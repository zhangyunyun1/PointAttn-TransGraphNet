import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .utils import MLP_Res, build_latent_flow, \
    reparameterize_gaussian, gaussian_entropy, \
    standard_normal_logprob, truncated_normal_, fps_subsample
from loss_functions import chamfer_3DDist # chamfer_l2 as chamfer
from loss_functions import emdModule  # EMD 
from .GraphSAGE import GraphSAGE

        
        
class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        out_points = self.mlp_4(x3)  # (b, 3, 256)
        return out_points

class Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_p0=512, radius=1, bounding=True, up_factors=None):
        super(Decoder, self).__init__()
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_p0)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            
            uppers.append(GraphSAGE(dim_feat=dim_feat, up_factor=factor, i=i, bounding=bounding, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, feat):
        feat = feat.unsqueeze(-1)
        arr_pcd = []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  # (B, 512, 3)

        arr_pcd.append(pcd)
        feat_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()

        for upper in self.uppers:
            pcd, feat_prev = upper(pcd, feat, feat_prev)  
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd

# PointAttention
class PointAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super(PointAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1)
        )
    
    def forward(self, x):
        # x: (B, C, N)
        scores = self.attention(x)  # (B, 1, N)
        weights = F.softmax(scores, dim=2)  # (B, 1, N)
        out = x * weights  # (B, C, N)
        out = torch.sum(out, dim=2)  # (B, C)
        return out

# Position encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (N, B, C)
        x = x + self.pe[:x.size(0), :]
        return x

# Transformer
class TransformerEncoderModule(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super(TransformerEncoderModule, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
    
    def forward(self, x):
        # x: (B, C, N) -> (N, B, C) for Transformer
        x = x.permute(2, 0, 1)  # (N, B, C)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (N, B, C)
        x = x.permute(1, 0, 2)  # (B, N, C)
        
        x = torch.mean(x, dim=1)  # (B, C)
        return x

# PointNet
class PointNetEncoder(nn.Module):
    def __init__(self, zdim=512, input_dim=3):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        # x: (B, N, 3)
        x = x.transpose(1, 2)  # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 128, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 256, N)
        x = self.bn4(self.conv4(x))          # (B, 512, N)
        x = torch.max(x, 2, keepdim=True)[0] # (B, 512, 1)
        x = x.view(-1, 512)                   # (B, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))  # (B, 256)
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))  # (B, 128)
        m = self.fc3_m(m)                           # (B, zdim)

        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))  # (B, 256)
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))  # (B, 128)
        v = self.fc3_v(v)                           # (B, zdim)

        return m, v

# Local-Global Dynamic Feature Fusion
class CombinedEncoder(nn.Module):
    def __init__(self, zdim=512, input_dim=3, transformer_layers=6, transformer_heads=8, dropout_p=0.3):
        super(CombinedEncoder, self).__init__()
        self.encoder = PointNetEncoder(zdim=zdim, input_dim=input_dim)  # (B, zdim), (B, zdim)
        self.attention = PointAttention(in_dim=zdim)  # (B, zdim)
        self.transformer = TransformerEncoderModule(d_model=zdim, nhead=transformer_heads, num_layers=transformer_layers)
        self.dropout = nn.Dropout(p=dropout_p)
        

        self.fc1_m = nn.Linear(zdim * 2, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)
        
        self.fc1_v = nn.Linear(zdim * 2, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        # x: (B, N, 3)
        z_mu, z_sigma = self.encoder(x)  # (B, zdim), (B, zdim)
        
        z_mu_expanded = z_mu.unsqueeze(-1)  # (B, zdim, 1)
        
        attended_features = self.attention(z_mu_expanded)  # (B, zdim)
        
        transformer_features = self.transformer(z_mu_expanded)  # (B, zdim)
        
        combined_features = torch.cat([attended_features, transformer_features], dim=1)  # (B, 2 * zdim)
        combined_features = self.dropout(combined_features)
        
        m = F.relu(self.fc_bn1_m(self.fc1_m(combined_features)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)  # (B, zdim)
        
        v = F.relu(self.fc_bn1_v(self.fc1_v(combined_features)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)  # (B, zdim)
        
        return m, v


class ModelVAE(nn.Module):

    def __init__(self, **kwargs):
        super(ModelVAE, self).__init__()
        dim_feat = kwargs.get('dim_feat', 512)
        num_p0 = kwargs.get('num_p0', 512)
        radius = kwargs.get('radius', 1)
        bounding = kwargs.get('bounding', True)
        up_factors = kwargs.get('up_factors', [2, 2])
        args = kwargs.get('args', None)
        
        self.encoder = CombinedEncoder(zdim=dim_feat, input_dim=3,
                                       transformer_layers=kwargs.get('transformer_layers', 6),
                                       transformer_heads=kwargs.get('transformer_heads', 8),
                                       dropout_p=kwargs.get('dropout_p', 0.3))
        self.flow = build_latent_flow(args)
        self.decoder = Decoder(dim_feat=dim_feat, num_p0=num_p0,
                               radius=radius, up_factors=up_factors, bounding=bounding)
        self.chamfer_dist = chamfer_3DDist()
        self.emd_calculator = emdModule().cuda()
        
    def chamfer_l2(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)
        return torch.mean(d1) + torch.mean(d2)

    def get_loss(self, x, kl_weight, writer=None, it=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        # print(x.size())
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)

        # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)  # (B, )

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)  # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)

        # Negative ELBO of P(X|z)
        p1, p2, p3 = self.decoder(z)

        x_512 = fps_subsample(x, 512)

        cd_1 = self.chamfer_l2(p1, x_512)

        cd_3 = self.chamfer_l2(p3, x)
        # EMD 
        emd_1 = self.compute_emd(p1, x_512).mean()
        emd_3 = self.compute_emd(p3, x).mean()
        loss_recons = cd_1 + cd_3 + emd_1 + emd_3  # + cd_2

        # Loss
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = loss_recons
        loss = kl_weight * (loss_entropy + loss_prior) + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', loss_entropy, it)
            writer.add_scalar('train/loss_prior', loss_prior, it)
            writer.add_scalar('train/loss_recons', loss_recons, it)
            writer.add_scalar('train/z_mean', z_mu.mean(), it)
            writer.add_scalar('train/z_mag', z_mu.abs().max(), it)
            writer.add_scalar('train/z_var', (0.5 * z_sigma).exp().mean(), it)

        return loss
    def compute_emd(self, pred_pcs, gt_pcs, eps=0.05, iters=3000):
        """
        Computes the Earth Mover's Distance (EMD) between predicted and ground truth point clouds.

        Args:
            pred_pcs (torch.Tensor): Predicted point clouds, shape (B, N, 3)
            gt_pcs (torch.Tensor): Ground truth point clouds, shape (B, N, 3)
            eps (float): Parameter to balance error rate and speed of convergence
            iters (int): Number of iterations for the auction algorithm

        Returns:
            torch.Tensor: EMD for each sample in the batch, shape (B,)
        """
        
        dist, assignment = self.emd_calculator(pred_pcs, gt_pcs, eps, iters)
        
       
        emd = torch.sqrt(dist).mean(dim=1)
        
        return emd

    def sample(self, w, truncate_std=None):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.decoder(z)[-1]
        return samples
