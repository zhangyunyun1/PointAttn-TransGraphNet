import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, knn_graph
from torch_geometric.utils import add_self_loops
from models.utils import MLP_Res, MLP_CONV
from models.transformer import SkipTransformer
from torch.nn import Dropout, BatchNorm1d

class GraphSAGE(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1, bounding=True, global_feat=True, aggr='mean'):
        """
        GraphSAGE
        
        Args:
            dim_feat (int): Dimension of global features.
            up_factor (int): Upsampling factor.
            i (int): Index for radius scaling.
            radius (float): Radius for bounding.
            bounding (bool): Whether to apply bounding.
            global_feat (bool): Whether to use global features.
            aggr (str): Aggregation method for SAGEConv ('mean', 'max', etc.).
        """
        super(GraphSAGE, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.bounding = bounding
        self.radius = radius
        self.global_feat = global_feat
        self.ps_dim = 32 if global_feat else 64


        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        # The graph convolutional layer employs SAGEConv
        self.sage_1 = SAGEConv(128, 128, aggr=aggr)
        self.bn1 = BatchNorm1d(128)
        self.dropout1 = Dropout(p=0.5)
        
        self.sage_2 = SAGEConv(128, 128, aggr=aggr)
        self.bn2 = BatchNorm1d(128)
        self.dropout2 = Dropout(p=0.5)
        
        self.mlp_2 = MLP_CONV(
            in_channel=128 * 2 + dim_feat if self.global_feat else 128, 
            layer_dims=[256, 128]
        )

        self.skip_transformer = SkipTransformer(in_channel=128, dim=64)
        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, self.ps_dim])
        self.ps = nn.ConvTranspose1d(self.ps_dim, 128, up_factor, up_factor, bias=False)
        self.up_sampler = nn.Upsample(scale_factor=up_factor, mode='nearest')
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)
        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def get_edge_index(self, pcd_prev, k=16):
        """
        Compute the adjacency matrix（edge_index）
        
        Args:
            pcd_prev (Tensor)
            k (int)
        
        Returns:
            edge_index (Tensor)
        """
        B, C, N = pcd_prev.shape
        edge_indices = []
        for b in range(B):
            x = pcd_prev[b].transpose(0, 1)  # (N, 3)
            edge_index = knn_graph(x, k=k, batch=None, loop=False)
            edge_index += b * N
            edge_indices.append(edge_index)

        edge_index = torch.cat(edge_indices, dim=1)  # (2, B*E)
        return edge_index

    def forward(self, pcd_prev, feat_global=None, K_prev=None):
        """
        
        Args:
            pcd_prev (Tensor): (B, 3, N_prev)
            feat_global (Tensor, optional): (B, dim_feat, 1)
            K_prev (Tensor, optional): (B, 128, N_prev)
        
        Returns:
            pcd_child (Tensor): (B, 3, N_prev * up_factor)
            K_curr (Tensor): (B, 128, N_prev * up_factor)
        """
        B, _, N_prev = pcd_prev.shape

        feat_1 = self.mlp_1(pcd_prev)  # (B, 128, N_prev)
        edge_index = self.get_edge_index(pcd_prev, k=16).to(pcd_prev.device)  # (2, B*E)
        feat_sage_1 = self.sage_1(feat_1.view(-1, 128), edge_index)  # (B*N_prev, 128)
        feat_sage_1 = F.relu(self.bn1(feat_sage_1))
        feat_sage_1 = self.dropout1(feat_sage_1)
        feat_sage_2 = self.sage_2(feat_sage_1, edge_index)  # (B*N_prev, 128)
        feat_sage_2 = F.relu(self.bn2(feat_sage_2))
        feat_sage_2 = self.dropout2(feat_sage_2)
        feat_sage_2 = feat_sage_2 + feat_sage_1  # (B*N_prev, 128)
        feat_sage_2 = feat_sage_2.view(B, N_prev, 128).transpose(1, 2)  # (B, 128, N_prev)

        if self.global_feat and feat_global is not None:
            global_feat_repeated = feat_global.repeat(1, 1, N_prev)  # (B, dim_feat, N_prev)
            feat_max = torch.max(feat_sage_2, 2, keepdim=True)[0].repeat(1, 1, N_prev)  # (B, 128, N_prev)
            feat_1 = torch.cat([feat_sage_2, feat_max, global_feat_repeated], dim=1)  # (B, 128*2 + dim_feat, N_prev)
        else:
            feat_1 = feat_sage_2  # (B, 128, N_prev)

        Q = self.mlp_2(feat_1)  # (B, 128, N_prev)
        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)  # (B, 64, N_prev)
        feat_child = self.mlp_ps(H)  # (B, ps_dim, N_prev)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)

        H_up = self.up_sampler(H)  # (B, 64, N_prev * up_factor)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], dim=1))  # (B, 128, N_prev * up_factor)

        delta = self.mlp_delta(torch.relu(K_curr))  # (B, 3, N_prev * up_factor)
        if self.bounding:
            delta = torch.tanh(delta) / (self.radius ** self.i)  # (B, 3, N_prev * up_factor)
      
        pcd_child = self.up_sampler(pcd_prev)  # (B, 3, N_prev * up_factor)
        pcd_child = pcd_child + delta  # (B, 3, N_prev * up_factor)

        return pcd_child, K_curr
