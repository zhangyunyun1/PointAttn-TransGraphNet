import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import sys
import time
sys.path.append('..')
import colorsys
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
import numpy as np
import open3d as o3d
import h5py


colors_points = (240 / 255, 183 / 255, 117 / 255)
colors_path1 = [0.7, 0.7, 0.7]
colors_path2 = [1, 0, 0]

def load_h5(path, verbose=False):
    if verbose:
        print("Loading %s \n" % (path))
    f = h5py.File(path, 'r')
    cloud_data = np.array(f['data'])
    f.close()
    return cloud_data.astype(np.float64)

def load_pcd(path):
    pc = o3d.io.read_point_cloud(path)
    ptcloud = np.array(pc.points)
    return ptcloud

def show_points(points, color=None):
    test_pcd = o3d.geometry.PointCloud()
    test_pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        test_pcd.paint_uniform_color(color)
    else:
        test_pcd.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([test_pcd], window_name="Open3D2", point_show_normal=True)

def create_sphere_at_xyz(xyz, colors=None, radius=0.12, resolution=4):
    """
    Create a spherical mesh at xyz to represent a point
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    if colors is None:
        sphere.paint_uniform_color([0.7, 0.1, 0.1])
    else:
        sphere.paint_uniform_color(colors)
    sphere = sphere.translate(xyz)
    return sphere

def create_pcd_mesh(point_cloud, colors=None):
    """
    Create each point in the point cloud as a small sphere to form a mesh.
    """
    mesh = []
    m = point_cloud.shape[0]
    if colors is None:
        colors = np.tile(np.array([1, 0, 0]), (m, 1))
    else:
        colors = np.array(colors)
        if colors.ndim == 1:
            colors = np.tile(colors, (m, 1))
    for i in range(m):
        mesh.append(create_sphere_at_xyz(point_cloud[i], colors=colors[i]))
    mesh_pcd = mesh[0]
    for i in range(1, len(mesh)):
        mesh_pcd += mesh[i]
    return mesh_pcd

def rotation_matrix_from_vectors(vec1, vec2):

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def get_line(point1, point2, radius=0.0003, resolution=7, colors=None):

    height = np.sqrt(np.sum((point1 - point2) ** 2))
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius,
                                                         height=height,
                                                         resolution=resolution)
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color(colors)
    mid = (point1 + point2) / 2
    vec1 = np.array([0, 0, height / 2])
    vec2 = point2 - mid
    T = np.eye(4)
    T[:3, :3] = rotation_matrix_from_vectors(vec1, vec2)
    T[:3, 3] = mid
    cylinder = cylinder.transform(T)
    return cylinder

def get_line_set(pcd1, pcd2, radius=0.001, resolution=10, colors=[0.7, 0.7, 0.7]):

    lines_mesh = []
    for i in range(pcd1.shape[0]):
        lines_mesh.append(get_line(pcd1[i], pcd2[i], radius=radius, resolution=resolution, colors=colors))
    mesh = lines_mesh[0]
    for i in range(1, len(lines_mesh)):
        mesh += lines_mesh[i]
    return mesh

def splitting_paths(pcd1, pcd2, inds=None, colors_points=colors_points, colors_paths=colors_path2):

    n1 = pcd1.shape[0]
    n2 = pcd2.shape[0]
    up_factor = n2 // n1
    pcd1 = np.tile(pcd1, (1, up_factor)).reshape((n2, 3))
    mesh_point1 = create_pcd_mesh(pcd1, colors=colors_points)
    displacements = None
    if inds is None:
        inds = np.arange(n1)
    for i in inds:
        new_dispacements = get_line_set(pcd1[i * up_factor: (i + 1) * up_factor],
                                        pcd2[i * up_factor: (i + 1) * up_factor], colors=colors_paths)
        if displacements is None:
            displacements = new_dispacements
        else:
            displacements += new_dispacements
    mesh_out = mesh_point1 + displacements
    return mesh_out

def splitting_paths_triple(pcd1, pcd2, pcd3, inds=None, colors_points=colors_points, colors_path1=colors_path1,
                           colors_path2=colors_path2):

    n1 = pcd1.shape[0]
    n2 = pcd2.shape[0]
    n3 = pcd3.shape[0]
    up_factor_1 = n2 // n1
    up_factor_2 = n3 // n2
    up_factor = up_factor_1 * up_factor_2
    pcd1_to_2 = np.tile(pcd1, (1, up_factor_1)).reshape((n2, 3))
    pcd2_to_3 = np.tile(pcd2, (1, up_factor_2)).reshape((n3, 3))
    if inds is None:
        inds = np.arange(n1)
    mesh_point1 = create_pcd_mesh(pcd1, colors=colors_points)
    displacements = None
    for j, i in enumerate(inds):
        new_dispacements = get_line_set(pcd1_to_2[i * up_factor_1: (i + 1) * up_factor_1],
                                        pcd2[i * up_factor_1: (i + 1) * up_factor_1], colors=colors_path1)
        new_dispacements += get_line_set(pcd2_to_3[i * up_factor: (i + 1) * up_factor],
                                         pcd3[i * up_factor: (i + 1) * up_factor], colors=colors_path2)
        if displacements is None:
            displacements = new_dispacements
        else:
            displacements += new_dispacements
    mesh_out = mesh_point1 + displacements
    return mesh_out


from utils.dataset import ShapeNetCore
from utils.misc import seed_all, get_logger, str_list, THOUSAND, get_new_log_dir, CheckpointManager, BlackHole, \
    get_linear_scheduler, log_hyperparams
from utils.data import DataLoader, get_data_iterator
from models.model_main import ModelVAE
from models.utils import add_spectral_norm, spectral_norm_power_iteration
from evaluation import EMD_CD, compute_all_metrics, jsd_between_point_cloud_sets

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # Arguments
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument('--model', type=str, default='flow', choices=['flow', 'gaussian'])
    parser.add_argument('--log_root', type=str, default='logs_gen/PointAttn_TransGraphNet/airplane')
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--sched_mode', type=str, default='linear')
    parser.add_argument('--flexibility', type=float, default=0.0)
    parser.add_argument('--truncate_std', type=float, default=2.0)
    parser.add_argument('--latent_flow_depth', type=int, default=14)
    parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--kl_weight', type=float, default=1e-3)
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

    # Datasets and loaders
    parser.add_argument('--dataset_path', type=str, default='data/shapenet.hdf5')
    parser.add_argument('--categories', type=str_list, default=['airplane'])
    parser.add_argument('--scale_mode', type=str, default='shape_unit')
    parser.add_argument('--train_batch_size', type=int, default=96)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # Transformer 
    parser.add_argument('--transformer_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--transformer_heads', type=int, default=8, help='Number of attention heads in transformer')
    parser.add_argument('--dropout_p', type=float, default=0.3, help='Dropout probability')

    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    parser.add_argument('--end_lr', type=float, default=1e-4)
    parser.add_argument('--sched_start_epoch', type=int, default=200 * THOUSAND)
    parser.add_argument('--sched_end_epoch', type=int, default=400 * THOUSAND)

    # Training
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_iters', type=int, default=float('inf'))
    parser.add_argument('--val_freq', type=int, default=10000)
    parser.add_argument('--test_freq', type=int, default=100000)
    parser.add_argument('--test_size', type=int, default=400)
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()
    seed_all(args.seed)

    # Logging
    if args.logging:
        log_dir = get_new_log_dir(args.log_root, prefix='GEN_', postfix='_' + args.tag if args.tag is not None else '')
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        ckpt_mgr = CheckpointManager(log_dir)
        log_hyperparams(writer, args)
    else:
        logger = get_logger('train', None)
        writer = BlackHole()
        ckpt_mgr = BlackHole()
    logger.info(args)

    # Datasets and loaders
    logger.info('Loading datasets...')
    train_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='train',
        scale_mode=args.scale_mode,
    )
    val_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='val',
        scale_mode=args.scale_mode,
    )
    train_iter = get_data_iterator(DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers
    ))

    # Model
    logger.info('Building model...')
    model = ModelVAE(dim_feat=args.latent_dim,
                     args=args,
                     up_factors=[2, 2]).to(args.device)
    logger.info(repr(model))
    if args.spectral_norm:
        add_spectral_norm(model, logger=logger)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=args.sched_start_epoch,
        end_epoch=args.sched_end_epoch,
        start_lr=args.lr,
        end_lr=args.end_lr)


    def train(it):
        batch = next(train_iter)
        x = batch['pointcloud'].to(args.device)
        optimizer.zero_grad()
        model.train()
        if args.spectral_norm:
            spectral_norm_power_iteration(model, n_power_iterations=1)
        kl_weight = args.kl_weight
        loss = model.get_loss(x, kl_weight=kl_weight, writer=writer, it=it)
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' %
                    (it, loss.item(), orig_grad_norm, kl_weight))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/kl_weight', kl_weight, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad_norm', orig_grad_norm, it)
        writer.flush()



    def validate_inspect(it):

        z = torch.randn([args.num_samples, args.latent_dim]).to(args.device)
        x = model.sample(z).detach().cpu().numpy()  # (num_samples, N, 3)

        all_meshes = []
        num_per_row = 5  
        offset_x = 10.0
        offset_y = 10.0


        theta = np.deg2rad(135)
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        Rx = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float32)
        R_fixed_new = Rz @ Rx

        control_colors = np.array([
            [1.0, 0.0, 0.2], 
            [1.0, 0.5, 0.0],  
            [0.2, 1.0, 0.2],  
            [0.0, 0.6, 1.0],  
            [0.8, 0.0, 1.0],  
        ]) * 0.85

        for i in range(x.shape[0]):
            pts = x[i]  

            center = pts.mean(axis=0)
            pts_centered = pts - center
            pts_rotated = (R_fixed_new @ pts_centered.T).T + center

            row = i // num_per_row
            col = i % num_per_row
            grid_offset = np.array([col * offset_x, -row * offset_y, 0.0])
            pts_rotated += grid_offset

            ys = pts_rotated[:, 1]
            t = (ys - ys.min()) / (ys.max() - ys.min() + 1e-8)
            r = np.interp(t, [0.0, 0.25, 0.5, 0.75, 1.0], control_colors[:, 0])
            g = np.interp(t, [0.0, 0.25, 0.5, 0.75, 1.0], control_colors[:, 1])
            b = np.interp(t, [0.0, 0.25, 0.5, 0.75, 1.0], control_colors[:, 2])
            colors = np.stack([r, g, b], axis=1)

            mesh = None
            for p, c in zip(pts_rotated, colors):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=4)
                sphere.paint_uniform_color(c)
                sphere.translate(p)
                mesh = sphere if mesh is None else (mesh + sphere)
            mesh.compute_vertex_normals()
            all_meshes.append(mesh)

        combined = all_meshes[0]
        for m in all_meshes[1:]:
            combined += m

        width, height = 7680, 4320  
        renderer = OffscreenRenderer(width, height)
        renderer.scene.set_background([1, 1, 1, 1])

        mat = MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("combined", combined, mat)

        bbox = combined.get_axis_aligned_bounding_box()
        center = np.asarray(bbox.get_center(), dtype=np.float32).reshape(3)
        eye = center + np.array([0.0, 0.0, 50.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        renderer.setup_camera(45.0, center, eye, up)  
        img = renderer.render_to_image()
        save_dir = args.log_root if os.path.exists(args.log_root) else './generated_images'
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, f"airplane_iter_{it}.png")
        o3d.io.write_image(img_path, img)
        logger.info(f"[Inspect] Saved airplane visualization to {img_path}")
        del renderer

    def test(it):
        ref_pcs = []
        for i, data in enumerate(val_dset):
            if i >= args.test_size:
                break
            ref_pcs.append(data['pointcloud'].unsqueeze(0))
        ref_pcs = torch.cat(ref_pcs, dim=0)
        gen_pcs = []
        for i in tqdm(range(0, math.ceil(args.test_size / args.val_batch_size)), 'Generate'):
            with torch.no_grad():
                z = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
                x = model.sample(z)
                gen_pcs.append(x.detach().cpu())
        gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_size]
        with torch.no_grad():
            results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.val_batch_size,
                                          accelerated_cd=True)
            results = {k: v.item() for k, v in results.items()}
            jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
            results['jsd'] = jsd
        writer.add_scalar('test/Coverage_CD', results['lgan_cov-CD'], global_step=it)
        writer.add_scalar('test/Coverage_EMD', results['lgan_cov-EMD'], global_step=it)
        writer.add_scalar('test/MMD_CD', results['lgan_mmd-CD'], global_step=it)
        writer.add_scalar('test/MMD_EMD', results['lgan_mmd-EMD'], global_step=it)
        writer.add_scalar('test/1NN_CD', results['1-NN-CD-acc'], global_step=it)
        writer.add_scalar('test/1NN_EMD', results['1-NN-EMD-acc'], global_step=it)
        writer.add_scalar('test/JSD', results['jsd'], global_step=it)
        logger.info('[Test] Coverage  | CD %.6f | EMD %.6f' % (results['lgan_cov-CD'], results['lgan_cov-EMD']))
        logger.info('[Test] MinMatDis | CD %.6f | EMD %.6f' % (results['lgan_mmd-CD'], results['lgan_mmd-EMD']))
        logger.info('[Test] 1NN-Accur | CD %.6f | EMD %.6f' % (results['1-NN-CD-acc'], results['1-NN-EMD-acc']))
        logger.info('[Test] JsnShnDis | %.6f ' % (results['jsd']))

    logger.info('Start training...')
    try:
        it = 1
        while it <= args.max_iters:
            train(it)
            if it % args.val_freq == 0 or it == args.max_iters:
                validate_inspect(it)
                opt_states = {
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
            if it % args.test_freq == 0 or it == args.max_iters:
                test(it)
            it += 1
    except KeyboardInterrupt:
        logger.info('Terminating...')
