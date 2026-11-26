import os
import sys
from pathlib import Path

# Allow direct invocation from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import argparse

# Parse --gpu argument early, before importing torch
# This allows us to set CUDA_VISIBLE_DEVICES before torch initializes
def _parse_gpu_early():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return '0'  # Default to GPU 0

_gpu_id = _parse_gpu_early()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_id

import torch
import json
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from train_generation import (
    Model, get_betas, get_dataset, remove_module_prefix
)
from datasets.dataset_generation import FDIS
from utils.file_utils import setup_logging, setup_output_subdirs, get_output_dir
from utils.visualize import export_dentition_npy

def chamfer_distance_numpy(points1, points2):
    """
    Compute Chamfer Distance between two point clouds using numpy

    Args:
        points1: (N, 3) numpy array
        points2: (M, 3) numpy array

    Returns:
        Chamfer distance (average of both directions)
    """
    # Convert to torch for efficient computation
    points1_t = torch.from_numpy(points1).float().unsqueeze(0)  # (1, N, 3)
    points2_t = torch.from_numpy(points2).float().unsqueeze(0)  # (1, M, 3)

    # Compute pairwise distances
    dist_matrix = torch.cdist(points1_t, points2_t)  # (1, N, M)

    # Forward: min distance from points1 to points2
    forward = dist_matrix.min(dim=2)[0].mean()

    # Backward: min distance from points2 to points1
    backward = dist_matrix.min(dim=1)[0].mean()

    return (forward + backward).item() / 2.0


def compute_root_axis_pca(points):
    """
    Compute tooth root axis using Principal Component Analysis.

    The first principal component represents the dominant direction of variation,
    which corresponds to the long axis (root axis) of the tooth.

    Reference: https://link.springer.com/article/10.1007/s12565-019-00506-1

    Args:
        points: numpy array of shape (N, 3) representing tooth point cloud

    Returns:
        dict containing:
            - 'axis': Unit vector along the root axis (3,)
            - 'centroid': Centroid of the point cloud (3,)
            - 'explained_variance': Variance explained by first PC
    """
    # Remove invalid points (zeros, NaNs)
    valid_mask = ~np.isnan(points).any(axis=1) & (np.abs(points).sum(axis=1) > 1e-6)
    valid_points = points[valid_mask]

    if len(valid_points) < 3:
        raise ValueError(f"Insufficient valid points ({len(valid_points)}) for PCA")

    # Compute centroid
    centroid = np.mean(valid_points, axis=0)

    # Center the points
    centered_points = valid_points - centroid

    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(centered_points)

    # First principal component is the root axis
    root_axis = pca.components_[0]

    # Ensure consistent direction (point towards positive z if possible)
    if root_axis[2] < 0:
        root_axis = -root_axis

    return {
        'axis': root_axis,
        'centroid': centroid,
        'explained_variance': pca.explained_variance_ratio_[0]
    }


def compute_angular_difference(axis1, axis2, in_degrees=True):
    """
    Compute angular difference between two axes (unsigned angle).

    Args:
        axis1: First axis vector (3,)
        axis2: Second axis vector (3,)
        in_degrees: Return angle in degrees (default: True)

    Returns:
        Angular difference in degrees or radians
    """
    # Normalize vectors
    axis1_norm = axis1 / np.linalg.norm(axis1)
    axis2_norm = axis2 / np.linalg.norm(axis2)

    # Compute dot product (cosine of angle)
    cos_angle = np.clip(np.dot(axis1_norm, axis2_norm), -1.0, 1.0)

    # Get unsigned angle
    angle_rad = np.arccos(np.abs(cos_angle))  # abs() for unsigned angle

    if in_degrees:
        return np.degrees(angle_rad)
    return angle_rad


def compute_axis_line_distance(axis_info1, axis_info2):
    """
    Compute minimum distance between two 3D lines (axes).

    This is the shortest distance between two skew lines in 3D space,
    measuring how parallel and aligned the tooth axes are.

    Args:
        axis_info1: Dict with 'axis' and 'centroid' for first tooth
        axis_info2: Dict with 'axis' and 'centroid' for second tooth

    Returns:
        Minimum distance between the two axis lines
    """
    # Line 1: P1 + t * d1
    P1 = axis_info1['centroid']
    d1 = axis_info1['axis']

    # Line 2: P2 + s * d2
    P2 = axis_info2['centroid']
    d2 = axis_info2['axis']

    # Vector between points
    w = P1 - P2

    # Compute coefficients
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w)
    e = np.dot(d2, w)

    # Compute parameters
    denom = a * c - b * b

    if abs(denom) < 1e-10:
        # Lines are parallel
        # Distance is perpendicular distance from P2 to line 1
        cross_product = np.cross(d1, w)
        distance = np.linalg.norm(cross_product) / np.linalg.norm(d1)
    else:
        # Lines are skew
        t = (b * e - c * d) / denom
        s = (a * e - b * d) / denom

        # Closest points
        closest1 = P1 + t * d1
        closest2 = P2 + s * d2

        distance = np.linalg.norm(closest1 - closest2)

    return distance


def centroid_distance(points1, points2):
    """
    Compute Euclidean distance between centroids of two point clouds

    This measures the translational offset between the generated and ground truth teeth,
    indicating positional accuracy independent of shape.

    Args:
        points1: (N, 3) numpy array
        points2: (M, 3) numpy array

    Returns:
        Euclidean distance between centroids
    """
    centroid1 = np.mean(points1, axis=0)  # (3,)
    centroid2 = np.mean(points2, axis=0)  # (3,)

    return np.linalg.norm(centroid1 - centroid2)


def earth_movers_distance(points1, points2, max_iter=100):
    """
    Compute Earth Mover's Distance (Wasserstein Distance) between two point clouds

    EMD measures the minimum cost of transforming one distribution into another,
    providing a more robust metric than Chamfer Distance for overall shape similarity.

    Uses approximate EMD via iterative linear programming relaxation for efficiency.

    Args:
        points1: (N, 3) numpy array
        points2: (M, 3) numpy array
        max_iter: Maximum iterations for optimization

    Returns:
        Earth Mover's Distance (approximate)
    """
    # For efficiency, we'll use a simplified EMD calculation
    # Full EMD requires solving a linear assignment problem which is O(N^3)
    # We use a greedy approximation that's O(N^2)

    N, M = points1.shape[0], points2.shape[0]

    # Convert to torch for efficient computation
    points1_t = torch.from_numpy(points1).float()  # (N, 3)
    points2_t = torch.from_numpy(points2).float()  # (M, 3)

    # Compute pairwise distance matrix
    dist_matrix = torch.cdist(points1_t.unsqueeze(0), points2_t.unsqueeze(0)).squeeze(0)  # (N, M)

    # Use Hungarian algorithm approximation via min-cost matching
    # For equal sized point clouds, this is the optimal transport cost
    if N == M:
        # Optimal transport: each point in points1 matched to exactly one in points2
        # Use greedy matching for efficiency (not optimal but fast)
        matched_distances = []
        available_indices = set(range(M))

        for i in range(N):
            # Find nearest unmatched point in points2
            distances_to_available = dist_matrix[i, list(available_indices)]
            min_idx_in_available = torch.argmin(distances_to_available).item()
            actual_idx = list(available_indices)[min_idx_in_available]

            matched_distances.append(dist_matrix[i, actual_idx].item())
            available_indices.remove(actual_idx)

        return np.mean(matched_distances)
    else:
        # Different sizes: use average min distance (approximation)
        # This is similar to Chamfer but with uniform weighting
        if N < M:
            # Match each point in points1 to closest in points2
            min_dists = dist_matrix.min(dim=1)[0]
        else:
            # Match each point in points2 to closest in points1
            min_dists = dist_matrix.min(dim=0)[0]

        return min_dists.mean().item()


def hausdorff_distance(points1, points2):
    """
    Compute Hausdorff Distance between two point clouds

    Hausdorff Distance measures the maximum distance from any point in one set
    to the nearest point in the other set. It captures the worst-case geometric error.

    Args:
        points1: (N, 3) numpy array
        points2: (M, 3) numpy array

    Returns:
        Hausdorff distance (max of both directions)
    """
    # Convert to torch for efficient computation
    points1_t = torch.from_numpy(points1).float().unsqueeze(0)  # (1, N, 3)
    points2_t = torch.from_numpy(points2).float().unsqueeze(0)  # (1, M, 3)

    # Compute pairwise distances
    dist_matrix = torch.cdist(points1_t, points2_t).squeeze(0)  # (N, M)

    # Forward: max of min distances from points1 to points2
    forward = dist_matrix.min(dim=1)[0].max()

    # Backward: max of min distances from points2 to points1
    backward = dist_matrix.min(dim=0)[0].max()

    # Hausdorff is the maximum of both directions
    return max(forward.item(), backward.item())


def f1_score_point_cloud(points1, points2, thresholds=[0.005, 0.01, 0.025, 0.05]):
    """
    Compute F1-score between two point clouds at multiple distance thresholds.

    F1-score is a threshold-based metric that balances precision and recall:
    - Precision: Fraction of generated points within threshold of a GT point
    - Recall: Fraction of GT points within threshold of a generated point
    - F1: Harmonic mean of precision and recall

    This metric is effective because:
    - More interpretable than Chamfer Distance (percentage of points within tolerance)
    - Less sensitive to outliers than Hausdorff Distance
    - Balances coverage (recall) and accuracy (precision)
    - Multi-threshold analysis shows performance at different clinical tolerances

    Args:
        points1: Generated point cloud (N, 3) numpy array
        points2: Ground truth point cloud (M, 3) numpy array
        thresholds: List of distance thresholds τ in normalized units
                    (default: [0.005, 0.01, 0.025, 0.05] ≈ [0.2, 0.4, 1.0, 2.0] mm)

    Returns:
        dict with precision, recall, f1_score for each threshold
    """
    # Convert to torch for efficient computation
    points1_t = torch.from_numpy(points1).float().unsqueeze(0)  # (1, N, 3)
    points2_t = torch.from_numpy(points2).float().unsqueeze(0)  # (1, M, 3)

    # Compute pairwise distances
    dist_matrix = torch.cdist(points1_t, points2_t).squeeze(0)  # (N, M)

    # Min distances in each direction
    min_dist_to_gt = dist_matrix.min(dim=1)[0]  # (N,) - for each gen point, dist to nearest GT
    min_dist_to_gen = dist_matrix.min(dim=0)[0]  # (M,) - for each GT point, dist to nearest gen

    results = {}
    for threshold in thresholds:
        # Precision: for each generated point, is there a GT point within threshold?
        precision = (min_dist_to_gt < threshold).float().mean().item()

        # Recall: for each GT point, is there a generated point within threshold?
        recall = (min_dist_to_gen < threshold).float().mean().item()

        # F1-score (harmonic mean)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    return results


def validate_checkpoint(checkpoint_path, data_path, output_dir=None,
                       num_samples=10, n_missing_teeth=1, compute_metrics=True, save_samples=True,
                       target_fdi=None):
    """
    Validate a trained checkpoint by generating samples and computing metrics

    Args:
        checkpoint_path: Path to .pth checkpoint file
        data_path: Path to dataset directory
        output_dir: Custom output directory (if None, auto-generated)
        num_samples: Number of validation samples to generate
        n_missing_teeth: Number of teeth to reconstruct per sample (1, 2, 3, etc.)
        compute_metrics: Whether to compute Chamfer Distance metrics
        save_samples: Whether to save generated point clouds to disk

    Returns:
        Dictionary containing validation metrics
    """

    # Create output directory
    if output_dir is None:
        output_dir = get_output_dir(os.path.dirname(__file__), 'validation')
    else:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)

    if save_samples:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    # GPU is already set via CUDA_VISIBLE_DEVICES at module load time
    # So cuda:0 refers to the GPU specified by --gpu argument
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device} (physical GPU {_gpu_id})')

    # Load checkpoint
    logger.info(f'Loading checkpoint from: {checkpoint_path}')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    epoch = ckpt.get('epoch', 'unknown')
    logger.info(f'Checkpoint epoch: {epoch}')

    # Load options from checkpoint directory (if available)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    options_file = os.path.join(checkpoint_dir, 'options.txt')

    if os.path.exists(options_file):
        logger.info(f'Loading options from: {options_file}')
        with open(options_file, 'r') as f:
            saved_opts = json.load(f)

        # Create argparse namespace from saved options
        opt = argparse.Namespace(**saved_opts)
    else:
        logger.warning('options.txt not found, using default parameters')
        # Use default parameters
        opt = argparse.Namespace(
            nc=3,
            extra_feature_nc=10,
            max_missing_teeth=6,
            tooth_npoints=1024,
            beta_start=0.0001,
            beta_end=0.02,
            schedule_type='linear',
            time_num=1000,
            attention=True,
            dropout=0.1,
            embed_dim=96,
            loss_type='mse',
            model_mean_type='eps',
            model_var_type='fixedsmall',
            vox_res_mult=1.0,
            width_mult=0.75
        )

    # Load validation dataset
    logger.info(f'Loading validation dataset from: {data_path}')
    val_dataset = get_dataset(path=data_path, tooth_npoints=opt.tooth_npoints, mode='val')
    logger.info(f'Validation dataset size: {len(val_dataset)}')

    # Create model
    logger.info('Creating model...')
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type,
                  opt.model_var_type, opt.width_mult, opt.vox_res_mult)

    # Load model weights (handle both 'model_state' and 'model_state_dict' keys)
    state_dict_key = 'model_state' if 'model_state' in ckpt else 'model_state_dict'
    clean_sd = remove_module_prefix(ckpt[state_dict_key])

    # Auto-detect extra_feature_nc from checkpoint to avoid shape mismatches
    # when the checkpoint was trained with a different feature config.
    try:
        voxel_key = next(k for k in clean_sd if 'sa_layers.0.0.voxel_layers.0.weight' in k)
        in_ch = clean_sd[voxel_key].shape[1]
        inferred_extra_nc = in_ch - opt.nc
        if inferred_extra_nc != opt.extra_feature_nc:
            logger.info(
                f'Auto-detected extra_feature_nc={inferred_extra_nc} from checkpoint '
                f'(options had {opt.extra_feature_nc}) — rebuilding model'
            )
            opt.extra_feature_nc = inferred_extra_nc
            betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
            model = Model(opt, betas, opt.loss_type, opt.model_mean_type,
                          opt.model_var_type, opt.width_mult, opt.vox_res_mult)
    except StopIteration:
        logger.warning('Could not find SA voxel layer — using extra_feature_nc from options')

    load_result = model.load_state_dict(clean_sd, strict=False)
    if load_result.missing_keys:
        logger.warning(
            f'{len(load_result.missing_keys)} model keys missing from checkpoint '
            f'(random init): {load_result.missing_keys[:5]}'
        )
    if load_result.unexpected_keys:
        logger.info(
            f'{len(load_result.unexpected_keys)} stale checkpoint keys ignored: '
            f'{load_result.unexpected_keys[:5]}'
        )
    if not load_result.missing_keys:
        logger.info('Model weights loaded successfully — all model parameters matched.')

    model = model.to(device)
    model.eval()

    # Initialize metrics storage
    chamfer_distances = []
    emd_distances = []
    hausdorff_distances = []
    centroid_distances = []
    angular_differences = []
    axis_line_distances = []
    # F1-score at multiple thresholds (in normalized units)
    # 0.005 ≈ 0.2mm, 0.01 ≈ 0.4mm, 0.025 ≈ 1mm, 0.05 ≈ 2mm (scale factor = 39.2)
    f1_thresholds = [0.005, 0.01, 0.025, 0.05]
    f1_scores = {t: [] for t in f1_thresholds}
    f1_precisions = {t: [] for t in f1_thresholds}
    f1_recalls = {t: [] for t in f1_thresholds}
    sample_info = []

    logger.info(f'Starting validation: generating {num_samples} samples with {n_missing_teeth} teeth per sample...')
    logger.info('='*80)

    # Generate validation samples
    for sample_idx in tqdm(range(num_samples), desc="Validating"):
        try:
            # Sample validation data
            sample_batch_size = 1
            val_sample_list = val_dataset.sample_patient(sample_batch_size)
            val_dentition_points = torch.stack([sample['dentition_points'] for sample in val_sample_list], 0).to(device)
            val_fdi_indices = torch.stack([sample['fdi_indices'] for sample in val_sample_list], 0).to(device)
            val_bound = None
            if 'bounds_cyl' in val_sample_list[0]:
                val_bound = torch.stack([sample['bounds_cyl'] for sample in val_sample_list], 0).to(device)
            val_dentition_ids = [sample['patient_id'] for sample in val_sample_list]

            # Handle original missing teeth
            val_original_missing_mask = None
            if 'original_missing_mask' in val_sample_list[0]:
                val_original_missing_mask = torch.stack([sample['original_missing_mask'] for sample in val_sample_list], 0).to(device)

            # Create latent mask (which teeth to reconstruct)
            latent_mask_val = torch.zeros_like(val_dentition_points[:,:,:1,:1]).to(device)

            # Get existing teeth for this patient
            if val_original_missing_mask is not None:
                existing_teeth_indices = torch.where(~val_original_missing_mask[0])[0]
            else:
                existing_teeth_indices = torch.arange(val_dentition_points.shape[1], device=device)

            if len(existing_teeth_indices) == 0:
                logger.warning(f"Sample {sample_idx}: No existing teeth, skipping")
                continue

            # Select teeth to reconstruct for validation
            if target_fdi is not None:
                fdi_to_pos = {fdi: pos for pos, fdi in enumerate(FDIS)}
                valid_positions = []
                for fdi_num in target_fdi:
                    pos = fdi_to_pos.get(fdi_num)
                    if pos is None:
                        continue
                    if val_original_missing_mask is not None and val_original_missing_mask[0, pos]:
                        logger.warning(f"Sample {sample_idx}: FDI {fdi_num} naturally missing, skipping")
                        continue
                    valid_positions.append(pos)
                if len(valid_positions) == 0:
                    logger.warning(f"Sample {sample_idx}: All target FDI teeth naturally missing, skipping")
                    continue
                selected_existing = torch.tensor(valid_positions, device=device)
                logger.info(f"Sample {sample_idx}: Targeting FDI {[FDIS[p] for p in valid_positions]}")
            else:
                n_missing_val = n_missing_teeth

                # Check if we have enough teeth to sample
                if len(existing_teeth_indices) < n_missing_val:
                    logger.warning(f"Sample {sample_idx}: Not enough teeth ({len(existing_teeth_indices)}) for {n_missing_val} missing teeth, skipping")
                    continue

                # Prefer non-wisdom teeth
                non_wisdom_existing = []
                for idx in existing_teeth_indices:
                    fdi = FDIS[idx] if idx < len(FDIS) else 0
                    if fdi % 10 != 8:  # Not a wisdom tooth
                        non_wisdom_existing.append(idx.item())

                if len(non_wisdom_existing) >= n_missing_val:
                    perm = torch.randperm(len(non_wisdom_existing), device=device)[:n_missing_val]
                    selected_existing = torch.tensor(non_wisdom_existing, device=device)[perm]
                else:
                    perm = torch.randperm(len(existing_teeth_indices), device=device)[:n_missing_val]
                    selected_existing = existing_teeth_indices[perm]

            latent_mask_val[0, selected_existing, 0, 0] = 1
            obs_mask_val = torch.ones_like(latent_mask_val) - latent_mask_val

            # Ensure we don't try to observe naturally missing teeth
            if val_original_missing_mask is not None:
                obs_mask_val = obs_mask_val * (~val_original_missing_mask.unsqueeze(-1).unsqueeze(-1))

            val_data_dict = {
                'x0': val_dentition_points,
                'fdi_indices': val_fdi_indices,
                'l_mask': latent_mask_val,
                'o_mask': obs_mask_val,
                'bound': val_bound,
                'original_missing_mask': val_original_missing_mask
            }

            # Generate samples
            with torch.no_grad():
                sample_out = model.gen_samples(model_kwargs=val_data_dict, use_ddim=False).detach().cpu()
                val_dentition_points = val_dentition_points.detach().cpu()
                obs_mask_val = obs_mask_val.detach().cpu()
                latent_mask_val = latent_mask_val.detach().cpu()

            # Get missing tooth indices
            missing_idx = np.where(latent_mask_val[0].squeeze().numpy())[0]

            # Compute metrics for each reconstructed tooth
            if compute_metrics:
                for tooth_idx in missing_idx:
                    gen_tooth = sample_out[0, tooth_idx].numpy().T  # (1024, 3)
                    gt_tooth = val_dentition_points[0, tooth_idx].numpy().T  # (1024, 3)

                    # Compute all metrics
                    cd = chamfer_distance_numpy(gen_tooth, gt_tooth)
                    emd = earth_movers_distance(gen_tooth, gt_tooth)
                    hausdorff = hausdorff_distance(gen_tooth, gt_tooth)
                    centroid_dist = centroid_distance(gen_tooth, gt_tooth)

                    # Compute F1-score at multiple thresholds
                    f1_results = f1_score_point_cloud(gen_tooth, gt_tooth, thresholds=f1_thresholds)

                    # Compute root axis metrics
                    try:
                        gt_axis_info = compute_root_axis_pca(gt_tooth)
                        gen_axis_info = compute_root_axis_pca(gen_tooth)

                        angular_diff = compute_angular_difference(
                            gt_axis_info['axis'],
                            gen_axis_info['axis']
                        )

                        axis_line_dist = compute_axis_line_distance(
                            gt_axis_info,
                            gen_axis_info
                        )
                    except Exception as e:
                        logger.warning(f"Failed to compute root axis for tooth {tooth_idx}: {str(e)}")
                        angular_diff = np.nan
                        axis_line_dist = np.nan

                    chamfer_distances.append(cd)
                    emd_distances.append(emd)
                    hausdorff_distances.append(hausdorff)
                    centroid_distances.append(centroid_dist)
                    angular_differences.append(angular_diff)
                    axis_line_distances.append(axis_line_dist)

                    # Store F1 metrics for each threshold
                    for t in f1_thresholds:
                        f1_scores[t].append(f1_results[t]['f1_score'])
                        f1_precisions[t].append(f1_results[t]['precision'])
                        f1_recalls[t].append(f1_results[t]['recall'])

                    sample_info.append({
                        'sample_idx': sample_idx,
                        'patient_id': val_dentition_ids[0],
                        'tooth_idx': int(tooth_idx),
                        'chamfer_distance': float(cd),
                        'emd': float(emd),
                        'hausdorff_distance': float(hausdorff),
                        'centroid_distance': float(centroid_dist),
                        'angular_difference_deg': float(angular_diff) if not np.isnan(angular_diff) else None,
                        'axis_line_distance': float(axis_line_dist) if not np.isnan(axis_line_dist) else None,
                        'f1_0.005': float(f1_results[0.005]['f1_score']),
                        'f1_0.01': float(f1_results[0.01]['f1_score']),
                        'f1_0.025': float(f1_results[0.025]['f1_score']),
                        'f1_0.05': float(f1_results[0.05]['f1_score']),
                        'precision_0.025': float(f1_results[0.025]['precision']),
                        'recall_0.025': float(f1_results[0.025]['recall'])
                    })

            # Save samples to disk
            if save_samples:
                patient_id = val_dentition_ids[0]
                sample_dir = os.path.join(outf_syn, f'sample_{sample_idx:03d}')
                os.makedirs(sample_dir, exist_ok=True)

                condition_points = val_dentition_points[0][obs_mask_val[0].view(-1).bool()]
                generated_points = sample_out[0][latent_mask_val[0].view(-1).bool()]
                ground_truth_points = val_dentition_points[0][latent_mask_val[0].view(-1).bool()]

                export_dentition_npy(f'{patient_id}_{missing_idx}_condition', sample_dir, condition_points.numpy())
                export_dentition_npy(f'{patient_id}_{missing_idx}_gen', sample_dir, generated_points.numpy())
                export_dentition_npy(f'{patient_id}_{missing_idx}_gt', sample_dir, ground_truth_points.numpy())

            # Clear GPU cache periodically
            if sample_idx % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {str(e)}")
            continue

    # Compute and log statistics
    logger.info('='*80)
    logger.info('VALIDATION RESULTS')
    logger.info('='*80)

    results = {
        'checkpoint_path': checkpoint_path,
        'checkpoint_epoch': epoch,
        'num_samples': len(sample_info),
        'n_missing_teeth': n_missing_teeth,
        'num_teeth_evaluated': len(chamfer_distances)
    }

    if compute_metrics and len(chamfer_distances) > 0:
        chamfer_distances = np.array(chamfer_distances)
        emd_distances = np.array(emd_distances)
        hausdorff_distances = np.array(hausdorff_distances)
        centroid_distances = np.array(centroid_distances)
        angular_differences = np.array(angular_differences)
        axis_line_distances = np.array(axis_line_distances)

        # Filter out NaN values for axis metrics
        valid_angular = angular_differences[~np.isnan(angular_differences)]
        valid_axis_dist = axis_line_distances[~np.isnan(axis_line_distances)]

        logger.info(f'Number of samples evaluated: {len(set([s["sample_idx"] for s in sample_info]))}')
        logger.info(f'Number of teeth evaluated: {len(chamfer_distances)}')

        logger.info(f'\n{"="*80}')
        logger.info(f'METRIC STATISTICS')
        logger.info(f'{"="*80}')

        logger.info(f'\nChamfer Distance (Average point-to-surface distance):')
        logger.info(f'  Mean:   {chamfer_distances.mean():.6f}')
        logger.info(f'  Std:    {chamfer_distances.std():.6f}')
        logger.info(f'  Median: {np.median(chamfer_distances):.6f}')
        logger.info(f'  Min:    {chamfer_distances.min():.6f}')
        logger.info(f'  Max:    {chamfer_distances.max():.6f}')

        logger.info(f'\nEarth Mover\'s Distance (Optimal transport cost):')
        logger.info(f'  Mean:   {emd_distances.mean():.6f}')
        logger.info(f'  Std:    {emd_distances.std():.6f}')
        logger.info(f'  Median: {np.median(emd_distances):.6f}')
        logger.info(f'  Min:    {emd_distances.min():.6f}')
        logger.info(f'  Max:    {emd_distances.max():.6f}')

        logger.info(f'\nHausdorff Distance (Worst-case error):')
        logger.info(f'  Mean:   {hausdorff_distances.mean():.6f}')
        logger.info(f'  Std:    {hausdorff_distances.std():.6f}')
        logger.info(f'  Median: {np.median(hausdorff_distances):.6f}')
        logger.info(f'  Min:    {hausdorff_distances.min():.6f}')
        logger.info(f'  Max:    {hausdorff_distances.max():.6f}')

        logger.info(f'\nCentroid Distance (Positional accuracy):')
        logger.info(f'  Mean:   {centroid_distances.mean():.6f}')
        logger.info(f'  Std:    {centroid_distances.std():.6f}')
        logger.info(f'  Median: {np.median(centroid_distances):.6f}')
        logger.info(f'  Min:    {centroid_distances.min():.6f}')
        logger.info(f'  Max:    {centroid_distances.max():.6f}')

        if len(valid_angular) > 0:
            logger.info(f'\nRoot Axis Angular Difference (Orientation accuracy):')
            logger.info(f'  Mean:   {valid_angular.mean():.4f}°')
            logger.info(f'  Std:    {valid_angular.std():.4f}°')
            logger.info(f'  Median: {np.median(valid_angular):.4f}°')
            logger.info(f'  Min:    {valid_angular.min():.4f}°')
            logger.info(f'  Max:    {valid_angular.max():.4f}°')
            logger.info(f'  Valid samples: {len(valid_angular)}/{len(angular_differences)}')

        if len(valid_axis_dist) > 0:
            logger.info(f'\nRoot Axis Line Distance (Axis alignment):')
            logger.info(f'  Mean:   {valid_axis_dist.mean():.6f}')
            logger.info(f'  Std:    {valid_axis_dist.std():.6f}')
            logger.info(f'  Median: {np.median(valid_axis_dist):.6f}')
            logger.info(f'  Min:    {valid_axis_dist.min():.6f}')
            logger.info(f'  Max:    {valid_axis_dist.max():.6f}')
            logger.info(f'  Valid samples: {len(valid_axis_dist)}/{len(axis_line_distances)}')

        # Log F1-score metrics at each threshold
        logger.info(f'\nF1-Score (Geometric accuracy at distance thresholds):')
        logger.info(f'  (Scale: 1 normalized unit ≈ 39.2 mm)')
        for t in f1_thresholds:
            f1_arr = np.array(f1_scores[t])
            prec_arr = np.array(f1_precisions[t])
            rec_arr = np.array(f1_recalls[t])
            mm_equiv = t * 39.2
            logger.info(f'\n  Threshold τ = {t} (≈{mm_equiv:.1f}mm):')
            logger.info(f'    F1-Score:  Mean={f1_arr.mean():.4f}, Std={f1_arr.std():.4f}, Median={np.median(f1_arr):.4f}')
            logger.info(f'    Precision: Mean={prec_arr.mean():.4f}, Std={prec_arr.std():.4f}')
            logger.info(f'    Recall:    Mean={rec_arr.mean():.4f}, Std={rec_arr.std():.4f}')

        results['chamfer_distance'] = {
            'mean': float(chamfer_distances.mean()),
            'std': float(chamfer_distances.std()),
            'median': float(np.median(chamfer_distances)),
            'min': float(chamfer_distances.min()),
            'max': float(chamfer_distances.max())
        }

        results['emd'] = {
            'mean': float(emd_distances.mean()),
            'std': float(emd_distances.std()),
            'median': float(np.median(emd_distances)),
            'min': float(emd_distances.min()),
            'max': float(emd_distances.max())
        }

        results['hausdorff_distance'] = {
            'mean': float(hausdorff_distances.mean()),
            'std': float(hausdorff_distances.std()),
            'median': float(np.median(hausdorff_distances)),
            'min': float(hausdorff_distances.min()),
            'max': float(hausdorff_distances.max())
        }

        results['centroid_distance'] = {
            'mean': float(centroid_distances.mean()),
            'std': float(centroid_distances.std()),
            'median': float(np.median(centroid_distances)),
            'min': float(centroid_distances.min()),
            'max': float(centroid_distances.max())
        }

        if len(valid_angular) > 0:
            results['angular_difference'] = {
                'mean': float(valid_angular.mean()),
                'std': float(valid_angular.std()),
                'median': float(np.median(valid_angular)),
                'min': float(valid_angular.min()),
                'max': float(valid_angular.max()),
                'valid_samples': int(len(valid_angular)),
                'total_samples': int(len(angular_differences))
            }

        if len(valid_axis_dist) > 0:
            results['axis_line_distance'] = {
                'mean': float(valid_axis_dist.mean()),
                'std': float(valid_axis_dist.std()),
                'median': float(np.median(valid_axis_dist)),
                'min': float(valid_axis_dist.min()),
                'max': float(valid_axis_dist.max()),
                'valid_samples': int(len(valid_axis_dist)),
                'total_samples': int(len(axis_line_distances))
            }

        # Add F1-score results for each threshold
        results['f1_scores'] = {}
        for t in f1_thresholds:
            f1_arr = np.array(f1_scores[t])
            prec_arr = np.array(f1_precisions[t])
            rec_arr = np.array(f1_recalls[t])
            mm_equiv = t * 39.2
            results['f1_scores'][f'threshold_{t}_({mm_equiv:.1f}mm)'] = {
                'f1_mean': float(f1_arr.mean()),
                'f1_std': float(f1_arr.std()),
                'f1_median': float(np.median(f1_arr)),
                'precision_mean': float(prec_arr.mean()),
                'precision_std': float(prec_arr.std()),
                'recall_mean': float(rec_arr.mean()),
                'recall_std': float(rec_arr.std())
            }

    logger.info('='*80)

    if save_samples:
        logger.info(f'\nValidation samples saved to: {outf_syn}')

    # Save results to JSON
    results_file = os.path.join(output_dir, 'validation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to: {results_file}')

    # Save detailed sample info to CSV
    if compute_metrics and len(sample_info) > 0:
        import csv
        csv_file = os.path.join(output_dir, 'validation_details.csv')
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['sample_idx', 'patient_id', 'tooth_idx', 'chamfer_distance',
                         'emd', 'hausdorff_distance', 'centroid_distance',
                         'angular_difference_deg', 'axis_line_distance',
                         'f1_0.005', 'f1_0.01', 'f1_0.025', 'f1_0.05', 'precision_0.025', 'recall_0.025']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample_info)
        logger.info(f'Detailed results saved to: {csv_file}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate a trained PCDIFF_CBCT checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pth file (e.g., output/train_generation/.../epoch_2999.pth)')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to dataset directory containing val_patients.txt')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Custom output directory for validation results (default: auto-generated)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of validation samples to generate (default: 10)')
    parser.add_argument('--n_missing', type=int, default=1,
                        help='Number of teeth to reconstruct per sample (default: 1)')
    parser.add_argument('--no_metrics', action='store_true',
                        help='Skip Chamfer Distance computation (faster, only generates samples)')
    parser.add_argument('--no_save', action='store_true',
                        help='Skip saving point clouds to disk (only compute metrics)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    parser.add_argument('--fdi', type=int, nargs='+', default=None,
                        help='FDI tooth number(s) to reconstruct per sample. '
                             'e.g. --fdi 11  or  --fdi 11 21. '
                             'If omitted, teeth are chosen randomly.')

    args = parser.parse_args()

    validate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        n_missing_teeth=args.n_missing,
        compute_metrics=not args.no_metrics,
        save_samples=not args.no_save,
        target_fdi=args.fdi,
    )
