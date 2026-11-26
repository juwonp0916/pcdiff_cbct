import os
import sys
from pathlib import Path

# Allow direct invocation from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import argparse

# Set CUDA_VISIBLE_DEVICES before torch is imported.
# Check --gpu flag first; if not given, honour any pre-existing CUDA_VISIBLE_DEVICES
# (e.g. set by a cluster scheduler) so we don't accidentally override it.
def _parse_gpu_early():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None   # None → don't touch CUDA_VISIBLE_DEVICES

_gpu_flag = _parse_gpu_early()
if _gpu_flag is not None:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_flag
elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # safe default

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


# ── Metric functions (unchanged from validate.py) ────────────────────────────

def chamfer_distance_numpy(points1, points2):
    points1_t = torch.from_numpy(points1).float().unsqueeze(0)
    points2_t = torch.from_numpy(points2).float().unsqueeze(0)
    dist_matrix = torch.cdist(points1_t, points2_t)
    forward = dist_matrix.min(dim=2)[0].mean()
    backward = dist_matrix.min(dim=1)[0].mean()
    return (forward + backward).item() / 2.0


def compute_root_axis_pca(points):
    valid_mask = ~np.isnan(points).any(axis=1) & (np.abs(points).sum(axis=1) > 1e-6)
    valid_points = points[valid_mask]
    if len(valid_points) < 3:
        raise ValueError(f"Insufficient valid points ({len(valid_points)}) for PCA")
    centroid = np.mean(valid_points, axis=0)
    centered_points = valid_points - centroid
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    root_axis = pca.components_[0]
    if root_axis[2] < 0:
        root_axis = -root_axis
    return {
        'axis': root_axis,
        'centroid': centroid,
        'explained_variance': pca.explained_variance_ratio_[0]
    }


def compute_angular_difference(axis1, axis2, in_degrees=True):
    axis1_norm = axis1 / np.linalg.norm(axis1)
    axis2_norm = axis2 / np.linalg.norm(axis2)
    cos_angle = np.clip(np.dot(axis1_norm, axis2_norm), -1.0, 1.0)
    angle_rad = np.arccos(np.abs(cos_angle))
    if in_degrees:
        return np.degrees(angle_rad)
    return angle_rad


def compute_axis_line_distance(axis_info1, axis_info2):
    P1 = axis_info1['centroid']
    d1 = axis_info1['axis']
    P2 = axis_info2['centroid']
    d2 = axis_info2['axis']
    w = P1 - P2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w)
    e = np.dot(d2, w)
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        cross_product = np.cross(d1, w)
        distance = np.linalg.norm(cross_product) / np.linalg.norm(d1)
    else:
        t = (b * e - c * d) / denom
        s = (a * e - b * d) / denom
        closest1 = P1 + t * d1
        closest2 = P2 + s * d2
        distance = np.linalg.norm(closest1 - closest2)
    return distance


def centroid_distance(points1, points2):
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    return np.linalg.norm(centroid1 - centroid2)


def earth_movers_distance(points1, points2, max_iter=100):
    N, M = points1.shape[0], points2.shape[0]
    points1_t = torch.from_numpy(points1).float()
    points2_t = torch.from_numpy(points2).float()
    dist_matrix = torch.cdist(points1_t.unsqueeze(0), points2_t.unsqueeze(0)).squeeze(0)
    if N == M:
        matched_distances = []
        available_indices = set(range(M))
        for i in range(N):
            distances_to_available = dist_matrix[i, list(available_indices)]
            min_idx_in_available = torch.argmin(distances_to_available).item()
            actual_idx = list(available_indices)[min_idx_in_available]
            matched_distances.append(dist_matrix[i, actual_idx].item())
            available_indices.remove(actual_idx)
        return np.mean(matched_distances)
    else:
        if N < M:
            min_dists = dist_matrix.min(dim=1)[0]
        else:
            min_dists = dist_matrix.min(dim=0)[0]
        return min_dists.mean().item()


def hausdorff_distance(points1, points2):
    points1_t = torch.from_numpy(points1).float().unsqueeze(0)
    points2_t = torch.from_numpy(points2).float().unsqueeze(0)
    dist_matrix = torch.cdist(points1_t, points2_t).squeeze(0)
    forward = dist_matrix.min(dim=1)[0].max()
    backward = dist_matrix.min(dim=0)[0].max()
    return max(forward.item(), backward.item())


def f1_score_point_cloud(points1, points2, thresholds=[0.005, 0.01, 0.025, 0.05]):
    points1_t = torch.from_numpy(points1).float().unsqueeze(0)
    points2_t = torch.from_numpy(points2).float().unsqueeze(0)
    dist_matrix = torch.cdist(points1_t, points2_t).squeeze(0)
    min_dist_to_gt = dist_matrix.min(dim=1)[0]
    min_dist_to_gen = dist_matrix.min(dim=0)[0]
    results = {}
    for threshold in thresholds:
        precision = (min_dist_to_gt < threshold).float().mean().item()
        recall = (min_dist_to_gen < threshold).float().mean().item()
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        results[threshold] = {'precision': precision, 'recall': recall, 'f1_score': f1}
    return results


# ── Shared helpers ────────────────────────────────────────────────────────────

def _default_opt():
    """Fallback model options when options.txt is absent."""
    return argparse.Namespace(
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
        width_mult=0.75,
        use_axis_conditioning=False,  # must be present so Model builds correctly
    )


def _load_opt(checkpoint_path, logger):
    """Load options.txt next to checkpoint; fall back to defaults."""
    options_file = os.path.join(os.path.dirname(checkpoint_path), 'options.txt')
    if os.path.exists(options_file):
        logger.info(f'Loading options from: {options_file}')
        with open(options_file, 'r') as f:
            saved = json.load(f)
        opt = argparse.Namespace(**saved)
        # Ensure the key exists even for older checkpoints
        if not hasattr(opt, 'use_axis_conditioning'):
            opt.use_axis_conditioning = False
        return opt
    logger.warning('options.txt not found, using default parameters')
    return _default_opt()


def _load_model(checkpoint_path, opt, device, logger):
    """Load checkpoint weights into a freshly built Model.

    Auto-detects use_axis_conditioning from the checkpoint keys so the
    correct architecture is built even when options.txt is absent or
    located in a different directory from the checkpoint file.
    """
    logger.info(f'Loading checkpoint: {checkpoint_path}')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    ckpt = torch.load(checkpoint_path, map_location=device)
    epoch = ckpt.get('epoch', 'unknown')

    # ── Auto-detect architecture flags from checkpoint keys ───────────────────
    state_dict_key = 'model_state' if 'model_state' in ckpt else 'model_state_dict'
    raw_keys = list(ckpt[state_dict_key].keys())

    # Infer extra_feature_nc from the first SA voxel layer's input channels.
    # The first voxel conv weight has shape (out_ch, in_ch, kx, ky, kz) where
    # in_ch = nc + extra_feature_nc.  This handles checkpoints trained with
    # different feature configs (e.g. extra=10 vs extra=13 for per-point root_axis).
    try:
        voxel_key = next(k for k in raw_keys if 'sa_layers.0.0.voxel_layers.0.weight' in k)
        in_ch = ckpt[state_dict_key][voxel_key].shape[1]
        inferred_extra_nc = in_ch - opt.nc
        if inferred_extra_nc != opt.extra_feature_nc:
            logger.info(
                f'Auto-detected extra_feature_nc={inferred_extra_nc} from checkpoint '
                f'(options had {opt.extra_feature_nc}) — using checkpoint value to avoid shape mismatch'
            )
            opt.extra_feature_nc = inferred_extra_nc
        else:
            logger.info(f'extra_feature_nc={opt.extra_feature_nc} matches checkpoint')
    except StopIteration:
        logger.warning('Could not find SA voxel layer in checkpoint — using extra_feature_nc from options')

    logger.info(f'Checkpoint epoch: {epoch}  |  extra_feature_nc: {opt.extra_feature_nc}')

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type,
                  opt.model_var_type, opt.width_mult, opt.vox_res_mult)

    # Always strip 'module.' prefix (DDP checkpoints save as 'model.module.*';
    # remove_module_prefix is a no-op for non-DDP checkpoints).
    # Use strict=False to silently ignore stale keys (e.g. bound_embedding) that
    # were removed from the current model architecture.
    clean_sd = remove_module_prefix(ckpt[state_dict_key])
    load_result = model.load_state_dict(clean_sd, strict=False)

    if load_result.missing_keys:
        logger.warning(
            f'{len(load_result.missing_keys)} model keys had NO matching tensor in checkpoint '
            f'(those layers are at random init!): {load_result.missing_keys[:5]}'
        )
    if load_result.unexpected_keys:
        logger.info(
            f'{len(load_result.unexpected_keys)} stale checkpoint keys ignored (not in current model): '
            f'{load_result.unexpected_keys[:5]}'
        )
    if not load_result.missing_keys:
        logger.info('Model weights loaded successfully — all model parameters matched.')

    model = model.to(device)
    model.eval()
    return model, epoch


def _empty_metrics():
    f1_thresholds = [0.005, 0.01, 0.025, 0.05]
    return ([], [], [], [], [], [],
            f1_thresholds,
            {t: [] for t in f1_thresholds},
            {t: [] for t in f1_thresholds},
            {t: [] for t in f1_thresholds})


def _compute_and_store_metrics(gen_tooth, gt_tooth, f1_thresholds,
                                chamfer_distances, emd_distances,
                                hausdorff_distances, centroid_distances,
                                angular_differences, axis_line_distances,
                                f1_scores, f1_precisions, f1_recalls,
                                logger, label=''):
    cd        = chamfer_distance_numpy(gen_tooth, gt_tooth)
    emd       = earth_movers_distance(gen_tooth, gt_tooth)
    hausdorff = hausdorff_distance(gen_tooth, gt_tooth)
    c_dist    = centroid_distance(gen_tooth, gt_tooth)
    f1_results = f1_score_point_cloud(gen_tooth, gt_tooth, thresholds=f1_thresholds)

    try:
        gt_axis_info  = compute_root_axis_pca(gt_tooth)
        gen_axis_info = compute_root_axis_pca(gen_tooth)
        ang_diff      = compute_angular_difference(gt_axis_info['axis'], gen_axis_info['axis'])
        ax_line_dist  = compute_axis_line_distance(gt_axis_info, gen_axis_info)
    except Exception as e:
        if logger:
            logger.warning(f'PCA axis failed {label}: {e}')
        ang_diff     = np.nan
        ax_line_dist = np.nan

    chamfer_distances.append(cd)
    emd_distances.append(emd)
    hausdorff_distances.append(hausdorff)
    centroid_distances.append(c_dist)
    angular_differences.append(ang_diff)
    axis_line_distances.append(ax_line_dist)
    for t in f1_thresholds:
        f1_scores[t].append(f1_results[t]['f1_score'])
        f1_precisions[t].append(f1_results[t]['precision'])
        f1_recalls[t].append(f1_results[t]['recall'])

    return cd, emd, hausdorff, c_dist, ang_diff, ax_line_dist, f1_results


def _aggregate_and_log(results, logger,
                        chamfer_distances, emd_distances,
                        hausdorff_distances, centroid_distances,
                        angular_differences, axis_line_distances,
                        f1_thresholds, f1_scores, f1_precisions, f1_recalls):
    cd_arr  = np.array(chamfer_distances)
    emd_arr = np.array(emd_distances)
    hd_arr  = np.array(hausdorff_distances)
    cen_arr = np.array(centroid_distances)
    ang_arr = np.array(angular_differences)
    axd_arr = np.array(axis_line_distances)
    valid_ang = ang_arr[~np.isnan(ang_arr)]
    valid_axd = axd_arr[~np.isnan(axd_arr)]

    def _stats(arr):
        return {'mean': float(arr.mean()), 'std': float(arr.std()),
                'median': float(np.median(arr)),
                'min': float(arr.min()), 'max': float(arr.max())}

    logger.info(f'Chamfer Distance:  mean={cd_arr.mean():.6f}  std={cd_arr.std():.6f}  median={np.median(cd_arr):.6f}')
    logger.info(f'EMD:               mean={emd_arr.mean():.6f}  std={emd_arr.std():.6f}')
    logger.info(f'Hausdorff:         mean={hd_arr.mean():.6f}  std={hd_arr.std():.6f}')
    logger.info(f'Centroid dist:     mean={cen_arr.mean():.6f}  std={cen_arr.std():.6f}')
    if len(valid_ang) > 0:
        logger.info(f'Angular diff:      mean={valid_ang.mean():.2f}°  std={valid_ang.std():.2f}°  ({len(valid_ang)}/{len(ang_arr)} valid)')
    if len(valid_axd) > 0:
        logger.info(f'Axis line dist:    mean={valid_axd.mean():.6f}  std={valid_axd.std():.6f}')
    logger.info('F1 scores:')
    for t in f1_thresholds:
        arr = np.array(f1_scores[t])
        logger.info(f'  τ={t} (~{t*39.2:.1f}mm): F1={arr.mean():.4f} ± {arr.std():.4f}')

    results['chamfer_distance']   = _stats(cd_arr)
    results['emd']                = _stats(emd_arr)
    results['hausdorff_distance'] = _stats(hd_arr)
    results['centroid_distance']  = _stats(cen_arr)
    if len(valid_ang) > 0:
        results['angular_difference'] = {**_stats(valid_ang),
            'valid': int(len(valid_ang)), 'total': int(len(ang_arr))}
    if len(valid_axd) > 0:
        results['axis_line_distance'] = {**_stats(valid_axd),
            'valid': int(len(valid_axd)), 'total': int(len(axd_arr))}
    results['f1_scores'] = {}
    for t in f1_thresholds:
        arr  = np.array(f1_scores[t])
        parr = np.array(f1_precisions[t])
        rarr = np.array(f1_recalls[t])
        results['f1_scores'][f'threshold_{t}_({t*39.2:.1f}mm)'] = {
            'f1_mean': float(arr.mean()), 'f1_std': float(arr.std()),
            'f1_median': float(np.median(arr)),
            'precision_mean': float(parr.mean()),
            'recall_mean':    float(rarr.mean()),
        }


# ── Validation functions ──────────────────────────────────────────────────────

def validate_checkpoint(checkpoint_path, data_path, output_dir=None,
                        num_samples=10, n_missing_teeth=1,
                        compute_metrics=True, save_samples=True,
                        target_fdi=None):
    """
    Random patient, random (or targeted) tooth validation with root axis conditioning.

    Mirrors validate.py:validate_checkpoint but passes root_axis from the
    dataset so the conditioned model receives the axis embedding it was trained with.
    """
    fdi_to_pos = {fdi: pos for pos, fdi in enumerate(FDIS)}

    if target_fdi is not None:
        invalid_fdis = [f for f in target_fdi if f not in fdi_to_pos]
        if invalid_fdis:
            raise ValueError(
                f"FDI number(s) {invalid_fdis} not in supported list.\n"
                f"Supported: {sorted(fdi_to_pos.keys())}"
            )

    output_dir = _make_output_dir('validation_cond', num_samples, 'samples', target_fdi, output_dir)

    logger = setup_logging(output_dir)
    if save_samples:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}  (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "not set")})')

    opt   = _load_opt(checkpoint_path, logger)
    model, epoch = _load_model(checkpoint_path, opt, device, logger)

    logger.info(f'Loading validation dataset from: {data_path}')
    val_dataset = get_dataset(path=data_path, tooth_npoints=opt.tooth_npoints, mode='val')
    logger.info(f'Validation dataset size: {len(val_dataset)}')

    (chamfer_distances, emd_distances, hausdorff_distances,
     centroid_distances, angular_differences, axis_line_distances,
     f1_thresholds, f1_scores, f1_precisions, f1_recalls) = _empty_metrics()
    sample_info = []

    logger.info(f'Generating {num_samples} samples with {n_missing_teeth} tooth/sample...')
    logger.info('='*80)

    for sample_idx in tqdm(range(num_samples), desc='Validating'):
        try:
            val_sample_list = val_dataset.sample_patient(1)

            val_dentition_points = torch.stack(
                [s['dentition_points'] for s in val_sample_list], 0).to(device)
            val_fdi_indices = torch.stack(
                [s['fdi_indices'] for s in val_sample_list], 0).to(device)
            val_dentition_ids = [s['patient_id'] for s in val_sample_list]

            val_bound = None
            if 'bounds_cyl' in val_sample_list[0]:
                val_bound = torch.stack(
                    [s['bounds_cyl'] for s in val_sample_list], 0).to(device)

            val_original_missing_mask = None
            if 'original_missing_mask' in val_sample_list[0]:
                val_original_missing_mask = torch.stack(
                    [s['original_missing_mask'] for s in val_sample_list], 0).to(device)

            # Root axis conditioning (the key addition vs validate.py)
            val_root_axis = None
            if 'root_axis' in val_sample_list[0]:
                val_root_axis = torch.stack(
                    [s['root_axis'] for s in val_sample_list], 0).to(device)  # (1, 28, 3)

            latent_mask_val = torch.zeros_like(val_dentition_points[:, :, :1, :1]).to(device)

            if val_original_missing_mask is not None:
                existing_teeth_indices = torch.where(~val_original_missing_mask[0])[0]
            else:
                existing_teeth_indices = torch.arange(val_dentition_points.shape[1], device=device)

            if len(existing_teeth_indices) == 0:
                logger.warning(f'Sample {sample_idx}: No existing teeth, skipping')
                continue

            if target_fdi is not None:
                valid_positions = []
                for fdi_num in target_fdi:
                    pos = fdi_to_pos[fdi_num]
                    if val_original_missing_mask is not None and val_original_missing_mask[0, pos]:
                        logger.warning(f'Sample {sample_idx}: FDI {fdi_num} naturally missing, skipping tooth')
                        continue
                    valid_positions.append(pos)
                if len(valid_positions) == 0:
                    logger.warning(f'Sample {sample_idx}: All target FDI teeth naturally missing, skipping')
                    continue
                selected_existing = torch.tensor(valid_positions, device=device)
                logger.info(f'Sample {sample_idx}: Targeting FDI {[FDIS[p] for p in valid_positions]}')
            else:
                n_missing_val = n_missing_teeth
                if len(existing_teeth_indices) < n_missing_val:
                    logger.warning(f'Sample {sample_idx}: Not enough teeth, skipping')
                    continue
                non_wisdom = [
                    idx.item() for idx in existing_teeth_indices
                    if (FDIS[idx] if idx < len(FDIS) else 0) % 10 != 8
                ]
                if len(non_wisdom) >= n_missing_val:
                    perm = torch.randperm(len(non_wisdom), device=device)[:n_missing_val]
                    selected_existing = torch.tensor(non_wisdom, device=device)[perm]
                else:
                    perm = torch.randperm(len(existing_teeth_indices), device=device)[:n_missing_val]
                    selected_existing = existing_teeth_indices[perm]

            latent_mask_val[0, selected_existing, 0, 0] = 1
            obs_mask_val = torch.ones_like(latent_mask_val) - latent_mask_val
            if val_original_missing_mask is not None:
                obs_mask_val = obs_mask_val * (~val_original_missing_mask.unsqueeze(-1).unsqueeze(-1))

            val_data_dict = {
                'x0':                    val_dentition_points,
                'fdi_indices':           val_fdi_indices,
                'l_mask':                latent_mask_val,
                'o_mask':                obs_mask_val,
                'bound':                 val_bound,
                'root_axis':             val_root_axis,       # ← conditioning
                'original_missing_mask': val_original_missing_mask,
            }

            with torch.no_grad():
                sample_out = model.gen_samples(model_kwargs=val_data_dict, use_ddim=False).detach().cpu()
                val_dentition_points = val_dentition_points.detach().cpu()
                obs_mask_val         = obs_mask_val.detach().cpu()
                latent_mask_val      = latent_mask_val.detach().cpu()

            missing_idx = np.where(latent_mask_val[0].squeeze().numpy())[0]

            if compute_metrics:
                for tooth_idx in missing_idx:
                    gen_tooth = sample_out[0, tooth_idx].numpy().T   # (N, 3)
                    gt_tooth  = val_dentition_points[0, tooth_idx].numpy().T

                    (cd, emd, hausdorff, c_dist, ang_diff, ax_line_dist, f1_results
                     ) = _compute_and_store_metrics(
                        gen_tooth, gt_tooth, f1_thresholds,
                        chamfer_distances, emd_distances,
                        hausdorff_distances, centroid_distances,
                        angular_differences, axis_line_distances,
                        f1_scores, f1_precisions, f1_recalls,
                        logger, label=f'tooth {tooth_idx}')

                    sample_info.append({
                        'sample_idx':             sample_idx,
                        'patient_id':             val_dentition_ids[0],
                        'tooth_idx':              int(tooth_idx),
                        'chamfer_distance':       float(cd),
                        'emd':                    float(emd),
                        'hausdorff_distance':     float(hausdorff),
                        'centroid_distance':      float(c_dist),
                        'angular_difference_deg': float(ang_diff) if not np.isnan(ang_diff) else None,
                        'axis_line_distance':     float(ax_line_dist) if not np.isnan(ax_line_dist) else None,
                        'f1_0.005':               float(f1_results[0.005]['f1_score']),
                        'f1_0.01':                float(f1_results[0.01]['f1_score']),
                        'f1_0.025':               float(f1_results[0.025]['f1_score']),
                        'f1_0.05':                float(f1_results[0.05]['f1_score']),
                        'precision_0.025':        float(f1_results[0.025]['precision']),
                        'recall_0.025':           float(f1_results[0.025]['recall']),
                    })

            if save_samples:
                patient_id = val_dentition_ids[0]
                sample_dir = os.path.join(outf_syn, f'sample_{sample_idx:03d}')
                os.makedirs(sample_dir, exist_ok=True)
                condition_points  = val_dentition_points[0][obs_mask_val[0].view(-1).bool()]
                generated_points  = sample_out[0][latent_mask_val[0].view(-1).bool()]
                ground_truth_points = val_dentition_points[0][latent_mask_val[0].view(-1).bool()]
                export_dentition_npy(f'{patient_id}_{missing_idx}_condition', sample_dir, condition_points.numpy())
                export_dentition_npy(f'{patient_id}_{missing_idx}_gen',       sample_dir, generated_points.numpy())
                export_dentition_npy(f'{patient_id}_{missing_idx}_gt',        sample_dir, ground_truth_points.numpy())

            if sample_idx % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            logger.error(f'Error processing sample {sample_idx}: {e}')
            logger.error(traceback.format_exc())
            continue

    logger.info('='*80)
    logger.info('VALIDATION RESULTS')
    logger.info('='*80)

    results = {
        'checkpoint_path':      checkpoint_path,
        'checkpoint_epoch':     epoch,
        'num_samples':          len(sample_info),
        'n_missing_teeth':      n_missing_teeth,
        'num_teeth_evaluated':  len(chamfer_distances),
        'use_axis_conditioning': opt.use_axis_conditioning,
    }

    if compute_metrics and len(chamfer_distances) > 0:
        _aggregate_and_log(results, logger,
                           chamfer_distances, emd_distances,
                           hausdorff_distances, centroid_distances,
                           angular_differences, axis_line_distances,
                           f1_thresholds, f1_scores, f1_precisions, f1_recalls)

    _save_results(output_dir, results, sample_info, compute_metrics, logger)
    return results


def validate_all_teeth(checkpoint_path, data_path, output_dir=None,
                       num_patients=None, compute_metrics=True, save_samples=True,
                       target_fdi=None):
    """
    Leave-one-out over every patient × every tooth, with root axis conditioning.

    If target_fdi is given (e.g. [17, 21]), only those FDI teeth are
    reconstructed per patient; all others are skipped.
    Saved point clouds are in normalized space (no denormalization).
    """
    n_label = num_patients if num_patients is not None else 'all'
    output_dir = _make_output_dir('validation_all_teeth_cond', n_label, 'patients', target_fdi, output_dir)

    logger = setup_logging(output_dir)
    if save_samples:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}  (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "not set")})')

    opt   = _load_opt(checkpoint_path, logger)
    model, epoch = _load_model(checkpoint_path, opt, device, logger)

    logger.info(f'Loading validation dataset from: {data_path}')
    val_dataset = get_dataset(path=data_path, tooth_npoints=opt.tooth_npoints, mode='val')
    n_patients = len(val_dataset)
    if num_patients is not None:
        n_patients = min(num_patients, n_patients)
    logger.info(f'Patients to process: {n_patients} / {len(val_dataset)}')

    (chamfer_distances, emd_distances, hausdorff_distances,
     centroid_distances, angular_differences, axis_line_distances,
     f1_thresholds, f1_scores, f1_precisions, f1_recalls) = _empty_metrics()
    sample_info = []
    total_teeth = 0

    logger.info('='*80)
    logger.info('Leave-one-out validation (all patients, all teeth)')
    logger.info('='*80)

    for patient_idx in tqdm(range(n_patients), desc='Patients'):
        try:
            sample = val_dataset[patient_idx]
        except Exception as e:
            logger.error(f'Failed to load patient {patient_idx}: {e}')
            continue

        patient_id            = sample['patient_id']
        original_missing_mask = sample['original_missing_mask']
        existing_teeth        = torch.where(~original_missing_mask)[0]
        n_existing            = len(existing_teeth)
        logger.info(f'\nPatient {patient_id}: {n_existing} teeth → {n_existing} reconstructions')

        dentition_points = sample['dentition_points'].unsqueeze(0).to(device)   # (1, 28, 3, N)
        fdi_indices      = sample['fdi_indices'].unsqueeze(0).to(device)
        orig_missing_dev = original_missing_mask.unsqueeze(0).to(device)

        # Root axis for the whole patient: (1, 28, 3)
        root_axis_patient = None
        if 'root_axis' in sample:
            root_axis_patient = sample['root_axis'].unsqueeze(0).to(device)

        for tooth_pos in existing_teeth:
            tooth_pos_item = tooth_pos.item()
            fdi_num        = FDIS[tooth_pos_item]

            # FDI filter: skip teeth not in the requested list
            if target_fdi is not None and fdi_num not in target_fdi:
                continue

            try:
                latent_mask = torch.zeros(1, 28, 1, 1, device=device)
                latent_mask[0, tooth_pos_item, 0, 0] = 1.0
                obs_mask = (1.0 - latent_mask) * (
                    (~orig_missing_dev).unsqueeze(-1).unsqueeze(-1).float())

                val_data_dict = {
                    'x0':                    dentition_points,
                    'fdi_indices':           fdi_indices,
                    'l_mask':                latent_mask,
                    'o_mask':                obs_mask,
                    'bound':                 None,
                    'root_axis':             root_axis_patient,   # ← conditioning
                    'original_missing_mask': orig_missing_dev,
                }

                with torch.no_grad():
                    sample_out = model.gen_samples(model_kwargs=val_data_dict, use_ddim=False).detach().cpu()

                gen_tooth = sample_out[0, tooth_pos_item].numpy().T
                gt_tooth  = sample['dentition_points'][tooth_pos_item].numpy().T
                total_teeth += 1

                if compute_metrics:
                    (cd, emd, hausdorff, c_dist, ang_diff, ax_line_dist, f1_results
                     ) = _compute_and_store_metrics(
                        gen_tooth, gt_tooth, f1_thresholds,
                        chamfer_distances, emd_distances,
                        hausdorff_distances, centroid_distances,
                        angular_differences, axis_line_distances,
                        f1_scores, f1_precisions, f1_recalls,
                        logger, label=f'FDI {fdi_num}')

                    ang_str = f'{ang_diff:.1f}°' if not np.isnan(ang_diff) else 'N/A'
                    logger.info(
                        f'  FDI {fdi_num:2d}: CD={cd:.4f}  HD={hausdorff:.4f}  '
                        f'centroid={c_dist:.4f}  angle={ang_str}  '
                        f'F1@1mm={f1_results[0.025]["f1_score"]:.3f}')

                    sample_info.append({
                        'patient_id':             patient_id,
                        'fdi':                    fdi_num,
                        'tooth_pos':              tooth_pos_item,
                        'chamfer_distance':       float(cd),
                        'emd':                    float(emd),
                        'hausdorff_distance':     float(hausdorff),
                        'centroid_distance':      float(c_dist),
                        'angular_difference_deg': float(ang_diff) if not np.isnan(ang_diff) else None,
                        'axis_line_distance':     float(ax_line_dist) if not np.isnan(ax_line_dist) else None,
                        'f1_0.005':               float(f1_results[0.005]['f1_score']),
                        'f1_0.01':                float(f1_results[0.01]['f1_score']),
                        'f1_0.025':               float(f1_results[0.025]['f1_score']),
                        'f1_0.05':                float(f1_results[0.05]['f1_score']),
                        'precision_0.025':        float(f1_results[0.025]['precision']),
                        'recall_0.025':           float(f1_results[0.025]['recall']),
                    })

                if save_samples:
                    patient_dir = os.path.join(outf_syn, patient_id)
                    os.makedirs(patient_dir, exist_ok=True)
                    latent_mask_cpu = latent_mask.cpu()
                    obs_mask_cpu    = obs_mask.cpu()
                    condition_pts = dentition_points[0].cpu()[obs_mask_cpu[0].view(-1).bool()]
                    generated_pts = sample_out[0][latent_mask_cpu[0].view(-1).bool()]
                    gt_pts        = sample['dentition_points'][latent_mask_cpu[0].view(-1).bool()]
                    export_dentition_npy(f'FDI_{fdi_num}_condition', patient_dir, condition_pts.numpy())
                    export_dentition_npy(f'FDI_{fdi_num}_gen',       patient_dir, generated_pts.numpy())
                    export_dentition_npy(f'FDI_{fdi_num}_gt',        patient_dir, gt_pts.numpy())

            except Exception as e:
                logger.error(f'  Error reconstructing FDI {fdi_num} for {patient_id}: {e}')
                continue

        torch.cuda.empty_cache()

    logger.info('='*80)
    logger.info(f'LEAVE-ONE-OUT COMPLETE  |  patients={n_patients}  teeth={total_teeth}')
    if target_fdi:
        logger.info(f'FDI filter applied: {sorted(target_fdi)}')
    logger.info('='*80)

    results = {
        'checkpoint_path':      checkpoint_path,
        'checkpoint_epoch':     epoch,
        'n_patients':           n_patients,
        'teeth_reconstructed':  total_teeth,
        'target_fdi':           sorted(target_fdi) if target_fdi else None,
        'use_axis_conditioning': opt.use_axis_conditioning,
    }
    if compute_metrics and len(chamfer_distances) > 0:
        _aggregate_and_log(results, logger,
                           chamfer_distances, emd_distances,
                           hausdorff_distances, centroid_distances,
                           angular_differences, axis_line_distances,
                           f1_thresholds, f1_scores, f1_precisions, f1_recalls)

    _save_results(output_dir, results, sample_info, compute_metrics, logger,
                  csv_fields=['patient_id', 'fdi', 'tooth_pos'])
    return results


def validate_sampled_patients_all_teeth(checkpoint_path, data_path, output_dir=None,
                                        num_patients=1, compute_metrics=True, save_samples=True,
                                        target_fdi=None):
    """
    Leave-one-out validation for randomly selected patients, with root axis conditioning.

    Randomly selects num_patients from the validation set. For each selected
    patient, every existing tooth is reconstructed one at a time using the
    remaining teeth as context.

    If target_fdi is given (e.g. [17, 21]), only those FDI teeth are
    reconstructed per patient; all others are skipped.
    Data stays in normalized space — no denormalization applied when saving.
    """
    output_dir = _make_output_dir('validation_sampled_cond', num_patients, 'patients', target_fdi, output_dir)

    logger = setup_logging(output_dir)
    if save_samples:
        outf_syn, = setup_output_subdirs(output_dir, 'syn')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}  (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "not set")})')

    opt   = _load_opt(checkpoint_path, logger)
    model, epoch = _load_model(checkpoint_path, opt, device, logger)

    logger.info(f'Loading validation dataset from: {data_path}')
    val_dataset = get_dataset(path=data_path, tooth_npoints=opt.tooth_npoints, mode='val')
    n_available = len(val_dataset)
    num_patients = min(num_patients, n_available)
    logger.info(f'Dataset size: {n_available}  |  Patients to sample: {num_patients}')

    patient_indices = torch.randperm(n_available)[:num_patients].tolist()
    logger.info(f'Selected patient indices: {patient_indices}')

    (chamfer_distances, emd_distances, hausdorff_distances,
     centroid_distances, angular_differences, axis_line_distances,
     f1_thresholds, f1_scores, f1_precisions, f1_recalls) = _empty_metrics()
    sample_info = []
    total_teeth = 0

    logger.info('='*80)
    logger.info('Sampled patient leave-one-out validation (with root axis conditioning)')
    logger.info('='*80)

    for patient_idx in patient_indices:
        try:
            sample = val_dataset[patient_idx]
        except Exception as e:
            logger.error(f'Failed to load patient index {patient_idx}: {e}')
            continue

        patient_id            = sample['patient_id']
        original_missing_mask = sample['original_missing_mask']
        existing_teeth        = torch.where(~original_missing_mask)[0]
        n_existing            = len(existing_teeth)
        logger.info(f'\nPatient {patient_id} (idx {patient_idx}): '
                    f'{n_existing} teeth → {n_existing} reconstructions')

        dentition_points = sample['dentition_points'].unsqueeze(0).to(device)
        fdi_indices      = sample['fdi_indices'].unsqueeze(0).to(device)
        orig_missing_dev = original_missing_mask.unsqueeze(0).to(device)

        # Root axis for the whole patient: (1, 28, 3)
        root_axis_patient = None
        if 'root_axis' in sample:
            root_axis_patient = sample['root_axis'].unsqueeze(0).to(device)

        for tooth_pos in existing_teeth:
            tooth_pos_item = tooth_pos.item()
            fdi_num        = FDIS[tooth_pos_item]

            # FDI filter: skip teeth not in the requested list
            if target_fdi is not None and fdi_num not in target_fdi:
                continue

            try:
                latent_mask = torch.zeros(1, 28, 1, 1, device=device)
                latent_mask[0, tooth_pos_item, 0, 0] = 1.0
                obs_mask = (1.0 - latent_mask) * (
                    (~orig_missing_dev).unsqueeze(-1).unsqueeze(-1).float())

                val_data_dict = {
                    'x0':                    dentition_points,
                    'fdi_indices':           fdi_indices,
                    'l_mask':                latent_mask,
                    'o_mask':                obs_mask,
                    'bound':                 None,
                    'root_axis':             root_axis_patient,   # ← conditioning
                    'original_missing_mask': orig_missing_dev,
                }

                with torch.no_grad():
                    sample_out = model.gen_samples(model_kwargs=val_data_dict, use_ddim=False).detach().cpu()

                gen_tooth = sample_out[0, tooth_pos_item].numpy().T
                gt_tooth  = sample['dentition_points'][tooth_pos_item].numpy().T
                total_teeth += 1

                if compute_metrics:
                    (cd, emd, hausdorff, c_dist, ang_diff, ax_line_dist, f1_results
                     ) = _compute_and_store_metrics(
                        gen_tooth, gt_tooth, f1_thresholds,
                        chamfer_distances, emd_distances,
                        hausdorff_distances, centroid_distances,
                        angular_differences, axis_line_distances,
                        f1_scores, f1_precisions, f1_recalls,
                        logger, label=f'FDI {fdi_num}')

                    ang_str = f'{ang_diff:.1f}°' if not np.isnan(ang_diff) else 'N/A'
                    logger.info(
                        f'  FDI {fdi_num:2d}: CD={cd:.4f}  HD={hausdorff:.4f}  '
                        f'centroid={c_dist:.4f}  angle={ang_str}  '
                        f'F1@1mm={f1_results[0.025]["f1_score"]:.3f}')

                    sample_info.append({
                        'patient_id':             patient_id,
                        'fdi':                    fdi_num,
                        'tooth_pos':              tooth_pos_item,
                        'chamfer_distance':       float(cd),
                        'emd':                    float(emd),
                        'hausdorff_distance':     float(hausdorff),
                        'centroid_distance':      float(c_dist),
                        'angular_difference_deg': float(ang_diff) if not np.isnan(ang_diff) else None,
                        'axis_line_distance':     float(ax_line_dist) if not np.isnan(ax_line_dist) else None,
                        'f1_0.005':               float(f1_results[0.005]['f1_score']),
                        'f1_0.01':                float(f1_results[0.01]['f1_score']),
                        'f1_0.025':               float(f1_results[0.025]['f1_score']),
                        'f1_0.05':                float(f1_results[0.05]['f1_score']),
                        'precision_0.025':        float(f1_results[0.025]['precision']),
                        'recall_0.025':           float(f1_results[0.025]['recall']),
                    })

                # Save in normalized space (no denormalization)
                if save_samples:
                    patient_dir = os.path.join(outf_syn, patient_id)
                    os.makedirs(patient_dir, exist_ok=True)
                    latent_mask_cpu = latent_mask.cpu()
                    obs_mask_cpu    = obs_mask.cpu()
                    condition_pts = dentition_points[0].cpu()[obs_mask_cpu[0].view(-1).bool()]
                    generated_pts = sample_out[0][latent_mask_cpu[0].view(-1).bool()]
                    gt_pts        = sample['dentition_points'][latent_mask_cpu[0].view(-1).bool()]
                    export_dentition_npy(f'FDI_{fdi_num}_condition', patient_dir, condition_pts.numpy())
                    export_dentition_npy(f'FDI_{fdi_num}_gen',       patient_dir, generated_pts.numpy())
                    export_dentition_npy(f'FDI_{fdi_num}_gt',        patient_dir, gt_pts.numpy())

            except Exception as e:
                logger.error(f'  Error reconstructing FDI {fdi_num} for {patient_id}: {e}')
                continue

        torch.cuda.empty_cache()

    logger.info('='*80)
    logger.info(f'SAMPLED PATIENT VALIDATION COMPLETE  |  patients={len(patient_indices)}  teeth={total_teeth}')
    if target_fdi:
        logger.info(f'FDI filter applied: {sorted(target_fdi)}')
    logger.info('='*80)

    results = {
        'checkpoint_path':      checkpoint_path,
        'checkpoint_epoch':     epoch,
        'n_patients':           len(patient_indices),
        'patient_indices':      patient_indices,
        'teeth_reconstructed':  total_teeth,
        'target_fdi':           sorted(target_fdi) if target_fdi else None,
        'use_axis_conditioning': opt.use_axis_conditioning,
    }
    if compute_metrics and len(chamfer_distances) > 0:
        _aggregate_and_log(results, logger,
                           chamfer_distances, emd_distances,
                           hausdorff_distances, centroid_distances,
                           angular_differences, axis_line_distances,
                           f1_thresholds, f1_scores, f1_precisions, f1_recalls)

    _save_results(output_dir, results, sample_info, compute_metrics, logger,
                  csv_fields=['patient_id', 'fdi', 'tooth_pos'])
    return results


# ── Output dir helper ─────────────────────────────────────────────────────────

def _make_output_dir(base_name, n_count, count_label, target_fdi, output_dir):
    """
    Build or create output directory.
    If output_dir is None, auto-generate a descriptive name that includes
    the sample/patient count and any FDI filter so different runs are
    distinguishable on disk.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    suffix = f'_n{n_count}{count_label}'
    if target_fdi:
        suffix += '_fdi' + '_'.join(str(f) for f in sorted(target_fdi))
    return get_output_dir(os.path.dirname(__file__), f'{base_name}{suffix}')


# ── I/O helper ────────────────────────────────────────────────────────────────

def _save_results(output_dir, results, sample_info, compute_metrics, logger,
                  csv_fields=None):
    results_file = os.path.join(output_dir, 'validation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to: {results_file}')

    if compute_metrics and len(sample_info) > 0:
        import csv
        base_fields = csv_fields or ['sample_idx', 'patient_id', 'tooth_idx']
        fieldnames = base_fields + [
            'chamfer_distance', 'emd', 'hausdorff_distance', 'centroid_distance',
            'angular_difference_deg', 'axis_line_distance',
            'f1_0.005', 'f1_0.01', 'f1_0.025', 'f1_0.05',
            'precision_0.025', 'recall_0.025',
        ]
        csv_file = os.path.join(output_dir, 'validation_details.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(sample_info)
        logger.info(f'Per-tooth details saved to: {csv_file}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate a PCDIFF_CBCT checkpoint trained with root axis conditioning')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pth file')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to dataset directory containing val_patients.txt')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='(default mode) Number of random samples (default: 10)')
    parser.add_argument('--n_missing', type=int, default=1,
                        help='(default mode) Teeth to reconstruct per sample (default: 1)')
    parser.add_argument('--no_metrics', action='store_true',
                        help='Skip metric computation')
    parser.add_argument('--no_save', action='store_true',
                        help='Skip saving point clouds')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--fdi', type=int, nargs='+', default=None,
                        help='FDI tooth number(s) to reconstruct. Works in all modes. '
                             'e.g. --fdi 17  or  --fdi 17 21. '
                             'In leave-one-out modes, only the specified teeth are '
                             'reconstructed (others are skipped). '
                             'Valid: 11-17, 21-27, 31-37, 41-47')
    parser.add_argument('--all_teeth', action='store_true',
                        help='Leave-one-out over ALL validation patients and teeth. '
                             'Combine with --fdi to restrict which teeth are reconstructed. '
                             'Use --num_patients to cap the number of patients processed.')
    parser.add_argument('--sampled_all_teeth', action='store_true',
                        help='Leave-one-out for randomly selected patients. '
                             'Combine with --fdi to restrict which teeth are reconstructed.')
    parser.add_argument('--num_patients', type=int, default=1,
                        help='Number of patients to process in --all_teeth or '
                             '--sampled_all_teeth modes (default: 1). '
                             'For --all_teeth, omit to process the full validation set.')

    args = parser.parse_args()

    if args.all_teeth:
        validate_all_teeth(
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_patients=args.num_patients,
            compute_metrics=not args.no_metrics,
            save_samples=not args.no_save,
            target_fdi=args.fdi,
        )
    elif args.sampled_all_teeth:
        validate_sampled_patients_all_teeth(
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_patients=args.num_patients,
            compute_metrics=not args.no_metrics,
            save_samples=not args.no_save,
            target_fdi=args.fdi,
        )
    else:
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
