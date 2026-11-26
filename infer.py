import os
import sys
import argparse

# Parse --gpu early so CUDA_VISIBLE_DEVICES is set before torch initializes
def _parse_gpu_early():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return '0'

_gpu_id = _parse_gpu_early()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_id

import torch
import json
import numpy as np
from glob import glob
from tqdm import tqdm

from train_generation import Model, get_betas, remove_module_prefix
from datasets.dataset_generation import FDIS, fdi_to_embedding_idx
from utils.file_utils import setup_logging, get_output_dir


# Fixed normalization constants (must match ToothDataset.normalize_dentition)
NORM_SHIFT = np.array([-0.72, 25.35, -6.64])
NORM_SCALE = 39.2


def load_jaw(jaw_dir: str, tooth_npoints: int):
    """
    Load all available teeth from a jaw directory.

    Args:
        jaw_dir: Path to jaw directory, expected to contain a `verts/` subfolder
                 with files named `*_FDI_XX.npy`.
        tooth_npoints: Number of points to sample per tooth.

    Returns:
        available_fdis: set of FDI numbers found in the directory
        raw_teeth: dict {fdi: (N, 3) numpy array}
    """
    verts_dir = os.path.join(jaw_dir, 'verts')
    if not os.path.isdir(verts_dir):
        raise FileNotFoundError(f"Expected a `verts/` subfolder inside {jaw_dir}")

    raw_teeth = {}
    for vert_path in glob(os.path.join(verts_dir, '*.npy')):
        fname = os.path.basename(vert_path)
        try:
            fdi = int(fname.split('_FDI_')[-1].replace('.npy', ''))
        except ValueError:
            continue
        if fdi not in FDIS:
            continue
        pts = np.load(vert_path)  # (N, 3)
        # Random sub-sample to tooth_npoints
        idx = np.random.randint(0, pts.shape[0], tooth_npoints)
        raw_teeth[fdi] = pts[idx]

    return set(raw_teeth.keys()), raw_teeth


def build_dentition_tensor(raw_teeth: dict, target_fdis: list, tooth_npoints: int):
    """
    Build the normalized dentition tensor and masks for inference.

    Args:
        raw_teeth: dict {fdi: (tooth_npoints, 3)} raw (unnormalized) tooth point clouds
        target_fdis: list of FDI numbers to generate (these are treated as latent)
        tooth_npoints: points per tooth

    Returns:
        dentition_points: (28, 3, tooth_npoints) float32 tensor — normalized
        fdi_indices: (28,) int64 tensor — positional embedding indices
        latent_mask: (28, 1, 1) float32 tensor  — 1 for teeth to generate
        obs_mask: (28, 1, 1) float32 tensor     — 1 for context teeth
        original_missing_mask: (28,) bool tensor — True for naturally absent teeth
    """
    target_fdi_set = set(target_fdis)
    available_fdis = set(raw_teeth.keys())

    dentition_arr = []    # (28, tooth_npoints, 3) before normalization
    fdi_list = []
    l_mask_list = []
    o_mask_list = []
    orig_missing_list = []

    for fdi in FDIS:
        embed_idx = fdi_to_embedding_idx(fdi)
        fdi_list.append(embed_idx)

        if fdi in target_fdi_set:
            # Tooth to generate — fill with tiny noise (model will overwrite this)
            noise = np.random.normal(0.0, 0.05, size=(tooth_npoints, 3)).astype(np.float32)
            dentition_arr.append(noise)
            l_mask_list.append(1.0)
            o_mask_list.append(0.0)
            orig_missing_list.append(False)  # Not naturally absent; we chose to generate it

        elif fdi in available_fdis:
            # Context tooth — use real data
            dentition_arr.append(raw_teeth[fdi].astype(np.float32))
            l_mask_list.append(0.0)
            o_mask_list.append(1.0)
            orig_missing_list.append(False)

        else:
            # Naturally missing tooth — fill with tiny noise and mark absent
            noise = np.random.normal(0.0, 0.05, size=(tooth_npoints, 3)).astype(np.float32)
            dentition_arr.append(noise)
            l_mask_list.append(0.0)
            o_mask_list.append(0.0)
            orig_missing_list.append(True)

    dentition_arr = np.stack(dentition_arr, axis=0)  # (28, tooth_npoints, 3)

    # Normalize using fixed constants (same as ToothDataset.normalize_dentition)
    dentition_arr_norm = (dentition_arr - NORM_SHIFT) / NORM_SCALE

    # Replace naturally missing teeth with small noise in normalized space
    for i, missing in enumerate(orig_missing_list):
        if missing:
            dentition_arr_norm[i] = np.random.normal(0.0, 0.01, size=(tooth_npoints, 3)).astype(np.float32)

    # Build tensors
    # dentition_points: transpose (28, tooth_npoints, 3) -> (28, 3, tooth_npoints)
    dentition_points = torch.from_numpy(dentition_arr_norm).transpose(1, 2).float()

    fdi_indices = torch.tensor(fdi_list, dtype=torch.long)
    l_mask = torch.tensor(l_mask_list, dtype=torch.float32).view(28, 1, 1)
    o_mask = torch.tensor(o_mask_list, dtype=torch.float32).view(28, 1, 1)
    original_missing_mask = torch.tensor(orig_missing_list, dtype=torch.bool)

    return dentition_points, fdi_indices, l_mask, o_mask, original_missing_mask


def denormalize(pts_norm: np.ndarray) -> np.ndarray:
    """Reverse the fixed normalization back to mm-scale coordinates."""
    return pts_norm * NORM_SCALE + NORM_SHIFT


def run_inference(
    checkpoint_path: str,
    jaw_dir: str,
    target_fdis: list,
    output_dir: str = None,
    tooth_npoints: int = 1024,
    use_ddim: bool = True,
    ddim_steps: int = 100,
    ddim_eta: float = 0.0,
    save_normalized: bool = False,
):
    """
    Run inference: load a trained checkpoint and generate target teeth for a jaw.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        jaw_dir: Path to the jaw directory (must contain a `verts/` subfolder).
        target_fdis: FDI numbers of teeth to generate (e.g. [11, 21]).
        output_dir: Directory where results are saved. Auto-generated if None.
        tooth_npoints: Points per tooth (must match training config; default 1024).
        use_ddim: Use DDIM (faster) instead of full DDPM sampling.
        ddim_steps: Number of DDIM denoising steps (ignored if use_ddim=False).
        ddim_eta: DDIM eta parameter (0 = deterministic, 1 = stochastic).
        save_normalized: Also save point clouds in normalized (model) coordinates.
    """
    if output_dir is None:
        output_dir = get_output_dir(os.path.dirname(__file__), 'inference')
    else:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device} (physical GPU {_gpu_id})')

    # ---- Load checkpoint ----
    logger.info(f'Loading checkpoint: {checkpoint_path}')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    epoch = ckpt.get('epoch', 'unknown')
    logger.info(f'Checkpoint epoch: {epoch}')

    # ---- Reconstruct training options ----
    checkpoint_dir = os.path.dirname(checkpoint_path)
    options_file = os.path.join(checkpoint_dir, 'options.txt')

    if os.path.exists(options_file):
        logger.info(f'Loading options from: {options_file}')
        with open(options_file, 'r') as f:
            saved_opts = json.load(f)
        opt = argparse.Namespace(**saved_opts)
    else:
        logger.warning('options.txt not found — using default parameters')
        opt = argparse.Namespace(
            nc=3,
            extra_feature_nc=13,
            max_missing_teeth=6,
            tooth_npoints=tooth_npoints,
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
        )

    # ---- Build model ----
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type,
                  opt.model_var_type, opt.width_mult, opt.vox_res_mult)

    # Load weights
    state_dict_key = 'model_state' if 'model_state' in ckpt else 'model_state_dict'
    clean_sd = remove_module_prefix(ckpt[state_dict_key])

    # Auto-detect extra_feature_nc from checkpoint weights to avoid shape mismatch
    try:
        voxel_key = next(k for k in clean_sd if 'sa_layers.0.0.voxel_layers.0.weight' in k)
        in_ch = clean_sd[voxel_key].shape[1]
        inferred_extra_nc = in_ch - opt.nc
        if inferred_extra_nc != opt.extra_feature_nc:
            logger.info(
                f'Auto-detected extra_feature_nc={inferred_extra_nc} '
                f'(options had {opt.extra_feature_nc}) — rebuilding model'
            )
            opt.extra_feature_nc = inferred_extra_nc
            betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
            model = Model(opt, betas, opt.loss_type, opt.model_mean_type,
                          opt.model_var_type, opt.width_mult, opt.vox_res_mult)
    except StopIteration:
        logger.warning('Could not find SA voxel layer key — using extra_feature_nc from options')

    load_result = model.load_state_dict(clean_sd, strict=False)
    if load_result.missing_keys:
        logger.warning(f'{len(load_result.missing_keys)} model keys missing (random init): '
                       f'{load_result.missing_keys[:5]}')
    if load_result.unexpected_keys:
        logger.info(f'{len(load_result.unexpected_keys)} stale keys ignored: '
                    f'{load_result.unexpected_keys[:5]}')
    if not load_result.missing_keys:
        logger.info('All model weights loaded successfully.')

    model = model.to(device)
    model.eval()

    # ---- Load jaw data ----
    jaw_id = os.path.basename(os.path.normpath(jaw_dir))
    logger.info(f'Loading jaw: {jaw_id}  from  {jaw_dir}')

    available_fdis, raw_teeth = load_jaw(jaw_dir, tooth_npoints)
    logger.info(f'Available teeth (FDI): {sorted(available_fdis)}')
    logger.info(f'Target teeth to generate (FDI): {target_fdis}')

    # Validate target FDIs are in the FDIS list
    invalid = [f for f in target_fdis if f not in FDIS]
    if invalid:
        raise ValueError(f"FDI numbers {invalid} are not in the supported FDIS list: {FDIS}")

    # ---- Build input tensors ----
    dentition_points, fdi_indices, l_mask, o_mask, original_missing_mask = build_dentition_tensor(
        raw_teeth, target_fdis, tooth_npoints
    )

    # Add batch dimension: (1, 28, ...)
    x0 = dentition_points.unsqueeze(0).to(device)           # (1, 28, 3, 1024)
    fdi_indices_b = fdi_indices.unsqueeze(0).to(device)     # (1, 28)
    l_mask_b = l_mask.unsqueeze(0).to(device)               # (1, 28, 1, 1)
    o_mask_b = o_mask.unsqueeze(0).to(device)               # (1, 28, 1, 1)
    orig_missing_b = original_missing_mask.unsqueeze(0).to(device)  # (1, 28)

    model_kwargs = {
        'x0': x0,
        'fdi_indices': fdi_indices_b,
        'l_mask': l_mask_b,
        'o_mask': o_mask_b,
        'bound': None,
        'original_missing_mask': orig_missing_b,
    }

    n_context = int(o_mask.sum().item())
    n_generate = int(l_mask.sum().item())
    logger.info(f'Context teeth: {n_context}  |  Teeth to generate: {n_generate}')
    logger.info(f'Sampling method: DDPM (full {opt.time_num} steps)')

    # ---- Run inference ----
    logger.info('Running p_sample_loop / ddim_sample_loop ...')
    with torch.no_grad():
        generated = model.gen_samples(
            model_kwargs=model_kwargs,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        ).detach().cpu()  # (1, 28, 3, 1024)

    # ---- Save results ----
    missing_positions = torch.where(l_mask.squeeze(-1).squeeze(-1))[0].numpy()  # positions in FDIS list

    for pos in missing_positions:
        fdi = FDIS[pos]
        # generated[0, pos] is (3, 1024) → transpose to (1024, 3)
        gen_pts_norm = generated[0, pos].numpy().T  # (1024, 3) normalized
        gen_pts_mm = denormalize(gen_pts_norm)       # (1024, 3) in mm

        out_name_mm = f'{jaw_id}_FDI_{fdi:02d}_generated.npy'
        out_path_mm = os.path.join(output_dir, out_name_mm)
        np.save(out_path_mm, gen_pts_mm)
        logger.info(f'Saved (mm-scale):    {out_path_mm}')

        if save_normalized:
            out_name_norm = f'{jaw_id}_FDI_{fdi:02d}_generated_norm.npy'
            out_path_norm = os.path.join(output_dir, out_name_norm)
            np.save(out_path_norm, gen_pts_norm)
            logger.info(f'Saved (normalized):  {out_path_norm}')

    logger.info('Inference complete.')
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inference: generate missing teeth using a trained PCDIFF_CBCT checkpoint'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pth file')
    parser.add_argument('--jaw_dir', type=str, required=True,
                        help='Path to jaw directory containing a verts/ subfolder '
                             '(e.g. data/DBT1_0001U)')
    parser.add_argument('--target_fdi', type=int, nargs='+', required=True,
                        help='FDI number(s) of teeth to generate, e.g. --target_fdi 11 21')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for generated point clouds (auto-generated if omitted)')
    parser.add_argument('--tooth_npoints', type=int, default=1024,
                        help='Points per tooth (must match training; default 1024)')
    parser.add_argument('--ddim_steps', type=int, default=100,
                        help='(unused — DDPM sampling is always used)')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                        help='(unused — DDPM sampling is always used)')
    parser.add_argument('--save_normalized', action='store_true',
                        help='Also save outputs in normalized (model-space) coordinates')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID to use (default: 0)')

    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        jaw_dir=args.jaw_dir,
        target_fdis=args.target_fdi,
        output_dir=args.output_dir,
        tooth_npoints=args.tooth_npoints,
        use_ddim=False,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        save_normalized=args.save_normalized,
    )
