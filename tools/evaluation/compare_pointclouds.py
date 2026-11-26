#!/usr/bin/env python3
"""
compare_pointclouds.py

Load two .npy point clouds, center both at the origin, then compute all
evaluation metrics used in validate_conditioning.py.

─── Quick run (edit the variables below, then just `python tools/evaluation/compare_pointclouds.py`) ───
"""

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  EDIT THESE to run without command-line arguments                           │
# └─────────────────────────────────────────────────────────────────────────────┘
FILE1 = None
FILE2 = None
NO_CENTER   = False  # set True to skip centering
SCALE       = 39.2   # mm-equivalent scale (39.2 for this project); set None to omit
OUT         = None   # e.g. 'results.json' to save; None to skip
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import numpy as np
import torch
from sklearn.decomposition import PCA


# ── Metric functions (identical to validate_conditioning.py) ─────────────────

def chamfer_distance_numpy(points1, points2):
    points1_t = torch.from_numpy(points1).float().unsqueeze(0)
    points2_t = torch.from_numpy(points2).float().unsqueeze(0)
    dist_matrix = torch.cdist(points1_t, points2_t)
    forward  = dist_matrix.min(dim=2)[0].mean()
    backward = dist_matrix.min(dim=1)[0].mean()
    return (forward + backward).item() / 2.0


def earth_movers_distance(points1, points2):
    N, M = points1.shape[0], points2.shape[0]
    points1_t = torch.from_numpy(points1).float()
    points2_t = torch.from_numpy(points2).float()
    dist_matrix = torch.cdist(points1_t.unsqueeze(0), points2_t.unsqueeze(0)).squeeze(0)
    if N == M:
        matched_distances = []
        available_indices = set(range(M))
        for i in range(N):
            distances_to_available = dist_matrix[i, list(available_indices)]
            min_idx_in_available   = torch.argmin(distances_to_available).item()
            actual_idx = list(available_indices)[min_idx_in_available]
            matched_distances.append(dist_matrix[i, actual_idx].item())
            available_indices.remove(actual_idx)
        return float(np.mean(matched_distances))
    else:
        min_dists = dist_matrix.min(dim=1)[0] if N < M else dist_matrix.min(dim=0)[0]
        return min_dists.mean().item()


def hausdorff_distance(points1, points2):
    points1_t = torch.from_numpy(points1).float().unsqueeze(0)
    points2_t = torch.from_numpy(points2).float().unsqueeze(0)
    dist_matrix = torch.cdist(points1_t, points2_t).squeeze(0)
    forward  = dist_matrix.min(dim=1)[0].max()
    backward = dist_matrix.min(dim=0)[0].max()
    return max(forward.item(), backward.item())


def centroid_distance(points1, points2):
    return float(np.linalg.norm(np.mean(points1, axis=0) - np.mean(points2, axis=0)))


def f1_score_point_cloud(points1, points2, thresholds=(0.005, 0.01, 0.025, 0.05)):
    points1_t = torch.from_numpy(points1).float().unsqueeze(0)
    points2_t = torch.from_numpy(points2).float().unsqueeze(0)
    dist_matrix     = torch.cdist(points1_t, points2_t).squeeze(0)
    min_dist_to_gt  = dist_matrix.min(dim=1)[0]
    min_dist_to_gen = dist_matrix.min(dim=0)[0]
    results = {}
    for t in thresholds:
        precision = (min_dist_to_gt  < t).float().mean().item()
        recall    = (min_dist_to_gen < t).float().mean().item()
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results[t] = {'precision': precision, 'recall': recall, 'f1_score': f1}
    return results


def compute_root_axis_pca(points):
    valid_mask   = ~np.isnan(points).any(axis=1) & (np.abs(points).sum(axis=1) > 1e-6)
    valid_points = points[valid_mask]
    if len(valid_points) < 3:
        raise ValueError(f'Insufficient valid points ({len(valid_points)}) for PCA')
    centroid        = np.mean(valid_points, axis=0)
    centered_points = valid_points - centroid
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    root_axis = pca.components_[0]
    if root_axis[2] < 0:
        root_axis = -root_axis
    return {'axis': root_axis, 'centroid': centroid,
            'explained_variance': pca.explained_variance_ratio_[0]}


def compute_angular_difference(axis1, axis2, in_degrees=True):
    axis1_norm = axis1 / np.linalg.norm(axis1)
    axis2_norm = axis2 / np.linalg.norm(axis2)
    cos_angle  = np.clip(np.dot(axis1_norm, axis2_norm), -1.0, 1.0)
    angle_rad  = np.arccos(np.abs(cos_angle))
    return float(np.degrees(angle_rad) if in_degrees else angle_rad)


def compute_axis_line_distance(ax1, ax2):
    P1, d1 = ax1['centroid'], ax1['axis']
    P2, d2 = ax2['centroid'], ax2['axis']
    w = P1 - P2
    a, b, c = np.dot(d1, d1), np.dot(d1, d2), np.dot(d2, d2)
    d, e    = np.dot(d1, w), np.dot(d2, w)
    denom   = a * c - b * b
    if abs(denom) < 1e-10:
        return float(np.linalg.norm(np.cross(d1, w)) / np.linalg.norm(d1))
    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    return float(np.linalg.norm((P1 + t * d1) - (P2 + s * d2)))


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_pointcloud(path):
    """
    Load a .npy file and return an (N, 3) float32 array.

    Handles formats written by export_dentition_npy (already (N, 3) after
    its internal reshape) as well as raw (3, N) single-tooth arrays.
    """
    arr = np.load(path)

    if arr.ndim == 3:
        # (T, 3, N) — e.g. raw dentition_points slice before export_dentition_npy
        arr = arr.transpose(0, 2, 1).reshape(-1, 3)
    elif arr.ndim == 2:
        if arr.shape[1] == 3:
            pass          # already (N, 3)
        elif arr.shape[0] == 3:
            arr = arr.T   # (3, N) → (N, 3)
        else:
            raise ValueError(f'Unexpected 2-D shape {arr.shape} in {path}')
    else:
        raise ValueError(f'Unexpected array ndim={arr.ndim} in {path}')

    return arr.astype(np.float32)


def center_pointcloud(points):
    """Subtract centroid; returns (centered_points, original_centroid)."""
    centroid = np.mean(points, axis=0)
    return points - centroid, centroid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare two .npy point clouds with optional centering and full metrics.')
    parser.add_argument('file1', nargs='?', default=None,
                        help='First point cloud .npy (treated as "generated"). '
                             'If omitted, FILE1 variable at top of script is used.')
    parser.add_argument('file2', nargs='?', default=None,
                        help='Second point cloud .npy (treated as "ground truth"). '
                             'If omitted, FILE2 variable at top of script is used.')
    parser.add_argument('--no_center', action='store_true',
                        help='Skip centering both clouds at the origin')
    parser.add_argument('--scale', type=float, default=None,
                        help='Scale factor for displaying mm equivalents '
                             '(e.g. 39.2 for this project\'s normalization). '
                             'Does NOT change computed values.')
    parser.add_argument('--out', type=str, default=None,
                        help='Optional path to save results as JSON')
    args = parser.parse_args()

    # Fall back to the hardcoded variables when CLI args are absent
    file1     = args.file1     or FILE1
    file2     = args.file2     or FILE2
    no_center = args.no_center or NO_CENTER
    scale     = args.scale     if args.scale is not None else SCALE
    out       = args.out       or OUT

    if not file1 or not file2:
        parser.error(
            'No input files specified. Either pass them as positional arguments '
            'or set FILE1 / FILE2 at the top of this script.'
        )

    # Rebind so the rest of the function is unchanged
    args.file1     = file1
    args.file2     = file2
    args.no_center = no_center
    args.scale     = scale
    args.out       = out

    # ── Load ──────────────────────────────────────────────────────────────────
    pts1 = load_pointcloud(args.file1)
    pts2 = load_pointcloud(args.file2)

    print(f'\nLoaded:')
    print(f'  file1 (gen): {args.file1}  →  shape {pts1.shape}')
    print(f'  file2 (gt):  {args.file2}  →  shape {pts2.shape}')

    # ── Center ────────────────────────────────────────────────────────────────
    if not args.no_center:
        pts1, c1 = center_pointcloud(pts1)
        pts2, c2 = center_pointcloud(pts2)
        print(f'\nCentered at origin:')
        print(f'  file1 centroid was: [{c1[0]:+.4f}, {c1[1]:+.4f}, {c1[2]:+.4f}]')
        print(f'  file2 centroid was: [{c2[0]:+.4f}, {c2[1]:+.4f}, {c2[2]:+.4f}]')
    else:
        print('\n(Centering skipped — using raw coordinates)')

    scale = args.scale
    def mm(v):
        return f'  ({v * scale:.4f} mm)' if scale else ''

    # ── Compute metrics ───────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('EVALUATION METRICS')
    print('=' * 65)

    cd  = chamfer_distance_numpy(pts1, pts2)
    emd = earth_movers_distance(pts1, pts2)
    hd  = hausdorff_distance(pts1, pts2)
    cen = centroid_distance(pts1, pts2)

    print(f'Chamfer Distance:    {cd:.6f}{mm(cd)}')
    print(f'EMD:                 {emd:.6f}{mm(emd)}')
    print(f'Hausdorff Distance:  {hd:.6f}{mm(hd)}')
    print(f'Centroid Distance:   {cen:.6f}{mm(cen)}'
          + ('  ← ~0 expected after centering' if not args.no_center else ''))

    # F1 scores
    f1_thresholds = [0.005, 0.01, 0.025, 0.05]
    f1_results = f1_score_point_cloud(pts1, pts2, thresholds=f1_thresholds)
    print('\nF1 Scores:')
    for t in f1_thresholds:
        r      = f1_results[t]
        mm_str = f'~{t * (scale or 1.0):.2f}{"mm" if scale else " (normalized)"}'
        print(f'  τ={t} ({mm_str}):  '
              f'F1={r["f1_score"]:.4f}  '
              f'precision={r["precision"]:.4f}  '
              f'recall={r["recall"]:.4f}')

    # PCA root axis
    print('\nPCA Root Axis:')
    ang_diff    = None
    axd         = None
    try:
        ax1 = compute_root_axis_pca(pts1)
        ax2 = compute_root_axis_pca(pts2)
        ang_diff = compute_angular_difference(ax1['axis'], ax2['axis'])
        axd      = compute_axis_line_distance(ax1, ax2)
        print(f'  file1 axis:         [{ax1["axis"][0]:+.4f}, {ax1["axis"][1]:+.4f}, {ax1["axis"][2]:+.4f}]'
              f'  (expl. var={ax1["explained_variance"]:.3f})')
        print(f'  file2 axis:         [{ax2["axis"][0]:+.4f}, {ax2["axis"][1]:+.4f}, {ax2["axis"][2]:+.4f}]'
              f'  (expl. var={ax2["explained_variance"]:.3f})')
        print(f'  Angular difference: {ang_diff:.2f}°')
        print(f'  Axis line distance: {axd:.6f}{mm(axd)}')
    except Exception as e:
        print(f'  PCA failed: {e}')

    print('=' * 65)

    # ── Optional JSON output ──────────────────────────────────────────────────
    if args.out:
        results = {
            'file1':              args.file1,
            'file2':              args.file2,
            'centered':           not args.no_center,
            'scale':              scale,
            'n_points_file1':     int(pts1.shape[0]),
            'n_points_file2':     int(pts2.shape[0]),
            'chamfer_distance':   cd,
            'emd':                emd,
            'hausdorff_distance': hd,
            'centroid_distance':  cen,
            'angular_difference_deg': ang_diff,
            'axis_line_distance': axd,
            'f1_scores': {
                f'threshold_{t}': {
                    'f1_score':  f1_results[t]['f1_score'],
                    'precision': f1_results[t]['precision'],
                    'recall':    f1_results[t]['recall'],
                } for t in f1_thresholds
            },
        }
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to: {args.out}')


if __name__ == '__main__':
    main()
