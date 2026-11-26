#!/usr/bin/env python3
"""
batch_compare.py

Walk through a directory of sample_XXX/ folders, pair _gen.npy with _gt.npy
in each, run four key metrics, and write a CSV summary.

─── Quick run (edit the variables below, or pass CLI args) ───
"""

SCALE = 39.2  # mm-equivalent scale (fixed for this project)

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

# ── Import metric + I/O helpers from compare_pointclouds.py ──────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

from compare_pointclouds import (
    load_pointcloud,
    center_pointcloud,
    chamfer_distance_numpy,
    hausdorff_distance,
    earth_movers_distance,
    compute_root_axis_pca,
    compute_angular_difference,
)

# ── CSV columns ───────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    'sample',
    'jaw_id',
    'tooth_id',
    'n_pts_gen',
    'n_pts_gt',
    'chamfer_distance',
    'hausdorff_distance',
    'emd',
    'angular_difference_deg',
]


def parse_ids_from_filename(filename):
    """
    Parse jaw_id and tooth_id from a filename like DBT2_3031L_[7]_gen.npy.

    Strategy: strip the trailing _gen.npy / _gt.npy suffix, then split on the
    last occurrence of '_[' to separate jaw_id from tooth_id.

    Returns (jaw_id, tooth_id) strings, or ('', '') on failure.
    """
    stem = re.sub(r'_(gen|gt)\.npy$', '', filename)
    # Find the last '_[' which starts the tooth bracket
    idx = stem.rfind('_[')
    if idx == -1:
        # Fall back: last '_'-delimited token is tooth_id
        parts = stem.rsplit('_', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return stem, ''
    jaw_id  = stem[:idx]
    tooth_id = stem[idx + 1:]   # keeps the '[7]' part (drop the leading '_')
    return jaw_id, tooth_id


def process_sample(sample_dir, no_center):
    """
    Find _gen.npy and _gt.npy inside sample_dir, compute metrics.

    Returns a dict with all CSV_COLUMNS values, or raises on hard errors.
    """
    gen_files = glob.glob(os.path.join(sample_dir, '*_gen.npy'))
    gt_files  = glob.glob(os.path.join(sample_dir, '*_gt.npy'))

    if not gen_files:
        raise FileNotFoundError(f'No *_gen.npy found in {sample_dir}')
    if not gt_files:
        raise FileNotFoundError(f'No *_gt.npy found in {sample_dir}')

    gen_path = gen_files[0]
    gt_path  = gt_files[0]

    jaw_id, tooth_id = parse_ids_from_filename(os.path.basename(gen_path))

    pts_gen = load_pointcloud(gen_path)
    pts_gt  = load_pointcloud(gt_path)

    if not no_center:
        pts_gen, _ = center_pointcloud(pts_gen)
        pts_gt,  _ = center_pointcloud(pts_gt)

    cd  = chamfer_distance_numpy(pts_gen, pts_gt)
    hd  = hausdorff_distance(pts_gen, pts_gt)
    emd = earth_movers_distance(pts_gen, pts_gt)

    ang_diff = float('nan')
    try:
        ax_gen = compute_root_axis_pca(pts_gen)
        ax_gt  = compute_root_axis_pca(pts_gt)
        ang_diff = compute_angular_difference(ax_gen['axis'], ax_gt['axis'])
    except Exception:
        pass  # leave as NaN

    return {
        'jaw_id':                 jaw_id,
        'tooth_id':               tooth_id,
        'n_pts_gen':              int(pts_gen.shape[0]),
        'n_pts_gt':               int(pts_gt.shape[0]),
        'chamfer_distance':       cd,
        'hausdorff_distance':     hd,
        'emd':                    emd,
        'angular_difference_deg': ang_diff,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Batch point-cloud comparison across sample_XXX/ subdirectories.')
    parser.add_argument('directory',
                        help='Root directory containing sample_000/, sample_001/, … subdirs.')
    parser.add_argument('--no_center', action='store_true',
                        help='Skip centering point clouds at the origin')
    parser.add_argument('--out', type=str, default=None,
                        help='Output CSV path (default: <directory>/batch_results.csv)')
    args = parser.parse_args()

    no_center = args.no_center
    out_path  = args.out or os.path.join(args.directory, 'batch_results.csv')

    # ── Discover sample_* subdirectories ─────────────────────────────────────
    pattern = os.path.join(args.directory, 'sample_*')
    candidates = sorted(
        (p for p in glob.glob(pattern) if os.path.isdir(p)),
        key=lambda p: int(re.search(r'sample_(\d+)', os.path.basename(p)).group(1))
                      if re.search(r'sample_(\d+)', os.path.basename(p)) else 0
    )

    if not candidates:
        print(f'[ERROR] No sample_* subdirectories found in: {args.directory}', file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(candidates)} sample directories in {args.directory}')
    print(f'Centering: {"disabled" if no_center else "enabled"}')
    print(f'Output CSV: {out_path}')
    print()

    # ── Process each sample ───────────────────────────────────────────────────
    rows = []
    n_ok = 0
    n_err = 0

    for sample_dir in candidates:
        sample_name = os.path.basename(sample_dir)
        try:
            result = process_sample(sample_dir, no_center)
            row = {'sample': sample_name}
            row.update(result)
            rows.append(row)
            n_ok += 1

            cd_mm = f'  ({result["chamfer_distance"] * SCALE:.4f} mm)'
            print(f'  {sample_name}  [{result["jaw_id"]}  {result["tooth_id"]}]'
                  f'  CD={result["chamfer_distance"]:.6f}{cd_mm}'
                  f'  ang={result["angular_difference_deg"]:.2f}°')
        except Exception as exc:
            print(f'  [WARN] {sample_name}: {exc}', file=sys.stderr)
            n_err += 1

    print(f'\nProcessed {n_ok} samples OK, {n_err} skipped.')

    if not rows:
        print('[ERROR] No results to write.', file=sys.stderr)
        sys.exit(1)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'CSV written to: {out_path}')

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('AGGREGATE SUMMARY')
    print('=' * 60)
    for metric in ('chamfer_distance', 'hausdorff_distance', 'emd', 'angular_difference_deg'):
        values = [r[metric] for r in rows if not np.isnan(r[metric])]
        if not values:
            print(f'  {metric}: no valid values')
            continue
        arr = np.array(values)
        mm_str = ''
        if metric != 'angular_difference_deg':
            mm_str = f'  mean={arr.mean() * SCALE:.4f} mm'
        unit = '°' if metric == 'angular_difference_deg' else ''
        print(f'  {metric}:')
        print(f'    mean={arr.mean():.6f}{unit}  std={arr.std():.6f}{unit}'
              f'  min={arr.min():.6f}{unit}  max={arr.max():.6f}{unit}{mm_str}')
    print('=' * 60)


if __name__ == '__main__':
    main()
