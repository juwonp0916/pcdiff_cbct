"""
Interactive viewer: context teeth from a patient folder + normalized generated tooth.

The generated .npy file is in the model's normalized coordinate space.
It is denormalized (× scale, default 39.2 mm) and Z-flipped before display.
Context teeth are loaded directly from verts/ in real mm — no transform applied.

Usage
-----
# Context arch + generated tooth (GT auto-loaded from verts/):
    python tools/visualization/visualize_gen_folder.py \\
        --patient_folder ./data/TADP0001U \\
        --target_fdi 16 \\
        --generated path/to/generated.npy

# Restrict context to specific teeth:
    python tools/visualization/visualize_gen_folder.py \\
        --patient_folder ./data/TADP0001U \\
        --context_fdi 14 15 17 24 25 26 27 \\
        --target_fdi 16 \\
        --generated path/to/generated.npy

# Override denorm scale:
    python tools/visualization/visualize_gen_folder.py \\
        --patient_folder ./data/TADP0001U \\
        --target_fdi 16 \\
        --generated path/to/generated.npy \\
        --scale 39.2

Key bindings:
    → / ←  — cycle display modes
    S      — toggle solo mode (target tooth only, no context)
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import matplotlib.patches as mpatches
from glob import glob


# ── Visual settings ───────────────────────────────────────────────────────────

ELEV = 10
AZIM = -90
ZOOM      = 30.0   # full arch half-range in mm
ZOOM_SOLO = 12.0   # single tooth half-range in mm

CONTEXT_COLOR      = '#aaaaaa'
CONTEXT_ALPHA      = 0.35   # when no target tooth is shown
CONTEXT_ALPHA_BACK = 0.12   # when target tooth is overlaid — fades context back
CONTEXT_SIZE       = 6

GT_COLOR     = '#2ecc71'   # green
GT_ALPHA     = 0.75
GT_SIZE      = 12
GT_SIZE_SOLO = 18          # larger in solo mode

GEN_COLOR     = "#0d82e2"  # red
GEN_ALPHA     = 0.75
GEN_SIZE      = 5
GEN_SIZE_SOLO = 2         # noticeably larger in solo so it isn't lost against GT

DENORM_SCALE = 10    # mm — global normalization scale used during training

AXIS_SCALE = 1.6        # root axis line half-length relative to tooth extent


# ── File helpers ──────────────────────────────────────────────────────────────

def find_file(directory, fdi, suffixes):
    for suffix in suffixes:
        matches = glob(os.path.join(directory, suffix.format(fdi=fdi)))
        if matches:
            return matches[0]
    return None


def get_all_fdis(verts_dir):
    fdis = []
    for f in glob(os.path.join(verts_dir, '*.npy')):
        m = re.search(r'FDI[_]?(\d+)', os.path.basename(f))
        if m:
            fdis.append(int(m.group(1)))
    return sorted(set(fdis))


def load_tooth_from_verts(verts_dir, fdi):
    path = find_file(verts_dir, fdi, [
        '*_FDI_{fdi}.npy', '*_FDI{fdi}.npy',
        'tooth_FDI_{fdi}.npy', 'tooth_FDI{fdi}.npy',
    ])
    return np.load(path) if path else None


def load_npy(path):
    """Load a .npy point cloud, ensuring shape (N, 3)."""
    pts = np.load(path)
    if pts.ndim == 2 and pts.shape[1] == 3:
        return pts
    if pts.ndim == 2 and pts.shape[0] == 3:
        return pts.T   # (3, N) → (N, 3)
    raise ValueError(f'Unexpected point cloud shape {pts.shape} in {path}')


# ── PCA helpers ───────────────────────────────────────────────────────────────

def compute_pca(pts):
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, eigvecs = np.linalg.eigh(np.cov(centered.T))
    return centroid, centered, eigvecs[:, -1]   # centroid, centered pts, long axis


def canonicalize_crown_up(axis):
    """Flip axis so its tip points toward the crown (+Y)."""
    return axis if axis[1] >= 0 else -axis


# ── Viewer ────────────────────────────────────────────────────────────────────

_MODE_LABELS = {
    'context':        'Context only  (gap visible)',
    'context+gt':     'Context + Ground Truth',
    'context+gen':    'Context + Generated',
    'gt+gen':         'Ground Truth  vs  Generated',
    'context+gt+gen': 'Context + GT + Generated',
}


class DentalGenViewer:
    """
    Interactive 3D viewer.

    Key bindings
    ────────────
    →  /  ←   cycle display modes
    S         toggle solo (hide context, zoom to target tooth)
    A         toggle root axis display on GT and generated tooth
    """

    def __init__(self, context_pts, gt_pts, gen_pts, title=''):
        self.context_pts = context_pts
        self.gt_pts      = gt_pts
        self.gen_pts     = gen_pts
        self.title       = title
        self.solo        = False
        self.show_axes   = False

        self.modes = ['context']
        if gt_pts is not None:
            self.modes.append('context+gt')
        if gen_pts is not None:
            self.modes.append('context+gen')
        if gt_pts is not None and gen_pts is not None:
            self.modes.append('gt+gen')
            self.modes.append('context+gt+gen')
        self.mode_idx = 0

        self.arch_centroid = context_pts.mean(axis=0)
        target = gt_pts if gt_pts is not None else gen_pts
        self.target_centroid = target.mean(axis=0) if target is not None else self.arch_centroid

        self.fig = plt.figure(figsize=(9, 8))
        self.fig.patch.set_facecolor('white')
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._draw()

    def _setup_ax(self):
        cx, cy, cz = self.target_centroid if self.solo else self.arch_centroid
        zoom = ZOOM_SOLO if self.solo else ZOOM
        self.ax.set_xlim([cx - zoom, cx + zoom])
        self.ax.set_ylim([cy - zoom, cy + zoom])
        self.ax.set_zlim([cz - zoom, cz + zoom])
        self.ax.set_axis_off()
        self.ax.set_facecolor('white')
        self.ax.view_init(elev=ELEV, azim=AZIM)

    def _draw_root_axis(self, pts, color):
        """Draw the PCA long axis (root axis) for a tooth point cloud."""
        centroid, centered, vec = compute_pca(pts)
        vec = canonicalize_crown_up(vec)
        proj   = centered @ vec
        extent = (proj.max() - proj.min()) * AXIS_SCALE * 0.5
        p0 = centroid - vec * extent
        p1 = centroid + vec * extent
        tip = extent * 0.15
        self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                     color=color, linewidth=2.0, linestyle=':', zorder=10)
        self.ax.quiver(*(p1 - vec * tip), *(vec * tip),
                       color=color, linewidth=2.0, arrow_length_ratio=0.99, zorder=11)
        self.ax.quiver(*(p0 + vec * tip), *(-vec * tip),
                       color=color, linewidth=2.0, arrow_length_ratio=0.99, zorder=11)
        self.ax.scatter(*centroid, c=color, s=30, zorder=12)

    def _scatter(self, pts, color, size, alpha, edge_color='#333333', edge_lw=0.2,
                 depthshade=True):
        self.ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=color, s=size, alpha=alpha,
            depthshade=depthshade,
            edgecolors=edge_color, linewidths=edge_lw,
            zorder=5,
        )

    def _draw(self):
        self.ax.cla()
        self._setup_ax()

        mode = self.modes[self.mode_idx]
        show_context   = (not self.solo) and ('gt+gen' not in mode or 'context' in mode)
        target_visible = 'gt' in mode or 'gen' in mode

        if show_context and 'context' in mode:
            ctx_alpha = CONTEXT_ALPHA_BACK if target_visible else CONTEXT_ALPHA
            self._scatter(self.context_pts, CONTEXT_COLOR, CONTEXT_SIZE, ctx_alpha,
                          edge_color='#666666', edge_lw=0.15)

        if 'gt' in mode and self.gt_pts is not None:
            gt_size = GT_SIZE_SOLO if self.solo else GT_SIZE
            self._scatter(self.gt_pts, GT_COLOR, gt_size, GT_ALPHA, depthshade=False)
            if self.show_axes:
                self._draw_root_axis(self.gt_pts, GT_COLOR)

        if 'gen' in mode and self.gen_pts is not None:
            gen_size = GEN_SIZE_SOLO if self.solo else GEN_SIZE
            self._scatter(self.gen_pts, GEN_COLOR, gen_size, GEN_ALPHA, depthshade=False)
            if self.show_axes:
                self._draw_root_axis(self.gen_pts, GEN_COLOR)

        legend_items = []
        if show_context and 'context' in mode:
            legend_items.append(mpatches.Patch(color=CONTEXT_COLOR, label='Context'))
        if 'gt' in mode:
            legend_items.append(mpatches.Patch(color=GT_COLOR,  label='Ground Truth'))
        if 'gen' in mode:
            legend_items.append(mpatches.Patch(color=GEN_COLOR, label='Generated'))
        if legend_items:
            self.ax.legend(handles=legend_items, loc='upper left',
                           fontsize=9, framealpha=0.85)

        solo_tag = '  [solo]' if self.solo else ''
        axes_tag = '  [axes]' if self.show_axes else ''
        nav_hint = '← → cycle mode  |  S = solo  |  A = root axes'
        label    = _MODE_LABELS.get(mode, mode)
        self.fig.suptitle(
            f'{self.title}{solo_tag}{axes_tag}\n{label}   ({nav_hint})',
            fontsize=9, y=0.97,
        )
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'right':
            self.mode_idx = (self.mode_idx + 1) % len(self.modes)
            self._draw()
        elif event.key == 'left':
            self.mode_idx = (self.mode_idx - 1) % len(self.modes)
            self._draw()
        elif event.key == 's':
            self.solo = not self.solo
            self._draw()
        elif event.key == 'a':
            self.show_axes = not self.show_axes
            self._draw()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize context teeth from a patient folder + normalized generated tooth'
    )
    parser.add_argument('--patient_folder', type=str, required=True,
                        help='Patient folder containing a verts/ subdirectory')
    parser.add_argument('--context_fdi', type=int, nargs='+', default=None,
                        help='FDI numbers to use as context. '
                             'Default: all teeth in verts/ except --target_fdi')
    parser.add_argument('--target_fdi', type=int, nargs='+', default=None,
                        help='FDI number(s) of the generated tooth — excluded from context. '
                             'GT is auto-loaded from verts/ when a single FDI is given.')
    parser.add_argument('--generated', type=str, required=True,
                        help='Path to the generated tooth .npy file (normalized coordinates)')
    parser.add_argument('--scale', type=float, default=DENORM_SCALE,
                        help=f'Denormalization scale in mm (default: {DENORM_SCALE})')
    parser.add_argument('--gt', type=str, default=None,
                        help='Path to ground-truth tooth .npy file (real mm, no transform). '
                             'Auto-loaded from verts/ when --target_fdi is a single tooth.')
    parser.add_argument('--rx', type=float, default=0.0,
                        help='Rotation around X axis in degrees (applied to generated tooth)')
    parser.add_argument('--ry', type=float, default=0.0,
                        help='Rotation around Y axis in degrees (applied to generated tooth)')
    parser.add_argument('--rz', type=float, default=0.0,
                        help='Rotation around Z axis in degrees (applied to generated tooth)')
    parser.add_argument('--spin', type=float, default=0.0,
                        help='Spin the generated tooth around its own long axis (degrees)')
    args = parser.parse_args()

    verts_dir  = os.path.join(args.patient_folder, 'verts')
    patient_id = os.path.basename(args.patient_folder)

    if not os.path.exists(verts_dir):
        print(f'ERROR: verts/ directory not found: {verts_dir}')
        return

    all_fdis    = get_all_fdis(verts_dir)
    target_fdis = args.target_fdi or []

    if not all_fdis:
        print('No .npy teeth found in verts/ directory.')
        return

    if args.context_fdi:
        context_fdis = [f for f in args.context_fdi if f not in target_fdis]
    else:
        context_fdis = [f for f in all_fdis if f not in target_fdis]

    print(f'Patient    : {patient_id}')
    print(f'All FDIs   : {all_fdis}')
    print(f'Context    : {context_fdis}')
    print(f'Target gap : {target_fdis}')
    print(f'Denorm ×   : {args.scale} mm  +  Z-flip')

    # Load context teeth (real mm, no transform)
    context_pts_list = []
    for fdi in context_fdis:
        pts = load_tooth_from_verts(verts_dir, fdi)
        if pts is not None and len(pts) > 0:
            context_pts_list.append(pts)
        else:
            print(f'  FDI {fdi}: not found in verts/, skipping')

    if not context_pts_list:
        print('ERROR: No context teeth could be loaded.')
        return

    context_pts = np.concatenate(context_pts_list, axis=0)

    # Load and denormalize generated tooth
    gen_pts = load_npy(args.generated) * args.scale
    if args.rx != 0.0 or args.ry != 0.0 or args.rz != 0.0:
        R = Rotation.from_euler('xyz', [args.rx, args.ry, args.rz], degrees=True)
        gen_pts = R.apply(gen_pts)
        print(f'Rotation   : rx={args.rx}°  ry={args.ry}°  rz={args.rz}°')

    if args.spin != 0.0:
        centroid  = gen_pts.mean(axis=0)
        centered  = gen_pts - centroid
        _, eigvecs = np.linalg.eigh(np.cov(centered.T))
        long_axis = eigvecs[:, -1]          # eigenvector with largest eigenvalue
        R = Rotation.from_rotvec(np.deg2rad(args.spin) * long_axis)
        gen_pts   = R.apply(centered) + centroid
        print(f'Spin       : {args.spin}° around long axis {np.round(long_axis, 3)}')
    print(f'Generated  : {args.generated}  shape={gen_pts.shape}  '
          f'x=[{gen_pts[:,0].min():.1f}, {gen_pts[:,0].max():.1f}]  '
          f'y=[{gen_pts[:,1].min():.1f}, {gen_pts[:,1].max():.1f}]  '
          f'z=[{gen_pts[:,2].min():.1f}, {gen_pts[:,2].max():.1f}] mm')

    # Load GT — explicit path or auto from verts/ for single target
    gt_pts = None
    if args.gt:
        gt_pts = load_npy(args.gt)
        print(f'GT         : {args.gt}  shape={gt_pts.shape}')
    elif len(target_fdis) == 1:
        pts = load_tooth_from_verts(verts_dir, target_fdis[0])
        if pts is not None:
            gt_pts = pts
            print(f'GT         : auto-loaded from verts/ for FDI {target_fdis[0]}  shape={gt_pts.shape}')
        else:
            print(f'GT         : FDI {target_fdis[0]} not in verts/ — skipping GT')

    if gt_pts is not None:
        offset = gt_pts.mean(axis=0) - gen_pts.mean(axis=0)
        gen_pts = gen_pts + offset
        print(f'Aligned    : centroid shift  dx={offset[0]:.2f}  dy={offset[1]:.2f}  dz={offset[2]:.2f} mm')

    title_parts = [patient_id]
    if target_fdis:
        title_parts.append(f'target FDI {", ".join(str(f) for f in target_fdis)}')
    title = '  —  '.join(title_parts)

    viewer = DentalGenViewer(context_pts, gt_pts, gen_pts, title=title)
    plt.show()


if __name__ == '__main__':
    main()
