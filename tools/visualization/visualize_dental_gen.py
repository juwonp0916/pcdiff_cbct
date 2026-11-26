"""
Interactive viewer for dental context + generated/GT tooth comparison.

Usage:
    # Context arch with tooth 16 missing (gap only):
    python tools/visualization/visualize_dental_gen.py --patient_folder ./data/TADP0001U --target_fdi 16

    # Show only specific context teeth around the gap:
    python tools/visualization/visualize_dental_gen.py --patient_folder ./data/TADP0001U \\
        --context_fdi 14 15 17 24 25 26 27 --target_fdi 16

    # Context + generated tooth overlay:
    python tools/visualization/visualize_dental_gen.py --patient_folder ./data/TADP0001U \\
        --target_fdi 16 --generated path/to/gen.npy

    # Full comparison (context + GT + generated):
    python tools/visualization/visualize_dental_gen.py --patient_folder ./data/TADP0001U \\
        --target_fdi 16 --gt path/to/gt.npy --generated path/to/gen.npy

Key bindings:
    → / ←  — cycle display modes
    S      — toggle solo mode (target tooth only, no context)
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
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

GT_COLOR  = '#2ecc71'   # green
GT_ALPHA  = 0.75
GT_SIZE   = 12

GEN_COLOR = '#e43636'   # red
GEN_ALPHA = 0.75
GEN_SIZE  = 12


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


# ── Viewer ────────────────────────────────────────────────────────────────────

# Human-readable labels for each mode
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
    """

    def __init__(self, context_pts, gt_pts, gen_pts, title=''):
        """
        Parameters
        ----------
        context_pts : (N, 3)       all context teeth combined (required)
        gt_pts      : (N, 3)|None  ground-truth target tooth
        gen_pts     : (N, 3)|None  generated target tooth
        """
        self.context_pts = context_pts
        self.gt_pts      = gt_pts
        self.gen_pts     = gen_pts
        self.title       = title
        self.solo        = False

        # Build mode list from available data
        self.modes = ['context']
        if gt_pts is not None:
            self.modes.append('context+gt')
        if gen_pts is not None:
            self.modes.append('context+gen')
        if gt_pts is not None and gen_pts is not None:
            self.modes.append('gt+gen')
            self.modes.append('context+gt+gen')
        self.mode_idx = 0

        # Camera centroids
        self.arch_centroid = context_pts.mean(axis=0)
        target = gt_pts if gt_pts is not None else gen_pts
        self.target_centroid = target.mean(axis=0) if target is not None else self.arch_centroid

        self.fig = plt.figure(figsize=(9, 8))
        self.fig.patch.set_facecolor('white')
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._draw()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _setup_ax(self):
        cx, cy, cz = self.target_centroid if self.solo else self.arch_centroid
        zoom = ZOOM_SOLO if self.solo else ZOOM
        self.ax.set_xlim([cx - zoom, cx + zoom])
        self.ax.set_ylim([cy - zoom, cy + zoom])
        self.ax.set_zlim([cz - zoom, cz + zoom])
        self.ax.set_axis_off()
        self.ax.set_facecolor('white')
        self.ax.view_init(elev=ELEV, azim=AZIM)

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
        show_context = (not self.solo) and ('gt+gen' not in mode or 'context' in mode)
        target_visible = 'gt' in mode or 'gen' in mode

        # Context teeth — fade back when a target tooth is overlaid
        if show_context and 'context' in mode:
            ctx_alpha = CONTEXT_ALPHA_BACK if target_visible else CONTEXT_ALPHA
            self._scatter(self.context_pts, CONTEXT_COLOR, CONTEXT_SIZE, ctx_alpha,
                          edge_color='#666666', edge_lw=0.15)

        # Ground truth tooth — depthshade=False keeps it vivid regardless of depth
        if 'gt' in mode and self.gt_pts is not None:
            self._scatter(self.gt_pts, GT_COLOR, GT_SIZE, GT_ALPHA, depthshade=False)

        # Generated tooth — same
        if 'gen' in mode and self.gen_pts is not None:
            self._scatter(self.gen_pts, GEN_COLOR, GEN_SIZE, GEN_ALPHA, depthshade=False)

        # Legend
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

        # Title
        solo_tag  = '  [solo]' if self.solo else ''
        nav_hint  = '← → cycle mode  |  S = solo'
        label     = _MODE_LABELS.get(mode, mode)
        self.fig.suptitle(
            f'{self.title}{solo_tag}\n{label}   ({nav_hint})',
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize dental context arch with optional generated/GT tooth overlay'
    )
    parser.add_argument('--patient_folder', type=str, required=True,
                        help='Patient folder containing a verts/ subdirectory')
    parser.add_argument('--context_fdi', type=int, nargs='+', default=None,
                        help='FDI numbers to display as context teeth. '
                             'Default: all teeth in verts/ except --target_fdi')
    parser.add_argument('--target_fdi', type=int, nargs='+', default=None,
                        help='FDI number(s) of the missing/target tooth — '
                             'excluded from context to show the gap. '
                             'If a single FDI is given and --gt is omitted, '
                             'the GT is auto-loaded from verts/.')
    parser.add_argument('--generated', type=str, default=None,
                        help='Path to generated tooth .npy file  (N, 3)')
    parser.add_argument('--gt', type=str, default=None,
                        help='Path to ground-truth tooth .npy file  (N, 3). '
                             'Auto-loaded from verts/ when --target_fdi is a single tooth.')
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

    # Determine which FDIs to show as context
    if args.context_fdi:
        context_fdis = [f for f in args.context_fdi if f not in target_fdis]
    else:
        context_fdis = [f for f in all_fdis if f not in target_fdis]

    print(f'Patient    : {patient_id}')
    print(f'All FDIs   : {all_fdis}')
    print(f'Context    : {context_fdis}')
    print(f'Target gap : {target_fdis}')

    # Load context teeth
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

    # Load generated
    gen_pts = None
    if args.generated:
        gen_pts = load_npy(args.generated)
        print(f'Generated  : {args.generated}  shape={gen_pts.shape}')

    if gt_pts is None and gen_pts is None:
        print('Note: no GT or generated tooth provided — showing context gap only.')

    # Build title
    title_parts = [patient_id]
    if target_fdis:
        title_parts.append(f'target FDI {", ".join(str(f) for f in target_fdis)}')
    title = '  —  '.join(title_parts)

    viewer = DentalGenViewer(context_pts, gt_pts, gen_pts, title=title)
    plt.show()


if __name__ == '__main__':
    main()
