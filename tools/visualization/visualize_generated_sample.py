"""
Visualize a normalized condition (context) and generated tooth directly from .npy files.

Both files are assumed to be in the model's normalized coordinate space.
They are denormalized here (× scale, default 39.2 mm) and Z-flipped before display.

Usage
-----
# Condition arch + generated tooth:
    python tools/visualization/visualize_generated_sample.py \\
        --condition path/to/condition.npy \\
        --generated path/to/generated.npy

# With optional ground-truth for comparison (in real mm, loaded from verts/):
    python tools/visualization/visualize_generated_sample.py \\
        --condition path/to/condition.npy \\
        --generated path/to/generated.npy \\
        --gt path/to/gt.npy

# Override denorm scale:
    python tools/visualization/visualize_generated_sample.py \\
        --condition path/to/condition.npy \\
        --generated path/to/generated.npy \\
        --scale 39.2

Key bindings
    → / ←  — cycle display modes
    S      — toggle solo (generated tooth only, hide context)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ── Visual settings ───────────────────────────────────────────────────────────

ELEV = 10
AZIM = -90
ZOOM      = 30.0   # full arch half-range in mm
ZOOM_SOLO = 12.0   # single tooth half-range in mm

CONTEXT_COLOR      = "#706666"
CONTEXT_ALPHA      = 0.50   # context-only mode
CONTEXT_ALPHA_BACK = 0.30   # fades context when generated/GT is overlaid
CONTEXT_SIZE       = 3

GT_COLOR  = '#2ecc71'   # green
GT_ALPHA  = 0.75
GT_SIZE   = 12

GEN_COLOR = "#e06e6e"   # red
GEN_ALPHA = 0.75
GEN_SIZE  = 12

DENORM_SCALE = 39.2     # mm — global normalization scale used during training

CONTEXT_MESH_K          = 5     # KNN neighbours per point for mesh lines
CONTEXT_MESH_LW         = 0.3   # line width
CONTEXT_MESH_ALPHA      = 0.18  # line alpha in context-only mode
CONTEXT_MESH_ALPHA_BACK = 0.08  # line alpha when target tooth is overlaid


# ── File helpers ──────────────────────────────────────────────────────────────

def load_npy(path):
    """Load a .npy point cloud, ensuring shape (N, 3)."""
    pts = np.load(path)
    if pts.ndim == 2 and pts.shape[1] == 3:
        return pts
    if pts.ndim == 2 and pts.shape[0] == 3:
        return pts.T   # (3, N) → (N, 3)
    raise ValueError(f'Unexpected point cloud shape {pts.shape} in {path}')


def denorm(pts, scale):
    """Denormalize and flip Z to convert from model space to display space."""
    out = pts * scale
    # out[:, 0] *= -1
    return out


# ── Viewer ────────────────────────────────────────────────────────────────────

_MODE_LABELS = {
    'context':        'Context only  (gap visible)',
    'context+gt':     'Context + Ground Truth',
    'context+gen':    'Context + Generated',
    'gt+gen':         'Ground Truth  vs  Generated',
    'context+gt+gen': 'Context + GT + Generated',
}


class GeneratedSampleViewer:
    """
    Interactive 3D viewer for a generated tooth alongside its condition arch.

    Key bindings
    ────────────
    →  /  ←   cycle display modes
    S         toggle solo (generated tooth only, hide arch context)
    """

    def __init__(self, context_pts, gt_pts, gen_pts, title=''):
        self.context_pts = context_pts
        self.gt_pts      = gt_pts
        self.gen_pts     = gen_pts
        self.title       = title
        self.solo        = False

        self.modes = ['context']
        if gt_pts is not None:
            self.modes.append('context+gt')
        if gen_pts is not None:
            self.modes.append('context+gen')
        if gt_pts is not None and gen_pts is not None:
            self.modes.append('gt+gen')
            self.modes.append('context+gt+gen')
        self.mode_idx = 0

        self.arch_centroid   = context_pts.mean(axis=0)
        target = gen_pts if gen_pts is not None else gt_pts
        self.target_centroid = target.mean(axis=0) if target is not None else self.arch_centroid

        # Precompute KNN mesh edges once so redraws stay fast
        self.context_segments = self._build_mesh_segments(context_pts)

        self.fig = plt.figure(figsize=(9, 8))
        self.fig.patch.set_facecolor('white')
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._draw()

    @staticmethod
    def _build_mesh_segments(pts, k=CONTEXT_MESH_K):
        """Return an (E, 2, 3) array of unique KNN edges for pts."""
        tree  = cKDTree(pts)
        _, idx = tree.query(pts, k=k + 1)   # (N, k+1) — first col is self
        N     = len(pts)
        i_idx = np.repeat(np.arange(N), k)
        j_idx = idx[:, 1:].ravel()
        mask  = i_idx < j_idx               # deduplicate: keep i < j only
        return np.stack([pts[i_idx[mask]], pts[j_idx[mask]]], axis=1)  # (E, 2, 3)

    def _setup_ax(self):
        cx, cy, cz = self.target_centroid if self.solo else self.arch_centroid
        zoom = ZOOM_SOLO if self.solo else ZOOM
        self.ax.set_xlim([cx - zoom, cx + zoom])
        self.ax.set_ylim([cy - zoom, cy + zoom])
        self.ax.set_zlim([cz - zoom, cz + zoom])
        self.ax.set_axis_off()
        self.ax.set_facecolor('white')
        self.ax.view_init(elev=ELEV, azim=AZIM)

    def _scatter(self, pts, color, size, alpha,
                 edge_color='#333333', edge_lw=0.2, depthshade=True):
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
            ctx_alpha  = CONTEXT_ALPHA_BACK      if target_visible else CONTEXT_ALPHA
            mesh_alpha = CONTEXT_MESH_ALPHA_BACK if target_visible else CONTEXT_MESH_ALPHA
            self._scatter(self.context_pts, CONTEXT_COLOR, CONTEXT_SIZE, ctx_alpha,
                          edge_color='#666666', edge_lw=0.15)
            lc = Line3DCollection(self.context_segments,
                                  linewidths=CONTEXT_MESH_LW,
                                  alpha=mesh_alpha,
                                  colors=CONTEXT_COLOR,
                                  zorder=2)
            self.ax.add_collection3d(lc)

        if 'gt' in mode and self.gt_pts is not None:
            self._scatter(self.gt_pts, GT_COLOR, GT_SIZE, GT_ALPHA, depthshade=False)

        if 'gen' in mode and self.gen_pts is not None:
            self._scatter(self.gen_pts, GEN_COLOR, GEN_SIZE, GEN_ALPHA, depthshade=False)

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
        nav_hint = '← → cycle mode  |  S = solo'
        label    = _MODE_LABELS.get(mode, mode)
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
        description='Visualize normalized condition + generated tooth .npy files'
    )
    parser.add_argument('--condition', type=str, required=True,
                        help='Path to condition (context arch) .npy file — normalized coordinates')
    parser.add_argument('--generated', type=str, required=True,
                        help='Path to generated tooth .npy file — normalized coordinates')
    parser.add_argument('--gt', type=str, default=None,
                        help='Optional ground-truth tooth .npy file for comparison '
                             '(normalized coordinates, same transform applied)')
    parser.add_argument('--scale', type=float, default=DENORM_SCALE,
                        help=f'Denormalization scale in mm (default: {DENORM_SCALE})')
    args = parser.parse_args()

    # Load and denormalize all inputs (× scale, then Z-flip)
    print(f'Denorm scale : ×{args.scale} mm  +  Z-flip')

    context_pts = load_npy(args.condition) * args.scale
    print(f'Condition    : {args.condition}  →  {context_pts.shape}  '
          f'x=[{context_pts[:,0].min():.1f}, {context_pts[:,0].max():.1f}]  '
          f'y=[{context_pts[:,1].min():.1f}, {context_pts[:,1].max():.1f}]  '
          f'z=[{context_pts[:,2].min():.1f}, {context_pts[:,2].max():.1f}] mm')

    gen_pts = denorm(load_npy(args.generated), args.scale)
    # gen_pts[:, 0] *= -1   # 180° rotation around Z: negate X and Y
    # gen_pts[:, 1] *= -1
    print(f'Generated    : {args.generated}  →  {gen_pts.shape}  '
          f'x=[{gen_pts[:,0].min():.1f}, {gen_pts[:,0].max():.1f}]  '
          f'y=[{gen_pts[:,1].min():.1f}, {gen_pts[:,1].max():.1f}]  '
          f'z=[{gen_pts[:,2].min():.1f}, {gen_pts[:,2].max():.1f}] mm')

    gt_pts = None
    if args.gt:
        gt_pts = denorm(load_npy(args.gt), args.scale)
        print(f'GT           : {args.gt}  →  {gt_pts.shape}  '
              f'x=[{gt_pts[:,0].min():.1f}, {gt_pts[:,0].max():.1f}]  '
              f'y=[{gt_pts[:,1].min():.1f}, {gt_pts[:,1].max():.1f}]  '
              f'z=[{gt_pts[:,2].min():.1f}, {gt_pts[:,2].max():.1f}] mm')

    import os
    cond_name = os.path.splitext(os.path.basename(args.condition))[0]
    gen_name  = os.path.splitext(os.path.basename(args.generated))[0]
    title = f'{cond_name}  —  {gen_name}'

    viewer = GeneratedSampleViewer(context_pts, gt_pts, gen_pts, title=title)
    plt.show()


if __name__ == '__main__':
    main()
