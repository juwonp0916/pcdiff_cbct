"""
Visualize pre-computed root axes within the full dentition context.

Usage:
    python tools/visualization/visualize_root_axes.py --patient_folder ./data/TADP0001U --fdi 11
    python tools/visualization/visualize_root_axes.py --patient_folder ./data/TADP0001U --fdi 11 21
    python tools/visualization/visualize_root_axes.py --patient_folder ./data/TADP0001U --fdi 11 --solo
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


# ── Visual settings ───────────────────────────────────────────────────────────

ELEV = 10       # camera elevation in degrees
AZIM = -90      # camera azimuth in degrees  (-90 = front of dentition, labial view)
ZOOM      = 30.0   # view half-range in mm — full arch mode
ZOOM_SOLO = 10.0   # view half-range in mm — solo (single tooth) mode

CONDITION_COLOR = '#aaaaaa'
CONDITION_ALPHA = 0.35
CONDITION_SIZE  = 6

FOCAL_COLORS = ["#2f97b1", "#71bef1", '#2ecc71', '#f39c12']   # one per focal FDI
FOCAL_ALPHA  = 0.55
FOCAL_SIZE   = 10

AXIS_SCALE      = 3.0    # axis line half-length relative to tooth extent
CENTROID_SIZE   = 10
CENTROID_COLOR  = '#1a1a1a'

# PC axis colours: PC1 (root/long axis) = red, PC2 (width) = green, PC3 (depth) = blue
PC_COLORS = ["#000000", "#ff0000", "#ffee00"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_pca(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    C = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    return centroid, eigenvalues[idx], eigenvectors[:, idx]


def canonicalize_crown_up(axis):
    """Flip axis so its tip points toward the crown (+y in y-up scenes)."""
    return axis.copy() if axis[1] >= 0 else -axis


def find_file(directory, fdi, suffixes):
    """Search directory for a file matching any of the given glob suffixes."""
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


def load_tooth(verts_dir, fdi):
    path = find_file(verts_dir, fdi, [
        '*_FDI_{fdi}.npy', '*_FDI{fdi}.npy',
        'tooth_FDI_{fdi}.npy', 'tooth_FDI{fdi}.npy',
    ])
    return np.load(path) if path else None


def load_root_axis(root_axes_dir, fdi):
    path = find_file(root_axes_dir, fdi, [
        '*_FDI_{fdi}_root_axis.npy', '*_FDI{fdi}_root_axis.npy',
        'tooth_FDI_{fdi}_root_axis.npy', 'tooth_FDI{fdi}_root_axis.npy',
    ])
    if path is None:
        return None
    data = np.load(path, allow_pickle=True).item()
    return data if np.linalg.norm(data.get('axis', np.zeros(3))) > 0.1 else None


# ── Interactive viewer ────────────────────────────────────────────────────────

class DentitionViewer:
    """
    Shows full dentition context (gray) with focal teeth and their root axes.

    Key bindings:
        A  — cycle axis mode: none → root axis only → all 3 PCs
        S  — toggle solo mode (hide/show context teeth)
    """

    MODES  = ['none', 'root', 'all']
    LABELS = ['No Axes  (press A)', 'Root Axis  (press A)', 'All 3 PCs  (press A)']

    def __init__(self, condition_pts, focal_data, title='', solo=False):
        """
        condition_pts : (N, 3) array — all non-focal teeth combined
        focal_data    : list of dicts, each with keys:
                          'points'     : (N, 3)
                          'fdi'        : int
                          'color'      : str
                          'root_axis'  : dict with 'axis', 'centroid', 'eigenvalues'
                                         (may be None if not available)
        solo          : if True, hide context teeth and zoom to focal tooth only
        """
        self.condition_pts = condition_pts
        self.focal_data    = focal_data
        self.title         = title
        self.mode_idx      = 0
        self.solo          = solo

        # Scene centroid: centre of focal teeth only in solo mode, all points otherwise
        focal_pts = np.concatenate([d['points'] for d in focal_data], axis=0)
        if solo or len(condition_pts) == 0:
            self.scene_centroid = focal_pts.mean(axis=0)
        else:
            all_pts = np.concatenate([condition_pts, focal_pts], axis=0)
            self.scene_centroid = all_pts.mean(axis=0)

        self.fig = plt.figure(figsize=(8, 8))
        self.fig.patch.set_facecolor('white')
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._draw()

    def _setup_ax(self):
        cx, cy, cz = self.scene_centroid
        zoom = ZOOM_SOLO if self.solo else ZOOM
        self.ax.set_xlim([cx - zoom, cx + zoom])
        self.ax.set_ylim([cy - zoom, cy + zoom])
        self.ax.set_zlim([cz - zoom, cz + zoom])
        self.ax.set_axis_off()
        self.ax.set_facecolor('white')
        self.ax.view_init(elev=ELEV, azim=AZIM)

    def _draw(self):
        self.ax.cla()
        self._setup_ax()

        # Condition (context) teeth — hidden in solo mode
        if not self.solo and len(self.condition_pts) > 0:
            self.ax.scatter(
                self.condition_pts[:, 0],
                self.condition_pts[:, 1],
                self.condition_pts[:, 2],
                s=CONDITION_SIZE, c=CONDITION_COLOR,
                alpha=CONDITION_ALPHA, zorder=1,
                depthshade=True, edgecolors='#555555', linewidths=0.2,
            )

        mode = self.MODES[self.mode_idx]

        for d in self.focal_data:
            pts   = d['points']
            color = d['color']
            fdi   = d['fdi']
            ra    = d['root_axis']   # may be None

            # Focal tooth point cloud
            self.ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=FOCAL_SIZE, c=color, alpha=FOCAL_ALPHA, zorder=5,
                label=f'FDI {fdi}',
                depthshade=True, edgecolors='#1a1a1a', linewidths=0.3,
            )

            if mode == 'none' or ra is None:
                continue

            centroid, eigenvalues, eigenvectors = compute_pca(pts)

            self.ax.scatter(*centroid,
                            c=CENTROID_COLOR, s=CENTROID_SIZE, zorder=11)

            n_axes = 1 if mode == 'root' else 3

            for i in range(n_axes):
                vec = eigenvectors[:, i].copy()
                if i == 0:
                    vec = canonicalize_crown_up(vec)

                # Scale length by projection extent of tooth along this axis
                proj = (pts - centroid) @ vec
                extent = (proj.max() - proj.min()) * AXIS_SCALE * 0.5

                p0 = centroid - vec * extent
                p1 = centroid + vec * extent

                lw    = [2.0, 1.8, 1.5][i]
                acolor = PC_COLORS[i]
                # Dotted shaft
                self.ax.plot(
                    [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=acolor,
                    linewidth=lw, linestyle=':', zorder=10,
                )
                # Arrowhead cones at both ends — quiver with arrow_length_ratio≈1
                # so virtually the whole quiver is the cone (no visible duplicate shaft)
                tip = extent * 0.15
                self.ax.quiver(
                    *(p1 - vec * tip), *(vec * tip),
                    color=acolor, linewidth=lw, arrow_length_ratio=0.99, zorder=11,
                )
                self.ax.quiver(
                    *(p0 + vec * tip), *(-vec * tip),
                    color=acolor, linewidth=lw, arrow_length_ratio=0.99, zorder=11,
                )


        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels, loc='upper left',
                           fontsize=8, framealpha=0.8)

        solo_tag = '  [solo]' if self.solo else ''
        self.fig.suptitle(
            f'{self.title}{solo_tag}\n{self.LABELS[self.mode_idx]}  |  S = toggle solo',
            fontsize=9, y=0.97,
        )
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'a':
            self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
            self._draw()
        elif event.key == 's':
            self.solo = not self.solo
            # Recompute centroid for the new mode
            focal_pts = np.concatenate([d['points'] for d in self.focal_data], axis=0)
            if self.solo or len(self.condition_pts) == 0:
                self.scene_centroid = focal_pts.mean(axis=0)
            else:
                all_pts = np.concatenate([self.condition_pts, focal_pts], axis=0)
                self.scene_centroid = all_pts.mean(axis=0)
            self._draw()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize root axes within the full dentition context'
    )
    parser.add_argument('--patient_folder', type=str, required=True,
                        help='Path to patient folder (e.g., ./data/TADP0001U)')
    parser.add_argument('--fdi', type=int, nargs='+', default=None,
                        help='FDI tooth numbers to highlight (default: first available)')
    parser.add_argument('--solo', action='store_true',
                        help='Show focal tooth only, hide context teeth (toggle with S key)')
    args = parser.parse_args()

    verts_dir     = os.path.join(args.patient_folder, 'verts')
    root_axes_dir = os.path.join(args.patient_folder, 'root_axes')
    patient_id    = os.path.basename(args.patient_folder)

    if not os.path.exists(verts_dir):
        print(f'Verts directory not found: {verts_dir}')
        return

    all_fdis = get_all_fdis(verts_dir)
    if not all_fdis:
        print('No teeth found in verts directory.')
        return

    focal_fdis = args.fdi if args.fdi else all_fdis[:1]
    focal_fdis = [f for f in focal_fdis if f in all_fdis]

    if not focal_fdis:
        print(f'None of the requested FDIs found. Available: {all_fdis}')
        return

    print(f'Patient : {patient_id}')
    print(f'All FDIs: {all_fdis}')
    print(f'Focal   : {focal_fdis}')

    # Load all teeth
    all_teeth = {}
    for fdi in all_fdis:
        pts = load_tooth(verts_dir, fdi)
        if pts is not None and len(pts) > 0:
            all_teeth[fdi] = pts

    # Condition = all non-focal teeth combined
    condition_pts_list = [pts for fdi, pts in all_teeth.items() if fdi not in focal_fdis]
    condition_pts = np.concatenate(condition_pts_list, axis=0) if condition_pts_list else np.zeros((0, 3))

    # Focal teeth with root axes
    focal_data = []
    for k, fdi in enumerate(focal_fdis):
        pts = all_teeth.get(fdi)
        if pts is None:
            print(f'  FDI {fdi}: tooth points not found, skipping')
            continue
        color = FOCAL_COLORS[k % len(FOCAL_COLORS)]
        ra    = None
        if os.path.exists(root_axes_dir):
            ra = load_root_axis(root_axes_dir, fdi)
            if ra is None:
                print(f'  FDI {fdi}: root axis not found or zero — axis display disabled')
        else:
            print(f'  Root axes dir not found ({root_axes_dir}) — axis display disabled')
        focal_data.append({'fdi': fdi, 'points': pts, 'color': color, 'root_axis': ra})

    if not focal_data:
        print('No focal teeth could be loaded.')
        return

    title = f'{patient_id}  —  FDI {", ".join(str(f) for f in focal_fdis)}'
    viewer = DentitionViewer(condition_pts, focal_data, title=title, solo=args.solo)
    plt.show()


if __name__ == '__main__':
    main()
