import polyscope as ps
import numpy as np
import sys
import argparse

# ── Config ───────────────────────────────────────────────────────────────────
TOOTH_NPOINTS = 1024   # change to 2048 if needed
N_TEETH       = 28

# ── Hardcoded files (fill these in to skip CLI args) ─────────────────────────
# Set FILES to a list of .npy paths to load without typing them on the command line.
# Set the corresponding label in LABELS (must be same length as FILES, or leave empty).
# Leave FILES = [] to use command-line arguments instead.
FILES = []
LABELS = []

# Visual settings (only used when FILES is non-empty)
SHOW_AXES    = True
ARROW_LENGTH = 2.0
POINT_RADIUS = 0.03

# Camera view — set CAMERA_POSITION / CAMERA_TARGET to fix the opening view.
# Leave both as None to let polyscope auto-fit the scene.
# Tip: position the camera how you like it in the GUI, then read off the values
# from the "View" panel (Camera extrinsics) and paste them here.
CAMERA_POSITION = (0.0845, -3.8248, 0.9373)
CAMERA_TARGET   = (0.0908, -0.0497, -0.1067)
CAMERA_UP       = (0.0, 1.0, 0.0)              # y-up matches set_up_dir("y_up")

# Per-label color overrides — set to None to use the default rainbow palette.
# Any label not listed here (or mapped to None) gets the rainbow per-tooth colors.
LABEL_COLORS = {
    "condition": (0.65, 0.65, 0.65),   # gray
    "generated": (0.90, 0.15, 0.15),   # red
    "gt":        None,                  # rainbow (default)
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def tooth_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        h = hue * 6
        x = 1 - abs(h % 2 - 1)
        if   h < 1: r,g,b = 1,x,0
        elif h < 2: r,g,b = x,1,0
        elif h < 3: r,g,b = 0,1,x
        elif h < 4: r,g,b = 0,x,1
        elif h < 5: r,g,b = x,0,1
        else:       r,g,b = 1,0,x
        colors.append([r, g, b])
    return np.array(colors)

COLORS = tooth_colors(N_TEETH)


def compute_root_axis(points):
    """PCA root axis: first principal component of the tooth point cloud."""
    centered = points - points.mean(axis=0)
    cov = (centered.T @ centered) / len(points)
    cov += np.eye(3) * 1e-6                          # numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
    axis = eigenvectors[:, -1]                        # largest eigenvector
    if axis[1] < 0:                                   # canonicalize to +y (crown up)
        axis = -axis
    return axis / np.linalg.norm(axis)


def register_axis_arrow(name, centroid, axis, color, arrow_length= 1.0, radius=0.0025):
    """
    Draw a root axis line as a curve network.
    The line runs from centroid - axis*length (apex) to centroid + axis*length (crown).
    """
    apex  = centroid - axis * arrow_length
    crown = centroid + axis * arrow_length

    nodes = np.array([apex, crown])
    edges = np.array([[0, 1]])

    ps.register_curve_network(name, nodes, edges, radius=radius,
                              color=tuple(color))


def load_and_register(npy_path, label, show_axes, point_radius=0.003, arrow_length=4.0,
                      color_override=None):
    pts = np.load(npy_path)                        # (N_total, 3)

    total_points = pts.shape[0]
    if total_points % TOOTH_NPOINTS != 0:
        raise ValueError(
            f"{npy_path}: total points ({total_points}) is not divisible by "
            f"--npoints ({TOOTH_NPOINTS}). Try passing --npoints <correct_value>."
        )
    n_teeth_in_file = total_points // TOOTH_NPOINTS
    pts = pts.reshape(n_teeth_in_file, TOOTH_NPOINTS, 3)
    print(f"  [{label}] {n_teeth_in_file} tooth/teeth detected")

    for i in range(n_teeth_in_file):
        tooth_pts = pts[i]                         # (npoints, 3)

        if np.allclose(tooth_pts, 0, atol=1e-6):
            continue

        color = color_override if color_override is not None else COLORS[i]
        name  = f"{label}_tooth{i:02d}"

        ps.register_point_cloud(name, tooth_pts, radius=point_radius,
                                color=tuple(color))

        # Root axis arrow (optional)
        if show_axes:
            axis     = compute_root_axis(tooth_pts)
            centroid = tooth_pts.mean(axis=0)
            register_axis_arrow(
                name         = f"{label}_axis_tooth{i:02d}",
                centroid     = centroid,
                axis         = axis,
                color        = color,
                arrow_length = arrow_length,
                radius       = point_radius * 0.2
            )


# ── Argument parsing ─────────────────────────────────────────────────────────
def parse_args(require_files=True):
    parser = argparse.ArgumentParser(description="Polyscope dental point cloud viewer")

    parser.add_argument("files", nargs="+" if require_files else "*",
                        help="One or more .npy dentition files to visualize")
    parser.add_argument("--labels", nargs="+", default=["condition", "generated", "gt"],
                        help="Display labels for each file (default: condition generated gt)")
    parser.add_argument("--axes", action="store_true",
                        help="Show PCA root axis arrows for each tooth")
    parser.add_argument("--arrow-length", type=float, default=0.8,
                        help="Length of root axis arrows in scene units (default: 1.5)")
    parser.add_argument("--point-radius", type=float, default=0.028,
                        help="Point cloud radius (default: 0.012 — increase for denser look)")
    parser.add_argument("--npoints", type=int, default=1024,
                        help="Points per tooth in the saved npy (default: 1024)")

    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Always parse CLI so option flags (--axes, --point-radius, etc.) work
    # regardless of whether FILES is hardcoded.
    args = parse_args(require_files=not bool(FILES))
    TOOTH_NPOINTS = args.npoints

    files        = FILES  if FILES  else args.files
    labels       = (LABELS if LABELS else [f"file{i}" for i in range(len(FILES))]) if FILES else args.labels
    show_axes    = args.axes    or SHOW_AXES
    point_radius = args.point_radius
    arrow_length = args.arrow_length

    # Pre-compute bounding box from teeth only so we can lock the length scale
    # after arrow registration (arrows inflate the bbox and shrink relative radii).
    all_raw = np.concatenate([np.load(f) for f in files], axis=0)
    nz = all_raw[~np.all(np.abs(all_raw) < 1e-6, axis=1)]
    teeth_scale = float(np.linalg.norm(nz.max(0) - nz.min(0))) if len(nz) else 1.0

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_navigation_style("free")   # unrestricted rotation — no gimbal lock
    ps.set_ground_plane_mode("shadow_only")

    for idx, fpath in enumerate(files):
        label = labels[idx] if idx < len(labels) else f"file{idx}"
        print(f"Loading [{label}]: {fpath}")
        load_and_register(
            npy_path       = fpath,
            label          = label,
            show_axes      = show_axes,
            point_radius   = point_radius,
            arrow_length   = arrow_length,
            color_override = LABEL_COLORS.get(label),
        )

    # Force length scale back to teeth-only extent so arrows don't affect point size
    ps.set_length_scale(teeth_scale)

    # Apply fixed camera view if configured
    scene_centroid = tuple(nz.mean(0)) if len(nz) else (0.0, 0.0, 0.0)
    if CAMERA_POSITION is not None:
        target = CAMERA_TARGET if CAMERA_TARGET is not None else scene_centroid
        ps.look_at_dir(CAMERA_POSITION, target, CAMERA_UP, fly_to=False)

    import polyscope.imgui as psim

    def camera_callback():
        if psim.Button("Print camera position"):
            view = ps.get_camera_view_matrix()    # 4x4 world-to-camera
            R, t = view[:3, :3], view[:3, 3]
            pos     = -R.T @ t                    # camera world position
            forward =  R.T @ np.array([0, 0, -1]) # forward direction in world space
            target  = np.array(scene_centroid)
            # refine: project scene centroid onto the look ray
            d = np.dot(target - pos, forward)
            target = pos + forward * max(d, 0.1)
            print(f"CAMERA_POSITION = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            print(f"CAMERA_TARGET   = ({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f})")

    ps.set_user_callback(camera_callback)
    ps.show()
