import numpy as np

from projection.slicer_slow import normalize_points, validate


def slice_volume(M, pts):
    """Fast (vectorized NumPy) version: slice volume M by the plane in pts."""
    a1, a2, b, c = normalize_points(pts)
    validate(a1, a2, b, c)

    dab = np.linalg.norm(b - a1)
    dac = np.linalg.norm(c - a2)
    dbc = np.linalg.norm(c - b)

    # Image size. The 1e-9 stops float noise from adding an extra row or column.
    rows = int(np.floor(dab + 1e-9))
    cols = int(np.ceil(dbc - 1e-9))

    # Unit vectors: one step along them is one voxel.
    dir1 = (b - a1) / dab
    dir2 = (c - a2) / dac

    # Step numbers 1..rows as a column, so every row is computed at once.
    k = np.arange(1, rows + 1)[:, None]
    # Both anchors after k steps (step first, then sample). One row per y.
    a1_y = a1 + k * dir1
    a2_y = a2 + k * dir2

    # Vector from the left anchor to the right anchor, for every row.
    diff = a2_y - a1_y
    # Length of each row's line.
    d12 = np.linalg.norm(diff, axis=1)
    # Unit step along each row's line.
    dp = diff / d12[:, None]
    # First column of each row; rounds half up and centers the line.
    start = np.floor(dbc / 2 - d12 / 2 + 0.5).astype(int)

    # Column grid: for each cell, how many steps it is from the line's start.
    x1 = np.arange(cols)[None, :] - start[:, None]  # (rows, cols)

    # Mask 1: keep only cells that are on the line (0 <= x1 <= d12).
    on_line = (x1 >= 0) & (x1 <= d12[:, None] + 1e-9)

    # Point on the plane for every cell, then its nearest voxel (half rounds up).
    p = a1_y[:, None, :] + x1[..., None] * dp[:, None, :]  # (rows, cols, 3)
    v = np.floor(p + 0.5).astype(int)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    # Mask 2: keep only voxels that are inside the volume.
    inside = (
        (vx >= 0) & (vx < M.shape[0])
        & (vy >= 0) & (vy < M.shape[1])
        & (vz >= 0) & (vz < M.shape[2])
    )

    # Write the volume values into E; every other cell stays 0.
    mask = on_line & inside
    E = np.zeros((rows, cols))
    E[mask] = M[vx[mask], vy[mask], vz[mask]]
    return E
