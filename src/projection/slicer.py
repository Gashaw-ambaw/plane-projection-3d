import numpy as np

from projection.geometry import normalize_points, output_shape


def slice_volume(M, pts, block_rows=64):
    """Fast (vectorized NumPy) version: slice volume M by the plane in pts.

    block_rows is how many rows are computed at once (small blocks stay in the CPU cache).
    """
    # Cheap check: a silent copy on every frame would be much slower.
    if not M.flags.c_contiguous:
        raise ValueError("M must be C-contiguous; call prepare_volume(M) once after loading")
    if block_rows < 1:
        raise ValueError(f"block_rows must be at least 1, got {block_rows}")

    a1, a2, b, c = normalize_points(pts)
    rows, cols = output_shape(pts)  # also validates the plane

    dab = np.linalg.norm(b - a1)
    dac = np.linalg.norm(c - a2)
    dbc = np.linalg.norm(c - b)

    # Unit vectors: one step along them is one voxel.
    dir1 = (b - a1) / dab
    dir2 = (c - a2) / dac

    E = np.zeros((rows, cols))

    # Compute the rows in small blocks of block_rows rows.
    for r0 in range(0, rows, block_rows):
        r1 = min(r0 + block_rows, rows)
        _slice_rows(M, E, r0, r1, a1, a2, dir1, dir2, dbc)
    return E


def _slice_rows(M, E, r0, r1, a1, a2, dir1, dir2, dbc):
    """Fill rows r0..r1-1 of E (all the math, on a block of rows)."""
    cols = E.shape[1]

    # Step numbers for these rows as a column, so every row is computed at once.
    k = np.arange(r0 + 1, r1 + 1)[:, None]
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
    x1 = np.arange(cols)[None, :] - start[:, None]  # (rows in block, cols)

    # Mask 1: keep only cells that are on the line (0 <= x1 <= d12).
    on_line = (x1 >= 0) & (x1 <= d12[:, None] + 1e-9)

    # Point on the plane for every cell, one axis at a time.
    px = a1_y[:, 0][:, None] + x1 * dp[:, 0][:, None]
    py = a1_y[:, 1][:, None] + x1 * dp[:, 1][:, None]
    pz = a1_y[:, 2][:, None] + x1 * dp[:, 2][:, None]

    # Nearest voxel on each axis (half rounds up).
    vx = np.floor(px + 0.5).astype(int)
    vy = np.floor(py + 0.5).astype(int)
    vz = np.floor(pz + 0.5).astype(int)

    # Mask 2: keep only voxels that are inside the volume.
    inside = (
        (vx >= 0) & (vx < M.shape[0])
        & (vy >= 0) & (vy < M.shape[1])
        & (vz >= 0) & (vz < M.shape[2])
    )
    mask = on_line & inside

    # One flat index per cell. Cells outside the volume get a wrong index,
    # but they are never read because the mask removes them first.
    Sy, Sz = M.shape[1], M.shape[2]
    idx = vx * Sy * Sz + vy * Sz + vz

    # Write into this block's rows of E (a view, so E itself is filled).
    # M is C-contiguous (checked in slice_volume), so reshape(-1) is a view.
    E_part = E[r0:r1]
    E_part[mask] = M.reshape(-1)[idx[mask]]
