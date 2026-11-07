"""Slow reference version of the slicer: plain loops, one pixel at a time.

It follows the assignment closely, so it is easy to check by hand. The demo does not use it.
The tests compare the fast versions (NumPy and Numba) with it and require the same image.
"""
import numpy as np

from projection.geometry import normalize_points, output_shape


def slice_slow(M, pts):
    """Slow reference version: slice volume M by the plane given in pts."""
    a1, a2, b, c = normalize_points(pts)
    rows, cols = output_shape(pts)  # also validates the plane

    # Side lengths: Dab and Dac are the two equal sides, Dbc is the base.
    dab = np.linalg.norm(b - a1)
    dac = np.linalg.norm(c - a2)
    dbc = np.linalg.norm(c - b)

    # Image: one row per step along a-b, one column per step along b-c.
    E = np.zeros((rows, cols))

    # Unit vectors: moving along them advances exactly one voxel.
    dir1 = (b - a1) / dab
    dir2 = (c - a2) / dac

    for y in range(rows):
        # Step first, then sample (decision 6, docs/decisions.md). No += so the original arrays stay safe.
        a1 = a1 + dir1
        a2 = a2 + dir2
        d12 = np.linalg.norm(a2 - a1)  # length of the current row
        dp = (a2 - a1) / d12  # unit step along the row

        # The 1e-9 keeps a whole-number d12 (like 8.0) from rounding down.
        n_points = int(np.floor(d12 + 1e-9)) + 1
        for x1 in range(n_points):
            # Center the row in the image. floor(v + 0.5) rounds half up (decision 7, docs/decisions.md).
            col = int(np.floor(dbc / 2 - d12 / 2 + x1 + 0.5))
            if col < 0 or col >= cols:
                continue

            p = a1 + x1 * dp  # point on the plane (no +=)
            v = np.floor(p + 0.5).astype(int)  # nearest voxel
            vx, vy, vz = v

            # Outside the volume the value stays 0.
            if not (0 <= vx < M.shape[0] and 0 <= vy < M.shape[1] and 0 <= vz < M.shape[2]):
                continue

            E[y, col] = M[vx, vy, vz]

    return E
