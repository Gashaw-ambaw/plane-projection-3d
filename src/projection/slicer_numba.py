import math

from numba import njit


@njit(cache=False)
def fill_image(M, E, a1, a2, dir1, dir2, dbc):
    """Numba version: fill the image E with the slice of M (a plain loop, like slicer_slow).

    The arithmetic is the same as in the NumPy version in slicer_numpy.py, so the image is identical.
    E must be zeros: cells that are not written stay 0 (outside the triangle or the volume).
    """
    rows, cols = E.shape
    sx, sy, sz = M.shape

    for y in range(rows):
        # Both anchors after k steps (step first, then sample, S1-2).
        k = y + 1
        a1x = a1[0] + k * dir1[0]
        a1y = a1[1] + k * dir1[1]
        a1z = a1[2] + k * dir1[2]
        a2x = a2[0] + k * dir2[0]
        a2y = a2[1] + k * dir2[1]
        a2z = a2[2] + k * dir2[2]

        # Vector from the left anchor to the right anchor, and its length.
        dx = a2x - a1x
        dy = a2y - a1y
        dz = a2z - a1z
        d12 = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Unit step along the row's line.
        dpx = dx / d12
        dpy = dy / d12
        dpz = dz / d12

        # First column of this row (rounds half up, centers the line).
        start = int(math.floor(dbc / 2 - d12 / 2 + 0.5))
        # Points on the line: x1 = 0 .. d12 (the 1e-9 keeps a whole-number d12 from rounding down).
        n_points = int(math.floor(d12 + 1e-9)) + 1

        for x1 in range(n_points):
            col = start + x1
            if col < 0 or col >= cols:
                continue

            # Point on the plane.
            px = a1x + x1 * dpx
            py = a1y + x1 * dpy
            pz = a1z + x1 * dpz

            # Nearest voxel on each axis (half rounds up). Checked as floats, before int().
            vx = math.floor(px + 0.5)
            vy = math.floor(py + 0.5)
            vz = math.floor(pz + 0.5)

            # Outside the volume the cell stays 0.
            if vx < 0 or vx >= sx or vy < 0 or vy >= sy or vz < 0 or vz >= sz:
                continue

            E[y, col] = M[int(vx), int(vy), int(vz)]
