import numpy as np


def normalize_points(pts):
    """Accepts 3 or 4 points, returns a1, a2, b, c as float arrays."""
    # NOTE: the trapezoid point order (a', b, a'', c) is our assumption;
    # it has not been verified with the lecturer.
    if len(pts) == 3:
        a, b, c = pts
        a1 = a2 = a
    elif len(pts) == 4:
        a1, b, a2, c = pts
    else:
        raise ValueError(f"expected 3 or 4 points, got {len(pts)}")
    # Turn each point into a NumPy array of decimal numbers (floats).
    return (
        np.array(a1, dtype=float),
        np.array(a2, dtype=float),
        np.array(b, dtype=float),
        np.array(c, dtype=float),
    )


def validate(a1, a2, b, c):
    """Check that the points make a valid plane, or raise ValueError (S1-8)."""
    u = b - a1
    dab = np.linalg.norm(u)
    if dab < 1:
        raise ValueError(f"Dab = {dab:.3f} is smaller than 1 voxel")

    dac = np.linalg.norm(c - a2)
    if abs(dab - dac) > 0.5:
        raise ValueError(
            f"not isosceles: Dab = {dab:.2f}, Dac = {dac:.2f} (difference above 0.5)"
        )

    # u x v is perpendicular to the plane of a1, b, c; its length is the
    # area of the parallelogram spanned by u and v.
    v = c - a1
    n = np.cross(u, v)
    dist_to_line = np.linalg.norm(n) / dab  # distance of c from the line a1-b
    if dist_to_line < 0.5:
        raise ValueError(
            f"a1, b, c are on one line (c is {dist_to_line:.3f} from the line a1-b)"
        )

    w = a2 - a1
    dist_to_plane = abs(np.dot(w, n)) / np.linalg.norm(n)  # distance of a2 from the plane
    if dist_to_plane > 0.5:
        raise ValueError(
            f"the 4 points are not coplanar (a2 is {dist_to_plane:.3f} from the plane)"
        )


def slice_slow(M, pts):
    """Slow reference version: slice volume M by the plane given in pts."""
    a1, a2, b, c = normalize_points(pts)
    validate(a1, a2, b, c)

    dab = np.linalg.norm(b - a1)
    dac = np.linalg.norm(c - a2)
    dbc = np.linalg.norm(c - b)

    # Image size: one row per step along a-b, one column per step along b-c.
    rows = int(np.floor(dab))
    cols = int(np.ceil(dbc))
    E = np.zeros((rows, cols))

    # Unit vectors: moving along them advances exactly one voxel.
    dir1 = (b - a1) / dab
    dir2 = (c - a2) / dac

    for y in range(rows):
        # Step first, then sample (S1-2). No += so the original arrays stay safe.
        a1 = a1 + dir1
        a2 = a2 + dir2
        d12 = np.linalg.norm(a2 - a1)  # length of the current row
        dp = (a2 - a1) / d12  # unit step along the row

        # The 1e-9 keeps a whole-number d12 (like 8.0) from rounding down.
        n_points = int(np.floor(d12 + 1e-9)) + 1
        for x1 in range(n_points):
            # Center the row in the image. floor(v + 0.5) rounds half up (S1-6).
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
