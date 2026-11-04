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


def output_shape(pts):
    """Return (rows, cols) of the image for the plane in pts (S1-3, S1-4).

    It also validates the plane, so invalid input raises ValueError here.
    """
    a1, a2, b, c = normalize_points(pts)
    validate(a1, a2, b, c)

    dab = np.linalg.norm(b - a1)
    dbc = np.linalg.norm(c - b)

    # The 1e-9 stops float noise from adding an extra row or column.
    rows = int(np.floor(dab + 1e-9))
    cols = int(np.ceil(dbc - 1e-9))
    return rows, cols
