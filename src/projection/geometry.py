import numpy as np

EPS = 0.001  # tolerance of the lecturer's Q&A (sections 1 and 3)
# Largest image side (rows or columns) we accept. Our own rule, not in the assignment: it keeps the
# video canvas and the H.264 encoder safe (see decisions.md, item 9).
MAX_IMAGE_SIDE = 2048


def normalize_points(pts):
    """Accepts 3 or 4 points, returns a1, a2, b, c as float arrays."""
    # NOTE: the trapezoid point order (a', b, a'', c) is our assumption, not confirmed with
    # the lecturer. Evidence: row 4 of the lecturer's CSV has equal legs only in this order.
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
    """Check that the points make a valid plane, or raise ValueError (decision 9, docs/decisions.md).

    Rules from the lecturer's Q&A (EPS = 0.001): equal legs, dist(a1, a2) <= dist(b, c),
    and the four points on one plane. Two more rules are ours: Dab of at least 1 voxel
    and a1, b, c not on one line.
    """
    if not all(np.all(np.isfinite(p)) for p in (a1, a2, b, c)):
        raise ValueError("a coordinate is not a finite number")

    u = b - a1
    dab = np.linalg.norm(u)
    if dab < 1:
        raise ValueError(f"Dab = {dab:.3f} is smaller than 1 voxel")

    dac = np.linalg.norm(c - a2)
    if not abs(dab - dac) < EPS:
        raise ValueError(
            f"not isosceles: Dab = {dab:.4f}, Dac = {dac:.4f} (difference not below {EPS})"
        )

    # u x v is perpendicular to the plane of a1, b, c; its length is the
    # area of the parallelogram spanned by u and v.
    v = c - a1
    n = np.cross(u, v)
    dist_to_line = np.linalg.norm(n) / dab  # distance of c from the line a1-b
    if dist_to_line < 0.5:  # our rule, not in the Q&A
        raise ValueError(
            f"a1, b, c are on one line (c is {dist_to_line:.3f} from the line a1-b)"
        )

    w = a2 - a1
    dist_to_plane = abs(np.dot(w, n)) / np.linalg.norm(n)  # distance of a2 from the plane
    if not dist_to_plane < EPS:
        raise ValueError(
            f"the 4 points are not coplanar (a2 is {dist_to_plane:.4f} from the plane)"
        )

    # The + EPS is needed: for a square the two bases are equal, and float noise
    # (about 1e-14) can make dist(a1, a2) a little longer than dist(b, c).
    d_top = np.linalg.norm(w)
    d_bottom = np.linalg.norm(c - b)
    if d_top > d_bottom + EPS:
        raise ValueError(
            f"dist(a1,a2) = {d_top:.2f} is longer than dist(b,c) = {d_bottom:.2f}"
        )


def output_shape(pts):
    """Return (rows, cols) of the image for the plane in pts (decision 5, docs/decisions.md).

    It also validates the plane, so invalid input raises ValueError here, and so does an image
    with more than MAX_IMAGE_SIDE rows or columns.
    """
    a1, a2, b, c = normalize_points(pts)
    validate(a1, a2, b, c)

    dab = np.linalg.norm(b - a1)
    dbc = np.linalg.norm(c - b)

    # The 1e-9 stops float noise from adding an extra row or column.
    rows = int(np.floor(dab + 1e-9))
    cols = int(np.ceil(dbc - 1e-9))
    if rows > MAX_IMAGE_SIDE or cols > MAX_IMAGE_SIDE:
        raise ValueError(f"image too large: {rows} x {cols}, maximum {MAX_IMAGE_SIDE} per side")
    return rows, cols
