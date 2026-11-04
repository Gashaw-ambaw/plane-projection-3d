import numpy as np


def make_ball_volume(size=300, radius=100, background=1000, ball=2000, dtype=np.uint16):
    """Return a C-contiguous size^3 volume with a ball centered on voxel size // 2."""
    center = size // 2  # 150 for size 300, like the lecturer's example
    M = np.full((size, size, size), background, dtype=dtype)

    axis = np.arange(size) - center
    yz = axis[:, None] ** 2 + axis[None, :] ** 2  # y^2 + z^2, computed once
    # One x-plane at a time keeps the memory small.
    for x in range(size):
        M[x][yz + axis[x] ** 2 <= radius ** 2] = ball
    return M


def plane_through_center(normal, size=300, center=None, half_width=120, up=None):
    """Return 4 points (a', b, a'', c) of a square lying on the plane through center.

    The plane is perpendicular to normal, the side is 2 * half_width. a'->b and
    a''->c are the equal legs of the trapezoid (the assumed order of the project).
    up (optional) fixes the direction of the legs, so a rotating plane keeps its
    orientation from frame to frame.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    if center is None:
        center = np.full(3, size // 2)  # the center of the ball of make_ball_volume
    center = np.asarray(center, dtype=float)

    if up is None:
        # Any axis that is not (nearly) parallel to n gives a valid first direction.
        up = np.eye(3)[np.argmin(np.abs(n))]
    up = np.asarray(up, dtype=float)
    v = up - np.dot(up, n) * n  # the part of up that lies on the plane
    v = v / np.linalg.norm(v)
    u = np.cross(v, n)  # unit, on the plane, perpendicular to v

    h = half_width
    a1 = center - u * h - v * h
    b = center - u * h + v * h
    a2 = center + u * h - v * h
    c = center + u * h + v * h
    return [tuple(p) for p in (a1, b, a2, c)]
