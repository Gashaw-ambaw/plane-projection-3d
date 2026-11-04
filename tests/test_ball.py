"""Check the slicer against the geometry of a ball (not against slice_slow).

A plane through the center of a ball of radius R cuts a disk of radius R, so the
image must show about pi*R^2 ball pixels, centered, with a width of about 2R.
"""
import numpy as np
import pytest

from projection.geometry import normalize_points, output_shape, validate
from projection.slicer import slice_volume
from projection.synthetic import make_ball_volume, plane_through_center

SIZE = 300
RADIUS = 100
BALL = 2000
HALF_WIDTH = 120

AXES = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
_rng = np.random.default_rng(12345)
RANDOM_NORMALS = [tuple(_rng.normal(size=3)) for _ in range(20)]
NORMALS = AXES + RANDOM_NORMALS


@pytest.fixture(scope="module")
def ball_volume():
    return make_ball_volume(SIZE, RADIUS, background=1000, ball=BALL)


def measure(M, normal):
    """Return (area ratio, centroid offset in pixels, width error in pixels)."""
    pts = plane_through_center(normal, SIZE, half_width=HALF_WIDTH)
    E = slice_volume(M, pts)
    rows_idx, cols_idx = np.nonzero(E == BALL)

    area_ratio = len(rows_idx) / (np.pi * RADIUS ** 2)
    rows, cols = E.shape
    centroid_offset = np.hypot(
        rows_idx.mean() - (rows - 1) / 2, cols_idx.mean() - (cols - 1) / 2
    )
    width = cols_idx.max() - cols_idx.min() + 1
    return area_ratio, centroid_offset, abs(width - 2 * RADIUS)


def test_plane_points_are_a_valid_trapezoid_with_equal_legs():
    pts = plane_through_center((1, 2, 3), SIZE, half_width=HALF_WIDTH)
    a1, a2, b, c = normalize_points(pts)

    # Legs a'->b and a''->c have the same length (2 * half_width).
    assert np.linalg.norm(b - a1) == pytest.approx(2 * HALF_WIDTH)
    assert np.linalg.norm(c - a2) == pytest.approx(2 * HALF_WIDTH)
    # The 4 points are on one plane, and validate accepts them.
    n = np.cross(b - a1, a2 - a1)
    assert abs(np.dot(c - a1, n)) < 1e-6
    validate(a1, a2, b, c)
    assert output_shape(pts) == (2 * HALF_WIDTH, 2 * HALF_WIDTH)


def test_plane_is_perpendicular_to_the_normal_and_through_the_center():
    normal = np.array([1.0, 2.0, 3.0])
    a1, a2, b, c = normalize_points(plane_through_center(normal, SIZE))

    assert np.dot(b - a1, normal) == pytest.approx(0, abs=1e-9)
    assert np.dot(a2 - a1, normal) == pytest.approx(0, abs=1e-9)
    # The center of the square is the center of the volume.
    assert (a1 + b + a2 + c) / 4 == pytest.approx(np.full(3, (SIZE - 1) / 2))


@pytest.mark.parametrize("normal", NORMALS)
def test_ball_area_is_pi_r_squared(ball_volume, normal):
    area_ratio, _, _ = measure(ball_volume, normal)
    assert 0.97 <= area_ratio <= 1.03, f"normal={normal}: ratio {area_ratio:.4f}"


@pytest.mark.parametrize("normal", NORMALS)
def test_ball_is_centered_in_the_image(ball_volume, normal):
    _, centroid_offset, _ = measure(ball_volume, normal)
    assert centroid_offset <= 1, f"normal={normal}: offset {centroid_offset:.3f} px"


@pytest.mark.parametrize("normal", NORMALS)
def test_ball_width_is_two_r(ball_volume, normal):
    _, _, width_error = measure(ball_volume, normal)
    assert width_error <= 3, f"normal={normal}: width error {width_error} px"
