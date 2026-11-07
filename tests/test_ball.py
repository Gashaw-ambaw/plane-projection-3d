"""Check the slicer against the geometry of a ball (not against slice_slow).

A plane through the center of a ball of radius R cuts a disk of radius R, so the
image must show about pi*R^2 ball pixels, centered, with a width of about 2R.
"""
from pathlib import Path

import numpy as np
import pytest

from projection.csv_planes import read_planes
from projection.geometry import normalize_points, output_shape, validate
from projection.slicer import slice_volume
from projection.slicer_slow import slice_slow
from projection.synthetic import make_ball_volume, plane_through_center
from projection.volume_loader import load_volume, prepare_volume

DATA = Path(__file__).resolve().parents[1] / "data"

pytestmark = pytest.mark.usefixtures("slicer_path")  # every test runs with Numba and with NumPy


SIZE = 300
RADIUS = 100
BALL = 2000
HALF_WIDTH = 120
CENTROID_TOLERANCE = 0.5  # pixels

AXES = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
_rng = np.random.default_rng(12345)
RANDOM_NORMALS = [tuple(_rng.normal(size=3)) for _ in range(20)]
NORMALS = AXES + RANDOM_NORMALS


@pytest.fixture(scope="module")
def ball_volume():
    return make_ball_volume(SIZE, RADIUS, background=1000, ball=BALL)


def expected_center_pixel(pts, center):
    """Where the algorithm puts the point `center` of the plane: (row, column).

    The row is k_center - 1 and the column is x1_center - start, as in slice_slow.
    The result is half a pixel away from the middle of the image, because:
    - a is not sampled, decision 6 (docs/decisions.md): row y is sampled after y + 1 steps,
      so the middle of the plane is in row h - 1, not h - 0.5;
    - the last column x1 = d12 is thrown away when d12 is a whole number, decision 6, so
      the columns are x1 = 0..cols - 1 and the middle is in column h.
    """
    a1, a2, b, c = normalize_points(pts)
    dir1 = (b - a1) / np.linalg.norm(b - a1)
    dir2 = (c - a2) / np.linalg.norm(c - a2)

    k_center = np.dot(center - a1, dir1)
    a1_y = a1 + k_center * dir1
    a2_y = a2 + k_center * dir2
    d12 = np.linalg.norm(a2_y - a1_y)
    dp = (a2_y - a1_y) / d12
    x1_center = np.dot(center - a1_y, dp)
    start = np.floor(np.linalg.norm(c - b) / 2 - d12 / 2 + 0.5)
    return k_center - 1, x1_center - start


def measure(M, normal):
    """Return (area ratio, centroid offset in pixels, width error in pixels).

    The centroid offset is measured from the expected position of the ball center.
    """
    pts = plane_through_center(normal, SIZE, half_width=HALF_WIDTH)
    E = slice_volume(M, pts)
    rows_idx, cols_idx = np.nonzero(E == BALL)

    area_ratio = len(rows_idx) / (np.pi * RADIUS ** 2)
    exp_row, exp_col = expected_center_pixel(pts, np.full(3, SIZE // 2))
    centroid_offset = np.hypot(rows_idx.mean() - exp_row, cols_idx.mean() - exp_col)
    width = cols_idx.max() - cols_idx.min() + 1
    return area_ratio, centroid_offset, abs(width - 2 * RADIUS)


def assert_centered(M, normal):
    _, centroid_offset, _ = measure(M, normal)
    assert centroid_offset <= CENTROID_TOLERANCE, (
        f"normal={normal}: offset {centroid_offset:.3f} px"
    )


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
    # The center of the square is the center of the ball (voxel SIZE // 2).
    assert (a1 + b + a2 + c) / 4 == pytest.approx(np.full(3, SIZE // 2))


@pytest.mark.parametrize("normal", NORMALS)
def test_ball_area_is_pi_r_squared(ball_volume, normal):
    area_ratio, _, _ = measure(ball_volume, normal)
    assert 0.97 <= area_ratio <= 1.03, f"normal={normal}: ratio {area_ratio:.4f}"


@pytest.mark.parametrize("normal", NORMALS)
def test_ball_is_centered_in_the_image(ball_volume, normal):
    assert_centered(ball_volume, normal)


# A ball shifted by one voxel along x must fail the centroid check (the check is not blind).
# Normal (1, 0, 0) is left out: x is perpendicular to that plane, so the disk stays centered.
@pytest.mark.parametrize("normal", [(0, 0, 1), (0, 1, 0), RANDOM_NORMALS[0]])
def test_centroid_check_fails_when_the_ball_is_one_voxel_off(ball_volume, normal):
    shifted = np.roll(ball_volume, 1, axis=0)  # the ball center moves to voxel 151 on x

    assert_centered(ball_volume, normal)  # control: the real ball passes
    with pytest.raises(AssertionError, match="offset"):
        assert_centered(shifted, normal)


@pytest.mark.parametrize("normal", NORMALS)
def test_ball_width_is_two_r(ball_volume, normal):
    _, _, width_error = measure(ball_volume, normal)
    assert width_error <= 3, f"normal={normal}: width error {width_error} px"


# --- Small checks of decisions 5 and 6 (docs/decisions.md) and the volume dtype (fast = slow) ---


@pytest.mark.parametrize(
    "pts",
    [
        # Rectangle: Dab = 4, Dbc = 5.2.
        [(0, 0, 0), (4, 0, 0), (0, 5.2, 0), (4, 5.2, 0)],
        # Isosceles triangle: Dab = Dac = 5, Dbc = 5.2.
        [(0, 0, 0), (5, 0, 0), (2.296, 4.441665, 0)],
    ],
)
def test_non_whole_dbc_gives_ceil_columns_and_equals_slow(pts):
    # Decision 5 (docs/decisions.md): Dbc = 5.2 gives 6 columns.
    x, y, z = np.indices((12, 12, 12))
    M = x * 100 + y * 10 + z + 1
    a1, a2, b, c = normalize_points(pts)
    assert np.linalg.norm(c - b) == pytest.approx(5.2, abs=0.01)

    E = slice_volume(M, pts)
    assert E.shape[1] == 6
    assert np.array_equal(E, slice_slow(M, pts))


def test_the_voxel_of_a_is_not_read():
    # Decision 6 (docs/decisions.md): a steps before sampling, so the voxel of a itself is never read.
    pts = [(2, 2, 2), (2, 7, 2), (7, 2, 2)]
    M = np.ones((12, 12, 12), dtype=np.uint16)
    M[2, 2, 2] = 9999

    assert 9999 not in slice_slow(M, pts)
    assert 9999 not in slice_volume(M, pts)

    # Control: the first voxel that is read (a after one step) does show up.
    M[2, 2, 2] = 1
    M[2, 3, 2] = 9999
    assert 9999 in slice_slow(M, pts)
    assert 9999 in slice_volume(M, pts)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint32])
def test_fast_equals_slow_for_other_dtypes(dtype):
    rng = np.random.default_rng(3)
    M = rng.integers(1, np.iinfo(dtype).max, size=(40, 40, 40), dtype=dtype)
    pts = [(10, 15, 10), (20, 25, 20), (20, 5, 20)]  # tilted isosceles triangle

    E = slice_volume(M, pts)
    assert np.array_equal(E, slice_slow(M, pts))
    if dtype == np.uint32:
        assert E.max() > 2 ** 24  # float32 could not hold such values exactly


def test_uint32_maximum_value_is_kept_exactly():
    # 2**32 - 1 is exact in float64 (a float32 would round it to 2**32).
    M = np.full((20, 20, 20), 2 ** 32 - 1, dtype=np.uint32)
    E = slice_volume(M, [(2, 2, 2), (2, 12, 2), (12, 2, 2)])

    assert E.max() == 2 ** 32 - 1


# --- data/ball.nrrd and data/ball_rotate.csv must match make_ball.py and the ball center ---


@pytest.fixture(scope="module")
def ball_nrrd():
    return prepare_volume(load_volume(str(DATA / "ball.nrrd")))


def test_ball_nrrd_has_the_ball_in_the_center(ball_nrrd):
    # scripts/make_ball.py writes background=65535 and ball=0, size 300, radius 100.
    assert ball_nrrd.shape == (SIZE, SIZE, SIZE)
    assert ball_nrrd.dtype == np.uint16
    assert ball_nrrd[150, 150, 150] == 0
    assert ball_nrrd[0, 0, 0] == 65535
    # The ball reaches voxel 250 (150 + 100) on an axis, and not 251.
    assert ball_nrrd[150, 150, 250] == 0
    assert ball_nrrd[150, 150, 251] == 65535


def test_first_ball_rotate_frame_is_a_disk_around_the_expected_center(ball_nrrd):
    pts = read_planes(DATA / "ball_rotate.csv")[0].points
    E = slice_volume(ball_nrrd, pts)

    # The white rectangle is the cut of the volume; the dark pixels inside it are the ball.
    white_rows, white_cols = np.nonzero(E == 65535)
    inside = np.zeros(E.shape, dtype=bool)
    inside[white_rows.min():white_rows.max() + 1, white_cols.min():white_cols.max() + 1] = True
    disk_rows, disk_cols = np.nonzero(inside & (E == 0))

    assert 0.97 <= len(disk_rows) / (np.pi * RADIUS ** 2) <= 1.03
    exp_row, exp_col = expected_center_pixel(pts, np.full(3, SIZE // 2))
    offset = np.hypot(disk_rows.mean() - exp_row, disk_cols.mean() - exp_col)
    assert offset <= CENTROID_TOLERANCE, f"offset {offset:.3f} px"
