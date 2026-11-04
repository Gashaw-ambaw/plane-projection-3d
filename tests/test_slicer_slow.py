import csv
from pathlib import Path

import numpy as np
import pytest

from projection.slicer_slow import normalize_points, slice_slow, validate

DATA = Path(__file__).resolve().parents[1] / "data"


def csv_row(number):
    """Return the points of the lecturer's CSV row (1-based, header skipped)."""
    with open(DATA / "Ex0.csv", newline="") as f:
        rows = list(csv.reader(f))
    values = [float(x) for x in rows[number][1:] if x != ""]
    return [tuple(values[i:i + 3]) for i in range(0, len(values), 3)]


def test_triangle_a1_equals_a2():
    # With 3 points, a1 and a2 are both the first point.
    a1, a2, b, c = normalize_points([(0, 0, 0), (1, 2, 3), (4, 5, 6)])

    assert np.array_equal(a1, [0, 0, 0])
    assert np.array_equal(a2, [0, 0, 0])
    assert np.array_equal(b, [1, 2, 3])
    assert np.array_equal(c, [4, 5, 6])


def test_trapezoid_order_a1_b_a2_c():
    # With 4 points, the order must be a1, b, a2, c.
    a1, a2, b, c = normalize_points([(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)])

    assert np.array_equal(a1, [1, 1, 1])
    assert np.array_equal(b, [2, 2, 2])
    assert np.array_equal(a2, [3, 3, 3])
    assert np.array_equal(c, [4, 4, 4])


def test_points_are_float_arrays():
    # Every returned point is a NumPy array of floats.
    a1, a2, b, c = normalize_points([(0, 0, 0), (1, 2, 3), (4, 5, 6)])

    for point in (a1, a2, b, c):
        assert isinstance(point, np.ndarray)
        assert point.dtype == float


@pytest.mark.parametrize("count", [0, 2, 5])
def test_wrong_number_of_points_raises(count):
    # 0, 2 or 5 points must raise an error.
    with pytest.raises(ValueError, match="expected 3 or 4 points"):
        normalize_points([(0, 0, 0)] * count)


@pytest.mark.parametrize("row", [1, 2, 4])
def test_validate_accepts_lecturer_csv_rows(row):
    # The valid rows from the lecturer's CSV must pass.
    # Row 1 is a triangle, rows 2 and 4 are trapezoids.
    validate(*normalize_points(csv_row(row)))


def test_validate_rejects_csv_row_3_not_isosceles():
    # Row 3 must fail: the two sides are too different.
    # Dab = 497.39 but Dac = 476.86.
    with pytest.raises(ValueError, match="not isosceles"):
        validate(*normalize_points(csv_row(3)))


def test_validate_rejects_coincident_points():
    # Points on the same spot (Dab = 0) must raise an error.
    with pytest.raises(ValueError, match="smaller than 1"):
        validate(*normalize_points([(5, 5, 5), (5, 5, 5), (9, 9, 9)]))


def test_validate_rejects_dab_below_one():
    # Dab of half a voxel is too short and must raise an error.
    with pytest.raises(ValueError, match="smaller than 1"):
        validate(*normalize_points([(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0)]))


def test_validate_rejects_points_on_one_line():
    # Three points on one line must raise an error.
    # b and c are on opposite sides of a, so Dab == Dac but all lie on a line.
    with pytest.raises(ValueError, match="one line"):
        validate(*normalize_points([(0, 0, 0), (10, 0, 0), (-10, 0, 0)]))


def test_validate_rejects_non_coplanar_trapezoid():
    # Four points that are not on one plane must raise an error.
    # a2 is 4.47 voxels away from the plane of a1, b, c.
    points = [(0, 0, 0), (10, 0, 0), (0, 0, 5), (0, 10, 5)]
    with pytest.raises(ValueError, match="not coplanar"):
        validate(*normalize_points(points))


def test_slice_slow_output_shape():
    # Triangle with Dab = 5 and Dbc = 8 gives an image of 5 rows and 8 columns.
    M = np.zeros((10, 10, 10))
    E = slice_slow(M, [(0, 0, 0), (3, 4, 0), (3, -4, 0)])
    assert E.shape == (5, 8)


def test_slice_slow_first_row_values():
    # M[x, y, z] = x*100 + y*10 + z, so every voxel value tells its position.
    x, y, z = np.indices((10, 10, 10))
    M = x * 100 + y * 10 + z
    E = slice_slow(M, [(0, 4, 0), (3, 8, 0), (3, 0, 0)])

    assert E.shape == (5, 8)
    assert E[0, 3] == 150
    assert E[0, 4] == 140
    # The rest of row 0 is empty (d12 = 1.6 gives only 2 pixels).
    assert np.count_nonzero(E[0]) == 2


def test_slice_slow_plane_outside_volume_is_zeros():
    # A plane far from the volume must give a black image, not an error.
    M = np.ones((10, 10, 10))
    E = slice_slow(M, [(100, 100, 100), (103, 104, 100), (103, 96, 100)])

    assert E.shape == (5, 8)
    assert not E.any()


def test_slice_slow_integer_rectangle_shape():
    # Rectangle with whole-number sides (4 x 4) must give exactly 4 x 4 pixels.
    M = np.ones((10, 10, 10))
    E = slice_slow(M, [(0, 0, 0), (4, 0, 0), (0, 4, 0), (4, 4, 0)])

    assert E.shape == (4, 4)
