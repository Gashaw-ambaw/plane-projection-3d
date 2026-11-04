from pathlib import Path

import numpy as np
import pytest

from projection.csv_planes import csv_row
from projection.geometry import normalize_points, output_shape, validate

CSV = Path(__file__).resolve().parents[1] / "data" / "Ex0.csv"


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
    validate(*normalize_points(csv_row(CSV, row)))


def test_validate_rejects_csv_row_3_not_isosceles():
    # Row 3 must fail: the two sides are too different.
    # Dab = 497.39 but Dac = 476.86.
    with pytest.raises(ValueError, match="not isosceles"):
        validate(*normalize_points(csv_row(CSV, 3)))


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


def test_output_shape_triangle():
    # Dab = 5 and Dbc = 8 give 5 rows and 8 columns.
    assert output_shape([(0, 0, 0), (3, 4, 0), (3, -4, 0)]) == (5, 8)


def test_output_shape_integer_rectangle():
    # Whole-number sides give exactly 4 x 4 (no extra row or column from float noise).
    assert output_shape([(0, 0, 0), (4, 0, 0), (0, 4, 0), (4, 4, 0)]) == (4, 4)


def test_output_shape_rejects_invalid_plane():
    # output_shape also validates, so CSV row 3 (not isosceles) must raise.
    with pytest.raises(ValueError, match="not isosceles"):
        output_shape(csv_row(CSV, 3))
