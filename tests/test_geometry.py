from pathlib import Path

import numpy as np
import pytest

from projection.csv_planes import csv_row
from projection.geometry import normalize_points, output_shape, validate
from projection.synthetic import plane_through_center

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


# --- The rules of the lecturer's Q&A, with EPS = 0.001 ---


@pytest.mark.parametrize("leg", [10.0009])
def test_validate_accepts_legs_that_differ_by_less_than_eps(leg):
    # Dab = 10 and Dac = 10.0009: the difference 0.0009 is below EPS = 0.001.
    validate(*normalize_points([(0, 0, 0), (10, 0, 0), (0, leg, 0)]))


@pytest.mark.parametrize("leg", [10.0011, 10.01])
def test_validate_rejects_legs_that_differ_by_more_than_eps(leg):
    # A difference of 0.0011 or 0.01 is above EPS = 0.001 (the old limit 0.5 accepted both).
    with pytest.raises(ValueError, match="not isosceles"):
        validate(*normalize_points([(0, 0, 0), (10, 0, 0), (0, leg, 0)]))


def test_validate_accepts_a2_slightly_off_the_plane():
    # a2 is 0.0005 above the plane z = 0 of a1, b, c: below EPS.
    validate(*normalize_points([(0, 0, 0), (0, 10, 0), (10, 0, 0.0005), (10, 10, 0)]))


def test_validate_rejects_a2_off_the_plane_by_more_than_eps():
    # a2 is 0.002 above the plane z = 0 of a1, b, c: above EPS.
    with pytest.raises(ValueError, match="not coplanar"):
        validate(*normalize_points([(0, 0, 0), (0, 10, 0), (10, 0, 0.002), (10, 10, 0)]))


def test_validate_rejects_top_base_longer_than_bottom_base():
    # Legs 5 and 5, but dist(a1, a2) = 10 is larger than dist(b, c) = 4.
    with pytest.raises(ValueError, match=r"dist\(a1,a2\)"):
        validate(*normalize_points([(0, 0, 0), (3, 4, 0), (10, 0, 0), (7, 4, 0)]))


def test_validate_accepts_square_with_float_noise_in_the_bases():
    # A square has dist(a1, a2) == dist(b, c), but float noise makes a1-a2 longer by about 1e-14.
    # The check must not reject it (this plane is one of the ball tests).
    normal = (-1.2093882869743162, -1.4122920134741184, 0.5415468299050529)
    a1, a2, b, c = normalize_points(plane_through_center(normal, 300, half_width=120))

    assert np.linalg.norm(a2 - a1) > np.linalg.norm(c - b)  # the noise is really there
    validate(a1, a2, b, c)


def test_lecturer_row_4_is_valid_only_in_our_point_order():
    # Our order is (a', b, a'', c): legs 502.49 and 502.49.
    p = csv_row(CSV, 4)
    validate(*normalize_points(p))
    # Read as (a1, a2, b, c), the order the Q&A lists them, the legs are 300 and 400.
    with pytest.raises(ValueError, match="not isosceles"):
        validate(*normalize_points([p[0], p[2], p[1], p[3]]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_validate_rejects_a_coordinate_that_is_not_finite(bad):
    with pytest.raises(ValueError, match="not a finite number"):
        validate(*normalize_points([(bad, 0, 0), (10, 0, 0), (0, 10, 0)]))


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
