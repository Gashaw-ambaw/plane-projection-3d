import numpy as np

from projection.slicer import slice_volume
from projection.slicer_slow import slice_slow, validate, normalize_points
from test_slicer_slow import csv_row


def assert_same_as_slow(M, pts):
    # The fast and the slow version must give exactly the same image.
    assert np.array_equal(slice_volume(M, pts), slice_slow(M, pts))


def test_slice_volume_output_shape():
    # Triangle with Dab = 5 and Dbc = 8 gives an image of 5 rows and 8 columns.
    M = np.zeros((10, 10, 10))
    E = slice_volume(M, [(0, 0, 0), (3, 4, 0), (3, -4, 0)])
    assert E.shape == (5, 8)


def test_fast_equals_slow_on_our_example():
    # M[x, y, z] = x*100 + y*10 + z, so every voxel value tells its position.
    x, y, z = np.indices((10, 10, 10))
    M = x * 100 + y * 10 + z
    assert_same_as_slow(M, [(0, 4, 0), (3, 8, 0), (3, 0, 0)])


def test_fast_equals_slow_on_integer_rectangle():
    # Rectangle with whole-number sides (4 x 4).
    x, y, z = np.indices((10, 10, 10))
    M = x * 100 + y * 10 + z + 1
    assert_same_as_slow(M, [(0, 0, 0), (4, 0, 0), (0, 4, 0), (4, 4, 0)])


def test_fast_equals_slow_on_csv_row_4_trapezoid():
    # Random volume big enough to contain the whole trapezoid.
    rng = np.random.default_rng(1)
    M = rng.integers(1, 65536, size=(202, 402, 502), dtype=np.uint16)
    assert_same_as_slow(M, csv_row(4))


def test_fast_equals_slow_when_plane_is_partly_outside():
    # The plane reaches past the volume edges, so some pixels must stay 0.
    x, y, z = np.indices((10, 10, 10))
    M = x * 100 + y * 10 + z + 1
    pts = [(5, 5, 5), (5, 5, 12), (12, 5, 5)]
    E = slice_volume(M, pts)

    assert E.any() and (E == 0).any()
    assert np.array_equal(E, slice_slow(M, pts))


def test_fast_equals_slow_on_random_tilted_triangles():
    # 5 random isosceles triangles (Dab == Dac) in a random 60^3 volume.
    rng = np.random.default_rng(2)
    M = rng.integers(1, 65536, size=(60, 60, 60), dtype=np.uint16)

    def random_unit():
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)

    checked = 0
    while checked < 5:
        a = rng.uniform(10, 50, size=3)
        length = rng.uniform(20, 40)
        b = a + length * random_unit()
        c = a + length * random_unit()
        try:
            validate(*normalize_points([a, b, c]))
        except ValueError:
            continue  # skip degenerate random triangles
        assert_same_as_slow(M, [a, b, c])
        checked += 1
