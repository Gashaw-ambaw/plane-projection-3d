from pathlib import Path

import numpy as np
import pytest

from projection.csv_planes import csv_row
from projection.geometry import normalize_points, validate
from projection.slicer import slice_volume
from projection.slicer_slow import slice_slow
from projection.volume_loader import load_volume, prepare_volume

DATA = Path(__file__).resolve().parents[1] / "data"
CSV = DATA / "Ex0.csv"

pytestmark = pytest.mark.usefixtures("slicer_path")  # every test runs with Numba and with NumPy


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
    assert_same_as_slow(M, csv_row(CSV, 4))


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


def test_fortran_volume_after_prepare_equals_slow():
    # A Fortran-ordered volume, once prepared, gives the same image as the slow version.
    x, y, z = np.indices((10, 10, 10))
    M = np.asfortranarray(x * 100 + y * 10 + z)
    pts = [(0, 4, 0), (3, 8, 0), (3, 0, 0)]

    assert not M.flags.c_contiguous
    assert np.array_equal(slice_volume(prepare_volume(M), pts), slice_slow(M, pts))


def test_fortran_volume_without_prepare_raises():
    # Forgetting prepare_volume must be an error, not a silent slowdown.
    M = np.asfortranarray(np.zeros((10, 10, 10)))
    with pytest.raises(ValueError, match="C-contiguous"):
        slice_volume(M, [(0, 4, 0), (3, 8, 0), (3, 0, 0)])


def test_loaded_mrhead_is_c_contiguous_after_prepare():
    # The NRRD loader returns a Fortran-ordered array; prepare_volume fixes it.
    M = load_volume(str(DATA / "MRHead_1.nrrd"))
    assert not M.flags.c_contiguous
    assert prepare_volume(M).flags.c_contiguous


@pytest.mark.parametrize("block_rows", [1, 3, 32])
def test_block_rows_equal_slow_on_example(block_rows):
    # Any block size (even one that does not divide the rows) gives the same image.
    x, y, z = np.indices((10, 10, 10))
    M = x * 100 + y * 10 + z
    pts = [(0, 4, 0), (3, 8, 0), (3, 0, 0)]
    assert np.array_equal(
        slice_volume(M, pts, block_rows=block_rows), slice_slow(M, pts)
    )


def test_block_rows_below_one_raises():
    with pytest.raises(ValueError, match="block_rows"):
        slice_volume(np.zeros((10, 10, 10)), [(0, 4, 0), (3, 8, 0), (3, 0, 0)], block_rows=0)
