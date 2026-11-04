import numpy as np

from projection.slicer_slow import slice_slow


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
