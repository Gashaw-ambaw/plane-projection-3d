"""Public entry point: slice_volume cuts the volume with a plane, using the fastest way available.

The slicer files:
- slicer.py: this file. Checks the input and picks the Numba or the NumPy version.
- slicer_numba.py: the Numba version (fastest, used when numba is installed).
- slicer_numpy.py: the NumPy version (the fallback when numba is missing).
- slicer_slow.py: the slow reference version. Only the tests use it, to check the fast ones.
"""
import numpy as np

from projection.geometry import normalize_points, output_shape
from projection.slicer_numpy import _slice_numpy

# Use the Numba kernel if numba is installed; otherwise the NumPy code in slicer_numpy.py.
try:
    from projection.slicer_numba import fill_image
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def slice_volume(M, pts, block_rows=64):
    """Fast version: slice volume M by the plane in pts (Numba if available, else NumPy).

    block_rows is how many rows the NumPy version computes at once (small blocks stay in the
    CPU cache). The Numba version works one cell at a time and ignores it.
    """
    # Cheap check: a silent copy on every frame would be much slower.
    if not M.flags.c_contiguous:
        raise ValueError("M must be C-contiguous; call prepare_volume(M) once after loading")
    if block_rows < 1:
        raise ValueError(f"block_rows must be at least 1, got {block_rows}")

    if HAVE_NUMBA:
        return _slice_numba(M, pts)
    return _slice_numpy(M, pts, block_rows)


def _slice_numba(M, pts):
    """Same setup as _slice_numpy, but the loop over the cells runs in the Numba kernel."""
    a1, a2, b, c = normalize_points(pts)
    rows, cols = output_shape(pts)  # also validates the plane

    # Side lengths: Dab and Dac are the two equal sides, Dbc is the base.
    dab = np.linalg.norm(b - a1)
    dac = np.linalg.norm(c - a2)
    dbc = np.linalg.norm(c - b)

    # Unit vectors: one step along them is one voxel.
    dir1 = (b - a1) / dab
    dir2 = (c - a2) / dac

    # Start with an all-zero image. The kernel fills it in place.
    E = np.zeros((rows, cols))
    fill_image(M, E, a1, a2, dir1, dir2, dbc)
    return E
