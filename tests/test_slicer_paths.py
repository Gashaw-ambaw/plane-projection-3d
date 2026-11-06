"""Check that slice_volume picks the right path (Numba or the NumPy fallback)."""
import numpy as np
import pytest

from projection import slicer
from projection.slicer_slow import slice_slow

PTS = [(2, 2, 2), (2, 12, 2), (12, 2, 2)]


def make_volume():
    x, y, z = np.indices((20, 20, 20))
    return x * 100 + y * 10 + z + 1


def fail(*args, **kwargs):
    raise AssertionError("this path must not be used")


def test_fallback_is_used_when_numba_is_missing(monkeypatch):
    # Force the "numba not installed" case: the Numba kernel must not be called.
    monkeypatch.setattr(slicer, "HAVE_NUMBA", False)
    monkeypatch.setattr(slicer, "fill_image", fail, raising=False)

    M = make_volume()
    assert np.array_equal(slicer.slice_volume(M, PTS), slice_slow(M, PTS))


def test_numba_is_used_when_it_is_available(monkeypatch):
    # The other way round: the NumPy code must not be called.
    pytest.importorskip("numba")
    monkeypatch.setattr(slicer, "HAVE_NUMBA", True)
    monkeypatch.setattr(slicer, "_slice_numpy", fail)

    M = make_volume()
    assert np.array_equal(slicer.slice_volume(M, PTS), slice_slow(M, PTS))


@pytest.mark.parametrize("have_numba", [True, False])
def test_both_paths_check_the_input_first(monkeypatch, have_numba):
    # Not C-contiguous and bad block_rows are rejected before any path is chosen.
    if have_numba:
        pytest.importorskip("numba")
    monkeypatch.setattr(slicer, "HAVE_NUMBA", have_numba)

    with pytest.raises(ValueError, match="C-contiguous"):
        slicer.slice_volume(np.asfortranarray(make_volume()), PTS)
    with pytest.raises(ValueError, match="block_rows"):
        slicer.slice_volume(make_volume(), PTS, block_rows=0)
