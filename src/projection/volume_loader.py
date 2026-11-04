import nrrd
import numpy as np


def load_volume(path: str) -> np.ndarray:
    """Load a NRRD file and return the 3D array indexed as M[x, y, z]."""
    data, _header = nrrd.read(path)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {data.ndim}D")
    return data


def prepare_volume(M: np.ndarray) -> np.ndarray:
    """Return M as a C-contiguous array (a view if it already is one)."""
    # Call once after loading, NOT per frame: a copy of a big volume is slow.
    return np.ascontiguousarray(M)
