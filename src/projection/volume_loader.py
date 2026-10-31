import nrrd
import numpy as np


def load_volume(path: str) -> np.ndarray:
    """Load a NRRD file and return the 3D array indexed as M[x, y, z]."""
    data, _header = nrrd.read(path)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {data.ndim}D")
    return data
