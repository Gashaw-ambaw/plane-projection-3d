from pathlib import Path

import nrrd
import numpy as np
import pytest

from projection.volume_loader import load_volume

DATA = Path(__file__).resolve().parents[1] / "data"


def test_mrhead_shape_and_slicer_probe_value():
    file_path = DATA / "MRHead_1.nrrd"     #File + fileName
    file_text = str(file_path)             #Convert from text to string
    m = load_volume(file_text)             # load the volume into the array m

    assert m.shape == (256, 256, 130)
    # Value shown by 3D Slicer's Data Probe in the assignment's screenshot.
    assert m[111, 125, 74] == 68


@pytest.mark.parametrize("shape", [(5, 4), (6,), (2, 3, 4, 5)])
def test_non_3d_nrrd_raises_value_error(tmp_path, shape):
    path = tmp_path / "not_3d.nrrd"
    nrrd.write(str(path), np.zeros(shape, dtype=np.uint8))

    with pytest.raises(ValueError, match="expected a 3D volume"):
        load_volume(str(path))


