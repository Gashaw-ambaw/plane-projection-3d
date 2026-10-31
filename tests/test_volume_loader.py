from pathlib import Path

from projection.volume_loader import load_volume

DATA = Path(__file__).resolve().parents[1] / "data"


def test_mrhead_shape_and_slicer_probe_value():
    file_path = DATA / "MRHead_1.nrrd"     #File + fileName
    file_text = str(file_path)             #Convert from text to string
    m = load_volume(file_text)             # load the volume into the array m

    assert m.shape == (256, 256, 130)
    # Value shown by 3D Slicer's Data Probe in the assignment's screenshot.
    assert m[111, 125, 74] == 68


