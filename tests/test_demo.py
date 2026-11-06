import importlib.util
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from projection.csv_planes import read_planes

ROOT = Path(__file__).resolve().parents[1]
EX0 = ROOT / "Ex0.py"
NRRD = ROOT / "data" / "MRHead_1.nrrd"
CSV = ROOT / "data" / "Ex0.csv"


def run_demo(*args):
    return subprocess.run(
        [sys.executable, str(EX0), *map(str, args)],
        capture_output=True, text=True, timeout=300,
    )


def test_demo_writes_one_frame_per_csv_row(tmp_path):
    mp4 = tmp_path / "out.mp4"
    result = run_demo(NRRD, CSV, mp4)

    assert result.returncode == 0, result.stderr
    assert mp4.is_file()

    frames = imageio.mimread(str(mp4), memtest=False)
    assert len(frames) == len(read_planes(CSV))
    # Every frame has the same even size (S1-10).
    sizes = {f.shape for f in frames}
    assert len(sizes) == 1
    height, width, _ = sizes.pop()
    assert height % 2 == 0 and width % 2 == 0


def test_demo_reports_the_invalid_row_as_an_error(tmp_path):
    result = run_demo(NRRD, CSV, tmp_path / "out.mp4")

    # Row 3 of the lecturer's CSV is not isosceles.
    assert "Error in row 3" in result.stdout
    assert "not isosceles" in result.stdout
    assert "errors: 1" in result.stdout
    assert "Error in row 1" not in result.stdout


def test_demo_skips_comment_lines_and_numbers_rows_without_them(tmp_path):
    csv_path = tmp_path / "commented.csv"
    csv_path.write_text(
        "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4\n"
        "# a comment, with commas, before the first plane\n"
        "3,200,200,0,0,200,400,400,200,400,,,\n"
        "# frame 2 is not isosceles\n"
        "\n"
        "3,200,200,-30,-50,200,400,400,250,400,,,\n"
    )
    mp4 = tmp_path / "out.mp4"
    result = run_demo(NRRD, csv_path, mp4)

    assert result.returncode == 0, result.stderr
    assert "Frames written: 2 (errors: 1)" in result.stdout
    assert "Error in row 2: not isosceles" in result.stdout
    assert len(imageio.mimread(str(mp4), memtest=False)) == 2


def test_demo_prints_volume_info_and_slicing_fps(tmp_path):
    result = run_demo(NRRD, CSV, tmp_path / "out.mp4")

    assert "shape=(256, 256, 130)" in result.stdout
    assert "FPS (slice_volume calls only)" in result.stdout


def test_demo_output_is_the_same_with_the_warm_up(tmp_path):
    # The warm-up must not add a frame, an error or change the printed lines.
    result = run_demo(NRRD, CSV, tmp_path / "out.mp4")

    assert result.returncode == 0, result.stderr
    assert "Frames written: 4 (errors: 1)" in result.stdout
    assert "Error in row 3: not isosceles" in result.stdout
    assert "Slicing: 3 slices in" in result.stdout  # only the 3 valid rows are counted
    assert "FPS (slice_volume calls only)" in result.stdout


def load_ex0():
    spec = importlib.util.spec_from_file_location("ex0_demo", EX0)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warm_up_calls_slice_volume_once_when_numba_is_used(monkeypatch):
    ex0 = load_ex0()
    calls = []
    monkeypatch.setattr(ex0.slicer, "HAVE_NUMBA", True)
    monkeypatch.setattr(ex0, "slice_volume", lambda M, pts: calls.append((M, pts)))
    M = np.zeros((10, 10, 10), dtype=np.int16)

    assert ex0._warm_up(M) is None  # the result is not returned
    assert len(calls) == 1
    assert calls[0][0] is M  # the same array, so the kernel is compiled for its dtype


def test_warm_up_does_nothing_without_numba(monkeypatch):
    ex0 = load_ex0()
    calls = []
    monkeypatch.setattr(ex0.slicer, "HAVE_NUMBA", False)
    monkeypatch.setattr(ex0, "slice_volume", lambda M, pts: calls.append(pts))

    ex0._warm_up(np.zeros((10, 10, 10), dtype=np.int16))

    assert calls == []


def test_warm_up_plane_is_valid_and_works_on_the_real_slicer():
    ex0 = load_ex0()
    M = np.ones((10, 10, 10), dtype=np.uint16)

    ex0._warm_up(M)  # must not raise
    E = ex0.slice_volume(M, ex0.WARM_UP_PLANE)
    assert E.shape == (4, 6)  # Dab = 4 rows, Dbc = 5.66 gives 6 columns


def test_demo_with_a_missing_argument_fails():
    result = run_demo(NRRD, CSV)

    assert result.returncode != 0
    assert "usage" in result.stderr


def test_demo_with_a_missing_input_file_fails(tmp_path):
    result = run_demo(tmp_path / "nope.nrrd", CSV, tmp_path / "out.mp4")

    assert result.returncode != 0
    assert "not found" in result.stderr


def test_demo_with_a_missing_output_folder_fails(tmp_path):
    result = run_demo(NRRD, CSV, tmp_path / "no_such_folder" / "out.mp4")

    assert result.returncode != 0
    assert "output folder" in result.stderr
