import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio

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


def test_demo_prints_volume_info_and_slicing_fps(tmp_path):
    result = run_demo(NRRD, CSV, tmp_path / "out.mp4")

    assert "shape=(256, 256, 130)" in result.stdout
    assert "FPS (slice_volume calls only)" in result.stdout


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
