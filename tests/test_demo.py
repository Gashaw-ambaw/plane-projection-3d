import importlib.util
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

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
    # Every frame has the same even size: decision 11 (docs/decisions.md).
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


HEADER = "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4\n"
SMALL_ROW = "3,200,200,0,0,200,400,400,200,400,,,\n"
HUGE_ROW = "3,99.5,0,80,-5000,179,80,5199,179,80,,,\n"  # a valid plane, image 5102 x 10199


def test_demo_gives_a_too_large_plane_an_error_frame_and_keeps_the_canvas_small(tmp_path):
    csv_path = tmp_path / "edge_like.csv"
    csv_path.write_text(
        HEADER + "# a comment\n" + SMALL_ROW + HUGE_ROW + "\n# another comment\n" + SMALL_ROW
    )
    mp4 = tmp_path / "out.mp4"
    result = run_demo(NRRD, csv_path, mp4)

    assert result.returncode == 0, result.stderr
    assert "Frames written: 3 (errors: 1)" in result.stdout
    assert "Error in row 2: image too large: 5102 x 10199, maximum 2048 per side" in result.stdout
    # The rejected row does not enlarge the canvas: it is the size of the small planes (447 x 400, made even).
    assert "Canvas: 448 rows x 400 cols" in result.stdout
    frames = imageio.mimread(str(mp4), memtest=False)
    assert len(frames) == 3
    assert {f.shape for f in frames} == {(448, 400, 3)}


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


def run_main_with_fake_writer(monkeypatch, tmp_path, fake_write_video):
    ex0 = load_ex0()
    monkeypatch.setattr(ex0, "write_video", fake_write_video)
    mp4 = tmp_path / "out.mp4"
    with pytest.raises(SystemExit) as stop:
        ex0.main(["Ex0.py", str(NRRD), str(CSV), str(mp4)])
    return stop.value.code, mp4


def test_demo_exits_with_a_message_and_leaves_no_mp4_when_the_encoder_fails(monkeypatch, tmp_path):
    def failing_write_video(frames, path, fps):
        path.write_bytes(b"half written")  # what ffmpeg leaves behind
        raise OSError("[Errno 22] Invalid argument\n\nFFMPEG STDERR OUTPUT:\nInvalid buffer size")

    code, mp4 = run_main_with_fake_writer(monkeypatch, tmp_path, failing_write_video)

    assert isinstance(code, str) and "could not write the video" in code
    assert "FFMPEG STDERR OUTPUT" in code  # ffmpeg's own text is shown
    assert not mp4.exists()


def test_demo_treats_an_empty_output_file_as_a_failure(monkeypatch, tmp_path):
    def silent_write_video(frames, path, fps):
        path.write_bytes(b"")  # ffmpeg exits at once, nothing is written, no exception (size above 16384)
        return 4

    code, mp4 = run_main_with_fake_writer(monkeypatch, tmp_path, silent_write_video)

    assert isinstance(code, str) and "could not write the video" in code
    assert not mp4.exists()


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
