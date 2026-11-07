"""Demo: python Ex0.py file.nrrd file.csv file.mp4

Writes one video frame for every plane (row) in the CSV, in the CSV order.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from projection import slicer  # noqa: E402
from projection.csv_planes import read_planes  # noqa: E402
from projection.geometry import output_shape  # noqa: E402
from projection.slicer import slice_volume  # noqa: E402
from projection.video import (  # noqa: E402
    canvas_shape,
    error_frame,
    normalize_to_uint8,
    pad_to_canvas,
    write_video,
)
from projection.volume_loader import load_volume, prepare_volume  # noqa: E402

PLAYBACK_FPS = 30  # speed of the output video, not related to the slicing speed
MAX_ERROR_CHARS = 200
USAGE = "usage: python Ex0.py file.nrrd file.csv file.mp4"
# A small valid triangle (Dab = Dac = 4) used only for the warm-up.
WARM_UP_PLANE = [(0, 0, 0), (4, 0, 0), (0, 4, 0)]


def check_arguments(argv):
    """Return (nrrd, csv, mp4) paths, or exit with a clear message."""
    if len(argv) != 4:
        sys.exit(f"expected 3 arguments, got {len(argv) - 1}\n{USAGE}")
    nrrd_path, csv_path, mp4_path = (Path(a) for a in argv[1:])
    for path in (nrrd_path, csv_path):
        if not path.is_file():
            sys.exit(f"input file not found: {path}\n{USAGE}")
    if not mp4_path.resolve().parent.is_dir():
        sys.exit(f"output folder does not exist: {mp4_path.resolve().parent}\n{USAGE}")
    return nrrd_path, csv_path, mp4_path


def _warm_up(volume):
    """Compile the Numba kernel for this volume's dtype before the timed loop (nothing to do without Numba)."""
    # Warm-up: Numba compiles the kernel on its first call (0.2 to 1.6 s, per dtype).
    # Calling it once here, outside the timed loop, keeps the printed FPS about the slicing only.
    # The result is discarded.
    if slicer.HAVE_NUMBA:
        slice_volume(volume, WARM_UP_PLANE)


def main(argv):
    # Step 0: check the arguments.
    nrrd_path, csv_path, mp4_path = check_arguments(argv)

    # Step 1: load the volume.
    M = prepare_volume(load_volume(str(nrrd_path)))
    vmax = float(M.max())  # one value for the whole volume (S1-9)
    print(f"Volume: shape={M.shape}, dtype={M.dtype}, min={M.min()}, max={M.dtype.type(vmax)}")

    # Step 2: read the planes (one per CSV row).
    planes = read_planes(csv_path)

    # Step 3, pass 1 (cheap): the image size of every valid row, no slicing yet.
    shapes = []
    for plane in planes:
        if plane.error:
            continue
        try:
            shapes.append(output_shape(plane.points))
        except ValueError:
            pass  # reported in pass 2
    canvas = canvas_shape(shapes)

    # Step 3b: warm up the slicer, so its compile time is not in the printed FPS.
    _warm_up(M)

    # Step 4, pass 2: one frame per row; frames are written one by one, not kept in memory.
    stats = {"slice_time": 0.0, "sliced": 0, "errors": []}

    def frames():
        for number, plane in enumerate(planes, start=1):
            message = plane.error
            if message is None:
                try:
                    start = time.perf_counter()
                    E = slice_volume(M, plane.points)
                    stats["slice_time"] += time.perf_counter() - start
                    stats["sliced"] += 1
                    yield pad_to_canvas(normalize_to_uint8(E, vmax), canvas)
                    continue
                except ValueError as e:
                    message = str(e)
            message = message[:MAX_ERROR_CHARS]
            stats["errors"].append((number, message))
            yield error_frame(canvas, f"row {number}: {message}")

    # Step 5: write the frames to the mp4 file.
    try:
        count = write_video(frames(), mp4_path, PLAYBACK_FPS)
        if mp4_path.stat().st_size == 0:  # ffmpeg can fail without an exception (a side above 16384)
            raise OSError("the encoder wrote an empty file")
    except (OSError, ValueError) as e:
        # write_video has already closed the writer (its finally block), so on Windows the
        # half-written file can be deleted: a truncated mp4 would look like a good result.
        message = f"could not write the video: {e}"
        try:
            mp4_path.unlink(missing_ok=True)
        except OSError:
            message += f"\n(the partial file {mp4_path} could not be deleted)"
        sys.exit(message)

    # Step 6: print a short summary.
    print(f"Frames written: {count} (errors: {len(stats['errors'])})")
    for number, message in stats["errors"]:
        print(f"  Error in row {number}: {message}")
    print(f"Canvas: {canvas[0]} rows x {canvas[1]} cols")
    print(f"Output: {mp4_path} ({mp4_path.stat().st_size} bytes, {PLAYBACK_FPS} FPS playback)")
    if stats["sliced"]:
        fps = stats["sliced"] / stats["slice_time"]
        print(
            f"Slicing: {stats['sliced']} slices in {stats['slice_time']:.3f} s = {fps:.1f} FPS "
            "(slice_volume calls only)"
        )
        print(
            "Note: with few rows this number is noisy (first call, small sample); "
            "the official FPS is measured in benchmarks/bench_final.py."
        )


if __name__ == "__main__":
    main(sys.argv)
