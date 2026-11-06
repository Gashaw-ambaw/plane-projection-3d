"""Compare the fast slicer (at several block sizes) with the slow one on a 500^3 volume."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from projection import slicer  # noqa: E402
from projection.slicer import slice_volume  # noqa: E402
from projection.slicer_slow import slice_slow  # noqa: E402

# These numbers are for the NumPy version (block_rows has no effect on the Numba one).
slicer.HAVE_NUMBA = False

RUNS = 20
BLOCK_ROWS = [8, 16, 32, 64, 128]

PLANES = {
    "axis-parallel": [(0, 0, 250), (499, 0, 250), (0, 499, 250), (499, 499, 250)],
    "tilted": [(100, 50, 100), (400, 50, 400), (100, 450, 100), (400, 450, 400)],
}

# The volume is created outside the timing.
rng = np.random.default_rng(0)
M = rng.integers(0, 65536, size=(500, 500, 500), dtype=np.uint16)


def mean_fps(pts, **kwargs):
    slice_volume(M, pts, **kwargs)  # warm-up run, not timed
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        slice_volume(M, pts, **kwargs)
        times.append(time.perf_counter() - start)
    return 1 / (sum(times) / RUNS)


for name, pts in PLANES.items():
    print(f"[{name}]")

    start = time.perf_counter()
    E = slice_slow(M, pts)
    slow_fps = 1 / (time.perf_counter() - start)
    print(f"  E shape: {E.shape}, slow (1 run): {slow_fps:.3f} FPS")

    print(f"  fast, default block_rows: {mean_fps(pts):.1f} FPS (mean of {RUNS})")
    for block_rows in BLOCK_ROWS:
        print(f"  fast, block_rows={block_rows}: {mean_fps(pts, block_rows=block_rows):.1f} FPS")
