"""Compare the fast slicer with the slow one on a 500x500x500 volume."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from projection.slicer import slice_volume  # noqa: E402
from projection.slicer_slow import slice_slow  # noqa: E402

RUNS = 20

PLANES = {
    "axis-parallel": [(0, 0, 250), (499, 0, 250), (0, 499, 250), (499, 499, 250)],
    "tilted": [(100, 50, 100), (400, 50, 400), (100, 450, 100), (400, 450, 400)],
}

# The volume is created outside the timing.
rng = np.random.default_rng(0)
M = rng.integers(0, 65536, size=(500, 500, 500), dtype=np.uint16)

ratios = {}
for name, pts in PLANES.items():
    slice_volume(M, pts)  # warm-up run, not timed

    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        E = slice_volume(M, pts)
        times.append(time.perf_counter() - start)
    fast_seconds = sum(times) / RUNS
    fast_fps = 1 / fast_seconds

    start = time.perf_counter()
    slice_slow(M, pts)
    slow_seconds = time.perf_counter() - start
    slow_fps = 1 / slow_seconds

    ratios[name] = fast_fps / slow_fps
    print(f"[{name}] E shape: {E.shape}")
    print(f"  fast: {fast_seconds:.4f} s/frame, {fast_fps:.1f} FPS (mean of {RUNS})")
    print(f"  slow: {slow_seconds:.3f} s/frame, {slow_fps:.3f} FPS (1 run)")

print()
for name, ratio in ratios.items():
    print(f"fast / slow [{name}]: {ratio:.0f}x")
