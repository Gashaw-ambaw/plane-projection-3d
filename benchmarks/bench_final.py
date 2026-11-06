"""Measure slice_volume (default settings) on a random 500^3 uint16 volume.

Two planes are measured in turns: ROUNDS rounds, each round slices every plane
SLICES times. slice_slow is not used here.
"""
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from projection import slicer  # noqa: E402
from projection.slicer import slice_volume  # noqa: E402

# benchmarks.md describes the NumPy version, so force it (bench_numba.py compares both paths).
slicer.HAVE_NUMBA = False

ROUNDS = 5
SLICES = 10

PLANES = {
    "z constant": [(0, 0, 250), (499, 0, 250), (0, 499, 250), (499, 499, 250)],
    "tilted": [(100, 50, 100), (400, 50, 400), (100, 450, 100), (400, 450, 400)],
}

# The volume is created outside the timing.
rng = np.random.default_rng(0)
M = rng.integers(0, 65536, size=(500, 500, 500), dtype=np.uint16)

# Warm-up: one slice per plane, not timed.
for pts in PLANES.values():
    slice_volume(M, pts)

fps = {name: [] for name in PLANES}
for _ in range(ROUNDS):
    for name, pts in PLANES.items():
        start = time.perf_counter()
        for _ in range(SLICES):
            slice_volume(M, pts)
        fps[name].append(SLICES / (time.perf_counter() - start))

print(f"Volume 500^3 uint16, default settings, {ROUNDS} rounds x {SLICES} slices, planes in turns")
print(f"{'plane':<12}{'image':>12}{'median FPS':>12}{'min FPS':>10}{'max FPS':>10}")
for name, pts in PLANES.items():
    shape = slice_volume(M, pts).shape
    values = fps[name]
    print(
        f"{name:<12}{str(shape):>12}{statistics.median(values):>12.1f}"
        f"{min(values):>10.1f}{max(values):>10.1f}"
    )
