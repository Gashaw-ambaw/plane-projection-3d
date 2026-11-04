"""Time one slice of the slow reference version on a 500x500x500 volume."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from projection.slicer_slow import slice_slow  # noqa: E402

rng = np.random.default_rng(0)
M = rng.integers(0, 65536, size=(500, 500, 500), dtype=np.uint16)
pts = [(0, 0, 250), (499, 0, 250), (0, 499, 250), (499, 499, 250)]

start = time.perf_counter()
E = slice_slow(M, pts)
seconds = time.perf_counter() - start

print(f"seconds per frame: {seconds:.3f}")
print(f"FPS: {1 / seconds:.3f}")
print(f"pixels: {E.size} ({E.shape[0]} x {E.shape[1]})")
