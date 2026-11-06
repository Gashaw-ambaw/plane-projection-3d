"""Compare slice_volume with Numba and with the NumPy fallback (slicing only, no video).

usage: python benchmarks/bench_numba.py croissant.nrrd croissant.csv

Both paths run in turns inside every round, and the order changes from round to round.
The first Numba call (JIT compile) is timed separately and is not part of any FPS below.
"""
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from projection import slicer  # noqa: E402
from projection.csv_planes import read_planes  # noqa: E402
from projection.volume_loader import load_volume, prepare_volume  # noqa: E402

ROUNDS = 10
SLICES = 10  # per round, for the two single-plane cases

Z_CONSTANT = [(0, 0, 250), (499, 0, 250), (0, 499, 250), (499, 499, 250)]
TILTED = [(100, 50, 100), (400, 50, 400), (100, 450, 100), (400, 450, 400)]


def slice_with(numba, M, pts):
    slicer.HAVE_NUMBA = numba  # slice_volume reads this flag at every call
    return slicer.slice_volume(M, pts)


def fps_of_pass(numba, M, planes, repeat):
    start = time.perf_counter()
    for _ in range(repeat):
        for pts in planes:
            slice_with(numba, M, pts)
    return repeat * len(planes) / (time.perf_counter() - start)


def run_case(title, M, planes, repeat):
    # Same image from both paths on every plane of the case.
    same = all(
        np.array_equal(slice_with(True, M, p), slice_with(False, M, p)) for p in planes
    )
    fps = {"NumPy": [], "Numba": []}
    for r in range(ROUNDS):
        order = [("NumPy", False), ("Numba", True)]
        if r % 2:
            order.reverse()  # alternate who goes first
        for name, flag in order:
            fps[name].append(fps_of_pass(flag, M, planes, repeat))
    med_np, med_nb = (statistics.median(fps[n]) for n in ("NumPy", "Numba"))
    ratios = [b / a for a, b in zip(fps["NumPy"], fps["Numba"])]
    print(
        f"| {title} | {med_np:.1f} | {min(fps['NumPy']):.1f} | {med_nb:.1f} | "
        f"{min(fps['Numba']):.1f} | {med_nb / med_np:.2f} | {statistics.median(ratios):.2f} | "
        f"{'yes' if same else 'NO'} |"
    )


def jit_cost(M):
    """Seconds of the first Numba call for this dtype (compile + one slice), and of a second call."""
    pts = TILTED
    t = time.perf_counter()
    slice_with(True, M, pts)
    first = time.perf_counter() - t
    t = time.perf_counter()
    slice_with(True, M, pts)
    return first, time.perf_counter() - t


def main(nrrd_path, csv_path):
    croissant = prepare_volume(load_volume(nrrd_path))
    random16 = np.random.default_rng(0).integers(0, 65536, size=(500, 500, 500), dtype=np.uint16)
    lecturer = [p.points for p in read_planes(csv_path) if p.error is None]

    print(f"croissant {croissant.shape} {croissant.dtype}, random {random16.shape} {random16.dtype}")
    print(f"lecturer planes (valid rows): {len(lecturer)}; {ROUNDS} rounds, {SLICES} slices per round")
    for name, M in (("croissant uint8", croissant), ("random uint16", random16)):
        first, second = jit_cost(M)
        print(f"first Numba call, {name}: {first:.2f} s (compile + 1 slice); second call: {second:.3f} s")

    print("\n| case | NumPy median FPS | NumPy min | Numba median FPS | Numba min "
          "| ratio of medians | median of per-round ratios | images equal |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    run_case("croissant.csv, 100 planes (croissant uint8)", croissant, lecturer, 1)
    run_case("z constant 499x499 (croissant uint8)", croissant, [Z_CONSTANT], SLICES)
    run_case("tilted 424x400 (croissant uint8)", croissant, [TILTED], SLICES)
    run_case("z constant 499x499 (random uint16)", random16, [Z_CONSTANT], SLICES)
    run_case("tilted 424x400 (random uint16)", random16, [TILTED], SLICES)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
