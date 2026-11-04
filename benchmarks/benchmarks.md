# Benchmarks: slicer

Machine: Windows 11, Intel Core i5-1235U, 15.7 GB RAM, Python 3.13.17.
Volume: random 500x500x500 uint16 (seed 0). FPS = `slice_volume` only (no video encoding).
Script: `benchmarks/bench_final.py` (final default settings, 5 rounds x 10 slices, planes taking turns).

## Final version: 4 runs (median FPS)

| Run | Conditions | z constant (499x499) | tilted (424x400) |
| --- | --- | --- | --- |
| 1 | not recorded | 88.8 | 146.3 |
| 2 | browsers and windows open | 68.7 | 112.3 |
| 3 | browsers and windows open | 72.8 | 113.3 |
| 4 | browsers and windows closed | 109.9 | 202.7 |

- 30 FPS (required) is met in every round of every run (lowest round: 51.4 FPS).
- 100 FPS (expected): the tilted plane is above 100 in every run. The large z-constant plane reached it only when the machine was idle.
- Background load changed the result by a factor of about 1.6. The code did not change between runs.

## How the speed was reached (FPS)

| Version | z constant | tilted |
| --- | --- | --- |
| `slice_slow` (loops, 1 run) | 0.37 to 0.48 | 0.67 to 0.73 |
| first vectorized version | 27.6 | 47.9 |
| per-axis (rows, cols) arrays instead of (rows, cols, 3) | 34.4 | 53.9 |
| flat index, one read | 37.7 | 57.6 |
| row blocks of 64 rows (single thread) | 97.0 | 146.9 |

The last row was measured in a separate session with a script that runs the settings in turns, so compare the steps, not the exact numbers.

## What was tried and removed

- Threads over row chunks: 59.7 and 111.4 FPS, but with small row blocks one thread was as fast (within noise). Removed to keep the code simple.
- `np.take` instead of indexing: no gain, reverted.
- `int32` flat index: inconclusive (inside the noise), removed.

## Findings

1. Bottleneck (first vectorized version): about 60% of the time went into building large temporary arrays. Reading from the volume was about 17%.
2. Row blocks keep the temporary arrays small. Likely reason: they fit in the CPU cache (not measured directly).
3. The NRRD loader returns a Fortran-ordered array. `prepare_volume(M)` makes it C-contiguous once after loading. `slice_volume` raises `ValueError` otherwise, because a silent copy on every frame would be much slower.
4. Measurements are noisy: differences of about 10% between settings are not reliable.

## Correctness

`slice_volume` equals `slice_slow` exactly (`np.array_equal`) on the tests. Independent check: on the synthetic ball, the disk area is 0.9979 to 1.0005 of pi*100^2 over 60 planes (see `docs/experiment`).
