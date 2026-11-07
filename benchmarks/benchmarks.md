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

## Numba kernel vs NumPy fallback (decision 14, docs/decisions.md)

Script: `benchmarks/bench_numba.py` (2 full runs, 10 rounds each; NumPy and Numba take turns inside every round, the order changes every round). FPS = `slice_volume` only, JIT compile excluded. Every number below is MEASURED on the same machine as above (i5-1235U, Python 3.13.17, numba 0.68.1, numpy 2.5.4).

Volumes: `croissant.nrrd` (500^3 uint8) and random 500^3 uint16 (seed 0). Planes: the 100 valid rows of the lecturer's `croissant.csv` (1 pass per round); the z-constant 499x499 and tilted 424x400 planes of `bench_final.py` (10 slices per round). The images of the two paths were compared with `np.array_equal` for every plane: equal in all cases of both runs.

Conditions: Microsoft Edge open (21 msedge/webview2 processes), power plan Balanced, on AC. CPU load right before the run: 18% (run 1), 24% (run 2). The NumPy numbers here are clearly lower than the 4 runs above (NumPy medians of 26.0 to 40.0 FPS on the z-constant and croissant.csv rows, 49.8 to 76.8 on the tilted rows); the cause was not investigated, background load is the only difference noted.

| Run | Case | NumPy median / min FPS | Numba median / min FPS | Ratio of medians | Median of per-round ratios |
| --- | --- | --- | --- | --- | --- |
| 1 | croissant.csv, 100 planes (uint8) | 30.8 / 24.2 | 163.5 / 110.5 | 5.31 | 5.54 |
| 1 | z constant 499x499 (uint8) | 36.6 / 25.7 | 120.8 / 92.9 | 3.30 | 3.64 |
| 1 | tilted 424x400 (uint8) | 55.0 / 48.7 | 182.3 / 154.5 | 3.32 | 3.30 |
| 1 | z constant 499x499 (random uint16) | 40.0 / 24.5 | 118.9 / 86.5 | 2.97 | 3.43 |
| 1 | tilted 424x400 (random uint16) | 49.8 / 43.9 | 158.3 / 141.5 | 3.18 | 3.22 |
| 2 | croissant.csv, 100 planes (uint8) | 26.3 / 16.8 | 163.3 / 88.7 | 6.21 | 5.73 |
| 2 | z constant 499x499 (uint8) | 28.2 / 24.3 | 94.6 / 83.5 | 3.35 | 3.16 |
| 2 | tilted 424x400 (uint8) | 52.2 / 42.8 | 184.2 / 152.3 | 3.53 | 3.59 |
| 2 | z constant 499x499 (random uint16) | 26.0 / 22.6 | 82.3 / 65.7 | 3.17 | 3.31 |
| 2 | tilted 424x400 (random uint16) | 76.8 / 50.5 | 233.7 / 143.9 | 3.04 | 3.10 |

- Numba ratio: 2.97 to 6.21 (ratio of medians over both runs). The lowest Numba round in any case is 65.7 FPS; the lowest NumPy round is 16.8 FPS (below the 30 FPS requirement, under this load).
- 100 FPS: Numba median is above 100 in 8 of 10 rows; the two exceptions are the z-constant planes in run 2 (94.6 uint8 and 82.3 uint16).

### JIT compile (first call, includes one slice; not inside any FPS above)

| Run | First call, croissant uint8 | First call, random uint16 (second dtype in the same process) |
| --- | --- | --- |
| 1 | 0.95 s | 0.24 s |
| 2 | 1.36 s | 0.80 s |

A separate quick check in a fresh process (uint16, 60^3 volume) took 1.58 s for the first call. A second call takes 0.002 to 0.004 s. Each new volume dtype is compiled once per process (`cache=False`).

### End to end: `Ex0.py` on croissant (100 frames, wall clock of the whole process)

| Run | Path | Wall time | `Slicing:` line printed by Ex0 (includes the JIT on Numba) |
| --- | --- | --- | --- |
| 1 | Numba | 4.85 s | 100 slices in 0.780 s = 128.2 FPS |
| 1 | NumPy | 5.79 s | 100 slices in 1.537 s = 65.1 FPS |
| 2 | Numba | 8.13 s | 100 slices in 1.713 s = 58.4 FPS |
| 2 | NumPy | 10.53 s | 100 slices in 3.559 s = 28.1 FPS |

The two output videos are byte-identical (same SHA-256). Run 2 was slower for both paths (machine load changed); compare the two paths inside a run, not across runs. The wall-time saving is 0.94 s (run 1) and 2.4 s (run 2); how the rest of the time splits between loading, slicing and encoding was not measured.

### Same benchmark again after closing the browser tabs (quieter machine)

Same script, same settings, 2 more runs. MEASURED. Conditions: Edge/webview2 processes 6 (was 21), Chrome 0; CPU load right before the benchmark runs 42% for both (the e2e runs below started at 14%); VS Code, Discord and the Claude CLI were still running. Images of the two paths equal in every case.

| Run | Case | NumPy median / min FPS | Numba median / min FPS | Ratio of medians | Median of per-round ratios |
| --- | --- | --- | --- | --- | --- |
| 3 | croissant.csv, 100 planes (uint8) | 76.5 / 70.8 | 498.3 / 386.6 | 6.51 | 5.84 |
| 3 | z constant 499x499 (uint8) | 83.8 / 64.5 | 316.1 / 236.8 | 3.77 | 3.77 |
| 3 | tilted 424x400 (uint8) | 108.0 / 93.0 | 629.7 / 576.3 | 5.83 | 5.52 |
| 3 | z constant 499x499 (random uint16) | 77.2 / 56.6 | 261.9 / 170.5 | 3.39 | 3.22 |
| 3 | tilted 424x400 (random uint16) | 124.5 / 80.2 | 460.1 / 282.5 | 3.70 | 3.62 |
| 4 | croissant.csv, 100 planes (uint8) | 62.6 / 58.8 | 471.8 / 346.2 | 7.54 | 7.40 |
| 4 | z constant 499x499 (uint8) | 56.9 / 15.1 | 205.2 / 23.9 | 3.61 | 4.09 |
| 4 | tilted 424x400 (uint8) | 75.8 / 27.0 | 338.9 / 158.9 | 4.47 | 5.04 |
| 4 | z constant 499x499 (random uint16) | 65.4 / 36.7 | 273.7 / 154.9 | 4.19 | 4.03 |
| 4 | tilted 424x400 (random uint16) | 117.6 / 111.2 | 556.7 / 400.7 | 4.73 | 4.63 |

- Run 4 had a disturbance in some rounds (NumPy min 15.1 and Numba min 23.9 FPS on the z-constant uint8 case, NumPy min 27.0 on tilted uint8); the cause was not identified. Run 3 is the cleaner one.
- Closing the tabs raised the NumPy medians (run 3: 76.5 to 124.5 FPS, against 26.0 to 76.8 in runs 1 and 2). The code did not change, so this is background load, as noted for the 4 runs at the top of this file.
- Ratio of medians over runs 3 and 4: 3.39 to 7.54. Over all four Numba runs of this section: 2.97 to 7.54.
- Numba median above 100 FPS in all 10 rows of runs 3 and 4 (lowest median 205.2). NumPy median above 100 FPS in 3 of 10 rows (run 3: tilted uint8 and tilted uint16; run 4: tilted uint16).
- The lowest Numba round in runs 3 and 4 is 23.9 FPS (the disturbed round in run 4); without it, 154.9.
- JIT first call in runs 3 and 4: 0.39 s and 0.57 s (croissant uint8), 0.18 s and 0.27 s (random uint16, second dtype in the process). All six measured first calls so far: 0.18 to 1.58 s.

End to end, `Ex0.py` on croissant, quieter machine (CPU load 14% before):

| Run | Path | Wall time | `Slicing:` line printed by Ex0 (includes the JIT on Numba) |
| --- | --- | --- | --- |
| 3 | Numba | 3.06 s | 100 slices in 0.606 s = 165.0 FPS |
| 3 | NumPy | 4.40 s | 100 slices in 1.564 s = 63.9 FPS |
| 4 | Numba | 4.58 s | 100 slices in 0.816 s = 122.5 FPS |
| 4 | NumPy | 5.08 s | 100 slices in 1.704 s = 58.7 FPS |

The videos are again byte-identical (same SHA-256). Wall-time saving: 1.34 s (run 3) and 0.50 s (run 4). Test suite on the quiet machine: 242 passed in 20.8 s (143 tests took 20.3 s before this change).
