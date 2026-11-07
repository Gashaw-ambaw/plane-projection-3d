# Testing: plan and test description

Requirements: [SRS.md](SRS.md). Algorithm: [algorithm.md](algorithm.md). Speed history: [benchmarks.md](../benchmarks/benchmarks.md).

## 1. Test plan
**Scope.** Every REQ and NFR of the SRS. The demo (`Ex0.py`) is tested as a whole.

**Levels.** *Unit:* `geometry`, `csv_planes`, `volume_loader`, `video`. *Integration:* the three slicers (Numba, NumPy, slow reference) compared with `np.array_equal`; `test_slicer.py` runs once with each path (`conftest.py`). *System:* `Ex0.py` as a subprocess on `MRHead_1.nrrd` + `Ex0.csv`: 4 frames, 1 error in row 3. *Acceptance:* the synthetic ball, the lecturer's CSV rows, the FPS benchmarks (scripts, not pytest), and the videos compared by eye with 3D Slicer and the lecturer's pictures.

**Approach.** pytest. The slow slicer is the oracle for the fast ones; the ball (known area pi*r^2 and center) checks them without it. Invalid input is tested with the lecturer's rows and values around `EPS`.

**Environment and data.** Windows 11, Python 3.13.17, `.venv`. `data/MRHead_1.nrrd`, `data/ball.nrrd`, `data/Ex0.csv`, `data/ball_rotate.csv`, seeded random volumes and `tmp_path` files. The lecturer's class volumes are not needed.

**Entry criteria.** `pip install -r requirements.txt`. **Exit criteria.** `python -m pytest` passes; every `bench_numba.py` median is 30 FPS or more; `Ex0.csv` gives 4 frames and 1 error.

## 2. Test description
`python -m pytest --collect-only -q` lists 279 tests; a full run gives 279 passed in 18 to 21 s (3 runs). With the Numba import blocked: 181 passed, 98 skipped in 17 s (2 runs).

| ID | Verifies | What is checked | Input | Expected | File (tests) |
|---|---|---|---|---|---|
| T-01 | REQ-01, REQ-06 | shape, and the value that 3D Slicer shows at `M[111,125,74]`; a 1D, 2D or 4D file is rejected | `MRHead_1.nrrd`, 3 generated NRRD files | shape (256,256,130), value 68; `ValueError` "expected a 3D volume" | `test_volume_loader.py` (4) |
| T-02 | REQ-02 | `#` and blank lines skipped, row numbers count data rows only; unreadable row kept as an error | `Ex0.csv`, temp CSVs | planes and messages as listed | `test_csv_planes.py` (9) |
| T-03 | REQ-03, REQ-04, REQ-07 | rows 1, 2, 4 of the lecturer's CSV valid, row 3 not; legs differing by 0.0009 / 0.0011; `a''` off the plane; top base longer; float noise in a square; NaN, inf; `Dab < 1`; collinear points; point order | points | accept or `ValueError` with the reason | `test_geometry.py` (31) |
| T-04 | REQ-05 | `floor(Dab)` x `ceil(Dbc)`, also for non-whole `Dbc`; side 2048 accepted, 2049 rejected ("image too large"), also frame 16 of `edge_cases.csv` (5102 x 10199) | planes | expected shape, or `ValueError` | `test_geometry.py`, `test_slicer_slow.py` (4), `test_ball.py` |
| T-05 | REQ-06, NFR-02 | fast = slow: plane partly outside, tilted triangles, other dtypes, Fortran order, `block_rows`; a too-large plane is rejected before the image is allocated | planes, volumes | `np.array_equal`; 0 outside | `test_slicer.py` (28), `test_ball.py` (164) |
| T-06 | NFR-02, NFR-03 | Numba used when available; NumPy fallback when missing; both check the input first | `HAVE_NUMBA` patched | right path called | `test_slicer_paths.py` (4) |
| T-07 | REQ-05, REQ-06 | ball area = pi*r^2, centered, width 2r, for 3 axes and 20 random normals; fails when the ball is 1 voxel off | `ball.nrrd`, generated planes | area within tolerance, centroid within 0.5 px | `test_ball.py` |
| T-08 | REQ-09, REQ-10 | brightness scale, no uint16 overflow, canvas size (even, largest plane), padding, error frame, write and read back | arrays | as listed | `test_video.py` (17) |
| T-09 | REQ-08, REQ-09, REQ-11 | one frame per row, equal even size, `Error in row 3`, FPS line, comment lines, warm-up; a too-large row gives one error frame and no larger canvas; encoder failure or empty output gives an exit message and no mp4 (fake writer); missing argument / file / folder | `Ex0.py` as a subprocess | exit code and output as listed | `test_demo.py` (14) |
| T-10 | REQ-02, REQ-04 | CSV generator output stays valid after rounding | generated CSVs | all rows valid | `test_make_csv.py` (4) |
| T-11 | NFR-01 | slicing FPS on 500^3 (random uint16 and `croissant.nrrd`) | planes of `bench_numba.py` | median of 30 FPS or more per case | `benchmarks/bench_numba.py` (not pytest) |

Test counts are in brackets; they add up to 279.

## 3. Risks and limits
- **Orientation (mirror or rotation)** is not caught by the ball test, because the ball is symmetric. It was checked by hand on `MRHead_1`: the voxel `M[111,125,74] = 68` matches 3D Slicer, and the head views show no mirror or rotation against the lecturer's reference pictures. The correlations with those pictures are moderate (0.5 to 0.9) because they have scale bars and a physical aspect ratio. This is a manual check, not an automatic test.
- Not tested yet: the volume size limit `< 2^10`. The 2048 image limit is our own rule; the real encoder failure was seen only by hand, the tests use a fake writer.
- The 4-point order and the coplanarity threshold are assumptions (SRS section 2).
- FPS is noisy: background load changed results by 1.6x or more. The NumPy path alone fell to 16.8 FPS in one loaded round.
- Coverage is not measured. The lecturer's volumes and CSVs are checked by eye (`scripts/run_examples.ps1`, `docs/experiment/`).
