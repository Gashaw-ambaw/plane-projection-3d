# Testing: plan and test description

Requirements: [SRS.md](SRS.md). Algorithm: [algorithm.md](algorithm.md). Speed history: [benchmarks.md](../benchmarks/benchmarks.md).

## 1. Test plan
**Scope.** Every REQ and NFR of the SRS. The demo (`Ex0.py`) is tested as a whole, as the assignment asks.

**Levels.**
- *Unit:* `geometry` (validation, output size), `csv_planes`, `volume_loader`, `video` helpers.
- *Integration:* the three slicers (Numba, NumPy, slow reference) compared with `np.array_equal`. The tests in `test_slicer.py` run once with Numba and once with NumPy (`conftest.py`).
- *System:* `Ex0.py` run as a subprocess on `MRHead_1.nrrd` + `Ex0.csv`: 4 frames, 1 error in row 3 (`test_demo.py`).
- *Acceptance:* the synthetic ball (size and position of the disk for 23 plane orientations), the lecturer's CSV rows, the FPS benchmarks (scripts, not pytest), and the visual comparison of the videos with 3D Slicer and the lecturer's pictures.

**Approach and tools.** pytest. The slow slicer is the oracle for the fast ones. The ball has a known area (pi*r^2) and a known center, so it checks the slicers without the oracle. Invalid input is tested with the lecturer's own rows and with boundary values around `EPS`.

**Environment and data.** Windows 11, Python 3.13.17, `.venv`. Data: `data/MRHead_1.nrrd`, `data/ball.nrrd`, `data/Ex0.csv`, `data/ball_rotate.csv`; seeded random volumes and `tmp_path` files made inside the tests. The lecturer's class volumes are not in the repo and no test needs them.

**Entry criteria.** `pip install -r requirements.txt` done. **Exit criteria.** All tests pass (`python -m pytest`); the median of every case of `bench_numba.py` is 30 FPS or more; the demo on `Ex0.csv` reports 4 frames and 1 error.

## 2. Test description
Run `python -m pytest --collect-only -q`: 270 tests (a full run passed in 13.2 s; in a fresh clean copy, 267 tests at that time, 21.3 s). Without Numba the 97 Numba variants are skipped.

| ID | Verifies | What is checked | Input | Expected | File (tests) |
|---|---|---|---|---|---|
| T-01 | REQ-01, REQ-06 | shape, and the value that 3D Slicer shows at `M[111,125,74]`; a 1D, 2D or 4D file is rejected | `MRHead_1.nrrd`, 3 generated NRRD files | shape (256,256,130), value 68; `ValueError` "expected a 3D volume" | `test_volume_loader.py` (4) |
| T-02 | REQ-02 | rows read; `#` and blank lines skipped, row numbers count data rows only; unreadable row kept as an error | `Ex0.csv`, temp CSVs | planes and messages as listed | `test_csv_planes.py` (9) |
| T-03 | REQ-03, REQ-04, REQ-07 | rows 1, 2, 4 of the lecturer's CSV valid, row 3 not; legs differing by 0.0009 / 0.0011; `a''` off the plane; top base longer than bottom; square with float noise; NaN, inf; `Dab < 1`; collinear points; order `(a',b,a'',c)` | points | accept or `ValueError` with the reason | `test_geometry.py` (27) |
| T-04 | REQ-05 | `floor(Dab)` x `ceil(Dbc)`, also for non-whole `Dbc` | planes | expected shape | `test_geometry.py`, `test_slicer_slow.py` (4), `test_ball.py` |
| T-05 | REQ-06, NFR-02 | fast = slow, with a plane partly outside, tilted triangles, other dtypes, Fortran-ordered input, `block_rows` | planes, volumes | `np.array_equal`; 0 outside | `test_slicer.py` (26), `test_ball.py` (164) |
| T-06 | NFR-02, NFR-03 | Numba used when available; NumPy fallback when missing; both check the input first | `HAVE_NUMBA` patched | right path called | `test_slicer_paths.py` (4) |
| T-07 | REQ-05, REQ-06 | ball area = pi*r^2, centered, width 2r, for 3 axes and 20 random normals; the check fails when the ball is 1 voxel off | `ball.nrrd`, generated planes | area within tolerance, centroid within 0.5 px | `test_ball.py` |
| T-08 | REQ-09, REQ-10 | brightness scale, no overflow on uint16, canvas size (even, largest plane), padding, error frame, write and read back | arrays | as listed | `test_video.py` (17) |
| T-09 | REQ-08, REQ-09, REQ-11 | one frame per row, equal even frame size, `Error in row 3`, `errors: 1`, FPS line printed, comment lines, same output with the warm-up, missing argument / file / folder | `Ex0.py` as a subprocess | exit code and output as listed | `test_demo.py` (11) |
| T-10 | REQ-02, REQ-04 | CSV generator output stays valid after rounding | generated CSVs | all rows valid | `test_make_csv.py` (4) |
| T-11 | NFR-01 | slicing FPS on 500^3 (random uint16 and `croissant.nrrd`) | planes of `bench_numba.py` | median of 30 FPS or more per case | `benchmarks/bench_numba.py` (not pytest) |

Test counts are in brackets; they add up to 270.

## 3. Risks and limits
- **Orientation (mirror or rotation)** is not caught by the ball test, because the ball is symmetric. It was checked by hand on `MRHead_1`: the voxel `M[111,125,74] = 68` matches 3D Slicer, and the head views show no mirror or rotation against the lecturer's reference pictures. The correlations with those pictures are moderate (0.5 to 0.9) because they have scale bars and a physical aspect ratio. This is a manual check, not an automatic test.
- Not tested yet: the volume size limit `< 2^10`, and a plane with a huge image (the encoder fails; undecided, no requirement).
- The 4-point order and the coplanarity threshold are assumptions (SRS section 2).
- FPS is measured by scripts and is noisy: background load changed results by 1.6x or more on this machine. The NumPy path alone fell below 30 FPS in one loaded round (16.8 FPS).
- Coverage is not measured. The lecturer's class volumes and CSVs are checked only by eye, through `scripts/run_examples.ps1` and the videos in `docs/experiment/`.
