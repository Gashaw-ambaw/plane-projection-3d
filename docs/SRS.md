# Software Requirements Specification: plane projection of a 3D volume

## 1. Introduction
**Purpose.** This document says what the system shall do, in statements that a test or a measurement can check. How it computes: [algorithm.md](algorithm.md). Why: [decisions.md](decisions.md). How it is verified: [testing.md](testing.md).

**Scope.** A command-line program that cuts a 3D volume with the planes of a CSV file and writes one video frame per plane. Out of scope: voxel spacing in mm, interpolation, a GUI, formats other than NRRD, CSV and mp4.

**Definitions.** *Volume* `M[x][y][z]`: a 3D array of integer values (unsigned 8/16/32 bit; signed 16 bit, as in MRHead_1, is also accepted). *Plane*: 3 points `a,b,c` (isosceles triangle) or 4 points `a',b,a'',c` (isosceles trapezoid, `a'→b` and `a''→c` the equal legs). `Dab = dist(a',b)`, `Dac = dist(a'',c)`, `Dbc = dist(b,c)`. *Slice* `E[y,x]`: the 2D image of one plane. `EPS = 0.001` (the lecturer's Q&A).

**References.** The assignment text, the lecturer's Q&A, 3D Slicer.

## 2. Overall description
**Environment.** Windows 11, Python 3.13; numpy, pynrrd, imageio(+ffmpeg), Pillow; numba is optional.
**Constraints.** Implementation language: Python ([decisions.md](decisions.md), item 1). A 500^3 volume must be sliced at 30 FPS at least (a must), about 100 FPS expected.
**Assumptions.** (1) The 4-point order is `(a',b,a'',c)`; this is not confirmed with the lecturer, it is the only order that fits row 4 of his `lecturer_example.csv`. (2) Coordinates are array indices, not mm. (3) Volumes satisfy `1 <= sx,sy,sz < 2^10`; the program does not check it.

## 3. Specific requirements
### 3.1 Functional

| ID | Requirement | Verified by |
|---|---|---|
| REQ-01 | The system shall load an NRRD file as `M[x,y,z]` and reject a file that is not 3D with a `ValueError`. | `test_volume_loader.py` (shape, probe value, non-3D file) |
| REQ-02 | The system shall read a CSV with a header and one plane per row (3 or 4 points), skipping blank lines and lines whose first cell starts with `#`. | `test_csv_planes.py` |
| REQ-03 | The system shall accept a plane only if `abs(Dab-Dac) < EPS`, `dist(a',a'') <= Dbc + EPS`, `a''` is within `EPS` of the plane of `a',b,c`, all coordinates are finite, `Dab >= 1`, and `a',b,c` are not on one line (distance of `c` from the line `a'b` at least 0.5). | `test_geometry.py` |
| REQ-04 | The system shall raise a `ValueError` for an invalid plane, and an invalid row shall not stop the processing of the other rows. | `test_geometry.py`, `test_csv_planes.py`, `test_demo.py` |
| REQ-05 | The system shall produce an image of `floor(Dab)` rows and `ceil(Dbc)` columns. | `test_geometry.py`, `test_slicer.py`, `test_ball.py` |
| REQ-06 | The system shall set each pixel to the voxel nearest to its position (`floor(p+0.5)` per axis) and to 0 where the position is outside the volume. | `test_slicer_slow.py`, `test_slicer.py`, `test_ball.py` |
| REQ-07 | The system shall treat a triangle as a trapezoid with `a' = a''`. | `test_geometry.py` |
| REQ-08 | The command `python Ex0.py file.nrrd file.csv file.mp4` shall write an mp4 with one frame per data row, in CSV order, and shall exit with a message when an argument or an input file is missing. | `test_demo.py` |
| REQ-09 | The system shall write, for an invalid row, a frame with the text `row n: <reason>` instead of an image. | `test_video.py`, `test_demo.py` |
| REQ-10 | All frames of a video shall have the same size (the largest plane, sides rounded up to even) and one brightness scale (`value*255/max(M)`). | `test_video.py`, `test_demo.py` |
| REQ-11 | The demo shall print the volume info, the number of frames, each error with its row number, and the FPS of the slicing calls only (the Numba compile excluded). | `test_demo.py` |

### 3.2 External interfaces
- **CLI:** `python Ex0.py file.nrrd file.csv file.mp4`; exit code 0 on success, a message and a non-zero code otherwise.
- **NRRD:** 3D, indexed `M[x][y][z]` as pynrrd returns it.
- **CSV:** header, then `n,x1,y1,z1,...,x4,y4,z4` (`n` = 3 or 4; the last three cells are empty for a triangle).
- **mp4:** H.264, played at 30 FPS, one frame per data row; error frames are black with white text.

### 3.3 Non-functional

| ID | Requirement | Verified by |
|---|---|---|
| NFR-01 | On a 500^3 volume, slicing shall run at a median of 30 FPS or more (goal: about 100 FPS) on the Numba path. | `benchmarks/bench_numba.py`, `benchmarks/bench_final.py`; results in [benchmarks.md](../benchmarks/benchmarks.md) (every median above 80 FPS, above 100 in 18 of 20 cases; one disturbed round fell to 23.9 FPS) |
| NFR-02 | Every slicer (Numba, NumPy) shall return an image equal (`np.array_equal`) to the slow reference. | `test_slicer.py`, `test_slicer_paths.py`, `test_ball.py` |
| NFR-03 | The system shall run on Python 3.13 without Numba, with the same images. | `test_slicer_paths.py`; the NumPy path alone can fall below 30 FPS under load (benchmarks.md) |

Open points (point order, coplanarity threshold, maximum image size, volume size limit): see "Open" in [decisions.md](decisions.md).
