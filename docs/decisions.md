# Design decisions

## Basics

1. **Language.** Python + NumPy: short code, ready libraries for NRRD and video.
2. **Slow reference first.** A slow version first, then a fast one, compared with `np.array_equal`. The reference is the oracle for the tests.
3. **Index order.** `M[x][y][z]` as pynrrd returns it. Verified: `M[111,125,74] == 68` matches 3D Slicer on `MRHead_1.nrrd`.
4. **CSV format.** A header and one row per frame: `n, x1,y1,z1, ..., x4,y4,z4`. `n` is 3 or 4.

## Algorithm

5. **Output shape.** `E[y, x]` with `floor(Dab)` rows and `ceil(Dbc)` columns. This is the order NumPy and video expect. With `ceil` rows (the assignment's `y < Dab`) the marker passes `b` and writes outside `E`.
6. **Advance first, then sample; column bounds.** The markers move before sampling, so `a` itself is not sampled and `d12 > 0` in the first row. A column outside `[0, cols)` is skipped, so nothing is written outside `E`.
7. **Rounding.** `floor(v + 0.5)`, not `round` (Python rounds exact halves to even). `1e-9` is used only when counting rows, columns and points, because of floating-point noise such as `7.999999999999999`.
8. **Triangle as a trapezoid.** A triangle is a trapezoid with `a' = a''`, so one code path serves both. The CSV point order `(a', b, a'', c)` is assumed.
9. **Invalid input.** The lecturer's Q&A gives `EPS = 0.001`, so an exception is raised when the legs differ by `EPS` or more, when `dist(a1,a2) > dist(b,c) + EPS` (the `+ EPS` is needed: in a square, float noise of about 3e-14 would otherwise reject valid rows, MEASURED), or when `a''` is farther than `EPS` from the plane. The Q&A gives no number for coplanarity; `EPS` is our reading. Our own extra rules: `Dab < 1` (an image of height 0 cannot be encoded to mp4), three points on a line (limit 0.5 voxel), and numbers that are not finite. With `EPS` no row of any CSV we have changes status (MEASURED). The demo draws an error frame and continues. A plane that misses the volume is a black image, not an error. The assignment has no limit on the image size. `MAX_IMAGE_SIDE = 2048` (rows and columns, one constant in `geometry.py`) is our own judgment call, to keep the canvas and the encoder safe: the canvas is the largest plane of the whole CSV, so one huge valid plane (frame 16 of the lecturer's `edge_cases.csv`, 5102 x 10199) made every frame 5102 x 10200 and the H.264 encoder failed intermittently, in about 40% of runs at that canvas (MEASURED on an idle machine: 8 of 19 trials; the encoder's hard limit is 16384 per side). The limit is not confirmed with the lecturer. A larger plane now gets the usual error frame ("image too large: R x C, maximum 2048 per side") and does not enter the canvas. `Ex0.py` also exits with a message, and deletes the partial mp4, when the encoder fails for any other reason or writes an empty file.
10. **Output type and units.** `E` is `float64`, which holds every input type exactly. Coordinates are array indices and the voxel spacing is ignored, so the picture can look stretched compared with 3D Slicer. `v(ac)` is divided by `Dac`, so a step is one unit on each leg.
11. **Video.** One brightness scale for all frames (`value * 255 / vmax`, with `vmax = M.max()` computed once). Every frame has the size of the largest plane, padded with black and never resized.

## Implementation

12. **Volume layout.** `prepare_volume` makes `M` C-contiguous once after loading. `slice_volume` raises `ValueError` otherwise, because a copy in every frame would be far slower.
13. **NumPy in 64-row blocks.** Computing 64 rows at a time gave the biggest speed-up of the NumPy version, about 2.5 times on the z-constant plane (MEASURED in a separate session, see `benchmarks/benchmarks.md`). Threads, `np.take` and an `int32` index gave no gain beyond the noise (about 10%) and were removed.
14. **Numba kernel and file structure.** Numba, foreseen from the start, is the default (`slicer_numba.py`). NumPy is the fallback (`slicer_numpy.py`) and the slow version is the reference (`slicer_slow.py`); `slicer.py` chooses. It is 3 to 7.5 times faster than NumPy with identical images (MEASURED). Rejected: `parallel=True` (more variance, and the serial kernel already passes 100 FPS) and NumPy only (it fell below 30 FPS under load). Cost: a dependency, a JIT delay of 0.2 to 1.6 s on the first call (`Ex0.py` compiles once before the timed loop, so the printed FPS excludes it), and no automatic bounds checking in Numba, so the checks before each access are required.

## Open

- Trapezoid point order in the CSV: not confirmed with the lecturer. The Q&A lists `(a1, a2, b, c)`, but only our order `(a', b, a'', c)` fits his own `lecturer_example.csv`: row 4 has legs 502.49 and 502.49 in our order, 300 and 400 in the listed order (MEASURED).
- Signed `int16` data versus the assignment's unsigned values.
- Array indices versus millimetres.
- Coplanarity threshold: the Q&A gives no number; `EPS = 0.001` is our reading (item 9).
- Volume size limit `1 <= sx,sy,sz < 2^10` of the assignment: not checked by `load_volume`.