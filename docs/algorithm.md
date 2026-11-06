# Algorithm: slice of a 3D volume along a plane

**Input:** a volume `M[x][y][z]` (NRRD, unsigned 8/16/32 bits or signed 16 bits) and a CSV with one plane per row. A plane is 3 points `a, b, c` (isosceles triangle) or 4 points `a', b, a'', c` (isosceles trapezoid).
**Output:** an mp4 with one frame per CSV row. Each frame is the 2D image `E[y, x]` of the volume cut by that plane. An invalid row gives an error frame, and the other frames stay correct.

## Pipeline

```
NRRD -> load_volume -> prepare_volume --+
CSV  -> read_planes --------------------+-> per plane: validate -> image size -> slice -> E
                                           E -> normalize -> pad to canvas -> write_video -> mp4
                                           bad row -> error frame
```

1. **Load.** pynrrd reads the file (must be 3D, `M[x, y, z]`). `prepare_volume` makes the array C-contiguous once, so `slice_volume` never copies per frame.
2. **Read planes.** Each row is `n, x1,y1,z1, ...` with `n` = 3 or 4. An unreadable row is kept as an error and does not stop the others. Order with 4 points: `(a', b, a'', c)`. With 3 points `a' = a'' = a`.
3. **Validate.** `Dab = |b - a'|`, `Dac = |c - a''|`, `Dbc = |c - b|`. With `EPS = 0.001` (the lecturer's Q&A), required: `|Dab - Dac| < EPS`, `|a'a''| <= Dbc + EPS`, `a''` on the plane of `a', b, c` (within `EPS`). Our own rules: `Dab >= 1`, `a', b, c` not on one line, all numbers finite. Otherwise `ValueError`.
4. **Image size.** `rows = floor(Dab + 1e-9)`, `cols = ceil(Dbc - 1e-9)`. The `1e-9` only absorbs float noise (e.g. `7.999999999999999`).
5. **Slice** (below).
6. **Video** (`Ex0.py`, two passes). Pass 1 computes all sizes; the canvas is the largest rows and columns, rounded up to even. Pass 2 slices each plane, scales `0..max(M)` to `0..255` (one maximum for the whole volume), centers the image on a black canvas and writes it at 30 FPS. A bad row gives a black frame with the error text.

## Slice (step 5)

`a1 = a'`, `a2 = a''`, `v(ab) = (b - a1) / Dab`, `v(ac) = (c - a2) / Dac` (one voxel per step). `E = zeros(rows, cols)`. For every row `y = 0 .. rows - 1`:

1. **Advance first:** `a1 += v(ab)`, `a2 += v(ac)`. So `a` itself is not sampled.
2. `d12 = |a2 - a1|` is the row length, `dp = (a2 - a1) / d12` the unit step along it.
3. For `x1 = 0 .. floor(d12)` (end included):
   - `p = a1 + x1 * dp`, voxel `= floor(p + 0.5)` on each axis (nearest, no interpolation; `round` is not used because it rounds exact halves to even);
   - column `= floor(Dbc/2 - d12/2 + 0.5) + x1` (centers the row); skip if outside `[0, cols)`;
   - `E[y, col] = M[voxel]` if the voxel is inside `M`, otherwise `0`.

`d12` grows linearly from `|a'a''|` (0 for a triangle) to `Dbc`, so the shape is a triangle or a trapezoid.

## Implementations

All give the same image (`np.array_equal`). `slice_volume` in `slicer.py` checks the input and picks Numba if installed, otherwise NumPy.

| File | Role |
|---|---|
| `slicer_slow.py` | Plain loops. The reference for the tests. |
| `slicer_numpy.py` | Blocks of 64 rows with masks. Fallback without Numba. |
| `slicer_numba.py` | The same loops, compiled. Default, fastest. |

## Edge cases and complexity

- A plane partly or fully outside the volume leaves those cells at 0 (black image, not an error).
- A non-3D or non-contiguous volume raises `ValueError`.
- `E` is `float64`, which holds every input type exactly.
- Time is proportional to the output pixels (about `Dab x Dbc`), not to the volume size: only the voxels on the plane are read. Speeds: `benchmarks/benchmarks.md`. Reasons for each choice: `docs/decisions.md`.