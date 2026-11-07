# Data: volumes and CSV files

All values below were read on 2026-11-07 from the NRRD headers and the CSV files (MEASURED), not typed from memory. Requirements: [SRS.md](SRS.md).

## 1. Volumes

| Volume | Where | Shape (x, y, z) | Header type | Voxel size (mm) | File size |
|---|---|---|---|---|---|
| `MRHead_1.nrrd` | repo `data/` and course folder (same size in bytes) | 256 x 256 x 130 | `short` (int16) | 1.0 x 1.0 x 1.3 | 6.9 MB |
| `ball.nrrd` | repo `data/` | 300 x 300 x 300 | `uint16` | none in the header | 0.13 MB |
| `onion.nrrd` | course folder only | 256 x 256 x 130 | `uint8` | 0.55 x 0.55 x 0.715 | 4.9 MB |
| `swiss_roll.nrrd` | course folder only | 256 x 256 x 130 | `uint32` | 1.1 x 1.1 x 1.43 | 15.1 MB |
| `biscuit_cake.nrrd` | course folder only | 256 x 256 x 130 | `uint16` | 1.0 x 1.0 x 1.3 | 9.1 MB |
| `croissant.nrrd` | course folder only | 500 x 500 x 500 | `uint8` | 0.37 x 0.37 x 0.37 | 63 MB |

- `ball.nrrd` is ours: a black ball (0, radius 100) on a white background (65535). `python scripts/make_ball.py` writes it.
- The five course volumes are in the space `left-posterior-superior`: array axis 0 runs to the posterior, axis 1 to the inferior, axis 2 (the short one, 130) to the right. So a plane of constant z is a sagittal view. Our code uses array indices and ignores the voxel size, so a picture can look stretched compared with 3D Slicer (decisions.md, item 10).

**Signed or unsigned.** The assignment speaks of unsigned integer values. `MRHead_1` is signed 16 bit. Its values are all non-negative (MEASURED by `Ex0.py`: minimum 0, maximum 279), so nothing is lost. `load_volume` does not check the type at all: it accepts any 3D array. A signed volume with negative values would show them as black, because the brightness scale is `value * 255 / max` and is clipped to 0..255. This is an open point (decisions.md, Open).

## 2. CSV files (data rows, without the header and the `#` lines)

| File | Where | Rows | Valid / invalid | Largest image (rows x cols) |
|---|---|---|---|---|
| `Ex0.csv` | repo | 4 | 3 / 1 (row 3) | 502 x 400 |
| `ball_rotate.csv` | repo | 60 | 60 / 0 | 520 x 521 |
| `mrhead_rotate.csv` | repo | 60 | 60 / 0 | 386 x 387 |
| `mrhead_sweep.csv` | repo | 60 | 60 / 0 | 255 x 255 |
| `lecturer_example.csv` | course folder | 4 | 3 / 1 (row 3) | 502 x 400 |
| `head.csv` | course folder | 280 | 280 / 0 | 255 x 256 |
| `onion.csv`, `swiss_roll.csv`, `biscuit_cake.csv` | course folder | 120 each | 120 / 0 | 255 x 256 |
| `croissant.csv` | course folder | 100 | 100 / 0 | 559 x 501 |
| `edge_cases.csv` | course folder | 16 (and 30 `#` comment lines) | 8 / 8 | 279 x 280 |

- `Ex0.csv` has the same text as `lecturer_example.csv` but is not byte-identical (230 and 228 bytes; the line endings differ).
- "Valid" is counted with the current code, including the image size limit of 2048 per side. In `edge_cases.csv` the invalid rows are 7 and 9 to 14 (the lecturer's cases) and 16 (a valid plane with a 5102 x 10199 image, rejected by the limit, decisions.md item 9). Rows 8 and 15 are valid.
- Every row of every file but frame 16 of `edge_cases.csv` has an image side below 560 (MEASURED), far below the limit.

## 3. Getting the course files

The lecturer's volumes and CSV files are course material and are not in this repo. Take them from the course folder of the assignment and unzip them into one folder that holds the `.nrrd` files and a `csv\` subfolder. Then:

```
powershell -ExecutionPolicy Bypass -File scripts\run_examples.ps1 -Drive <that folder>
```

It writes one mp4 per pair into `out_examples\` (not tracked by git). Add `-IncludeEdgeCases` to also run `edge_cases.csv`.

## 4. Regenerating our own files

```
python scripts/make_ball.py
python scripts/make_csv.py rotate --shape 256 256 130 --frames 60 --out data/mrhead_rotate.csv
python scripts/make_csv.py sweep  --shape 256 256 130 --frames 60 --out data/mrhead_sweep.csv
python scripts/make_csv.py rotate --shape 300 300 300 --center 150 150 150 --out data/ball_rotate.csv
```

The regenerated CSV files are not byte-identical to the committed ones: the committed files have 3 decimals and the generator now writes up to 6. The values agree to within 0.0005 (MEASURED for the three files above). `scripts/make_views.py` writes a CSV with the three central planes of a volume; no file in `data/` comes from it.
