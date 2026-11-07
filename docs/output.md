# Output of Ex0.py

`python Ex0.py file.nrrd file.csv file.mp4` writes the video and prints this summary (here for `data/MRHead_1.nrrd` and `data/Ex0.csv`):

```
Volume: shape=(256, 256, 130), dtype=int16, min=0, max=279
Frames written: 4 (errors: 1)
  Error in row 3: not isosceles: Dab = 497.3932, Dac = 476.8648 (difference not below 0.001)
Canvas: 502 rows x 400 cols
Output: out.mp4 (<size> bytes, 30 FPS playback)
Slicing: 3 slices in <s> s = <F> FPS (slice_volume calls only)
Note: with few rows this number is noisy (first call, small sample); the official FPS is measured in benchmarks/bench_final.py.
```

`<size>`, `<s>` and `<F>` vary between runs.

| Line | Meaning |
|---|---|
| `Volume` | shape (x, y, z), type, minimum and maximum; the maximum sets the brightness scale. |
| `Frames written` | one frame per CSV data row; `errors` counts the rows that became error frames. |
| `Error in row N` | why row N (data rows only) is invalid. |
| `Canvas` | size of every frame: the largest image, even sides. Smaller images are centered; a rejected row does not enlarge it. |
| `Output` | mp4 path, size, playback speed (fixed, 30 FPS). |
| `Slicing` | valid rows and FPS of the `slice_volume` calls only (no compile, encoding or writing). With few rows the number is noisy and must not be compared with the 30 FPS requirement (NFR-01); the official measurement is in `benchmarks/` and [experiment/README.md](experiment/README.md). |

**The video:** one frame per row, in CSV order: the slice, or for an invalid row white text `row N: <reason>` on black ([SRS.md](SRS.md), REQ-08 to REQ-11).

**Failures:** a missing argument or file prints a usage message and exits with a non-zero code. If the video cannot be written, the program prints `could not write the video: ...` with ffmpeg's text, deletes the partial mp4 and exits with a non-zero code.
