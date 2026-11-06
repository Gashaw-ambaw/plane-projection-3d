"""Write a planes CSV with three axis-aligned planes through the center of a volume.

Rows: y = SY // 2, x = SX // 2, z = SZ // 2. Each plane covers the whole volume, with the
same corner pattern as the reference views of MRHead (256 x 256 x 130 gives centers 128, 128, 65).

Example:
  python scripts/make_views.py --shape 256 256 130 --out data/onion_views.csv
"""
import argparse
from pathlib import Path

from make_csv import write_csv


def views_planes(shape):
    sx, sy, sz = shape
    cx, cy, cz = sx // 2, sy // 2, sz // 2
    mx, my, mz = sx - 1, sy - 1, sz - 1
    return [
        [(0, cy, mz), (mx, cy, mz), (0, cy, 0), (mx, cy, 0)],
        [(cx, 0, mz), (cx, my, mz), (cx, 0, 0), (cx, my, 0)],
        [(0, 0, cz), (0, my, cz), (mx, 0, cz), (mx, my, cz)],
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shape", type=int, nargs=3, required=True, metavar=("SX", "SY", "SZ"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    write_csv(Path(args.out), views_planes(args.shape))
    print(f"wrote {args.out}: 3 rows, shape {tuple(args.shape)}")


if __name__ == "__main__":
    main()
