"""Generate a planes CSV in the project format, decision 4 (docs/decisions.md).

Every row has 4 points (a', b, a'', c). Two modes:
  sweep   planes with constant z, moving from z=0 to z=sz-1
  rotate  planes through the center of the volume, rotating around an axis (half a turn)

Example:
  python scripts/make_csv.py rotate --shape 256 256 130 --frames 60 --out data/mrhead_rotate.csv
Use --bad-row N to make row N an invalid plane (not isosceles), to show an error frame.
Use --center X Y Z to rotate around another point. The default is the middle of the
volume, (shape - 1) / 2 (MRHead). For the ball of make_ball.py use --center 150 150 150
(size // 2 on every axis):
  python scripts/make_csv.py rotate --shape 300 300 300 --center 150 150 150 --out data/ball_rotate.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from projection.geometry import normalize_points, validate  # noqa: E402
from projection.synthetic import plane_through_center  # noqa: E402

HEADER = ["Number of Points (3 or 4)"] + [f"{axis}{i}" for i in range(1, 5) for axis in "xyz"]
BAD_PLANE = [(0, 0, 0), (10, 0, 0), (0, 20, 0)]  # Dab = 10 but Dac = 20


def sweep_planes(shape, frames):
    sx, sy, sz = shape
    zs = np.linspace(0, sz - 1, frames)
    # a'->b runs along y and a''->c too, so the legs have equal length.
    return [
        [(0, 0, z), (0, sy - 1, z), (sx - 1, 0, z), (sx - 1, sy - 1, z)] for z in zs
    ]


def rotate_planes(shape, frames, axis, center=None):
    if center is None:
        center = (np.array(shape) - 1) / 2
    half_width = np.ceil(np.linalg.norm(shape) / 2)  # half the diagonal covers the volume
    up = np.eye(3)[axis]
    first, second = [i for i in range(3) if i != axis]

    planes = []
    for k in range(frames):
        angle = np.pi * k / frames  # half a turn: the other half repeats the planes
        normal = np.zeros(3)
        normal[first], normal[second] = np.cos(angle), np.sin(angle)
        planes.append(plane_through_center(normal, center=center, half_width=half_width, up=up))
    return planes


def write_csv(path, planes, bad_row=None):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for number, pts in enumerate(planes, start=1):
            if number == bad_row:
                pts = BAD_PLANE
            cells = [len(pts)]
            for p in pts:
                cells.extend(f"{float(x):.6f}".rstrip("0").rstrip(".") for x in p)
            if number != bad_row:
                # Validate the numbers as written (rounded), because the demo reads those.
                written = [tuple(float(x) for x in cells[1 + 3 * i:4 + 3 * i]) for i in range(len(pts))]
                validate(*normalize_points(written))  # a generated plane must be valid
            cells.extend([""] * (13 - len(cells)))  # point 4 empty for a triangle
            writer.writerow(cells)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["sweep", "rotate"])
    parser.add_argument("--shape", type=int, nargs=3, required=True, metavar=("SX", "SY", "SZ"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=2, help="rotation axis")
    parser.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="rotation center (default: the middle of the volume)")
    parser.add_argument("--bad-row", type=int, help="make this row (1-based) an invalid plane")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.mode == "sweep":
        planes = sweep_planes(args.shape, args.frames)
    else:
        center = None if args.center is None else np.array(args.center)
        planes = rotate_planes(args.shape, args.frames, args.axis, center)
    write_csv(args.out, planes, args.bad_row)
    print(f"wrote {args.out}: {len(planes)} rows, mode {args.mode}, shape {tuple(args.shape)}")


if __name__ == "__main__":
    main()
