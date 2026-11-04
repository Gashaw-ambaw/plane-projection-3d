"""Write data/ball.nrrd: a black ball (0) on a white background (65535), uint16, gzip.

Run from the repo root: python scripts/make_ball.py
"""
import sys
from pathlib import Path

import nrrd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from projection.synthetic import make_ball_volume  # noqa: E402

SIZE = 300
RADIUS = 100
OUT = ROOT / "data" / "ball.nrrd"


def main():
    M = make_ball_volume(SIZE, RADIUS, background=65535, ball=0, dtype=np.uint16)
    nrrd.write(str(OUT), M, header={"encoding": "gzip"})
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes), shape {M.shape}, dtype {M.dtype}")


if __name__ == "__main__":
    main()
