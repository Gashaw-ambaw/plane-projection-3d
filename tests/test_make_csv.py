import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_csv import rotate_planes, sweep_planes, write_csv  # noqa: E402
from projection.csv_planes import read_planes  # noqa: E402
from projection.geometry import normalize_points, validate  # noqa: E402
from projection.synthetic import plane_through_center  # noqa: E402


def test_written_rotate_rows_are_valid_after_reading_back(tmp_path):
    path = tmp_path / "rotate.csv"
    write_csv(path, rotate_planes((256, 256, 130), 12, axis=2))

    planes = read_planes(path)
    assert len(planes) == 12
    for plane in planes:
        assert plane.error is None
        validate(*normalize_points(plane.points))


def test_written_sweep_rows_are_valid_after_reading_back(tmp_path):
    path = tmp_path / "sweep.csv"
    write_csv(path, sweep_planes((100, 120, 30), 5))

    assert all(p.error is None for p in read_planes(path))


def test_rows_of_arbitrary_normals_stay_valid_after_rounding(tmp_path):
    # Rounding to 3 decimals broke about 5% of such planes; 6 decimals must not.
    rng = np.random.default_rng(0)
    planes = [
        plane_through_center(rng.normal(size=3), center=rng.uniform(0, 500, 3),
                             half_width=rng.uniform(10, 400))
        for _ in range(300)
    ]
    path = tmp_path / "random.csv"
    write_csv(path, planes)

    for plane in read_planes(path):
        validate(*normalize_points(plane.points))


def test_bad_row_is_written_as_an_invalid_plane(tmp_path):
    path = tmp_path / "bad.csv"
    write_csv(path, sweep_planes((50, 50, 50), 3), bad_row=2)

    planes = read_planes(path)
    validate(*normalize_points(planes[0].points))
    with pytest.raises(ValueError, match="not isosceles"):
        validate(*normalize_points(planes[1].points))
