from pathlib import Path

import pytest

from projection.csv_planes import csv_row, read_planes

CSV = Path(__file__).resolve().parents[1] / "data" / "Ex0.csv"


def test_read_lecturer_csv():
    # Rows 1 and 3 are triangles, rows 2 and 4 are trapezoids.
    planes = read_planes(CSV)

    assert [len(p.points) for p in planes] == [3, 4, 3, 4]
    assert all(p.error is None for p in planes)
    assert planes[0].points[0] == (200.0, 200.0, 0.0)


def test_bad_row_is_kept_as_error_and_does_not_stop_the_others(tmp_path):
    path = tmp_path / "planes.csv"
    path.write_text(
        "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4\n"
        "3,0,0,0,1,0,0,0,1,0,,,\n"
        "3,0,0,abc,1,0,0,0,1,0,,,\n"
        "5,0,0,0,1,0,0,0,1,0,,,\n"
        "4,0,0,0,1,0,0,0,1,0,,,\n"
        "\n"
        "3,0,0,0,1,0,0,0,1,0,,,\n"
    )
    planes = read_planes(path)

    assert len(planes) == 5  # the blank line is skipped
    assert planes[0].error is None
    assert "not a number" in planes[1].error
    assert "3 or 4" in planes[2].error
    assert "coordinates" in planes[3].error
    assert planes[4].error is None


def test_csv_row_returns_points_and_raises_for_bad_rows(tmp_path):
    assert csv_row(CSV, 4)[1] == (200.0, 0.0, 500.0)

    with pytest.raises(ValueError, match="does not exist"):
        csv_row(CSV, 5)

    path = tmp_path / "bad.csv"
    path.write_text("header\n3,0,0,x,1,0,0,0,1,0,,,\n")
    with pytest.raises(ValueError, match="row 1"):
        csv_row(path, 1)
