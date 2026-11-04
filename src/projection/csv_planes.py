import csv
from collections import namedtuple

# One CSV row: either points (a list of 3 or 4 (x, y, z) tuples) or an error message.
Plane = namedtuple("Plane", ["points", "error"])


def read_planes(path):
    """Read the planes CSV (header skipped) and return one Plane per data row.

    A row that cannot be read is kept as Plane(None, message) and does not stop
    the other rows from being read.
    """
    planes = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue  # blank line
            planes.append(_parse_row(row))
    return planes


def csv_row(path, number):
    """Return the points of CSV data row `number` (1-based); raise ValueError if it has an error."""
    planes = read_planes(path)
    if not 1 <= number <= len(planes):
        raise ValueError(f"row {number} does not exist (the file has {len(planes)} rows)")
    plane = planes[number - 1]
    if plane.error:
        raise ValueError(f"row {number}: {plane.error}")
    return plane.points


def _parse_row(row):
    try:
        count = int(row[0])
    except ValueError:
        return Plane(None, f"the first cell must be 3 or 4, got {row[0]!r}")
    if count not in (3, 4):
        return Plane(None, f"the first cell must be 3 or 4, got {count}")

    cells = row[1:1 + 3 * count]
    if len(cells) < 3 * count or any(cell.strip() == "" for cell in cells):
        return Plane(None, f"expected {3 * count} coordinates for {count} points")
    try:
        values = [float(cell) for cell in cells]
    except ValueError:
        return Plane(None, "a coordinate is not a number")

    points = [tuple(values[i:i + 3]) for i in range(0, len(values), 3)]
    return Plane(points, None)
