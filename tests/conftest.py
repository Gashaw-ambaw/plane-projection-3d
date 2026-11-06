# conftest.py is loaded automatically by pytest before the tests run.
# Our code lives in the "src" folder, which Python does not search by default.
# The line below adds "src" to Python's import path,
# so a test can write: from projection.volume_loader import load_volume

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(params=["numba", "numpy"])
def slicer_path(request, monkeypatch):
    """Run a test twice: with the Numba kernel, and with the NumPy fallback.

    A test file asks for it with: pytestmark = pytest.mark.usefixtures("slicer_path")
    """
    from projection import slicer

    if request.param == "numba":
        pytest.importorskip("numba")
    # slice_volume reads this flag at every call, so the test sees the chosen path.
    monkeypatch.setattr(slicer, "HAVE_NUMBA", request.param == "numba")
    return request.param
