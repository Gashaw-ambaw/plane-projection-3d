# conftest.py is loaded automatically by pytest before the tests run.
# Our code lives in the "src" folder, which Python does not search by default.
# The line below adds "src" to Python's import path,
# so a test can write: from projection.volume_loader import load_volume

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))