# plane-projection-3d

Ex1 (Intro to Software Engineering, Ariel University): given a 3D raster volume (NRRD)
and a plane defined by 3 or 4 points, compute the 2D image of the volume projected on that plane,
and render a video with one frame per plane listed in a CSV file.

> Status: Stage 0 - repository skeleton. Problem definition, scope and SRS are written in later stages.

## Quick start (Windows PowerShell)
```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
```

## Planned demo
```
python Ex0.py data/MRHead_1.nrrd data/Ex0.csv out.mp4
```

## Structure
- `src/projection/` - source code
- `tests/` - automated tests (pytest)
- `docs/` - SRS, algorithm description, test strategy, `decisions.md`
- `data/` - NRRD volumes and CSV plane files
- `AI_USAGE.md` - log of AI usage (course policy)
