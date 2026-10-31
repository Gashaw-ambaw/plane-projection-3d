# Design decisions
Format: decision - reason.

1. Language: Python + NumPy (Numba if performance requires) - short code, ready libraries for NRRD and video.
2. Development order: slow reference implementation first, then a fast one, and compare - the reference is the oracle for tests.
3. Array index order `M[x][y][z]` as returned by pynrrd - verified: `M[111,125,74] == 68` matches 3D Slicer's Data Probe on MRHead_1.nrrd.
4. Plane CSV: header + one row per frame: `n, x1,y1,z1, ..., x4,y4,z4`; n is 3 or 4, point 4 empty when n=3.
