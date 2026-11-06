# החלטות שלב 1: האלגוריתם על נייר

תאריך: 3.11.2026. לשילוב ב-`docs/decisions.md` (שם כבר יש D1 עד D4: Python+NumPy, גרסת ייחוס איטית קודם, סדר אינדקסים M[x][y][z], פורמט CSV). כאן ההחלטות החדשות, ממוספרות S1-1 ואילך.

עיקרון מנחה (החלטה שלי): **פשוט ועובד קודם, שיפור אחר כך.** הדרישה: 30 FPS חובה, מעל 100 אין טעם לשפר, והעיצוב חשוב יותר מהמימוש.

נקודות ייחוס: "המטלה" היא `I2SE_Exe1.docx.pdf`. בה E היא Dab\*Dbc, הלולאה החיצונית `y<Dab`, הפנימית `x1<=d12`, והכתיבה `E[x][y]`.

## S1-1. סדר E: `E[y, x]`, גודל `(floor(Dab), ceil(Dbc))`
- **במטלה:** `E[x][y]`.
- **החלטה:** y הוא השורה, x הוא העמודה, והאחסון `E[y, x]` (שורה, עמודה). המשמעות של x ו-y נשארת כמו במטלה.
- **למה:** כך מוגדר מערך ב-NumPy, כך מצפים OpenCV/imageio/ffmpeg (גובה, רוחב), ו-"Dab\*Dbc" במטלה מתאים ל-Dab שורות ו-Dbc עמודות. אחרת התמונה יוצאת מסובבת ויש צורך ב-`E.T`.
- **דוגמה:** Dab=5, Dbc=8 נותן `shape == (5, 8)`.

## S1-2. מתקדמים ואז דוגמים (כמו המטלה)
- a1 ו-a2 מתקדמים צעד **לפני** הדגימה, ולכן הנקודה a עצמה לא נדגמת (פיקסל אחד).
- **למה:** כך כתוב במטלה, ובשורה הראשונה d12 כבר גדול מאפס ולכן `dp = vector/d12` לא מחלק באפס.

## S1-3. מספר השורות: `floor(Dab)`
- **במטלה:** `y<Dab` עם y שלם, כלומר `ceil(Dab)` איטרציות (448 כש-Dab=447.2).
- **סטייה מכוונת:** `floor(Dab)` שורות.
- **למה:** עם ceil הסמן a1 עובר את b, d12 גדול מ-Dbc והעמודה יוצאת שלילית (כתיבה מחוץ ל-E). עם floor לא חורגים מהמשולש. המחיר: פחות משורה אחת.
- **קצה:** Dab קטן מ-1 נותן אפס שורות, ולכן הוא קלט לא תקין (ראו S1-8).

## S1-4. מספר העמודות: `ceil(Dbc)`
- Dbc לא שלם (למשל 400.2) נותן 401 עמודות.

## S2-8. Numba kernel for `slice_volume`, NumPy code kept as the fallback

Language: English (this section only).

- **Decision:** `slice_volume` keeps its signature and its checks (`ValueError` for a non-C-contiguous `M` and for `block_rows < 1`). After the checks, it calls the Numba kernel `fill_image` (`src/projection/slicer_numba.py`) if numba can be imported (`HAVE_NUMBA`), otherwise `_slice_numpy` (the previous code, unchanged, in `src/projection/slicer_numpy.py`; `slicer.py` keeps only `slice_volume`, the Numba import check and the small `_slice_numba` wrapper). The kernel is a plain loop over rows and columns with the same arithmetic as the NumPy version (`a1 + k*dir1`, `floor(p + 0.5)`, the same start column), so the image is identical (S1-1 to S1-10 are not touched). `numba` is in `requirements.txt`. No `parallel=True`.
- **Why:** the NumPy version builds about 35 to 40 temporary arrays per block of rows (counted in the code, not profiled); the loop does one pass per cell. Measured: about 3 to 7.5 times the FPS (ratio of medians 2.97 to 7.54 over 4 runs, 2 on a loaded and 2 on a quieter machine), identical images (`benchmarks/benchmarks.md`, section "Numba kernel vs NumPy fallback"). D1 already said "Numba if performance requires".
- **Alternatives:**
  - NumPy only (previous state): simplest, no dependency, already above 30 FPS in earlier runs; but under load its median was 26.0 to 76.8 FPS (lowest round 16.8, below the 30 FPS requirement) and the z-constant plane reached 100 FPS only on an idle machine.
  - `parallel=True` (prange over rows): measured in a throw-away test at 4.8 to 5.9 times NumPy against 2.6 to 4.3 for serial (not re-measured here); more complexity and more variance, and serial already passes the 100 FPS target in most rows.
  - Small NumPy changes (one unsigned compare in `inside`, in-place arithmetic, one `flatnonzero`): ESTIMATE only, expected to be inside the measurement noise (about 10%), so they could not be shown to help. Not done.
- **Consequences:**
  - A dependency (numba and llvmlite, installed with `pip install -r requirements.txt`).
  - JIT delay: 0.18 to 1.58 s on the first call for each volume dtype in a process (MEASURED, see benchmarks), not hidden in the FPS numbers. `cache=False`, so it is paid in every run of `Ex0.py`.
  - Two code paths that must stay equal. The tests run every `slice_volume` test with both paths (fixture `slicer_path` in `tests/conftest.py`), and `tests/test_slicer_paths.py` checks the choice between the paths (forced fallback, forced Numba).
  - `block_rows` has no effect on the Numba path (it is still validated).
