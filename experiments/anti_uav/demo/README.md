# VALIDATION-ONLY demo — Anti-UAV (DFSC / GlobalTrack) reproduction

Launch (from `A:\AEROTRACK_OMEN`):

    venv\Scripts\python.exe -m streamlit run experiments\anti_uav\demo\app.py

Open http://localhost:8501 (Streamlit prints the URL). Stop with Ctrl+C.

Sidebar: **1 · Sequence** (67 IR val sequences with attributes), **2 · Checkpoint** (every run with `weights/last.pt`;
toggle *compare* for a second one on the same frames), **3 · Playback** (speed, stride, 3× inset on/off), optional live
tracking. Main page:

* **Frame inspector** — `⏮ Previous · ▶ PLAY / ⏸ PAUSE · Next ⏭ · First target · Middle · Last`, a *low-IoU frame* picker
  (failures are never hidden), the frame slider, the frame with the **GT box (green)** and **prediction (orange)**, a 3× zoom
  inset on the target, the metrics block (frame, IoU, centre error, confidence, sequence SA, status) and an IoU-over-time
  strip with the current frame marked. Continuous playback uses `st.fragment(run_every=…)` (only the viewer re-renders).
  It opens at the first tracked frame, not frame 0.
* **Annotated video** — the same frames and boxes rendered once into an H.264 MP4 (25 fps, from the start frame) and
  played by the browser's own player (play / pause / scrub / speed); cached under `demo/cache/videos/`.
* **Checkpoint summary** — method, seed, checkpoint SHA, this sequence's SA, the run's full-val mSA and the paper's
  Table V value with the delta.

Data contract (checked by `test_demo.py`): label index t ↔ `{t+1:06d}.jpg`; GT and predictions are `[x, y, w, h]` in original
image pixels (the tracker rescales its boxes back before saving), so nothing is converted for display.

Integrity: split fixed to `val` (`demo_core.SPLIT`, purpose `tune`; the loader refuses the test split — tested). Ground truth
is read only for the init box and for scoring. Predictions are the read-only files of `scripts/evaluate.py --split val`;
live tracking writes only to `demo/cache/`. No metric value is typed anywhere.

Tests (no training, no long evaluation):

    venv\Scripts\python.exe experiments\anti_uav\demo\test_demo.py      :: data contract, rendering, short MP4, headless app run
    python experiments\anti_uav\demo\browser_test.py                    :: real Chromium (system python + playwright): play/pause/step/compare/video

Dependencies (venv): `streamlit`, `imageio-ffmpeg` (bundled ffmpeg for the MP4) — `pip install -r experiments\anti_uav\demo\requirements.txt`.
