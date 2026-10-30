# Operating guide — unified workspace (Windows, OMEN)

Every command below was run on 2026-10-13 in `A:\AEROTRACK_OMEN` with the outcome noted. Run them from the repository root.

## 1. Environment

The project venv is `venv\` (Python 3.11.9, torch 2.13.0+cu130, Ultralytics 8.4.115, Streamlit 1.64.0). Nothing new was installed for
the workspace. Activation is optional: every command calls `venv\Scripts\python.exe` directly.

```bat
venv\Scripts\activate.bat                     :: optional
venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'.'); from aerolab.health import environment; print(environment())"
```
Verified: CUDA available, RTX 5050 Laptop (8 GB). Without CUDA, detector and GlobalTrack inference fall back to CPU (correct but slow);
forecasters always run on CPU. To simulate a machine without CUDA, set `CUDA_VISIBLE_DEVICES=-1`; an empty value does not
hide the GPU on this Windows setup. TensorRT is not importable in this venv, so the TensorRT engines are registered but not runnable from
the studio. The environment freeze is in `docs/environments/venv_freeze_2026-10-05.txt` (HPC: `results/artifacts/hpc_recovery_20261013/aerotrack_server/environments/at311_freeze.txt`).

## 2. Launch / stop the studio

```bat
studio.bat                 :: http://localhost:8501   (studio.bat 8600 for another port)
```
Verified: environment line printed, server bound to 127.0.0.1 only, `/_stcore/health` = ok, page HTTP 200. **Stop**: Ctrl+C in that
window, or close it. Equivalent without the launcher:
`venv\Scripts\python.exe -m streamlit run app\app.py --server.address localhost`.

### Flight Deck (the main interface since 2026-10-13)

The app opens on the **Flight Deck** home page. The five primary sections are Flight Deck, Test Bench, Live Comparison, Results &
Failures and Model Vault; the pages described above live under *Workspace tools*. The step-by-step first experiment and comparison,
the seven categories, presets and rules are in `docs/workspace/FLIGHT_DECK.md`.

## 3. Registry

```bat
venv\Scripts\python.exe scripts\build_workspace_registry.py            :: ~1 min first time (hashes ~80 GB), seconds afterwards
venv\Scripts\python.exe scripts\build_workspace_registry.py --rehash   :: re-hash every workspace file
venv\Scripts\python.exe scripts\verify_adapters.py                     :: load + tiny VAL inference of every runnable model except epoch checkpoints (~5 min; --all ~25 min)
venv\Scripts\python.exe scripts\verify_adapters.py --family aerotrack.forecaster.c1
venv\Scripts\python.exe scripts\render_workspace_docs.py               :: regenerate the result tables in research/*/README.md
```
Verified: 314 model entries, 498 files (372 weights, 20 exports, 106 detection caches), 0 unclassified, 0 integrity mismatches;
228 / 228 runnable checkpoints passed load + inference (`--all`, incl. 126 intermediate epochs; ~25 min).

**Register a model**: put it where its series stores runs, add or extend a family in `results/registry/catalog/families.json` if no pattern
matches, rebuild, verify its adapter, rebuild again (`docs/workspace/ARCHITECTURE.md` §Adding a new model).

## 4. Integrity checks (read-only)

```bat
venv\Scripts\python.exe scripts\hpc\recovery_manifest.py results\artifacts\hpc_recovery_20261013 --out results\runs\studio\health\recovery_check
venv\Scripts\python.exe scripts\restore_from_recovery.py                :: dry run: what would be copied out of the backup
```
The first command re-hashes the whole backup against the HPC-side hashes (~1 min) and writes its outputs outside the backup. **Without
`--out` it rewrites the manifest files inside the backup directory**, which is the original behaviour. The studio's Health page runs the
`--out` form. `--apply` on the second command copies files and never overwrites existing ones.

## 5. A small allowed evaluation (VAL / TRAIN only)

In the studio: **Inference** → model version → dataset → VAL sequence → *Run inference*. From Python:

```python
from aerolab.evaluation import evaluate_sot, evaluate_forecast
evaluate_sot("aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last", "cst2025", "val", "IR", ["building_41"], max_frames=100)
evaluate_sot("antiuav_repro.globaltrack:normal_ir@last", "antiuav2021", "val", "IR", ["20190925_101846_1_4"], max_frames=50)
evaluate_forecast("aerotrack.forecaster.c1:c1_tfm_abs_s0@weights", ["cst_val_building_41"], max_anchors=300)
```
Each call writes `results/runs/studio/evals/<id>/` (manifest, per-sequence results, predictions, log) and one line in `results/runs/studio/index.jsonl`.
`split="test"` raises `TestSplitLocked` before anything is loaded. Verified: the X1 full-sequence SA on `building_41` / `cn_mountains_21`
(74.12 / 73.50) equals the tracker run on the HPC-cached detections to within 0.02.

## 6. Saved results

Studio → **Experiments** (filters, per-sequence chart, the verbatim source of every historical record) or **Compare** (protocol-matched
tables with paired bootstrap deltas). Files: `results/registry/experiments.jsonl` (historical), `results/runs/studio/index.jsonl` (studio runs).

## 7. Tests

```bat
venv\Scripts\python.exe -m unittest tests.test_workspace_registry -v      :: 30 tests, ~2 s
venv\Scripts\python.exe -m unittest tests.test_workspace_adapters -v      :: 12 tests, ~20 s (local data; CUDA if present, plus one CPU-fallback run)
venv\Scripts\python.exe -m unittest tests.test_studio_ui -v               :: 10 tests, ~20 s (headless Streamlit AppTest)
venv\Scripts\python.exe -m unittest tests.test_research_program tests.test_cst_protocol   :: existing suites (34 tests)
venv\Scripts\python.exe experiments\anti_uav\demo\test_demo.py            :: existing Direction B demo smoke test
venv\Scripts\python.exe -m unittest tests.test_flight_deck -v               :: Flight Deck core (runner, presets, replay, evidence, streaming)
venv\Scripts\python.exe -m unittest tests.test_flight_deck_ui -v            :: Flight Deck UI journeys (AppTest)
venv\Scripts\python.exe toolserify_evaluations.py                        :: fresh vs recorded VAL evaluation per checkpoint (~6 min)
```
`pytest` is not installed; the repository uses `unittest`.

## 8. Troubleshooting

| symptom | cause / action |
|---|---|
| "results/registry/models.json not found" | run `scripts\build_workspace_registry.py` |
| model shown as "recorded results only" | its family has no adapter (repair grid, historical, LSTM, exports) — see its records under Experiments |
| "no usable workspace copy … with run_meta.json" | the checkpoint is only in the backup (restore with `scripts\restore_from_recovery.py --apply`) or its run directory lacks `run_meta.json` |
| "SHA-256 … does not match the registry" | the file changed since the build: investigate before rebuilding, because this is exactly what the check is for |
| `TestSplitLocked` | intended: TEST is never evaluated by the workspace layer |
| CUDA out of memory | one model is kept in memory at a time; close other GPU jobs; choose device `cpu` |
| Health: "registry freshness" warning | a checkpoint changed after the build: rebuild |
| port in use | `studio.bat 8600` |
