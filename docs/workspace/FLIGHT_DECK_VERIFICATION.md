# Flight Deck — verification record (2026-10-13 / 14, OMEN, RTX 5050, torch 2.13.0+cu130, Streamlit 1.64.0)

Every claim below was checked on this machine. Commands are in `FLIGHT_DECK.md` § Commands.

## Models (counting rule: one registry entry per checkpoint identity `family:run@tag`, `results/registry/models.json`)

| quantity | value | how |
|---|---:|---|
| model families | 13 | `results/registry/catalog/families.json` |
| registry entries (checkpoint versions), all kept in the Vault archive | 314 | distinct weights+exports SHA-256: 299 |
| featured models (curated cards) | 7 | 6 tracking-side (one recorded-only: released GlobalTrack) + 1 forecaster; 14 checkpoint versions |
| load- and inference-verified entries (runnable here) | 228 | `scripts/verify_adapters.py --all` |
| families with verified local inference | 7 | cst_x, f1_std, gen, v2 detectors; C1, C1c forecasters; Direction B GlobalTrack |
| evaluation-verified (fresh whole-sequence VAL run = recorded per-sequence SA) | 5 | X1/X2 s0 (74.126 vs 74.138, tol 0.5); normal_ir 72.245 vs 72.239, dfsc_ir 72.403 vs 72.409, seed-1 normal 71.595 vs 71.590, seed-1 DFSC 71.658 vs 71.655 (tol 0.05) — `results/registry/verification/evaluation_checks.json` |
| not runnable here | 86 | 33 recorded-results-only families (repair grid, historical SPD, LSTM, exports, bases), 29 backup-only checkpoints, 20 smoke, 3 forecaster ONNX exports, 1 invalid — each with its blocker in the Vault |
| recorded per-frame prediction sources | 41 | `src/aerolab/recorded.py` |

## Fidelity of the live path

- GlobalTrack `stream()` reproduces `predict()` box for box: 60 of 60 frames identical.
- AeroTrack detector `stream()` (batch 1) vs `predict()` (batch 16), on 60 frames:
  - presence agrees on 60 / 60 frames;
  - box IoU median 0.997, minimum 0.984 (bf16 autocast numerics).
- Sequential and lockstep execution give identical boxes.
- A saved prediction file equals the stream that was shown live; SA recomputed from the file equals the manifest's value.
- Recorded replay recomputes the registered per-sequence SA exactly (|Δ| = 0, 9 decimals):
  - CST TEST: X1, X2 seeds and all 21 released trackers (JSON and text formats), on `jungle_18` and `building_66`;
  - CST VAL: X1 STE on all 40 sequences;
  - Anti-UAV VAL: Direction B normal on all 67 sequences.
- `jungle_18` (TEST, replay): X1 22.81, single-frame s0 5.86, released GlobalTrack 28.87. These agree with FINAL_REPORT §4 (X1 22.8, GlobalTrack 28.9).

## Values shown in the app (read from records, pinned by tests)

- X1: 60.08 STE, 59.70 argmax.
- X2 temporal: 60.08 / 60.26 / 60.60, mean 60.31 (SD 0.26).
- Single-frame control: 46.83 / 49.81 / 49.79, mean 48.81.
- T3: +11.51 [+8.36, +14.86].
- X2 − X1: +0.23 [−0.66, +1.14], 32 / 28 (recomputed).
- P1, run s0, instant gate at τ 0.7:
  - false alarms 27.6 % → 4.7 %;
  - visible misses 12.1 % → 38.7 %;
  - verdict NO_GAIN.
- Anti-UAV Protocol II: val 79.29 / 79.26 vs paper 79.60 / 80.09 (reproduced); test 70.58 / 70.29 vs 65.36 / 66.04 (not reproduced, higher).
- CST Table 3: reproduced from the released predictions, max |Δ| 0.004.

## Speed (controlled benchmark, `building_41`, 40 timed frames, 5 warm-up; PyTorch eager, batch 1)

| model | device | init s | first frame ms | median ms | model-only median ms | peak GPU MB |
|---|---|---:|---:|---:|---:|---:|
| X1 (STE) | cuda | 0.31 | 918 | 29.3 | 24.9 | 86 |
| X1 (STE) | cpu | 0.13 | 70 | 41.3 | 37.6 | — |

Live observed rates (not a benchmark): about 24–29 FPS for the detector on CUDA and about 11 FPS for GlobalTrack. The TensorRT
deployment gate (deploy.latency.v1, 3.2 ms model p50) is a different setup and is shown separately.

## Tests

| suite | result |
|---|---|
| `tests.test_flight_deck` | 27 OK: catalog, eligibility, evidence labels, presets vs raw labels, frame ordering, TEST lock (data, runner, benchmark, forecasting, evaluation service), replay = records, evidence values, episode rules, presence curves, latency gating, streaming parity, two-model run + reopen, lockstep = sequential, cancellation, one failing model, 9 invalid selections, explicit CUDA refused without CUDA, CPU benchmark, forecasting anchors identical + domain skip, job manager run + cancel |
| `tests.test_flight_deck_ui` | 9 OK: home, every category configures, presence excludes GlobalTrack with reason, invalid selection disables Start, full journey (start → LIVE → completed → synchronised replay, 2 panels, step, overlay, difference, metrics; rerenders start nothing; exactly one index record per model), cancel from the UI, recorded TEST replay (jungle_18, \|Δ\| = 0, no threshold control), results + vault, explicit CUDA in a CUDA-hidden process (error, Start disabled) |
| existing suites | 86 OK (`test_research_program`, `test_cst_protocol`, `test_workspace_registry`, `test_workspace_adapters`, `test_studio_ui`) |
| total | **122 OK**; plus the Direction B demo smoke test and `render_workspace_docs.py --check` |
| CPU only | the journey and runner tests pass with `CUDA_VISIBLE_DEVICES=-1` |

## Real browser (headless Edge over the DevTools protocol, `studio.bat 8611`)

- Bound to 127.0.0.1 only; `/_stcore/health` returned ok; HTTP 200.
- The full journey ran with no exception on any page:
  - home → Test Bench → Start;
  - LIVE view with real frames, latency and rolling FPS → completed;
  - synchronised replay (side by side, overlay);
  - recorded TEST replay of `jungle_18`;
  - Results; Vault.
- Three defects that AppTest cannot see were found in the browser and fixed:
  - "Open the synchronised replay" sat inside a fragment and reran only the fragment; it now reruns the app.
  - A direct URL on a fresh server took Streamlit's legacy `pages/` route. The directory is now `app/views/`.
  - The status strip rendered empty. It is now drawn as an image.

## Not tested / limits

- Execution through TensorRT / ONNX: not installed in this venv.
- Whole-split GlobalTrack evaluations: about 90 s per 1000-frame sequence on CUDA.
- Multi-user or remote access: localhost by design.
- Real-browser play / pause timing at speeds above 5 frames per tick.
- AppTest does not follow a script-initiated `st.switch_page`, so those tests switch pages explicitly. The browser run covers the real navigation.
