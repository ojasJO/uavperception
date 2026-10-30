# Track C — AeroTrack temporal / tracking / prediction system

The third research track of the program, kept apart from the other two:

| track | what | where | run / checkpoint names |
|---|---|---|---|
| A | Anti-UAV (DFSC / GlobalTrack) reproduction and controlled replication | `experiments/anti_uav/` | `normal_ir`, `dfsc_ir`, `phaseB_*` |
| B | AeroTrack detector (YOLO11 stride layout, stacked frames, corpus v1 / v2) | `experiments/aerotrack_v2/`, `experiments/aerotrack_v3/` | `gen_*`, `v2_<config>_s<seed>_e20` |
| **C** | **video frames → temporal window → detections → association → track history → temporal state → future path → uncertainty → re-acquisition** | **`experiments/aerotrack_temporal/`** | **`c1_<arch>_<variant>_s<seed>` (forecasters); detector windows `v2_causal_g<g>_s<seed>_e20` (C2)** |

Track C reuses Track B's detector code and the systems layer (`experiments/systems/scripts`: STE, detection caches, SOT / MOT
harnesses) but owns its experiments, registrations and reports. Experiment IDs: `C1`, `C2`, … (below). TEST is sealed for every
Track C experiment: `tc_data.load_split("test")` raises unless a registered final evaluation asks for it.

## Experiments

| id | question | status | registration |
|---|---|---|---|
| C1 | trajectory forecasting from causal histories: learned vs strong classical forecasters, calibrated uncertainty, absolute position (camera re-centring) | registered (grids amended pre-VAL); learned models trained (transformer chosen on dev); VAL pass waits for the classical re-tuning | `preregistration/C1_trajectory_forecasting.json` |
| C1b | forecasting on TRACKER histories (causal detector -> cross-fitted STE -> forecaster vs ground truth; noise-augmented training): decides the deployed prediction output | registered; `tc_c1b.py`; runs after C3 on the causal caches | `preregistration/C1b_forecasting_on_tracker_histories.json` |
| C2 | causal frame-stacking window for the detector: [t-2,t-1,t], [t-4,t-2,t], [t-8,t-4,t], [t-16,t-8,t] vs single frame and the symmetric window | registered; GPU queue (after v2, post-v2, H3); `run_c2.py` / `analyze_c2.py` | `preregistration/C2_causal_windows.json` |
| C3 | tracking with domain / learned motion models: argmax vs constant-velocity STE vs forecaster-driven STE (domain classical / C1 model), cross-fitted; SOT mSA, HOTA / IDF1 / MOTA, fragmentation, recovery time | registered; `tc_ste.py`, `tc_c3.py` (machinery tested); runs after C1's verdict and the causal-detector caches | `preregistration/C3_tracking_motion_models.json` |
| CF | the complete causal system (rule-chosen detector window, tracker, forecaster) scored once on Anti-UAV + CST TEST: SOT mSA, HOTA / IDF1, recovery, forecast calibration; deployment gate on VAL inputs | registered; runs after C2 / C3 / C1b | `preregistration/CF_final_causal_system.json` |
| C4 | temporal alternatives to input stacking (feature fusion, motion-conditioned input, recurrent state) — only if C2 shows a temporal signal worth the cost | not started | — |

## Components and the 9-question gate

Every new component must state: the measured failure it addresses, why it fails, the baseline, what is new, its cost, the ablation
that isolates it, whether it runs on the RTX 5050, whether it exports, and what would falsify it. The answers live in each
registration (`component_gate`).

## Code

| file | role |
|---|---|
| `scripts/tc_data.py` | ground-truth trajectories (Anti-UAV IR / RGB, complete CST) → `results/runs/aerotrack_temporal/_data/` (sha256-checked) |
| `scripts/tc_forecast.py` | causal forecast samples, ten classical forecasters (incl. mean reversion and a vectorised Kalman bank), metrics, cluster bootstrap |
| `scripts/tc_learned.py` | MLP / GRU / transformer forecasters with diagonal-Gaussian heads, deterministic CPU training |
| `scripts/tc_c1.py`, `scripts/tc_c1_val.py` | C1 stages: TRAIN tuning, training, the single VAL pass + report |
| `scripts/tc_track_metrics.py` | HOTA (TrackEval algorithm, re-implemented; identical to TrackEval on 60 random sequences, `check_hota_vs_trackeval.py`), MT / PT / ML, fragmentation, longest-run continuity, recovery time, ids per target |
| `scripts/tc_ste.py` | forecaster-driven STE (C3): STE decisions with predictions from a classical (gap-robust) or learned (ONNX) forecaster |
| `scripts/tc_c3.py`, `scripts/tc_track_eval.py` | C3 runner (prep on TRAIN, cross-fitted VAL run, MOT-protocol metrics); tracker identity / continuity evaluation on any cache |
| `scripts/run_c2.py`, `scripts/analyze_c2.py` | C2 runner on the v2 machinery; rule-driven analysis |
| `scripts/tc_export.py` | forecaster ONNX export + parity |

Regression tests: `tests/test_research_program.py` (`TestTrackCForecasting`, `TestHOTA`).
