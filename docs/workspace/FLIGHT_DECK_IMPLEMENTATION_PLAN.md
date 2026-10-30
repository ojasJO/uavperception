# AeroTrack Flight Deck — implementation plan (Phase 0 understanding, 2026-10-13)

Starting point: branch `research-program`, HEAD `26e8873` (on `b396b25`). `git status --short` before any change:
`?? runs/_deploy/`, `?? runs/tmp_wheels/`, `?? runs/v2_causal_g4_s0_e20/` (pre-existing, untouched). No HPC command is run in this
work (not even read-only); `/tmp/yr_freeze_at311.txt` on the HPC is left alone.

## 1. The two research directions (as found in the code, not as named in the brief)

| | Direction A — AeroTrack multi-source system | Direction B — paper-specific reproduction |
|---|---|---|
| what | temporal-window YOLO11n detector (`c_model/`), single-target state estimator STE / argmax (`experiments/systems/scripts/ste.py`, `track_eval.py`), Track C forecasters (`experiments/aerotrack_temporal/`), CF final causal system, deployment | *Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking* (Jiang et al., arXiv:2101.08466): GlobalTrack QG-RCNN fine-tuned on Anti-UAV IR, normal vs DFSC, Protocol II (`experiments/anti_uav/`) |
| papers | YOLO11, SPD-Conv (historical), SORT/Kalman, ByteTrack (retired), CST benchmark protocol (arXiv:2507.23473), HOTA | Anti-UAV 2021; GlobalTrack (AAAI 2020) as the tracker it builds on |
| main results | X1 CST TEST 60.08 (STE) / 59.70 (argmax); X2 temporal seeds 60.08 / 60.26 / 60.60 (mean 60.31); single-frame control 46.83 / 49.81 / 49.79 (mean 48.81); CF TEST Anti-UAV IR 68.89 / RGB 74.10 / CST 56.55 | normal / DFSC val 79.29 / 79.26 (paper 79.60 / 80.09), test 70.58 / 70.29 (paper 65.36 / 66.04); DFSC gain not reproduced |

**Correction to the brief.** The brief calls Direction B the "CST paper reproduction". The repository's paper-specific
reproduction is the Anti-UAV (2021) DFSC/GlobalTrack study above (`experiments/anti_uav/README.md`). The CST paper
(arXiv:2507.23473) is a benchmark. Its *published baseline table* is reproduced in the repository — the 21 released
CST-trained trackers' TEST predictions scored by the official toolkit equal the paper's Table 3 to rounding
(`experiments/cst_anti_uav/reports/BASELINE_RESULTS.md`) — and its protocol is what Direction A's X1 / X2 / CF are scored with.
The Flight Deck therefore shows **two reproductions** in Category G, each with its own provenance: (G1) the Anti-UAV 2021
Protocol II reproduction = Direction B; (G2) the CST benchmark baseline-score reproduction = the CST study's protocol check (shared).
Nothing is relabelled or merged; this is the evidence-based reading, not new science.

## 2. Current application architecture (what is reused)

- `aerolab/` (shared layer): `registry.Registry` (314 model entries, 13 families, 317 historical records), `adapters/`
  (`aerotrack_detector`, `globaltrack_repro`, `trackc_forecaster`: SHA-256 gate, never loads from the backup), `evaluation`
  (manifest + `runs/studio/evals/<id>/` + `runs/studio/index.jsonl`), `test_lock` (TEST refused), `compare` (protocol compatibility,
  sequence-paired bootstrap), `data` (Anti-UAV / CST / Track C loaders via the experiments' own code), `failures`, `health`.
- `studio/` (Streamlit 1.64, `st.navigation`): 8 diagnostic-style pages; `studio.bat` launcher (localhost only).
- `experiments/anti_uav/demo/` (Direction B VAL demo, own smoke test) — left untouched.

## 3. Model families and real capabilities (traced in code)

| family | registry adapter | input | temporal state | output | per-frame incremental? |
|---|---|---|---|---|---|
| AeroTrack detector (cst_x X1/X2, v2/CF, gen, f1) | `aerotrack_detector` | 3 frames stacked to 9 channels (`cache_detections.windows`: causal [t, t-g, t-2g] / symmetric / ttt = single frame), letterbox 640 | window of past frames + STE Kalman state | detections (NMS 0.01 / 0.7 / 10) → STE / argmax box or *absent* | yes: window per frame + `STE.step` (same calls as `track_eval.run_tracker`) |
| GlobalTrack (Direction B normal / DFSC / seed 1 / phaseB) | `globaltrack_repro` | single frame, TestTransform (scale 1333×800 rule) | query from the init frame only | top-1 box + score every frame; **never reports absent** | yes: `track_init` + `track_update` per frame (same calls as `GlobalTrackTracker.track_sequence`) |
| Track C forecasters (C1 pooled, C1c domain-aware; 27 registered) | `trackc_forecaster` | 16-frame GT centre/size history (C1 oracle protocol) | none (per anchor) | 24-step mean + variance | per anchor (not a video stream) |
| repair grid, historical SPD, LSTM, ONNX/TensorRT exports, YOLO11 base, official GlobalTrack weights | none | — | — | recorded results only | no (blocker per family in the Vault) |

Score semantics: the detector's per-frame score is the highest drone-class detection score (the STE's association score is not
exported); GlobalTrack's score is its top-1 box score. Neither is a calibrated presence probability.

## 4. Datasets and annotations (local, verified)

| dataset | splits on disk | labels | annotations usable for presets |
|---|---|---|---|
| CST Anti-UAV 2025 | train 120 / val 40 / test 60, 640×512 IR JPEG | `IR_label.json`: `exist`, `gt`, **per-frame** `OC OV SV TC DBC CDB FM SIZE` (official) | absent frames, reappearance (exist transitions), occlusion / out-of-view / scale variation / thermal crossover / clutter / fast motion (official per-frame flags), size from GT diagonal (= release SIZE codes) |
| Anti-UAV 2021 | train 160 / val 67 / test 91, IR + RGB | `infrared.json` / `visible.json`: `exist`, `gt_rect`; `label_new/<split>.json` **sequence-level** `OV OC FM SV LI TC(-EASY/MID/HARD) LR` | absent, reappearance, size (GT), official sequence attributes |
| Track C trajectories | VAL (`runs/aerotrack_temporal/_data/traj_val.json`) | GT tracks per sequence and domain | forecasting only |

Frame rate: Anti-UAV 25 FPS is paper-explicit (`docs/ANTI_UAV_REPRODUCTION_SPEC.md`); CST reports none
(`PAPER_PROTOCOL.md`) → CST timestamps are shown as unavailable, never guessed. Manifests already hold per-sequence counts,
attribute counts and size bins (`experiments/cst_anti_uav/reports/dataset_manifest.json`, `experiments/anti_uav/manifests/`), so
the browser does not scan frames.

Recorded per-frame predictions (playback sources, verified on disk): CST TEST — X1 (`runs/aerotrack_v3/x1_test_predictions/{ste,argmax}`),
X2 temporal s1/s2 and single-frame s0–s2 (`runs/cst_anti_uav/x2_test_predictions/`), 21 released CST-trained trackers
(`data/cst_anti_uav_official/tracking_results/TrainedCST/`); CST VAL — X1 / X2 s0 (`runs/cst_anti_uav/val_predictions/`);
Anti-UAV VAL / TEST — normal_ir, dfsc_ir, seed-1 VAL (`experiments/anti_uav/runs/*/eval_*_IR_last/predictions/`).

## 5. Protocol restrictions (enforced in code, not only shown)

- TEST: every TEST split has been exposed by registered passes (8 ledgers in `aerolab/test_lock.py`). Inference / evaluation /
  threshold exploration on TEST is refused by `assert_split_allowed`. New: a **read-only replay** purpose may open TEST frames and
  GT only to draw *recorded* predictions (X1 / X2 / CF / released baselines). No model is loaded, no new number is registered.
- Consumed passes never re-run: X1, X2, CF, Anti-UAV Protocol II, v2, repair, generalization, v3/F1.
- Historical files are read, never rewritten. New runs get their own id under `runs/studio/evals/<id>/`.
- Verified historical values used by the UI (read from their records at render time): P1 run s0 at τ = 0.7 (instant gate):
  false alarms 27.6 % → 4.7 %, visible misses 12.1 % → 38.7 %, verdict NO_GAIN (`P1_presence_gate_VAL.json`); X2 T1–T3
  (`X2_test_official.json`); X2 − X1 +0.23 [−0.66, +1.14] (FINAL_REPORT §2, recomputed from per-sequence records);
  `jungle_18` is a **TEST** sequence: the distractor lock is seed-specific (X1 22.8, seeds 1 / 2 44.6 / 39.8) → replay only.

## 6. The seven categories: state before this work

| category | before | gap |
|---|---|---|
| A tracking | single-model run + frame slider (Inference page) | live streaming, multi-model synchronized views, CLE / loss / reacquisition / drift episodes |
| B presence | false-alarm and located rates only | score distributions, PR curve, threshold trade-off, confusion matrix, P1 curve |
| C temporal | recorded TEST rows in Compare | seed-aware view, T1–T3 from records, live temporal vs single-frame on VAL, runtime overhead |
| D robustness | size bins in Failure page | grounded scenario presets, per-attribute SA, failure intervals |
| E speed | one ms/frame number | controlled benchmark (warm-up, stages, sync, percentiles, memory) vs live demo FPS |
| F forecasting | one anchor plot | anchor browser with uncertainty, multi-forecaster comparison, Track C verdicts |
| G reproduction | records only | dedicated paper-vs-local view (G1, G2), config differences, fresh-vs-recorded VAL consistency |

## 7. Interface architecture

Five primary sections (Streamlit, `studio/app.py`): **Flight Deck** (home), **Test Bench** (category → dataset → preset → sequences →
models → options → summary → start), **Live Comparison** (central synchronized video; live job monitor; replay of saved runs and of
recorded predictions; side-by-side / overlay / difference / metrics views), **Results & Failures** (category-specific result views,
historical evidence, failure intervals), **Model Vault** (featured cards + full searchable archive + detail). The previous pages stay
reachable under a secondary "Workspace tools" group (Datasets, Inference, Compare records, Experiments, Failure analysis, Health).

New shared modules: `aerolab/sequences.py` (sequence index + grounded presets), `aerolab/categories.py` (7 category definitions,
eligibility), `aerolab/flightdeck.py` (multi-model experiment runner: config validation, sequential / lockstep execution,
cancellation, manifests, predictions, replay), `aerolab/jobs.py` (one background worker; UI polls), `aerolab/analysis.py`
(tracking episodes, presence analysis, seed summaries, latency statistics), `aerolab/recorded.py` (recorded-prediction catalog),
`aerolab/benchmark.py` (controlled speed benchmark), `registry/catalog/featured.json` (curated, no metrics).
Adapters gain `stream()` (incremental, per-frame timing) built from the same calls as their `predict()`.

## 8. Implementation order and acceptance

1. Contracts: experiment config / manifest schema `aerolab/flightdeck_experiment/1`, preset definition, frame record. 
2. Streaming adapters + parity tests (stream == predict on the same frames: GlobalTrack identical; detector within batch-size numerics).
3. Vertical slice: one model, CST VAL sequence, live stream → saved → reopened → replay equals saved → add a second model, synchronized.
4. Presets + sequence browser; categories A–G one by one, each with an eligibility check, a smoke run, a result view, a failure path.
5. Comparison views, cancellation, rerender safety; UI polish.
6. Tests: existing 86 kept; new unit + AppTest suites; real launch with `studio.bat`.

## 9. Risks and controls

| risk | control |
|---|---|
| live timing looks like a benchmark | separate Category E benchmark with warm-up / sync / repetitions; live FPS labelled "observed" |
| GPU memory with several models | sequential execution by default; lockstep loads all and warns; one job at a time |
| replay mistaken for inference | every view carries a mode banner (LIVE / SAVED RUN / RECORDED); recorded TEST predictions never trigger a model |
| TEST used for tuning | split lock in data + runner + UI; presence thresholds only on TRAIN / VAL runs; TEST replay has no threshold controls |
| derived scenario labels mistaken for annotations | each preset states its basis: official annotation, derived from GT (rule shown), recorded results, or unavailable |
| detector stream ≠ evaluated path | stream reuses `windows`, letterbox, autocast, NMS, `STE.step`; parity test against `predict` |
