# Direction A — AeroTrack: multi-source research system

**Question.** Can an integrated detection → single-target state estimation → forecasting system, assembled from several published
components plus new logic, track tiny UAVs in thermal and visible video? Which components actually help?

**Nature.** An integrated research system. Some components come from published work, some are adapted, and some are new. A component
counts as an improvement only when a controlled, pre-registered experiment supports it. Combining papers does not make a better method
by itself, and several published components were evaluated here and *retired*.

The code stays where the experiments put it; this directory is the entry point and index. Browse models, results and failures in the
studio (`studio.bat`, see `docs/workspace/OPERATING_GUIDE.md`).

## Source → component provenance matrix

| component | source | how it is used here | status (evidence) |
|---|---|---|---|
| Detector network (YOLO11n layout, Ultralytics 8.4.115) | Ultralytics YOLO11 (software) | **adapted**: 9-channel temporal input, function-preserving COCO initialisation; as-built network has the `c3k=True` erratum (64 % COCO load) | VALIDATED as the base detector; standard layout F1 **NOT SUPPORTED** (`experiments/aerotrack_v3/reports/F1_standard_yolo11n.md`, `experiments/aerotrack_v2/ERRATUM_architecture.md`) |
| Temporal frame window (t−g, t, t+g → causal [t, t−g, t−2g]) | **new design** in this project | stacked frames at the detector input; causal window removes the 160 ms look-ahead | **SUPPORTED**: g = 4 (GEN_REPORT §9); causal_g4 selected (`experiments/aerotrack_v2/hpc/reports/v2_decisions.md`); X2 T3 temporal vs single frame |
| SPD-Conv downsampling | Sunkara & Luo 2022 (citation not recorded in repo) | **adapted**: replaces stride-2 convolutions | **RETIRED as a claim**: no effect once initialisation is fair (`experiments/aerotrack_generalization/reports/GEN_REPORT.md`) |
| ByteTrack / SORT association | Zhang et al. 2022 / Bewley et al. 2016 (citations not recorded in repo) | faithful re-implementation as MOT trackers | **RETIRED**: lose to single-target argmax (`docs/AEROTRACK_REPAIR_REPORT.md`, `experiments/systems/reports/mot_*`) |
| Single-target state estimator (STE: Kalman CV state, Mahalanobis gate, coasting, re-acquisition, absence) | classical Kalman filtering; **new** single-target logic | `experiments/systems/scripts/ste.py`; settings chosen on VAL (cross-fitted) | VALIDATED; reports absence (matters under Eq. 1) |
| LSTM forecaster | Review-2 design | `c_model/lstm_forecaster.py` | **RETIRED** (repair report) |
| Learned Gaussian forecasters (mlp / gru / tfm, C1; domain-aware C1c) | **new design** (Track C) | `experiments/aerotrack_temporal/scripts/tc_learned.py` | **MIXED**: better on Anti-UAV, not on CST (`experiments/aerotrack_temporal/reports/TRACK_C_SUMMARY.md`) |
| Classical forecasters (cp, cv, damped velocity, mean reversion, Kalman) | classical | TRAIN-tuned references (`tc_forecast.py`) | references; damped velocity best on CST |
| HOTA / CLEAR-MOT metrics | Luiten et al. 2021 (TrackEval) | re-implemented, cross-checked | `experiments/aerotrack_temporal/reports/hota_crosscheck.json` |
| State accuracy mSA (Eq. 1) | Anti-UAV paper (arXiv:2101.08466) | evaluation metric (Track A's code) | equals the official CST toolkit per sequence (`tests/test_cst_protocol.py`) |
| CST benchmark protocol + toolkit | CST Anti-UAV (arXiv:2507.23473) | **evaluation protocol only** (the paper proposes no tracker) | X1 / X2 / CF scored with the unmodified toolkit |
| Deployment (ONNX / TensorRT) | ONNX Runtime / TensorRT | export + latency gate | measured on the RTX 5050 (`experiments/deployment/reports/`) |

## Model families (registry ids)

`aerotrack.detector.cst_x` (X1 / X2), `aerotrack.detector.v2` (v2 campaign, CF detector), `aerotrack.detector.f1_std`,
`aerotrack.detector.gen`, `aerotrack.detector.repair`, `aerotrack.detector.historical`, `aerotrack.forecaster.c1`,
`aerotrack.forecaster.c1c`, `aerotrack.forecaster.lstm`, `aerotrack.deploy`. X1 and X2 temporal seed 0 are **one checkpoint**
(`v3_n__causal_g4_cst_s0_e20/weights/last.pt`, sha256 `e52a5645…`), registered under both roles.

## Headline results (generated — do not edit by hand)

<!-- BEGIN GENERATED: tools/render_workspace_docs.py -->
| record | model (checkpoint) | dataset / split | protocol | mSA | 95 % CI | kind |
|---|---|---|---|---:|---|---|
| `cst:released:GlobalTrack` | cst_released:GlobalTrack | cst2025 / test | `sot.mSA.cst_official.v1` | 35.92 |  | released_baseline (reproduced_from_released_predictions) |
| `cst:paper:GlobalTrack` | cst_released:GlobalTrack | cst2025 / test | `sot.mSA.cst_official.v1` | 35.92 |  | paper_reported (paper_reported) |
| `cst:x1:ste` | v3_n__causal_g4_cst_s0_e20@last | cst2025 / test | `sot.mSA.cst_official.v1` | 60.08 | [56.13, 63.97] | registered_test_pass (historical_import) |
| `cst:x1:argmax` | v3_n__causal_g4_cst_s0_e20@last | cst2025 / test | `sot.mSA.cst_official.v1` | 59.70 | [55.80, 63.48] | registered_test_pass (historical_import) |
| `cst:x2:v3_n__causal_g4_cst_s0_e20:ste` | v3_n__causal_g4_cst_s0_e20@last | cst2025 / test | `sot.mSA.cst_official.v1` | 60.08 |  | registered_test_pass (historical_import) |
| `cst:x2:v3_n__causal_g4_cst_s1_e20:ste` | v3_n__causal_g4_cst_s1_e20@last | cst2025 / test | `sot.mSA.cst_official.v1` | 60.26 |  | registered_test_pass (historical_import) |
| `cst:x2:v3_n__causal_g4_cst_s2_e20:ste` | v3_n__causal_g4_cst_s2_e20@last | cst2025 / test | `sot.mSA.cst_official.v1` | 60.60 |  | registered_test_pass (historical_import) |
| `cst:x2:v3_n__ttt_cst_s0_e20:ste` | v3_n__ttt_cst_s0_e20@last | cst2025 / test | `sot.mSA.cst_official.v1` | 46.83 |  | registered_test_pass (historical_import) |
| `cst:x2:v3_n__ttt_cst_s1_e20:ste` | v3_n__ttt_cst_s1_e20@last | cst2025 / test | `sot.mSA.cst_official.v1` | 49.81 |  | registered_test_pass (historical_import) |
| `cst:x2:v3_n__ttt_cst_s2_e20:ste` | v3_n__ttt_cst_s2_e20@last | cst2025 / test | `sot.mSA.cst_official.v1` | 49.79 |  | registered_test_pass (historical_import) |
| `cf:test:sot:anti_uav_IR` | v2_causal_g4_s0_e20@last | antiuav2021 / test | `sot.mSA.antiuav_eq1.v1` | 68.89 |  | registered_test_pass (historical_import) |
| `cf:test:sot:anti_uav_RGB` | v2_causal_g4_s0_e20@last | antiuav2021 / test | `sot.mSA.antiuav_eq1.v1` | 74.10 |  | registered_test_pass (historical_import) |
| `cf:test:sot:cst_IR` | v2_causal_g4_s0_e20@last | cst2025 / test | `sot.mSA.antiuav_eq1.v1` | 56.55 |  | registered_test_pass (historical_import) |

Seed means: temporal causal_g4 60.31, single frame 48.81. The registered paired statistics (X2 vs X1: no improvement, CI spans zero; temporal vs single frame: supported) are in `experiments/cst_anti_uav/reports/FINAL_REPORT.md`. All rows above are TEST passes (used up; read-only).

P1 (VAL, exploratory):

| run | frozen mSA | gated mSA | Δ | verdict |
|---|---:|---:|---:|---|
| v3_n__causal_g4_cst_s0_e20 | 63.61 | 63.61 | 0.00 | NO_GAIN |
| v3_n__causal_g4_cst_s1_e20 | 65.08 | 65.08 | 0.00 | NO_GAIN |
| v3_n__causal_g4_cst_s2_e20 | 64.46 | 64.46 | 0.00 | NO_GAIN |

<!-- END GENERATED -->

## Reproduction status, limitations, next steps

- All registered TEST budgets (generalization detector, X1, X2, CF, v2) are **used up**. Any new TEST pass needs a new registration.
- Limitations: large targets (> 50 px) are weak, with only 6 large frames in CST train. Absent-frame false alarms and wrong-object boxes
  are a large share of the remaining loss; P1's score gate did not fix them (NO_GAIN). VAL shares videos with TRAIN.
- Next justified step (TRAIN / VAL only, register first): a *learned* absence / presence signal. Templates:
  `experiments/cst_anti_uav/preregistration/P1_presence_gate_VAL.json` for the registration and `docs/MASTER_STATE.md` for the plan.
