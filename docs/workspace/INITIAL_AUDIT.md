# Workspace baseline audit — 2026-10-13 (before the unified-workspace changes)

Written before any file of the workspace initiative was created or changed. Every statement was checked on the OMEN on
2026-10-13; nothing here was copied from older documents without a check.

## 0. Original state

| item | value |
|---|---|
| branch / HEAD | `research-program` @ `7f84f55` (local only; `origin` = github.com/ojasJO/uavperception, never pushed by an agent) |
| other branches | `main`, `aerotrack-repair`; remote refs `origin/main`, `hpc/cd433bb`; no tags, no stashes |
| `git status --short` | `?? runs/_deploy/` · `?? runs/tmp_wheels/` · `?? runs/v2_causal_g4_s0_e20/` (pre-existing, left untouched) |
| disk | A: 300 GB, 86 GB free |
| Python | `venv\` Python 3.11.9 · torch 2.13.0+cu130 · torchvision 0.28.0+cu130 · ultralytics 8.4.115 · streamlit 1.64.0 · onnxruntime 1.30.0 · numpy 2.4.6 · pandas 3.0.5 · scipy 1.17.1 · opencv 5.0.0 · altair 6.3.0 · jsonschema 4.26.0. Not installed: pytest, plotly, tensorrt (in this venv) |
| GPU | RTX 5050 Laptop 8 GB, driver 592.19; `torch.cuda.is_available()` = True and a CUDA tensor op succeeds (the earlier driver-compatibility error does not reproduce) |
| tests | `venv\Scripts\python.exe -m unittest tests.test_research_program tests.test_cst_protocol` → **34 tests OK (10.7 s)**; separate demo smoke test `experiments/anti_uav/demo/test_demo.py` |

## 1. The two research directions (identified from the sources, not from names)

**Direction A — AeroTrack (multi-source research system).** Detection → tracking → prediction → deployment.
Code: `c_model/`, `b_pipeline/`, `experiments/aerotrack_{repair,generalization,v2,v3}`, `experiments/systems` (STE tracker),
`experiments/aerotrack_temporal` (Track C forecasting / tracking metrics / CF final system), `experiments/deployment`,
`experiments/cst_anti_uav` (AeroTrack X1/X2 scored under the CST benchmark's official protocol).
Sources: YOLO11 / Ultralytics (detector), SPD-Conv (Sunkara & Luo 2022; evaluated, retired as a claim), ByteTrack / SORT
(evaluated, retired), Kalman single-target state estimation (classical), learned Gaussian forecasters (own design), the CST
Anti-UAV benchmark protocol (arXiv:2507.23473) and the Anti-UAV mSA metric (arXiv:2101.08466) as evaluation protocols.

**Direction B — single-paper reproduction: *Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking*** (arXiv:2101.08466v3,
IEEE TMM; PDF `anti_uav_research_paper.pdf`): DFSC training on GlobalTrack, Protocol II, IR (and visible follow-up A2).
Code: `experiments/anti_uav/` (self-contained package `src/anti_uav_repro`, configs, frozen manifests, append-only test ledger,
existing VAL-only Streamlit demo `experiments/anti_uav/demo/`). Docs: `docs/ANTI_UAV_REPRODUCTION_SPEC.md` (paper vs inherited vs
substituted), `docs/ANTI_UAV_EXPERIMENT_PROTOCOL.md`, `experiments/anti_uav/reports/FINAL_PROTOCOL_II_IR_BENCHMARK_REPORT.md`.

**Correction to the task brief.** The brief assumed the CST Anti-UAV work is the paper reproduction. The sources say otherwise:
the CST paper (arXiv:2507.23473) is a *benchmark* paper with no method of its own; this repository (a) reproduces its Table 3
from the authors' released predictions with the official toolkit (an evaluation reproduction, `experiments/cst_anti_uav/reports/BASELINE_RESULTS.md`)
and (b) scores **AeroTrack** X1 / X2 under its protocol (`FINAL_REPORT.md`: "AeroTrack under the benchmark's own protocol").
X1 / X2 / CF / Track C therefore belong to Direction A; the CST protocol and its reproduced published baselines are registered
as a shared benchmark / protocol, not as Direction B. The paper-specific reproduction (a trained method) is the Anti-UAV DFSC work.

## 2. Source modules and entry points

| area | entry points |
|---|---|
| A detector training / eval | `experiments/aerotrack_v3/scripts/{train_v3,eval_v3,x1_final}.py`, `experiments/aerotrack_v2/scripts/`, `c_model/train.py` |
| A tracker + metrics | `experiments/systems/scripts/{ste,track_eval,cache_detections}.py`; SA metric = Track A's `anti_uav_repro.eval.metrics` |
| A streaming inference | `experiments/deployment/scripts/aerotrack_stream.py` (frames/video → window → detector → NMS → STE/FSTE → JSONL), `backends.py` |
| A Track C | `experiments/aerotrack_temporal/scripts/tc_*.py` (`tc_data.load_split` seals TEST; `tc_learned.load/predict`; `tc_forecast.make_samples`) |
| A CST study | `experiments/cst_anti_uav/scripts/{official_eval,x2_final,failure_analysis,explore_p1}.py` |
| B reproduction | `experiments/anti_uav/scripts/{train,evaluate,aggregate_results}.py`; tracker `anti_uav_repro.engine.track.GlobalTrackTracker` |
| existing UI | `experiments/anti_uav/demo/app.py` (Streamlit, Direction B, IR VAL only; core logic `demo_core.py`) — the only interface |
| ledgers | `tools/build_master_ledger.py` → `docs/EXPERIMENT_LEDGER.{jsonl,csv}` (179 rows); `tools/build_registry.py` → `artifacts/experiment_registry.json`; `tools/render_results.py` → `docs/RESULTS.md`; per-study test ledgers (`*_test_ledger.jsonl`) |
| recovery | `tools/hpc/recovery_manifest.py`, `artifacts/hpc_recovery_20261013/` (5,725 files, 17.03 GB, SHA-256 verified 2026-10-13) |

## 3. Model checkpoint inventory (file level)

| location | model files (`.pt .onnx .engine .pkl …`, excluding `.npz`) |
|---|---|
| local workspace (excl. venv, data, recovery) | 297 files: 62.7 GB in `experiments/anti_uav/runs` (8 IR runs × 19 checkpoints), ~2 GB of detector runs in `runs/`, Track C C1 forecasters, ONNX / TensorRT engines (`artifacts/deploy`, `deployment/`) |
| recovery backup | 1,348 files (16.37 GB) incl. 1,193 `.npz` |
| **only in the backup** | X1 / X2 / F1 runs `v3_*` (8 dirs), v2 seeds `v2_causal_g{1,2,8}_s{0,1}`, `v2_causal_g4_s1`, `v2_stock_s1`, `v2_sym_g4_s1`, `omen_v2_stock_s0`; C1c forecasters (15); Anti-UAV RGB `normal_rgb` / `dfsc_rgb` (epochs 1-12 + last) and `largescale_globaltrack` |

X1 shares its checkpoint with X2 temporal seed 0 (`v3_n__causal_g4_cst_s0_e20/weights/last.pt`, sha256 `e52a5645…`).

## 4. Datasets and splits (local)

| dataset | location | splits (sequences) |
|---|---|---|
| Anti-UAV (2021) labels + raw | `data/anti_uav/{train,val,test}` (6.3 GB) | 160 / 67 / 91 (IR + RGB per sequence); frozen manifests `experiments/anti_uav/manifests/antiuav_{IR,RGB}_{train,val,test}.manifest.json` |
| Anti-UAV decoded frames | `b_pipeline/data_unified/anti_uav/images/` | used by Direction B and A |
| CST Anti-UAV (complete release) | `data/cst_anti_uav_full/CST-AntiUAV/CST-AntiUAV/{train,val,test}` | 120 / 40 / 60; manifest `experiments/cst_anti_uav/reports/dataset_manifest.json` |
| CST (truncated v1 copy) | `data/cst_anti_uav/` | historical (84 train, 0 val) — never used for new work |
| Det-Fly | `data/det_fly/{train,valid,test}` | images (Roboflow export) |
| unified YOLO corpora | `b_pipeline/data_unified/` (v1), `b_pipeline/data_unified_v2/` (v2, complete CST) | `splits.json`, `manifest.csv` |
| Track C trajectories | `runs/aerotrack_temporal/_data/traj_{train,val}.{npz,json}` | sha-checked by `tc_data.load_split` |

## 5. Experiment and evaluation records

Master ledger (179 rows: 45 train, 89 eval, 45 HPC jobs) covers Track A, the generalization / repair / v2 detector series,
deployment latency and C1. **Not in the master ledger as metrics**: CST X1 / X2 (`experiments/aerotrack_v3/reports/x1_final_test.json`,
`experiments/cst_anti_uav/reports/x2_final_test.json`), CF (`experiments/aerotrack_temporal/reports/CF_final_causal_system.json`),
P1 (`experiments/cst_anti_uav/reports/explore/P1_presence_gate_VAL.json`), Track C C1b/C1c/C2/C3 summaries. Consumed TEST budgets:
Anti-UAV IR (normal / DFSC seeds 0-1, ledger `experiments/anti_uav/reports/test_evaluation_ledger.jsonl`), generalization detector,
X1, X2, CF, v2 final test.

## 6. Problems found

- No single model registry; checkpoint identity spread over `run_meta.json`, frozen configs, learned registries and ledgers.
- Many registered checkpoints are present only in the recovery backup (section 3) → not runnable by the existing scripts.
- The only UI covers Direction B, IR VAL only. No cross-direction browsing, no comparison laboratory, no health panel.
- `README.md` (generated by `render_report.py`) is marked SUPERSEDED in parts; the entry docs are `docs/MASTER_STATE.md` / `AGENT_CONTEXT.md`.
- `pytest` is not installed; tests run with `unittest`.
- Pre-existing untracked `runs/v2_causal_g4_s0_e20/` holds one `.pt`; its HPC twin holds best + last. Left untouched.

## 7. Plan chosen (safest)

1. **No moves.** Checkpoints keep their established paths; a generated registry (`registry/`) is the one place to discover them.
2. Checkpoints that exist only in the backup are **copied** (never moved) back to their original repo-relative path when that path
   is free, SHA-256 verified, and logged in `registry/migrations/restore_manifest.json`. The backup is read-only.
3. A small shared package `aerolab/` (registry access, schemas, adapters wrapping the existing pipelines, evaluation service with
   the test-split lock, health checks) and one Streamlit studio `studio/` replacing nothing (the Direction B demo stays).
4. Tests with `unittest` (existing convention) + Streamlit `AppTest`.
