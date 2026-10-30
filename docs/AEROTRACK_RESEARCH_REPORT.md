# AeroTrack — research report

*Generated 2026-10-05 16:24 by `scripts/render_aerotrack_report.py` from the campaigns' JSON (index: `results/artifacts/experiment_registry.json`). No number is typed by hand; sections without evidence yet are marked pending.*

## 1. Problem definition

Detect, track and predict a single small UAV in thermal (IR) and visible (RGB) video, in real time on an RTX 5050 Laptop GPU (8 GB), and report what each component buys against the obvious alternative. Datasets: Anti-UAV (gimbal-tracked IR + RGB clips), CST Anti-UAV (static-camera IR, tiny targets in complex scenes), Det-Fly (air-to-air RGB images incl. birds).

## 2. Dataset characterization

From `results/artifacts/dataset_card_data_unified_v2.json` (full table: `docs/DATASETS_data_unified_v2.md`).

| dataset / modality / split | seq | frames | box diag median px | tiny (≤10 px diag) | micro area | motion px/frame (median) |
|---|---:|---:|---:|---:|---:|---:|
| anti_uav/IR/test | 91 | 85,374 | 55.6 | 0.0 % | 0.0 % | 4.03 |
| anti_uav/IR/train | 160 | 149,528 | 58.6 | 0.0 % | 0.0 % | 3.64 |
| anti_uav/RGB/test | 91 | 85,374 | 144.1 | 0.0 % | 0.2 % | 9.12 |
| anti_uav/RGB/train | 160 | 149,528 | 124.7 | 0.0 % | 0.5 % | 9.01 |
| cst/IR/test | 60 | 72,175 | 14.3 | 17.5 % | 57.4 % | 0.76 |
| cst/IR/train | 120 | 127,991 | 15.7 | 20.5 % | 45.6 % | 0.60 |
| det_fly/RGB/test | 1,567 | 1,567 | 33.8 | 4.6 % | 11.3 % | — |
| det_fly/RGB/train | 4,698 | 4,698 | 33.1 | 3.4 % | 11.2 % | — |

Known defects: **anti_uav** — IR 640x512 + RGB 1920x1080 per clip; official 160/67/91 sequences per modality; released box count ~2 % below the paper's Table I; VAL shares all 26 recording sessions with TRAIN **cst** — corpus v1 was built from parts z01-z07 only (84/120 train, 0/40 val, 60/60 test sequences); corpus v2 from all 11 parts (CRC-verified). Official release 120/40/60. **det_fly** — random image-level export split (VAL frames adjacent to TRAIN frames: 1,224 of 1,469); images stretched to 640x640; the only source of birds

## 3. Hardness analysis

Hardness is reported per dataset, per size bin and per CST scene category / frame attribute rather than as one mean: see sections 7–8 (CST micro AP, per-category AP, attribute SA) and `experiments/systems/reports/failures_*` (deterministic failure categories with traceable examples).

## 4. Architecture

Detector (YOLO11n-family, 9-channel temporal input, function-preserving initialisation) -> NMS -> single-target state estimator (Kalman state, gating, coasting, re-acquisition, existence) -> forecast with covariance -> streaming runtime (`experiments/deployment/scripts/aerotrack_stream.py`). Component status: `docs/AGENT_CONTEXT.md` §A.

## 5. Design rationale (each kept component answers a measured failure)

- generalization series, `spd_fp` vs `stock`: mean Δ CST drone AP50 +0.0139, CI [-0.0046, +0.0337] -> not supported: (1) mean delta > 0.02, (3) bootstrap CI excludes 0
- generalization series, `temporal_fp_g1` vs `stock`: mean Δ CST drone AP50 +0.0089, CI [-0.0054, +0.0232] -> not supported: (1) mean delta > 0.02, (3) bootstrap CI excludes 0
- generalization series, `temporal_fp_g4` vs `stock`: mean Δ CST drone AP50 +0.0448, CI [+0.0208, +0.0717] -> supported
- repair grid (initialisation-confounded, superseded by the generalization series): SPD-Conv (not credited: micro_AP50 Δ best -0.016 / last -0.078, cst_micro_AP50 Δ best -0.039 / last -0.102); temporal stack (not credited: cst_AP50 Δ best -0.045 / last -0.015, drone_AP50 Δ best -0.027 / last -0.003); full recipe vs stock YOLO11n (not credited: drone_AP50 Δ best -0.055 / last -0.035)
- retired by pre-registered rules: ByteTrack (loses to a per-frame argmax), the LSTM forecaster (collapsed; loses to zero-velocity / Kalman), SPD-Conv as a claimed contribution (no supported gain once initialisation is fair), the t±1 stack.

## 6. Experimental protocol

`docs/EXPERIMENT_PROTOCOL.md`: pre-registration committed before runs, last.pt fixed, one TEST pass per final candidate, sequence-level bootstrap, cross-fitted tracker parameters, idle-GPU latency.

## 7. Baselines, ablations and detection analysis

**v2 (complete CST corpus; windows)**: pending (`experiments/aerotrack_v2/`)

**v3 (capacity allocation)**: pending (`experiments/aerotrack_v3/`)

## 8. Tracking analysis

From `experiments\systems\reports\track_gen_temporal_fp_g4_full_e20_data_unified.json` (SOT protocol, mSA by Track A's code, cross-fitted parameters):

| tracker | group | mSA | located | recover median (frames) |
|---|---|---:|---:|---:|
| argmax | anti_uav_IR | 80.56 | 98.6 % | 1 |
| argmax | anti_uav_RGB | 81.40 | 97.9 % | 0 |
| argmax | cst_IR | 72.11 | 85.5 % | 3 |
| ste | anti_uav_IR | 80.67 | 98.7 % | 1 |
| ste | anti_uav_RGB | 81.07 | 97.4 % | 0 |
| ste | cst_IR | 73.96 | 88.1 % | 1 |

Our GlobalTrack (Track A) on the same 67 Anti-UAV IR VAL sequences: mSA 79.29; STE − GlobalTrack +1.39 [-0.24, +4.35].

Established trackers under CLEAR-MOT / IDF1: `experiments\systems\reports\mot_gen_temporal_fp_g4_full_e20_data_unified.md`.

Identity persistence, continuity and recovery (Track C, MOT protocol; HOTA re-implementation identical to TrackEval), from `experiments\aerotrack_temporal\reports\tracking_gen_temporal_fp_g4_full_e20_data_unified.md` (groups ['anti_uav_IR', 'anti_uav_RGB']):

| tracker | group | HOTA | AssA | IDF1 | ID sw | ids / target | loss -> recovered median / p90 (frames) |
|---|---|---:|---:|---:|---:|---:|---|
| bytetrack | anti_uav_IR | 0.574 | 0.445 | 0.604 | 1391 | 14.5 | 1 / 4 |
| bytetrack | anti_uav_RGB | 0.543 | 0.414 | 0.561 | 2011 | 17.1 | 1 / 4 |
| sort | anti_uav_IR | 0.575 | 0.445 | 0.605 | 1339 | 14.3 | 1 / 4 |
| sort | anti_uav_RGB | 0.544 | 0.403 | 0.550 | 1965 | 17.7 | 1 / 4 |
| argmax@cf | anti_uav_IR | 0.789 | 0.800 | 0.982 | 0 | 1.0 | 1 / 5 |
| argmax@cf | anti_uav_RGB | 0.798 | 0.816 | 0.969 | 0 | 1.0 | 1 / 6 |
| ste@cf | anti_uav_IR | 0.789 | 0.800 | 0.982 | 0 | 1.0 | 1 / 4 |
| ste@cf | anti_uav_RGB | 0.795 | 0.815 | 0.964 | 0 | 1.0 | 1 / 8 |
| ste@cf/segments | anti_uav_IR | 0.774 | 0.770 | 0.951 | 8 | 1.1 | 1 / 4 |
| ste@cf/segments | anti_uav_RGB | 0.767 | 0.758 | 0.902 | 28 | 1.4 | 1 / 8 |

**Corpus caveat:** these rows were computed on corpus v1 (`data_unified`), whose CST part is the truncated archive (84/120 train sequences, no official val; the `cst_IR` group is a carved v1 holdout). Anti-UAV rows are corpus-independent; CST rows are development evidence only and are superseded by the corpus-v2 analyses.

## 9. Trajectory prediction

From `experiments\systems\reports\forecast_gen_temporal_fp_g4_full_e20_data_unified.json` (forecasts from the tracker's own outputs, horizon 12):

| group | ADE Kalman / zero-vel / last-vel / mean-vel4 (px) | Kalman 95 % ellipse coverage |
|---|---|---:|
| anti_uav_IR | 28.0 / 27.8 / 36.8 / 37.5 | 0.73 |
| all | 45.7 / 46.4 / 63.6 / 62.7 | 0.75 |
| anti_uav_RGB | 72.2 / 73.5 / 102.4 / 100.5 | 0.73 |
| cst_IR | 4.2 / 7.1 / 7.8 / 3.9 | 0.91 |

**Corpus caveat:** these rows were computed on corpus v1 (`data_unified`), whose CST part is the truncated archive (84/120 train sequences, no official val; the `cst_IR` group is a carved v1 holdout). Anti-UAV rows are corpus-independent; CST rows are development evidence only and are superseded by the corpus-v2 analyses.

Learned vs classical forecasters on ground-truth histories (Track C, C1, registered; `experiments\aerotrack_temporal\reports\C1_trajectory_forecasting.md`): transformer chosen on the dev split; ADE over 1-24 frames, VAL, 3 seeds:

| domain | classical reference (TRAIN) | reference ADE px | learned ADE px (s0 / s1 / s2) | H1 point | H2 uncertainty | H3 abs. position |
|---|---|---:|---|---|---|---|
| antiuav_IR | oudcv | 37.3 | 33.7 / 34.3 / 33.7 | SUPPORTED | NOT_SUPPORTED | NOT_SUPPORTED |
| antiuav_RGB | oudcv | 97.7 | 89.1 / 90.9 / 89.0 | SUPPORTED | SUPPORTED | NOT_SUPPORTED |
| cst | dcv | 12.1 | 17.5 / 17.6 / 17.9 | LOSES | SUPPORTED | NOT_SUPPORTED |

## 10. Failure analysis

`experiments\systems\reports\failures_gen_temporal_fp_g4_full_e20_data_unified.md` (13 rule-defined categories, quantile-selected examples).

**Corpus caveat:** these rows were computed on corpus v1 (`data_unified`), whose CST part is the truncated archive (84/120 train sequences, no official val; the `cst_IR` group is a carved v1 holdout). Anti-UAV rows are corpus-independent; CST rows are development evidence only and are superseded by the corpus-v2 analyses.

## 11–13. Latency, memory, deployment (RTX 5050)

| run | backend | model p50 / p99 ms | end-to-end p50 / p99 ms | device MB | AP50 parity Δ vs bf16 |
|---|---|---|---|---:|---:|
| gen_temporal_fp_g4_full_e20 | torch_fp32 | 18.52 / 24.15 | 28.96 / 36.35 | 126 | +0.0032 |
| gen_temporal_fp_g4_full_e20 | torch_bf16 | 23.53 / 30.02 | 19.74 / 31.40 | 68 | +0.0000 |
| gen_temporal_fp_g4_full_e20 | torch_fp16 | 17.81 / 22.87 | 14.53 / 23.58 | 60 | -0.0016 |
| gen_temporal_fp_g4_full_e20 | ort_cuda | 9.71 / 14.41 | 9.83 / 14.84 | 80 | +0.0037 |
| gen_temporal_fp_g4_full_e20 | trt_fp32 | 3.54 / 5.13 | 7.30 / 9.78 | 104 | +0.0028 |
| gen_temporal_fp_g4_full_e20 | trt_fp16 | 3.49 / 5.48 | 5.93 / 8.32 | 100 | -0.0027 |

Accuracy vs latency-to-output (incl. the window's look-ahead): `docs/PARETO.md`.

## 14. Cross-domain generalization

Per-dataset and per-scene-category AP are reported in every campaign; training-domain subsets (Anti-UAV only, CST only, without Det-Fly, IR only, RGB only) are prepared in `experiments/aerotrack_v3/` (`--train-domains`) — pending.

## 15. Final test results (one pass per candidate)

- generalization candidate `gen_temporal_fp_g4_full_e20` (corpus v1): CST drone AP50 0.521 [0.442, 0.597]
- v2 candidate: pending (`experiments/aerotrack_v2/final_protocol.json`)

## 16. Limitations

- Single laptop GPU: 1–2 seeds per configuration, 20-epoch budget; HPC access was not available.
- Anti-UAV VAL shares recording sessions with TRAIN (optimistic for Anti-UAV); Det-Fly VAL is near-duplicate of TRAIN.
- The temporal window helps static-camera thermal data (CST); gimbal footage (Anti-UAV) gains nothing measurable.

## 17. Future work

- Pending campaigns listed in `docs/progress/RESEARCH_PROGRAM_PROGRESS.md` (v3 capacity, cross-domain, more seeds).

## 18. Reproducibility

Registry `results/artifacts/experiment_registry.json` (git commit, config hash, data / split hashes, seed, checkpoint SHA-256, environment per run); environments `docs/environments/`; every report regenerable by its named script.
