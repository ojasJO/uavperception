# RESULTS (generated 2026-10-13 17:48 +0530 by `tools/render_results.py` — do not edit)

Every number below is read from the artifact named next to it. Status of running work: `docs/MASTER_STATE.md`.

## 1. Track A — Anti-UAV reproduction (Jiang et al. 2021, arXiv:2101.08466v3; Protocol II, IR, mSA)

PAPER = Table V; OURS — FAITHFUL = Phase A (seed 0, frozen protocol); OURS — CONTROLLED = seed-1 replication (same protocol). Columns are never mixed; differences are OURS − PAPER.

| method | split | PAPER | OURS — FAITHFUL (s0) | OURS — CONTROLLED (s1) | seed mean ± half-range | Δ mean vs paper |
|---|---|---:|---:|---:|---:|---:|
| normal | val | 79.60 | 79.29 | 79.36 | 79.33 ± 0.04 | -0.27 |
| normal | test | 65.36 | 70.58 | 70.22 | 70.40 ± 0.18 | +5.04 |
| DFSC | val | 80.09 | 79.26 | 79.43 | 79.34 ± 0.09 | -0.75 |
| DFSC | test | 66.04 | 70.29 | 70.14 | 70.21 ± 0.07 | +4.17 |

TEST ledger lines: 12 (`experiments/anti_uav/reports/test_evaluation_ledger.jsonl`). Seed-0 checkpoints have been scored on TEST four times (registered variants; see `docs/PROJECT_KNOWLEDGE.md` §A.4).

**Exploratory / Phase B variants (not reproduction columns)**

| id | split | mSA |
|---|---|---:|
| expl/absence_thr/normal_ir/val | val | 80.80 |
| expl/h4_native_scale/normal_ir/val | val | 75.26 |
| expl/absence_thr/normal_ir/test | test | 70.87 |
| expl/h4_native_scale/normal_ir/test | test | 66.15 |
| expl/absence_thr/dfsc_ir/val | val | 80.76 |
| expl/h4_native_scale/dfsc_ir/val | val | 74.65 |
| expl/absence_thr/dfsc_ir/test | test | 70.70 |
| expl/h4_native_scale/dfsc_ir/test | test | 66.01 |

**Claims (from the ledger)**: val_reproduced; test_above_paper; dfsc_effect_not_observed; absence_not_cause

**Host check HC3** (amendment A1: Phase A `normal_ir` VAL re-scored on a V100; pass if |Δ| ≤ 0.05): HPC 79.28739 vs OMEN 79.28740 (Δ -0.00001) -> PASS

**H3 (optimisation regime)**: see `experiments/anti_uav/reports/H3_optimisation_regime.md`

## 2. Track B — AeroTrack detector

### 2.1 Selected detector of the generalization series (`gen_temporal_fp_g4_full_e20`, corpus v1, TEST spent once)

| metric | TEST |
|---|---:|
| cst_drone_AP50 | 0.521 |
| anti_uav_drone_AP50 | 0.919 |
| det_fly_drone_AP50 | 0.538 |
| drone_AP50 | 0.718 |
| cst_drone_micro_AP50 | 0.425 |
| mAP50 | 0.633 |

weights sha256 `d26641b2cedb…`, set `37e944793773…` (corpus v1, CST train truncated to 84/120 sequences — data defect, see AGENT_CONTEXT §B).

### 2.2 v2 campaign on corpus v2 VAL (complete CST; HPC, fp16; amendment A1)

**HPC (registered comparisons)** — `experiments/aerotrack_v2/hpc/reports/v2_results.json`

| run | config | seed | CST AP50 | Anti-UAV AP50 | Det-Fly AP50 | pooled AP50 | CST micro AP50 | CST urban-areas |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gen_temporal_fp_g4_full_e20 | temporal_fp_g4 | 0 | 0.599 | 0.980 | 0.498 | 0.784 | 0.462 | 0.238 |
| omen_v2_stock_s0_e20 | stock | 0 | 0.566 | 0.977 | 0.471 | 0.758 | 0.450 | 0.285 |
| v2_stock_s0_e20 | stock | 0 | 0.564 | 0.979 | 0.472 | 0.755 | 0.454 | 0.276 |
| v2_sym_g4_s0_e20 | sym_g4 | 0 | 0.678 | 0.979 | 0.440 | 0.803 | 0.552 | 0.353 |
| v2_causal_g4_s0_e20 | causal_g4 | 0 | 0.676 | 0.982 | 0.445 | 0.805 | 0.555 | 0.377 |
| v2_stock_s1_e20 | stock | 1 | 0.576 | 0.981 | 0.501 | 0.768 | 0.474 | 0.284 |
| v2_sym_g4_s1_e20 | sym_g4 | 1 | 0.669 | 0.981 | 0.485 | 0.806 | 0.539 | 0.343 |
| v2_causal_g4_s1_e20 | causal_g4 | 1 | 0.686 | 0.980 | 0.487 | 0.812 | 0.539 | 0.364 |
| v2_causal_g1_s0_e20 | causal_g1 | 0 | 0.615 | 0.980 | 0.463 | 0.780 | 0.487 | 0.306 |
| v2_causal_g2_s0_e20 | causal_g2 | 0 | 0.670 | 0.976 | 0.461 | 0.800 | 0.538 | 0.382 |
| v2_causal_g8_s0_e20 | causal_g8 | 0 | 0.667 | 0.982 | 0.444 | 0.798 | 0.528 | 0.348 |
| v2_causal_g1_s1_e20 | causal_g1 | 1 | 0.589 | 0.980 | 0.488 | 0.771 | 0.458 | 0.314 |
| v2_causal_g2_s1_e20 | causal_g2 | 1 | 0.655 | 0.981 | 0.465 | 0.799 | 0.528 | 0.352 |
| v2_causal_g8_s1_e20 | causal_g8 | 1 | 0.691 | 0.983 | 0.474 | 0.812 | 0.545 | 0.367 |

**OMEN (historical, bf16; not part of any comparison)** — `experiments/aerotrack_v2/reports/v2_results.json`

| run | config | seed | CST AP50 | Anti-UAV AP50 | Det-Fly AP50 | pooled AP50 | CST micro AP50 | CST urban-areas |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gen_temporal_fp_g4_full_e20 | temporal_fp_g4 | 0 | 0.601 | 0.981 | 0.490 | 0.784 | 0.465 | 0.244 |
| v2_stock_s0_e20 | stock | 0 | 0.567 | 0.976 | 0.470 | 0.758 | 0.451 | 0.288 |

**Host checks (amendment A1; `experiments/hpc/reports/host_checks.json`)**

| check | weights | Δ CST AP50 | Δ Anti-UAV AP50 | Δ pooled AP50 | verdict |
|---|---|---:|---:|---:|---|
| HC1 eval host (v2_stock_s0_e20) | 5bba24c8dece | -0.0011 | +0.0005 | -0.0005 | PASS (|Δ| ≤ 0.005) |
| HC1 eval host (gen_temporal_fp_g4_full_e20) | d26641b2cedb | -0.0015 | -0.0004 | -0.0005 | PASS (|Δ| ≤ 0.005) |
| HC2 training host (descriptive) | v2_stock_s0 HPC − OMEN | -0.0026 | +0.0019 | -0.0027 | one sample, no decision |

**Registered decisions (HPC; `experiments/aerotrack_v2/hpc/reports/v2_decisions.json`, bootstrap B = 1000)**

| id | question | per-seed Δ CST AP50 | mean Δ | 95 % CI | Anti-UAV Δ | status |
|---|---|---|---:|---|---:|---|
| C1 | causal window vs single frame | +0.112, +0.110 | +0.1110 | [+0.071, +0.155] | +0.0009 | SUPPORTED |
| C2 | symmetric window vs single frame (replication on complete data) | +0.114, +0.093 | +0.1036 | [+0.065, +0.146] | +0.0002 | SUPPORTED |
| C3 | causal vs symmetric window (cost of removing the look-ahead) | -0.002, +0.017 | +0.0074 | [-0.000, +0.016] | +0.0007 | NON-INFERIOR (conclusive) |
| C4 | data effect: complete vs truncated CST training data (same window, seed 0; n = 1 seed, CI = sampling noise only) | +0.079 | +0.0789 | [+0.056, +0.102] | -0.0010 | SUPPORTED |

**Selected candidate (registered selection rule)**: `causal_g4`. Caveat: bootstrap = sampling noise over VAL sequences; seed noise enters only through the per-seed deltas (2 seeds); C4 compares an HPC-trained with an OMEN-trained model (training-host effect HC2 above).

### 2.3 v2 final TEST — the single registered pass (`experiments/aerotrack_v2/final_protocol.json`)

Candidate `v2_causal_g4_s0_e20` (weights `1118cf3290ae…`), tracker parameters frozen on VAL before the pass. **Note**: the pass crashed after its detection part and was finished with `--resume-after-crash` (F-05; nothing chosen after the pass).

| detection (TEST) | drone AP50 |
|---|---:|
| cst_drone_AP50 | 0.587 |
| anti_uav_drone_AP50 | 0.909 |
| det_fly_drone_AP50 | 0.466 |
| drone_AP50 | 0.732 |
| cst_drone_micro_AP50 | 0.487 |
| cst_urban-areas_drone_AP50 | 0.380 |

| tracker | group | n seq | mSA | located | false alarms (invisible frames) |
|---|---|---:|---:|---:|---:|
| ste | anti_uav_IR | 91 | 69.12 | 87.3 % | 19.6 % |
| ste | anti_uav_RGB | 91 | 74.34 | 91.9 % | 12.1 % |
| ste | cst_IR | 60 | 56.55 | 67.1 % | 33.1 % |
| argmax | anti_uav_IR | 91 | 68.89 | 87.0 % | 21.6 % |
| argmax | anti_uav_RGB | 91 | 74.10 | 91.5 % | 11.3 % |
| argmax | cst_IR | 60 | 56.28 | 67.6 % | 35.8 % |

**F1** STE vs our faithful GlobalTrack on the 91 Anti-UAV IR TEST sequences (GlobalTrack 70.58): Δ -1.46 mSA, 95 % CI [-3.00, +0.47], wins 29 / losses 62 → **no detectable difference**.

**F3** STE vs argmax: anti_uav_IR Δ +0.22 [+0.02, +0.48] → better; anti_uav_RGB Δ +0.24 [+0.01, +0.48] → better; cst_IR Δ +0.27 [-0.27, +0.79] → no detectable difference

**F2** (descriptive only): CST benchmark best published mSA 35.92 (GlobalTrack retrained on CST, arXiv:2507.23473 Table 3) vs ours 56.55 — different training data (4 domains vs CST only); the protocol-matched claim is X1's.

## 3. Deployment (OMEN RTX 5050, batch 1)

**gen_temporal_fp_g4_full_e20** (window symmetric, gap 4) — `experiments/deployment/reports/gate_gen_temporal_fp_g4_full_e20.json`

| backend | model p50 / p99 ms | end-to-end p50 / p99 ms | e2e FPS (mean) | device MB |
|---|---:|---:|---:|---:|
| torch_fp32 | 18.5 / 24.1 | 29.0 / 36.3 | 36.8 | 126 |
| torch_bf16 | 23.5 / 30.0 | 19.7 / 31.4 | 44.0 | 68 |
| torch_fp16 | 17.8 / 22.9 | 14.5 / 23.6 | 58.3 | 60 |
| ort_cuda | 9.7 / 14.4 | 9.8 / 14.8 | 91.0 | 80 |
| trt_fp32 | 3.5 / 5.1 | 7.3 / 9.8 | 137.5 | 104 |
| trt_fp16 | 3.5 / 5.5 | 5.9 / 8.3 | 156.8 | 100 |

**v2_causal_g4_s0_e20** (window causal, gap 4) — `experiments/deployment/reports/gate_v2_causal_g4_s0_e20.json`

| backend | model p50 / p99 ms | end-to-end p50 / p99 ms | e2e FPS (mean) | device MB |
|---|---:|---:|---:|---:|
| torch_fp32 | 8.4 / 17.7 | 26.3 / 33.1 | 40.3 | 126 |
| torch_bf16 | 11.8 / 23.3 | 17.1 / 30.8 | 49.4 | 68 |
| torch_fp16 | 7.9 / 16.4 | 12.9 / 22.8 | 65.2 | 54 |
| ort_cuda | 5.8 / 9.7 | 9.4 / 14.2 | 97.0 | 80 |
| trt_fp32 | 3.1 / 4.5 | 6.5 / 8.1 | 157.6 | 106 |
| trt_fp16 | 3.2 / 4.7 | 5.4 / 8.0 | 170.5 | 100 |

