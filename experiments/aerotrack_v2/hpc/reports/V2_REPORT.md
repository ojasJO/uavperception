# AeroTrack v2 series — complete CST corpus, causal temporal window

*Generated 2026-10-06 14:13 by `scripts/report_v2.py` from the series' JSON; rules frozen in `preregistration.json` (committed before the corpus existed). No number here is typed by hand.*

## 1. Questions (pre-registered)

- **C1** — causal window vs single frame (rule: superiority, metric: cst)
- **C2** — symmetric window vs single frame (replication on complete data) (rule: superiority, metric: cst)
- **C3** — causal vs symmetric window (cost of removing the look-ahead) (rule: non_inferiority, metric: cst)
- **C4** — data effect: complete vs truncated CST training data (same window, seed 0; n = 1 seed, CI = sampling noise only) (rule: superiority, metric: cst)

Configurations: `stock` YOLO11n stride layout, input [t, t, t]; `sym_g4` same network, input [t-4, t, t+4] (the generalization series' supported window); `causal_g4` same network, input [t-4, t, t-8]: past frames only, zero look-ahead

Held fixed: c_model/train.py STAGES['baseline'] recipe (20 epochs x 25,000 weighted samples, batch 32 x accumulate 2, AdamW 1e-3 cosine), letterbox 640, augmentation, sampler, function-preserving initialisation (every configuration starts as COCO YOLO11n on frame t), checkpoint = last.pt

## 2. Data: corpus v2

Corpus v2 (`data_unified_v2`, manifest `f842d03909cf`): CST sequences train 120, val 40, test 60; CST TEST labels identical to corpus v1: True; official CST val split used: True.

| split | CST categories |
|---|---|
| train | building 51, cn_mountains 20, cn_sky 8, jungle 5, urban-areas 23, water 13 |
| val | building 14, cn_mountains 9, cn_sky 2, jungle 1, urban-areas 8, water 6 |
| test | building 9, cn_mountains 9, cn_sky 6, jungle 13, urban-areas 17, water 6 |

## 3. Runs

| run | config | seed | epochs | training h | init check | last.pt sha256 |
|---|---|---|---:|---:|---|---|
| `gen_temporal_fp_g4_full_e20` | temporal_fp_g4 | 0 | 20 | 2.21 | ok | `d26641b2cedb` |
| `laptop_v2_stock_s0_e20` | stock | 0 | 20 | 2.14 | — | `5bba24c8dece` |
| `v2_causal_g4_s0_e20` | causal_g4 | 0 | 20 | 1.02 | ok | `1118cf3290ae` |
| `v2_causal_g4_s1_e20` | causal_g4 | 1 | 20 | 1.05 | ok | `2d6aceab5ccf` |
| `v2_stock_s0_e20` | stock | 0 | 20 | 1.03 | ok | `5287cda7ddf7` |
| `v2_stock_s1_e20` | stock | 1 | 20 | 1.02 | ok | `344047cf648e` |
| `v2_sym_g4_s0_e20` | sym_g4 | 0 | 20 | 1.03 | ok | `4c80bcf44ec4` |
| `v2_sym_g4_s1_e20` | sym_g4 | 1 | 20 | 1.02 | ok | `8eb2667145a1` |

## 4. VAL results (corpus v2, every dataset capped at 5,000 images, same frames for every run)

| run | drone AP50 | CST | CST micro | Anti-UAV | Det-Fly | mAP50 | CST building | CST cn_mountains | CST cn_sky | CST jungle | CST urban-areas | CST water |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gen_temporal_fp_g4_full_e20` | 0.7839 | 0.5992 | 0.4621 | 0.9801 | 0.4981 | 0.6611 | 0.5796 | 0.7321 | 0.7516 | 1.0000 | 0.2380 | 0.8870 |
| `laptop_v2_stock_s0_e20` | 0.7579 | 0.5662 | 0.4498 | 0.9768 | 0.4708 | 0.6423 | 0.5446 | 0.5985 | 0.7498 | 0.9998 | 0.2851 | 0.9573 |
| `v2_causal_g4_s0_e20` | 0.8054 | 0.6760 | 0.5551 | 0.9820 | 0.4447 | 0.6627 | 0.6382 | 0.7876 | 0.8238 | 1.0000 | 0.3767 | 0.9508 |
| `v2_causal_g4_s1_e20` | 0.8124 | 0.6857 | 0.5389 | 0.9800 | 0.4869 | 0.6628 | 0.6433 | 0.8399 | 0.7226 | 1.0000 | 0.3635 | 0.9411 |
| `v2_stock_s0_e20` | 0.7552 | 0.5636 | 0.4537 | 0.9787 | 0.4720 | 0.6337 | 0.5470 | 0.6113 | 0.7606 | 0.9985 | 0.2764 | 0.9426 |
| `v2_stock_s1_e20` | 0.7678 | 0.5761 | 0.4743 | 0.9814 | 0.5013 | 0.6529 | 0.5594 | 0.5968 | 0.7165 | 1.0000 | 0.2841 | 0.9420 |
| `v2_sym_g4_s0_e20` | 0.8030 | 0.6780 | 0.5515 | 0.9791 | 0.4400 | 0.6546 | 0.6479 | 0.7971 | 0.7855 | 1.0000 | 0.3535 | 0.9557 |
| `v2_sym_g4_s1_e20` | 0.8056 | 0.6689 | 0.5387 | 0.9814 | 0.4847 | 0.6446 | 0.6334 | 0.8001 | 0.7773 | 1.0000 | 0.3429 | 0.9496 |

## 5. Decisions (pre-registered rules)

| id | question | per seed | mean Δ | 95 % CI | Anti-UAV Δ | status |
|---|---|---|---:|---|---:|---|
| C1 | causal window vs single frame | +0.112, +0.110 | +0.1110 | [+0.071, +0.155] | +0.0009 | SUPPORTED |
| C2 | symmetric window vs single frame (replication on complete data) | +0.114, +0.093 | +0.1036 | [+0.065, +0.146] | +0.0002 | SUPPORTED |
| C3 | causal vs symmetric window (cost of removing the look-ahead) | -0.002, +0.017 | +0.0074 | [-0.000, +0.016] | +0.0007 | NON-INFERIOR (conclusive) |
| C4 | data effect: complete vs truncated CST training data (same window, seed 0; n = 1 seed, CI = sampling noise only) | +0.079 | +0.0789 | [+0.056, +0.102] | -0.0010 | SUPPORTED |

**Selected candidate (selection rule): `causal_g4`**

Per-category CST deltas (seed mean):

- C1: building +0.088, cn_mountains +0.210, cn_sky +0.035, jungle +0.001, urban-areas +0.090, water +0.004
- C2: building +0.087, cn_mountains +0.195, cn_sky +0.043, jungle +0.001, urban-areas +0.068, water +0.010
- C3: building +0.000, cn_mountains +0.015, cn_sky -0.008, jungle +0.000, urban-areas +0.022, water -0.007
- C4: building +0.068, cn_mountains +0.065, cn_sky +0.034, jungle +0.000, urban-areas +0.115, water +0.069

## 6. Selected candidate as a system (VAL)

Deployment gate on NVIDIA GeForce RTX 5050 Laptop GPU (window `causal`, batch 1):

| backend | model p50 / p95 / p99 ms | end-to-end p50 / p99 ms | device MB | drone AP50 (Δ vs bf16) |
|---|---|---|---:|---|
| torch_fp32 | 8.36 / 13.65 / 17.74 | 26.27 / 33.06 | 126 | 0.7300 (+0.0026) |
| torch_bf16 | 11.79 / 19.65 / 23.33 | 17.10 / 30.83 | 68 | 0.7274 (+0.0000) |
| torch_fp16 | 7.92 / 14.44 / 16.44 | 12.94 / 22.75 | 54 | 0.7288 (+0.0013) |
| ort_cuda | 5.82 / 9.33 / 9.73 | 9.41 / 14.21 | 80 | 0.7298 (+0.0024) |
| trt_fp32 | 3.11 / 3.99 / 4.49 | 6.45 / 8.05 | 106 | 0.7279 (+0.0005) |
| trt_fp16 | 3.20 / 4.34 / 4.66 | 5.43 / 7.99 | 100 | 0.7290 (+0.0015) |

Single-object tracking on VAL (cross-fitted parameters):

| tracker | group | mSA | located | false alarms | recover median |
|---|---|---:|---:|---:|---:|
| argmax | anti_uav_IR | 80.46 | 98.5 % | 13.5 % | 1.0 |
| argmax | anti_uav_RGB | 81.38 | 98.2 % | 26.4 % | 0.0 |
| argmax | cst_IR | 62.18 | 73.3 % | 20.2 % | 14.0 |
| ste | anti_uav_IR | 80.53 | 98.5 % | 9.8 % | 1.0 |
| ste | anti_uav_RGB | 81.20 | 97.9 % | 25.8 % | 1.0 |
| ste | cst_IR | 63.02 | 73.7 % | 17.0 % | 18.0 |

GlobalTrack (Track A) on the same 67 Anti-UAV IR VAL sequences: mSA 79.29; STE − GlobalTrack +1.24 [-0.44, +4.29].

- Established trackers (MOT protocol): `experiments/systems/reports/mot_v2_causal_g4_s0_e20_data_unified_v2.md`
- Forecasting: `experiments/systems/reports/forecast_v2_causal_g4_s0_e20_data_unified_v2.md`
- Failure analysis: `experiments/systems/reports/failures_v2_causal_g4_s0_e20_data_unified_v2.md`

## 7. Limitations (structural)

- 2 seeds per configuration; the bootstrap measures sampling noise over VAL sequences, seed noise enters only through the per-seed deltas.
- Anti-UAV VAL shares all recording sessions with TRAIN (optimistic for Anti-UAV); CST val is the official split.
- C4 (data effect) has one seed per arm.
- TEST is untouched by this report; the candidate's single pass follows `final_protocol.json`.
