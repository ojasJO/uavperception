# DRONE-Track — every existing result, in one place

**Compiled 2026-10-30.** Every number below is copied from a generated artifact in this
repository; the artifact is named under each table. Nothing here is typed from memory, and
nothing is re-derived. Where a result is **descriptive** (no pre-registered rule behind it) or
**exploratory**, it says so, because the two must not be read as equal evidence.

Two research directions share this repository and **never share a result row**:

| | Direction A — AeroTrack | Direction B — Anti-UAV (2021) reproduction |
|---|---|---|
| question | can a temporal-window detector plus a simple state estimator beat the published trackers on tiny-UAV benchmarks? | can the results of Jiang et al., *Anti-UAV* (arXiv:2101.08466) be reproduced? |
| metric | state accuracy (mSA), AP50, ADE | state accuracy (mSA), Protocol II |
| sections | 1, 3, 4, 5, 6 | 2 |

**Metric definitions.** *mSA* (state accuracy, CST Eq. 1): per frame, the IoU with the ground
truth when the target is visible, and 1 / 0 for a correct / incorrect "absent" when it is not;
averaged over frames, then over sequences. *AP50*: average precision at IoU 0.5. *ADE*: average
displacement error in pixels. *micro* target: box area below 0.03 % of the frame.

---

## 1. Headline result — CST Anti-UAV benchmark (Direction A)

The one table to put on a slide. CST Anti-UAV is a 2025 benchmark whose own paper reports that
the strongest of 21 evaluated trackers reaches 35.92 mSA. Scored with the **authors' official
toolkit**, unmodified, on the official 60-sequence TEST split.

| method | training data | mSA [95 % CI] | success AUC | P@20px | tiny-target SA | seeds |
|---|---|---:|---:|---:|---:|---:|
| GlobalTrack (released) | CST train | 35.92 *(published best)* | 35.38 | 58.72 | 15.83 | — |
| SiamDT (released) | CST train | 35.84 | 35.28 | 57.96 | 14.98 | — |
| PrDiMP (released) | CST train | 29.76 | 29.28 | 42.55 | **28.21** | — |
| **AeroTrack-X1** (causal_g4, STE) | CST train only | **60.08** [56.13, 63.97] | 58.53 | 77.42 | 39.88 | 1 |
| **AeroTrack-X2** (causal_g4, STE) | CST train only | **60.31** mean (60.08 / 60.26 / 60.60, sd 0.26) | 58.53–59.03 | 77.42–77.79 | 39.88–42.45 | 3 |
| AeroTrack-X2 single-frame control | CST train only | 48.81 mean (46.83 / 49.81 / 49.79, sd 1.72) | 45.58–48.49 | 58.60–62.33 | 31.03–31.38 | 3 |

All 21 published trackers in the benchmark table were reproduced from the authors' released
predictions and match their paper to rounding, so the 35.92 baseline is verified, not quoted.

### 1.1 The registered comparisons

| comparison | Δ mSA [95 % CI] | wins / losses | Wilcoxon p | verdict |
|---|---|---|---|---|
| X1 STE − GlobalTrack (paired, 60 seq) | **+24.16** [+20.78, +27.63] | 59 / 1 | 1.8e-11 | exceeds |
| X2 causal_g4 seed mean − GlobalTrack | **+24.39** [+21.08, +27.80] | **60 / 0** | — | exceeds |
| X2 every seed vs 35.92 | lower CI bounds 56.13 / 56.29 / 56.66 | — | — | **seed-robust** |
| X2 causal window − single frame | **+11.51** [+8.36, +14.86] | 45 / 15 | 1.2e-07 | **SUPPORTED** |
| X2 seed mean − X1 | +0.23 [−0.66, +1.14] | 32 / 28 | 0.71 | X2 confirms X1, does not improve it |
| absence-neutral: X1 − GlobalTrack *(descriptive)* | +13.00 [+10.47, +15.52] | 56 / 4 | 1.2e-10 | — |
| visible-frame IoU: X1 − GlobalTrack *(descriptive)* | +14.88 [+11.93, +17.78] | 56 / 4 | — | — |

The absence-neutral and visible-frame rows matter for honesty: roughly half the headline margin
is a better box on visible frames, and the rest is better handling of absent frames.

Source: `experiments/cst_anti_uav/reports/FINAL_REPORT.md`. Both authorised TEST passes
(PBS 93019, 93029) are spent; the TEST split is locked.

### 1.2 Known weaknesses of the headline system (from the same TEST evidence)

| weakness | magnitude | can VAL select a fix? |
|---|---|---|
| large targets | 23–27 mSA vs 34.5 overall | **no** — CST train holds 6 large-target frames; 337 of 347 VAL large frames sit in one sequence |
| false alarms on absent frames | 29–38 % of absent frames | yes — 6,250 absent frames across 30 VAL sequences |
| wrong-object boxes | about 10 % | partly |
| seed variance on hard sequences (`jungle_18`) | seed-specific lock | yes, 3 seeds exist |

---

## 2. Direction B — reproduction of Anti-UAV (Jiang et al. 2021)

Protocol II, infrared, mSA. PAPER = the paper's Table V. FAITHFUL = seed 0 under the frozen
protocol; CONTROLLED = the seed-1 replication. Columns are never mixed.

| method | split | PAPER | OURS — FAITHFUL (s0) | OURS — CONTROLLED (s1) | seed mean ± half-range | Δ vs paper |
|---|---|---:|---:|---:|---:|---:|
| normal | val | 79.60 | 79.29 | 79.36 | 79.33 ± 0.04 | **−0.27** |
| normal | test | 65.36 | 70.58 | 70.22 | 70.40 ± 0.18 | **+5.04** |
| DFSC | val | 80.09 | 79.26 | 79.43 | 79.34 ± 0.09 | −0.75 |
| DFSC | test | 66.04 | 70.29 | 70.14 | 70.21 ± 0.07 | +4.17 |

**Findings as registered**: `val_reproduced` (within 1 mSA) · `test_above_paper` (our TEST sits
about 5 mSA *above* the paper, in both arms — not explained by the conversion, which could only
lower ours) · `dfsc_effect_not_observed` (DFSC − normal: val −0.02, test −0.64; the paper's
claimed DFSC gain does not appear) · `absence_not_cause`.

**H3 (optimisation regime)**: H3a NOT_SUPPORTED — an 8-GPU regime still leaves TEST +5.04 / +3.72
above the paper, so the surplus is not an optimisation artefact. H3b NOT_OBSERVED.

| visible (RGB) arm | sequences | frames | mSA | P@20px | success AUC | FPS |
|---|---:|---:|---:|---:|---:|---:|
| normal_rgb | 91 | 83,858 | 72.88 | 93.11 | 75.43 | 6.95 |
| dfsc_rgb | 91 | 83,858 | 72.90 | 93.23 | 75.49 | 9.77 |

**Large-scale row (VAL, registered A2a)**: IR 72.32 vs paper 72.00 (+0.32, within ±1 — the
conversion is faithful); RGB 70.50 vs 67.28 (**+3.22**, outside the band, reported as a
discrepancy, not as an improvement).

**Exploratory variants** (not reproduction columns): absence-threshold normal_ir val 80.80 /
test 70.87; native-scale (H4) normal_ir val 75.26 / test 66.15.

**Host check HC3**: the seed-0 VAL run re-scored on a V100 gives 79.28739 against the laptop's
79.28740 (Δ −0.00001) → PASS. The two machines agree to five decimals.

Sources: `docs/RESULTS.md` §1, `experiments/anti_uav/reports/` (ledger: 12 TEST lines),
`experiments/anti_uav/reports/H3_optimisation_regime.md`.

---

## 3. Direction A — detector

### 3.1 The window ablation on corpus v2 VAL (complete CST, HPC, fp16)

Drone-class AP50. `stock` = single frame; `sym_g4` = symmetric window [t−4, t, t+4];
`causal_g4` = causal window [t−8, t−4, t] (no look-ahead).

| run | config | seed | CST | Anti-UAV | Det-Fly | pooled | CST micro | CST urban |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v2_stock_s0_e20 | stock | 0 | 0.564 | 0.979 | 0.472 | 0.755 | 0.454 | 0.276 |
| v2_stock_s1_e20 | stock | 1 | 0.576 | 0.981 | 0.501 | 0.768 | 0.474 | 0.284 |
| v2_causal_g1_s0_e20 | causal_g1 | 0 | 0.615 | 0.980 | 0.463 | 0.780 | 0.487 | 0.306 |
| v2_causal_g1_s1_e20 | causal_g1 | 1 | 0.589 | 0.980 | 0.488 | 0.771 | 0.458 | 0.314 |
| v2_causal_g2_s0_e20 | causal_g2 | 0 | 0.670 | 0.976 | 0.461 | 0.800 | 0.538 | 0.382 |
| v2_causal_g2_s1_e20 | causal_g2 | 1 | 0.655 | 0.981 | 0.465 | 0.799 | 0.528 | 0.352 |
| **v2_causal_g4_s0_e20** | **causal_g4** | 0 | **0.676** | 0.982 | 0.445 | 0.805 | 0.555 | 0.377 |
| **v2_causal_g4_s1_e20** | **causal_g4** | 1 | **0.686** | 0.980 | 0.487 | 0.812 | 0.539 | 0.364 |
| v2_causal_g8_s0_e20 | causal_g8 | 0 | 0.667 | 0.982 | 0.444 | 0.798 | 0.528 | 0.348 |
| v2_causal_g8_s1_e20 | causal_g8 | 1 | 0.691 | 0.983 | 0.474 | 0.812 | 0.545 | 0.367 |
| v2_sym_g4_s0_e20 | sym_g4 | 0 | 0.678 | 0.979 | 0.440 | 0.803 | 0.552 | 0.353 |
| v2_sym_g4_s1_e20 | sym_g4 | 1 | 0.669 | 0.981 | 0.485 | 0.806 | 0.539 | 0.343 |
| gen_temporal_fp_g4_full_e20 | temporal_fp_g4 | 0 | 0.599 | 0.980 | 0.498 | 0.784 | 0.462 | 0.238 |
| laptop_v2_stock_s0_e20 | stock | 0 | 0.566 | 0.977 | 0.471 | 0.758 | 0.450 | 0.285 |

Source: `experiments/aerotrack_v2/hpc/reports/v2_results.json`.

### 3.2 The registered decisions behind the selected detector (bootstrap B = 1000)

| id | question | per-seed Δ CST AP50 | mean | 95 % CI | Anti-UAV Δ | status |
|---|---|---|---:|---|---:|---|
| C1 | causal window vs single frame | +0.112, +0.110 | **+0.1110** | [+0.071, +0.155] | +0.0009 | **SUPPORTED** |
| C2 | symmetric window vs single frame | +0.114, +0.093 | +0.1036 | [+0.065, +0.146] | +0.0002 | SUPPORTED |
| C3 | causal vs symmetric (cost of dropping the look-ahead) | −0.002, +0.017 | +0.0074 | [−0.000, +0.016] | +0.0007 | **NON-INFERIOR** |
| C4 | complete vs truncated CST training data (1 seed) | +0.079 | +0.0789 | [+0.056, +0.102] | −0.0010 | SUPPORTED |

**Selected**: `causal_g4` — the full temporal gain with **zero frames of look-ahead**, which is
what makes the system deployable on a live feed.

*Caveat as registered*: the bootstrap is sampling noise over VAL sequences; seed noise enters
only through two per-seed deltas. C4 compares an HPC-trained with a laptop-trained model.

### 3.3 The single registered TEST pass of the v2 detector

Candidate `v2_causal_g4_s0_e20`, tracker parameters frozen on VAL before the pass.

| detection (TEST) | drone AP50 | | tracker | group | n | mSA | located | false alarms |
|---|---:|---|---|---|---:|---:|---:|---:|
| CST | 0.587 | | ste | anti_uav IR | 91 | 69.12 | 87.3 % | 19.6 % |
| Anti-UAV | 0.909 | | ste | anti_uav RGB | 91 | 74.34 | 91.9 % | 12.1 % |
| Det-Fly | 0.466 | | ste | cst IR | 60 | 56.55 | 67.1 % | 33.1 % |
| pooled | 0.732 | | argmax | anti_uav IR | 91 | 68.89 | 87.0 % | 21.6 % |
| CST micro | 0.487 | | argmax | anti_uav RGB | 91 | 74.10 | 91.5 % | 11.3 % |
| CST urban | 0.380 | | argmax | cst IR | 60 | 56.28 | 67.6 % | 35.8 % |

**F1**: the state estimator vs our own faithful GlobalTrack on 91 Anti-UAV IR TEST sequences
(GlobalTrack 70.58) — Δ −1.46 [−3.00, +0.47], 29 wins / 62 losses → **no detectable difference**.
**F3**: STE vs argmax — better on IR (+0.22 [+0.02, +0.48]) and RGB (+0.24 [+0.01, +0.48]), no
detectable difference on CST.

*Procedural note*: the pass crashed after its detection stage and was completed with a guarded
resume (failure F-05). Nothing was selected after the pass.

Source: `experiments/aerotrack_v2/hpc/reports/final_test.json`.

### 3.4 Multi-object tracking metrics (TEST)

| group | HOTA | AssA | IDF1 | MOTA |
|---|---:|---:|---:|---:|
| Anti-UAV IR | 0.675 | 0.711 | 0.896 | 0.820 |
| Anti-UAV RGB | 0.717 | 0.740 | 0.916 | 0.866 |
| CST | 0.479 | 0.537 | 0.694 | 0.438 |

### 3.5 Host agreement

| check | subject | Δ CST AP50 | Δ Anti-UAV | Δ pooled | verdict |
|---|---|---:|---:|---:|---|
| HC1 | eval host, `v2_stock_s0_e20` | −0.0011 | +0.0005 | −0.0005 | PASS (\|Δ\| ≤ 0.005) |
| HC1 | eval host, `gen_temporal_fp_g4_full_e20` | −0.0015 | −0.0004 | −0.0005 | PASS |
| HC2 | training host, HPC − laptop | −0.0026 | +0.0019 | −0.0027 | one sample, no decision |

---

## 4. Direction A — trajectory forecasting and state estimation (Track C)

All stages are VAL except where the single registered TEST pass is named. These are the results
that **did not** support the obvious hypotheses, which is why they are reported in full.

### 4.1 Does a learned forecaster beat a classical motion model? (C1 / C1c, VAL ADE px)

| domain | learned vs classical (pooled C1) | per-domain arm (C1c) vs classical | C1c vs pooled C1 |
|---|---|---|---|
| Anti-UAV IR | **SUPPORTED** −3.54 [−4.31, −2.80] | **SUPPORTED** −2.53 [−3.26, −1.81] | NOT_SUPPORTED |
| Anti-UAV RGB | **SUPPORTED** −8.55 [−10.33, −6.82] | **SUPPORTED** −9.68 [−11.85, −7.64] | NOT_SUPPORTED |
| CST | **LOSES** +5.35 [+3.73, +7.01] | NOT_SUPPORTED −0.14 [−0.84, +0.52] | **SUPPORTED** −5.49 [−6.76, −4.29] |

Reading: learned forecasting helps on Anti-UAV and **loses** on CST. Training per domain removes
CST's negative transfer (−5.5 px) but still does not beat the classical `dcv` model there, so
**no learned forecaster was adopted for CST**.

### 4.2 Does a motion model improve tracking? (C3, VAL mSA)

| hypothesis | Anti-UAV IR | Anti-UAV RGB | CST |
|---|---|---|---|
| H1 temporal state (STE − argmax) | +0.07 [−0.06, +0.25] no difference | −0.18 [−0.57, +0.04] no difference | **+0.84 [+0.24, +1.61] SUPPORTED** (25/13) |
| H2 domain motion model | +0.00 no difference | +0.03 no difference | +0.02 no difference |
| H3 learned motion model | −0.10 no difference | +0.19 no difference | not run |

Reading: **motion models never improve tracking.** Keeping a temporal state helps only on CST,
and only with the temporal detector.

### 4.3 Forecasting on tracker histories (C1b) and augmentation (H4)

| domain | anchors | point forecast (learned − classical ADE) | uncertainty (learned − Kalman NLL) | H4 augmentation |
|---|---:|---|---|---|
| Anti-UAV IR | 59,939 | **SUPPORTED** −3.44 [−4.20, −2.72] | **SUPPORTED** −1.33 [−1.63, −1.02] | **REVERSED** (hurts) +0.64 |
| Anti-UAV RGB | 57,498 | **SUPPORTED** −8.03 [−9.88, −6.34] | NOT_SUPPORTED −0.69 [−1.49, +0.36] | **REVERSED** +1.37 |
| CST | 32,251 | NOT_SUPPORTED −2.10 [−6.11, +1.54] | **LOSES** +745.9 [+274.7, +1348.0] | NOT_SUPPORTED |

The CST uncertainty head is badly miscalibrated — a loss of three orders of magnitude, reported
rather than hidden. Augmentation **hurts** on both Anti-UAV modalities.

### 4.4 The complete causal final system (CF) — single registered TEST pass

| group | tracker | sequences | mSA | located | false alarms | median recovery |
|---|---|---:|---:|---:|---:|---:|
| Anti-UAV IR | argmax | 91 | 68.89 | 87.0 % | 21.6 % | 0 frames |
| Anti-UAV RGB | argmax | 91 | 74.10 | 91.5 % | 11.3 % | 3 frames |
| CST | ste_cv | 60 | 56.55 | 67.1 % | 33.1 % | 20 frames |

CF-a vs Track A GlobalTrack: −1.69 [−3.25, +0.06] (no detectable difference).
CF-c vs argmax on CST: +0.27 [−0.27, +0.79] (no detectable difference).

| domain | anchors | ADE px: adopted / Kalman / constant-position | vs Kalman | vs constant position |
|---|---:|---|---|---|
| Anti-UAV IR | 75,530 | 38.9 / 43.6 / 43.6 | **SUPPORTED** −4.65 [−5.30, −3.98] | **SUPPORTED** |
| Anti-UAV RGB | 75,081 | 94.7 / 106.4 / 106.4 | **SUPPORTED** −11.66 [−13.42, −9.81] | **SUPPORTED** |
| CST | 47,679 | 49.9 / 51.2 / **47.5** | NOT_SUPPORTED −1.26 (below the 5 % margin) | NOT_SUPPORTED |

On CST, **predicting that the target does not move beats every forecaster we trained.**

**P1 presence gate** (exploratory, CST VAL only): NO_GAIN — the gate trades absent-frame false
alarms for visible misses one for one; cross-fitted selection keeps the threshold at 0. A
detection-score gate is not the fix.

Source: `experiments/aerotrack_temporal/reports/TRACK_C_SUMMARY.md`.

---

## 5. Deployment — measured latency

Laptop RTX 5050 (8 GB, Blackwell sm_120), batch 1, 640×640, end-to-end = JPEG decode →
letterbox → window assembly → inference → NMS.

**Selected model `v2_causal_g4_s0_e20`** (causal window, 0 frames of look-ahead):

| backend | model p50 / p99 ms | end-to-end p50 / p99 ms | e2e FPS | device MB |
|---|---:|---:|---:|---:|
| torch fp32 | 8.4 / 17.7 | 26.3 / 33.1 | 40.3 | 126 |
| torch bf16 | 11.8 / 23.3 | 17.1 / 30.8 | 49.4 | 68 |
| torch fp16 | 7.9 / 16.4 | 12.9 / 22.8 | 65.2 | 54 |
| ONNX Runtime CUDA | 5.8 / 9.7 | 9.4 / 14.2 | 97.0 | 80 |
| TensorRT fp32 | 3.1 / 4.5 | 6.5 / 8.1 | 157.6 | 106 |
| **TensorRT fp16** | **3.2 / 4.7** | **5.4 / 8.0** | **170.5** | 100 |

**Symmetric-window model `gen_temporal_fp_g4_full_e20`** (needs 4 frames of look-ahead):

| backend | model p50 / p99 ms | end-to-end p50 / p99 ms | e2e FPS | device MB |
|---|---:|---:|---:|---:|
| torch fp32 | 18.5 / 24.1 | 29.0 / 36.3 | 36.8 | 126 |
| torch fp16 | 17.8 / 22.9 | 14.5 / 23.6 | 58.3 | 60 |
| ONNX Runtime CUDA | 9.7 / 14.4 | 9.8 / 14.8 | 91.0 | 80 |
| TensorRT fp32 | 3.5 / 5.1 | 7.3 / 9.8 | 137.5 | 104 |
| TensorRT fp16 | 3.5 / 5.5 | 5.9 / 8.3 | 156.8 | 100 |

**AP50 parity after export**: ≤ 0.003 against the PyTorch model. **Look-ahead**: 0 frames for
`causal_g4`. **Server-to-laptop parity**: presence 100 %, top-1 IoU mean 0.985 (min 0.944),
|Δ score| 0.0003. **Tracker cost**: STE 0.126–0.145 ms/frame, argmax 0.012–0.017 ms/frame.

Sources: `experiments/deployment/reports/gate_v2_causal_g4_s0_e20.md`,
`experiments/deployment/reports/gate_gen_temporal_fp_g4_full_e20.json`,
`deployment/aerotrack_causal_g4_v2/` (17 sha256-verified files).

---

## 6. Datasets and data integrity

| dataset | content | splits (sequences) | role |
|---|---|---|---|
| Anti-UAV (2021) | 318 IR/RGB sequence pairs, 593,802 frames | train 160 / val 67 / test 91 | Direction B reproduction; Direction A training + evaluation |
| CST Anti-UAV (2025) | 144 IR sequences in the corpus (220 in the official release), 162,187 frames | train 120 / val 40 / test 60 | the headline benchmark |
| Det-Fly | 7,830 daylight images, `Bird` / `Drone` labels | image-level | hard-negative source (bird rejection) |

- Splits are **sequence-level** (seed 1337), with leakage assertions in the pipeline; no sequence
  name appears in two splits.
- CST ships no VAL split, so a fixed-seed holdout is carved from its train split; **CST TEST is
  never used for selection.**
- `DATA_VERSION` = `corpus_v2_f842d03909cf_cst_full_antiuav_hpcdecode`.
- The CST extraction on the HPC gives 246,131 files with **0** missing / size / CRC mismatches,
  identical to the laptop; the data gate passes on 602,613 TRAIN/VAL paths.
- The Anti-UAV frames were independently re-decoded on the HPC: **byte-identical** to the laptop
  across all 636 sequences (frame counts, 1,908 boundary hashes, a 2,000-frame random sample, and
  the labels).
- Known caveats, as registered: Anti-UAV VAL shares all 26 recording sessions with TRAIN, so VAL
  is optimistic; CST VAL shares source videos with TRAIN; CST TRAIN holds only 6 large-target
  frames.

Live, filesystem-checked version of this table: `venv\Scripts\python.exe tools\show_datasets.py`.

---

## 7. What was tested and retired

Reporting only the wins would misrepresent the work. These components were built, measured
against a trivial baseline under a pre-registered rule, and **retired**:

| component | the rule it failed | what beat it |
|---|---|---|
| ByteTrack association | MOTA / IDF1 on every dataset | a no-association argmax box |
| LSTM forecaster | ADE on CST | mean-velocity / Kalman; on gimbal footage, predicting no motion |
| SPD-Conv (as an isolated factor) | > 0.02 AP50 on both `best.pt` and `last.pt` | no measurable effect on micro AP50 |
| learned CST forecaster | ADE vs classical `dcv` | the classical model |
| P1 presence gate | cross-fitted CST VAL mSA | the ungated output |
| H4 trajectory augmentation | ADE | no augmentation (augmentation *hurts*) |

An earlier revision of this project claimed "+43 % mAP from fine-tuning", "ByteTrack occlusion
recovery" and "LSTM beats Kalman by 26 %". A controlled re-measurement overturned all three; the
audit that forced it is `audit/AUDIT_REPORT.md` and the item-by-item repair log is
`docs/AEROTRACK_REPAIR_REPORT.md`. The numbers in this document are the post-audit ones.

---

## 8. Where every artifact lives

| what | path |
|---|---|
| this table set | `SUBMISSION/RESULTS_ALL.md` |
| auto-generated results (regenerate with `tools/render_results.py`) | `docs/RESULTS.md` |
| CST benchmark study | `experiments/cst_anti_uav/reports/FINAL_REPORT.md` |
| Anti-UAV reproduction | `experiments/anti_uav/reports/`, `experiments/anti_uav/README.md` |
| detector campaign | `experiments/aerotrack_v2/hpc/reports/` |
| Track C | `experiments/aerotrack_temporal/reports/TRACK_C_SUMMARY.md` |
| deployment gates | `experiments/deployment/reports/` |
| every checkpoint identity (SHA-256) | `registry/models.json`, `registry/checkpoints.csv` |
| experiment ledger | `docs/EXPERIMENT_LEDGER.csv`, `docs/EXPERIMENT_LEDGER.jsonl` |
| decisions and their reasons | `docs/DECISIONS.md` |
| failures and recoveries | `docs/FAILURES_AND_RECOVERY.md` |
| HPC job records (46 jobs: command, environment, events, stdout) | `~/DRONE-Track/logs/pbs_jobs/` on the HPC |
