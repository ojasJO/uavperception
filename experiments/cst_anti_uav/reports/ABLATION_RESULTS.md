# CST Anti-UAV — ablations (X2: training seed × temporal window vs single frame)

Registered experiment: `preregistration/X2_seeds_single_frame.json`.
- Frozen settings `frozen_x2_params.json` (commit 3ffee11), written before any X2 TEST access.
- Single TEST pass: PBS 93029, node02, exit 0, 74.5 min. Ledger: `x2_test_ledger.jsonl`.
- Official scoring: `X2_test_official.{json,md}`. Descriptive analysis: `X2_failure_analysis.json`.
- Every TEST number below is the unmodified official toolkit's. It equals each pass's own Track A value to max per-sequence
  |Δ| = 0 across 12 tracker outputs.

**One factor, everything else fixed.**

| | causal_g4 arm | ttt arm (single-frame control) |
|---|---|---|
| network | YOLO11n 'n', 9-channel window stem, function-preserving init from COCO `yolo11n.pt` | identical |
| input slots | frames (t−4, t, t−8) | (t, t, t) |
| training | CST train only (120 sequences), 20 epochs × 25,000 samples, batch 32, AdamW 1e-3 cosine, fp16, `last.pt` | identical |
| tracker | STE / argmax, settings tuned on ALL CST VAL per run | identical procedure |
| seeds | 0, 1, 2 (seed 0 = X1's run, its TEST pass reused, never re-scored) | 0, 1, 2 |

Training jobs: PBS 93016 (exit 0) and 93018 (exit 0).
- Every run is a single attempt with no resume.
- `args.json` is identical across runs except name and seed.
- Every `last.pt` sha256 equals its `run_meta`, re-checked on the HPC.
- Weights sha256: causal_g4 s0 `e52a5645…`, s1 `b716d1ae…`, s2 `5f5c9e40…`; ttt s0 `c04cba26…`, s1 `ee3da997…`, s2 `3b062e49…`.

## 1. VAL design (PBS 93020): descriptive

CST VAL shares source videos with CST TRAIN (paper §3.3), so VAL is optimistic. No model selection was done: all six runs go to
TEST.

| run | CST AP50 | STE mSA cross-fitted | STE mSA tuned on all VAL | argmax cross-fitted |
|---|---:|---:|---:|---:|
| causal_g4 s0 (X1) | 0.694 | 63.61 | 64.03 | 62.60 |
| causal_g4 s1 | 0.700 | 65.08 | 65.76 | 64.38 |
| causal_g4 s2 | 0.703 | 64.46 | 65.20 | 63.52 |
| ttt s0 | 0.540 | 55.40 | 55.94 | 54.17 |
| ttt s1 | 0.575 | 55.47 | 56.66 | 54.84 |
| ttt s2 | 0.540 | 55.15 | 55.85 | 54.26 |

V1, causal_g4 minus single frame (seed-mean, cross-fitted STE, 40 sequences): **+9.04 SA [+6.38, +12.03]**, 36 wins / 4 losses.
Per seed: +8.21, +9.62, +9.30. Detection AP50 per seed: +0.154, +0.126, +0.163.

## 2. TEST: registered tests T1–T3

| run | official mSA | 95 % CI | vs 35.92 (T1) | success AUC | P@20px | tiny SA | argmax mSA |
|---|---:|---|---|---:|---:|---:|---:|
| causal_g4 s0 (= X1) | 60.08 | [56.13, 63.97] | exceeds | 58.53 | 77.42 | 39.88 | 59.70 |
| causal_g4 s1 | 60.26 | [56.29, 64.17] | exceeds | 58.72 | 77.61 | 40.89 | 59.68 |
| causal_g4 s2 | **60.60** | [56.66, 64.40] | exceeds | 59.03 | 77.79 | 42.45 | 60.49 |
| ttt s0 | 46.83 | [40.80, 52.84] | exceeds | 45.58 | 58.60 | 31.26 | 48.68 |
| ttt s1 | 49.81 | [44.10, 55.47] | exceeds | 48.49 | 62.33 | 31.03 | 51.29 |
| ttt s2 | 49.79 | [43.86, 55.74] | exceeds | 48.46 | 61.94 | 31.38 | 49.26 |

- **T1, seed robustness:** every seed of both arms exceeds the best published CST tracker (35.92), so the claim is seed-robust.
  - causal_g4: seed mean **60.31**, sd 0.26.
  - ttt: seed mean 48.81, sd 1.72.
- **T2, seed-mean paired vs the released CST-trained GlobalTrack (60 sequences):**
  - causal_g4: **+24.39 SA [+21.08, +27.80], 60 wins / 0 losses**.
  - ttt: +12.88 [+8.78, +16.98], 43 / 17.
- **T3, temporal window vs single frame (seed means, paired):** **+11.51 SA [+8.36, +14.86]**, 45 / 15, Wilcoxon p = 1.2e-07 →
  **SUPPORTED**. Per seed: +13.26 [+9.35, +17.22], +10.46 [+7.39, +13.75], +10.81 [+7.12, +14.76].

The VAL estimate of the window effect (+9.04) agrees in sign and size with TEST (+11.51).

## 3. What the temporal window changes (TEST, descriptive; `X2_failure_analysis.json`)

| per run | visible: IoU ≥ 0.5 | visible: box at IoU 0 | visible: no box | absent: correct | re-appearance recovered ≤ 1 s | never within 10 s |
|---|---:|---:|---:|---:|---:|---:|
| causal_g4 s0 / s1 / s2 | 70.9 / 72.4 / 71.6 % | **9.6 / 10.4 / 10.1 %** | 13.5 / 11.4 / 12.4 % | 71.0 / 62.3 / 69.3 % | **56.4 / 61.5 / 59.2 %** | 25 / 21 / 21 |
| ttt s0 / s1 / s2 | 54.0 / 58.4 / 57.0 % | 26.9 / 21.4 / 23.9 % | 16.4 / 17.8 / 16.5 % | 56.5 / 60.6 / 64.2 % | 35.4–38.5 % | 65 / 76 / 77 |

The window's gain is mostly **fewer wrong-object boxes**: IoU-0 frames fall from about 24 % to about 10 % of visible frames.
Re-acquisition after re-appearance is also faster, with the median falling from 31–34 to 10–13 frames. Box precision on frames
that are already correct changes little. A plausible interpretation, not tested here: the t−4 and t−8 frames give the
detector motion evidence that separates a moving drone from static hot spots.

Frame-level attributes, seed mean (min–max), official frame-score definition:

| arm | OC | OV | SV | TC | DBC | CDB | FM (93 frames) | tiny | small | normal | large (484) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| causal_g4 | 67.71 (60.60–74.51) | 66.90 | 52.81 | 50.89 | 63.85 | 50.44 | 30.18 | **41.07** (39.88–42.45) | 60.67 | 68.39 | 25.48 (23.38–27.08) |
| ttt | 65.68 | 53.99 | 40.29 | 39.88 | 56.02 | 39.01 | 22.38 | 31.22 | 47.91 | 51.15 | 14.72 |

## 4. X2 vs X1

X1 is the causal_g4 seed-0 run. The causal_g4 seed mean minus X1, paired per sequence, is **+0.23 SA [−0.66, +1.14]**: 32 wins /
28 losses, Wilcoxon p = 0.71. Per seed: s1 +0.18 [−1.58, +1.84], s2 +0.52 [−0.75, +1.84].

X2 does **not** improve on X1; it shows X1 was representative of its configuration. The best single seed (s2, 60.60) is not a
selected candidate, because the registration selects nothing on TEST, and must not be reported as "the" result.

## 5. Other ablations in this project (cited, not re-run)

- **Tracker (STE vs argmax) on CST TEST:** paired with the same detections, STE − argmax is +0.38 (X1), +0.58 (s1), +0.11 (s2) mSA.
  The registered VAL test of the Kalman state is Track C C3 (H1 SUPPORTED on CST: +0.84 [+0.24, +1.61], v2 detector).
- **Training data (CST only vs CST + Anti-UAV + Det-Fly):** the v2 all-domain run `v2_causal_g4_s0_e20` scored 56.55 CST TEST STE
  mSA. It is unpaired with X1 and from a separate pass, so it is descriptive only: extra-domain data did not help CST.
- **Motion models, learned forecasters, causal window widths:** Track C (`experiments/aerotrack_temporal/reports/TRACK_C_SUMMARY.md`).
  Those are VAL-only and measured in Track C's metrics, never mixed with the CST benchmark.
