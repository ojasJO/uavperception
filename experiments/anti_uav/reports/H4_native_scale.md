# H4 — does the test-time input scale explain the test surplus?

*Generated 2026-10-05T06:03:57 by `scripts/analyze_h4.py` from saved `metrics.json` files; rule and predictions frozen in `experiments\anti_uav\preregistration\H4_native_scale.json` before any full run. Class: EXPLORATORY EXPERIMENT (Phase B additional control on frozen checkpoints; never merged into the reproduction columns). Do not edit numbers by hand.*

**Verdict (pre-registered rule): NOT_SUPPORTED_VAL_BREAKS** — native evaluation no longer reproduces the paper's validation for at least one model.

Controlled variable: test-time rescale target only — reference = the Phase A evaluations (IR 640×512 rescaled to 1000×800 by GlobalTrack's 1333×800 rule), native = 640×512 without resizing. Same frozen final-epoch checkpoints, tracker, metric and split manifests.

| model | split | paper | reference (1000×800) | native (640×512) | Δ native − reference | native − paper | fps ref → native |
|---|---|---:|---:|---:|---:|---:|---|
| normal_ir | val | 79.60 | 79.29 | 75.26 | -4.02 | -4.34 | 9.7 → 12.8 |
| normal_ir | test | 65.36 | 70.58 | 66.15 | -4.43 | +0.79 | 10.0 → 12.5 |
| dfsc_ir | val | 80.09 | 79.26 | 74.65 | -4.61 | -5.44 | 9.9 → 12.1 |
| dfsc_ir | test | 66.04 | 70.29 | 66.01 | -4.28 | -0.03 | 9.1 → 11.9 |

Val/test asymmetry of the scale effect (Δtest − Δval; the pre-registered trigger for a train-at-native follow-up is ≤ −2 pt for both models): normal_ir -0.41, dfsc_ir +0.33

## Per-attribute effect (Δ native − reference, mSA pt)

| attribute | normal_ir val (n) | normal_ir test (n) | dfsc_ir val (n) | dfsc_ir test (n) |
|---|---:|---:|---:|---:|
| OV | -2.63 (4) | -5.48 (18) | -2.32 (4) | -5.93 (18) |
| OC | -2.09 (5) | -7.97 (2) | -2.99 (5) | -9.44 (2) |
| FM | -3.78 (23) | -9.45 (31) | -4.31 (23) | -8.99 (31) |
| SV | -7.55 (17) | -9.14 (18) | -8.45 (17) | -8.27 (18) |
| LI | -2.07 (29) | -14.83 (18) | -2.66 (29) | -14.26 (18) |
| TC-EASY | -2.34 (17) | +0.90 (14) | -2.95 (17) | +0.61 (14) |
| TC-MID | -4.51 (8) | -4.93 (31) | -5.24 (8) | -4.74 (31) |
| TC-HARD | -11.92 (7) | -6.53 (18) | -13.53 (7) | -5.39 (18) |
| TC | -4.98 (32) | -4.09 (63) | -5.84 (32) | -3.74 (63) |
| LR | -10.89 (10) | -16.23 (15) | -11.97 (10) | -15.67 (15) |

## Known confound (stated before the runs)

eval-only probe: the checkpoints were trained at the 1000x800 scale, so native evaluation mixes 'lower resolution' with 'train/test scale mismatch'. A symmetric drop on both splits is most likely the mismatch and says little about the paper. Decisive follow-up only if a pronounced asymmetry appears (delta_test - delta_val <= -2 pt for both models): train at native scale and evaluate at native scale (separate pre-registration).

## Provenance

| model | split | eval dir | checkpoint sha256 | git | started |
|---|---|---|---|---|---|
| normal_ir | val | `experiments\anti_uav\runs\normal_ir\phaseB_test_scale_640x512\val` | `ab9890d52684730e` | `4e14b979fe` (dirty) | 2026-10-04T23:24:11+0530 |
| normal_ir | test | `experiments\anti_uav\runs\normal_ir\phaseB_test_scale_640x512\test` | `ab9890d52684730e` | `d89d36bb43` (dirty) | 2026-10-05T00:45:51+0530 |
| dfsc_ir | val | `experiments\anti_uav\runs\dfsc_ir\phaseB_test_scale_640x512\val` | `1e8219c56d4b9bfe` | `e010d63173` (dirty) | 2026-10-05T02:39:25+0530 |
| dfsc_ir | test | `experiments\anti_uav\runs\dfsc_ir\phaseB_test_scale_640x512\test` | `1e8219c56d4b9bfe` | `e010d63173` (dirty) | 2026-10-05T04:04:20+0530 |
