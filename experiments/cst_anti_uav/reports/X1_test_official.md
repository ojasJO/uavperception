# X1 on CST TEST — official toolkit scoring and paired comparison (amendment X1-A1)

Candidate `v3_n__causal_g4_cst_s0_e20` (weights `e52a5645b852…`), single registered pass `experiments/aerotrack_v3/reports/x1_final_test.json`.

| tracker | official mSA | 95 % CI | success AUC | P@20px | tiny SA | rule X1-a (vs 35.92) |
|---|---:|---|---:|---:|---:|---|
| AeroTrack-X1 STE | **60.08** | [56.13, 63.97] | 58.53 | 77.42 | 39.88 | exceeds |
| AeroTrack-X1 argmax | **59.70** | [55.80, 63.48] | 58.16 | 77.08 | 40.43 | exceeds |
| GlobalTrack (released, CST-trained) | 35.92 | — | | | 15.83 | |
| SiamDT (released, CST-trained) | 35.84 | — | | | 14.98 | |
| PrDiMP (released, CST-trained) | 29.76 | — | | | 28.21 | |

Official toolkit vs the pass's Track A values: max per-sequence |Δ| = 0.0e+00 (STE), 0.0e+00 (argmax).

**X1-a2 (paired, 60 sequences):** STE − GlobalTrack = +24.16 SA, 95 % CI [+20.78, +27.63], wins 59 / losses 1, Wilcoxon p = 1.8e-11 → **exceeds GlobalTrack (paired)**.
Descriptive: STE − SiamDT = +24.24 [+20.72, +27.90], wins 59 / losses 1.

Tracker cost (from cached detections, HPC CPU): {'ste': 0.12636477974797353, 'argmax': 0.01312163223908936} ms / frame.

**Decomposition (descriptive).** Visible frames: mean IoU STE 56.67 vs GlobalTrack 41.33; absent frames: correct 'absent' STE 71.03 % vs GlobalTrack 0.00 % (no released tracker ever reports absence). Sequence-level visible-frame IoU, STE − GlobalTrack (paired): +14.88 [+11.93, +17.78], wins 56 / losses 4.
