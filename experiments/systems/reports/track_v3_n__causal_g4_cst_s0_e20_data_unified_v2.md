# Single-target tracking on VAL — v3_n__causal_g4_cst_s0_e20 (data_unified_v2)

*Generated 2026-10-10T07:30:36+0530 by `scripts/track_eval.py` from cached detections (set `ec07f695df26`, window `causal`). SOT protocol: GT box at the official start frame, then box or 'absent' per frame; mSA = paper Eq. (1) computed by Track A's code on the official annotations. Hyper-parameters cross-fitted over two halves of VAL (by recording group).*

| tracker | setting | group | n seq | mSA | located (IoU ≥ .5) | false alarms on invisible frames | recover median / mean (frames) |
|---|---|---|---:|---:|---:|---:|---|
| argmax | cross-fitted | cst_IR | 40 | 62.60 | 75.2 % | 36.0 % | 14 / 56.5 |
| argmax | tuned on all VAL | cst_IR | 40 | 63.42 | 74.9 % | 27.2 % | 14 / 59.8 |
| ste | cross-fitted | cst_IR | 40 | 63.61 | 75.3 % | 27.6 % | 20 / 76.4 |
| ste | tuned on all VAL | cst_IR | 40 | 64.03 | 75.0 % | 22.5 % | 22 / 77.8 |
| ste | defaults (untuned) | cst_IR | 40 | 61.70 | 74.8 % | 46.0 % | 18 / 75.8 |

Selected parameters (per half, then on all VAL):

- argmax: half0 ← {'thr': 0.15}; half1 ← {'thr': 0.4}; all ← {'thr': 0.3}
- ste: half0 ← {'det_thr': 0.05, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 60, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; half1 ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 20, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; all ← {'det_thr': 0.15, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}
