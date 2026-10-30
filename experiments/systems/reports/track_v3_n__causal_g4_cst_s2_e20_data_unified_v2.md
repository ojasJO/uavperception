# Single-target tracking on VAL — v3_n__causal_g4_cst_s2_e20 (data_unified_v2)

*Generated 2026-10-12T10:12:02+0530 by `scripts/track_eval.py` from cached detections (set `ec07f695df26`, window `causal`). SOT protocol: GT box at the official start frame, then box or 'absent' per frame; mSA = paper Eq. (1) computed by Track A's code on the official annotations. Hyper-parameters cross-fitted over two halves of VAL (by recording group).*

| tracker | setting | group | n seq | mSA | located (IoU ≥ .5) | false alarms on invisible frames | recover median / mean (frames) |
|---|---|---|---:|---:|---:|---:|---|
| argmax | cross-fitted | cst_IR | 40 | 63.52 | 75.0 % | 28.6 % | 10 / 46.2 |
| argmax | tuned on all VAL | cst_IR | 40 | 64.31 | 74.8 % | 20.2 % | 14 / 52.2 |
| ste | cross-fitted | cst_IR | 40 | 64.46 | 75.0 % | 20.8 % | 16 / 78.2 |
| ste | tuned on all VAL | cst_IR | 40 | 65.20 | 75.9 % | 20.4 % | 18 / 80.1 |
| ste | defaults (untuned) | cst_IR | 40 | 63.05 | 75.0 % | 40.2 % | 17 / 74.5 |

Selected parameters (per half, then on all VAL):

- argmax: half0 ← {'thr': 0.15}; half1 ← {'thr': 0.4}; all ← {'thr': 0.3}
- ste: half0 ← {'det_thr': 0.05, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 60, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; half1 ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; all ← {'det_thr': 0.05, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 20, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}
