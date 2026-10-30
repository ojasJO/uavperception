# Single-target tracking on VAL — v3_n__causal_g4_cst_s1_e20 (data_unified_v2)

*Generated 2026-10-12T09:41:09+0530 by `scripts/track_eval.py` from cached detections (set `ec07f695df26`, window `causal`). SOT protocol: GT box at the official start frame, then box or 'absent' per frame; mSA = paper Eq. (1) computed by Track A's code on the official annotations. Hyper-parameters cross-fitted over two halves of VAL (by recording group).*

| tracker | setting | group | n seq | mSA | located (IoU ≥ .5) | false alarms on invisible frames | recover median / mean (frames) |
|---|---|---|---:|---:|---:|---:|---|
| argmax | cross-fitted | cst_IR | 40 | 64.38 | 75.5 % | 23.3 % | 12 / 50.9 |
| argmax | tuned on all VAL | cst_IR | 40 | 64.57 | 75.0 % | 18.6 % | 16 / 54.6 |
| ste | cross-fitted | cst_IR | 40 | 65.08 | 76.5 % | 23.3 % | 18 / 61.2 |
| ste | tuned on all VAL | cst_IR | 40 | 65.76 | 76.9 % | 20.4 % | 18 / 60.7 |
| ste | defaults (untuned) | cst_IR | 40 | 63.18 | 76.3 % | 44.3 % | 18 / 63.5 |

Selected parameters (per half, then on all VAL):

- argmax: half0 ← {'thr': 0.3}; half1 ← {'thr': 0.4}; all ← {'thr': 0.4}
- ste: half0 ← {'det_thr': 0.05, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 20, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; half1 ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; all ← {'det_thr': 0.15, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 20, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}
