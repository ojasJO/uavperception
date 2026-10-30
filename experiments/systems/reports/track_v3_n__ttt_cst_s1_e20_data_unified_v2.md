# Single-target tracking on VAL — v3_n__ttt_cst_s1_e20 (data_unified_v2)

*Generated 2026-10-12T11:15:46+0530 by `scripts/track_eval.py` from cached detections (set `ec07f695df26`, window `ttt`). SOT protocol: GT box at the official start frame, then box or 'absent' per frame; mSA = paper Eq. (1) computed by Track A's code on the official annotations. Hyper-parameters cross-fitted over two halves of VAL (by recording group).*

| tracker | setting | group | n seq | mSA | located (IoU ≥ .5) | false alarms on invisible frames | recover median / mean (frames) |
|---|---|---|---:|---:|---:|---:|---|
| argmax | cross-fitted | cst_IR | 40 | 54.84 | 63.4 % | 29.6 % | 38 / 116.2 |
| argmax | tuned on all VAL | cst_IR | 40 | 55.73 | 62.4 % | 16.5 % | 50 / 134.2 |
| ste | cross-fitted | cst_IR | 40 | 55.47 | 63.9 % | 29.7 % | 50 / 136.1 |
| ste | tuned on all VAL | cst_IR | 40 | 56.66 | 64.6 % | 22.3 % | 50 / 134.8 |
| ste | defaults (untuned) | cst_IR | 40 | 54.32 | 64.8 % | 49.5 % | 50 / 130.9 |

Selected parameters (per half, then on all VAL):

- argmax: half0 ← {'thr': 0.3}; half1 ← {'thr': 0.5}; all ← {'thr': 0.5}
- ste: half0 ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': True, 'gate_chi2': 25.0}; half1 ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.7, 'max_coast': 60, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; all ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 0, 'output': 'det', 'coast_output': True, 'gate_chi2': 25.0}
