# Single-target tracking on VAL — v3_n__ttt_cst_s2_e20 (data_unified_v2)

*Generated 2026-10-12T11:47:32+0530 by `scripts/track_eval.py` from cached detections (set `ec07f695df26`, window `ttt`). SOT protocol: GT box at the official start frame, then box or 'absent' per frame; mSA = paper Eq. (1) computed by Track A's code on the official annotations. Hyper-parameters cross-fitted over two halves of VAL (by recording group).*

| tracker | setting | group | n seq | mSA | located (IoU ≥ .5) | false alarms on invisible frames | recover median / mean (frames) |
|---|---|---|---:|---:|---:|---:|---|
| argmax | cross-fitted | cst_IR | 40 | 54.26 | 61.6 % | 24.5 % | 50 / 121.0 |
| argmax | tuned on all VAL | cst_IR | 40 | 54.36 | 61.0 % | 20.3 % | 50 / 122.7 |
| ste | cross-fitted | cst_IR | 40 | 55.15 | 63.3 % | 29.9 % | 53 / 128.0 |
| ste | tuned on all VAL | cst_IR | 40 | 55.85 | 63.7 % | 25.1 % | 50 / 127.0 |
| ste | defaults (untuned) | cst_IR | 40 | 52.20 | 62.7 % | 50.8 % | 76 / 145.1 |

Selected parameters (per half, then on all VAL):

- argmax: half0 ← {'thr': 0.4}; half1 ← {'thr': 0.5}; all ← {'thr': 0.5}
- ste: half0 ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': True, 'gate_chi2': 25.0}; half1 ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.7, 'max_coast': 20, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; all ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}
