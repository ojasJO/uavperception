# Single-target tracking on VAL — v3_n__ttt_cst_s0_e20 (data_unified_v2)

*Generated 2026-10-12T10:44:10+0530 by `scripts/track_eval.py` from cached detections (set `ec07f695df26`, window `ttt`). SOT protocol: GT box at the official start frame, then box or 'absent' per frame; mSA = paper Eq. (1) computed by Track A's code on the official annotations. Hyper-parameters cross-fitted over two halves of VAL (by recording group).*

| tracker | setting | group | n seq | mSA | located (IoU ≥ .5) | false alarms on invisible frames | recover median / mean (frames) |
|---|---|---|---:|---:|---:|---:|---|
| argmax | cross-fitted | cst_IR | 40 | 54.17 | 62.7 % | 30.2 % | 40 / 105.2 |
| argmax | tuned on all VAL | cst_IR | 40 | 54.88 | 61.8 % | 18.8 % | 41 / 113.8 |
| ste | cross-fitted | cst_IR | 40 | 55.40 | 63.6 % | 27.8 % | 52 / 119.2 |
| ste | tuned on all VAL | cst_IR | 40 | 55.94 | 63.9 % | 22.8 % | 54 / 119.3 |
| ste | defaults (untuned) | cst_IR | 40 | 52.69 | 62.9 % | 51.3 % | 64 / 127.8 |

Selected parameters (per half, then on all VAL):

- argmax: half0 ← {'thr': 0.3}; half1 ← {'thr': 0.5}; all ← {'thr': 0.5}
- ste: half0 ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': True, 'gate_chi2': 25.0}; half1 ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.7, 'max_coast': 60, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; all ← {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}
