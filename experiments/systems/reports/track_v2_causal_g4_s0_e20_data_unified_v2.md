# Single-target tracking on VAL — v2_causal_g4_s0_e20 (data_unified_v2)

*Generated 2026-10-06T13:55:54+0530 by `scripts/track_eval.py` from cached detections (set `12711ffc0083`, window `causal`). SOT protocol: GT box at the official start frame, then box or 'absent' per frame; mSA = paper Eq. (1) computed by Track A's code on the official annotations. Hyper-parameters cross-fitted over two halves of VAL (by recording group).*

| tracker | setting | group | n seq | mSA | located (IoU ≥ .5) | false alarms on invisible frames | recover median / mean (frames) |
|---|---|---|---:|---:|---:|---:|---|
| argmax | cross-fitted | anti_uav_IR | 67 | 80.46 | 98.5 % | 13.5 % | 1 / 3.6 |
| argmax | cross-fitted | anti_uav_RGB | 67 | 81.38 | 98.2 % | 26.4 % | 0 / 1.3 |
| argmax | cross-fitted | cst_IR | 40 | 62.18 | 73.3 % | 20.2 % | 14 / 48.0 |
| argmax | tuned on all VAL | anti_uav_IR | 67 | 80.46 | 98.5 % | 13.5 % | 1 / 3.6 |
| argmax | tuned on all VAL | anti_uav_RGB | 67 | 81.38 | 98.2 % | 26.4 % | 0 / 1.3 |
| argmax | tuned on all VAL | cst_IR | 40 | 62.18 | 73.3 % | 20.2 % | 14 / 48.0 |
| ste | cross-fitted | anti_uav_IR | 67 | 80.53 | 98.5 % | 9.8 % | 1 / 3.7 |
| ste | cross-fitted | anti_uav_RGB | 67 | 81.20 | 97.9 % | 25.8 % | 1 / 2.9 |
| ste | cross-fitted | cst_IR | 40 | 63.02 | 73.7 % | 17.0 % | 18 / 71.0 |
| ste | tuned on all VAL | anti_uav_IR | 67 | 80.53 | 98.5 % | 9.9 % | 1 / 3.7 |
| ste | tuned on all VAL | anti_uav_RGB | 67 | 81.20 | 97.9 % | 25.6 % | 1 / 2.9 |
| ste | tuned on all VAL | cst_IR | 40 | 63.19 | 73.8 % | 16.0 % | 18 / 96.8 |
| ste | defaults (untuned) | anti_uav_IR | 67 | 80.09 | 98.3 % | 30.0 % | 1 / 3.6 |
| ste | defaults (untuned) | anti_uav_RGB | 67 | 80.13 | 97.1 % | 37.3 % | 1 / 3.0 |
| ste | defaults (untuned) | cst_IR | 40 | 60.71 | 72.5 % | 35.7 % | 19 / 73.3 |

GlobalTrack (Track A faithful reproduction, normal_ir seed 0) on the same 67 Anti-UAV IR VAL sequences: mSA 79.29.
- argmax (cross-fitted) − GlobalTrack: +1.17 mSA, sequence-bootstrap 95 % CI [-0.48, +4.15], wins 30 / losses 37
- ste (cross-fitted) − GlobalTrack: +1.24 mSA, sequence-bootstrap 95 % CI [-0.44, +4.29], wins 30 / losses 37

Selected parameters (per half, then on all VAL):

- argmax: half0 ← {'thr': 0.3}; half1 ← {'thr': 0.3}; all ← {'thr': 0.3}
- ste: half0 ← {'det_thr': 0.15, 'init_thr': 0.3, 'reacq_thr': 0.5, 'max_coast': 60, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; half1 ← {'det_thr': 0.15, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 20, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; all ← {'det_thr': 0.15, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 60, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}
