# Single-target tracking on VAL — gen_temporal_fp_g4_full_e20 (data_unified)

*Generated 2026-10-05T07:14:57+0530 by `scripts/track_eval.py` from cached detections (set `3b15c413bed5`, window `symmetric`). SOT protocol: GT box at the official start frame, then box or 'absent' per frame; mSA = paper Eq. (1) computed by Track A's code on the official annotations. Hyper-parameters cross-fitted over two halves of VAL (by recording group).*

| tracker | setting | group | n seq | mSA | located (IoU ≥ .5) | false alarms on invisible frames | recover median / mean (frames) |
|---|---|---|---:|---:|---:|---:|---|
| argmax | cross-fitted | anti_uav_IR | 67 | 80.56 | 98.6 % | 12.5 % | 1 / 3.6 |
| argmax | cross-fitted | anti_uav_RGB | 67 | 81.40 | 97.9 % | 23.3 % | 0 / 1.5 |
| argmax | cross-fitted | cst_IR | 15 | 72.11 | 85.5 % | 3.0 % | 3 / 18.8 |
| argmax | tuned on all VAL | anti_uav_IR | 67 | 80.64 | 98.7 % | 12.5 % | 1 / 3.6 |
| argmax | tuned on all VAL | anti_uav_RGB | 67 | 81.40 | 98.1 % | 25.8 % | 0 / 1.2 |
| argmax | tuned on all VAL | cst_IR | 15 | 72.28 | 87.0 % | 7.8 % | 2 / 16.8 |
| ste | cross-fitted | anti_uav_IR | 67 | 80.67 | 98.7 % | 11.0 % | 1 / 4.3 |
| ste | cross-fitted | anti_uav_RGB | 67 | 81.07 | 97.4 % | 25.7 % | 0 / 2.0 |
| ste | cross-fitted | cst_IR | 15 | 73.96 | 88.1 % | 3.6 % | 1 / 15.6 |
| ste | tuned on all VAL | anti_uav_IR | 67 | 80.68 | 98.7 % | 10.8 % | 1 / 4.3 |
| ste | tuned on all VAL | anti_uav_RGB | 67 | 81.10 | 97.4 % | 25.1 % | 0 / 2.2 |
| ste | tuned on all VAL | cst_IR | 15 | 74.54 | 88.8 % | 3.2 % | 1 / 15.6 |
| ste | defaults (untuned) | anti_uav_IR | 67 | 80.30 | 98.6 % | 29.4 % | 1 / 3.7 |
| ste | defaults (untuned) | anti_uav_RGB | 67 | 80.19 | 97.0 % | 37.5 % | 0 / 7.0 |
| ste | defaults (untuned) | cst_IR | 15 | 72.57 | 88.9 % | 15.9 % | 2 / 15.8 |

GlobalTrack (Track A faithful reproduction, normal_ir seed 0) on the same 67 Anti-UAV IR VAL sequences: mSA 79.29.
- argmax (cross-fitted) − GlobalTrack: +1.27 mSA, sequence-bootstrap 95 % CI [-0.34, +4.20], wins 30 / losses 37
- ste (cross-fitted) − GlobalTrack: +1.39 mSA, sequence-bootstrap 95 % CI [-0.24, +4.35], wins 32 / losses 35

Selected parameters (per half, then on all VAL):

- argmax: half0 ← {'thr': 0.4}; half1 ← {'thr': 0.3}; all ← {'thr': 0.3}
- ste: half0 ← {'det_thr': 0.05, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': False, 'gate_chi2': 9.2103}; half1 ← {'det_thr': 0.05, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 60, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0}; all ← {'det_thr': 0.05, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 20, 'output': 'det', 'coast_output': False, 'gate_chi2': 9.2103}
