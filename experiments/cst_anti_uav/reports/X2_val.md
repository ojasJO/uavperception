# X2 — CST VAL design (seeds × causal_g4 / single frame, CST-only training)

*Generated 2026-10-12T11:53:32+0530 by `scripts/run_x2.py --val-design`; descriptive (VAL shares source videos with TRAIN).*

| run | arm | seed | CST AP50 | STE mSA cross-fitted | STE mSA tuned on all VAL | argmax cross-fitted | frozen STE settings |
|---|---|---:|---:|---:|---:|---:|---|
| v3_n__causal_g4_cst_s0_e20 | causal_g4 | 0 | 0.694 | 63.61 | 64.03 | 62.60 | {'det_thr': 0.15, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0} |
| v3_n__causal_g4_cst_s1_e20 | causal_g4 | 1 | 0.700 | 65.08 | 65.76 | 64.38 | {'det_thr': 0.15, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 20, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0} |
| v3_n__causal_g4_cst_s2_e20 | causal_g4 | 2 | 0.703 | 64.46 | 65.20 | 63.52 | {'det_thr': 0.05, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 20, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0} |
| v3_n__ttt_cst_s0_e20 | ttt | 0 | 0.540 | 55.40 | 55.94 | 54.17 | {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0} |
| v3_n__ttt_cst_s1_e20 | ttt | 1 | 0.575 | 55.47 | 56.66 | 54.84 | {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 0, 'output': 'det', 'coast_output': True, 'gate_chi2': 25.0} |
| v3_n__ttt_cst_s2_e20 | ttt | 2 | 0.540 | 55.15 | 55.85 | 54.26 | {'det_thr': 0.3, 'init_thr': 0.5, 'reacq_thr': 0.5, 'max_coast': 5, 'output': 'det', 'coast_output': False, 'gate_chi2': 25.0} |

**V1** causal_g4 − single frame (STE cross-fitted, seed-mean per sequence, 40 sequences): +9.04 SA, 95 % CI [+6.38, +12.03], wins 36 / losses 4; per seed: s0 +8.21 [+5.08, +11.79] (AP50 +0.154), s1 +9.62 [+6.97, +12.44] (AP50 +0.126), s2 +9.30 [+6.26, +12.55] (AP50 +0.163)
