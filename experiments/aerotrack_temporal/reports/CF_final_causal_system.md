# CF — the complete causal system (single registered TEST pass)

*Generated 2026-10-12T14:35:11+0530 by `scripts/tc_cf.py --stage test` (TEST) under `experiments/aerotrack_temporal/preregistration/CF_final_causal_system.json` (+ CF-A2); frozen `experiments/aerotrack_temporal/frozen_cf.json` (sha256 6c5c83b8378e…); detector `v2_causal_g4_s0_e20`.*

| group | tracker | sequences | mSA | located | false alarms | recover median | argmax (frozen thr) mSA | CF-c Δ [95 % CI] |
|---|---|---:|---:|---:|---:|---:|---:|---|
| anti_uav_IR | argmax | 91 | 68.89 | 87.0 % | 21.6 % | 0 | 68.89 | +0.00 [+0.00, +0.00] |
| anti_uav_RGB | argmax | 91 | 74.10 | 91.5 % | 11.3 % | 3 | 74.10 | +0.00 [+0.00, +0.00] |
| cst_IR | ste | 60 | 56.55 | 67.1 % | 33.1 % | 20 | 56.28 | +0.27 [-0.27, +0.79] |

**CF-a** (Anti-UAV IR, 91 sequences) CF − Track A GlobalTrack normal_ir (70.58): -1.69 SA [-3.25, +0.06], wins 28 / losses 63 → **no detectable difference** (caveat: training data differ).

MOT protocol (no init box):

| group | HOTA | AssA | IDF1 | MOTA | Frag |
|---|---:|---:|---:|---:|---:|
| anti_uav_IR | 0.675 | 0.711 | 0.896 | 0.820 | 1111 |
| anti_uav_RGB | 0.717 | 0.740 | 0.916 | 0.866 | 1475 |
| cst_IR | 0.479 | 0.537 | 0.694 | 0.438 | 1643 |

Forecasting on the CF tracker's own TEST histories (CF-b):

| domain | anchors | deployed | ADE px (deployed / Kalman / CP) | FDE@24 px | NLL (deployed / Kalman) | vs Kalman | vs CP |
|---|---:|---|---|---|---|---|---|
| antiuav_IR | 75,530 | c1_tfm_abs_s0 | 38.9 / 43.6 / 43.6 | 54.1 / 58.8 / 58.8 | 10.95 / 11.08 | SUPPORTED | SUPPORTED |
| antiuav_RGB | 75,081 | c1_tfm_abs_s0 | 94.7 / 106.4 / 106.4 | 131.4 / 141.4 / 141.3 | 12.45 / 12.99 | SUPPORTED | SUPPORTED |
| cst | 47,679 | dcv | 49.9 / 51.2 / 47.5 | 69.4 / 73.8 / 62.9 | 167.07 / 97.39 | NOT_SUPPORTED | NOT_SUPPORTED |

CF-d: not applicable: the v2 candidate v2_causal_g4_s0_e20 is causal (CF-A2); its v2 final TEST pass, descriptively.
