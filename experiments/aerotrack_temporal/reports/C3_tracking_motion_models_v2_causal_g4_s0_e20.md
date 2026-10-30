# C3 — tracking with domain / learned motion models (v2_causal_g4_s0_e20, data_unified_v2, VAL)

*Generated 2026-10-12T04:57:18+0530 by `scripts/tc_c3.py` under `preregistration/C3_tracking_motion_models.json`; SOT protocol, mSA by Track A's code; every tracker cross-fitted over the two VAL halves. Learned forecaster (C1 / C1c rule): {'antiuav_IR': 'C1:c1_tfm_abs_s0', 'antiuav_RGB': 'C1:c1_tfm_abs_s0'}.*

| tracker | group | n | mSA | located | false alarms | recover median (frames) |
|---|---|---:|---:|---:|---:|---:|
| argmax | anti_uav_IR | 67 | 80.46 | 98.5 % | 13.5 % | 1 |
| argmax | anti_uav_RGB | 67 | 81.38 | 98.2 % | 26.4 % | 0 |
| argmax | cst_IR | 40 | 62.18 | 73.3 % | 20.2 % | 14 |
| ste | anti_uav_IR | 67 | 80.53 | 98.5 % | 9.8 % | 1 |
| ste | anti_uav_RGB | 67 | 81.20 | 97.9 % | 25.8 % | 1 |
| ste | cst_IR | 40 | 63.02 | 73.7 % | 17.0 % | 18 |
| fste_classical | anti_uav_IR | 67 | 80.53 | 98.5 % | 13.8 % | 1 |
| fste_classical | anti_uav_RGB | 67 | 81.22 | 98.3 % | 30.6 % | 0 |
| fste_classical | cst_IR | 40 | 63.04 | 74.3 % | 20.1 % | 16 |
| fste_learned | anti_uav_IR | 67 | 80.43 | 98.5 % | 12.2 % | 1 |
| fste_learned | anti_uav_RGB | 67 | 81.42 | 98.1 % | 23.1 % | 0 |

MOT protocol (no init box) at the same settings:

| tracker | group | HOTA | AssA | IDF1 | MOTA | Frag | MT / ML | loss → recovered median / never | re-entry → recovered median / never |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| argmax | anti_uav_IR | 0.787 | 0.798 | 0.980 | 0.974 | 302 | 0.99 / 0.00 | 1 / 0 of 299 | 0 / 0 of 12 |
| argmax | anti_uav_RGB | 0.798 | 0.816 | 0.970 | 0.952 | 390 | 0.99 / 0.00 | 1 / 1 of 386 | 0 / 0 of 26 |
| argmax | cst_IR | 0.544 | 0.594 | 0.769 | 0.575 | 1178 | 0.55 / 0.00 | 2 / 9 of 1169 | 12 / 7 of 166 |
| ste | anti_uav_IR | 0.787 | 0.798 | 0.981 | 0.974 | 301 | 0.99 / 0.00 | 1 / 0 of 298 | 0 / 0 of 12 |
| ste | anti_uav_RGB | 0.797 | 0.815 | 0.969 | 0.951 | 385 | 0.99 / 0.00 | 1 / 1 of 380 | 0 / 0 of 26 |
| ste | cst_IR | 0.558 | 0.609 | 0.777 | 0.598 | 1047 | 0.57 / 0.03 | 2 / 9 of 1035 | 16 / 13 of 166 |
| fste_classical | anti_uav_IR | 0.787 | 0.798 | 0.980 | 0.973 | 294 | 0.99 / 0.00 | 1 / 0 of 291 | 0 / 0 of 12 |
| fste_classical | anti_uav_RGB | 0.797 | 0.815 | 0.969 | 0.949 | 363 | 0.99 / 0.00 | 1 / 1 of 358 | 0 / 0 of 26 |
| fste_classical | cst_IR | 0.556 | 0.608 | 0.776 | 0.586 | 1017 | 0.57 / 0.03 | 2 / 8 of 1001 | 15 / 13 of 166 |
| fste_learned | anti_uav_IR | 0.787 | 0.798 | 0.980 | 0.975 | 290 | 0.99 / 0.00 | 1 / 0 of 285 | 1 / 0 of 12 |
| fste_learned | anti_uav_RGB | 0.799 | 0.817 | 0.971 | 0.954 | 352 | 0.99 / 0.00 | 1 / 1 of 348 | 0 / 0 of 26 |

| hypothesis | group | Δ mSA [95 % CI] | wins / losses | verdict |
|---|---|---|---|---|
| H1_temporal_state | anti_uav_IR | +0.07 [-0.06, +0.25] | 9 / 14 | **NO_DIFFERENCE** |
| H1_temporal_state | anti_uav_RGB | -0.18 [-0.57, +0.04] | 9 / 13 | **NO_DIFFERENCE** |
| H1_temporal_state | cst_IR | +0.84 [+0.24, +1.61] | 25 / 13 | **SUPPORTED** |
| H2_domain_motion_model | anti_uav_IR | +0.00 [-0.18, +0.12] | 12 / 3 | **NO_DIFFERENCE** |
| H2_domain_motion_model | anti_uav_RGB | +0.03 [-0.35, +0.54] | 9 / 8 | **NO_DIFFERENCE** |
| H2_domain_motion_model | cst_IR | +0.02 [-0.27, +0.29] | 17 / 11 | **NO_DIFFERENCE** |
| H3_learned_motion_model | anti_uav_IR | -0.10 [-0.34, +0.06] | 8 / 13 | **NO_DIFFERENCE** |
| H3_learned_motion_model | anti_uav_RGB | +0.19 [-0.21, +0.65] | 10 / 11 | **NO_DIFFERENCE** |
| H3_learned_motion_model | cst_IR | — | — | NOT RUN (no learned forecaster adopted by C1 / C1c) |
