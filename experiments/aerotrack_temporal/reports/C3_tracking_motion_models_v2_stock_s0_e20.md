# C3 — tracking with domain / learned motion models (v2_stock_s0_e20, data_unified_v2, VAL)

*Generated 2026-10-12T09:44:55+0530 by `scripts/tc_c3.py` under `preregistration/C3_tracking_motion_models.json`; SOT protocol, mSA by Track A's code; every tracker cross-fitted over the two VAL halves. Learned forecaster (C1 / C1c rule): {'antiuav_IR': 'C1:c1_tfm_abs_s0', 'antiuav_RGB': 'C1:c1_tfm_abs_s0'}.*

| tracker | group | n | mSA | located | false alarms | recover median (frames) |
|---|---|---:|---:|---:|---:|---:|
| argmax | anti_uav_IR | 67 | 80.29 | 98.4 % | 13.0 % | 1 |
| argmax | anti_uav_RGB | 67 | 81.42 | 97.9 % | 22.3 % | 0 |
| argmax | cst_IR | 40 | 52.59 | 58.6 % | 21.3 % | 56 |
| ste | anti_uav_IR | 67 | 80.43 | 98.5 % | 12.0 % | 1 |
| ste | anti_uav_RGB | 67 | 81.38 | 98.0 % | 23.6 % | 1 |
| ste | cst_IR | 40 | 52.76 | 59.3 % | 23.6 % | 75 |
| fste_classical | anti_uav_IR | 67 | 80.36 | 98.3 % | 9.0 % | 1 |
| fste_classical | anti_uav_RGB | 67 | 81.33 | 98.0 % | 25.7 % | 0 |
| fste_classical | cst_IR | 40 | 52.87 | 59.6 % | 24.0 % | 74 |
| fste_learned | anti_uav_IR | 67 | 80.27 | 98.4 % | 15.3 % | 1 |
| fste_learned | anti_uav_RGB | 67 | 81.48 | 98.0 % | 22.1 % | 1 |

MOT protocol (no init box) at the same settings:

| tracker | group | HOTA | AssA | IDF1 | MOTA | Frag | MT / ML | loss → recovered median / never | re-entry → recovered median / never |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| argmax | anti_uav_IR | 0.785 | 0.797 | 0.979 | 0.973 | 326 | 0.97 / 0.00 | 1 / 0 of 323 | 2 / 0 of 12 |
| argmax | anti_uav_RGB | 0.798 | 0.816 | 0.970 | 0.950 | 423 | 0.99 / 0.00 | 1 / 2 of 419 | 0 / 0 of 26 |
| argmax | cst_IR | 0.467 | 0.560 | 0.657 | 0.413 | 833 | 0.33 / 0.17 | 3 / 15 of 828 | 44 / 24 of 166 |
| ste | anti_uav_IR | 0.786 | 0.797 | 0.980 | 0.974 | 330 | 0.99 / 0.00 | 1 / 0 of 325 | 1 / 0 of 12 |
| ste | anti_uav_RGB | 0.797 | 0.816 | 0.970 | 0.949 | 407 | 0.99 / 0.00 | 2 / 2 of 401 | 0 / 0 of 26 |
| ste | cst_IR | 0.471 | 0.576 | 0.658 | 0.393 | 641 | 0.35 / 0.17 | 3 / 12 of 635 | 55 / 24 of 166 |
| fste_classical | anti_uav_IR | 0.785 | 0.796 | 0.980 | 0.972 | 330 | 0.99 / 0.00 | 2 / 0 of 325 | 1 / 0 of 12 |
| fste_classical | anti_uav_RGB | 0.797 | 0.815 | 0.969 | 0.948 | 408 | 0.99 / 0.00 | 1 / 2 of 404 | 0 / 0 of 26 |
| fste_classical | cst_IR | 0.475 | 0.577 | 0.663 | 0.407 | 670 | 0.35 / 0.17 | 3 / 12 of 662 | 56 / 25 of 166 |
| fste_learned | anti_uav_IR | 0.785 | 0.797 | 0.979 | 0.974 | 298 | 0.97 / 0.00 | 1 / 0 of 293 | 2 / 0 of 12 |
| fste_learned | anti_uav_RGB | 0.799 | 0.817 | 0.971 | 0.951 | 381 | 0.99 / 0.00 | 1 / 2 of 376 | 1 / 0 of 26 |

| hypothesis | group | Δ mSA [95 % CI] | wins / losses | verdict |
|---|---|---|---|---|
| H1_temporal_state | anti_uav_IR | +0.14 [+0.01, +0.35] | 15 / 9 | **NO_DIFFERENCE** |
| H1_temporal_state | anti_uav_RGB | -0.05 [-0.12, +0.02] | 11 / 13 | **NO_DIFFERENCE** |
| H1_temporal_state | cst_IR | +0.17 [-0.73, +1.10] | 23 / 14 | **NO_DIFFERENCE** |
| H2_domain_motion_model | anti_uav_IR | -0.08 [-0.40, +0.19] | 11 / 7 | **NO_DIFFERENCE** |
| H2_domain_motion_model | anti_uav_RGB | -0.04 [-0.20, +0.15] | 8 / 12 | **NO_DIFFERENCE** |
| H2_domain_motion_model | cst_IR | +0.11 [-0.27, +0.58] | 16 / 13 | **NO_DIFFERENCE** |
| H3_learned_motion_model | anti_uav_IR | -0.08 [-0.52, +0.35] | 8 / 16 | **NO_DIFFERENCE** |
| H3_learned_motion_model | anti_uav_RGB | +0.15 [-0.02, +0.34] | 12 / 11 | **NO_DIFFERENCE** |
| H3_learned_motion_model | cst_IR | — | — | NOT RUN (no learned forecaster adopted by C1 / C1c) |
