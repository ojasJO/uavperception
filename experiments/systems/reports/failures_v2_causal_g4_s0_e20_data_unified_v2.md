# Failure analysis — v2_causal_g4_s0_e20 (data_unified_v2, VAL)

*Generated 2026-10-06T14:04:51+0530 by `scripts/failure_cases.py`. Rules in the script docstring; examples = the 0.1, 0.3, 0.5, 0.7, 0.9 quantiles of each category's severity (no manual selection); contact sheets in `failures_v2_causal_g4_s0_e20_data_unified_v2/`.*

| category | anti_uav_IR | anti_uav_RGB | cst_IR |
|---|---:|---:|---:|
| true_positive | 59610 | 57028 | 28743 |
| false_positive | 114 | 878 | 1062 |
| false_negative | 927 | 1195 | 10272 |
| micro_miss | 78 | 116 | 7370 |
| occlusion_failure | 51 | 81 | 2598 |
| reacquisition_success | 7 | 19 | 64 |
| reacquisition_failure | 0 | 0 | 69 |
| identity_failure | 10 | 4 | 17 |
| trajectory_overshoot | 7301 | 8995 | 7372 |
| trajectory_undershoot | 44677 | 45609 | 16416 |
| high_uncertainty | 50 | 7346 | 0 |
| low_confidence_correct | 32 | 41 | 344 |
| high_confidence_wrong | 22 | 330 | 211 |

Examples (sequence @ frame, score, IoU):

- **true_positive**: `antiuav_val_20190925_222534_1_7__RGB`@657 (s 0.79, IoU 0.91); `antiuav_val_20190926_130341_1_5__IR`@310 (s 0.83, IoU 0.86); `cst_val_building_49`@830 (s 0.60, IoU 0.82); `antiuav_val_20190925_194211_1_4__IR`@43 (s 0.76, IoU 0.77); `antiuav_val_20190926_195921_1_1__IR`@907 (s 0.74, IoU 0.68)
- **false_positive**: `cst_val_urban-areas_29`@1307 (s 0.24, IoU 0.00); `antiuav_val_20190926_200510_1_8__RGB`@479 (s 0.44, IoU 0.00); `cst_val_cn_sky_9`@960 (s 0.55, IoU 0.00); `antiuav_val_20190926_195921_1_5__RGB`@837 (s 0.63, IoU 0.00); `antiuav_val_20190925_222534_1_5__RGB`@796 (s 0.75, IoU 0.00)
- **false_negative**: `antiuav_val_20190926_183941_1_5__RGB`@954 (s 0.00, IoU 0.00); `cst_val_cn_mountains_24`@910 (s 0.00, IoU 0.00); `cst_val_urban-areas_32`@846 (s 0.28, IoU 0.38); `cst_val_urban-areas_28`@263 (s 0.56, IoU 0.23); `cst_val_building_52`@55 (s 0.00, IoU 0.00)
- **micro_miss**: `cst_val_urban-areas_30`@55 (s 0.00, IoU 0.00); `cst_val_building_41`@887 (s 0.29, IoU 0.00); `cst_val_building_48`@194 (s 0.00, IoU 0.00); `cst_val_building_45`@825 (s 0.00, IoU 0.00); `cst_val_building_52`@135 (s 0.00, IoU 0.00)
- **occlusion_failure**: `cst_val_urban-areas_33`@931 (s 0.61, IoU 0.00); `cst_val_urban-areas_30`@413 (s 0.00, IoU 0.00); `cst_val_urban-areas_33`@848 (s 0.65, IoU 0.00); `cst_val_urban-areas_30`@1068 (s 0.00, IoU 0.00); `cst_val_building_47`@826 (s 0.00, IoU 0.00)
- **reacquisition_success**: `antiuav_val_20190925_213001_1_1__RGB`@615 (s 0.83, IoU 0.00); `cst_val_cn_mountains_25`@1413 (s 0.77, IoU 0.00); `cst_val_water_19`@162 (s 0.67, IoU 0.00); `cst_val_water_15`@739 (s 0.00, IoU 0.00); `antiuav_val_20190925_152412_1_3__RGB`@81 (s 0.49, IoU 0.00)
- **reacquisition_failure**: `cst_val_urban-areas_27`@569 (s 0.00, IoU 0.00); `cst_val_urban-areas_29`@942 (s 0.00, IoU 0.00); `cst_val_urban-areas_30`@1485 (s 0.16, IoU 0.00); `cst_val_urban-areas_33`@830 (s 0.50, IoU 0.00); `cst_val_urban-areas_34`@1313 (s 0.50, IoU 0.00)
- **identity_failure**: `cst_val_cn_mountains_29`@882 (s 0.36, IoU 0.00); `cst_val_urban-areas_30`@838 (s 0.52, IoU 0.00); `cst_val_cn_mountains_21`@744 (s 0.58, IoU 0.00); `antiuav_val_20190925_143900_1_2__IR`@564 (s 0.65, IoU 0.00); `antiuav_val_20190925_222534_1_7__IR`@655 (s 0.78, IoU 0.00)
- **trajectory_overshoot**: `cst_val_water_19`@626 (s 0.72, IoU 0.84); `antiuav_val_20190925_140917_1_3__IR`@737 (s 0.80, IoU 0.83); `antiuav_val_20190926_143632_1_4__RGB`@159 (s 0.80, IoU 0.86); `antiuav_val_20190925_205804_1_6__RGB`@696 (s 0.81, IoU 0.92); `antiuav_val_20190925_130434_1_8__IR`@502 (s 0.79, IoU 0.74)
- **trajectory_undershoot**: `cst_val_building_53`@452 (s 0.74, IoU 0.84); `antiuav_val_20190926_144550_1_4__IR`@674 (s 0.76, IoU 0.88); `antiuav_val_20190926_141816_1_6__IR`@341 (s 0.78, IoU 0.84); `antiuav_val_20190926_144550_1_1__RGB`@148 (s 0.78, IoU 0.77); `antiuav_val_20190925_152412_1_5__IR`@288 (s 0.60, IoU 0.90)
- **high_uncertainty**: `antiuav_val_20190925_210802_1_8__RGB`@246 (s 0.81, IoU 0.89); `antiuav_val_20190926_141816_1_4__RGB`@766 (s 0.79, IoU 0.78); `antiuav_val_20190926_141816_1_2__IR`@200 (s 0.75, IoU 0.84); `antiuav_val_20190926_142435_1_9__RGB`@233 (s 0.81, IoU 0.90); `antiuav_val_20190926_141816_1_2__RGB`@531 (s 0.75, IoU 0.86)
- **low_confidence_correct**: `cst_val_building_42`@148 (s 0.29, IoU 0.65); `cst_val_cn_mountains_28`@915 (s 0.26, IoU 0.53); `cst_val_urban-areas_32`@537 (s 0.23, IoU 0.68); `cst_val_building_52`@967 (s 0.20, IoU 0.71); `cst_val_urban-areas_29`@1397 (s 0.17, IoU 0.70)
- **high_confidence_wrong**: `antiuav_val_20190925_141417_1_1__RGB`@152 (s 0.71, IoU 0.00); `antiuav_val_20190925_222534_1_5__RGB`@808 (s 0.73, IoU 0.00); `antiuav_val_20190925_222534_1_5__RGB`@802 (s 0.75, IoU 0.00); `antiuav_val_20190925_222534_1_5__RGB`@447 (s 0.77, IoU 0.00); `antiuav_val_20190925_222534_1_5__RGB`@429 (s 0.80, IoU 0.00)
