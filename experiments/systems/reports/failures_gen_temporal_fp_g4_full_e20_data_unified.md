# Failure analysis — gen_temporal_fp_g4_full_e20 (data_unified, VAL)

*Generated 2026-10-05T07:19:41+0530 by `scripts/failure_cases.py`. Rules in the script docstring; examples = the 0.1, 0.3, 0.5, 0.7, 0.9 quantiles of each category's severity (no manual selection); contact sheets in `failures_gen_temporal_fp_g4_full_e20_data_unified/`.*

| category | anti_uav_IR | anti_uav_RGB | cst_IR |
|---|---:|---:|---:|
| true_positive | 59737 | 56687 | 10572 |
| false_positive | 127 | 874 | 108 |
| false_negative | 800 | 1536 | 1430 |
| micro_miss | 78 | 116 | 758 |
| occlusion_failure | 55 | 81 | 119 |
| reacquisition_success | 7 | 20 | 17 |
| reacquisition_failure | 0 | 0 | 1 |
| identity_failure | 9 | 5 | 0 |
| trajectory_overshoot | 7732 | 8988 | 2249 |
| trajectory_undershoot | 44269 | 45568 | 5331 |
| high_uncertainty | 37 | 6309 | 0 |
| low_confidence_correct | 59 | 85 | 255 |
| high_confidence_wrong | 24 | 250 | 4 |

Examples (sequence @ frame, score, IoU):

- **true_positive**: `antiuav_val_20190926_103046_1_2__RGB`@752 (s 0.82, IoU 0.91); `antiuav_val_20190925_222534_1_7__RGB`@335 (s 0.79, IoU 0.87); `antiuav_val_20190926_144550_1_7__IR`@831 (s 0.80, IoU 0.83); `antiuav_val_20190926_144550_1_4__RGB`@56 (s 0.79, IoU 0.78); `antiuav_val_20190926_183400_1_1__RGB`@630 (s 0.71, IoU 0.69)
- **false_positive**: `antiuav_val_20190926_195921_1_5__RGB`@537 (s 0.15, IoU 0.00); `antiuav_val_20190925_152412_1_3__RGB`@74 (s 0.45, IoU 0.00); `antiuav_val_20190925_222534_1_5__RGB`@309 (s 0.58, IoU 0.00); `antiuav_val_20190925_222534_1_5__RGB`@477 (s 0.66, IoU 0.00); `antiuav_val_20190925_222534_1_5__RGB`@833 (s 0.75, IoU 0.00)
- **false_negative**: `antiuav_val_20190925_143900_1_2__RGB`@461 (s 0.00, IoU 0.00); `antiuav_val_20190926_183941_1_5__RGB`@845 (s 0.00, IoU 0.00); `antiuav_val_20190925_133630_1_9__IR`@697 (s 0.00, IoU 0.00); `cst_train_cn_mountains_9`@998 (s 0.41, IoU 0.19); `cst_train_building_1`@906 (s 0.66, IoU 0.50)
- **micro_miss**: `cst_train_cn_mountains_18`@385 (s 0.77, IoU 0.48); `antiuav_val_20190926_195921_1_6__RGB`@105 (s 0.00, IoU 0.00); `cst_train_building_20`@710 (s 0.00, IoU 0.00); `cst_train_building_4`@421 (s 0.00, IoU 0.00); `antiuav_val_20190925_213001_1_1__IR`@625 (s 0.00, IoU 0.00)
- **occlusion_failure**: `cst_train_cn_mountains_8`@665 (s 0.00, IoU 0.00); `cst_train_building_38`@841 (s 0.00, IoU 0.00); `cst_train_cn_mountains_8`@671 (s 0.00, IoU 0.00); `cst_train_building_34`@454 (s 0.00, IoU 0.00); `cst_train_building_34`@418 (s 0.00, IoU 0.00)
- **reacquisition_success**: `antiuav_val_20190925_152412_1_3__IR`@59 (s 0.53, IoU 0.00); `antiuav_val_20190926_195921_1_5__RGB`@509 (s 0.31, IoU 0.00); `cst_train_cn_mountains_17`@191 (s 0.71, IoU 0.00); `cst_train_building_34`@416 (s 0.00, IoU 0.00); `antiuav_val_20190925_152412_1_3__RGB`@81 (s 0.66, IoU 0.00)
- **reacquisition_failure**: `cst_train_building_7`@93 (s 0.58, IoU 0.00)
- **identity_failure**: `antiuav_val_20190925_213001_1_1__IR`@613 (s 0.48, IoU 0.00); `antiuav_val_20190926_195921_1_6__RGB`@82 (s 0.53, IoU 0.00); `antiuav_val_20190926_141816_1_6__IR`@938 (s 0.64, IoU 0.00); `antiuav_val_20190926_183941_1_3__IR`@723 (s 0.68, IoU 0.00); `antiuav_val_20190925_222534_1_7__IR`@655 (s 0.79, IoU 0.00)
- **trajectory_overshoot**: `antiuav_val_20190926_144550_1_7__IR`@391 (s 0.80, IoU 0.84); `cst_train_building_11`@44 (s 0.84, IoU 0.85); `antiuav_val_20190925_205804_1_5__IR`@54 (s 0.83, IoU 0.90); `antiuav_val_20190925_183946_1_3__IR`@460 (s 0.78, IoU 0.86); `antiuav_val_20190926_195921_1_1__RGB`@270 (s 0.80, IoU 0.89)
- **trajectory_undershoot**: `antiuav_val_20190926_133516_1_2__IR`@775 (s 0.85, IoU 0.80); `antiuav_val_20190925_210802_1_3__RGB`@814 (s 0.83, IoU 0.92); `antiuav_val_20190926_141816_1_2__IR`@900 (s 0.78, IoU 0.75); `antiuav_val_20190926_183400_1_4__RGB`@627 (s 0.68, IoU 0.78); `antiuav_val_20190925_213001_1_3__IR`@733 (s 0.80, IoU 0.80)
- **high_uncertainty**: `antiuav_val_20190925_143900_1_7__RGB`@650 (s 0.81, IoU 0.74); `antiuav_val_20190925_130434_1_1__RGB`@887 (s 0.75, IoU 0.82); `antiuav_val_20190926_142435_1_9__RGB`@111 (s 0.79, IoU 0.90); `antiuav_val_20190925_130434_1_5__RGB`@654 (s 0.84, IoU 0.91); `antiuav_val_20190926_141816_1_2__RGB`@253 (s 0.75, IoU 0.87)
- **low_confidence_correct**: `cst_train_building_38`@164 (s 0.28, IoU 0.62); `cst_train_cn_mountains_9`@669 (s 0.22, IoU 0.69); `cst_train_building_1`@292 (s 0.17, IoU 0.67); `antiuav_val_20190925_133630_1_9__IR`@692 (s 0.12, IoU 0.84); `antiuav_val_20190925_140917_1_3__RGB`@922 (s 0.07, IoU 0.55)
- **high_confidence_wrong**: `cst_train_cn_mountains_15`@1121 (s 0.71, IoU 0.00); `antiuav_val_20190925_101846_1_4__RGB`@498 (s 0.74, IoU 0.08); `antiuav_val_20190925_222534_1_5__RGB`@435 (s 0.75, IoU 0.00); `antiuav_val_20190925_143900_1_2__IR`@393 (s 0.78, IoU 0.00); `antiuav_val_20190925_222534_1_5__RGB`@429 (s 0.80, IoU 0.00)
