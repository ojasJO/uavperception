# C1b — forecasting on tracker histories (v2_stock_s0_e20, data_unified_v2, VAL)

*Generated 2026-10-12T09:53:54+0530 by `scripts/tc_c1b.py`; histories = ste_cv outputs with C3's cross-fitted settings; targets = ground truth. Learned family: tfm.*

## antiuav_IR (67 sequences, 59,932 anchors; classical reference `oudcv`)

H1b **SUPPORTED** · H2b **SUPPORTED** · H4 (aug) **HARMFUL**

| forecaster | ADE px | FDE@24 px | NLL | 95 % coverage | calib. error |
|---|---:|---:|---:|---:|---:|
| kalman | 38.4 | 52.4 | 10.49 | 0.928 | 0.256 |
| oudcv | 37.5 | 52.1 | — | — | — |
| c1_tfm_abs_s0 | 34.0 | 48.6 | 9.15 | 0.926 | 0.201 |
| c1_tfm_abs_s1 | 34.6 | 49.7 | 9.27 | 0.940 | 0.240 |
| c1_tfm_abs_s2 | 34.1 | 48.7 | 9.24 | 0.917 | 0.166 |
| c1_tfm_aug_s0 | 34.7 | 48.9 | 9.44 | 0.929 | 0.183 |
| c1_tfm_aug_s1 | 37.3 | 51.5 | 10.29 | 0.892 | 0.101 |
| c1_tfm_aug_s2 | 34.9 | 49.1 | 9.78 | 0.919 | 0.148 |

## antiuav_RGB (67 sequences, 57,719 anchors; classical reference `oudcv`)

H1b **SUPPORTED** · H2b **SUPPORTED** · H4 (aug) **HARMFUL**

| forecaster | ADE px | FDE@24 px | NLL | 95 % coverage | calib. error |
|---|---:|---:|---:|---:|---:|
| kalman | 101.9 | 138.3 | 12.62 | 0.942 | 0.282 |
| oudcv | 98.4 | 134.6 | — | — | — |
| c1_tfm_abs_s0 | 90.4 | 128.6 | 11.51 | 0.933 | 0.233 |
| c1_tfm_abs_s1 | 92.1 | 131.9 | 11.58 | 0.942 | 0.260 |
| c1_tfm_abs_s2 | 90.4 | 128.5 | 11.65 | 0.926 | 0.202 |
| c1_tfm_aug_s0 | 91.7 | 129.6 | 11.77 | 0.930 | 0.211 |
| c1_tfm_aug_s1 | 98.9 | 136.3 | 12.68 | 0.902 | 0.127 |
| c1_tfm_aug_s2 | 92.3 | 130.1 | 12.41 | 0.921 | 0.177 |

## cst (40 sequences, 28,762 anchors; classical reference `dcv`)

H1b **NOT_SUPPORTED** · H2b **LOSES** · H4 (aug) **NOT_SUPPORTED**

| forecaster | ADE px | FDE@24 px | NLL | 95 % coverage | calib. error |
|---|---:|---:|---:|---:|---:|
| kalman | 56.4 | 75.1 | 83.12 | 0.819 | 0.277 |
| dcv | 55.3 | 71.2 | — | — | — |
| c1_tfm_abs_s0 | 49.7 | 59.4 | 1395.18 | 0.800 | 0.188 |
| c1_tfm_abs_s1 | 49.5 | 59.2 | 813.72 | 0.801 | 0.191 |
| c1_tfm_abs_s2 | 49.9 | 59.5 | 789.07 | 0.799 | 0.179 |
| c1_tfm_aug_s0 | 48.8 | 57.6 | 514.12 | 0.796 | 0.167 |
| c1_tfm_aug_s1 | 51.2 | 62.0 | 1311.92 | 0.788 | 0.119 |
| c1_tfm_aug_s2 | 48.4 | 57.0 | 636.97 | 0.782 | 0.151 |

