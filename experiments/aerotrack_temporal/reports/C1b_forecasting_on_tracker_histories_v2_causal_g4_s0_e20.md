# C1b — forecasting on tracker histories (v2_causal_g4_s0_e20, data_unified_v2, VAL)

*Generated 2026-10-12T07:27:04+0530 by `scripts/tc_c1b.py`; histories = ste_cv outputs with C3's cross-fitted settings; targets = ground truth. Learned family: tfm.*

## antiuav_IR (67 sequences, 59,939 anchors; classical reference `oudcv`)

H1b **SUPPORTED** · H2b **SUPPORTED** · H4 (aug) **HARMFUL**

| forecaster | ADE px | FDE@24 px | NLL | 95 % coverage | calib. error |
|---|---:|---:|---:|---:|---:|
| kalman | 38.5 | 52.4 | 10.49 | 0.928 | 0.255 |
| oudcv | 37.5 | 52.1 | — | — | — |
| c1_tfm_abs_s0 | 34.1 | 48.6 | 9.16 | 0.926 | 0.199 |
| c1_tfm_abs_s1 | 34.7 | 49.7 | 9.28 | 0.939 | 0.238 |
| c1_tfm_abs_s2 | 34.1 | 48.7 | 9.25 | 0.917 | 0.164 |
| c1_tfm_aug_s0 | 34.7 | 48.9 | 9.46 | 0.928 | 0.180 |
| c1_tfm_aug_s1 | 37.4 | 51.5 | 10.32 | 0.891 | 0.098 |
| c1_tfm_aug_s2 | 34.9 | 49.1 | 9.81 | 0.918 | 0.144 |

## antiuav_RGB (67 sequences, 57,498 anchors; classical reference `oudcv`)

H1b **SUPPORTED** · H2b **NOT_SUPPORTED** · H4 (aug) **HARMFUL**

| forecaster | ADE px | FDE@24 px | NLL | 95 % coverage | calib. error |
|---|---:|---:|---:|---:|---:|
| kalman | 101.9 | 138.4 | 12.62 | 0.942 | 0.281 |
| oudcv | 98.4 | 134.7 | — | — | — |
| c1_tfm_abs_s0 | 90.3 | 128.8 | 11.94 | 0.933 | 0.231 |
| c1_tfm_abs_s1 | 92.0 | 132.0 | 11.64 | 0.941 | 0.259 |
| c1_tfm_abs_s2 | 90.4 | 128.6 | 11.60 | 0.926 | 0.201 |
| c1_tfm_aug_s0 | 91.7 | 129.7 | 11.88 | 0.929 | 0.209 |
| c1_tfm_aug_s1 | 98.9 | 136.5 | 13.16 | 0.901 | 0.123 |
| c1_tfm_aug_s2 | 92.3 | 130.4 | 12.61 | 0.921 | 0.175 |

## cst (40 sequences, 32,251 anchors; classical reference `dcv`)

H1b **NOT_SUPPORTED** · H2b **LOSES** · H4 (aug) **NOT_SUPPORTED**

| forecaster | ADE px | FDE@24 px | NLL | 95 % coverage | calib. error |
|---|---:|---:|---:|---:|---:|
| kalman | 33.4 | 51.8 | 54.29 | 0.917 | 0.357 |
| dcv | 32.5 | 48.5 | — | — | — |
| c1_tfm_abs_s0 | 30.4 | 43.2 | 800.17 | 0.914 | 0.240 |
| c1_tfm_abs_s1 | 30.5 | 43.5 | 533.60 | 0.915 | 0.245 |
| c1_tfm_abs_s2 | 30.9 | 43.5 | 525.79 | 0.910 | 0.228 |
| c1_tfm_aug_s0 | 29.5 | 41.3 | 266.71 | 0.908 | 0.219 |
| c1_tfm_aug_s1 | 32.0 | 46.0 | 440.97 | 0.900 | 0.167 |
| c1_tfm_aug_s2 | 29.1 | 40.7 | 449.36 | 0.894 | 0.193 |

