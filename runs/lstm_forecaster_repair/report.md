# Trajectory forecasting — repaired comparison (ADE in px, lower is better)

Generated 2026-09-21T16:42:27+0530 (git d1e468551f). Windows: train 89,699, val 29,229, test 54,651. Selection: per variant: epoch with the lowest VAL ADE; Kalman (q,r) by VAL sweep; TEST scored once for all methods.

## VAL

| subset | n | zero_vel | last_vel | mean_vel_4 | mean_vel_8 | mean_vel_16 | kalman_project(q=1e-2,r=1e-1) | kalman_val_tuned(q=1,r=1e-05) | lstm_original | lstm_residual_lv | lstm_residual_mv4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 29229 | 49.85 | 66.21 | 67.94 | 68.58 | 65.23 | 68.38 | 64.51 | 45.84 | 47.60 | 47.35 |
| sudden-jump (>30px in horizon) | 6074 | 126.24 | 188.07 | 168.68 | 160.46 | 154.69 | 166.59 | 176.04 | 115.70 | 119.21 | 117.98 |
| smooth | 23155 | 29.81 | 34.24 | 41.51 | 44.47 | 41.76 | 42.62 | 35.25 | 27.51 | 28.82 | 28.82 |
| Anti-UAV | 29229 | 49.85 | 66.21 | 67.94 | 68.58 | 65.23 | 68.38 | 64.51 | 45.84 | 47.60 | 47.35 |
| CST Anti-UAV | 0 | — | — | — | — | — | — | — | — | — | — |
| CST sudden-jump | 0 | — | — | — | — | — | — | — | — | — | — |
| Anti-UAV sudden-jump | 6074 | 126.24 | 188.07 | 168.68 | 160.46 | 154.69 | 166.59 | 176.04 | 115.70 | 119.21 | 117.98 |

## TEST

| subset | n | zero_vel | last_vel | mean_vel_4 | mean_vel_8 | mean_vel_16 | kalman_project(q=1e-2,r=1e-1) | kalman_val_tuned(q=1,r=1e-05) | lstm_original | lstm_residual_lv | lstm_residual_mv4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 54651 | 41.88 | 57.63 | 56.45 | 56.38 | 52.32 | 56.50 | 54.79 | 38.78 | 40.15 | 39.78 |
| sudden-jump (>30px in horizon) | 10752 | 109.86 | 165.01 | 151.66 | 143.48 | 134.10 | 149.16 | 155.41 | 99.89 | 103.45 | 102.36 |
| smooth | 43899 | 25.23 | 31.33 | 33.13 | 35.05 | 32.29 | 33.80 | 30.15 | 23.81 | 24.64 | 24.46 |
| Anti-UAV | 40314 | 52.31 | 71.90 | 73.36 | 73.39 | 67.67 | 73.69 | 70.13 | 47.44 | 49.34 | 48.91 |
| CST Anti-UAV | 14337 | 12.55 | 17.52 | 8.90 | 8.55 | 9.15 | 8.16 | 11.67 | 14.45 | 14.29 | 14.12 |
| CST sudden-jump | 721 | 60.44 | 89.78 | 52.44 | 53.24 | 56.79 | 49.98 | 63.96 | 62.97 | 62.71 | 63.01 |
| Anti-UAV sudden-jump | 10031 | 113.42 | 170.41 | 158.80 | 149.96 | 139.66 | 156.29 | 161.99 | 102.55 | 106.38 | 105.19 |

## Does the model move? (test windows, normalised units)

- true mean future step: 0.00705; mean last observed step: 0.00708
- lstm_original: mean predicted step 0.00180, corr(step1, last obs) +0.814
- lstm_residual_lv: mean predicted step 0.00190, corr(step1, last obs) +0.393
- lstm_residual_mv4: mean predicted step 0.00184, corr(step1, last obs) +0.584

## Reading rule (pre-registered)

The forecaster earns its place only if, on the TEST table, it beats the best trivial baseline (min over zero/last/mean-vel) AND the val-tuned Kalman on the CST sudden-jump subset — the population the citation rests on. Otherwise the LSTM is retired from the pipeline and the README says so.
