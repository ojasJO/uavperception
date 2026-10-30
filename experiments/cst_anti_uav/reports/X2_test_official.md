# X2 on CST TEST — seeds and the single-frame control (official toolkit)

| run | arm | seed | official mSA | 95 % CI | vs 35.92 | S-AUC | P@20 | tiny SA |
|---|---|---:|---:|---|---|---:|---:|---:|
| v3_n__causal_g4_cst_s0_e20 | causal_g4 | 0 | 60.08 | [56.13, 63.97] | exceeds | 58.53 | 77.42 | 39.88 |
| v3_n__causal_g4_cst_s1_e20 | causal_g4 | 1 | 60.26 | [56.29, 64.17] | exceeds | 58.72 | 77.61 | 40.89 |
| v3_n__causal_g4_cst_s2_e20 | causal_g4 | 2 | 60.60 | [56.66, 64.40] | exceeds | 59.03 | 77.79 | 42.45 |
| v3_n__ttt_cst_s0_e20 | ttt | 0 | 46.83 | [40.80, 52.84] | exceeds | 45.58 | 58.60 | 31.26 |
| v3_n__ttt_cst_s1_e20 | ttt | 1 | 49.81 | [44.10, 55.47] | exceeds | 48.49 | 62.33 | 31.03 |
| v3_n__ttt_cst_s2_e20 | ttt | 2 | 49.79 | [43.86, 55.74] | exceeds | 48.46 | 61.94 | 31.38 |

Official toolkit vs each pass's Track A values: max per-sequence |Δ| = 0.0e+00 over 12 tracker outputs.

- **T1 causal_g4:** seed mean 60.31 (sd 0.26); all seeds exceed 35.92: **True**. **T2** seed-mean − GlobalTrack (paired): +24.39 [+21.08, +27.80], wins 60 / losses 0 → exceeds
- **T1 ttt:** seed mean 48.81 (sd 1.72); all seeds exceed 35.92: **True**. **T2** seed-mean − GlobalTrack (paired): +12.88 [+8.78, +16.98], wins 43 / losses 17 → exceeds
- **T3** causal_g4 − single frame (seed means, paired): +11.51 [+8.36, +14.86], wins 45 / losses 15, Wilcoxon p = 1.2e-07 → **SUPPORTED**; per seed: s0 +13.26, s1 +10.46, s2 +10.81
