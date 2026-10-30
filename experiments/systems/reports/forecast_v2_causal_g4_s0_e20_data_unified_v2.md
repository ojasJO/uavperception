# Forecasting from the tracker's own state — v2_causal_g4_s0_e20 (data_unified_v2)

*Generated 2026-10-06T14:00:59+0530 by `scripts/forecast_eval.py`; STE parameters: cross-fitted per half from track_v2_causal_g4_s0_e20_data_unified_v2.json. Errors in pixels at the original resolution; every forecaster sees only the tracker's past outputs.*

| group | windows | points | ADE kalman / zero / last-vel / mean-vel4 (px) | FDE@12 kalman / zero / last / mean4 | Kalman 95 % ellipse coverage (all / jumps) |
|---|---|---:|---|---|---|
| anti_uav_IR | all | 715,781 | 28.3 / 27.8 / 36.6 / 37.2 | 49.2 / 43.7 / 72.3 / 70.5 | 0.72 |
| anti_uav_IR | sudden_jump | 106,185 | 85.1 / 74.9 / 111.0 / 96.2 | 152.5 / 111.3 / 220.0 / 182.8 | 0.41 |
| anti_uav_IR | smooth | 609,596 | 18.3 / 19.6 / 23.6 / 27.0 | 31.2 / 31.9 / 46.6 / 51.0 | 0.77 |
| all | all | 1,795,380 | 44.6 / 45.6 / 61.4 / 60.0 | 75.2 / 69.7 / 115.8 / 109.5 | 0.73 |
| all | sudden_jump | 302,298 | 140.4 / 125.6 / 186.0 / 165.9 | 248.0 / 186.3 / 363.3 / 312.4 | 0.41 |
| all | smooth | 1,493,082 | 25.3 / 29.3 / 36.2 / 38.5 | 40.1 / 46.1 / 65.6 / 68.3 | 0.80 |
| anti_uav_RGB | all | 686,667 | 73.0 / 73.9 / 103.2 / 101.6 | 127.8 / 115.6 / 200.9 / 191.2 | 0.72 |
| anti_uav_RGB | sudden_jump | 182,408 | 175.5 / 156.8 / 234.3 / 211.7 | 311.3 / 233.0 / 460.1 / 401.2 | 0.43 |
| anti_uav_RGB | smooth | 504,259 | 36.0 / 43.8 / 55.8 / 61.8 | 61.3 / 73.0 / 106.9 / 115.1 | 0.83 |
| cst_IR | all | 392,932 | 24.9 / 28.4 / 33.5 / 28.7 | 30.4 / 36.9 / 46.0 / 37.4 | 0.77 |
| cst_IR | sudden_jump | 13,705 | 101.9 / 103.1 / 123.9 / 96.2 | 143.3 / 145.2 / 181.4 / 131.6 | 0.13 |
| cst_IR | smooth | 379,227 | 22.1 / 25.7 / 30.2 / 26.3 | 26.4 / 33.1 / 41.2 / 34.1 | 0.79 |
