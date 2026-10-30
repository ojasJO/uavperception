# Forecasting from the tracker's own state — gen_temporal_fp_g4_full_e20 (data_unified)

*Generated 2026-10-05T07:17:55+0530 by `scripts/forecast_eval.py`; STE parameters: cross-fitted per half from track_gen_temporal_fp_g4_full_e20_data_unified.json. Errors in pixels at the original resolution; every forecaster sees only the tracker's past outputs.*

| group | windows | points | ADE kalman / zero / last-vel / mean-vel4 (px) | FDE@12 kalman / zero / last / mean4 | Kalman 95 % ellipse coverage (all / jumps) |
|---|---|---:|---|---|---|
| anti_uav_IR | all | 716,508 | 28.0 / 27.8 / 36.8 / 37.5 | 49.1 / 43.7 / 72.7 / 71.0 | 0.73 |
| anti_uav_IR | sudden_jump | 106,294 | 84.7 / 75.0 / 111.1 / 96.2 | 151.2 / 111.3 / 220.4 / 182.9 | 0.41 |
| anti_uav_IR | smooth | 610,214 | 18.1 / 19.6 / 23.8 / 27.3 | 31.3 / 31.9 / 46.9 / 51.5 | 0.79 |
| all | all | 1,535,703 | 45.7 / 46.4 / 63.6 / 62.7 | 80.0 / 73.0 / 124.0 / 118.1 | 0.75 |
| all | sudden_jump | 288,269 | 141.1 / 126.6 / 187.6 / 168.1 | 250.3 / 188.1 / 369.0 / 318.6 | 0.43 |
| all | smooth | 1,247,434 | 23.6 / 27.9 / 34.9 / 38.4 | 40.6 / 46.3 / 67.3 / 71.7 | 0.82 |
| anti_uav_RGB | all | 686,053 | 72.2 / 73.5 / 102.4 / 100.5 | 126.4 / 115.3 / 199.0 / 189.0 | 0.73 |
| anti_uav_RGB | sudden_jump | 181,525 | 174.5 / 157.0 / 232.7 / 210.5 | 308.8 / 233.4 / 456.7 / 398.8 | 0.44 |
| anti_uav_RGB | smooth | 504,528 | 35.4 / 43.5 / 55.5 / 60.9 | 60.6 / 72.6 / 106.0 / 113.4 | 0.84 |
| cst_IR | all | 133,142 | 4.2 / 7.1 / 7.8 / 3.9 | 7.2 / 12.9 / 13.9 / 6.5 | 0.91 |
| cst_IR | sudden_jump | 450 | 33.5 / 37.0 / 38.2 / 32.6 | 60.8 / 67.2 / 68.6 / 57.8 | 0.35 |
| cst_IR | smooth | 132,692 | 4.1 / 7.0 / 7.7 / 3.8 | 7.0 / 12.7 / 13.7 / 6.3 | 0.91 |
