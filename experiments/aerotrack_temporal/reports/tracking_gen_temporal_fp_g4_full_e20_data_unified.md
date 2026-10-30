# Tracking: identity persistence, continuity, recovery — gen_temporal_fp_g4_full_e20 (data_unified, VAL)

*Generated 2026-10-05T14:04:05+0530 by `scripts/tc_track_eval.py`; MOT (no init box); drone detections conf >= 0.05; one GT identity per sequence; groups: ['anti_uav_IR', 'anti_uav_RGB']; cross-fitted parameters from `track_gen_temporal_fp_g4_full_e20_data_unified.json`. Recovery in frames (25 = 1 s at 25 fps); 'never' = not recovered before the sequence ends.*

| tracker | group | HOTA | DetA | AssA | MOTA | IDF1 | ID sw | Frag | MT / ML | longest run | ids / target | loss → recovered: median / p90 / never | re-entry → recovered: median / p90 / never |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| bytetrack | anti_uav_IR | 0.574 | 0.741 | 0.445 | 0.926 | 0.604 | 1391 | 908 | 1.00 / 0.00 | 0.48 | 14.5 | 1 / 4 / 2 of 903 | 2 / 17 / 0 of 12 |
| bytetrack | anti_uav_RGB | 0.543 | 0.716 | 0.414 | 0.852 | 0.561 | 2011 | 1135 | 0.97 / 0.00 | 0.50 | 17.1 | 1 / 4 / 3 of 1131 | 0 / 6 / 0 of 26 |
| sort | anti_uav_IR | 0.575 | 0.745 | 0.445 | 0.932 | 0.605 | 1339 | 908 | 0.99 / 0.00 | 0.48 | 14.3 | 1 / 4 / 2 of 905 | 5 / 17 / 0 of 12 |
| sort | anti_uav_RGB | 0.544 | 0.737 | 0.403 | 0.882 | 0.550 | 1965 | 1232 | 0.97 / 0.00 | 0.48 | 17.7 | 1 / 4 / 4 of 1230 | 0 / 9 / 0 of 26 |
| argmax@cf | anti_uav_IR | 0.789 | 0.779 | 0.800 | 0.977 | 0.982 | 0 | 289 | 0.99 / 0.00 | 0.68 | 1.0 | 1 / 5 / 0 of 286 | 1 / 15 / 0 of 12 |
| argmax@cf | anti_uav_RGB | 0.798 | 0.781 | 0.816 | 0.951 | 0.969 | 0 | 454 | 0.99 / 0.00 | 0.75 | 1.0 | 1 / 6 / 2 of 450 | 0 / 6 / 0 of 26 |
| ste@cf | anti_uav_IR | 0.789 | 0.779 | 0.800 | 0.977 | 0.982 | 0 | 286 | 1.00 / 0.00 | 0.68 | 1.0 | 1 / 4 / 0 of 282 | 1 / 16 / 0 of 12 |
| ste@cf | anti_uav_RGB | 0.795 | 0.777 | 0.815 | 0.940 | 0.964 | 0 | 438 | 0.97 / 0.00 | 0.75 | 1.0 | 1 / 8 / 1 of 432 | 0 / 8 / 0 of 26 |
| ste@cf/segments | anti_uav_IR | 0.774 | 0.779 | 0.770 | 0.977 | 0.951 | 8 | 286 | 1.00 / 0.00 | 0.68 | 1.1 | 1 / 4 / 0 of 282 | 1 / 16 / 0 of 12 |
| ste@cf/segments | anti_uav_RGB | 0.767 | 0.777 | 0.758 | 0.940 | 0.902 | 28 | 438 | 0.97 / 0.00 | 0.75 | 1.4 | 1 / 8 / 1 of 432 | 0 / 8 / 0 of 26 |
