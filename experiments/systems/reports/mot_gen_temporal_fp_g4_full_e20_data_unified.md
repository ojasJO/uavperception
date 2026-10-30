# Established trackers vs single-target layers (MOT protocol) — gen_temporal_fp_g4_full_e20 (data_unified)

*Generated 2026-10-05T07:22:16+0530 by `scripts/mot_eval.py`; MOT (no init box); detections conf >= 0.05, drone class; pooled CLEAR-MOT, IDF1 mean per sequence.*

| tracker | group | sequences | MOTA | IDF1 | ID switches | at re-entry | predicted IDs | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bytetrack | anti_uav_IR | 67 | 0.9257 | 0.6042 | 1391 | 1 | 1029 | 1102 | 2011 |
| bytetrack | anti_uav_RGB | 67 | 0.8524 | 0.5615 | 2011 | 11 | 1232 | 4077 | 2516 |
| bytetrack | cst_IR | 15 | 0.7105 | 0.5953 | 296 | 3 | 96 | 1772 | 1411 |
| sort | anti_uav_IR | 67 | 0.9325 | 0.6054 | 1339 | 2 | 980 | 566 | 2187 |
| sort | anti_uav_RGB | 67 | 0.8821 | 0.5500 | 1965 | 10 | 1267 | 1827 | 3083 |
| sort | cst_IR | 15 | 0.7571 | 0.6116 | 113 | 4 | 101 | 483 | 2323 |
| argmax@0.25 | anti_uav_IR | 67 | 0.9765 | 0.9811 | 0 | 0 | 67 | 639 | 786 |
| argmax@0.25 | anti_uav_RGB | 67 | 0.9490 | 0.9686 | 0 | 0 | 67 | 1870 | 1106 |
| argmax@0.25 | cst_IR | 15 | 0.7905 | 0.8813 | 0 | 0 | 15 | 1015 | 1502 |
| argmax@cf | anti_uav_IR | 67 | 0.9770 | 0.9816 | 0 | 0 | 67 | 538 | 857 |
| argmax@cf | anti_uav_RGB | 67 | 0.9506 | 0.9692 | 0 | 0 | 67 | 1646 | 1232 |
| argmax@cf | cst_IR | 15 | 0.8070 | 0.8844 | 0 | 0 | 15 | 578 | 1741 |
| ste@cf | anti_uav_IR | 67 | 0.9771 | 0.9822 | 0 | 0 | 67 | 564 | 825 |
| ste@cf | anti_uav_RGB | 67 | 0.9404 | 0.9644 | 0 | 0 | 67 | 1929 | 1546 |
| ste@cf | cst_IR | 15 | 0.8171 | 0.8972 | 0 | 0 | 15 | 767 | 1431 |
