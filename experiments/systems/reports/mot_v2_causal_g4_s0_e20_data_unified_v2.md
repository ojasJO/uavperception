# Established trackers vs single-target layers (MOT protocol) — v2_causal_g4_s0_e20 (data_unified_v2)

*Generated 2026-10-06T14:13:23+0530 by `scripts/mot_eval.py`; MOT (no init box); detections conf >= 0.05, drone class; pooled CLEAR-MOT, IDF1 mean per sequence.*

| tracker | group | sequences | MOTA | IDF1 | ID switches | at re-entry | predicted IDs | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bytetrack | anti_uav_IR | 67 | 0.9192 | 0.6125 | 1485 | 1 | 1040 | 1307 | 2108 |
| bytetrack | anti_uav_RGB | 67 | 0.8610 | 0.5455 | 2054 | 9 | 1221 | 3674 | 2373 |
| bytetrack | cst_IR | 40 | 0.3892 | 0.4265 | 1352 | 7 | 959 | 12178 | 10326 |
| sort | anti_uav_IR | 67 | 0.9293 | 0.6165 | 1358 | 1 | 986 | 636 | 2293 |
| sort | anti_uav_RGB | 67 | 0.8855 | 0.5625 | 1993 | 6 | 1249 | 1633 | 3047 |
| sort | cst_IR | 40 | 0.5482 | 0.4520 | 823 | 3 | 1075 | 4101 | 12722 |
| argmax@0.25 | anti_uav_IR | 67 | 0.9734 | 0.9793 | 0 | 0 | 67 | 722 | 890 |
| argmax@0.25 | anti_uav_RGB | 67 | 0.9517 | 0.9688 | 0 | 0 | 67 | 1812 | 1005 |
| argmax@0.25 | cst_IR | 40 | 0.5587 | 0.7658 | 0 | 0 | 40 | 6918 | 10316 |
| argmax@cf | anti_uav_IR | 67 | 0.9740 | 0.9799 | 0 | 0 | 67 | 662 | 911 |
| argmax@cf | anti_uav_RGB | 67 | 0.9525 | 0.9695 | 0 | 0 | 67 | 1737 | 1033 |
| argmax@cf | cst_IR | 40 | 0.5746 | 0.7686 | 0 | 0 | 40 | 6149 | 10463 |
| ste@cf | anti_uav_IR | 67 | 0.9738 | 0.9806 | 0 | 0 | 67 | 645 | 943 |
| ste@cf | anti_uav_RGB | 67 | 0.9513 | 0.9695 | 0 | 0 | 67 | 1640 | 1200 |
| ste@cf | cst_IR | 40 | 0.5975 | 0.7773 | 0 | 0 | 40 | 5417 | 10302 |
