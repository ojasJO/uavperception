# Detector comparison — val set `6fb41d44d45b` (11,565 images: {'anti_uav': 5000, 'cst': 5000, 'det_fly': 1565}, 1714 sequences)

Every row is scored on exactly this image set (hash above) with conf 0.001 / IoU 0.7 / max_det 30; numbers come from `evaluation_report.json` files written by `evaluate_detector.py`. Size bins: micro < 0.03 % of source-frame area, small 0.03–1 %, standard > 1 %; pixel equivalents: anti_uav/IR micro ≤ 9.9 px, anti_uav/RGB micro ≤ 24.9 px, cst/IR micro ≤ 9.9 px, det_fly/RGB micro ≤ 11.1 px. Input: each checkpoint is scored on its own training input (t,t,t for temporal_gap-0 runs; rows without the field predate input modes and are 3-frame t-1,t,t+1).

| run | ckpt | epoch | input | mAP50 | mAP50-95 | drone AP50 | bird AP50 | micro AP50 (n) | small | standard | Anti-UAV | CST | CST micro | Det-Fly | bird-only imgs w/ drone alarm @.25 | GT birds boxed as drone @.25 | empty-frame drone FP/img @.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aerotrack_spd_finetuned | best | 23 | t-1,t,t+1 | 0.7788 | 0.4140 | 0.8027 | 0.7548 | 0.4078 (2008) | 0.7933 | 0.8454 | 0.9534 | 0.6392 | 0.6091 | 0.7388 | 0.0000 % | 0.0000 % | 0.0869 |
| aerotrack_spd_finetuned | last | 24 | t-1,t,t+1 | 0.7799 | 0.4158 | 0.8049 | 0.7549 | 0.4062 (2008) | 0.7943 | 0.8547 | 0.9533 | 0.6436 | 0.6112 | 0.7407 | 0.0000 % | 0.0000 % | 0.0877 |
| aerotrack_spd_v1 | best | 15 | t-1,t,t+1 | 0.6074 | 0.2991 | 0.7925 | 0.4224 | 0.2929 (2008) | 0.6240 | 0.4932 | 0.9775 | 0.6767 | 0.6594 | 0.3974 | 20.6 % | 0.8036 % | 0.0704 |
| aerotrack_spd_v1 | last | 21 | t-1,t,t+1 | 0.6505 | 0.3188 | 0.8286 | 0.4724 | 0.3293 (2008) | 0.6602 | 0.5358 | 0.9813 | 0.7420 | 0.6846 | 0.4454 | 3.1 % | 0.1507 % | 0.0778 |
| repair_abl_baseline_ctrl_e20 | best | 13 | t-1,t,t+1 | 0.6052 | 0.3009 | 0.7791 | 0.4313 | 0.2790 (2008) | 0.6170 | 0.5105 | 0.9751 | 0.6374 | 0.6187 | 0.4095 | 4.1 % | 0.1005 % | 0.1200 |
| repair_abl_baseline_ctrl_e20 | last | 19 | t-1,t,t+1 | 0.6409 | 0.3233 | 0.8030 | 0.4788 | 0.2804 (2008) | 0.6585 | 0.6494 | 0.9816 | 0.6677 | 0.6269 | 0.4650 | 4.1 % | 0.0502 % | 0.1018 |
| repair_abl_no_spd_e20 | best | 10 | t-1,t,t+1 | 0.6279 | 0.3055 | 0.7975 | 0.4582 | 0.2949 (2008) | 0.6452 | 0.4780 | 0.9718 | 0.7098 | 0.6579 | 0.3908 | 16.5 % | 0.3516 % | 0.2409 |
| repair_abl_no_spd_e20 | last | 19 | t-1,t,t+1 | 0.6996 | 0.3453 | 0.8415 | 0.5576 | 0.3588 (2008) | 0.7114 | 0.5387 | 0.9815 | 0.7621 | 0.7288 | 0.5163 | 4.1 % | 0.2009 % | 0.1316 |
| repair_abl_single_frame_e20 | best | 19 | t,t,t | 0.6533 | 0.3244 | 0.8065 | 0.5002 | 0.2873 (2008) | 0.6707 | 0.5609 | 0.9807 | 0.6825 | 0.6119 | 0.4559 | 7.2 % | 0.1005 % | 0.0778 |
| repair_abl_single_frame_e20 | last | 19 | t,t,t | 0.6533 | 0.3244 | 0.8065 | 0.5002 | 0.2873 (2008) | 0.6707 | 0.5609 | 0.9807 | 0.6825 | 0.6119 | 0.4559 | 7.2 % | 0.1005 % | 0.0778 |
| repair_abl_stock_yolo11n_e20 | best | 18 | t,t,t | 0.6708 | 0.3371 | 0.8337 | 0.5079 | 0.3182 (2008) | 0.6837 | 0.5894 | 0.9820 | 0.7305 | 0.7078 | 0.5208 | 1.0309 % | 0.0000 % | 0.1167 |
| repair_abl_stock_yolo11n_e20 | last | 19 | t,t,t | 0.6790 | 0.3415 | 0.8379 | 0.5200 | 0.3184 (2008) | 0.6937 | 0.5879 | 0.9828 | 0.7327 | 0.7091 | 0.5336 | 3.1 % | 0.0502 % | 0.1126 |
