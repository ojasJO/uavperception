# Detector comparison — test set `37e944793773` (11,567 images: {'anti_uav': 5000, 'cst': 5000, 'det_fly': 1567}, 1809 sequences)

Every row is scored on exactly this image set (hash above) with conf 0.001 / IoU 0.7 / max_det 30; numbers come from `evaluation_report.json` files written by `evaluate_detector.py`. Size bins: micro < 0.03 % of source-frame area, small 0.03–1 %, standard > 1 %; pixel equivalents: anti_uav/IR micro ≤ 9.9 px, anti_uav/RGB micro ≤ 24.9 px, cst/IR micro ≤ 9.9 px, det_fly/RGB micro ≤ 11.1 px. Input: each checkpoint is scored on its own training input (t,t,t for temporal_gap-0 runs; rows without the field predate input modes and are 3-frame t-1,t,t+1).

| run | ckpt | epoch | input | mAP50 | mAP50-95 | drone AP50 | bird AP50 | micro AP50 (n) | small | standard | Anti-UAV | CST | CST micro | Det-Fly | bird-only imgs w/ drone alarm @.25 | GT birds boxed as drone @.25 | empty-frame drone FP/img @.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aerotrack_spd_finetuned | best | 23 | t-1,t,t+1 | 0.7344 | 0.3776 | 0.6745 | 0.7944 | 0.2862 (2722) | 0.7678 | 0.9079 | 0.8787 | 0.3933 | 0.3174 | 0.7685 | 2.1 % | 0.0615 % | 0.4349 |
| aerotrack_spd_finetuned | last | 24 | t-1,t,t+1 | 0.7339 | 0.3774 | 0.6755 | 0.7924 | 0.2877 (2722) | 0.7676 | 0.9073 | 0.8816 | 0.3931 | 0.3182 | 0.7668 | 2.1 % | 0.0615 % | 0.4377 |
| aerotrack_spd_v1 | best | 15 | t-1,t,t+1 | 0.5491 | 0.2428 | 0.6590 | 0.4391 | 0.2049 (2722) | 0.5693 | 0.4997 | 0.9015 | 0.4471 | 0.3950 | 0.4145 | 18.6 % | 0.7376 % | 0.2696 |
| aerotrack_spd_v1 | last | 21 | t-1,t,t+1 | 0.5858 | 0.2629 | 0.6861 | 0.4855 | 0.2543 (2722) | 0.6007 | 0.5630 | 0.9064 | 0.4876 | 0.4236 | 0.4613 | 0.0000 % | 0.0000 % | 0.4700 |
| repair_abl_baseline_ctrl_e20 | best | 13 | t-1,t,t+1 | 0.5396 | 0.2402 | 0.6557 | 0.4234 | 0.1726 (2722) | 0.5636 | 0.5429 | 0.9012 | 0.4245 | 0.3399 | 0.4017 | 5.2 % | 0.0615 % | 0.3878 |
| repair_abl_baseline_ctrl_e20 | last | 19 | t-1,t,t+1 | 0.5693 | 0.2584 | 0.6607 | 0.4779 | 0.1641 (2722) | 0.5982 | 0.6250 | 0.9087 | 0.4118 | 0.3290 | 0.4692 | 4.1 % | 0.0000 % | 0.6362 |
| repair_abl_no_spd_e20 | best | 10 | t-1,t,t+1 | 0.5567 | 0.2515 | 0.6528 | 0.4607 | 0.2033 (2722) | 0.5723 | 0.5111 | 0.8992 | 0.4365 | 0.3422 | 0.4026 | 15.5 % | 0.6146 % | 0.6953 |
| repair_abl_no_spd_e20 | last | 19 | t-1,t,t+1 | 0.6124 | 0.2776 | 0.6679 | 0.5570 | 0.2129 (2722) | 0.6260 | 0.4870 | 0.9175 | 0.4384 | 0.3353 | 0.5293 | 11.3 % | 0.5532 % | 0.7091 |
| repair_abl_single_frame_e20 | best | 19 | t,t,t | 0.5838 | 0.2642 | 0.6740 | 0.4937 | 0.2045 (2722) | 0.6064 | 0.5641 | 0.9012 | 0.4672 | 0.4055 | 0.4688 | 5.2 % | 0.0000 % | 0.5891 |
| repair_abl_single_frame_e20 | last | 19 | t,t,t | 0.5838 | 0.2642 | 0.6740 | 0.4937 | 0.2045 (2722) | 0.6064 | 0.5641 | 0.9012 | 0.4672 | 0.4055 | 0.4688 | 5.2 % | 0.0000 % | 0.5891 |
| repair_abl_stock_yolo11n_e20 | best | 18 | t,t,t | 0.5778 | 0.2692 | 0.6738 | 0.4819 | 0.2147 (2722) | 0.5860 | 0.6019 | 0.9078 | 0.4508 | 0.3987 | 0.5190 | 1.0309 % | 0.0615 % | 0.5476 |
| repair_abl_stock_yolo11n_e20 | last | 19 | t,t,t | 0.5913 | 0.2756 | 0.6837 | 0.4988 | 0.2131 (2722) | 0.6061 | 0.6152 | 0.9105 | 0.4561 | 0.3996 | 0.5345 | 2.1 % | 0.0615 % | 0.5503 |
