# Complete-stack deployment gate — v2_causal_g4_s0_e20 (trt_fp16, tracker ste)

*Generated 2026-10-12T14:39:53+0530 by `scripts/stack_gate.py`: 6 VAL sequences, 300 frames each, drawing off; structural look-ahead 0 ms on top of the per-frame compute. GPU: NVIDIA GeForce RTX 5050 Laptop GPU, 592.19, 62, 2580 MHz, 31.16 W, 822 MiB.*

| group | stage | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|
| anti_uav_IR | read_preprocess | 1.09 | 1.34 | 1.58 |
| anti_uav_IR | window | 0.13 | 0.21 | 0.27 |
| anti_uav_IR | infer | 2.64 | 3.69 | 4.04 |
| anti_uav_IR | nms | 0.90 | 1.27 | 1.38 |
| anti_uav_IR | tracker | 0.29 | 0.51 | 0.55 |
| anti_uav_IR | total | 5.09 | 6.82 | 7.29 |
| anti_uav_RGB | read_preprocess | 3.90 | 4.93 | 5.14 |
| anti_uav_RGB | window | 0.14 | 0.28 | 0.35 |
| anti_uav_RGB | infer | 2.65 | 3.84 | 4.55 |
| anti_uav_RGB | nms | 0.88 | 1.31 | 1.54 |
| anti_uav_RGB | tracker | 0.29 | 0.53 | 0.59 |
| anti_uav_RGB | total | 7.86 | 10.58 | 10.98 |
| cst_IR | read_preprocess | 1.07 | 1.28 | 1.49 |
| cst_IR | window | 0.13 | 0.21 | 0.28 |
| cst_IR | infer | 2.58 | 3.43 | 3.73 |
| cst_IR | nms | 0.85 | 1.25 | 1.58 |
| cst_IR | tracker | 0.29 | 0.47 | 0.52 |
| cst_IR | total | 4.95 | 6.37 | 6.99 |
| pooled | read_preprocess | 1.16 | 4.79 | 4.98 |
| pooled | window | 0.13 | 0.24 | 0.30 |
| pooled | infer | 2.62 | 3.69 | 4.17 |
| pooled | nms | 0.87 | 1.29 | 1.53 |
| pooled | tracker | 0.29 | 0.51 | 0.56 |
| pooled | total | 5.89 | 10.17 | 10.76 |
