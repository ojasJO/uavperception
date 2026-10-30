# Latency — aerotrack_spd_finetuned (epoch 23), NVIDIA GeForce RTX 5050 Laptop GPU, batch 1, 640×640, 200 iterations

| stage | mean ms | p50 ms | p95 ms |
|---|---|---|---|
| forward_bf16 | 12.39 | 11.09 | 20.12 |
| forward_fp32 | 8.17 | 7.57 | 10.62 |
| preprocess_3frame_letterbox | 5.76 | 5.67 | 6.36 |
| nms_cpu_demo_conf.25_iou.45 | 0.5 | 0.51 | 0.63 |
| nms_cpu_eval_conf.001_iou.7 | 11.39 | 8.47 | 16.98 |
| bytetrack_update | 0.17 | 0.16 | 0.19 |

- end-to-end fp32 (deployed, no tracker): **14.43 ms ≈ 69.3 FPS**
- end-to-end bf16 (not used by inference.py): 18.65 ms ≈ 53.6 FPS
- plus structural look-ahead of one source frame: +40.0 ms at 25 FPS
