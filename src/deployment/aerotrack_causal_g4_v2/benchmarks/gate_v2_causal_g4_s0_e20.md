# Deployment gate — v2_causal_g4_s0_e20

*Generated 2026-10-06T11:40:21+0530 by `scripts/deploy_gate.py` on NVIDIA GeForce RTX 5050 Laptop GPU (driver 592.19), window `causal` (gap 4), batch 1, 640×640. GPU before: 44 °C, SM 180 MHz; after: 65 °C, SM 180 MHz.*

| backend | model p50 / p95 / p99 ms | FPS (mean) | end-to-end p50 / p95 / p99 ms | device MB | startup s | drone AP50 (Δ vs bf16) |
|---|---|---:|---|---:|---:|---|
| torch_fp32 | 8.36 / 13.65 / 17.74 | 111 | 26.27 / 30.27 / 33.06 | 126 | 1.1 | 0.7300 (+0.0026) |
| torch_bf16 | 11.79 / 19.65 / 23.33 | 79 | 17.10 / 28.52 / 30.83 | 68 | 1.1 | 0.7274 (+0.0000) |
| torch_fp16 | 7.92 / 14.44 / 16.44 | 115 | 12.94 / 21.55 / 22.75 | 54 | 0.9 | 0.7288 (+0.0013) |
| ort_cuda | 5.82 / 9.33 / 9.73 | 152 | 9.41 / 13.24 / 14.21 | 80 | 0.7 | 0.7298 (+0.0024) |
| trt_fp32 | 3.11 / 3.99 / 4.49 | 326 | 6.45 / 7.54 / 8.05 | 106 | 30.3 | 0.7279 (+0.0005) |
| trt_fp16 | 3.20 / 4.34 / 4.66 | 327 | 5.43 / 7.39 / 7.99 | 100 | 22.5 | 0.7290 (+0.0015) |

Structural look-ahead of the `causal` window: 0 frames = 0 ms at 25 fps, on top of the per-frame compute above.
End-to-end loop: sequence `antiuav_val_20190925_101846_1_4__IR`, 380 timed frames (decode from disk -> letterbox -> window -> inference -> NMS at conf 0.05).
