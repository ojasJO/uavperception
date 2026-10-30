# Deployment gate — gen_temporal_fp_g4_full_e20

*Generated 2026-10-05T06:04:24+0530 by `scripts/deploy_gate.py` on NVIDIA GeForce RTX 5050 Laptop GPU (driver 592.19), window `symmetric` (gap 4), batch 1, 640×640. GPU before: 52 °C, SM 180 MHz; after: 59 °C, SM 180 MHz.*

| backend | model p50 / p95 / p99 ms | FPS (mean) | end-to-end p50 / p95 / p99 ms | device MB | startup s | drone AP50 (Δ vs bf16) |
|---|---|---:|---|---:|---:|---|
| torch_fp32 | 18.52 / 21.49 / 24.15 | 60 | 28.96 / 33.57 / 36.35 | 126 | 2.0 | 0.7993 (+0.0032) |
| torch_bf16 | 23.53 / 28.28 / 30.02 | 46 | 19.74 / 30.48 / 31.40 | 68 | 1.6 | 0.7961 (+0.0000) |
| torch_fp16 | 17.81 / 21.04 / 22.87 | 62 | 14.53 / 23.20 / 23.58 | 60 | 1.5 | 0.7945 (-0.0016) |
| ort_cuda | 9.71 / 12.58 / 14.41 | 103 | 9.83 / 14.23 / 14.84 | 80 | 1.1 | 0.7998 (+0.0037) |
| trt_fp32 | 3.54 / 4.53 / 5.13 | 278 | 7.30 / 8.85 / 9.78 | 104 | 40.8 | 0.7989 (+0.0028) |
| trt_fp16 | 3.49 / 4.82 / 5.48 | 271 | 5.93 / 7.82 / 8.32 | 100 | 39.1 | 0.7935 (-0.0027) |

Structural look-ahead of the `symmetric` window: 4 frames = 160 ms at 25 fps, on top of the per-frame compute above.
End-to-end loop: sequence `antiuav_val_20190925_101846_1_4__IR`, 380 timed frames (decode from disk -> letterbox -> window -> inference -> NMS at conf 0.05).
