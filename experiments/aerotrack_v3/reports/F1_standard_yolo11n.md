# F1 — standard YOLO11n blocks vs the as-built network (corpus-v2 VAL)

*Generated 2026-10-06T23:35:36+0530 by `scripts/analyze_f1.py` (B = 1000); window from v2's selection: `causal_g4`.*

| pair | CST AP50 delta |
|---|---:|
| v3_n_std__causal_g4_s0_e20 vs v2_causal_g4_s0_e20 | +0.0044 |
| v3_n_std__causal_g4_s1_e20 vs v2_causal_g4_s1_e20 | -0.0243 |

Seed-mean delta -0.0100, 95 % CI [-0.0259, +0.0064], Anti-UAV -0.0002 -> **NOT SUPPORTED** ((1) mean delta -0.0100 <= 0.02; (2) not positive for every seed; (3) CI includes 0)

Adoption: X1 and CF stay on the as-built network.
