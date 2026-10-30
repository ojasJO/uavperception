# C1c — camera-mode-aware learned forecasters (Track C, oracle histories, VAL)

*Generated 2026-10-12T02:15:03+0530 by `scripts/tc_c1c.py --stage val` under `preregistration/C1c_domain_aware_forecasting.json`.*

| domain | dev-chosen arm | H1c vs classical | H2c vs pooled C1 | H3c uncertainty | ADE: C1c s0/s1/s2 · pooled s0/s1/s2 · reference |
|---|---|---|---|---|---|
| antiuav_IR | perdom | **SUPPORTED** | **NOT_SUPPORTED** | **NOT_SUPPORTED** | 34.8 / 33.6 / 34.4 · 33.7 / 34.3 / 33.7 · 37.3 (oudcv) |
| antiuav_RGB | domtok | **SUPPORTED** | **NOT_SUPPORTED** | **NOT_SUPPORTED** | 88.0 / 88.6 / 88.4 · 89.1 / 90.9 / 89.0 · 97.7 (oudcv) |
| cst | perdom | **NOT_SUPPORTED** | **SUPPORTED** | **SUPPORTED** | 12.0 / 11.7 / 12.1 · 17.5 / 17.6 / 17.9 · 12.1 (dcv) |
