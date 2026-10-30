# H3 — optimisation regime (8-GPU convention) and the test surplus / DFSC effect

*Generated 2026-10-06T12:29:34 by `scripts/analyze_h3.py`; rules frozen in `experiments/anti_uav/preregistration/H3_optimisation_regime.json`. Do not edit numbers.*

**H3a (test surplus): NOT_SUPPORTED** — the 8-GPU regime does not reconcile val and test with the paper for both arms.  
**H3b (DFSC effect in the paper's regime): NOT_OBSERVED** — DFSC - normal <= +0.1 on at least one split.

| arm | split | paper | 1 GPU x b2, LR 0.0025 (Phase A) | 8-GPU regime (b16, LR 0.02) | regime − Phase A | regime − paper |
|---|---|---:|---:|---:|---:|---:|
| normal | val | 79.60 | 79.29 | 79.16 | -0.13 | -0.44 |
| normal | test | 65.36 | 70.58 | 70.40 | -0.18 | +5.04 |
| DFSC | val | 80.09 | 79.26 | 79.14 | -0.12 | -0.95 |
| DFSC | test | 66.04 | 70.29 | 69.76 | -0.52 | +3.72 |

DFSC − normal: 8-GPU regime val -0.02, test -0.64; Phase A regime val -0.03, test -0.29; paper val +0.49, test +0.68.
