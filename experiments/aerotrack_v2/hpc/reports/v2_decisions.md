# AeroTrack v2 series — decisions (pre-registered rules)

*Generated 2026-10-06T11:32:37+0530 by `scripts/analyze_v2.py` from cached VAL records (B = 1000); rules frozen in `preregistration.json`.*

| id | question | metric | per seed | mean Δ | 95 % CI | Anti-UAV Δ | status |
|---|---|---|---|---:|---|---:|---|
| C1 | causal window vs single frame | cst | +0.112, +0.110 | +0.1110 | [+0.071, +0.155] | +0.0009 | SUPPORTED |
| C2 | symmetric window vs single frame (replication on complete data) | cst | +0.114, +0.093 | +0.1036 | [+0.065, +0.146] | +0.0002 | SUPPORTED |
| C3 | causal vs symmetric window (cost of removing the look-ahead) | cst | -0.002, +0.017 | +0.0074 | [-0.000, +0.016] | +0.0007 | NON-INFERIOR (conclusive) |
| C4 | data effect: complete vs truncated CST training data (same window, seed 0; n = 1 seed, CI = sampling noise only) | cst | +0.079 | +0.0789 | [+0.056, +0.102] | -0.0010 | SUPPORTED |

**Selected candidate (selection rule): causal_g4**

Per-category CST drone AP50 deltas (seed-mean):

- C1: building +0.088, cn_mountains +0.210, cn_sky +0.035, jungle +0.001, urban-areas +0.090, water +0.004
- C2: building +0.087, cn_mountains +0.195, cn_sky +0.043, jungle +0.001, urban-areas +0.068, water +0.010
- C3: building +0.000, cn_mountains +0.015, cn_sky -0.008, jungle +0.000, urban-areas +0.022, water -0.007
- C4: building +0.068, cn_mountains +0.065, cn_sky +0.034, jungle +0.000, urban-areas +0.115, water +0.069
