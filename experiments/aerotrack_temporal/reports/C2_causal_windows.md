# C2 — causal frame-stacking windows (Track C, corpus-v2 VAL)

*Generated 2026-10-10T13:26:00+0530 by `scripts/analyze_c2.py` (B = 1000); rules: `experiments/aerotrack_temporal/preregistration/C2_causal_windows.json` with the v2 margins (superiority 0.02, Anti-UAV non-inferiority 0.02).*

## W2 — CST drone AP50 by window (seed mean, bootstrap 95 % CI)

| window | frames | look-ahead | seed-mean CST AP50 | 95 % CI | per seed | Anti-UAV AP50 |
|---|---|---:|---:|---|---|---:|
| stock | [t, t, t] | 0 | 0.5698 | [0.4752, 0.6614] | 0.5636, 0.5761 | 0.9800 |
| causal_g1 | [t-2, t-1, t] | 0 | 0.6020 | [0.5078, 0.6875] | 0.6148, 0.5892 | 0.9796 |
| causal_g2 | [t-4, t-2, t] | 0 | 0.6627 | [0.5773, 0.7442] | 0.6705, 0.6550 | 0.9787 |
| causal_g4 | [t-8, t-4, t] | 0 | 0.6808 | [0.5975, 0.7599] | 0.6760, 0.6857 | 0.9810 |
| causal_g8 | [t-16, t-8, t] | 0 | 0.6791 | [0.5933, 0.7601] | 0.6668, 0.6915 | 0.9823 |
| sym_g4 | [t-4, t, t+4] | 4 frames | 0.6735 | [0.5902, 0.7543] | 0.6780, 0.6689 | 0.9803 |

## W1 — each causal window vs single frame (superiority)

| window | per seed Δ | mean Δ | 95 % CI | Anti-UAV Δ | status |
|---|---|---:|---|---:|---|
| causal_g1 | +0.0512, +0.0131 | +0.0322 | [+0.0145, +0.0505] | -0.0005 | **SUPPORTED** |
| causal_g2 | +0.1069, +0.0789 | +0.0929 | [+0.0593, +0.1298] | -0.0013 | **SUPPORTED** |
| causal_g4 | +0.1124, +0.1096 | +0.1110 | [+0.0710, +0.1546] | +0.0009 | **SUPPORTED** |
| causal_g8 | +0.1032, +0.1154 | +0.1093 | [+0.0663, +0.1589] | +0.0023 | **SUPPORTED** |

## W3 — cost of causality

best W1-superior window `causal_g4` vs `sym_g4`: mean Δ +0.0074, CI [-0.0002, +0.0161] → **NON-INFERIOR (conclusive)**

**Selected window for the causal system:** causal_g4

*bootstrap = sampling noise over VAL sequences; seed noise only via the per-seed deltas (2 seeds); the selected window's VAL delta is optimistic (winner's curse) until a TEST pass*
