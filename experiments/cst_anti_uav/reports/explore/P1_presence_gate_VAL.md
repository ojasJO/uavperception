# P1 — presence gate decoupled from STE association (EXPLORATORY, CST VAL only)

*Registration `experiments/cst_anti_uav/preregistration/P1_presence_gate_VAL.json` (committed before any P1 computation). Cross-fitted over track_eval's VAL halves; sequence bootstrap B = 10000. No TEST access; VAL shares source videos with TRAIN.*

| run | frozen STE (cross-fitted) | gated | Δ [95 % CI] | wins / losses | gate chosen (half 0 / half 1) | false-alarm rate frozen → gated | visible miss rate frozen → gated |
|---|---:|---:|---|---|---|---|---|
| v3_n__causal_g4_cst_s0_e20 | 63.61 | 63.61 | +0.00 [+0.00, +0.00] | 0 / 0 | instant 0.0 / instant 0.0 | 27.6 → 27.6 % | 12.1 → 12.1 % |
| v3_n__causal_g4_cst_s1_e20 | 65.08 | 65.08 | +0.00 [+0.00, +0.00] | 0 / 0 | instant 0.0 / instant 0.0 | 23.3 → 23.3 % | 12.8 → 12.8 % |
| v3_n__causal_g4_cst_s2_e20 | 64.46 | 64.46 | +0.00 [+0.00, +0.00] | 0 / 0 | instant 0.0 / instant 0.0 | 20.8 → 20.8 % | 15.4 → 15.4 % |

**Verdict (registered rule): NO_GAIN** (3-run mean Δ +0.00; CI > 0 for 0 / 3 runs, < 0 for 0 / 3).
