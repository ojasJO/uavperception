# Track C — synthesis of C1 / C1c / C3 / C1b (VAL) and the CF final-system TEST pass

Track C's own registered experiments on `data_unified_v2`: §1–4 on **VAL**, §5 the single CF TEST pass:
- Anti-UAV IR / RGB: 67 sequences each.
- CST: 40 sequences.

The exact values, with each source report's sha256, are in `TRACK_C_SUMMARY.json` (`scripts/tc_summary.py`, which reads the
reports and recomputes nothing).

**Track C is not the CST benchmark.** Its forecasting metrics (ADE / FDE in pixels, Gaussian NLL) and its tracking mSA (Track A's
code, cross-fitted over two VAL halves, all-domain v2 detectors) are never comparable to the CST Anti-UAV state accuracy of X1 / X2
(`experiments/cst_anti_uav/`).

Sections 1–4 are VAL only. The single exception is §5, the registered CF final-system TEST pass, which ran once after its
settings were frozen on VAL.

Statistics are as recorded by each producing script:
- C1-family: paired sequence bootstrap per seed, three seeds;
- C3: sequence bootstrap B = 2000, verdict SUPPORTED if Δ > 0.5 SA and CI lower bound > 0.

## 1. Learned trajectory forecasting on oracle (ground-truth) histories

C1 is the pooled model; C1c adds the camera-mode-aware arms. Each cell shows the verdict and the per-seed paired difference
[95 % CI]: ADE px for point error, NLL for uncertainty. Negative is better for the learned model.

| domain | C1 H1 pooled vs classical (ADE) | C1 H2 pooled vs Kalman (NLL) | C1c arm (dev) | C1c H1c vs classical (ADE) | C1c H2c vs pooled C1 (ADE) | C1c H3c uncertainty |
|---|---|---|---|---|---|---|
| Anti-UAV IR (ref. `oudcv`) | **SUPPORTED**: −3.54 [−4.31, −2.80], −2.94, −3.54 | NOT_SUPPORTED | perdom | **SUPPORTED**: −2.53 [−3.26, −1.81], −3.67, −2.84 | NOT_SUPPORTED: +1.01, −0.73, +0.70 | NOT_SUPPORTED |
| Anti-UAV RGB (ref. `oudcv`) | **SUPPORTED**: −8.55 [−10.33, −6.82], −6.74, −8.62 | **SUPPORTED** | domtok | **SUPPORTED**: −9.68 [−11.85, −7.64], −9.08, −9.28 | NOT_SUPPORTED: −1.13, −2.33, −0.66 | NOT_SUPPORTED |
| CST (ref. `dcv`) | **LOSES**: +5.35 [+3.73, +7.01], +5.45, +5.81 | **SUPPORTED** | perdom | NOT_SUPPORTED: −0.14 [−0.84, +0.52], −0.42 [−1.10, +0.23], −0.02 [−0.75, +0.67] | **SUPPORTED**: −5.49 [−6.76, −4.29], −5.86, −5.83 | **SUPPORTED** |

What this shows:
- On CST, the pooled learned forecaster suffers **negative transfer**: about +5.5 px worse than the classical `dcv`. Per-domain
  training (C1c `perdom`) **removes the negative transfer**. ADE falls from about 17.6 px pooled to about 11.9 px, which beats the
  pooled model by about 5.7 px.
- However, it **does not beat the classical forecaster**: 11.7–12.1 vs `dcv` 12.1 px, with every CI spanning 0. "Negative transfer
  resolved" is accurate; "learned beats classical on CST" is not supported.
- On Anti-UAV, the learned forecaster beats the classical reference with either arm. The domain-aware arm adds nothing over the
  pooled one (H2c not supported).

**Adoption (registered rule, `scripts/tc_adoption.py`).** A domain gets a learned forecaster downstream only if H1 / H1c is
SUPPORTED. Hence:
- Anti-UAV IR and RGB use **C1 pooled** (`c1_tfm_abs_s*`).
- **CST uses none**. C1c's CST model fails H1c, and C1's H1 on CST is LOSES.

This is why C3's `fste_learned` and C1b's CST rows below carry no C1c model.

## 2. C3 — tracking with motion models (SOT protocol, VAL, cross-fitted)

### Primary cache `v2_causal_g4_s0_e20`

Run as stage 3a of PBS 93015; report generated 2026-10-12T04:57.

| hypothesis | Anti-UAV IR | Anti-UAV RGB | CST |
|---|---|---|---|
| H1 temporal state (STE − argmax) | +0.07 [−0.06, +0.25] NO_DIFFERENCE | −0.18 [−0.57, +0.04] NO_DIFFERENCE | **+0.84 [+0.24, +1.61] SUPPORTED** (25 / 13) |
| H2 domain motion model (fste_classical − STE) | +0.00 [−0.18, +0.12] NO_DIFFERENCE | +0.03 [−0.35, +0.54] NO_DIFFERENCE | +0.02 [−0.27, +0.29] NO_DIFFERENCE |
| H3 learned motion model (fste_learned − fste_classical) | −0.10 [−0.34, +0.06] NO_DIFFERENCE | +0.19 [−0.21, +0.65] NO_DIFFERENCE | NOT RUN (no learned forecaster adopted) |

mSA for STE: Anti-UAV IR 80.53, Anti-UAV RGB 81.20, CST 63.02 (40 VAL sequences).

The Kalman state estimator helps only on CST. There targets are small and detections intermittent, so coasting and gating
matter. On Anti-UAV's gimbal footage, per-frame argmax is already near the ceiling. Neither a domain-specific nor a learned
motion model improves tracking anywhere.

### Secondary cache `v2_stock_s0_e20`

Single-frame detector; stage 3b of PBS 93021; report generated 2026-10-12T09:44:55.

| hypothesis | Anti-UAV IR | Anti-UAV RGB | CST |
|---|---|---|---|
| H1 temporal state (STE − argmax) | +0.14 [+0.01, +0.35] NO_DIFFERENCE (Δ below the registered 0.5 threshold) | −0.05 [−0.12, +0.02] NO_DIFFERENCE | +0.17 [−0.73, +1.10] NO_DIFFERENCE |
| H2 domain motion model | −0.08 [−0.40, +0.19] NO_DIFFERENCE | −0.04 [−0.20, +0.15] NO_DIFFERENCE | +0.11 [−0.27, +0.58] NO_DIFFERENCE |
| H3 learned motion model | −0.08 [−0.52, +0.35] NO_DIFFERENCE | +0.15 [−0.02, +0.34] NO_DIFFERENCE | NOT RUN (no learned forecaster adopted) |

mSA for STE: Anti-UAV IR 80.43, Anti-UAV RGB 81.38, CST **52.76**. For comparison, argmax on CST scores 52.59, recovers in a
median of 56 frames and locates the target in 58.6 % of frames.

**Primary vs secondary cache.** These are the same tracker grid on the same 40 CST VAL sequences, with different detectors,
and the comparison is descriptive (not a registered contrast):
- With the temporal-window detector, CST STE mSA is 63.02. With the single-frame detector it is 52.76.
- H1 (Kalman state over argmax) is SUPPORTED only with the temporal detector. With single-frame detections the state estimator
  gains nothing measurable on CST.

On VAL tracking, then, the causal temporal window matters more for CST than any motion model does. All three motion-model
hypotheses (H1–H3) are NO_DIFFERENCE on both caches for Anti-UAV.

## 3. C1b — forecasting on the tracker's own (noisy) histories

### Primary cache

Stage 4a, run by PBS 93021; report generated 2026-10-12T07:27. Histories are STE outputs at C3's cross-fitted settings; targets
are ground truth. Learned arm per the adoption rule: C1 pooled everywhere, since CST adopts none and falls back to C1 pooled.

| domain | sequences / anchors | H1b point (learned − classical ADE) | H2b uncertainty (learned − Kalman NLL) | H4 augmentation (aug − abs ADE) |
|---|---|---|---|---|
| Anti-UAV IR (ref. `oudcv`) | 67 / 59,939 | **SUPPORTED**: −3.44 [−4.20, −2.72], −2.84, −3.41 | **SUPPORTED**: −1.33 [−1.63, −1.02], −1.21, −1.23 | REVERSED (augmentation hurts): +0.64, +2.73, +0.82 |
| Anti-UAV RGB (ref. `oudcv`) | 67 / 57,498 | **SUPPORTED**: −8.03 [−9.88, −6.34], −6.33, −8.00 | NOT_SUPPORTED: −0.69 [−1.49, +0.36], −0.98, −1.02 | REVERSED: +1.37, +6.90, +1.99 |
| CST (ref. `dcv`) | 40 / 32,251 | NOT_SUPPORTED: −2.10 [−6.11, +1.54], −2.01, −1.64 | **LOSES**: +745.9 [+274.7, +1348.0], +479.3, +471.5 | NOT_SUPPORTED (mixed signs) |

The producing script's markdown labels the augmentation verdict "HARMFUL"; its JSON verdict is `REVERSED`. Both mean
augmentation made ADE worse.

Absolute ADE / FDE@24:
- Anti-UAV IR: learned 34.1 / 48.6 px vs `oudcv` 37.5 / 52.1;
- Anti-UAV RGB: 90.3 / 128.8 vs 98.4 / 134.7;
- CST: pooled 30.4–30.9 / 43.2–43.5 vs `dcv` 32.5 / 48.5.

What this shows:
- The Anti-UAV point-forecast gain from oracle histories (C1) **survives the move to real tracker histories**: −2.8 to −8.0 px,
  every CI excluding 0.
- On CST, the pooled model's **uncertainty is badly miscalibrated** on tracker histories: NLL 526–800 vs Kalman 54. This is a
  documented regression, and a learned CST forecaster must not be deployed for uncertainty.

### Secondary cache `v2_stock_s0_e20`

Stage 4b of PBS 93021, 09:51–09:54. Histories are STE outputs on the single-frame cache.

| domain | sequences / anchors | H1b point (ADE) | H2b uncertainty (NLL) | H4 augmentation |
|---|---|---|---|---|
| Anti-UAV IR (ref. `oudcv`) | 67 / 59,932 | **SUPPORTED**: −3.43 [−4.19, −2.70], −2.85, −3.40 | **SUPPORTED**: −1.34 [−1.63, −1.04], −1.22, −1.24 | REVERSED: +0.62, +2.71, +0.80 |
| Anti-UAV RGB (ref. `oudcv`) | 67 / 57,719 | **SUPPORTED**: −8.00 [−9.83, −6.31], −6.34, −7.97 | **SUPPORTED**: −1.12 [−1.60, −0.58], −1.04 [−1.51, −0.51], −0.97 [−1.58, −0.27] | REVERSED: +1.32, +6.88, +1.90 |
| CST (ref. `dcv`) | 40 / 28,762 | NOT_SUPPORTED: −5.65 [−12.03, +0.14], −5.84 [−12.24, −0.17], −5.42 [−11.92, +0.35] | **LOSES**: +1312 [+491, +2391], +731, +706 | NOT_SUPPORTED (mixed signs) |

Absolute ADE / FDE@24 on CST: pooled learned 49.5–49.9 / 59.2–59.5 px, `dcv` 55.3 / 71.2, Kalman 56.4 / 75.1. Learned NLL is
789–1395 against Kalman's 83.1.

**Both caches together.**
- Anti-UAV: H1b and H4 replicate exactly in direction and size, so the learned point forecaster beats `oudcv` on tracker
  histories with either detector.
- Anti-UAV RGB H2b differs between caches: NOT_SUPPORTED on the temporal cache (one seed's CI crosses 0), SUPPORTED on the
  single-frame cache. Treat RGB uncertainty gains as **not robust**.
- CST: the point forecast is directionally better but never meets the registered rule on either cache (CIs reach 0). The
  uncertainty regression is large on both. The secondary cache's CST histories are noisier (STE mSA 52.76 vs 63.02), so every
  CST forecaster has higher ADE there.

## 4. Conclusions (VAL; Track C only)

1. **Learned point forecasting helps on gimbal footage (Anti-UAV), not on CST.** This holds with oracle histories (C1, C1c) and
   with real tracker histories on both detector caches (C1b).
2. **Domain-aware training (C1c) removes CST's negative transfer** (−5.5 to −5.9 px vs pooled) **but does not beat the
   classical `dcv`** (H1c NOT_SUPPORTED). Under the registered adoption rule, CST therefore uses no learned forecaster
   downstream.
3. **Learned uncertainty on CST tracker histories is badly miscalibrated** (NLL 10–17× Kalman's) on both caches. This is a
   documented regression; do not use learned CST uncertainty.
4. **Motion models do not improve tracking** (C3 H2 and H3 NO_DIFFERENCE everywhere). The Kalman state helps only on CST
   with the temporal-window detector (H1 +0.84 [+0.24, +1.61]).
5. **Augmentation of the learned forecaster hurts** on Anti-UAV tracker histories (REVERSED on both caches).

**Limitations.**
- VAL only. CST VAL shares source videos with CST TRAIN, so CST numbers are optimistic.
- One detector seed per cache.
- The C3 bootstrap uses B = 2000.
- 93015 and 93021 exceeded their memory requests (82 GB / 32 GB and 154 GB / 64 GB). The results are unaffected, but future
  submissions should request ≥ 96 GB / ≥ 192 GB respectively.

## 5. CF: the complete causal system, the single registered TEST pass

Sources:
- Report: `CF_final_causal_system.{json,md}`.
- Registration: `preregistration/CF_final_causal_system.json` + amendment **CF-A2** (77839f0), committed before any CF TEST access.
- Frozen settings: `frozen_cf.json` (bfea36b). Produced on VAL only by PBS 93030 (node04, CPU, exit 0) from the committed, clean
  `scripts/tc_cf.py` @ 6b9415a.
- Pass: run once on the OMEN CPU, 188 s, `cf_test_ledger.jsonl`. A second pass is refused.
- Before the pass, the script was checked by a smoke freeze and a dry run of the whole test path on VAL (no production output).

**Components, all fixed by the registered rules before the pass:**
- **Detector** `v2_causal_g4_s0_e20`: C2 selected causal_g4; F1 NOT SUPPORTED → as-built network.
- **Trackers:**
  - Anti-UAV IR / RGB: **argmax** (thr 0.3). C3 H1 was not SUPPORTED, and the `fste_classical`-vs-argmax clause gave
    NO_DIFFERENCE: +0.07 [−0.04, +0.25] / −0.16 [−0.40, +0.04].
  - CST: **ste_cv** (C3 H1 SUPPORTED), re-tuned on all CST VAL over 504 settings.
- **Forecasters:**
  - Anti-UAV: learned `c1_tfm_abs_s0` (C1b H1b SUPPORTED, H2b not LOSES, H4 REVERSED → clean variant).
  - CST: classical `dcv` with TRAIN-residual variance.

**Detections** are the verified v2 TEST cache: weights `1118cf32…`, set `7a867239…`.

**Cross-check:** CF's SOT numbers equal the earlier v2 final pass to four decimals in every group:

| | IR argmax | RGB argmax | CST argmax | CST STE |
|---|---:|---:|---:|---:|
| value | 68.8919 | 74.1046 | 56.2846 | 56.5541 |

The re-tuned CST STE setting coincides with v2's frozen one. Two independent code paths agree.

| group (TEST) | tracker | sequences | mSA | located | false alarms | recover median (frames) | CF-c: tracker − argmax at its frozen threshold |
|---|---|---:|---:|---:|---:|---:|---|
| Anti-UAV IR | argmax | 91 | 68.89 | 87.0 % | 21.6 % | 0 | identical by construction |
| Anti-UAV RGB | argmax | 91 | 74.10 | 91.5 % | 11.3 % | 3 | identical by construction |
| CST | ste_cv | 60 | 56.55 | 67.1 % | 33.1 % | 20 | +0.27 [−0.27, +0.79], 32 / 26 → no detectable difference |

- **CF-a** (Anti-UAV IR vs our faithful GlobalTrack, Track A `normal_ir`, 70.58; same 91 sequences): **−1.69 SA [−3.25, +0.06]**,
  28 / 63 → **no detectable difference**. Caveat: the training data differ.
- **CF-c:** on CST TEST the Kalman state's VAL advantage (C3 H1 +0.84) does not reach significance (+0.27 [−0.27, +0.79]).
  Under the registered rule, the simpler argmax is deployed on Anti-UAV. Descriptively, the v2 pass's STE was slightly better
  there on TEST (+0.22 / +0.24); that is not used.
- **Falsifier** ("CF-a worse and CF-c no detectable difference"): **not triggered**, because CF-a is not "worse".
- **CF-d:** not applicable. The v2 candidate is causal.

**MOT protocol (no initialisation box):**

| group | HOTA | AssA | IDF1 | MOTA |
|---|---:|---:|---:|---:|
| Anti-UAV IR | 0.675 | 0.711 | 0.896 | 0.820 |
| Anti-UAV RGB | 0.717 | 0.740 | 0.916 | 0.866 |
| CST | 0.479 | 0.537 | 0.694 | 0.438 |

**CF-b, forecasting on the CF tracker's own TEST histories** (C1 rules, paired cluster bootstrap, three seeds of the adopted family):

| domain | anchors | ADE px: adopted s0 / Kalman / constant position | point vs Kalman (per seed Δ [CI]) | point vs constant position | NLL vs Kalman |
|---|---:|---|---|---|---|
| Anti-UAV IR | 75,530 | 38.9 / 43.6 / 43.6 | **SUPPORTED**: −4.65 [−5.30, −3.98], −3.81, −4.57 | **SUPPORTED** (−4.64, −3.80, −4.56) | NOT_SUPPORTED (−0.12 [−0.94, +0.93], −0.77, −0.43) |
| Anti-UAV RGB | 75,081 | 94.7 / 106.4 / 106.4 | **SUPPORTED**: −11.66 [−13.42, −9.81], −9.88, −11.74 | **SUPPORTED** | NOT_SUPPORTED (−0.54 [−1.39, +0.68], −1.15, −0.96) |
| CST (classical `dcv`) | 47,679 | 49.9 / 51.2 / **47.5** | NOT_SUPPORTED: −1.26 [−1.93, −0.64] (−2.5 %, below the 5 % margin) | NOT_SUPPORTED: +2.39 [−0.80, +5.92] | **LOSES**: +69.7 [+22.1, +119.9] |

- The learned Anti-UAV forecaster's point gain from VAL (C1b) **holds on TEST**, at 8.7–11 % lower ADE. Its uncertainty gain
  does not hold.
- On CST no forecaster beats constant position. The deployed `dcv` uncertainty is worse calibrated than Kalman's.

**Deployment** (RTX 5050 laptop, VAL inputs, `experiments/deployment/reports/stack_gate_v2_causal_g4_s0_e20_ste.{json,md}`):
- Setup: TensorRT fp16 detector, causal window, look-ahead 0, STE at CF's CST settings, 6 VAL sequences × 300 frames.
- Per-frame p50 / p99:
  - inference 2.62 / 4.17 ms;
  - tracker 0.29 / 0.56 ms;
  - end to end 5.89 / 10.76 ms pooled: IR 5.09 / 7.29, CST 4.95 / 6.99, RGB 7.86 / 10.98 (1920×1080 decode).
- Learned forecaster `c1_tfm_abs_s0`, ONNX Runtime CPU, one thread, batch 1: 0.11 / 0.17 ms per call
  (`forecaster_latency_c1_tfm_abs_s0.json`).
- The runtime has no argmax + learned-forecaster mode. CF's Anti-UAV stack is therefore bounded by the measured STE stack
  (STE costs more than argmax) plus about 0.2 ms, which is far inside 40 ms per frame at 25 fps.
- Two silent fallbacks in the runtime were found and are now refused by `stack_gate.py`. A mislabelled report was deleted
  before use.
  - When no engine exists for the weights' run name, it fell back to torch fp32.
  - When `run_meta.json` is missing, it defaulted to a symmetric window with 1 frame of look-ahead.

**CF is not the CST benchmark.** CF's CST group uses the all-domain v2 detector. Its 56.55 is a system result and is never
compared with the paper's Table 3 (`experiments/cst_anti_uav/reports/FINAL_REPORT.md` §2b).

## 4. Execution record

| stage | what | PBS | status |
|---|---|---|---|
| 1 | C1c training + dev selection (12 forecasters) | 93015 (node04) | DONE 02:15 |
| 2 | C1c VAL pass | 93015 | DONE (report 02:15:03) |
| 3a | C3 primary cache | 93015 | DONE (report 04:57:18) |
| 3b | C3 secondary cache | 93015 → **93021** | 93015 hit its 03:30:00 walltime at the start of 3b (Exit_status 271, no 3b output); re-run in 93021 (node01) from 07:27 |
| 4a | C1b primary cache | 93021 | DONE 07:27:04 |
| 4b | C1b secondary cache | 93021 | after 3b |

`scripts/run_track_c_part2.sh` (commit b58610e) runs 4a before 3b and gives 3b 8 workers. This reorder is dependency-safe:
`tc_c1b.py` needs the C3 report of the same cache. The worker change does not alter results (`Pool.map` keeps order and
scoring is deterministic). It does raise memory: about 154 GB used against a 64 GB request on node01, which has 256 GB and was
the only PBS job there.
