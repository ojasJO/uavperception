# CST Anti-UAV — final report (AeroTrack under the benchmark's own protocol)

**Question.** Under the CST Anti-UAV benchmark's official protocol, does the AeroTrack stack (a causal temporal-window detector
plus a light state estimator) beat the best published CST tracker (35.92 mSA)? Is the result seed-robust, and is it caused by
the temporal window?

**Answer.** Yes, with paired, sequence-level evidence on the official 60 TEST sequences, scored by the unmodified official
toolkit:
- **Single registered pass, X1:** **60.08** mSA.
- **Seed mean, X2:** **60.31** (60.08 / 60.26 / 60.60).
- **Against the released CST-trained GlobalTrack (35.92):** +24.39 SA [+21.08, +27.80], winning **60 of 60 sequences**.
- **The temporal window is causal to the gain:** +11.51 SA [+8.36, +14.86] over the same network fed a single frame.
- **Not from absence reporting alone:** with every target-absent frame scored 0 for every tracker, the lead is still
  +13.63 [+11.10, +16.14].
- **Weakness:** large targets (> 50 px), where the released template trackers remain better. CST train holds only 6 such frames.

## 1. Protocol (verified, `PAPER_PROTOCOL.md`, `PROTOCOL_COMPATIBILITY.md`)

| item | benchmark (paper arXiv 2507.23473 + official toolkit `PCwenyue/CST-Anti-UAV` @ bd92b984) | this study |
|---|---|---|
| data / split | 220 sequences: train 120 / val 40 / test 60, official lists | identical; complete release, sha256-manifested (`dataset_manifest.json`); no name overlap between splits |
| protocol | one-pass evaluation from the frame-0 ground-truth box (frame 0 is visible in all 220 sequences) | identical: STE reset from the frame-0 box, no later ground truth |
| primary metric | state accuracy, Eq. 1: per frame IoU if the target is visible, 1 / 0 for a correct / incorrect "absent" when it is not; mean over frames, then over the 60 sequences (mSA) | the official toolkit itself (`official_eval.py` calls it unmodified); Track A's re-implementation agrees per sequence to \|Δ\| = 0 |
| training data | the published trackers: "official implementations, pre-trained models and default configurations", retrained on CST train (§4.1) | detector trained on **CST train only**, from COCO `yolo11n.pt`; tracker settings tuned on CST VAL |
| strongest published result | **GlobalTrack 35.92** (Table 3; abstract: "the state-of-the-art method achieves only 35.92 % state accuracy"); SiamDT 35.84 | reproduced exactly from the authors' released predictions (all 21 released trackers match Table 3 to rounding) |

The paper is a benchmark paper and proposes no tracker of its own. No later CST result was found: a search on 2026-10-12 turned up
only the benchmark paper itself, and UAUTrack (arXiv 2512.02668) evaluates Anti-UAV, Anti-UAV410 and DUT Anti-UAV, not CST.
**35.92 is the strongest relevant published result.** The 67.69 figure is SiamDT on Anti-UAV410, a different benchmark, and is
never compared here.

**Comparability caveats (all disclosed, none favour us silently):**
1. Our comparison is against the authors' **released per-frame predictions**, scored by the same toolkit. Their checkpoints
   were not released.
2. Both sides use external pre-training. The published trackers use their official pre-trained models; our detector uses
   COCO-pretrained YOLO11n.
3. Our tracker settings were tuned on CST VAL. The paper uses default configurations.
4. **Absence reporting.** Eq. 1 credits a correct "absent". None of the 21 released trackers ever outputs "absent", so they score
   0 on all 12,925 absent TEST frames (17.9 %). Our STE does report absence. We therefore always show the absence-neutral
   variant (§3), where our lead remains about 13 SA.
5. Two official frame-level script quirks are reproduced but not used for ranking (`PAPER_PROTOCOL.md`):
   - OC / OV score only the last event window;
   - the paper's OV 65.77 is not reproducible (79.74 from the released predictions).

## 2. Results

### 2a. Official-protocol comparison: CST TEST, 60 sequences, official toolkit

| method | protocol | official mSA [95 % CI] | success AUC | P@20px | tiny SA | seeds | status |
|---|---|---|---:|---:|---:|---:|---|
| GlobalTrack (released, CST-trained) | official | **35.92** (paper 35.92) | 35.38 | 58.72 | 15.83 | — | published best, reproduced |
| SiamDT (released, CST-trained) | official | 35.84 | 35.28 | 57.96 | 14.98 | — | reproduced |
| PrDiMP (released, CST-trained) | official | 29.76 | 29.28 | 42.55 | 28.21 | — | best published tiny-target SA |
| **AeroTrack-X1** (causal_g4 s0, STE) | official, CST train only | **60.08** [56.13, 63.97] | 58.53 | 77.42 | 39.88 | 1 | single registered pass (PBS 93019) |
| AeroTrack-X1 argmax | official | 59.70 [55.80, 63.48] | 58.16 | 77.08 | 40.43 | 1 | same pass |
| **AeroTrack-X2 causal_g4** (STE), seeds 0 / 1 / 2 | official, CST train only | 60.08 / 60.26 / **60.60**; mean **60.31** (sd 0.26) | 58.53 / 58.72 / 59.03 | 77.42 / 77.61 / 77.79 | 39.88 / 40.89 / 42.45 | 3 | single registered pass (PBS 93029) |
| AeroTrack-X2 single-frame control (STE) | official, CST train only | 46.83 / 49.81 / 49.79; mean 48.81 (sd 1.72) | 45.58 / 48.49 / 48.46 | 58.60 / 62.33 / 61.94 | 31.26 / 31.03 / 31.38 | 3 | same pass |

The full table of all 21 released trackers is in `BASELINE_RESULTS.md`.

### 2b. AeroTrack system experiments: not protocol-matched, never compared with Table 3

| system | difference from the protocol | CST TEST STE mSA | source |
|---|---|---:|---|
| `v2_causal_g4_s0_e20` (all-domain detector) | detector trained on CST + Anti-UAV + Det-Fly | 56.55 | `experiments/aerotrack_v2/hpc/reports/final_test.json` |
| CF complete causal system (Track C) | different registration; Anti-UAV + CST groups, forecasting | see `experiments/aerotrack_temporal/` | — |

## 3. Statistics: sequence-level, paired (the statistical unit is the sequence, never the frame)

Method: sequence bootstrap, B = 10000, `numpy.default_rng(0)`, percentile 95 % CI; two-sided Wilcoxon signed-rank alongside
(zero differences dropped). Rules were registered before each pass (X1 + amendment X1-A1; X2).

| comparison | Δ mSA [95 % CI] | wins / losses | Wilcoxon p | registered verdict |
|---|---|---|---|---|
| X1 STE − GlobalTrack (X1-a2) | +24.16 [+20.78, +27.63] | 59 / 1 | 1.8e-11 | exceeds (paired) |
| X1 STE vs 35.92 (X1-a, unpaired CI) | 60.08 [56.13, 63.97] | — | — | exceeds |
| X2 causal_g4 seed mean − GlobalTrack (T2) | **+24.39 [+21.08, +27.80]** | **60 / 0** | — | exceeds (paired) |
| X2 every causal_g4 seed vs 35.92 (T1) | lower CI bounds 56.13 / 56.29 / 56.66 | — | — | **seed-robust** |
| X2 causal_g4 − single frame, seed means (T3) | **+11.51 [+8.36, +14.86]** | 45 / 15 | 1.2e-07 | **SUPPORTED** |
| X2 causal_g4 seed mean − X1 (descriptive) | +0.23 [−0.66, +1.14] | 32 / 28 | 0.71 | no difference: X2 confirms, does not improve |
| absence-neutral: X1 − GlobalTrack (descriptive) | +13.00 [+10.47, +15.52] | 56 / 4 | 1.2e-10 | — |
| absence-neutral: causal_g4 seed mean − GlobalTrack (descriptive) | +13.63 [+11.10, +16.14] | 56 / 4 | — | — |
| visible-frame IoU: X1 − GlobalTrack (descriptive) | +14.88 [+11.93, +17.78] | 56 / 4 | — | — |

Checks:
- Official toolkit equals each pass's own per-sequence SA: \|Δ\| = 0 for X1 (2 outputs) and X2 (12 outputs).
- Every set of weights is in its TEST ledger exactly once (`experiments/aerotrack_v3/reports/x1_test_ledger.jsonl`,
  `x2_test_ledger.jsonl`, `v3_test_ledger.jsonl`).

**Does the evidence support a paper-beating claim?**

**Yes**, for this precise statement:

> Under the official CST protocol (same 60 TEST sequences, official toolkit, one-pass evaluation from the frame-0 box, CST-only
> training of the learned component), the AeroTrack causal-window detector + STE scores 60.1–60.6 mSA over three training seeds.
> That is +24.4 SA above the best published CST tracker, paired over all 60 sequences.

What the claim does **not** cover:
- It is not a claim about the published trackers' architectures retrained by us.
- About 11 of the 24 points come from correctly reporting absence, which Eq. 1 rewards and the released trackers never attempt.
- It is not a claim about large targets (§4).

## 4. Attributes and failures (`ATTRIBUTE_ANALYSIS.md`, `FAILURE_ANALYSIS.md`)

- **Gains** on thermal crossover, scale variation, dynamic background clutter and complex dynamic background (14–48 SA over
  GlobalTrack), and on tiny targets: 39.9–42.5 vs GlobalTrack 15.8 and the best published PrDiMP 28.2.
- **Large targets (> 50 px, 484 frames in 8 sequences): the configuration is weaker** than the template trackers, 23.4–27.1 vs
  GlobalTrack 34.5 and SiamDT 38.4. The CIs are extremely wide.
  - Cause: a training-distribution gap. CST train has **6** large-target frames; VAL has 347, 337 of them in one sequence.
  - X2 does not change it.
- **`jungle_18`** (X1's only loss to GlobalTrack): the visible-frame distractor lock was **seed-specific**. Seeds 1 and 2 score
  44.6 and 39.8 vs X1 22.8 and GlobalTrack 28.9. Absent-frame false alarms (92–100 %) persist for every seed and every tracker.
- **Failure modes, ranked** (every seed):
  1. misses on visible frames (11–14 %);
  2. wrong-object boxes (about 10 %; about 24 % without the window);
  3. false alarms on absent frames (29–38 %).
- **What the window does:** it mainly cuts wrong-object boxes (about 24 % → about 10 %) and speeds re-acquisition (median 31–34
  → 10–13 frames; `ABLATION_RESULTS.md` §3).

## 5. Speed and cost

| stage | measurement | value |
|---|---|---|
| tracker (STE) | from cached detections, HPC CPU, per frame | 0.126–0.145 ms (argmax 0.012–0.017 ms) — `x2_final_test.json`, `x1_final_test.json` |
| detector (RTX 5050 laptop, TensorRT fp16, batch 1, 640) | deployment gate of the **same architecture** (`v2_causal_g4_s0_e20` weights; X1/X2 weights not separately gated) | model p50 / p99 3.20 / 4.66 ms (327 FPS); end-to-end loop (decode → letterbox → window → inference → NMS) p50 / p99 5.43 / 7.99 ms — `experiments/deployment/reports/gate_v2_causal_g4_s0_e20.md` |
| look-ahead | causal window [t−8, t−4, t] | 0 frames |
| training | one run, V100, 20 epochs | 0.72–0.79 h |
| TEST detection cache (V100, fp16, CST TEST 72,175 frames) | including JPEG decode | about 150 frames/s |

For context only (our measurements of the published trackers are not available): the paper does not report speed for its CST
table.

## 6. What was not done, and why: research continuation decision

The registrations allow no TEST-driven selection, so the next step must be justified on VAL. The remaining weaknesses and their
VAL measurability:

| weakness (TEST evidence) | candidate remedy | can VAL select it? | decision |
|---|---|---|---|
| large targets (23–27 vs 34.5) | scale augmentation / size-balanced sampling | **no**: 347 VAL large frames, 337 in one sequence; CST train has 6 | not trained. A defensible test needs a held-out large-target set (e.g. a registered split of large-target sequences) before any TEST access |
| absent-frame false alarms (29–38 %) | a VAL-tuned absence / confidence model on the STE output | yes: 30 of the 40 VAL sequences contain absent frames (6,250 frames), measurable cross-fitted | **best-justified next experiment**: register it, then cross-fit on VAL halves. It would need its own registered TEST pass, not authorised here |
| wrong-object boxes (about 10 %) | appearance re-identification term in the STE | partly: VAL distractor events are few; descriptive only so far | candidate, lower priority |
| seed variance on hard sequences (`jungle_18`) | seed ensembling of the detector | VAL can measure it (3 seeds exist) | cost 3× inference; would need registration + a new TEST pass |

No further TEST pass was run, because none is authorised by an existing registration. No exploratory training was launched: the
two remedies VAL could validate are tracker-level and cheap, but each still needs a registration and a separate TEST budget. The
large-target remedy cannot be validated with the available splits.

## 7. Limitations

- VAL shares source videos with TRAIN (paper §3.3), so every VAL number is optimistic. TEST is independent.
- One network family and size (YOLO11n). The STE is a constant-velocity Kalman tracker with no appearance model.
- The published trackers are represented by their released predictions only. ARTrackV2 (Table 3: 22.14) is missing from the
  archive.
- The speed numbers mix two devices (V100 for training and caching, RTX 5050 for deployment). The detector latency is that of
  the same architecture with different weights.
- The X2 prediction files (31 MB) are git-ignored (`results/runs/cst_anti_uav/x2_test_predictions/`, also on the HPC). Their per-file
  sha256 values are recorded in `X2_test_official/official_scores.json`. X1's predictions are committed
  (`results/runs/aerotrack_v3/x1_test_predictions/`).

## 8. Reproduction

```
venv\Scripts\python.exe experiments\cst_anti_uav\scripts\official_eval.py --published --subset test --out <dir>   # Table 3
venv\Scripts\python.exe experiments\cst_anti_uav\scripts\analyze_test.py --x1     # X1 official scoring + paired tests
venv\Scripts\python.exe experiments\cst_anti_uav\scripts\analyze_test.py --x2     # X2 T1-T3
venv\Scripts\python.exe experiments\cst_anti_uav\scripts\failure_analysis.py --x1 / --x2
venv\Scripts\python.exe -m unittest tests.test_cst_protocol                     # metric / format tests (7)
```

| artifact | identifier |
|---|---|
| X1 pass | PBS 93019, commit f9bbd3a, weights `e52a5645…`, frozen `experiments/aerotrack_v3/frozen_x1_params.json` (`c534211b…`) |
| X2 pass | PBS 93029, commit 3ffee11, frozen `experiments/cst_anti_uav/frozen_x2_params.json` (git blob `b0c3bba`), 5 weights sha256 in `x2_test_ledger.jsonl` |
| training | PBS 92973 (X1), 93016 + 93018 (X2), VAL design 93020 |
| released predictions | `TrainedCST.zip` sha256 `1c886465…` |
