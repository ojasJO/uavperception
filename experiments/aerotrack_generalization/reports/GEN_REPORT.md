# AeroTrack generalization investigation — report

*Generated 2026-09-28 16:36 by `scripts/report_gen.py` from `reports/*.json` and `ledger.jsonl`. Labels: [fact] read from files · [measured] computed in this series · [rule] outcome of the pre-registered rule · [inference] interpretation.*

## 1. Original problem

[fact] The repair grid (`experiments/aerotrack_repair/reports/ablation_grid_e20.md`) found no factor credited: in all five pre-registered comparisons the ablated model scored higher on validation, and stock YOLO11n beat the full AeroTrack recipe. The question here: is there a change with a plausible mechanism that improves generalisation, verified without tuning to the validation or test sets?

## 2–4. Evidence, diagnosis, confounds

- [measured] COCO-initialised parameters: SPD layout 48.4 % of 4.085 M, stride layout 93.7 % of 2.63 M (D2).
- [measured] function-preserving init: SPD vs stride max |Δ| 2.38e-03 on outputs up to 639; centre-frame stem ignores t±1 (max |Δ| 0.0e+00); the grid's tiled stem responds to the frame mean (max |Δ| 404) (D2).
- [fact] grid training loss epoch 0 → 19: baseline_ctrl 6.932 → 2.261; no_spd 5.746 → 2.143; single_frame 6.536 → 2.279; stock_yolo11n 5.68 → 2.156 (D2).
- [measured] Anti-UAV VAL shares 26 of 26 recording sessions with TRAIN; TEST shares 0 (D1).
- [measured] CST TEST scene categories absent from TRAIN: urban-areas, water (D1).
- [measured] Det-Fly VAL frames whose adjacent frame (id ±1) is in TRAIN: 1224 of 1469 (D1).
- [measured] median target motion per frame at the network input: CST 0.53 px (68 % < 1 px), Anti-UAV IR 3.64 px, RGB 3.0 px (D1).
- [measured] grid VAL comparisons with sequence-bootstrap 95 % CIs (sampling noise only): SPD-Conv micro_AP50 best -0.016 [-0.045, +0.005]; SPD-Conv micro_AP50 last -0.078 [-0.114, -0.050]; SPD-Conv cst_micro_AP50 best -0.039 [-0.104, +0.009]; SPD-Conv cst_micro_AP50 last -0.102 [-0.155, -0.067]; temporal stack cst_AP50 best -0.045 [-0.080, -0.005]; temporal stack cst_AP50 last -0.015 [-0.034, +0.008]; temporal stack drone_AP50 best -0.027 [-0.041, -0.013]; temporal stack drone_AP50 last -0.003 [-0.012, +0.006]; full recipe vs stock YOLO11n drone_AP50 best -0.055 [-0.068, -0.040]; full recipe vs stock YOLO11n drone_AP50 last -0.035 [-0.044, -0.025]; [exploratory] temporal within the stride layout cst_AP50 best -0.021 [-0.040, -0.005]; [exploratory] temporal within the stride layout cst_AP50 last +0.029 [+0.008, +0.053] (D4).
- [measured] CST micro drones at last.pt, found-but-mislocalised vs missed: baseline_ctrl 0.070 / 0.210; no_spd 0.081 / 0.134; single_frame 0.087 / 0.189; stock_yolo11n 0.065 / 0.162 (D5).

## 5–6. Experiments and exact configurations

[fact] Pre-registered plan (ledger P1, `scripts/plan_gen.py`): reference `stock`; candidates spd_fp, temporal_fp_g1, temporal_fp_g4; 20 epochs, recipe `src/model/train.py` STAGES['baseline'] (25,000 weighted samples/epoch, batch 32 × acc 2, AdamW 1e-3 cosine); function-preserving init for every configuration; seed = fold index; checkpoint last.pt (no selection); primary cst_drone_AP50 on the held-out DEV fold; non-inferiority anti_uav_drone_AP50; margin 0.02; rules: (1) fold-mean delta > +0.02; (2) delta > 0 in each of the 3 folds; (3) fold-pooled paired sequence bootstrap 95% CI of the mean delta excludes 0; (4) fold-mean Anti-UAV drone AP50 delta > -0.02; futility: a candidate with delta <= 0 on fold 0 is rejected (rule (2) cannot hold); folds 1-2 are run only for survivors.

| run | config | fold | seed | epochs | launches | training time | last.pt sha | init check |
|---|---|---|---|---|---|---|---|---|
| `gen_stock_f0_e20` | stock | 0 | 0 | 20 | 1 | 1.44 h | 711b43546223 | ok |
| `gen_spd_fp_f0_e20` | spd_fp | 0 | 0 | 20 | 1 | 1.59 h | 83d771065108 | ok |
| `gen_temporal_fp_g1_f0_e20` | temporal_fp_g1 | 0 | 0 | 20 | 1 | 1.94 h | cec8530415c6 | ok |
| `gen_temporal_fp_g4_f0_e20` | temporal_fp_g4 | 0 | 0 | 20 | 1 | 1.99 h | 4bdaf3f50fd1 | ok |
| `gen_stock_f1_e20` | stock | 1 | 1 | 20 | 1 | 1.59 h | 21ac66700e2b | ok |
| `gen_spd_fp_f1_e20` | spd_fp | 1 | 1 | 20 | 1 | 1.61 h | a5f883bd347d | ok |
| `gen_temporal_fp_g1_f1_e20` | temporal_fp_g1 | 1 | 1 | 20 | 1 | 1.96 h | 6644e48d6a24 | ok |
| `gen_temporal_fp_g4_f1_e20` | temporal_fp_g4 | 1 | 1 | 20 | 1 | 2.04 h | 1e2ca653aae1 | ok |
| `gen_stock_f2_e20` | stock | 2 | 2 | 20 | 1 | 1.58 h | e40b569c7344 | ok |
| `gen_spd_fp_f2_e20` | spd_fp | 2 | 2 | 20 | 1 | 1.66 h | e27497e356e6 | ok |
| `gen_temporal_fp_g1_f2_e20` | temporal_fp_g1 | 2 | 2 | 20 | 1 | 1.95 h | f918d081366d | ok |
| `gen_temporal_fp_g4_f2_e20` | temporal_fp_g4 | 2 | 2 | 20 | 1 | 2.03 h | 8e49e2421df2 | ok |
| `gen_temporal_fp_g4_full_e20` | temporal_fp_g4 | full | 0 | 20 | 1 | 2.21 h | d26641b2cedb | ok |

## 7–8. Development results (held-out DEV folds: unseen Anti-UAV sessions, unseen CST sequences, held-out Det-Fly flight chunks)

| run | set | CST drone AP50 (primary) | Anti-UAV drone AP50 | Det-Fly drone AP50 | CST drone-micro AP50 | pooled drone AP50 | images |
|---|---|---|---|---|---|---|---|
| `gen_spd_fp_f0_e20` | dev | 0.7415 | 0.9415 | 0.3494 | 0.7100 | 0.7877 | 11566 |
| `gen_spd_fp_f0_e20` | val | 0.7330 | 0.9721 | 0.4886 | 0.7103 | 0.8212 | 11565 |
| `gen_spd_fp_f1_e20` | dev | 0.7549 | 0.9472 | 0.4695 | 0.6813 | 0.8199 | 11566 |
| `gen_spd_fp_f1_e20` | val | 0.7270 | 0.9704 | 0.5027 | 0.6546 | 0.8224 | 11565 |
| `gen_spd_fp_f2_e20` | dev | 0.7223 | 0.9067 | 0.4555 | 0.6846 | 0.7843 | 11566 |
| `gen_spd_fp_f2_e20` | val | 0.7158 | 0.9600 | 0.5608 | 0.6841 | 0.8235 | 11565 |
| `gen_stock_f0_e20` | dev | 0.7264 | 0.9489 | 0.3249 | 0.6904 | 0.7906 | 11566 |
| `gen_stock_f0_e20` | val | 0.7382 | 0.9652 | 0.4678 | 0.7240 | 0.8231 | 11565 |
| `gen_stock_f1_e20` | dev | 0.7384 | 0.9429 | 0.4617 | 0.6400 | 0.8033 | 11566 |
| `gen_stock_f1_e20` | val | 0.6952 | 0.9703 | 0.4874 | 0.6136 | 0.8097 | 11565 |
| `gen_stock_f2_e20` | dev | 0.7124 | 0.9292 | 0.4282 | 0.6684 | 0.7904 | 11566 |
| `gen_stock_f2_e20` | val | 0.7363 | 0.9647 | 0.5388 | 0.7299 | 0.8319 | 11565 |
| `gen_temporal_fp_g1_f0_e20` | dev | 0.7367 | 0.9512 | 0.2506 | 0.6938 | 0.7719 | 11566 |
| `gen_temporal_fp_g1_f0_e20` | val | 0.7583 | 0.9767 | 0.3966 | 0.7199 | 0.8161 | 11565 |
| `gen_temporal_fp_g1_f1_e20` | dev | 0.7511 | 0.9356 | 0.4810 | 0.6790 | 0.8132 | 11566 |
| `gen_temporal_fp_g1_f1_e20` | val | 0.7208 | 0.9654 | 0.5024 | 0.6196 | 0.8193 | 11565 |
| `gen_temporal_fp_g1_f2_e20` | dev | 0.7159 | 0.9339 | 0.3833 | 0.6592 | 0.7879 | 11566 |
| `gen_temporal_fp_g1_f2_e20` | val | 0.7343 | 0.9672 | 0.4921 | 0.6866 | 0.8274 | 11565 |
| `gen_temporal_fp_g4_f0_e20` | dev | 0.7709 | 0.9446 | 0.3410 | 0.7286 | 0.8054 | 11566 |
| `gen_temporal_fp_g4_f0_e20` | val | 0.8323 | 0.9751 | 0.4671 | 0.7924 | 0.8619 | 11565 |
| `gen_temporal_fp_g4_f1_e20` | dev | 0.7746 | 0.9428 | 0.4899 | 0.7221 | 0.8286 | 11566 |
| `gen_temporal_fp_g4_f1_e20` | val | 0.7873 | 0.9687 | 0.5085 | 0.6767 | 0.8525 | 11565 |
| `gen_temporal_fp_g4_f2_e20` | dev | 0.7660 | 0.9292 | 0.3589 | 0.7008 | 0.8037 | 11566 |
| `gen_temporal_fp_g4_f2_e20` | val | 0.8059 | 0.9658 | 0.5171 | 0.7506 | 0.8573 | 11565 |
| `gen_temporal_fp_g4_full_e20` | test | 0.5209 | 0.9185 | 0.5382 | 0.4255 | 0.7183 | 11567 |
| `gen_temporal_fp_g4_full_e20` | val | 0.8106 | 0.9805 | 0.4897 | 0.7455 | 0.8655 | 11565 |

[measured] secondary, official VAL (optimistic set, see §2–4), candidate − stock per fold, CST drone AP50: spd_fp -0.005 / +0.032 / -0.021; temporal_fp_g1 +0.020 / +0.026 / -0.002; temporal_fp_g4 +0.094 / +0.092 / +0.070.

[measured] training curves (held-out-fold monitoring mAP50 and training loss, slope per epoch over the last 5 epochs): spd_fp_f0 +0.0069 / -0.0151; spd_fp_f1 +0.0049 / -0.0159; spd_fp_f2 +0.0046 / -0.0110; stock_f0 +0.0092 / -0.0169; stock_f1 +0.0056 / -0.0160; stock_f2 +0.0055 / -0.0108; temporal_fp_g1_f0 +0.0010 / -0.0155; temporal_fp_g1_f1 +0.0089 / -0.0161; temporal_fp_g1_f2 +0.0031 / -0.0130; temporal_fp_g4_f0 +0.0095 / -0.0149; temporal_fp_g4_f1 +0.0072 / -0.0179; temporal_fp_g4_f2 -0.0035 / -0.0136; temporal_fp_g4_full +0.0082 / -0.0154.

## 9. Decisions and statistical uncertainty

| candidate | vs | fold deltas (primary) | fold-mean Δ | sd | bootstrap 95 % CI | Anti-UAV mean Δ | [rule] status |
|---|---|---|---|---|---|---|---|
| spd_fp | stock | f0: +0.0152 (+2.1 %), f1: +0.0165 (+2.2 %), f2: +0.0099 (+1.4 %) | +0.0139 | 0.0035 | [-0.0046, +0.0337] | -0.0085 | not supported — failed rule (1) mean delta > 0.02, (3) bootstrap CI excludes 0 |
| temporal_fp_g1 | stock | f0: +0.0103 (+1.4 %), f1: +0.0128 (+1.7 %), f2: +0.0035 (+0.5 %) | +0.0089 | 0.0048 | [-0.0054, +0.0232] | -0.0001 | not supported — failed rule (1) mean delta > 0.02, (3) bootstrap CI excludes 0 |
| temporal_fp_g4 | stock | f0: +0.0446 (+6.1 %), f1: +0.0362 (+4.9 %), f2: +0.0536 (+7.5 %) | +0.0448 | 0.0087 | [+0.0208, +0.0717] | -0.0015 | supported |
| stock_e40 | stock | — | — | — | — | — | not run: skipped by the pre-registered 24 h development time guard |

## 10. Final test

| run | test set | CST drone AP50 | Anti-UAV drone AP50 | pooled drone AP50 | when |
|---|---|---|---|---|---|
| `gen_temporal_fp_g4_full_e20` | `37e944793773` | 0.5209 | 0.9185 | 0.7183 | 2026-09-27T11:54:58+0530 |
[fact] scored once; the candidate was selected on the development folds before the test split was read.

[measured] uncertainty of that single test result (sequence-level bootstrap of its recorded predictions, B = 2000): cst_drone_AP50 0.5209 [0.4415, 0.5973] (60 sequences); cst_drone_micro_AP50 0.4255 [0.3283, 0.5144] (60 sequences); anti_uav_drone_AP50 0.9185 [0.8939, 0.9419] (182 sequences); pooled_drone_AP50 0.7183 [0.6848, 0.7523] (1809 sequences).
[measured] CST test by scene type: categories seen in TRAIN 0.6172 (37 sequences), categories absent from TRAIN 0.3724 (23 sequences); per category: building 0.391 (9), cn_mountains 0.626 (9), cn_sky 0.884 (6), jungle 0.723 (13), urban-areas 0.242 (17, unseen), water 0.829 (6, unseen).

[fact] descriptive reference — stock YOLO11n's single test evaluation (repair series, same image set, same protocol; a different training run of the same recipe, so NOT a paired or seed-controlled comparison, and no per-image records exist for a paired CI):

| metric | selected (last.pt) | stock YOLO11n last | stock YOLO11n best | Δ vs stock last |
|---|---|---|---|---|
| drone_AP50 | 0.7183 | 0.6837 | 0.6738 | +0.0345 |
| cst_drone_AP50 | 0.5209 | 0.4561 | 0.4508 | +0.0648 |
| cst_drone_micro_AP50 | 0.4255 | 0.3996 | 0.3987 | +0.0259 |
| anti_uav_drone_AP50 | 0.9185 | 0.9105 | 0.9078 | +0.0080 |
| mAP50 | 0.6327 | 0.5913 | 0.5778 | +0.0414 |

## 11–13. What improved, what did not, credibility

- [rule] supported: temporal_fp_g4; not supported / rejected: spd_fp (not supported — failed rule (1) mean delta > 0.02, (3) bootstrap CI excludes 0), temporal_fp_g1 (not supported — failed rule (1) mean delta > 0.02, (3) bootstrap CI excludes 0).
- [rule] training budget (H4, 40 vs 20 epochs): not run: skipped by the pre-registered 24 h development time guard.

### [inference] Interpretation, limitations and recommended next step

**What improved.**
- [measured] Once every configuration starts from the same pretrained function, all three candidates are slightly ahead of
  `stock` in every fold (9 of 9 fold comparisons positive), i.e. the large penalties the repair grid attributed to SPD-Conv and
  to the temporal stack were produced by the initialisation, not by the architecture.
- [rule] Only the motion-matched temporal stack (`temporal_fp_g4`: t−4, t, t+4 frames, COCO stem on the centre frame) passes the
  pre-registered rule: CST drone AP50 +0.045 on held-out folds (+0.045 / +0.036 / +0.054; bootstrap 95 % CI of the mean
  [+0.021, +0.072]), Anti-UAV unchanged (−0.002). It is also ahead on the official VAL in every fold (CST +0.070 to +0.094) and,
  on its single test evaluation, above stock YOLO11n's single test evaluation (CST 0.521 vs 0.456, pooled drone 0.718 vs 0.684).

**What did not improve.**
- [rule] SPD-Conv with a fair start (`spd_fp`): +0.014, CI [−0.005, +0.034] — below the margin; not supported. With a 55 %
  larger network and no reproducible gain, it is not justified.
- [rule] The original t−1, t, t+1 stack with a fair start (`temporal_fp_g1`): +0.009, CI [−0.005, +0.023] — not supported.
- [fact] The training-budget question (40 vs 20 epochs) was not run: the pre-registered 24 h guard skipped it.

**Why the gap matters (inference).** CST cameras are static and its drones move about 0.5 px per frame (D1). At g = 1 the three
frames are nearly identical, so there is no motion to learn; at g = 4 the displacement is about 2 px, enough for the network to
use the motion of small, low-contrast thermal targets against a still background. Anti-UAV targets are large and visible in a
single frame, which matches the unchanged Anti-UAV score. [hypothesis] The gain comes from motion contrast, not from extra
capacity: `temporal_fp_g1` has the same network and input size and gains almost nothing.

**Credibility.** [inference] Moderate-to-good for the development claim: one change, chosen from a measurement (not a search),
pre-registered, paired against the reference with the same seed and the same starting function, replicated on three
session/sequence-disjoint folds, and the CI excludes zero. The test result agrees in direction. It is **not** a paired test
comparison (stock's test predictions were not kept, and re-scoring stock on test would break the once-only rule): stock's CST
test value (0.456) lies just inside the selected model's own CST CI [0.442, 0.597], so the test alone would not prove the gain.

**Limitations.**
1. One seed per fold (seed = fold index); fold-to-fold spread mixes data and optimisation noise (sd of Δ 0.009).
2. The CST development folds contain only the four training scene categories. On the test split the selected model scores 0.617
   on seen categories but 0.372 on the two unseen ones (`urban-areas` 0.242) — whether the temporal gain holds on new scene types
   is untested, and new scene types are the dominant remaining failure.
3. Deployment cost: g = 4 needs 4 future frames (+160 ms at 25 fps, vs +40 ms for g = 1 and 0 for a single frame); training reads
   three distinct frames per sample (≈ 1.35× the stock training time).
4. All models were still improving at epoch 20 (monitoring mAP50 slopes ≈ +0.005 to +0.010 per epoch); rankings at a longer
   budget are untested.
5. Det-Fly offers little independent evidence (two flights; its "temporal" input is always t,t,t).
6. Only g ∈ {0, 1, 4} was tested; the optimum gap is unknown and was deliberately not searched.

**Recommended next step.** Adopt stock YOLO11n with a centre-frame-initialised t−4, t, t+4 input as the AeroTrack detector; drop
SPD-Conv. Before claiming more, test — on the same folds, pre-registered — (a) a causal variant (t−8, t−4, t) that removes the
look-ahead latency, (b) three seeds per fold at 20 and 40 epochs, and (c) a leave-category-out CST protocol to measure unseen
scene types. Do not re-evaluate the test split for any of these; a new final candidate would need its own single test pass.

**Final conclusion.** [conclusion] The investigation found an implementation/training issue (initialisation) that invalidated
the repair grid's architectural verdicts, fixed it, and under a controlled protocol found one change with credible evidence of
better generalisation to unseen sequences — a temporal gap matched to the measured motion (+0.045 CST drone AP50, CI
[+0.021, +0.072]) — while SPD-Conv and the original t±1 stack remain unsupported.
