# Anti-UAV / DFSC — comparison with the paper

*Generated 2026-10-05T06:24:40 by `experiments/anti_uav/scripts/aggregate_results.py`. Do not edit by hand: every number in the 'Paper' column is transcribed from the paper's tables (see `reports/paper_reported_results.json`, with table references), every number in 'Our reproduction' is read from a `metrics.json` produced by `scripts/evaluate.py` from saved tracker outputs. Empty cells = experiment not run yet.*

## Verdict rules (pre-registered before any run)

- |Δ mSA| ≤ 1.0 pt → **reproduced**; ≤ 3.0 pt → **approximately reproduced**; larger → **not reproduced**; no paper value → *not reproducible from available information*; no run → *not yet run*.
- The DFSC − normal *gain* counts as reproduced only if it has the paper's sign and |Δgain| ≤ 1.0 pt. The paper's gains are +0.49 (val) / +0.68 (test) mSA on IR — smaller than plausible seed-to-seed variance, so a single seed cannot settle this; Phase B seed runs are required before any claim.

## Protocol differences (apply to every row)

Same: paper epochs/LR steps/batch-per-GPU/losses/init source, GlobalTrack architecture & hyper-parameters, Eq.(1) metric, official start frames. Different (spec §11): torchvision re-implementation (S1), torchvision COCO Faster R-CNN init instead of the mmdet checkpoint (S2), 1 GPU x batch 2 with LR 0.0025 = linear-scaled 0.02 (S3), JPEG-decoded frames (S4), batches forced to distinct sequences (S5). Reproduction columns = seed 0; seed 1 runs are listed under seed variance.

## Table — mSA (%) on Anti-UAV infrared, Protocol II

| Method | Split | Metric | Paper | Our reproduction | Difference | Verdict | Our evaluation dir |
|---|---|---|---|---|---|---|---|
| normal | val | mSA OV | 78.09 | 56.05 | -22.04 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_val_IR_last |
| normal | val | mSA OC | 81.42 | 78.26 | -3.16 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_val_IR_last |
| normal | val | mSA FM | 78.66 | 79.47 | +0.81 | reproduced | experiments\anti_uav\runs\normal_ir\eval_val_IR_last |
| normal | val | mSA SV | 76.61 | 72.89 | -3.72 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_val_IR_last |
| normal | val | mSA LI | 79.95 | 79.94 | -0.01 | reproduced | experiments\anti_uav\runs\normal_ir\eval_val_IR_last |
| normal | val | mSA TC-EASY | 81.23 | 81.62 | +0.39 | reproduced | experiments\anti_uav\runs\normal_ir\eval_val_IR_last |
| normal | val | mSA TC-MID | 76.56 | 77.78 | +1.22 | approximately reproduced | experiments\anti_uav\runs\normal_ir\eval_val_IR_last |
| normal | val | mSA TC-HARD | 74.04 | 76.71 | +2.67 | approximately reproduced | experiments\anti_uav\runs\normal_ir\eval_val_IR_last |
| normal | val | mSA TC | 78.49 | 79.58 | +1.09 | approximately reproduced | experiments\anti_uav\runs\normal_ir\eval_val_IR_last |
| normal | val | mSA LR | 73.55 | 76.94 | +3.39 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_val_IR_last |
| normal | val | mSA All | 79.60 | 79.29 | -0.31 | reproduced | experiments\anti_uav\runs\normal_ir\eval_val_IR_last |
| normal | test | mSA OV | 70.44 | 69.57 | -0.87 | reproduced | experiments\anti_uav\runs\normal_ir\eval_test_IR_last |
| normal | test | mSA OC | 68.94 | 64.52 | -4.42 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_test_IR_last |
| normal | test | mSA FM | 60.66 | 70.43 | +9.77 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_test_IR_last |
| normal | test | mSA SV | 55.48 | 64.64 | +9.16 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_test_IR_last |
| normal | test | mSA LI | 59.78 | 75.51 | +15.73 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_test_IR_last |
| normal | test | mSA TC-EASY | 77.07 | 77.01 | -0.06 | reproduced | experiments\anti_uav\runs\normal_ir\eval_test_IR_last |
| normal | test | mSA TC-MID | 64.64 | 72.31 | +7.67 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_test_IR_last |
| normal | test | mSA TC-HARD | 44.61 | 51.63 | +7.02 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_test_IR_last |
| normal | test | mSA TC | 61.68 | 67.45 | +5.77 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_test_IR_last |
| normal | test | mSA LR | 52.94 | 70.24 | +17.30 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_test_IR_last |
| normal | test | mSA All | 65.36 | 70.58 | +5.22 | not reproduced | experiments\anti_uav\runs\normal_ir\eval_test_IR_last |
| DFSC | val | mSA OV | 77.72 | 56.03 | -21.69 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_val_IR_last |
| DFSC | val | mSA OC | 82.70 | 78.24 | -4.46 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_val_IR_last |
| DFSC | val | mSA FM | 79.34 | 79.41 | +0.07 | reproduced | experiments\anti_uav\runs\dfsc_ir\eval_val_IR_last |
| DFSC | val | mSA SV | 77.58 | 72.91 | -4.67 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_val_IR_last |
| DFSC | val | mSA LI | 80.31 | 79.88 | -0.43 | reproduced | experiments\anti_uav\runs\dfsc_ir\eval_val_IR_last |
| DFSC | val | mSA TC-EASY | 81.48 | 81.68 | +0.20 | reproduced | experiments\anti_uav\runs\dfsc_ir\eval_val_IR_last |
| DFSC | val | mSA TC-MID | 76.83 | 77.76 | +0.93 | reproduced | experiments\anti_uav\runs\dfsc_ir\eval_val_IR_last |
| DFSC | val | mSA TC-HARD | 75.65 | 76.72 | +1.07 | approximately reproduced | experiments\anti_uav\runs\dfsc_ir\eval_val_IR_last |
| DFSC | val | mSA TC | 79.04 | 79.61 | +0.57 | reproduced | experiments\anti_uav\runs\dfsc_ir\eval_val_IR_last |
| DFSC | val | mSA LR | 74.33 | 76.94 | +2.61 | approximately reproduced | experiments\anti_uav\runs\dfsc_ir\eval_val_IR_last |
| DFSC | val | mSA All | 80.09 | 79.26 | -0.83 | reproduced | experiments\anti_uav\runs\dfsc_ir\eval_val_IR_last |
| DFSC | test | mSA OV | 70.16 | 69.46 | -0.70 | reproduced | experiments\anti_uav\runs\dfsc_ir\eval_test_IR_last |
| DFSC | test | mSA OC | 68.07 | 64.78 | -3.29 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_test_IR_last |
| DFSC | test | mSA FM | 60.95 | 69.77 | +8.82 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_test_IR_last |
| DFSC | test | mSA SV | 55.55 | 63.87 | +8.32 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_test_IR_last |
| DFSC | test | mSA LI | 60.13 | 74.50 | +14.37 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_test_IR_last |
| DFSC | test | mSA TC-EASY | 77.96 | 77.34 | -0.62 | reproduced | experiments\anti_uav\runs\dfsc_ir\eval_test_IR_last |
| DFSC | test | mSA TC-MID | 65.85 | 72.00 | +6.15 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_test_IR_last |
| DFSC | test | mSA TC-HARD | 45.59 | 50.80 | +5.21 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_test_IR_last |
| DFSC | test | mSA TC | 62.75 | 67.13 | +4.38 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_test_IR_last |
| DFSC | test | mSA LR | 53.10 | 69.11 | +16.01 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_test_IR_last |
| DFSC | test | mSA All | 66.04 | 70.29 | +4.25 | not reproduced | experiments\anti_uav\runs\dfsc_ir\eval_test_IR_last |

## DFSC − normal gain (IR, mSA All)

| Split | Paper gain | Our gain | Same sign |
|---|---|---|---|
| val | +0.49 | -0.03 | False |
| test | +0.68 | -0.29 | False |

## Seed variance (plain training runs only; α = 0 → normal, α = 1 → DFSC)

| Split | Method | runs (seed → mSA) | n | mean | min | max | spread |
|---|---|---|---|---|---|---|---|
| val | normal | normal_ir (seed 0) → 79.29, phaseB_normal_ir_seed1 (seed 1) → 79.36 | 2 | 79.33 | 79.29 | 79.36 | 0.08 |
| val | DFSC | dfsc_ir (seed 0) → 79.26, phaseB_dfsc_ir_seed1 (seed 1) → 79.43 | 2 | 79.34 | 79.26 | 79.43 | 0.18 |
| test | normal | normal_ir (seed 0) → 70.58 | 1 | 70.58 | 70.58 | 70.58 | 0.00 |
| test | DFSC | dfsc_ir (seed 0) → 70.29 | 1 | 70.29 | 70.29 | 70.29 | 0.00 |

| Split | Paper gain | Gain of seed-means | Paired-by-seed gains | n pairs |
|---|---|---|---|---|
| val | +0.49 | +0.02 | -0.03, +0.07 | 2 |
| test | +0.68 | -0.29 | -0.29 | 1 |

Reading rule: the paper's gain is 'observed' only if every paired-by-seed gain has the paper's sign and the gain of seed-means is within 1 pt of the paper's value; otherwise the effect is reported as not observed at the available seeds.

## Paper rows not reproducible here

- **large-scale** (public GlobalTrack checkpoint, COCO+LaSOT+GOT-10k): needs the mmdet checkpoint + verified conversion → *not reproducible from available information* in this environment (spec S6).
- **Visible (RGB) Protocol II rows**: same dependency (init from the GlobalTrack checkpoint) → deferred; COCO-init RGB runs, if made, are ADDITIONAL CONTROLS, not reproductions.
- **Table VII precision/success**: split unspecified in the paper; ours are reported in each `report.md` for orientation only.

## Other evaluations (Phase B / additional controls / deferred) — never merged into the reproduction columns

| Run | Phase | α | Variant (single changed factor) | Split | mSA | Δ vs Phase A same run/split | prec@20 | success | Eval dir |
|---|---|---|---|---|---|---|---|---|---|
| dfsc_ir | B_additional_control | 1.0 | absence_by_score_threshold tau=0.3 | test | 70.70 | +0.42 | 89.91 | 69.71 | experiments\anti_uav\runs\dfsc_ir\phaseB_absence_thr\test |
| dfsc_ir | B_additional_control | 1.0 | absence_by_score_threshold tau=0.3 | val | 80.76 | +1.50 | 99.37 | 79.20 | experiments\anti_uav\runs\dfsc_ir\phaseB_absence_thr\val |
| dfsc_ir | B_additional_control | 1.0 | test_scale_640x512 | test | 66.01 | -4.28 | 90.33 | 66.61 | experiments\anti_uav\runs\dfsc_ir\phaseB_test_scale_640x512\test |
| dfsc_ir | B_additional_control | 1.0 | test_scale_640x512 | val | 74.65 | -4.61 | 99.32 | 74.88 | experiments\anti_uav\runs\dfsc_ir\phaseB_test_scale_640x512\val |
| normal_ir | B_additional_control | 0.0 | absence_by_score_threshold tau=0.3 | test | 70.87 | +0.29 | 89.87 | 69.87 | experiments\anti_uav\runs\normal_ir\phaseB_absence_thr\test |
| normal_ir | B_additional_control | 0.0 | absence_by_score_threshold tau=0.3 | val | 80.80 | +1.51 | 99.41 | 79.24 | experiments\anti_uav\runs\normal_ir\phaseB_absence_thr\val |
| normal_ir | B_additional_control | 0.0 | test_scale_640x512 | test | 66.15 | -4.43 | 90.30 | 66.76 | experiments\anti_uav\runs\normal_ir\phaseB_test_scale_640x512\test |
| normal_ir | B_additional_control | 0.0 | test_scale_640x512 | val | 75.26 | -4.02 | 99.33 | 75.44 | experiments\anti_uav\runs\normal_ir\phaseB_test_scale_640x512\val |
| phaseB_dfsc_ir_seed1 | B_controlled_improvement | 1.0 | — | val | 79.43 | — | 99.57 | 79.48 | experiments\anti_uav\runs\phaseB_dfsc_ir_seed1\eval_val_IR_last |
| phaseB_normal_ir_seed1 | B_controlled_improvement | 0.0 | — | val | 79.36 | — | 99.57 | 79.39 | experiments\anti_uav\runs\phaseB_normal_ir_seed1\eval_val_IR_last |

## Provenance of our numbers

| Run | Split | Checkpoint epoch | Checkpoint sha256 | git commit | seed | sequences |
|---|---|---|---|---|---|---|
| dfsc_ir | test | 18 | `1e8219c56d4b9bfe` | `77accee6c4` | 0 | 91 |
| dfsc_ir | val | 18 | `1e8219c56d4b9bfe` | `77accee6c4` | 0 | 67 |
| dfsc_ir | test | 18 | `1e8219c56d4b9bfe` | `0a8a482df9` | 0 | 91 |
| dfsc_ir | val | 18 | `1e8219c56d4b9bfe` | `0a8a482df9` | 0 | 67 |
| dfsc_ir | test | 18 | `1e8219c56d4b9bfe` | `e010d63173` | 0 | 91 |
| dfsc_ir | val | 18 | `1e8219c56d4b9bfe` | `e010d63173` | 0 | 67 |
| normal_ir | test | 18 | `ab9890d52684730e` | `0a8a482df9` | 0 | 91 |
| normal_ir | val | 18 | `ab9890d52684730e` | `77accee6c4` | 0 | 67 |
| normal_ir | test | 18 | `ab9890d52684730e` | `0a8a482df9` | 0 | 91 |
| normal_ir | val | 18 | `ab9890d52684730e` | `0a8a482df9` | 0 | 67 |
| normal_ir | test | 18 | `ab9890d52684730e` | `d89d36bb43` | 0 | 91 |
| normal_ir | val | 18 | `ab9890d52684730e` | `4e14b979fe` | 0 | 67 |
| phaseB_dfsc_ir_seed1 | val | 18 | `dc0b3e643b53222b` | `619449e3b0` | 1 | 67 |
| phaseB_normal_ir_seed1 | val | 18 | `414ef0164e2bfc65` | `619449e3b0` | 1 | 67 |

## Discrepancy analysis (written 2026-09-20 after the Phase A test evaluations; all numbers are filled in from `comparison.json` by `aggregate_results.py`)

### What was reproduced and what was not

| Quantity (IR, mSA %) | Paper | Ours | Δ | Pre-registered verdict |
|---|---|---|---|---|
| normal, val | 79.60 | 79.29 | -0.31 | reproduced |
| DFSC, val | 80.09 | 79.26 | -0.83 | reproduced |
| normal, test | 65.36 | 70.58 | +5.22 | not reproduced |
| DFSC, test | 66.04 | 70.29 | +4.25 | not reproduced |
| gain DFSC − normal, val | +0.49 | -0.03 | | not reproduced (sign differs; single seed) |
| gain DFSC − normal, test | +0.68 | -0.29 | | not reproduced (sign differs; single seed) |

1. **The GlobalTrack fine-tuning baseline and the DFSC model are reproduced on the validation split** (both within the
   1-pt band). The whole pipeline — data, architecture, training recipe, tracker, metric — lands where the paper lands on
   the split that shares source videos with training.
2. **On the independent test split both our models are +5.22/+4.25 pt *above* the paper.**
   The pre-registered rule labels this "not reproduced" because the difference is material, even though it is in our favour.
   Our val→test drop is 8.71 (normal) / 8.97 (DFSC) pt against the paper's 14.24 / 14.05 pt.
3. **The DFSC effect is not observed.** Seed 0 gives -0.03 (val) / -0.29 (test) against the paper's
   +0.49 / +0.68. With the Phase B seed-1 runs the val picture is: normal 79.29, 79.36 (mean
   79.33, spread 0.08), DFSC 79.26, 79.43 (mean 79.34, spread
   0.18); gain of seed-means +0.02, paired-by-seed gains -0.03, +0.07. The paper's
   +0.49 val gain lies outside everything we observe, while the seed-to-seed spread of a single method is of the same order
   as the paper's claimed effect. The training runs have near-identical loss trajectories and the DFSC cross-sequence RPN
   losses equal the intra-sequence ones (≈ 0.001) from epoch 3 on — the "any UAV" proposal task adds almost no gradient
   signal once the RPN has adapted to the IR domain. Honest summary: **no measurable DFSC effect in our reproduction**
   (n = 2 seeds per arm on val; test has seed 0 only unless the seed-1 test evaluations are run).

### Where the test surplus comes from — hypotheses, evidence, status

| # | Hypothesis | Evidence for / against | Status |
|---|---|---|---|
| H1 | **Annotation version.** The paper's runs used an earlier label set; the released labels were revised (the authors publish a start-frame correction table; the released box count is ~2 % below Table I: 573.4k vs 585.9k). Attribute pattern fits: the surplus concentrates on attributes where a few pixels of box error move IoU most — LR +17.30, LI +15.73, FM +9.77, SV +9.16 — while OV (-0.87) and TC-easy (-0.06) match. Val/test asymmetry (val matches, test does not) is consistent with a revision that touched the harder test videos more. Released labels are frame-accurate (no 25-frame interpolation signature; `reports/dataset_verification.md`). | plausible; not testable without the original labels |
| H2 | **Absence handling.** Phase A never declares "absent" (`[INFERRED]`). The Phase B control (score < τ ⇒ absent, τ chosen on val) changes val by +1.51/+1.50 pt but test by only +0.29/+0.42 pt. It explains the val OV/OC/SV attribute deficits (one val sequence has the UAV in 77/1000 frames), **not** the test surplus — any absence rule can only raise our test number further. | tested; rejected as cause of the surplus |
| H3 | **Optimisation regime.** 1 GPU × batch 2 for 18 epochs = 14,400 optimizer steps at LR 0.0025; an 8-GPU × 2 run (the mmdet convention behind "LR 0.02") would take 1,800 steps. More, smaller steps may fit the IR domain better and generalise better to independent videos (smaller val→test drop). | untested; `configs/phase_b/dfsc_ir_lr0.02_accum8.yaml` probes it |
| H4 | **Input scale.** GlobalTrack's transform up-scales IR 640×512 → 1000×800 (spec S1/§4). If the authors evaluated IR at native resolution, small (LR) and low-contrast (LI) targets would be harder for them — the two attributes with the largest surplus. | untested; a native-resolution eval of the same checkpoints would isolate it (Phase B, evaluation-only) |
| H5 | **Framework / weights.** torchvision Faster R-CNN (aligned RoIAlign, FrozenBN, anchor centring) and torchvision COCO_V1 weights instead of mmdet 1.x / mmdet 2x checkpoint. Val agreement within 1 pt argues against a large systematic advantage. | plausible minor contributor; not isolable here |
| H6 | **Evaluation implementation.** The challenge reference's `not_exist` quirk scores absent frames as 0; our Eq.(1) implementation scores an absent-declared frame as 1 — irrelevant in Phase A (no absence declared) and bounded by the 1.4 % absent frames in IR test. Start-frame handling changes our numbers by ≤ 0.05 pt (all-frames variant in each `report.md`). | tested; negligible |
| H7 | **Paper reporting.** Table VII precision/success for "normal" are 87.61/62.88; ours on test are 91.85/71.01, a surplus of the same order as mSA — so it is not a metric-specific artefact. A transcription error in the paper cannot be excluded but nothing indicates one. | no evidence |

No hypothesis is declared the cause. H1 is the most consistent with all observations; H3/H4 are the testable ones and are
queued as single-factor Phase B runs.

### What may be claimed (and what may not)

- May: "The GlobalTrack/DFSC training recipe was reproduced end-to-end; validation mSA agrees with the paper within 1 pt
  for both the baseline and DFSC; on the independent test split our implementation scores +5.22–+4.25 pt higher
  than the paper for reasons that could not be isolated (candidates H1, H3, H4)."
- May: "DFSC gives no measurable improvement over plain fine-tuning in our reproduction: gain of seed-means
  +0.02 pt on val (2 seeds per arm), -0.29 pt on test (seed 0)."
- May not: "DFSC does not work in general" (two seeds, one modality, one implementation), "our tracker is better than the
  paper's" (H1/H3/H4 unresolved), anything about the visible modality or the large-scale baseline (not run).

