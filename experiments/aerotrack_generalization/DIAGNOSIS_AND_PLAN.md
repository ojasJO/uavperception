# AeroTrack generalization investigation — diagnosis and proposed plan

*2026-09-26. Diagnosis only: **nothing has been trained** in this series and **the test split has not been read** (no test
image, label or prediction; test sequence *names* were read once for a composition/leakage check). The completed repair grid
(`runs/repair_abl_*_e20`, `experiments/aerotrack_repair/reports/*`) is read, never modified. Every number below comes from
`reports/*.json` in this directory, produced by `scripts/*.py`, and each run is logged in `ledger.jsonl`.*

| id | script | what | output |
|---|---|---|---|
| D1 | `diag_data.py` | split structure / leakage, neighbour adjacency, target size and motion, camera motion, label flicker, Det-Fly duplicates | `reports/diag_data.json` |
| D2 | `diag_training.py` | per-epoch curves of the grid, COCO-initialisation audit, function-preserving-init check | `reports/diag_training.json` |
| D3 | `collect_val_records.py` | per-image VAL predictions of the 8 grid checkpoints (registered protocol; reproduces the registered mAP50) | `runs/aerotrack_generalization/_cache/` |
| D4 | `bootstrap_val.py` | paired, dataset-stratified, sequence-level bootstrap of the grid's comparisons (B = 1000) | `reports/bootstrap_val.json` |
| D5 | `diag_localization.py` | CST micro drones: missed vs found-but-mislocalised | `reports/diag_localization.json` |

## 1. Short answers

**Most plausible bottleneck.** Not a missing component. Three measured problems stack up:

1. **Initialization, not architecture, decides the grid.** The SPD layout replaces all five pretrained downsampling convolutions
   with randomly initialised SPD convolutions: **48.4 %** of its 4.09 M parameters start from COCO, against **93.7 %** of the
   stride layout's 2.63 M (D2). The SPD runs trail from epoch 0 (capped-val mAP50 0.18–0.21 vs 0.27–0.28), stay behind, and
   end with a *higher training loss* (2.26–2.28 vs 2.14–2.16) — they fit even the training data worse at equal budget. So
   "SPD vs stride" and "full recipe vs stock" at 20 epochs mostly measure half-random vs nearly-pretrained initialisation.
2. **The temporal stack is fed almost no motion.** CST drones move a median **0.53 px per frame** (68 % of frames < 1 px) under a
   static camera (median global shift 0.02 px); Anti-UAV drones move ~3 px on 36–51 px targets (≈ 7–8 % of their size) and are
   plainly visible in one frame; Det-Fly images have no neighbours at all (every "sequence" is one image) (D1). In addition,
   the grid's 9-channel *stride* stem (tiled /3) starts as a filter on the **mean** of the three frames, i.e. a moving target is
   first seen as three faint copies (D2).
3. **The development signal cannot resolve effects of the size we care about, and it is optimistic.** All 26 Anti-UAV VAL
   recording sessions also occur in TRAIN (the 10 test sessions occur in neither); CST VAL is 15 sequences carved from CST-train
   categories while CST test adds two scene types absent from training (`water`, `urban-areas`: 23 of 60 test sequences); 1,224
   of 1,469 Det-Fly VAL frames have their *adjacent* frame in TRAIN (D1). The pre-registered "micro AP50" is the mean of a
   drone-micro AP (1,579 of 1,860 boxes from 14 CST sequences) and a bird-micro AP (148 near-duplicate Det-Fly birds, 50 % of the
   weight). Sampling noise alone gives CST per-model 95 % CIs of about ±0.12 (D4), and checkpoint-to-checkpoint noise is as large
   as the effects: the same no_spd-vs-stock comparison flips sign between best.pt and last.pt (D4).

**Is there a credible modification?** One, and it is a fix rather than an addition: a **function-preserving initialisation**
(each SPDConv initialised to compute exactly the pretrained stride-2 convolution; the COCO stem placed on the centre frame with
zeros on t±1). With it, AeroTrack at initialisation *is* stock YOLO11n (verified numerically, D2), so its extra capacity can only
be learned on top, and the ablation finally compares architectures instead of initialisations. A second, conditional idea — a
temporal gap matched to the measured motion — has a mechanism for the static-camera thermal data only. The honest prior for
both is modest: a 3×3 stride-2 convolution already sees every pixel (the "lossless" argument for SPD is weak here), and the
motion cue is intrinsically small. **If the fair test shows no reproducible gain, the architectural additions are not justified
and the stock YOLO11n baseline should be adopted.**

**Decision (section 7): A** — there is a strong training/initialisation issue (and an evaluation-protocol issue) worth fixing
before any architectural conclusion is drawn. On the evidence available today, the simpler baseline is the better-supported
model (D); the plan below is the cheapest test that can overturn that.

## 2. The fifteen questions

| # | question | answer (evidence) |
|---|---|---|
| 1 | Does the architecture implement the idea? | Mechanically yes: SPDConv at all five downsampling layers, 9-ch early fusion of t−1/t/t+1, augmentation identical across the three frames, neighbours are the true adjacent frames (all gaps = 1), T6 fixed. But both premises fail on this data at g = 1 (Q2, Q4) and the SPD variant discards the pretrained downsampling layers (D1, D2). |
| 2 | Is SPD-Conv useful where it is used? | SPD addresses information loss in strided layers that *skip* pixels (kernel = stride, pooling). YOLO's 3×3 stride-2 convs overlap and see every pixel; an SPD conv can represent exactly the same function (proved by construction, D2). SPD is applied at all five stages, adds 1.46 M parameters (+55 %), all random. Only CST has genuinely small targets at the input (median 12–15 px; Anti-UAV 36–51 px, D1). The measured effect at this budget is negative (last.pt Δ micro −0.078 [−0.114, −0.050], CST micro −0.102 [−0.155, −0.067], D4) but confounded with initialisation. |
| 3 | Is the temporal stack implemented correctly? | Yes: same letterbox geometry, one affine/flip/photometric draw for all three frames, copy-paste lifts the same rect from all frames, boundary frames duplicated, neighbours consecutive, label flicker 18 isolated frames in 298k (D1). |
| 4 | Does the representation destroy information or add redundancy? | Redundancy: at g = 1 the three frames are near-copies (CST motion 0.53 px/frame; Anti-UAV ~7 % of target size; Det-Fly t,t,t always), so the 9-ch input triples the stem input for little information. Nothing is destroyed — frame t is intact — but the stride stem's tiled init averages the frames (D2). |
| 5 | Loss / augmentation / sampling / resolution / boxes / imbalance / schedule? | Loss: stock v8 gains; the installed assigner already enlarges sub-8-px boxes for candidate selection (no starvation of tiny targets — checked in `ultralytics/utils/tal.py`). Augmentation: no mosaic/mixup/v-flip (reasonable for tiny targets). Sampler: 18.9 % micro, 78.8 % small, 0.87 % empty frames (5.4 % of the corpus). Resolution: 640 letterbox is native for CST/IR; RGB is downscaled 3× but its targets stay large. Det-Fly was stretched to 640×640 by its Roboflow export (aspect ratio not preserved). Schedule: 20 of the originally planned 150 epochs; cosine completes. |
| 6 | Under/over-training, instability, mismatch? | Undertraining at the end of the schedule (training loss still falling 0.018/epoch; no val decline, so no overfitting signal). No instability in the loss. Representation mismatch: random SPD layers between COCO blocks, and the averaging stem (D2). The capped-val metric that picks best.pt is noise-dominated (stock's micro AP50 jumps 0.30 → 0.79 → 0.30 between epochs; best.pt = a spike, D2). |
| 7 | Why does stock YOLO11n outperform the full model? | Initialisation (94 % vs 48 % pretrained) at a short budget; near-zero motion signal at g = 1; SPD's extra random capacity. Val Δ (full − stock) drone AP50: −0.055 [−0.068, −0.040] best, −0.035 [−0.044, −0.025] last (D4) — significant for *this* seed, not attributable to the architecture. |
| 8 | Why does no_spd outperform baseline_ctrl? | Same initialisation gap: no_spd keeps the COCO downsampling convs; lower training loss from epoch 0 to 19 (D2). |
| 9 | Why is single_frame competitive? | The removed information is small: sub-pixel motion on CST, redundant appearance on Anti-UAV, none on Det-Fly (D1). Temporal Δ at last.pt: CST −0.015 [−0.034, +0.008], drone −0.004 [−0.012, +0.006] (D4) — indistinguishable. |
| 10 | Is val/test distribution different? | Yes, substantially: Anti-UAV VAL shares all 26 sessions with TRAIN, TEST shares none; CST VAL = 15 sequences from 3 training categories, CST TEST adds 2 unseen categories (23/60 sequences); Det-Fly VAL is interleaved with TRAIN (D1). VAL is an optimistic, in-distribution set. |
| 11 | Duplicates / near-duplicates across splits? | Sequence-level splits are clean for Anti-UAV/CST, but session-level overlap exists (Anti-UAV train↔val) and Det-Fly VAL is near-duplicate of TRAIN (1,478/1,565 within 2 bits of perceptual hash; 1,224 adjacent frame ids). Det-Fly TEST comes from the same random export split and is presumably the same; not checked, to keep the test untouched. |
| 12 | Are the three frames aligned? | Yes (same letterbox, same augmentation, consecutive frames; camera shift per frame is ≤ 0.35 px median, D1). |
| 13 | Are labels temporally consistent? | Yes: isolated on/off label flips 18/298,416 (Anti-UAV) and 6/74,867 (CST) training frames (D1). |
| 14 | Is the metric sensitive to the objective? | Partly not. "micro AP50" averages drone-micro and bird-micro AP (148 near-duplicate birds weigh 50 %); CST micro rests on 14 sequences; pooled AP mixes datasets. IoU 0.5 on 6–10 px boxes is harsh but not decisive: 7–9 % of CST micro drones are found-but-mislocalised (0.25 ≤ IoU < 0.5) vs 13–21 % missed, and the model ranking is the same by IoU ≥ 0.5 and by centre hit (D5). |
| 15 | Inherited details to reconsider | (a) tiled /3 stem for temporal input; (b) SPD layers initialised at random; (c) best.pt chosen by a fitness dominated by a noisy two-class micro AP on 2,000 images; (d) CST holdout of 15 sequences; (e) Det-Fly Roboflow split used as if sequence-level; (f) empty frames sampled at 0.87 %; (g) the 150-epoch recipe (3 warm-up epochs) run at 20 epochs. |

## 3. Explanations checked and rejected

| suspicion | verdict | evidence |
|---|---|---|
| tiny boxes never get a positive anchor (assigner starvation) | rejected | the installed `TaskAlignedAssigner.select_candidates_in_gts` widens boxes < 8 px to 16 px for candidate selection |
| t−1/t/t+1 misaligned or not adjacent | rejected | every neighbour gap is 1 frame; same letterbox; one augmentation draw per sample (D1, code) |
| label noise across frames | rejected | isolated flips ≤ 6e-5 of frames (D1) |
| the metric (IoU 0.5 on tiny boxes) produces the ranking | rejected | same ranking by centre hit; mislocalised share 6.5–8.7 % (D5) |
| sequence-level leakage in Anti-UAV / CST | rejected (but session-level overlap in Anti-UAV VAL) | D1 |

## 4. Hypotheses

### H1 — The ablation measures initialisation, not architecture (fix: function-preserving initialisation)

| | |
|---|---|
| problem | SPD layout 48.4 % pretrained vs stride 93.7 %; temporal stride stem averages the frames at init (D2). |
| why it could help generalisation | With ~400 training sequences, COCO features are the main source of robustness to unseen scenes. The modification should not throw them away; with the fix, any deviation from stock YOLO11n must be learned from data. |
| change | New init mode for this series only (the historical pipeline and the completed grid are untouched): after `transfer_pretrained`, (i) stem = COCO weights on the centre-frame channels, zeros on t−1/t+1; (ii) each SPDConv weight = `spd_from_stride(COCO stride-2 kernel)` + that layer's BatchNorm. Verified: SPD model = stride model to float32 rounding (max \|Δ\| 2.4e-3 on outputs up to 639); the centre-frame model ignores t±1 exactly (D2). Recipe otherwise identical to the grid (baseline stage, 20 epochs, batch 32). |
| expected failure mode | zero-initialised neighbour weights stay near zero (model ≈ stock); SPD's extra parameters fit training sessions and do not transfer. |
| supports it | `aerotrack_fp` − `stock` > 0.02 AP50 on the primary metric, averaged over the 3 grouped dev folds, **positive in every fold**, fold-pooled paired sequence bootstrap CI excluding 0; Anti-UAV drone AP50 not lower by more than 0.02 (non-inferiority). |
| rejects it | anything else — in particular Δ within ±0.02 or a sign change across folds. |
| cost | 2 configurations × 3 folds = 6 training runs ≈ 13 GPU-h (+ ~0.5 h evaluation). |
| test set needed? | no |

### H2 — The temporal cue is below resolution at g = 1 (change: a motion-matched temporal gap), tested without SPD

| | |
|---|---|
| problem | CST drones move 0.53 px/frame under a static camera; the stack carries almost no motion where appearance is weakest (D1). |
| why it could help generalisation | For small, low-contrast thermal targets against static backgrounds, motion is a scene-independent cue (the principle behind background subtraction); it should transfer to unseen scene types better than appearance. |
| change | temporal gap **g = 4**, derived from training labels (smallest g with median CST displacement ≥ 2 px: 4 × 0.53 ≈ 2.1 px), **not searched**; stride layout with centre-frame init (so SPD is not involved and the model starts as stock YOLO11n). Deployment cost stated up front: look-ahead grows from 1 to 4 frames (+160 ms at 25 fps). |
| expected failure mode | on gimbal Anti-UAV footage the target appears as separated copies over 4 frames and the network learns to ignore t±g; no CST gain. |
| supports it | CST drone AP50 (primary) and CST drone-micro AP50 improve over `stock` by > 0.02 in the fold mean, positive in every fold; Anti-UAV non-inferior. |
| rejects it | no consistent CST gain. |
| cost | 1 configuration × 3 folds = 3 runs ≈ 6.5 GPU-h. |
| test set needed? | no |

### H3 — The null: the additions add nothing measurable once confounds are removed

Supported if H1 and H2 are both rejected under the rule below. Action: adopt stock YOLO11n (COCO, single frame) as the
AeroTrack detector, retire SPD-Conv and the temporal stack as evaluated-and-rejected components (as was done for ByteTrack and
the LSTM), and redirect effort to the measured failure on unseen data — domain shift to new sessions and scene types — which
none of the additions targets. No new test evaluation is needed in this case: stock YOLO11n was already scored once.

## 5. Controlled experiment plan (proposed; to be frozen in `ledger.jsonl` before the first run)

**Development protocol — grouped folds from TRAIN only.** The official VAL cannot measure generalisation to new sessions or scenes
(Q10), so development uses 3 folds carved from the TRAIN split, grouped so that every held-out fold is *novel* at the level that
matters: Anti-UAV by recording session (27 sessions → 9 per fold, IR/RGB pairs kept together), CST by sequence stratified by scene
category (69 sequences), Det-Fly by flight (5 flights). Fold assignment by a fixed seed, balanced by frame count, written to a
hashed file before any run. Each run trains on two folds and is evaluated on the third. The official VAL is scored as a secondary,
continuity check only. The test split is not used until section 6.

**Configurations** (only these; no search): `stock` (reference: stride, single frame), `aerotrack_fp` (SPD + 9-ch g = 1,
function-preserving init) for H1, `temporal_fp_g4` (stride + 9-ch g = 4, centre-frame init) for H2. Recipe = the grid's
(`baseline` stage, 20 epochs × 25,000 samples, batch 32 × acc 2, same sampler/augmentation). Seed = fold index. 9 runs.

**Metrics, fixed now.** Primary: drone AP50 on the held-out fold per dataset — CST (primary for H1 and H2) and Anti-UAV
(non-inferiority) — and CST drone-micro AP50 (drone class only). Checkpoint: **last.pt only** (best.pt selection is noise-driven,
D2). Secondary, for continuity: the four registered metrics on the official VAL.

**Decision rule.** A configuration is supported only if, versus `stock`: fold-mean Δ > 0.02 on its primary metric, Δ > 0 in all
three folds, and the fold-pooled paired sequence bootstrap 95 % CI excludes 0; plus Anti-UAV Δ > −0.02. Report absolute and
relative Δ, per-fold values, mean ± sd, CIs. Anything else is "not supported" — no re-runs, no extra configurations, no epoch
picking.

**Ledger.** Every run appends configuration, fold, seed, data/fold-file hashes, git state, checkpoint hashes, metrics and rationale
to `ledger.jsonl`; code reuses the repair runner's resume/provenance machinery; new runs go to `runs/gen_<config>_f<k>_e20/`.

## 6. Test-set protection and the final step

The test split was scored once per grid checkpoint already and is not regenerated. Only if H1 or H2 is supported: retrain that one
configuration on the full TRAIN split with the registered recipe (seed 0, last.pt, decided now), score it on the official VAL, then
score it **once** on the test split through the existing ledger-guarded path. If nothing is supported, the test split is not
touched again.

## 6b. Plan as frozen before the first training run (ledger `P1`, `scripts/plan_gen.py`; supersedes §4–5 where they differ)

Refinements made before any training, for cleaner attribution:

- **One factor per candidate, identical starting function.** Instead of one combined `aerotrack_fp`, each candidate differs from
  `stock` in exactly one factor, and every configuration starts as COCO YOLO11n on frame t: `spd_fp` (SPD layout, t,t,t, mapped
  init — H1), `temporal_fp_g1` (stride, t−1,t,t+1, centre-frame init — the original temporal design with a fair start), and
  `temporal_fp_g4` (stride, t−4,t,t+4 — H2). The combined model is only worth testing if a single factor survives.
- **Futility at fold 0**: a candidate not ahead of `stock` on fold 0 cannot satisfy "Δ > 0 in every fold" and is rejected without
  running folds 1–2 (stopping only for failure cannot create a false positive).
- **H4 (training budget)** after the architecture stages, only if time permits: `stock` at 40 vs 20 epochs, same rule.
- **Folds** (`folds.json`, sha `ea48bbd1e1f6`): Anti-UAV 9 recording sessions per fold (~100k frames), CST 22–24 sequences per fold
  with every category represented, Det-Fly: its two drone flights are cut into three contiguous time chunks each (one per fold;
  only two flights exist), web bird photos grouped by base name (merges the export's duplicates).
- Seeds = fold index, shared by all configurations of a fold (paired comparison with common random numbers); torch/numpy seeded
  before the model is built; checkpoint = last.pt; per-epoch monitoring on the held-out fold selects nothing.
- Runner: `scripts/run_gen.py` (unattended, resumable, time-guarded: development runs must end within 24 h of the campaign start,
  the final step within 28 h); machinery smoke-tested end to end, including a killed-and-resumed run (`reports/smoke/`).

## 7. Cost and decision

| step | GPU time | notes |
|---|---|---|
| engineering (fold file, init mode, series runner, fold evaluation) | 0 | ~3–4 h of work, no training |
| H1: stock + aerotrack_fp × 3 folds | ≈ 13 h | answers the main question |
| H2: temporal_fp_g4 × 3 folds | ≈ 6.5 h | optional, can be dropped |
| final (only if supported) | ≈ 2.5 h + 1 test pass | none if H3 holds |

**Conclusion: A** — "There is a strong implementation/training issue worth fixing": the ablation is confounded by initialisation
(48 % vs 94 % pretrained parameters) and judged on an optimistic, low-resolution development set; the fix is small, verified and
cheap to test. Until that test is run, the measured evidence supports **D** — the stock YOLO11n baseline is the better-supported
approach — and that is also the most likely outcome of the test.
