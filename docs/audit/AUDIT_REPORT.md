# AeroTrack-Net — Critical Audit & Independent Benchmark

*Audit date: 2026-09-19. Scope: `rev1.ipynb`, `rev2.ipynb`, `README.md`, `results/runs/`, and the full code path. Every number below was either re-derived from the committed artifacts or re-measured on the RTX 5050 with the scripts in `audit/`. Outputs are in `results/runs/audit/`.*

---

## 0. The one-paragraph version

Review 2 claims four research pillars are "EMPIRICALLY VERIFIED ✓" and reports a "+43.2% mAP leap". The evidence does not support the attribution of any of the four pillars, and the headline comparison is invalid: **v1 was scored on 150 images and v2 on 3,000 different images**, v1 was an **unconverged run killed at epoch 21 of 150**, and the "fine-tune" changed **eight** things simultaneously. When both models are scored on the *same* images, the drone-class gain is **+0.01 AP50**; the entire mAP jump is the bird class. Independent benchmarks show: the 9-channel temporal stack contributes **nothing** (reversing time order *raises* mAP); the Social-LSTM has **collapsed to predicting zero motion** (ADE identical to a zero-velocity baseline to three significant figures); ByteTrack is beaten by a **no-tracker argmax baseline** by 2.8× IDF1; and on the held-out test split v2 is **worse than v1 on the thermal micro-drone dataset (CST) that motivated the architecture**. Meanwhile, the rev2 LSTM "visual proof" is hand-typed coordinates, and the demo GIF's "ByteTrack lock" contains no tracker. The data pipeline, split hygiene, and evaluation code are genuinely good — the problem is what was claimed on top of them.

---

## 1. What a reviewer sees immediately (why rev2 landed badly)

| # | Issue | Where |
|---|---|---|
| 1 | **No ablation, no baseline.** `train.py` defines `ABLATIONS = {no_spd, single_frame, no_sampler, scratch, ...}` and `yolo11-stride.yaml` exists with a docstring saying it "exists to answer one question and only one". The roadmap (§7.4) titles them *"each one defends a design decision to the panel."* **None were run.** No stock YOLO11n was trained. Therefore no gain can be attributed to SPD-Conv, temporal stacking, ByteTrack, or the LSTM. | `src/model/train.py:121`, `00_TRAINING_ROADMAP.md:380` |
| 2 | **v1 vs v2 compared on different evaluation sets.** v1 `eval_val` = 150 images (50/dataset; **31 micro GT boxes, 10 standard boxes**). v2 `eval_val` = 3,000 images. rev2's own caption says "150 validation frames". Every delta in the README table and rev2 §4 is across sets of unequal size and composition. | `results/runs/aerotrack_spd_v1/eval_val/evaluation_report.json`, rev2 cell 3 |
| 3 | **"EMPIRICALLY VERIFIED ✓" ×4 and "DEPLOYMENT READY ✓"** with no controlled experiment behind any of them. Marketing register ("PERFECT", "exceeds theoretical bounds", "leap") on top of uncontrolled comparisons. | rev2 cells 16, 20 |
| 4 | **The Pillar 4 plot is fabricated.** `x_obs, y_obs, x_gt, x_kf, x_lstm` are literal lists typed into the notebook, presented under "Direct Visual Proof". The real forecaster is never loaded in rev2. | rev2 cell 10 |
| 5 | **Negative results reported as positive.** SORT beats ByteTrack on MOTA (0.646 vs 0.598), IDF1, and ID switches (375 vs 578); rev2 leads with FN count. CST AP fell (0.693→0.648 even in the project's own numbers); rev2 says "gains across all previously depressed metrics". The LSTM is 54.5% *worse* than Kalman on CST in `forecast_report.json`; rev2 shows only the favourable subset. | `tracking_report.json`, `forecast_report.json` |
| 6 | **Cherry-picked demo from the test split.** `make_model_comparison_demo.py:104` hard-codes frames 104–134 of a *test* sequence chosen because "v1 drops track and v2 locks on". rev2's "35% vs 100% lock rate" is 20 frames of the same window. The project's own `val.py` says the test split is to be touched "once. ever." | `src/model/make_model_comparison_demo.py` |
| 7 | **The demo's "ByteTrack trail" is not ByteTrack.** Zero `BYTETracker(` instantiations in `inference.py`, `visual_demo.py`, or either demo generator. The trail is a `deque` of raw NMS centres. The HUD label "BYTETRACK OCCLUSION RECOVERY … 0 Identity Drops" is drawn over plain detector output. The LSTM is likewise never consumed by anything. The README's architecture diagram describes an integrated pipeline that does not exist. | `grep -c "BYTETracker(" src/model/*.py` |

---

## 2. Independent benchmark results

All runs: identical frozen sample sets (deterministic `subsample`, seed 0), the project's own `evaluation.py` metrics, bf16, conf 0.001 / IoU 0.7 / max_det 30 — i.e. the reported protocol. Scripts: `scripts/audit/bench_detector.py`, `bench_forecaster.py`, `bench_tracker.py`, `bench_latency.py`, `bench_bird_confusion.py`.

### 2.1 Same-set detector comparison — val split

**3,000 images (1000/dataset), the same images for both models:**

| | v1 (ep 15) | v2 (ep 23) | Δ |
|---|---|---|---|
| mAP50 | 0.5662 | 0.7881 | +0.222 |
| **drone AP50** | 0.6992 | 0.7880 | +0.089 |
| bird AP50 | 0.4332 | 0.7883 | +0.355 |
| micro AP50 (n=558) | 0.2566 | 0.4634 | +0.207 |
| Anti-UAV AP50 | 0.9761 | 0.9531 | **−0.023** |
| **CST AP50** | 0.6729 | 0.6412 | **−0.032** |
| CST micro AP50 | 0.6819 | 0.6089 | **−0.073** |
| Det-Fly AP50 | 0.3966 | 0.7579 | +0.361 |
| RGB AP50 | 0.9744 | 0.9387 | −0.036 |
| empty-frame FP/img @0.25 (n=239) | 0.042 | 0.071 | **+69%** (README: −22%) |

**11,565 images (5000/5000/all 1565):**

| | v1 | v2 |
|---|---|---|
| mAP50 | 0.6074 | 0.7788 |
| **drone AP50** | **0.7925** | **0.8027** (+0.010) |
| bird AP50 | 0.4224 | 0.7548 |
| micro AP50 (n=2008) | 0.2929 | 0.4078 |
| CST AP50 / micro | 0.6767 / 0.6594 | 0.6392 / 0.6091 |
| Det-Fly AP50 / micro | 0.3974 / 0.1235 | 0.7388 / 0.3307 |

Reading: the "+43% mAP" is the two-class average moving because bird AP doubled. The drone class barely moved. The pooled "micro" gain is Det-Fly micro (birds and daylight drones); **thermal micro drones (CST) got worse.**

### 2.2 Held-out test split (11,567 images; 2,722 micro GT)

| | v1 | v2 | README (v2, val) |
|---|---|---|---|
| mAP50 | 0.5491 | 0.7344 | 0.7904 |
| drone AP50 | 0.6590 | 0.6745 | 0.7916 |
| micro AP50 | 0.2049 | 0.2862 | 0.4776 |
| **CST AP50** | **0.4471** | **0.3933** | 0.6478 |
| CST micro AP50 | 0.3950 | 0.3174 | 0.6187 |
| Anti-UAV AP50 | 0.9015 | 0.8787 | 0.9534 |
| empty-frame FP/img @0.25 (n=1083) | 0.270 | **0.435** | 0.0711 |

`val.py`'s docstring calls CST-test "the one measurement that catches" a centre-bias model. On it, v2 is 5 points worse than v1 and 25 points below the advertised number. The val→test gap on CST (0.64 → 0.39) says the CST val holdout (carved from `cst[train]`) is not representative of the non-gimbal test footage.

### 2.3 Inference-time temporal ablation (Pillar 2)

Same 3,000 val images; the 9-channel input is rewritten before the model:

| input | v1 mAP50 | v2 mAP50 | v2 micro AP50 |
|---|---|---|---|
| `[t−1, t, t+1]` (trained config) | 0.5662 | 0.7881 | 0.4634 |
| `[t, t, t]` — no motion | 0.5617 | 0.7853 | **0.4693** |
| `[t+1, t, t−1]` — time reversed | **0.5671** | **0.7883** | 0.4670 |
| `[0, t, 0]` — neighbours blanked | 0.4363 | 0.1023 | 0.0004 |

Removing all temporal information costs ≤0.005 mAP; reversing time *improves* it. The network learned nothing from inter-frame motion; it learned that channels 0–2 and 6–8 look like natural images (hence the collapse when they're zeroed). **The TrackNet pillar is not doing what rev2 says it does** ("early kernels compute spatio-temporal derivatives, extracting velocity vectors and propeller flicker" — unsupported). The structural cost of the stack is real, though: `inference.py:191` notes "+1 frame of structural latency from the t+1 lookahead" — 40 ms at 25 fps, omitted from the README latency table.

### 2.4 Trajectory forecasting (Pillar 4) — LSTM vs trivial baselines

Same 54,651 test windows as `forecast_report.json`, project's own `ade_fde`:

| subset | n | **zero-velocity** | last-vel | mean-vel(4) | Kalman (project q,r) | Kalman (tuned on val) | **LSTM** |
|---|---|---|---|---|---|---|---|
| all | 54,651 | **41.88** | 57.63 | 56.45 | 56.50 | 54.79 | **41.85** |
| sudden-jump | 10,752 | **109.86** | 165.01 | 151.66 | 149.16 | 155.41 | **109.80** |
| smooth | 43,899 | **25.23** | 31.33 | 33.13 | 33.80 | 30.15 | **25.21** |
| Anti-UAV | 40,314 | **52.31** | 71.90 | 73.36 | 73.69 | 70.13 | **52.25** |
| CST | 14,337 | 12.55 | 17.52 | 8.90 | **8.16** | 11.67 | 12.61 |

Diagnostics: LSTM mean predicted step = 0.00021 (normalised); true mean step = 0.0071; corr(LSTM step 1, last observed step) = −0.24. **The LSTM outputs ≈0 regardless of input.** The "+26.4% over Kalman" is exactly the margin by which *predicting no motion* beats constant-velocity extrapolation on gimbal-tracked, mean-reverting jitter. Where motion is persistent (CST), the LSTM loses to Kalman and to a 4-frame mean. `train_mse` flatlining at 0.00193 from epoch 1 in `forecast_report.json` was the tell. Pillar 4 is unverified and, as trained, useless.

### 2.5 Tracking (Pillar 3) — ByteTrack vs no tracker

Same 10 sequences / same detections as `tracking_report.json` (reproduced to 3 decimals):

| tracker | MOTA | IDF1 | ID sw | FP | FN | pred IDs (10 GT) |
|---|---|---|---|---|---|---|
| ByteTrack (reported) | 0.596 | 0.309 | 578 | 1326 | 1898 | 279 |
| SORT (control) | 0.645 | 0.307 | 375 | 368 | 2599 | 258 |
| **argmax@0.25 — no tracker** | **0.783** | **0.862** | **0** | 350 | **1689** | 10 |
| argmax@0.1 | 0.778 | 0.869 | 0 | 601 | 1484 | 10 |
| greedy-IoU single ID @0.25 | 0.777 | 0.859 | 0 | 381 | 1720 | 10 |

Per domain (MOTA / IDF1): Anti-UAV — ByteTrack 0.635/0.167, argmax 0.955/0.977. CST (occlusion-heavy, ByteTrack's motivating case) — ByteTrack 0.551/0.450, argmax 0.588/0.747.

A single-target corpus does not need a multi-object tracker; the one deployed *destroys* identity (64–94 IDs for one drone in a 1000-frame clip) and *adds* false positives. The "0 ID switches at occlusion re-entry" is 0 for SORT too — and `track_eval.py`'s own docstring says that if ByteTrack doesn't reduce it "the citation is decoration and should be dropped."

### 2.6 Latency (README: 11.8 ms / 84.7 FPS "bfloat16")

RTX 5050 Laptop, batch 1, 640², fused:

| stage | README | measured |
|---|---|---|
| forward, **bf16** | 8.4 ms | **12.48 ms** (p95 20.0) |
| forward, fp32 | — | **8.91 ms** (p95 14.8) |
| preprocess (1 new frame, rolling deque, IR) | 1.8 ms | 0.26 ms (1080p RGB: 1.97) |
| NMS @0.25 (CPU) | 0.9 ms | 0.34 ms |
| ByteTrack update | 0.7 ms | 0.17 ms |
| structural lookahead latency | — | **+1 frame (40 ms @25 fps)** |

`inference.py` runs fp32 (no autocast anywhere), so the total is plausible; the "bfloat16" label is wrong (bf16 is *slower* here), the breakdown was not measured, and there is no benchmark script in the repo to reproduce it. `scripts/audit/bench_latency.py` now is one.

### 2.7 Bird→drone confusion, measured correctly (all 1,565 Det-Fly val images)

`evaluation.py::fp_rate` counts *every* prediction on a bird-only image as an FP, including correct bird detections — so the shipped `bird_fp` (16–19 FP/img) is meaningless. Correct measurement:

| | v1 | v2 |
|---|---|---|
| bird-only images with ≥1 drone alarm @0.25 | 20.6% | **0.0%** |
| drone alarms per bird-only image @0.25 | 0.526 | **0.000** |
| GT birds boxed as drone (IoU≥.5) | 0.8% | **0.0%** |

**This is the one clear thing the fine-tune bought, and it is real.** Caveats a panel will raise: Det-Fly has no mixed images (97 bird-only, 1,468 drone-only), ~20 birds per bird image (flocks), and is the only source of birds — so the model can solve it from image-level domain cues, and *nothing* tests a lone bird in a thermal or Anti-UAV-style scene. The price paid is in §2.1–2.2 (CST, Anti-UAV, empty-frame FP).

---

## 3. Confounds in the v1 → v2 comparison (why nothing is attributable)

1. **v1 is not converged.** `args.json` says 150 epochs; `results.csv` stops at epoch 21 with LR still 0.00095 (cosine barely started). Best checkpoint = epoch 15. v2 then trains 25 more epochs with a *full* anneal (2e-4 → 4e-6). A large share of "v2 gain" is "finish training".
2. **Fine-tune epoch 0 scored 0.12 mAP50** (v1 was at 0.67). Something reset — likely `hyp cls=1.5` (3× stock) × class weight 4.5 × sampler bird 35 blew up the classification loss (39.6 at epoch 0 vs 0.45). It re-learned from near scratch.
3. **Eight simultaneous changes** in `STAGES["finetune"]`, not the three rev2 lists: lr0/lrf/warmup; batch 8→32 (baseline stage says 8; `args.json` says 32); `domain_balanced` 40/35/25; `sampler_weights` → `FINETUNE_WEIGHTS` (**bird 35.0**, not the 8.0 shown in rev1); `class_weights [1, 4.5]`; `hyp cls 0.5→1.5`; different `augment_kwargs`; different training-time `val_caps`. Effective bird gradient weight ≈ 35 × 4.5 × 3 ≈ **470×** a standard drone.
4. **Det-Fly over-exposure.** 25% of 25k samples/epoch on 4,698 images ≈ 33 passes over Det-Fly train in 25 epochs vs ≈0.8 passes over Anti-UAV. The Det-Fly jump is at least partly memorisation of a small daylight dataset.
5. **Checkpoint selection on a noisy metric.** Fitness weights micro AP at 35%; `results.csv` shows v2 micro_AP50 flipping 0.33 ↔ 0.77 on consecutive epochs at LR ≈ 5e-6 (weights essentially frozen). "best" (epoch 23) was picked on a high swing. Any selected checkpoint is optimistically biased on this metric.
6. **60% of weights are COCO-pretrained** (`yolo11n.pt`, 388/649 tensors). Never mentioned in rev2; the hero tile "vs Base YOLOv11" implies a comparison that never happened.

---

## 4. Definitional / reporting inconsistencies

| Claim | Reality |
|---|---|
| README: size bins "strictly adhere to MS-COCO thresholds" | Code: relative area (<0.03%, 0.03–1%, >1%). COCO is absolute 32²/96². rev2 labels the same bins "<16 px / 16–32 px / >32 px". 0.03% of frame is ~10 px on IR, ~25 px on 1080p RGB, ~11 px on Det-Fly. Three definitions across three documents. |
| README: v1 "15 epochs" | rev2: "22 epochs". (best.pt is epoch 15 of a 22-epoch run; say so.) |
| README/rev1: sampler weights bird 8, micro 6, small 3 | v2 used bird 35, micro 8, small 4, empty 0.8. |
| rev1/rev2: "stride-2 conv discards 75% of pixels", "81 samples → 1.27" | A 3×3/s2 conv reads every input pixel (overlapping windows). Nothing is "discarded"; resolution halves — same as SPD. The 64× bar chart is not a real quantity. |
| "SPD-Conv is 100% lossless" | The *reshape* is lossless. The block then projects 4C→c2 with c2 < 4C (layer 0: 36→16, layer 3: 256→64). SPD+3×3/s1 ≡ a stride-2 conv with a 6×6 effective receptive field and more parameters (4.06M vs 2.61M). Any benefit is capacity/RF, not losslessness — which is exactly what the unrun `no_spd` ablation would measure. |
| README latency table "bfloat16" | `inference.py` is fp32; bf16 is slower at batch 1. |
| README: "Empty Frame FP −21.78%" | v1 n=11 images vs v2 n=239. Same-set: +69% (val), +61% (test). |
| README "Bird Precision 96.32%" as rejection rate | Precision of the *bird* class ≠ bird→drone false-alarm rate. The correct number (§2.7) is better than what was claimed, but was never computed. |
| rev2: on-disk `eval_val/evaluation_report.json` for v2 | Currently overwritten by a **6-image** smoke run (`n_images: 6`, `git diff`). Restore or regenerate before anyone opens it. |

---

## 5. What is genuinely good (say this to the panel too)

- **Split hygiene is excellent.** Sequence-level splits, fixed-seed CST holdout, hard leakage assertions (T4), IR/RGB scene pairs verified to land in the same split (318/318). This is better than most student projects and most papers.
- **The evaluation code is honest and well-reasoned** (size stratification on source-frame area, COCO ignore semantics, NMS time-budget bug caught and neutralised, measured NMS-on-CPU decision).
- **The forecaster and tracker scripts set falsifiable criteria in their docstrings** — the science was designed correctly; the presentation then ignored the outcomes.
- **Data engineering** (headless-archive recovery, 723k-box EDA, materialised corpus, label cache) is substantial, real work.
- **The bird-rejection result (§2.7) is a real, measurable win** once the right metric is used.

---

## 6. What to do before Review 3 (priority order)

Real training throughput on the 5050 is 15–21 img/s (`results.csv`), i.e. ≈19–27 min per 25k-sample epoch (the journal's 124 samples/s was the data loader alone). Budget accordingly: a **40-epoch reduced ablation ≈ 13–18 h** each. Two or three of them are the most valuable thing you can do before Review 3.

1. **Run the ablations you already wrote.** `--ablation no_spd` and `--ablation single_frame` at the *same* recipe/budget as a fresh, converged baseline (not v1). If SPD doesn't beat `no_spd` on CST micro AP, drop the claim. If `single_frame` matches 9-ch (my §2.3 predicts it will), drop the stack and recover 40 ms of latency.
2. **Add a stock YOLO11n baseline** (3-ch, COCO init, same data, same epochs). Without it there is no "vs standard detection".
3. **One evaluation set, one protocol, both models.** Full val for everything; test once at the end, on the final model, both v1 and v2 side by side. Report per-domain first (CST is the headline for a *thermal micro-drone* system), pooled second. Report `n_gt` per bin and don't quote AP on <100 boxes.
4. **Change one thing at a time from a converged baseline**, or at minimum present the fine-tune as "a recipe change" and never attribute its gains to an architecture.
5. **Retire Pillar 4 or fix it.** Add zero-velocity and last-velocity baselines to `lstm_forecaster.py` (they're 2 lines); train with a loss that can't collapse (e.g. predict residual over last velocity, or standardise displacements per-sequence); report per-domain. If it can't beat mean-vel(4) on CST, say so and drop it.
6. **Retire ByteTrack for this corpus** or reframe it: report argmax/greedy-single-ID as the baseline it must beat. Fix `bird_fp` to count drone-class predictions only.
7. **Delete the hand-typed LSTM plot**, regenerate every "visual proof" from model output, and pick demo clips by a stated rule (e.g. median-difficulty val sequence), not by "where v1 fails". Don't render demos from the test split.
8. **Make the demo honest about the pipeline**: either wire `BYTETracker` into `inference.py`/the demo or remove "ByteTrack" from the captions and the README diagram.
9. **Consistency pass:** one definition of micro (state it in px at network input *and* as area), one epoch count, one sampler table, remove "MS-COCO thresholds" and "bfloat16", add p95 latency and the +1-frame lookahead.
10. **Add `audit/` to the repo and cite it.** A reviewer trusts a project that benchmarks itself against trivial baselines far more than one that reports "PERFECT ✓".

---

## Appendix — reproducing this audit

```bash
# same-set detector comparison + temporal ablation (val, 3000 imgs)
venv/Scripts/python.exe scripts/audit/bench_detector.py --split val --cap 1000 --modes normal static reversed zero_nb \
  --weights results/runs/aerotrack_spd_v1/weights/best.pt results/runs/aerotrack_spd_finetuned/weights/best.pt
# larger val and held-out test samples
venv/Scripts/python.exe scripts/audit/bench_detector.py --split val  --cap 5000 --weights <both>
venv/Scripts/python.exe scripts/audit/bench_detector.py --split test --cap 5000 --weights <both>
# forecaster vs trivial baselines
venv/Scripts/python.exe scripts/audit/bench_forecaster.py
# tracker vs no-tracker
venv/Scripts/python.exe scripts/audit/bench_tracker.py --weights results/runs/aerotrack_spd_finetuned/weights/best.pt
# latency
venv/Scripts/python.exe scripts/audit/bench_latency.py
# bird -> drone confusion, measured correctly
venv/Scripts/python.exe scripts/audit/bench_bird_confusion.py --weights <both>
```

Raw outputs: `results/runs/audit/*.json`, logs in `results/runs/audit/log_*.txt`.
