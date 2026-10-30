# Can AeroTrack be scored under the CST Anti-UAV single-object-tracking protocol?

Short answer: **yes, as a UAV-specific single-object tracker**, with the differences below stated next to every number. The
adapter already exists (`experiments/systems/scripts/track_eval.py::run_tracker` + `ste.py`); this study only adds the official
output format (`scripts/cstlib.py`) and scoring by the official toolkit (`scripts/official_eval.py`).

## 1. Requirement-by-requirement

| CST protocol requirement (`PAPER_PROTOCOL.md` §2) | AeroTrack SOT harness | status |
|---|---|---|
| initialise from the GT box of frame 0, once | `STE.reset(init_box)` with the GT box of the official start frame; frame 0 is the start frame in all 220 sequences (verified) | **match** |
| no GT after initialisation, no re-initialisation | the tracker sees only cached detections from frame 1 on; `ste.py` reads no labels | **match** |
| online / causal | detector input window = frames (t−4, t, t−8) (`cache_detections.windows`, causal branch; clipped to frame 0 at the start) — no future frame; the STE is a forward Kalman filter | **match** (the symmetric `sym_g4` window would use t+4 and is **not** eligible; it is never used for CST claims) |
| per frame: a box or "absent" | STE outputs the associated detection box or `absent` (after `max_coast` misses, or below the init threshold) | **match** — and, unlike the released trackers, it can score the absent frames |
| result format | JSON `{"res": [box or []]}` per sequence (`cstlib.write_official`), read by the toolkit's own loader (tested) | **match** |
| metric | official toolkit `ExperimentCSTAntiUAV.report` (and Track A's identical implementation) | **match** (exact, `PAPER_PROTOCOL.md` §3) |
| training data: CST train (120 sequences) | X1 / X2 detectors train on CST train only (`train_v3.py --train-domains cst`); the v2 detector does **not** (it adds Anti-UAV + Det-Fly) | **match for X1 / X2 only** |
| model selection | benchmark: default configurations, no tuning. Ours: STE / argmax thresholds tuned on the official CST **val** split (40 sequences), frozen and committed before TEST | allowed use of val; stated |
| TEST used once | registered single passes with ledgers that refuse a second pass of the same weights (`x1_final.py`) | **match** |

## 2. Differences that remain (stated with every number)

1. **Paradigm.** The 21 released trackers are class-agnostic template trackers. AeroTrack is a class-specific (UAV) detector + a
   single-target state estimator: the first-frame box initialises the state and the gate, but no appearance template is kept.
   This is what a UAV-specific tracker (e.g. SiamDT, which the paper includes) is entitled to do with the CST training split, and the
   one-UAV-per-sequence design of CST makes "the target" and "the UAV" coincide. It would **not** be a valid entry on a generic SOT
   benchmark. We therefore label our row "UAV-specific detector-based SOT".
2. **Pretraining.** YOLO11n from COCO weights (function-preserving 9-channel stem); the released trackers use their own published
   pretraining (ImageNet / COCO / large SOT corpora). Comparable in kind, not identical.
3. **Validation tuning.** The benchmark trackers used default configurations; ours were tuned on CST val. Because CST val is cut from
   the same source videos as CST train (paper §3.3), val scores are optimistic; TEST is the independent measurement.
4. **Hardware / speed.** The paper ran on A100s and reports no CST FPS. We report end-to-end latency of the same network on the
   deployment GPU (RTX 5050) and the HPC V100 timing of the TEST pass; speed is never part of the accuracy comparison.
5. **Statistical reference.** The published table has one number per tracker; the released per-frame predictions give per-sequence
   SA, so the comparison with GlobalTrack / SiamDT can be **paired by sequence** (registered in amendment X1-A1 before the TEST pass).

## 3. What is NOT protocol-compatible and is kept in a separate table

| experiment | why not an official CST comparison |
|---|---|
| v2 final TEST (all-domain causal_g4, CST STE mSA 56.55) | detector trained on CST + Anti-UAV + Det-Fly (extra training data) |
| detection AP50 (any run) | a detection metric on all frames, not OPE state accuracy |
| C1 / C1b / C1c forecasting (ADE / FDE / NLL) | trajectory-forecast metrics on oracle or tracker histories |
| C3 (forecaster-driven STE vs CV-STE vs argmax) on VAL | VAL only; all-domain detectors; development evidence for motion models |
| Anti-UAV IR / RGB (Track A, 410-style protocols) | different benchmark |
| any symmetric (t+4) window | uses a future frame — violates OPE causality |
