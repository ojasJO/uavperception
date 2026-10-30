# DRONE-Track — system architecture

Companion to `docs/REVIEWER_GUIDE.md`, and the material for the architecture slides: each section
below is written to go on a slide more or less as it stands. Every component names the file it
lives in, so each box can be opened in the code.

---

## 1. The problem the architecture is shaped by

| constraint | consequence for the design |
|---|---|
| the target is a few pixels across (micro = under 0.03 % of the frame area; 9.9 px at the threshold on 640×512 IR) | the detector must keep spatial resolution where it is cheapest to lose it, and must be scored on a size-stratified metric, not on aggregate mAP |
| a single frame of a 9-pixel drone against thermal clutter is nearly indistinguishable from noise | the input is a **window of frames**: motion is the signal |
| the system must run on a live feed | the window may use **only past frames** — zero look-ahead |
| the target disappears behind buildings and terrain, sometimes for 90 frames | the tracker needs an explicit *absent* state, and the metric must score "correctly absent" |
| two sensors (thermal IR, visible) and three datasets | one unified corpus, one evaluation protocol, sequence-level splits |
| a laptop GPU is the deployment target | about 2.6 M parameters, and export to TensorRT must not change the result |

---

## 2. End-to-end pipeline

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0   DATA PREPARATION                                   src/pipeline/         │
├──────────────────────────────────────────────────────────────────────────────────┤
│  Anti-UAV 2021        CST Anti-UAV              Det-Fly                          │
│  318 IR/RGB pairs     144 IR sequences          7,830 daylight images            │
│  593,802 frames       162,187 frames            Bird / Drone                     │
│         │                     │                       │                          │
│         └─────────────────────┼───────────────────────┘                          │
│                               ▼                                                  │
│   1_standardize_annotations.py   one YOLO corpus, two classes                    │
│   2_verify_pipeline.py           counts, hashes, label round-trips               │
│   3_prepare_training.py          sequence-level splits, seed 1337,               │
│                                  leakage assertions                              │
│                               ▼                                                  │
│   data_unified_v2/   manifest.csv · splits.json · label_cache.npz · hardness.npz │
│   DATA_VERSION = corpus_v2_f842d03909cf_cst_full_antiuav_hpcdecode               │
└──────────────────────────────────────────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1   SAMPLING                        src/pipeline/imbalance_sampler.py        │
├──────────────────────────────────────────────────────────────────────────────────┤
│  25,000 frames per epoch, drawn by a hardness-aware weighted sampler             │
│  weights  bird 8.0 · micro 6.0 · small 3.0 · standard 1.0 · empty 0.5            │
│  (without it, the loss is dominated by easy large targets and empty frames)      │
└──────────────────────────────────────────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2   CAUSAL TEMPORAL WINDOW          src/pipeline/stacked_dataset.py          │
│                                           experiments/aerotrack_v2/scripts/v2lib │
├──────────────────────────────────────────────────────────────────────────────────┤
│   frame t-8 ─┐                                                                   │
│   frame t-4 ─┼─► letterbox to 640×640, aspect kept, pad value 114                │
│   frame t   ─┘    BGR as OpenCV reads it, float32 = uint8 / 255                   │
│                                                                                  │
│   channel order   0-2 = t-4   │   3-5 = t   │   6-8 = t-8                        │
│   tensor          [1, 9, 640, 640]                                               │
│   look-ahead      0 frames          gap g = 4, so the window spans 8 frames      │
│   sequence start  a missing past frame is replaced by the earliest available one  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3   DETECTOR                        src/model/yolo11-stride.yaml             │
│                                           Ultralytics 8.4.115 · 2.613 M params   │
├──────────────────────────────────────────────────────────────────────────────────┤
│   input  [9, 640, 640]                                                           │
│     │                                                                            │
│   BACKBONE   Conv s2 ──► 64    P1/2   320²                                       │
│              Conv s2 ──► 128   P2/4   160²                                       │
│              C3k2 ×2 ──► 256                                                     │
│              Conv s2 ──► 256   P3/8    80²  ────────────────┐ skip               │
│              C3k2 ×2 ──► 512                                │                    │
│              Conv s2 ──► 512   P4/16   40²  ──────────┐ skip│                    │
│              C3k2 ×2 ──► 512                          │     │                    │
│              Conv s2 ──► 1024  P5/32   20²            │     │                    │
│              C3k2 ×2 ──► 1024                         │     │                    │
│              SPPF    ──► 1024                         │     │                    │
│              C2PSA   ──► 1024  ──────────────┐ skip   │     │                    │
│     │                                        │        │     │                    │
│   NECK  (standard YOLO11 PAN-FPN, unchanged) │        │     │                    │
│         upsample ─► concat P4 ◄──────────────────────┘     │                    │
│         C3k2 ─► upsample ─► concat P3 ◄─────────────────────┘                    │
│         C3k2 ─► P3/8  small   ──────────────────────────► head                   │
│         Conv s2 ─► concat ─► C3k2 ─► P4/16 medium ───────► head                   │
│         Conv s2 ─► concat ◄──────────────────┘                                   │
│                   C3k2 ─► P5/32 large  ─────────────────► head                   │
│     │                                                                            │
│   HEAD  Detect, 3 scales, 2 classes:  0 = uav_drone   1 = bird_hard_negative     │
│                                                                                  │
│   initialisation  COCO yolo11n.pt, every name- and shape-matching tensor         │
│                   417 / 649 tensors = 64.3 % (see §6, the erratum)               │
└──────────────────────────────────────────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4   NON-MAXIMUM SUPPRESSION                                                │
├──────────────────────────────────────────────────────────────────────────────────┤
│   evaluation   conf 0.001 · IoU 0.7 · max 30 detections per image                │
│   streaming    conf 0.05  · IoU 0.7                                              │
│   boxes xyxy in letterboxed pixels ─► letterbox undone ─► source pixels          │
└──────────────────────────────────────────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5   SINGLE-TARGET STATE ESTIMATOR (STE)   src/model/track_eval.py            │
│                                                 experiments/systems/             │
├──────────────────────────────────────────────────────────────────────────────────┤
│   A Kalman single-target state estimator, reset from the frame-0 ground-truth     │
│   box and given no later ground truth.                                           │
│                                                                                  │
│   per frame:   detections ─► gate by predicted position and score                │
│                           ─► accept the best candidate, or declare ABSENT        │
│                           ─► update the state  (constant-velocity model)          │
│                                                                                  │
│   output   one box per frame in source pixels, or the explicit state "absent"     │
│   params   frozen on VAL on 2026-10-06, before any TEST pass                     │
│            src/deployment/aerotrack_causal_g4_v2/tracker/ste_params.json             │
│   cost     0.126–0.145 ms per frame (the argmax baseline: 0.012–0.017 ms)         │
└──────────────────────────────────────────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6   DEPLOYMENT          experiments/deployment/scripts/aerotrack_stream.py │
├──────────────────────────────────────────────────────────────────────────────────┤
│   PyTorch ─► ONNX (opset 18) ─► TensorRT fp16 engine                             │
│   laptop RTX 5050, batch 1:  model p50 3.2 ms / p99 4.7 ms                        │
│                              end-to-end p50 5.4 ms / p99 8.0 ms = 170 FPS        │
│   AP50 parity after export ≤ 0.003 · look-ahead 0 frames                         │
│   server↔laptop parity: presence 100 %, top-1 IoU mean 0.985, |Δscore| 0.0003    │
│   package  src/deployment/aerotrack_causal_g4_v2/  (17 files, sha256-verified)        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Why the causal window is the architecture's one real idea

Everything else in the detector is stock YOLO11. The window is what the measurements support, and
it is the only component that survived its ablation:

| configuration | window | look-ahead | CST drone AP50 (VAL, seed mean) |
|---|---|---:|---:|
| `stock` | frame *t* only | 0 | 0.570 |
| `causal_g1` | t−2, t−1, t | 0 | 0.602 |
| `causal_g2` | t−4, t−2, t | 0 | 0.663 |
| **`causal_g4`** | **t−8, t−4, t** | **0** | **0.681** |
| `causal_g8` | t−16, t−8, t | 0 | 0.679 |
| `sym_g4` | t−4, t, **t+4** | **4 frames** | 0.674 |

Three results, each pre-registered:

1. **The window helps, and the effect is large**: causal vs single frame **+0.1110 AP50**
   [+0.071, +0.155] (decision C1, SUPPORTED). On the benchmark metric the same comparison is
   **+11.51 mSA** [+8.36, +14.86] across three seeds.
2. **The look-ahead is not needed**: causal vs symmetric **+0.0074** [−0.000, +0.016] — the causal
   window is **non-inferior** (decision C3, conclusive). Dropping the future frame costs nothing
   measurable, and it is what makes the system deployable.
3. **The gap matters more than the count**: a one-frame gap captures too little motion; past
   gap 4 the curve flattens. `causal_g4` was selected by the registered rule.

The companion negative result is just as important: on Anti-UAV the window changes nothing
(|Δ| ≤ 0.001 AP50). The gain is specific to CST's tiny, cluttered thermal targets — exactly where
a single frame carries too little evidence.

---

## 4. The two research directions

```
                        corpus v2  (Anti-UAV · CST Anti-UAV · Det-Fly)
                                        │
          ┌─────────────────────────────┴─────────────────────────────┐
          ▼                                                           ▼
DIRECTION A — AeroTrack                              DIRECTION B — reproduction
"can a causal temporal window plus a simple          "can Anti-UAV (Jiang et al. 2021,
 state estimator beat the published trackers?"        arXiv:2101.08466) be reproduced?"

  causal-window detector  (aerotrack_v2, v3)           GlobalTrack, authors' public
  single-target state estimator  (systems/)             checkpoint, converted 322/322
  trajectory forecasting  (aerotrack_temporal)          two arms: normal · DFSC
  deployment  (src/deployment/)                             Protocol II, IR and RGB
          │                                                           │
          ▼                                                           ▼
  CST TEST mSA 60.31 vs published best 35.92            VAL −0.27 vs paper (reproduced)
  +24.39 [+21.08, +27.80], 60/60 sequences              TEST +5.04 (above the paper)
  170 FPS, 0 look-ahead                                 DFSC gain NOT observed
```

They share the corpus, the evaluation code and the workspace. **They never share a result row** —
Direction B's GlobalTrack is also Direction A's baseline, but a number produced under one
registration is never reported under the other.

---

## 5. Workspace architecture

```
results/registry/              one identity for every artifact — 314 model identities, 498 checkpoint
                       files, each with a SHA-256; 0 integrity mismatches; 228 load- and
                       inference-verified                      (scripts/build_workspace_registry.py)
src/aerolab/               the shared layer: adapters that wrap each experiment's own code, plus an
                       evaluation service restricted to TRAIN and VAL — TEST is locked
app/                the AeroTrack Flight Deck (Streamlit): 7 experiment categories, live
                       inference, frame-synchronised multi-model comparison, recorded TEST
                       replay (read-only), controlled speed benchmark      (studio.bat :8501)
experiments/<study>/   one directory per study: preregistration/ · scripts/ · reports/
scripts/hpc/             PBS submission, data and environment gates, sync, result pull
docs/                  MASTER_STATE · DECISIONS · EXPERIMENT_LEDGER · FAILURES_AND_RECOVERY
```

Three rules hold the workspace together, and they are the reason the numbers can be trusted:

- **The TEST lock.** The evaluation service reaches TRAIN and VAL only. A TEST pass needs a
  written registration, is logged in a ledger with the weights' SHA-256, and is spent once. Every
  budget is now spent.
- **Comparability.** Each checkpoint is scored on the input it was trained with — the window it
  was trained with, or three copies of frame *t* for the single-frame controls. Scoring a
  single-frame model on a window, or the reverse, would make the comparison meaningless.
- **No moves.** Artifacts are never relocated; the registry records where each one already is.
  This is why nothing in this project needed re-running after the HPC disk filled up.

---

## 6. The erratum, stated on the slide

`src/model/yolo11-stride.yaml` carries flat multipliers and no scale letter. Ultralytics 8.4.115
therefore resolves `scale = ''`, and `parse_model`'s `if scale in "mlx"` is true for the empty
string, so **every C3k2 block is built with the m/l/x inner C3k** rather than YOLO11n's
Bottleneck. The network has 2.613 M parameters and only **417 of 649 tensors (64.3 %)** of
`yolo11n.pt` load into it; the C3k2 internals of layers 2, 4, 13, 16, 19 start from random
initialisation.

**What stays valid**: every registered comparison is between configurations built the same way
from identical starting weights, so the window verdicts are internally valid. The wording "starts
as COCO YOLO11n" should read "starts as the same partially COCO-initialised network".

**What was done about it**: `src/model/yolo11-stride-std.yaml` builds standard YOLO11n (2.591 M
parameters, 448 / 499 tensors = 89.8 % loading). Its effect was measured as registered experiment
**F1** and came out **NOT SUPPORTED** — seed-mean −0.0100 CST AP50, 95 % CI [−0.0259, +0.0064],
not positive for every seed. The headline systems therefore stay on the as-built network.

Source: `experiments/aerotrack_v2/ERRATUM_architecture.md`,
`experiments/aerotrack_v3/reports/F1_standard_yolo11n.md`.

---

## 7. What was built, measured, and removed

An architecture is also what it does **not** contain. Each of these was implemented, compared
against the dumbest baseline that could work under a pre-registered rule, lost, and was retired —
the code stays in the repository so the measurement can be repeated.

| component | the architecture it would have joined | the rule it failed | what beat it |
|---|---|---|---|
| **SPD-Conv** (space-to-depth downsampling) | the five stride-2 backbone convolutions | > 0.02 AP50 on both `best.pt` and `last.pt` | no measurable effect on micro AP50 — ordinary stride-2 Conv is deployed |
| **ByteTrack** association | stage 5, instead of the STE | MOTA / IDF1 on every dataset | a no-association argmax box |
| **LSTM forecaster** | a trajectory predictor after stage 5 | ADE on CST | mean-velocity / Kalman; on gimbal footage, predicting no motion |
| **learned CST forecaster** (C1 / C1c) | the same slot, per domain | ADE vs the classical `dcv` model | the classical model |
| **learned uncertainty head** | confidence on the forecast | NLL vs Kalman | Kalman, by three orders of magnitude on CST |
| **P1 presence gate** | a confidence gate on the STE output | cross-fitted CST VAL mSA | the ungated output — the gate trades absent-frame false alarms for visible misses one for one |
| **trajectory augmentation** (H4) | the forecaster's training | ADE | no augmentation — augmentation *hurts* |

An earlier revision of this work claimed "+43 % mAP from fine-tuning", "ByteTrack occlusion
recovery" and "LSTM beats Kalman by 26 %". A controlled re-measurement overturned all three. The
independent audit that forced it is `docs/audit/AUDIT_REPORT.md`; the repair log is
`docs/AEROTRACK_REPAIR_REPORT.md`.

---

## 8. Training and compute

| | |
|---|---|
| training hardware | Amrita HPC, PBS, Tesla V100 nodes; fp16 autocast (amendment A1) |
| one run | 20 epochs, 25,000 sampled frames per epoch, batch 32, 12 dataloader workers, 0.72–0.79 h on a V100 |
| optimiser | lr0 0.001 → ×0.01, class weights [1, 1] |
| job records | 46 PBS jobs preserved with command, environment, event log and full stdout — `~/DRONE-Track/logs/pbs_jobs/` |
| evaluation | CST TEST detection cache about 150 frames/s on a V100 (including JPEG decode) |
| host agreement | the same VAL run on the V100 and the laptop: 79.28739 vs 79.28740 mSA (Δ 0.00001) |
| deployment | laptop RTX 5050 (8 GB, Blackwell sm_120), TensorRT fp16, 170 FPS end-to-end |

---

## 9. One slide, if only one is possible

> **DRONE-Track.** A drone that matters is a few pixels across, and one frame of it is almost
> noise. We give a 2.6 M-parameter YOLO11 detector a **causal window of three frames**,
> `[t−8, t−4, t]`, stacked into nine channels, and read its boxes with a single-target state
> estimator that can say *absent*. On the CST Anti-UAV benchmark — whose own paper reports the
> best of 21 published trackers at **35.92** mean state accuracy — this reaches **60.31**, a
> paired **+24.39** [+21.08, +27.80] over GlobalTrack, winning **60 of 60** sequences, scored by
> the benchmark authors' own unmodified toolkit. The window itself is worth **+11.51 mSA**, and
> removing its look-ahead costs nothing measurable, so the whole system runs on a live feed at
> **170 FPS** on a laptop GPU with **zero frames of look-ahead**. Six other components we built
> lost to a trivial baseline and were retired.
