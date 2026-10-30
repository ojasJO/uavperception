# Anti-UAV reproduction — progress log

Concise, factual state of the reproduction task. Verified facts only; anything not yet verified is marked TODO.
Last update: 2026-09-21 (Phase A complete; Phase B: absence control + seed-1 runs done and evaluated on val).

## Paper identified
- File: `anti_uav_research_paper.pdf` (repo root), 13 pages, PDF created 2021-02-09.
- *Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking* (Jiang, Wang, Peng, Yu, Wang, Xing, Li, Zhao,
  Guo, Han) — arXiv:2101.08466v3, 8 Feb 2021; journal version IEEE TMM (ieeexplore 9615243).
  **Not a 2025 paper** — discrepancy with the task brief ("published in 2025") flagged to the user.
- Task in the paper: **single-object tracking** (init box on the start frame → per-frame box), metric mSA
  (state accuracy, Eq. 1) + OTB precision/success. Method: **DFSC** training strategy on **GlobalTrack**
  (query-guided Faster R-CNN, ResNet-50-FPN). The existing AeroTrack-Net (YOLO detector) is not an
  implementation of this paper.
- Official code: `ucas-vg/Anti-UAV` = README + LICENSE only ("Code coming") → DFSC code never released.
  GlobalTrack code (mmdet 1.x, PyTorch 1.1, CUDA 9) inspected file by file; not buildable here → re-implemented
  on torchvision 0.28 primitives (documented substitution S1).

## Environment (verified)
- Windows 11, RTX 5050 Laptop 8.5 GB (sm_120); `venv/Scripts/python.exe` = Python 3.11.9, torch 2.13.0+cu130,
  torchvision 0.28.0+cu130, numpy 2.4.6, opencv 5.0.0. COCO Faster R-CNN weights cached in `~/.cache/torch/hub`.

## Dataset (verified, manifests frozen)
- Official splits 160/67/91 sequences (= paper). Labels `exist` + `gt_rect` [x,y,w,h]; attributes in `label_new/`.
- Decoded frames exist for all 636 clips (frame count == label count). IR 640×512, RGB 1920×1080.
- Boxes: IR 148,368 / 60,620 / 84,221 (train/val/test); total both modalities 573,427 vs paper 585.9k (~2 %, documented).
- Cross-split: 0 name/label/frame collisions; test shares no source video with train/val; train/val share 26 videos (by paper design).
- GlobalTrack pair filter keeps 99.8 % of IR frames. Manifest SHA-256s in `experiments/anti_uav/manifests/`.
- Report: `experiments/anti_uav/reports/dataset_verification.md`.

## Completed
- [x] Repository + environment inspection; paper read completely; GlobalTrack + challenge eval code inspected
- [x] `docs/ANTI_UAV_REPRODUCTION_SPEC.md` (tagged parameters, reported results, substitutions S1–S6)
- [x] `experiments/anti_uav/reports/paper_reported_results.json` (Tables V, VI, VII + Protocol I highlights)
- [x] Implementation `experiments/anti_uav/src/anti_uav_repro/` (data guard + manifests, pair sampler,
      GlobalTrack transforms, QG-RCNN + DFSC losses, trainer, tracker, metrics, evaluation, repro utils)
- [x] Configs: `_base_ir.yaml`, `normal_ir.yaml` (α=0), `dfsc_ir.yaml` (α=1), `smoke_ir.yaml`, `phase_b/*`, `deferred/*`
- [x] Scripts: `verify_dataset.py` (run; manifests created+verified), `smoke_test.py` (**14/14 pass**), `train.py`
      (`--dry-run` verified for both Phase A configs), `evaluate.py`, `aggregate_results.py` (comparison doc generated, 0/44 cells filled)
- [x] `docs/ANTI_UAV_EXPERIMENT_PROTOCOL.md`, `docs/ANTI_UAV_COMPARISON.md` (generated), `docs/ANTI_UAV_INTEGRITY_AUDIT.md`,
      `docs/AEROTRACK_REPAIR_PLAN.md` (second task: planned only), `experiments/anti_uav/README.md`
- [x] Measured: 800 it/epoch, ≈0.58 s/it fp32, peak 3.9–6.4 GB → ≈2.5 h per 18-epoch run

## Bugs found and fixed during smoke testing (recorded so they are not reintroduced)
1. torchvision changes the *architecture* with the init source (BatchNorm2d + no freezing without weights;
   FrozenBN eps 0.0 for COCO_V1) → `build_base_faster_rcnn` builds FrozenBN + frozen stem/res2 always and the
   eps travels in the state_dict (`_frozen_bn_eps` buffer) → checkpoint round-trip is exact.
2. DataLoader workers hold a stale dataset copy → `set_epoch` did not reach them → identical pairs every epoch.
   Fixed by encoding the epoch into the sampler index (`epoch*N + i`); regression-tested in smoke check 14.
3. `--dry-run` used to create empty run dirs that then blocked the real launch → directories created lazily.
4. Windows cp1252 console could not print report symbols → scripts force UTF-8 stdout.

## Verified commands (all executed successfully in this session)
```
venv\Scripts\python.exe experiments\anti_uav\scripts\verify_dataset.py
venv\Scripts\python.exe experiments\anti_uav\scripts\smoke_test.py
venv\Scripts\python.exe experiments\anti_uav\scripts\train.py --config experiments\anti_uav\configs\dfsc_ir.yaml --dry-run
venv\Scripts\python.exe experiments\anti_uav\scripts\train.py --config experiments\anti_uav\configs\normal_ir.yaml --dry-run
venv\Scripts\python.exe experiments\anti_uav\scripts\aggregate_results.py
```

## Phase A results so far (2026-09-20)
- `dfsc_ir`: 18/18 epochs, 2h20 (git 77accee, dirty tree = user's unrelated pending edits), final ckpt sha 1e8219c56d4b…
- `normal_ir`: 18/18 epochs, 2h01, final ckpt sha ab9890d52684…
- Loss curves near-identical between the runs; DFSC cross terms ≈ same terms (~0.001) from epoch 3.
- **VAL IR (full protocol):** DFSC mSA **79.26** (paper 80.09, Δ −0.83 → reproduced); normal **79.29** (paper 79.60,
  Δ −0.31 → reproduced). Gain DFSC − normal = −0.03 (paper +0.49) → not reproduced as a gain (single seed).
- Attribute rows: TC*, LI, FM within ~1 pt; OV −22 (4 seqs), OC −3/−4 (5 seqs), SV −4: driven by val sequence
  `20190925_130434_1_2` (UAV present in 77/1000 frames; tracker never predicts absence → SA 5 %). Absence handling is
  `[INFERRED]` (none) — the paper's OV ≈ 78 implies its trackers did output absence (or annotations differed).
  → Phase B additional control: post-hoc score-threshold absence rule on saved predictions (val-chosen τ).
- Incident: the dfsc_ir val evaluation was accidentally launched twice; the duplicate was stopped at 15/67 after
  overwriting 15 prediction files. Folder re-scored from files (`--predictions`); mSA identical (79.2559);
  re-tracked sequences differ from the first pass by ≤ 5.5e-5 SA (cuDNN noise) — recorded as a determinism check.
- Tracking speed ≈ 9.5–11 fps → val ≈ 1.6–1.8 h, test ≈ 2–2.5 h per model.

- **TEST IR (full protocol, once each, ledger entries 1–2):** DFSC mSA **70.29** (paper 66.04, Δ +4.25 → not reproduced,
  above); normal **70.58** (paper 65.36, Δ +5.22 → not reproduced, above). Gain −0.29 (paper +0.68). Precision/success
  91.85/71.01 (normal) vs paper Table VII 87.61/62.88. Attribute pattern: LR +17, LI +16, FM +10, SV +9, TC-mid/hard +7;
  OV and TC-easy within 1 pt. Released labels are frame-accurate (no 25-frame interpolation signature).
- **Phase B control (post-hoc absence rule, τ chosen on val = 0.3 for both):** val +1.50/+1.51, test +0.42/+0.29
  (ledger entries 3–4, flagged phase B). Explains the val OV deficit, not the test surplus.
- Discrepancy analysis written as a template (`reports/discrepancy_analysis.md.tmpl`) rendered into
  `docs/ANTI_UAV_COMPARISON.md` by `aggregate_results.py` (hypotheses H1–H7; H1 annotation version most consistent;
  H3 optimisation regime and H4 input scale are the testable ones).

- **Phase B seed-1 runs (2026-09-20/21):** `phaseB_dfsc_ir_seed1` (18/18, 2h20, ckpt dc0b3e643b53…) val mSA **79.43**;
  `phaseB_normal_ir_seed1` (18/18, 2h04, ckpt 414ef0164e2b…) val mSA **79.36**. Seed spread: normal 0.08, DFSC 0.18.
  Gain of seed-means +0.02 (paired −0.03, +0.07) vs paper +0.49 → DFSC effect not observed (n = 2 seeds/arm, val).
  Test not evaluated for the seed-1 models (optional; would add 2 ledger entries, ~2.3 h each).
- `aggregate_results.py` now renders a seed-variance section; discrepancy template updated with seed placeholders.

- **Demo (2026-09-21, commit 4f518ff on branch aerotrack-repair):** `experiments/anti_uav/demo/` — Streamlit,
  VALIDATION ONLY, saved predictions of all 4 runs, per-frame IoU/CE/score, sequence SA (== metrics.json), full-val mSA vs
  paper Table V, compare mode, optional live tracking (cache under demo/cache/). `test_demo.py` 9/9 checks; server launch verified.

## NOT done (by instruction / by dependency)
- Optional: seed-1 TEST evaluations (once each) to put a seed spread on the test gain.
- Phase B H3 (`phase_b/dfsc_ir_lr0.02_accum8.yaml`) and H4 (native-resolution evaluation of the same checkpoints; script TODO).
- Visible-modality runs and the "large-scale" baseline — need the GlobalTrack mmdet checkpoint + conversion (S6).
- AeroTrack-Net repair — planned in `docs/AEROTRACK_REPAIR_PLAN.md`, not started.

## Important decisions
- IR Protocol II is the faithful target; visible deferred (init dependency).
- Single GPU, batch 2, LR 0.0025 (= 0.02 × 2/16); `accumulate_steps` exposed; sensitivity run in Phase B.
- Final-epoch checkpoint is the model; monitor line is informational only.
- Primary mSA over frames ≥ official start frame; all-frames variant reported too.
- Verdict thresholds pre-registered in `aggregate_results.py` (1 pt / 3 pt).

## Unresolved
- Whether the paper's authors used GlobalTrack's `pairs_per_seq=10` epoch definition (1600 pairs/epoch) — assumed.
- Absence prediction at inference (none, as GlobalTrack) — assumed from "trackers used without modification".
- The paper is 2021, not 2025 — the user must confirm this is the intended paper.

## Next action
1. Optional GPU work in priority order: (a) seed-1 test evals, (b) H4 native-resolution evaluation of existing checkpoints (script TODO), (c) H3 `dfsc_ir_lr0.02_accum8` run + val + test.
2. Otherwise: Review 3 write-up from the docs; AeroTrack repair per `docs/AEROTRACK_REPAIR_PLAN.md`.
3. Review 3 write-up from `docs/ANTI_UAV_COMPARISON.md` + spec + protocol; AeroTrack repair per `docs/AEROTRACK_REPAIR_PLAN.md`.
