# AeroTrack-Net controlled ablation grid — final report (20 epochs)

*Generated 2026-09-26 09:17 by `experiments/aerotrack_repair/scripts/grid_report.py` from run_meta.json / args.json / results.csv, `reports/detector_val_cap5000.json`, `reports/detector_test_cap5000.json`, the test ledger and `results/runs/repair_grid_e20/` (events, pre-flight). No number in this file is typed by hand. Machine-readable twin: `ablation_grid_e20.json`.*

## Outcome

- **Workflow: COMPLETE** — 4/4 runs finished, 8/8 grid checkpoints evaluated on validation, 8/8 grid checkpoints (+ 4 historical) evaluated once on test.
- **Pre-registered rule:** a factor is credited only if the full model (`baseline_ctrl`) beats the ablated model by > 0.02 AP50 on the target metric for BOTH best.pt and last.pt, on the validation set; otherwise it has no measurable effect at this budget.
- SPD-Conv: no measurable effect on micro AP50 and CST micro AP50
- temporal stack: no measurable effect on CST AP50 and drone AP50
- full recipe vs stock YOLO11n: no measurable effect on drone AP50

## 1. Training status

| run | factor changed | input | status | epochs | launches | training time (sum of launches) | wall clock | median train time / epoch | workers used | COCO transfer | best.pt (epoch, capped-val fitness) | last.pt (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `repair_abl_baseline_ctrl_e20` | full recipe (SPD + 9-ch stack) | t−1,t,t+1 | completed | 20/20 | 2 | 2.14 h | 2.39 h | 5.7 min | 4 | 388/649 (59.8 %) | epoch 13 (0.4578) | epoch 19 |
| `repair_abl_no_spd_e20` | SPDConv → stride-2 Conv | t−1,t,t+1 | completed | 20/20 | 1 | 2.45 h | 2.46 h | 6.4 min | 4 | 418/649 (64.4 %) | epoch 10 (0.4948) | epoch 19 |
| `repair_abl_single_frame_e20` | t−1 = t = t+1 (no motion) | t,t,t | completed | 20/20 | 1 | 1.89 h | 1.89 h | 5.1 min | 4 | 388/649 (59.8 %) | epoch 19 (0.4436) | epoch 19 |
| `repair_abl_stock_yolo11n_e20` | stride-2 AND single frame (stock YOLO11n) | t,t,t | completed | 20/20 | 1 | 1.86 h | 1.86 h | 5.1 min | 4 | 418/649 (64.4 %) | epoch 18 (0.6160) | epoch 19 |

Checkpoints (sha256 prefix): `results\runs\repair_abl_baseline_ctrl_e20\weights\best.pt` 815ebed2d9a2, `results\runs\repair_abl_baseline_ctrl_e20\weights\last.pt` 2e02ebc5ef23; `results\runs\repair_abl_no_spd_e20\weights\best.pt` 084f1dd8f0f6, `results\runs\repair_abl_no_spd_e20\weights\last.pt` 2aebbb3bcb6c; `results\runs\repair_abl_single_frame_e20\weights\best.pt` deccf5e70105, `results\runs\repair_abl_single_frame_e20\weights\last.pt` d6138d094438; `results\runs\repair_abl_stock_yolo11n_e20\weights\best.pt` 07744c380618, `results\runs\repair_abl_stock_yolo11n_e20\weights\last.pt` eda773e9c1f9

**Training configuration (identical for all four runs; read from `args.json`)** — stage `baseline` recipe, 20 epochs × 25,000 sampled frames, batch 32 × accumulate 2, AdamW lr0 0.001 → ×0.01 (cosine, 3.0 warm-up epochs), weight decay 0.0005, bf16 autocast, EMA True, grad clip 10.0, image 640, seed 0, per-epoch validation on the capped val set at conf 0.005 (checkpoint selection only; fitness weights [0.3, 0.35, 0.35]), patience 30. Sampler weights {'bird': 8.0, 'micro': 6.0, 'small': 3.0, 'standard': 1.0, 'empty': 0.5}; augmentation (TemporalAugment defaults) {'hflip': 0.5, 'scale': (0.9, 1.25), 'translate': 0.1, 'brightness': 0.15, 'contrast': 0.15, 'copy_paste': 0.3, 'copy_paste_max': 3, 'noise_std': 0.0, 'gray_p': 0.0}.
`args.json` keys that differ between the four runs: ['name'] — the run name; none of these changes what is trained.
Factor per run: `no_spd` swaps the five SPDConv downsamplers for stride-2 convolutions (`yolo11-stride.yaml`); `single_frame` feeds t,t,t instead of t−1,t,t+1; `stock_yolo11n` does both. COCO initialisation uses the same rule everywhere (tensors whose name and shape match; a 3-channel stem is tiled /3 into a 9-channel stride stem).

Pre-flight on each exact job (train.py T1–T8, nothing trained): baseline_ctrl: yolo11-spd.yaml, 5 SPDConv, first conv 36 ch, 781 it/epoch, VRAM peak 4.77 GB (headroom 2.0 GB); no_spd: yolo11-stride.yaml, 0 SPDConv, first conv 9 ch, 781 it/epoch, VRAM peak 4.76 GB (headroom 2.02 GB); single_frame: yolo11-spd.yaml, 5 SPDConv, first conv 36 ch, 781 it/epoch, VRAM peak 4.77 GB (headroom 2.01 GB); stock_yolo11n: yolo11-stride.yaml, 0 SPDConv, first conv 9 ch, 781 it/epoch, VRAM peak 4.76 GB (headroom 2.02 GB).

## 2. Validation — every run, both checkpoints

**Validation** — image set `6fb41d44d45b`: 11,565 images (anti_uav 5,000, cst 5,000, det_fly 1,565), 1714 sequences. Every row uses exactly these images.

| run | ckpt | epoch | input | mAP50 | mAP50-95 | drone AP50 | bird AP50 | micro AP50 (n GT) | small | standard | Anti-UAV | CST | CST micro | Det-Fly | bird-only imgs raising a drone alarm | GT birds boxed as drone | empty-frame drone FP / img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aerotrack_spd_v1 | best | 15 | t-1,t,t+1 | 0.6074 | 0.2991 | 0.7925 | 0.4224 | 0.2929 (2008) | 0.6240 | 0.4932 | 0.9775 | 0.6767 | 0.6594 | 0.3974 | 20.6 % | 0.8 % | 0.0704 |
| aerotrack_spd_v1 | last | 21 | t-1,t,t+1 | 0.6505 | 0.3188 | 0.8286 | 0.4724 | 0.3293 (2008) | 0.6602 | 0.5358 | 0.9813 | 0.7420 | 0.6846 | 0.4454 | 3.1 % | 0.2 % | 0.0778 |
| aerotrack_spd_finetuned | best | 23 | t-1,t,t+1 | 0.7788 | 0.4140 | 0.8027 | 0.7548 | 0.4078 (2008) | 0.7933 | 0.8454 | 0.9534 | 0.6392 | 0.6091 | 0.7388 | 0.0 % | 0.0 % | 0.0869 |
| aerotrack_spd_finetuned | last | 24 | t-1,t,t+1 | 0.7799 | 0.4158 | 0.8049 | 0.7549 | 0.4062 (2008) | 0.7943 | 0.8547 | 0.9533 | 0.6436 | 0.6112 | 0.7407 | 0.0 % | 0.0 % | 0.0877 |
| repair_abl_baseline_ctrl_e20 | best | 13 | t-1,t,t+1 | 0.6052 | 0.3009 | 0.7791 | 0.4313 | 0.2790 (2008) | 0.6170 | 0.5105 | 0.9751 | 0.6374 | 0.6187 | 0.4095 | 4.1 % | 0.1 % | 0.1200 |
| repair_abl_baseline_ctrl_e20 | last | 19 | t-1,t,t+1 | 0.6409 | 0.3233 | 0.8030 | 0.4788 | 0.2804 (2008) | 0.6585 | 0.6494 | 0.9816 | 0.6677 | 0.6269 | 0.4650 | 4.1 % | 0.1 % | 0.1018 |
| repair_abl_no_spd_e20 | best | 10 | t-1,t,t+1 | 0.6279 | 0.3055 | 0.7975 | 0.4582 | 0.2949 (2008) | 0.6452 | 0.4780 | 0.9718 | 0.7098 | 0.6579 | 0.3908 | 16.5 % | 0.4 % | 0.2409 |
| repair_abl_no_spd_e20 | last | 19 | t-1,t,t+1 | 0.6996 | 0.3453 | 0.8415 | 0.5576 | 0.3588 (2008) | 0.7114 | 0.5387 | 0.9815 | 0.7621 | 0.7288 | 0.5163 | 4.1 % | 0.2 % | 0.1316 |
| repair_abl_single_frame_e20 | best | 19 | t,t,t | 0.6533 | 0.3244 | 0.8065 | 0.5002 | 0.2873 (2008) | 0.6707 | 0.5609 | 0.9807 | 0.6825 | 0.6119 | 0.4559 | 7.2 % | 0.1 % | 0.0778 |
| repair_abl_single_frame_e20 | last | 19 | t,t,t | 0.6533 | 0.3244 | 0.8065 | 0.5002 | 0.2873 (2008) | 0.6707 | 0.5609 | 0.9807 | 0.6825 | 0.6119 | 0.4559 | 7.2 % | 0.1 % | 0.0778 |
| repair_abl_stock_yolo11n_e20 | best | 18 | t,t,t | 0.6708 | 0.3371 | 0.8337 | 0.5079 | 0.3182 (2008) | 0.6837 | 0.5894 | 0.9820 | 0.7305 | 0.7078 | 0.5208 | 1.0 % | 0.0 % | 0.1167 |
| repair_abl_stock_yolo11n_e20 | last | 19 | t,t,t | 0.6790 | 0.3415 | 0.8379 | 0.5200 | 0.3184 (2008) | 0.6937 | 0.5879 | 0.9828 | 0.7327 | 0.7091 | 0.5336 | 3.1 % | 0.1 % | 0.1126 |

v1/v2 rows are the historical models, shown for reference; the grid runs are compared only with each other.

## 3. Comparison with the full model (pre-registered rule, validation)

| factor | ablated run | target metric | full best.pt | ablated best.pt | Δ best | full last.pt | ablated last.pt | Δ last | > 0.02 on both? | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| SPD-Conv | no_spd | micro AP50 | 0.2790 | 0.2949 | -0.0159 | 0.2804 | 0.3588 | -0.0784 | no | no measurable effect |
| SPD-Conv | no_spd | CST micro AP50 | 0.6187 | 0.6579 | -0.0392 | 0.6269 | 0.7288 | -0.1019 | no | no measurable effect |
| temporal stack | single_frame | CST AP50 | 0.6374 | 0.6825 | -0.0451 | 0.6677 | 0.6825 | -0.0148 | no | no measurable effect |
| temporal stack | single_frame | drone AP50 | 0.7791 | 0.8065 | -0.0274 | 0.8030 | 0.8065 | -0.0035 | no | no measurable effect |
| full recipe vs stock YOLO11n | stock_yolo11n | drone AP50 | 0.7791 | 0.8337 | -0.0547 | 0.8030 | 0.8379 | -0.0349 | no | no measurable effect |

## 4. What the measurements support

- **SPD-Conv — not supported on micro AP50** (Δ -0.0159 best.pt, -0.0784 last.pt): the ablated model scores higher on both checkpoints. Under the rule this is 'no measurable effect at this budget'. The rule only credits a factor; it defines no verdict for a difference in the other direction, so none is drawn from this single-seed comparison.
- **SPD-Conv — not supported on CST micro AP50** (Δ -0.0392 best.pt, -0.1019 last.pt): the ablated model scores higher on both checkpoints. Under the rule this is 'no measurable effect at this budget'. The rule only credits a factor; it defines no verdict for a difference in the other direction, so none is drawn from this single-seed comparison.
- **temporal stack — not supported on CST AP50** (Δ -0.0451 best.pt, -0.0148 last.pt): the ablated model scores higher on both checkpoints. Under the rule this is 'no measurable effect at this budget'. The rule only credits a factor; it defines no verdict for a difference in the other direction, so none is drawn from this single-seed comparison.
- **temporal stack — not supported on drone AP50** (Δ -0.0274 best.pt, -0.0035 last.pt): the ablated model scores higher on both checkpoints. Under the rule this is 'no measurable effect at this budget'. The rule only credits a factor; it defines no verdict for a difference in the other direction, so none is drawn from this single-seed comparison.
- **full recipe vs stock YOLO11n — not supported on drone AP50** (Δ -0.0547 best.pt, -0.0349 last.pt): the ablated model scores higher on both checkpoints. Under the rule this is 'no measurable effect at this budget'. The rule only credits a factor; it defines no verdict for a difference in the other direction, so none is drawn from this single-seed comparison.

Scope: one seed per configuration and a 20-epoch budget, so a difference inside the margin cannot be separated from run-to-run noise; the verdicts are about this budget, not about convergence.

## 5. Final test — evaluated once per checkpoint

**Test (final, once per model)** — image set `37e944793773`: 11,567 images (anti_uav 5,000, cst 5,000, det_fly 1,567), 1809 sequences. Every row uses exactly these images.

| run | ckpt | epoch | input | mAP50 | mAP50-95 | drone AP50 | bird AP50 | micro AP50 (n GT) | small | standard | Anti-UAV | CST | CST micro | Det-Fly | bird-only imgs raising a drone alarm | GT birds boxed as drone | empty-frame drone FP / img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aerotrack_spd_v1 | best | 15 | t-1,t,t+1 | 0.5491 | 0.2428 | 0.6590 | 0.4391 | 0.2049 (2722) | 0.5693 | 0.4997 | 0.9015 | 0.4471 | 0.3950 | 0.4145 | 18.6 % | 0.7 % | 0.2696 |
| aerotrack_spd_v1 | last | 21 | t-1,t,t+1 | 0.5858 | 0.2629 | 0.6861 | 0.4855 | 0.2543 (2722) | 0.6007 | 0.5630 | 0.9064 | 0.4876 | 0.4236 | 0.4613 | 0.0 % | 0.0 % | 0.4700 |
| aerotrack_spd_finetuned | best | 23 | t-1,t,t+1 | 0.7344 | 0.3776 | 0.6745 | 0.7944 | 0.2862 (2722) | 0.7678 | 0.9079 | 0.8787 | 0.3933 | 0.3174 | 0.7685 | 2.1 % | 0.1 % | 0.4349 |
| aerotrack_spd_finetuned | last | 24 | t-1,t,t+1 | 0.7339 | 0.3774 | 0.6755 | 0.7924 | 0.2877 (2722) | 0.7676 | 0.9073 | 0.8816 | 0.3931 | 0.3182 | 0.7668 | 2.1 % | 0.1 % | 0.4377 |
| repair_abl_baseline_ctrl_e20 | best | 13 | t-1,t,t+1 | 0.5396 | 0.2402 | 0.6557 | 0.4234 | 0.1726 (2722) | 0.5636 | 0.5429 | 0.9012 | 0.4245 | 0.3399 | 0.4017 | 5.2 % | 0.1 % | 0.3878 |
| repair_abl_baseline_ctrl_e20 | last | 19 | t-1,t,t+1 | 0.5693 | 0.2584 | 0.6607 | 0.4779 | 0.1641 (2722) | 0.5982 | 0.6250 | 0.9087 | 0.4118 | 0.3290 | 0.4692 | 4.1 % | 0.0 % | 0.6362 |
| repair_abl_no_spd_e20 | best | 10 | t-1,t,t+1 | 0.5567 | 0.2515 | 0.6528 | 0.4607 | 0.2033 (2722) | 0.5723 | 0.5111 | 0.8992 | 0.4365 | 0.3422 | 0.4026 | 15.5 % | 0.6 % | 0.6953 |
| repair_abl_no_spd_e20 | last | 19 | t-1,t,t+1 | 0.6124 | 0.2776 | 0.6679 | 0.5570 | 0.2129 (2722) | 0.6260 | 0.4870 | 0.9175 | 0.4384 | 0.3353 | 0.5293 | 11.3 % | 0.6 % | 0.7091 |
| repair_abl_single_frame_e20 | best | 19 | t,t,t | 0.5838 | 0.2642 | 0.6740 | 0.4937 | 0.2045 (2722) | 0.6064 | 0.5641 | 0.9012 | 0.4672 | 0.4055 | 0.4688 | 5.2 % | 0.0 % | 0.5891 |
| repair_abl_single_frame_e20 | last | 19 | t,t,t | 0.5838 | 0.2642 | 0.6740 | 0.4937 | 0.2045 (2722) | 0.6064 | 0.5641 | 0.9012 | 0.4672 | 0.4055 | 0.4688 | 5.2 % | 0.0 % | 0.5891 |
| repair_abl_stock_yolo11n_e20 | best | 18 | t,t,t | 0.5778 | 0.2692 | 0.6738 | 0.4819 | 0.2147 (2722) | 0.5860 | 0.6019 | 0.9078 | 0.4508 | 0.3987 | 0.5190 | 1.0 % | 0.1 % | 0.5476 |
| repair_abl_stock_yolo11n_e20 | last | 19 | t,t,t | 0.5913 | 0.2756 | 0.6837 | 0.4988 | 0.2131 (2722) | 0.6061 | 0.6152 | 0.9105 | 0.4561 | 0.3996 | 0.5345 | 2.1 % | 0.1 % | 0.5503 |

Ledger `reports/test_evaluation_ledger.jsonl`: 12 entries for 12 distinct checkpoints (one per checkpoint), test set `37e944793773`, first 2026-09-26T08:44:40+0530, last 2026-09-26T09:10:28+0530. The test split was scored only after all four runs and their validation evaluations had finished, and it is not used for any choice or verdict.

Descriptive only — the same full-minus-ablated differences on TEST (the rule is applied on validation; nothing is credited here):

| factor | metric | Δ best.pt (test) | Δ last.pt (test) |
|---|---|---|---|
| SPD-Conv | micro AP50 | -0.0308 | -0.0488 |
| SPD-Conv | CST micro AP50 | -0.0023 | -0.0063 |
| temporal stack | CST AP50 | -0.0427 | -0.0554 |
| temporal stack | drone AP50 | -0.0182 | -0.0133 |
| full recipe vs stock YOLO11n | drone AP50 | -0.0180 | -0.0231 |

## 6. Generated artefacts

- this report: `experiments/aerotrack_repair/reports/ablation_grid_e20.md` and `.json`
- validation / test tables: `experiments/aerotrack_repair/reports/detector_val_cap5000.{json,md}`, `detector_test_cap5000.{json,md}`; per-checkpoint reports + PR / size plots in `results/runs/<run>/eval_repair_<split>_cap5000/<ckpt>/`
- test ledger: `experiments/aerotrack_repair/reports/test_evaluation_ledger.jsonl`
- regenerated `README.md` and `docs/AEROTRACK_REPAIR_REPORT.md` (`scripts/render_report.py`)
- runs: `results/runs/repair_abl_<ablation>_e20/` (run_meta.json, args.json, results.csv, training_curves.png; weights are git-ignored)
- grid runner: `results/runs/repair_grid_e20/` — events.jsonl, grid_state.json, preflight_*.json, provenance/ (code diff + copies of untracked sources), logs/ (every step's full log; git-ignored by the repository's `*.log` rule)

## 7. Errors, interruptions and recoveries

- pre-flight (preflight_20260926_091703.json): passed; no warnings
- grid runner launches: 6 (2026-09-25T23:24:34+0530 pre-flight only, 2026-09-25T23:25:07+0530 pre-flight only, 2026-09-25T23:29:44+0530, 2026-09-25T23:44:35+0530, 2026-09-26T01:56:36+0530, 2026-09-26T09:13:00+0530)
- 2026-09-25T23:24:46+0530 `preflight` ok: False, record: preflight_20260925_232446.json
- 2026-09-25T23:33:32+0530 `train_failed` run: repair_abl_baseline_ctrl_e20, launch: 1, rc: 1, kind: host_memory, detail: LoaderWorkerDied, epochs_before: 0, epochs_after: 0, log: train_baseline_ctrl_launch1_20260925_232951.log
- 2026-09-25T23:33:32+0530 `train_recover` run: repair_abl_baseline_ctrl_e20, action: resume after 180s, workers: 2, kind: host_memory
- 2026-09-25T23:36:32+0530 `stopped` reason: STOP requested
- 2026-09-25T23:52:52+0530 `operator_note` text: Operator stop of the first baseline_ctrl launch (23:33-23:36): its 4 DataLoader workers grew ~1.7 GB private per 30 s each and the machine reached 98.6 % RAM; the OS killed a worker at 23:33:32 (event above). Reproduced in isolation: training-config loader, 1 worker, +1.25 GB per 10 batches (one batch retained per batch); dataset alone in one process: flat. Fix: src/pipeline/stacked_dataset.py::aerotrack_collate stacks into shared memory inside workers (bit-exact vs the old stack; 4 workers flat at ~1.7 GB over 60 batches). A loader-memory probe was added to the pre-flight; the grid was relaunched with the same command at 23:44 and baseline_ctrl restarted from scratch (no epoch had finished).
- 2026-09-25T23:53:01+0530 `operator_note` text: The pre-flight at 23:24:46 (ok: False) failed only its 'no other GPU job' check: on Windows the venv python.exe is a launcher whose child is the real interpreter, and the check flagged the runner's own launcher (its parent) as a foreign job. Fixed in run_grid.py (ancestors count as the runner's own tree) before any training; the next pre-flight passed every check.
- 2026-09-26T01:56:15+0530 `operator_note` text: The runner started at 23:44:35 (pid 16996) was killed externally between 01:50:50 and 01:51:52 (no traceback, no event; its console log simply stops). Its training child kept running and finished baseline_ctrl cleanly at 01:53:32 (20/20 epochs, rc 0, recorded by train_ablation.py). Most likely the agent tool that launched the runner cleaned up the process it had started. Mitigations: --detach now starts the runner through a short-lived spawner (never a process of the calling shell), and a relaunched runner first waits for any still-running step of an earlier runner (orphan adoption). Relaunched with the same command; baseline_ctrl is verified from its files, not retrained.
- `repair_abl_baseline_ctrl_e20` needed 2 launches: #1 rc 1 epochs 0→0; #2 rc 0 epochs 0→20

## 8. What changed in this task

- `src/model/preflight.py` — T6 required a 36-channel first convolution (the SPD stem). The stride-2 stem of the `no_spd` and `stock_yolo11n` layouts has 9, so both runs would have aborted before epoch 0. T6 now takes the expected width from the live model (and still requires 9 or 36, so a reload that fell back to 3 channels still fails).
- `src/pipeline/stacked_dataset.py::aerotrack_collate` — inside a DataLoader worker the batch is now stacked straight into shared memory, as torch's own `default_collate` does. On this machine the old stack-then-share path left every batch behind in the worker (+118 MB per batch per worker, measured); the first launch of `baseline_ctrl` ran the machine out of memory 3.5 min in and lost a worker. Verified bit-exact against the old stack, flat memory at 4 workers; a loader-memory probe now guards it in the pre-flight.
- `src/model/aerotrack_trainer.py` — checkpoints are written to a temporary file and renamed atomically, so an interruption can never leave a truncated `last.pt` (the resume point).
- `src/model/train.py` — `--preflight-only`: build the exact job and run T1–T8, train nothing (used to check all four jobs up front).
- `scripts/train_ablation.py` — a resumed run keeps ONE `run_meta.json` (first start time, one `attempts` entry per launch, summed training time, workers used) and refuses a changed recipe; `--resume-from`; `--preflight-only`.
- `scripts/evaluate_detector.py` — runs trained with temporal_gap 0 (`single_frame`, `stock_yolo11n`) are scored on t,t,t, the input they were trained and validated on (`src/model/train.py::build`), instead of the 3-frame stack; the image set and its hash are unchanged and each row records its `input`. The summary is written after every checkpoint and before the test-ledger line; `--smoke`.
- `scripts/render_report.py`, `README.template.md` — grid runs in the run-facts table, an `input` column, verdicts only from finished, hash-matched runs at one budget (rule and target metrics unchanged), plan items 2–4 computed from the grid, a test-ledger note, the one-command reproduction, and a corrected sentence on which COCO tensors are transferred.
- new `scripts/run_grid.py` (the one command: pre-flight, sequential training with resume/watchdog, evaluations, test once, reports) and `scripts/grid_report.py` (this report).
- Not changed: metrics, splits, image sets, caps, thresholds, the training recipe, the epoch budget, the checkpoint-selection rule, the attribution rule, the COCO-transfer rule.

## 9. Reproduce

```bat
venv\Scripts\python.exe experiments\aerotrack_repair\scripts\run_grid.py
```

It re-verifies instead of re-running: finished runs are checked by checkpoint hash, evaluated checkpoints by their rows, and a checkpoint with a test-ledger line is never scored on test again. To retrain from scratch, move `results/runs/repair_abl_*_e20/` aside first.
