# AeroTrack-Net repair — progress log

Branch `aerotrack-repair` (from main d1e4685). Plan: `docs/AEROTRACK_REPAIR_PLAN.md`. Namespace: `experiments/aerotrack_repair/`.
Historical runs (`results/runs/aerotrack_spd_v1`, `results/runs/aerotrack_spd_finetuned`) are never modified; new runs go to `results/runs/repair_*`.
The user's pre-existing uncommitted edits (src/model/val.py, track_eval.py, visual_demo.py, scripts/run_aerotrack.py, train.ipynb,
results/runs/aerotrack_spd_finetuned/eval_val/*) are left untouched and uncommitted.

## Inspection findings (2026-09-21, beyond the audit)
- `train.py::build()` supports `ablation` with `no_spd` (yolo11-stride.yaml == stock YOLO11n layout, 9-ch) and
  `single_frame` (temporal_gap=0). Stock-YOLO11n baseline == stride yaml + single frame + COCO transfer (new entry).
- `scripts/audit/bench_*.py` hold the correct measurements; the repair promotes them into project scripts.
- `lstm_forecaster.py`: loss = MSE on cumulative positions of near-zero-mean displacements -> collapse to zero motion.
- Throughput: 19-28 min/epoch (25k samples) -> 20-epoch run ~6.5-9 h; 40-epoch ~13-19 h.

## Done (2026-09-21)
- item 1/9/11/12 `scripts/evaluate_detector.py`: same-set val evaluation (11,565 images, set sha 6fb41d44d45b) of
  v1 best/last and v2 best/last with the corrected bird->drone metric and pixel-stated size bins ->
  `reports/detector_val_cap5000.{json,md}`. Findings: v1 LAST (ep21) > v1 BEST (ep15) on every metric; v2 < v1-last on
  drone AP / CST / Anti-UAV; v2 gains are bird AP + Det-Fly; bird-only images with a drone alarm: 20.6 % (v1 best),
  3.1 % (v1 last), 0.0 % (v2).
- item 6 `scripts/tracker_repair.py`: ByteTrack vs SORT vs argmax/greedy baselines on 10 val sequences (10,578 frames):
  argmax@0.25 MOTA 0.783 / IDF1 0.862 vs ByteTrack 0.596 / 0.309 -> **ByteTrack retired** (pre-registered rule).
- item 10 `scripts/latency_repair.py`: fp32 (deployed) 14.4 ms ≈ 69 FPS end-to-end without tracker; bf16 slower;
  +40 ms structural look-ahead at 25 FPS -> `reports/latency_aerotrack_spd_finetuned.{json,md}`.
- item 5 `scripts/lstm_repair.py`: 3 parameterisations x trivial baselines x Kalman, per domain, val selection, one test
  pass -> `results/runs/lstm_forecaster_repair/`. **LSTM retired** (loses to mean-vel/Kalman on CST; beats zero-velocity only on
  gimbal-jittered Anti-UAV). Also found: the forecaster val split has no CST windows.
- item 7 `scripts/plot_forecast_example.py`: model-output figure `docs/assets/repair_forecast_examples_val.png` (+ JSON of
  every plotted coordinate), examples chosen by median-ADE rule on val.
- item 8 `scripts/make_demo.py`: side-by-side val demo, sequence by label rule (first IR val seq with >=20 absent
  frames), no tracker/trail/magnifier, per-frame IoU JSON -> `docs/assets/repair_demo_val_<seq>.gif`.
- items 2/3/13 `scripts/train_ablation.py` + `ABLATIONS[baseline_ctrl, stock_yolo11n]` in `src/model/train.py`;
  `transfer_pretrained` tiles the 3-ch COCO stem into 9-ch stride stems (/3, bit-exact on identical frames).
  Dry-run OK (stock: 0 SPDConv, temporal_gap 0, 781 it/epoch @ batch 32, 64 % COCO transfer). NOT trained yet.

- `scripts/render_report.py` + `README.template.md`: README.md is now GENERATED from the JSONs (Review-2 README kept at
  `docs/historical/README_review2.md`); `docs/AEROTRACK_REPAIR_REPORT.md` lists the 14 plan items with status.
  Attribution rule for the grid pre-registered in the script (> 0.02 AP50 on both best.pt and last.pt).

## Done (2026-09-25 → 26): ablation grid + final test, one command
- `scripts/run_grid.py` runs everything that was pending, unattended and resumable: pre-flight → baseline_ctrl, no_spd,
  single_frame, stock_yolo11n (20 epochs, batch 32, sequential) → val eval of best.pt AND last.pt after each → TEST once
  per checkpoint (v1/v2 + the grid's 8, ledger-guarded) → `render_report.py` + `grid_report.py`. Completed 2026-09-26 09:10.
- Result (numbers only in the generated `reports/ablation_grid_e20.md`): under the pre-registered rule none of SPD-Conv,
  the temporal stack, or the full recipe vs stock YOLO11n is credited — "no measurable effect at this budget".
- Needed on the way (details in §7/§8 of that report): T6 pre-flight rejected stride-2 stems (would have aborted two runs);
  `aerotrack_collate` leaked one batch per batch per DataLoader worker (fixed: shared-memory stacking; first launch lost to
  it); single-frame runs are scored on their own t,t,t input; atomic checkpoints; one run_meta.json across resumes; the
  runner itself was once killed externally and is now started through a detached spawner and adopts orphaned steps.

## Next
- Review-3 write-up from the generated report. All code/report changes of 2026-09-25/26 are uncommitted.
