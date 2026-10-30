# Generalization investigation — progress log

Concise milestones only; every number lives in `reports/` or `ledger.jsonl`. Credit/usage balance is not observable from the
agent, so compute is budgeted by wall-clock (plan time guards) and by keeping agent turns few while jobs run.

| when | milestone | learned / changed | next, and why |
|---|---|---|---|
| 2026-09-26 09:30–10:20 | Phase A: diagnostics D1–D5 (no training; test untouched) | init confound 48 % vs 94 % pretrained; ~0.5 px/frame CST motion at g = 1; VAL optimistic (session overlap, Det-Fly adjacency); micro metric two-class; sampling CIs ±0.1 on CST | build a fair, fold-based protocol |
| 2026-09-26 10:20–11:00 | Phase B engineering: folds, function-preserving init (runtime-checked), train/eval/analysis/report scripts, runner | smoke-tested end to end incl. kill-and-resume; metrics reproduce the registered values; 3 bugs found and fixed in the smoke (recursion in env wrapper, smoke-ledger path, f-string) | launch Stage A (fold 0 × 4 configs, ≈ 10 GPU-h) |
| 2026-09-26 18:18 | Stage A done (fold 0 x 4 configs, 7.4 h incl. evals) | all three candidates ahead of stock on fold 0 (reports/gen_decisions.json): spd_fp +0.015, temporal_fp_g1 +0.010, temporal_fp_g4 +0.045 CST drone AP50; single fold = not evidence | Stage B: folds 1-2 for stock + 3 survivors (8 runs, ~14.8 h); H4 will not fit the 24 h guard |
| 2026-09-27 09:35 | Stage B done (folds 1-2 x 4 configs); decisions by the frozen rule | temporal_fp_g4 SUPPORTED (+0.045, CI [+0.021, +0.072], Anti-UAV -0.002); spd_fp +0.014 and temporal_fp_g1 +0.009 not supported (mean <= 0.02, CI includes 0) | H4 (stock 40 vs 20 epochs) skipped by the 24 h guard (needs 4.85 h); final candidate = temporal_fp_g4 |
| 2026-09-27 11:55 | final retrain on full TRAIN (seed 0, 1 launch, 2.21 h) + the single TEST evaluation | TEST (37e944793773): CST 0.521, Anti-UAV 0.919, pooled 0.718; integrity of the repair series verified at start and end (31 files) | no further test use; descriptive analysis only |
| 2026-09-28 16:35 | A1 final analysis (no new inference) + GEN_REPORT.md | test bootstrap CIs (CST [0.442, 0.597]); CST unseen scene types 0.372 vs seen 0.617; stock YOLO11n's single test run is 0.456 CST (unpaired reference) | campaign closed; next steps in the report section 15 |
