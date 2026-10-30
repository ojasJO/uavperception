# Anti-UAV paper benchmark run — 2026-09-22T19:38:53+0530

An immutable benchmark execution of our trained Anti-UAV (DFSC/GlobalTrack) models against the
protocols of `anti_uav_research_paper.pdf` (arXiv:2101.08466v3). Nothing in here is hand-edited; every
number is produced by a script in `experiments/anti_uav/scripts/` and every figure is drawn from these
files.

Read the assembled result at **`experiments/anti_uav/reports/PAPER_BENCHMARK_RESULTS.md`**.
The inventory of what was and was not run is **`reports/benchmark_execution_plan.md`**.

> **Documentation correction, 2026-09-23 (post-shutdown reconciliation).** This README was hand-edited to fix the
> `ablation/` and `seed_variance/` rows below. No experimental artifact in this directory was changed. Authoritative
> summary: `experiments/anti_uav/reports/FINAL_PROTOCOL_II_IR_BENCHMARK_REPORT.md`.

## What was executed

| Stage | Contents | Training? | Status |
|---|---|---|---|
| `normal_ir__{val,test}_IR/`, `dfsc_ir__{val,test}_IR/` | the four primary, **authorised** benchmark evaluations (Table V IR rows, Table VII IR rows) | no — frozen checkpoints | **complete, independently verified**; both TEST runs valid final evaluations (ledger lines 5–6) |
| `seed_variance/` | the same two configurations at seed 1, TEST only, so the DFSC effect can be read against run-to-run noise | no — checkpoints trained 2026-09-20/21 | *[corrected]* complete and verified, but **supplementary / unscoped**: outside the four-run scope, no authorisation record found (ledger lines 7–8) |
| `ablation/` | Table VI variant `DFSC-all` (`dfsc.ism_modulation: cross`) | **yes** | *[corrected; previously "trained … then evaluated on TEST once"]* DFSC-all training was completed, but its test evaluation was interrupted before completion and therefore produces no valid final test benchmark result: 54/91 prediction files, no `metrics.json`, no `report.md`, no ledger entry. Kept as an **incomplete/aborted artifact**. Outside the authorised scope; no authorisation record found. |

Each evaluation directory holds `predictions/<sequence>.json` (raw per-frame boxes and scores),
`metrics.json`, `eval_meta.json` (checkpoint SHA-256, manifest SHA-256, git commit, environment, GPU,
command, timing) and `report.md`.

## Assembled artifacts in this directory

| File | Contents |
|---|---|
| `benchmark_results.json` / `.md` | paper-vs-ours comparison for every metric and attribute |
| `mSA_by_attribute.csv` | the attribute table in long format |
| `precision_success.json` | Table VII comparison |
| `per_sequence_metrics.json` | per-sequence metrics for the four primary evaluations |
| `benchmark_run_metadata.json` | provenance of the four primary evaluations |
| `absence_diagnostic.json` | decomposition of Eq. (1) into its exist / absent terms |
| `metric_independent_audit.json` | from-scratch re-computation of every metric, and its diff vs the reported values |
| `curve_data.json` | the plotted OTB success / precision curves, as numbers |
| `_status.jsonl`, `_driver.log`, `*.log` | exit codes and stdout of every run |

## Reproducing it

```bash
bash run_benchmarks.sh          # the four primary evaluations   (~7.9 h GPU)
bash run_seed_variance.sh       # seed-1 TEST evaluations        (~4.2 h GPU)   [supplementary / unscoped]
bash run_ablation_dfsc_all.sh   # DFSC-all: train + TEST eval    (~4.6 h GPU)   [unscoped; the TEST eval never completed]

venv/Scripts/python.exe experiments/anti_uav/scripts/verify_metrics_independently.py --bench-dir <this dir>
venv/Scripts/python.exe experiments/anti_uav/scripts/make_curve_figures.py          --bench-dir <this dir>
venv/Scripts/python.exe experiments/anti_uav/scripts/make_paper_benchmark_report.py --bench-dir <this dir>
venv/Scripts/python.exe experiments/anti_uav/scripts/make_final_paper_report.py     --bench-dir <this dir>
```

## Rules observed

- TEST was used only to evaluate finished, frozen models — never for training, tuning, thresholding,
  checkpoint selection or any debugging decision. Every TEST evaluation went through
  `evaluate.py --final-eval` and is recorded in the append-only ledger
  `reports/test_evaluation_ledger.jsonl`.
- No earlier result was modified or overwritten; the pre-existing `runs/*/eval_*` directories are
  untouched.
- No model was retrained to improve a score, and no sequence, label or attribute was excluded or
  altered.
- The `DFSC-all` ablation was trained *after* the primary benchmark was complete and reported; its
  purpose is to test the implementation, not to change any number above. *[corrected]* Its TEST evaluation was
  interrupted at 54/91 sequences by an unexpected machine shutdown (2026-09-23 ~20:04), so it produced no result.
