# Phase B — controlled improvements / sensitivity runs (NOT part of the faithful reproduction)

Each config here changes exactly ONE factor relative to `../dfsc_ir.yaml` (or `../normal_ir.yaml`) and says
which. Run them only after Phase A is complete and evaluated. Results go into separate run directories and
are reported in separate columns ("OUR IMPROVEMENT" / "ADDITIONAL CONTROL") — never merged with Phase A.

| config | single factor changed | why |
|---|---|---|
| `dfsc_ir_alpha0.25.yaml` | alpha 1 -> 0.25 | paper's best ablation value (Table VI) |
| `dfsc_ir_lr0.02_accum8.yaml` | effective batch 2 -> 16 with LR 0.02 (gradient accumulation) | sensitivity to the unspecified GPU count (spec S3) |
| `dfsc_ir_seed1.yaml` | seed 0 -> 1 | run-to-run variance; the paper reports single runs |
| `normal_ir_seed1.yaml` | seed 0 -> 1 | same, for the baseline |
| `dfsc_ir_ism_cross.yaml` | ISM modulation same -> cross (paper Table VI "DFSC-all") | the one Table VI variant with a large paper effect (-2.05 pt) |

Status as of 2026-09-23 (documentation reconciliation; see `reports/FINAL_PROTOCOL_II_IR_BENCHMARK_REPORT.md`):

- `dfsc_ir_seed1` / `normal_ir_seed1`: trained (2026-09-20/21) and evaluated on val. They were also evaluated on TEST on
  2026-09-23, outside the authorised four-run benchmark scope. Those TEST results are **supplementary / unscoped**.
- `dfsc_ir_ism_cross` (DFSC-all): training completed, but its test evaluation was interrupted before completion and
  therefore produces no valid final test benchmark result. The config's header comment says "reproduced", which is
  inaccurate. The config file is left unchanged because the training run's provenance records it.
- `dfsc_ir_alpha0.25`, `dfsc_ir_lr0.02_accum8`: not run.

Launch exactly like Phase A: `venv\Scripts\python.exe experiments\anti_uav\scripts\train.py --config experiments\anti_uav\configs\phase_b\<file>`.
