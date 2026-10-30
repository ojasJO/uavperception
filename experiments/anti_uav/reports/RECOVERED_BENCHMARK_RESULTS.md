# Recovered benchmark results: paper_benchmarks_20260922T193853

Generated 2026-09-23T20:36:59+0530 by the post-shutdown forensic inspection. **Nothing was re-run.**

Source label for every number: **RECOVERED_FROM_EXISTING_ARTIFACT** (the live run's `metrics.json`) **+ RECOMPUTED_FROM_SAVED_PREDICTIONS**. An independent re-implementation scored the saved predictions against the ground-truth labels and matched every figure exactly (max abs diff 0). No number here is `LIVE_RUN`.

Units: percent. mSA follows paper Eq. 1, scored from the official start frame. Precision@20 px and success AUC are computed over exist=1 frames. Attribute sequence counts are in the JSON.

## 1. The four authorised evaluations

| Evaluation | Model | Split | mSA ALL | OV | OC | FM | SV | LI | TC-EASY | TC-MID | TC-HARD | TC | LR | Prec@20 | Success AUC | Seqs | Frames (all / scored) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `normal_ir/val` | normal (DFSC alpha=0) | val | **79.29** | 56.05 | 78.26 | 79.47 | 72.89 | 79.94 | 81.62 | 77.78 | 76.71 | 79.58 | 76.94 | 99.56 | 79.32 | 67 | 61999 / 61763 |
| `dfsc_ir/val` | DFSC (alpha=1) | val | **79.26** | 56.03 | 78.24 | 79.41 | 72.91 | 79.88 | 81.68 | 77.76 | 76.72 | 79.61 | 76.94 | 99.57 | 79.29 | 67 | 61999 / 61763 |
| `normal_ir/test` | normal (DFSC alpha=0) | test | **70.58** | 69.57 | 64.52 | 70.43 | 64.64 | 75.51 | 77.01 | 72.31 | 51.63 | 67.45 | 70.24 | 91.85 | 71.01 | 91 | 85374 / 85312 |
| `dfsc_ir/test` | DFSC (alpha=1) | test | **70.29** | 69.46 | 64.78 | 69.77 | 63.87 | 74.50 | 77.34 | 71.99 | 50.80 | 67.13 | 69.11 | 91.65 | 70.72 | 91 | 85374 / 85312 |

### Provenance

| Evaluation | Checkpoint (epoch) | SHA-256 | Eval start → end | Predictions | Metrics source | Verified | Test verdict |
|---|---|---|---|---|---|---|---|
| `normal_ir/val` | `experiments/anti_uav/runs/normal_ir/weights/last.pt` (18) | `ab9890d52684730e…` | 2026-09-22T19:41:28+0530 → 2026-09-22T21:29:29+0530 | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/normal_ir__val_IR/predictions` | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/normal_ir__val_IR/metrics.json` | yes, exact | n/a (val) |
| `dfsc_ir/val` | `experiments/anti_uav/runs/dfsc_ir/weights/last.pt` (18) | `1e8219c56d4b9bfe…` | 2026-09-22T21:29:53+0530 → 2026-09-22T23:03:13+0530 | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/dfsc_ir__val_IR/predictions` | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/dfsc_ir__val_IR/metrics.json` | yes, exact | n/a (val) |
| `normal_ir/test` | `experiments/anti_uav/runs/normal_ir/weights/last.pt` (18) | `ab9890d52684730e…` | 2026-09-22T23:03:17+0530 → 2026-09-23T01:34:14+0530 | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/normal_ir__test_IR/predictions` | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/normal_ir__test_IR/metrics.json` | yes, exact | VALID FINAL TEST RESULT |
| `dfsc_ir/test` | `experiments/anti_uav/runs/dfsc_ir/weights/last.pt` (18) | `1e8219c56d4b9bfe…` | 2026-09-23T01:34:18+0530 → 2026-09-23T03:38:10+0530 | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/dfsc_ir__test_IR/predictions` | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/dfsc_ir__test_IR/metrics.json` | yes, exact | VALID FINAL TEST RESULT |

## 2. Completed evaluations OUTSIDE the stated authorised scope

Your brief authorised only the four evaluations above. The same run directory also contains two seed-1 TEST evaluations (`seed_variance/`, 09:26–13:59 on 2026-09-23). Their artifacts are complete and verified. `benchmark_execution_plan.md` says they were "added under a later authorisation", but this inspection found no record of that authorisation. Whether to use them is your decision.

| Evaluation | Model | Split | mSA ALL | OV | OC | FM | SV | LI | TC-EASY | TC-MID | TC-HARD | TC | LR | Prec@20 | Success AUC | Seqs | Frames (all / scored) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `phaseB_normal_ir_seed1/test` | normal (alpha=0), seed 1 | test | **70.22** | 69.22 | 63.63 | 69.77 | 63.96 | 75.01 | 77.15 | 72.15 | 50.11 | 66.96 | 69.66 | 91.30 | 70.66 | 91 | 85374 / 85312 |
| `phaseB_dfsc_ir_seed1/test` | DFSC (alpha=1), seed 1 | test | **70.14** | 69.40 | 63.50 | 70.00 | 64.28 | 75.15 | 76.90 | 71.96 | 50.69 | 66.98 | 69.81 | 91.36 | 70.58 | 91 | 85374 / 85312 |

### Provenance

| Evaluation | Checkpoint (epoch) | SHA-256 | Eval start → end | Predictions | Metrics source | Verified | Test verdict |
|---|---|---|---|---|---|---|---|
| `phaseB_normal_ir_seed1/test` | `experiments/anti_uav/runs/phaseB_normal_ir_seed1/weights/last.pt` (18) | `414ef0164e2bfc65…` | 2026-09-23T09:26:21+0530 → 2026-09-23T11:50:44+0530 | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/seed_variance/phaseB_normal_ir_seed1__test_IR/predictions` | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/seed_variance/phaseB_normal_ir_seed1__test_IR/metrics.json` | yes, exact | VALID FINAL TEST RESULT |
| `phaseB_dfsc_ir_seed1/test` | `experiments/anti_uav/runs/phaseB_dfsc_ir_seed1/weights/last.pt` (18) | `dc0b3e643b53222b…` | 2026-09-23T11:51:08+0530 → 2026-09-23T13:59:28+0530 | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/seed_variance/phaseB_dfsc_ir_seed1__test_IR/predictions` | `experiments/anti_uav/runs/paper_benchmarks_20260922T193853/seed_variance/phaseB_dfsc_ir_seed1__test_IR/metrics.json` | yes, exact | VALID FINAL TEST RESULT |

## 3. Not results

- `phaseB_dfsc_ir_ism_cross/test (Table VI DFSC-all)`: **C_PARTIAL**, 54/91 sequences. NOT A VALID FINAL TEST RESULT. No number from it is reported here.

## 4. Historical results (for comparison only; not modified, not replaced)

| Evaluation | Historical source | Hist. timestamp | Hist. mSA | Recovered mSA | Hist. prec / succ | Relationship |
|---|---|---|---:|---:|---|---|
| `normal_ir/val` | `experiments/anti_uav/runs/normal_ir/eval_val_IR_last` | 2026-09-20T13:42:37+0530 | 79.2874 | 79.2874 | 99.56 / 79.32 | same checkpoint re-measured; agrees within GPU non-determinism (mSA |diff| = 0.000000 pt); historical NOT replaced |
| `dfsc_ir/val` | `experiments/anti_uav/runs/dfsc_ir/eval_val_IR_last` | 2026-09-20T13:52:35+0530 | 79.2559 | 79.2559 | 99.57 / 79.29 | same checkpoint re-measured; agrees within GPU non-determinism (mSA |diff| = 0.000018 pt); historical NOT replaced |
| `normal_ir/test` | `experiments/anti_uav/runs/normal_ir/eval_test_IR_last` | 2026-09-20T19:00:41+0530 | 70.5796 | 70.5796 | 91.85 / 71.01 | same checkpoint re-measured; agrees within GPU non-determinism (mSA |diff| = 0.000005 pt); historical NOT replaced |
| `dfsc_ir/test` | `experiments/anti_uav/runs/dfsc_ir/eval_test_IR_last` | 2026-09-20T16:29:25+0530 | 70.2868 | 70.2866 | 91.65 / 70.72 | same checkpoint re-measured; agrees within GPU non-determinism (mSA |diff| = 0.000162 pt); historical NOT replaced |

The overnight run re-measured the same checkpoints and got the same results (to within 0.0002 pt mSA) using a fresh, fully provenanced pipeline. These are **duplicate/confirmatory** results, not new or different ones. The seed-1 TEST numbers are the only numbers with no historical counterpart.
