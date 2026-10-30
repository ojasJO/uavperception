# Final report: Anti-UAV Protocol II, infrared benchmark

**Status date:** 2026-09-23 (post-shutdown documentation reconciliation)
**This is the authoritative summary of the benchmark.** It is written from existing, verified artifacts only. No
experiment, training, evaluation, rescoring run or GPU job was executed to produce it.

Supporting evidence, all in `experiments/anti_uav/reports/`:
`POST_SHUTDOWN_BENCHMARK_STATUS.md` (forensic status), `benchmark_artifact_integrity.{md,json}` (per-sequence integrity
and independent rescore), `RECOVERED_BENCHMARK_RESULTS.{md,json}` (every verified number with provenance),
`test_evaluation_ledger.jsonl` (append-only test ledger). Benchmark run directory:
`experiments/anti_uav/runs/paper_benchmarks_20260922T193853/`.

---

## 1. Scope

The authorised benchmark scope was **exactly four evaluations** of two frozen, already-trained checkpoints:

| # | Model | Split | Mode |
|---|---|---|---|
| 1 | `normal_ir` | val | validation |
| 2 | `dfsc_ir` | val | validation |
| 3 | `normal_ir` | test | `--final-eval` |
| 4 | `dfsc_ir` | test | `--final-eval` |

The authorised scope included no training, no Table VI ablations and no seed-1 runs. Work that nevertheless exists in
the benchmark directory is reported separately: seed-1 test evaluations in §8, and the DFSC-all ablation in §9. None of
it enters the primary tables.

## 2. Paper reference

*Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking*, Jiang et al., arXiv:2101.08466v3 (2021)
(`anti_uav_research_paper.pdf`). Protocol II = GlobalTrack fine-tuned on the Anti-UAV train split; "normal" = plain
fine-tuning, "DFSC" = fine-tuning with the paper's DFSC strategy. Values below are from Table V, infrared "All" column, mSA %.
They were checked against the PDF text and against `reports/paper_reported_results.json`.

| Model | Split | Paper mSA (IR, All) |
|---|---|---:|
| normal | val | 79.60 |
| DFSC | val | 80.09 |
| normal | test | 65.36 |
| DFSC | test | 66.04 |

Paper DFSC effect (DFSC − normal): **+0.49** (val), **+0.68** (test).
Paper Table VII (IR precision / success, split not stated in the paper): normal 87.61 / 62.88, DFSC 87.77 / 63.24.

## 3. Exact reproduction configuration

| Item | Value |
|---|---|
| Implementation | `experiments/anti_uav/src/anti_uav_repro/`: torchvision re-implementation of GlobalTrack QG-RCNN (R50-FPN) + DFSC |
| Configs | `configs/normal_ir.yaml` (DFSC α = 0) and `configs/dfsc_ir.yaml` (α = 1). Both inherit `configs/_base_ir.yaml` and differ only in α |
| Training | 18 epochs, LR ×0.1 at epochs 12 and 15 (paper-explicit); 1 GPU × batch 2, LR 0.0025 (inferred linear scaling of 0.02; spec S3); SGD; COCO Faster R-CNN init (torchvision COCO_V1, spec S2); seed 0 |
| Checkpoints | final epoch, no selection. `results/runs/normal_ir/weights/last.pt` (epoch 18, SHA-256 `ab9890d52684730e…`); `results/runs/dfsc_ir/weights/last.pt` (epoch 18, `1e8219c56d4b9bfe…`) |
| Input | IR 640×512 rescaled to 1000×800 (GlobalTrack 1333×800 rule); JPEG-decoded frames (spec S4) |
| Data | official Anti-UAV IR splits, frozen manifests: val `a688bf92c46dfc52…` (67 seq), test `1fc351c74f4b29cc…` (91 seq). Label files re-hashed on 2026-09-23: unchanged |
| Metric | mSA = paper Eq. (1), scored from the official start frame; precision@20 px and success AUC over exist = 1 frames (OTB convention; the paper does not specify) |
| Evaluator | `scripts/evaluate.py`, git commit `3b2d48b` (evaluation code unmodified at that commit), `cudnn.benchmark=True`, AMP off |
| Hardware | NVIDIA RTX 5050 Laptop GPU (8.5 GB), torch 2.13.0+cu130 |

All documented substitutions from the paper/GlobalTrack setup (S1–S6) are listed in `docs/ANTI_UAV_REPRODUCTION_SPEC.md` §11.

## 4. The four authorised benchmark results

Source: the live run's `metrics.json`, **independently re-scored from the saved predictions with exact agreement**
(max abs diff 0; `benchmark_artifact_integrity.md`).

| Model | Split | mSA | Precision@20 | Success AUC | Sequences | Frames (all / scored) | Test verdict |
|---|---|---:|---:|---:|---:|---:|---|
| normal_ir | val  | **79.29** | 99.56 | 79.32 | 67 | 61,999 / 61,763 | n/a (validation) |
| dfsc_ir   | val  | **79.26** | 99.57 | 79.29 | 67 | 61,999 / 61,763 | n/a (validation) |
| normal_ir | test | **70.58** | 91.85 | 71.01 | 91 | 85,374 / 85,312 | **VALID FINAL TEST RESULT** (ledger line 5) |
| dfsc_ir   | test | **70.29** | 91.65 | 70.72 | 91 | 85,374 / 85,312 | **VALID FINAL TEST RESULT** (ledger line 6) |

Evaluation windows: normal/val 2026-09-22 19:41–21:29; dfsc/val 21:29–23:03; normal/test 23:03–01:34 (23 Sep);
dfsc/test 01:34–03:38. All four exited with rc = 0.

## 5. Paper vs reproduction

| Model | Split | Paper | Ours | Δ (ours − paper) | Pre-registered verdict (\|Δ\| ≤ 1 reproduced; ≤ 3 approx.; else not) |
|---|---|---:|---:|---:|---|
| normal | val  | 79.60 | 79.29 | **−0.31** | reproduced |
| DFSC   | val  | 80.09 | 79.26 | **−0.83** | reproduced |
| normal | test | 65.36 | 70.58 | **+5.22** | not reproduced (higher than paper) |
| DFSC   | test | 66.04 | 70.29 | **+4.25** | not reproduced (higher than paper) |

Validation agrees with the paper within 1 pt for both models. On test, both models score 4–5 pt **above** the paper. By the
project's pre-registered rule that is "not reproduced", even though the difference is in our favour.

Precision/success (Table VII; the paper does not say which split): ours on test are 91.85 / 71.01 (normal) and
91.65 / 70.72 (DFSC), against the paper's 87.61 / 62.88 and 87.77 / 63.24.

## 6. DFSC effect (DFSC − normal, mSA pt)

| Split | Paper | This reproduction |
|---|---:|---:|
| val  | +0.49 | **−0.03** (79.26 − 79.29) |
| test | +0.68 | **−0.29** (70.29 − 70.58) |

**The positive DFSC effect reported in the paper was not reproduced under this implementation, checkpoint and evaluation
setup.** Each figure is a difference between two single-seed runs, both smaller than 1 pt. This does **not** show that DFSC
is ineffective in general. It shows that the paper's gain was not observed here.

## 7. Per-attribute results (verified)

`n_seq` = sequences in our split carrying the attribute. Attributes with few sequences (e.g. OV val = 4, OC test = 2)
move by tens of points when one sequence changes.

#### `normal_ir` / val (paper row *normal*)

| Attribute | n_seq | Paper | Ours | Δ (ours − paper) |
|---|---:|---:|---:|---:|
| OV | 4 | 78.09 | 56.05 | -22.04 |
| OC | 5 | 81.42 | 78.26 | -3.16 |
| FM | 23 | 78.66 | 79.47 | +0.81 |
| SV | 17 | 76.61 | 72.89 | -3.72 |
| LI | 29 | 79.95 | 79.94 | -0.01 |
| TC-EASY | 17 | 81.23 | 81.62 | +0.39 |
| TC-MID | 8 | 76.56 | 77.78 | +1.22 |
| TC-HARD | 7 | 74.04 | 76.71 | +2.67 |
| TC | 32 | 78.49 | 79.58 | +1.09 |
| LR | 10 | 73.55 | 76.94 | +3.39 |
| **ALL** | 67 | 79.60 | 79.29 | -0.31 |

#### `dfsc_ir` / val (paper row *DFSC*)

| Attribute | n_seq | Paper | Ours | Δ (ours − paper) |
|---|---:|---:|---:|---:|
| OV | 4 | 77.72 | 56.03 | -21.69 |
| OC | 5 | 82.70 | 78.24 | -4.46 |
| FM | 23 | 79.34 | 79.41 | +0.07 |
| SV | 17 | 77.58 | 72.91 | -4.67 |
| LI | 29 | 80.31 | 79.88 | -0.43 |
| TC-EASY | 17 | 81.48 | 81.68 | +0.20 |
| TC-MID | 8 | 76.83 | 77.76 | +0.93 |
| TC-HARD | 7 | 75.65 | 76.72 | +1.07 |
| TC | 32 | 79.04 | 79.61 | +0.57 |
| LR | 10 | 74.33 | 76.94 | +2.61 |
| **ALL** | 67 | 80.09 | 79.26 | -0.83 |

#### `normal_ir` / test (paper row *normal*)

| Attribute | n_seq | Paper | Ours | Δ (ours − paper) |
|---|---:|---:|---:|---:|
| OV | 18 | 70.44 | 69.57 | -0.87 |
| OC | 2 | 68.94 | 64.52 | -4.42 |
| FM | 31 | 60.66 | 70.43 | +9.77 |
| SV | 18 | 55.48 | 64.64 | +9.16 |
| LI | 18 | 59.78 | 75.51 | +15.73 |
| TC-EASY | 14 | 77.07 | 77.01 | -0.06 |
| TC-MID | 31 | 64.64 | 72.31 | +7.67 |
| TC-HARD | 18 | 44.61 | 51.63 | +7.02 |
| TC | 63 | 61.68 | 67.45 | +5.77 |
| LR | 15 | 52.94 | 70.24 | +17.30 |
| **ALL** | 91 | 65.36 | 70.58 | +5.22 |

#### `dfsc_ir` / test (paper row *DFSC*)

| Attribute | n_seq | Paper | Ours | Δ (ours − paper) |
|---|---:|---:|---:|---:|
| OV | 18 | 70.16 | 69.46 | -0.70 |
| OC | 2 | 68.07 | 64.78 | -3.29 |
| FM | 31 | 60.95 | 69.77 | +8.82 |
| SV | 18 | 55.55 | 63.87 | +8.32 |
| LI | 18 | 60.13 | 74.50 | +14.37 |
| TC-EASY | 14 | 77.96 | 77.34 | -0.62 |
| TC-MID | 31 | 65.85 | 71.99 | +6.14 |
| TC-HARD | 18 | 45.59 | 50.80 | +5.21 |
| TC | 63 | 62.75 | 67.13 | +4.38 |
| LR | 15 | 53.10 | 69.11 | +16.01 |
| **ALL** | 91 | 66.04 | 70.29 | +4.25 |

## 8. Supplementary seed-1 results (unscoped; not primary)

Two seed-1 checkpoints (trained 2026-09-20/21, frozen) were evaluated on TEST on 2026-09-23 09:26–13:59. That was
**after** the four authorised evaluations had finished and **outside the four-run scope**. The generated plan describes
them as "added under a later authorisation". **No authorisation record was found** in the repository, in git history or
in the project notes. They are therefore **supplementary / unscoped**, are not used in any primary table, and must not be
presented as benchmark results. Their artifacts are complete and independently verified. They added two permanent entries
to the test ledger (lines 7–8).

| Run | Checkpoint | Split | mSA | Precision@20 | Success |
|---|---|---|---:|---:|---:|
| `phaseB_normal_ir_seed1` | `414ef0164e2bfc65…` | test | 70.22 | 91.30 | 70.66 |
| `phaseB_dfsc_ir_seed1` | `dc0b3e643b53222b…` | test | 70.14 | 91.36 | 70.58 |

Descriptively, the seed-1 DFSC − normal difference on test is −0.08 pt (seed 0: −0.29). Seed-0 → seed-1 changes are
0.36 pt (normal) and 0.15 pt (DFSC). With two seeds per arm this is a range, not a variance estimate. Historical seed-1
**val** results from 2026-09-21 (79.36 normal, 79.43 DFSC; `docs/ANTI_UAV_COMPARISON.md`) are pre-existing Phase B
artifacts and are likewise not primary results.

## 9. Incomplete DFSC-all ablation (no valid result)

DFSC-all training was completed, but its test evaluation was interrupted before completion and therefore produces no valid
final test benchmark result.

- **Training:** `results/runs/phaseB_dfsc_ir_ism_cross/`, config `configs/phase_b/dfsc_ir_ism_cross.yaml` (ISM modulation =
  cross), 18/18 epochs, 2026-09-23 14:00–16:33, rc = 0, `last.pt` SHA-256 `708ce9666ff84f18…`. It was outside the authorised
  scope, and no authorisation record was found.
- **Test evaluation:** started 16:33:26 with `--final-eval`. It completed 54 of 91 sequences (51,009 of 85,374 frames;
  last file `20190925_200805_1_8` at 20:03:46) and was killed while tracking sequence 55 by an unexpected machine shutdown.
  The Windows System log shows Kernel-Power event 41 and EventLog 6008, with a reboot at 20:12. No `metrics.json`, no
  `report.md` and no ledger entry was written.
- **Status:** `NOT A VALID FINAL TEST RESULT`. The 54 prediction files pass the integrity checks and are preserved
  unmodified as an **incomplete/aborted artifact** in
  `results/runs/paper_benchmarks_20260922T193853/ablation/phaseB_dfsc_ir_ism_cross__test_IR/`. They cover a non-random prefix of
  the test split and must not be scored, quoted or compared.
- **Paper Table VI therefore has no reproduced row.** The paper's DFSC-all value (63.99 IR test) has no counterpart here.

## 10. Artifact integrity and verification status

| Check (four authorised evaluations) | Result |
|---|---|
| Prediction files present | 67 / 67 / 91 / 91, exact prefix-complete match with the frozen manifests |
| Per-file schema (JSON parse, sequence ID, split, IR modality, stride 1, lengths = label length, start frame = manifest, finite boxes with w,h > 0, finite scores, processed flags) | 0 problems in 316 files |
| Missing / duplicate / truncated frames or files | none |
| Checkpoint SHA-256 on disk = value recorded at evaluation time = value in the 2026-09-20 ledger entries | yes, all four |
| Test manifest SHA-256 = frozen manifest; label files unchanged since freeze | yes |
| Independent rescore (from-scratch re-implementation) vs reported mSA, precision, success, 10 attributes, every per-sequence SA | exact match (max abs diff 0). A second, earlier independent audit (`metric_independent_audit.json`) also agrees |
| Test runs invoked with `--final-eval`; ledger entries present with matching mSA | yes (lines 5, 6) |
| Agreement with the historical 2026-09-20 evaluations of the same checkpoints | within 0.00016 pt mSA; predictions not bit-identical (`cudnn.benchmark=True`). These are confirmatory re-measurements; historical results were not overwritten |

**Test-set isolation.** Both checkpoints are the fixed final epoch. No checkpoint, threshold, hyper-parameter or rule
was selected using test data. The test split is served only to `--final-eval`. Every test evaluation is in the
append-only ledger. The ledger records each authorised checkpoint as evaluated on test twice: on 2026-09-20 (lines 1–2)
and in this benchmark (lines 5–6). Both are evaluations of the same frozen weights and involved no selection. Ledger
lines 3–4 are the historical Phase B post-hoc absence-threshold control, a separate additional control that is not part
of these results.

## 11. Known discrepancies

1. **Test split: +5.22 (normal) / +4.25 (DFSC) pt above the paper**, while validation matches within 1 pt. The val → test
   drop is 8.71 / 8.97 pt here against the paper's 14.24 / 14.05 pt.
2. **The DFSC effect is not reproduced**: −0.03 / −0.29 pt here vs the paper's +0.49 / +0.68.
3. **Validation attribute deficits on OV (−22), OC (−3 to −4) and SV (−4 to −5).** A post-hoc decomposition in
   `PAPER_BENCHMARK_RESULTS.md` §H0/H3/H5 shows these val deficits come from `exist == 0` frames. The tracker always emits a
   box, so those frames score 0. This is an established, measured property of *our* output on *val* attributes. It does not
   explain the test surplus, and it does not justify altering any reported number.
4. **Test attribute surpluses** are concentrated in LR (+16 to +17), LI (+14 to +16), FM (+9 to +10) and SV (+8 to +9), while
   OV and TC-EASY are close to the paper. Unexplained.
5. **Precision/success** exceed the paper's Table VII values. The paper does not state the split for Table VII.
6. **Dataset box count**: the released labels carry ~2 % fewer boxes than the paper's Table I
   (`reports/dataset_verification.json`). Observed; cause not established.

## 12. Possible explanations (HYPOTHESES, not established causes)

**The current evidence does not isolate the cause of the test-split discrepancy or of the missing DFSC effect.** Possible
explanations include the following (full statements in `docs/ANTI_UAV_COMPARISON.md` §Discrepancy analysis and
`PAPER_BENCHMARK_RESULTS.md` §H):

| Hypothesis | What is known | Status |
|---|---|---|
| Annotation / label-version differences (released labels revised after the paper) | consistent with the ~2 % box shortfall and with the attribute pattern | **hypothesis; untestable without the original labels** |
| Absence handling in Eq. (1) | measured: explains the val OV/OC/SV deficits; any absence rule could only *raise* our test score further | **tested as a cause of the test surplus: not supported** |
| Optimisation / training regime (1 GPU × batch 2 × LR 0.0025 = 8× more steps than an 8-GPU run) | not varied | **hypothesis; untested** |
| Input scale (IR up-scaled to 1000×800; the paper's test-time scale is unknown) | not varied | **hypothesis; untested** |
| Framework / initial weights (torchvision vs mmdetection 1.x; COCO_V1 vs mmdet init) | val agreement argues against a large systematic effect | **hypothesis; not isolable here** |
| Evaluation implementation | two independent metric implementations agree exactly; start-frame handling changes results by ≤ 0.05 pt | **checked; no evidence of a metric error** |
| Paper reporting (e.g. transcription or split ambiguity, especially Table VII) | nothing indicates an error | **hypothesis; no evidence** |
| Seed variability (for the DFSC effect) | two seeds per arm only (seed-1 on test is supplementary) | **not resolved** |

No hypothesis is declared to be the cause.

## 13. What can legitimately be claimed

- The GlobalTrack fine-tuning ("normal") and DFSC training recipes were re-implemented and trained end-to-end on
  Anti-UAV IR (Protocol II), and evaluated with the paper's mSA metric.
- **Validation mSA agrees with the paper within 1 pt** for both models (79.29 vs 79.60; 79.26 vs 80.09).
- **On the test split both models score higher than the paper** (70.58 vs 65.36; 70.29 vs 66.04). The cause has not been
  isolated.
- **The paper's positive DFSC effect was not reproduced** in this setup (−0.03 val, −0.29 test, single seed).
- All four results are complete, integrity-checked, independently re-scored and consistent with earlier evaluations of the
  same checkpoints to within 0.0002 pt. Both test results are valid final evaluations under the repository's procedure.

## 14. What cannot legitimately be claimed

- That DFSC is ineffective in general (one implementation, one modality, one seed in the primary results).
- That our tracker is better than the paper's (the test surplus is unexplained; hypotheses H1/H3/H4 are open).
- Any specific cause for the discrepancy.
- Any Table VI ablation result. DFSC-all has **no** valid result, and its 54-sequence partial predictions must not be used.
- That the seed-1 test numbers belong to the authorised benchmark, or any significance or variance statement from two seeds.
- Anything about the visible modality, the "large-scale" row, Protocol I or Protocol III (not run; blocked or not
  applicable; see `benchmark_execution_plan.md`).
- That the evaluation is bit-exactly deterministic. It is reproducible to ≤ 0.00016 pt mSA, not bit-identical.

## 15. Reproducibility notes

- Exact commands, environment, timestamps and hashes for each evaluation: `PAPER_BENCHMARK_RESULTS.md` §I and each
  `results/runs/paper_benchmarks_20260922T193853/<run>__<split>_IR/eval_meta.json`.
- Driver script for the four authorised evaluations: `results/runs/paper_benchmarks_20260922T193853/run_benchmarks.sh`
  (≈ 7.9 GPU-hours measured).
- Rescoring without the model: `evaluate.py --predictions <dir>/predictions --split <split> --checkpoint <same ckpt>`.
  This is a CPU-only rescore and was **not** run for this report. Verification used the forensic scripts described in
  `benchmark_artifact_integrity.md`.
- Remaining non-determinism: cuDNN benchmark kernel selection (GlobalTrack default). Re-evaluation of the same checkpoint
  reproduced mSA to ≤ 0.00016 pt.
- `evaluate.py` has **no sequence-level resume**; an interrupted evaluation must be rerun in full.
- The generated documents `PAPER_BENCHMARK_RESULTS.md` and `benchmark_execution_plan.{md,json}` were **hand-corrected** on
  2026-09-23. Re-running `scripts/make_final_paper_report.py` or `scripts/make_benchmark_plan.py` would reintroduce the
  incorrect DFSC-all / seed-1 statements. `make_benchmark_plan.py` hard-codes DFSC-all as "trained and evaluated".

## 16. Final status

| Item | Status |
|---|---|
| Four authorised Protocol II IR evaluations | **COMPLETE** |
| Their prediction artifacts | **COMPLETE and VERIFIED** |
| Their metrics | **independently re-scored; MATCH exactly** |
| normal_ir / test, dfsc_ir / test | **VALID FINAL TEST RESULTS** under the repository's evaluation procedure |
| Further runs required for the authorised Phase A scope | **NONE** |
| DFSC-all ablation | **INCOMPLETE; NO VALID FINAL TEST RESULT** (training done, test evaluation 54/91) |
| Seed-1 test evaluations | **SUPPLEMENTARY**, separate from the four-run scope; no authorisation record found |
| Paper vs reproduction discrepancy (test surplus; missing DFSC effect) | **UNRESOLVED** |
| Causal explanation of the discrepancy | **none asserted**; hypotheses only (§12) |
