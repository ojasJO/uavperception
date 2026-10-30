# Post-shutdown benchmark status: `paper_benchmarks_20260922T193853`

Forensic inspection, 2026-09-23. **Read-only. Nothing was resumed, re-run, trained, modified or deleted.** The only files written
are this report and four new companion files in `experiments/anti_uav/reports/`:
`benchmark_artifact_integrity.{json,md}` and `RECOVERED_BENCHMARK_RESULTS.{json,md}`.

## 1. Executive summary

- **4 authorised evaluations. 4 completed, 0 partial, 0 not started.** All four had finished and exited `rc=0` by
  **2026-09-23 03:38:10**, about 16.5 hours before the machine went down.
- **4 valid authorised results recovered** (2 val, 2 test). Each one passed full artifact integrity checks and an independent
  re-score from the saved predictions that matches the reported metrics exactly.
- **Both authorised test evaluations are `VALID FINAL TEST RESULT`s.**
- **0 authorised evaluations need continuation or rerun.** Remaining authorised GPU work: **0 h**.
- **The shutdown did not interrupt any authorised run.** It interrupted work outside the scope stated in your brief, which ran
  in the same benchmark directory after the authorised suite finished:
  1. `seed_variance/`: seed-1 TEST evaluations of `phaseB_normal_ir_seed1` and `phaseB_dfsc_ir_seed1` (09:26–13:59). **Completed and verified.**
  2. `ablation/`: **training** of Table VI variant DFSC-all (`phaseB_dfsc_ir_ism_cross`, 14:00–16:33, completed, 18/18 epochs),
     followed by a TEST `--final-eval` of that checkpoint (started 16:33:26). **This evaluation was interrupted at 54/91 sequences
     and is `NOT A VALID FINAL TEST RESULT`.**
  `reports/benchmark_execution_plan.md` labels items 1 and 2 as "added under a later authorisation". This inspection found no record
  of that authorisation. It contradicts the scope in your brief, and you should settle it.
- **Shutdown:** Windows System log, Kernel-Power **event 41** ("rebooted without cleanly shutting down") and EventLog **6008**
  ("previous system shutdown at 8:01:08 PM on 9/23/2026 was unexpected"), reboot at 20:12:33. The 6008 time comes from a periodic
  heartbeat. The last artifact write was 20:03:46, so the machine died between 20:03:46 and 20:12:33. No Python process is running now.

## 2. Run-by-run status (authorised scope)

Every field below is computed from the artifacts. Frames = all label frames in the prediction files; the scored count
(from the official start frame, which excludes pre-start frames) is in brackets.

| Run | Started? | Completed? | Sequences processed | Frames processed | Predictions complete? | Metrics valid? | Report valid? | Can resume? |
|---|---|---|---:|---:|---|---|---|---|
| normal_ir / val  | yes, 19:41:04 (22 Sep) | **yes**, rc=0, 21:29:29 | 67 / 67 | 61,999 / 61,999 (61,763 scored) | **yes**, 67 files, 0 schema problems | **yes**, independent re-score = reported exactly | yes | n/a (complete) |
| dfsc_ir / val    | yes, 21:29:29 | **yes**, rc=0, 23:03:14 | 67 / 67 | 61,999 / 61,999 (61,763) | **yes**, 67 / 0 problems | **yes**, exact | yes | n/a |
| normal_ir / test | yes, 23:03:14 | **yes**, rc=0, 01:34:15 (23 Sep) | 91 / 91 | 85,374 / 85,374 (85,312) | **yes**, 91 / 0 problems | **yes**, exact | yes | n/a |
| dfsc_ir / test   | yes, 01:34:15 | **yes**, rc=0, 03:38:10 | 91 / 91 | 85,374 / 85,374 (85,312) | **yes**, 91 / 0 problems | **yes**, exact | yes | n/a |

Classification (A–E scheme): all four are **A. COMPLETED AND VERIFIED**.

For each run, the evidence chain covers: process started (`_driver.log` START line and `eval_meta.json`
`timestamp_start`), every sequence completed (log lines `[067/067]` / `[091/091]`), predictions written (all files present, prefix-complete),
predictions verified (this inspection), metrics calculated (`metrics.json`; `eval_meta.json` got `timestamp_end` and `mSA_percent`,
which the evaluator writes only after scoring), report generated (`report.md`), normal exit (`rc=0` in `_driver.log` and `_status.jsonl`,
plus `ALL DONE 2026-09-23T03:38:10`).

### Outside the stated scope (same directory)

| Run | Started? | Completed? | Sequences | Frames | Predictions complete? | Metrics valid? | Report valid? | Class |
|---|---|---|---:|---:|---|---|---|---|
| phaseB_normal_ir_seed1 / test | 09:26:21 | yes, rc=0, 11:50:44 | 91/91 | 85,374 | yes | yes, exact | yes | A |
| phaseB_dfsc_ir_seed1 / test | 11:51:08 | yes, rc=0, 13:59:28 | 91/91 | 85,374 | yes | yes, exact | yes | A |
| phaseB_dfsc_ir_ism_cross **training** | 14:00:28 | yes, rc=0, 16:33:22, 18/18 epochs, `last.pt` SHA `708ce966…` | — | — | — | — | — | complete |
| phaseB_dfsc_ir_ism_cross / test | 16:33:26 | **NO, killed by the shutdown** | **54/91** | **51,009 / 85,374** | **no** (54 valid files, prefix of manifest order, 0 corrupt) | **no metrics.json** | **no report.md** | **C. PARTIAL** |

## 3. Valid results already obtained

All numbers are percentages. Source: **RECOVERED_FROM_EXISTING_ARTIFACT** (the live run's `metrics.json`), confirmed by
**RECOMPUTED_FROM_SAVED_PREDICTIONS** (independent re-implementation, max abs diff 0). Nothing here is `LIVE_RUN`, because nothing was re-run.
Full tables with provenance: `RECOVERED_BENCHMARK_RESULTS.md`.

| Run | Checkpoint (SHA-256 prefix) | Split | mSA ALL | OV | OC | FM | SV | LI | TC-EASY | TC-MID | TC-HARD | TC | LR | Prec@20 | Success |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal_ir | `ab9890d52684…` ep 18 | val  | **79.29** | 56.05 | 78.26 | 79.47 | 72.89 | 79.94 | 81.62 | 77.78 | 76.71 | 79.58 | 76.94 | 99.56 | 79.32 |
| dfsc_ir   | `1e8219c56d4b…` ep 18 | val  | **79.26** | 56.03 | 78.24 | 79.41 | 72.91 | 79.88 | 81.68 | 77.76 | 76.72 | 79.61 | 76.94 | 99.57 | 79.29 |
| normal_ir | `ab9890d52684…` ep 18 | test | **70.58** | 69.57 | 64.52 | 70.43 | 64.64 | 75.51 | 77.01 | 72.31 | 51.63 | 67.45 | 70.24 | 91.85 | 71.01 |
| dfsc_ir   | `1e8219c56d4b…` ep 18 | test | **70.29** | 69.46 | 64.78 | 69.77 | 63.87 | 74.50 | 77.34 | 71.99 | 50.80 | 67.13 | 69.11 | 91.65 | 70.72 |

Outside stated scope (valid artifacts; include them only if you accept that authorisation):

| Run | Checkpoint | Split | mSA ALL | Prec@20 | Success |
|---|---|---|---:|---:|---:|
| phaseB_normal_ir_seed1 | `414ef0164e2b…` ep 18 | test | 70.22 | 91.30 | 70.66 |
| phaseB_dfsc_ir_seed1   | `dc0b3e643b53…` ep 18 | test | 70.14 | 91.36 | 70.58 |

### Relationship to historical results (nothing overwritten)

| Run | Historical source | Historical mSA | Recovered mSA | Relationship |
|---|---|---:|---:|---|
| normal_ir / val  | `runs/normal_ir/eval_val_IR_last` (2026-09-20) | 79.2874 | 79.2874 | duplicate: same ckpt, 3/67 seqs differ by ≤0.03 px, identical metrics |
| dfsc_ir / val    | `runs/dfsc_ir/eval_val_IR_last` (2026-09-20) | 79.2559 | 79.2559 | duplicate: same ckpt, Δ = 0.00002 pt |
| normal_ir / test | `runs/normal_ir/eval_test_IR_last` (2026-09-20) | 70.5796 | 70.5796 | duplicate: same ckpt, Δ = 0.000005 pt; boxes differ ≤1 px |
| dfsc_ir / test   | `runs/dfsc_ir/eval_test_IR_last` (2026-09-20) | 70.2868 | 70.2866 | duplicate: same ckpt, Δ = 0.00016 pt; a few frames diverge (max 137 px) |

The overnight run produced **confirmatory re-measurements** of the historical numbers, not new or different results. Predictions are
not bit-identical (GPU non-determinism with `cudnn.benchmark=True`), but every metric agrees to within 0.0002 pt. No file under
`runs/normal_ir/`, `runs/dfsc_ir/`, `runs/phaseB_*_seed1/` or the historical reports was modified after the benchmark started (checked by mtime).
The seed-1 **test** numbers have no historical counterpart.

## 4. Partial results

Only one evaluation is partial: **`ablation/phaseB_dfsc_ir_ism_cross__test_IR`** (outside the stated scope).

- Started 2026-09-23 16:33:26 (`eval_meta.json`). `--final-eval`, frozen test manifest verified, checkpoint `708ce966…` (epoch 18).
- Last completed sequence: **054/91 `20190925_200805_1_8`**, file written **20:03:46**.
- Interrupted **inside sequence 055/91 `20190925_200805_1_9`** (475 frames). No file was written for it. The evaluator writes each
  sequence file once, after that sequence finishes tracking, so nothing is half-written. No frame-level progress is logged within a sequence.
- Remaining: 37 sequences, 34,365 frames.
- It did not reach metric aggregation: no `metrics.json`, no `report.md`, no `timestamp_end` in `eval_meta.json`, no ledger entry.
  The evaluator did not exit normally, and the driver did not append an `eval` line to `ablation/_status.jsonl`.
- Anomaly: sequence 053 (`20190925_200805_1_7`) logged **0.1 fps**, about 124 min of wall time from 17:58 to 20:02, versus about 1.6 min normally.
  The machine stalled during that sequence. Its prediction file is structurally valid.
- The integrity script necessarily scored the 54 files to check them. That partial aggregate is stored in
  `benchmark_artifact_integrity.json` under a `NOT A RESULT` warning. It covers a non-random prefix of the test split, is not comparable
  to any 91-sequence number, and must not be reported or used.

## 5. Artifact integrity

Details: `benchmark_artifact_integrity.md` / `.json`. For all 7 evaluation directories, every prediction file that exists passed:
JSON parse (0 truncated/corrupt); sequence field = filename, unique; sequence belongs to the split's frozen manifest; split and
modality = `IR` recorded correctly; `frame_stride`=1; per-frame arrays (`res`, `scores`, `processed`) have exactly the label length
(frame index = list position, so 0 missing and 0 duplicate frames); `start_frame` = manifest; all boxes are 4 finite numbers with
w,h > 0, and every other output is an explicit absent marker; all scores finite; `processed` flags consistent with the start frame.
Checkpoint SHA-256 on disk now = the value recorded at evaluation time, for all 7. Manifest SHA-256 = frozen manifest, for all 7.
Label files still hash to the frozen manifests (val 0/67 changed, test 0/91 changed).

| Evaluation | Files | Valid | Problems | Complete |
|---|---:|---:|---:|---|
| normal_ir / val | 67 | 67 | 0 | yes |
| dfsc_ir / val | 67 | 67 | 0 | yes |
| normal_ir / test | 91 | 91 | 0 | yes |
| dfsc_ir / test | 91 | 91 | 0 | yes |
| seed1 normal / test | 91 | 91 | 0 | yes |
| seed1 dfsc / test | 91 | 91 | 0 | yes |
| DFSC-all / test | 54 | 54 | 0 | **no, 37 sequences missing** |

The evaluation code (`scripts/evaluate.py`, `src/anti_uav_repro/`) is unmodified relative to commit `3b2d48b`, the commit recorded in every
`eval_meta.json`. The working tree was dirty only in unrelated files.

### Inconsistencies in existing reports

> *Update 2026-09-23 (documentation reconciliation):* the statements below have since been corrected in place and marked
> *[corrected]*. See `FINAL_PROTOCOL_II_IR_BENCHMARK_REPORT.md`, now the authoritative summary.

- `reports/PAPER_BENCHMARK_RESULTS.md` (last written 16:33, after ablation training but before its evaluation) contradicts itself. Its
  inventory marks Table VI DFSC-all as executed "**YES**", and its file list says `ablation/` holds a "test evaluation". Its §F says
  "No ablation was trained or evaluated in this task", and it gives no DFSC-all number. **No DFSC-all test result exists.**
- The benchmark `README.md` says DFSC-all was "trained … then evaluated on TEST once". Training finished; the evaluation did not.
- `reports/benchmark_execution_plan.md` still says of Table VI "None of these was executed" while listing DFSC-all as a follow-up.

## 6. Test-split status

| Test evaluation | `--final-eval` in recorded command | Frozen manifest verified | Checkpoint = frozen last.pt (SHA unchanged since first test eval 2026-09-20) | Predictions complete | Ledger | Verdict |
|---|---|---|---|---|---|---|
| normal_ir / test | yes | yes (`1fc351c7…`) | yes, `ab9890d5…`, same SHA as ledger line 2 | 91/91 | line 5, mSA matches | **VALID FINAL TEST RESULT** |
| dfsc_ir / test | yes | yes | yes, `1e8219c5…`, same SHA as ledger line 1 | 91/91 | line 6, matches | **VALID FINAL TEST RESULT** |
| seed1 normal / test *(outside scope)* | yes | yes | `414ef016…` (trained 2026-09-20/21) | 91/91 | line 7, matches | VALID FINAL TEST RESULT (authorisation disputed) |
| seed1 dfsc / test *(outside scope)* | yes | yes | `dc0b3e64…` (trained 2026-09-20) | 91/91 | line 8, matches | VALID FINAL TEST RESULT (authorisation disputed) |
| DFSC-all / test *(outside scope)* | yes | yes | `708ce966…` | **54/91** | none (correct: the ledger is written only on completion) | **NOT A VALID FINAL TEST RESULT** |

Test not used for tuning: all checkpoints are the fixed final epoch (`last.pt`, epoch 18). No checkpoint selection, threshold or rule
was applied in this benchmark: raw evaluator, primary scoring. The two authorised checkpoints are byte-identical to the ones evaluated
on test on 2026-09-20. The ledger (`reports/test_evaluation_ledger.jsonl`, 8 lines, append-only, +4 lines vs git HEAD) now records each
authorised checkpoint as evaluated on test **twice**: once on 2026-09-20 and once in this benchmark. Earlier lines 3–4 are the
historical Phase B post-hoc absence-threshold rule and are not part of this benchmark.

## 7. Resume / rerun requirements

**Authorised scope: nothing remains.** No rerun is necessary.

Evaluator resume semantics, checked in `scripts/evaluate.py`: **no sequence-level resume.** The script always tracks every sequence in
manifest order and overwrites `predictions/<seq>.json`. It never checks for existing files. The `--predictions` mode only re-scores, and it
requires a file for every sequence (`load_prediction` for all 91), so it cannot finish a partial directory.

| Run | Resume possible? | Why | Exact command (NOT executed) | Risk |
|---|---|---|---|---|
| normal_ir / val, dfsc_ir / val, normal_ir / test, dfsc_ir / test | not needed | complete and verified | — | Re-running would add two more test-ledger entries for the same checkpoints and change nothing |
| DFSC-all / test *(outside scope)* | **No safe resume with current code** | no skip-existing logic; pointing it at the same `--out-dir` re-tracks all 91 and overwrites the 54 partial files and `eval_meta.json` | clean rerun into a **new** directory, which keeps the partial evidence: `venv/Scripts/python.exe experiments/anti_uav/scripts/evaluate.py --checkpoint experiments/anti_uav/runs/phaseB_dfsc_ir_ism_cross/weights/last.pt --split test --modality IR --final-eval --out-dir experiments/anti_uav/runs/paper_benchmarks_20260922T193853/ablation/phaseB_dfsc_ir_ism_cross__test_IR_rerun` | (a) it is a test evaluation of an ablation your brief did not authorise; (b) it becomes this checkpoint's first ledger entry; (c) a true resume would need a code change (skip existing, validated files), which is itself an unauthorised modification |

## 8. GPU work remaining

- **Authorised benchmark: 0 h.** Measured tracking time already spent: 1.80 + 1.55 + 2.52 + 2.06 = **7.93 h**.
- Only if you decide to complete DFSC-all/test (outside the stated scope): clean rerun ≈ **2.4–2.6 h**, from 85,374 frames at the
  ≈9.7 fps this checkpoint achieved. A hypothetical resume of the remaining 34,365 frames would be ≈1.0 h, but it needs an evaluator change.

## 9. Recommended next execution (not executed)

1. **Run nothing for the authorised scope.** The four authorised results are complete, valid and verified. Use the Section 3 numbers.
2. Decide whether the post-03:38 work (seed-1 test evals, DFSC-all training) is accepted as authorised. If not, keep the seed-1 numbers out
   of reported results. Their ledger entries (lines 7–8) are permanent and should be disclosed either way.
3. If DFSC-all is wanted: do a clean rerun of its test evaluation into a new directory (command in Section 7), after explicit authorisation.
   If it is not wanted: leave the partial directory as-is, as evidence, and report no DFSC-all number.
4. Separately, and only on instruction: correct the three inconsistent report statements listed in Section 5.
