# Experiment protocol — research program (both tracks)

*Rules that every experiment of the research program follows, and where the code enforces each one. Track A's detailed rules
remain in `docs/ANTI_UAV_EXPERIMENT_PROTOCOL.md`; this document adds the program-wide ones and the AeroTrack specifics.*

## 1. Data

| rule | enforced by |
|---|---|
| Three concepts kept apart: SOURCE (`datasets/` archives, `data/` extractions), DERIVED (`data/derived/data_unified*/`), EXPERIMENT (fold files, frozen manifests) | directory layout; `docs/AGENT_CONTEXT.md` §D.4 |
| A derived corpus is never rewritten; a new version gets a new directory and is selected per process with `AEROTRACK_CORPUS` | `src/pipeline/corpus.py`; `build_corpus_v2.py` refuses to overwrite |
| Sequence-level splits; train ∩ val ∩ test = ∅ asserted before every run (roadmap T4) | `3_prepare_training.py`, `stacked_dataset.assert_no_leakage`, preflight |
| CST val = the official split when the corpus has it, else v1's fixed-seed holdout | `3_prepare_training.build_splits` |
| Archive integrity by CRC-32 of every member and SHA-256 of every part, never by file size | `extract_and_verify.py`, `fetch_missing_volumes.py` |

## 2. Pre-registration

- Before the first run of a campaign: question, configurations, held-fixed factors, seeds, metric, checkpoint, decision rule,
  margin, futility rule, test usage — in a JSON committed to git (the commit time is the registration time).
- Runners refuse to start if the registration is uncommitted or modified (`run_v2.py::preconditions`).
- Rules are applied by code (`analyze_*.py`), never by reading tables; a registered rule is never relabelled after the fact.
- Anything run outside a registration is labelled EXPLORATORY and never merged into confirmatory numbers.

## 3. Test-set sealing

| series | guard | ledger |
|---|---|---|
| Anti-UAV (Track A) | `load_sequences(purpose='final_eval')` only; `evaluate.py --split test` needs `--final-eval`; Phase B variants flagged | `experiments/anti_uav/reports/test_evaluation_ledger.jsonl` |
| AeroTrack repair grid | once per checkpoint | `experiments/aerotrack_repair/reports/test_evaluation_ledger.jsonl` |
| AeroTrack generalization | refuses a second pass per checkpoint | `experiments/aerotrack_generalization/reports/gen_test_ledger.jsonl` |
| AeroTrack v2 | `eval_v2.py --set test --final-eval`, refuses a second pass, smoke runs never | `experiments/aerotrack_v2/reports/v2_test_ledger.jsonl` |
| systems / deployment | VAL only (scripts refuse TEST) | — |

A final candidate gets ONE test pass that computes every metric it will ever report (detection AP, SOT mSA, tracking,
forecasting) in that pass. Existing test results are historical evidence and are never recomputed under another protocol.

## 4. Training

- Recipe locked per series (`src/model/train.py` STAGES["baseline"]); a resumed run must match its `run_meta.json`.
- Function-preserving initialisation, proved at runtime (the run aborts if the check fails).
- Pre-flight T1–T8 on the exact job (tensor shape/range, temporal boundaries, labels, leakage, collate, checkpoint round
  trip, throughput, VRAM); a smoke run (2 epochs × 512 samples) before every campaign.
- Checkpoints written atomically; `last.pt` is the scored checkpoint, fixed before any run (`best.pt` kept, never used:
  the fitness metric picked a worse checkpoint in the repair audit).
- Seed = recorded and set before the model is built; multi-seed claims report per-seed values, mean, spread.

## 5. Evaluation

| what | protocol |
|---|---|
| detection | letterbox 640, conf 0.001, NMS IoU 0.7, max_det 30, bf16, every dataset capped at 5,000 images evenly over its sequences; same frames for every configuration (identity hash checked) |
| SOT / tracking | GT box at the official start frame, then box or "absent" per frame; mSA = Anti-UAV paper Eq. (1) by Track A's code on the official annotations; hyper-parameters cross-fitted over two halves of VAL |
| forecasting | from the tracker's own outputs; Kalman vs zero / last / mean velocity; error by horizon; ellipse coverage |
| deployment | idle GPU only; batch 1; p50/p95/p99 not means alone; end-to-end loop timed as one loop; AP parity per backend vs the evaluated protocol |
| statistics | paired, dataset-stratified, sequence-level bootstrap (B = 1000); seed effects via per-seed deltas |

## 6. Long runs

- Every long job is a detached runner with RUNNING / DONE / FAILED sentinels, an events log and per-job logs; it is never
  polled. Completion is observed through a background waiter's notification and checked once.
- Short jobs that need an idle GPU run in a chain between campaigns (`scripts/chain.py`, `scripts/chains/*.json`).

## 7. Reporting

- Numbers come from generated JSON; Markdown reports are generated from it; README.md is generated
  (`experiments/aerotrack_repair/README.template.md` → `render_report.py`).
- One registry indexes everything: `results/artifacts/experiment_registry.json` (`scripts/build_registry.py`); Track A's evidence
  classes are in `results/artifacts/anti_uav_ledger.json` (`scripts/build_anti_uav_ledger.py`).
- Negative and failed results stay in the ledgers and reports.

## 8. Execution on the HPC (from 2026-10-06; amendment A1, `experiments/AMENDMENT_A1_HPC.json`)

| rule | enforced by |
|---|---|
| No training on the OMEN; every training run and multi-hour research evaluation runs as a PBS job on the HPC | `docs/DECISIONS.md` D-2026-10-06-01; `scripts/hpc/submit.sh` |
| One GPU per job; a job refuses to start (GPU_BUSY, exit 75) if any other process is on the node's V100 | `scripts/hpc/gpu_guard.sh` |
| Identical data on both machines: archives sha256-equal, extractions CRC-verified, Anti-UAV frames byte-identical, corpus files sha256-equal; manifests are never rewritten (load-time `AEROTRACK_PATH_MAP`) | `scripts/hpc/{push_files.sh,extract_cst.py,verify_antiuav_frames.py,env_gate.py --part data}` |
| AeroTrack autocast = fp16 on the V100 (bf16 on sm ≥ 8.0); Track A fp32; recorded per run | `src/model/precision.py`; job `env.json`; `run_meta.json` |
| Every comparison is within one host type; cross-host numbers only appear as labelled host checks (HC1–HC3) | amendment A1 |
| Every job: unique id, `cmd.txt`, `env.json` (commit, dirty, torch, GPU, env vars), `events.jsonl`, `job.log`, RUNNING/DONE/FAILED sentinel; a DONE id is immutable | `scripts/hpc/run_job.pbs` |
| Code reaches the HPC only as committed history (fast-forward); results return via `pull_results.py` (namespaced; Track A append-only check) | `scripts/hpc/sync_code.sh`, `scripts/hpc/pull_results.py` |
| The master ledger is generated, never edited | `scripts/build_master_ledger.py` → `docs/EXPERIMENT_LEDGER.{jsonl,csv}` |
