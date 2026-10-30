# MASTER_STATE — one screen (update at every milestone)

**Updated**: 2026-10-13 ~19:30 IST · **branch** `research-program` · OMEN ahead of the HPC checkout (HPC at 6b9415a + untracked reports) · never pushed

**HPC model backup (2026-10-13)**: every trained weight / cache / job record in `~/aerotrack_server` (all but `environments/` and the
`shared_data` dataset store) is copied to `artifacts/hpc_recovery_20261013/` (git-ignored) with per-file SHA-256 verification;
see its `RECOVERY_README.md` + `ARTIFACT_MANIFEST.csv`. HPC `/home` is 100 % full (57 GB free of 25 TB) -- treat HPC files as at risk.

**AeroTrack Flight Deck (2026-10-14)**: `studio.bat` now opens a local multi-model laboratory (7 experiment categories, live inference with synchronised multi-model replay, recorded TEST replay read-only, controlled speed benchmark, forecasting, paper reproduction); 5 checkpoints evaluation-verified (fresh VAL = recorded); 122 tests. `docs/workspace/FLIGHT_DECK.md`.

**Unified workspace (2026-10-13 evening)**: one registry + one studio for both directions, nothing moved.
`studio.bat` → http://localhost:8501 (Overview · Models · Datasets · Inference · Compare · Experiments · Failure analysis · Health).
Registry `registry/` (`tools/build_workspace_registry.py`): 314 checkpoint identities, 13 families, 498 files (workspace + backup), 0
integrity mismatches, 228 checkpoints load- and inference-verified (all runnable ones in the workspace), 317 historical result records (verbatim, sourced). Shared layer
`aerolab/` (adapters wrap the experiments' own code; evaluation service VAL/TRAIN only, TEST locked). Backup-only checkpoints restored
by copy (`registry/migrations/restore_manifest.json`, 137 files); the backup itself is unchanged. **Direction B = the Anti-UAV (2021)
DFSC/GlobalTrack reproduction; the CST paper is a benchmark/protocol used by Direction A.** Docs: `docs/workspace/` (INITIAL_AUDIT,
ARCHITECTURE, OPERATING_GUIDE, FINAL_AUDIT), `research/*/README.md`. Tests: 122 unittest + demo smoke test pass.

## Where we are
Two separate workstreams -- never mix their metrics (CST benchmark state accuracy vs Track C VAL ADE / NLL / mSA).

**CST Anti-UAV benchmark study** (`experiments/cst_anti_uav/`, entry point `reports/FINAL_REPORT.md`) -- **DONE**
- Official toolkit + authors' released predictions reproduce Table 3 exactly; best published CST tracker GlobalTrack **35.92**
  (paper is a benchmark paper; no later CST result found).
- **X1** single registered TEST pass (PBS 93019): STE **60.08** [56.13, 63.97], argmax 59.70; vs GlobalTrack paired
  +24.16 [+20.78, +27.63], 59 / 1.
- **X2** (seeds 0-2 x causal_g4 / single frame; PBS 93016 + 93018 training, 93020 VAL design, 93029 single TEST pass, all exit 0):
  causal_g4 60.08 / 60.26 / 60.60 (mean 60.31, sd 0.26) -- T1 seed-robust; T2 +24.39 [+21.08, +27.80], 60 / 0;
  T3 window vs single frame +11.51 [+8.36, +14.86] SUPPORTED. X2 confirms X1 (+0.23 [-0.66, +1.14]), does not improve it.
- Absence-neutral lead +13.6 SA; weakness: large targets (CST train has 6 large frames) -- not resolved; jungle_18 lock seed-specific.
- Official toolkit == each pass's Track A values (|d| = 0). Next justified experiment (not authorised here): a registered VAL-tuned
  absence model (FINAL_REPORT §6).

**Track C** (`experiments/aerotrack_temporal/`, entry point `reports/TRACK_C_SUMMARY.md`) -- **all planned VAL stages DONE**
- C1c (93015 stages 1-2): removes CST negative transfer (-5.5..-5.9 px vs pooled C1) but does **not** beat classical `dcv`
  (H1c NOT_SUPPORTED) -> no learned forecaster adopted for CST.
- C3 (93015 3a primary, 93021 3b secondary): motion models never improve tracking; Kalman state helps only on CST with the
  temporal detector (+0.84 [+0.24, +1.61]).
- C1b (93021 4a / 4b): learned point forecasting helps Anti-UAV on tracker histories (both caches); learned CST uncertainty badly
  miscalibrated (LOSES on both caches).
- **CF final system: DONE.** CF-A2 (77839f0), `tc_cf.py` (6b9415a), freeze PBS 93030 (node04, exit 0, VAL only; frozen_cf.json bfea36b),
  single registered TEST pass on the OMEN (da1b3e6, `cf_test_ledger.jsonl` -- used up, never rerun): SOT IR 68.89 / RGB 74.10 / CST 56.55;
  CF-a vs Track A GlobalTrack -1.69 [-3.25, +0.06]; CF-c CST +0.27 [-0.27, +0.79]; deployment gate e2e 5.89 / 10.76 ms p50 / p99.
- **P1 presence gate (exploratory, CST VAL only; 27d45e0 registration, 6cc6810 result)**: NO_GAIN -- the gate trades absent-frame
  false alarms for visible misses one-for-one; cross-fitted selection keeps tau_out = 0. A detection-score gate is not the fix.

## Running now / queued
| PBS | run_id | what | target | state |
|---|---|---|---|:---:|
| -- | -- | nothing of ours queued or running (2026-10-13 17:50; 93029 / 93030 F, exit 0). `ENV_BACKUP` 93035 belongs to `~/env_backup`, not this project | -- | -- |

## Next concrete actions
1. All registered TEST budgets (X1, X2, CF) are used up. New work = TRAIN/VAL only; any TEST needs a new registration + authorisation.
2. Next justified experiment (not launched; one e20 CST run = ~46 min on a V100 + VAL STE eval): a *learned* absence / presence
   signal (P1 showed the detection score alone cannot separate absent-frame false alarms from correct boxes at the SA level).
   Register on VAL first (pattern: `experiments/cst_anti_uav/preregistration/P1_presence_gate_VAL.json`), then train via
   `tools/hpc/submit.sh` (see `jobs/x2_train_a/cmd.txt` for the train_v3.py + `--resume` pattern).
3. HPC disk is full: before any new HPC training, check `df -h ~` (57 GB free on 2026-10-13).

## Blocked / risks
- PBS does not enforce memory: 93015 / 93021 used 82 / 154 GB against 32 / 64 GB requests -- request realistically.
- node03 offline; node01's V100 is shared with another user (gpu_guard.sh).
- The OMEN has 24 GB RAM: parallel CPU jobs there were killed by memory pressure (2026-10-12) -- run heavy CPU work on the HPC.

## Done (verified)
- **C2 (`c2_causal_windows_runner`, 92977)**: all four causal windows superior to single frame on CST AP50; `causal_g4` best
  (+11.10 % CST AP50), non-inferior to symmetric `sym_g4` (+0.74 %). `experiments/aerotrack_temporal/reports/C2_causal_windows.md`.
- **Anti-UAV visible TEST (Track A, both arms)**: `normal_rgb` 91 sequences, 83,858 frames, **mSA 72.88 %** (P@20px 93.11 %,
  success AUC 75.43 %, 6.95 FPS); `dfsc_rgb` (PBS 93014, node04, 143.1 min) **mSA 72.90 %** (P@20px 93.23 %, AUC 75.49 %, 9.77 FPS).
  Anti-UAV benchmark -- not CST.
- **X1 VAL freeze (`x1_cst_runner`, 92976)**: `experiments/aerotrack_v3/frozen_x1_params.json` (cross-fitted CST VAL mSA STE 63.61,
  argmax 62.60).
- **A2a (registered) large-scale row, VAL**: IR 72.32 vs paper 72.00 (+0.32, within ±1: conversion faithful); RGB 70.50 vs 67.28 (**+3.22**, outside the band, ABOVE the paper — not explicable by the conversion conventions, which could only lower ours; reported as a discrepancy alongside the IR test surplus).
- **M11 deployment package** `deployment/aerotrack_causal_g4_v2/` (17 files, sha256-verified): model card (causal window channels [t-4, t, t-8], letterbox 640 / pad 114, BGR, classes), frozen STE params, gate, live smoke on a VAL clip (TensorRT fp16, per-frame compute p99 6.9 ms, 0 ms look-ahead), server (V100 fp16) vs laptop (TRT fp16) parity: presence 100 %, top-1 IoU mean 0.985 (min 0.944), |Δscore| 0.0003.
- Track C causal caches (v2_stock_s0, v2_causal_g4_s0) on full VAL: done (inputs for C1b / C3).
- **v2 final TEST (single registered pass, causal_g4)**: detection AP50 CST 0.587 / Anti-UAV 0.909 / pooled 0.732; SOT mSA (STE) IR 69.12 / RGB 74.34 / CST 56.55; F1 STE vs our GlobalTrack (IR test 70.58) −1.46 [−3.00, +0.47] no detectable difference; F3 STE vs argmax better on IR / RGB (+0.22 / +0.24), n.d. on CST. Pass crashed after detection and was finished by a guarded resume (F-05). `experiments/aerotrack_v2/hpc/reports/final_test.json`.
- **post-v2 (HPC)**: causal_g4 VAL SOT mSA (cross-fitted STE / argmax): Anti-UAV IR 80.53 / 80.46, RGB 81.20 / 81.38, CST 63.02 / 62.18; STE − our GlobalTrack on IR VAL +1.24 [−0.44, +4.29] (no detectable difference); `frozen_ste_params.json` committed (e6d5180) before TEST.
- **A2a IR**: converted public GlobalTrack, large-scale row, IR VAL 72.32 vs paper 72.00 (within ±1).
- **H3 (registered)**: H3a NOT_SUPPORTED (8-GPU regime leaves test +5.04 / +3.72 above the paper), H3b NOT_OBSERVED (DFSC − normal val −0.02, test −0.64). `experiments/anti_uav/reports/H3_optimisation_regime.{json,md}`; TEST ledger 12 lines.
- **Deployment gate of causal_g4 (RTX 5050)**: TensorRT fp16 e2e p99 7.99 ms (model p99 4.66 ms), 0 ms look-ahead, AP50 parity ≤ 0.003. `experiments/deployment/reports/gate_v2_causal_g4_s0_e20.md`.
- **v2 campaign (registered, HPC, 2 seeds, VAL)**: CST AP50 seed-mean deltas vs single frame: causal_g4 +0.111 [+0.071, +0.155], sym_g4 +0.104 [+0.065, +0.146]; causal vs sym +0.007 [−0.000, +0.016] (non-inferior); complete vs truncated CST data +0.079 [+0.056, +0.102] (1 seed); Anti-UAV unchanged (|Δ| ≤ 0.001). Selected: causal_g4. `experiments/aerotrack_v2/hpc/reports/v2_decisions.{json,md}`.
- **v2 stage 1 (HPC, seed 0, VAL; descriptive until the registered analysis)**: CST drone AP50 stock 0.564 / sym_g4 0.678 / causal_g4 0.676
  (futility: both candidates survive, deltas +0.114 / +0.112) -> stage 2 (seed 1: stock, sym_g4, causal_g4) running. Each run ~62 min.
- **H3**: VAL evaluations of both arms done (100 / 105 min); TEST passes running (registered, --final-eval).
- **Host checks**: HC1 PASS (2 checkpoints, |Δ| ≤ 0.0015 AP50), HC3 PASS, HC2 −0.0026 (descriptive).
- **M6**: loader 233 / 638 / 553 samples/s at 8 / 12 / 16 workers (GPU-only fp16 231 img/s) -> 12 workers; /local staging 78.7 GB in 55 min (once per node).
- **A2 sanity gate PASS**: converted GlobalTrack zero-shot 74.41 vs untrained control 4.70 mSA (8 RGB TRAIN sequences).
- **CST extracted on the HPC**: 246,131 files, 0 missing / size / CRC mismatches (= OMEN); **data gate PASS on the HPC** (602,613 TRAIN/VAL paths, leakage train/val/test none); `DATA_VERSION` = corpus_v2_f842d03909cf_cst_full_antiuav_hpcdecode.
- **HC3 PASS**: Phase A `normal_ir` VAL on the V100 79.287393 vs OMEN 79.287413 (Δ −0.00002).
- A2 prepared: GlobalTrack checkpoint downloaded (sha256 e5950365…), converted (322/322, exact), registered (`experiments/anti_uav/preregistration/A2_visible_and_largescale.json`).
- HPC forensics (no prior AeroTrack work there); key auth; workspace; env `at311` (torch 2.6.0+cu118, OpenCV 5.0.0.93 = OMEN).
- Archives Det-Fly / Anti-UAV sha256-equal; extracted (CRC) — member lists = OMEN extraction.
- **Anti-UAV frames re-decoded on the HPC (593,802) and byte-identical to the OMEN** (all 636 sequences: counts, 1,908 boundary
  hashes, 2,000-frame random sample, labels) — `reports/hpc/m3_antiuav_frame_identity.json`.
- Corpus v2 core files on the HPC: manifest `f842d039…`, splits `30f0e35f…`, label cache `9e10fb75…` = OMEN.
- **Env gate core PASS on node03** (kernel, builds 418/649 + 448/499 COCO transfer, fp16 GradScaler, checkpoint round trip;
  GPU-only throughput fp16 231 img/s, fp32 172 img/s at batch 32).
- Found + fixed **F-04**: Track A's data loader package had never been committed (ignore rule); committed unchanged.
- Tools: path map, precision helper, PBS framework, sync / pull, ledger generator (127 rows), RUNBOOK.

## Next concrete actions (historical, 2026-10-06 -- superseded; kept as record)
1. HC3 result → if within ±0.05 pt: submit the registered H3 evaluations (VAL ×2, TEST ×2 `--final-eval`, analysis).
2. CST push done → `tools/hpc/extract_cst.py` (PBS CPU job) → `DATA_VERSION` → env gate data part on the HPC.
3. M6 loader benchmark (`train_v2 --preflight-only`, workers 8/12/16) → v2 smoke (causal_g4) → v2 stage 1 (3 runs; GPU-limited:
   run sequentially on node03 + any idle GPU) + v1 reference / HC1 evaluations.

## Blocked / risks (historical, 2026-10-06 -- superseded)
- PBS GPU allocation broken cluster-wide (F-02); GPUs shared with other users without scheduler protection.
- `/home` 99 % full (~437 GB free, shared).
