# CST Anti-UAV study — initial audit (2026-10-12)

Scope: environment, server and job state; identity and integrity of the local CST Anti-UAV release; how CST has been used so far
in this project; which existing artifacts can be reused. Machine-readable companion: `dataset_manifest.json`
(`scripts/build_manifest.py`). Everything below was checked live on 2026-10-12 between 02:30 and 03:30 IST unless marked otherwise.

## 1. Environment and server

| item | verified value |
|---|---|
| local repository | `A:\AEROTRACK_OMEN` (git root), branch `research-program`, HEAD `1745604` |
| SSH alias | `amritahpc` → `y_rajeshwar@172.17.18.100:22`, key `~/.ssh/aerotrack_hpc_ed25519` (BatchMode login works; no password used) |
| remote project | `~/aerotrack_server/source` (git, branch `research-program`, HEAD `cd433bb`), jobs in `~/aerotrack_server/jobs/<run_id>/` |
| remote env | `at311` (conda, `~/aerotrack_server/environments/at311`): Python 3.11.13, torch 2.6.0+cu118, numpy 1.26.4, OpenCV 5.0.0, ultralytics 8.4.115; activated by `tools/hpc/job_env.sh` |
| scheduler | PBS Pro, queue `workq`; 4 compute nodes × 28 CPUs × 1 V100 (`ngpus` cannot be requested — `RUNBOOK.md`; GPU exclusivity via `gpu_guard.sh`) |
| storage | `/home` 25 T, **100 % used, 97 G free** (shared by all users) — keep new outputs small |
| repo instruction files | `HANDOFF.md`, `ENTRY.md`, `CONVENTIONS.md` do **not** exist (local or remote); the project's equivalents are `docs/MASTER_STATE.md`, `docs/SESSION_HANDOFF.md`, `docs/AGENT_CONTEXT.md`, `tools/hpc/RUNBOOK.md` |

### Job 93015.amritahpc (`track_c_pipeline`)
Live at 02:30 IST: **R**unning on node04 since 01:42 (16 CPUs, 32 GB requested, walltime 03:30:00, CPU-only, `gpu 0`).
Stages: C1c train (done 02:15; the job itself committed `cd433bb` "c1c: record dev-chosen forecaster models" in the HPC
checkout) → C1c VAL (done 02:15) → C3 tracking evaluation (running since 02:15) → C1b. Observed memory use 67.8 GB (above
the 32 GB request; PBS does not enforce it). Not touched. Its final state is recorded in `FINAL_REPORT.md`.

### Local / remote repository divergence
Common ancestor `168c27d`. OMEN has 4 commits the HPC lacks (`90a3561`, `3f8c189`, `141577f`, `1745604`); the HPC has 1 commit
the OMEN lacks (`cd433bb`, made by job 93015). Uncommitted on the HPC: `track_eval.py` (same 2-line label-path fallback as the OMEN
copy, CRLF line endings), `docs/MASTER_STATE.md`, `docs/SESSION_HANDOFF.md`, Track C files rewritten by the running job
(`runs/aerotrack_temporal/_data/traj_*.json`, `_c2_runner/events.jsonl`) and several untracked reports. `tools/hpc/sync_code.sh`
fast-forwards only, so code cannot be shipped until `cd433bb` is merged on the OMEN and the HPC-modified files are reconciled —
which must wait for 93015 (it rewrites those files and spawns fresh processes from the checkout).

## 2. Dataset identity and integrity

| check | result |
|---|---|
| release | CST Anti-UAV (Xie et al., ICCVW 2025, arXiv:2507.23473 v2), Google Drive folder 1 of the official README; 11 archive parts, sha256 of each in `experiments/cst_completion/reports/volumes.json` |
| extraction | 246,131 archive members, 0 missing / size / CRC mismatches (`experiments/cst_completion/reports/extraction.md`, 2026-10-05); identical tree on the HPC |
| location | `data/cst_anti_uav_full/CST-AntiUAV/CST-AntiUAV/{train,val,test}` (git-ignored) |
| official repo | `github.com/PCwenyue/CST-Anti-UAV` commit `bd92b984` (2026-09-06) cloned to `data/cst_anti_uav_official/toolkit` |
| sequences | 220 = train 120 / val 40 / test 60 (= paper §3.3) |
| frames | 245,471 (train 127,991 / val 45,305 / test 72,175); frames on disk = label length for every sequence |
| visible boxes | 208,221 (paper: "over 240k bounding box annotations" counts all annotated frames incl. target-absent ones) |
| target absent | 37,250 frames (train 18,075 / val 6,250 / test 12,925), `exist = 0` with an empty box |
| image size | 640×512 single-channel-content JPEG for all 220 sequences |
| OPE start frame | frame 0 is visible in **all** 220 sequences |
| visible frames without a valid box | 0 |
| agreement with the official repo's `annos/<split>/<seq>.txt` | all 220 sequences: same length, identical boxes on every visible frame |
| sequence attributes | official `annos/<split>/att/*.txt` (11 flags: OC OV SV TC DBC CDB FM TS SS MS NS) |
| frame attributes | `IR_label.json` keys OC OV SV TC DBC CDB FM (+ `SIZE` 1–4; one TEST sequence carries TS/SS/MS/NS keys instead) |
| size bins | `SIZE` 1/2/3/4 = GT diagonal ≤10 / (10,30] / (30,50] / >50 px (verified: code 1 spans 3.3–10.0 px, 4 spans 50.0–84.8 px) |
| test size distribution | tiny 10,354 / small 45,812 / normal 2,600 / large 484 visible frames |
| name overlap between splits | none |
| independence | paper §3.3: "the test set remains strictly independent, while training and validation sets derive from non-overlapping clips of shared sequences" — **VAL is not independent of TRAIN** (same source videos), so VAL scores are expected to be optimistic relative to TEST |

## 3. How CST has been used in this project so far

| use | split | evidence |
|---|---|---|
| detector training, corpus v2 (all 120 CST train sequences, + Anti-UAV + Det-Fly) | train | `experiments/aerotrack_v2/` (v2 campaign) |
| detector training, CST only (X1) | train | `runs/v3_n__causal_g4_cst_s0_e20` (HPC; last.pt sha256 `e52a5645…`) |
| window / tracker selection | val (official 40) | v2 decisions, `frozen_ste_params.json`, `frozen_x1_params.json` (HPC, untracked) |
| corpus v1 generalization TEST (detection AP50 only; CST train truncated to 84 seq) | test | `docs/RESULTS.md` §2.1 (CST AP50 0.521) |
| v2 final TEST (registered single pass; all-domain causal_g4; SOT STE mSA **56.55** on the 60 CST test sequences) | test | `experiments/aerotrack_v2/hpc/reports/final_test.json` |
| X1 TEST (protocol-matched, CST-only) | test | **never run** (`x1_test_ledger.jsonl` absent; no CST test cache for the X1 run) |

CST TEST has therefore been scored twice before this study, both times by registered single passes after which nothing was
tuned. The v2 number is **not** a protocol-matched benchmark result (its detector also trained on Anti-UAV and Det-Fly); the
registration already labelled it descriptive (`F2`).

## 4. Reusable artifacts (verified)

| artifact | status |
|---|---|
| X1 candidate `v3_n__causal_g4_cst_s0_e20` (YOLO11n as built, causal window [t-4, t, t-8], CST train only, seed 0, 20 epochs) | trained on the HPC 2026-10-10 (44.7 min), VAL AP50 0.694; VAL detection cache present |
| `frozen_x1_params.json` (STE `det_thr 0.15, init_thr 0.5, reacq_thr 0.5, max_coast 5, output det, coast_output false, gate_chi2 25`) | written 2026-10-10 07:30 by `run_x1.py`, tuned on all CST VAL; VAL cross-fitted STE mSA 63.61 (argmax 62.60) — **not yet committed** (the TEST script refuses until it is) |
| X1 registration (`experiments/aerotrack_v3/preregistration_X1_cst_protocol_matched.json`) | committed 2026-10-05, before any CST-only training |
| `x1_final.py` (single registered TEST pass) | present on both machines (identical md5); does not save per-frame predictions or per-sequence SA |
| Track A SA implementation | identical to the official toolkit on CST (see `PAPER_PROTOCOL.md` §4) |
| Anti-UAV visible TEST (normal_rgb 72.880 / dfsc_rgb 72.904 mSA, 91 seq, 83,858 frames) | matches `experiments/anti_uav/runs/*/eval_test_RGB_last/metrics.json` and ledger rows 13–14 — a **different benchmark**, not used for any CST claim |
| deployment gate of `v2_causal_g4_s0_e20` (same network and window as X1) | TensorRT fp16 end-to-end p99 8.0 ms on the RTX 5050 (`experiments/deployment/reports/gate_v2_causal_g4_s0_e20.md`) |

## 5. Gaps that this study closes

1. The published 35.92 had never been reproduced with the official evaluator → done (`PAPER_PROTOCOL.md`).
2. No per-sequence published results were available, so X1-a could only compare against a fixed number → the official
   "TrainedCST" result files give per-sequence SA for all 21 released trackers, enabling paired comparisons.
3. X1 is single-seed and has no single-frame control under the SOT protocol → registered extension X2 (seeds 1–2, CST-only
   single-frame arm).
