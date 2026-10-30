# Research program — progress log (branch `research-program`)

Mission: the 2026-10-04 "AeroTrack Research Program" brief (Track A Anti-UAV reproduction closure, Track B AeroTrack-Net as an
integrated, deployable perception stack). Entry point for anyone picking this up: `docs/AGENT_CONTEXT.md`. Every number lives in
a generated JSON / report; this file only records milestones, running jobs and decisions.

## Running / queued jobs (no polling: each has sentinels; check once when notified)

GPU queue (each step waits for the previous one's sentinel; rebuilt 13:44 on 2026-10-05):

| # | job | state dir | waits for | what | expected |
|---|---|---|---|---|---|
| 1 | aerotrack_v2 campaign (Track B), attempt 2 | `runs/aerotrack_v2/_runner/` | — (GPU since 13:31) | smoke OK; v1 reference on v2 VAL; stage 1 (stock / sym_g4 / causal_g4 s0), futility, stage 2, analysis | 7-28 h |
| 2 | post-v2 chain | `runs/_chains/post_v2/` | v2 runner | export, gate, VAL cache, tracking, forecasting, failures, MOT, freeze STE, report | ~1.5 h |
| 3 | causal-detector VAL caches (Track C inputs) | `runs/_chains/post_v2_causal_caches/` | post-v2 | v2_stock_s0, v2_causal_g4_s0 full VAL caches | ~1.5 h |
| 4 | **F1** standard YOLO11n vs as-built (Track B; architecture erratum) | `runs/_chains/f1_std/` | caches | n_std of the v2-selected window, seeds 0 / 1 (futility after seed 0), analyze_f1 | 2.5-9 h |
| 5 | **X1** CST protocol-matched arm (Track B) | `runs/_chains/x1_cst_only/` | F1 | CST-only training of the v2-selected config on the network F1 adopts (s0), VAL eval, CST VAL cache, tracker cross-fit, frozen TEST settings | 3.5-6 h |
| 6 | H3 evaluations (Track A) | `runs/_chains/h3_resume_after_v2/` | X1 | consume the pause flag, run_h3 (4 evals + analysis) | >= 6.7 h |
| 7 | Track C C2 causal windows | `runs/_chains/c2_launch/` -> `runs/aerotrack_temporal/_c2_runner/` | h3_resume_after_v2 (+ v2 decisions guard) | stage 1 g1 / g2 / g8 s0, futility, stage 2, analysis | 7-32 h |

CPU (Windows commit is the binding resource: heavy CPU work runs only in the GPU's no-training window):

| job | state | then |
|---|---|---|
| `cpu_trackc_window` (C1c train -> commit registry -> C1c VAL -> C3 x 2 -> C1b x 2) | waits for X1 (= start of the H3 evaluations, inference only) | Track C adoption decisions |
| C1 VAL pass, Track C tracking evaluation, C3 prep | DONE 2026-10-05 | — |
| X1 TEST pass script (`x1_final.py`) | ready | after X1's VAL design + frozen settings committed |
| event waiter (`tools/wait_events.py`) | reaped twice by the memory reaper; not restarted per the reaper notice | the 2-hourly heartbeat covers sentinel checks |

## Milestones

| when | milestone | evidence |
|---|---|---|
| 2026-10-04 22:5x | context audit; previous sessions' uncommitted repair + generalization work committed unchanged (584846d, 6c840a0); branch `research-program` | `docs/AGENT_CONTEXT.md` |
| 2026-10-04 23:0x | **CST data defect found**: local archive = z01–z07 of 11 parts; we hold 84/120 train, 0/40 val, 60/60 test CST sequences; no urban-areas / water training scenes | AGENT_CONTEXT §B, D.4; dataset README; arXiv:2507.23473 |
| 2026-10-04 23:2x | Anti-UAV H4 pre-registered (4e14b97) and launched; evaluate.py --scale (Phase B routing, overwrite guard); aggregator regression byte-identical | `experiments/anti_uav/preregistration/H4_native_scale.json` |
| 2026-10-04 23:3x | seed-1 TEST results adopted (not re-run) for mission §9.1 P1; Track A ledger with 5 evidence classes | `artifacts/anti_uav_ledger.json` |
| 2026-10-04 23:4x | ONNX export of the selected detector: parity 0.005 px / 1.2e-6; SPD-Conv also exports (4.06 M params, 11.7 GFLOPs vs 2.60 M, 6.6) | `experiments/deployment/reports/export_*.json` |
| 2026-10-04 23:5x | corpus switch (AEROTRACK_CORPUS), split builder uses official CST val when present (v1 reproduced exactly); CST extract/verify/build scripts; novelty audit | 787c7eb; `docs/research/NOVELTY_GAP.md` |
| 2026-10-05 00:0x | aerotrack_v2 campaign pre-registered (6c645b1): stock / sym_g4 / causal_g4 x seeds 0,1 on corpus v2; causal window verified (no future frame) | `experiments/aerotrack_v2/preregistration.json` |
| 2026-10-05 00:1x | systems: STE (Kalman single-target state), SOT-metric harness with cross-fitting, backends + deployment gate; registry; chains armed | 582ea79, ef95980 |
| 2026-10-05 06:57 | **H3 first launch crashed** (Windows commit limit, error 1455, while 4 CPU analysis chains ran) and the retry refused the non-empty run dir; GPU idle 06:57-08:24. Runner now keeps a crashed attempt (`_h3_optimisation/crashed_*`) and retries; relaunched 08:24 | d41d06c; `_h3_optimisation/events.jsonl` |
| 2026-10-05 08:4x | **Track C started** (`experiments/aerotrack_temporal/`): trajectory data from official annotations (complete CST), forecasting core, HOTA + continuity metrics, C1 + C2 registered (e2b051f); regression suite 18 tests | Track C README |
| 2026-10-05 10:13 | **corpus v2 gate passed**: archive COMPLETE_AND_CONSISTENT (246,131 files, 0 missing / size / CRC mismatches; v1 partial extraction byte-identical); corpus v2 OK at 11:09 (CST 120 / 40 / 60, official val, CST TEST labels identical to v1) | ece7fcb; `experiments/cst_completion/reports/` |
| 2026-10-05 11:11 | **v2 attempt 1 failed in its registered smoke run** (pre-flight T2 accepted only symmetric windows); T2 made window-aware (single / symmetric / causal, no frame after t), verified on real corpus-v2 samples; queue rebuilt in the registered order (C2 runner, which would have started before v2, stopped while waiting) | 244587f |
| 2026-10-05 14:0x | **knowledge audit**: architecture erratum (every AeroTrack detector built as C3k2(c3k=True), 64 % COCO load) -> F1 registered; CST protocol audit -> X1; C1 VAL pass: learned forecaster SUPPORTED on Anti-UAV IR / RGB, LOSES on CST; record in `docs/PROJECT_KNOWLEDGE.md` | 47f3c2b, 4c749d6, 897bc0f |
| 2026-10-05 15:59 | **C1c stopped to protect the v2 GPU training** (system commit 1.6 GB free); cause: Track C loader kept ~400 decompressed copies (~6 GB) -> fixed (bit-identical data, +21 MB); C1c relaunched 16:0x; event waiter reaped again by the memory reaper (not restarted; heartbeat + detached chains carry the queue) | 67e944c |

## Execution mechanics (window-safe)

- Every GPU / long job is a detached process with sentinels and resumable state; none depends on the agent session, so a
  usage-window boundary can only delay the agent's reaction, never lose work.
- One event waiter (`tools/wait_events.py`, background task, 2 h cap) wakes the agent when ANY job writes DONE / FAILED;
  after handling, `--ack` marks the sentinels as seen and the waiter is re-armed.
- A session cron (`17 */2 * * *`, auto-expires after 7 days) re-invokes the agent every 2 h as a fallback, so work resumes
  automatically after a usage-window refresh even if a notification was missed. Usage budget itself is not observable from
  inside the session, so it is managed by keeping agent turns event-driven (no polling).

## Results 2026-10-05

- **Anti-UAV H4 (native test-time scale): NOT_SUPPORTED_VAL_BREAKS** (b9433cb). Native 640x512 evaluation lowers val -4.0/-4.6 and
  test -4.4/-4.3 (normal/DFSC): symmetric, i.e. train/test scale mismatch; the train-at-native trigger (asymmetry <= -2) not met.
  Descriptive: native test mSA 66.15/66.01 lands on the paper's 65.36/66.04 (LR/LI drop most) but val falls 4-5 pt below the
  paper. Review-3, ledger, comparison, registry regenerated (d5a9a1e).
- **Anti-UAV H3 pre-registered** after H4 (b235945) and queued after gpu_slot_1 (`experiments/anti_uav/runs/_h3_optimisation/`).

## Data facts established 2026-10-05

- Complete CST release on disk (`data/cst_anti_uav_full/`, streaming extraction of all 11 parts, 246,131 files, every CRC-32 ok):
  train 120 = building 51, cn_mountains 20, cn_sky 8, jungle 5, **urban-areas 23, water 13**; val 40 = building 14,
  cn_mountains 9, cn_sky 2, jungle 1, urban-areas 8, water 6; test 60 (as before); 220 label files. The old corpus lacked
  exactly the 36 urban-areas / water training sequences and the whole val split.
- The independent verification (central directory + byte comparison with the v1 extraction) was stopped by the memory reaper
  under memory pressure (02:0x) and must not be restarted without the user's go-ahead; corpus v2 waits on it.

## Decisions (with reasons)

- **v2 campaign before Anti-UAV H3 (2026-10-05, deviation from the brief's default Phase 6 -> 7 order)**: (i) the CST defect makes
  the complete-corpus campaign the largest information gain available (the dominant AeroTrack test failure coincides with the
  missing data, and the causal-window question decides whether the selected detector is deployable in real time); (ii) H3's
  value depends on H4's verdict (if H4 explains the test surplus, H3 reduces to the DFSC-effect question), so ordering H3 after
  H4's verdict lets that information shape H3's protocol; (iii) the GPU queue is serial on the OMEN, so H3 starts after v2
  unless HPC access arrives. H3 is pre-registered after H4 and before v2 finishes.

- **Seed-1 test not re-run**: the evaluations exist and are verified; re-running only adds test exposure. Adopted with a note.
- **H4 before H3**: evaluation-only (~5 GPU-h) vs ~13 GPU-h of training; H3 to be pre-registered after H4's verdict.
- **Corpus v2 is a new directory**, never an in-place rewrite: every historical run references v1's manifest hash.
- **v2 campaign on official VAL with 2 seeds** instead of 3 dev folds: the official CST val split now exists (40 sequences,
  disjoint by design); folds were introduced because v1's CST val was a carved holdout.
- **Causal window [t-4, t, t-8]**: zero look-ahead; the g=4 symmetric window costs +160 ms at 25 fps.
- **ONNX Runtime 1.30 CUDA EP works** with the venv's CUDA 13 libraries (IO binding); TensorRT 11.3 cu13 to be installed after the
  CST download (wheels on pypi.nvidia.com, sha256 published).
- **HPC**: blocked — no host / user / key anywhere on this machine. Everything runs on the OMEN until access details arrive.

## Prepared, waiting on a decision (code committed, NOT registered / launched)

| item | waits for | where |
|---|---|---|
| final TEST pass of the v2 candidate (detection + SOT + tracking + forecasting, one pass) | v2 decision, post_v2 VAL analyses, commit of `frozen_ste_params.json` | `experiments/aerotrack_v2/final_protocol.json` (registered), `scripts/final_eval.py` |
| post-v2 VAL follow-up (export, gate, cache, tracking, forecast, failures, MOT, freeze, report) | v2 runner DONE | `tools/chains/post_v2.json` (armed) |
| v3 capacity allocation: n_p2 (function-preserving P2, exact proof) / s (YOLO11s) / s single-frame around the v2 window | v2 winner -> write + commit `experiments/aerotrack_v3/preregistration.json` | `experiments/aerotrack_v3/scripts/` |
| cross-dataset generalisation (Anti-UAV only, CST only, no Det-Fly, IR only, RGB only) | same machinery (`--train-domains`), registration after v3 design | `experiments/aerotrack_v3/scripts/` |
| Anti-UAV H3 (8-GPU regime: accumulate 8, LR 0.02; warm-up 500 optimizer steps = mmdet convention) normal + DFSC arms | H4 verdict, then pre-registration | `experiments/anti_uav/configs/phase_b/` (normal arm config to add) |

## Next actions

1. When the CST download reports DONE: `extract_and_verify.py` -> review `reports/extraction.md` -> `build_corpus_v2.py` -> review
   `reports/corpus_v2.md` (counts 120/40/60, CST test labels identical to v1) -> commit -> `run_v2.py --detach --wait-for runs/_chains/gpu_slot_1`.
2. TensorRT: resumable download of the cu13 11.3 libs + bindings wheels, sha256 check, pip install from file; re-run deploy_gate if
   slot 1 ran before it.
3. When H4 is DONE: read `experiments/anti_uav/reports/H4_native_scale.md`, rebuild `artifacts/anti_uav_ledger.json` and the registry,
   decide/pre-register H3.
4. Deployment runtime (one-command video inference with the STE overlay) and failure-analysis tooling (deterministic example
   selection) — engineering while the GPU is busy.
