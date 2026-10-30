# MASTER_PLAN — AeroTrack research program (from 2026-10-06: HPC-trained, OMEN-deployed)

*Roadmap, gates and acceptance criteria. Live status lives in `docs/MASTER_STATE.md`; reasons in `docs/DECISIONS.md`. The science
(registrations, metrics, rules) is unchanged from the 2026-10-04 program; what changes on 2026-10-06 is where it executes.*

## 1. What the project is (reconstructed from rev1/rev2/train notebooks + the repository)

Holistic AeroTrack = **multi-modal (IR + RGB) UAV perception stack**: detection (with a temporal input) → single-target state /
tracking → trajectory prediction with uncertainty → real-time deployment. Review-1 (`rev1.ipynb`) proposed four pillars: SPD-Conv,
TrackNet-style temporal stacking, ByteTrack, Social-LSTM. Controlled evidence since then (see `docs/AGENT_CONTEXT.md` §B):
SPD-Conv **retired** (not supported with fair init), t±1 stack **retired**, t±4 stack **supported** (+0.045 CST AP50), ByteTrack
**retired** (loses to single-target layers), LSTM **retired** (collapsed); replacements: STE Kalman single-target estimator
(validated on v1 VAL), Track C learned/classical forecasters (C1 done on VAL). Track A = faithful reproduction of Jiang et al.
2021 (arXiv:2101.08466v3; DFSC on GlobalTrack, Protocol II IR) — `experiments/anti_uav/`.

### Task matrix (only tasks the data and protocol support)

| task | input | target / labels | data (split) | metric | implementation | status |
|---|---|---|---|---|---|---|
| T1 drone detection (+ bird hard negatives) | 3-frame 9-ch stack, letterbox 640 | YOLO boxes, 2 classes | corpus v2: Anti-UAV IR+RGB 160/67/91, CST 120/40/60, Det-Fly 4,698/1,565/1,567 (sequence-level) | drone AP50 per dataset (CST primary), micro / size-stratified AP, empty-frame FP | `c_model/` + `experiments/aerotrack_v2/` | v2 campaign: 1/6 runs done on OMEN (historical); **re-run on HPC** |
| T2 temporal window choice | gap g ∈ {1,2,4,8}, symmetric vs causal | as T1 | as T1 (VAL) | as T1 + look-ahead latency | v2 (g4 sym/causal), Track C C2 (causal g1/g2/g8) | registered, queued |
| T3 backbone erratum | standard YOLO11n vs as-built | as T1 | as T1 | as T1 | F1 (`experiments/aerotrack_v3/preregistration_F1_*`) | registered, queued |
| T4 CST protocol-matched benchmark | CST-only training | CST boxes | CST 120/40/60 | SA (Anti-UAV Eq. 1) vs CST paper | X1 | registered, queued |
| T5 single-target tracking | detections per frame | exist + gt_rect per frame | Anti-UAV IR/RGB, CST (VAL; TEST once) | mSA, HOTA, IDF1, recovery time | STE + Track C FSTE (`experiments/systems`, `aerotrack_temporal`) | v1 VAL done; v2 after detector |
| T6 trajectory forecasting | track history (oracle / tracker) | future centres 1–24 frames | official annotations (TRAIN tune, VAL once) | ADE/FDE, NLL, coverage, calibration | Track C C1 / C1b / C1c | C1 done (learned SUPPORTED on Anti-UAV, loses on CST); C1b/C1c queued |
| T7 deployment | live / video frames | — | — | p50/p95/p99 latency, FPS, memory, AP parity | `experiments/deployment/` (ONNX, TensorRT, stream) | gate done for v1 detector; redo for final |
| A1 Anti-UAV reproduction | IR frames, GlobalTrack pairs | gt_rect / exist | Anti-UAV IR 160/67/91 | mSA (paper Eq. 1) | `experiments/anti_uav/` | val reproduced; test +4–5 above paper; H4 done; **H3 evals pending** |
| A2 visible / large-scale rows | — | — | — | — | needs GlobalTrack mmdet checkpoint conversion | **blocked** (component unavailable; documented) |

Not supported by the data (not implemented): multi-target MOT benchmarks (one target per sequence), bird tracking (Det-Fly is
single images), CST RGB (CST is IR only).

## 2. Milestones and gates

| M | milestone | acceptance criterion (gate) | depends on |
|---|---|---|---|
| M0 | local reconstruction | state documented; tests pass on OMEN | — | 
| M1 | remote forensics | `docs/REMOTE_STATE.md` §3 inventory | — |
| M2 | control documents | MASTER_STATE / PLAN / DECISIONS / REMOTE_STATE / FAILURES / HANDOFF / ledger generator | M0, M1 |
| M3 | remote workspace + data | archives sha256-equal; extraction counts = OMEN (Anti-UAV 636 clips; CST 246,131 files CRC ok; Det-Fly 7,830 images); derived bundle sha256-equal; manifest/splits sha256 = recorded (`f842d039…`, `30f0e35f…`); Anti-UAV frames byte-identical to OMEN (Track A manifest `frame_sha256` first/last of every sequence + a random 2,000-frame sample) | M1 |
| M4 | environment gate | in a PBS job on a V100: CUDA kernel, model build (as-built + std YOLO11n), fp16 forward/backward, GradScaler step, checkpoint save/load round-trip, real-data DataLoader, `c_model/preflight.py` suite, `tests/test_research_program.py` all pass | M3 |
| M5 | protocol freeze | amendment A1 (HPC host, fp16, parallel execution; nothing scientific changed) committed before the first HPC run; host-equivalence checks defined | M4 |
| M6 | throughput benchmark | 300-iteration timing per (workers, batch) on a V100 node; choose loader settings; GPU util and img/s recorded | M4 |
| M7 | Track A H3 | host check: `normal_ir` VAL mSA on V100 within 0.05 pt of 79.29; then the registered H3 runner (4 evaluations + analysis) | M4 |
| M8 | Track B v2 campaign | 6 runs (3 configs × 2 seeds), VAL evaluation, `analyze` applies the registered rules (+ host check stock_s0 HPC vs OMEN, descriptive) | M5, M6 |
| M9 | post-v2 | caches, tracking, forecasting, failures, freeze STE; F1 → X1 → C2; Track C C1b/C1c/C3 | M8 |
| M10 | sealed final test | one pre-registered pass per final candidate (`final_protocol.json`) | M9 |
| M11 | deployment package | export (ONNX/TensorRT), load + parity on the RTX 5050, latency/FPS, checksums, same-clip server-vs-laptop parity | M10 |
| M12 | final report + reproducibility | RESULTS.md and the report generated from JSON; reproduction table PAPER / OURS-FAITHFUL / OURS-CONTROLLED | all |

## 3. Execution rules on the HPC (details: `docs/EXPERIMENT_PROTOCOL.md` §HPC, runbook `tools/hpc/RUNBOOK.md`)

- PBS jobs, CPU-only request + idle-GPU guard, one GPU per job, data staged to `/local`, checkpoints on `/home` (NFS).
- Every run: unique run ID, immutable run directory, `run_meta.json` (command, commit, host, GPU, env, precision, data hashes),
  stdout/stderr log, sentinel (`RUNNING`/`DONE`/`FAILED`), resumable `last.pt` each epoch (atomic write).
- No polling: jobs are checked when a dependency needs their output; status via `qstat` + sentinels.
- Order by information per GPU-hour: host checks (minutes) → benchmark → v2 stage 1 in parallel → H3 evaluations alongside.
