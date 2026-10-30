# Unified workspace — architecture

The research code stays where the experiments put it. The workspace adds **one registry**, **one thin shared layer** (`src/aerolab/`)
and **one interface** (`app/`) on top, without moving checkpoints, data or experiment code.

```text
                 results/registry/catalog/*.json  (hand-curated: directions, papers, families, protocols, datasets, findings — no metrics)
                              │
 files on disk ──────┐        ▼
 results/runs/, experiments/, ├──► scripts/build_workspace_registry.py ──► results/registry/models.json        one entry per checkpoint identity
 src/deployment/,         │      (hash cache, recorded-hash cross-     results/registry/checkpoints.csv    every physical file (workspace + backup)
 src/model/, results/artifacts/ │       check, report importers)             results/registry/model_lineage.json parent / same-bytes / export edges
 recovery backup ─────┘                                            results/registry/duplicates.json    byte-identical groups (never deleted)
 report JSONs, ledgers ─────────────────────────────────────────►  results/registry/experiments.jsonl  historical results (verbatim + source)
                                                                   results/registry/datasets.json      roots, splits, manifests, counts
 scripts/verify_adapters.py ──► results/registry/verification/adapter_checks.json (load / inference evidence, merged at the next build)
 scripts/restore_from_recovery.py ──► results/registry/migrations/restore_manifest.json (copies out of the backup, verified)

 src/aerolab/  (no torch at import time)
   paths · hashing · schemas (versioned JSON schemas) · registry (read-only queries) · test_lock · compare · data
   adapters/  base ─ aerotrack_detector (Direction A detector + STE / argmax)
                   ─ globaltrack        (Direction B reproduction tracker)
                   ─ forecaster         (Track C learned forecasters)
   evaluation (VAL/TRAIN only → results/runs/studio/evals/<id>/ + results/runs/studio/index.jsonl) · failures · health
   Flight Deck layer: categories (7 experiment categories, eligibility, evidence labels) · sequences (sequence index + grounded
   presets) · recorded (catalog of recorded per-frame predictions, read-only replay) · flightdeck (multi-model runner: sequential /
   lockstep, cancellation, manifests) · jobs (one background worker) · benchmark (controlled speed) · forecasting (Category F runner)
   · analysis (tracking episodes, presence curves, seed and latency statistics) · evidence (record-derived numbers)
   adapters gain stream(): per-frame incremental inference built from the same calls as predict()

 app/app.py  (Streamlit, st.navigation)
   Flight Deck: Flight Deck (home) · Test Bench · Live Comparison · Results & Failures · Model Vault       (app/fd.py components)
   Workspace tools: overview · registry table · datasets · single-model inference · recorded comparison · history · failures · health
```

The Flight Deck (2026-10-13) is described in `docs/workspace/FLIGHT_DECK.md`; its design notes are in
`docs/workspace/FLIGHT_DECK_IMPLEMENTATION_PLAN.md`.

## The two directions and what is shared

| | Direction A — AeroTrack system | Direction B — Anti-UAV paper reproduction |
|---|---|---|
| registry `direction` | `aerotrack` | `antiuav_repro` |
| code | `src/model/`, `src/pipeline/`, `experiments/aerotrack_*`, `systems`, `deployment`, `cst_anti_uav` | `experiments/anti_uav/` |
| adapters | `aerotrack_detector`, `trackc_forecaster` | `globaltrack_repro` |
| README | `docs/research/aero_multi_paper/README.md` | `docs/research/anti_uav_paper_reproduction/README.md` |

Shared (`direction = shared`): the CST benchmark protocol and its 21 released baselines (reproduced from the authors' predictions),
COCO YOLO11 bases, the official GlobalTrack weights. Shared *software*: registry, schemas, evaluation service, metric code (Track A's
`anti_uav_repro.eval.metrics`, shown equal to the official CST toolkit). The directions never share a result row. A cross-direction
comparison is numeric only when `aerolab.compare.compatibility` finds the same protocol, split and comparability keys.

## Model registry

- **Identity** = SHA-256. `model_id = <family>:<run_id>@<checkpoint_tag>`, for example
  `aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last`.
- **Same name, different bytes**: when the workspace and the backup hold different files at the same path (a run id trained on the OMEN
  and re-trained on the HPC), the backup copy becomes its own model `…@last~hpc`. It is verified against its own `run_meta.json`.
- **Same bytes, different names** (for example `normal_ir@last` = `normal_ir@epoch_18`): both ids are kept, linked by `same_bytes_as`
  and listed in `duplicates.json`. Nothing is deleted.
- **One checkpoint, several roles**: X1 = X2 temporal seed 0 is a single entry with both aliases and both registrations
  (`results/registry/catalog/families.json` → `run_roles`).
- **Status**: registered / validated / experimental / legacy / retired / invalid / smoke / external. **Verification levels**: integrity
  (SHA-256 equals an independent record), load, inference (`scripts/verify_adapters.py`), evaluation.
- **Queries** (`aerolab.registry.Registry`): `versions(family)`; `latest(family)` (file time only); `demonstration_version(family)`
  (registered); `best_validated(family, protocol, metric, split, **match)`, which never ranks across protocols.

## Adapters (inference contract)

`ModelAdapter.can_run / load / validate_inputs / predict / get_metadata / unload` (`src/aerolab/adapters/base.py`). Each adapter wraps the
experiment's own evaluated code, never a re-implementation:

- `aerotrack_detector`: `aerotrack_trainer.load_checkpoint` → `cache_detections.windows` (the run's window from `run_meta.json`)
  → letterbox → NMS 0.01 / 0.7 / 10 → `track_eval.run_tracker` (frame-0 GT init). Tracker settings come from the frozen VAL selection
  when one exists. Parity with the HPC VAL detection cache: median top-box IoU 0.99. SA matches the cached-detection run to within
  0.02 (fp16 vs bf16).
- `globaltrack_repro`: `QGRCNN` + `GlobalTrackTracker` exactly as `experiments/anti_uav/scripts/evaluate.py` / the VAL demo.
- `trackc_forecaster`: `tc_learned.load/predict` on `tc_forecast.make_samples`, with classical references (cp, cv, TRAIN-tuned dcv).

Every adapter verifies the checkpoint's SHA-256 before loading. It never loads from the recovery backup, and it refuses checkpoints
whose temporal window cannot be determined (no `run_meta.json`). In that case it uses a byte-identical copy that has one, for example
the deployment package for the CF detector.

## Evaluation service and manifests

`aerolab.evaluation.evaluate_sot / evaluate_forecast`: TEST refused → model resolved → checksum verified → adapter → task metric →
`results/runs/studio/evals/<id>/{manifest.json, per_sequence.json, predictions/, log.txt}` + one line in `results/runs/studio/index.jsonl`.
The manifest (`src/aerolab/studio_manifest/1`) records model, checkpoint path and SHA-256, config hash, dataset manifest hash, split,
protocol id, metric definition version, preprocessing, tracker settings and their source, seed, git commit and dirty flag, environment,
device, timestamps, status, metrics, per-sample path, log path and warnings. A truncated run uses protocol `studio.sot_smoke.v1`, which
is never comparable with a full-sequence mSA. Failures are recorded as `failed` with the error; they are never silently dropped.

## Test lock

`aerolab.test_lock` lists every TEST ledger in the repository. `assert_split_allowed("test")` raises in the data layer, in the
evaluation service, in the Flight Deck runner, benchmark and forecasting runner, and through the UI. Recorded TEST results are shown
read-only, with their registration. One purpose may open TEST frames and labels: `REPLAY_PURPOSE` (`replay_recorded`), used only by the
Live Comparison's recorded-prediction replay to draw the registered passes' own predictions; no model is loaded on that path and the
runners never accept it. A new final evaluation needs its own pre-registration and script outside the studio.

## Artifact paths and migration strategy

- **No moves.** The registry gives stable discovery; physical paths stay where every existing script expects them.
- Checkpoints that existed **only in the recovery backup** were **copied** (never moved) back to their original repo-relative path
  (`scripts/restore_from_recovery.py`; 137 files, 2.32 GB, each SHA-256-verified). The copy is never made where a file already exists or
  into the pre-existing untracked directories, and each restore is logged in `results/registry/migrations/restore_manifest.json`.
- 32 model entries remain **backup-only** on purpose (browsable, not runnable):
  - 24 intermediate RGB epoch checkpoints;
  - 6 `~hpc` re-runs, whose OMEN namesakes occupy the same path, so restoring them would overwrite different weights;
  - `v2_causal_g4_s0_e20@best`, which would go into the protected pre-existing `results/runs/v2_causal_g4_s0_e20/`;
  - the official GlobalTrack weights, which were stored outside the HPC source tree.
- The backup is read-only; `scripts/hpc/recovery_manifest.py --out …` verifies it without writing into it.
- A future canonical store (for example `models/checkpoints/…`) would need a copy, verification, a path-consumer update and a
  migration-ledger entry per file. With 86 GB free it is possible but not needed, because the registry already provides one place to
  find everything.

## Adding a new model

1. Train / save it where its series puts runs (for example `results/runs/<run_id>/weights/last.pt` + `run_meta.json`).
2. If its path matches no family, add a family (or a pattern) to `results/registry/catalog/families.json`. Use `run_roles` for registered roles.
3. `venv\Scripts\python.exe scripts\build_workspace_registry.py`, then `scripts\verify_adapters.py --family <id>`, then rebuild.
4. It appears in the studio. Results enter the history through the evaluation service (VAL) or through a report importer in
   `scripts/build_workspace_registry.py` (registered passes).
