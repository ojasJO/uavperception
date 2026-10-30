# Unified workspace — final audit (2026-10-13)

Companion to `INITIAL_AUDIT.md` (state before) and `ARCHITECTURE.md` / `OPERATING_GUIDE.md` (how it works / how to run it).

## 1. What changed

**New, version-controlled**

| path | what |
|---|---|
| `src/aerolab/` | shared layer: `paths`, `hashing`, `schemas` (versioned JSON schemas), `registry`, `test_lock`, `compare`, `data`, `adapters/` (`aerotrack_detector`, `globaltrack`, `forecaster`), `evaluation`, `failures`, `health` |
| `app/` + `studio.bat` | Streamlit studio, 8 pages, localhost-only launcher |
| `results/registry/catalog/*.json` | hand-curated provenance: directions, papers (citation status), 13 model families + run roles, 7 protocols, 7 datasets, findings / limitations |
| `results/registry/*.json, *.csv, *.jsonl` | generated registry (models, files, lineage, duplicates, datasets, experiments, build summary), adapter evidence, restore ledger |
| `scripts/build_workspace_registry.py`, `scripts/verify_adapters.py`, `scripts/restore_from_recovery.py`, `scripts/render_workspace_docs.py` | registry build, adapter verification, backup restore (copy-only), generated README tables |
| `tests/test_workspace_registry.py` (30), `tests/test_workspace_adapters.py` (12), `tests/test_studio_ui.py` (10) | new tests |
| `docs/research/aero_multi_paper/README.md`, `docs/research/anti_uav_paper_reproduction/README.md` | per-direction READMEs (provenance matrix; generated result tables) |
| `docs/workspace/{INITIAL_AUDIT,ARCHITECTURE,OPERATING_GUIDE,FINAL_AUDIT}.md`, `results/registry/README.md` | documentation |
| `results/runs/<18 restored run dirs>/{args.json, init_check.json, results.csv, run_meta.json, training_curves.png}` | run metadata copied from the backup (2.7 MB; tracked like every other run's metadata; weights git-ignored) |

**Modified**

| path | change |
|---|---|
| `scripts/hpc/recovery_manifest.py` | optional `--out DIR` (default behaviour unchanged) so the backup can be verified without writing into it |
| `README.md` + `experiments/aerotrack_repair/README.template.md` | identical "Start here: the unified research workspace" block (no numbers), so a future `render_report.py` run keeps it |
| `.gitignore` | `results/runs/studio/`, `results/registry/.hash_cache.json` |
| `docs/MASTER_STATE.md`, `docs/SESSION_HANDOFF.md` | current state; the stale 2026-10-06 trailing sections of MASTER_STATE are now labelled historical (text kept) |

**Copied out of the recovery backup (git-ignored weights)**: 137 files, 2.32 GB, each SHA-256-verified before being moved into place
(`results/registry/migrations/restore_manifest.json`). This covers the X1 / X2 / F1 `v3_*` runs, 10 v2-campaign runs, `laptop_v2_stock_s0_e20`, 15
C1c forecasters, 3 ONNX exports, and the `normal_rgb` / `dfsc_rgb` / `largescale_globaltrack` final checkpoints. Nothing was moved,
renamed or deleted anywhere.

## 2. Compatibility decisions

- **Direction B is the Anti-UAV (2021) DFSC / GlobalTrack reproduction**, not the CST work; CST (arXiv:2507.23473) is a benchmark
  protocol used by Direction A (evidence: `experiments/anti_uav/README.md`, `experiments/cst_anti_uav/reports/FINAL_REPORT.md`).
- No physical model store: the registry is the single discovery point. Every path stays where existing scripts expect it.
- Adapters wrap the evaluated pipelines (`cache_detections` / `track_eval` / `GlobalTrackTracker` / `tc_learned`); no model was
  re-implemented. Root `README.md` stays generated; only an identical navigation block was added to it and its template.
- The original Direction B VAL demo (`experiments/anti_uav/demo/`) is unchanged and still passes its smoke test.

## 3. Registry counts (method)

| quantity | value | counting rule |
|---|---:|---|
| physical files registered | 498 | `.pt .pth .ckpt .safetensors .onnx .engine .bin .joblib .pkl` under results/runs/, experiments/, src/deployment/, src/model/, results/artifacts/deploy/ + the backup; `.npz` arrays excluded (1,193 in the backup) |
| — workspace / backup copies | 343 / 155 | a file present in both places counts twice |
| — model weights / exports / detection caches | 372 / 20 / 106 | |
| distinct SHA-256 (all files / weights + exports) | 373 / 299 | byte identity |
| **model entries** (checkpoint identities) | **314** | one per `family:run@tag`; workspace and backup copies of identical bytes are one entry; different bytes at one path = two entries (`~hpc`) |
| model families | 13 | `results/registry/catalog/families.json` |
| integrity verified against an independent record | 131 | SHA-256 equals run metadata / frozen config / learned registry / export record / ledger / HPC hash |
| hashed, no independent record | 183 | mostly intermediate epoch checkpoints (hash recorded now) |
| not verified | 29 | exactly the backup-only entries without a workspace copy (24 RGB epochs, `v2_causal_g4_s0_e20@best`, 4 `~hpc` re-runs) |
| integrity mismatches | 0 | the 6 apparent mismatches were same-name different-weight re-runs, now separate `~hpc` entries |
| load + inference verified | 228 | `scripts/verify_adapters.py --all` (2 identical runs, structure checks): every runnable checkpoint in the workspace, incl. 126 intermediate epochs |
| not applicable | 57 | no adapter (repair, historical, LSTM, exports, bases), smoke, invalid, forecaster ONNX |
| duplicate-content groups | 31 | listed in `results/registry/duplicates.json`, nothing deleted (e.g. `normal_ir@last` = `@epoch_18`; CF detector = deployment copy) |
| backup-only model entries | 32 | 24 intermediate RGB epochs, 6 `~hpc` re-runs, `v2_causal_g4_s0_e20@best` (protected dir), official GlobalTrack weights |
| historical result records | 317 | 179 master-ledger rows verbatim + CST official / released / paper, CF, P1, Track C verdicts, Anti-UAV paper Table V |

## 4. Scientific integrity

- Historical metrics are imported verbatim with their `source_file`. Tests pin the master ledger rows, the CST official scores
  (X1 60.08 / 59.70, X2 60.08 / 60.26 / 60.60, mean 60.31, single frame 46.83 / 49.81 / 49.79, mean 48.81) and the paper value 35.92.
  The FINAL_REPORT paired result X1 vs released GlobalTrack (+24.16, 59 / 1) is recomputed from the recorded per-sequence values.
- TEST is locked in code (data layer, evaluation service, UI). Recorded TEST rows are labelled and read-only.
- Comparisons are numeric only under one protocol + split + comparability keys. Paper-transcribed numbers, forecasting vs tracking
  metrics and truncated (`studio.sot_smoke.v1`) vs full runs are always NON-EQUIVALENT.
- Findings on the overview are qualitative statements whose numbers are read from the cited records at render time.

## 5. Validation

| check | result |
|---|---|
| existing `tests.test_research_program`, `tests.test_cst_protocol` | 34 OK before and after |
| `tests.test_workspace_registry` | 30 OK |
| `tests.test_workspace_adapters` | 12 OK (detector parity vs HPC cache: median top-box IoU 0.991, 0 presence disagreements; SA fidelity; checksum gate; refusals; CPU fallback with CUDA hidden) |
| `tests.test_studio_ui` (AppTest) | 10 OK: every page renders; search; recorded-only path; detector, GlobalTrack and forecaster inference from the UI; default recorded comparison; fresh two-model comparison; failure page |
| `experiments/anti_uav/demo/test_demo.py` | passed |
| full-sequence fidelity | X1 studio SA vs tracker on HPC-cached detections: 74.12 vs 74.14, 73.50 vs 73.49 |
| adapter verification | 228 / 228 runnable checkpoints passed (102 final / best + 126 intermediate epochs), all deterministic, 0 failures |
| real launch | `studio.bat`: 127.0.0.1 only, `/_stcore/health` ok, HTTP 200, clean stop |
| recovery backup | `recovery_manifest.py --out`: 5,725 / 5,725 verified, 0 mismatch; fingerprint (path, size, mtime) of all 5,742 files identical before / after |
| CPU fallback (`CUDA_VISIBLE_DEVICES=-1`) | `device=auto` resolves to CPU, an explicit `cuda` request is refused with a message; detector, GlobalTrack and forecaster runs complete on CPU; `tests.test_studio_ui` 10 / 10 OK without CUDA. Small runs (CST `building_41` 30 frames, Anti-UAV 10 frames): detector SA 90.75 CPU vs 90.79 CUDA (fp32 vs bf16 autocast), GlobalTrack 79.67 on both |
| `git diff --check` | clean |

Not tested: TensorRT / ONNX execution from the studio (not installed in this venv), CPU throughput on full sequences (only short runs were timed: GlobalTrack 10 frames ~15 s on CPU vs ~5 s on CUDA, both including model load), a
multi-user or remote browser session (localhost only by design).

## 5a. HPC-side exception (one temporary file)

The Phase 0 inventory (2026-10-13 12:13 UTC, `ssh amritahpc`) ran `pip freeze` for each HPC environment and redirected the output to
`/tmp/yr_freeze_<env>.txt`. One temporary file was written outside the project: **`/tmp/yr_freeze_at311.txt`**. Nothing under
`~/aerotrack_server/` was modified or deleted. The file was never copied to the OMEN. It is left in place by decision (2026-10-13), and
no further HPC-modifying commands are allowed. The package list of the `at311` environment is preserved locally at
`results/artifacts/hpc_recovery_20261013/aerotrack_server/environments/at311_freeze.txt` (written on the HPC 2026-10-06; SHA-256
`6733a996…5d61`, matches the HPC-side hash in `_verification/remote_sha_tier3_small.txt`). That local copy was not compared byte for
byte with the `/tmp` file.

## 6. Open issues

- 5 backup-only model entries (non-epoch) are not runnable without a deliberate restore decision. The `~hpc` re-runs would collide
  with OMEN files of different content, and `results/runs/v2_causal_g4_s0_e20/` is pre-existing untracked work left untouched.
- `results/runs/v2_causal_g4_s0_e20/` lacks `run_meta.json`; the CF detector runs from the byte-identical deployment copy instead.
- 183 entries have no independent hash record (their hash is recorded from now on).
- Citations for SPD-Conv, ByteTrack, SORT, HOTA and GlobalTrack are recorded by title only (`citation_status: not_recorded_in_repo`).
- Studio evaluations are single-process. A whole-split VAL evaluation of GlobalTrack takes hours (~10 fps) and belongs in
  `experiments/anti_uav/scripts/evaluate.py`.

## 7. Next steps

Research (separate from maintenance): register a TRAIN / VAL-only *learned* absence / presence signal before any computation (P1
showed a score gate only trades false alarms for misses), train it on the HPC, compare candidates in the studio on VAL, and spend no
TEST without a registration. Maintenance: rebuild → verify adapters → render docs → tests after every new artifact.
