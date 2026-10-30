# HPC → OMEN recovery backup, 2026-10-13

**Source**: `amritahpc:~/aerotrack_server/` (user y_rajeshwar). HPC `/home` was 100 % full (57 GB → 51 GB free of 25 TB during the copy).
**Destination**: `A:\AEROTRACK_OMEN\artifacts\hpc_recovery_20261013\aerotrack_server\` (same relative layout as the HPC).
Git-ignored (`.gitignore`: `artifacts/hpc_recovery_*/`) so it can never be committed and `git clean -fd` leaves it alone
(`git clean -x` would NOT: never run it here). Independent of the live `runs/` tree.

Method: `ssh amritahpc 'tar cf - …' | tar xf -` in three batches (no rsync on the OMEN; nothing deleted or modified on the HPC).
Verification: SHA-256 computed **on the HPC** (`_verification/remote_sha_*.txt`), re-computed locally and compared file by file
by `tools/hpc/recovery_manifest.py` (rerun any time:
`venv/Scripts/python.exe tools/hpc/recovery_manifest.py artifacts/hpc_recovery_20261013`).

## Verified (SHA-256 local == HPC)
| set | files | bytes |
|---|---:|---:|
| model / checkpoint / cache files (`.pt .pth .ckpt .onnx .pkl .npz …`) | 1,348 | 16.37 GB |
| everything else in the copied tree (code incl. `source/.git`, job records, logs, reports, configs, manifests, bundles) | 4,377 | 0.65 GB |
| **total in `ARTIFACT_MANIFEST.csv` / `ALL_FILES_SHA256_VERIFIED.txt`** | **5,725** | **17.03 GB** — 0 mismatches, 0 missing |
| tier 3 extras (`_verification/remote_sha_tier3_small.txt`, `sha256sum -c` OK): `environments/at311_freeze.txt`, `build_env.sh`, `build_env_at311.log`, `shared_data/derived_bundles/{corpus_v2_core,derived_v2}.tar.gz`, `shared_data/logs/*`, `shared_data/root/DATA_VERSION`, onnxruntime 1.31 wheel | 9 | 0.07 GB |

Not checksum-verified: none of the copied files.

### Registered checkpoints (hash also recorded in a frozen config / registry → identity proven)
Paths relative to `aerotrack_server/source/runs/`.

| role | file | sha256 (prefix) |
|---|---|---|
| **X1** (STE 60.08 / argmax 59.70, PBS 93019) = **X2 temporal seed 0** (60.08, PBS 93029) | `v3_n__causal_g4_cst_s0_e20/weights/last.pt` | e52a5645 |
| X2 temporal causal_g4 seed 1 (60.26) | `v3_n__causal_g4_cst_s1_e20/weights/last.pt` | b716d1ae |
| X2 temporal causal_g4 seed 2 (60.60) | `v3_n__causal_g4_cst_s2_e20/weights/last.pt` | 5f5c9e40 |
| X2 single-frame control seeds 0 / 1 / 2 (46.83 / 49.81 / 49.79) | `v3_n__ttt_cst_s{0,1,2}_e20/weights/last.pt` | c04cba26 / ee3da997 / 3b062e49 |
| CF detector (`frozen_cf.json`, CF-A2) | `v2_causal_g4_s0_e20/weights/last.pt` | 1118cf32 |
| Track C C1 (gru / mlp / tfm abs, aug, noabs × s0-2) + C1c (domtok; perdom CST / Anti-UAV IR / RGB × s0-2): 27 forecasters, all in `learned_registry{,_c1c}.json` | `aerotrack_temporal/c1/*.pt` (+ `onnx/` exports) | see manifest |

Sources of the hashes: `experiments/aerotrack_v3/reports/x1_final_test.json`, `experiments/cst_anti_uav/frozen_x2_params.json`,
`experiments/cst_anti_uav/reports/x2_final_test.json`, `experiments/aerotrack_temporal/frozen_cf.json`,
`runs/aerotrack_temporal/c1/learned_registry{,_c1c}.json`.

### Other models preserved (all verified)
- `best.pt` + `last.pt` of every other CST/AeroTrack run: v2 campaign (stock / sym_g4 / causal_g1/g2/g4/g8 × seeds 0-1, smoke),
  F1 standard yolo11n (`v3_n_std__causal_g4_s{0,1}`), `gen_temporal_fp_g4_full_e20`, `laptop_v2_stock_s0_e20`, retired `lstm_forecaster`.
- Anti-UAV Track A (`source/experiments/anti_uav/runs/`): `normal_ir`, `dfsc_ir`, `phaseB_*_lr0.02_accum8`, `largescale_globaltrack`
  (last.pt) and `normal_rgb` / `dfsc_rgb` (epoch_01-12 + last.pt); GlobalTrack init `checkpoints/globaltrack/…torchvision.pt`;
  base `c_model/yolo11{n,s}.pt`. Training-state checkpoints are the Ultralytics / GlobalTrack `.pt` files themselves (resume-capable).
- Detection caches (`runs/aerotrack_v2/_cache`, `runs/aerotrack_v3/_cache`, incl. the TEST caches of X2 / CF) and 1,193 `.npz` run outputs.
- Job records `jobs/<run_id>/` (cmd, env, events, pbs.out, job.log), `manifests/`, `reports/`, `logs/`, `backups/`, `source_bundles/`,
  and the full HPC source checkout incl. `.git` (HPC HEAD 6b9415a + untracked HPC-side reports such as
  `experiments/aerotrack_v2/reports/{V2_REPORT.md,final_test.json,v2_test_ledger.jsonl}`, `experiments/aerotrack_v3/ledger.jsonl`).

## Not copied (deliberately) — reproducible
| item | HPC size | why / how to regenerate |
|---|---:|---|
| `environments/at311` (Python 3.11 venv) | 6.7 GB | `environments/build_env.sh` + `at311_freeze.txt` (copied) |
| `shared_data/source_archives/` (2_cst-antiuav 22 G, 1_anti-uav 5.7 G, 3_det-fly 0.3 G, globaltrack 0.45 G) | 28 GB | public downloads (CST: arXiv:2507.23473 / github.com/PCwenyue/CST-Anti-UAV); the OMEN also holds data in `data/` and `b_pipeline/data_unified*` |
| `shared_data/root/{data,b_pipeline}` (unpacked + derived corpora) | 79 GB | rebuilt by `b_pipeline/` + `experiments/cst_completion/scripts/`; derived bundles copied |
| symlinks `source/data`, `source/b_pipeline/data_unified{,_v2}` | — | point into `shared_data/root`; tar could not recreate them on Windows (the only tar errors) |

Outside `~/aerotrack_server` (not this project, not copied): `~/env_backup` (running job 93035 ENV_BACKUP), `~/final_year_project_29`, `~/anti_uav`.

## Missing / blocked
None of the inventoried model or research artifacts is missing. Files created on the HPC after 2026-10-13 ~17:30 IST are not included.
