# registry/ — the one place to discover every model, dataset and result

| file | made by | content |
|---|---|---|
| `catalog/*.json` | hand-curated (version-controlled) | directions, papers (citation status), model families + run roles, evaluation protocols, datasets, key findings. **No metrics.** |
| `models.json` | `tools/build_workspace_registry.py` | one entry per checkpoint identity (`<family>:<run>@<tag>`, SHA-256, locations, status, integrity, verification, config, lineage, linked results) |
| `checkpoints.csv` | same | every physical model / export / cache file, workspace and recovery backup |
| `model_lineage.json`, `duplicates.json` | same | `initialised_from` / `same_bytes` / `exported_from` edges; byte-identical groups (never deleted) |
| `datasets.json` | same | roots, splits, manifests, sequence counts, existence checks |
| `experiments.jsonl` | same | historical results imported verbatim with their `source_file` (master ledger, CST official scores, CF, P1, Track C, paper-reported) |
| `build_summary.json` | same | counts, unclassified files, backup-only models |
| `verification/adapter_checks.json` | `tools/verify_adapters.py` | load + inference evidence per model |
| `verification/evaluation_checks.json` | `tools/verify_evaluations.py` | fresh whole-sequence VAL run vs recorded per-sequence SA (evaluation-verified label) |
| `catalog/featured.json` | hand-curated | Flight Deck featured models (versions, purpose, failure modes with sources, evidence record ids; no metrics) |
| `migrations/restore_manifest.json` | `tools/restore_from_recovery.py` | every file copied out of the recovery backup (source, destination, SHA-256, status) |
| `.hash_cache.json` | build (git-ignored) | SHA-256 cache keyed by path, size and mtime |

Rebuild after any change: `venv\Scripts\python.exe tools\build_workspace_registry.py`. Read it with `aerolab.registry.Registry`.
Schemas: `aerolab/schemas.py` (`aerolab/model/1`, `aerolab/experiment/1`, `aerolab/dataset/1`, `aerolab/studio_manifest/1`).
Studio evaluations live in `runs/studio/index.jsonl` (git-ignored outputs) and are merged at read time.
