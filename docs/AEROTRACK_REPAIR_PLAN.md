# AeroTrack-Net repair — task plan (SECOND task; not started)

Status: **planned, not executed.** Priority order set by the task brief: the Anti-UAV reproduction pipeline
came first (done, see `ANTI_UAV_EXPERIMENT_PROTOCOL.md`). Nothing in the historical code (`c_model/`,
`b_pipeline/`, `run_aerotrack.py`, notebooks, `runs/aerotrack_*`) has been modified. Historical results stay
on disk unchanged; repairs produce **new** run directories and **new** documents.

Inputs: `audit/AUDIT_REPORT.md` (independent audit, 2026-09-19, benchmarks in `runs/audit/`) and the re-scan
in `ANTI_UAV_INTEGRITY_AUDIT.md` §B. Each item below names the exact code location inspected today, what has
to change, and how the fix will be verified. Do the items in order; each is one commit on a branch
`aerotrack-repair`.

| # | Audit finding | Inspected location | Required change | Verification |
|---|---|---|---|---|
| 1 | v1 and v2 evaluated on different image sets (150 vs 3000 images) | `c_model/val.py`, `runs/*/eval_val/evaluation_report.json`; `audit/bench_detector.py` already does same-set eval | Make `val.py` evaluate a **fixed, manifest-hashed** image set (full val); re-evaluate v1 and v2 on it; regenerate `runs/<run>/eval_val_full/`; the README table must be regenerated from those JSONs, not edited | `evaluation_report.json` of both models carries the same `image_set_sha256`; a script produces the README table |
| 2 | no ablations run; `ABLATIONS` dict and `yolo11-stride.yaml` exist unused | `c_model/train.py` (`ABLATIONS`), `c_model/yolo11-stride.yaml` | Run, from one converged baseline recipe at equal budget: `no_spd` (stride baseline), `single_frame` (3-ch), and the **stock YOLO11n** (COCO init, 3-ch, same data/epochs). One factor per run; identical eval set (item 1) | four `runs/ablation_*` dirs with `args.json`, identical epochs/data manifest; a generated ablation table |
| 3 | v1 unconverged (killed at epoch 21/150) and fine-tune changed ~8 variables at once | `c_model/train.py::STAGES`, `runs/aerotrack_spd_v1/results.csv` | Present v2 honestly as "recipe change from an unconverged checkpoint"; for any attribution, train the converged baseline of item 2 and change one thing at a time | each run's `args.json` differs from the baseline in exactly one key |
| 4 | temporal 9-channel stack contributes nothing (static / reversed ≈ same mAP) | `audit/bench_detector.py --modes normal static reversed zero_nb` | Report the inference-time ablation as-is; the `single_frame` training ablation (item 2) decides whether the stack stays; if dropped, remove the +1-frame lookahead latency claim | table from `runs/audit/val_cap*_*.json` |
| 5 | LSTM forecaster collapsed to zero motion; beaten by trivial baselines | `c_model/lstm_forecaster.py`, `audit/bench_forecaster.py`, `runs/audit/forecaster_benchmark.json` | Add zero-velocity / last-velocity / mean-velocity(4) / Kalman baselines **inside** `lstm_forecaster.py`'s report; retrain with a residual-over-last-velocity target and per-sequence normalisation; report per domain (CST separately). If it does not beat mean-vel(4) on CST: state so and retire the pillar | `forecast_report.json` contains all baselines; ADE/FDE per domain |
| 6 | ByteTrack loses to a no-tracker argmax baseline; demos claim ByteTrack without a tracker | `c_model/track_eval.py`, `c_model/inference.py`, `c_model/visual_demo.py`, `c_model/make_*_demo.py`; `BYTETracker(` only in `bytetrack.py`/`track_eval.py` | Either wire `BYTETracker` into `inference.py`/demos or remove "ByteTrack" from every caption, HUD string and the README diagram; add argmax / greedy-single-ID baselines to `track_eval.py`'s report (they exist in `audit/bench_tracker.py`) | grep for "ByteTrack" in demo/inference code returns only real usages; tracking report lists the baselines |
| 7 | hand-typed LSTM trajectory plot | `rev2.ipynb` cell 10 (`x_obs, y_obs, x_gt, x_kf, x_lstm` literals) | Delete the cell; regenerate the figure from `runs/lstm_forecaster/` predictions by script; store the script | the notebook has no literal coordinate lists (scan in `ANTI_UAV_INTEGRITY_AUDIT.md` §B returns 0) |
| 8 | test-set demo cherry-picked (frames 104–134 of a test sequence chosen where v1 fails) | `c_model/make_model_comparison_demo.py:104`, `visual_demo.py:146-194` | Select demo clips from **val** by a stated rule (e.g. median-difficulty sequence, first 100 tracked frames); never render test; remove hard-coded frame ranges and the "LATENCY: 11.8 ms" HUD string unless measured live | demo script takes the rule as an argument and logs the chosen sequence |
| 9 | bird→drone false-alarm metric wrong (`fp_rate` counts every prediction on bird-only images) | `c_model/evaluation.py::fp_rate`; correct version in `audit/bench_bird_confusion.py` | Count **drone-class** predictions on bird-only images and GT birds boxed as drone (IoU ≥ .5); report both models | numbers reproduce `runs/audit/bird_confusion.json` |
| 10 | latency table mislabelled (bf16 claimed; fp32 run; breakdown not measured; lookahead omitted) | `c_model/inference.py`, README latency table, `audit/bench_latency.py` | Replace the table with `bench_latency.py` output incl. p95 and the +1-frame structural latency; label precision correctly | README table generated from `runs/audit/latency_benchmark.json` |
| 11 | three definitions of "micro" across README / rev2 / code; epoch counts and sampler tables inconsistent | `c_model/evaluation.py` (relative-area bins), README, rev2 | One definition (relative area, stated also in px at input for each source), one epoch count per model, one sampler table taken from `args.json` | consistency check script over README/notebook text |
| 12 | checkpoint selected on a noisy fitness metric (micro AP swings 0.33↔0.77) | `runs/aerotrack_spd_finetuned/results.csv`, `c_model/aerotrack_trainer.py` fitness | Report the last-epoch checkpoint alongside "best"; if "best" is kept, select on a smoothed or larger-n metric and say so | both checkpoints evaluated on the item-1 set |
| 13 | reproducible experiment manifests missing | `b_pipeline/data_unified/splits.json` (good), no per-run manifest hash | Add a run-manifest (data hash, git commit, env, command) to `aerotrack_trainer.py` — the pattern is in `experiments/anti_uav/src/anti_uav_repro/utils/repro.py` and can be imported | every new run dir has `run_meta.json` |
| 14 | `runs/aerotrack_spd_finetuned/eval_val/evaluation_report.json` overwritten by a 6-image smoke run (uncommitted modification in the working tree) | `git status` shows it modified | Restore from git (`git checkout -- <file>`) or regenerate with item 1; **ask the user** before touching, as it is their pending change | file content is a full evaluation again |

## Ground rules for the repair (same as the Anti-UAV protocol)

- New runs go to new directories; `runs/aerotrack_spd_v1` and `runs/aerotrack_spd_finetuned` are never modified.
- Every reported number comes from a generated JSON; README tables are regenerated by script.
- Test split (`splits.json` test) is touched once, at the very end, for the final table.
- Claims are downgraded to what the controlled experiments support ("recipe change", "no measurable temporal
  effect", "tracker not beneficial on single-target data") — see `audit/AUDIT_REPORT.md` §5 for what can be
  said positively (split hygiene, evaluation code, bird-rejection result).

## Estimated cost

Items 1, 7, 8, 9, 10, 11, 13 are code/doc work (≈ 1 day). Items 2–3 need four training runs of ~13–18 h each on
this GPU (audit §6 estimate); item 5 needs one LSTM retraining (< 1 h); item 6 is evaluation only.
