# experiments/anti_uav — reproduction of *Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking* (DFSC on GlobalTrack)

This directory is a **self-contained, clean experimental namespace**. It shares only the decoded frames
with the historical AeroTrack-Net code and never writes outside `experiments/anti_uav/` and `docs/`.

## What is reproduced

Protocol II of the paper on the **infrared** modality: GlobalTrack (query-guided Faster R-CNN, ResNet-50-FPN)
fine-tuned on the Anti-UAV train split with (`dfsc_ir`) and without (`normal_ir`) the paper's DFSC training
strategy, evaluated with the paper's state-accuracy metric (mSA) on val and — once, at the end — on test.
Paper reference numbers (IR All mSA): normal 79.60 / 65.36, DFSC 80.09 / 66.04 (val / test).

Everything the paper states, everything inherited from GlobalTrack's code, and every substitution we had to
make are tagged in **`docs/ANTI_UAV_REPRODUCTION_SPEC.md`**. The operating rules are in
**`docs/ANTI_UAV_EXPERIMENT_PROTOCOL.md`**; the comparison table is generated into
**`docs/ANTI_UAV_COMPARISON.md`**.

Note: the paper is arXiv:2101.08466v3 (2021; journal version IEEE TMM). It is a **single-object tracking**
paper; the older AeroTrack-Net code in this repository is a detector and is not an implementation of it.

## Final results (Protocol II, infrared)

The authoritative result document is **`reports/FINAL_PROTOCOL_II_IR_BENCHMARK_REPORT.md`**. It covers the four authorised
evaluations (normal / DFSC × val / test), which are complete and independently verified, and keeps them separate from the
supplementary seed-1 test results and from the incomplete DFSC-all ablation, which has no valid result.

## Where things are

| Path | Content |
|---|---|
| `../../anti_uav_research_paper.pdf` | the paper (repo root) |
| `configs/_base_ir.yaml` | all shared settings (tagged); `normal_ir.yaml` / `dfsc_ir.yaml` differ only in `dfsc.alpha` |
| `configs/smoke_ir.yaml` | plumbing test only |
| `configs/phase_b/` | one-factor-at-a-time follow-ups (run only after Phase A) |
| `configs/deferred/` | visible-modality configs; **not faithful** until the GlobalTrack checkpoint is converted |
| `src/anti_uav_repro/` | the implementation (data / models / engine / eval / utils) |
| `scripts/` | `verify_dataset.py`, `smoke_test.py`, `train.py`, `evaluate.py`, `aggregate_results.py` |
| `manifests/` | frozen split identities (SHA-256), created by `verify_dataset.py` |
| `reports/paper_reported_results.json` | the paper's numbers, transcribed with table references |
| `reports/dataset_verification.{json,md}` | dataset audit output |
| `reports/test_evaluation_ledger.jsonl` | append-only record of every TEST evaluation |
| `runs/<run_name>/` | one directory per training run (see below) |

## Dataset configuration

`data.root: data/anti_uav` holds the official labels (`<split>/<seq>/infrared.json`, `visible.json`: `exist`,
`gt_rect` = `[x, y, w, h]`) and attributes (`label_new/<split>.json`). `data.frames_root:
b_pipeline/data_unified/anti_uav/images/antiuav_<split>_<seq>__{IR,RGB}/NNNNNN.jpg` holds the decoded frames
(1-indexed, made by the repository's earlier pipeline). Splits are the official ones: 160 / 67 / 91 sequences.

Verify (and freeze the manifests the first time):

```bat
venv\Scripts\python.exe experiments\anti_uav\scripts\verify_dataset.py
```

## Smoke tests (no full training)

```bat
venv\Scripts\python.exe experiments\anti_uav\scripts\smoke_test.py
```

14 checks: compile/imports, manifests, sample loading, preprocessing, forward, loss, backward, optimizer step,
checkpoint round-trip, tiny end-to-end training, tiny val evaluation, metrics recomputed from predictions,
TEST-split guard, seed → metadata, config saved with run. Artefacts go to `runs/_smoke/` (disposable).

## Training (Phase A — run both; order does not matter)

```bat
venv\Scripts\python.exe experiments\anti_uav\scripts\train.py --config experiments\anti_uav\configs\dfsc_ir.yaml
venv\Scripts\python.exe experiments\anti_uav\scripts\train.py --config experiments\anti_uav\configs\normal_ir.yaml
```

Validate a command without training: append `--dry-run`. Resume an interrupted run:
`--resume experiments\anti_uav\runs\dfsc_ir\weights\last.pt`.

Live terminal output, every 20 iterations: epoch, iteration, mean loss and its six components (RPN same /
cross, RCNN cls / reg), LR, gradient norm, it/s, peak GPU memory, ETA for the epoch and the whole run. Per epoch:
checkpoint name + SHA-256; every 3 epochs a `[monitor]` line (8 fixed val sequences, every 10th frame — progress
signal only, never used for selection). Expected: 800 it/epoch, ≈ 8 min/epoch, 18 epochs ≈ 2.5 h.

Run directory `runs/<run_name>/`: `config.effective.yaml`, `run_meta.json` (git, env, GPU, seed, command,
manifest hash, model summary, timestamps), `train_manifest.json`, `metrics.jsonl` (one record per epoch),
`train_log.txt` (mirror of the terminal), `weights/epoch_XX.pt` + `last.pt`.

## Evaluation (independent of training)

```bat
:: validation — allowed any time
venv\Scripts\python.exe experiments\anti_uav\scripts\evaluate.py --checkpoint experiments\anti_uav\runs\dfsc_ir\weights\last.pt --split val
venv\Scripts\python.exe experiments\anti_uav\scripts\evaluate.py --checkpoint experiments\anti_uav\runs\normal_ir\weights\last.pt --split val

:: FINAL test — once per finished model, after both val evaluations exist
venv\Scripts\python.exe experiments\anti_uav\scripts\evaluate.py --checkpoint experiments\anti_uav\runs\dfsc_ir\weights\last.pt --split test --final-eval
venv\Scripts\python.exe experiments\anti_uav\scripts\evaluate.py --checkpoint experiments\anti_uav\runs\normal_ir\weights\last.pt --split test --final-eval

:: regenerate the comparison document
venv\Scripts\python.exe experiments\anti_uav\scripts\aggregate_results.py
```

Output `runs/<run>/eval_<split>_IR_last/`: `predictions/<seq>.json` (per-frame boxes + scores, raw), `metrics.json`,
`report.md`, `eval_meta.json`. The tracker reads ground truth only for the init box of the start frame.

## How test isolation works

- `load_sequences(..., purpose=...)` serves the test split **only** for `purpose='final_eval'`; training,
  monitoring and tuning code paths use other purposes and get a `TestSplitAccessError`.
- `validate_config` rejects any config whose `train_split` or `monitor_split` is `test`.
- `evaluate.py --split test` requires `--final-eval`, refuses strided/limited (partial) runs, checks the split
  against the frozen manifest, prints a banner and appends to `reports/test_evaluation_ledger.jsonl`.
- Smoke check 13 exercises all of the above.

## Reproducing the exact experiment

1. Same data (manifest SHA-256s in `manifests/`, verified by `verify_dataset.py`).
2. Same code: the git commit in `run_meta.json` (with `dirty: false`).
3. Same config: `runs/<run>/config.effective.yaml` (or the config file under `configs/`).
4. Same seed (0) — pair sampling and augmentation are a pure function of (seed, epoch, index);
   remaining non-determinism: cuDNN benchmark kernels (as in GlobalTrack).
5. Same environment: `run_meta.json` → `env` (Python 3.11.9, torch 2.13.0+cu130, torchvision 0.28.0, RTX 5050).

## Not done here (documented, not hidden)

- Visible-modality runs (need the GlobalTrack pretrained checkpoint + conversion) — `configs/deferred/README.md`.
- The paper's "large-scale" row (public GlobalTrack checkpoint) — same dependency.
- Protocol I (40+ off-the-shelf trackers) and Protocol III — out of scope.
