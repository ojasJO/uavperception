# Anti-UAV / DFSC — Experiment Protocol

Operating rules for the reproduction in `experiments/anti_uav/`. The *what* is in
`ANTI_UAV_REPRODUCTION_SPEC.md`; this document is the *how*, written so another student can run
and defend the experiment without asking questions.

## 1. Three phases, never mixed

| Phase | Purpose | Configs | Output columns |
|---|---|---|---|
| **A — faithful reproduction** | answer "can we reproduce Table V (IR)?" | `configs/normal_ir.yaml` (PAPER BASELINE "normal"), `configs/dfsc_ir.yaml` (OUR REPRODUCTION) | "Our reproduction" |
| **B — controlled changes** | one factor at a time from a finished Phase A run | `configs/phase_b/*.yaml` | "OUR IMPROVEMENT" / "ADDITIONAL CONTROL" |
| **C — comparison** | paper vs A vs B (vs AeroTrack-Net where a common metric exists — currently none: AeroTrack-Net is a detector) | `scripts/aggregate_results.py` → `docs/ANTI_UAV_COMPARISON.md` | all, side by side |

Phase A must be complete (both runs trained, evaluated on val, then once on test) before any
Phase B config is trained. Phase B results never overwrite or replace Phase A results.

## 2. Data rules

| Split | Allowed use | Enforced by |
|---|---|---|
| TRAIN (160 seqs) | optimisation only | `validate_config` rejects `train_split: test`; `load_sequences(purpose='train')` |
| VAL (67 seqs) | training-time monitor (frame-strided subset, printed as "monitor only"); hyper-parameter / checkpoint decisions **in Phase B only** | `purpose='monitor'` / `'tune'` |
| TEST (91 seqs) | final evaluation of a *finished* model, once per model | `load_sequences` raises `TestSplitAccessError` for any other purpose; `evaluate.py --split test` requires `--final-eval`, refuses strided/limited runs, verifies the frozen manifest, and appends to `reports/test_evaluation_ledger.jsonl` |

- Split identity is frozen in `experiments/anti_uav/manifests/antiuav_{IR,RGB}_{train,val,test}.manifest.json`
  (label-file SHA-256, frame counts, boundary-frame hashes; manifest SHA-256 recorded in every run and
  evaluation). `scripts/verify_dataset.py` re-verifies; it never overwrites a manifest without `--force-rebuild`.
- No data cleaning is applied. The only frame filter is GlobalTrack's training-pair filter (spec §4),
  which drops 0.2 % of IR frames from *pair sampling* only; evaluation uses every frame.
- Frames come from the JPEG decodes already in the repository (spec S4). Nothing is re-extracted.

## 3. Training rules (Phase A)

- Exactly the settings in `configs/_base_ir.yaml`; the two Phase A configs differ **only** in `dfsc.alpha`
  (0 = normal, 1 = DFSC).
- Single seed (0) for Phase A, recorded in `run_meta.json`; Phase B adds seed 1 for both runs.
- **No checkpoint selection**: the final-epoch checkpoint (`weights/epoch_18.pt` == `weights/last.pt`) is
  the model, as in GlobalTrack and the paper. The training-time monitor is a progress signal only.
- Every epoch checkpoint is kept (`epoch_XX.pt`) so the training trajectory can be audited later;
  none of them is evaluated on TEST.
- A run directory is never overwritten: `train.py` refuses a non-empty run dir unless `--resume` or
  `--allow-existing` is given. Failed / aborted runs stay on disk and are listed in the progress log.
- Anything that changes a run's behaviour (config, code) must be committed; `run_meta.json` records
  the git commit and a `dirty` flag listing uncommitted files.

## 4. Evaluation rules

1. `scripts/evaluate.py --checkpoint <run>/weights/last.pt --split val` — full val, every frame,
   every sequence. Produces `predictions/*.json` (raw per-frame boxes + scores), `metrics.json`,
   `report.md`, `eval_meta.json` (checkpoint SHA-256, config, manifest SHA-256, git, env, seed, command,
   timestamps).
2. Only when both Phase A runs are finished and their val numbers are recorded:
   `scripts/evaluate.py --checkpoint ... --split test --final-eval` — **once per model**.
3. `scripts/aggregate_results.py` regenerates `docs/ANTI_UAV_COMPARISON.md`. Nobody types a number
   into that file.
4. Metric = state accuracy Eq. (1) → mSA, over frames from the official start frame onward (primary),
   plus the all-frames variant, precision@20 px and success AUC, and per-attribute mSA — all from the
   saved prediction files (`eval/metrics.py`). Predictions include scores so that any later
   absence-threshold study (Phase B) re-uses the same files without re-running the model.
5. Verdict labels are pre-registered in `aggregate_results.py`: |Δ mSA| ≤ 1 pt reproduced; ≤ 3 pt
   approximately reproduced; else not reproduced; missing paper value → not reproducible from available
   information.

## 5. Demos and figures

- Any figure is generated from `predictions/*.json` or `metrics.jsonl`; no coordinates, trajectories or
  metric values are typed by hand.
- Demo clips, if ever made, come from **VAL** sequences chosen by a stated rule fixed in advance
  (e.g. the median-SA sequence of the DFSC run), never by browsing results; TEST sequences are never
  rendered.

## 6. Reporting language

- "reproduced" / "approximately reproduced" / "not reproduced" / "not reproducible from available
  information" — as defined above, with the number next to the word.
- The gain DFSC − normal is a difference of two single-seed runs until the Phase B seed runs exist;
  until then it is reported as such, without a claim.
- Never describe an `[INFERRED]` or `[UNSPECIFIED]` setting as "the paper's setting".

## 7. Compute budget (measured on the RTX 5050 Laptop, 8.5 GB)

- Training: 800 iterations/epoch at ≈ 0.55–0.6 s/iteration (batch 2, fp32, peak ≈ 4–6.5 GB) → ≈ 8 min/epoch,
  ≈ 2.5 h per 18-epoch run, plus ~1 min per monitor pass (every 3 epochs).
- Tracking: ≈ 10–20 frames/s → full IR val (62k frames) ≈ 1–1.5 h; full IR test (85k frames) ≈ 1.5–2.5 h.
- Phase A total (2 runs + 2 val + 2 test evaluations): roughly 10–12 h of GPU time.

## 8. Checklist before claiming anything

- [ ] both Phase A runs finished 18/18 epochs (metrics.jsonl has 18 records, `run_meta.json` has `timestamp_end`)
- [ ] val evaluations exist for both, full protocol (`partial: false`)
- [ ] test evaluations exist for both, exactly one entry each in the ledger
- [ ] `aggregate_results.py` run; comparison table filled; discrepancy analysis written for every |Δ| > 1 pt
- [ ] git commit hash of the code that produced the numbers is recorded and the tree was clean (or the dirty files are listed and irrelevant)
