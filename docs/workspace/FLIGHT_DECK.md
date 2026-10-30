# AeroTrack Flight Deck — user guide

The Flight Deck is the local research laboratory on top of the unified workspace (`results/registry/`, `src/aerolab/`). It covers both research
directions:
- Direction A, the AeroTrack multi-source system.
- Direction B, the Anti-UAV (2021) DFSC / GlobalTrack paper reproduction.

Design notes: `FLIGHT_DECK_IMPLEMENTATION_PLAN.md`. Layout: `ARCHITECTURE.md`. Verified evidence: `FLIGHT_DECK_VERIFICATION.md`.

## Launch

```bat
studio.bat                    :: http://localhost:8501 (localhost only); studio.bat 8600 for another port
```

The same thing without the launcher: `venv\Scripts\python.exe -m streamlit run app\app.py --server.address localhost`.
Stop it with Ctrl+C in that window.

## First experiment, then a comparison

1. **Flight Deck** (home) → *A · Tracking performance* → **Open in Test Bench**.
2. **Data.** CST Anti-UAV / VAL / IR, preset *Standard quick set*. Keep one or two sequences (e.g. `building_41`).
   - The preview shows the start frame with the GT box used for initialisation.
3. **Models.** The defaults are X1 / X2 temporal seed 0 and the single-frame control, both with STE.
   - Add or remove models (2–4; up to 6 with *Extended comparison*).
4. **Execution.**
   - *Frames after the init frame*: e.g. 150. *Whole sequences* gives full Eq. 1 mSA.
   - Device: auto / cuda / cpu.
   - Execution: *sequential* is the default; *lockstep* processes each frame by every model in turn.
5. Check the **Summary**, then click **▶ Start experiment**.
   - The run executes in the background, and **Live Comparison** shows each model's newest frame.
   - Live counters: frames done / total, per-frame latency, rolling FPS, elapsed time, device.
   - **⏹ Cancel run** stops after the current frame.
6. When the run finishes, click **Open the synchronised replay**.
   - **Side by side / Overlay / Difference** views of the same frame index for every model.
   - Play / pause, step, speed, jump to a frame, GT and per-model toggles, zoom on the target.
   - *Next disagreement* jumps to frames where one model is right and another is not.
   - Below the player: metrics per model, paired differences from the reference model, failure intervals (each opens at its frame).
7. To reopen later: **Live Comparison → Saved run**, or **Results & Failures**. Replays read the saved predictions; no model is re-run.

## The seven categories

| | category | models | data / presets | what you get |
|---|---|---|---|---|
| A | Tracking performance | AeroTrack detectors + STE / argmax, Direction B GlobalTrack | CST, Anti-UAV VAL / TRAIN; all presets | live + replay; SA, P@20, success AUC, IoU, centre error, located / false-alarm rates, loss events, failure intervals (drift / sudden / wrong target / no output), recovery |
| B | Target presence & detection | AeroTrack detectors only (GlobalTrack never reports absent → excluded with the reason) | absent / reappearance presets | false-alarm vs miss curves over the detector score threshold, PR curve, confusion matrix, score histograms, tracker fixed point; the recorded P1 trade-off and verdict NO_GAIN |
| C | Temporal & ablation | `cst_x` (X1/X2 temporal, single-frame control, seeds) and v2 windows | CST VAL | live comparison; recorded X2 per-seed TEST table, T1–T3, X2 − X1 recomputed, per-sequence TEST differences (read-only), VAL cross-fitted; window overhead |
| D | Robustness & scenarios | trackers | occlusion / reappearance, absent, small, large, fast motion, clutter, thermal crossover, long, confusing objects, difficult cases, custom | per-condition frame-level SA heatmap, failure intervals, re-acquisition; recorded CST TEST attribute table |
| E | Speed & deployment | trackers | one sequence | controlled benchmark (warm-up discarded, CUDA-synchronised stages, percentiles only with enough samples, init / first-frame, peak GPU, RSS, CPU cores) vs live demonstration (observed FPS); TensorRT gate records kept separate |
| F | Forecasting & trajectories | Track C forecasters (C1 / C1c, 27 registered) | Track C VAL trajectories by domain | ADE / FDE / NLL vs classical cp / cv / dcv on identical anchors; anchor browser with ±2σ; Track C verdicts |
| G | Paper reproduction | Direction B GlobalTrack (normal, DFSC, seed-1 variants) | Anti-UAV VAL | G1 paper vs reproduction (Table V, verdict rule, DFSC effect, config and substitutions S1–S6, fresh-vs-recorded VAL check); G2 CST Table 3 reproduced from released predictions |

## Presets: what each one is based on

Every preset shows its **basis**. Nothing is invented to fill a preset: a preset with no qualifying sequence says why
(`src/aerolab/sequences.py`).

| basis | presets |
|---|---|
| official annotation | target absent (`exist`), occlusion & reappearance (exist transitions + CST OC / OV, Anti-UAV OC / OV), clutter (CST DBC / CDB), thermal crossover (CST TC, Anti-UAV TC) |
| derived from ground truth (rule shown) | small targets (≥ 25 % of visible frames with diagonal ≤ 10 px), large targets (≥ 20 frames > 50 px), fast motion (fastest quarter by 95th-percentile GT-centre speed, plus official FM) |
| recorded results | confusing objects (≥ 20 recorded wrong-object frames), difficult cases (8 lowest recorded VAL SA of a named model) |
| fixed rule | standard (the whole official split), standard quick (one per CST scene category / 8 evenly spaced Anti-UAV), long sequences (≥ max(Q3, 1.2 × median)) |

Unavailable by design:
- Anti-UAV clutter: no such attribute is shipped.
- Anti-UAV long sequences: they are capped at 1000 frames.

Frame rate: Anti-UAV 25 FPS is paper-explicit. CST reports none, so CST timestamps are shown as unavailable.

## Modes: live, saved, recorded

Every view says which mode it is in:
- **LIVE INFERENCE.** The loaded checkpoint produces each box now.
- **SAVED RUN.** A Flight Deck run's predictions, replayed from disk.
- **RECORDED PREDICTIONS.** Earlier registered evaluations or authors' releases, replayed read-only.

Recorded sources (41): X1 and X2 seeds / single-frame controls on CST TEST, 21 released CST-trained trackers on CST TEST, X1 on CST
VAL, and Direction B normal / DFSC / seed-1 on Anti-UAV VAL / TEST. The replay table recomputes each sequence's SA from the recorded
boxes and shows |Δ| against the registered record (0 on every check).

## Rules enforced in code

- **TEST locked.** The runners and the data layer refuse TEST. TEST frames are opened only by the recorded replay
  (`REPLAY_PURPOSE`), which loads no model and offers no threshold or parameter control. Consumed passes (X1, X2, CF, Anti-UAV
  Protocol II, …) are never re-run.
- **Same frames for every model**: the start frame (GT init) .. start + N, or the whole sequence.
  - A truncated run uses `studio.sot_smoke.v1`, which is never comparable with full mSA.
  - A presence run uses `presence.frame.v1`; a benchmark uses `speed.bench.local.v1`.
- **Comparisons.**
  - Differences are in percentage points, paired by sequence.
  - A bootstrap CI is shown only with at least 5 sequences, and no significance claim is made from short runs.
  - A cross-direction pair is labelled as such (different training data and provenance).
- **Outputs.** Each experiment writes `results/runs/studio/evals/<fd|bench|fc>-<timestamp>-<id>/`: `config.json`, `manifest.json`,
  `per_sequence.json`, `timing.json`, `log.txt`, and `predictions/<model>/<sequence>.json`. Each model also gets one line in
  `results/runs/studio/index.jsonl`.
  - Historical files are never rewritten.
  - A cancelled sequence is saved with `complete: false` and no metric.
- **Resources.**
  - One worker thread runs one experiment at a time.
  - Sequential execution keeps one model in memory.
  - An explicit CUDA request without CUDA is refused with a message.
  - A page rerender never starts a run.

## Evidence labels (Model Vault)

From weakest to strongest claim:
- **registered**: the checkpoint is the identity of a reported, pre-registered result.
- **integrity-verified**: its SHA-256 equals an independent record.
- **load-verified** and **inference-verified**: `scripts/verify_adapters.py` loaded it and ran a small VAL inference on this machine.
- **evaluation-verified**: `scripts/verify_evaluations.py` ran a fresh whole-sequence evaluation that reproduces a recorded per-sequence VAL SA within tolerance.
- **historical results**: the checkpoint has imported result records.
- **recorded results only**: no inference adapter exists.
- **not runnable locally**: the Vault shows the exact blocker.

## Commands

```bat
venv\Scripts\python.exe scripts\verify_evaluations.py            :: fresh vs recorded VAL per checkpoint -> results/registry/verification/evaluation_checks.json
venv\Scripts\python.exe -m unittest tests.test_flight_deck -v
venv\Scripts\python.exe -m unittest tests.test_flight_deck_ui -v
```
From Python: `aerolab.flightdeck.run(build_config(...))`, `aerolab.benchmark.run_benchmark(...)` and
`aerolab.forecasting.run_forecast(...)`. Each module's docstring has a worked example.
