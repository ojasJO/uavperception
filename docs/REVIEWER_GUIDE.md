# DRONE-Track — micro-UAV detection and tracking in thermal infrared and visible video

**Reviewer's guide.** This is the one page to read first. It covers how to run the code, where
every dataset directory is, what the system is, and what it achieves. Nothing here needs to be
installed or configured beyond the one-time environment setup in §2.

| you want to… | do this |
|---|---|
| **see everything working, without typing anything** | double-click **`DEMO.bat`** and press **A** |
| know where the data is | `venv\Scripts\python.exe scripts\show_datasets.py --tree` |
| know what was trained | `venv\Scripts\python.exe scripts\show_models.py` |
| read every result | **`docs\RESULTS_ALL.md`** |
| understand the system | **`docs\ARCHITECTURE.md`** — the architecture in text and diagrams, written to be lifted into slides |
| explore it yourself | `studio.bat` → <http://localhost:8501> |
| re-run the training | §6 |
| find it on the HPC | `~/DRONE-Track` on `y_rajeshwar@172.17.18.100` — see §7 |

---

## 1. What this is, in one paragraph

A drone is a few pixels across when it matters. This project detects and tracks such targets in
thermal infrared and visible video, and it does so **causally** — using only frames that have
already arrived, so it can run on a live feed. The detector is a YOLO11n-shaped network whose
input is not one frame but a **causal temporal window** of three frames, `[t−8, t−4, t]`, stacked
into nine channels; a single-target state estimator turns its per-frame boxes into one trajectory
with an explicit "the target is absent" state. On the CST Anti-UAV benchmark, whose own paper
reports that the best of 21 published trackers reaches **35.92** mean state accuracy, this system
reaches **60.31** (three seeds, 60.08 / 60.26 / 60.60), a paired margin of **+24.39 mSA
[+21.08, +27.80]** winning **60 of 60** sequences — scored by the benchmark authors' own
unmodified toolkit. It runs at **170 FPS** end-to-end on a laptop RTX 5050 under TensorRT fp16,
with **zero frames of look-ahead**.

A second, independent line of work in the same repository reproduces *Anti-UAV* (Jiang et al.,
arXiv:2101.08466): validation reproduces to within 0.3 mSA, while our test score lands about
5 mSA **above** the paper's, and the paper's claimed DFSC gain **does not appear**. Both outcomes
are reported as found.

---

## 2. Setup (once)

**Needed**: Windows 10/11, Python 3.11, about 90 GB of free disk for the datasets, and — for
training only — an NVIDIA GPU. Inference, evaluation and every part of the demo also run on CPU.

```bat
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

On the Blackwell laptop GPU (RTX 50-series, sm_120) install the CUDA 13 build of PyTorch instead
of the default wheel — `requirements_omen.txt` pins the versions that were used:

```bat
venv\Scripts\python.exe -m pip install -r requirements_omen.txt
```

Check the result:

```bat
venv\Scripts\python.exe scripts\show_env.py
```

It prints the Python, PyTorch, CUDA, GPU and free disk the project will actually use. The
reference machine reports python 3.11.9, torch 2.13.0+cu130, CUDA True, RTX 5050 Laptop GPU.

---

## 3. One click: `DEMO.bat`

Double-click `DEMO.bat` and press **A**. It walks through six sections in order, pausing between
them, and is meant to be screen-recorded:

| # | section | what you see |
|---|---|---|
| 1 | environment and GPU check | the machine, PyTorch, CUDA, the GPU, free disk |
| 2 | **dataset directory information** | every dataset root, split and frame-path pattern, each verified against the disk, with a sample sequence listed |
| 3 | trained models and checkpoints | every headline checkpoint with its size and SHA-256, and the 314 model identities by family |
| 4 | all results | `docs\RESULTS_ALL.md`, paged in the console |
| 5 | **live detection** | a video window: eight real sequences — daylight, thermal IR, a sub-16-pixel target, urban thermal clutter, a 65-frame building occlusion, mountain terrain, evasive manoeuvres, night — with boxes, the trajectory trail and a 4× magnifier, numbers streaming to the console |
| 6 | the Flight Deck | the full research laboratory in the browser |

Single sections are available by number, and **7** runs the 122-test suite.

Nothing in the demo trains a model, writes to a dataset, or touches a TEST split: every section
is read-only.

In the video window: `SPACE`/`ENTER` next sequence · `G` ground-truth box · `M` magnifier ·
`S` screenshot · `Q`/`ESC` quit.

---

## 4. Where every dataset directory is

Run `venv\Scripts\python.exe scripts\show_datasets.py --tree` for the live, disk-checked version of
this section. It reads `results/registry/datasets.json` and marks every path `[ok]` or `[MISSING]`.

### 4.1 The three sources

```
A:\AEROTRACK_OMEN\                           <- project root
├─ data\                                     <- RAW datasets, as released
│  ├─ anti_uav\                              Anti-UAV 2021 (Jiang et al.)
│  │  ├─ train\  val\  test\                 160 / 67 / 91 sequences
│  │  │   └─ <sequence>\                     IR.mp4, RGB.mp4, infrared.json, visible.json
│  │  └─ label_new\<split>.json              per-sequence attributes
│  ├─ cst_anti_uav_full\CST-AntiUAV\CST-AntiUAV\      CST Anti-UAV, complete release
│  │  ├─ train\  val\  test\                 120 / 40 / 60 sequences
│  │  │   └─ <sequence>\                     000001.jpg … , IR_label.json
│  ├─ cst_anti_uav\                          truncated historical copy (corpus v1 only)
│  ├─ cst_anti_uav_official\                 official toolkit + the authors' released predictions
│  └─ det_fly\                               Det-Fly, 7,830 daylight images, Bird / Drone
└─ src\pipeline\
   ├─ data_unified\                          UNIFIED corpus v1 (built from data\)
   │  ├─ anti_uav\images\<seq>__{IR|RGB}\     636 sequence directories, 000001.jpg …
   │  ├─ cst\       det_fly\
   │  ├─ splits.json                         sequence-level splits, seed 1337
   │  ├─ manifest.csv  label_cache.npz  hardness.npz
   │  └─ standardization_summary.json
   └─ data_unified_v2\                       UNIFIED corpus v2 (complete CST) + DATA_VERSION
```

### 4.2 Label formats

| dataset | label file | content |
|---|---|---|
| Anti-UAV | `<seq>/infrared.json`, `<seq>/visible.json` | `exist[]` (0/1 per frame), `gt_rect[]` = `[x, y, w, h]` in pixels |
| CST Anti-UAV | `<seq>/IR_label.json` | `exist[]`, `gt_rect[]` = `[x, y, w, h]`, plus SIZE and attribute codes |
| Det-Fly | YOLO `.txt` per image | `class cx cy w h`, normalised |
| unified corpus | `label_cache.npz` | one array of `[class, cx, cy, w, h]` per frame path, plus `hardness.npz` |

### 4.3 Frame sizes and classes

Anti-UAV IR 640×512, Anti-UAV RGB 1920×1080, CST IR 640×512, Det-Fly RGB varies.
Two classes throughout: `0 = uav_drone`, `1 = bird_hard_negative`.

### 4.4 The rules the splits obey

- Splits are **sequence-level**, seed 1337, with leakage assertions in
  `src/pipeline/3_prepare_training.py`: no sequence name appears in two splits.
- CST ships no VAL split, so a fixed-seed holdout is carved out of its train split.
- **CST TEST is never used for selection.** Every TEST pass in this project is pre-registered,
  logged in a ledger, and spent exactly once. All of them are now spent; new work is TRAIN/VAL
  only.
- `DATA_VERSION` = `corpus_v2_f842d03909cf_cst_full_antiuav_hpcdecode`.

### 4.5 Integrity, checked on two machines

The CST extraction yields 246,131 files with **0** missing / size / CRC mismatches, identical on
the HPC and the laptop. The Anti-UAV frames were independently re-decoded on the HPC and are
**byte-identical** to the laptop's across all 636 sequences. The data gate passes on 602,613
TRAIN/VAL paths with no train/val/test leakage.

### 4.6 Building the corpus from the raw releases

```bat
venv\Scripts\python.exe src\pipeline\1_standardize_annotations.py
venv\Scripts\python.exe src\pipeline\2_verify_pipeline.py
venv\Scripts\python.exe src\pipeline\3_prepare_training.py
```

Step 1 converts all three releases into one YOLO corpus; step 2 verifies it (counts, hashes,
label round-trips); step 3 writes the splits and asserts no leakage. Full documentation:
`docs/DATASETS_data_unified_v2.md`.

---

## 5. The system

Detail and diagrams: `docs/ARCHITECTURE.md`.

```
video / sequence
      |
      v
[ causal temporal window ]   frames t-4, t, t-8 -> letterbox 640x640 (pad 114)
      |                      -> channels 0-2, 3-5, 6-8;  BGR, uint8/255
      |                      0 frames of look-ahead
      v
[ detector ]                 YOLO11n stride layout (src/model/yolo11-stride.yaml,
      |                      Ultralytics 8.4.115), 2.613 M parameters, 9-channel
      |                      input, PAN-FPN neck and Detect head unchanged;
      |                      2 classes: uav_drone, bird_hard_negative
      v
[ NMS ]                      evaluation conf 0.001, IoU 0.7, max 30 det/image
      |                      streaming conf 0.05, IoU 0.7
      v
[ single-target state estimator (STE) ]   a Kalman single-target state estimator:
      |                                   one trajectory plus an explicit ABSENT
      |                                   state; parameters frozen on VAL
      |                                   (2026-10-06) before any TEST pass
      v
box per frame in source pixels, or "absent"
```

Initialised from COCO `yolo11n.pt`: every tensor whose name and shape match. *Erratum, stated
plainly*: these YAMLs carry flat multipliers and no scale letter, so Ultralytics resolves an empty
scale and builds every C3k2 block with the m/l/x inner C3k instead of YOLO11n's Bottleneck. Only
**417 of 649 tensors (64.3 %)** of `yolo11n.pt` load; the C3k2 internals of layers 2, 4, 13, 16, 19
and the class head start from random initialisation. Every registered comparison is between
configurations built the same way, so the window verdicts are internally valid, but "starts as COCO
YOLO11n" should read "starts as the same partially COCO-initialised network".
`experiments/aerotrack_v2/ERRATUM_architecture.md` is the full statement; the fix was then tested
as experiment **F1** (`yolo11-stride-std.yaml`, true YOLO11n, 89.8 % of the weights loading) and
came out **NOT SUPPORTED** (seed-mean −0.0100 CST AP50, CI [−0.0259, +0.0064]), so the headline
systems stay on the as-built network.

**Retired by pre-registered rules, and kept in the repository for the record**: ByteTrack
association (a no-association argmax box beat it on MOTA and IDF1 on every dataset), the LSTM
forecaster (mean-velocity / Kalman beat it), **SPD-Conv** — space-to-depth downsampling, which
earlier revisions of this work credited for micro-target performance and which a controlled
ablation found to have no measurable effect on micro AP50, so the deployed detector uses ordinary
stride-2 convolutions — the learned CST forecaster, the P1 presence gate, and trajectory
augmentation (which *hurt*). `docs/RESULTS_ALL.md` §7 lists each with the rule it failed.

---

## 6. Running the code

Everything below runs from the project root. `run.bat` forwards to `scripts/run_aerotrack.py` using the
project's virtual environment, never a system Python.

### 6.1 Inference on your own video

```bat
run.bat infer --source path\to\video.mp4
```

### 6.2 The interactive laboratory

```bat
studio.bat
```

Opens <http://localhost:8501>: seven experiment categories, live inference, several models
compared frame-synchronised, the recorded TEST replay (read-only), a controlled speed benchmark,
forecasting, the paper reproduction, and the model and dataset registries.
Guide: `docs/workspace/FLIGHT_DECK.md`.

### 6.3 Evaluation (VAL / TRAIN; TEST is locked)

```bat
run.bat evaluate --split val
```

### 6.4 Training

Training needs a CUDA GPU. One 20-epoch run is about 46 minutes on a V100, roughly 1 hour on the
RTX 5050. Configuration is `<model>__<window>`, for example `n__causal_g4`.

```bat
venv\Scripts\python.exe experiments\aerotrack_v3\scripts\train_v3.py ^
    --config n__causal_g4 --seed 0 --batch 32 --train-domains cst
```

- `--train-domains` — `all`, or a comma list of `anti_uav_IR`, `anti_uav_RGB`, `cst`, `det_fly`
- `--config` — window `ttt` (single frame, the control), `causal_g1/g2/g4/g8`, `sym_g4`
- `--resume` — resume an interrupted run from its last checkpoint
- `--epochs`, `--batch`, `--workers`

On the HPC the same run goes through PBS:

```bash
scripts/hpc/submit.sh <job_name>        # the command is read from jobs/<job_name>/cmd.txt
```

Every job directory keeps its command, its environment, an event log and the full stdout — 46 of
them are preserved in `~/DRONE-Track/logs/pbs_jobs/`. `scripts/hpc/RUNBOOK.md` has the procedure.

### 6.5 Tests

```bat
venv\Scripts\python.exe -m unittest discover -s tests -v
```

122 tests covering the data pipeline, the model, the evaluation metrics, the registry and the
Flight Deck, including a CPU-only fallback path.

### 6.6 Regenerating the reports

No number in the generated documents is typed by hand:

```bat
venv\Scripts\python.exe scripts\render_results.py              :: docs\RESULTS.md
venv\Scripts\python.exe scripts\build_workspace_registry.py    :: results\registry\
```

---

## 7. The HPC deliverable: `~/DRONE-Track`

Training ran on the Amrita HPC (`y_rajeshwar@172.17.18.100`, PBS, V100 nodes). Everything a
reviewer needs is collected in one folder there:

```
~/DRONE-Track/
├─ README.md  RESULTS_ALL.md  ARCHITECTURE.md  INVENTORY.md
├─ code/            272 files — b_pipeline (data preparation), c_model (model, training,
│                   evaluation, inference), experiments (per-study scripts, configs and
│                   pre-registrations), tools (PBS submission, ledgers, report renderers)
├─ checkpoints/      77 weight files, 15 GB — detector runs, the Anti-UAV reproduction runs,
│                   the converted public GlobalTrack baseline, and the COCO initialisation
├─ logs/          2,577 files — 46 PBS job records (command, environment, events, full stdout)
│                   and the per-run training records (results.csv, args.yaml, event logs)
├─ results/         824 files — every report JSON and Markdown, the benchmark tree, the
│                   manifests and the project documentation
├─ data          -> the dataset store (Anti-UAV 6.3 GB, CST 23 GB, Det-Fly 360 MB)
├─ data_archives -> the original release archives, as downloaded
├─ data_derived  -> the derived label caches and unified corpus
└─ DATA_VERSION  -> the corpus version string
```

The files are **hardlinked** from the original locations, not copied: they cost no extra disk on a
filesystem that is 99 % full, and each one keeps a second, independent name, so it survives a
later cleanup of the original folder. Only the large dataset trees are symlinks.

To rebuild the folder: `bash scripts/hpc/build_drone_track.sh` on the HPC.

`INVENTORY.md` inside the folder lists the file count and size of each part and every checkpoint
path, with its build timestamp.

---

## 8. Headline numbers

| | result | source |
|---|---|---|
| CST Anti-UAV TEST, mean state accuracy | **60.31** (3 seeds: 60.08 / 60.26 / 60.60) vs the benchmark's published best **35.92** | `docs/RESULTS_ALL.md` §1 |
| paired margin over GlobalTrack | **+24.39 mSA** [+21.08, +27.80], **60 / 60** sequences won | §1.1 |
| the temporal window's own contribution | **+11.51 mSA** [+8.36, +14.86] over a single frame | §1.1 |
| cost of removing the look-ahead | **non-inferior** (+0.007 AP50 [−0.000, +0.016]) | §3.2 |
| speed, laptop RTX 5050, TensorRT fp16 | **170 FPS** end-to-end, p99 8.0 ms, **0 frames** of look-ahead | §5 |
| Anti-UAV 2021 reproduction, VAL | 79.33 vs the paper's 79.60 (**−0.27**) — reproduced | §2 |
| Anti-UAV 2021 reproduction, TEST | 70.40 vs the paper's 65.36 (**+5.04**) — above the paper, reported as a discrepancy | §2 |
| the paper's DFSC gain | **not observed** (val −0.02, test −0.64) | §2 |

Full tables, confidence intervals, the registered verdicts, the known weaknesses and the retired
components: **`docs/RESULTS_ALL.md`**.

---

## 9. How to trust these numbers

- **Pre-registration.** Every comparison that decides something was written down — hypothesis,
  metric, threshold, analysis — before the run. The files are in each experiment's
  `preregistration/` directory.
- **The TEST split is a budget.** Each TEST pass is registered, logged in a ledger with the
  weights' SHA-256, and spent once. All budgets are now spent.
- **Trivial baselines.** Every component had to beat the dumbest thing that could work — a
  no-association box, predicting no motion, a single frame. Six components lost and were retired.
- **Generated reports.** `docs/RESULTS.md` and the README of the repair series are rendered from
  evaluation JSON by `scripts/render_results.py`; no number is hand-entered.
- **Checkpoint identity.** 498 checkpoint files carry a SHA-256 in `results/registry/checkpoints.csv`,
  with 0 integrity mismatches; 228 were load- and inference-verified.
- **Two machines agree.** The same VAL run scored on the HPC V100 and on the laptop differs by
  0.00001 mSA; evaluation-host checks pass at |Δ| ≤ 0.005 AP50.
- **An independent audit** rejected the previous revision of this work; `docs/audit/AUDIT_REPORT.md`
  is that audit and `docs/AEROTRACK_REPAIR_REPORT.md` is the item-by-item repair. Three
  headline claims of the earlier revision were overturned by controlled re-measurement and are
  now reported as retired.

---

## 10. Document map

| document | what it is |
|---|---|
| `docs/REVIEWER_GUIDE.md` | this page |
| `docs/RESULTS_ALL.md` | every existing result, one document |
| `docs/ARCHITECTURE.md` | the architecture, in text and diagrams |
| `README.md` | the generated record of the September repair series |
| `docs/MASTER_STATE.md` | the current state of the research, on one screen |
| `docs/RESULTS.md` | auto-generated results |
| `docs/DECISIONS.md` | every decision and its reason |
| `docs/FAILURES_AND_RECOVERY.md` | what broke and how it was recovered |
| `docs/EXPERIMENT_LEDGER.csv` | every run ever launched |
| `docs/workspace/ARCHITECTURE.md` | the workspace layout and its rules |
| `docs/workspace/OPERATING_GUIDE.md` | operating the workspace |
| `docs/workspace/FLIGHT_DECK.md` | the Flight Deck application |
| `docs/audit/AUDIT_REPORT.md` | the independent audit |
| `experiments/*/reports/` | the per-study reports every number comes from |

---

## 11. Credits and licence

Datasets: Anti-UAV (Jiang et al., arXiv:2101.08466), CST Anti-UAV (arXiv:2507.23473), Det-Fly.
Detector architecture after YOLO11 (Ultralytics 8.4); space-to-depth downsampling after SPD-Conv.
Baseline tracker: GlobalTrack, from the authors' public checkpoint. Each dataset remains under
its own licence; this repository's code is under `LICENSE`.
