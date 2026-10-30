# DECISIONS — engineering and scientific decisions (append-only)

*Earlier decisions (2026-09 → 2026-10-05) are recorded in `docs/progress/RESEARCH_PROGRAM_PROGRESS.md` §Decisions,
`docs/progress/AEROTRACK_REPAIR_PROGRESS.md`, `docs/progress/ANTI_UAV_PROGRESS.md` and the series reports; they remain valid unless
superseded below. Format: id · date · decision · evidence · alternatives · why.*

---

### D-2026-10-06-01 · No training on the OMEN; the HPC is the sole training host
- **Decision**: from 2026-10-06 every training run (and every multi-hour research evaluation) runs on the HPC. The OMEN RTX 5050
  is used only for deployment, export, inference benchmarks and smoke tests. The local GPU queue of 2026-10-05
  (`runs/aerotrack_v2/_runner`, chains `post_v2`, `post_v2_causal_caches`, `f1_std`, `x1_cst_only`, `h3_resume_after_v2`,
  `c2_launch`, `cpu_trackc_window`) is **retired, not relaunched**; its stale `RUNNING` sentinels are left untouched as evidence.
- **Evidence**: user directive (mission brief 2026-10-06, rules 9–10); the OMEN hung with bugcheck 0x19C at 19:59 on 2026-10-05
  after `v2_sym_g4_s0_e20`'s throughput fell from ~61 to ~0.5 img/s at ~17:17 (see `docs/FAILURES_AND_RECOVERY.md` F-01).
- **Alternatives**: resume the local queue (rejected: directive). **Why**: directive + machine instability under load.

### D-2026-10-06-02 · Data reach the HPC as pristine archives and are rebuilt there; nothing derived is trusted without a check
- **Decision**: push `datasets/` archives (Anti-UAV-RGBT.zip 6.0 GB, CST z01–z10 + .zip 23.3 GB, Det-Fly 0.33 GB) with
  `tools/hpc/push_files.sh` (resumable append, sha256 on both sides); extract on Linux into a layout mirroring the OMEN's
  `data/`; transfer the small derived files (manifest, splits, label/hardness caches, label txt) as a tar; obtain Anti-UAV frames
  by re-running the project's own decode on the HPC **only if** a byte comparison against OMEN frames passes, else transfer them.
- **Evidence**: OMEN→HPC throughput 5.7 MB/s (≈ 20 GB/h); archives 30 GB vs ~1.07 M decoded frames; manifest rows are keyed by
  row number in the label cache (path-independent); CST/Det-Fly images are read directly from `data/` (no junction in any
  manifest path); Windows junctions (`b_pipeline/data_unified*/{cst,det_fly}`) are therefore never copied.
- **Alternatives**: rsync the derived tree (no rsync on the OMEN; ~10× the bytes; junction hazards); download from the original
  hosts on the HPC (possible, but bytes would still have to be compared with ours). **Why**: least bytes, exact provenance.

### D-2026-10-06-03 · GPUs are used through CPU-only PBS jobs with an idle-GPU guard
- **Decision**: jobs request `select=1:ncpus=N:mem=M:host=<node>` without `ngpus`, check at start that no other compute process
  is on the node's GPU (exit 75 otherwise), and use exactly one GPU. Training data are staged to the node's `/local`.
- **Evidence**: every `ngpus` request fails in the `pbs_cgroups` hook (probe 92820; the user's jobs 92354/92486/92525 held for the
  same reason); CPU-only jobs see the GPU (probes 92823/92824); all users run this way (`ngpus assigned = 0` everywhere); node04's GPU
  was occupied by another user's CPU-only job.
- **Alternatives**: ask the administrators to fix the hook (recommended to the user; not blocking); run on the login node outside
  PBS (rejected: no accounting, survives no reboot, impolite on a shared head node). **Why**: the only working path; the guard
  prevents us from colliding with another user's GPU process.

### D-2026-10-06-04 · The v2 campaign is re-run in full on the HPC under one recipe (protocol amendment A1)
- **Decision**: all six registered runs (stock / sym_g4 / causal_g4 × seeds 0, 1) are trained on V100s with **fp16 AMP +
  GradScaler** (the trainer's existing scaler path) instead of bf16; every arm of every comparison uses the same host type,
  precision and software. The OMEN runs (`v2_stock_s0_e20` complete; `v2_sym_g4_s0_e20` interrupted at epoch 10/20) are historical
  evidence, never mixed into the comparison; `v2_stock_s0_e20` (OMEN) vs its HPC twin is reported as a host/precision
  reproducibility check. The amendment is committed before the first HPC run (`experiments/aerotrack_v2/amendment_A1_hpc.json`).
- **Evidence**: `c_model/aerotrack_trainer.py:246` defaults to `amp_dtype="bf16"`; the V100 has no native bf16 (emulation is
  slow and numerically different); torch on the HPC is 2.6.0+cu118 vs 2.13.0+cu130 on the OMEN (driver 510 cannot run cu13).
- **Alternatives**: resume `sym_g4_s0` from its OMEN epoch-10 checkpoint on the HPC (rejected: one run would span two hosts and
  precisions); fp32 on the V100 (rejected: ~3–5× slower, and still differs from the bf16 runs). **Why**: comparisons stay
  within-host; nothing registered about the hypotheses, metrics, splits, seeds or decision rules changes.

### D-2026-10-06-05 · Machine-portable paths by a load-time prefix map, manifests unchanged
- **Decision**: `AEROTRACK_PATH_MAP="<old prefix>=><new prefix>"` rewrites `image_path` / `label_path` when a manifest is loaded
  (backslashes → `/`). Manifests, splits and caches are byte-identical on both machines, so the recorded manifest hashes stay
  valid. Unset → behaviour unchanged.
- **Alternatives**: write Linux copies of the manifests (rejected: two hashes for one corpus).

### D-2026-10-06-06 · SSH key authentication; the password is never stored
- **Decision**: ed25519 key `~/.ssh/aerotrack_hpc_ed25519` (OMEN) installed in the remote `authorized_keys`; the password was
  used once (process environment only, never written to disk or to a repository file).

### D-2026-10-06-07 · Close Track A gap A2 with the converted public GlobalTrack checkpoint (registered)
- **Decision**: download the public GlobalTrack checkpoint (sha256 e5950365…), convert it key-by-key into our torchvision
  QG-RCNN (322/322 tensors, strict load, exact round trip), and run the paper's **large-scale row** (no training; VAL then one
  TEST pass per modality) and the **faithful visible rows** (12 epochs from that init) under
  `experiments/anti_uav/preregistration/A2_visible_and_largescale.json`.
- **Evidence**: the embedded mmdet config matches our architecture (2 classes bg-first, class-specific regression, RoIAlign 7/2,
  anchors 8 × (0.5,1,2), RCNN stds 0.1/0.2 = torchvision weights 10/5, ImageNet RGB normalisation, scale 1333×800); two
  convention differences (anchor centre offset, legacy +1) are documented and left uncorrected (the paper fine-tunes; the
  large-scale VAL row measures their effect on an un-fine-tuned model).
- **Alternatives**: install mmdetection 0.6 to run the original (rejected: CUDA-extension build for 2019 mmcv on this cluster is
  a project of its own and would not be the code our IR rows used); leave the rows "not reproduced" (rejected: the component is
  available). **Why**: turns the last "component unavailable" rows into a measured result with a built-in fidelity check.

### D-2026-10-06-08 · GPU priority while only one V100 is reliably free
- **Order**: (1) aerotrack_v2 registered runner (Track B headline: complete-CST effect + causal window), (2) Anti-UAV H3
  evaluations, (3) A2 (large-scale VAL, visible trainings), (4) post-v2 / F1 / X1 / C2. Any additional idle GPU takes the next
  Track A item (evaluation-type jobs need few CPUs). Short gates (HC3, zero-shot) run first because they can stop wasted work.
- **Evidence**: `docs/REMOTE_STATE.md` §8 (node01/02 memory reserved by others, head and node04 GPUs in use by others).

### D-2026-10-06-09 · Our own jobs may share one V100
- **Decision**: the GPU guard blocks only other users' processes; two of our jobs may share a GPU (memory budget: v2 training
  ~5 GB + Track A evaluation ~3 GB of 16 GB). Timing figures of shared runs are labelled as such; results are unaffected.
- **Evidence**: HC3 passed (V100 VAL mSA 79.287393 vs OMEN 79.287413); node04 / head GPUs are taken by other users; the H3
  evaluations (~7 h) would otherwise wait for the whole v2 campaign. **Alternative**: strict serial use (rejected: idles the
  GPU while the v2 trainer is loader-bound).
