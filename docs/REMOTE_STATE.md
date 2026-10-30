# REMOTE_STATE — the HPC (`amritahpc`, 172.17.18.100)

*Facts verified live on 2026-10-06 (00:55–01:15 IST) from read-only inspection; update this file whenever the remote state
changes. Credentials are never written to the repository: access is by the SSH key `~/.ssh/aerotrack_hpc_ed25519` (OMEN) and the
host alias `amritahpc` in `~/.ssh/config`.*

## 1. Machine

| item | value |
|---|---|
| cluster | Bright-managed PBS Pro cluster: head `amritahpc` (login + execution host) + `node01`–`node04` |
| OS | RHEL 7.9 (kernel 3.10.0-693), **glibc 2.17** → only manylinux2014 wheels install |
| CPU / RAM per host | 2× Xeon Gold 5120 (28 cores, no SMT); head 125 GB, nodes 251 GB |
| GPU per host | **1× Tesla V100-PCIE-16GB** (sm_70; fp16 tensor cores; **no native bf16**) — 5 GPUs in the cluster |
| driver | 510.47.03 (**CUDA ≤ 11.6**); no system nvcc; CUDA 11.8 runtime wheels run via minor-version compatibility |
| storage | `/home` = NFS v3 from `master` (xfs, 25 TB, **99 % full, ~437 GB free, shared by all users**); per-node **`/local` 479 GB xfs, empty** (node03/node04 verified); `/tmp` 2 GB on nodes |
| network | internet works from the login node (PyPI, download.pytorch.org, GitHub: HTTP 200); OMEN → HPC scp ≈ **5.7 MB/s** over campus Wi-Fi |
| python | `~/anaconda3` (conda 24.11.3, Python 3.12.2); modules: gcc 7.2.0, python3, openmpi (no CUDA module) |

## 2. Scheduler (PBS Pro) — verified behaviour

| queue | walltime max | per-user limits |
|---|---|---|
| `workq` | 48 h | 98 cpus, 5 gpus |
| `testq` | 96 h | 14 cpus, 1 gpu |
| `matlabq` | 96 h | 2 cpus |

- **Requesting `ngpus` fails**: the `pbs_cgroups` hook rejects every GPU request (`CgroupProcessingError ('Failed to assign
  resources')`, tracejob of probe 92820, run_count 21 → held). The user's older jobs 92354 / 92486 / 92525 are held for the same
  reason. `resources_assigned.ngpus = 0` on every host for every user.
- **CPU-only jobs see the GPU**: probes 92823 (node03) and 92824 (node04) without `ngpus` ran `nvidia-smi` successfully; no device
  cgroup is enforced and `CUDA_VISIBLE_DEVICES` is empty.
- **Consequence**: PBS does not give GPU exclusivity. At 01:12 node04's V100 was 99 % busy (2.9 GB) under another user's CPU-only
  job (`s_divya2`, 92812). Our jobs therefore (a) request CPUs/memory via PBS and pin a host with `-l select=...:host=<node>`,
  (b) run a **GPU-idle guard** at start (abort, exit 75, if any other compute process or > 500 MiB is on the GPU), (c) use one
  GPU per job. See `docs/DECISIONS.md` D-2026-10-06-03.
- SLURM client tools exist but are unconfigured (`squeue`: DNS SRV lookup failed) — not usable.

Load snapshot 2026-10-06 01:10 (other users): node01 23/28 cpus, node02 27/28, node03 0/28, node04 4/28 (GPU busy), head 16/28.

## 3. Pre-existing content (forensic inventory 2026-10-05, full index of `$HOME`, 4,143,918 entries)

- **No AeroTrack / OMEN / Anti-UAV-reproduction work had ever been executed on the HPC.** No matching paths, no git repo, no
  checkpoint, no PBS job (server job history enabled; none named aero/uav/omen/yolo/track, any user). Newest non-cache file
  before this program: 2026-10-01 00:00 (`final_year_project_29`).
- `~/anti_uav` (7.4 GB, 2026-09-30): Kaggle download of the **DUT Anti-UAV-Tracking-V0** dataset (20 videos + GT) and env
  `anaconda3/envs/anti_uav` (Python 3.10 + kagglehub only). Different dataset from ours (Anti-UAV410/300 RGBT, CST, Det-Fly);
  unused by this project; left untouched.
- Other projects (untouched): `final_year_project_29` (291 GB), `vlil_40`, `akhils_hpc`, `Team_4_Dynamic_Scene_Graph`,
  `lane_detection_project`, `datasets`, `anomaly_robot`, ….

## 4. Changes made by this program on the HPC

| when | change |
|---|---|
| 2026-10-05 21:02 | `/tmp/tmux-1189` created by a read-only `tmux ls`; `/var/tmp/.pbs_last_query_1189` by `qstat` |
| 2026-10-06 01:04 | public key appended to `~/.ssh/authorized_keys` (backup `authorized_keys.bak_aerotrack_20261006010434`) |
| 2026-10-06 01:05 | workspace `~/aerotrack_server/` created (layout §5) |
| 2026-10-06 01:07 | source archives pushed (sha256-verified) → `shared_data/source_archives/` |
| 2026-10-06 01:09 | env build started: `docs/environments/at311` (conda, Python 3.11) |
| 2026-10-06 01:11 | probe jobs 92820–92822 (GPU request: held → deleted by us), 92823–92824 (CPU-only: ran) |

## 5. Workspace layout (`~/aerotrack_server/`)

```text
source/                 git checkout of this repository (transferred as a git bundle; commit recorded per run)
shared_data/
  source_archives/      pristine archives, byte-identical to the OMEN's datasets/ (sha256 in results/runs/_hpc_sync/*/sha256.tsv)
  data/                 extractions (Linux; no junctions) mirroring the OMEN's data/ layout
  derived/              corpus files mirroring data/derived/data_unified*/ (labels, manifest, caches, Anti-UAV frames)
docs/environments/           at311 (conda prefix) + pip freeze
holistic/  anti_uav_replication/   per-track run roots (immutable run directories)
checkpoints/ logs/ reports/ manifests/ src/deployment/ jobs/
```
Training reads images from a per-node copy in `/local/y_rajeshwar/` (staged by the job; NFS is the source of truth).

## 6. Environment (`docs/environments/at311`, freeze copied to `experiments/hpc/env/at311_freeze.txt`)

| item | HPC `at311` | OMEN `venv` |
|---|---|---|
| Python | 3.11.13 | 3.11.9 |
| torch / torchvision | 2.6.0+cu118 / 0.21.0+cu118 (2.7.1 has no glibc-2.17 wheel) | 2.13.0+cu130 / 0.28.0+cu130 |
| precision (AeroTrack) | **fp16 autocast + GradScaler** (V100) | bf16 autocast (RTX 5050) |
| precision (Track A) | fp32 (`amp: none`) | fp32 |
| Ultralytics | 8.4.115 | 8.4.115 |
| OpenCV | **5.0.0.93 (same build)** | 5.0.0.93 |
| numpy / pandas / scipy | 2.2.6 / 2.3.2 / 1.16.3 (newer need glibc 2.28) | 2.4.6 / 3.0.5 / 1.17.1 |

Gate M4 core **PASS** (node03, `experiments/hpc/reports/m4_env_gate_core_node03_r3.json`): fp32/fp16 kernels, both detector
builds with the registered COCO-transfer counts, fp16 GradScaler training steps, checkpoint round trip; GPU-only training
throughput at batch 32 / 640: **fp16 231 img/s (5.0 GB)**, fp32 172 img/s (8.7 GB). (Two earlier FAILs were test flaws: see
commits 3379276 and the C3 note in `scripts/hpc/env_gate.py`.)

Side effect: Ultralytics migrated the user's pre-existing `~/.config/Ultralytics/settings.json` to its current schema
(values preserved, per its own message) on first import.

## 7. Checkout specifics
- `source/data`, `source/b_pipeline/data_unified`, `source/b_pipeline/data_unified_v2` are symlinks into `shared_data/root/`
  (listed in `.git/info/exclude`).
- Sparse checkout excludes the OMEN's v2 artifacts (`results/runs/v2_stock_s0_e20/`, `results/runs/v2_sym_g4_s0_e20/`,
  `results/runs/v2_causal_g4_s0_e2_smoke/`, `results/runs/aerotrack_v2/`, `experiments/aerotrack_v2/{ledger.jsonl,reports/v2_results*.json}`)
  so the series re-runs from scratch under its registered names (amendment A1). Re-hosted OMEN checkpoints for evaluation only:
  `results/runs/laptop_v2_stock_s0_e20/` (HC1), `results/runs/gen_temporal_fp_g4_full_e20/weights/` (v1 reference).

## 8. Practical GPU capacity (2026-10-06 ~02:00)
Of 5 V100s: node03 free; head node GPU used by another user's process (01:55); node04 GPU used by another user's CPU-only job;
node01 / node02 have their memory fully reserved by another user's 250 GB jobs. **Plan for 1 GPU, opportunistically 2.**

## 9. Active jobs
See `docs/MASTER_STATE.md` (live table) and `experiments/hpc/jobs/` (pulled records).

## 10. Provenance note: `git.dirty` in HPC run records
`experiments/aerotrack_v2/scripts/v2series.git()` and Track A's run metadata use `git status --porcelain`, which also lists
untracked files. On the HPC checkout the run outputs (`results/runs/…`, `experiments/anti_uav/runs/<new evals>`) are untracked, so those
records say `dirty: true` even though no tracked file differs. Tracked-file cleanliness is checked separately by
`scripts/hpc/sync_code.sh` (`--untracked-files=no`, refuses otherwise) and recorded per job in `env.json` (`git_dirty`, tracked only).
