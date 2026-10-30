# HPC RUNBOOK (amritahpc, PBS) — AeroTrack

Everything below runs from the OMEN in Git Bash unless marked **[HPC]**. Access is by SSH key only (`ssh amritahpc`); never type
or store the password in any file.

## 0. Layout
```
~/aerotrack_server/source          git checkout (same commit as the OMEN; sparse: OMEN v2 artifacts excluded)
  data -> shared_data/root/data                    (symlink, Linux; never a copied Windows junction)
  b_pipeline/data_unified{,_v2} -> shared_data/root/b_pipeline/...
~/aerotrack_server/shared_data/source_archives     pristine archives (sha256 = OMEN datasets/)
~/aerotrack_server/shared_data/root                 data tree mirroring the OMEN's relative paths (AEROTRACK_PATH_MAP target)
/local/$USER/at_root                                per-node copy of shared_data/root (training reads here; STAGED_<version> marker)
~/aerotrack_server/environments/at311               conda env (+ at311_freeze.txt)
~/aerotrack_server/jobs/<run_id>/                   one dir per job: cmd.txt, env.json, events.jsonl, job.log, RUNNING/DONE/FAILED/GPU_BUSY
~/aerotrack_server/reports/hpc/                     gate / verification JSON reports
```

## 1. Ship code
Commit on the OMEN, then `bash tools/hpc/sync_code.sh` (incremental bundle, fast-forward only; refuses on a dirty HPC tree).

## 2. Launch a job (detached by construction: PBS owns the process; laptop / SSH can disappear)
**[HPC]** `cd ~/aerotrack_server/source && tools/hpc/submit.sh <run_id> <host> <ncpus> <mem_gb> <HH:MM:SS> <gpu 0|1> <stage 0|1> -- <command>`
- pick a host whose CPUs are free (`pbsnodes -a | grep -E "^[a-z]|assigned.ncpus"`); GPU jobs abort with `GPU_BUSY` (exit 75)
  if another process is on that node's V100 — resubmit on another host.
- `AT_DEPEND=<pbs id>` makes the job wait for another job to succeed.
- never request `ngpus` (cluster cgroup hook rejects it).
From the OMEN: `ssh amritahpc 'cd ~/aerotrack_server/source && tools/hpc/submit.sh ...'`.

## 3. Status (cheap; do not poll)
`ssh amritahpc 'qstat -u $USER; ls ~/aerotrack_server/jobs/<run_id>; tail -5 ~/aerotrack_server/jobs/<run_id>/job.log'`

## 4. Resume after a failure
- Training (`train_v2.py` / `train_v3.py`): resubmit the same command with `--resume` (continues from `runs/<name>/weights/last.pt`,
  written every epoch; attempts are appended to `run_meta.json`). Check `last.pt` loads first. Record the incident in
  `docs/FAILURES_AND_RECOVERY.md`.
- Evaluations: re-run; outputs are written atomically and complete results are never overwritten (scripts refuse).
- A run_id with `DONE` is immutable; use a new run_id for anything new.

## 5. Retrieve results
`venv/Scripts/python.exe tools/hpc/pull_results.py` copies the HPC's result files (reports, ledgers, run metadata, job records — never weights unless
asked) into the OMEN under `experiments/hpc/` and `runs/hpc/`, then `python tools/build_master_ledger.py`, commit.

## 6. Data (re)build order (done once; documented in docs/REMOTE_STATE.md)
archives (`tools/hpc/push_files.sh`) → unzip Anti-UAV / Det-Fly → `tools/hpc/decode_antiuav.py` → `tools/hpc/verify_antiuav_frames.py`
→ `tools/hpc/extract_cst.py` → corpus core bundle → `tools/hpc/env_gate.py --part data` → `DATA_VERSION`.
