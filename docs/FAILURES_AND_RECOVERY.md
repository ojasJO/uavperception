# FAILURES_AND_RECOVERY

*Every meaningful crash, OOM, data or infrastructure problem: symptom · evidence · root cause · fix · protocol impact. Earlier
incidents are summarised at the end with pointers to their full records.*

---

### F-01 · 2026-10-05 · OMEN hang and bugcheck during `v2_sym_g4_s0_e20`
- **Symptom**: training throughput fell from ~61 img/s (epochs 0–10, ~410 s/epoch) to 0.4–0.7 img/s at the start of epoch 11
  (~17:17); the child exited with rc 4294967295 (−1) at 19:20 after 3.51 h; the automatic resume (`--resume`, 19:20) loaded and
  verified the model but logged no step; Windows hung at 19:59 and rebooted at 20:53 with **bugcheck 0x19C** (0x50, …) =
  WIN32K_POWER_WATCHDOG_TIMEOUT (System log events 41 / 6008 / 1001).
- **Evidence**: `runs/aerotrack_v2/_runner/train_v2_sym_g4_s0_e20.log` (tail), `…_retry.log`, `runs/aerotrack_v2/_runner/events.jsonl`;
  no other project process wrote any file between 16:06 and 20:00 (C1c had been stopped at 16:05); `last.pt` = epoch 10 (17:17),
  loads cleanly (`epoch 10`, keys model/ema/optimizer).
- **Root cause**: **undetermined**. Not a project CPU job (none running). The earlier memory-pressure pattern (error 1455 on
  2026-10-05 06:57) is a candidate but no memory counter was logged; a display/power-driver stall is consistent with 0x19C.
- **Fix / recovery**: none on the OMEN — training moved to the HPC (D-2026-10-06-01). The interrupted run is kept as is
  (historical, never resumed, never scored); the campaign is re-run on the HPC (D-2026-10-06-04). Work lost: 0 registered
  results (the run's 11 completed epochs are not reused).
- **Protocol impact**: host/precision amendment A1 (all arms re-run on one host type).

### F-02 · 2026-10-06 · PBS rejects every GPU request
- **Symptom**: jobs with `ngpus=1` go R → H, comment "job held, too many failed attempts to run", run_count 21.
- **Evidence**: `tracejob 92820`: "Processing error in pbs_cgroups handling execjob_begin event … CgroupProcessingError ('Failed
  to assign resources')"; the user's jobs 92354 / 92486 / 92525 (Sep 18–25) held identically.
- **Root cause**: cluster-side `pbs_cgroups` hook misconfiguration for the `ngpus` resource (not fixable by a user).
- **Fix**: CPU-only PBS jobs + idle-GPU guard (D-2026-10-06-03). Our probe jobs 92820–92822 deleted; the user's held jobs left
  untouched. Recommended to the user: report the hook error to the cluster administrators.
- **Protocol impact**: none (execution mechanics only).

### F-03 · 2026-10-06 · first archive-push launch ran in WSL instead of Git Bash
- **Symptom**: the detached push exited immediately without output. **Cause**: `Get-Command bash.exe` resolved to the WSL launcher
  (`WindowsApps\bash.exe`). **Fix**: launch with `C:\Program Files\Git\bin\bash.exe` explicitly. **Impact**: none (nothing written).

### F-04 · found 2026-10-06 · Track A's data-loading package was never under version control
- **Symptom**: on the HPC checkout, `import anti_uav_repro.data` failed (ModuleNotFoundError).
- **Evidence**: `.gitignore:18` `data/` matched `experiments/anti_uav/src/anti_uav_repro/data/` (`antiuav.py`, `pairs.py`,
  `transforms.py`, `__init__.py`); `git ls-files` listed none of them; no other source file was affected
  (`git ls-files --others --ignored -- '*.py'` outside data/run trees: only these four and the intentionally ignored
  `rebuild_junctions.py`).
- **Root cause**: an over-broad ignore pattern; the runs' `git.dirty` flag could not reveal it (ignored files are invisible to git).
- **Provenance**: file mtimes 2026-09-19 19:34–20:02, i.e. before Phase A training started and never modified since, so the
  committed content is the code every Track A run used. sha256 prefixes: antiuav.py `9a0fd1e44009`, pairs.py `5ca0604f6077`,
  transforms.py `f1fef0170027`.
- **Fix**: `.gitignore` exception + files committed unchanged (2026-10-06). **Protocol impact**: none (no code change); the
  reproducibility package now contains the data loader.

### F-05 · 2026-10-06 14:17–15:05 · the single v2 TEST pass crashed after its detection part
- **Symptom**: `final_eval.py --final-eval` (PBS 92877) finished (a) detection AP (ledger line written 14:23) and (b) the full TEST
  detection cache (242 sequences, 242,923 frames), then raised `AttributeError: module 'track_eval' has no attribute 'load'`.
- **Root cause**: module shadowing — `cache_detections` (imported in step b) puts `c_model/` first on `sys.path`, and
  `c_model/track_eval.py` (the old Review-2 evaluator) then won over `experiments/systems/scripts/track_eval.py`. Latent: the
  `--final-eval` path had never been executed before (registered, single use).
- **Fix**: the systems modules are imported by explicit file path (asserted); `--resume-after-crash` finishes the SAME pass: it
  requires exactly one detection ledger record and no systems record for these weights, the protocol and frozen parameters
  committed and unchanged, reuses the complete TEST cache of these exact weights, and computes only the never-computed SOT /
  paired comparisons / forecasting with the parameters frozen before the pass; output and ledger line carry `resumed_after_crash`.
  `cache_detections` now records its real autocast dtype (it said "bf16" unconditionally).
- **Protocol impact**: none on what is measured or selected (nothing remained to choose: detector, tracker parameters and metrics
  were frozen before the pass). Disclosure: the detection TEST metrics were displayed (by the operator) before the systems part was
  computed; they cannot influence it.

---

## Earlier incidents (full records elsewhere)

| date | incident | record |
|---|---|---|
| 2026-08-04 | CST archive delivered as 7 of 11 parts; sequence-key collision; Anti-UAV 9.6 % materialised | `docs/PROJECT_KNOWLEDGE.md` §B; `00_PROJECT_JOURNAL.md` §5 |
| 2026-09-03 | standardisation without Pillow produced 162,187 empty CST labels | `00_PROJECT_JOURNAL.md` §5 |
| 2026-09-23 | OMEN crash (Kernel-Power 41) interrupted the DFSC-all TEST pass at 54/91 (no valid result) | `docs/PROJECT_KNOWLEDGE.md` §A.5 |
| 2026-09-24 | DataLoader collate memory leak | `docs/progress/AEROTRACK_REPAIR_PROGRESS.md` |
| 2026-10-05 06:57 | H3 launch killed by Windows commit limit (error 1455) beside CPU chains | `docs/progress/RESEARCH_PROGRAM_PROGRESS.md` milestones |
| 2026-10-05 11:11 | v2 attempt 1 failed its smoke run (window-unaware pre-flight T2) | same |
| 2026-10-05 15:59 | Track C loader kept ~400 decompressed copies (~6 GB) | `docs/PROJECT_KNOWLEDGE.md` §A.10 |
