#!/usr/bin/env bash
# Assemble ~/DRONE-Track on the HPC: one self-contained deliverable folder holding
# every file a reviewer needs to re-run or audit this project, and nothing else.
#
#   code/        data preparation, model, training, evaluation, inference, HPC tooling
#   checkpoints/ every trained weight (hardlinked -> no extra disk, survives a later cleanup)
#   logs/        PBS job records (command, environment, events, stdout) + per-run training records
#   results/     every report JSON/MD + the benchmark tree
#   data, data_archives  symlinks to the dataset store (107 GB, never copied)
#
# Files are HARDLINKED from ~/aerotrack_server, not copied: same filesystem, so the
# 29 GB of weights costs 0 bytes and each file keeps a second, independent name.
# Excluded by design: __pycache__, .git, git sync bundles, smoke / dry-run / self-check
# scripts, GPU probe jobs, the superseded review notebooks, and the conda environment.
#
# Idempotent: rebuilds the tree from scratch each run. Read-only with respect to the source.
set -euo pipefail

SRC="$HOME/aerotrack_server"
S="$SRC/source"
DST="$HOME/DRONE-Track"

[ -d "$S" ] || { echo "[FATAL] $S not found"; exit 1; }

echo "[1/8] fresh tree at $DST"
rm -rf "$DST"
mkdir -p "$DST"/{code,checkpoints,logs/pbs_jobs,logs/training,results/reports,results/benchmark}

# ---- link helpers -----------------------------------------------------------
# hl_tree <src dir> <dst dir> [extra find-prune args...]
# Hardlinks a whole tree, dropping caches and the excluded script names.
hl_tree() {
  local src="$1" dst="$2"; shift 2
  [ -d "$src" ] || { echo "   (skip, absent: $src)"; return 0; }
  mkdir -p "$dst"
  ( cd "$src" && find . \
      \( -name __pycache__ -o -name .git -o -name .ipynb_checkpoints -o -path '*/anti_uav/demo' \) -prune -o \
      -type d -print ) | while read -r d; do mkdir -p "$dst/$d"; done
  ( cd "$src" && find . \
      \( -name __pycache__ -o -name .git -o -name .ipynb_checkpoints -o -path '*/anti_uav/demo' \) -prune -o \
      -type f ! -name '*.pyc' \
      ! -name 'smoke_test.py' ! -name 'selfcheck.py' ! -name 'preflight.py' \
      ! -name '1_cpu_dry_run.py' ! -name 'browser_test.py' \
      ! -name 'cpu_trackc_restart.json' ! -name 'cpu_trackc_window.json' \
      ! -name 'hota_crosscheck.json' \
      -print ) | while read -r f; do
        ln -f "$src/$f" "$dst/$f" 2>/dev/null || cp -p "$src/$f" "$dst/$f"
      done
}

# hl_glob <dst dir> <files...>
hl_glob() {
  local dst="$1"; shift; mkdir -p "$dst"
  for f in "$@"; do
    [ -f "$f" ] || continue
    ln -f "$f" "$dst/$(basename "$f")" 2>/dev/null || cp -p "$f" "$dst/"
  done
}

# ---- 2. code ----------------------------------------------------------------
echo "[2/8] code: data pipeline, model, experiments, tooling"
hl_tree "$S/b_pipeline" "$DST/code/b_pipeline"
# the data_unified* dirs under b_pipeline are derived label caches, not code
rm -rf "$DST/code/b_pipeline/data_unified" "$DST/code/b_pipeline/data_unified_v2"
hl_tree "$S/c_model"   "$DST/code/c_model"
hl_tree "$S/tools"     "$DST/code/tools"
hl_glob "$DST/code" "$S/run_aerotrack.py" "$S/requirements.txt"

# experiments: scripts / configs / preregistrations / source packages only.
for e in aerotrack_generalization aerotrack_repair aerotrack_temporal \
         aerotrack_v2 aerotrack_v3 anti_uav cst_anti_uav cst_completion \
         deployment hpc systems; do
  [ -d "$S/experiments/$e" ] || continue
  for sub in scripts src configs preregistration manifests demo; do
    [ -d "$S/experiments/$e/$sub" ] && hl_tree "$S/experiments/$e/$sub" "$DST/code/experiments/$e/$sub"
  done
  # top-level files of the experiment (READMEs, frozen parameter sets, ledgers)
  for f in "$S/experiments/$e"/*; do
    [ -f "$f" ] && hl_glob "$DST/code/experiments/$e" "$f"
  done
done
hl_glob "$DST/code/experiments" "$S/experiments/AMENDMENT_A1_HPC.json"

# ---- 3. checkpoints ---------------------------------------------------------
echo "[3/8] checkpoints (hardlinked)"
# detector runs: runs/<name>/weights/*.pt  -> checkpoints/detector/<name>/
find "$S/runs" -type f -name '*.pt' -size +1M 2>/dev/null | while read -r w; do
  run=$(basename "$(dirname "$(dirname "$w")")")
  case "$run" in *smoke*|*dry_run*) continue;; esac
  mkdir -p "$DST/checkpoints/detector/$run"
  ln -f "$w" "$DST/checkpoints/detector/$run/$(basename "$w")" 2>/dev/null || true
done
# Anti-UAV reproduction runs (GlobalTrack / DFSC): keep .pth too
find "$S/experiments/anti_uav/runs" -type f \( -name '*.pt' -o -name '*.pth' \) -size +1M 2>/dev/null | while read -r w; do
  rel=${w#"$S/experiments/anti_uav/runs/"}
  mkdir -p "$DST/checkpoints/anti_uav_repro/$(dirname "$rel")"
  ln -f "$w" "$DST/checkpoints/anti_uav_repro/$rel" 2>/dev/null || true
done
# the converted public GlobalTrack baseline
hl_tree "$SRC/checkpoints" "$DST/checkpoints/globaltrack_baseline"
# COCO initialisation used by every detector run
hl_glob "$DST/checkpoints/pretrained" "$S/c_model/yolo11n.pt" "$S/c_model/yolo11s.pt"

# ---- 4. logs: PBS job records ----------------------------------------------
echo "[4/8] logs: PBS job records"
for j in "$SRC/jobs"/*; do
  [ -d "$j" ] || continue
  n=$(basename "$j")
  case "$n" in probe|gpuprobe_*|*.withdrawn_*|*.gpu_busy_*|*.failed_import) continue;; esac
  hl_tree "$j" "$DST/logs/pbs_jobs/$n"
done
hl_tree "$SRC/logs" "$DST/logs/server_logs"

# ---- 5. logs: per-run training records (no weights) -------------------------
echo "[5/8] logs: per-run training records"
for base in "$S/runs" "$S/experiments/anti_uav/runs"; do
  [ -d "$base" ] || continue
  tag=$([ "$base" = "$S/runs" ] && echo detector || echo anti_uav_repro)
  find "$base" -type f \( -name 'results.csv' -o -name 'args.yaml' -o -name 'events.jsonl' \
      -o -name '*.json' -o -name '*.md' -o -name '*.log' -o -name '*.txt' -o -name '*.jsonl' \
      -o -name '*.png' -o -name '*.yaml' \) 2>/dev/null | while read -r f; do
    rel=${f#"$base/"}
    case "$rel" in *smoke*|*dry_run*|*__pycache__*) continue;; esac
    mkdir -p "$DST/logs/training/$tag/$(dirname "$rel")"
    ln -f "$f" "$DST/logs/training/$tag/$rel" 2>/dev/null || true
  done
done

# ---- 6. results: reports + benchmark tree ----------------------------------
echo "[6/8] results: reports and benchmark tables"
hl_tree "$SRC/reports"   "$DST/results/reports/server"
hl_tree "$SRC/manifests" "$DST/results/manifests"
for e in aerotrack_generalization aerotrack_repair aerotrack_temporal aerotrack_v2 \
         aerotrack_v3 anti_uav cst_anti_uav cst_completion deployment hpc systems; do
  [ -d "$S/experiments/$e/reports" ] && hl_tree "$S/experiments/$e/reports" "$DST/results/reports/$e"
  [ -d "$S/experiments/$e/hpc/reports" ] && hl_tree "$S/experiments/$e/hpc/reports" "$DST/results/reports/${e}_hpc"
done
hl_tree "$S/benchmark" "$DST/results/benchmark"
# Only the documents that substantiate a published number travel with the deliverable:
# the protocol, the experiment ledger, the dataset specification and the results table.
# The working documents (plans, state, handoffs, audits, reviews) stay on the laptop.
mkdir -p "$DST/results/project_docs"
hl_glob "$DST/results/project_docs" \
  "$S/docs/RESULTS.md" "$S/docs/EXPERIMENT_PROTOCOL.md" \
  "$S/docs/EXPERIMENT_LEDGER.csv" "$S/docs/EXPERIMENT_LEDGER.jsonl" \
  "$S/docs/ANTI_UAV_EXPERIMENT_PROTOCOL.md" "$S/docs/ANTI_UAV_REPRODUCTION_SPEC.md" \
  "$S/docs/DATASETS_data_unified_v2.md"

# ---- 7. dataset symlinks ----------------------------------------------------
echo "[7/8] dataset symlinks (nothing copied)"
ln -sfn "$SRC/shared_data/root/data"            "$DST/data"
ln -sfn "$SRC/shared_data/source_archives"      "$DST/data_archives"
ln -sfn "$SRC/shared_data/root/b_pipeline"      "$DST/data_derived"
ln -sfn "$SRC/shared_data/root/DATA_VERSION"    "$DST/DATA_VERSION"

# ---- 8. inventory -----------------------------------------------------------
echo "[8/8] inventory"
{
  echo "# DRONE-Track inventory"
  echo
  echo "Built $(date -Is) on $(hostname) from \$HOME/aerotrack_server (hardlinks; dataset trees symlinked)."
  echo
  echo '| part | files | apparent size |'
  echo '|---|---:|---:|'
  for p in code checkpoints logs results; do
    printf '| %s | %s | %s |
' "$p" \
      "$(find "$DST/$p" -type f 2>/dev/null | wc -l)" \
      "$(du -sh --apparent-size "$DST/$p" 2>/dev/null | cut -f1)"
  done
  echo
  echo '## Symlinked dataset trees'
  echo '```'
  ls -l "$DST" | grep '^l'
  echo '```'
  echo
  echo '## Checkpoints'
  echo '```'
  find "$DST/checkpoints" -type f \( -name '*.pt' -o -name '*.pth' \) | sort
  echo '```'
} > "$DST/INVENTORY.md"

echo
echo "DONE -> $DST"
du -sh --apparent-size "$DST" 2>/dev/null
df -h "$HOME" | tail -1
