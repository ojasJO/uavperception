#!/usr/bin/env bash
# Seed-1 TEST evaluations: gives a 2-seed spread on the test split so the DFSC effect
# (paper +0.68 pt; our seed-0 pair gave -0.29 pt) can be read against seed noise.
# Evaluation only - both checkpoints were trained on 2026-09-20/21 and are frozen.
set -u
cd /a/AEROTRACK_OMEN || exit 1
B=experiments/anti_uav/runs/paper_benchmarks_20260922T193853/seed_variance
mkdir -p "$B"
# write this script's own stdout/stderr into the driver log. Doing it HERE (after mkdir)
# rather than piping the launch command through `tee` avoids depending on the directory
# existing before the script starts.
exec > >(tee -a "$B/_driver.log") 2>&1
PY=venv/Scripts/python.exe
EV=experiments/anti_uav/scripts/evaluate.py

run () {
  local name=$1 split=$2; shift 2
  local out="$B/${name}__${split}_IR"
  echo "=== START $name $split $(date -Is) ==="
  "$PY" "$EV" --checkpoint "experiments/anti_uav/runs/$name/weights/last.pt" \
       --split "$split" --modality IR --out-dir "$out" "$@" \
       > "$B/${name}__${split}_IR.log" 2>&1
  local rc=$?
  echo "=== END $name $split rc=$rc $(date -Is) ==="
  echo "{\"run\":\"$name\",\"split\":\"$split\",\"rc\":$rc,\"finished\":\"$(date -Is)\"}" >> "$B/_status.jsonl"
}

run phaseB_normal_ir_seed1 test --final-eval
run phaseB_dfsc_ir_seed1   test --final-eval

echo "SEED VARIANCE DONE $(date -Is)"
