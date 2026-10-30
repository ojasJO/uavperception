#!/usr/bin/env bash
# The four AUTHORISED live benchmark evaluations for the Anti-UAV paper benchmark task.
# Sequential on purpose: one GPU, one tracker at a time.
set -u
cd /a/AEROTRACK_OMEN || exit 1
B=experiments/anti_uav/runs/paper_benchmarks_20260922T193853
PY=venv/Scripts/python.exe
EV=experiments/anti_uav/scripts/evaluate.py

run () {  # run <run_name> <split> <extra...>
  local name=$1 split=$2; shift 2
  local out="$B/${name}__${split}_IR"
  echo "=== START $name $split $(date -Is) ==="
  "$PY" "$EV" --checkpoint "experiments/anti_uav/runs/$name/weights/last.pt" \
       --split "$split" --modality IR --out-dir "$out" "$@" \
       > "$B/${name}__${split}_IR.log" 2>&1
  local rc=$?
  echo "=== END $name $split rc=$rc $(date -Is) ==="
  echo "{\"run\":\"$name\",\"split\":\"$split\",\"rc\":$rc,\"finished\":\"$(date -Is)\"}" >> "$B/_status.jsonl"
  return $rc
}

run normal_ir val
run dfsc_ir   val
run normal_ir test --final-eval
run dfsc_ir   test --final-eval

echo "ALL DONE $(date -Is)"
