#!/usr/bin/env bash
# Paper Table VI ablation "DFSC-all": train (18 epochs) then evaluate on TEST once.
# Waits for the seed-variance evaluations to release the GPU first - the 8 GB card cannot hold a
# training run and a tracker at the same time.
set -u
cd /a/AEROTRACK_OMEN || exit 1
ROOT=experiments/anti_uav/runs/paper_benchmarks_20260922T193853
SV="$ROOT/seed_variance/_status.jsonl"
B="$ROOT/ablation"
mkdir -p "$B"
PY=venv/Scripts/python.exe
CFG=experiments/anti_uav/configs/phase_b/dfsc_ir_ism_cross.yaml
RUN=phaseB_dfsc_ir_ism_cross

echo "=== waiting for seed-variance evals to finish $(date -Is) ==="
while [ "$( [ -f "$SV" ] && wc -l < "$SV" || echo 0 )" -lt 2 ]; do sleep 60; done
echo "=== GPU free $(date -Is) ==="

echo "=== TRAIN $RUN $(date -Is) ==="
"$PY" experiments/anti_uav/scripts/train.py --config "$CFG" > "$B/train_${RUN}.log" 2>&1
rc=$?
echo "{\"stage\":\"train\",\"run\":\"$RUN\",\"rc\":$rc,\"finished\":\"$(date -Is)\"}" >> "$B/_status.jsonl"
echo "=== TRAIN END rc=$rc $(date -Is) ==="
[ $rc -ne 0 ] && { echo "TRAINING FAILED - not evaluating"; exit $rc; }

echo "=== EVAL $RUN test $(date -Is) ==="
"$PY" experiments/anti_uav/scripts/evaluate.py     --checkpoint "experiments/anti_uav/runs/$RUN/weights/last.pt"     --split test --modality IR --final-eval     --out-dir "$B/${RUN}__test_IR" > "$B/${RUN}__test_IR.log" 2>&1
rc=$?
echo "{\"stage\":\"eval\",\"run\":\"$RUN\",\"split\":\"test\",\"rc\":$rc,\"finished\":\"$(date -Is)\"}" >> "$B/_status.jsonl"
echo "=== EVAL END rc=$rc $(date -Is) ==="
echo "ABLATION DONE $(date -Is)"
