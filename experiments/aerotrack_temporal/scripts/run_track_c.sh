#!/usr/bin/env bash
# Track C end-to-end execution pipeline (C1C -> C3 -> C1B)
# Submitted to PBS via tools/hpc/submit.sh
set -euo pipefail

source "$HOME/aerotrack_server/source/tools/hpc/job_env.sh"
cd "$HOME/aerotrack_server/source"

echo "=== [$(date -Iseconds)] STAGE 1: C1C Forecaster Training & Dev Selection ==="
python experiments/aerotrack_temporal/scripts/tc_c1c.py --stage train

echo "=== [$(date -Iseconds)] Committing learned_registry_c1c.json ==="
git add runs/aerotrack_temporal/c1/learned_registry_c1c.json
git commit -m "c1c: record dev-chosen forecaster models" || true

echo "=== [$(date -Iseconds)] STAGE 2: C1C Validation Pass ==="
python experiments/aerotrack_temporal/scripts/tc_c1c.py --stage val

echo "=== [$(date -Iseconds)] STAGE 3a: C3 Tracking Evaluation (primary: v2_causal_g4_s0_e20) ==="
python experiments/aerotrack_temporal/scripts/tc_c3.py --stage run --run v2_causal_g4_s0_e20 --workers 4

echo "=== [$(date -Iseconds)] STAGE 3b: C3 Tracking Evaluation (secondary: v2_stock_s0_e20) ==="
python experiments/aerotrack_temporal/scripts/tc_c3.py --stage run --run v2_stock_s0_e20 --workers 4

echo "=== [$(date -Iseconds)] STAGE 4a: C1B Tracker History Forecasting (primary: v2_causal_g4_s0_e20) ==="
python experiments/aerotrack_temporal/scripts/tc_c1b.py --run v2_causal_g4_s0_e20 --corpus data_unified_v2

echo "=== [$(date -Iseconds)] STAGE 4b: C1B Tracker History Forecasting (secondary: v2_stock_s0_e20) ==="
python experiments/aerotrack_temporal/scripts/tc_c1b.py --run v2_stock_s0_e20 --corpus data_unified_v2

echo "=== [$(date -Iseconds)] ALL TRACK C STAGES COMPLETED CLEANLY ==="
