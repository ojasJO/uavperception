#!/usr/bin/env bash
# Track C Part 2: Replication tracking evaluation (stock) and tracker history forecasting (causal + stock)
set -euo pipefail

source "$HOME/aerotrack_server/source/tools/hpc/job_env.sh"
cd "$HOME/aerotrack_server/source"

echo "=== [$(date -Iseconds)] STAGE 4a: C1B Tracker History Forecasting (primary: v2_causal_g4_s0_e20) ==="
python experiments/aerotrack_temporal/scripts/tc_c1b.py --run v2_causal_g4_s0_e20 --corpus data_unified_v2

echo "=== [$(date -Iseconds)] STAGE 3b: C3 Tracking Evaluation (secondary: v2_stock_s0_e20) ==="
python experiments/aerotrack_temporal/scripts/tc_c3.py --stage run --run v2_stock_s0_e20 --workers 8

echo "=== [$(date -Iseconds)] STAGE 4b: C1B Tracker History Forecasting (secondary: v2_stock_s0_e20) ==="
python experiments/aerotrack_temporal/scripts/tc_c1b.py --run v2_stock_s0_e20 --corpus data_unified_v2

echo "=== [$(date -Iseconds)] ALL TRACK C PART 2 STAGES COMPLETED CLEANLY ==="
