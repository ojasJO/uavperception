#!/usr/bin/env bash
# [HPC] Queue the aerotrack_v2 campaign (amendment A1) on one node: M6 loader benchmark -> registered runner (run_v2.py).
#   tools/hpc/queue_v2.sh <host> [after_pbs_job_id]
# M6: train_v2.py --preflight-only (the production pre-flight incl. P.check_throughput) for 8 / 12 / 16 loader workers, one job,
#     sequentially; nothing is trained. Choose AEROTRACK_WORKERS from its log, then submit the runner with
#     tools/hpc/queue_v2.sh runner <host> <workers> [after]
set -eu
cd "$HOME/aerotrack_server/source"
if [ "$1" = "runner" ]; then
  HOST="$2"; W="$3"; AFTER="${4:-}"
  AT_DEPEND="$AFTER" tools/hpc/submit.sh v2_runner_hpc "$HOST" $((W + 4)) 96 47:00:00 1 1 -- \
    "AEROTRACK_WORKERS=$W python -u experiments/aerotrack_v2/scripts/run_v2.py"
  exit 0
fi
HOST="$1"; AFTER="${2:-}"
CMD=""
for W in 8 12 16; do
  CMD="$CMD python -u experiments/aerotrack_v2/scripts/train_v2.py --config sym_g4 --seed 0 --preflight-only --workers $W --name m6bench_w$W 2>&1 | tail -60;"
done
AT_DEPEND="$AFTER" tools/hpc/submit.sh m6_loader_bench "$HOST" 20 96 02:00:00 1 1 -- "$CMD"
