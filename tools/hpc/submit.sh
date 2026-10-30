#!/usr/bin/env bash
# Submit one AeroTrack job to PBS (run on the HPC login node).
#   tools/hpc/submit.sh <run_id> <host> <ncpus> <mem_gb> <walltime HH:MM:SS> <gpu 0|1> <stage 0|1> -- <command ...>
# Example:
#   tools/hpc/submit.sh m4_env_gate node03 8 32 00:30:00 1 0 -- python tools/hpc/env_gate.py
# Optional: AT_DEPEND=<pbs job id> (afterok) or a raw PBS spec such as afterany:<id>[:<id>] / afterok:<a>:<b>.
# Refuses to reuse a run_id whose state dir has DONE (runs are immutable); a FAILED / GPU_BUSY run_id may be resubmitted.
# No ngpus request: the cluster's cgroup hook rejects it (docs/REMOTE_STATE.md §2); the job's guard checks the GPU instead.
set -eu
[ $# -ge 9 ] && [ "$8" = "--" ] || { sed -n 2,8p "$0"; exit 2; }
RUN_ID="$1"; HOST="$2"; NCPUS="$3"; MEM="$4"; WALL="$5"; GPU="$6"; STAGE="$7"; shift 8
CMD="$*"
STATE="$HOME/aerotrack_server/jobs/$RUN_ID"
[ -f "$STATE/DONE" ] && { echo "refusing: $RUN_ID already DONE ($STATE)"; exit 3; }
[ -f "$STATE/RUNNING" ] && { echo "refusing: $RUN_ID has a RUNNING sentinel ($STATE)"; exit 3; }
mkdir -p "$STATE"
printf '%s\n' "$CMD" > "$STATE/cmd.txt"     # the job reads the command from here (PBS -v splits on commas)
cd "$HOME/aerotrack_server/source"
DEP=(); if [ -n "${AT_DEPEND:-}" ]; then case "$AT_DEPEND" in after*) DEP=(-W "depend=$AT_DEPEND");; *) DEP=(-W "depend=afterok:$AT_DEPEND");; esac; fi
SEL="select=1:ncpus=$NCPUS:mem=${MEM}gb"
[ "$HOST" != "any" ] && [ -n "$HOST" ] && SEL="$SEL:host=$HOST"
JOB=$(qsub ${DEP[@]+"${DEP[@]}"} -N "at_${RUN_ID:0:12}" -q workq -l "$SEL" -l "walltime=$WALL" \
      -o "$STATE/pbs.out" -v "AT_RUN_ID=$RUN_ID,AT_GPU=$GPU,AT_STAGE=$STAGE,OMP_NUM_THREADS=$(( NCPUS < 4 ? NCPUS : 4 ))" \
      tools/hpc/run_job.pbs 2>&1) || { echo "qsub failed: $JOB"; exit 1; }
echo "{\"t\": \"$(date -Iseconds)\", \"job\": \"$RUN_ID\", \"kind\": \"submitted\", \"pbs_job\": \"$JOB\", \"host\": \"$HOST\", \"ncpus\": $NCPUS, \"mem_gb\": $MEM, \"walltime\": \"$WALL\", \"gpu\": $GPU, \"stage\": $STAGE, \"depend\": \"${AT_DEPEND:-}\", \"commit\": \"$(git rev-parse HEAD)\"}" >> "$STATE/events.jsonl"
echo "$JOB"
