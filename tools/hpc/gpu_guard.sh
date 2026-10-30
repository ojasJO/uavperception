#!/usr/bin/env bash
# Refuse to start a GPU job on a node whose V100 is used by ANOTHER user (PBS cannot allocate GPUs on this cluster, so it gives
# no exclusivity: D-2026-10-06-03). Our own processes may share the GPU (D-2026-10-06-09: e.g. a loader-bound training beside a
# tracking evaluation; timing changes, results do not). Exit 0 = usable; exit 75 = foreign use (EX_TEMPFAIL). One JSON line.
set -u
me=$(id -un)
foreign=0; foreign_mib=0; own=0
while IFS=, read -r pid mib; do
  pid=$(echo "$pid" | tr -d ' '); mib=$(echo "$mib" | tr -d ' MiB')
  [ -z "$pid" ] && continue
  owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ "$owner" = "$me" ]; then own=$((own + 1)); else foreign=$((foreign + 1)); foreign_mib=$((foreign_mib + ${mib:-0})); fi
done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "{\"gpu_guard\": {\"host\": \"$(hostname)\", \"gpu\": \"$name\", \"mem_used_mib\": ${used:-null}, \"util\": ${util:-null}, \"own_procs\": $own, \"foreign_procs\": $foreign, \"foreign_mib\": $foreign_mib}}"
# a process we cannot attribute (e.g. in another container) counts as foreign; memory without any process = foreign too
if [ "$foreign" -gt 0 ] || { [ "$own" -eq 0 ] && [ "${used:-0}" -gt 500 ]; }; then exit 75; fi
exit 0
