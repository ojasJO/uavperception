#!/usr/bin/env bash
# Resumable, verified push of large files from the OMEN to the HPC (ssh host alias `amritahpc`, key auth only;
# no password is ever used or stored by this script).
#
#   tools/hpc/push_files.sh <state_dir> <remote_dir> <local_file>...
#
# Per file: compare remote size with local size, append only the missing tail (so an interrupted transfer resumes
# where it stopped), then compare sha256 on both sides. A mismatch renames the remote copy to *.bad_<ts> (kept, not
# deleted) and the file is pushed again once. Writes <state_dir>/events.jsonl, <state_dir>/sha256.tsv and a DONE or
# FAILED sentinel. Safe to re-run: verified files are skipped by checksum.
set -u
STATE="$1"; RDIR="$2"; shift 2
mkdir -p "$STATE"; rm -f "$STATE/DONE" "$STATE/FAILED"; echo "pid $$ started $(date -Iseconds)" > "$STATE/RUNNING"
ev() { echo "{\"t\": \"$(date -Iseconds)\", $1}" >> "$STATE/events.jsonl"; }
ssh -o BatchMode=yes amritahpc "mkdir -p '$RDIR'" || { ev "\"kind\": \"failed\", \"error\": \"mkdir\""; touch "$STATE/FAILED"; exit 1; }
fail=0
for f in "$@"; do
  name=$(basename "$f"); lsize=$(stat -c %s "$f")
  lsha=$(grep -P "^[0-9a-f]{64}\t$name\t" "$STATE/sha256_local.tsv" 2>/dev/null | cut -f1)
  if [ -z "$lsha" ]; then lsha=$(sha256sum "$f" | cut -d' ' -f1); printf '%s\t%s\t%s\n' "$lsha" "$name" "$lsize" >> "$STATE/sha256_local.tsv"; fi
  for attempt in 1 2; do
    rsize=$(ssh -o BatchMode=yes amritahpc "stat -c %s '$RDIR/$name' 2>/dev/null || echo 0")
    if [ "$rsize" -gt "$lsize" ]; then ssh amritahpc "mv '$RDIR/$name' '$RDIR/$name.bad_$(date +%s)'"; rsize=0; fi
    if [ "$rsize" -lt "$lsize" ]; then
      ev "\"kind\": \"push\", \"file\": \"$name\", \"from_byte\": $rsize, \"size\": $lsize, \"attempt\": $attempt"
      t0=$(date +%s)
      tail -c +$((rsize + 1)) "$f" | ssh -o BatchMode=yes amritahpc "cat >> '$RDIR/$name'"
      ev "\"kind\": \"pushed\", \"file\": \"$name\", \"seconds\": $(( $(date +%s) - t0 ))"
    fi
    rsha=$(ssh -o BatchMode=yes amritahpc "sha256sum '$RDIR/$name' | cut -d' ' -f1")
    if [ "$rsha" = "$lsha" ]; then
      printf '%s\t%s\t%s\n' "$lsha" "$name" "$lsize" >> "$STATE/sha256.tsv"
      ev "\"kind\": \"verified\", \"file\": \"$name\", \"sha256\": \"$lsha\""; break
    fi
    ev "\"kind\": \"mismatch\", \"file\": \"$name\", \"local\": \"$lsha\", \"remote\": \"$rsha\", \"attempt\": $attempt"
    ssh amritahpc "mv '$RDIR/$name' '$RDIR/$name.bad_$(date +%s)'"
    [ $attempt -eq 2 ] && fail=1
  done
done
rm -f "$STATE/RUNNING"
if [ $fail -eq 0 ]; then ev "\"kind\": \"done\""; touch "$STATE/DONE"; else ev "\"kind\": \"failed\""; touch "$STATE/FAILED"; fi
