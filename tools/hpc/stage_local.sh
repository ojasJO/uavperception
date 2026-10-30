#!/usr/bin/env bash
# Copy the training data tree from NFS ($AT_NFS_ROOT) to this node's /local ($AT_LOCAL_ROOT), once per node and data version.
# Idempotent and safe for concurrent jobs on one node (flock). The copy is verified by file count + total bytes against the
# NFS tree; a STAGED_<version> marker records success. NFS stays the source of truth; /local is a cache that may be deleted.
#   tools/hpc/stage_local.sh   (after: source tools/hpc/job_env.sh)
set -eu
VERSION_FILE="$AT_NFS_ROOT/DATA_VERSION"
[ -f "$VERSION_FILE" ] || { echo "stage: $VERSION_FILE missing (data tree not finalised)"; exit 3; }
VER=$(cat "$VERSION_FILE")
mkdir -p "$AT_LOCAL_ROOT"
exec 9> "$AT_LOCAL_ROOT/.stage.lock"
flock 9
if [ -f "$AT_LOCAL_ROOT/STAGED_$VER" ]; then echo "stage: $(hostname) already has $VER"; exit 0; fi
t0=$(date +%s)
echo "stage: copying $VER to $(hostname):$AT_LOCAL_ROOT"
rsync -a --delete --exclude '.stage.lock' --exclude 'STAGED_*' "$AT_NFS_ROOT/" "$AT_LOCAL_ROOT/"
n_src=$(find -L "$AT_NFS_ROOT" -type f ! -name '.stage.lock' ! -name 'STAGED_*' | wc -l)
n_dst=$(find "$AT_LOCAL_ROOT" -type f ! -name '.stage.lock' ! -name 'STAGED_*' | wc -l)
b_src=$(du -sbL "$AT_NFS_ROOT" | cut -f1); b_dst=$(du -sb --exclude='.stage.lock' "$AT_LOCAL_ROOT" | cut -f1)
echo "stage: files $n_src -> $n_dst, bytes $b_src -> $b_dst, $(( $(date +%s) - t0 )) s"
[ "$n_src" = "$n_dst" ] || { echo "stage: FILE COUNT MISMATCH"; exit 4; }
echo "{\"version\": \"$VER\", \"files\": $n_dst, \"bytes\": $b_dst, \"host\": \"$(hostname)\", \"t\": \"$(date -Iseconds)\"}" > "$AT_LOCAL_ROOT/STAGED_$VER"
