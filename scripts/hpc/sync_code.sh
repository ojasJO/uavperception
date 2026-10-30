#!/usr/bin/env bash
# Bring the HPC checkout (~/aerotrack_server/source) to the OMEN's committed HEAD of the current branch.
# Incremental git bundle (only commits the HPC lacks) -> scp -> fetch -> fast-forward.
# A tracked file modified on the HPC (e.g. a Track A runner appending to its events.jsonl) is allowed ONLY if its bytes equal
# the version in the incoming commit (i.e. the OMEN already committed exactly what the HPC wrote, via pull_results.py);
# anything else refuses. Such a file is restored to HEAD and the fast-forward writes the identical bytes back. Do not run
# while a job is still appending to such a file. An UNTRACKED HPC file that the incoming commit adds (a result pulled
# and committed on the OMEN) is removed only if byte-identical, so the fast-forward writes it back; otherwise it refuses.
#   scripts/hpc/sync_code.sh
set -eu
cd "$(dirname "$0")/../.."
BR=$(git rev-parse --abbrev-ref HEAD)
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "refusing: commit local changes first"; exit 2; }
REMOTE_HEAD=$(ssh -o BatchMode=yes amritahpc "cd ~/aerotrack_server/source && git rev-parse HEAD")
LOCAL_HEAD=$(git rev-parse HEAD)
[ "$REMOTE_HEAD" = "$LOCAL_HEAD" ] && { echo "HPC already at $LOCAL_HEAD"; exit 0; }
git merge-base --is-ancestor "$REMOTE_HEAD" "$LOCAL_HEAD" || { echo "refusing: HPC head $REMOTE_HEAD is not an ancestor of $LOCAL_HEAD"; exit 3; }
B="results/runs/_hpc_sync/repo/inc_${LOCAL_HEAD:0:12}.bundle"
git bundle create "$B" "$REMOTE_HEAD..$BR" >/dev/null 2>&1
scp -q "$B" amritahpc:aerotrack_server/source_bundles/
ssh -o BatchMode=yes amritahpc "cd ~/aerotrack_server/source && \
  git fetch -q ../source_bundles/$(basename "$B") $BR:refs/remotes/omen/$BR && \
  for f in \$(git diff --name-only); do \
    want=\$(git rev-parse omen/$BR:\"\$f\" 2>/dev/null || echo none); have=\$(git hash-object \"\$f\"); \
    if [ \"\$want\" = \"\$have\" ]; then git checkout -q HEAD -- \"\$f\"; \
    else echo \"refusing: \$f modified on the HPC and differs from the incoming commit\"; exit 4; fi; \
  done && \
  for f in \$(git diff --name-only --diff-filter=A HEAD omen/$BR); do \
    [ -f \"\$f\" ] || continue; \
    if [ \"\$(git rev-parse omen/$BR:\"\$f\")\" = \"\$(git hash-object \"\$f\")\" ]; then rm -f \"\$f\"; \
    else echo \"refusing: untracked \$f on the HPC differs from the version the OMEN committed\"; exit 5; fi; \
  done && git merge -q --ff-only omen/$BR && git log --oneline -1 && git status --short --untracked-files=no | wc -l"
