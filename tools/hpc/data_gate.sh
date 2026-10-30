#!/usr/bin/env bash
# [HPC job] M3/M4 data gate: env_gate --part data on the NFS tree; on PASS write shared_data/root/DATA_VERSION (staging key).
set -eu
REP="$HOME/aerotrack_server/reports/hpc/m4_env_gate_data.json"
python tools/hpc/env_gate.py --part data --out "$REP"
python - "$REP" > "$AT_NFS_ROOT/DATA_VERSION.tmp" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); assert r["ok"]
m = next(c for c in r["checks"] if c["check"].startswith("D1"))["detail"]["manifest"]
print(f"corpus_v2_{m[:12]}_cst_full_antiuav_hpcdecode")
PY
mv "$AT_NFS_ROOT/DATA_VERSION.tmp" "$AT_NFS_ROOT/DATA_VERSION"; cat "$AT_NFS_ROOT/DATA_VERSION"
