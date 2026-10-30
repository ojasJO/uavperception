"""Paths, corpus pin and ledger for the AeroTrack v2 series (complete CST corpus; experiments/aerotrack_v2).

Importing this module pins the process -- and every child it starts -- to corpus v2 (b_pipeline/data_unified_v2) BEFORE any
pipeline module is imported, because b_pipeline/corpus.py reads AEROTRACK_CORPUS at import time.

Test-set rule: nothing in this series reads TEST images, labels or predictions except the single, pre-registered final
evaluation of the selected candidate (its own ledger, refused a second time)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CORPUS = "data_unified_v2"
if os.environ.get("AEROTRACK_CORPUS", CORPUS) != CORPUS:
    raise SystemExit(f"the v2 series runs on {CORPUS}; AEROTRACK_CORPUS is set to {os.environ['AEROTRACK_CORPUS']!r}")
os.environ["AEROTRACK_CORPUS"] = CORPUS

REPO = Path(__file__).resolve().parents[3]
SERIES = REPO / "experiments" / "aerotrack_v2"
REPORTS = SERIES / "reports"
CACHE = REPO / "runs" / "aerotrack_v2" / "_cache"            # derived per-image records (git-ignored)
STATE = REPO / "runs" / "aerotrack_v2" / "_runner"           # runner sentinels / events / logs
LEDGER = SERIES / "ledger.jsonl"
TEST_LEDGER = REPORTS / "v2_test_ledger.jsonl"
for p in (REPO / "c_model", REPO / "b_pipeline", REPO / "experiments" / "aerotrack_repair" / "scripts",
          REPO / "experiments" / "aerotrack_generalization" / "scripts", REPO / "experiments" / "anti_uav" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def git():
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def ledger(entry):
    entry = {"timestamp": now(), "git": git(), "command": " ".join(sys.argv), **entry}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
