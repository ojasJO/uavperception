"""Paths + ledger for the AeroTrack generalization investigation (a separate series: the completed repair grid in
experiments/aerotrack_repair and results/runs/repair_abl_* is read, never modified).

Test-set rule for this series: no test image, label or prediction is read. The only test information any script uses
is split METADATA (sequence names), for leakage/composition checks."""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SERIES = REPO / "experiments" / "aerotrack_generalization"
REPORTS = SERIES / "reports"
CACHE = REPO / "results/runs" / "aerotrack_generalization" / "_cache"   # derived bulk data (git-ignored)
def ledger_path():
    """Smoke tests of the machinery (GEN_SMOKE set) log to a separate file; decided at call time, not import time."""
    import os
    return SERIES / ("ledger_smoke.jsonl" if os.environ.get("GEN_SMOKE") else "ledger.jsonl")


LEDGER = SERIES / "ledger.jsonl"
for p in (REPO / "src/model", REPO / "src/pipeline", REPO / "experiments" / "aerotrack_repair" / "scripts",
          REPO / "experiments" / "anti_uav" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def git():
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def write_report(name, obj):
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / name
    p.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    return p


def ledger(entry):
    """Append one record (an experiment or a diagnostic) to the series ledger."""
    entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git(), "command": " ".join(sys.argv), **entry}
    with open(ledger_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
