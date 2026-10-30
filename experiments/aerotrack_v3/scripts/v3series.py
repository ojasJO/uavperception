"""Paths + ledger of the AeroTrack v3 series; corpus pinned to v2 (via v2series, imported first)."""
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[2] / "aerotrack_v2" / "scripts"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))
import v2series as _v2  # noqa: E402  (pins AEROTRACK_CORPUS=data_unified_v2, sets sys.path)

CORPUS, REPO, git, now, write_json_atomic = _v2.CORPUS, _v2.REPO, _v2.git, _v2.now, _v2.write_json_atomic
SERIES = REPO / "experiments" / "aerotrack_v3"
REPORTS = SERIES / "reports"
CACHE = REPO / "results/runs" / "aerotrack_v3" / "_cache"
STATE = REPO / "results/runs" / "aerotrack_v3" / "_runner"
LEDGER = SERIES / "ledger.jsonl"
TEST_LEDGER = REPORTS / "v3_test_ledger.jsonl"


def ledger(entry):
    import sys as _s
    entry = {"timestamp": now(), "git": git(), "command": " ".join(_s.argv), **entry}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
