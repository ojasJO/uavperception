"""Path setup for the AeroTrack repair scripts: project modules (c_model, b_pipeline) + the shared repro utilities
from the Anti-UAV experiment package (git/env snapshot, hashing) so provenance is recorded the same way everywhere."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REPAIR_DIR = REPO_ROOT / "experiments" / "aerotrack_repair"
REPORTS_DIR = REPAIR_DIR / "reports"
for p in (REPO_ROOT / "c_model", REPO_ROOT / "b_pipeline", REPO_ROOT / "experiments" / "anti_uav" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
