"""Make `anti_uav_repro` importable from the scripts directory without installing the package,
and force UTF-8 console output (Windows defaults to cp1252, which cannot print report symbols)."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - not a TTY / already configured
        pass
