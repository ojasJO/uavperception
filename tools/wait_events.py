#!/usr/bin/env python
"""Block until a NEW job sentinel (DONE / FAILED) appears anywhere in the research program, print it, exit.

    venv\\Scripts\\python.exe tools\\wait_events.py            (run as a background task; one notification per event batch)
    venv\\Scripts\\python.exe tools\\wait_events.py --ack       (mark every current sentinel as seen; run after handling)
    venv\\Scripts\\python.exe tools\\wait_events.py --list      (show watched sentinels and whether they are new)

One waiter replaces per-job polling: the agent is woken only when some job finishes or fails. Watched: every
runs/_chains/*/ state dir, the Anti-UAV H4/H3 runner dirs, the aerotrack_v2 runner, the CST fetch dir, and any extra
directory listed in runs/_chains/_watch.json. Seen sentinels (path + mtime) are kept in runs/_chains/_events_seen.json.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEEN = REPO / "runs" / "_chains" / "_events_seen.json"
EXTRA = REPO / "runs" / "_chains" / "_watch.json"
FIXED = ["experiments/anti_uav/runs/_h4_native_scale", "experiments/anti_uav/runs/_h3_optimisation", "runs/aerotrack_v2/_runner",
         "datasets/2_cst-antiuav_part2"]


def watched_dirs():
    dirs = [REPO / d for d in FIXED]
    chains = REPO / "runs" / "_chains"
    if chains.exists():
        dirs += [p for p in chains.iterdir() if p.is_dir()]
    if EXTRA.exists():
        dirs += [REPO / d for d in json.loads(EXTRA.read_text(encoding="utf-8"))]
    return dirs


def current():
    out = {}
    for d in watched_dirs():
        for s in ("DONE", "FAILED"):
            p = d / s
            if p.exists():
                out[str(p.relative_to(REPO))] = p.stat().st_mtime
    return out


def seen():
    return json.loads(SEEN.read_text(encoding="utf-8")) if SEEN.exists() else {}


def main():
    SEEN.parent.mkdir(parents=True, exist_ok=True)
    if "--ack" in sys.argv:
        SEEN.write_text(json.dumps(current(), indent=1), encoding="utf-8")
        print(f"acknowledged {len(current())} sentinels")
        return
    if "--list" in sys.argv:
        s = seen()
        for k, v in sorted(current().items()):
            print(("NEW  " if s.get(k) != v else "seen ") + k)
        return
    while True:
        s = seen()
        new = {k: v for k, v in current().items() if s.get(k) != v}
        if new:
            for k in sorted(new):
                p = REPO / k
                print(f"EVENT {k}: {p.read_text(encoding='utf-8', errors='replace').strip()[:300]}", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
