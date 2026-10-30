#!/usr/bin/env python
"""Block until a runner's events.jsonl contains a line with ALL --contains substrings and NONE of the --exclude ones, then sleep
--after seconds; also returns when the runner's state dir holds DONE / FAILED (nothing left to protect). Exit 0 either way.

    {python} tools/wait_for_event.py --file runs/aerotrack_v2/_runner/events.jsonl --contains '"kind": "start"' --contains train_v2_
                                     --exclude _smoke --after 240

Why: a GPU training sizes its data-loader workers from the free RAM / commit at its start (c_model/train.py planner); a CPU job
running at that moment permanently costs the training workers (the v2 smoke got 2 instead of 4: ~2x slower epochs). Chains of
CPU jobs therefore start just AFTER a training has planned its workers. Polls a file every --poll seconds (a detached process,
not the agent).
"""
import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--contains", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--after", type=int, default=0)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--timeout-h", type=float, default=72.0)
    ap.add_argument("--also-done", action="append", default=[], help="also return when this state dir holds DONE / FAILED")
    a = ap.parse_args()
    f = Path(a.file)
    t0 = time.time()
    while time.time() - t0 < a.timeout_h * 3600:
        lines = f.read_text(encoding="utf-8").splitlines() if f.exists() else []
        if any(all(c in ln for c in a.contains) and not any(x in ln for x in a.exclude) for ln in lines):
            print(f"matched in {f}; sleeping {a.after} s", flush=True)
            time.sleep(a.after)
            return
        for d in [f.parent] + [Path(x) for x in a.also_done]:
            if (d / "DONE").exists() or (d / "FAILED").exists():
                print(f"{d} finished without a match; proceeding", flush=True)
                return
        time.sleep(a.poll)
    print("timeout; proceeding", flush=True)


if __name__ == "__main__":
    sys.exit(main())
