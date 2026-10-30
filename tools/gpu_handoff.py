#!/usr/bin/env python
"""GPU hand-off after the corpus-v2 gate (run as a job of tools/chains/gpu_handoff.json, which waits for runs/_chains/corpus_v2).

The user's 2026-10-05 instruction: once the corpus-v2 verification gate is satisfied, proceed with the complete-CST corpus and the
pre-registered v2 campaign (v2 has GPU priority over the Anti-UAV H3 remainder; reasons in progress/RESEARCH_PROGRAM_PROGRESS.md).
Steps, each recorded in runs/_chains/gpu_handoff/handoff.json:
  1. experiments/cst_completion/reports/corpus_v2.json must say verdict OK (else exit 1: nothing is launched, H3 continues).
  2. H3 yields the GPU at its next arm boundary: write experiments/anti_uav/runs/_h3_optimisation/PAUSE_AFTER_ARM (reason inside).
  3. run_v2.py --detach --wait-for experiments/anti_uav/runs/_h3_optimisation   (starts when H3 writes PAUSED / DONE / FAILED)
  4. chains post_v2_causal_caches (after post_v2) and h3_resume (after the caches: consumes the flag, resumes H3), detached
  5. run_c2.py --detach --wait-for runs/_chains/h3_resume                       (Track C causal-window study, last in the queue)
Every launched runner checks its own registration / corpus preconditions again; this script only sequences them.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = REPO / "runs" / "_chains" / "gpu_handoff" / "handoff.json"


def run(cmd):
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    return {"cmd": " ".join(str(c) for c in cmd), "rc": r.returncode, "out": (r.stdout + r.stderr).strip()[-600:]}


def main():
    rec = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": []}

    def save():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")

    rep = REPO / "experiments" / "cst_completion" / "reports" / "corpus_v2.json"
    verdict = json.loads(rep.read_text(encoding="utf-8")).get("verdict") if rep.exists() else None
    rec["corpus_v2_verdict"] = verdict
    h3 = REPO / "experiments" / "anti_uav" / "runs" / "_h3_optimisation"
    if verdict != "OK":
        # an anticipatory pause flag (written after the archive verification passed, before the build finished) is withdrawn
        # (renamed, kept) and a paused H3 is resumed, so the GPU never waits for a campaign that cannot start
        flag = h3 / "PAUSE_AFTER_ARM"
        if flag.exists():
            flag.rename(h3 / f"PAUSE_AFTER_ARM.withdrawn_{time.strftime('%Y%m%dT%H%M%S')}")
            rec["steps"].append({"pause_flag": "withdrawn"})
        if (h3 / "PAUSED").exists() and not (h3 / "RUNNING").exists():
            rec["steps"].append(run([PY, "experiments/anti_uav/scripts/run_h3.py", "--detach"]))
        rec["result"] = "corpus v2 not verified: v2 not launched; H3 keeps / regains the GPU"
        save()
        sys.exit(1)
    if (h3 / "RUNNING").exists():
        (h3 / "PAUSE_AFTER_ARM").write_text(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} corpus v2 verified (verdict OK): the registered aerotrack_v2 campaign takes the GPU "
            "(user instruction 2026-10-05); H3 resumes via tools/chains/h3_resume.json after v2, post-v2 and the causal caches\n",
            encoding="utf-8")
        rec["steps"].append({"pause_flag": "written", "h3_state": "RUNNING"})
        wait_dir = "experiments/anti_uav/runs/_h3_optimisation"
    else:
        rec["steps"].append({"pause_flag": "not needed", "h3_state": [p.name for p in h3.iterdir() if p.name.isupper()]})
        wait_dir = None
    rec["steps"].append(run([PY, "experiments/aerotrack_v2/scripts/run_v2.py", "--detach"] + (["--wait-for", wait_dir] if wait_dir else [])))
    for spec in ("post_v2_causal_caches", "h3_resume"):
        rec["steps"].append(run([PY, "tools/chain.py", "--spec", f"tools/chains/{spec}.json", "--detach"]))
    rec["steps"].append(run([PY, "experiments/aerotrack_temporal/scripts/run_c2.py", "--detach", "--wait-for", "runs/_chains/h3_resume"]))
    rec["result"] = "launched" if all(s.get("rc", 0) == 0 for s in rec["steps"]) else "partially launched (see rc)"
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    print(json.dumps(rec, indent=1))
    sys.exit(0 if rec["result"] == "launched" else 1)


if __name__ == "__main__":
    main()
