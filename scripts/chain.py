#!/usr/bin/env python
"""Run a fixed list of jobs sequentially and unattended, optionally after another job's sentinel appears.

    venv\\Scripts\\python.exe scripts\\chain.py --spec <jobs.json> --detach

jobs.json: {"name": ..., "state_dir": "results/runs/<...>", "wait_for": "<state dir with DONE/FAILED>" | null,
            "jobs": [{"name": ..., "cmd": [...], "env": {...}, "required": true, "when_exists": "<path>" | null}, ...]}
A job with `when_exists` runs only if that path exists (else it is recorded as skipped); a failed `required` job stops the
chain (FAILED), a failed optional job is recorded and the chain continues. State dir: RUNNING / DONE / FAILED sentinels,
events.jsonl, one log per job. Used to give the GPU to short jobs that need it idle (latency benchmarks) between campaigns.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def main():
    spec_path = Path(sys.argv[sys.argv.index("--spec") + 1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    state = REPO / spec["state_dir"]
    state.mkdir(parents=True, exist_ok=True)

    def event(**kw):
        kw["t"] = now()
        with open(state / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\n")

    if "--detach" in sys.argv:
        if (state / "RUNNING").exists():
            sys.exit(f"already running: {(state / 'RUNNING').read_text(encoding='utf-8').strip()}")
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0
        with open(state / "chain.log", "a", encoding="utf-8") as out:
            p = subprocess.Popen([sys.executable, "-u", __file__, "--spec", str(spec_path.resolve())], cwd=str(REPO), stdout=out,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        print(f"chain '{spec['name']}' detached, pid {p.pid}; state in {state}")
        return
    for s in ("DONE", "FAILED"):
        (state / s).unlink(missing_ok=True)
    (state / "RUNNING").write_text(f"pid {os.getpid()} started {now()}\n", encoding="utf-8")
    (state / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    try:
        if spec.get("wait_for"):
            wd = REPO / spec["wait_for"]
            event(kind="waiting", dir=str(wd))
            while not ((wd / "DONE").exists() or (wd / "FAILED").exists()):
                time.sleep(60)
            event(kind="go", sentinel="DONE" if (wd / "DONE").exists() else "FAILED")
        for job in spec["jobs"]:
            if job.get("when_exists") and not (REPO / job["when_exists"]).exists():
                event(kind="skipped", job=job["name"], missing=job["when_exists"])
                continue
            cmd = [sys.executable if c == "{python}" else c for c in job["cmd"]]   # absolute interpreter (Windows path rules)
            event(kind="start", job=job["name"], cmd=" ".join(cmd))
            t0 = time.time()
            with open(state / f"{job['name']}.log", "a", encoding="utf-8") as f:
                rc = subprocess.call(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, **job.get("env", {})))
            event(kind="end", job=job["name"], rc=rc, minutes=round((time.time() - t0) / 60, 1))
            if rc != 0 and job.get("required", True):
                raise RuntimeError(f"required job {job['name']} failed (rc {rc})")
        (state / "DONE").write_text(f"finished {now()}\n", encoding="utf-8")
    except BaseException as e:  # noqa: BLE001
        (state / "FAILED").write_text(f"{now()} {type(e).__name__}: {e}\n", encoding="utf-8")
        event(kind="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        (state / "RUNNING").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
