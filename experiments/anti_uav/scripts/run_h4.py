#!/usr/bin/env python
"""Run the pre-registered H4 experiment (preregistration/H4_native_scale.json) unattended, then analyse it.

    venv\\Scripts\\python.exe experiments\\anti_uav\\scripts\\run_h4.py --detach     (returns at once; survives the shell)
    venv\\Scripts\\python.exe experiments\\anti_uav\\scripts\\run_h4.py --status

Four native-scale evaluations in the registered order (normal val, normal test, dfsc val, dfsc test), each a separate
`evaluate.py --scale 640 512` process (TEST with --final-eval, so each lands in the test ledger flagged phase B). An
evaluation whose full-protocol metrics.json already exists is skipped (re-running the runner resumes); a failed one is
re-run once from scratch, as registered. State lives in runs/_h4_native_scale/: RUNNING / DONE / FAILED sentinels,
events.jsonl, one log per evaluation. Finally `analyze_h4.py` applies the frozen rule.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent
STATE = EXP / "runs" / "_h4_native_scale"
PY = sys.executable
ORDER = [("normal_ir", "val"), ("normal_ir", "test"), ("dfsc_ir", "val"), ("dfsc_ir", "test")]
VARIANT_DIR = "phaseB_test_scale_640x512"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def event(**kw):
    kw["t"] = now()
    with open(STATE / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def done(model, split):
    m = EXP / "runs" / model / VARIANT_DIR / split / "metrics.json"
    return m.exists() and not json.loads(m.read_text(encoding="utf-8"))["meta"].get("partial")


def run_one(model, split, attempt):
    cmd = [PY, "-u", str(HERE / "evaluate.py"), "--checkpoint", str(EXP / "runs" / model / "weights" / "last.pt"),
           "--split", split, "--scale", "640", "512"] + (["--final-eval"] if split == "test" else [])
    log = STATE / f"{model}_{split}_attempt{attempt}.log"
    event(kind="start", model=model, split=split, attempt=attempt, cmd=" ".join(cmd), log=log.name)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    event(kind="end", model=model, split=split, attempt=attempt, rc=rc, minutes=round((time.time() - t0) / 60, 1))
    return rc == 0 and done(model, split)


def work():
    STATE.mkdir(parents=True, exist_ok=True)
    for s in ("DONE", "FAILED"):
        (STATE / s).unlink(missing_ok=True)
    (STATE / "RUNNING").write_text(f"pid {os.getpid()} started {now()}\n", encoding="utf-8")
    try:
        for model, split in ORDER:
            if done(model, split):
                event(kind="skip_complete", model=model, split=split)
                continue
            if not run_one(model, split, 1) and not run_one(model, split, 2):
                raise RuntimeError(f"{model}/{split} failed twice")
        rc = subprocess.call([PY, str(HERE / "analyze_h4.py")], cwd=str(REPO))
        event(kind="analysis", rc=rc)
        (STATE / "DONE").write_text(f"finished {now()}\n", encoding="utf-8")
    except BaseException as e:  # noqa: BLE001 — every exit leaves a sentinel
        (STATE / "FAILED").write_text(f"{now()} {type(e).__name__}: {e}\n", encoding="utf-8")
        event(kind="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        (STATE / "RUNNING").unlink(missing_ok=True)


def main():
    if "--status" in sys.argv:
        for s in ("RUNNING", "DONE", "FAILED"):
            if (STATE / s).exists():
                print(s, (STATE / s).read_text(encoding="utf-8").strip())
        for model, split in ORDER:
            print(f"{model:10s} {split:5s} {'complete' if done(model, split) else 'pending'}")
        return
    if "--detach" in sys.argv:
        STATE.mkdir(parents=True, exist_ok=True)
        if (STATE / "RUNNING").exists():
            sys.exit(f"already running: {(STATE / 'RUNNING').read_text(encoding='utf-8').strip()}")
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0       # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        with open(STATE / "runner.log", "a", encoding="utf-8") as out:
            p = subprocess.Popen([PY, "-u", __file__], cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        print(f"H4 runner detached, pid {p.pid}; state in {STATE}")
        return
    work()


if __name__ == "__main__":
    main()
