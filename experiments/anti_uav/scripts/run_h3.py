#!/usr/bin/env python
"""Run the pre-registered H3 experiment (preregistration/H3_optimisation_regime.json) unattended, then analyse it.

    venv\\Scripts\\python.exe experiments\\anti_uav\\scripts\\run_h3.py --detach [--wait-for <state dir>]
    venv\\Scripts\\python.exe experiments\\anti_uav\\scripts\\run_h3.py --status

Order (as registered): train phaseB_normal_ir_lr0.02_accum8 and phaseB_dfsc_ir_lr0.02_accum8 (an interrupted training resumes
from its last.pt), evaluate both on VAL, then both on TEST once (--final-eval, ledger-recorded), then analyze_h3.py. A finished
training (final checkpoint hash in run_meta.json) or evaluation (metrics.json present) is never redone. Refuses to start unless the
registration is committed and unmodified. State: results/runs/_h3_optimisation/ (RUNNING / DONE / FAILED, events.jsonl, logs).

A training that crashes before its first checkpoint is moved (never deleted) to results/runs/_h3_optimisation/crashed_<run>_<time>/ and
retried from scratch. Writing results/runs/_h3_optimisation/PAUSE_AFTER_ARM (text = reason) makes the runner stop at the next arm boundary
and write PAUSED instead of DONE (scheduling only: the registered order and every rule are unchanged; a relaunch resumes).
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
STATE = EXP / "runs" / "_h3_optimisation"
PRE = EXP / "preregistration" / "H3_optimisation_regime.json"
PAUSE_FLAG = STATE / "PAUSE_AFTER_ARM"   # written by the operator to hand the GPU to another campaign at the next arm boundary
PY = sys.executable
ARMS = [("phaseB_normal_ir_lr0.02_accum8", EXP / "configs" / "phase_b" / "normal_ir_lr0.02_accum8.yaml"),
        ("phaseB_dfsc_ir_lr0.02_accum8", EXP / "configs" / "phase_b" / "dfsc_ir_lr0.02_accum8.yaml")]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def event(**kw):
    kw["t"] = now()
    with open(STATE / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def sh(cmd, name):
    event(kind="start", job=name, cmd=" ".join(cmd))
    t0 = time.time()
    with open(STATE / f"{name}.log", "a", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    event(kind="end", job=name, rc=rc, minutes=round((time.time() - t0) / 60, 1))
    return rc


def trained(run):
    m = EXP / "runs" / run / "run_meta.json"
    return m.exists() and bool(json.loads(m.read_text(encoding="utf-8")).get("final_checkpoint_sha256")) and \
        (EXP / "runs" / run / "weights" / "last.pt").exists()


def evaluated(run, split):
    m = EXP / "runs" / run / f"eval_{split}_IR_last" / "metrics.json"
    return m.exists() and not json.loads(m.read_text(encoding="utf-8"))["meta"].get("partial")


def set_aside_crashed(run):
    """A run dir without last.pt is a training that crashed inside epoch 1: nothing to resume from. It is moved (never deleted) into
    the state dir so the retry starts clean and the failed attempt's logs stay on record."""
    rd = EXP / "runs" / run
    if rd.exists() and any(rd.iterdir()) and not (rd / "weights" / "last.pt").exists():
        dst = STATE / f"crashed_{run}_{time.strftime('%Y%m%dT%H%M%S')}"
        rd.rename(dst)
        event(kind="crashed_attempt_set_aside", run=run, moved_to=str(dst))


def train(run, cfg):
    if trained(run):
        event(kind="skip_trained", run=run)
        return
    last = EXP / "runs" / run / "weights" / "last.pt"
    for attempt in (1, 2, 3):
        set_aside_crashed(run)
        cmd = [PY, "-u", str(HERE / "train.py"), "--config", str(cfg)] + (["--resume", str(last)] if last.exists() else [])
        sh(cmd, f"train_{run}_attempt{attempt}")
        if trained(run):
            return
    raise RuntimeError(f"training {run} did not finish")


def evaluate(run, split):
    if evaluated(run, split):
        event(kind="skip_evaluated", run=run, split=split)
        return
    cmd = [PY, "-u", str(HERE / "evaluate.py"), "--checkpoint", str(EXP / "runs" / run / "weights" / "last.pt"), "--split", split]
    cmd += ["--final-eval"] if split == "test" else []
    for attempt in (1, 2):
        sh(cmd, f"eval_{run}_{split}_attempt{attempt}")
        if evaluated(run, split):
            return
    raise RuntimeError(f"evaluation {run}/{split} failed twice")


def preconditions():
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PRE)], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(PRE)], cwd=REPO).returncode == 0
    if not (tracked and clean):
        raise SystemExit("preregistration/H3_optimisation_regime.json must be committed and unmodified before H3 starts")


def work(wait_for):
    STATE.mkdir(parents=True, exist_ok=True)
    for s in ("DONE", "FAILED", "PAUSED"):
        (STATE / s).unlink(missing_ok=True)
    (STATE / "RUNNING").write_text(f"pid {os.getpid()} started {now()}\n", encoding="utf-8")
    try:
        preconditions()
        if wait_for:
            wd = Path(wait_for)
            event(kind="waiting_for_gpu_job", dir=str(wd))
            while not ((wd / "DONE").exists() or (wd / "FAILED").exists()):
                time.sleep(60)
        for run, cfg in ARMS:
            train(run, cfg)
            if PAUSE_FLAG.exists():
                event(kind="paused_at_arm_boundary", after=run, reason=PAUSE_FLAG.read_text(encoding="utf-8").strip())
                (STATE / "PAUSED").write_text(f"{now()} after {run}\n", encoding="utf-8")
                return
        for split in ("val", "test"):
            for run, _ in ARMS:
                evaluate(run, split)
        rc = sh([PY, "-u", str(HERE / "analyze_h3.py")], "analyze_h3")
        event(kind="analysis", rc=rc)
        (STATE / "DONE").write_text(f"finished {now()}\n", encoding="utf-8")
    except BaseException as e:  # noqa: BLE001
        (STATE / "FAILED").write_text(f"{now()} {type(e).__name__}: {e}\n", encoding="utf-8")
        event(kind="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        (STATE / "RUNNING").unlink(missing_ok=True)


def main():
    wait_for = sys.argv[sys.argv.index("--wait-for") + 1] if "--wait-for" in sys.argv else None
    if "--status" in sys.argv:
        for s in ("RUNNING", "DONE", "FAILED", "PAUSED", "PAUSE_AFTER_ARM"):
            if (STATE / s).exists():
                print(s, (STATE / s).read_text(encoding="utf-8").strip())
        for run, _ in ARMS:
            print(f"{run:34s} trained {trained(run)}  val {evaluated(run, 'val')}  test {evaluated(run, 'test')}")
        return
    if "--detach" in sys.argv:
        STATE.mkdir(parents=True, exist_ok=True)
        if (STATE / "RUNNING").exists():
            sys.exit(f"already running: {(STATE / 'RUNNING').read_text(encoding='utf-8').strip()}")
        preconditions()
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0
        args = [PY, "-u", __file__] + (["--wait-for", wait_for] if wait_for else [])
        with open(STATE / "runner.log", "a", encoding="utf-8") as out:
            p = subprocess.Popen(args, cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 creationflags=flags, close_fds=True)
        print(f"H3 runner detached, pid {p.pid}; state in {STATE}")
        return
    work(wait_for)


if __name__ == "__main__":
    main()
