#!/usr/bin/env python
"""Run the registered C2 causal-window study (preregistration/C2_causal_windows.json) unattended on the GPU, then analyse it.

    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\run_c2.py --detach [--wait-for <state dir>]
    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\run_c2.py --status

Trains / evaluates with the v2 series' own scripts (experiments/aerotrack_v2/scripts/train_v2.py, eval_v2.py; corpus v2, the
registered recipe and evaluation), so every C2 run is a `v2_<config>_s<seed>_e20` run and shares the VAL frame set and cache
with the v2 campaign. Order (as registered): stage 1 = causal_g1 / g2 / g8 at seed 0 (+ v2_stock_s0 and v2_causal_g4_s0 if the
v2 campaign did not produce them) -> futility (seed-0 CST delta vs stock <= 0 drops a window) -> stage 2 = seed 1 of the
surviving windows (+ stock_s1, and sym_g4_s1 when W3 needs it) -> scripts/analyze_c2.py. Finished trainings / evaluations are
never redone. Refuses to start unless the registration is committed and unmodified and corpus v2 is verified.
State: runs/aerotrack_temporal/_c2_runner/ (RUNNING / DONE / FAILED / PAUSED-compatible waiting, events.jsonl, logs).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V2 = REPO / "experiments" / "aerotrack_v2" / "scripts"
STATE = REPO / "runs" / "aerotrack_temporal" / "_c2_runner"
PRE = REPO / "experiments" / "aerotrack_temporal" / "preregistration" / "C2_causal_windows.json"
CORPUS_REPORT = REPO / "experiments" / "cst_completion" / "reports" / "corpus_v2.json"
PY = sys.executable
NEW = ("causal_g1", "causal_g2", "causal_g8")
WINDOWS = ("causal_g1", "causal_g2", "causal_g4", "causal_g8")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def event(**kw):
    kw["t"] = now()
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(kw, default=str) + "\n")
    print(json.dumps(kw, default=str), flush=True)


def sh(cmd, name):
    event(kind="start", job=name, cmd=" ".join(cmd))
    t0 = time.time()
    with open(STATE / f"{name}.log", "a", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, AEROTRACK_CORPUS="data_unified_v2"))
    event(kind="end", job=name, rc=rc, minutes=round((time.time() - t0) / 60, 1))
    return rc


def _v2():
    sys.path.insert(0, str(V2))
    import run_v2 as R          # trained() / evaluated() / cst(): the v2 runner's own completion tests
    return R


def train(config, seed):
    R = _v2()
    name = f"v2_{config}_s{seed}_e20"
    if R.trained(name):
        event(kind="skip_trained", run=name)
        return name
    base = [PY, "-u", str(V2 / "train_v2.py"), "--config", config, "--seed", str(seed)]
    exists = (REPO / "runs" / name / "run_meta.json").exists()
    rc = sh(base + (["--resume"] if exists else []), f"train_{name}")
    if rc != 0 and not R.trained(name):
        event(kind="retry_resume", run=name)
        rc = sh(base + ["--resume"], f"train_{name}_retry")
    if not R.trained(name):
        raise RuntimeError(f"training {name} did not finish (rc {rc})")
    return name


def evaluate(name):
    R = _v2()
    if R.evaluated(name):
        event(kind="skip_evaluated", run=name)
        return
    cmd = [PY, "-u", str(V2 / "eval_v2.py"), "--run", name]
    if sh(cmd, f"eval_{name}") != 0 and sh(cmd, f"eval_{name}_retry") != 0:
        raise RuntimeError(f"evaluation of {name} failed twice")


def ensure(config, seed):
    evaluate(train(config, seed))
    return f"v2_{config}_s{seed}_e20"


def preconditions():
    rep = json.loads(CORPUS_REPORT.read_text(encoding="utf-8")) if CORPUS_REPORT.exists() else {}
    if rep.get("verdict") != "OK":
        raise SystemExit(f"corpus v2 not verified ({CORPUS_REPORT.name}: {rep.get('verdict')!r})")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PRE)], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(PRE)], cwd=REPO).returncode == 0
    if not (tracked and clean):
        raise SystemExit("C2_causal_windows.json must be committed and unmodified before C2 starts")
    sys.path.insert(0, str(V2))
    import v2lib
    missing = [c for c in WINDOWS if c not in v2lib.CONFIGS]
    if missing:
        raise SystemExit(f"v2lib.CONFIGS lacks {missing}")


def work(wait_for):
    STATE.mkdir(parents=True, exist_ok=True)
    for s in ("DONE", "FAILED"):
        (STATE / s).unlink(missing_ok=True)
    (STATE / "RUNNING").write_text(f"pid {os.getpid()} started {now()}\n", encoding="utf-8")
    try:
        preconditions()
        if wait_for:
            wd = Path(wait_for)
            event(kind="waiting_for_gpu_job", dir=str(wd))
            free = ("DONE", "FAILED", "PAUSED")
            while not any((wd / x).exists() for x in free):
                time.sleep(60)
            event(kind="gpu_free", sentinel=next(x for x in free if (wd / x).exists()))
        # C2 shares v2's stock / causal_g4 runs and must follow the v2 campaign, never replace it: if v2 has not reached its
        # registered analysis (decisions file), stop here without GPU work (the queue was released by an upstream failure)
        if not (REPO / "experiments" / "aerotrack_v2" / "reports" / "v2_decisions.json").exists():
            raise SystemExit("v2 campaign has no decisions yet (upstream failure released the queue): C2 not started")
        R = _v2()
        stock0 = ensure("stock", 0)
        for c in WINDOWS:
            ensure(c, 0)
        survivors = [c for c in WINDOWS if R.cst(f"v2_{c}_s0_e20") - R.cst(stock0) > 0]
        event(kind="futility", survivors=survivors, seed0_delta={c: R.cst(f"v2_{c}_s0_e20") - R.cst(stock0) for c in WINDOWS})
        if survivors:
            stock1 = ensure("stock", 1)
            for c in survivors:
                ensure(c, 1)
            # W3 compares the best W1-superior window with sym_g4 at both seeds: train what it needs only if some window passes
            # W1's point conditions (1) seed-mean delta > margin and (2) every seed positive (necessary for superiority)
            margin = json.loads(PRE.read_text(encoding="utf-8")).get("margin", 0.02)
            d = {c: [R.cst(f"v2_{c}_s0_e20") - R.cst(stock0), R.cst(f"v2_{c}_s1_e20") - R.cst(stock1)] for c in survivors}
            if any(sum(v) / 2 > margin and min(v) > 0 for v in d.values()):
                event(kind="w3_needed", deltas=d)
                ensure("sym_g4", 0)
                ensure("sym_g4", 1)
        rc = sh([PY, "-u", str(HERE / "analyze_c2.py")], "analyze_c2")
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
        for s in ("RUNNING", "DONE", "FAILED"):
            if (STATE / s).exists():
                print(s, (STATE / s).read_text(encoding="utf-8").strip())
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
        print(f"C2 runner detached, pid {p.pid}; state in {STATE}")
        return
    work(wait_for)


if __name__ == "__main__":
    main()
