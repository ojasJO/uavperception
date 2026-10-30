#!/usr/bin/env python
"""Run the pre-registered v2 series unattended (preregistration.json), resumable, with sentinels.

    venv\\Scripts\\python.exe experiments\\aerotrack_v2\\scripts\\run_v2.py --detach [--wait-for <runner state dir>]
    venv\\Scripts\\python.exe experiments\\aerotrack_v2\\scripts\\run_v2.py --status

Preconditions (checked, else the runner refuses): corpus v2 verified (experiments/cst_completion/reports/corpus_v2.json verdict
OK) and preregistration.json committed and unmodified. Order: optional wait for another GPU job's DONE/FAILED/PAUSED sentinel ->
machinery smoke test (2 epochs x 512 samples, scored on 50 images/dataset, results kept apart) -> v1 reference evaluation ->
stage 1 (seed 0) -> pre-registered futility check -> stage 2 (seed 1, survivors) -> analyze_v2.py.
A training run interrupted mid-way is resumed from its last.pt (one retry); a finished run / evaluation is never redone.
State: results/runs/aerotrack_v2/_runner/ (RUNNING / DONE / FAILED, events.jsonl, one log per job).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import v2series as S

from anti_uav_repro.utils.repro import file_sha256

HERE = Path(__file__).resolve().parent
PY = sys.executable
PRE = S.SERIES / "preregistration.json"
CORPUS_REPORT = S.REPO / "experiments" / "cst_completion" / "reports" / "corpus_v2.json"
REF_RUN = "gen_temporal_fp_g4_full_e20"
CANDIDATES = ("sym_g4", "causal_g4")


def event(**kw):
    kw["t"] = S.now()
    S.STATE.mkdir(parents=True, exist_ok=True)
    with open(S.STATE / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(kw, default=str) + "\n")
    print(json.dumps(kw, default=str), flush=True)


def sh(cmd, log_name):
    log = S.STATE / f"{log_name}.log"
    event(kind="start", job=log_name, cmd=" ".join(cmd))
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=str(S.REPO), stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, AEROTRACK_CORPUS=S.CORPUS))
    event(kind="end", job=log_name, rc=rc, minutes=round((time.time() - t0) / 60, 1))
    return rc


def trained(name):
    d = S.REPO / "results/runs" / name
    try:
        m = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    w = d / "weights" / "last.pt"
    return m.get("return_code") == 0 and w.exists() and m.get("last_pt_sha256") == file_sha256(w)


def evaluated(name, smoke=False):
    p = S.REPORTS / ("v2_results_smoke.json" if smoke else "v2_results.json")
    if not p.exists():
        return False
    w = S.REPO / "results/runs" / name / "weights" / "last.pt"
    sha = file_sha256(w) if w.exists() else None
    return any(r["run"] == name and r["set"] == "val" and r["weights_sha256"] == sha
               for r in json.loads(p.read_text(encoding="utf-8"))["rows"])


def train(config, seed, smoke=False):
    name = f"v2_{config}_s{seed}_e{2 if smoke else 20}" + ("_smoke" if smoke else "")   # train_v2 --smoke = 2 epochs
    if trained(name):
        event(kind="skip_trained", run=name)
        return name
    base = [PY, "-u", str(HERE / "train_v2.py"), "--config", config, "--seed", str(seed)] + (["--smoke"] if smoke else [])
    exists = (S.REPO / "results/runs" / name / "run_meta.json").exists()
    rc = sh(base + (["--resume"] if exists else []), f"train_{name}")
    if rc != 0 and not trained(name):
        event(kind="retry_resume", run=name)
        rc = sh(base + ["--resume"], f"train_{name}_retry")
    if not trained(name):
        raise RuntimeError(f"training {name} did not finish (rc {rc})")
    return name


def evaluate(name, smoke=False):
    if evaluated(name, smoke):
        event(kind="skip_evaluated", run=name)
        return
    cmd = [PY, "-u", str(HERE / "eval_v2.py"), "--run", name] + (["--cap", "50"] if smoke else [])
    if sh(cmd, f"eval_{name}") != 0 and sh(cmd, f"eval_{name}_retry") != 0:
        raise RuntimeError(f"evaluation of {name} failed twice")


def cst(name):
    rows = json.loads((S.REPORTS / "v2_results.json").read_text(encoding="utf-8"))["rows"]
    return next(r["metrics"]["cst_drone_AP50"] for r in rows if r["run"] == name and r["set"] == "val")


def preconditions():
    rep = json.loads(CORPUS_REPORT.read_text(encoding="utf-8")) if CORPUS_REPORT.exists() else {}
    if rep.get("verdict") != "OK":
        raise SystemExit(f"corpus v2 not verified ({CORPUS_REPORT.name}: {rep.get('verdict')!r})")
    corpus_dir = S.REPO / "src/pipeline" / S.CORPUS
    if file_sha256(corpus_dir / "manifest.csv") != rep["manifest_sha256"]:
        raise SystemExit("corpus v2 manifest differs from the verified one")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PRE)], cwd=S.REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(PRE)], cwd=S.REPO).returncode == 0
    if not (tracked and clean):
        raise SystemExit("preregistration.json must be committed and unmodified before the campaign starts")
    return {"corpus_manifest_sha256": rep["manifest_sha256"], "preregistration_sha256": file_sha256(PRE), "git": S.git()}


def work(wait_for):
    S.STATE.mkdir(parents=True, exist_ok=True)
    for s in ("DONE", "FAILED"):
        (S.STATE / s).unlink(missing_ok=True)
    (S.STATE / "RUNNING").write_text(f"pid {os.getpid()} started {S.now()}\n", encoding="utf-8")
    try:
        info = preconditions()
        event(kind="campaign_start", **info)
        S.ledger({"id": "P:v2_campaign_start", "type": "plan", **info})
        if wait_for:
            wd = Path(wait_for)
            event(kind="waiting_for_gpu_job", dir=str(wd))
            free = ("DONE", "FAILED", "PAUSED")          # PAUSED: a runner that yielded the GPU at a clean boundary (run_h3.py)
            while not any((wd / x).exists() for x in free):
                time.sleep(60)
            event(kind="gpu_free", sentinel=next(x for x in free if (wd / x).exists()))
        smoke = train("causal_g4", 0, smoke=True)
        evaluate(smoke, smoke=True)
        evaluate(REF_RUN)
        for c in ("stock",) + CANDIDATES:
            evaluate(train(c, 0))
        d0 = {c: cst(f"v2_{c}_s0_e20") - cst("v2_stock_s0_e20") for c in CANDIDATES}
        survivors = [c for c in CANDIDATES if d0[c] > 0]
        event(kind="futility", seed0_cst_delta_vs_stock=d0, survivors=survivors)
        S.ledger({"id": "F:v2_futility", "type": "decision", "seed0_cst_delta_vs_stock": d0, "survivors": survivors})
        if survivors:
            for c in ("stock",) + tuple(survivors):
                evaluate(train(c, 1))
        rc = sh([PY, "-u", str(HERE / "analyze_v2.py")], "analyze_v2")
        event(kind="analysis", rc=rc)
        (S.STATE / "DONE").write_text(f"finished {S.now()}\n", encoding="utf-8")
    except BaseException as e:  # noqa: BLE001 — every exit leaves a sentinel
        (S.STATE / "FAILED").write_text(f"{S.now()} {type(e).__name__}: {e}\n", encoding="utf-8")
        event(kind="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        (S.STATE / "RUNNING").unlink(missing_ok=True)


def main():
    wait_for = sys.argv[sys.argv.index("--wait-for") + 1] if "--wait-for" in sys.argv else None
    if "--status" in sys.argv:
        for s in ("RUNNING", "DONE", "FAILED"):
            if (S.STATE / s).exists():
                print(s, (S.STATE / s).read_text(encoding="utf-8").strip())
        ev = S.STATE / "events.jsonl"
        if ev.exists():
            print("".join(ev.read_text(encoding="utf-8").splitlines(keepends=True)[-5:]))
        return
    if "--detach" in sys.argv:
        S.STATE.mkdir(parents=True, exist_ok=True)
        if (S.STATE / "RUNNING").exists():
            sys.exit(f"already running: {(S.STATE / 'RUNNING').read_text(encoding='utf-8').strip()}")
        preconditions()                                  # fail fast in the foreground
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0
        args = [PY, "-u", __file__] + (["--wait-for", wait_for] if wait_for else [])
        with open(S.STATE / "runner.log", "a", encoding="utf-8") as out:
            p = subprocess.Popen(args, cwd=str(S.REPO), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 creationflags=flags, close_fds=True, env=dict(os.environ, AEROTRACK_CORPUS=S.CORPUS))
        print(f"v2 runner detached, pid {p.pid}; state in {S.STATE}")
        return
    work(wait_for)


if __name__ == "__main__":
    main()
