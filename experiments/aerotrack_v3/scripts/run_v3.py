#!/usr/bin/env python
"""Run the pre-registered v3 series unattended (experiments/aerotrack_v3/preregistration.json), resumable, with sentinels.

    venv\\Scripts\\python.exe experiments\\aerotrack_v3\\scripts\\run_v3.py --detach [--wait-for <state dir>]
    venv\\Scripts\\python.exe experiments\\aerotrack_v3\\scripts\\run_v3.py --status

The registration lists `runs` ([config, seed, batch] in execution order) and, optionally, `futility` rules
({"after": [config, seed], "compare": [config_a, config_b_or_v2_run], "drop_if_delta_le": 0.0, "drop": [[config, seed], ...]}).
Order: preconditions (corpus v2 verified, registration committed) -> optional wait -> smoke test of the first configuration ->
each registered run trained (resumed once on failure) and evaluated -> analyze_v3.py. A finished run / evaluation is never redone.
State: results/runs/aerotrack_v3/_runner/ (RUNNING / DONE / FAILED, events.jsonl, logs).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import v3series as S
import v3lib as L

from anti_uav_repro.utils.repro import file_sha256

HERE = Path(__file__).resolve().parent
PY = sys.executable
PRE = S.SERIES / "preregistration.json"
CORPUS_REPORT = S.REPO / "experiments" / "cst_completion" / "reports" / "corpus_v2.json"


def event(**kw):
    kw["t"] = S.now()
    S.STATE.mkdir(parents=True, exist_ok=True)
    with open(S.STATE / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(kw, default=str) + "\n")
    print(json.dumps(kw, default=str), flush=True)


def sh(cmd, name):
    event(kind="start", job=name, cmd=" ".join(cmd))
    t0 = time.time()
    with open(S.STATE / f"{name}.log", "a", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=str(S.REPO), stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, AEROTRACK_CORPUS=S.CORPUS))
    event(kind="end", job=name, rc=rc, minutes=round((time.time() - t0) / 60, 1))
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
    p = S.REPORTS / ("v3_results_smoke.json" if smoke else "v3_results.json")
    if not p.exists():
        return False
    w = S.REPO / "results/runs" / name / "weights" / "last.pt"
    sha = file_sha256(w) if w.exists() else None
    return any(r["run"] == name and r["set"] == "val" and r["weights_sha256"] == sha for r in json.loads(p.read_text(encoding="utf-8"))["rows"])


def train(config, seed, batch, smoke=False, domains="all"):
    doms = L.parse_domains(domains)
    name = L.run_name(config + L.domain_tag(doms), seed, 2 if smoke else 20) + ("_smoke" if smoke else "")
    if trained(name):
        event(kind="skip_trained", run=name)
        return name
    base = [PY, "-u", str(HERE / "train_v3.py"), "--config", config, "--seed", str(seed), "--batch", str(batch),
            "--train-domains", ",".join(doms)] + (["--smoke"] if smoke else [])
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
    cmd = [PY, "-u", str(HERE / "eval_v3.py"), "--run", name] + (["--cap", "50"] if smoke else [])
    if sh(cmd, f"eval_{name}") != 0 and sh(cmd, f"eval_{name}_retry") != 0:
        raise RuntimeError(f"evaluation of {name} failed twice")


def cst(name):
    for p in (S.REPORTS / "v3_results.json", S.REPO / "experiments" / "aerotrack_v2" / "reports" / "v2_results.json"):
        if p.exists():
            for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
                if r["run"] == name and r["set"] == "val":
                    return r["metrics"]["cst_drone_AP50"]
    raise KeyError(name)


def preconditions():
    rep = json.loads(CORPUS_REPORT.read_text(encoding="utf-8")) if CORPUS_REPORT.exists() else {}
    if rep.get("verdict") != "OK":
        raise SystemExit(f"corpus v2 not verified ({rep.get('verdict')!r})")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PRE)], cwd=S.REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(PRE)], cwd=S.REPO).returncode == 0
    if not (tracked and clean):
        raise SystemExit("experiments/aerotrack_v3/preregistration.json must be committed and unmodified")
    return {"corpus_manifest_sha256": rep["manifest_sha256"], "preregistration_sha256": file_sha256(PRE), "git": S.git()}


def work(wait_for):
    S.STATE.mkdir(parents=True, exist_ok=True)
    for s in ("DONE", "FAILED"):
        (S.STATE / s).unlink(missing_ok=True)
    (S.STATE / "RUNNING").write_text(f"pid {os.getpid()} started {S.now()}\n", encoding="utf-8")
    try:
        info = preconditions()
        pre = json.loads(PRE.read_text(encoding="utf-8"))
        event(kind="campaign_start", **info)
        S.ledger({"id": "P:v3_campaign_start", "type": "plan", **info})
        if wait_for:
            wd = Path(wait_for)
            event(kind="waiting_for_gpu_job", dir=str(wd))
            while not ((wd / "DONE").exists() or (wd / "FAILED").exists()):
                time.sleep(60)
        first = pre["runs"][0]
        evaluate(train(first[0], 0, first[2], smoke=True), smoke=True)
        dropped = set()
        for entry in pre["runs"]:
            cfg, seed, batch = entry[:3]
            domains = entry[3] if len(entry) > 3 else "all"          # optional training-domain subset (cross-dataset runs)
            if (cfg, seed) in dropped:
                event(kind="dropped_by_futility", config=cfg, seed=seed)
                continue
            evaluate(train(cfg, seed, batch, domains=domains))
            for fu in pre.get("futility", []):
                if [cfg, seed] == fu["after"]:
                    a, b = fu["compare"]
                    d = cst(a) - cst(b)
                    event(kind="futility_check", compare=fu["compare"], delta=d, threshold=fu["drop_if_delta_le"])
                    if d <= fu["drop_if_delta_le"]:
                        dropped |= {tuple(x) for x in fu["drop"]}
                        S.ledger({"id": f"F:v3:{a}", "type": "decision", "delta": d, "dropped": fu["drop"]})
        rc = sh([PY, "-u", str(HERE / "analyze_v3.py")], "analyze_v3")
        event(kind="analysis", rc=rc)
        (S.STATE / "DONE").write_text(f"finished {S.now()}\n", encoding="utf-8")
    except BaseException as e:  # noqa: BLE001
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
        return
    if "--detach" in sys.argv:
        S.STATE.mkdir(parents=True, exist_ok=True)
        if (S.STATE / "RUNNING").exists():
            sys.exit(f"already running: {(S.STATE / 'RUNNING').read_text(encoding='utf-8').strip()}")
        preconditions()
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0
        args = [PY, "-u", __file__] + (["--wait-for", wait_for] if wait_for else [])
        with open(S.STATE / "runner.log", "a", encoding="utf-8") as out:
            p = subprocess.Popen(args, cwd=str(S.REPO), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 creationflags=flags, close_fds=True, env=dict(os.environ, AEROTRACK_CORPUS=S.CORPUS))
        print(f"v3 runner detached, pid {p.pid}; state in {S.STATE}")
        return
    work(wait_for)


if __name__ == "__main__":
    main()
