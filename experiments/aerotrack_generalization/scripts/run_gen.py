#!/usr/bin/env python
r"""ONE command for the generalization campaign — unattended, resumable, applying only the pre-registered plan (plan_gen.py).

    venv\Scripts\python.exe experiments\aerotrack_generalization\scripts\run_gen.py --detach     (background)
    venv\Scripts\python.exe experiments\aerotrack_generalization\scripts\run_gen.py --status
    venv\Scripts\python.exe experiments\aerotrack_generalization\scripts\run_gen.py --stop
    venv\Scripts\python.exe experiments\aerotrack_generalization\scripts\run_gen.py --smoke      (machinery test: tiny runs, separate
                                                                                                  names, ledger and results)

Stage A: fold 0 for stock, spd_fp, temporal_fp_g1, temporal_fp_g4 (train_gen.py, then eval_gen.py on DEV + official VAL).
Futility (analyze_gen.py): a candidate not ahead of stock on fold 0 is rejected. Stage B: folds 1-2 for stock + survivors.
Stage C (H4, only if time permits): stock at 40 epochs. Final: only if a candidate is supported, retrain it on the full TRAIN
split and score it once on TEST. Time guards from the plan; every run is resumable; the completed repair experiment is hashed
at the first start and verified at every start and at the end. The parent never imports torch."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import _series as S
from plan_gen import PLAN

sys.path.insert(0, str(S.REPO / "experiments" / "aerotrack_repair" / "scripts"))
import run_grid as RG  # noqa: E402  (process supervision shared with the repair series)

HERE = Path(__file__).resolve().parent
TRAIN, EVAL, ANALYZE, REPORT = HERE / "train_gen.py", HERE / "eval_gen.py", HERE / "analyze_gen.py", HERE / "report_gen.py"
INNER = r"train_gen\.py.*--_child"
PLAN_FILE = HERE / "plan_gen.py"


class Campaign(RG.Grid):
    def __init__(self, smoke):
        self.epochs = None
        self.smoke = smoke
        self.suffix = self.tag = "_smoke" if smoke else ""
        self.dir = S.REPO / "runs" / "aerotrack_generalization" / f"campaign{self.suffix}"
        self.logs = self.dir / "logs"
        self.state_path = self.dir / "state.json"
        self.events_path = self.dir / "events.jsonl"
        self.driver_log = self.logs / "driver.log"
        self.lock = self.dir / "campaign.lock"
        self.stop_file = self.dir / "STOP"

    def run_name(self, cfg, fold, epochs):
        return f"gen_{cfg}_{'full' if fold == 'full' else f'f{fold}'}_e{epochs}{self.suffix}"


def env():
    return RG.child_env()               # carries GEN_SMOKE from os.environ when smoke-testing (separate ledger)


def run(c, cmd, log, what, a, inner=None):
    return RG.run_logged(c, cmd, log, a.stall_minutes, what, inner=inner)


# ------------------------------------------------------------------------------------------------ bookkeeping
def plan_digest():
    return hashlib.sha256(PLAN_FILE.read_bytes()).hexdigest()


def freeze_plan(c):
    """P1: the plan and the hash of plan_gen.py, written once; any later difference stops the campaign."""
    L = S.ledger_path()
    prior = [json.loads(x) for x in L.read_text(encoding="utf-8").splitlines() if x.strip()] if L.exists() else []
    p1 = [x for x in prior if x.get("id") == "P1"]
    folds_sha = json.loads((S.SERIES / "folds.json").read_text(encoding="utf-8"))["sha256"]
    if not p1:
        S.ledger({"id": "P1", "type": "pre-registration", "plan": PLAN, "plan_file_sha256": plan_digest(), "folds_sha256": folds_sha,
                  "note": "frozen before the first training run of the series"})
        c.event("plan_frozen", sha=plan_digest()[:12])
    elif p1[0]["plan_file_sha256"] != plan_digest() or p1[0]["folds_sha256"] != folds_sha:
        raise RG.GridFatal("plan_gen.py or folds.json changed after the plan was frozen (ledger P1); refusing to continue")


def repair_fingerprint():
    files = []
    for d in sorted((S.REPO / "runs").glob("repair_abl_*_e20")):
        files += [d / n for n in ("run_meta.json", "results.csv", "args.json", "weights/best.pt", "weights/last.pt")]
    files += sorted((S.REPO / "experiments" / "aerotrack_repair" / "reports").glob("*"))
    h = hashlib.sha256()
    for f in files:
        if f.is_file():
            h.update(str(f.relative_to(S.REPO)).encode())
            h.update(RG.sha256(f).encode())
    return h.hexdigest(), len(files)


def integrity(c, when):
    fp, n = repair_fingerprint()
    st = c.state()
    if "repair_fingerprint" not in st:
        c.update_state(repair_fingerprint=fp)
        S.ledger({"id": "INT0", "type": "integrity", "what": "hash of the completed repair experiment (runs + reports)", "files": n, "sha256": fp})
    elif st["repair_fingerprint"] != fp:
        c.event("integrity_changed", when=when, files=n)
        raise RG.GridFatal("the completed repair experiment changed (runs/repair_abl_* or experiments/aerotrack_repair/reports)")
    c.event("integrity_ok", when=when, files=n)


def results(c):
    return RG.jload(S.REPORTS / f"gen_results{c.tag}.json") or {"rows": []}


def budget_ok(c, hours, kind):
    if c.smoke:
        return True
    limit = PLAN["time_budget_h"]["final_ends_within" if kind == "final" else "dev_runs_end_within"]
    return time.time() + hours * 3600 <= c.state()["campaign_start_epoch"] + limit * 3600


# ------------------------------------------------------------------------------------------------ steps
def inspect(rd):
    out = subprocess.run([str(RG.VENV_PY), str(Path(RG.__file__).resolve()), "--_inspect", str(rd)], cwd=str(S.REPO), capture_output=True,
                         text=True, timeout=900, env=env(), creationflags=RG.CHILD_FLAGS)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise RG.GridFatal(f"could not inspect {rd.name}: {out.stderr[-1500:]}")


def ensure_run(c, cfg, fold, epochs, a):
    name = c.run_name(cfg, fold, epochs)
    rd = S.REPO / "runs" / name
    info = inspect(rd) if rd.exists() else {"exists": False}
    if info.get("exists") and RG.run_complete(info, epochs):
        c.log(f"[train] {name}: complete — verified, not re-run")
        return name
    workers, launches, failures, no_progress, vram_waits = a.workers, 0, 0, 0, 0
    while True:
        if c.stop_file.exists():
            raise RG.GridStopped("STOP requested")
        done_before = (info.get("last", {}).get("epoch", -1) + 1) if info.get("last", {}).get("ok") else 0
        cmd = [RG.VENV_PY, "-u", TRAIN, "--config", cfg, "--fold", str(fold), "--epochs", str(epochs), "--workers", str(workers)]
        if c.smoke:
            cmd += ["--name-suffix", c.suffix, "--samples-per-epoch", "320", "--monitor-cap", "60"]
        mode = "fresh start"
        if rd.exists() and any(rd.iterdir()):
            last, best = info.get("last", {}), info.get("best", {})
            if last.get("exists") and not last.get("ok"):
                os.replace(rd / "weights" / "last.pt", rd / "weights" / f"last.unreadable_{RG.stamp()}.pt")
                if not best.get("ok"):
                    raise RG.GridFatal(f"{name}: neither last.pt nor best.pt is loadable")
                cmd += ["--resume-from", str(rd / "weights" / "best.pt")]
                mode = f"resume from best.pt (epoch {best.get('epoch')}; last.pt unreadable)"
                done_before = best.get("epoch", -1) + 1
            else:
                cmd += ["--resume"]
                mode = f"resume from last.pt (epoch {last.get('epoch')})" if last.get("ok") else "restart in the existing dir"
        if failures >= a.max_launches:
            raise RG.GridFatal(f"{name}: {failures} failed launches")
        wait_gpu(c, f"train {name}")
        launches += 1
        c.update_state(phase=f"train:{name}")
        c.event("train_launch", run=name, launch=launches, workers=workers, mode=mode)
        lp = c.logs / f"train_{name}_launch{launches}_{RG.stamp()}.log"
        rc, stalled = run(c, cmd, lp, f"train {name}", a, inner=INNER)
        info = inspect(rd)
        done_after = (info.get("last", {}).get("epoch", -1) + 1) if info.get("last", {}).get("ok") else 0
        if rc == 0 and RG.run_complete(info, epochs):
            meta = RG.jload(rd / "run_meta.json") or {}
            c.event("train_complete", run=name, epochs=done_after, train_seconds=meta.get("train_seconds"))
            S.ledger({"id": f"TR:{name}", "type": "training", "run": name, "config": cfg, "fold": fold, "seed": meta.get("seed"),
                      "epochs": epochs, "train_seconds": meta.get("train_seconds"), "launches": len(meta.get("attempts", [])),
                      "last_pt_sha256": meta.get("last_pt_sha256"), "best_pt_sha256": meta.get("best_pt_sha256"),
                      "init_check": RG.jload(rd / "init_check.json"), "rationale": PLAN_RATIONALE.get(cfg, "")})
            return name
        kind, detail = ("inconsistent", f"exit 0, {done_after}/{epochs} epochs") if rc == 0 else RG.classify(RG.log_tail(lp), rc, stalled)
        c.event("train_failed", run=name, launch=launches, rc=rc, kind=kind, detail=detail, epochs_before=done_before, epochs_after=done_after, log=lp.name)
        if kind in RG.FATAL_KINDS or kind == "inconsistent":
            raise RG.GridFatal(f"{name}: {kind}: {detail} (log {lp})")
        if kind == "vram" and done_after <= done_before:
            vram_waits += 1
            if vram_waits > 12:
                raise RG.GridFatal(f"{name}: VRAM too tight on {vram_waits} launches")
            pause = 600
        else:
            failures += 1
            no_progress = 0 if done_after > done_before else no_progress + 1
            if no_progress >= 3:
                raise RG.GridFatal(f"{name}: three failed launches without a finished epoch ({kind}: {detail})")
            workers, pause = (max(1, workers // 2), 180) if kind == "host_memory" else (workers, 120)
        c.event("train_recover", run=name, action=f"resume after {pause}s", workers=workers, kind=kind)
        time.sleep(pause)


def ensure_eval(c, name, sets, a):
    w = S.REPO / "runs" / name / "weights" / "last.pt"
    sha12 = RG.sha256(w)[:12]
    for attempt in range(1, 4):
        have = {(r["run"], r["set"]) for r in results(c)["rows"] if r.get("weights_sha256") == sha12}
        need = [s for s in sets if (name, s) not in have]
        if not need:
            return
        wait_gpu(c, f"eval {name}")
        c.update_state(phase=f"eval:{name}")
        cmd = [RG.VENV_PY, "-u", EVAL, "--run", name, "--sets", *need, "--tag", c.tag] + (["--cap", "60"] if c.smoke else [])
        lp = c.logs / f"eval_{name}_attempt{attempt}_{RG.stamp()}.log"
        rc, stalled = run(c, cmd, lp, f"eval {name} {need}", a)
        if rc:
            kind, detail = RG.classify(RG.log_tail(lp), rc, stalled)
            c.event("eval_failed", run=name, sets=need, rc=rc, kind=kind, detail=detail, log=lp.name)
            if kind in RG.FATAL_KINDS:
                raise RG.GridFatal(f"eval {name}: {kind}: {detail} (log {lp})")
            time.sleep(60)
    have = {(r["run"], r["set"]) for r in results(c)["rows"] if r.get("weights_sha256") == sha12}
    if any((name, s) not in have for s in sets):
        raise RG.GridFatal(f"evaluation of {name} failed three times")


def analyze(c, a, e_main, e_long):
    lp = c.logs / f"analyze_{RG.stamp()}.log"
    rc, _ = run(c, [RG.VENV_PY, "-u", ANALYZE, "--tag", c.tag, "--suffix", c.suffix, "--e-main", str(e_main), "--e-long", str(e_long)], lp, "analyze", a)
    if rc:
        raise RG.GridFatal(f"analyze_gen.py failed (log {lp})")
    dec = RG.jload(S.REPORTS / f"gen_decisions{c.tag}.json")
    c.event("decisions", **{k: v["status"] for k, v in dec["candidates"].items()}, budget_h4=dec["budget_h4"]["status"])
    return dec


def wait_gpu(c, what, max_wait_h=6.0):
    t0, announced = time.time(), False
    while True:
        found = jobs()
        if not found:
            if announced:
                c.event("gpu_free", waited_min=round((time.time() - t0) / 60, 1))
            return
        if not announced:
            c.event("gpu_busy_waiting", before=what, processes=found)
            announced = True
        if time.time() - t0 > max_wait_h * 3600:
            raise RG.GridFatal(f"another GPU job still running after {max_wait_h} h: {found}")
        if c.stop_file.exists():
            raise RG.GridStopped("stopped while waiting for the GPU")
        time.sleep(120)


def jobs():
    """Python ML processes outside this runner's tree (both series), and python GPU clients."""
    ps = RG._psutil()
    mine = RG.own_tree_pids()
    pat = re.compile(r"(train_gen\.py|eval_gen\.py|run_gen\.py|c_model[\\/]train\.py|train_ablation\.py|evaluate_detector\.py|run_grid\.py|"
                     r"collect_val_records\.py|anti_uav.*[\\/](train|evaluate)\.py|audit[\\/]bench_)")
    helper = re.compile(r"run_(gen|grid)\.py.*(--status|--stop|--note|--_inspect|--_spawn|--detach)")
    found = []
    for p in ps.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in mine or "python" not in (p.info["name"] or "").lower():
                continue
            cl = " ".join(p.info["cmdline"] or [])
            if pat.search(cl) and not helper.search(cl):
                found.append({"pid": p.info["pid"], "cmdline": cl[:200]})
        except (ps.NoSuchProcess, ps.AccessDenied):
            continue
    return found


def final_test(c, name, a):
    L = S.REPORTS / "gen_test_ledger.jsonl"
    sha = RG.sha256(S.REPO / "runs" / name / "weights" / "last.pt")
    if L.exists() and any(json.loads(x).get("weights_sha256") == sha for x in L.read_text(encoding="utf-8").splitlines() if x.strip()):
        c.log(f"[test] {name} already has its single TEST evaluation — not re-run")
        return
    c.event("test_eval_start", run=name)
    lp = c.logs / f"test_{name}_{RG.stamp()}.log"
    rc, stalled = run(c, [RG.VENV_PY, "-u", EVAL, "--run", name, "--sets", "test", "--final-eval"], lp, f"TEST {name}", a)
    if rc:
        c.event("test_eval_failed", run=name, rc=rc, detail=RG.classify(RG.log_tail(lp), rc, stalled), log=lp.name)
        raise RG.GridFatal(f"TEST evaluation of {name} failed (log {lp}); no result recorded")
    c.event("test_eval_done", run=name)


PLAN_RATIONALE = {c: q for c, q in {
    "stock": "reference: stock YOLO11n layout, single frame, COCO (= function-preserving init)",
    "spd_fp": "H1: does SPD-Conv help once it starts from the pretrained function (removes the 48 % vs 94 % init confound)?",
    "temporal_fp_g1": "H1b: does the original t-1,t,t+1 stack help once its stem starts as YOLO11n on frame t?",
    "temporal_fp_g4": "H2: does a motion-matched gap (g=4 from CST's 0.53 px/frame) make the temporal cue useful?"}.items()}


# ------------------------------------------------------------------------------------------------ campaign
def campaign(c, a):
    freeze_plan(c)
    integrity(c, "start")
    E, E2 = (1, 2) if c.smoke else (PLAN["epochs"], PLAN["budget_h4_epochs"])
    est = PLAN["estimated_hours"]
    sets = ["dev"] if c.smoke else ["dev", "val"]

    def stage_run(cfg, fold, epochs, hours, kind="dev"):
        if not budget_ok(c, hours, kind):
            c.event("skipped_time_budget", run=c.run_name(cfg, fold, epochs), hours=hours)
            return None
        n = ensure_run(c, cfg, fold, epochs, a)
        ensure_eval(c, n, sets if fold != "full" else ["val"], a)
        return n

    for cfg in PLAN["stage_a_order"]:                                   # Stage A: fold 0
        stage_run(cfg, 0, E, est["run_e20"] + est["eval"])
    dec = analyze(c, a, E, E2)
    survivors = [x for x in PLAN["candidates"] if not dec["candidates"][x]["status"].startswith("rejected")]
    c.event("stage_a_done", survivors=survivors)
    for k in (1, 2):                                                     # Stage B: folds 1-2 for stock + survivors
        for cfg in ([PLAN["reference"]] + survivors if survivors else []):
            stage_run(cfg, k, E, est["run_e20"] + est["eval"])
    dec = analyze(c, a, E, E2)
    if a.h4:                                                             # Stage C: training budget (H4)
        stage_run(PLAN["reference"], 0, E2, est["run_e40"] + est["eval"])
        dec = analyze(c, a, E, E2)
        h4 = dec["budget_h4"]
        if h4["status"] == "pending" and "0" in h4["folds"]:            # ahead on fold 0 (JSON keys are strings)
            for k in (1, 2):
                stage_run(PLAN["reference"], k, E, est["run_e20"] + est["eval"])
                stage_run(PLAN["reference"], k, E2, est["run_e40"] + est["eval"])
            dec = analyze(c, a, E, E2)
    supported = [x for x in PLAN["candidates"] if dec["candidates"][x]["status"] == "supported"]
    c.event("development_done", supported=supported)
    if supported:                                                        # Final: only a supported candidate touches TEST
        winner = max(supported, key=lambda x: dec["candidates"][x]["mean_delta"])
        c.event("final_candidate", candidate=winner, mean_delta=dec["candidates"][winner]["mean_delta"])
        if c.smoke:
            c.log("[smoke] final step (full-TRAIN retrain + TEST) not exercised in smoke mode")
        elif budget_ok(c, est["run_e20"] + 2 * est["eval"], "final"):
            n = stage_run(winner, "full", E, est["run_e20"] + est["eval"], kind="final")
            if n:
                final_test(c, n, a)
        else:
            c.event("final_skipped_time_budget", candidate=winner)
    if REPORT.exists():
        lp = c.logs / f"report_{RG.stamp()}.log"
        rc, _ = run(c, [RG.VENV_PY, "-u", REPORT, "--tag", c.tag], lp, "report", a)
        c.event("report" if rc == 0 else "report_failed", rc=rc, log=lp.name)
    integrity(c, "end")


# ------------------------------------------------------------------------------------------------ CLI
def acquire_lock(c):
    c.mkdirs()
    info = RG.jload(c.lock) or {}
    if c.lock.exists() and RG.pid_alive(info.get("pid", -1), "run_gen.py"):
        raise SystemExit(f"[ABORT] a campaign runner is already active (pid {info.get('pid')})")
    c.lock.unlink(missing_ok=True)
    fd = os.open(str(c.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "started": RG.now()}, f)
    import atexit
    atexit.register(lambda: c.lock.unlink(missing_ok=True))


def status(c):
    st = c.state()
    info = RG.jload(c.lock) or {}
    alive = RG.pid_alive(info.get("pid", -1), "run_gen.py")
    el = (time.time() - st["campaign_start_epoch"]) / 3600 if st.get("campaign_start_epoch") else 0
    print(f"campaign {RG.rel(c.dir)}: runner {'RUNNING pid ' + str(info.get('pid')) if alive else 'not running'}; phase {st.get('phase')}; "
          f"outcome {st.get('outcome')}; {el:.1f} h since start")
    for r in results(c)["rows"]:
        m = r["metrics"]
        print(f"  {r['run']:34s} {r['set']:4s} CST drone {m.get('cst_drone_AP50') or 0:.4f}  Anti-UAV drone {m.get('anti_uav_drone_AP50') or 0:.4f}")
    cur = st.get("current") or {}
    if cur.get("log"):
        print(f"current: {cur.get('what')}\n" + "\n".join(RG.log_tail(cur["log"], 3000).splitlines()[-4:]))
    if c.events_path.exists():
        print("--- last events\n" + "\n".join(x[:220] for x in c.events_path.read_text(encoding="utf-8").splitlines()[-6:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-launches", type=int, default=8)
    ap.add_argument("--stall-minutes", type=float, default=60)
    ap.add_argument("--no-h4", dest="h4", action="store_false", help="skip the training-budget check (Stage C)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--note", default=None)
    ap.add_argument("--_spawn", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.smoke:
        os.environ["GEN_SMOKE"] = "1"
    c = Campaign(a.smoke)
    if a._spawn:
        rest = sys.argv[1:]
        k = rest.index("--_spawn")
        time.sleep(20)
        with open(a._spawn, "ab") as f:
            RG._popen_detached([str(RG.VENV_PY), "-u", str(Path(__file__).resolve())] + rest[:k] + rest[k + 2:], f)
        return
    if a.status:
        return status(c)
    if a.stop:
        c.stop_file.write_text(RG.now(), encoding="utf-8")
        print("STOP requested; the runner stops the current step within ~30 s and keeps every finished result")
        return
    if a.note:
        c.mkdirs()
        c.event("operator_note", text=a.note)
        return
    if a.detach:
        c.mkdirs()
        log = c.logs / f"console_{RG.stamp()}.log"
        rest = [x for x in sys.argv[1:] if x != "--detach"]
        RG._popen_detached([str(RG.VENV_PY), str(Path(__file__).resolve()), "--_spawn", str(log)] + rest, subprocess.DEVNULL)
        print(f"[detach] campaign runner starts in ~20 s\n  timeline: {c.driver_log}\n  status  : run_gen.py --status")
        return
    acquire_lock(c)
    c.stop_file.unlink(missing_ok=True)
    RG.keep_awake(True)
    if not c.state().get("campaign_start_epoch"):
        c.update_state(campaign_start_epoch=time.time(), campaign_start=RG.now())
    c.update_state(phase="starting", outcome=None, current=None)
    c.event("start", pid=os.getpid(), smoke=c.smoke, argv=" ".join(sys.argv))
    orphans = jobs()
    if orphans:                                                            # an earlier runner's step still running
        c.event("orphans_adopted", processes=orphans)
        wait_gpu(c, "orphans", max_wait_h=10)
    try:
        snap = c.dir / "provenance"
        snap.mkdir(parents=True, exist_ok=True)
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=str(S.REPO), capture_output=True, text=True).stdout
        (snap / f"code_{RG.stamp()}.patch").write_text(diff, encoding="utf-8")
        campaign(c, a)
        c.update_state(phase="done", outcome="complete")
        c.event("complete", hours=round((time.time() - c.state()["campaign_start_epoch"]) / 3600, 2))
        return 0
    except RG.GridStopped as e:
        c.update_state(phase="stopped", outcome=str(e))
        c.event("stopped", reason=str(e))
        return 3
    except RG.GridFatal as e:
        c.update_state(phase="failed", outcome=str(e))
        c.event("fatal", reason=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
