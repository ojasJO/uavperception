#!/usr/bin/env python
"""X2 (preregistration/X2_seeds_single_frame.json) -- training and VAL design. Never touches TEST.

    python experiments/cst_anti_uav/scripts/run_x2.py --train v3_n__ttt_cst_s0_e20 [v3_n__causal_g4_cst_s1_e20 ...]   (GPU)
    python experiments/cst_anti_uav/scripts/run_x2.py --val-design                                                  (GPU, sequential)

--train: v3's train() for each named X2 run (resumed once on failure; a finished run is skipped). Each training writes only its own
results/runs/<name>/ and one ledger line, so several --train jobs may run on different nodes at once.
--val-design (ONE job; eval_v3 rewrites a shared results file): for all six runs, eval_v3 on VAL, the full CST VAL detection cache,
track_eval; then per-sequence cross-fitted STE SA (recomputed with track_eval's per-half settings and checked against its summary),
experiments/cst_anti_uav/frozen_x2_params.json (settings tuned on all CST VAL, per run) and the V1 analysis
experiments/cst_anti_uav/reports/X2_val.{json,md}. State / logs: results/runs/cst_anti_uav/_x2/<mode>/.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V3 = REPO / "experiments" / "aerotrack_v3" / "scripts"
for p in (HERE, V3, REPO / "experiments" / "systems" / "scripts"):
    sys.path.insert(0, str(p))
import x2lib as X  # noqa: E402
import run_v3 as R  # noqa: E402
import v3series as S  # noqa: E402  (pins AEROTRACK_CORPUS=data_unified_v2)

PY = sys.executable
B = 10000


def _sys_module(name):
    """Explicit-path import of a systems module (src/model/track_eval.py would shadow systems/scripts/track_eval.py: F-05)."""
    import importlib.util
    p = REPO / "experiments" / "systems" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"systems_{name}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    assert Path(m.__file__).resolve() == p.resolve()
    return m


def committed_clean(p):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(p)], cwd=REPO).returncode == 0
    return tracked and clean


def parse(run):
    for arm in X.ARMS:
        for s in X.SEEDS:
            if X.run_name(arm, s) == run:
                return arm, s
    raise SystemExit(f"{run} is not an X2 run ({X.all_runs()})")


def step(job, cmd):
    R.event(kind="start", job=job)
    t0 = time.time()
    with open(S.STATE / f"{job}.log", "a", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, AEROTRACK_CORPUS=S.CORPUS))
    R.event(kind="end", job=job, rc=rc, minutes=round((time.time() - t0) / 60, 1))
    if rc != 0:
        raise RuntimeError(f"{job} failed (rc {rc})")


def cache_ok(run):
    from anti_uav_repro.utils.repro import file_sha256
    idx = REPO / "results/runs" / "systems" / "_cache" / run / S.CORPUS / "val" / "index.json"
    if not idx.exists():
        return False
    i = json.loads(idx.read_text(encoding="utf-8"))
    n = sum(1 for v in i["sequences"].values() if v["dataset"] == "cst")
    return n == 40 and not i.get("limited") and i["weights_sha256"] == file_sha256(REPO / "results/runs" / run / "weights" / "last.pt")


def track_report(run):
    return REPO / "experiments" / "systems" / "reports" / f"track_{run}_{S.CORPUS}.json"


def track_ok(run):
    p = track_report(run)
    if not p.exists():
        return False
    idx = json.loads((REPO / "results/runs" / "systems" / "_cache" / run / S.CORPUS / "val" / "index.json").read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))["cache_set_sha256"] == idx["set_sha256"]


def cross_fitted_per_sequence(run, rep):
    """Per-sequence cross-fitted SA (percent, official names) for STE and argmax, recomputed from track_eval's per-half settings."""
    TE = _sys_module("track_eval")
    from anti_uav_repro.eval.metrics import sequence_metrics
    _, seqs = TE.load(run, S.CORPUS, split="val")
    out = {}
    for kind in ("ste", "argmax"):
        half_params = {int(h): p for h, p in rep["trackers"][kind]["params_per_half"].items()}
        sa = {}
        for k, it in seqs.items():
            if it["group"] != "cst_IR":
                continue
            pred = TE.run_tracker(kind, half_params[it["half"]], it)
            mask = np.arange(it["seq"].n_frames) >= pred["start_frame"]
            sa[it["seq"].label_path.parent.name] = 100 * sequence_metrics(pred["res"], it["seq"].exist, it["seq"].gt_rect, mask=mask)["SA"]
        want = rep["trackers"][kind]["cross_fitted"]["cst_IR"]["mSA"]
        if abs(np.mean(list(sa.values())) - want) > 1e-6:
            raise RuntimeError(f"{run} {kind}: recomputed cross-fitted mSA {np.mean(list(sa.values()))} != track_eval {want}")
        out[kind] = sa
    return out


def boot(d):
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    b = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return float(d.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def val_design():
    for run in X.all_runs():
        if not R.trained(run):
            raise SystemExit(f"{run} is not trained yet")
    for run in X.all_runs():
        R.evaluate(run)
        if not cache_ok(run):
            step(f"cache_val_{run}", [PY, "-u", str(REPO / "experiments/systems/scripts/cache_detections.py"), "--run", run,
                                      "--corpus", S.CORPUS, "--datasets", "cst"])
        if not track_ok(run):
            step(f"track_eval_{run}", [PY, "-u", str(REPO / "experiments/systems/scripts/track_eval.py"), "--run", run,
                                       "--corpus", S.CORPUS, "--workers", "4"])
    v3rows = {r["run"]: r for r in json.loads((S.REPORTS / "v3_results.json").read_text(encoding="utf-8"))["rows"] if r["set"] == "val"}
    frozen = {"registration": str(X.REGISTRATION).replace("\\", "/"), "written": S.now(),
              "rule": "per run: STE / argmax settings tuned on ALL CST VAL by systems/scripts/track_eval.py (maximum mSA); commit before the TEST pass",
              "runs": {}}
    per = {}
    from anti_uav_repro.utils.repro import file_sha256
    for run in X.all_runs():
        rep = json.loads(track_report(run).read_text(encoding="utf-8"))
        per[run] = cross_fitted_per_sequence(run, rep)
        frozen["runs"][run] = {"arm": parse(run)[0], "seed": parse(run)[1],
                               "weights_sha256": file_sha256(REPO / "results/runs" / run / "weights" / "last.pt"),
                               "ste": rep["trackers"]["ste"]["tuned_on_all_val"]["params"],
                               "argmax": rep["trackers"]["argmax"]["tuned_on_all_val"]["params"],
                               "val_cross_fitted_mSA": {k: rep["trackers"][k]["cross_fitted"]["cst_IR"]["mSA"] for k in ("ste", "argmax")},
                               "val_tuned_on_all_mSA": {k: rep["trackers"][k]["tuned_on_all_val"]["summary"]["cst_IR"]["mSA"]
                                                        for k in ("ste", "argmax")},
                               "val_cst_drone_AP50": v3rows[run]["metrics"]["cst_drone_AP50"]}
    x1 = json.loads((REPO / "experiments" / "aerotrack_v3" / "frozen_x1_params.json").read_text(encoding="utf-8"))
    if frozen["runs"][X.X1_RUN]["ste"] != x1["ste"] or frozen["runs"][X.X1_RUN]["argmax"] != x1["argmax"]:
        raise RuntimeError("X1's run: settings differ from frozen_x1_params.json")
    S.write_json_atomic(frozen, REPO / X.FROZEN)
    names = sorted(per[X.X1_RUN]["ste"])
    sm = {arm: {n: float(np.mean([per[X.run_name(arm, s)]["ste"][n] for s in X.SEEDS])) for n in names} for arm in X.ARMS}
    d = [sm["causal_g4"][n] - sm["ttt"][n] for n in names]
    m, ci = boot(d)
    v1 = {"STE_cross_fitted_seed_mean": {"mean_delta": m, "ci95": ci, "wins": int(sum(x > 0 for x in d)), "losses": int(sum(x < 0 for x in d))},
          "per_seed": {}}
    for s in X.SEEDS:
        a, b = per[X.run_name("causal_g4", s)]["ste"], per[X.run_name("ttt", s)]["ste"]
        ms, cis = boot([a[n] - b[n] for n in names])
        v1["per_seed"][s] = {"mean_delta": ms, "ci95": cis,
                             "AP50_delta": frozen["runs"][X.run_name("causal_g4", s)]["val_cst_drone_AP50"]
                             - frozen["runs"][X.run_name("ttt", s)]["val_cst_drone_AP50"]}
    rep = {"generated": S.now(), "registration": frozen["registration"], "runs": frozen["runs"], "V1": v1,
           "per_sequence_cross_fitted_SA": per, "note": "VAL shares source videos with TRAIN (paper §3.3): descriptive"}
    S.write_json_atomic(rep, REPO / X.VAL_REPORT.with_suffix(".json"))
    L = ["# X2 — CST VAL design (seeds × causal_g4 / single frame, CST-only training)", "",
         f"*Generated {rep['generated']} by `scripts/run_x2.py --val-design`; descriptive (VAL shares source videos with TRAIN).*", "",
         "| run | arm | seed | CST AP50 | STE mSA cross-fitted | STE mSA tuned on all VAL | argmax cross-fitted | frozen STE settings |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
    for run, r in frozen["runs"].items():
        L.append(f"| {run} | {r['arm']} | {r['seed']} | {r['val_cst_drone_AP50']:.3f} | {r['val_cross_fitted_mSA']['ste']:.2f} | "
                 f"{r['val_tuned_on_all_mSA']['ste']:.2f} | {r['val_cross_fitted_mSA']['argmax']:.2f} | {r['ste']} |")
    L += ["", f"**V1** causal_g4 − single frame (STE cross-fitted, seed-mean per sequence, 40 sequences): {m:+.2f} SA, 95 % CI "
          f"[{ci[0]:+.2f}, {ci[1]:+.2f}], wins {v1['STE_cross_fitted_seed_mean']['wins']} / losses {v1['STE_cross_fitted_seed_mean']['losses']}; "
          "per seed: " + ", ".join(f"s{s} {v['mean_delta']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] (AP50 {v['AP50_delta']:+.3f})"
                                    for s, v in v1["per_seed"].items())]
    (REPO / X.VAL_REPORT.with_suffix(".md")).write_text("\n".join(L) + "\n", encoding="utf-8")
    R.event(kind="x2_val_design_done")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--train", nargs="+")
    g.add_argument("--val-design", action="store_true")
    a = ap.parse_args()
    if not committed_clean(REPO / X.REGISTRATION):
        raise SystemExit("X2 registration must be committed and unmodified")
    S.STATE = REPO / "results/runs" / "cst_anti_uav" / "_x2" / ("val_design" if a.val_design else "train_" + "+".join(parse(r)[0] + str(parse(r)[1]) for r in a.train))
    S.STATE.mkdir(parents=True, exist_ok=True)
    R.event(kind="x2_start", mode="val_design" if a.val_design else "train", runs=a.train, git=S.git())
    if a.train:
        for run in a.train:
            arm, seed = parse(run)
            if run == X.X1_RUN:
                raise SystemExit("X1's run is not retrained by X2")
            name = R.train(X.ARMS[arm], seed, 32, domains=X.DOMAINS)
            assert name == run, (name, run)
        R.event(kind="x2_train_done", runs=a.train)
    else:
        val_design()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
