#!/usr/bin/env python
"""Pre-registered decisions of the generalization series (the runner calls this; the rules below are fixed before any run).

    python experiments/aerotrack_generalization/scripts/analyze_gen.py [--tag _smoke]

Primary metric: CST drone AP50 on the held-out DEV fold (drone class only; last.pt). Non-inferiority: Anti-UAV drone AP50.
A candidate is SUPPORTED versus `stock` (same fold, same seed) only if
  (1) fold-mean delta > +0.02, (2) delta > 0 in every one of the 3 folds, (3) the fold-pooled paired sequence bootstrap 95 % CI
  of the mean delta excludes 0, and (4) the fold-mean Anti-UAV delta > -0.02.
Futility: a candidate not ahead on fold 0 (delta <= 0) cannot satisfy (2) and is REJECTED without running folds 1-2.
Budget check (H4): `stock` at 40 epochs vs `stock` at 20 epochs under the same rule.
Output: reports/gen_decisions<tag>.json."""
import argparse
import json
import pickle
import time
from collections import defaultdict

import _series as S
import gen_lib as L
import numpy as np

import evaluation as E

from plan_gen import PLAN

PRIMARY, NONINF, MARGIN = PLAN["primary_metric"], PLAN["noninferiority_metric"], PLAN["margin_ap50"]
REFERENCE, CANDIDATES = PLAN["reference"], PLAN["candidates"]
IOU = np.array([0.5])


def rows(tag):
    p = S.REPORTS / f"gen_results{tag}.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"rows": []}
    return {(r["run"], r["set"]): r for r in data["rows"]}


def cst_seq_parts(run, tag):
    """CST drone-class match results of a run's DEV records, per sequence (for the paired bootstrap)."""
    d = pickle.load(open(S.CACHE / "gen_records" / f"{run}__dev{tag}.pkl", "rb"))
    by = defaultdict(list)
    for r in d["records"]:
        if r["dataset"] == "cst":
            by[r["seq_key"]].append(E.match_image(r["pred"], r["gt"], IOU, None))
    return {k: tuple(np.concatenate([x[i] for x in v], 0) for i in range(4)) for k, v in by.items()}


def ap0(parts):
    if not parts:
        return 0.0
    tp, cf, pc, tc = (np.concatenate([p[i] for p in parts], 0) for i in range(4))
    res = E.ap_per_class(tp, cf, pc, tc, classes=(0,))
    return float(res["ap"][0, 0]) if res["n_gt"][0] > 0 else 0.0


def boot_mean_delta(pairs, tag, B=1000, seed=0):
    """pairs: [(candidate_run, reference_run)] one per fold -> distribution of the fold-mean CST delta (paired resampling of each
    fold's CST sequences)."""
    prepared = [(cst_seq_parts(c, tag), cst_seq_parts(r, tag)) for c, r in pairs]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        ds = []
        for pc, pr in prepared:
            keys = sorted(pr)
            pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            ds.append(ap0([pc[k] for k in pick if k in pc]) - ap0([pr[k] for k in pick]))
        out.append(float(np.mean(ds)))
    return np.array(out)


def compare(cand, ref, R, tag, epochs, suffix="", e_main=20):
    folds, deltas, nonin = {}, [], []
    for k in (0, 1, 2):
        rc, rr = R.get((L.run_name(cand, k, epochs[0], suffix), "dev")), R.get((L.run_name(ref, k, epochs[1], suffix), "dev"))
        if rc and rr:
            d = rc["metrics"][PRIMARY] - rr["metrics"][PRIMARY]
            n = rc["metrics"][NONINF] - rr["metrics"][NONINF]
            folds[k] = {"candidate": rc["metrics"][PRIMARY], "reference": rr["metrics"][PRIMARY], "delta": d,
                        "relative_pct": 100 * d / rr["metrics"][PRIMARY] if rr["metrics"][PRIMARY] else None, "anti_uav_delta": n,
                        "secondary_deltas": {m: rc["metrics"][m] - rr["metrics"][m] for m in rc["metrics"] if isinstance(rc["metrics"][m], float)
                                             and isinstance(rr["metrics"].get(m), float)}}
    out = {"candidate": cand if epochs[0] == e_main else f"{cand}_e{epochs[0]}", "reference": ref if epochs[1] == e_main else f"{ref}_e{epochs[1]}",
           "primary": PRIMARY, "folds": folds}
    if 0 in folds and folds[0]["delta"] <= 0:
        out["status"] = "rejected: not ahead on fold 0 (pre-registered futility; rule (2) cannot hold)"
    elif len(folds) == 3:
        ds = [folds[k]["delta"] for k in (0, 1, 2)]
        boot = boot_mean_delta([(L.run_name(cand, k, epochs[0], suffix), L.run_name(ref, k, epochs[1], suffix)) for k in (0, 1, 2)], tag)
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        mean_n = float(np.mean([folds[k]["anti_uav_delta"] for k in (0, 1, 2)]))
        rules = {"(1) mean delta > 0.02": float(np.mean(ds)) > MARGIN, "(2) delta > 0 in every fold": all(d > 0 for d in ds),
                 "(3) bootstrap CI excludes 0": ci[0] > 0, "(4) Anti-UAV non-inferior": mean_n > -MARGIN}
        out.update(mean_delta=float(np.mean(ds)), sd_delta=float(np.std(ds, ddof=1)), ci95_mean_delta=ci, anti_uav_mean_delta=mean_n,
                   rules=rules, status="supported" if all(rules.values()) else "not supported: " + ", ".join(k for k, v in rules.items() if not v))
    else:
        out["status"] = "pending"
    return out


def decide(tag="", suffix="", e_main=PLAN["epochs"], e_long=PLAN["budget_h4_epochs"]):
    R = rows(tag)
    dec = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rule": __doc__.split("Output:")[0].strip(),
           "candidates": {c: compare(c, REFERENCE, R, tag, (e_main, e_main), suffix, e_main) for c in CANDIDATES},
           "budget_h4": compare(REFERENCE, REFERENCE, R, tag, (e_long, e_main), suffix, e_main)}
    S.REPORTS.mkdir(parents=True, exist_ok=True)
    (S.REPORTS / f"gen_decisions{tag}.json").write_text(json.dumps(dec, indent=1, default=str), encoding="utf-8")
    return dec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--e-main", type=int, default=PLAN["epochs"])
    ap.add_argument("--e-long", type=int, default=PLAN["budget_h4_epochs"])
    a = ap.parse_args()
    d = decide(a.tag, a.suffix, a.e_main, a.e_long)
    for c, v in list(d["candidates"].items()) + [("budget_h4", d["budget_h4"])]:
        print(f"{c:16s} {v['status']}  " + "  ".join(f"f{k}: {f['delta']:+.4f}" for k, f in v["folds"].items()))
