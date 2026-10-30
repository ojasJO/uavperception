#!/usr/bin/env python
"""Track C synthesis: one table of every registered hypothesis outcome (C1, C1c, C3, C1b) read from the committed reports.

    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_summary.py

Reads only experiments/aerotrack_temporal/reports/*.json (never re-computes a metric) and writes
experiments/aerotrack_temporal/reports/TRACK_C_SUMMARY.json with, per experiment / cache / domain / hypothesis: the verdict, the
paired statistic(s) with their 95 % CIs exactly as the producing script recorded them, and the source file + its sha256.
Track C metrics (ADE / FDE px, Gaussian NLL, mSA on Track C's VAL cross-fit) are VAL-only and never mixed with the CST benchmark.
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REPORTS = REPO / "experiments" / "aerotrack_temporal" / "reports"
CACHES = ("v2_causal_g4_s0_e20", "v2_stock_s0_e20")


def rd(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pairs(block):
    """Seed pairs of a C1-family hypothesis block -> compact list (statistic, mean, CI, n)."""
    out = []
    for p in block.get("pairs", []):
        out.append({k: p[k] for k in p if k in ("a", "b", "diff", "ci95", "rel", "rel_ci95", "n_sequences", "stat", "mean_delta", "wins", "losses")})
    return out


def main():
    res = {"note": "VAL-only Track C outcomes, read verbatim from the producing reports", "sources": {}, "C1": {}, "C1c": {}, "C3": {}, "C1b": {},
           "missing": []}
    p = REPORTS / "C1_trajectory_forecasting.json"
    r = rd(p)
    if r:
        res["sources"]["C1"] = {"file": p.relative_to(REPO).as_posix(), "sha256": sha(p)}
        for dom, d in r["domains"].items():
            res["C1"][dom] = {h: {"verdict": d[h].get("verdict", "descriptive"), "pairs": pairs(d[h])} for h in d if h.startswith("H") and isinstance(d[h], dict)}
    p = REPORTS / "C1c_domain_aware_forecasting.json"
    r = rd(p)
    if r:
        res["sources"]["C1c"] = {"file": p.relative_to(REPO).as_posix(), "sha256": sha(p)}
        for dom, d in r["domains"].items():
            res["C1c"][dom] = {"arm": d.get("arm"),
                               **{h: {"verdict": d[h].get("verdict", "descriptive"), "pairs": pairs(d[h])} for h in d if h.startswith("H") and isinstance(d[h], dict)}}
    for cache in CACHES:
        p = REPORTS / f"C3_tracking_motion_models_{cache}.json"
        r = rd(p)
        if not r:
            res["missing"].append(p.name)
        else:
            res["sources"][f"C3_{cache}"] = {"file": p.relative_to(REPO).as_posix(), "sha256": sha(p)}
            res["C3"][cache] = {"hypotheses": r["hypotheses"],
                                "mSA": {k: {g: v["mSA"] for g, v in t["cross_fitted"].items()} for k, t in r["trackers"].items()},
                                "adopted_learned": r.get("adopted_learned"), "seconds": r.get("seconds")}
        p = REPORTS / f"C1b_forecasting_on_tracker_histories_{cache}.json"
        r = rd(p)
        if not r:
            res["missing"].append(p.name)
        else:
            res["sources"][f"C1b_{cache}"] = {"file": p.relative_to(REPO).as_posix(), "sha256": sha(p)}
            res["C1b"][cache] = {dom: {"learned_arm": d["learned_arm"], "n_sequences": d["n_sequences"], "n_samples": d["n_samples"],
                                       "reference": d["reference"],
                                       **{h: {"verdict": d[h].get("verdict", "descriptive"), "pairs": pairs(d[h])}
                                          for h in ("H1b_point", "H2b_uncertainty", "H4_augmentation")}}
                                 for dom, d in r["domains"].items()}
    out = REPORTS / "TRACK_C_SUMMARY.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1)[:6000])
    print("missing:", res["missing"])


if __name__ == "__main__":
    main()
