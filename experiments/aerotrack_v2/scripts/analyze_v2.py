#!/usr/bin/env python
"""Apply the pre-registered rules of the v2 series (preregistration.json) to the VAL records. Never runs a model.

    venv\\Scripts\\python.exe experiments\\aerotrack_v2\\scripts\\analyze_v2.py [B=1000]

Paired, dataset-stratified, sequence-level bootstrap (the generalization series' method): per-image matches at IoU 0.5 are
grouped by sequence; each replicate resamples sequences with replacement inside each dataset and recomputes drone AP50 for
every run on the SAME resampled sequences; a comparison's statistic is the seed-mean of the per-seed deltas.
Writes reports/v2_decisions.json (+ markdown) and appends a ledger record.
"""
import json
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

import v2series as S
import v2lib as L
import evaluation as E

IOU = np.array([0.5])
PRE = S.SERIES / "preregistration.json"
G = {}


def prep(path):
    d = pickle.load(open(path, "rb"))
    by, ds_of = defaultdict(list), {}
    for r in d["records"]:
        ds_of[r["seq_key"]] = r["dataset"]
        by[r["seq_key"]].append(E.match_image(r["pred"], r["gt"], IOU, None))
    cat = {k: tuple(np.concatenate([x[i] for x in v], 0) for i in range(4)) for k, v in by.items()}
    return {"cat": cat, "ds_of": ds_of, "identity": d["identity"]["sha256"]}


def drone_ap(parts):
    if not parts:
        return None
    tp, cf, pc, tc = (np.concatenate([p[i] for p in parts], 0) for i in range(4))
    res = E.ap_per_class(tp, cf, pc, tc, classes=(0, 1))
    return next((float(res["ap"][i, 0]) for i, c in enumerate(res["classes"]) if c == 0 and res["n_gt"][i] > 0), None)


def stats(run, seqs):
    g = G[run]
    out = {}
    for name, keep in (("cst", lambda k: g["ds_of"][k] == "cst"), ("anti_uav", lambda k: g["ds_of"][k] == "anti_uav"),
                       ("pooled", lambda k: True)):
        out[name] = drone_ap([g["cat"][k] for k in seqs if keep(k)])
    for c in L.CST_CATEGORIES:
        out[f"cst_{c}"] = drone_ap([g["cat"][k] for k in seqs if g["ds_of"][k] == "cst" and L.cst_category(k) == c])
    return out


def init(files):
    for f in files:
        G[f.stem.split("__")[0]] = prep(f)


def one(seed):
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for k, d in next(iter(G.values()))["ds_of"].items():
        strata[d].append(k)
    sample = [ks[i] for _, ks in sorted(strata.items()) for i in rng.integers(0, len(ks), len(ks))]
    return {n: stats(n, sample) for n in G}


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    files = sorted(f for f in S.CACHE.glob("*__val.pkl"))
    init(files)
    have = set(G)
    ids = {G[n]["identity"] for n in G}
    if len(ids) != 1:
        sys.exit(f"runs were scored on different frame sets: {ids}")
    every = list(next(iter(G.values()))["ds_of"])
    point = {n: stats(n, every) for n in G}
    with Pool(6, initializer=init, initargs=(files,)) as pool:
        reps = pool.map(one, range(B), chunksize=10)
    margin, ni = pre["rules"]["margin"], pre["rules"]["non_inferiority_anti_uav"]
    decisions = []
    for comp in pre["comparisons"]:
        pairs = [(a, b) for a, b in comp["pairs"] if a in have and b in have]
        rec = {"id": comp["id"], "question": comp["question"], "metric": comp["metric"], "pairs_available": pairs,
               "pairs_registered": comp["pairs"]}
        if len(pairs) < len(comp["pairs"]):
            rec["status"] = "INCOMPLETE"
            decisions.append(rec)
            continue
        m = comp["metric"]

        def delta(st, metric):
            vals = [st[a][metric] - st[b][metric] for a, b in pairs if st[a][metric] is not None and st[b][metric] is not None]
            return float(np.mean(vals)) if vals else None
        per_pair = [point[a][m] - point[b][m] for a, b in pairs]
        d = delta(point, m)
        boots = np.array([v for v in (delta(r, m) for r in reps) if v is not None])
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        au = delta(point, "anti_uav")
        rec.update(per_pair=per_pair, mean_delta=d, ci95=ci, anti_uav_mean_delta=au,
                   per_category={c: delta(point, f"cst_{c}") for c in L.CST_CATEGORIES})
        kind = comp["rule"]
        if kind == "superiority":
            fails = []
            if not d > margin:
                fails.append(f"(1) mean delta {d:+.4f} <= {margin}")
            if not all(x > 0 for x in per_pair):
                fails.append("(2) not positive for every seed")
            if not ci[0] > 0:
                fails.append("(3) CI includes 0")
            if au is not None and not au > -ni:
                fails.append(f"(4) Anti-UAV {au:+.4f} <= -{ni}")
            rec["status"] = "SUPPORTED" if not fails else "NOT SUPPORTED"
            rec["failed"] = fails
        elif kind == "non_inferiority":
            rec["status"] = ("NON-INFERIOR (conclusive)" if ci[0] >= -margin else "NON-INFERIOR (point estimate only)"
                             if d >= -margin else "INFERIOR")
        else:
            rec["status"] = "DESCRIPTIVE"
        decisions.append(rec)
    by = {r["id"]: r for r in decisions}
    sel = pre["selection_rule"]
    chosen = None
    if by.get("C1", {}).get("status") == "SUPPORTED" and str(by.get("C3", {}).get("status", "")).startswith("NON-INFERIOR"):
        chosen = sel["if_C1_and_C3"]
    elif by.get("C2", {}).get("status") == "SUPPORTED":
        chosen = sel["elif_C2"]
    elif all(by.get(k, {}).get("status") not in (None, "INCOMPLETE") for k in ("C1", "C2", "C3")):
        chosen = sel["else"]
    out = {"generated": S.now(), "B": B, "preregistration": str(PRE.relative_to(S.REPO)), "runs": sorted(have),
           "frame_set_sha256": ids.pop(), "point": point, "decisions": decisions, "selected_candidate": chosen,
           "caveat": "bootstrap = sampling noise over VAL sequences; seed noise enters only through the per-seed deltas (2 seeds)"}
    S.write_json_atomic(out, S.REPORTS / "v2_decisions.json")
    L_ = [f"# AeroTrack v2 series — decisions (pre-registered rules)", "", f"*Generated {out['generated']} by `scripts/analyze_v2.py` "
          f"from cached VAL records (B = {B}); rules frozen in `preregistration.json`.*", "",
          "| id | question | metric | per seed | mean Δ | 95 % CI | Anti-UAV Δ | status |", "|---|---|---|---|---:|---|---:|---|"]
    for r in decisions:
        if r.get("status") == "INCOMPLETE":
            L_.append(f"| {r['id']} | {r['question']} | {r['metric']} | — | — | — | — | INCOMPLETE |")
            continue
        L_.append(f"| {r['id']} | {r['question']} | {r['metric']} | {', '.join(f'{x:+.3f}' for x in r['per_pair'])} | "
                  f"{r['mean_delta']:+.4f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | "
                  f"{'—' if r['anti_uav_mean_delta'] is None else format(r['anti_uav_mean_delta'], '+.4f')} | {r['status']} |")
    L_ += ["", f"**Selected candidate (selection rule): {chosen or 'not decidable yet'}**", "",
           "Per-category CST drone AP50 deltas (seed-mean):", ""]
    for r in decisions:
        if r.get("per_category"):
            L_.append(f"- {r['id']}: " + ", ".join(f"{c} {v:+.3f}" for c, v in r["per_category"].items() if v is not None))
    (S.REPORTS / "v2_decisions.md").write_text("\n".join(L_) + "\n", encoding="utf-8")
    S.ledger({"id": "A:v2_decisions", "type": "analysis", "output": "reports/v2_decisions.json", "selected": chosen,
              "statuses": {r["id"]: r.get("status") for r in decisions}})
    print("\n".join(L_))


if __name__ == "__main__":
    main()
