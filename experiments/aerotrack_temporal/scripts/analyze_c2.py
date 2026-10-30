#!/usr/bin/env python
"""Apply C2's registered rules (preregistration/C2_causal_windows.json) to the cached corpus-v2 VAL records. Never runs a model.

    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\analyze_c2.py [B=1000]

Statistics are the v2 series' (experiments/aerotrack_v2/scripts/analyze_v2.py: per-image matches at IoU 0.5 grouped by sequence;
paired, dataset-stratified, sequence-level bootstrap; a comparison's statistic is the seed-mean of the per-seed deltas), with the
v2 registration's margins. W1: each causal window vs stock (superiority); W2: gap dependence (descriptive); W3: the best
W1-superior window vs the symmetric window (non-inferiority). Output: experiments/aerotrack_temporal/reports/C2_causal_windows.*
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V2 = REPO / "experiments" / "aerotrack_v2" / "scripts"
sys.path.insert(0, str(V2))
import analyze_v2 as A  # noqa: E402
import v2lib as L  # noqa: E402
import v2series as S  # noqa: E402

PRE = REPO / "experiments" / "aerotrack_temporal" / "preregistration" / "C2_causal_windows.json"
REPORTS = REPO / "experiments" / "aerotrack_temporal" / "reports"
WINDOWS = ("causal_g1", "causal_g2", "causal_g4", "causal_g8")
REACH = {"causal_g1": "[t-2, t-1, t]", "causal_g2": "[t-4, t-2, t]", "causal_g4": "[t-8, t-4, t]", "causal_g8": "[t-16, t-8, t]",
         "stock": "[t, t, t]", "sym_g4": "[t-4, t, t+4]"}


def run(cfg, seed):
    return f"v2_{cfg}_s{seed}_e20"


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    v2pre = json.loads((S.SERIES / "preregistration.json").read_text(encoding="utf-8"))
    margin, ni = v2pre["rules"]["margin"], v2pre["rules"]["non_inferiority_anti_uav"]
    files = sorted(f for f in S.CACHE.glob("*__val.pkl"))
    A.init(files)
    have = set(A.G)
    ids = {A.G[n]["identity"] for n in A.G}
    if len(ids) != 1:
        sys.exit(f"runs were scored on different frame sets: {ids}")
    every = list(next(iter(A.G.values()))["ds_of"])
    point = {n: A.stats(n, every) for n in A.G}
    with Pool(6, initializer=A.init, initargs=(files,)) as pool:
        reps = pool.map(A.one, range(B), chunksize=10)

    def delta(st, pairs, metric):
        vals = [st[a][metric] - st[b][metric] for a, b in pairs if st[a][metric] is not None and st[b][metric] is not None]
        return float(np.mean(vals)) if vals else None

    def compare(pairs, rule):
        per = [point[a]["cst"] - point[b]["cst"] for a, b in pairs]
        d = delta(point, pairs, "cst")
        boots = np.array([v for v in (delta(r, pairs, "cst") for r in reps) if v is not None])
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        au = delta(point, pairs, "anti_uav")
        rec = {"pairs": pairs, "per_seed": per, "mean_delta": d, "ci95": ci, "anti_uav_mean_delta": au,
               "per_category": {c: delta(point, pairs, f"cst_{c}") for c in L.CST_CATEGORIES}}
        if rule == "superiority":
            fails = []
            if not d > margin:
                fails.append(f"(1) mean delta {d:+.4f} <= {margin}")
            if not all(x > 0 for x in per):
                fails.append("(2) not positive for every seed")
            if not ci[0] > 0:
                fails.append("(3) CI includes 0")
            if au is not None and not au > -ni:
                fails.append(f"(4) Anti-UAV {au:+.4f} <= -{ni}")
            rec["status"], rec["failed"] = ("SUPPORTED" if not fails else "NOT SUPPORTED"), fails
        else:
            rec["status"] = ("NON-INFERIOR (conclusive)" if ci[0] >= -margin else
                             "NON-INFERIOR (point estimate only)" if d >= -margin else "INFERIOR")
        return rec

    w1 = {}
    for c in WINDOWS:
        pairs = [(run(c, s), run("stock", s)) for s in (0, 1) if run(c, s) in have and run("stock", s) in have]
        if not pairs:
            w1[c] = {"status": "NOT RUN"}
        elif len(pairs) == 1:
            p = pairs[0]
            w1[c] = {"status": "STOPPED AFTER SEED 0 (futility)" if point[p[0]]["cst"] - point[p[1]]["cst"] <= 0 else "INCOMPLETE",
                     "per_seed": [point[p[0]]["cst"] - point[p[1]]["cst"]], "pairs": pairs}
        else:
            w1[c] = compare(pairs, "superiority")
    levels = {}
    for c in ("stock",) + WINDOWS + ("sym_g4",):
        runs = [run(c, s) for s in (0, 1) if run(c, s) in have]
        if runs:
            vals = [point[r]["cst"] for r in runs]
            boots = [np.mean([rep[r]["cst"] for r in runs]) for rep in reps]
            levels[c] = {"runs": runs, "seed_mean_cst_ap50": float(np.mean(vals)), "per_seed": vals,
                         "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         "anti_uav_seed_mean": float(np.mean([point[r]["anti_uav"] for r in runs])), "reach": REACH[c]}
    sup = [c for c in WINDOWS if w1[c].get("status") == "SUPPORTED"]
    best = max(sup, key=lambda c: levels[c]["seed_mean_cst_ap50"]) if sup else None
    w3 = None
    if best:
        pairs = [(run(best, s), run("sym_g4", s)) for s in (0, 1) if run("sym_g4", s) in have]
        w3 = dict(compare(pairs, "non_inferiority"), window=best) if len(pairs) == 2 else {"status": "INCOMPLETE", "window": best}
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "B": B, "registration": str(PRE.relative_to(REPO)),
           "frame_set_sha256": ids.pop(), "margins": {"superiority": margin, "anti_uav_non_inferiority": ni},
           "W1_superiority_vs_stock": w1, "W2_levels": levels, "W3_cost_of_causality": w3,
           "selected_causal_window": best or "stock (no causal window is W1-superior)",
           "caveat": "bootstrap = sampling noise over VAL sequences; seed noise only via the per-seed deltas (2 seeds); the selected "
                     "window's VAL delta is optimistic (winner's curse) until a TEST pass"}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "C2_causal_windows.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# C2 — causal frame-stacking windows (Track C, corpus-v2 VAL)", "",
          f"*Generated {out['generated']} by `scripts/analyze_c2.py` (B = {B}); rules: `{out['registration']}` with the v2 margins "
          f"(superiority {margin}, Anti-UAV non-inferiority {ni}).*", "", "## W2 — CST drone AP50 by window (seed mean, bootstrap 95 % CI)", "",
          "| window | frames | look-ahead | seed-mean CST AP50 | 95 % CI | per seed | Anti-UAV AP50 |", "|---|---|---:|---:|---|---|---:|"]
    for c, v in levels.items():
        la = "4 frames" if c == "sym_g4" else "0"
        md.append(f"| {c} | {v['reach']} | {la} | {v['seed_mean_cst_ap50']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | "
                  f"{', '.join(f'{x:.4f}' for x in v['per_seed'])} | {v['anti_uav_seed_mean']:.4f} |")
    md += ["", "## W1 — each causal window vs single frame (superiority)", "", "| window | per seed Δ | mean Δ | 95 % CI | Anti-UAV Δ | status |",
           "|---|---|---:|---|---:|---|"]
    for c, r in w1.items():
        ps = ", ".join(f"{x:+.4f}" for x in r.get("per_seed", []))
        md.append(f"| {c} | {ps} | {r['mean_delta']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['anti_uav_mean_delta']:+.4f} | "
                  f"**{r['status']}** |" if "ci95" in r else f"| {c} | {ps} | — | — | — | **{r['status']}** |")
    md += ["", "## W3 — cost of causality", "",
           (f"best W1-superior window `{w3['window']}` vs `sym_g4`: mean Δ {w3['mean_delta']:+.4f}, CI [{w3['ci95'][0]:+.4f}, "
            f"{w3['ci95'][1]:+.4f}] → **{w3['status']}**" if w3 and "ci95" in w3 else
            "no causal window is W1-superior: W3 not applicable" if not w3 else f"`{w3['window']}`: {w3['status']}"),
           "", f"**Selected window for the causal system:** {out['selected_causal_window']}", "", f"*{out['caveat']}*"]
    (REPORTS / "C2_causal_windows.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
