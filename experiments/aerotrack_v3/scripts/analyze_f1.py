#!/usr/bin/env python
"""Apply F1's registered rule (preregistration_F1_standard_yolo11n.json) to cached corpus-v2 VAL records. Never runs a model.

Statistics are analyze_v2's (per-image matches at IoU 0.5 grouped by sequence; paired, dataset-stratified, sequence-level bootstrap;
seed-mean of per-seed deltas) with the v2 margins; v2 and v3 caches are read together (run keys use the full run name: v3 names
contain '__'). Output: experiments/aerotrack_v3/reports/F1_standard_yolo11n.{json,md}; the verdict drives X1's and CF's network.
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
import v2series as S2  # noqa: E402

V3_CACHE = REPO / "results/runs" / "aerotrack_v3" / "_cache"
OUT = REPO / "experiments" / "aerotrack_v3" / "reports" / "F1_standard_yolo11n"
WINDOW_OF = {"causal_g4": "causal_g4", "sym_g4": "sym_g4", "stock": "ttt"}


def files_for(runs):
    out = []
    for r in runs:
        for base in (S2.CACHE, V3_CACHE):
            f = base / f"{r}__val.pkl"
            if f.exists():
                out.append(f)
    return out


def init(files):
    for f in files:
        A.G[f.stem.rsplit("__", 1)[0]] = A.prep(f)


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    v2pre = json.loads((S2.SERIES / "preregistration.json").read_text(encoding="utf-8"))
    margin, ni = v2pre["rules"]["margin"], v2pre["rules"]["non_inferiority_anti_uav"]
    sel = json.loads((S2.REPORTS / "v2_decisions.json").read_text(encoding="utf-8"))["selected_candidate"]
    std = [f"v3_n_std__{WINDOW_OF[sel]}_s{s}_e20" for s in (0, 1)]
    built = [f"v2_{sel}_s{s}_e20" for s in (0, 1)]
    files = files_for(std + built)
    init(files)
    have = set(A.G)
    pairs = [(a, b) for a, b in zip(std, built) if a in have and b in have]
    if not pairs:
        raise SystemExit(f"no F1 pair cached yet (have {sorted(have)})")
    ids = {A.G[n]["identity"] for n in A.G}
    if len(ids) != 1:
        raise SystemExit(f"runs were scored on different frame sets: {ids}")
    every = list(next(iter(A.G.values()))["ds_of"])
    point = {n: A.stats(n, every) for n in A.G}
    with Pool(4, initializer=init, initargs=(files,)) as pool:
        reps = pool.map(A.one, range(B), chunksize=10)

    def delta(st, metric):
        vals = [st[a][metric] - st[b][metric] for a, b in pairs if st[a][metric] is not None and st[b][metric] is not None]
        return float(np.mean(vals)) if vals else None
    per = [point[a]["cst"] - point[b]["cst"] for a, b in pairs]
    d = delta(point, "cst")
    boots = np.array([v for v in (delta(r, "cst") for r in reps) if v is not None])
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    au = delta(point, "anti_uav")
    fails = []
    if len(pairs) < 2:
        fails.append("stopped after seed 0 (futility: seed-0 delta <= 0)" if per and per[0] <= 0 else "seed 1 missing")
    if not (d is not None and d > margin):
        fails.append(f"(1) mean delta {d:+.4f} <= {margin}")
    if not all(x > 0 for x in per):
        fails.append("(2) not positive for every seed")
    if not ci[0] > 0:
        fails.append("(3) CI includes 0")
    if au is not None and not au > -ni:
        fails.append(f"(4) Anti-UAV {au:+.4f} <= -{ni}")
    status = "SUPPORTED" if not fails else "NOT SUPPORTED"
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "B": B, "v2_selected": sel, "pairs": pairs, "per_seed_cst_delta": per,
           "mean_delta": d, "ci95": ci, "anti_uav_mean_delta": au, "micro_mean_delta": delta(point, "cst_micro") if "cst_micro" in point[pairs[0][0]] else None,
           "per_category": {c: delta(point, f"cst_{c}") for c in L.CST_CATEGORIES}, "point": {n: point[n] for n in have},
           "status": status, "failed": fails, "frame_set_sha256": ids.pop(),
           "adoption": "X1 and CF use n_std" if status == "SUPPORTED" else "X1 and CF stay on the as-built network"}
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# F1 — standard YOLO11n blocks vs the as-built network (corpus-v2 VAL)", "",
          f"*Generated {out['generated']} by `scripts/analyze_f1.py` (B = {B}); window from v2's selection: `{sel}`.*", "",
          "| pair | CST AP50 delta |", "|---|---:|"] + [f"| {a} vs {b} | {x:+.4f} |" for (a, b), x in zip(pairs, per)] + [
          "", f"Seed-mean delta {d:+.4f}, 95 % CI [{ci[0]:+.4f}, {ci[1]:+.4f}], Anti-UAV {au:+.4f} -> **{status}** "
          f"{'(' + '; '.join(fails) + ')' if fails else ''}", "", f"Adoption: {out['adoption']}."]
    OUT.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
