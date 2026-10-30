#!/usr/bin/env python
"""D4 — how much of the grid's VAL differences is sampling noise? Paired, dataset-stratified, sequence-level (cluster) bootstrap
of the pre-registered comparisons, from the D3 per-image records, using evaluation.match_image / ap_per_class at IoU 0.5 so the
point estimates reproduce the registered numbers.

    venv\\Scripts\\python.exe experiments\\aerotrack_generalization\\scripts\\bootstrap_val.py [B=1000]

What this does NOT measure: training (seed) noise -- each configuration was trained once. Output: reports/bootstrap_val.json."""
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool

import _series as S
import numpy as np

import evaluation as E

D = S.CACHE / "val_records"
IOU = np.array([0.5])
LO, HI = E.SIZE_BINS["micro"]
METRICS = ("drone_AP50", "micro_AP50", "cst_AP50", "cst_micro_AP50")
COMPS = [("SPD-Conv", "baseline_ctrl", "no_spd", "micro_AP50"), ("SPD-Conv", "baseline_ctrl", "no_spd", "cst_micro_AP50"),
         ("temporal stack", "baseline_ctrl", "single_frame", "cst_AP50"), ("temporal stack", "baseline_ctrl", "single_frame", "drone_AP50"),
         ("full recipe vs stock YOLO11n", "baseline_ctrl", "stock_yolo11n", "drone_AP50"),
         ("[exploratory] temporal within the stride layout", "no_spd", "stock_yolo11n", "cst_AP50")]
G = {}


def prep(path):
    d = pickle.load(open(path, "rb"))
    out = {m: defaultdict(list) for m in ("all", "micro")}
    ds_of = {}
    for r in d["records"]:
        ds_of[r["seq_key"]] = r["dataset"]
        out["all"][r["seq_key"]].append(E.match_image(r["pred"], r["gt"], IOU, None))
        keep = (r["gt_area"] >= LO) & (r["gt_area"] < HI)
        if not (len(r["gt"]) and not keep.any() and len(r["pred"]) == 0):   # evaluation.score's skip rule
            out["micro"][r["seq_key"]].append(E.match_image(r["pred"], r["gt"], IOU, keep))
    cat = {m: {k: tuple(np.concatenate([x[i] for x in v], 0) for i in range(4)) for k, v in by.items()} for m, by in out.items()}
    return cat, ds_of


def ap(parts):
    tp, cf, pc, tc = (np.concatenate([p[i] for p in parts], 0) for i in range(4))
    res = E.ap_per_class(tp, cf, pc, tc, classes=(0, 1))
    present = [i for i, c in enumerate(res["classes"]) if res["n_gt"][i] > 0]
    drone = next((float(res["ap"][i, 0]) for i, c in enumerate(res["classes"]) if c == 0 and res["n_gt"][i] > 0), None)
    return float(res["ap"][present, 0].mean()) if present else 0.0, drone


def metrics(cat, ds_of, seqs):
    cst = [k for k in seqs if ds_of.get(k) == "cst"]
    return {"drone_AP50": ap([cat["all"][k] for k in seqs if k in cat["all"]])[1],
            "micro_AP50": ap([cat["micro"][k] for k in seqs if k in cat["micro"]])[0],
            "cst_AP50": ap([cat["all"][k] for k in cst if k in cat["all"]])[0],
            "cst_micro_AP50": ap([cat["micro"][k] for k in cst if k in cat["micro"]])[0]}


def init(files):
    for f in files:
        G[f.stem] = prep(f)


def one(seed):
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for k, d in next(iter(G.values()))[1].items():
        strata[d].append(k)
    sample = [ks[i] for _, ks in sorted(strata.items()) for i in rng.integers(0, len(ks), len(ks))]
    return {n: metrics(v[0], v[1], sample) for n, v in G.items()}


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    files = sorted(D.glob("*.pkl"))
    assert len(files) == 8, f"expected 8 record files in {D}, found {len(files)} (run collect_val_records.py)"
    init(files)
    point = {n: metrics(v[0], v[1], list(v[1])) for n, v in G.items()}
    with Pool(6, initializer=init, initargs=(files,)) as pool:
        reps = pool.map(one, range(B), chunksize=10)
    out = {"B": B, "resampling": "sequences with replacement, stratified by dataset, paired across models", "point": point,
           "per_model_ci95": {n: {m: [float(np.percentile([r[n][m] for r in reps], q)) for q in (2.5, 97.5)] for m in METRICS} for n in point},
           "comparisons": [], "caveat": "sampling noise over VAL sequences only; training/seed noise not included (one run per config)"}
    for factor, full, abl, m in COMPS:
        for ck in ("best", "last"):
            a, b = f"repair_abl_{full}_e20__{ck}", f"repair_abl_{abl}_e20__{ck}"
            d = np.array([r[a][m] - r[b][m] for r in reps])
            out["comparisons"].append({"factor": factor, "full": a, "ablated": b, "metric": m, "delta": point[a][m] - point[b][m],
                                       "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                       "p_delta_gt_0.02": float((d > 0.02).mean()), "p_delta_lt_0": float((d < 0).mean())})
    p = S.write_report("bootstrap_val.json", out)
    S.ledger({"id": "D4", "type": "diagnostic", "what": f"paired sequence-level bootstrap of the grid's VAL comparisons (B={B})",
              "script": "scripts/bootstrap_val.py", "output": str(p.relative_to(S.REPO)), "test_usage": "none"})
    for c in out["comparisons"]:
        print(f"{c['factor'][:40]:40s} {c['ablated'].split('__')[1]:4s} {c['metric']:15s} {c['delta']:+.4f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
