#!/usr/bin/env python
"""A1 — final descriptive analysis from files already on disk (no new inference, the test split is NOT scored again).

    python experiments/aerotrack_generalization/scripts/final_analysis.py

1. uncertainty of the single TEST result of the selected model: sequence-level, dataset-stratified bootstrap (B = 2000) of its
   recorded per-image predictions (results/runs/aerotrack_generalization/_cache/gen_records/<run>__test.pkl);
2. CST TEST by scene category (seen in TRAIN vs absent from TRAIN) for the selected model — absolute values only: stock YOLO11n's
   single test evaluation (repair series) kept aggregates, not per-image records, so no paired comparison is possible;
3. descriptive TEST reference: stock YOLO11n (repair series, evaluated once) on the metrics both series define identically;
4. official-VAL secondary deltas per fold (candidate - stock);
5. training curves of every series run (held-out-fold monitoring mAP50 and training loss over the last 5 epochs).
Output: reports/final_analysis.json (+ ledger line A1)."""
import csv
import json
import pickle
import re
from collections import defaultdict

import _series as S
import numpy as np

import evaluation as E

SELECTED = "gen_temporal_fp_g4_full_e20"
IOU = np.array([0.5])
B = 2000


def parts_by_seq(records, dataset=None, micro=False):
    lo, hi = E.SIZE_BINS["micro"]
    by = defaultdict(list)
    for r in records:
        if dataset and r["dataset"] != dataset:
            continue
        keep = ((r["gt_area"] >= lo) & (r["gt_area"] < hi)) if micro else None
        if micro and len(r["gt"]) and not keep.any() and len(r["pred"]) == 0:
            continue
        by[r["seq_key"]].append(E.match_image(r["pred"], r["gt"], IOU, keep))
    return {k: tuple(np.concatenate([x[i] for x in v], 0) for i in range(4)) for k, v in by.items()}


def ap0(parts):
    if not parts:
        return None
    tp, cf, pc, tc = (np.concatenate([p[i] for p in parts], 0) for i in range(4))
    res = E.ap_per_class(tp, cf, pc, tc, classes=(0,))
    return float(res["ap"][0, 0]) if res["n_gt"][0] > 0 else None


def boot(parts, strata, seed=0):
    rng = np.random.default_rng(seed)
    keys_by = defaultdict(list)
    for k in parts:
        keys_by[strata[k]].append(k)
    point = ap0(list(parts.values()))
    reps = []
    for _ in range(B):
        pick = [ks[i] for _, ks in sorted(keys_by.items()) for i in rng.integers(0, len(ks), len(ks))]
        reps.append(ap0([parts[k] for k in pick]))
    reps = np.array([x for x in reps if x is not None])
    return {"point": point, "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], "sequences": len(parts)}


def main():
    out = {"selected": SELECTED}
    d = pickle.load(open(S.CACHE / "gen_records" / f"{SELECTED}__test.pkl", "rb"))
    recs = d["records"]
    strata = {r["seq_key"]: r["dataset"] for r in recs}
    out["test_set"] = d["identity"]
    out["test_bootstrap"] = {
        "cst_drone_AP50": boot(parts_by_seq(recs, "cst"), strata),
        "cst_drone_micro_AP50": boot(parts_by_seq(recs, "cst", micro=True), strata),
        "anti_uav_drone_AP50": boot(parts_by_seq(recs, "anti_uav"), strata),
        "pooled_drone_AP50": boot(parts_by_seq(recs), strata)}
    cat = lambda k: re.match(r"cst_[a-z]+_(.+)_(\d+)$", k).group(1)
    train_cats = set(json.loads((S.REPORTS / "diag_data.json").read_text(encoding="utf-8"))["cst"]["scene_categories"]["train"])
    cst = parts_by_seq(recs, "cst")
    by_cat = defaultdict(dict)
    for k, v in cst.items():
        by_cat[cat(k)][k] = v
    out["test_cst_by_category"] = {c: {"seen_in_train": c in train_cats, "sequences": len(v), "drone_AP50": ap0(list(v.values()))} for c, v in sorted(by_cat.items())}
    seen = {k: v for c, kv in by_cat.items() if c in train_cats for k, v in kv.items()}
    unseen = {k: v for c, kv in by_cat.items() if c not in train_cats for k, v in kv.items()}
    out["test_cst_seen_vs_unseen"] = {"seen_categories_AP50": ap0(list(seen.values())), "seen_sequences": len(seen),
                                      "unseen_categories_AP50": ap0(list(unseen.values())), "unseen_sequences": len(unseen)}
    rep = json.loads((S.REPO / "experiments" / "aerotrack_repair" / "reports" / "detector_test_cap5000.json").read_text(encoding="utf-8"))
    assert rep["eval_set"]["sha256"] == d["identity"]["registered_sha256"], "different test image sets"
    sel = [json.loads(x) for x in (S.REPORTS / "gen_test_ledger.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(sel) == 1 and sel[0]["run"] == SELECTED
    m = sel[0]["metrics"]
    same = {"drone_AP50": "drone_AP50", "cst_drone_AP50": "cst_AP50", "cst_drone_micro_AP50": "cst_micro_AP50",
            "anti_uav_drone_AP50": "anti_uav_AP50", "mAP50": "mAP50"}           # identical definitions in both series
    ref = {}
    for r in rep["rows"]:
        if r["run"] == "repair_abl_stock_yolo11n_e20":
            ref[r["checkpoint"]] = {k: r[v] for k, v in same.items()}
    out["test_reference_stock_yolo11n"] = {"source": "experiments/aerotrack_repair/reports/detector_test_cap5000.json (evaluated once, 2026-09-26)",
                                           "selected": {k: m[k] for k in same}, "stock": ref,
                                           "delta_vs_stock_last": {k: m[k] - ref["last"][k] for k in same},
                                           "note": "different training runs of the same recipe (not seed-paired); descriptive only"}
    res = {(r["run"], r["set"]): r for r in json.loads((S.REPORTS / "gen_results.json").read_text(encoding="utf-8"))["rows"]}
    out["val_secondary_deltas"] = {c: {k: {mm: res[(f"gen_{c}_f{k}_e20", "val")]["metrics"][mm] - res[(f"gen_stock_f{k}_e20", "val")]["metrics"][mm]
                                           for mm in ("cst_drone_AP50", "anti_uav_drone_AP50", "drone_AP50")} for k in (0, 1, 2)}
                                   for c in ("spd_fp", "temporal_fp_g1", "temporal_fp_g4")}
    curves = {}
    for rd in sorted(S.REPO.joinpath("results/runs").glob("gen_*_e20")):
        rows = list(csv.DictReader(open(rd / "results.csv", encoding="utf-8")))
        m50 = [float(x["mAP50"]) for x in rows]
        loss = [float(x["box_loss"]) + float(x["cls_loss"]) + float(x["dfl_loss"]) for x in rows]
        slope = lambda y: float(np.polyfit(np.arange(5), y[-5:], 1)[0])
        curves[rd.name] = {"epochs": len(rows), "monitor_mAP50_last": m50[-1], "monitor_mAP50_slope_last5": slope(m50),
                           "train_loss_last": loss[-1], "train_loss_slope_last5": slope(loss)}
    out["training_curves"] = curves
    p = S.write_report("final_analysis.json", out)
    S.ledger({"id": "A1", "type": "analysis", "what": "final descriptive analysis (no new inference; test not re-scored)",
              "script": "scripts/final_analysis.py", "output": str(p.relative_to(S.REPO)), "test_usage": "recorded predictions of the single test evaluation"})
    print(json.dumps({k: out[k] for k in ("test_bootstrap", "test_cst_seen_vs_unseen", "test_cst_by_category", "test_reference_stock_yolo11n")}, indent=1))
    print("val secondary deltas:", json.dumps(out["val_secondary_deltas"])[:900])
    print("curve slopes (monitor mAP50 per epoch, last 5):", {k: round(v["monitor_mAP50_slope_last5"], 4) for k, v in curves.items()})
    print("train-loss slopes (last 5):", {k: round(v["train_loss_slope_last5"], 4) for k, v in curves.items()})


if __name__ == "__main__":
    main()
