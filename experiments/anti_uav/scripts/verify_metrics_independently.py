#!/usr/bin/env python
"""INDEPENDENT re-implementation of the Anti-UAV metrics, used to audit the reported numbers.

    python experiments/anti_uav/scripts/verify_metrics_independently.py \
        --bench-dir experiments/anti_uav/runs/paper_benchmarks_20260922T193853

This script deliberately imports NOTHING from `anti_uav_repro`. It re-reads:
  * the raw dataset labels      data/anti_uav/<split>/<seq>/infrared.json   (keys: exist, gt_rect)
  * the raw attribute lists     data/anti_uav/label_new/<split>.json
  * the raw tracker predictions <bench>/<run>__<split>_IR/predictions/<seq>.json (key: res)
and recomputes state accuracy straight from the paper's Eq. (1)

    SA = (1/T) * sum_t [ IoU_t * 1(v_t > 0) + p_t * 1(v_t == 0) ]

with IoU written out from scratch in the challenge reference's x/y/w/h convention, plus OTB precision
(centre error <= 20 px) and success (AUC over IoU thresholds 0, .05, ..., 1). It then diffs its own
numbers against the `metrics.json` the pipeline produced.

A disagreement means the reported numbers are wrong. Agreement means two independent implementations
of the paper's formula, reading the same raw files, land on the same value.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LABELS = REPO / "data" / "anti_uav"
TOL = 1e-9


def jload(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def iou_xywh(a, b) -> float:
    """From scratch, challenge convention: boxes are [x, y, w, h], x1 = x + w."""
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def centre_error(a, b) -> float:
    return math.hypot((a[0] + a[2] / 2.0) - (b[0] + b[2] / 2.0),
                      (a[1] + a[3] / 2.0) - (b[1] + b[3] / 2.0))


def is_box(p) -> bool:
    return isinstance(p, (list, tuple)) and len(p) == 4


def score_sequence(res: list, exist: list, gt: list, start: int) -> dict:
    """Paper Eq. (1) over frames t >= start (the frames the tracker actually tracked)."""
    thresholds = [i * 0.05 for i in range(21)]
    sa_terms, ious, ces = [], [], []
    for t in range(start, len(exist)):
        p = res[t]
        if exist[t]:
            if is_box(p) and is_box(gt[t]):
                v = iou_xywh(p, gt[t])
                ious.append(v)
                ces.append(centre_error(p, gt[t]))
            else:
                v = 0.0
                ious.append(0.0)
                ces.append(float("inf"))
            sa_terms.append(v)
        else:
            # p_t = 1 only when the tracker declared absence (emitted a non-box)
            sa_terms.append(0.0 if is_box(p) else 1.0)
    n = len(sa_terms)
    sa = sum(sa_terms) / n if n else float("nan")
    prec = (sum(1 for c in ces if c <= 20.0) / len(ces)) if ces else float("nan")
    if ious:
        curve = [sum(1 for v in ious if v >= th) / len(ious) for th in thresholds]
        succ = sum(curve) / len(curve)
    else:
        succ = float("nan")
    return {"SA": sa, "precision": prec, "success": succ,
            "n_frames": n, "n_exist": len(ious), "n_absent": n - len(ious)}


def audit(bench: Path) -> dict:
    attrs_cache: dict[str, dict] = {}
    report: dict = {}
    for d in sorted(bench.glob("*__*_IR")) + sorted(bench.glob("*/*__*_IR")):
        mp, pd = d / "metrics.json", d / "predictions"
        if not mp.exists() or not pd.is_dir():
            continue
        reported = jload(mp)
        split = reported["meta"]["split"]
        if split not in attrs_cache:
            attrs_cache[split] = jload(LABELS / "label_new" / f"{split}.json")
        attributes = attrs_cache[split]

        per_seq, worst_seq, worst_d = {}, None, 0.0
        for pf in sorted(pd.glob("*.json")):
            name = pf.stem
            pred = jload(pf)
            lab = jload(LABELS / split / name / "infrared.json")
            mine = score_sequence(pred["res"], lab["exist"], lab["gt_rect"], int(pred["start_frame"]))
            theirs = reported["per_sequence"][name]
            dsa = abs(mine["SA"] - theirs["SA"])
            if dsa > worst_d:
                worst_d, worst_seq = dsa, name
            per_seq[name] = mine
            # structural cross-checks against the pipeline's own bookkeeping
            assert mine["n_frames"] == theirs["n_frames"], f"{name}: frame count mismatch"
            assert mine["n_exist"] == theirs["n_exist"], f"{name}: exist count mismatch"

        names = list(per_seq)
        my_msa = 100 * sum(per_seq[n]["SA"] for n in names) / len(names)
        my_prec = 100 * sum(per_seq[n]["precision"] for n in names) / len(names)
        my_succ = 100 * sum(per_seq[n]["success"] for n in names) / len(names)
        agg = reported["aggregate"]

        by_attr = {}
        all_attrs = sorted({a for n in names for a in attributes.get(n, [])})
        for att in all_attrs:
            mem = [n for n in names if att in attributes.get(n, [])]
            mine_v = 100 * sum(per_seq[n]["SA"] for n in mem) / len(mem)
            rep_v = agg["by_attribute"].get(att, {}).get("mSA_percent")
            by_attr[att] = {"n_sequences": len(mem), "independent": mine_v, "reported": rep_v,
                            "abs_diff": None if rep_v is None else abs(mine_v - rep_v)}

        key = str(d.relative_to(bench)).replace("\\", "/")
        report[key] = {
            "n_sequences": len(names),
            "mSA": {"independent": my_msa, "reported": agg["mSA_percent"],
                    "abs_diff": abs(my_msa - agg["mSA_percent"])},
            "precision_20px": {"independent": my_prec, "reported": 100 * agg["precision_20px"],
                               "abs_diff": abs(my_prec - 100 * agg["precision_20px"])},
            "success_auc": {"independent": my_succ, "reported": 100 * agg["success_auc"],
                            "abs_diff": abs(my_succ - 100 * agg["success_auc"])},
            "worst_per_sequence_SA_abs_diff": worst_d,
            "worst_sequence": worst_seq,
            "by_attribute": by_attr,
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--tolerance", type=float, default=1e-9)
    args = ap.parse_args()
    bench = Path(args.bench_dir)
    if not bench.is_absolute():
        bench = REPO / bench

    rep = audit(bench)
    if not rep:
        print("no evaluations found", file=sys.stderr)
        return 2

    ok = True
    print(f"{'evaluation':42s} {'mSA ind.':>9s} {'mSA rep.':>9s} {'|d|':>10s}  {'prec |d|':>9s} {'succ |d|':>9s} {'worst seq |d|':>13s}")
    for k, v in rep.items():
        d = v["mSA"]["abs_diff"]
        bad = max(d, v["precision_20px"]["abs_diff"], v["success_auc"]["abs_diff"],
                  v["worst_per_sequence_SA_abs_diff"])
        attr_bad = max([x["abs_diff"] for x in v["by_attribute"].values() if x["abs_diff"] is not None] or [0.0])
        bad = max(bad, attr_bad)
        ok &= bad <= args.tolerance
        print(f"{k:42s} {v['mSA']['independent']:9.5f} {v['mSA']['reported']:9.5f} {d:10.2e}  "
              f"{v['precision_20px']['abs_diff']:9.2e} {v['success_auc']['abs_diff']:9.2e} "
              f"{v['worst_per_sequence_SA_abs_diff']:13.2e}")
    out = bench / "metric_independent_audit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tolerance": args.tolerance, "all_within_tolerance": bool(ok), "results": rep}, f, indent=2)
    print()
    print(f"all differences within {args.tolerance:g}: {'YES' if ok else 'NO'}")
    print("wrote", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
