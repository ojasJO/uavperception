#!/usr/bin/env python
"""EXPLORATORY (VAL only; not registered): where does the frozen STE lose state accuracy on CST VAL, and do absent-frame false
alarms have lower detection scores than correct track frames? (motivates a presence / output threshold decoupled from association)

    venv\\Scripts\\python.exe experiments\\cst_anti_uav\\scripts\\explore_presence.py --run v3_n__causal_g4_cst_s0_e20

Per CST VAL sequence, the STE runs with track_eval's cross-fitted settings for that sequence's half (the other half's choice, as
in the registered VAL design). Per frame: STE mode, associated detection score, GT visibility, IoU. Output:
experiments/cst_anti_uav/reports/explore/presence_<run>.json. Never reads TEST.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "aerotrack_v3" / "scripts"))
CORPUS = "data_unified_v2"


def _sys_module(name):
    import importlib.util
    p = REPO / "experiments" / "systems" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"systems_{name}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def run_frames(T, params, it):
    """SOT run of the STE -> per-frame (mode, score, box_xywh|None) from the start frame on."""
    s, dets = it["seq"], it["dets"]
    trk = T.STE(T.STEParams(**{**T.STEParams().as_dict(), **params}))
    st = s.start_frame
    init = s.box_xywh(st)
    trk.reset(np.array([init[0], init[1], init[0] + init[2], init[1] + init[3]], dtype=np.float64))
    out = [("init", 1.0, list(init))]
    for t in range(st + 1, s.n_frames):
        b, mode, sc = trk.step(*dets[t])
        out.append((mode, sc, None if b is None else [b[0], b[1], b[2] - b[0], b[3] - b[1]]))
    return st, out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    TE = _sys_module("track_eval")
    T = _sys_module("ste")
    rep = json.loads((REPO / "experiments" / "systems" / "reports" / f"track_{a.run}_{CORPUS}.json").read_text(encoding="utf-8"))
    pph = {int(h): p for h, p in rep["trackers"]["ste"]["params_per_half"].items()}
    _, seqs = TE.load(a.run, CORPUS, split="val")
    rows = []                                     # (seq, exist, mode, score, iou)
    for k, it in sorted(seqs.items()):
        if it["group"] != "cst_IR":
            continue
        st, out = run_frames(T, pph[it["half"]], it)
        s = it["seq"]
        for i, (mode, sc, box) in enumerate(out):
            t = st + i
            ex = bool(s.exist[t])
            iou = iou_xywh(box, s.gt_rect[t]) if (ex and box is not None and len(s.gt_rect[t]) == 4) else 0.0
            rows.append((k, ex, mode, float(sc), box is not None, iou))
    ex = np.array([r[1] for r in rows]); has = np.array([r[4] for r in rows]); iou = np.array([r[5] for r in rows])
    sc = np.array([r[3] for r in rows]); mode = np.array([r[2] for r in rows])
    n = len(rows)
    loss = {"visible_no_box": float((ex & ~has).sum()), "visible_box_iou0": float((ex & has & (iou == 0)).sum()),
            "visible_partial_1_minus_iou": float(((1 - iou) * (ex & has & (iou > 0))).sum()), "absent_false_alarm": float((~ex & has).sum())}
    tot = sum(loss.values())
    fa = ~ex & has
    tp = ex & has & (iou >= 0.5)
    wrong = ex & has & (iou == 0)
    q = lambda m: {p: float(np.percentile(sc[m], p)) for p in (10, 25, 50, 75, 90)} if m.any() else None  # noqa: E731
    res = {"run": a.run, "split": "val (CST, 40 sequences)", "frames_scored": n, "note": "EXPLORATORY, VAL only",
           "frame_SA_loss_share": {k2: v / tot for k2, v in loss.items()}, "frame_SA_loss_frames": loss,
           "false_alarm_modes": {m: int((fa & (mode == m)).sum()) for m in np.unique(mode[fa])} if fa.any() else {},
           "score_quantiles": {"absent_false_alarm": q(fa), "visible_correct_iou_ge_0_5": q(tp), "visible_wrong_iou0": q(wrong & (mode != "init"))},
           "auc_score_correct_vs_false_alarm": None}
    if fa.any() and tp.any():
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(sc[tp], sc[fa], alternative="two-sided").statistic
        res["auc_score_correct_vs_false_alarm"] = float(u / (tp.sum() * fa.sum()))
    out = REPO / "experiments" / "cst_anti_uav" / "reports" / "explore"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"presence_{a.run}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
