"""Frame-level failure categories for single-target tracking predictions (descriptive only: a category says WHAT happened, never why).

Categories (per scored frame; a frame can be in several):
    false_alarm          target absent (GT exist = 0) but a box is reported
    miss                 target visible but no box
    poor_localisation    target visible, box reported, IoU < 0.5
    low_confidence       box reported with score < 0.3 (only when the source records scores)
    sudden_disappearance GT visible at t-1 and absent at t (a box at t is then a false alarm)
    reappearance         GT absent at t-1 and visible at t
    size_tiny/small/normal/large   GT box diagonal (0,10], (10,30], (30,50], > 50 px  (CST SIZE convention, track_eval.CST_SIZE_BINS)
"""
import json
from pathlib import Path

import numpy as np

from .paths import REPO, STUDIO_RUNS

SIZE_BINS = (("tiny", 0, 10), ("small", 10, 30), ("normal", 30, 50), ("large", 50, np.inf))
CATEGORIES = ("false_alarm", "miss", "poor_localisation", "low_confidence", "sudden_disappearance", "reappearance")


def _iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0])) * max(0.0, min(ay2, by2) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - ix
    return ix / u if u > 0 else 0.0


def frame_table(res_xywh, scores, exist, gt_rect, start, processed=None):
    """-> list of per-frame dicts with flags; only frames > start that were processed are 'scored'."""
    out = []
    n = min(len(res_xywh), len(exist))
    for t in range(n):
        r = res_xywh[t]
        box = list(map(float, r)) if isinstance(r, (list, tuple)) and len(r) == 4 else None
        ex = bool(exist[t])
        g = gt_rect[t] if ex and isinstance(gt_rect[t], (list, tuple)) and len(gt_rect[t]) == 4 else None
        scored = t > start and (processed is None or bool(processed[t]))
        sc = None if scores is None or t >= len(scores) else (None if scores[t] is None else float(scores[t]))
        iou = _iou_xywh(box, g) if (box and g) else None
        diag = float(np.hypot(g[2], g[3])) if g else None
        f = {"frame": t, "scored": scored, "exist": ex, "box": box, "score": sc, "iou": iou, "diag": diag,
             "false_alarm": scored and not ex and box is not None, "miss": scored and ex and box is None,
             "poor_localisation": scored and iou is not None and iou < 0.5,
             "low_confidence": scored and box is not None and sc is not None and sc < 0.3,
             "sudden_disappearance": scored and t > 0 and bool(exist[t - 1]) and not ex,
             "reappearance": scored and t > 0 and not bool(exist[t - 1]) and ex}
        for name, lo, hi in SIZE_BINS:
            f[f"size_{name}"] = diag is not None and lo < diag <= hi
        out.append(f)
    return out


def summary(table) -> dict:
    sc = [f for f in table if f["scored"]]
    vis = [f for f in sc if f["exist"]]
    out = {"n_scored": len(sc), "n_visible": len(vis), "n_absent": len(sc) - len(vis)}
    for c in CATEGORIES:
        out[c] = sum(f[c] for f in sc)
    for name, _, _ in SIZE_BINS:
        fs = [f for f in vis if f[f"size_{name}"]]
        ious = [f["iou"] or 0.0 for f in fs]
        out[f"size_{name}"] = {"n": len(fs), "mean_iou_incl_misses": float(np.mean(ious)) if ious else None}
    return out


def prediction_sources() -> list[dict]:
    """recorded per-frame predictions that can be analysed (VAL only): studio evaluations + Direction B saved VAL predictions."""
    out = []
    for man_p in sorted((STUDIO_RUNS / "evals").glob("*/manifest.json")):
        try:
            man = json.loads(man_p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if man.get("status") != "completed" or man.get("evaluation_protocol_id", "").startswith("forecast"):
            continue
        for p in sorted((man_p.parent / "predictions").glob("*.json")):
            out.append({"source": f"studio:{man['experiment_id']}", "model_id": man["model_id"], "dataset_id": man["dataset_id"],
                        "split": man["split"], "modality": man.get("modality"), "sequence": p.stem, "path": p, "format": "studio"})
    for d in sorted((REPO / "experiments" / "anti_uav" / "runs").glob("*/eval_val_*_last/predictions")):
        run, mod = d.parent.parent.name, d.parent.name.split("_")[2]
        for p in sorted(d.glob("*.json")):
            out.append({"source": f"recorded:{run}/{d.parent.name}", "model_id": f"antiuav_repro.globaltrack:{run}@last", "dataset_id": "antiuav2021",
                        "split": "val", "modality": mod, "sequence": p.stem, "path": p, "format": "track_a"})
    return out


def load_prediction(src: dict) -> dict:
    """-> {res_xywh, scores, start_frame, processed}"""
    d = json.loads(Path(src["path"]).read_text(encoding="utf-8"))
    if src["format"] == "studio":
        return {"res_xywh": d["res_xywh"], "scores": [f["score"] for f in d["frames"]] + [None] * (len(d["res_xywh"]) - len(d["frames"])),
                "start_frame": d["start_frame"], "processed": d["processed"], "frames": d["frames"]}
    return {"res_xywh": d["res"], "scores": d.get("scores"), "start_frame": int(d["start_frame"]), "processed": d.get("processed"), "frames": None}
