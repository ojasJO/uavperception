"""Anti-UAV metrics — computed ONLY from saved tracker outputs and ground-truth labels.

State accuracy (paper Eq. 1, challenge `Eval/test.py::eval`):
    SA = mean_t [ IoU_t * 1[v_t > 0] + p_t * 1[v_t == 0] ]
        v_t : GT exist flag;  p_t = 1 if the tracker output an EMPTY box (absent), else 0.
    mSA = mean over sequences.
Precision / success as in OTB (Wu et al. 2013): precision = fraction of frames with centre error <= 20 px,
success = AUC of the success-rate vs IoU-threshold curve (thresholds 0, .05, ..., 1). These two are computed
over frames with exist == 1 (convention documented in the spec §6; the paper does not specify).

A prediction is a per-frame list: [x, y, w, h] (box) or a list of length != 4 (e.g. [0] or []) meaning "absent".
Frames before the tracker's start frame are stored as absent by the tracker.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

ATTRIBUTE_ORDER = ["OV", "OC", "FM", "SV", "LI", "TC-EASY", "TC-MID", "TC-HARD", "TC", "LR"]  # names as in label_new/*.json
SUCCESS_THRESHOLDS = np.arange(0.0, 1.05, 0.05)
PRECISION_THRESHOLD_PX = 20.0


def iou_xywh(a, b) -> float:
    """Faithful to the challenge reference `iou()` (x0 + w convention; 0 when no positive overlap)."""
    x0_1, y0_1, w1, h1 = [float(v) for v in a]
    x0_2, y0_2, w2, h2 = [float(v) for v in b]
    x1_1, y1_1 = x0_1 + w1, y0_1 + h1
    x1_2, y1_2 = x0_2 + w2, y0_2 + h2
    ox0, oy0 = max(x0_1, x0_2), max(y0_1, y0_2)
    ox1, oy1 = min(x1_1, x1_2), min(y1_1, y1_2)
    if ox1 - ox0 <= 0 or oy1 - oy0 <= 0:
        return 0.0
    inter = (ox1 - ox0) * (oy1 - oy0)
    union = w1 * h1 + w2 * h2 - inter
    return float(inter / union) if union > 0 else 0.0


def is_box(pred) -> bool:
    return isinstance(pred, (list, tuple, np.ndarray)) and len(pred) == 4


def center_error(a, b) -> float:
    ax, ay = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
    bx, by = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
    return float(np.hypot(ax - bx, ay - by))


def per_frame_scores(preds: list, exist: list[int], gt_rect: list) -> dict:
    """Per-frame arrays for one sequence. Lengths must match the label length."""
    n = len(exist)
    if len(preds) != n:
        raise ValueError(f"prediction length {len(preds)} != label length {n}")
    sa = np.zeros(n, dtype=np.float64)
    iou = np.full(n, np.nan)
    ce = np.full(n, np.nan)
    for t in range(n):
        p = preds[t]
        if exist[t]:
            gt = gt_rect[t]
            if is_box(p) and is_box(gt):
                iou[t] = iou_xywh(p, gt)
                ce[t] = center_error(p, gt)
                sa[t] = iou[t]
            else:
                iou[t] = 0.0
                ce[t] = np.inf
                sa[t] = 0.0
        else:
            sa[t] = 0.0 if is_box(p) else 1.0
    return {"sa": sa, "iou": iou, "ce": ce, "exist": np.asarray(exist, dtype=bool)}


def sequence_metrics(preds: list, exist: list[int], gt_rect: list, mask=None) -> dict:
    """mask: optional boolean array selecting the frames to score (used only by the training-time
    monitor, which tracks a frame-strided subset). Full evaluations pass mask=None (all frames)."""
    f = per_frame_scores(preds, exist, gt_rect)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape[0] != len(exist):
            raise ValueError("mask length mismatch")
        f = {k: v[m] for k, v in f.items()}
    ex = f["exist"]
    iou_e = f["iou"][ex]
    ce_e = f["ce"][ex]
    success_curve = np.array([(iou_e >= th).mean() if len(iou_e) else np.nan for th in SUCCESS_THRESHOLDS])
    return {
        "SA": float(f["sa"].mean()),
        "n_frames": int(len(ex)),
        "n_exist": int(ex.sum()),
        "n_absent": int((~ex).sum()),
        "absent_frames_scored_correct": int(((~ex) & (f["sa"] == 1.0)).sum()),
        "mean_iou_exist": float(np.nanmean(iou_e)) if len(iou_e) else float("nan"),
        "precision_20px": float((ce_e <= PRECISION_THRESHOLD_PX).mean()) if len(ce_e) else float("nan"),
        "success_auc": float(np.nanmean(success_curve)) if len(iou_e) else float("nan"),
        "success_curve": success_curve.tolist(),
    }


def aggregate(seq_results: "OrderedDict[str, dict]", attributes: dict[str, list[str]]) -> dict:
    """seq_results: name -> sequence_metrics(...); attributes: name -> list of attribute strings."""
    names = list(seq_results)
    sa = np.array([seq_results[n]["SA"] for n in names])
    prec = np.array([seq_results[n]["precision_20px"] for n in names])
    succ = np.array([seq_results[n]["success_auc"] for n in names])
    out = {
        "n_sequences": len(names),
        "mSA": float(sa.mean()) if len(sa) else float("nan"),
        "mSA_percent": float(100 * sa.mean()) if len(sa) else float("nan"),
        "precision_20px": float(np.nanmean(prec)) if len(prec) else float("nan"),
        "success_auc": float(np.nanmean(succ)) if len(succ) else float("nan"),
        "frames_total": int(sum(seq_results[n]["n_frames"] for n in names)),
        "frames_exist": int(sum(seq_results[n]["n_exist"] for n in names)),
        "by_attribute": {},
    }
    all_attrs = set(ATTRIBUTE_ORDER)
    for n in names:
        all_attrs.update(attributes.get(n, []))
    for attr in ATTRIBUTE_ORDER + sorted(all_attrs - set(ATTRIBUTE_ORDER)):
        members = [n for n in names if attr in attributes.get(n, [])]
        if not members:
            out["by_attribute"][attr] = {"n_sequences": 0, "mSA_percent": None}
            continue
        vals = np.array([seq_results[n]["SA"] for n in members])
        out["by_attribute"][attr] = {"n_sequences": len(members), "mSA_percent": float(100 * vals.mean())}
    return out
