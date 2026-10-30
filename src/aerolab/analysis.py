"""Analyses behind the Flight Deck result views. Descriptive, rule-based, reproducible; every rule is written here once.

Tracking (single target, per sequence; ground truth available, so every episode below is annotation-confirmed as to WHAT happened,
never as to WHY):
    frame status    init | tracked (IoU >= 0.5) | poor (0 < IoU < 0.5) | wrong_object (box with IoU 0 on a visible target) |
                    miss (visible, no box) | false_alarm (absent, box) | correct_absent (absent, no box) | not_processed
    SA / P@20 / success AUC   Track A's anti_uav_repro.eval.metrics.sequence_metrics (= the official CST toolkit per sequence)
    CLE             centre-location error (px) on visible frames that have a box
    failure interval  >= MIN_FAIL consecutive visible frames with IoU < 0.5 (no box counts as IoU 0)
        sudden      the IoU fell by >= 0.4 between the last tracked frame and the first failed frame
        drift       otherwise, if the IoU slope over the 10 visible frames before the interval is <= -0.02 / frame
        wrong_target  >= 50 % of the interval's frames are wrong_object; no_output: >= 50 % are misses
    recovery        frames from the end of a failure interval or an absence episode to the next tracked frame
    loss event      tracked -> not tracked while the target stays visible

Presence (frame level, TRAIN / VAL only): positive = target visible (exist = 1). The detector's top drone-class score at a threshold
gives a presence decision; the tracker's reported box is a second, fixed decision. Neither score is a calibrated probability.
"""
import numpy as np

MIN_FAIL = 5
SA_STATUS = ("init", "tracked", "poor", "wrong_object", "miss", "false_alarm", "correct_absent", "not_processed")


def _metrics_module():
    import sys
    from .paths import REPO
    p = str(REPO / "experiments" / "anti_uav" / "src")
    if p not in sys.path:
        sys.path.insert(0, p)
    from anti_uav_repro.eval import metrics
    return metrics


def iou_xywh(a, b):
    if not (isinstance(a, (list, tuple)) and len(a) == 4 and isinstance(b, (list, tuple)) and len(b) == 4):
        return None
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - ix
    return ix / u if u > 0 else 0.0


def frame_table(res_xywh, exist, gt_rect, start, end=None, processed=None):
    """-> list of per-frame dicts (frame, status, iou, cle, has_box, visible) for frames 0..end-1"""
    n = min(len(res_xywh), len(exist)) if end is None else min(end, len(res_xywh), len(exist))
    rows = []
    for t in range(n):
        r = res_xywh[t]
        box = list(r) if isinstance(r, (list, tuple)) and len(r) == 4 else None     # Track A is_box: any 4-vector is a reported box
        vis = bool(exist[t]) and isinstance(gt_rect[t], (list, tuple)) and len(gt_rect[t]) == 4
        g = gt_rect[t] if vis else None
        iou = iou_xywh(box, g) if (box and g) else None
        cle = (float(np.hypot(box[0] + box[2] / 2 - g[0] - g[2] / 2, box[1] + box[3] / 2 - g[1] - g[3] / 2)) if (box and g) else None)
        if t < start or (processed is not None and t < len(processed) and not processed[t]):
            st = "not_processed"
        elif t == start:
            st = "init"
        elif vis:
            st = "miss" if box is None else ("tracked" if iou >= 0.5 else ("wrong_object" if iou == 0 else "poor"))
        else:
            st = "false_alarm" if box is not None else "correct_absent"
        rows.append({"frame": t, "status": st, "iou": iou, "cle": cle, "has_box": box is not None, "visible": vis})
    return rows


def sequence_summary(res_xywh, exist, gt_rect, start, end=None, processed=None) -> dict:
    """Track A metrics over frames start..end-1 + the episode analysis."""
    M = _metrics_module()
    n = min(len(res_xywh), len(exist)) if end is None else min(end, len(res_xywh), len(exist))
    proc = np.ones(n, bool) if processed is None else np.asarray(list(processed[:n]) + [False] * (n - len(processed[:n])), bool)
    mask = proc & (np.arange(n) >= start)
    res = [r if isinstance(r, (list, tuple)) and len(r) == 4 else [] for r in res_xywh[:n]]
    m = M.sequence_metrics(res, list(exist[:n]), list(gt_rect[:n]), mask=mask)
    rows = frame_table(res_xywh, exist, gt_rect, start, n, processed)
    sc = [r for r in rows if r["status"] not in ("not_processed", "init")]
    vis = [r for r in sc if r["visible"]]
    ab = [r for r in sc if not r["visible"]]
    cles = [r["cle"] for r in vis if r["cle"] is not None]
    ious = [r["iou"] or 0.0 for r in vis]
    counts = {s: sum(r["status"] == s for r in rows) for s in SA_STATUS}
    nan = lambda v: None if v is None or v != v else float(v)  # noqa: E731
    return {"SA": 100 * m["SA"], "precision_20px": nan(100 * m["precision_20px"]) if m["precision_20px"] == m["precision_20px"] else None,
            "success_auc": nan(100 * m["success_auc"]) if m["success_auc"] == m["success_auc"] else None,
            "mean_iou_visible": float(np.mean(ious)) if ious else None, "mean_cle_px": float(np.mean(cles)) if cles else None,
            "median_cle_px": float(np.median(cles)) if cles else None,
            "false_alarm_rate": (counts["false_alarm"] / len(ab)) if ab else None,
            "located_rate": (sum(r["has_box"] for r in vis) / len(vis)) if vis else None,
            "n_scored": len(sc) + int(start < n), "n_visible": len(vis), "n_absent": len(ab), "status_counts": counts,
            "episodes": episodes(rows), "full_sequence": n == len(exist), "end_frame": n}


def episodes(rows) -> dict:
    """failure intervals (typed), loss events, recoveries -- rules in the module docstring"""
    vis_rows = [r for r in rows if r["status"] not in ("not_processed", "init")]
    fails, cur = [], []
    for r in vis_rows:                      # an absent frame or a tracked frame closes the current interval
        if r["visible"] and r["status"] != "tracked":
            cur.append(r)
            continue
        if len(cur) >= MIN_FAIL:
            fails.append(cur)
        cur = []
    if len(cur) >= MIN_FAIL:
        fails.append(cur)
    by_frame = {r["frame"]: r for r in rows}
    out = []
    for seg in fails:
        f0, f1 = seg[0]["frame"], seg[-1]["frame"]
        prev = [by_frame[t] for t in range(max(0, f0 - 30), f0) if t in by_frame and by_frame[t]["visible"] and by_frame[t]["status"] != "not_processed"][-10:]
        prev_iou = [(p["iou"] or 0.0) for p in prev]
        first = seg[0]["iou"] or 0.0
        kind = "sudden"
        if prev_iou and prev_iou[-1] - first < 0.4:
            slope = np.polyfit(np.arange(len(prev_iou)), prev_iou, 1)[0] if len(prev_iou) >= 3 else 0.0
            kind = "drift" if slope <= -0.02 else "unstable"
        if not prev_iou:
            kind = "from_start"
        wrong = sum(r["status"] == "wrong_object" for r in seg) / len(seg)
        miss = sum(r["status"] == "miss" for r in seg) / len(seg)
        after = next((t for t in range(f1 + 1, max(by_frame) + 1) if by_frame.get(t, {}).get("status") == "tracked"), None)
        out.append({"start": f0, "end": f1, "length": len(seg), "onset": kind,
                    "dominant": "wrong_target" if wrong >= 0.5 else ("no_output" if miss >= 0.5 else "poor_overlap"),
                    "wrong_object_share": round(wrong, 3), "miss_share": round(miss, 3),
                    "recovered_at": after, "frames_to_recover": None if after is None else after - f1})
    absences, cur = [], []
    for r in vis_rows:
        if not r["visible"]:
            cur.append(r)
        elif cur:
            absences.append(cur)
            cur = []
    if cur:
        absences.append(cur)
    abs_out = []
    for seg in absences:
        f0, f1 = seg[0]["frame"], seg[-1]["frame"]
        after = next((t for t in range(f1 + 1, max(by_frame) + 1) if by_frame.get(t, {}).get("status") == "tracked"), None)
        abs_out.append({"start": f0, "end": f1, "length": len(seg), "false_alarm_share": round(sum(r["status"] == "false_alarm" for r in seg) / len(seg), 3),
                        "target_returns_at": f1 + 1 if (f1 + 1) in by_frame else None,
                        "reacquired_at": after, "frames_to_reacquire": None if after is None else after - (f1 + 1)})
    losses = sum(1 for a, b in zip(vis_rows, vis_rows[1:]) if a["status"] == "tracked" and b["visible"] and b["status"] != "tracked")
    return {"failure_intervals": out, "absence_episodes": abs_out, "loss_events": losses,
            "rules": "failure interval >= 5 visible frames with IoU < 0.5; onset sudden if IoU dropped >= 0.4 at the first failed frame, drift "
                     "if the IoU slope over the previous 10 visible frames <= -0.02 / frame; dominant = wrong_target / no_output if >= 50 % of its frames"}


# ------------------------------------------------------------------------------------------------------------------- presence
def presence_frames(frames, exist, gt_rect, start, end):
    """-> arrays over scored frames (start < t < end): visible, top score (0 if no detection), top-box IoU, tracker has-box"""
    vis, score, top_iou, trk = [], [], [], []
    for f in frames:
        t = f["frame"]
        if t <= start or t >= end:
            continue
        v = bool(exist[t]) and isinstance(gt_rect[t], (list, tuple)) and len(gt_rect[t]) == 4
        dets = f.get("detections") or []
        s = f.get("score")
        vis.append(v)
        score.append(0.0 if s is None else float(s))
        if v and dets:
            d = dets[0]
            top_iou.append(iou_xywh([d[0], d[1], d[2] - d[0], d[3] - d[1]], gt_rect[t]) or 0.0)
        else:
            top_iou.append(0.0)
        trk.append(f.get("box_xyxy") is not None)
    return {"visible": np.array(vis, bool), "score": np.array(score, float), "top_iou": np.array(top_iou, float), "tracker_box": np.array(trk, bool)}


def presence_curve(P, thresholds=None) -> list[dict]:
    th = np.round(np.arange(0.0, 1.0001, 0.01), 2) if thresholds is None else np.asarray(thresholds, float)
    v, s, i = P["visible"], P["score"], P["top_iou"]
    nv, na = int(v.sum()), int((~v).sum())
    out = []
    for t in th:
        pos = s >= t if t > 0 else np.ones_like(v)
        tp, fp = int((pos & v).sum()), int((pos & ~v).sum())
        out.append({"threshold": float(t), "false_alarm_rate": fp / na if na else None, "miss_rate": (nv - tp) / nv if nv else None,
                    "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / nv if nv else None,
                    "correct_localisation_rate": float((pos & v & (i >= 0.5)).sum() / nv) if nv else None, "tp": tp, "fp": fp,
                    "fn": nv - tp, "tn": na - fp})
    return out


def presence_auc(P) -> float | None:
    """ROC AUC of the top score for visible vs absent frames (Mann-Whitney), None if one class is empty"""
    v, s = P["visible"], P["score"]
    if v.all() or (~v).all():
        return None
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = int(v.sum()), int((~v).sum())
    return float((r[v].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tracker_presence(P) -> dict:
    v, b = P["visible"], P["tracker_box"]
    nv, na = int(v.sum()), int((~v).sum())
    return {"false_alarm_rate": float((b & ~v).sum() / na) if na else None, "miss_rate": float((~b & v).sum() / nv) if nv else None,
            "n_visible": nv, "n_absent": na}


# ------------------------------------------------------------------------------------------------------------------- seeds
def seed_summary(values: dict) -> dict:
    """values {seed: metric}. Mean, SD (ddof 1), range. No seed-level CI: with 2-3 seeds it would not be informative; the registered
    uncertainty is the sequence-paired bootstrap reported with each test."""
    xs = np.array([v for v in values.values() if v is not None], float)
    if not len(xs):
        return {"n": 0}
    return {"n": int(len(xs)), "mean": float(xs.mean()), "sd": float(xs.std(ddof=1)) if len(xs) > 1 else None,
            "min": float(xs.min()), "max": float(xs.max()), "values": {str(k): v for k, v in values.items()}}


# ------------------------------------------------------------------------------------------------------------------- latency
def latency_stats(ms) -> dict:
    x = np.asarray([v for v in ms if v is not None], float)
    if not len(x):
        return {"n": 0}
    out = {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
           "min": float(x.min()), "max": float(x.max()), "fps_from_mean": float(1000 / x.mean()) if x.mean() > 0 else None}
    if len(x) >= 20:
        out["p90"] = float(np.percentile(x, 90))
        out["p95"] = float(np.percentile(x, 95))
    if len(x) >= 100:
        out["p99"] = float(np.percentile(x, 99))
    return out
