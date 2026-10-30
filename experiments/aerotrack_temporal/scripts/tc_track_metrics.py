"""Track C tracking metrics beyond CLEAR-MOT / IDF1: HOTA (with DetA / AssA / LocA), MT / PT / ML, continuity, recovery time.

Input format per sequence (single class): gt = {t: (ids[n], boxes[n, 4] xyxy)}, pr = {t: (ids[m], boxes[m, 4] xyxy)}, frames 0..T-1.

HOTA follows TrackEval's reference implementation (Luiten et al., IJCV 2021; github.com/JonathonLuiten/TrackEval, MIT licence,
hota.py: eval_sequence / combine_sequences / _compute_final_fields), re-implemented here: alphas 0.05..0.95, Hungarian matching on
global-alignment-weighted IoU, per-alpha TP / FN / FP, AssA / AssRe / AssPr from the per-(gt id, tracker id) match counts, LocA;
sequences are combined by summing the counts and TP-weighting AssA / AssRe / AssPr / LocA. HOTA = mean over alphas of
sqrt(DetA * AssA). Cross-checked against TrackEval itself on 60 random multi-target sequences: identical per sequence and
combined (max |diff| 0.0; scripts/check_hota_vs_trackeval.py, reports/hota_crosscheck.json).

Per-target continuity and recovery (CLEAR matching at IoU >= 0.5, one target id per GT trajectory):
    MT / PT / ML        fraction of a GT trajectory's visible frames that are matched: >= 0.8 / in between / <= 0.2
    fragments           tracked -> untracked transitions while the target stays visible (CLEAR "Frag")
    longest_run_frac    longest run of consecutive matched visible frames / visible frames
    recovery            events: re-entry (target visible after >= 1 invisible frame) and loss (visible, matched at t-1, unmatched
                        at t); delay = frames until the next match (censored at the sequence end -> "not recovered")
    ids_per_target      number of distinct tracker ids ever matched to the target (identity persistence; 1 is perfect)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 0.99, 0.05)
EPS = np.finfo("float").eps


def iou_xyxy(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _remap(by_frame, T):
    ids = sorted({int(i) for t in range(T) for i in by_frame.get(t, (np.zeros(0), None))[0]})
    m = {v: k for k, v in enumerate(ids)}
    return m, len(ids)


def _zeros():
    return {k: np.zeros(len(ALPHAS)) for k in ("HOTA_TP", "HOTA_FN", "HOTA_FP", "AssA", "AssRe", "AssPr", "LocA")}


def _final(res):
    tp, fn, fp = res["HOTA_TP"], res["HOTA_FN"], res["HOTA_FP"]
    res["DetRe"] = tp / np.maximum(1, tp + fn)
    res["DetPr"] = tp / np.maximum(1, tp + fp)
    res["DetA"] = tp / np.maximum(1, tp + fn + fp)
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    return res


def hota_sequence(gt, pr, T):
    """TrackEval HOTA.eval_sequence for one class. -> per-alpha arrays."""
    gmap, ng = _remap(gt, T)
    pmap, npr = _remap(pr, T)
    res = _zeros()
    n_gt = sum(len(gt.get(t, (np.zeros(0),))[0]) for t in range(T))
    n_pr = sum(len(pr.get(t, (np.zeros(0),))[0]) for t in range(T))
    if n_pr == 0:
        res["HOTA_FN"] = n_gt * np.ones(len(ALPHAS))
        res["LocA"] = np.ones(len(ALPHAS))
        return _final(res)
    if n_gt == 0:
        res["HOTA_FP"] = n_pr * np.ones(len(ALPHAS))
        res["LocA"] = np.ones(len(ALPHAS))
        return _final(res)
    frames = []
    for t in range(T):
        gi, gb = gt.get(t, (np.zeros(0), np.zeros((0, 4))))
        pi, pb = pr.get(t, (np.zeros(0), np.zeros((0, 4))))
        g_ids = np.array([gmap[int(i)] for i in gi], dtype=int)
        p_ids = np.array([pmap[int(i)] for i in pi], dtype=int)
        frames.append((g_ids, p_ids, iou_xyxy(gb, pb)))
    potential = np.zeros((ng, npr))
    gt_count = np.zeros((ng, 1))
    pr_count = np.zeros((1, npr))
    for g_ids, p_ids, sim in frames:
        if len(g_ids) and len(p_ids):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            s_iou = np.zeros_like(sim)
            mk = denom > 0 + EPS
            s_iou[mk] = sim[mk] / denom[mk]
            potential[g_ids[:, None], p_ids[None, :]] += s_iou
        gt_count[g_ids] += 1
        pr_count[0, p_ids] += 1
    global_align = potential / (gt_count + pr_count - potential)
    matches = [np.zeros_like(potential) for _ in ALPHAS]
    for g_ids, p_ids, sim in frames:
        if not len(g_ids):
            res["HOTA_FP"] += len(p_ids)
            continue
        if not len(p_ids):
            res["HOTA_FN"] += len(g_ids)
            continue
        score = global_align[g_ids[:, None], p_ids[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = sim[rows, cols] >= alpha - EPS
            r, c = rows[ok], cols[ok]
            k = len(r)
            res["HOTA_TP"][a] += k
            res["HOTA_FN"][a] += len(g_ids) - k
            res["HOTA_FP"][a] += len(p_ids) - k
            if k:
                res["LocA"][a] += sim[r, c].sum()
                matches[a][g_ids[r], p_ids[c]] += 1
    for a in range(len(ALPHAS)):
        mc = matches[a]
        tp = np.maximum(1, res["HOTA_TP"][a])
        res["AssA"][a] = np.sum(mc * (mc / np.maximum(1, gt_count + pr_count - mc))) / tp
        res["AssRe"][a] = np.sum(mc * (mc / np.maximum(1, gt_count))) / tp
        res["AssPr"][a] = np.sum(mc * (mc / np.maximum(1, pr_count))) / tp
    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
    return _final(res)


def hota_combine(seq_results):
    """TrackEval HOTA.combine_sequences."""
    res = _zeros()
    for k in ("HOTA_TP", "HOTA_FN", "HOTA_FP"):
        res[k] = sum(r[k] for r in seq_results)
    for k in ("AssRe", "AssPr", "AssA"):
        res[k] = sum(r[k] * r["HOTA_TP"] for r in seq_results) / np.maximum(1.0, res["HOTA_TP"])
    loc = sum(r["LocA"] * r["HOTA_TP"] for r in seq_results)
    res["LocA"] = np.maximum(1e-10, loc) / np.maximum(1e-10, res["HOTA_TP"])
    return _final(res)


def hota_summary(res):
    return {"HOTA": float(res["HOTA"].mean()), "DetA": float(res["DetA"].mean()), "AssA": float(res["AssA"].mean()),
            "LocA": float(res["LocA"].mean()), "DetRe": float(res["DetRe"].mean()), "DetPr": float(res["DetPr"].mean()),
            "AssRe": float(res["AssRe"].mean()), "AssPr": float(res["AssPr"].mean()), "HOTA(0)": float(res["HOTA"][0])}


def continuity(gt, pr, T, iou_thr=0.5, recover_window=25):
    """Per-sequence continuity / recovery for every GT id (CLEAR-style Hungarian matching at IoU >= iou_thr per frame)."""
    matched_pid = {}                                         # (t, gid) -> pid
    for t in range(T):
        gi, gb = gt.get(t, (np.zeros(0), np.zeros((0, 4))))
        pi, pb = pr.get(t, (np.zeros(0), np.zeros((0, 4))))
        if not len(gi) or not len(pi):
            continue
        iou = iou_xyxy(gb, pb)
        r, c = linear_sum_assignment(-iou)
        for a, b in zip(r, c):
            if iou[a, b] >= iou_thr:
                matched_pid[(t, int(gi[a]))] = int(pi[b])
    out = []
    gids = sorted({int(i) for t in range(T) for i in gt.get(t, (np.zeros(0),))[0]})
    for g in gids:
        vis = np.array([g in set(int(i) for i in gt.get(t, (np.zeros(0),))[0]) for t in range(T)])
        hit = np.array([(t, g) in matched_pid for t in range(T)])
        nv = int(vis.sum())
        cover = float(hit[vis].mean()) if nv else 0.0
        frag, run, best = 0, 0, 0
        for t in range(T):
            if vis[t]:
                if hit[t]:
                    run += 1
                    best = max(best, run)
                else:
                    if run > 0:
                        frag += 1
                    run = 0
        events = []
        for t in range(1, T):
            kind = None
            if vis[t] and not vis[t - 1]:
                kind = "reentry"
            elif vis[t] and vis[t - 1] and hit[t - 1] and not hit[t]:
                kind = "loss"
            if kind:
                j = np.flatnonzero(hit[t:])
                events.append({"kind": kind, "t": t, "delay": int(j[0]) if len(j) else None})
        out.append({"gid": g, "visible": nv, "coverage": cover, "class": "MT" if cover >= 0.8 else ("ML" if cover <= 0.2 else "PT"),
                    "fragments": frag, "longest_run_frac": best / nv if nv else 0.0,
                    "ids": len({matched_pid[(t, g)] for t in range(T) if (t, g) in matched_pid}), "events": events})
    return out


def continuity_summary(targets, recover_window=25):
    if not targets:
        return None
    ev = [e for x in targets for e in x["events"]]

    def rec(kind):
        d = [e["delay"] for e in ev if e["kind"] == kind]
        got = [x for x in d if x is not None]
        return {"events": len(d), "recovered": len(got), "never": len(d) - len(got),
                "median_frames": float(np.median(got)) if got else None, "p90_frames": float(np.percentile(got, 90)) if got else None,
                "within_window": float(np.mean([x is not None and x <= recover_window for x in d])) if d else None}
    n = len(targets)
    return {"targets": n, "MT": sum(x["class"] == "MT" for x in targets) / n, "PT": sum(x["class"] == "PT" for x in targets) / n,
            "ML": sum(x["class"] == "ML" for x in targets) / n, "fragments": int(sum(x["fragments"] for x in targets)),
            "fragments_per_target": float(np.mean([x["fragments"] for x in targets])),
            "longest_run_frac_mean": float(np.mean([x["longest_run_frac"] for x in targets])),
            "coverage_mean": float(np.mean([x["coverage"] for x in targets])),
            "ids_per_target_mean": float(np.mean([x["ids"] for x in targets])),
            "recovery_reentry": rec("reentry"), "recovery_loss": rec("loss"), "recover_window_frames": recover_window}
