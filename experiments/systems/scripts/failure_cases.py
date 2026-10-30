#!/usr/bin/env python
"""Automatic, deterministic failure analysis of the integrated stack (detector cache + STE) on VAL.

    venv\\Scripts\\python.exe experiments\\systems\\scripts\\failure_cases.py --run gen_temporal_fp_g4_full_e20 [--corpus auto]

Every frame / event is classified by an explicit rule (below); per category the rates are reported per dataset group and
EXAMPLES ARE CHOSEN BY RULE, never by eye: candidates are sorted by the category's severity measure and the ones at the
10th / 30th / 50th / 70th / 90th percentile are rendered (ties broken by sequence name, then frame). Every example records its
sequence, frame, source cache file and STE parameters, so any figure can be traced back to data.

    true_positive            visible target, output IoU >= 0.5                       severity: 1 - IoU
    false_positive           target invisible (exist 0), a box is output              severity: output score
    false_negative           visible target, no output or IoU < 0.5                  severity: GT box size (smaller = harder)
    micro_miss               false negative on a micro target (< 0.03 % of the frame) severity: GT area
    occlusion_failure        false negative in the first 30 frames after the target reappears
    reacquisition_success    reappearance recovered (IoU >= 0.5) within 10 frames     severity: frames to recover
    reacquisition_failure    reappearance not recovered within 30 frames
    identity_failure         previous frame correct, now a box with IoU 0 while the target is visible (track hijack)
    trajectory_overshoot / _undershoot   12-step Kalman forecast error projected on the true motion direction (> 0 / < 0),
                             only where the target moved >= 5 px over the horizon     severity: |projection|
    high_uncertainty         Kalman 95 % ellipse area at h = 12 in the top 5 %        severity: area
    low_confidence_correct   true positive with output score < 0.3                    severity: 0.3 - score
    high_confidence_wrong    output score >= 0.7 but IoU < 0.1 (or target invisible)  severity: score
Output: experiments/systems/reports/failures_<run>_<corpus>.{json,md} and contact sheets in
experiments/systems/reports/failures_<run>_<corpus>/<category>.png (crops around the target, GT white, output green/red).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import ste as T  # noqa: E402
import track_eval as TE  # noqa: E402
from anti_uav_repro.eval.metrics import iou_xywh  # noqa: E402

QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
MICRO = 0.0003
CATS = ["true_positive", "false_positive", "false_negative", "micro_miss", "occlusion_failure", "reacquisition_success",
        "reacquisition_failure", "identity_failure", "trajectory_overshoot", "trajectory_undershoot", "high_uncertainty",
        "low_confidence_correct", "high_confidence_wrong"]


def analyse(k, it, params):
    s, dets = it["seq"], it["dets"]
    W, H = s.image_size()
    start = s.start_frame
    init = s.box_xywh(start)
    trk = T.STE(T.STEParams(**{**T.STEParams().as_dict(), **params}))
    trk.reset(np.array([init[0], init[1], init[0] + init[2], init[1] + init[3]], dtype=np.float64))
    ev = defaultdict(list)
    out = {start: ([float(v) for v in init], 1.0, "init")}
    fc = {}
    for t in range(start + 1, s.n_frames):
        b, sc, cl = dets[t]
        box, mode, score = trk.step(b, sc, cl)
        out[t] = (None if box is None else [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])], score, mode)
        if trk.kf is not None and mode in ("init", "track", "reacquired"):
            m, c = trk.kf.forecast(12)
            fc[t] = (m[-1], c[-1])
    prev_ok = True
    for t in range(start + 1, s.n_frames):
        box, score, mode = out[t]
        vis = bool(s.exist[t]) and s.box_xywh(t) is not None
        g = s.box_xywh(t)
        iou = iou_xywh(box, list(g)) if (vis and box is not None) else 0.0
        base = {"seq": k, "frame": t, "score": score, "mode": mode, "iou": iou, "gt": None if g is None else [float(v) for v in g], "out": box}
        if vis and iou >= 0.5:
            ev["true_positive"].append({**base, "sev": 1 - iou})
            if score < 0.3:
                ev["low_confidence_correct"].append({**base, "sev": 0.3 - score})
        if not vis and box is not None:
            ev["false_positive"].append({**base, "sev": score})
        if vis and iou < 0.5:
            area = float(g[2] * g[3]) / (W * H)
            ev["false_negative"].append({**base, "sev": -max(g[2], g[3])})
            if area < MICRO:
                ev["micro_miss"].append({**base, "sev": -area})
        if box is not None and score >= 0.7 and (not vis or iou < 0.1):
            ev["high_confidence_wrong"].append({**base, "sev": score})
        if vis and prev_ok and box is not None and iou == 0.0:
            ev["identity_failure"].append({**base, "sev": score})
        prev_ok = vis and iou >= 0.5
        if t in fc and t + 12 < s.n_frames and s.box_xywh(t + 12) is not None and g is not None:
            g12 = s.box_xywh(t + 12)
            c0 = np.array([g[0] + g[2] / 2, g[1] + g[3] / 2])
            c12 = np.array([g12[0] + g12[2] / 2, g12[1] + g12[3] / 2])
            motion = c12 - c0
            if np.linalg.norm(motion) >= 5:
                proj = float((fc[t][0] - c12) @ (motion / np.linalg.norm(motion)))
                cat = "trajectory_overshoot" if proj > 0 else "trajectory_undershoot"
                ev[cat].append({**base, "sev": abs(proj), "forecast_h12": fc[t][0].tolist(), "gt_h12": c12.tolist()})
            ev["_ellipse"].append({**base, "sev": float(np.pi * 5.9915 * np.sqrt(max(np.linalg.det(fc[t][1]), 0)))})
    # reappearance events
    for t in range(start + 1, s.n_frames):
        if s.exist[t] and not s.exist[t - 1]:
            rec = None
            for u in range(t, min(t + 31, s.n_frames)):
                b = out.get(u, (None,))[0]
                g = s.box_xywh(u)
                if b is not None and g is not None and iou_xywh(b, list(g)) >= 0.5:
                    rec = u - t
                    break
            base = {"seq": k, "frame": t, "score": out.get(t, (None, 0.0))[1], "mode": out.get(t, (None, 0, "?"))[2],
                    "iou": 0.0, "gt": None if s.box_xywh(t) is None else [float(v) for v in s.box_xywh(t)], "out": out.get(t, (None,))[0]}
            if rec is not None and rec <= 10:
                ev["reacquisition_success"].append({**base, "sev": rec, "frames_to_recover": rec})
            if rec is None:
                ev["reacquisition_failure"].append({**base, "sev": 30})
            for u in range(t, min(t + 30, s.n_frames)):
                b, g = out.get(u, (None,))[0], s.box_xywh(u)
                if g is not None and (b is None or iou_xywh(b, list(g)) < 0.5):
                    ev["occlusion_failure"].append({**base, "frame": u, "sev": u - t})
    return ev


def pick(cands):
    if not cands:
        return []
    c = sorted(cands, key=lambda e: (e["sev"], e["seq"], e["frame"]))
    idx = sorted({min(int(q * (len(c) - 1) + 0.5), len(c) - 1) for q in QUANTILES})
    return [dict(c[i], quantile=QUANTILES[min(j, len(QUANTILES) - 1)]) for j, i in enumerate(idx)]


def crop(seq, e, size=160):
    img = cv2.imread(str(seq.frame_path(e["frame"])), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((size, size, 3), np.uint8)
    ref = e["gt"] or e["out"]
    cx, cy = (ref[0] + ref[2] / 2, ref[1] + ref[3] / 2) if ref else (img.shape[1] / 2, img.shape[0] / 2)
    half = max(40, int(3 * max(ref[2], ref[3]))) if ref else 80
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, img.shape[1])), int(min(cy + half, img.shape[0]))
    for box, col in ((e["gt"], (255, 255, 255)), (e["out"], (0, 220, 0) if e["iou"] >= 0.5 else (0, 0, 255))):
        if box:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), col, 1)
    c = cv2.resize(img[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.putText(c, f"{e['frame']} s{e['score']:.2f} iou{e['iou']:.2f}", (2, size - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
    return c


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", default="auto")
    a = ap.parse_args()
    if a.corpus == "auto":
        base = REPO / "results/runs" / "systems" / "_cache" / a.run
        a.corpus = "data_unified_v2" if (base / "data_unified_v2" / "val" / "index.json").exists() else "data_unified"
    t0 = time.time()
    index, seqs = TE.load(a.run, a.corpus)
    rep = REPO / "experiments" / "systems" / "reports" / f"track_{a.run}_{a.corpus}.json"
    halves = json.loads(rep.read_text(encoding="utf-8"))["trackers"]["ste"]["params_per_half"] if rep.exists() else {}
    allev = defaultdict(list)
    counts = defaultdict(lambda: defaultdict(int))
    frames = defaultdict(int)
    for k, it in sorted(seqs.items()):
        p = halves.get(str(it["half"]), halves.get(it["half"], {})) if halves else {}
        ev = analyse(k, it, p)
        frames[it["group"]] += it["seq"].n_frames
        for c, lst in ev.items():
            counts[it["group"]][c] += len(lst)
            for e in lst:
                e["group"] = it["group"]
            allev[c] += lst
    if allev["_ellipse"]:
        thr = np.percentile([e["sev"] for e in allev["_ellipse"]], 95)
        allev["high_uncertainty"] = [e for e in allev["_ellipse"] if e["sev"] >= thr]
        for e in allev["high_uncertainty"]:
            counts[e["group"]]["high_uncertainty"] += 1
    out_dir = REPO / "experiments" / "systems" / "reports" / f"failures_{a.run}_{a.corpus}"
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = {}
    for c in CATS:
        ex = pick(allev.get(c, []))
        examples[c] = ex
        if ex:
            tiles = [crop(seqs[e["seq"]]["seq"], e) for e in ex]
            cv2.imwrite(str(out_dir / f"{c}.png"), np.hstack(tiles))
    cache = f"results/runs/systems/_cache/{a.run}/{a.corpus}/val"
    rec = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": a.run, "corpus": a.corpus, "cache": cache,
           "ste_params": "cross-fitted per half from " + rep.name if halves else "defaults", "selection": f"quantiles {QUANTILES} of the severity measure",
           "frames_per_group": dict(frames), "counts": {g: dict(v) for g, v in counts.items()},
           "examples": {c: [{k2: v for k2, v in e.items() if k2 != "sev"} | {"severity": e["sev"], "source": f"{cache}/{e['seq']}.npz"} for e in ex]
                        for c, ex in examples.items()}, "seconds": round(time.time() - t0, 1)}
    path = REPO / "experiments" / "systems" / "reports" / f"failures_{a.run}_{a.corpus}.json"
    path.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    groups = sorted(frames)
    md = [f"# Failure analysis — {a.run} ({a.corpus}, VAL)", "", f"*Generated {rec['generated']} by `scripts/failure_cases.py`. "
          f"Rules in the script docstring; examples = the {', '.join(str(q) for q in QUANTILES)} quantiles of each category's severity "
          f"(no manual selection); contact sheets in `{out_dir.name}/`.*", "",
          "| category | " + " | ".join(groups) + " |", "|---|" + "---:|" * len(groups)]
    for c in CATS:
        md.append(f"| {c} | " + " | ".join(str(counts[g].get(c, 0)) for g in groups) + " |")
    md += ["", "Examples (sequence @ frame, score, IoU):", ""]
    for c in CATS:
        if examples[c]:
            md.append(f"- **{c}**: " + "; ".join(f"`{e['seq']}`@{e['frame']} (s {e['score']:.2f}, IoU {e['iou']:.2f})" for e in examples[c]))
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:8 + len(CATS)]))


if __name__ == "__main__":
    main()
