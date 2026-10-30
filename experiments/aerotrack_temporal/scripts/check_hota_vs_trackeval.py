#!/usr/bin/env python
"""Numerical cross-check of tc_track_metrics.hota_* against TrackEval's own HOTA (github.com/JonathonLuiten/TrackEval, MIT).

    venv\Scripts\python.exe experiments\aerotrack_temporal\scripts\check_hota_vs_trackeval.py <path to an unpacked TrackEval-master>

TrackEval is NOT vendored or installed: point the script at a downloaded copy (e.g. the master zip unpacked in a scratch folder).
60 random multi-target sequences (1-3 targets, 15 % invisibility, 15 % misses, 5 % identity swaps, 20 % false-positive frames,
box noise sigma 3 px); every per-alpha array (HOTA, DetA, AssA, LocA, DetRe, DetPr, AssRe, AssPr) is compared per sequence and
after combine_sequences. Output: experiments/aerotrack_temporal/reports/hota_crosscheck.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    te = Path(sys.argv[1])
    for k, v in (("float", float), ("int", int)):
        if k not in np.__dict__:
            setattr(np, k, v)                      # TrackEval predates numpy 1.24 (np.float / np.int removed)
    sys.path.insert(0, str(te))
    sys.path.insert(0, str(HERE))
    from trackeval.metrics.hota import HOTA
    import tc_track_metrics as M
    rng = np.random.default_rng(0)
    ref, worst, seq = HOTA(), 0.0, []
    keys = ("HOTA", "DetA", "AssA", "LocA", "AssRe", "AssPr", "DetRe", "DetPr")
    for _ in range(60):
        T = int(rng.integers(20, 120)); ng = int(rng.integers(1, 4)); nt = int(rng.integers(1, 6))
        gt, pr = {}, {}
        pos = rng.uniform(50, 500, (ng, 2)); vel = rng.normal(0, 3, (ng, 2))
        for t in range(T):
            pos += vel
            gi = np.flatnonzero(rng.random(ng) > 0.15)
            gb = np.array([[p[0], p[1], p[0] + 20, p[1] + 15] for p in pos[gi]]) if len(gi) else np.zeros((0, 4))
            if len(gi):
                gt[t] = (gi.astype(int) + 100, gb)
            ids, boxes = [], []
            for k, g in enumerate(gi):
                if rng.random() < 0.85:
                    tid = (g + (1 if rng.random() < 0.05 else 0)) % nt
                    if tid in ids:
                        continue
                    ids.append(tid); boxes.append(gb[k] + rng.normal(0, 3, 4))
            if rng.random() < 0.2:
                fid = int(rng.integers(0, nt))
                if fid not in ids:
                    ids.append(fid); boxes.append(np.array([*rng.uniform(0, 600, 2), 0, 0]) + np.array([0, 0, 18, 14]))
            if ids:
                b = np.array(boxes); b[:, 2:] = np.maximum(b[:, 2:], b[:, :2] + 1)
                pr[t] = (np.array(ids) + 7, b)
        gmap = {v: i for i, v in enumerate(sorted({int(x) for t in gt for x in gt[t][0]}))}
        pmap = {v: i for i, v in enumerate(sorted({int(x) for t in pr for x in pr[t][0]}))}
        data = {"gt_ids": [], "tracker_ids": [], "similarity_scores": []}
        for t in range(T):
            gi, gb = gt.get(t, (np.zeros(0, int), np.zeros((0, 4))))
            pi, pb = pr.get(t, (np.zeros(0, int), np.zeros((0, 4))))
            data["gt_ids"].append(np.array([gmap[int(x)] for x in gi], int))
            data["tracker_ids"].append(np.array([pmap[int(x)] for x in pi], int))
            data["similarity_scores"].append(M.iou_xyxy(gb, pb))
        data.update(num_gt_ids=len(gmap), num_tracker_ids=len(pmap), num_gt_dets=sum(len(x) for x in data["gt_ids"]),
                    num_tracker_dets=sum(len(x) for x in data["tracker_ids"]), num_timesteps=T)
        r = ref.eval_sequence(data)
        m = M.hota_sequence(gt, pr, T)
        worst = max([worst] + [float(np.abs(np.asarray(r[k]) - m[k]).max()) for k in keys])
        seq.append((r, m))
    cr = ref.combine_sequences({i: r for i, (r, _) in enumerate(seq)})
    cm = M.hota_combine([m for _, m in seq])
    wc = max(float(np.abs(np.asarray(cr[k]) - cm[k]).max()) for k in ("HOTA", "DetA", "AssA", "LocA", "AssRe", "AssPr"))
    out = {"checked": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "trackeval": str(te), "sequences": len(seq),
           "max_abs_diff_per_sequence": worst, "max_abs_diff_combined": wc,
           "combined_HOTA": {"trackeval": float(np.mean(cr["HOTA"])), "ours": float(np.mean(cm["HOTA"]))},
           "verdict": "IDENTICAL" if worst < 1e-12 and wc < 1e-12 else "DIFFERENT"}
    if len(sys.argv) > 2:
        out["trackeval_zip_sha256"] = sys.argv[2]
    p = REPO / "experiments" / "aerotrack_temporal" / "reports" / "hota_crosscheck.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
