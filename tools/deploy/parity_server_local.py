#!/usr/bin/env python
"""Server-vs-laptop inference parity on the same clip (M11): the HPC's VAL detection cache (V100, PyTorch fp16 autocast, NMS conf 0.01)
vs the deployment runtime's per-frame detections (RTX 5050, the packaged backend, NMS conf 0.05) for one sequence.

Per frame, the top-scoring drone detection of each side is compared (both sides restricted to class 0 and score >= 0.05, the
stricter of the two NMS thresholds): presence agreement, IoU of the two boxes, |score difference|.
    venv\\Scripts\\python.exe tools\\deploy\\parity_server_local.py --server <seq.npz> --local <stem.frames.jsonl> --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--server", required=True); ap.add_argument("--local", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--conf", type=float, default=0.05)
    a = ap.parse_args()
    z = np.load(a.server)
    off, B, S, C = z["offsets"], z["boxes"], z["scores"], z["classes"]
    loc = [json.loads(x) for x in open(a.local, encoding="utf-8") if x.strip()]
    n = min(len(off) - 1, len(loc))
    agree = both = 0
    ious, dscore = [], []
    for i in range(n):
        s = [(S[j], B[j]) for j in range(off[i], off[i + 1]) if C[j] == 0 and S[j] >= a.conf]
        l = [(d[4], d[:4]) for d in loc[i]["detections"] if int(d[5]) == 0 and d[4] >= a.conf]
        if bool(s) == bool(l):
            agree += 1
        if s and l:
            both += 1
            ss, sb = max(s, key=lambda t: t[0]); ls, lb = max(l, key=lambda t: t[0])
            ious.append(iou(list(map(float, sb)), list(map(float, lb)))); dscore.append(abs(float(ss) - float(ls)))
    ious, dscore = np.array(ious), np.array(dscore)
    rep = {"server": Path(a.server).name, "local": Path(a.local).name, "frames": n, "presence_agreement": agree / n,
           "frames_with_both": both, "top1_iou_mean": float(ious.mean()), "top1_iou_p05": float(np.percentile(ious, 5)),
           "top1_iou_min": float(ious.min()), "frac_iou_ge_0.9": float((ious >= 0.9).mean()),
           "abs_score_diff_mean": float(dscore.mean()), "abs_score_diff_p95": float(np.percentile(dscore, 95)), "conf": a.conf}
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
