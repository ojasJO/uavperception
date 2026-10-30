#!/usr/bin/env python
"""Latency measurement for the README table (repair item 10) — measured, labelled, with p95 and provenance.

    python experiments/aerotrack_repair/scripts/latency_repair.py --weights runs/aerotrack_spd_finetuned/weights/best.pt

Protocol (from audit/bench_latency.py): batch 1, 640x640 letterboxed 9-channel input, fused model, 30 warm-up +
`--iters` timed iterations; forward timed with CUDA events in both bf16 autocast and fp32; preprocessing (letterbox
3 frames + stack + upload), CPU NMS at the demo threshold (0.25/0.45) and at the evaluation threshold (0.001/0.7),
and one tracker update timed with perf_counter. Reports mean / p50 / p95 per stage, the end-to-end sum for the
precision actually used by inference.py (fp32 — it has no autocast), and the structural +1-frame latency of the
[t-1, t, t+1] stack (the t+1 frame must exist before frame t can be processed: one frame period at the source rate).
Output: experiments/aerotrack_repair/reports/latency_<run>.json (+ .md), never typed into the README by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression

from aerotrack_trainer import load_checkpoint
from bytetrack import BYTETracker
from stacked_dataset import letterbox_image, letterbox_params

from anti_uav_repro.utils.repro import env_snapshot, file_sha256, git_info, write_json

from _bootstrap import REPORTS_DIR


def pct(xs):
    xs = np.asarray(xs) * 1000
    return {"mean_ms": round(float(xs.mean()), 2), "p50_ms": round(float(np.median(xs)), 2), "p95_ms": round(float(np.percentile(xs, 95)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--source-fps", type=float, default=25.0, help="frame rate of the video source (Anti-UAV: 25)")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    model, ck = load_checkpoint(a.weights, dev, prefer_ema=True, fuse=True)
    frames = [np.random.randint(0, 255, (512, 640, 3), np.uint8) for _ in range(3)]
    r, dw, dh, nw, nh = letterbox_params(512, 640, 640, 640)

    def preprocess():
        st = np.concatenate([letterbox_image(f, 640, 640, r, dw, dh, nw, nh)[:, :, ::-1] for f in frames], 2)
        t = torch.from_numpy(np.ascontiguousarray(st.transpose(2, 0, 1))).to(dev).float().div_(255).unsqueeze(0)
        torch.cuda.synchronize()
        return t

    x = preprocess()
    res = {}
    for dtype in (torch.bfloat16, None):
        tag = "bf16" if dtype else "fp32"
        for _ in range(30):
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                model(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                model(x)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) / 1000)
        res[f"forward_{tag}"] = pct(ts)
    with torch.no_grad():
        preds = model(x)
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        preprocess()
        ts.append(time.perf_counter() - t0)
    res["preprocess_3frame_letterbox"] = pct(ts)
    for name, conf, iou in (("nms_cpu_demo_conf.25_iou.45", 0.25, 0.45), ("nms_cpu_eval_conf.001_iou.7", 0.001, 0.7)):
        ts = []
        for _ in range(a.iters):
            p = preds.detach().float().cpu()
            t0 = time.perf_counter()
            non_max_suppression(p.clone(), conf, iou, max_det=30)
            ts.append(time.perf_counter() - t0)
        res[name] = pct(ts)
    tk = BYTETracker(track_thresh=.5, low_thresh=.1, track_buffer=90)
    ts = []
    for i in range(a.iters):
        d = np.array([[300 + i, 250, 320 + i, 265, 0.8, 0], [100, 100, 108, 106, 0.3, 0]], np.float32)
        t0 = time.perf_counter()
        tk.update(d)
        ts.append(time.perf_counter() - t0)
    res["bytetrack_update"] = pct(ts)
    for prec in ("fp32", "bf16"):
        s = sum(res[k]["mean_ms"] for k in ("preprocess_3frame_letterbox", f"forward_{prec}", "nms_cpu_demo_conf.25_iou.45"))
        res[f"end_to_end_{prec}_no_tracker_ms"] = round(s, 2)
        res[f"end_to_end_{prec}_no_tracker_fps"] = round(1000 / s, 1)
    res["structural_lookahead_latency_ms"] = round(1000 / a.source_fps, 1)
    res["notes"] = ["inference.py runs fp32 (no autocast): the fp32 row is the deployed configuration",
                    "the 9-channel stack needs frame t+1 before frame t is processed: add one source frame period",
                    "tracker row shown for reference only; see the tracker comparison for whether it is used"]
    res["_meta"] = {"weights": a.weights, "weights_sha256": file_sha256(a.weights), "epoch": ck.get("epoch"), "iters": a.iters,
                    "gpu": torch.cuda.get_device_name(0), "git": git_info(), "env": env_snapshot(), "command": " ".join(sys.argv),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    run = a.weights.replace("\\", "/").split("/")[-3]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(res, REPORTS_DIR / f"latency_{run}.json")
    md = [f"# Latency — {run} (epoch {ck.get('epoch')}), {res['_meta']['gpu']}, batch 1, 640×640, {a.iters} iterations", "",
          "| stage | mean ms | p50 ms | p95 ms |", "|---|---|---|---|"]
    for k, v in res.items():
        if isinstance(v, dict) and "mean_ms" in v:
            md.append(f"| {k} | {v['mean_ms']} | {v['p50_ms']} | {v['p95_ms']} |")
    md += ["", f"- end-to-end fp32 (deployed, no tracker): **{res['end_to_end_fp32_no_tracker_ms']} ms ≈ {res['end_to_end_fp32_no_tracker_fps']} FPS**",
           f"- end-to-end bf16 (not used by inference.py): {res['end_to_end_bf16_no_tracker_ms']} ms ≈ {res['end_to_end_bf16_no_tracker_fps']} FPS",
           f"- plus structural look-ahead of one source frame: +{res['structural_lookahead_latency_ms']} ms at {a.source_fps:g} FPS", ""]
    (REPORTS_DIR / f"latency_{run}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
