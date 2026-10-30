#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_latency.py -- measure what the README's latency table claims (11.8 ms / 84.7 FPS).

Stages timed separately with CUDA events / perf_counter, batch=1, 640x640, 9ch:
  preprocess (letterbox 3 frames + stack + to-device), model forward (bf16 & fp32),
  NMS (CPU, as in evaluation.py), ByteTrack update. 200 warm iterations after 30 warmup.
"""
import sys, time, json, argparse
from pathlib import Path
import numpy as np, torch, cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "c_model")); sys.path.insert(0, str(ROOT / "b_pipeline"))
from aerotrack_trainer import load_checkpoint
from stacked_dataset import letterbox_params, letterbox_image
from bytetrack import BYTETracker
from ultralytics.utils.nms import non_max_suppression


def pct(xs): xs = np.asarray(xs) * 1000; return {"mean_ms": round(float(xs.mean()), 2), "p50_ms": round(float(np.median(xs)), 2), "p95_ms": round(float(np.percentile(xs, 95)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/aerotrack_spd_finetuned/weights/best.pt"))
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    model, _ = load_checkpoint(a.weights, dev, prefer_ema=True, fuse=True)
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__)

    frames = [np.random.randint(0, 255, (512, 640, 3), np.uint8) for _ in range(3)]
    r, dw, dh, nw, nh = letterbox_params(512, 640, 640, 640)

    def preprocess():
        st = np.concatenate([letterbox_image(f, 640, 640, r, dw, dh, nw, nh)[:, :, ::-1] for f in frames], 2)
        t = torch.from_numpy(np.ascontiguousarray(st.transpose(2, 0, 1))).to(dev).float().div_(255).unsqueeze(0)
        torch.cuda.synchronize(); return t

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
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                out = model(x)
            e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / 1000)
        res[f"forward_{tag}"] = pct(ts)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        preds = model(x); preds = preds[0] if isinstance(preds, (list, tuple)) else preds
    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter(); preprocess(); ts.append(time.perf_counter() - t0)
    res["preprocess_3frame_letterbox"] = pct(ts)
    ts = []
    for _ in range(a.iters):
        p = preds.detach().float().cpu(); t0 = time.perf_counter()
        non_max_suppression(p.clone(), 0.25, 0.45, max_det=30); ts.append(time.perf_counter() - t0)
    res["nms_cpu_conf.25"] = pct(ts)
    ts = []
    for _ in range(a.iters):
        p = preds.detach().float().cpu(); t0 = time.perf_counter()
        non_max_suppression(p.clone(), 0.001, 0.7, max_det=30); ts.append(time.perf_counter() - t0)
    res["nms_cpu_conf.001(eval)"] = pct(ts)
    tk = BYTETracker(track_thresh=.5, low_thresh=.1, track_buffer=90); ts = []
    for i in range(a.iters):
        d = np.array([[300 + i, 250, 320 + i, 265, 0.8, 0], [100, 100, 108, 106, 0.3, 0]], np.float32)
        t0 = time.perf_counter(); tk.update(d); ts.append(time.perf_counter() - t0)
    res["bytetrack_update"] = pct(ts)

    e2e = sum(res[k]["mean_ms"] for k in ("preprocess_3frame_letterbox", "forward_bf16", "nms_cpu_conf.25", "bytetrack_update"))
    res["end_to_end_sum_bf16_ms"] = round(e2e, 2); res["end_to_end_fps"] = round(1000 / e2e, 1)
    res["README_claim"] = {"total_ms": 11.8, "fps": 84.7, "forward_ms": 8.4, "preprocess_ms": 1.8, "nms_ms": 0.9, "bytetrack_ms": 0.7}
    print(json.dumps(res, indent=1))
    (ROOT / "runs/audit/latency_benchmark.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
