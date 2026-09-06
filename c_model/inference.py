#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py - AeroTrack-Net Streaming Video Inference
======================================================
Applies the 3-frame temporal rolling window (t-1, t, t+1 -> 9-channel tensor)
on video streams and exports annotated detections with latency and FPS telemetry.

Streaming Architecture:
  A 3-slot rolling buffer maintains one future frame (t+1) while detecting
  on the centre frame (t), matching the training window with single-frame lag.
  Boundary frames are duplicated (t=0 -> t-1=t; last -> t+1=t).
"""

import sys
import time
import inspect
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "b_pipeline"))
from custom_modules import SPDConv
from stacked_dataset import letterbox_params, letterbox_image, PAD_VALUE

IMG = 640
NAMES = {0: "uav_drone", 1: "bird_hard_negative"}
COLORS = {0: (0, 220, 0), 1: (0, 140, 255)}          # BGR: drone=green, bird=orange


def register_spdconv():
    import ultralytics.nn.tasks as T
    T.SPDConv = SPDConv
    try:
        import ultralytics.nn.modules as M
        M.SPDConv = SPDConv
    except Exception:
        pass
    src = inspect.getsource(T.parse_model)
    if "or m is SPDConv" not in src:
        patched = src.replace("if m in base_modules:",
                              "if m in base_modules or m is SPDConv:", 1)
        g = dict(vars(T)); g["SPDConv"] = SPDConv
        exec(compile(patched, "<patched_parse_model>", "exec"), g)
        T.parse_model = g["parse_model"]


def letterbox_geometry(H, W):
    """The (r, dw, dh) mapping this stream uses, computed once per video."""
    r, dw, dh, nw, nh = letterbox_params(H, W, IMG, IMG)
    return r, dw, dh, nw, nh


def preprocess(frame_bgr, device, geom):
    """BGR HxWx3 uint8 -> [3,640,640] float [0,1] on device (RGB).

    LETTERBOX, not resize. Training uses an aspect-preserving pad
    (stacked_dataset.letterbox_image); stretching here instead would hand the
    network a geometry it never saw -- a 1920x1080 drone squashed to 0.93 aspect
    when the model learned it at 1.66. Train/serve preprocessing must match
    exactly or the deployed numbers are not the validated numbers.
    """
    r, dw, dh, nw, nh = geom
    img = letterbox_image(frame_bgr, IMG, IMG, r, dw, dh, nw, nh)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(device)
    return t.float().div_(255.0).permute(2, 0, 1).contiguous()


def draw(frame_bgr, dets, W, H, latency_ms, fps_inst, fps_avg, geom):
    """dets [n,6] (xyxy on the 640 letterbox canvas) -> annotated original frame.

    Inverse of the letterbox: subtract the pad, divide by the scale. Using
    W/640 here (the old stretch inverse) would displace every box by the pad
    width -- up to 90 px on a 16:9 source."""
    out = frame_bgr.copy()
    r, dw, dh, _, _ = geom
    unx = lambda v: int(round(min(max((v - dw) / r, 0), W - 1)))
    uny = lambda v: int(round(min(max((v - dh) / r, 0), H - 1)))
    for x1, y1, x2, y2, conf, cls in dets:
        cls = int(cls)
        p1 = (unx(x1), uny(y1))
        p2 = (unx(x2), uny(y2))
        color = COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(out, f"{NAMES.get(cls, cls)} {conf:.2f}",
                    (p1[0], max(0, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    # HUD: latency + FPS (real-time viability proof)
    hud = f"latency {latency_ms:5.1f} ms | {fps_inst:5.1f} FPS (avg {fps_avg:5.1f})"
    cv2.rectangle(out, (0, 0), (min(W, 560), 34), (0, 0, 0), -1)
    cv2.putText(out, hud, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input .mp4")
    ap.add_argument("--weights", default=str(ROOT / "runs" / "aerotrack_spd_v1" /
                                             "weights" / "best.pt"))
    ap.add_argument("--out", default=str(HERE / "inference_output.mp4"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[WARN] No CUDA device detected; running inference on CPU.")
    if not Path(args.weights).exists():
        raise SystemExit(f"\n[ERROR] Weights not found at: {args.weights}\n")

    register_spdconv()
    from aerotrack_trainer import load_checkpoint
    model, ck = load_checkpoint(args.weights, device, prefer_ema=True, fuse=True)
    print(f"[model] {Path(args.weights).name} — epoch {ck.get('epoch')}, "
          f"fitness {ck.get('best_fitness')}, EMA weights, Conv+BN fused")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"[ABORT] cannot open source: {args.source}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_in, (W, H))
    geom = letterbox_geometry(H, W)          # constant for the whole stream
    print(f"[stream] {W}x{H} @ {fps_in:.1f} fps -> letterbox r={geom[0]:.4f} "
          f"pad=({geom[1]:.1f}, {geom[2]:.1f})")

    tens = deque(maxlen=3)      # preprocessed [3,640,640] tensors
    raws = deque(maxlen=3)      # matching raw BGR frames
    lat_hist = deque(maxlen=30)      # rolling, for the on-screen HUD
    all_lat = []                     # every measurement, for the p99 report
    n_out = 0

    def emit_centre():
        nonlocal n_out
        stack = torch.cat([tens[0], tens[1], tens[2]], dim=0).unsqueeze(0)  # [1,9,640,640]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            preds = model(stack)
        dets = non_max_suppression(preds, args.conf, args.iou)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000.0
        lat_hist.append(latency); all_lat.append(latency)
        fps_inst = 1000.0 / latency if latency > 0 else 0.0
        fps_avg = 1000.0 / (sum(lat_hist) / len(lat_hist))
        annotated = draw(raws[1], dets.cpu().numpy(), W, H, latency, fps_inst,
                         fps_avg, geom)
        writer.write(annotated)
        n_out += 1

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = preprocess(frame, device, geom)
        if idx == 0:                         # prime t-1 with a duplicate of frame 0
            tens.append(t); raws.append(frame)
        tens.append(t); raws.append(frame)
        if len(tens) == 3:
            emit_centre()                    # detects the CENTRE (lag-1) frame
        idx += 1

    if idx > 0:                              # flush: last frame as centre, t+1 dup
        tens.append(tens[-1]); raws.append(raws[-1])
        if len(tens) == 3:
            emit_centre()

    cap.release(); writer.release()
    lat = np.array(all_lat, dtype=np.float64)
    avg = float(lat.mean()) if len(lat) else 0.0
    p99 = float(np.percentile(lat, 99)) if len(lat) else 0.0
    print("=" * 60)
    print(f"[done] wrote {n_out} annotated frames -> {args.out}")
    print(f"       mean latency {avg:.1f} ms  ->  {1000.0/avg if avg else 0:.1f} FPS")
    print(f"       p99  latency {p99:.1f} ms  ->  {1000.0/p99 if p99 else 0:.1f} FPS "
          f"(worst case is what a counter-UAS system is judged on)")
    print(f"       + 1 frame of structural latency from the t+1 lookahead "
          f"({1000.0/fps_in:.1f} ms at {fps_in:.0f} fps)")
    print(f"       GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
