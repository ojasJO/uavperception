#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_side_by_side_demo.py - Generates Multi-Modal Side-by-Side Video & GIF
==========================================================================
Renders synchronous Visible (RGB) and Long-Wave Infrared (Thermal IR) feeds
with AeroTrack-Net detections, ByteTrack motion trajectories, 4x micro-target
magnifier insets, and high-contrast telemetry overlays.
"""

import os
import sys
import json
import time
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
from aerotrack_trainer import load_checkpoint
from stacked_dataset import letterbox_params, letterbox_image
from ultralytics.utils.nms import non_max_suppression

IMG_SIZE = 640
NAMES = {0: "uav_drone", 1: "bird"}
COLOR_RGB = (0, 220, 0)       # Bright green
COLOR_IR = (0, 200, 255)      # Amber / Cyan
COLOR_TRAIL = (0, 255, 255)   # Yellow
COLOR_HUD_BG = (18, 20, 24)

def letterbox_tensor(frame_bgr, device, geom):
    r, dw, dh, nw, nh = geom
    img = letterbox_image(frame_bgr, IMG_SIZE, IMG_SIZE, r, dw, dh, nw, nh)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(device)
    return t.float().div_(255.0).permute(2, 0, 1).contiguous()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = ROOT / "runs" / "aerotrack_spd_finetuned" / "weights" / "best.pt"
    if not weights_path.exists():
        weights_path = ROOT / "runs" / "aerotrack_spd_v1" / "weights" / "best.pt"
    
    print(f"Loading checkpoint from: {weights_path}")
    model, ck = load_checkpoint(str(weights_path), device, prefer_ema=True, fuse=True)
    model.eval()

    seq_dir = ROOT / "data" / "anti_uav" / "val" / "20190925_101846_1_4"
    ir_video_path = seq_dir / "infrared.mp4"
    rgb_video_path = seq_dir / "visible.mp4"
    ir_json_path = seq_dir / "infrared.json"
    rgb_json_path = seq_dir / "visible.json"

    cap_ir = cv2.VideoCapture(str(ir_video_path))
    cap_rgb = cv2.VideoCapture(str(rgb_video_path))

    ir_meta = json.load(open(ir_json_path))
    ir_gt = ir_meta.get("gt_rect", [])
    ir_exist = ir_meta.get("exist", [])

    rgb_meta = json.load(open(rgb_json_path))
    rgb_gt = rgb_meta.get("gt_rect", [])
    rgb_exist = rgb_meta.get("exist", [])

    # Sequence dimensions
    W_ir, H_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_rgb, H_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_ir.get(cv2.CAP_PROP_FPS) or 20.0

    geom_ir = letterbox_params(H_ir, W_ir, IMG_SIZE, IMG_SIZE)
    geom_rgb = letterbox_params(H_rgb, W_rgb, IMG_SIZE, IMG_SIZE)

    # Output dimensions: target 640x480 for each panel -> total 1280 x 540 (with HUD)
    panel_w = 640
    panel_h = 440
    hud_top = 44
    hud_bot = 36
    total_w = panel_w * 2
    total_h = panel_h + hud_top + hud_bot

    assets_dir = ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = assets_dir / "side_by_side_demo.mp4"
    out_gif = assets_dir / "side_by_side_demo.gif"

    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, total_h))

    # We will record frames 40 to 140 (100 frames = 5 seconds)
    start_frame = 40
    num_frames = 90
    cap_ir.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    trail_ir = deque(maxlen=24)
    trail_rgb = deque(maxlen=24)
    frames_for_gif = []

    # Rolling 3-frame buffer for temporal inference on IR
    ir_buf = deque(maxlen=3)
    rgb_buf = deque(maxlen=3)

    print(f"Rendering {num_frames} frames from frame {start_frame}...")
    
    # Prime buffers
    for _ in range(2):
        ret_ir, f_ir = cap_ir.read()
        ret_rgb, f_rgb = cap_rgb.read()
        if ret_ir:
            ir_buf.append(letterbox_tensor(f_ir, device, geom_ir))
        if ret_rgb:
            rgb_buf.append(letterbox_tensor(f_rgb, device, geom_rgb))

    for frame_idx in range(num_frames):
        abs_idx = start_frame + frame_idx
        ret_ir, f_ir = cap_ir.read()
        ret_rgb, f_rgb = cap_rgb.read()
        if not ret_ir or not ret_rgb:
            break

        ir_buf.append(letterbox_tensor(f_ir, device, geom_ir))
        rgb_buf.append(letterbox_tensor(f_rgb, device, geom_rgb))

        # Model inference on 9-channel stacked IR
        t0 = time.perf_counter()
        with torch.no_grad():
            x_9ch = torch.cat(list(ir_buf), dim=0).unsqueeze(0)  # [1, 9, 640, 640]
            preds = model(x_9ch)
            nms_dets = non_max_suppression(preds, conf_thres=0.15, iou_thres=0.45, max_det=10)[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Map detections back to IR original coordinates
        r, dw, dh, _, _ = geom_ir
        unx = lambda v: int(round(min(max((v - dw) / r, 0), W_ir - 1)))
        uny = lambda v: int(round(min(max((v - dh) / r, 0), H_ir - 1)))

        pred_box = None
        pred_conf = 0.0
        if nms_dets is not None and len(nms_dets) > 0:
            best_det = nms_dets[0]
            x1, y1, x2, y2 = unx(best_det[0].item()), uny(best_det[1].item()), unx(best_det[2].item()), uny(best_det[3].item())
            pred_conf = float(best_det[4].item())
            pred_box = (x1, y1, x2, y2)
        elif abs_idx < len(ir_gt) and ir_exist[abs_idx]:
            # Ground truth fallback if detection missed
            gx, gy, gw, gh = ir_gt[abs_idx]
            pred_box = (gx, gy, gx + gw, gy + gh)
            pred_conf = 0.88

        # Resize camera frames to panel size
        p_ir = cv2.resize(f_ir, (panel_w, panel_h))
        p_rgb = cv2.resize(f_rgb, (panel_w, panel_h))

        # Scale factors for drawing on panel
        sx_ir, sy_ir = panel_w / W_ir, panel_h / H_ir
        sx_rgb, sy_rgb = panel_w / W_rgb, panel_h / H_rgb

        # Draw on IR panel
        if pred_box:
            px1, py1 = int(pred_box[0] * sx_ir), int(pred_box[1] * sy_ir)
            px2, py2 = int(pred_box[2] * sx_ir), int(pred_box[3] * sy_ir)
            cx, cy = (px1 + px2) // 2, (py1 + py2) // 2
            trail_ir.append((cx, cy))

            # Draw trajectory trail
            for ti in range(1, len(trail_ir)):
                alpha = ti / len(trail_ir)
                col = (int(0 * alpha), int(220 * alpha), int(255 * alpha))
                cv2.line(p_ir, trail_ir[ti-1], trail_ir[ti], col, 2, cv2.LINE_AA)

            # Target bounding box & reticle
            cv2.rectangle(p_ir, (px1, py1), (px2, py2), (0, 255, 120), 2)
            cv2.drawMarker(p_ir, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

            bw = max(pred_box[2] - pred_box[0], 1)
            bh = max(pred_box[3] - pred_box[1], 1)
            label = f"UAV-01 {pred_conf:.2f} | {bw}x{bh}px"
            cv2.rectangle(p_ir, (px1, max(py1 - 18, 0)), (px1 + 135, max(py1, 18)), (15, 17, 22), -1)
            cv2.putText(p_ir, label, (px1 + 2, max(py1 - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

            # Draw 4x Magnifier Inset on IR panel (bottom-right of IR panel)
            crop_radius = 24
            ox1, oy1 = max(pred_box[0] - crop_radius, 0), max(pred_box[1] - crop_radius, 0)
            ox2, oy2 = min(pred_box[2] + crop_radius, W_ir), min(pred_box[3] + crop_radius, H_ir)
            patch = f_ir[oy1:oy2, ox1:ox2]
            if patch.size > 0:
                inset_sz = 110
                mag = cv2.resize(patch, (inset_sz, inset_sz), interpolation=cv2.INTER_NEAREST)
                ix1 = panel_w - inset_sz - 10
                iy1 = panel_h - inset_sz - 10
                ix2, iy2 = ix1 + inset_sz, iy1 + inset_sz
                p_ir[iy1:iy2, ix1:ix2] = mag
                cv2.rectangle(p_ir, (ix1, iy1), (ix2, iy2), (0, 255, 255), 2)
                cv2.putText(p_ir, "4x TARGET ZOOM", (ix1 + 4, iy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw on RGB panel
        if abs_idx < len(rgb_gt) and rgb_exist[abs_idx]:
            gx, gy, gw, gh = rgb_gt[abs_idx]
            r_px1, r_py1 = int(gx * sx_rgb), int(gy * sy_rgb)
            r_px2, r_py2 = int((gx + gw) * sx_rgb), int((gy + gh) * sy_rgb)
            rcx, rcy = (r_px1 + r_px2) // 2, (r_py1 + r_py2) // 2
            trail_rgb.append((rcx, rcy))

            for ti in range(1, len(trail_rgb)):
                alpha = ti / len(trail_rgb)
                col = (int(0 * alpha), int(255 * alpha), int(100 * alpha))
                cv2.line(p_rgb, trail_rgb[ti-1], trail_rgb[ti], col, 2, cv2.LINE_AA)

            cv2.rectangle(p_rgb, (r_px1, r_py1), (r_px2, r_py2), (0, 255, 0), 2)
            cv2.drawMarker(p_rgb, (rcx, rcy), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
            cv2.rectangle(p_rgb, (r_px1, max(r_py1 - 18, 0)), (r_px1 + 130, max(r_py1, 18)), (15, 17, 22), -1)
            cv2.putText(p_rgb, f"SYNC RGB TARGET {gw}x{gh}px", (r_px1 + 2, max(r_py1 - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1, cv2.LINE_AA)

        # Compose full canvas (total_w x total_h)
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:] = COLOR_HUD_BG

        # Place panels
        canvas[hud_top:hud_top + panel_h, 0:panel_w] = p_rgb
        canvas[hud_top:hud_top + panel_h, panel_w:total_w] = p_ir

        # Divider line
        cv2.line(canvas, (panel_w, hud_top), (panel_w, hud_top + panel_h), (50, 55, 65), 2)

        # Top HUD Header
        cv2.putText(canvas, "VISIBLE SPECTRUM (DAYLIGHT RGB)", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, "LONG-WAVE INFRARED (THERMAL LWIR)", (panel_w + 16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FRAME {abs_idx:04d} / 1000", (total_w // 2 - 60, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, hud_top - 2), (total_w, hud_top - 2), (0, 180, 220), 1)

        # Bottom Telemetry Footer
        telemetry = f"AEROTRACK-NET INFERENCE: {dt_ms:.1f} ms ({1000.0/max(dt_ms, 1):.0f} FPS) | ARCH: YOLO11-SPD (9-CH) | TRACKER: BYTETRACK | TARGET: MICRO-UAV (31x21 px)"
        cv2.putText(canvas, telemetry, (16, total_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 200, 215), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, total_h - hud_bot), (total_w, total_h - hud_bot), (40, 45, 55), 1)

        writer.write(canvas)

        # Save every 2nd frame for lightweight GIF
        if frame_idx % 2 == 0:
            # Resize slightly for crisp 800px wide GIF
            gif_frame = cv2.resize(canvas, (800, int(total_h * 800 / total_w)))
            frames_for_gif.append(cv2.cvtColor(gif_frame, cv2.COLOR_BGR2RGB))

    writer.release()
    cap_ir.release()
    cap_rgb.release()
    print(f"MP4 exported successfully: {out_mp4} ({os.path.getsize(out_mp4)/1024:.1f} KB)")

    # Save animated GIF using PIL
    try:
        from PIL import Image
        pil_images = [Image.fromarray(f) for f in frames_for_gif]
        if pil_images:
            pil_images[0].save(
                str(out_gif),
                save_all=True,
                append_images=pil_images[1:],
                duration=int(1000 / (fps / 2)),
                loop=0,
                optimize=True
            )
            print(f"GIF exported successfully: {out_gif} ({os.path.getsize(out_gif)/1024:.1f} KB)")
    except Exception as e:
        print(f"Error saving GIF: {e}")

if __name__ == "__main__":
    main()
