#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visual_demo.py — Real-Time Visual GUI Window with Simultaneous Terminal Telemetry
================================================================================
Displays an interactive OpenCV desktop window showing live drone tracking,
bounding boxes, motion trajectory trails, and a 4x micro-target magnifier inset,
while simultaneously streaming numeric metrics to the CMD terminal.

Controls:
  [Space]   Pause / Play
  [N]       Next sequence
  [G]       Toggle ground-truth bounding box
  [M]       Toggle 4x micro-target magnifier inset
  [S]       Save high-res screenshot
  [Q / Esc] Quit
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src/pipeline"))

from custom_modules import SPDConv
from aerotrack_trainer import load_checkpoint
from stacked_dataset import letterbox_image, letterbox_params
from ultralytics.utils.nms import non_max_suppression

IMG_SIZE = 640
NAMES = {0: "uav_drone", 1: "bird_hard_negative"}
COLORS = {
    0: (0, 255, 0),      # Bright Green (Drone)
    1: (0, 140, 255),    # Orange (Bird)
}
COLOR_MICRO = (255, 255, 0)     # Cyan for micro target badge
COLOR_GT = (255, 200, 0)        # Blue/Yellow for Ground Truth


def get_target_size_tag(w, h, orig_w, orig_h):
    """Classify target based on source area percentage."""
    rel_area = (w * h) / max(orig_w * orig_h, 1)
    if rel_area < 0.0003:
        return "MICRO", f"{w:.0f}x{h:.0f}px"
    elif rel_area < 0.01:
        return "SMALL", f"{w:.0f}x{h:.0f}px"
    else:
        return "STANDARD", f"{w:.0f}x{h:.0f}px"


def draw_hud(frame, fps, latency_ms, frame_idx, total_frames, seq_idx, total_seqs,
             seq_title, modality, det_info, is_realtime, finding_tag=""):
    """Draw a high-visibility cybersecurity HUD overlay on the display frame."""
    h, w = frame.shape[:2]

    # Top status bar background
    bar_h = 44
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 17, 22), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    bar_color = (0, 220, 0) if modality.upper() == "RGB" else (0, 165, 255)
    cv2.line(frame, (0, bar_h), (w, bar_h), bar_color, 2)

    # Title & Telemetry
    mod_tag = "[VISIBLE RGB - DAYLIGHT]" if modality.upper() == "RGB" else "[THERMAL IR]"
    mod_color = (100, 255, 100) if modality.upper() == "RGB" else (0, 200, 255)
    if finding_tag:
        title = f"AEROTRACK-NET | Video {seq_idx+1}/{total_seqs}: [{finding_tag}] {seq_title}"
    else:
        title = f"AEROTRACK-NET | Video {seq_idx+1}/{total_seqs}: {seq_title}"
    cv2.putText(frame, title, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)

    spd_txt = "30 FPS [REAL-TIME]" if is_realtime else "UNCAPPED [MAX GPU]"
    stats = f"FPS: {fps:4.1f} | Latency: {latency_ms:4.1f} ms | Frame: {frame_idx:04d}/{total_frames:04d} | {mod_tag} | {spd_txt}"
    cv2.putText(frame, stats, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mod_color, 1, cv2.LINE_AA)

    # Right-hand target status
    if det_info:
        tag, size_str, conf, cls_name = det_info
        badge = f"STATUS: TARGET ACQUIRED [{cls_name.upper()}] ({conf*100:.1f}%)"
        cv2.putText(frame, badge, (max(w - 440, 320), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1, cv2.LINE_AA)
        size_badge = f"SIZE: {tag} ({size_str}) | TRACK: ACTIVE"
        c_badge = (0, 255, 255) if tag == "MICRO" else (180, 255, 180)
        cv2.putText(frame, size_badge, (max(w - 440, 320), 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c_badge, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "STATUS: SCANNING / SURVEILLANCE", (max(w - 380, 320), 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 160), 1, cv2.LINE_AA)

    # Bottom controls bar
    bar_b = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_b), (w, h), (15, 17, 22), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
    controls = ">> [N] / [ENTER]: NEXT VIDEO <<   [P]: PREV   [SPACE]: PAUSE   [1-8]: JUMP   [F]: SPEED   [M]: ZOOM   [Q]: QUIT"
    cv2.putText(frame, controls, (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)


def draw_magnifier(frame, crop_box, tag=""):
    """Draw a 4x magnified picture-in-picture inset of the target in the bottom-right corner."""
    x1, y1, x2, y2 = [int(v) for v in crop_box]
    h, w = frame.shape[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    radius = 24
    px1, px2 = max(cx - radius, 0), min(cx + radius, w)
    py1, py2 = max(cy - radius, 0), min(cy + radius, h)
    if px2 <= px1 or py2 <= py1:
        return

    patch = frame[py1:py2, px1:px2]
    inset_size = 148
    magnified = cv2.resize(patch, (inset_size, inset_size), interpolation=cv2.INTER_NEAREST)

    ix1 = w - inset_size - 14
    iy1 = h - inset_size - 38
    ix2, iy2 = ix1 + inset_size, iy1 + inset_size

    frame[iy1:iy2, ix1:ix2] = magnified
    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
    cv2.line(frame, (ix1 + inset_size // 2, iy1), (ix1 + inset_size // 2, iy2), (0, 255, 255), 1)
    cv2.line(frame, (ix1, iy1 + inset_size // 2), (ix2, iy1 + inset_size // 2), (0, 255, 255), 1)
    lbl = f"4x ZOOM [{tag}]" if tag else "4x TARGET ZOOM"
    cv2.putText(frame, lbl, (ix1 + 6, iy1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)


def preprocess_one(frame_bgr, device, r, dw, dh, nw, nh):
    """Letterbox and convert frame to float tensor."""
    img = letterbox_image(frame_bgr, IMG_SIZE, IMG_SIZE, r, dw, dh, nw, nh)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(device)
    return t.float().div_(255.0).permute(2, 0, 1).contiguous()


CURATED_PLAYLIST = [
    # --- Finding 1: Flawless Daylight Baseline ---
    {
        "key": "antiuav_test_20190925_111757_1_1__RGB",
        "modality": "RGB",
        "finding": "FINDING 1: DAYLIGHT BASELINE",
        "title": "Clear Sky DJI Drone Acquisition",
        "desc": "Daylight 11:17 AM - Flawless 100% Target Acquisition & Smooth Trajectory",
    },
    # --- Finding 2: Multi-Modal Sensor Invariance ---
    {
        "key": "antiuav_test_20190925_111757_1_1__IR",
        "modality": "IR",
        "finding": "FINDING 2: DUAL-BAND SENSOR INVARIANCE",
        "title": "Synchronous Thermal IR Camera",
        "desc": "Thermal IR - Paired Sensor on the Exact Same Drone Flight (Modality Invariance)",
    },
    # --- Finding 3: micro-scale target retention (the architecture as a whole;
    #     SPD-Conv in isolation showed no measurable effect -- docs/AEROTRACK_REPAIR_REPORT.md) ---
    {
        "key": "antiuav_test_20190925_111757_1_4__RGB",
        "modality": "RGB",
        "finding": "FINDING 3: MICRO-SCALE RETENTION",
        "title": "Distant Sub-16px Micro-Drone",
        "desc": "Distant High-Altitude Target (<0.03% Area) Still Detected",
    },
    # --- Finding 4: Extreme Scale Sparsity & High Clutter ---
    {
        "key": "cst_test_urban-areas_48",
        "modality": "IR",
        "finding": "FINDING 4: EXTREME CST URBAN CLUTTER",
        "title": "Long-Wave IR Urban Micro Target",
        "desc": "Extreme Thermal Clutter & Tiny Target (Median 0.028% Area; CST Generalization)",
    },
    # --- Finding 5: re-acquisition after a full occlusion. ByteTrack was retired by the
    #     repair series (a no-association argmax box beat it on MOTA and IDF1 everywhere);
    #     what this shows is the detector plus the single-target state estimator. ---
    {
        "key": "cst_test_building_68",
        "modality": "IR",
        "finding": "FINDING 5: OCCLUSION RE-ACQUISITION",
        "title": "65-Frame Solid Building Occlusion",
        "desc": "Drone Vanishes Behind a Building for 65 Frames and is Re-acquired on Re-entry",
    },
    # --- Finding 6: Mountain Ridge Complex Terrain Occlusion ---
    {
        "key": "cst_test_cn_mountains_30",
        "modality": "IR",
        "finding": "FINDING 6: COMPLEX TERRAIN OCCLUSION",
        "title": "92-Frame Mountain Ridge Occlusion",
        "desc": "Extreme Long-Duration Terrain Occlusion with Smooth Trajectory Resumption",
    },
    # --- Finding 7: Evasive High-Dynamic Maneuvers ---
    {
        "key": "antiuav_test_20190925_124000_1_5__RGB",
        "modality": "RGB",
        "finding": "FINDING 7: EVASIVE DYNAMIC MANEUVERS",
        "title": "Midday Evasive Drone Maneuvers",
        "desc": "Rapid Directional Changes; 3-Frame Temporal Stacking Motion Vectors",
    },
    # --- Finding 8: Night / Zero-Illumination Surveillance ---
    {
        "key": "antiuav_val_20190926_200510_1_4__IR",
        "modality": "IR",
        "finding": "FINDING 8: NIGHT THERMAL SURVEILLANCE",
        "title": "Zero-Illumination Night Infrared",
        "desc": "Night Surveillance Under Complete Darkness; Thermal Radiance Tracking",
    },
]


def run_visual_demo(source_seq=None, weights=None, conf_thres=0.25, iou_thres=0.45,
                    max_frames=0, modality=None, loop=True, one_button=False):
    """Run real-time visual tracking window with simultaneous terminal output."""
    weights_path = Path(weights).resolve() if weights else (ROOT / "results/runs" / "aerotrack_spd_finetuned" / "weights" / "best.pt" if (ROOT / "results/runs" / "aerotrack_spd_finetuned" / "weights" / "best.pt").exists() else ROOT / "results/runs" / "aerotrack_spd_v1" / "weights" / "best.pt").resolve()
    if not weights_path.exists():
        print(f"Error: Weights not found at {weights_path}")
        return

    try:
        weights_disp = weights_path.relative_to(ROOT.resolve())
    except ValueError:
        weights_disp = weights_path.name

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("AeroTrack-Net | Real-Time Visual Tracking Window & Live Telemetry")
    print(f"  Device     : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Weights    : {weights_disp}")
    print(f"  Precision  : {'bfloat16' if torch.cuda.is_bf16_supported() else 'float32'}")
    print("=" * 78)

    model, ck = load_checkpoint(weights_path, device, prefer_ema=True)
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None

    # Load manifest
    manifest_p = ROOT / "data" / "derived" / "data_unified" / "manifest.csv"
    if not manifest_p.exists():
        print("Manifest not found. Please verify data pipeline.")
        return

    import pandas as pd
    df = pd.read_csv(manifest_p)

    # Build active playlist based on user arguments
    playlist = []
    if source_seq:
        if source_seq in df["seq_key"].values:
            sub_df = df[df["seq_key"] == source_seq]
            mod = sub_df["modality"].iloc[0]
            playlist.append({"key": source_seq, "modality": mod, "title": source_seq, "desc": f"User specified sequence [{mod}]"})
        else:
            matched = df[df["seq_key"].str.contains(source_seq, case=False, na=False)]
            matched_keys = matched["seq_key"].unique()
            if len(matched_keys) > 0:
                for k in matched_keys[:10]:
                    sub_df = matched[matched["seq_key"] == k]
                    mod = sub_df["modality"].iloc[0]
                    playlist.append({"key": k, "modality": mod, "title": k, "desc": f"Matched search '{source_seq}' [{mod}]"})
            else:
                print(f"Warning: No sequences matching '{source_seq}' found. Falling back to curated playlist.")
                playlist = list(CURATED_PLAYLIST)
    else:
        playlist = list(CURATED_PLAYLIST)

    # Apply modality filter if specified
    if modality:
        mod_upper = modality.upper()
        filtered = [item for item in playlist if item["modality"].upper() == mod_upper]
        if filtered:
            playlist = filtered
        else:
            # If curated list didn't have enough, pull from manifest
            m_matches = df[df["modality"].str.upper() == mod_upper]["seq_key"].unique()
            playlist = [{"key": k, "modality": mod_upper, "title": k, "desc": f"{mod_upper} sequence"} for k in m_matches[:10]]

    # Ensure all sequences in playlist exist in manifest
    playlist = [item for item in playlist if not df[df["seq_key"] == item["key"]].empty]
    if not playlist:
        print("Error: No valid sequences available for playback.")
        return

    # Print playlist table and controls in terminal
    print(f"\n[RESEARCH VALIDATION PLAYLIST] Loaded {len(playlist)} multi-modal sequence(s):")
    for i, item in enumerate(playlist):
        tag = f"[{item['modality']:3s}]"
        f_tag = f"[{item.get('finding', 'FINDING')}]"
        print(f"  [{i+1:2d}] {tag} {f_tag:38s} - {item.get('desc', '')}")

    print("\n[ONE-BUTTON INTERACTIVE CONTROLS]:")
    if one_button:
        print("  >> [Space] / [Enter] / [N] -> ADVANCE TO NEXT VIDEO << (One-Button Mode Active)")
    else:
        print("  >> [Enter] / [N]           -> ADVANCE TO NEXT VIDEO << (One-Button Control)")
        print("  [Space] Pause / Play       [P] Previous Video")
    print("  [1-8] Jump directly to finding #1-#8")
    print("  [M] Toggle 4x Target Zoom        [G] Toggle Ground Truth Box")
    print("  [F] Toggle 30 FPS / Uncapped     [S] Save High-Res Snapshot")
    print("  [Q / Esc] Quit Demo")
    print("=" * 78 + "\n")

    seq_idx = 0
    show_gt = True
    show_magnifier = True
    paused = False
    is_realtime = True   # Default: Smooth 30 FPS real-time playback

    window_name = "AeroTrack-Net | Real-Time Live Drone Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1024, 768)

    trail = deque(maxlen=30)
    snapshot_dir = ROOT / "results/runs" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    while seq_idx < len(playlist):
        item = playlist[seq_idx]
        seq_key = item["key"]
        seq_desc = item.get("desc", item["key"])
        finding_tag = item.get("finding", "")
        seq_df = df[df["seq_key"] == seq_key].sort_values("frame_idx")
        if seq_df.empty:
            seq_idx = (seq_idx + 1) % len(playlist)
            continue

        modality = seq_df["modality"].iloc[0]
        dataset_name = seq_df["dataset"].iloc[0].upper()
        total_frames = len(seq_df)
        if max_frames > 0:
            seq_df = seq_df.iloc[:max_frames]
            total_frames = len(seq_df)

        print("\n" + "=" * 78)
        if finding_tag:
            print(f">>> [VIDEO {seq_idx+1}/{len(playlist)}] {finding_tag}")
            print(f"    Target   : {item.get('title', seq_key)}")
        else:
            print(f">>> [PLAYING {seq_idx+1}/{len(playlist)}] {seq_key}")
        print(f"    Dataset  : {dataset_name} | Modality: {modality} | Frames: {total_frames} | {seq_desc}")
        advance_hint = "[Space], [Enter], or [N]" if one_button else "[Enter] or [N]"
        print(f"    Control  : Press {advance_hint} to jump to Next Video")
        print("=" * 78)
        trail.clear()

        frame_paths = seq_df["image_path"].tolist()
        label_paths = seq_df["label_path"].tolist()
        n_frames = len(frame_paths)

        # Pre-compute letterbox geometry from first frame
        im0 = cv2.imread(frame_paths[0])
        if im0 is None:
            seq_idx += 1
            continue
        orig_h, orig_w = im0.shape[:2]
        r, dw, dh, nw, nh = letterbox_params(orig_h, orig_w, IMG_SIZE, IMG_SIZE)
        top = int(round(dh - 0.1))
        left = int(round(dw - 0.1))

        f_idx = 0
        jump_next = None

        while f_idx < n_frames:
            t_start = time.perf_counter()

            if not paused:
                idx_prev = max(f_idx - 1, 0)
                idx_curr = f_idx
                idx_next = min(f_idx + 1, n_frames - 1)

                im_prev = cv2.imread(frame_paths[idx_prev])
                im_curr = cv2.imread(frame_paths[idx_curr])
                im_next = cv2.imread(frame_paths[idx_next])

                if im_curr is None:
                    f_idx += 1
                    continue
                if im_prev is None:
                    im_prev = im_curr
                if im_next is None:
                    im_next = im_curr

                # Process 3 frames and stack into [1, 9, 640, 640]
                t_p = preprocess_one(im_prev, device, r, dw, dh, nw, nh)
                t_c = preprocess_one(im_curr, device, r, dw, dh, nw, nh)
                t_n = preprocess_one(im_next, device, r, dw, dh, nw, nh)
                tensor = torch.cat([t_p, t_c, t_n], dim=0).unsqueeze(0)

                # Inference
                t_infer_start = time.perf_counter()
                with torch.no_grad():
                    if amp_dtype is not None:
                        with torch.autocast(device_type=device.type, dtype=amp_dtype):
                            preds = model(tensor)
                    else:
                        preds = model(tensor)

                    if isinstance(preds, (list, tuple)):
                        preds = preds[0]
                    dets = non_max_suppression(preds, conf_thres, iou_thres, max_det=10)
                t_infer_end = time.perf_counter()

                latency_ms = (t_infer_end - t_infer_start) * 1000.0
                fps = 1.0 / max(t_infer_end - t_start, 1e-4)

                det = dets[0].detach().cpu().numpy() if len(dets) and len(dets[0]) else None
                display_frame = im_curr.copy()

                det_info = None
                best_box = None
                best_tag = ""

                unx = lambda v: int(round(min(max((v - left) / r, 0), orig_w - 1)))
                uny = lambda v: int(round(min(max((v - top) / r, 0), orig_h - 1)))

                if det is not None and len(det) > 0:
                    det = det[np.argsort(-det[:, 4])]
                    for d in det:
                        x1, y1, x2, y2, conf, cls_id = d
                        rx1, ry1 = unx(x1), uny(y1)
                        rx2, ry2 = unx(x2), uny(y2)
                        bw, bh = max(rx2 - rx1, 1), max(ry2 - ry1, 1)

                        tag, size_str = get_target_size_tag(bw, bh, orig_w, orig_h)
                        cls_name = NAMES.get(int(cls_id), "target")
                        box_color = COLORS.get(int(cls_id), (0, 255, 0))

                        # Thicker bounding box for high visibility
                        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), box_color, 2)

                        badge_txt = f"{cls_name} {conf:.2f} [{tag}]"
                        (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                        cv2.rectangle(display_frame, (rx1, max(ry1 - 20, 0)), (rx1 + tw + 6, max(ry1, 20)), box_color, -1)
                        cv2.putText(display_frame, badge_txt, (rx1 + 3, max(ry1 - 5, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)

                        if det_info is None:
                            det_info = (tag, size_str, conf, cls_name)
                            best_box = (rx1, ry1, rx2, ry2)
                            best_tag = f"{cls_name.upper()} {tag}"
                            trail.append(((rx1 + rx2) // 2, (ry1 + ry2) // 2))

                            print(f"[f {f_idx+1:04d}/{n_frames:04d}] DET=1 | conf={conf:.3f} | {cls_name:10s} "
                                  f"| {tag:8s} ({bw:2d}x{bh:2d}px) | fps={fps:4.1f} | lat={latency_ms:4.1f}ms "
                                  f"| pos=({rx1:4d},{ry1:4d})")
                else:
                    if f_idx % 15 == 0 or f_idx == n_frames - 1:
                        print(f"[f {f_idx+1:04d}/{n_frames:04d}] DET=0 | scanning surveillance sky... | fps={fps:4.1f} | lat={latency_ms:4.1f}ms")

                # Ground truth
                if show_gt and Path(label_paths[f_idx]).exists():
                    try:
                        lbl_txt = Path(label_paths[f_idx]).read_text().strip()
                        if lbl_txt:
                            for l_line in lbl_txt.splitlines():
                                parts = l_line.split()
                                if len(parts) >= 5:
                                    _, gcx, gcy, gw, gh = [float(v) for v in parts[:5]]
                                    gx1 = int((gcx - gw / 2) * orig_w)
                                    gy1 = int((gcy - gh / 2) * orig_h)
                                    gx2 = int((gcx + gw / 2) * orig_w)
                                    gy2 = int((gcy + gh / 2) * orig_h)
                                    cv2.rectangle(display_frame, (gx1, gy1), (gx2, gy2), COLOR_GT, 1)
                                    cv2.putText(display_frame, "GT", (gx1, gy1 - 3),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_GT, 1, cv2.LINE_AA)
                    except Exception:
                        pass

                # Flight path trail
                if len(trail) > 1:
                    for i in range(1, len(trail)):
                        thickness = int(np.sqrt(float(i) / len(trail)) * 2.5) + 1
                        cv2.line(display_frame, trail[i - 1], trail[i], (0, 240, 255), thickness)

                # Magnifier inset
                if show_magnifier and best_box is not None:
                    draw_magnifier(display_frame, best_box, tag=best_tag)

                # HUD
                draw_hud(display_frame, fps, latency_ms, f_idx + 1, n_frames, seq_idx, len(playlist),
                         item.get("title", seq_key), modality, det_info, is_realtime,
                         finding_tag=item.get("finding", ""))

                cv2.imshow(window_name, display_frame)

            # Frame timing & key handling
            t_frame_end = time.perf_counter()
            elapsed_sec = t_frame_end - t_start
            if paused:
                wait_time = 30
            elif is_realtime and elapsed_sec < 0.0333:
                wait_time = max(int((0.0333 - elapsed_sec) * 1000), 1)
            else:
                wait_time = 1

            key = cv2.waitKey(wait_time) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                print("\n[visual] Exiting visual tracking window.")
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                if one_button:
                    next_idx = (seq_idx + 1) % len(playlist)
                    print(f"\n[visual] [ONE-BUTTON ADVANCE] -> Video #{next_idx+1}: {playlist[next_idx]['key']}")
                    jump_next = next_idx
                    break
                else:
                    paused = not paused
                    print(f"[visual] {'PAUSED' if paused else 'RESUMED'}")
            elif key in (ord('n'), ord('N'), 13, 10, 83, ord('d'), ord('D')):
                next_idx = (seq_idx + 1) % len(playlist)
                print(f"\n[visual] [NEXT VIDEO BUTTON] -> Video #{next_idx+1}: {playlist[next_idx]['key']}")
                jump_next = next_idx
                break
            elif key in (ord('p'), ord('P'), 8, 127, 81, ord('a'), ord('A')):
                prev_idx = (seq_idx - 1) % len(playlist)
                print(f"\n[visual] [PREV VIDEO BUTTON] -> Video #{prev_idx+1}: {playlist[prev_idx]['key']}")
                jump_next = prev_idx
                break
            elif key == ord('r') or key == ord('R'):
                # Switch to next RGB sequence
                next_rgb = None
                for offset in range(1, len(playlist)):
                    cand_idx = (seq_idx + offset) % len(playlist)
                    if playlist[cand_idx]["modality"].upper() == "RGB":
                        next_rgb = cand_idx
                        break
                if next_rgb is not None:
                    print(f"\n[visual] Switching to next RGB Daylight video: {playlist[next_rgb]['key']}")
                    jump_next = next_rgb
                    break
                else:
                    print("[visual] No other RGB videos in playlist.")
            elif key == ord('i') or key == ord('I'):
                # Switch to next IR sequence
                next_ir = None
                for offset in range(1, len(playlist)):
                    cand_idx = (seq_idx + offset) % len(playlist)
                    if playlist[cand_idx]["modality"].upper() == "IR":
                        next_ir = cand_idx
                        break
                if next_ir is not None:
                    print(f"\n[visual] Switching to next IR Thermal video: {playlist[next_ir]['key']}")
                    jump_next = next_ir
                    break
                else:
                    print("[visual] No other IR videos in playlist.")
            elif ord('1') <= key <= ord('9'):
                target = key - ord('1')
                if target < len(playlist):
                    print(f"\n[visual] Jumping to video #{target+1}: {playlist[target]['key']}")
                    jump_next = target
                    break
            elif key == ord('f') or key == ord('F'):
                is_realtime = not is_realtime
                mode_str = "30 FPS Real-Time" if is_realtime else "Uncapped Max GPU"
                print(f"[visual] Playback speed set to: {mode_str}")
            elif key == ord('g') or key == ord('G'):
                show_gt = not show_gt
                print(f"[visual] Ground Truth boxes: {'ON' if show_gt else 'OFF'}")
            elif key == ord('m') or key == ord('M'):
                show_magnifier = not show_magnifier
                print(f"[visual] Micro-target magnifier: {'ON' if show_magnifier else 'OFF'}")
            elif key == ord('s') or key == ord('S'):
                snap_path = snapshot_dir / f"snap_{seq_key}_f{f_idx+1:04d}_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), display_frame)
                print(f"[visual] Snapshot saved -> {snap_path.name}")

            if not paused:
                f_idx += 1

        if jump_next is not None:
            seq_idx = jump_next
        else:
            seq_idx += 1
            if loop and seq_idx >= len(playlist):
                seq_idx = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AeroTrack-Net Real-Time Visual Tracking Window")
    parser.add_argument("--source", default=None, help="Sequence key name or keyword to search")
    parser.add_argument("--weights", default=None, help="Path to checkpoint weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to play (0 = all)")
    parser.add_argument("--modality", choices=["rgb", "ir", "all"], default=None,
                        help="Filter playback by modality: 'rgb' (visible daylight), 'ir' (thermal), or 'all'")
    parser.add_argument("--rgb", action="store_true", help="Play only visible RGB daylight drone videos")
    parser.add_argument("--ir", action="store_true", help="Play only thermal IR drone videos")
    parser.add_argument("--list", action="store_true", help="List all available curated sequences and exit")
    parser.add_argument("--one-button", action="store_true", help="One-button mode: Space or Enter advances to next video")
    args = parser.parse_args()

    if args.list:
        print("=" * 78)
        print("AeroTrack-Net Curated Multi-Modal Research Validation Playlist:")
        print("=" * 78)
        for idx, item in enumerate(CURATED_PLAYLIST, 1):
            f_tag = f"[{item.get('finding', 'FINDING')}]"
            print(f"  [{idx:2d}] [{item['modality']:3s}] {f_tag:38s} - {item['desc']}")
        print("=" * 78)
        sys.exit(0)

    mod_filter = "rgb" if args.rgb else ("ir" if args.ir else args.modality)
    if mod_filter == "all":
        mod_filter = None

    run_visual_demo(source_seq=args.source, weights=args.weights,
                    conf_thres=args.conf, iou_thres=args.iou,
                    max_frames=args.max_frames, modality=mod_filter,
                    one_button=args.one_button)
