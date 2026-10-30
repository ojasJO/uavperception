#!/usr/bin/env python
"""Honest side-by-side demo on a VAL sequence chosen by a label-based rule (repair item 8).

    python experiments/aerotrack_repair/scripts/make_demo.py \
        --left results/runs/aerotrack_spd_v1/weights/last.pt --right results/runs/aerotrack_spd_finetuned/weights/best.pt

Sequence rule (no model output involved): among Anti-UAV **IR VAL** sequences, the first in sorted order whose labels
contain at least `--min-occluded` exist=0 frames (i.e. a genuine out-of-view event). The clip starts `--before` frames
before the first exist=0 frame and lasts `--frames` frames. Both models see identical frames; detections are the raw
NMS output at conf `--conf` (no tracker, no trail, no magnifier). The HUD shows, per frame, the top drone box's IoU with
the ground truth (or "absent"/"no det") and the running detection rate for each model. The per-frame numbers are
written to a JSON next to the GIF so the animation can be audited.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import torch

from aerotrack_trainer import load_checkpoint
from evaluation import box_iou_np
from stacked_dataset import AeroTrackDataset, build_split_samples, load_label_cache
from track_eval import detect_sequence
from train import LABEL_CACHE_PATH, MANIFEST, SPLITS

from anti_uav_repro.utils.repro import file_sha256, git_info, write_json

from _bootstrap import REPO_ROOT


def choose_sequence(min_occluded: int):
    samples = build_split_samples(MANIFEST, SPLITS, "val", datasets=["anti_uav"])
    by_seq = defaultdict(list)
    for s in samples:
        if s.get("modality") == "IR":
            by_seq[s["seq_key"]].append(s)
    cache = load_label_cache(LABEL_CACHE_PATH)
    for key in sorted(by_seq):
        frames = sorted(by_seq[key], key=lambda s: int(s["frame_idx"]))
        # exist == 0 <-> empty label (0-byte file); use the on-disk label to stay model-free
        exist = []
        for s in frames:
            p = Path(s["label_path"])
            exist.append(1 if (p.exists() and p.stat().st_size > 0) else 0)
        n_occ = sum(1 for e in exist if e == 0)
        if n_occ >= min_occluded:
            first_occ = exist.index(0)
            return key, frames, exist, first_occ, cache
    raise SystemExit("no IR val sequence satisfies the rule")


def top_drone_box(dets, conf):
    d = dets[dets[:, 4] >= conf] if len(dets) else dets
    return d[np.argmax(d[:, 4])] if len(d) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--min-occluded", type=int, default=20)
    ap.add_argument("--before", type=int, default=60)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=2, help="render every k-th clip frame (detections are still computed on every frame)")
    a = ap.parse_args()
    key, frames, exist, first_occ, cache = choose_sequence(a.min_occluded)
    start = max(0, first_occ - a.before)
    clip = frames[start:start + a.frames]
    print(f"[demo] sequence {key}: {len(frames)} frames, {sum(1 for e in exist if not e)} exist=0; clip frames {start}..{start+len(clip)-1}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if device.type == "cuda" else None
    results = {}
    for side, w in (("left", a.left), ("right", a.right)):
        model, ck = load_checkpoint(w, device, prefer_ema=True)
        dets, gt, occ = detect_sequence(model, clip, device, conf=0.05, batch=16, workers=0, amp_dtype=amp, label_cache=cache)
        results[side] = {"weights": w, "sha256": file_sha256(w)[:12], "run": Path(w).parent.parent.name, "ckpt": Path(w).stem,
                         "epoch": ck.get("epoch"), "dets": dets, "gt": gt}
        del model
        torch.cuda.empty_cache()
    ds = AeroTrackDataset(clip, img_size=(640, 640), augment=None, letterbox=True, out_dtype="uint8", return_meta=True, label_cache=cache)
    gif_frames, per_frame = [], []
    hits = {"left": 0, "right": 0}
    n_exist = 0
    for i in range(len(clip)):
        render = (i % a.stride == 0)
        img9, _, _ = ds[i]
        cur = img9[3:6].numpy().transpose(1, 2, 0)[:, :, ::-1].copy()      # current frame, BGR uint8, letterboxed 640
        gt = results["left"]["gt"][i]
        rec = {"frame_idx": int(clip[i]["frame_idx"]), "exist": int(len(gt) > 0)}
        n_exist += rec["exist"]
        panels = []
        for side in ("left", "right"):
            pan = cur.copy()
            d = top_drone_box(results[side]["dets"][i], a.conf)
            if len(gt):
                g = gt[0]
                cv2.rectangle(pan, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 200, 0), 1)
            iou = None
            if d is not None:
                cv2.rectangle(pan, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 165, 255), 2)
                if len(gt):
                    iou = float(box_iou_np(d[None, :4], gt[:, :4])[0, 0])
                    hits[side] += int(iou >= 0.5)
            r = results[side]
            status = ("absent frame: " + ("FALSE ALARM" if d is not None else "no det (correct)")) if not len(gt) else \
                     (f"IoU {iou:.2f}" if iou is not None else "MISSED")
            rate = 100 * hits[side] / max(n_exist, 1)
            cv2.rectangle(pan, (0, 0), (640, 44), (20, 20, 20), -1)
            cv2.putText(pan, f"{r['run']}/{r['ckpt']} (epoch {r['epoch']})  conf>={a.conf}  no tracker", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(pan, f"{status} | hit rate so far {rate:.0f}% (IoU>=.5 on exist frames)", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            rec[side] = {"iou": iou, "det": None if d is None else [float(v) for v in d[:5]]}
            panels.append(pan)
        canvas = np.concatenate(panels, 1)
        cv2.rectangle(canvas, (0, 640 - 26), (1280, 640), (20, 20, 20), -1)
        cv2.putText(canvas, f"VAL {key}  frame {rec['frame_idx']}  GT: {'target present (green box)' if rec['exist'] else 'target absent (exist=0)'}  |  rule: first IR val seq with >={a.min_occluded} absent frames",
                    (8, 640 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
        if render:
            if a.scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
            gif_frames.append(canvas[:, :, ::-1])
        per_frame.append(rec)
    out = REPO_ROOT / "docs/assets" / f"repair_demo_val_{key}.gif"
    try:
        import imageio.v2 as imageio
        imageio.mimsave(out, gif_frames, duration=1.0 / a.fps, loop=0)
    except ImportError:
        from PIL import Image
        ims = [Image.fromarray(f) for f in gif_frames]
        ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(1000 / a.fps), loop=0)
    summary = {side: {"run": results[side]["run"], "ckpt": results[side]["ckpt"], "epoch": results[side]["epoch"], "sha256": results[side]["sha256"],
                      "hit_rate_iou50_on_exist_frames": hits[side] / max(n_exist, 1),
                      "false_alarms_on_absent_frames": sum(1 for r in per_frame if not r["exist"] and r[side]["det"] is not None)}
               for side in ("left", "right")}
    write_json({"sequence": key, "rule": f"first Anti-UAV IR VAL sequence (sorted) with >= {a.min_occluded} exist=0 frames; clip starts {a.before} frames before the first absent frame",
                "clip_frames": [start, start + len(clip) - 1], "n_frames": len(clip), "n_exist": n_exist, "conf": a.conf, "git": git_info().get("commit"),
                "summary": summary, "per_frame": per_frame}, out.with_suffix(".json"))
    print(f"wrote {out} ({len(gif_frames)} frames) and {out.with_suffix('.json')}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
