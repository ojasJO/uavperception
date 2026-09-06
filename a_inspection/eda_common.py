#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda_common.py
=============
Shared infrastructure for the UAV dataset inspection pipeline.

Provides:
  * canonical paths + constants (micro-target threshold, etc.)
  * robust video probing (cv2) and image-size probing (PIL, header-only)
  * format-adaptive discovery of the three datasets:
        - Anti-UAV / CST Anti-UAV : per-sequence (video|frame-folder) + JSON
                                    annotations of the form {exist[], gt_rect[]}
        - Det-Fly                 : Roboflow YOLO export (images + txt labels)
  * build_master_index(): a single pass that produces two tables persisted to
        a_inspection/artifacts/:
            media_meta.csv   -> one row per video/image (resolution, #frames…)
            bboxes.pkl        -> long-form per-frame bounding-box table
        Downstream scripts (3_/4_/5_) just load these, so the expensive
        annotation parse happens exactly once.

Bounding-box convention (unified, pixel space, top-left origin):
    x, y, w, h  ->  centre = (x + w/2, y + h/2)
    area_ratio  =  (w*h) / (W*H)         with (W,H) the frame resolution
"""

import os
import io
import cv2
import json
import time
import pickle
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Paths & constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent            # D:\Drone_Maxxing
DATA = ROOT / "data"
INSP = ROOT / "a_inspection"
ART = INSP / "artifacts"
PLOTS = INSP / "plots"
LOGS = INSP / "logs"
for d in (ART, PLOTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

DATASET_DIRS = {
    "anti_uav": DATA / "anti_uav",
    "cst_anti_uav": DATA / "cst_anti_uav",
    "det_fly": DATA / "det_fly",
}
DATASET_LABEL = {
    "anti_uav": "Anti-UAV",
    "cst_anti_uav": "CST Anti-UAV",
    "det_fly": "Det-Fly",
}

MICRO_THRESHOLD = 0.0003        # 0.03 % of frame area  -> "micro" target
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

MEDIA_CSV = ART / "media_meta.csv"
BBOX_PKL = ART / "bboxes.pkl"


def log(msg):
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:,.1f} {u}"
        n /= 1024
    return f"{n:,.1f} PB"


# --------------------------------------------------------------------------- #
#  Low-level probing
# --------------------------------------------------------------------------- #
_video_cache = {}


def probe_video(path):
    """Return dict(w,h,n,fps) or None if unreadable. Cached by path."""
    key = str(path)
    if key in _video_cache:
        return _video_cache[key]
    res = None
    try:
        cap = cv2.VideoCapture(key)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if w > 0 and h > 0:
                res = {"w": w, "h": h, "n": n, "fps": fps}
        cap.release()
    except Exception:
        res = None
    _video_cache[key] = res
    return res


def image_size(path):
    """(w,h) using PIL header-only read; None if corrupt."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size            # (w, h)
    except Exception:
        return None


def read_frame(video_path, frame_idx):
    """Decode a single frame (BGR ndarray) from a video, or None."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Annotation parsing (Anti-UAV / CST schema)
# --------------------------------------------------------------------------- #
_MODALITY_MAP = [
    (("infrared", "thermal", "_ir", "ir_", "/ir", "lwir"), "IR"),
    (("visible", "rgb", "color", "colour", "vis"), "RGB"),
]


def infer_modality(name_lower):
    for keys, mod in _MODALITY_MAP:
        if any(k in name_lower for k in keys):
            return mod
    return "NA"


def infer_split(path_parts_lower):
    for p in path_parts_lower:
        if p in ("train", "training"):
            return "train"
        if p in ("val", "valid", "validation"):
            return "val"
        if p in ("test", "testing"):
            return "test"
    return "unknown"


def looks_like_antiuav_ann(path, max_head=4096):
    """Cheap check: does this json carry the {exist / gt_rect / res / gt} schema?"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(max_head)
    except Exception:
        return False
    return any(k in head for k in ('"gt_rect"', '"exist"', '"gt"', '"res"', '"rect"'))


def parse_antiuav_ann(path):
    """Return (exist_list, rect_list) with rect as [x,y,w,h]; adaptive to the
    common Anti-UAV key variants."""
    try:
        d = json.load(open(path, "r", encoding="utf-8", errors="ignore"))
    except Exception:
        return None, None
    if not isinstance(d, dict):
        return None, None
    rects = None
    for k in ("gt_rect", "gt", "res", "rect", "bbox", "boxes"):
        if k in d and isinstance(d[k], list):
            rects = d[k]
            break
    exist = d.get("exist")
    if rects is None:
        return exist, None
    if exist is None:
        # derive existence from rect validity
        exist = [1 if (isinstance(r, (list, tuple)) and len(r) >= 4
                       and r[2] and r[3]) else 0 for r in rects]
    return exist, rects


# --------------------------------------------------------------------------- #
#  Dataset discovery
# --------------------------------------------------------------------------- #
def discover_video_dataset(dataset_key):
    """Discover sequence/modality clips for a video-sequence dataset
    (Anti-UAV, CST). Returns a list of clip dicts."""
    root = DATASET_DIRS[dataset_key]
    clips = []
    if not root.exists():
        return clips
    for jp in root.rglob("*.json"):
        # skip split-manifest jsons (label_new/{train,val,test}.json)
        if jp.parent.name.lower() in ("label_new", "label", "labels", "meta"):
            continue
        if not looks_like_antiuav_ann(jp):
            continue
        rel = jp.relative_to(root)
        parts_lower = [p.lower() for p in rel.parts]
        stem_lower = jp.stem.lower()
        modality = infer_modality(stem_lower)
        if modality == "NA":
            modality = infer_modality("/".join(parts_lower))
        # sequence identity = parent directory relative path
        seq = str(jp.parent.relative_to(root)).replace("\\", "/")
        if seq in (".", ""):
            seq = jp.stem
        # locate media: sibling video with same stem, else frames folder
        media, media_kind = None, None
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            cand = jp.with_suffix(ext)
            if cand.exists():
                media, media_kind = cand, "video"
                break
        if media is None:
            cand_dir = jp.parent / jp.stem
            if cand_dir.is_dir():
                media, media_kind = cand_dir, "frames"
        if media is None:
            # CST layout: numbered frames live directly beside the annotation
            for f in jp.parent.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    media, media_kind = jp.parent, "frames"
                    break
        clips.append({
            "dataset": dataset_key,
            "split": infer_split(parts_lower),
            "sequence": seq,
            "modality": modality,
            "ann_path": str(jp),
            "media_path": str(media) if media else None,
            "media_kind": media_kind,
        })
    return clips


def discover_detfly():
    """Discover the Det-Fly YOLO export: classes + (split,image,label) triples."""
    root = DATASET_DIRS["det_fly"]
    info = {"classes": None, "samples": []}
    if not root.exists():
        return info
    # data.yaml -> class names
    import yaml
    yml = None
    for cand in root.rglob("*.yaml"):
        yml = cand
        break
    if yml is None:
        for cand in root.rglob("*.yml"):
            yml = cand
            break
    if yml is not None:
        try:
            y = yaml.safe_load(open(yml, "r", encoding="utf-8", errors="ignore"))
            names = y.get("names")
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            info["classes"] = names
            info["yaml_path"] = str(yml)
        except Exception:
            pass
    # images + labels
    for img in root.rglob("*"):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        parts_lower = [p.lower() for p in img.relative_to(root).parts]
        if "images" not in parts_lower:
            continue
        # matching label: swap images/ -> labels/ and ext -> .txt
        rel = img.relative_to(root)
        lbl_parts = ["labels" if p == "images" else p for p in rel.parts]
        lbl = root.joinpath(*lbl_parts).with_suffix(".txt")
        info["samples"].append({
            "split": infer_split(parts_lower),
            "image_path": str(img),
            "label_path": str(lbl) if lbl.exists() else None,
        })
    return info


# --------------------------------------------------------------------------- #
#  Master index construction
# --------------------------------------------------------------------------- #
def build_master_index(video_datasets=("anti_uav", "cst_anti_uav"),
                       do_detfly=True, verbose=True):
    """Parse every annotation once; build media_meta + bbox tables; persist."""
    media_rows = []
    bbox_rows = []
    corrupt = []

    # ---- video-sequence datasets (Anti-UAV, CST) ------------------------- #
    for dk in video_datasets:
        if not DATASET_DIRS[dk].exists():
            if verbose:
                log(f"[{dk}] directory absent -> skipped.")
            continue
        clips = discover_video_dataset(dk)
        if verbose:
            log(f"[{dk}] discovered {len(clips)} clips "
                f"(IR={sum(c['modality']=='IR' for c in clips)}, "
                f"RGB={sum(c['modality']=='RGB' for c in clips)}, "
                f"NA={sum(c['modality']=='NA' for c in clips)})")
        for i, c in enumerate(clips):
            exist, rects = parse_antiuav_ann(c["ann_path"])
            # resolution
            W = H = nframes_media = 0
            fps = 0.0
            if c["media_kind"] == "video" and c["media_path"]:
                pv = probe_video(c["media_path"])
                if pv:
                    W, H, nframes_media, fps = pv["w"], pv["h"], pv["n"], pv["fps"]
                else:
                    corrupt.append(c["media_path"])
            elif c["media_kind"] == "frames" and c["media_path"]:
                frame_files = sorted(Path(c["media_path"]).glob("*"))
                frame_files = [f for f in frame_files if f.suffix.lower() in IMG_EXTS]
                nframes_media = len(frame_files)
                if frame_files:
                    sz = image_size(frame_files[0])
                    if sz:
                        W, H = sz
            n_ann = len(rects) if rects else 0
            exist_count = int(sum(1 for e in (exist or []) if e))
            media_rows.append({
                "dataset": dk, "split": c["split"], "sequence": c["sequence"],
                "modality": c["modality"], "kind": c["media_kind"] or "none",
                "width": W, "height": H, "fps": round(fps, 2),
                "n_frames_media": nframes_media, "n_frames_ann": n_ann,
                "exist_count": exist_count,
                "media_path": c["media_path"], "ann_path": c["ann_path"],
            })
            # per-frame bbox rows
            if rects and W and H:
                for fi, r in enumerate(rects):
                    e = 1
                    if exist is not None and fi < len(exist):
                        e = int(bool(exist[fi]))
                    if not (isinstance(r, (list, tuple)) and len(r) >= 4):
                        continue
                    x, y, w, h = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                    if e == 0 or w <= 0 or h <= 0:
                        continue
                    bbox_rows.append((
                        dk, c["split"], c["sequence"], c["modality"], fi,
                        x, y, w, h, x + w / 2.0, y + h / 2.0, W, H,
                        -1, "uav"))
            if verbose and (i + 1) % 200 == 0:
                log(f"[{dk}]  indexed {i+1}/{len(clips)} clips")

    # ---- Det-Fly (YOLO) -------------------------------------------------- #
    if do_detfly and DATASET_DIRS["det_fly"].exists():
        info = discover_detfly()
        classes = info["classes"] or []
        if verbose:
            log(f"[det_fly] classes={classes}  images={len(info['samples'])}")
        for i, s in enumerate(info["samples"]):
            sz = image_size(s["image_path"])
            if sz is None:
                corrupt.append(s["image_path"])
                continue
            W, H = sz
            stem = Path(s["image_path"]).stem
            n_boxes = 0
            if s["label_path"]:
                try:
                    for line in open(s["label_path"], "r", encoding="utf-8",
                                     errors="ignore"):
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cid = int(float(parts[0]))
                        cx, cy, ww, hh = map(float, parts[1:5])
                        bw, bh = ww * W, hh * H
                        bx, by = cx * W - bw / 2.0, cy * H - bh / 2.0
                        cname = classes[cid] if 0 <= cid < len(classes) else str(cid)
                        bbox_rows.append((
                            "det_fly", s["split"], stem, "RGB", 0,
                            bx, by, bw, bh, cx * W, cy * H, W, H, cid, cname))
                        n_boxes += 1
                except Exception:
                    pass
            media_rows.append({
                "dataset": "det_fly", "split": s["split"], "sequence": stem,
                "modality": "RGB", "kind": "image", "width": W, "height": H,
                "fps": 0.0, "n_frames_media": 1, "n_frames_ann": n_boxes,
                "exist_count": 1 if n_boxes else 0,
                "media_path": s["image_path"], "ann_path": s["label_path"],
            })
            if verbose and (i + 1) % 2000 == 0:
                log(f"[det_fly]  indexed {i+1}/{len(info['samples'])} images")
        # stash class list
        (ART / "detfly_classes.json").write_text(
            json.dumps(info.get("classes"), indent=2), encoding="utf-8")

    media_df = pd.DataFrame(media_rows)
    bbox_df = pd.DataFrame(bbox_rows, columns=[
        "dataset", "split", "sequence", "modality", "frame_idx",
        "x", "y", "w", "h", "cx", "cy", "frame_w", "frame_h",
        "class_id", "class_name"])

    # derived per-box columns
    if len(bbox_df):
        bbox_df["bbox_area"] = bbox_df["w"] * bbox_df["h"]
        bbox_df["frame_area"] = bbox_df["frame_w"] * bbox_df["frame_h"]
        bbox_df["area_ratio"] = bbox_df["bbox_area"] / bbox_df["frame_area"]
        bbox_df["aspect"] = bbox_df["w"] / bbox_df["h"]
        bbox_df["cx_norm"] = bbox_df["cx"] / bbox_df["frame_w"]
        bbox_df["cy_norm"] = bbox_df["cy"] / bbox_df["frame_h"]
        bbox_df["is_micro"] = bbox_df["area_ratio"] < MICRO_THRESHOLD

    media_df.to_csv(MEDIA_CSV, index=False)
    with open(BBOX_PKL, "wb") as f:
        pickle.dump(bbox_df, f)
    # also a compact csv sample of bboxes for portability
    bbox_df.head(50000).to_csv(ART / "bboxes_sample.csv", index=False)
    with open(ART / "corrupt_media.json", "w", encoding="utf-8") as f:
        json.dump(corrupt, f, indent=2)

    if verbose:
        log(f"master index: media_rows={len(media_df):,}  bbox_rows={len(bbox_df):,}"
            f"  corrupt={len(corrupt)}")
    return media_df, bbox_df


def load_master():
    media_df = pd.read_csv(MEDIA_CSV)
    with open(BBOX_PKL, "rb") as f:
        bbox_df = pickle.load(f)
    return media_df, bbox_df


if __name__ == "__main__":
    build_master_index()
