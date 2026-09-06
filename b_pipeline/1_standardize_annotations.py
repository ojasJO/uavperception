#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_standardize_annotations.py  --  Phase B, Step 2: Unified Label Converter
=========================================================================
Converts the three heterogeneous annotation schemas into a single normalized
YOLO representation and lays out a unified dataset:

    b_pipeline/data_unified/
        anti_uav/{images,labels}/<seq>__<MOD>/NNNNNN.{jpg,txt}
        cst/{images,labels}/<seq>/NNNNNN.{jpg,txt}
        det_fly/{images,labels}/<split>/<stem>.{jpg,txt}
        manifest.csv                <- one row per frame; drives temporal windowing
        standardization_summary.json

YOLO label format (per line):  <class> <x_center> <y_center> <w> <h>   (all in [0,1])
Frames with no visible target (exist==0 / out-of-view / degenerate box) get an
EMPTY .txt file, exactly as required for the "no target" signal.

Class scheme (single primary target + retained hard negatives):
    0 = UAV / Drone          (Anti-UAV, CST, Det-Fly 'Drone')
    1 = Bird (hard negative) (Det-Fly 'Bird', retained per spec)

Image materialization:
  * CST / Det-Fly frames already exist on disk  -> exposed via directory
    JUNCTIONS (zero duplication); manifest points at the real files.
  * Anti-UAV frames live inside MP4s -> a bounded, representative subset of
    sequences is DECODED to real jpgs (both IR+RGB). Physical YOLO labels are
    written for EVERY Anti-UAV frame regardless; the manifest's `materialized`
    flag marks which frames also have an image on disk.  --full-antiuav decodes
    all sequences.
"""

import os
import sys
import csv
import json
import time
import argparse
import datetime
import subprocess
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "a_inspection"))
import eda_common as C                      # reuse discovery / parsing / probing

UNIFIED = HERE / "data_unified"
LOGS = HERE / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

CLASS_UAV = 0
CLASS_BIRD = 1

_log_lines = []


def log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    _log_lines.append(line)


def clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def rect_to_yolo(x, y, w, h, W, H, cls=CLASS_UAV):
    """[x,y,w,h] top-left pixel box -> clipped normalized YOLO tuple, or None."""
    if W <= 0 or H <= 0:
        return None
    x1 = max(0.0, float(x)); y1 = max(0.0, float(y))
    x2 = min(float(W), float(x) + float(w)); y2 = min(float(H), float(y) + float(h))
    bw = x2 - x1; bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    xc = (x1 + bw / 2.0) / W
    yc = (y1 + bh / 2.0) / H
    return (cls, clamp01(xc), clamp01(yc), clamp01(bw / W), clamp01(bh / H))


def write_label(path: Path, lines):
    """Write YOLO lines (possibly empty -> 0-byte file)."""
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(f"{ln[0]} {ln[1]:.6f} {ln[2]:.6f} {ln[3]:.6f} {ln[4]:.6f}\n")


def make_junction(link: Path, target: Path):
    """Create a Windows directory junction link->target (no admin needed).
    Returns True on success. Idempotent."""
    try:
        if link.exists() or link.is_symlink():
            return True
        link.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True, text=True)
        return r.returncode == 0
    except Exception as e:
        log(f"   junction failed {link} -> {target}: {e!r}")
        return False


def sanitize(seq):
    return seq.replace("\\", "/").strip("/").replace("/", "_")


# --------------------------------------------------------------------------- #
#  CST Anti-UAV
# --------------------------------------------------------------------------- #
def standardize_cst(writer, summary):
    clips = C.discover_video_dataset("cst_anti_uav")
    img_root = UNIFIED / "cst" / "images"
    lbl_root = UNIFIED / "cst" / "labels"
    img_root.mkdir(parents=True, exist_ok=True)
    lbl_root.mkdir(parents=True, exist_ok=True)
    n_frames = n_target = n_empty = n_junc = 0
    t0 = time.time()
    for ci, clip in enumerate(clips):
        frames_dir = Path(clip["media_path"]) if clip["media_path"] else None
        if frames_dir is None or not frames_dir.is_dir():
            continue
        jpgs = sorted(p for p in frames_dir.glob("*") if p.suffix.lower() in C.IMG_EXTS)
        if not jpgs:
            continue
        exist, rects = C.parse_antiuav_ann(clip["ann_path"])
        pv = None
        sz = C.image_size(jpgs[0])
        W, H = (sz if sz else (0, 0))
        split = clip["split"]
        seq_name = clip["sequence"].replace("\\", "/").rstrip("/").split("/")[-1]
        seqkey = f"cst_{split}_{seq_name}"          # full sequence folder name (unique)
        lbl_dir = lbl_root / seqkey
        lbl_dir.mkdir(parents=True, exist_ok=True)
        # junction images/<seqkey> -> source frames dir
        if make_junction(img_root / seqkey, frames_dir):
            n_junc += 1
        N = len(jpgs)
        for i, jp in enumerate(jpgs):
            e = 1
            if exist is not None and i < len(exist):
                e = int(bool(exist[i]))
            lines = []
            if e and rects is not None and i < len(rects):
                r = rects[i]
                if isinstance(r, (list, tuple)) and len(r) >= 4:
                    y = rect_to_yolo(r[0], r[1], r[2], r[3], W, H, CLASS_UAV)
                    if y:
                        lines.append(y)
            lbl_path = lbl_dir / (jp.stem + ".txt")
            write_label(lbl_path, lines)
            has_t = 1 if lines else 0
            n_target += has_t; n_empty += (0 if has_t else 1); n_frames += 1
            writer.writerow(["cst", seqkey, "IR", split, i, N,
                             str(jp), str(lbl_path), has_t, len(lines), 1])
        if (ci + 1) % 20 == 0:
            log(f"[CST] {ci+1}/{len(clips)} seqs, {n_frames:,} frames "
                f"({time.time()-t0:.0f}s)")
    summary["cst"] = {"sequences": len(clips), "frames": n_frames,
                      "with_target": n_target, "empty": n_empty,
                      "junctions": n_junc, "materialized": n_frames}
    log(f"[CST] done: {n_frames:,} labels ({n_target:,} with target, "
        f"{n_empty:,} empty), {n_junc} junctions")


# --------------------------------------------------------------------------- #
#  Det-Fly (YOLO remap)
# --------------------------------------------------------------------------- #
def standardize_detfly(writer, summary):
    info = C.discover_detfly()
    classes = info["classes"] or []
    # source class ids -> unified ids
    remap = {}
    for cid, name in enumerate(classes):
        nl = str(name).lower()
        remap[cid] = CLASS_UAV if ("drone" in nl or "uav" in nl) else CLASS_BIRD
    img_root = UNIFIED / "det_fly" / "images"
    lbl_root = UNIFIED / "det_fly" / "labels"
    img_root.mkdir(parents=True, exist_ok=True)
    lbl_root.mkdir(parents=True, exist_ok=True)
    n_img = n_box_uav = n_box_bird = n_bg = 0
    junized = set()
    for s in info["samples"]:
        img = Path(s["image_path"])
        split = s["split"]
        # junction the whole split image dir once
        if split not in junized:
            src_split_imgs = img.parent               # .../<split>/images
            make_junction(img_root / split, src_split_imgs)
            junized.add(split)
        lbl_dir = lbl_root / split
        lbl_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        if s["label_path"] and Path(s["label_path"]).exists():
            for raw in open(s["label_path"], "r", encoding="utf-8", errors="ignore"):
                p = raw.split()
                if len(p) < 5:
                    continue
                cid = int(float(p[0]))
                ncid = remap.get(cid, CLASS_UAV)
                lines.append((ncid, float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        for ln in lines:
            if ln[0] == CLASS_UAV:
                n_box_uav += 1
            else:
                n_box_bird += 1
        lbl_path = lbl_dir / (img.stem + ".txt")
        write_label(lbl_path, lines)
        has_uav = 1 if any(l[0] == CLASS_UAV for l in lines) else 0
        if not lines:
            n_bg += 1
        n_img += 1
        seqkey = f"detfly_{split}_{img.stem}"
        writer.writerow(["det_fly", seqkey, "RGB", split, 0, 1,
                         str(img), str(lbl_path), has_uav, len(lines), 1])
    summary["det_fly"] = {"images": n_img, "drone_boxes": n_box_uav,
                          "bird_boxes_hard_neg": n_box_bird, "background_images": n_bg,
                          "class_remap": {classes[c] if c < len(classes) else c: remap[c]
                                          for c in remap}}
    log(f"[Det-Fly] done: {n_img:,} images, {n_box_uav:,} drone / "
        f"{n_box_bird:,} bird(hard-neg) boxes")


# --------------------------------------------------------------------------- #
#  Anti-UAV (subset frame extraction; full labels)
# --------------------------------------------------------------------------- #
def standardize_antiuav(writer, summary, seq_limit, modalities, full):
    clips = C.discover_video_dataset("anti_uav")
    seqs = sorted(set(c["sequence"] for c in clips))
    chosen = set(seqs) if full else set(seqs[:seq_limit])
    img_root = UNIFIED / "anti_uav" / "images"
    lbl_root = UNIFIED / "anti_uav" / "labels"
    img_root.mkdir(parents=True, exist_ok=True)
    lbl_root.mkdir(parents=True, exist_ok=True)
    n_frames = n_target = n_empty = n_mat = 0
    mat_seqs = set()
    t0 = time.time()
    for ci, clip in enumerate(clips):
        if clip["modality"] not in modalities:
            continue
        exist, rects = C.parse_antiuav_ann(clip["ann_path"])
        if rects is None:
            continue
        pv = C.probe_video(clip["media_path"]) if clip["media_path"] else None
        if not pv:
            continue
        W, H, N = pv["w"], pv["h"], len(rects)
        materialize = clip["sequence"] in chosen
        seqkey = f"antiuav_{sanitize(clip['sequence'])}__{clip['modality']}"
        lbl_dir = lbl_root / seqkey
        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_dir = img_root / seqkey
        if materialize:
            img_dir.mkdir(parents=True, exist_ok=True)
            mat_seqs.add(clip["sequence"])

        # pre-compute all labels
        def label_lines(i):
            e = 1
            if exist is not None and i < len(exist):
                e = int(bool(exist[i]))
            if not e:
                return []
            r = rects[i] if i < len(rects) else None
            if isinstance(r, (list, tuple)) and len(r) >= 4:
                y = rect_to_yolo(r[0], r[1], r[2], r[3], W, H, CLASS_UAV)
                return [y] if y else []
            return []

        # idempotency: if this sequence's frames were already extracted, reuse
        # them instead of decoding the video again.
        existing = len(list(img_dir.glob("*.jpg"))) if (materialize and img_dir.exists()) else 0
        reuse = materialize and existing >= N
        cap = cv2.VideoCapture(clip["media_path"]) if (materialize and not reuse) else None
        for i in range(N):
            lines = label_lines(i)
            stem = f"{i+1:06d}"
            write_label(lbl_dir / (stem + ".txt"), lines)
            has_t = 1 if lines else 0
            n_target += has_t; n_empty += (0 if has_t else 1); n_frames += 1
            img_path = ""
            if materialize:
                ip = img_dir / (stem + ".jpg")
                if reuse:
                    if ip.exists():
                        img_path = str(ip); n_mat += 1
                elif cap is not None:
                    ok, frame = cap.read()
                    if ok:
                        cv2.imwrite(str(ip), frame)
                        img_path = str(ip); n_mat += 1
            writer.writerow(["anti_uav", seqkey, clip["modality"], clip["split"],
                             i, N, img_path, str(lbl_dir / (stem + ".txt")),
                             has_t, len(lines), 1 if img_path else 0])
        if cap is not None:
            cap.release()
        if (ci + 1) % 40 == 0:
            log(f"[Anti-UAV] {ci+1}/{len(clips)} clips, {n_frames:,} labels, "
                f"{n_mat:,} frames extracted ({time.time()-t0:.0f}s)")
    summary["anti_uav"] = {
        "clips": sum(1 for c in clips if c["modality"] in modalities),
        "total_sequences": len(seqs), "materialized_sequences": len(mat_seqs),
        "modalities": sorted(modalities), "frames_labelled": n_frames,
        "with_target": n_target, "empty": n_empty, "frames_materialized": n_mat,
        "full_extraction": full,
    }
    log(f"[Anti-UAV] done: {n_frames:,} labels ({n_target:,} target / {n_empty:,} empty), "
        f"{n_mat:,} frames extracted from {len(mat_seqs)}/{len(seqs)} sequences")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--antiuav-seq-limit", type=int, default=30,
                    help="number of Anti-UAV sequences to decode to frames")
    ap.add_argument("--antiuav-modalities", default="IR,RGB")
    ap.add_argument("--full-antiuav", action="store_true",
                    help="decode ALL Anti-UAV sequences (~100 GB, slow)")
    ap.add_argument("--only", default="all",
                    help="comma list of {cst,det_fly,anti_uav} or 'all'")
    args = ap.parse_args()

    UNIFIED.mkdir(parents=True, exist_ok=True)
    which = args.only.split(",") if args.only != "all" else ["cst", "det_fly", "anti_uav"]
    mods = set(m.strip().upper() for m in args.antiuav_modalities.split(","))

    log("=" * 78)
    log("Phase B / Step 2: Unified label standardization")
    log(f"  datasets={which}  antiuav_mods={sorted(mods)}  "
        f"seq_limit={args.antiuav_seq_limit}  full={args.full_antiuav}")
    log("=" * 78)

    summary = {"generated": datetime.datetime.now().isoformat(),
               "class_scheme": {"0": "uav_drone", "1": "bird_hard_negative"}}
    manifest_path = UNIFIED / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["dataset", "seq_key", "modality", "split", "frame_idx",
                         "n_frames", "image_path", "label_path", "has_target",
                         "n_boxes", "materialized"])
        if "cst" in which:
            standardize_cst(writer, summary)
        if "det_fly" in which:
            standardize_detfly(writer, summary)
        if "anti_uav" in which:
            standardize_antiuav(writer, summary, args.antiuav_seq_limit, mods,
                                args.full_antiuav)

    summary["manifest"] = str(manifest_path)
    (UNIFIED / "standardization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (LOGS / "1_standardize.log").write_text("\n".join(_log_lines), encoding="utf-8")
    log("Wrote manifest.csv + standardization_summary.json")
    log("Standardization complete.")


if __name__ == "__main__":
    main()
