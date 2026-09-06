#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_basic_eda.py  --  STEP 3: Fundamental Computer-Vision & Dataset EDA
====================================================================
Builds the canonical master index (once) and computes:

  1. Frame & resolution counts   : frames/images per dataset, resolution
                                    distribution (W x H), aspect-ratio spread.
  2. Annotation integrity        : total boxes, missing-annotation ratio
                                    (empty/out-of-view frames), corrupt media,
                                    out-of-bounds bbox coordinate sanity checks.
  3. Modalities & classes        : RGB vs Thermal/IR breakdown; Det-Fly class
                                    histogram.

Writes a_inspection/artifacts/basic_eda.json  (+ persists the master index for
downstream scripts).
"""

import json
import numpy as np
import pandas as pd
from collections import Counter

import eda_common as C
from eda_common import log, human, DATASET_LABEL


def resolution_stats(media_df):
    out = {}
    for dk, g in media_df.groupby("dataset"):
        rescnt = Counter()
        for _, r in g.iterrows():
            if r["width"] and r["height"]:
                rescnt[f"{int(r['width'])}x{int(r['height'])}"] += 1
        # frames weighted by resolution
        frame_res = Counter()
        for _, r in g.iterrows():
            if r["width"] and r["height"]:
                frame_res[f"{int(r['width'])}x{int(r['height'])}"] += int(r["n_frames_ann"] or 0)
        # aspect ratios
        ar = []
        for _, r in g.iterrows():
            if r["width"] and r["height"]:
                ar.append(round(r["width"] / r["height"], 4))
        out[dk] = {
            "n_media": int(len(g)),
            "resolutions_by_media": dict(sorted(rescnt.items(), key=lambda kv: -kv[1])),
            "resolutions_by_frame": dict(sorted(frame_res.items(), key=lambda kv: -kv[1])),
            "distinct_resolutions": len(rescnt),
            "aspect_ratio_unique": dict(Counter(ar)),
        }
    return out


def frame_counts(media_df):
    out = {}
    for dk, g in media_df.groupby("dataset"):
        if dk == "det_fly":
            # detection dataset: one image == one "frame"; n_frames_ann holds
            # the per-image BOX count, so aggregate the two quantities separately.
            n_imgs = int(len(g))
            with_target = int((g["exist_count"] > 0).sum())
            total_boxes = int(g["n_frames_ann"].fillna(0).sum())
            empty = n_imgs - with_target
            out[dk] = {
                "n_sequences_or_images": n_imgs,
                "total_frames": n_imgs,
                "annotated_frames": with_target,
                "empty_frames": empty,
                "missing_annotation_ratio": round(empty / n_imgs, 5) if n_imgs else 0.0,
                "total_boxes": total_boxes,
                "note": "detection: 'frames'=images; empty=background images (no target)",
            }
        else:
            total_frames = int(g["n_frames_ann"].fillna(0).sum())
            annotated = int(g["exist_count"].fillna(0).sum())
            empty = total_frames - annotated
            out[dk] = {
                "n_sequences_or_images": int(len(g)),
                "total_frames": total_frames,
                "annotated_frames": annotated,
                "empty_frames": int(empty),
                "missing_annotation_ratio": round(empty / total_frames, 5) if total_frames else 0.0,
                "note": "video: empty=out-of-view/occluded frames (exist=0)",
            }
    return out


def integrity_stats(bbox_df, media_df, corrupt):
    out = {}
    for dk, g in bbox_df.groupby("dataset"):
        n = len(g)
        oob_x = int(((g["x"] < 0) | (g["y"] < 0)).sum())
        oob_far = int(((g["x"] + g["w"] > g["frame_w"] + 1) |
                       (g["y"] + g["h"] > g["frame_h"] + 1)).sum())
        degenerate = int(((g["w"] <= 1) | (g["h"] <= 1)).sum())
        out[dk] = {
            "total_bboxes": int(n),
            "oob_negative_origin": oob_x,
            "oob_exceeds_frame": oob_far,
            "oob_total": oob_x + oob_far,
            "oob_ratio": round((oob_x + oob_far) / n, 5) if n else 0.0,
            "degenerate_boxes_<=1px": degenerate,
        }
    # corrupt media per dataset
    cc = Counter()
    for p in corrupt:
        for dk in C.DATASET_DIRS:
            if f"{dk}" in p.replace("\\", "/").replace("data/", ""):
                cc[dk] += 1
                break
    for dk in out:
        out[dk]["corrupt_media"] = int(cc.get(dk, 0))
    return out


def modality_class_stats(media_df, bbox_df):
    out = {}
    for dk, g in media_df.groupby("dataset"):
        mod_media = Counter(g["modality"])
        # frame-weighted modality
        mod_frames = Counter()
        for _, r in g.iterrows():
            mod_frames[r["modality"]] += int(r["n_frames_ann"] or 0)
        out[dk] = {
            "modality_by_media": dict(mod_media),
            "modality_by_frame": dict(mod_frames),
        }
    # Det-Fly class histogram
    try:
        classes = json.load(open(C.ART / "detfly_classes.json"))
    except Exception:
        classes = None
    df = bbox_df[bbox_df["dataset"] == "det_fly"]
    if len(df):
        ch = Counter(df["class_name"])
        out.setdefault("det_fly", {})["class_histogram"] = dict(
            sorted(ch.items(), key=lambda kv: -kv[1]))
        out["det_fly"]["class_names"] = classes
        # per-split class breakdown
        split_cls = {}
        for sp, sg in df.groupby("split"):
            split_cls[sp] = dict(Counter(sg["class_name"]))
        out["det_fly"]["class_by_split"] = split_cls
    return out


def main():
    log("=" * 78)
    log("STEP 3 : Fundamental CV & dataset EDA")
    log("=" * 78)

    media_df, bbox_df = C.build_master_index(verbose=True)
    try:
        corrupt = json.load(open(C.ART / "corrupt_media.json"))
    except Exception:
        corrupt = []

    fc = frame_counts(media_df)
    report = {
        "generated": pd.Timestamp.now().isoformat(),
        "datasets_present": sorted(media_df["dataset"].unique().tolist()),
        "frame_counts": fc,
        "resolution": resolution_stats(media_df),
        "integrity": integrity_stats(bbox_df, media_df, corrupt),
        "modalities_classes": modality_class_stats(media_df, bbox_df),
        "grand_totals": {
            "total_media": int(len(media_df)),
            "total_bboxes": int(len(bbox_df)),
            "total_frames": int(sum(v["total_frames"] for v in fc.values())),
            "total_video_frames": int(
                media_df[media_df.dataset != "det_fly"]["n_frames_ann"].fillna(0).sum()),
        },
    }

    (C.ART / "basic_eda.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- console summary ------------------------------------------------- #
    log("")
    log("---- FRAME / IMAGE COUNTS ----")
    for dk, v in report["frame_counts"].items():
        log(f"  {DATASET_LABEL.get(dk,dk):14s}: seqs/imgs={v['n_sequences_or_images']:,}  "
            f"frames={v['total_frames']:,}  annotated={v['annotated_frames']:,}  "
            f"empty={v['empty_frames']:,}  missing_ratio={v['missing_annotation_ratio']}")
    log("---- RESOLUTIONS (top by media) ----")
    for dk, v in report["resolution"].items():
        top = list(v["resolutions_by_media"].items())[:4]
        log(f"  {DATASET_LABEL.get(dk,dk):14s}: distinct={v['distinct_resolutions']}  top={top}")
    log("---- INTEGRITY ----")
    for dk, v in report["integrity"].items():
        log(f"  {DATASET_LABEL.get(dk,dk):14s}: boxes={v['total_bboxes']:,}  "
            f"oob={v['oob_total']:,} ({v['oob_ratio']})  "
            f"degenerate={v['degenerate_boxes_<=1px']:,}  corrupt={v['corrupt_media']}")
    log("---- MODALITIES / CLASSES ----")
    for dk, v in report["modalities_classes"].items():
        log(f"  {DATASET_LABEL.get(dk,dk):14s}: {v.get('modality_by_media', v)}")
        if "class_histogram" in v:
            log(f"       classes={v['class_histogram']}")
    log("")
    log("Wrote a_inspection/artifacts/basic_eda.json")
    log("STEP 3 complete.")


if __name__ == "__main__":
    main()
