#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_tracking_metrics.py  --  STEP 4: Tracking & Spatial-Temporal Dynamics
=======================================================================
Consumes the master index (bboxes.pkl) and computes tracking-relevant metrics
for high-speed tiny-object tracking:

  1. Relative target area  A_bbox / A_frame        (percentile spread)
  2. Micro-drone visibility ratio  (share of targets < 0.03 % of frame area)
  3. Bounding-box aspect ratio  W_bbox / H_bbox     (distribution)
  4. Inter-frame velocity / motion vectors          (px/frame between centres
     of consecutive frames of a sequence) + max speed + sudden-jump detection.

Writes artifacts/tracking_metrics.json and artifacts/velocities.pkl.
"""

import json
import numpy as np
import pandas as pd

import eda_common as C
from eda_common import log, DATASET_LABEL

JUMP_PX = 30.0          # a frame-to-frame centre jump above this = "sudden jump"
DIR_FLIP_DEG = 120.0    # angle between successive motion vectors above this = reversal


def pct(series, ps=(1, 5, 25, 50, 75, 90, 95, 99, 99.9)):
    s = series.to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {}
    return {f"p{p}": float(np.percentile(s, p)) for p in ps}


def scale_and_aspect(bbox_df):
    out = {}
    for dk, g in bbox_df.groupby("dataset"):
        micro = float((g["area_ratio"] < C.MICRO_THRESHOLD).mean())
        tiny_1pct = float((g["area_ratio"] < 0.01).mean())
        out[dk] = {
            "n_boxes": int(len(g)),
            "relative_area": {
                "mean": float(g["area_ratio"].mean()),
                "median": float(g["area_ratio"].median()),
                "percentiles": pct(g["area_ratio"]),
            },
            "micro_visibility_ratio_lt_0.03pct": round(micro, 5),
            "share_lt_1pct": round(tiny_1pct, 5),
            "aspect_ratio": {
                "mean": float(g["aspect"].mean()),
                "median": float(g["aspect"].median()),
                "percentiles": pct(g["aspect"]),
                "share_wider_than_tall": float((g["aspect"] > 1).mean()),
            },
        }
        # per-modality micro ratio (IR vs RGB)
        modo = {}
        for mod, mg in g.groupby("modality"):
            modo[mod] = {
                "n": int(len(mg)),
                "micro_ratio": round(float((mg["area_ratio"] < C.MICRO_THRESHOLD).mean()), 5),
                "median_area_ratio": float(mg["area_ratio"].median()),
            }
        out[dk]["by_modality"] = modo
    return out


def compute_velocities(bbox_df):
    """Per (dataset, sequence, modality) centre displacement between
    consecutive (frame_idx diff == 1) annotated frames."""
    rows = []
    vids = bbox_df[bbox_df["dataset"] != "det_fly"]
    for (dk, seq, mod), g in vids.groupby(["dataset", "sequence", "modality"], sort=False):
        g = g.sort_values("frame_idx")
        fidx = g["frame_idx"].to_numpy()
        cx = g["cx"].to_numpy()
        cy = g["cy"].to_numpy()
        fw = g["frame_w"].to_numpy()
        fh = g["frame_h"].to_numpy()
        if len(g) < 2:
            continue
        dfi = np.diff(fidx)
        dx = np.diff(cx)
        dy = np.diff(cy)
        speed = np.sqrt(dx * dx + dy * dy)
        diag = np.sqrt(fw[1:] ** 2 + fh[1:] ** 2)
        consecutive = dfi == 1
        # direction reversal detection
        for i in range(len(speed)):
            if not consecutive[i]:
                continue
            rows.append((dk, mod, seq, float(speed[i]),
                         float(speed[i] / diag[i]) if diag[i] else 0.0,
                         float(dx[i]), float(dy[i])))
    vel = pd.DataFrame(rows, columns=["dataset", "modality", "sequence",
                                      "speed_px", "speed_norm", "dx", "dy"])
    return vel


def velocity_stats(vel):
    out = {}
    for dk, g in vel.groupby("dataset"):
        jumps = int((g["speed_px"] > JUMP_PX).sum())
        out[dk] = {
            "n_transitions": int(len(g)),
            "speed_px_per_frame": {
                "mean": float(g["speed_px"].mean()),
                "median": float(g["speed_px"].median()),
                "max": float(g["speed_px"].max()),
                "percentiles": pct(g["speed_px"]),
            },
            "speed_norm_diag": {
                "median": float(g["speed_norm"].median()),
                "p99": float(np.percentile(g["speed_norm"], 99)),
                "max": float(g["speed_norm"].max()),
            },
            "sudden_jumps_gt_%dpx" % int(JUMP_PX): jumps,
            "sudden_jump_ratio": round(jumps / len(g), 5) if len(g) else 0.0,
        }
        modo = {}
        for mod, mg in g.groupby("modality"):
            modo[mod] = {
                "n": int(len(mg)),
                "median_px": float(mg["speed_px"].median()),
                "max_px": float(mg["speed_px"].max()),
                "p99_px": float(np.percentile(mg["speed_px"], 99)),
            }
        out[dk]["by_modality"] = modo
    return out


def main():
    log("=" * 78)
    log("STEP 4 : Tracking & spatio-temporal dynamics")
    log("=" * 78)
    media_df, bbox_df = C.load_master()
    log(f"loaded master index: {len(bbox_df):,} bboxes, {len(media_df):,} media")

    sa = scale_and_aspect(bbox_df)
    vel = compute_velocities(bbox_df)
    with open(C.ART / "velocities.pkl", "wb") as f:
        import pickle
        pickle.dump(vel, f)
    vs = velocity_stats(vel) if len(vel) else {}

    report = {
        "generated": pd.Timestamp.now().isoformat(),
        "micro_threshold_pct": C.MICRO_THRESHOLD * 100,
        "scale_and_aspect": sa,
        "velocity": vs,
        "jump_threshold_px": JUMP_PX,
    }
    (C.ART / "tracking_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    log("")
    log("---- RELATIVE AREA & MICRO-TARGET RATIO ----")
    for dk, v in sa.items():
        log(f"  {DATASET_LABEL.get(dk,dk):14s}: median_area={v['relative_area']['median']:.6f}  "
            f"micro(<0.03%)={v['micro_visibility_ratio_lt_0.03pct']*100:.2f}%  "
            f"<1%={v['share_lt_1pct']*100:.1f}%  aspect_med={v['aspect_ratio']['median']:.2f}")
    log("---- INTER-FRAME VELOCITY (px/frame) ----")
    for dk, v in vs.items():
        sp = v["speed_px_per_frame"]
        log(f"  {DATASET_LABEL.get(dk,dk):14s}: median={sp['median']:.2f}  p99={sp['percentiles'].get('p99',0):.1f}  "
            f"max={sp['max']:.1f}  jumps>{int(JUMP_PX)}px={v['sudden_jumps_gt_%dpx'%int(JUMP_PX)]:,}")
    log("")
    log("Wrote artifacts/tracking_metrics.json + velocities.pkl")
    log("STEP 4 complete.")


if __name__ == "__main__":
    main()
