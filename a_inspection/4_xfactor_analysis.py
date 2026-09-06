#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4_xfactor_analysis.py  --  STEP 5: "X-Factor" & Advanced Academic Insight
=========================================================================
Deep, non-obvious analytics that demonstrate research-grade rigor.

  1. Spatial target-density heatmap  : 2-D histogram of normalised centres
                                        (x_c, y_c) -> quantifies centre-bias.
  2. Target blur / contrast metric   : Laplacian-variance sharpness of the
                                        target ROI vs a same-size background
                                        patch (motion-blur & thermal-crossover
                                        proxies). Sampled + decoded.
  3. Speed-vs-Scale hardness scatter : inter-frame speed vs relative area to
                                        isolate the "hard-negative quadrant"
                                        (micro AND fast).
  4. Autonomous feature discovery    : within-sequence aspect morphing, scale
                                        variation, out-of-view / occlusion
                                        duration statistics, lighting variance.

Persists numeric artifacts + sampled dataframes for the report step.
"""

import json
import pickle
import numpy as np
import pandas as pd
import cv2

import eda_common as C
from eda_common import log, DATASET_LABEL, parse_antiuav_ann, read_frame

RNG = np.random.default_rng(20260804)
BLUR_SAMPLES_PER_DATASET = 1200      # decoded ROIs per dataset (sampling cap)
HEATMAP_BINS = 100


# --------------------------------------------------------------------------- #
#  1. Spatial density heatmap + centre-bias
# --------------------------------------------------------------------------- #
def spatial_heatmaps(bbox_df):
    heatmaps = {}
    centre_bias = {}
    for dk, g in bbox_df.groupby("dataset"):
        x = g["cx_norm"].to_numpy()
        y = g["cy_norm"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        x, y = x[m], y[m]
        H, xe, ye = np.histogram2d(x, y, bins=HEATMAP_BINS, range=[[0, 1], [0, 1]])
        heatmaps[dk] = H
        # centre vs edge vs corner
        central = np.mean((x > 0.25) & (x < 0.75) & (y > 0.25) & (y < 0.75))
        corner = np.mean(((x < 0.2) | (x > 0.8)) & ((y < 0.2) | (y > 0.8)))
        centre_bias[dk] = {
            "n": int(x.size),
            "central_50pct_box_share": round(float(central), 4),
            "corner_share": round(float(corner), 4),
            "mean_cx": round(float(x.mean()), 4),
            "mean_cy": round(float(y.mean()), 4),
            "std_cx": round(float(x.std()), 4),
            "std_cy": round(float(y.std()), 4),
        }
    np.savez_compressed(C.ART / "heatmaps.npz",
                        **{k: v for k, v in heatmaps.items()})
    return centre_bias


# --------------------------------------------------------------------------- #
#  2. Blur / contrast / lighting sampling (requires decoding)
# --------------------------------------------------------------------------- #
def _neighbor_patch(gray, x, y, w, h):
    """Return a same-size in-frame background patch adjacent to the ROI."""
    H, W = gray.shape[:2]
    x, y, w, h = int(x), int(y), max(1, int(w)), max(1, int(h))
    for dx, dy in [(w, 0), (-w, 0), (0, h), (0, -h), (w, h), (-w, -h)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx and nx + w <= W and 0 <= ny and ny + h <= H:
            return gray[ny:ny + h, nx:nx + w]
    return None


def _lapvar(patch):
    if patch is None or patch.size < 9:
        return np.nan
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def sample_blur_contrast(media_df, bbox_df):
    rows = []
    med = media_df[["dataset", "sequence", "modality", "media_path", "kind"]].drop_duplicates()
    for dk in bbox_df["dataset"].unique():
        g = bbox_df[bbox_df["dataset"] == dk]
        if len(g) == 0:
            continue
        take = min(BLUR_SAMPLES_PER_DATASET, len(g))
        samp = g.sample(take, random_state=int(RNG.integers(1 << 30)))
        samp = samp.merge(med, on=["dataset", "sequence", "modality"], how="left")
        # group by media to reuse decoders
        n_done = 0
        for media_path, sub in samp.groupby("media_path"):
            if not isinstance(media_path, str):
                continue
            kind = sub["kind"].iloc[0]
            cap = None
            img = None
            frame_list = None
            if kind == "image":
                img = cv2.imread(media_path)
            elif kind == "video":
                cap = cv2.VideoCapture(media_path)
            elif kind == "frames":
                from pathlib import Path as _P
                frame_list = sorted(p for p in _P(media_path).glob("*")
                                    if p.suffix.lower() in C.IMG_EXTS)
            for _, r in sub.iterrows():
                frame = None
                if kind == "image":
                    frame = img
                elif kind == "video" and cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame_idx"]))
                    ok, frame = cap.read()
                    if not ok:
                        frame = None
                elif kind == "frames" and frame_list is not None:
                    fi = int(r["frame_idx"])
                    if 0 <= fi < len(frame_list):
                        frame = cv2.imread(str(frame_list[fi]))
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                Hf, Wf = gray.shape[:2]
                x = max(0, min(int(r["x"]), Wf - 1))
                y = max(0, min(int(r["y"]), Hf - 1))
                w = max(1, min(int(r["w"]), Wf - x))
                h = max(1, min(int(r["h"]), Hf - y))
                roi = gray[y:y + h, x:x + w]
                if roi.size < 9:
                    continue
                bg = _neighbor_patch(gray, x, y, w, h)
                roi_sharp = _lapvar(roi)
                bg_sharp = _lapvar(bg)
                rows.append({
                    "dataset": dk, "modality": r["modality"],
                    "area_ratio": float(r["area_ratio"]),
                    "roi_sharp": roi_sharp, "bg_sharp": bg_sharp,
                    "rel_sharp": roi_sharp / (bg_sharp + 1e-6) if np.isfinite(bg_sharp) else np.nan,
                    "roi_mean": float(roi.mean()),
                    "bg_mean": float(bg.mean()) if bg is not None else np.nan,
                    "target_bg_contrast": (abs(float(roi.mean()) - float(bg.mean()))
                                           if bg is not None else np.nan),
                    "frame_mean": float(gray.mean()),
                })
                n_done += 1
            if cap is not None:
                cap.release()
        log(f"[blur] {DATASET_LABEL.get(dk,dk)}: decoded {n_done} ROIs")
    df = pd.DataFrame(rows)
    with open(C.ART / "blur_samples.pkl", "wb") as f:
        pickle.dump(df, f)
    # summaries
    summ = {}
    for dk, g in df.groupby("dataset"):
        summ[dk] = {
            "n": int(len(g)),
            "roi_sharp_median": float(np.nanmedian(g["roi_sharp"])),
            "bg_sharp_median": float(np.nanmedian(g["bg_sharp"])),
            "rel_sharp_median": float(np.nanmedian(g["rel_sharp"])),
            "share_roi_softer_than_bg": float(np.nanmean(g["rel_sharp"] < 1.0)),
            "target_bg_contrast_median": float(np.nanmedian(g["target_bg_contrast"])),
            "frame_mean_intensity_median": float(np.nanmedian(g["frame_mean"])),
            "frame_mean_intensity_std": float(np.nanstd(g["frame_mean"])),
        }
        modo = {}
        for mod, mg in g.groupby("modality"):
            modo[mod] = {
                "n": int(len(mg)),
                "roi_sharp_median": float(np.nanmedian(mg["roi_sharp"])),
                "target_bg_contrast_median": float(np.nanmedian(mg["target_bg_contrast"])),
            }
        summ[dk]["by_modality"] = modo
    return summ


# --------------------------------------------------------------------------- #
#  3. Speed-vs-Scale hardness
# --------------------------------------------------------------------------- #
def speed_vs_scale(bbox_df):
    """Pair each consecutive-frame speed with the (later) box's area_ratio."""
    rows = []
    vids = bbox_df[bbox_df["dataset"] != "det_fly"]
    for (dk, seq, mod), g in vids.groupby(["dataset", "sequence", "modality"], sort=False):
        g = g.sort_values("frame_idx")
        fidx = g["frame_idx"].to_numpy()
        cx = g["cx"].to_numpy(); cy = g["cy"].to_numpy()
        ar = g["area_ratio"].to_numpy()
        if len(g) < 2:
            continue
        dfi = np.diff(fidx)
        speed = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
        for i in range(len(speed)):
            if dfi[i] == 1:
                rows.append((dk, mod, float(speed[i]), float(ar[i + 1])))
    df = pd.DataFrame(rows, columns=["dataset", "modality", "speed_px", "area_ratio"])
    with open(C.ART / "speed_scale.pkl", "wb") as f:
        pickle.dump(df, f)
    hard = {}
    for dk, g in df.groupby("dataset"):
        if len(g) == 0:
            continue
        sp_thr = float(np.percentile(g["speed_px"], 90))
        micro = g["area_ratio"] < C.MICRO_THRESHOLD
        fast = g["speed_px"] > sp_thr
        hard[dk] = {
            "n_transitions": int(len(g)),
            "speed_p90_px": round(sp_thr, 3),
            "hard_quadrant_share_micro_and_fast": round(float((micro & fast).mean()), 5),
            "micro_share": round(float(micro.mean()), 5),
            "fast_share": round(float(fast.mean()), 5),
            "corr_speed_area": round(float(np.corrcoef(
                g["speed_px"], g["area_ratio"])[0, 1]) if len(g) > 2 else 0.0, 4),
        }
    return hard


# --------------------------------------------------------------------------- #
#  4. Autonomous feature discovery
# --------------------------------------------------------------------------- #
def autonomous_discovery(media_df, bbox_df):
    findings = {}

    # (a) within-sequence aspect morphing & scale variation
    morph = {}
    vids = bbox_df[bbox_df["dataset"] != "det_fly"]
    for dk, g in vids.groupby("dataset"):
        per_seq = g.groupby(["sequence", "modality"]).agg(
            aspect_std=("aspect", "std"),
            aspect_range=("aspect", lambda s: float(s.max() - s.min())),
            area_cv=("area_ratio", lambda s: float(s.std() / (s.mean() + 1e-9))),
            n=("aspect", "size"),
        ).reset_index()
        per_seq = per_seq[per_seq["n"] >= 5]
        morph[dk] = {
            "median_aspect_std": float(per_seq["aspect_std"].median()),
            "median_aspect_range": float(per_seq["aspect_range"].median()),
            "median_scale_cv": float(per_seq["area_cv"].median()),
            "p95_scale_cv": float(np.percentile(per_seq["area_cv"], 95)) if len(per_seq) else 0.0,
            "n_sequences": int(len(per_seq)),
        }
    findings["aspect_morphing_and_scale_variation"] = morph

    # (b) out-of-view / occlusion duration from exist flags (re-parse)
    occ = {}
    for dk in ("anti_uav", "cst_anti_uav"):
        # CST sequences are stored as frame-folders (kind == 'frames'), Anti-UAV
        # as videos (kind == 'video'); occlusion is read from the annotation
        # either way, so accept both.
        sub = media_df[(media_df["dataset"] == dk)
                       & (media_df["kind"].isin(["video", "frames"]))]
        if len(sub) == 0:
            continue
        tot_frames = 0; tot_oov = 0; events = []; seqs_with_oov = 0; nseq = 0
        for _, r in sub.iterrows():
            exist, _ = parse_antiuav_ann(r["ann_path"])
            if not exist:
                continue
            nseq += 1
            e = np.asarray([1 if x else 0 for x in exist])
            tot_frames += e.size
            tot_oov += int((e == 0).sum())
            # runs of zeros
            had = False
            run = 0
            for v in e:
                if v == 0:
                    run += 1
                else:
                    if run > 0:
                        events.append(run); had = True
                    run = 0
            if run > 0:
                events.append(run); had = True
            if had:
                seqs_with_oov += 1
        events = np.asarray(events) if events else np.asarray([0])
        occ[dk] = {
            "n_sequences": nseq,
            "total_frames": int(tot_frames),
            "out_of_view_frames": int(tot_oov),
            "oov_frame_ratio": round(tot_oov / tot_frames, 5) if tot_frames else 0.0,
            "n_oov_events": int((events > 0).sum()),
            "mean_oov_event_len": round(float(events[events > 0].mean()), 2) if (events > 0).any() else 0.0,
            "max_oov_event_len": int(events.max()),
            "share_sequences_with_oov": round(seqs_with_oov / nseq, 4) if nseq else 0.0,
        }
    findings["occlusion_out_of_view"] = occ

    # (c) modality resolution asymmetry (IR vs RGB) — a training-critical quirk
    resasym = {}
    for dk in ("anti_uav", "cst_anti_uav"):
        sub = media_df[media_df["dataset"] == dk]
        if len(sub) == 0:
            continue
        d = {}
        for mod, mg in sub.groupby("modality"):
            wh = mg[["width", "height"]].mode()
            if len(wh):
                d[mod] = f"{int(wh.iloc[0]['width'])}x{int(wh.iloc[0]['height'])}"
        resasym[dk] = d
    findings["modality_resolution_asymmetry"] = resasym

    return findings


def main():
    log("=" * 78)
    log("STEP 5 : X-Factor & advanced academic insight analysis")
    log("=" * 78)
    media_df, bbox_df = C.load_master()
    log(f"loaded master: {len(bbox_df):,} boxes")

    log("[1/4] spatial density heatmaps...")
    centre_bias = spatial_heatmaps(bbox_df)

    log("[2/4] speed-vs-scale hardness...")
    hard = speed_vs_scale(bbox_df)

    log("[3/4] autonomous feature discovery...")
    auto = autonomous_discovery(media_df, bbox_df)

    log("[4/4] blur / contrast / lighting sampling (decoding ROIs)...")
    blur = sample_blur_contrast(media_df, bbox_df)

    report = {
        "generated": pd.Timestamp.now().isoformat(),
        "centre_bias": centre_bias,
        "speed_vs_scale_hardness": hard,
        "blur_contrast_lighting": blur,
        "autonomous_findings": auto,
    }
    (C.ART / "xfactor_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    log("")
    log("---- CENTRE BIAS ----")
    for dk, v in centre_bias.items():
        log(f"  {DATASET_LABEL.get(dk,dk):14s}: central50%={v['central_50pct_box_share']*100:.1f}%  "
            f"corner={v['corner_share']*100:.1f}%  mean=({v['mean_cx']},{v['mean_cy']})")
    log("---- HARD-NEGATIVE QUADRANT (micro & fast) ----")
    for dk, v in hard.items():
        log(f"  {DATASET_LABEL.get(dk,dk):14s}: hard_share={v['hard_quadrant_share_micro_and_fast']*100:.2f}%  "
            f"corr(speed,area)={v['corr_speed_area']}")
    log("---- BLUR / CONTRAST ----")
    for dk, v in blur.items():
        log(f"  {DATASET_LABEL.get(dk,dk):14s}: roi_sharp_med={v['roi_sharp_median']:.1f}  "
            f"rel_sharp_med={v['rel_sharp_median']:.2f}  contrast_med={v['target_bg_contrast_median']:.1f}")
    log("---- OCCLUSION / OOV ----")
    for dk, v in auto["occlusion_out_of_view"].items():
        log(f"  {DATASET_LABEL.get(dk,dk):14s}: oov_ratio={v['oov_frame_ratio']*100:.1f}%  "
            f"max_event={v['max_oov_event_len']}  seqs_with_oov={v['share_sequences_with_oov']*100:.0f}%")
    log("")
    log("Wrote artifacts/xfactor_analysis.json")
    log("STEP 5 complete.")


if __name__ == "__main__":
    main()
