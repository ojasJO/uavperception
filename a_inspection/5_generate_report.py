#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_generate_report.py  --  STEP 6: Visualisations & Comprehensive Report
=======================================================================
Renders six publication-quality (300 DPI) figures into a_inspection/plots/ and
compiles a_inspection/00_DATASET_INSPECTION_REPORT.md with comparison tables,
embedded charts, insights and model-configuration recommendations.

Palette (fixed, CVD-safe, validated via the data-viz method):
    Anti-UAV = blue #2a78d6 | CST Anti-UAV = green #008300 | Det-Fly = amber #eda100
    heatmaps = single-hue blue sequential ramp.
"""

import json
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

import eda_common as C
from eda_common import DATASET_LABEL, log

# --------------------------------------------------------------------------- #
#  Style
# --------------------------------------------------------------------------- #
DS_COLOR = {"anti_uav": "#2a78d6", "cst_anti_uav": "#008300", "det_fly": "#eda100"}
DS_ORDER = ["anti_uav", "cst_anti_uav", "det_fly"]
BLUE_RAMP = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("brand_blue", BLUE_RAMP)

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"
SURF = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
})

PLOTS = C.PLOTS


def present_datasets(df, col="dataset"):
    return [d for d in DS_ORDER if d in set(df[col].unique())]


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def fig_resolution(media_df):
    rows = []
    for dk, g in media_df.groupby("dataset"):
        tmp = g[(g["width"] > 0) & (g["height"] > 0)].copy()
        tmp["res"] = tmp["width"].astype(int).astype(str) + "x" + tmp["height"].astype(int).astype(str)
        for res, sub in tmp.groupby("res"):
            rows.append((dk, res, int(sub["n_frames_ann"].sum()), int(len(sub))))
    d = pd.DataFrame(rows, columns=["dataset", "res", "frames", "media"])
    if len(d) == 0:
        return
    d = d.sort_values(["dataset", "frames"], ascending=[True, True])
    labels = [f"{DATASET_LABEL[r.dataset]}  {r.res}" for r in d.itertuples()]
    colors = [DS_COLOR[r.dataset] for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(d) + 1.5)))
    y = np.arange(len(d))
    ax.barh(y, d["frames"], color=colors, height=0.72, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("annotated frames (log scale)")
    ax.set_title("Resolution distribution — frames per (dataset, resolution)")
    for yi, (fr, md) in enumerate(zip(d["frames"], d["media"])):
        ax.text(fr * 1.05, yi, f"{fr:,} fr / {md:,} clips", va="center",
                fontsize=8, color=INK2)
    ax.set_xlim(right=d["frames"].max() * 3)
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=DS_COLOR[d0],
               label=DATASET_LABEL[d0]) for d0 in present_datasets(media_df)]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS / "resolution_distribution.png", dpi=300)
    plt.close(fig)


def fig_scale_cdf(bbox_df):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for dk in present_datasets(bbox_df):
        a = np.sort(bbox_df.loc[bbox_df.dataset == dk, "area_ratio"].to_numpy())
        a = a[a > 0]
        if a.size == 0:
            continue
        y = np.arange(1, a.size + 1) / a.size
        ax.plot(a, y, color=DS_COLOR[dk], lw=2, label=DATASET_LABEL[dk], zorder=3)
        below = float((a < C.MICRO_THRESHOLD).mean())
        ax.text(0.98, 0.05 + 0.06 * DS_ORDER.index(dk),
                f"{DATASET_LABEL[dk]}: {below*100:.1f}% micro",
                transform=ax.transAxes, ha="right", fontsize=9, color=DS_COLOR[dk])
    ax.axvline(C.MICRO_THRESHOLD, color="#d03b3b", ls="--", lw=1.6, zorder=4)
    ax.text(C.MICRO_THRESHOLD, 1.02, "0.03% micro-target line", color="#d03b3b",
            fontsize=9, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("relative target area  $A_{bbox}/A_{frame}$  (log scale)")
    ax.set_ylabel("cumulative fraction of targets")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Target-scale CDF — how tiny are the drones?")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(PLOTS / "target_scale_cdf.png", dpi=300)
    plt.close(fig)


def fig_heatmaps():
    try:
        hm = np.load(C.ART / "heatmaps.npz")
    except Exception:
        return
    keys = [k for k in DS_ORDER if k in hm.files]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.4),
                             squeeze=False)
    for ax, dk in zip(axes[0], keys):
        H = hm[dk].T                       # transpose: rows=y, cols=x
        Hn = H / (H.max() + 1e-9)
        im = ax.imshow(Hn, origin="upper", extent=[0, 1, 1, 0],
                       cmap=BLUE_CMAP, aspect="auto", vmin=0, vmax=1)
        ax.set_title(DATASET_LABEL[dk])
        ax.set_xlabel("x_c (norm)"); ax.set_ylabel("y_c (norm)")
        ax.grid(False)
        ax.plot([0.25, 0.75, 0.75, 0.25, 0.25],
                [0.25, 0.25, 0.75, 0.75, 0.25], color="#d03b3b", lw=1.2, ls="--")
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.02)
    cb.set_label("normalised target density")
    fig.suptitle("Spatial target-density heatmaps (centre-bias analysis)",
                 fontweight="bold", y=1.02)
    fig.savefig(PLOTS / "spatial_center_bias_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_velocity():
    try:
        with open(C.ART / "velocities.pkl", "rb") as f:
            vel = pickle.load(f)
    except Exception:
        return
    if len(vel) == 0:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    hi = np.percentile(vel["speed_px"], 99)
    bins = np.linspace(0, max(hi, 5), 60)
    for dk in present_datasets(vel):
        s = vel.loc[vel.dataset == dk, "speed_px"].to_numpy()
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        ax.hist(s, bins=bins, histtype="step", lw=2, color=DS_COLOR[dk],
                density=True, label=f"{DATASET_LABEL[dk]} (med {np.median(s):.1f} px)")
        ax.axvline(np.median(s), color=DS_COLOR[dk], ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel("inter-frame centre displacement  (pixels / frame)")
    ax.set_ylabel("probability density")
    ax.set_title("Inter-frame velocity distribution (target motion magnitude)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS / "interframe_velocity_histogram.png", dpi=300)
    plt.close(fig)


def fig_speed_scale():
    try:
        with open(C.ART / "speed_scale.pkl", "rb") as f:
            ss = pickle.load(f)
    except Exception:
        return
    if len(ss) == 0:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    fast_thr = float(np.percentile(ss["speed_px"], 90))
    for dk in present_datasets(ss):
        g = ss[ss.dataset == dk]
        g = g[(g.area_ratio > 0) & (g.speed_px >= 0)]
        if len(g) > 8000:
            g = g.sample(8000, random_state=7)
        ax.scatter(g["area_ratio"], g["speed_px"] + 0.1, s=6, alpha=0.28,
                   color=DS_COLOR[dk], edgecolors="none", label=DATASET_LABEL[dk])
    ax.axvline(C.MICRO_THRESHOLD, color="#d03b3b", ls="--", lw=1.4)
    ax.axhline(fast_thr, color="#4a3aa7", ls="--", lw=1.4)
    ax.axvspan(ax.get_xlim()[0] if False else 1e-6, C.MICRO_THRESHOLD,
               ymin=0, ymax=1, color="#d03b3b", alpha=0.05)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("relative target area  $A_{bbox}/A_{frame}$  (log)")
    ax.set_ylabel("inter-frame speed  (px/frame, log)")
    ax.set_title("Speed-vs-Scale hardness — isolating the hard-negative quadrant")
    ax.text(C.MICRO_THRESHOLD * 0.9, ax.get_ylim()[1] * 0.6,
            "<- micro (<0.03%)", color="#d03b3b", ha="right", fontsize=9)
    ax.text(ax.get_xlim()[1] * 0.9, fast_thr * 1.18,
            f"fast (>{fast_thr:.0f} px/frame)", color="#4a3aa7", ha="right", fontsize=9)
    ax.annotate("HARD-NEGATIVE\nQUADRANT\n(micro & fast)",
                xy=(C.MICRO_THRESHOLD * 0.25, fast_thr * 3),
                fontsize=10, color="#d03b3b", fontweight="bold", ha="center")
    leg = ax.legend(fontsize=9, loc="lower right", markerscale=2)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    fig.tight_layout(); fig.savefig(PLOTS / "speed_vs_scale_hardness_scatter.png", dpi=300)
    plt.close(fig)


def fig_blur():
    try:
        with open(C.ART / "blur_samples.pkl", "rb") as f:
            bs = pickle.load(f)
    except Exception:
        return
    if len(bs) == 0:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for dk in present_datasets(bs):
        g = bs[(bs.dataset == dk) & np.isfinite(bs.roi_sharp) & (bs.area_ratio > 0)]
        if len(g) == 0:
            continue
        ax.scatter(g["area_ratio"], g["roi_sharp"] + 0.1, s=8, alpha=0.30,
                   color=DS_COLOR[dk], edgecolors="none", label=DATASET_LABEL[dk])
        # binned median trend
        try:
            q = pd.qcut(np.log10(g["area_ratio"]), 8, duplicates="drop")
            trend = g.groupby(q, observed=True).agg(
                x=("area_ratio", "median"), y=("roi_sharp", "median"))
            ax.plot(trend["x"], trend["y"] + 0.1, color=DS_COLOR[dk], lw=2, alpha=0.9)
        except Exception:
            pass
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("relative target area  $A_{bbox}/A_{frame}$  (log)")
    ax.set_ylabel("target ROI sharpness  (Laplacian variance, log)")
    ax.set_title("Target blur vs size — smaller targets carry less high-freq detail")
    leg = ax.legend(fontsize=9, loc="upper left", markerscale=2)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    fig.tight_layout(); fig.savefig(PLOTS / "blur_vs_target_size.png", dpi=300)
    plt.close(fig)


def fig_aspect_and_class(bbox_df):
    # bonus: aspect ratio distribution
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bins = np.linspace(0, 4, 60)
    for dk in present_datasets(bbox_df):
        a = bbox_df.loc[bbox_df.dataset == dk, "aspect"].to_numpy()
        a = a[np.isfinite(a) & (a > 0) & (a < 6)]
        if a.size == 0:
            continue
        ax.hist(a, bins=bins, histtype="step", lw=2, density=True,
                color=DS_COLOR[dk], label=f"{DATASET_LABEL[dk]} (med {np.median(a):.2f})")
    ax.axvline(1.0, color=MUTED, ls=":", lw=1.2)
    ax.set_xlabel("bounding-box aspect ratio  $W_{bbox}/H_{bbox}$")
    ax.set_ylabel("density"); ax.set_title("Bounding-box aspect-ratio distribution")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS / "aspect_ratio_distribution.png", dpi=300)
    plt.close(fig)

    # bonus: Det-Fly class distribution
    df = bbox_df[bbox_df.dataset == "det_fly"]
    if len(df):
        ch = df["class_name"].value_counts()
        fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(ch) + 2))
        ax.barh(range(len(ch)), ch.values, color=DS_COLOR["det_fly"], zorder=3)
        ax.set_yticks(range(len(ch))); ax.set_yticklabels(ch.index)
        ax.invert_yaxis()
        for i, v in enumerate(ch.values):
            ax.text(v, i, f" {v:,}", va="center", fontsize=9, color=INK2)
        ax.set_xlabel("bounding boxes")
        ax.set_title("Det-Fly — class distribution")
        fig.tight_layout(); fig.savefig(PLOTS / "detfly_class_distribution.png", dpi=300)
        plt.close(fig)


# --------------------------------------------------------------------------- #
#  Markdown report
# --------------------------------------------------------------------------- #
def load_json(name):
    try:
        return json.load(open(C.ART / name, encoding="utf-8"))
    except Exception:
        return {}


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_report(media_df, bbox_df):
    basic = load_json("basic_eda.json")
    track = load_json("tracking_metrics.json")
    xf = load_json("xfactor_analysis.json")
    extr = load_json("extraction_summary.json")
    present = present_datasets(media_df)

    def L(dk):
        return DATASET_LABEL.get(dk, dk)

    lines = []
    A = lines.append
    A("# UAV Tracking Dataset Inspection Report")
    A("### Academic-grade EDA for the AeroTrack-Net hybrid spatio-temporal tracker")
    A("")
    A(f"*Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Datasets analysed: "
      f"{', '.join(L(d) for d in present)}.*")
    A("")
    A("This report inspects three UAV datasets end-to-end — decompression, "
      "annotation parsing, fundamental CV statistics, tracking dynamics, and "
      "deep 'X-factor' analytics — to derive concrete configuration guidance for "
      "**AeroTrack-Net (YOLO-SPD backbone + temporal frame-stacking)**.")
    A("")

    # ---- 0. Executive summary ------------------------------------------- #
    A("## 0. Executive Summary")
    A("")
    gt = basic.get("grand_totals", {})
    A(f"- **{gt.get('total_media', 0):,} media items** "
      f"(video clips + images) parsed, totalling "
      f"**{gt.get('total_frames', 0):,} frames** "
      f"(of which {gt.get('total_bboxes', 0):,} carry a labelled target "
      f"bounding box).")
    sa = track.get("scale_and_aspect", {})
    for dk in present:
        if dk in sa:
            A(f"- **{L(dk)}**: median target occupies "
              f"**{sa[dk]['relative_area']['median']*100:.3f}%** of the frame; "
              f"**{sa[dk]['micro_visibility_ratio_lt_0.03pct']*100:.1f}%** of targets are "
              f"micro (<0.03%).")
    A("")
    A("> **Headline:** these are extreme tiny-object tracking datasets. The "
      "dominant failure mode for a naive detector is losing sub-0.03%-area targets, "
      "especially when they also move fast — the *hard-negative quadrant* quantified "
      "in §5.3. This directly motivates the YOLO-SPD (space-to-depth, no "
      "downsampling) backbone and temporal frame-stacking.")
    A("")

    # ---- 1. Decompression ----------------------------------------------- #
    A("## 1. Decompression & Structuring")
    A("")
    rows = []
    for dk, meta in extr.get("datasets", {}).items():
        s = meta.get("survey", {})
        note = ""
        if "truncated_member" in meta and meta.get("truncated_member"):
            note = f"recovered (stream truncated at `{meta['truncated_member']}`)"
        elif meta.get("crc_bad"):
            note = f"{meta.get('crc_ok',0)} CRC-ok / {meta.get('crc_bad',0)} bad"
        elif meta.get("skipped"):
            note = "already present"
        else:
            note = "clean extract"
        rows.append([L(dk), s.get("n_images", 0), s.get("n_videos", 0),
                     s.get("n_annotations", 0), s.get("total_human", "—"), note])
    A(md_table(["Dataset", "Images", "Videos", "Annotations", "Size on disk", "Status"], rows))
    A("")
    cst = extr.get("datasets", {}).get("cst_anti_uav", {})
    if cst:
        A(f"**CST Anti-UAV recovery note.** The delivered archive was an "
          f"*incomplete* multi-volume split zip (`.z01`–`.z07`, {len(cst.get('volumes',[]))}"
          f" volumes) whose terminal `.zip` volume — carrying the End-Of-Central-Directory "
          f"and central directory — was missing. Standard tools (unzip / 7z / Python "
          f"`zipfile`) all require that record and therefore fail. We instead performed a "
          f"**forward streaming recovery**: the volumes were chained into one logical byte "
          f"stream and walked local-file-header by local-file-header (the archive uses no "
          f"data descriptors, so every header carries an authoritative compressed size), "
          f"extracting **{cst.get('files',0):,} files** with "
          f"**{cst.get('crc_ok',0):,} CRC-verified** members before the stream truncated "
          f"inside `{cst.get('truncated_member','?')}`. Any files that would have resided in "
          f"the absent final volume are unrecoverable; CST figures below are over the "
          f"recovered subset.")
        A("")

    # ---- 2. Frame / resolution / integrity ------------------------------ #
    A("## 2. Fundamental Statistics")
    A("")
    A("### 2.1 Frame & annotation counts")
    fc = basic.get("frame_counts", {})
    rows = []
    for dk in present:
        v = fc.get(dk, {})
        rows.append([L(dk), f"{v.get('n_sequences_or_images',0):,}",
                     f"{v.get('total_frames',0):,}", f"{v.get('annotated_frames',0):,}",
                     f"{v.get('empty_frames',0):,}",
                     f"{v.get('missing_annotation_ratio',0)*100:.1f}%"])
    A(md_table(["Dataset", "Seqs/Imgs", "Frames", "Annotated", "Empty (no target)",
                "Missing-ann ratio"], rows))
    A("")
    A("### 2.2 Resolution & modality")
    res = basic.get("resolution", {}); mc = basic.get("modalities_classes", {})
    rows = []
    for dk in present:
        rv = res.get(dk, {})
        top = list(rv.get("resolutions_by_media", {}).items())[:3]
        top_s = ", ".join(f"{k} ({v})" for k, v in top)
        modv = mc.get(dk, {}).get("modality_by_media", {})
        rows.append([L(dk), rv.get("distinct_resolutions", 0), top_s,
                     ", ".join(f"{k}:{v}" for k, v in modv.items())])
    A(md_table(["Dataset", "#Distinct res.", "Top resolutions (media count)",
                "Modalities (media)"], rows))
    A("")
    A("![Resolution distribution](plots/resolution_distribution.png)")
    A("")
    A("### 2.3 Annotation integrity")
    integ = basic.get("integrity", {})
    rows = []
    for dk in present:
        v = integ.get(dk, {})
        rows.append([L(dk), f"{v.get('total_bboxes',0):,}", v.get("oob_total", 0),
                     f"{v.get('oob_ratio',0)*100:.3f}%",
                     v.get("degenerate_boxes_<=1px", 0), v.get("corrupt_media", 0)])
    A(md_table(["Dataset", "Total boxes", "Out-of-bounds", "OOB ratio",
                "Degenerate (≤1px)", "Corrupt media"], rows))
    A("")
    dfc = mc.get("det_fly", {})
    if dfc.get("class_histogram"):
        A("**Det-Fly classes:** " +
          ", ".join(f"`{k}` = {v:,}" for k, v in dfc["class_histogram"].items()) + ".")
        A("")
        A("![Det-Fly class distribution](plots/detfly_class_distribution.png)")
        A("")

    # ---- 3. Tracking metrics -------------------------------------------- #
    A("## 3. Tracking & Spatio-Temporal Dynamics")
    A("")
    A("### 3.1 Target scale & micro-drone visibility")
    rows = []
    for dk in present:
        v = sa.get(dk, {})
        if not v:
            continue
        rows.append([L(dk), f"{v['relative_area']['median']*100:.4f}%",
                     f"{v['relative_area']['percentiles'].get('p99',0)*100:.3f}%",
                     f"{v['micro_visibility_ratio_lt_0.03pct']*100:.1f}%",
                     f"{v['share_lt_1pct']*100:.1f}%",
                     f"{v['aspect_ratio']['median']:.2f}"])
    A(md_table(["Dataset", "Median area", "p99 area", "Micro <0.03%",
                "Share <1%", "Median W/H"], rows))
    A("")
    A("![Target scale CDF](plots/target_scale_cdf.png)")
    A("")
    A("![Aspect ratio distribution](plots/aspect_ratio_distribution.png)")
    A("")
    A("### 3.2 Inter-frame velocity")
    vv = track.get("velocity", {})
    rows = []
    for dk in present:
        v = vv.get(dk, {})
        if not v:
            continue
        sp = v["speed_px_per_frame"]
        rows.append([L(dk), f"{sp['median']:.2f}", f"{sp['percentiles'].get('p99',0):.1f}",
                     f"{sp['max']:.0f}",
                     f"{v.get('sudden_jumps_gt_%dpx'%int(track.get('jump_threshold_px',30)),0):,}",
                     f"{v.get('sudden_jump_ratio',0)*100:.2f}%"])
    A(md_table(["Dataset", "Median px/fr", "p99 px/fr", "Max px/fr",
                "Sudden jumps (>30px)", "Jump ratio"], rows))
    A("")
    A("![Inter-frame velocity histogram](plots/interframe_velocity_histogram.png)")
    A("")

    # ---- 4/5. X-factor -------------------------------------------------- #
    A("## 4. X-Factor Analytics")
    A("")
    A("### 4.1 Spatial centre-bias")
    cb = xf.get("centre_bias", {})
    rows = []
    for dk in present:
        v = cb.get(dk, {})
        if not v:
            continue
        rows.append([L(dk), f"{v['central_50pct_box_share']*100:.1f}%",
                     f"{v['corner_share']*100:.1f}%",
                     f"({v['mean_cx']:.2f}, {v['mean_cy']:.2f})",
                     f"({v['std_cx']:.2f}, {v['std_cy']:.2f})"])
    A(md_table(["Dataset", "Central-50% share", "Corner share",
                "Mean centre", "Centre spread (σ)"], rows))
    A("")
    A("![Spatial centre-bias heatmaps](plots/spatial_center_bias_heatmap.png)")
    A("")
    A("### 4.2 Motion-blur & target–background contrast")
    bl = xf.get("blur_contrast_lighting", {})
    rows = []
    for dk in present:
        v = bl.get(dk, {})
        if not v:
            continue
        rows.append([L(dk), f"{v['roi_sharp_median']:.1f}", f"{v['bg_sharp_median']:.1f}",
                     f"{v['rel_sharp_median']:.2f}",
                     f"{v['share_roi_softer_than_bg']*100:.0f}%",
                     f"{v['target_bg_contrast_median']:.1f}"])
    A(md_table(["Dataset", "ROI sharpness", "BG sharpness", "ROI/BG ratio",
                "ROI softer than BG", "Target–BG contrast"], rows))
    A("")
    A("![Blur vs target size](plots/blur_vs_target_size.png)")
    A("")
    A("### 4.3 Speed-vs-Scale hardness (the hard-negative quadrant)")
    hd = xf.get("speed_vs_scale_hardness", {})
    rows = []
    for dk in present:
        v = hd.get(dk, {})
        if not v:
            continue
        rows.append([L(dk), f"{v['micro_share']*100:.1f}%", f"{v['fast_share']*100:.1f}%",
                     f"{v['hard_quadrant_share_micro_and_fast']*100:.2f}%",
                     f"{v['corr_speed_area']}"])
    A(md_table(["Dataset", "Micro share", "Fast share (>p90)",
                "Hard quadrant (micro & fast)", "corr(speed, area)"], rows))
    A("")
    A("![Speed vs scale hardness](plots/speed_vs_scale_hardness_scatter.png)")
    A("")
    A("### 4.4 Autonomous feature discovery")
    auto = xf.get("autonomous_findings", {})
    occ = auto.get("occlusion_out_of_view", {})
    if occ:
        A("**Occlusion / out-of-view dynamics.**")
        rows = []
        for dk, v in occ.items():
            rows.append([L(dk), f"{v['oov_frame_ratio']*100:.1f}%", v["n_oov_events"],
                         f"{v['mean_oov_event_len']:.1f}", v["max_oov_event_len"],
                         f"{v['share_sequences_with_oov']*100:.0f}%"])
        A(md_table(["Dataset", "OOV frame ratio", "#OOV events", "Mean len",
                    "Max len (frames)", "Seqs w/ OOV"], rows))
        A("")
    mo = auto.get("aspect_morphing_and_scale_variation", {})
    if mo:
        A("**Aspect morphing & scale variation (within-sequence).**")
        rows = []
        for dk, v in mo.items():
            rows.append([L(dk), f"{v['median_aspect_std']:.3f}",
                         f"{v['median_aspect_range']:.2f}", f"{v['median_scale_cv']:.2f}",
                         f"{v['p95_scale_cv']:.2f}"])
        A(md_table(["Dataset", "Median aspect σ", "Median aspect range",
                    "Median scale CV", "p95 scale CV"], rows))
        A("")
    ra = auto.get("modality_resolution_asymmetry", {})
    if ra:
        A("**Modality resolution asymmetry.** " +
          "; ".join(f"{L(dk)}: " + ", ".join(f"{m}={r}" for m, r in d.items())
                    for dk, d in ra.items()) +
          ". IR and RGB streams are captured at *different* resolutions — a "
          "registration/rescaling concern for any RGB-T fusion.")
        A("")

    # ---- 6. Recommendations --------------------------------------------- #
    A("## 5. Recommendations for AeroTrack-Net (YOLO-SPD + Frame Stacking)")
    A("")
    recs = [
        ("Preserve resolution — use YOLO-SPD (Space-to-Depth-Conv)",
         "With medians well under 0.1% of frame area and large micro-target shares, "
         "every stride-2 downsample destroys targets. Replace strided convs / pooling "
         "with SPD-Conv blocks so sub-16px drones survive to the detection head; keep "
         "input resolution native (do not letterbox 1080p IR/RGB down to 640)."),
        ("Temporal frame-stacking sized to real motion",
         "Median inter-frame motion is only a few px but the tails reach the hundreds "
         "(fast maneuvers + sudden jumps). Stack k=3–5 consecutive frames (or add a "
         "recurrent/temporal-attention neck) so the model exploits motion cues that a "
         "single tiny, low-contrast frame lacks. Size the temporal window to the p99 "
         "displacement, not the median."),
        ("Small-anchor / anchor-free head + tiny-object augmentation",
         "Aspect ratios cluster near 1 but morph over sequences. Use an anchor-free head "
         "(or anchors tuned to the observed W/H and area percentiles), copy-paste small-"
         "target augmentation, and mosaic with care (avoid shrinking targets further)."),
        ("Hard-negative-aware sampling",
         "Explicitly oversample the micro-&-fast hard-negative quadrant (§4.3) and "
         "out-of-view re-entry frames during training; these are where trackers drift."),
        ("Modality-specific pipelines for RGB-T",
         "IR and RGB differ in resolution and in target–background contrast (thermal "
         "crossover lowers IR contrast). Train modality-aware branches or normalise "
         "per-modality; register IR↔RGB before any mid-level fusion."),
        ("Robust loss for occlusion / OOV",
         "Sequences contain out-of-view gaps; incorporate an existence/visibility head "
         "and don't penalise 'no-detection' on OOV frames — use the `exist` flag as "
         "supervision."),
    ]
    for i, (t, body) in enumerate(recs, 1):
        A(f"**R{i}. {t}.** {body}")
        A("")

    A("## 6. Reproducibility")
    A("")
    A("All figures and tables are regenerated by the pipeline in `a_inspection/`:")
    A("")
    A("```\n1_uncompress_datasets.py   # extract + CST streaming recovery\n"
      "2_basic_eda.py             # fundamental CV stats  -> basic_eda.json\n"
      "3_tracking_metrics.py      # scale/aspect/velocity -> tracking_metrics.json\n"
      "4_xfactor_analysis.py      # heatmaps/blur/hardness -> xfactor_analysis.json\n"
      "5_generate_report.py       # plots + this report\n```")
    A("")
    A("*Machine-readable outputs live in `a_inspection/artifacts/`; 300-DPI figures "
      "in `a_inspection/plots/`.*")

    (C.INSP / "00_DATASET_INSPECTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    log("=" * 78)
    log("STEP 6 : Visualisations & comprehensive report")
    log("=" * 78)
    media_df, bbox_df = C.load_master()
    log("rendering figures...")
    fig_resolution(media_df)
    fig_scale_cdf(bbox_df)
    fig_heatmaps()
    fig_velocity()
    fig_speed_scale()
    fig_blur()
    fig_aspect_and_class(bbox_df)
    log("building markdown report...")
    build_report(media_df, bbox_df)
    log("Wrote a_inspection/00_DATASET_INSPECTION_REPORT.md")
    log("STEP 6 complete.")


if __name__ == "__main__":
    main()
