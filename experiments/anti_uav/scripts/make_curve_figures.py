#!/usr/bin/env python
"""OTB success / precision curves and the attribute heatmap, recomputed from RAW predictions.

    python experiments/anti_uav/scripts/make_curve_figures.py \
        --bench-dir experiments/anti_uav/runs/paper_benchmarks_20260922T193853

Like verify_metrics_independently.py this reads only raw label + prediction files and recomputes
everything from scratch, so the curves are not a re-plot of a stored summary:
  * success curve   : fraction of exist==1 frames with IoU >= tau, tau in [0, 1]      (AUC = the paper's "success")
  * precision curve : fraction of exist==1 frames with centre error <= d px, d in [0, 50]  (d = 20 is the paper's "precision")
Both are averaged per sequence first, then across sequences, matching the aggregation used for mSA.

Writes reports/figures/fig6_success_curve.png, fig7_precision_curve.png, fig8_attribute_heatmap.png
and <bench>/curve_data.json (the plotted numbers, so the figures are auditable).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
LABELS = REPO / "data" / "anti_uav"
FIGS = REPO / "experiments" / "anti_uav" / "reports" / "figures"

ATTRS = ["OV", "OC", "FM", "SV", "LI", "TC-EASY", "TC-MID", "TC-HARD", "TC", "LR"]
IOU_TAU = np.linspace(0.0, 1.0, 21)
CE_TAU = np.arange(0.0, 51.0, 1.0)

# dataviz reference palette, light mode: config -> hue, split -> line style (secondary encoding)
COL = {"normal_ir": "#2a78d6", "dfsc_ir": "#eb6834"}
STYLE = {"val": "-", "test": "--"}
C_TEXT, C_MUTED, C_GRID, C_SURFACE = "#0b0b0b", "#52514e", "#dcdcd8", "#fcfcfb"


def jload(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 - ix0 <= 0 or iy1 - iy0 <= 0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def is_box(p) -> bool:
    return isinstance(p, (list, tuple)) and len(p) == 4


def sequence_curves(res, exist, gt, start):
    ious, ces = [], []
    for t in range(start, len(exist)):
        if not exist[t]:
            continue
        p = res[t]
        if is_box(p) and is_box(gt[t]):
            ious.append(iou_xywh(p, gt[t]))
            ces.append(math.hypot((p[0] + p[2] / 2) - (gt[t][0] + gt[t][2] / 2),
                                  (p[1] + p[3] / 2) - (gt[t][1] + gt[t][3] / 2)))
        else:
            ious.append(0.0)
            ces.append(float("inf"))
    if not ious:
        return None, None
    iou = np.asarray(ious)
    ce = np.asarray(ces)
    return ((iou[None, :] >= IOU_TAU[:, None]).mean(axis=1),
            (ce[None, :] <= CE_TAU[:, None]).mean(axis=1))


def collect(bench: Path) -> dict:
    out = {}
    for d in sorted(bench.glob("*__*_IR")):
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        rep = jload(mp)
        split = rep["meta"]["split"]
        run = rep["meta"]["train_run_name"]
        succ, prec, per_seq_sa, attrs = [], [], {}, {}
        lab_attr = jload(LABELS / "label_new" / f"{split}.json")
        for pf in sorted((d / "predictions").glob("*.json")):
            name = pf.stem
            pred = jload(pf)
            lab = jload(LABELS / split / name / "infrared.json")
            s, p = sequence_curves(pred["res"], lab["exist"], lab["gt_rect"], int(pred["start_frame"]))
            if s is None:
                continue
            succ.append(s)
            prec.append(p)
            per_seq_sa[name] = rep["per_sequence"][name]["SA"]
            attrs[name] = lab_attr.get(name, [])
        out[f"{run}/{split}"] = {
            "run": run, "split": split,
            "success_curve": np.mean(succ, axis=0),
            "precision_curve": np.mean(prec, axis=0),
            "per_sequence_SA": per_seq_sa, "attributes": attrs,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True)
    args = ap.parse_args()
    bench = Path(args.bench_dir)
    if not bench.is_absolute():
        bench = REPO / bench

    data = collect(bench)
    if not data:
        print("no evaluations found", file=sys.stderr)
        return 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)

    def style(ax, xlab, ylab, title):
        ax.set_facecolor(C_SURFACE)
        ax.figure.set_facecolor(C_SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        ax.grid(True, color=C_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlab, color=C_MUTED, fontsize=9)
        ax.set_ylabel(ylab, color=C_MUTED, fontsize=9)
        ax.set_title(title, color=C_TEXT, fontsize=11, pad=12)

    def curve_figure(kind: str, x, xlab, title, fname, summary):
        """Left panel: the four curves. Right panel: DFSC − normal, per split.

        The two models track each other so closely that on the left panel one line hides the other;
        the right panel is the only place the difference is readable, and it is the quantity the
        paper's Table V/VII claim is about."""
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=200,
                                      gridspec_kw={"width_ratios": [1.45, 1]})
        for k, v in sorted(data.items()):
            ax.plot(x, 100 * v[kind], STYLE[v["split"]], color=COL[v["run"]], linewidth=2,
                    label=f"{v['run']} / {v['split']}  ({summary(v)})")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 102)
        style(ax, xlab, f"{'success rate' if kind=='success_curve' else 'precision'} (%)", title)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=C_MUTED,
                  loc="lower left" if kind == "success_curve" else "lower right")

        for split, ls in STYLE.items():
            a_ = data.get(f"dfsc_ir/{split}")
            b_ = data.get(f"normal_ir/{split}")
            if a_ is None or b_ is None:
                continue
            ax2.plot(x, 100 * (a_[kind] - b_[kind]), ls, color=C_TEXT, linewidth=2,
                     label=f"{split}")
        ax2.axhline(0, color=C_MUTED, linewidth=1)
        style(ax2, xlab, "DFSC − normal (pt)", "Difference between the two models")
        ax2.legend(frameon=False, fontsize=8.5, labelcolor=C_MUTED)
        fig.tight_layout()
        fig.savefig(FIGS / fname)
        plt.close(fig)

    curve_figure("success_curve", IOU_TAU, "IoU threshold",
                 "Success plot — Anti-UAV IR, Protocol II", "fig6_success_curve.png",
                 lambda v: f"AUC {100*float(np.mean(v['success_curve'])):.2f}")
    curve_figure("precision_curve", CE_TAU, "centre-location error threshold (px)",
                 "Precision plot — Anti-UAV IR, Protocol II", "fig7_precision_curve.png",
                 lambda v: f"@20px {100*float(v['precision_curve'][20]):.2f}")

    # ---------------- attribute heatmap (ours - paper, per attribute)
    paper = jload(REPO / "experiments" / "anti_uav" / "reports" / "paper_reported_results.json")
    cols = paper["attribute_columns"]
    rows, labels = [], []
    for k in sorted(data):
        v = data[k]
        prow = {"normal_ir": "normal", "dfsc_ir": "DFSC"}[v["run"]]
        pv = paper["table_V_protocol_II"][v["split"]][prow]
        r = []
        for att in ATTRS:
            mem = [n for n, al in v["attributes"].items() if att in al]
            ours = 100 * float(np.mean([v["per_sequence_SA"][n] for n in mem])) if mem else np.nan
            r.append(ours - pv[cols.index(att)])
        rows.append(r)
        labels.append(k)
    M = np.array(rows)
    lim = float(np.nanmax(np.abs(M)))
    fig, ax = plt.subplots(figsize=(9.2, 2.6 + 0.42 * len(labels)), dpi=200)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(ATTRS)))
    ax.set_xticklabels(ATTRS, fontsize=8.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if abs(M[i, j]) > 0.62 * lim else C_TEXT)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("ours − paper (mSA pt)", color=C_MUTED, fontsize=9)
    cb.ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.set_title("Deviation from the paper, by attribute (red = we score higher)",
                 color=C_TEXT, fontsize=11, pad=12)
    ax.tick_params(colors=C_MUTED)
    fig.set_facecolor(C_SURFACE)
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_attribute_heatmap.png")
    plt.close(fig)

    with open(bench / "curve_data.json", "w", encoding="utf-8") as f:
        json.dump({
            "iou_thresholds": IOU_TAU.tolist(),
            "centre_error_thresholds_px": CE_TAU.tolist(),
            "curves": {k: {"success_curve": v["success_curve"].tolist(),
                           "precision_curve": v["precision_curve"].tolist(),
                           "success_auc_percent": 100 * float(np.mean(v["success_curve"])),
                           "precision_20px_percent": 100 * float(v["precision_curve"][20])}
                       for k, v in data.items()},
            "attribute_delta_vs_paper": {labels[i]: dict(zip(ATTRS, rows[i])) for i in range(len(labels))},
        }, f, indent=2)

    for k, v in sorted(data.items()):
        print(f"{k:22s} success AUC {100*float(np.mean(v['success_curve'])):6.2f}  "
              f"precision@20 {100*float(v['precision_curve'][20]):6.2f}")
    print("wrote fig6_success_curve.png, fig7_precision_curve.png, fig8_attribute_heatmap.png")
    print("wrote", bench / "curve_data.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
