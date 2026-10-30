#!/usr/bin/env python
"""Assemble the paper-benchmark artifacts from a completed benchmark run directory.

    python experiments/anti_uav/scripts/make_paper_benchmark_report.py \
        --bench-dir experiments/anti_uav/runs/paper_benchmarks_20260922T193853

Reads ONLY machine-generated inputs:
  <bench>/<run>__<split>_IR/{metrics.json,eval_meta.json}   this task's LIVE executions   (column C)
  runs/<run>/eval_<split>_IR_last/metrics.json              the earlier reproduction evals (column B)
  reports/paper_reported_results.json                       the paper's transcribed numbers (column A)
  reports/benchmark_execution_plan.json                     the inventory + classifications

Writes (all deterministic, nothing hand-typed):
  <bench>/benchmark_results.json .md, mSA_by_attribute.csv, precision_success.json,
  <bench>/per_sequence_metrics.json, benchmark_run_metadata.json
  reports/PAPER_BENCHMARK_RESULTS.md
  reports/figures/*.png

Verdict thresholds are the project's PRE-REGISTERED ones (scripts/aggregate_results.py):
  |delta| <= 1.0 pt "reproduced"; <= 3.0 pt "approximately reproduced"; else "not reproduced".
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from anti_uav_repro.paths import REPO_ROOT, REPORTS_DIR, RUNS_DIR

THRESH_REPRODUCED, THRESH_APPROX = 1.0, 3.0
ATTRS = ["OV", "OC", "FM", "SV", "LI", "TC-EASY", "TC-MID", "TC-HARD", "TC", "LR"]
PAPER_COLS = ["OV", "OC", "FM", "SV", "LI", "TC-EASY", "TC-MID", "TC-HARD", "TC", "LR", "ALL_IR", "ALL_VIS"]
# our run_name -> the paper's row label in Table V / Table VII
RUN_TO_PAPER = {"normal_ir": "normal", "dfsc_ir": "DFSC"}
SPLITS = ["val", "test"]
RUNS = ["normal_ir", "dfsc_ir"]

# dataviz: categorical slots 1 and 2 of the validated reference palette (light mode).
C_PAPER, C_OURS, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
C_TEXT, C_MUTED, C_GRID, C_SURFACE = "#0b0b0b", "#52514e", "#dcdcd8", "#fcfcfb"


def verdict(delta: float | None) -> str:
    if delta is None:
        return "not reproducible from available information"
    d = abs(delta)
    if d <= THRESH_REPRODUCED:
        return "reproduced"
    if d <= THRESH_APPROX:
        return "approximately reproduced"
    return "not reproduced"


def load(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------
# gather
# ------------------------------------------------------------------------------------------------
def gather(bench: Path) -> dict:
    paper = load(REPORTS_DIR / "paper_reported_results.json")
    plan_path = REPORTS_DIR / "benchmark_execution_plan.json"
    plan = load(plan_path) if plan_path.exists() else None

    live: dict = {}
    for run in RUNS:
        for split in SPLITS:
            d = bench / f"{run}__{split}_IR"
            mp, ep = d / "metrics.json", d / "eval_meta.json"
            if not mp.exists():
                live[(run, split)] = None
                continue
            m, e = load(mp), load(ep)
            live[(run, split)] = {"dir": d, "metrics": m, "meta": e}

    hist: dict = {}
    for run in RUNS:
        for split in SPLITS:
            p = RUNS_DIR / run / f"eval_{split}_IR_last" / "metrics.json"
            hist[(run, split)] = load(p)["aggregate"] if p.exists() else None

    return {"paper": paper, "plan": plan, "live": live, "hist": hist}


def paper_val(paper: dict, row: str, split: str, col: str) -> float | None:
    try:
        return paper["table_V_protocol_II"][split][row][PAPER_COLS.index(col)]
    except Exception:  # noqa: BLE001
        return None


def paper_ps(paper: dict, row: str) -> tuple[float | None, float | None]:
    try:
        v = paper["table_VII_precision_success"][row]
        return v[0], v[1]           # IR_precision, IR_success
    except Exception:  # noqa: BLE001
        return None, None


def ours_attr(agg: dict, attr: str) -> float | None:
    if attr == "ALL":
        return agg["mSA_percent"]
    d = agg["by_attribute"].get(attr)
    return None if d is None else d["mSA_percent"]


def ours_nseq(agg: dict, attr: str) -> int:
    if attr == "ALL":
        return agg["n_sequences"]
    d = agg["by_attribute"].get(attr)
    return 0 if d is None else d["n_sequences"]


def sub(a, b):
    return None if (a is None or b is None) else round(a - b, 2)


# ------------------------------------------------------------------------------------------------
# build the result structure
# ------------------------------------------------------------------------------------------------
def build_results(g: dict) -> dict:
    paper, live, hist = g["paper"], g["live"], g["hist"]
    out: dict = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "paper": paper["paper"],
        "protocol": "Protocol II (Anti-UAV train split used for fine-tuning), infrared modality",
        "metric_definitions": {
            "mSA": "paper Eq. (1), averaged over frames of a sequence then over sequences; scored from "
                   "the official start frame onward (spec SS6 primary convention)",
            "precision": "OTB: fraction of exist==1 frames with centre-location error <= 20 px",
            "success": "OTB: AUC of the success-rate vs IoU-threshold curve (thresholds 0, .05, ..., 1), "
                       "over exist==1 frames",
        },
        "verdict_thresholds": {"reproduced_pt": THRESH_REPRODUCED, "approximately_reproduced_pt": THRESH_APPROX},
        "column_semantics": {
            "A_paper": "the value the paper reports (Table V / Table VII)",
            "B_prior_reproduction": "what this implementation measured in the earlier reproduction evaluations",
            "C_live_benchmark": "what the current trained checkpoint measures in THIS task's live execution",
        },
        "runs": {},
        "table_V_mSA": {},
        "table_VII_precision_success": {},
    }

    for run in RUNS:
        prow = RUN_TO_PAPER[run]
        for split in SPLITS:
            key = f"{run}/{split}"
            L = live[(run, split)]
            if L is None:
                out["runs"][key] = {"executed": False, "reason": "evaluation did not produce metrics.json"}
                continue
            agg, meta = L["metrics"]["aggregate"], L["meta"]
            out["runs"][key] = {
                "executed": True,
                "our_run": run, "paper_row": prow, "split": split, "modality": "IR",
                "checkpoint": meta["checkpoint"], "checkpoint_sha256": meta["checkpoint_sha256"],
                "checkpoint_epoch": meta["checkpoint_epoch"], "seed": meta["seed"],
                "manifest_sha256": meta["manifest"].get("sha256") or meta["manifest"].get("sha256_live"),
                "manifest_verified": meta["manifest"].get("verified"),
                "git_commit": meta["git"]["commit"], "git_dirty": meta["git"]["dirty"],
                "command": meta["command"], "device": meta["device"], "amp": meta["amp"],
                "env": meta["env"],
                "timestamp_start": meta["timestamp_start"], "timestamp_end": meta.get("timestamp_end"),
                "tracking_seconds": meta.get("tracking_seconds"),
                "n_sequences": agg["n_sequences"], "n_frames": agg["frames_total"],
                "n_frames_exist": agg["frames_exist"],
                "n_predictions": agg["frames_total"],
                "n_fallback_frames": agg["n_fallback_frames"],
                "tracker_fps_mean": agg["tracker_fps_mean"],
                "prestart_frames_excluded": agg["prestart_frames_excluded"],
                "scoring": agg["scoring"],
                "out_dir": str(L["dir"]),
                "mSA_percent": agg["mSA_percent"],
                "mSA_all_frames_percent": agg["mSA_all_frames_percent"],
                "precision_20px_percent": 100 * agg["precision_20px"],
                "success_auc_percent": 100 * agg["success_auc"],
            }

            # --- Table V: attribute-wise
            rows = {}
            for attr in ATTRS + ["ALL"]:
                pcol = "ALL_IR" if attr == "ALL" else attr
                pv = paper_val(paper, prow, split, pcol)
                ov = ours_attr(agg, attr)
                bv = ours_attr(hist[(run, split)], attr) if hist[(run, split)] else None
                rows[attr] = {
                    "paper": pv,
                    "ours_live": None if ov is None else round(ov, 2),
                    "ours_prior": None if bv is None else round(bv, 2),
                    # unrounded, for the live-vs-prior determinism check (differences are ~1e-6)
                    "ours_live_exact": ov,
                    "ours_prior_exact": bv,
                    "delta_live_minus_paper": sub(ov, pv),
                    "delta_live_minus_prior": sub(ov, bv),
                    "n_sequences": ours_nseq(agg, attr),
                    "verdict": verdict(sub(ov, pv)),
                }
            out["table_V_mSA"][key] = rows

            # --- Table VII: precision / success
            pp, psu = paper_ps(paper, prow)
            op, osu = 100 * agg["precision_20px"], 100 * agg["success_auc"]
            bp = 100 * hist[(run, split)]["precision_20px"] if hist[(run, split)] else None
            bs = 100 * hist[(run, split)]["success_auc"] if hist[(run, split)] else None
            out["table_VII_precision_success"][key] = {
                "_paper_split_note": paper["table_VII_precision_success"]["_split"],
                "precision": {"paper": pp, "ours_live": round(op, 2),
                              "ours_prior": None if bp is None else round(bp, 2),
                              "delta_live_minus_paper": sub(op, pp), "verdict": verdict(sub(op, pp))},
                "success": {"paper": psu, "ours_live": round(osu, 2),
                            "ours_prior": None if bs is None else round(bs, 2),
                            "delta_live_minus_paper": sub(osu, psu), "verdict": verdict(sub(osu, psu))},
            }

    # --- the DFSC effect (difference of two single-seed runs)
    eff = {}
    for split in SPLITS:
        n, d = live[("normal_ir", split)], live[("dfsc_ir", split)]
        if n and d:
            on = n["metrics"]["aggregate"]["mSA_percent"]
            od = d["metrics"]["aggregate"]["mSA_percent"]
            pn = paper_val(paper, "normal", split, "ALL_IR")
            pd_ = paper_val(paper, "DFSC", split, "ALL_IR")
            eff[split] = {
                "ours_live_DFSC_minus_normal": round(od - on, 2),
                "paper_DFSC_minus_normal": None if (pn is None or pd_ is None) else round(pd_ - pn, 2),
                "note": "difference of two single-seed runs; no variance estimate was measured in this task",
            }
    out["dfsc_effect_mSA_pt"] = eff
    return out


# ------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------
def make_figures(res: dict, fig_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    def style(ax):
        ax.set_facecolor(C_SURFACE)
        ax.figure.set_facecolor(C_SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    def barlabels(ax, bars, fmt="%.1f"):
        for b in bars:
            h = b.get_height()
            if np.isnan(h):
                continue
            ax.annotate(fmt % h, (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom",
                        fontsize=7.5, color=C_TEXT, xytext=(0, 2), textcoords="offset points")

    keys = [k for k in res["table_V_mSA"]]

    # --- Fig 1: overall mSA, paper vs ours, per run/split
    labels, pv, ov = [], [], []
    for k in keys:
        labels.append(k.replace("_ir", "").replace("/", "\n"))
        r = res["table_V_mSA"][k]["ALL"]
        pv.append(r["paper"]); ov.append(r["ours_live"])
    if labels:
        x = np.arange(len(labels)); w = 0.38
        fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
        b1 = ax.bar(x - w/2, pv, w, label="Paper (Table V)", color=C_PAPER)
        b2 = ax.bar(x + w/2, ov, w, label="Ours (live)", color=C_OURS)
        barlabels(ax, b1); barlabels(ax, b2)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("mSA (%)", color=C_MUTED, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title("Overall mSA — Anti-UAV Protocol II, infrared", color=C_TEXT, fontsize=11, pad=12)
        ax.legend(frameon=False, fontsize=9, labelcolor=C_MUTED)
        style(ax); fig.tight_layout()
        p = fig_dir / "fig1_mSA_overall_paper_vs_ours.png"; fig.savefig(p); plt.close(fig)
        made.append(p.name)

    # --- Fig 2: attribute-wise mSA, one panel per run/split
    if keys:
        fig, axes = plt.subplots(len(keys), 1, figsize=(9.5, 3.0 * len(keys)), dpi=200, squeeze=False)
        for ax, k in zip(axes[:, 0], keys):
            rows = res["table_V_mSA"][k]
            attrs = ATTRS + ["ALL"]
            p = [rows[a]["paper"] if rows[a]["paper"] is not None else np.nan for a in attrs]
            o = [rows[a]["ours_live"] if rows[a]["ours_live"] is not None else np.nan for a in attrs]
            x = np.arange(len(attrs)); w = 0.38
            b1 = ax.bar(x - w/2, p, w, label="Paper", color=C_PAPER)
            b2 = ax.bar(x + w/2, o, w, label="Ours (live)", color=C_OURS)
            barlabels(ax, b1); barlabels(ax, b2)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{a}\n(n={rows[a]['n_sequences']})" for a in attrs], fontsize=7.5)
            ax.set_ylabel("mSA (%)", color=C_MUTED, fontsize=9)
            ax.set_ylim(0, 105)
            ax.set_title(f"{k}  —  mSA by attribute", color=C_TEXT, fontsize=10, pad=8)
            ax.legend(frameon=False, fontsize=8, labelcolor=C_MUTED, ncol=2)
            style(ax)
        fig.tight_layout()
        p = fig_dir / "fig2_mSA_by_attribute.png"; fig.savefig(p); plt.close(fig)
        made.append(p.name)

    # --- Fig 3: delta (ours - paper) per attribute, diverging around 0
    if keys:
        fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=200)
        attrs = ATTRS + ["ALL"]
        x = np.arange(len(attrs)); n = len(keys); w = 0.8 / max(n, 1)
        cols = [C_PAPER, C_OURS, C_THIRD, "#4a3aa7"]
        for i, k in enumerate(keys):
            rows = res["table_V_mSA"][k]
            d = [rows[a]["delta_live_minus_paper"] if rows[a]["delta_live_minus_paper"] is not None else np.nan
                 for a in attrs]
            ax.bar(x + (i - (n - 1) / 2) * w, d, w * 0.92, label=k, color=cols[i % len(cols)])
        ax.axhline(0, color=C_TEXT, linewidth=1)
        for thr, ls in ((THRESH_REPRODUCED, ":"), (THRESH_APPROX, "--")):
            for s in (thr, -thr):
                ax.axhline(s, color=C_MUTED, linewidth=0.8, linestyle=ls)
        ax.set_xticks(x); ax.set_xticklabels(attrs, fontsize=8)
        ax.set_ylabel("ours − paper (mSA percentage points)", color=C_MUTED, fontsize=9)
        ax.set_title("Deviation from the paper by attribute  (dotted ±1 pt = 'reproduced', dashed ±3 pt)",
                     color=C_TEXT, fontsize=10, pad=12)
        ax.legend(frameon=False, fontsize=8, labelcolor=C_MUTED, ncol=2)
        style(ax); fig.tight_layout()
        p = fig_dir / "fig3_mSA_delta_by_attribute.png"; fig.savefig(p); plt.close(fig)
        made.append(p.name)

    # --- Fig 4: precision & success
    pk = list(res["table_VII_precision_success"])
    if pk:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=200)
        for ax, metric, title in ((axes[0], "precision", "Precision @20 px (%)"),
                                  (axes[1], "success", "Success AUC (%)")):
            labs = [k.replace("_ir", "").replace("/", "\n") for k in pk]
            p = [res["table_VII_precision_success"][k][metric]["paper"] or np.nan for k in pk]
            o = [res["table_VII_precision_success"][k][metric]["ours_live"] for k in pk]
            x = np.arange(len(pk)); w = 0.38
            b1 = ax.bar(x - w/2, p, w, label="Paper (Table VII)", color=C_PAPER)
            b2 = ax.bar(x + w/2, o, w, label="Ours (live)", color=C_OURS)
            barlabels(ax, b1); barlabels(ax, b2)
            ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
            ax.set_ylim(0, 110); ax.set_title(title, color=C_TEXT, fontsize=10, pad=10)
            ax.legend(frameon=False, fontsize=8, labelcolor=C_MUTED)
            style(ax)
        fig.suptitle("OTB precision / success — infrared", color=C_TEXT, fontsize=11)
        fig.tight_layout()
        p = fig_dir / "fig4_precision_success.png"; fig.savefig(p); plt.close(fig)
        made.append(p.name)

    return made


def make_per_sequence_figure(bench: Path, fig_dir: Path) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    series = []
    for run in RUNS:
        for split in SPLITS:
            p = bench / f"{run}__{split}_IR" / "metrics.json"
            if p.exists():
                sa = [100 * d["SA"] for d in load(p)["per_sequence"].values()]
                series.append((f"{run}/{split}", sa))
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=200)
    bp = ax.boxplot([s for _, s in series], tick_labels=[n for n, _ in series],
                    patch_artist=True, widths=0.55, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(C_PAPER); patch.set_alpha(0.35); patch.set_edgecolor(C_PAPER)
    for k in ("whiskers", "caps", "medians"):
        for e in bp[k]:
            e.set_color(C_PAPER)
    rng = np.random.default_rng(0)
    for i, (_, sa) in enumerate(series, 1):
        ax.scatter(rng.normal(i, 0.055, len(sa)), sa, s=13, color=C_OURS, alpha=0.75,
                   edgecolors=C_SURFACE, linewidths=0.6, zorder=3)
    ax.set_ylabel("per-sequence SA (%)", color=C_MUTED, fontsize=9)
    ax.set_ylim(-3, 103)
    ax.set_title("Per-sequence state accuracy — each dot is one sequence", color=C_TEXT, fontsize=11, pad=12)
    ax.set_facecolor(C_SURFACE); fig.set_facecolor(C_SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    p = fig_dir / "fig5_per_sequence_SA_distribution.png"
    fig.savefig(p); plt.close(fig)
    return p.name


# ------------------------------------------------------------------------------------------------
# markdown
# ------------------------------------------------------------------------------------------------
def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def signed(v, nd=2):
    return "—" if v is None else f"{v:+.{nd}f}"


def results_markdown(res: dict) -> str:
    L: list[str] = []
    a = L.append
    a("# Anti-UAV paper benchmark — measured results")
    a("")
    a(f"Generated from `benchmark_results.json` on {res['generated']}. Do not edit by hand.")
    a("")
    a(f"- Paper: {res['paper']}")
    a(f"- Protocol: {res['protocol']}")
    a("")
    a("## Overall")
    a("")
    a("| Run | Paper row | Split | Metric | Paper | Ours (live) | Delta | Verdict |")
    a("|---|---|---|---|---:|---:|---:|---|")
    for k, rows in res["table_V_mSA"].items():
        run, split = k.split("/")
        r = rows["ALL"]
        a(f"| `{run}` | {RUN_TO_PAPER[run]} | {split} | mSA | {fmt(r['paper'])} | {fmt(r['ours_live'])} | "
          f"{signed(r['delta_live_minus_paper'])} | {r['verdict']} |")
    for k, d in res["table_VII_precision_success"].items():
        run, split = k.split("/")
        for m in ("precision", "success"):
            a(f"| `{run}` | {RUN_TO_PAPER[run]} | {split} | {m} | {fmt(d[m]['paper'])} | "
              f"{fmt(d[m]['ours_live'])} | {signed(d[m]['delta_live_minus_paper'])} | {d[m]['verdict']} |")
    a("")
    a("## mSA by attribute")
    a("")
    for k, rows in res["table_V_mSA"].items():
        a(f"### {k}")
        a("")
        a("| Attribute | n_seq | Paper | Ours (live) | Delta | Verdict |")
        a("|---|---:|---:|---:|---:|---|")
        for attr in ATTRS + ["ALL"]:
            r = rows[attr]
            a(f"| {attr} | {r['n_sequences']} | {fmt(r['paper'])} | {fmt(r['ours_live'])} | "
              f"{signed(r['delta_live_minus_paper'])} | {r['verdict']} |")
        a("")
    return "\n".join(L) + "\n"


def write_csv(res: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "paper_row", "split", "modality", "attribute", "n_sequences",
                    "paper_mSA_percent", "ours_live_mSA_percent", "ours_prior_mSA_percent",
                    "delta_live_minus_paper", "delta_live_minus_prior", "verdict"])
        for k, rows in res["table_V_mSA"].items():
            run, split = k.split("/")
            for attr in ATTRS + ["ALL"]:
                r = rows[attr]
                w.writerow([run, RUN_TO_PAPER[run], split, "IR", attr, r["n_sequences"],
                            r["paper"], r["ours_live"], r["ours_prior"],
                            r["delta_live_minus_paper"], r["delta_live_minus_prior"], r["verdict"]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()
    bench = Path(args.bench_dir)
    if not bench.is_absolute():
        bench = REPO_ROOT / bench

    g = gather(bench)
    res = build_results(g)

    missing = [k for k, v in res["runs"].items() if not v.get("executed")]
    if missing:
        print(f"WARNING: no metrics for {missing} — they will appear as not executed", file=sys.stderr)

    with open(bench / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    (bench / "benchmark_results.md").write_text(results_markdown(res), encoding="utf-8")
    write_csv(res, bench / "mSA_by_attribute.csv")
    with open(bench / "precision_success.json", "w", encoding="utf-8") as f:
        json.dump(res["table_VII_precision_success"], f, indent=2)
    with open(bench / "benchmark_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"generated": res["generated"], "runs": res["runs"],
                   "verdict_thresholds": res["verdict_thresholds"]}, f, indent=2)

    per_seq = {}
    for run in RUNS:
        for split in SPLITS:
            p = bench / f"{run}__{split}_IR" / "metrics.json"
            if p.exists():
                per_seq[f"{run}/{split}"] = load(p)["per_sequence"]
    with open(bench / "per_sequence_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_seq, f, indent=2)

    figs: list[str] = []
    if not args.skip_figures:
        fig_dir = REPORTS_DIR / "figures"
        figs = make_figures(res, fig_dir)
        f5 = make_per_sequence_figure(bench, fig_dir)
        if f5:
            figs.append(f5)

    with open(bench / "figures_written.json", "w", encoding="utf-8") as f:
        json.dump(figs, f, indent=2)

    print("wrote artifacts in", bench)
    for k, rows in res["table_V_mSA"].items():
        r = rows["ALL"]
        print(f"  {k:20s} mSA ours {fmt(r['ours_live'])}  paper {fmt(r['paper'])}  "
              f"delta {signed(r['delta_live_minus_paper'])}  -> {r['verdict']}")
    print("figures:", figs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
