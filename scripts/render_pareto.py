#!/usr/bin/env python
"""Compute-aware model selection (mission §33): accuracy vs latency to output on the RTX 5050, generated from JSON only.

    venv\\Scripts\\python.exe scripts\\render_pareto.py

Points = configurations (model family x temporal window) trained on corpus v2, seed-averaged (bars = seed min-max), from
experiments/aerotrack_v2/reports/v2_results.json and experiments/aerotrack_v3/reports/v3_results.json (domain-subset runs
excluded). Latency = what a deployed system experiences for frame t: per-frame end-to-end compute p99 measured by
deploy_gate.py for that architecture (same backend for every architecture: TensorRT fp16 when every architecture has it,
else PyTorch fp32) PLUS the window's structural look-ahead (symmetric g: g frames at 25 fps; causal / single frame: 0).
Outputs: docs/PARETO.md (table incl. Anti-UAV / micro AP, params, GFLOPs, device memory, frontier membership) and
docs/figures/pareto_cst_latency.png. Configurations without a latency measurement are listed but not plotted.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEP = REPO / "experiments" / "deployment" / "reports"
FPS = 25.0
# reference palette (dataviz skill, validated all-pairs for 3 slots, light surface): blue, orange, aqua
COLORS = {"n": "#2a78d6", "n_p2": "#eb6834", "s": "#1baf7a"}
MARKERS = {"ttt": "o", "symmetric": "s", "causal": "^"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def rd(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run_info(run):
    m = rd(REPO / "results/runs" / run / "run_meta.json") or {}
    family = m.get("model") or "n"                         # v2 / generalization runs are the stock-scale stride layout
    window = m.get("window") or ("ttt" if int(m.get("gap", 1)) == 0 else "symmetric")
    return family, window, int(m.get("gap", 0) or 0), m.get("train_domains"), (m.get("data") or {}).get("corpus")


def main():
    rows = []
    for p in (REPO / "experiments" / "aerotrack_v2" / "reports" / "v2_results.json", REPO / "experiments" / "aerotrack_v3" / "reports" / "v3_results.json"):
        rows += [r for r in (rd(p) or {}).get("rows", []) if r["set"] == "val"]
    cfg = defaultdict(list)
    for r in rows:
        family, window, gap, doms, corpus = run_info(r["run"])
        if corpus != "data_unified_v2" or (doms and len(doms) < 4):
            continue                                       # v1-trained references and domain-subset runs are not candidates
        cfg[(family, window, gap)].append(r)
    gates = {}
    for g in DEP.glob("gate_*.json"):
        d = rd(g)
        family = run_info(d["run"])[0]
        gates.setdefault(family, d)
    backend = "trt_fp16" if gates and all(v["backends"].get("trt_fp16", {}).get("available") for v in gates.values()) else "torch_fp32"
    exports = {}
    for e in DEP.glob("export_*.json"):
        d = rd(e)
        if d.get("precision", "fp32") == "fp32":
            exports.setdefault(run_info(d["run"])[0], d["model"])
    pts = []
    for (family, window, gap), rs in sorted(cfg.items()):
        cst = [r["metrics"]["cst_drone_AP50"] for r in rs]
        rec = {"family": family, "window": window, "gap": gap, "seeds": len(rs), "runs": sorted(r["run"] for r in rs),
               "cst": sum(cst) / len(cst), "cst_min": min(cst), "cst_max": max(cst),
               "anti_uav": sum(r["metrics"]["anti_uav_drone_AP50"] for r in rs) / len(rs),
               "cst_micro": sum(r["metrics"]["cst_drone_micro_AP50"] for r in rs) / len(rs),
               "params": (exports.get(family) or {}).get("params"), "gflops": (exports.get(family) or {}).get("gflops_640")}
        gb = (gates.get(family) or {}).get("backends", {}).get(backend, {})
        if gb.get("available"):
            look = gap * 1000.0 / FPS if window == "symmetric" else 0.0
            rec.update(compute_p99=gb["e2e"]["per_frame_total"]["p99_ms"], look_ahead=look, device_mb=gb.get("device_memory_used_mb"))
            rec["latency"] = rec["compute_p99"] + look
        pts.append(rec)
    with_lat = [p for p in pts if p.get("latency") is not None]
    for p in with_lat:
        p["frontier"] = not any(q is not p and q["latency"] <= p["latency"] and q["cst"] >= p["cst"]
                                and (q["latency"] < p["latency"] or q["cst"] > p["cst"]) for q in with_lat)
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "backend": backend, "fps_for_look_ahead": FPS, "points": pts}
    (REPO / "results/artifacts").mkdir(exist_ok=True)
    (REPO / "results/artifacts" / "pareto.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    f = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"    # noqa: E731
    L = ["# Compute-aware model selection (accuracy vs latency on the RTX 5050)", "",
         f"*Generated {out['generated']} by `scripts/render_pareto.py` from the campaign results and deployment-gate JSON "
         f"(`results/artifacts/pareto.json`). Latency = per-frame end-to-end compute p99 ({backend}) + the window's structural look-ahead "
         f"at {FPS:.0f} fps. CST drone AP50 on corpus-v2 VAL, seed mean [min, max]. Never hand-edit.*", "",
         "| model | window | seeds | CST AP50 | Anti-UAV AP50 | CST micro AP50 | params (M) | GFLOPs | compute p99 ms | look-ahead ms | latency ms | device MB | frontier |",
         "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for p in pts:
        L.append(f"| {p['family']} | {p['window']} (g={p['gap']}) | {p['seeds']} | {f(p['cst'])} [{f(p['cst_min'])}, {f(p['cst_max'])}] | "
                 f"{f(p['anti_uav'])} | {f(p['cst_micro'])} | {f(p['params'] / 1e6 if p['params'] else None, 2)} | {f(p['gflops'], 1)} | "
                 f"{f(p.get('compute_p99'), 1)} | {f(p.get('look_ahead'), 0)} | {f(p.get('latency'), 1)} | {f(p.get('device_mb'), 0)} | "
                 f"{'yes' if p.get('frontier') else ('no' if 'frontier' in p else 'not measured')} |")
    if not pts:
        L.append("| *no corpus-v2 results yet* | | | | | | | | | | | | |")
    (REPO / "docs").mkdir(exist_ok=True)
    if with_lat:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200, facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        fr = sorted((p for p in with_lat if p["frontier"]), key=lambda p: p["latency"])
        if len(fr) > 1:
            ax.plot([p["latency"] for p in fr], [p["cst"] for p in fr], color=MUTED, lw=1.5, ls="--", zorder=1, label="Pareto frontier")
        for p in with_lat:
            ax.errorbar(p["latency"], p["cst"], yerr=[[p["cst"] - p["cst_min"]], [p["cst_max"] - p["cst"]]], fmt="none",
                        ecolor=AXIS, elinewidth=1.5, capsize=0, zorder=2)
            ax.scatter(p["latency"], p["cst"], s=90, marker=MARKERS.get(p["window"], "o"), color=COLORS.get(p["family"], INK2),
                       edgecolors=SURFACE, linewidths=2, zorder=3)
            ax.annotate(f"{p['family']} · {p['window']}", (p["latency"], p["cst"]), xytext=(7, 5), textcoords="offset points",
                        fontsize=8, color=INK2)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor=SURFACE, markersize=8, label=f"model {k}")
                   for k, c in COLORS.items() if any(p["family"] == k for p in with_lat)]
        handles += [Line2D([], [], marker=m, ls="", color=MUTED, markersize=8, label=f"window {k}")
                    for k, m in MARKERS.items() if any(p["window"] == k for p in with_lat)]
        ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
        ax.set_xlabel(f"latency to output for frame t, ms (log): compute p99 ({backend}) + look-ahead at {FPS:.0f} fps", color=INK2, fontsize=9)
        ax.set_ylabel("CST drone AP50 (corpus-v2 VAL)", color=INK2, fontsize=9)
        ax.set_title("Accuracy vs real-time latency on the RTX 5050", color=INK, fontsize=11, loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
        lo, hi = min(p["latency"] for p in with_lat), max(p["latency"] for p in with_lat)
        if hi / max(lo, 1e-6) > 4:                         # a look-ahead window can sit an order of magnitude to the right
            from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
            ax.set_xscale("log")
            ticks = [t for t in (5, 10, 20, 40, 80, 160, 320, 640) if lo / 1.5 <= t <= hi * 1.5]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xlim(lo / 1.4, hi * 1.6)
        (REPO / "docs" / "figures").mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(REPO / "docs" / "figures" / "pareto_cst_latency.png", facecolor=SURFACE)
        L += ["", "![Accuracy vs latency](figures/pareto_cst_latency.png)", "",
              "*Colour = model family, marker = temporal window (secondary encoding), direct labels on every point; the table above "
              "is the data view.*"]
    (REPO / "docs" / "PARETO.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote docs/PARETO.md ({len(pts)} configurations, {len(with_lat)} with latency; backend {backend})")


if __name__ == "__main__":
    main()
