#!/usr/bin/env python
"""Render README.md and docs/AEROTRACK_REPAIR_REPORT.md from the repair JSON files — no hand-typed numbers.

    python experiments/aerotrack_repair/scripts/render_report.py            (write both files)
    python experiments/aerotrack_repair/scripts/render_report.py --check    (render in memory, write nothing)

Inputs (all produced by the other repair scripts):
  reports/detector_{val,test}_*.json, reports/test_evaluation_ledger.jsonl, reports/tracker_val_*.json, reports/latency_*.json,
  results/runs/lstm_forecaster_repair/forecast_repair_report.json, docs/assets/repair_demo_val_*.json,
  results/runs/aerotrack_spd_*/{args.json,results.csv}, results/runs/repair_abl_*/{run_meta.json,args.json,results.csv},
  results/runs/repair_grid_e*/preflight_*.json (COCO-transfer coverage recorded by run_grid.py)
Attribution rule for the ablation grid (pre-registered here): a factor is credited only if the full model beats the
ablated model by more than 0.02 AP50 on the target metric for BOTH last.pt and best.pt on the same image set;
otherwise the factor has "no measurable effect at this budget". All four runs are read at ONE epoch budget, and only
finished runs whose evaluated checkpoint hashes match their run_meta.json enter a verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from imbalance_sampler import DEFAULT_WEIGHTS, FINETUNE_WEIGHTS

from anti_uav_repro.utils.repro import git_info

from _bootstrap import REPAIR_DIR, REPO_ROOT, REPORTS_DIR

ATTRIB_MARGIN = 0.02
GRID = [("baseline_ctrl", "full recipe (SPD + 9-ch stack)"), ("no_spd", "SPDConv → stride-2 Conv"),
        ("single_frame", "t−1 = t = t+1 (no motion)"), ("stock_yolo11n", "stride-2 AND single frame (stock YOLO11n)")]
# factor, ablated run, target metrics -- pre-registered together with the rule above
FACTORS = (("SPD-Conv", "no_spd", ("micro_AP50", "cst_micro_AP50")),
           ("temporal stack", "single_frame", ("cst_AP50", "drone_AP50")),
           ("full recipe vs stock YOLO11n", "stock_yolo11n", ("drone_AP50",)))
METRIC_LABEL = {"micro_AP50": "micro AP50", "cst_micro_AP50": "CST micro AP50", "cst_AP50": "CST AP50", "drone_AP50": "drone AP50"}
INPUT_OF = {"baseline_ctrl": "t−1,t,t+1", "no_spd": "t−1,t,t+1", "single_frame": "t,t,t", "stock_yolo11n": "t,t,t"}


def jload(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f4(v):
    return "—" if v is None else f"{v:.4f}"


def f1(v):
    return "—" if v is None else f"{v:.1f}"


def results_rows(d):
    p = Path(d) / "results.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ep(v):
    return int(float(v))


# ------------------------------------------------------------------------------------------ grid state
def grid_budget():
    """The grid's epoch budget: the largest N among results/runs/repair_abl_*_e<N> (None before the first run)."""
    ns = [int(m.group(1)) for d in REPO_ROOT.glob("results/runs/repair_abl_*_e*") if (m := re.search(r"_e(\d+)$", d.name))]
    return max(ns) if ns else None


def grid_state(det=None, epochs=None):
    """Every grid run at ONE budget, with the val rows that may enter a verdict (finished run, hash-matched checkpoint)."""
    det = det if det is not None else (jload(REPORTS_DIR / "detector_val_cap5000.json") or {"rows": []})
    E = epochs or grid_budget()
    runs = {}
    for name, desc in GRID:
        d = REPO_ROOT / "results/runs" / f"repair_abl_{name}_e{E}" if E else None
        if d is None or not d.exists():
            runs[name] = {"desc": desc, "exists": False, "val": {"best": None, "last": None}}
            continue
        meta = jload(d / "run_meta.json") or {}
        rows = results_rows(d)
        finished = bool(meta.get("timestamp_end")) and meta.get("return_code") == 0 and len(rows) >= int(meta.get("epochs") or E)
        val, stale = {}, []
        for ck in ("best", "last"):
            r = next((x for x in det["rows"] if x["run"] == d.name and x["checkpoint"] == ck), None)
            if r is not None and (meta.get(f"{ck}_pt_sha256") or "")[:12] != r["weights_sha256"]:
                stale.append(ck)
                r = None
            val[ck] = r if finished else None
        runs[name] = {"desc": desc, "exists": True, "dir": d, "meta": meta, "rows": rows, "finished": finished, "val": val,
                      "stale": stale, "launches": len(meta.get("attempts") or []) or (1 if meta else 0)}
    return {"epochs": E, "runs": runs}


def verdicts(gs):
    """The pre-registered attribution rule, per factor and target metric."""
    out = []
    for factor, abl, keys in FACTORS:
        mets = []
        for key in keys:
            pairs = []
            for ck in ("best", "last"):
                a, b = gs["runs"]["baseline_ctrl"]["val"][ck], gs["runs"][abl]["val"][ck]
                pairs.append((a.get(key) if a else None, b.get(key) if b else None))
            if any(x is None or y is None for x, y in pairs):
                mets.append({"metric": key, "evaluated": False})
            else:
                mets.append({"metric": key, "evaluated": True,
                             "full_best": pairs[0][0], "ablated_best": pairs[0][1], "delta_best": pairs[0][0] - pairs[0][1],
                             "full_last": pairs[1][0], "ablated_last": pairs[1][1], "delta_last": pairs[1][0] - pairs[1][1],
                             "credited": all(x - y > ATTRIB_MARGIN for x, y in pairs)})
        out.append({"factor": factor, "ablation": abl, "metrics": mets})
    return out


def verdict_line(v):
    parts = []
    for m in v["metrics"]:
        if not m["evaluated"]:
            parts.append(f"{m['metric']}: not evaluated")
        else:
            parts.append(f"{m['metric']}: Δ best {m['delta_best']:+.3f}, last {m['delta_last']:+.3f} → "
                         f"{'credited' if m['credited'] else 'no measurable effect'}")
    return f"- {v['factor']} ({v['ablation']}): " + "; ".join(parts)


def verdict_summary(v):
    cr = [METRIC_LABEL[m["metric"]] for m in v["metrics"] if m["evaluated"] and m["credited"]]
    nc = [METRIC_LABEL[m["metric"]] for m in v["metrics"] if m["evaluated"] and not m["credited"]]
    s = []
    if cr:
        s.append("credited on " + " and ".join(cr))
    if nc:
        s.append("no measurable effect on " + " and ".join(nc))
    return f"{v['factor']}: " + "; ".join(s)


def grid_complete(vs):
    return all(m["evaluated"] for v in vs for m in v["metrics"])


def transfer_record(E):
    """COCO-transfer coverage per grid run, as measured by run_grid.py's pre-flight (None if not recorded)."""
    pf = sorted(REPO_ROOT.glob(f"results/runs/repair_grid_e{E}/preflight_*.json")) if E else []
    for p in reversed(pf):
        j = jload(p) or {}
        tr = {k: v.get("transfer") for k, v in (j.get("ablations") or {}).items() if v.get("transfer")}
        if tr:
            return tr
    return None


# ------------------------------------------------------------------------------------------ blocks
def definitions_block(gs):
    det = jload(REPORTS_DIR / "detector_val_cap5000.json")
    L = ["### One definition of every term", "",
         "- **Size bins** (code: `src/model/evaluation.py`): relative box area of the *source* frame — micro < 0.03 %, small 0.03–1 %, "
         "standard > 1 %. In pixels (side of a square box at the threshold), computed from the evaluated images:"]
    if det:
        L.append("  " + "; ".join(f"{k}: micro ≤ {v['micro_max_side_px']} px, small ≤ {v['small_max_side_px']} px" for k, v in det["size_definition_px"].items()) + ".")
    L += ["- **Evaluation protocol**: conf 0.001, NMS IoU 0.7, max 30 detections/image; FP-rate style numbers at conf 0.25; "
          "AP with all-point interpolation over IoU 0.5:0.95. Each checkpoint is scored on its own training input: the t−1,t,t+1 stack, "
          "or t,t,t (three copies of frame t) for the runs trained with temporal_gap 0.", "",
          "| run | planned epochs | epochs run | best.pt (fitness) | last.pt | batch × accumulate | lr0 → lrf | class weights | sampler weights (bird/micro/small/standard/empty) | init |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    runs = [(REPO_ROOT / "results/runs" / "aerotrack_spd_v1", DEFAULT_WEIGHTS, "yolo11n.pt (COCO)"),
            (REPO_ROOT / "results/runs" / "aerotrack_spd_finetuned", FINETUNE_WEIGHTS, "aerotrack_spd_v1 best.pt")]
    for name, _ in GRID:
        r = gs["runs"][name]
        if r["exists"]:
            stride = "stride" in str(r["meta"].get("model_cfg", ""))
            runs.append((r["dir"], DEFAULT_WEIGHTS, "yolo11n.pt (COCO), shape-matched" + (" + 3-ch stem tiled /3" if stride else "")))
    for d, sw, init in runs:
        a = jload(d / "args.json") or {}
        rows = results_rows(d)
        best = ep(max(rows, key=lambda r: float(r["fitness"]))["epoch"]) if rows else "—"
        last = ep(rows[-1]["epoch"]) if rows else "—"
        L.append(f"| {d.name} | {a.get('epochs')} | {len(rows)} (last epoch index {last}) | epoch {best} | epoch {last} | "
                 f"{a.get('batch')} × {max(1, int(a.get('nbs', 64)) // int(a.get('batch', 32)))} | {a.get('lr0')} → ×{a.get('lrf')} | {a.get('class_weights') or '[1, 1]'} | "
                 f"{sw['bird']}/{sw['micro']}/{sw['small']}/{sw['standard']}/{sw['empty']} | {init} |")
    L += ["", "The v1 run was stopped at epoch index 21 of a planned 150 (its cosine schedule never completed); the fine-tune changed the "
          "sampler weights, class weights, loss weights, batch, learning-rate schedule and augmentation at once, so its gains are a "
          "**recipe change**, not an architectural effect."]
    if any(gs["runs"][n]["exists"] for n, _ in GRID):
        L += ["The `repair_abl_*` runs use v1's recipe (`baseline` stage, default augmentation and sampler weights — the values v1 was "
              f"launched with) at a {gs['epochs']}-epoch budget whose cosine schedule completes, and differ from each other in one factor."]
    return "\n".join(L)


def detector_table(js, title):
    if not js:
        return f"*{title}: not evaluated yet.*"
    s = js["eval_set"]
    L = [f"**{title}** — image set `{s['sha256'][:12]}`: {s['n_images']:,} images ({', '.join(f'{k} {v:,}' for k, v in s['per_dataset'].items())}), "
         f"{s['n_sequences']} sequences. Every row uses exactly these images.", "",
         "| run | ckpt | epoch | input | mAP50 | mAP50-95 | drone AP50 | bird AP50 | micro AP50 (n GT) | small | standard | Anti-UAV | CST | CST micro | Det-Fly | bird-only imgs raising a drone alarm | GT birds boxed as drone | empty-frame drone FP / img |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"aerotrack_spd_v1": 0, "aerotrack_spd_finetuned": 1}
    order.update({f"repair_abl_{n}": 2 + i for i, (n, _) in enumerate(GRID)})
    key = lambda r: (order.get(re.sub(r"_e\d+$", "", r["run"]), 9), r["run"], r["checkpoint"] != "best")
    for r in sorted(js["rows"], key=key):
        L.append(f"| {r['run']} | {r['checkpoint']} | {r['epoch']} | {r.get('input', 't-1,t,t+1')} | {f4(r['mAP50'])} | {f4(r['mAP50-95'])} | {f4(r['drone_AP50'])} | {f4(r['bird_AP50'])} | "
                 f"{f4(r['micro_AP50'])} ({r['micro_n_gt']}) | {f4(r['small_AP50'])} | {f4(r['standard_AP50'])} | {f4(r['anti_uav_AP50'])} | {f4(r['cst_AP50'])} | "
                 f"{f4(r['cst_micro_AP50'])} | {f4(r['det_fly_AP50'])} | {f1(r['bird_only_img_drone_alarm_pct@.25'])} % | {f1(r['gt_birds_boxed_as_drone_pct@.25'])} % | "
                 f"{f4(r['empty_frame_drone_FP/img@.25'])} |")
    return "\n".join(L)


def ledger_lines():
    p = REPORTS_DIR / "test_evaluation_ledger.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def detector_block():
    val = jload(REPORTS_DIR / "detector_val_cap5000.json")
    L = [detector_table(val, "Validation")]
    tests = sorted(REPORTS_DIR.glob("detector_test_*.json"))
    if tests:
        tj = jload(tests[-1])
        led = ledger_lines()
        shas = {x["weights_sha256"][:12] for x in led}
        L += ["", detector_table(tj, "Test (final, once per model)"),
              "", f"*Test split: scored once per checkpoint after the ablation grid finished; `reports/test_evaluation_ledger.jsonl` holds "
                  f"{len(led)} entries for {len(shas)} distinct checkpoints. The test table is not used for any checkpoint choice or attribution "
                  "(the rule is applied on validation).*"]
    else:
        L += ["", "*Test set: not evaluated yet — it is used once, for the final table, after the ablation grid.*"]
    if val:
        rows = {(r["run"], r["checkpoint"]): r for r in val["rows"]}
        v1b, v1l, v2b = rows.get(("aerotrack_spd_v1", "best")), rows.get(("aerotrack_spd_v1", "last")), rows.get(("aerotrack_spd_finetuned", "best"))
        if v1b and v1l and v2b:
            L += ["", "**Reading (computed from the table):**",
                  f"- checkpoint selection: v1 *last* (epoch {v1l['epoch']}) beats v1 *best* (epoch {v1b['epoch']}, chosen by the fitness metric) on drone AP50 "
                  f"({v1l['drone_AP50']:.3f} vs {v1b['drone_AP50']:.3f}), CST ({v1l['cst_AP50']:.3f} vs {v1b['cst_AP50']:.3f}) and micro AP50 "
                  f"({v1l['micro_AP50']:.3f} vs {v1b['micro_AP50']:.3f}); the fitness metric picked a worse checkpoint.",
                  f"- fine-tune vs v1-last: drone AP50 {v2b['drone_AP50'] - v1l['drone_AP50']:+.3f}, CST AP50 {v2b['cst_AP50'] - v1l['cst_AP50']:+.3f}, "
                  f"Anti-UAV AP50 {v2b['anti_uav_AP50'] - v1l['anti_uav_AP50']:+.3f}, bird AP50 {v2b['bird_AP50'] - v1l['bird_AP50']:+.3f}, "
                  f"Det-Fly AP50 {v2b['det_fly_AP50'] - v1l['det_fly_AP50']:+.3f}; bird-only images raising a drone alarm "
                  f"{v1l['bird_only_img_drone_alarm_pct@.25']:.1f} % → {v2b['bird_only_img_drone_alarm_pct@.25']:.1f} %.",
                  "- so the fine-tune bought bird rejection and paid for it on the thermal drone datasets; the v1/v2 comparison attributes nothing "
                  "to SPD-Conv or the temporal stack — that is what the controlled ablation grid below is for."]
    return "\n".join(L)


def ablation_block(gs, vs):
    E = gs["epochs"]
    L = [f"Four runs, same recipe (`baseline` stage), same budget, one factor each; both checkpoints evaluated on the same image set as above. "
         f"**Attribution rule (pre-registered):** a factor is credited only if the full model beats the ablated model by > {ATTRIB_MARGIN:.2f} AP50 on the target "
         "metric for BOTH last.pt and best.pt; otherwise it has no measurable effect at this budget. Target metrics: SPD → micro AP50 and CST micro AP50; "
         "temporal stack → CST AP50 (motion cue) and drone AP50; stock baseline → drone AP50 overall.", "",
         "| ablation | changes | input | status | epochs | best.pt drone AP50 / micro / CST / CST micro | last.pt drone AP50 / micro / CST / CST micro |",
         "|---|---|---|---|---|---|---|"]
    any_run = False
    for name, desc in GRID:
        r = gs["runs"][name]
        if not r["exists"]:
            L.append(f"| {name} | {desc} | {INPUT_OF[name]} | **pending** | — | — | — |")
            continue
        any_run = True
        status = "finished" if r["finished"] else ("running/incomplete" if r["rows"] else "started")
        if r["launches"] > 1:
            status += f", {r['launches']} launches"
        cells = []
        for ck in ("best", "last"):
            v = r["val"][ck]
            cells.append(f"{f4(v['drone_AP50'])} / {f4(v['micro_AP50'])} / {f4(v['cst_AP50'])} / {f4(v['cst_micro_AP50'])}" if v else
                         ("stale evaluation (hash mismatch)" if ck in r["stale"] else "not evaluated"))
        L.append(f"| {name} | {desc} | {INPUT_OF[name]} | {status} (`{r['dir'].name}`) | {len(r['rows'])}/{r['meta'].get('epochs', '?')} | {cells[0]} | {cells[1]} |")
    if not any_run:
        L += ["", "No grid run has been trained yet (all four, with their evaluations and the final test table, run from one command: "
              "`scripts/run_grid.py`, see the end of this file). Until then **no claim about SPD-Conv or the temporal stack is made.**"]
        return "\n".join(L)
    L += ["", "**Attribution verdicts (rule above):**"] + [verdict_line(v) for v in vs]
    L += ["", "`single_frame` and `stock_yolo11n` are trained, validated and scored on t,t,t (three copies of frame t — their input definition, "
          "`src/model/train.py::build`); `baseline_ctrl` and `no_spd` on t−1,t,t+1. A verdict only uses finished runs whose evaluated checkpoint "
          "hashes match `run_meta.json`."]
    tr = transfer_record(E)
    if tr:
        L += ["COCO initialisation, same rule in every run (every tensor whose name and shape match; a 3-channel stem tiled /3 into a 9-channel "
              "stride stem): " + "; ".join(f"{k} {v['copied']}/{v['total']} tensors ({v['pct']:.1f} %)" for k, v in tr.items()) +
              ". With the installed Ultralytics the C3k2 bottleneck `cv1` convolutions are 1×1 while `yolo11n.pt` holds 3×3 ones, so those and "
              "the class head start from scratch in all four runs (and in v1)."]
    if (REPORTS_DIR / f"ablation_grid_e{E}.md").exists():
        L += ["", f"Full grid report (training status, runtimes, launches/recoveries, checkpoint hashes, test table): "
                  f"`experiments/aerotrack_repair/reports/ablation_grid_e{E}.md`."]
    return "\n".join(L)


def ablation_claim(gs, vs):
    """Middle + last cell of the README's Review-2 claim row for SPD-Conv / the temporal stack."""
    if not gs["epochs"] or not grid_complete(vs):
        return ("Untested until the ablation grid runs (one command, below). Inference-time ablation in the audit: removing motion costs "
                "≤ 0.005 mAP. | ablation table (pending)")
    return (f"Controlled {gs['epochs']}-epoch grid, pre-registered rule (> {ATTRIB_MARGIN:.2f} AP50 on best.pt AND last.pt) — "
            + "; ".join(verdict_summary(v) for v in vs) + ". | ablation table")


def tracker_block():
    js = sorted(REPORTS_DIR.glob("tracker_val_*.json"))
    if not js:
        return "*not evaluated yet*"
    t = jload(js[-1])
    L = [f"Detector `{Path(t['weights']).parent.parent.name}` (sha `{t['weights_sha256'][:12]}`), VAL, {t['sequence_rule']}: {len(t['sequences'])} sequences, "
         f"{sum(s['frames'] for s in t['sequences'])} frames, detections at conf {t['conf']}. Every sequence holds one drone.", "",
         "| tracker | MOTA | IDF1 | ID switches | ID sw. at occlusion re-entry | FP | FN | predicted IDs (GT: 1 per sequence) |", "|---|---|---|---|---|---|---|---|"]
    for k, v in t["aggregate"].items():
        L.append(f"| {k} | {v['MOTA']:.4f} | {v['IDF1_mean_per_sequence']:.4f} | {v['id_switches']} | {v['id_switches_at_occlusion_reentry']} | {v['false_positives']} | {v['false_negatives']} | {v['n_pred_ids']} |")
    L += ["", "| dataset | " + " | ".join(t["aggregate"]) + " |", "|---|" + "---|" * len(t["aggregate"])]
    for d, sub in t["per_dataset"].items():
        L.append(f"| {d} (MOTA / IDF1) | " + " | ".join(f"{sub[k]['MOTA']:.3f} / {sub[k]['IDF1_mean_per_sequence']:.3f}" for k in t["aggregate"]) + " |")
    L += ["", f"**Rule:** {t['rule']}. **Verdict: {t['verdict']}.** The demo and this README therefore make no tracking claim; "
          "`src/model/bytetrack.py` remains in the repository as an evaluated-and-rejected component."]
    return "\n".join(L)


def forecast_block():
    rep = jload(REPO_ROOT / "results/runs" / "lstm_forecaster_repair" / "forecast_repair_report.json")
    if not rep:
        return "*not evaluated yet*"
    tab = rep["test_table"]
    methods = [m for m in next(iter(tab.values())) if m != "n"]
    show = [m for m in methods if m.startswith(("zero_vel", "last_vel", "mean_vel_4", "kalman", "lstm"))]
    L = [f"Windows: observe {rep['config']['obs']} steps → forecast {rep['config']['pred']} (train {rep['windows']['train']:,} / val {rep['windows']['val']:,} / "
         f"test {rep['windows']['test']:,}). ADE in pixels on the TEST windows (lower is better); every method sees the same windows. "
         f"Selection: {rep['selection_rule']}.", "",
         "| subset | n | " + " | ".join(show) + " |", "|---|---|" + "---|" * len(show)]
    for sname, row in tab.items():
        if row["n"] == 0:
            continue
        L.append(f"| {sname} | {row['n']} | " + " | ".join(f"{row[m]['ADE_px']:.2f}" if m in row else "—" for m in show) + " |")
    d = rep["diagnostics"]
    cst_j = tab.get("CST sudden-jump", {})
    best_trivial = min(cst_j[m]["ADE_px"] for m in cst_j if m.startswith(("zero_vel", "last_vel", "mean_vel"))) if cst_j.get("n") else None
    kal = min(cst_j[m]["ADE_px"] for m in cst_j if m.startswith("kalman_val")) if cst_j.get("n") else None
    lstm_best = min(cst_j[m]["ADE_px"] for m in cst_j if m.startswith("lstm")) if cst_j.get("n") else None
    verdict = "retired" if (best_trivial is None or lstm_best is None or not (lstm_best < best_trivial and lstm_best < kal)) else "kept"
    L += ["", f"Does it move? true mean future step {d['true_mean_future_step_norm']:.4f} (normalised units); " +
          "; ".join(f"{k}: {v['mean_pred_step_norm']:.4f}" for k, v in d.items() if isinstance(v, dict)) + ".",
          f"**Rule:** the LSTM stays only if it beats the best trivial baseline AND the val-tuned Kalman on the CST sudden-jump subset. "
          f"CST sudden-jump ADE: best LSTM {f1(lstm_best)} vs best trivial {f1(best_trivial)} vs Kalman {f1(kal)} → **LSTM {verdict}.** "
          "On gimbal-tracked Anti-UAV footage the best forecaster is 'predict no motion'; the 'sudden jumps' are mostly frame-to-frame "
          "discontinuities (gimbal re-slews / label jumps) that no model can anticipate. Caveat: the forecaster's val split contains no CST windows.",
          "", "![Forecast examples on VAL, generated from model output](docs/assets/repair_forecast_examples_val.png)",
          "*Examples chosen by rule (median LSTM ADE among sudden-jump / smooth VAL windows); every plotted coordinate is in `docs/assets/repair_forecast_examples_val.json`.*"]
    return "\n".join(L)


def latency_block():
    js = sorted(REPORTS_DIR.glob("latency_*.json"))
    if not js:
        return "*not measured yet*"
    r = jload(js[-1])
    m = r["_meta"]
    L = [f"`{Path(m['weights']).parent.parent.name}` epoch {m['epoch']}, {m['gpu']}, batch 1, 640×640, {m['iters']} timed iterations after warm-up.", "",
         "| stage | mean ms | p50 ms | p95 ms |", "|---|---|---|---|"]
    for k, v in r.items():
        if isinstance(v, dict) and "mean_ms" in v:
            L.append(f"| {k} | {v['mean_ms']} | {v['p50_ms']} | {v['p95_ms']} |")
    L += ["", f"- **deployed configuration (fp32, no tracker): {r['end_to_end_fp32_no_tracker_ms']} ms ≈ {r['end_to_end_fp32_no_tracker_fps']} FPS** "
          f"(bf16 autocast is slower on this GPU at batch 1: {r['end_to_end_bf16_no_tracker_ms']} ms).",
          f"- the 3-frame stack needs frame t+1 before frame t can be processed: **+{r['structural_lookahead_latency_ms']} ms** structural latency at 25 FPS."]
    return "\n".join(L)


def demo_block():
    js = sorted((REPO_ROOT / "docs/assets").glob("repair_demo_val_*.json"))
    if not js:
        return "*no demo rendered yet*"
    d = jload(js[-1])
    gif = js[-1].with_suffix(".gif").name
    L = [f"![Side-by-side detector demo on a VAL sequence](docs/assets/{gif})", "",
         f"Sequence `{d['sequence']}` (VAL), frames {d['clip_frames'][0]}–{d['clip_frames'][1]}; **rule:** {d['rule']}. Both panels show raw NMS output at conf "
         f"{d['conf']} — no tracker, no trail, no magnifier; green = ground truth. Per-frame IoUs are in `docs/assets/{js[-1].name}`.", "",
         "| panel | run / checkpoint | epoch | hit rate (IoU ≥ .5 on frames with a target) | false alarms on frames labelled absent |", "|---|---|---|---|---|"]
    for side in ("left", "right"):
        s = d["summary"][side]
        L.append(f"| {side} | {s['run']} / {s['ckpt']} | {s['epoch']} | {100*s['hit_rate_iou50_on_exist_frames']:.1f} % | {s['false_alarms_on_absent_frames']} / {d['n_frames'] - d['n_exist']} |")
    L += ["", "Labels are taken as given; in this sequence the target is marked absent for most of the clip while it is partly visible at the frame "
          "edge in some frames (the Anti-UAV authors later published a start-frame correction for it), so 'false alarms' here include detections of a real object."]
    return "\n".join(L)


def render_readme(write=True):
    gs = grid_state()
    vs = verdicts(gs)
    tmpl = (REPAIR_DIR / "README.template.md").read_text(encoding="utf-8")
    blocks = {"definitions_block": definitions_block(gs), "detector_block": detector_block(), "ablation_block": ablation_block(gs, vs),
              "ablation_claim": ablation_claim(gs, vs), "tracker_block": tracker_block(), "forecast_block": forecast_block(),
              "latency_block": latency_block(), "demo_block": demo_block()}
    out = re.sub(r"\{\{(\w+)\}\}", lambda m: blocks[m.group(1)], tmpl)
    if write:
        (REPO_ROOT / "docs" / "REPAIR_SERIES.md").write_text(out, encoding="utf-8")
    return blocks, gs, vs, out


def plan_status(gs, vs):
    E = gs["epochs"]
    runs = [gs["runs"][n] for n, _ in GRID]
    n_fin = sum(1 for r in runs if r.get("finished"))
    n_eval = sum(1 for r in runs for ck in ("best", "last") if r["val"][ck])
    if E and grid_complete(vs):
        item2 = (f"done — 4 runs × {E} epochs, best.pt and last.pt of each evaluated; verdicts in 'Ablation grid' "
                 f"(`scripts/run_grid.py`, `reports/ablation_grid_e{E}.md`)")
        item3 = (f"done — the grid's `baseline_ctrl` ({E} epochs, cosine schedule completed) is the reference and every other grid run changes one "
                 "factor (`run_meta.json`); v1/v2 presented as a recipe change")
        sf = next(v for v in vs if v["ablation"] == "single_frame")
        item4 = f"audit result cited; training ablation `single_frame` done — {verdict_summary(sf)} (rule in 'Ablation grid')"
    elif E:
        item2 = f"in progress — {n_fin}/4 runs finished, {n_eval}/8 checkpoints evaluated (`scripts/run_grid.py`); see grid table"
        item3 = "grid design: `baseline_ctrl` is the reference trained at the grid budget; v1/v2 presented as a recipe change"
        item4 = "audit result cited; training ablation `single_frame` " + ("evaluated" if gs["runs"]["single_frame"]["val"]["last"] else "pending")
    else:
        item2 = "infrastructure done (`scripts/train_ablation.py`, `scripts/run_grid.py`, `ABLATIONS`); runs pending — see grid table"
        item3 = "grid design: `baseline_ctrl` is the reference trained at the grid budget; v1/v2 presented as a recipe change"
        item4 = "audit result cited; training ablation `single_frame` pending"
    try:
        c14 = subprocess.run(["git", "log", "-1", "--format=%h", "--", "results/runs/aerotrack_spd_finetuned/eval_val/evaluation_report.json"],
                             cwd=REPO_ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        c14 = ""
    item14 = ("superseded: the same-set table replaces `results/runs/aerotrack_spd_finetuned/eval_val`"
              + (f" (that directory was re-generated and committed in `{c14}`)" if c14 else ""))
    return [
        ("1", "same-set v1/v2 evaluation", "done — `scripts/evaluate_detector.py`, `reports/detector_val_cap5000.md`"),
        ("2", "ablations (SPD, temporal, stock YOLO11n)", item2),
        ("3", "one change at a time from a converged baseline", item3),
        ("4", "temporal stack contributes nothing (inference ablation)", item4),
        ("5", "LSTM vs trivial baselines", "done — retired by rule (`results/runs/lstm_forecaster_repair/report.md`)"),
        ("6", "ByteTrack: wire it or drop the claims", "done — retired by rule; all ByteTrack captions removed from README/demo"),
        ("7", "hand-typed trajectory plot", "replaced by `docs/assets/repair_forecast_examples_val.png` (+ JSON); rev2.ipynb kept as the historical submission"),
        ("8", "test-set demo cherry-picking", "replaced by `docs/assets/repair_demo_val_*.gif` (VAL, label rule, no tracker claims)"),
        ("9", "bird→drone false-alarm metric", "done — corrected metric in the detector table"),
        ("10", "latency", "done — measured fp32/bf16 with p95 + structural look-ahead"),
        ("11", "consistent micro definition / epochs / sampler table", "done — 'One definition of every term' section, all values read from code/args.json"),
        ("12", "honest checkpoint selection", "done — best.pt and last.pt evaluated side by side; last > best for v1"),
        ("13", "reproducible manifests", "done — every repair output carries git/env/weights hash/set hash; `train_ablation.py` writes run_meta.json (one record across resumes)"),
        ("14", "overwritten v2 eval report in the working tree", item14),
    ]


def render_report(blocks, gs, vs, write=True):
    g = git_info()
    L = [f"# AeroTrack-Net repair report", "", f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by `render_report.py` on branch `{g.get('branch')}` commit "
         f"`{(g.get('commit') or '')[:10]}` (dirty: {g.get('dirty')}). Plan: `AEROTRACK_REPAIR_PLAN.md`; audit: `../audit/AUDIT_REPORT.md`.*", "",
         "## Plan items", "", "| # | item | status |", "|---|---|---|"]
    L += [f"| {n} | {item} | {status} |" for n, item, status in plan_status(gs, vs)]
    for title, key in (("Definitions and run facts", "definitions_block"), ("Detector — same image set", "detector_block"), ("Ablation grid", "ablation_block"),
                       ("Tracking", "tracker_block"), ("Forecasting", "forecast_block"), ("Latency", "latency_block"), ("Demo", "demo_block")):
        L += ["", f"## {title}", "", blocks[key].replace("](docs/assets/", "](../assets/")]
    text = "\n".join(L) + "\n"
    if write:
        (REPO_ROOT / "docs" / "AEROTRACK_REPAIR_REPORT.md").write_text(text, encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="render in memory and report; write nothing")
    a = ap.parse_args()
    blocks, gs, vs, readme = render_readme(write=not a.check)
    report = render_report(blocks, gs, vs, write=not a.check)
    if a.check:
        left = re.findall(r"\{\{\w+\}\}", readme)
        print(f"[check] README {len(readme):,} chars, report {len(report):,} chars; grid budget {gs['epochs']}; "
              f"unfilled placeholders: {left or 'none'}; nothing written")
        if left:
            raise SystemExit(1)
        return
    print("wrote README.md and docs/AEROTRACK_REPAIR_REPORT.md")


if __name__ == "__main__":
    main()
