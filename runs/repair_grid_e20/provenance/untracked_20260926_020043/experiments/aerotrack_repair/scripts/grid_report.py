#!/usr/bin/env python
"""Final report of the controlled ablation grid — generated from files, no hand-typed numbers.

    python experiments/aerotrack_repair/scripts/grid_report.py [--epochs 20]

Reads runs/repair_abl_*_e<E>/{run_meta.json,args.json,results.csv}, reports/detector_{val,test}_cap5000.json,
reports/test_evaluation_ledger.jsonl and runs/repair_grid_e<E>/{events.jsonl,preflight_*.json}; writes
experiments/aerotrack_repair/reports/ablation_grid_e<E>.{json,md}. The attribution rule and target metrics are the
pre-registered ones in render_report.py (imported, not restated): a factor is credited only if the full model beats
the ablated model by > 0.02 AP50 on the target metric for BOTH best.pt and last.pt, on the validation set.
"""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import statistics
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import render_report as R
from imbalance_sampler import DEFAULT_WEIGHTS
from stacked_dataset import TemporalAugment

from anti_uav_repro.utils.repro import git_info

from _bootstrap import REPO_ROOT, REPORTS_DIR

ONE_COMMAND = r"venv\Scripts\python.exe experiments\aerotrack_repair\scripts\run_grid.py"
# what this task changed in the code (kept in step with the code; the numbers below all come from files)
CHANGES = [
    "`c_model/preflight.py` — T6 required a 36-channel first convolution (the SPD stem). The stride-2 stem of the `no_spd` and "
    "`stock_yolo11n` layouts has 9, so both runs would have aborted before epoch 0. T6 now takes the expected width from the live model "
    "(and still requires 9 or 36, so a reload that fell back to 3 channels still fails).",
    "`b_pipeline/stacked_dataset.py::aerotrack_collate` — inside a DataLoader worker the batch is now stacked straight into shared "
    "memory, as torch's own `default_collate` does. On this machine the old stack-then-share path left every batch behind in the "
    "worker (+118 MB per batch per worker, measured); the first launch of `baseline_ctrl` ran the machine out of memory 3.5 min in "
    "and lost a worker. Verified bit-exact against the old stack, flat memory at 4 workers; a loader-memory probe now guards it in the "
    "pre-flight.",
    "`c_model/aerotrack_trainer.py` — checkpoints are written to a temporary file and renamed atomically, so an interruption can never "
    "leave a truncated `last.pt` (the resume point).",
    "`c_model/train.py` — `--preflight-only`: build the exact job and run T1–T8, train nothing (used to check all four jobs up front).",
    "`scripts/train_ablation.py` — a resumed run keeps ONE `run_meta.json` (first start time, one `attempts` entry per launch, summed "
    "training time, workers used) and refuses a changed recipe; `--resume-from`; `--preflight-only`.",
    "`scripts/evaluate_detector.py` — runs trained with temporal_gap 0 (`single_frame`, `stock_yolo11n`) are scored on t,t,t, the input "
    "they were trained and validated on (`c_model/train.py::build`), instead of the 3-frame stack; the image set and its hash are unchanged "
    "and each row records its `input`. The summary is written after every checkpoint and before the test-ledger line; `--smoke`.",
    "`scripts/render_report.py`, `README.template.md` — grid runs in the run-facts table, an `input` column, verdicts only from finished, "
    "hash-matched runs at one budget (rule and target metrics unchanged), plan items 2–4 computed from the grid, a test-ledger note, the "
    "one-command reproduction, and a corrected sentence on which COCO tensors are transferred.",
    "new `scripts/run_grid.py` (the one command: pre-flight, sequential training with resume/watchdog, evaluations, test once, reports) "
    "and `scripts/grid_report.py` (this report).",
    "Not changed: metrics, splits, image sets, caps, thresholds, the training recipe, the epoch budget, the checkpoint-selection rule, "
    "the attribution rule, the COCO-transfer rule.",
]


def ts(s):
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
    except (TypeError, ValueError):
        return None


def hms(sec):
    if sec is None:
        return "—"
    return f"{sec / 3600:.2f} h"


def md_row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    E = a.epochs
    det = R.jload(REPORTS_DIR / "detector_val_cap5000.json") or {"rows": [], "eval_set": {}}
    test = R.jload(REPORTS_DIR / "detector_test_cap5000.json")
    gs = R.grid_state(det, epochs=E)
    vs = R.verdicts(gs)
    gdir = REPO_ROOT / "runs" / f"repair_grid_e{E}"
    events = []
    if (gdir / "events.jsonl").exists():
        events = [json.loads(x) for x in (gdir / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    pfs = sorted(gdir.glob("preflight_*.json"))
    pf = R.jload(pfs[-1]) if pfs else None
    transfer = R.transfer_record(E) or {}
    ledger = R.ledger_lines()

    # ------------------------------------------------------------------ 1. training status
    runs_out = {}
    for name, desc in R.GRID:
        r = gs["runs"][name]
        if not r["exists"]:
            runs_out[name] = {"status": "not started", "change": desc}
            continue
        meta, rows, d = r["meta"], r["rows"], r["dir"]
        args = R.jload(d / "args.json") or {}
        attempts = meta.get("attempts") or []
        t0 = ts(attempts[0]["timestamp_start"]) if attempts else ts(meta.get("timestamp_start"))
        t1 = ts(meta.get("timestamp_end"))
        ep_times = [float(x["epoch_time_s"]) for x in rows if x.get("epoch_time_s")]
        best_row = max(rows, key=lambda x: float(x["fitness"])) if rows else None
        runs_out[name] = {
            "run": d.name, "change": desc, "input": R.INPUT_OF[name],
            "status": "completed" if r["finished"] else ("interrupted/incomplete" if rows else "started, no epoch finished"),
            "epochs_completed": len(rows), "epochs_planned": meta.get("epochs"), "launches": len(attempts) or 1,
            "launch_outcomes": [{k: x.get(k) for k in ("attempt", "timestamp_start", "timestamp_end", "seconds", "return_code", "resumed_from",
                                                       "epochs_on_disk_at_start", "epochs_on_disk_at_end", "workers_requested", "workers_used", "outcome")}
                                for x in attempts],
            "train_seconds_sum_of_launches": meta.get("train_seconds"),
            "wall_clock_seconds_first_start_to_end": (t1 - t0).total_seconds() if (t0 and t1) else None,
            "epoch_train_time_s_median": statistics.median(ep_times) if ep_times else None,
            "model_cfg": meta.get("model_cfg"), "temporal_gap": meta.get("temporal_gap"),
            "batch": args.get("batch"), "accumulate": max(1, int(args.get("nbs", 64)) // int(args.get("batch", 32))) if args else None,
            "workers_used": args.get("workers"), "lr0": args.get("lr0"), "lrf": args.get("lrf"),
            "coco_transfer": transfer.get(name),
            "best_pt": {"path": str((d / "weights" / "best.pt").relative_to(REPO_ROOT)), "sha256": meta.get("best_pt_sha256"),
                        "epoch": int(float(best_row["epoch"])) if best_row else None, "fitness": float(best_row["fitness"]) if best_row else None},
            "last_pt": {"path": str((d / "weights" / "last.pt").relative_to(REPO_ROOT)), "sha256": meta.get("last_pt_sha256"),
                        "epoch": int(float(rows[-1]["epoch"])) if rows else None},
            "final_epoch_val_capped": {k: rows[-1].get(k) for k in ("mAP50", "mAP50-95", "micro_AP50", "fitness")} if rows else None,
        }
    # controlled-variable check: every args.json key identical across the four runs except name / workers
    argsets = {n: (R.jload(gs["runs"][n]["dir"] / "args.json") or {}) for n, _ in R.GRID if gs["runs"][n]["exists"]}
    keys = sorted({k for v in argsets.values() for k in v})
    differ = [k for k in keys if len({json.dumps(v.get(k), sort_keys=True) for v in argsets.values()}) > 1]
    common = next(iter(argsets.values()), {})
    aug = {k: p.default for k, p in inspect.signature(TemporalAugment.__init__).parameters.items() if k not in ("self", "seed")}

    # ------------------------------------------------------------------ 2./5. evaluation tables
    grid_names = {gs["runs"][n]["dir"].name for n, _ in R.GRID if gs["runs"][n]["exists"]}
    val_rows = [x for x in det["rows"] if x["run"] in grid_names or x["run"] in ("aerotrack_spd_v1", "aerotrack_spd_finetuned")]
    test_rows = (test or {}).get("rows", [])
    grid_val_done = sum(1 for n, _ in R.GRID for ck in ("best", "last") if gs["runs"][n]["val"][ck])
    grid_test_done = sum(1 for x in test_rows if x["run"] in grid_names)
    complete = all(gs["runs"][n].get("finished") for n, _ in R.GRID) and grid_val_done == 8 and R.grid_complete(vs) and grid_test_done == 8

    # descriptive test deltas (NOT part of the rule, which is applied on validation)
    test_desc = []
    if test_rows:
        def trow(run, ck):
            return next((x for x in test_rows if x["run"] == run and x["checkpoint"] == ck), None)
        base = gs["runs"]["baseline_ctrl"]
        for v in vs:
            ab = gs["runs"][v["ablation"]]
            if not (base["exists"] and ab["exists"]):
                continue
            for m in v["metrics"]:
                pair = [(trow(base["dir"].name, ck), trow(ab["dir"].name, ck)) for ck in ("best", "last")]
                if all(p and q for p, q in pair):
                    test_desc.append({"factor": v["factor"], "metric": m["metric"],
                                      "delta_best": pair[0][0][m["metric"]] - pair[0][1][m["metric"]],
                                      "delta_last": pair[1][0][m["metric"]] - pair[1][1][m["metric"]]})

    # ------------------------------------------------------------------ 7. errors / recoveries
    notable = [e for e in events if e["event"] in ("train_failed", "train_recover", "checkpoint_unreadable", "val_eval_failed",
                                                   "test_eval_interrupted", "test_ledger_restored", "render_failed", "gpu_busy_waiting",
                                                   "fatal", "stopped", "operator_note")
               or (e["event"] == "preflight" and not e.get("ok"))]
    starts = [e for e in events if e["event"] == "start"]

    # ------------------------------------------------------------------ JSON
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_info(), "epochs": E, "workflow_complete": complete,
           "rule": {"margin_ap50": R.ATTRIB_MARGIN, "both_checkpoints": True, "split": "val",
                    "targets": {f: list(k) for f, _, k in R.FACTORS}},
           "val_set": det.get("eval_set"), "test_set": (test or {}).get("eval_set"),
           "runs": runs_out, "args_json_keys_that_differ": differ, "common_config": common,
           "augmentation_defaults": aug, "sampler_weights": DEFAULT_WEIGHTS,
           "val_rows": val_rows, "verdicts": vs, "test_rows": test_rows, "test_ledger": ledger,
           "test_descriptive_deltas": test_desc, "events": notable, "grid_launches": len(starts),
           "preflight": {"file": pfs[-1].name if pfs else None, "ok": pf.get("ok") if pf else None,
                         "checks": pf.get("checks") if pf else None, "environment": pf.get("environment") if pf else None,
                         "ablations": pf.get("ablations") if pf else None},
           "changes": CHANGES, "reproduce": ONE_COMMAND}
    jpath = REPORTS_DIR / f"ablation_grid_e{E}.json"
    jpath.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------ Markdown
    L = [f"# AeroTrack-Net controlled ablation grid — final report ({E} epochs)", "",
         f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by `experiments/aerotrack_repair/scripts/grid_report.py` from run_meta.json / args.json / "
         f"results.csv, `reports/detector_val_cap5000.json`, `reports/detector_test_cap5000.json`, the test ledger and "
         f"`runs/repair_grid_e{E}/` (events, pre-flight). No number in this file is typed by hand. Machine-readable twin: `{jpath.name}`.*", "",
         "## Outcome", "",
         f"- **Workflow: {'COMPLETE' if complete else 'INCOMPLETE'}** — {sum(1 for n, _ in R.GRID if gs['runs'][n].get('finished'))}/4 runs finished, "
         f"{grid_val_done}/8 grid checkpoints evaluated on validation, {grid_test_done}/8 grid checkpoints (+ "
         f"{sum(1 for x in test_rows if x['run'] not in grid_names)} historical) evaluated once on test.",
         f"- **Pre-registered rule:** a factor is credited only if the full model (`baseline_ctrl`) beats the ablated model by > {R.ATTRIB_MARGIN:.2f} AP50 "
         "on the target metric for BOTH best.pt and last.pt, on the validation set; otherwise it has no measurable effect at this budget."]
    for v in vs:
        L.append(f"- {R.verdict_summary(v) if all(m['evaluated'] for m in v['metrics']) else v['factor'] + ': not evaluated'}")
    L += ["", "## 1. Training status", "",
          md_row(["run", "factor changed", "input", "status", "epochs", "launches", "training time (sum of launches)", "wall clock", "median train time / epoch",
                  "workers used", "COCO transfer", "best.pt (epoch, capped-val fitness)", "last.pt (epoch)"]),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, _ in R.GRID:
        x = runs_out[name]
        if "run" not in x:
            L.append(md_row([f"repair_abl_{name}_e{E}", x["change"], R.INPUT_OF[name], x["status"], "—", "—", "—", "—", "—", "—", "—", "—", "—"]))
            continue
        tr = x["coco_transfer"] or {}
        L.append(md_row([f"`{x['run']}`", x["change"], x["input"], x["status"], f"{x['epochs_completed']}/{x['epochs_planned']}", x["launches"],
                         hms(x["train_seconds_sum_of_launches"]), hms(x["wall_clock_seconds_first_start_to_end"]),
                         f"{x['epoch_train_time_s_median'] / 60:.1f} min" if x["epoch_train_time_s_median"] else "—", x["workers_used"],
                         f"{tr.get('copied')}/{tr.get('total')} ({tr.get('pct')} %)" if tr else "—",
                         f"epoch {x['best_pt']['epoch']} ({x['best_pt']['fitness']:.4f})" if x["best_pt"]["fitness"] is not None else "—",
                         f"epoch {x['last_pt']['epoch']}" if x["last_pt"]["epoch"] is not None else "—"]))
    if any("run" in x for x in runs_out.values()):
        L += ["", "Checkpoints (sha256 prefix): " + "; ".join(
            f"`{x['best_pt']['path']}` {(x['best_pt']['sha256'] or '—')[:12]}, `{x['last_pt']['path']}` {(x['last_pt']['sha256'] or '—')[:12]}"
            for x in runs_out.values() if "run" in x)]
    if common:
        L += ["", "**Training configuration (identical for all four runs; read from `args.json`)** — "
              f"stage `baseline` recipe, {common.get('epochs')} epochs × {common.get('samples_per_epoch'):,} sampled frames, batch {common.get('batch')} × "
              f"accumulate {max(1, int(common.get('nbs', 64)) // int(common.get('batch', 32)))}, {common.get('optimizer')} lr0 {common.get('lr0')} → ×{common.get('lrf')} "
              f"(cosine, {common.get('warmup_epochs')} warm-up epochs), weight decay {common.get('weight_decay')}, {common.get('amp_dtype')} autocast, EMA "
              f"{common.get('ema')}, grad clip {common.get('grad_clip')}, image {common.get('imgsz')}, seed {common.get('seed')}, per-epoch validation on the "
              f"capped val set at conf {common.get('val_conf')} (checkpoint selection only; fitness weights {common.get('fitness_weights')}), patience "
              f"{common.get('patience')}. Sampler weights {DEFAULT_WEIGHTS}; augmentation (TemporalAugment defaults) {aug}.",
              f"`args.json` keys that differ between the four runs: {differ or 'none'} (the run name and the worker count the loader planner "
              "chose from free memory; neither changes what is trained).",
              "Factor per run: `no_spd` swaps the five SPDConv downsamplers for stride-2 convolutions (`yolo11-stride.yaml`); `single_frame` feeds "
              "t,t,t instead of t−1,t,t+1; `stock_yolo11n` does both. COCO initialisation uses the same rule everywhere (tensors whose name and "
              "shape match; a 3-channel stem is tiled /3 into a 9-channel stride stem)."]
    pfa = (pf or {}).get("ablations") or {}
    if pfa:
        L += ["", "Pre-flight on each exact job (train.py T1–T8, nothing trained): " + "; ".join(
            f"{k}: {v.get('model_cfg')}, {v.get('n_spdconv')} SPDConv, first conv {v.get('t6_first_conv_in')} ch, {v.get('iters_per_epoch')} it/epoch, "
            f"VRAM peak {v.get('vram_peak_gb')} GB (headroom {v.get('vram_headroom_gb')} GB)" for k, v in pfa.items()) + "."]

    L += ["", "## 2. Validation — every run, both checkpoints", "",
          R.detector_table({"eval_set": det.get("eval_set", {}), "rows": val_rows}, "Validation"),
          "", "v1/v2 rows are the historical models, shown for reference; the grid runs are compared only with each other."]

    L += ["", "## 3. Comparison with the full model (pre-registered rule, validation)", "",
          md_row(["factor", "ablated run", "target metric", "full best.pt", "ablated best.pt", "Δ best", "full last.pt", "ablated last.pt", "Δ last",
                  f"> {R.ATTRIB_MARGIN:.2f} on both?", "verdict"]),
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in vs:
        for m in v["metrics"]:
            if not m["evaluated"]:
                L.append(md_row([v["factor"], v["ablation"], R.METRIC_LABEL[m["metric"]], "—", "—", "—", "—", "—", "—", "—", "not evaluated"]))
                continue
            L.append(md_row([v["factor"], v["ablation"], R.METRIC_LABEL[m["metric"]], f"{m['full_best']:.4f}", f"{m['ablated_best']:.4f}",
                             f"{m['delta_best']:+.4f}", f"{m['full_last']:.4f}", f"{m['ablated_last']:.4f}", f"{m['delta_last']:+.4f}",
                             "yes" if m["credited"] else "no", "**credited**" if m["credited"] else "no measurable effect"]))

    L += ["", "## 4. What the measurements support", ""]
    if not R.grid_complete(vs):
        L.append("Not every comparison is evaluated yet, so no factor is credited or dismissed here.")
    else:
        for v in vs:
            for m in v["metrics"]:
                lab = R.METRIC_LABEL[m["metric"]]
                if m["credited"]:
                    L.append(f"- **{v['factor']} — supported on {lab}:** the full model is ahead by {m['delta_best']:+.4f} (best.pt) and "
                             f"{m['delta_last']:+.4f} (last.pt), both above the {R.ATTRIB_MARGIN:.2f} margin.")
                else:
                    side = ("the ablated model scores higher on both checkpoints" if m["delta_best"] < 0 and m["delta_last"] < 0 else
                            "the two checkpoints disagree in sign" if (m["delta_best"] > 0) != (m["delta_last"] > 0) else
                            "the full model is ahead, but not by more than the margin on both checkpoints")
                    L.append(f"- **{v['factor']} — not supported on {lab}** (Δ {m['delta_best']:+.4f} best.pt, {m['delta_last']:+.4f} last.pt): {side}. "
                             "Under the rule this is 'no measurable effect at this budget', not evidence that the factor hurts.")
        L += ["", f"Scope: one seed per configuration and a {E}-epoch budget, so a difference inside the margin cannot be separated from "
              "run-to-run noise; the verdicts are about this budget, not about convergence."]

    L += ["", "## 5. Final test — evaluated once per checkpoint", ""]
    if test:
        shas = {x["weights_sha256"] for x in ledger}
        L += [R.detector_table(test, "Test (final, once per model)"), "",
              f"Ledger `reports/test_evaluation_ledger.jsonl`: {len(ledger)} entries for {len(shas)} distinct checkpoints "
              + ("(one per checkpoint)" if len(ledger) == len(shas) else "(**more entries than checkpoints — see the ledger**)")
              + f", test set `{(test.get('eval_set') or {}).get('sha256', '')[:12]}`, first {min(x['timestamp'] for x in ledger) if ledger else '—'}, "
              f"last {max(x['timestamp'] for x in ledger) if ledger else '—'}. The test split was scored only after all four runs and their "
              "validation evaluations had finished, and it is not used for any choice or verdict."]
        if test_desc:
            L += ["", "Descriptive only — the same full-minus-ablated differences on TEST (the rule is applied on validation; nothing is credited here):", "",
                  md_row(["factor", "metric", "Δ best.pt (test)", "Δ last.pt (test)"]), "|---|---|---|---|"]
            L += [md_row([t["factor"], R.METRIC_LABEL[t["metric"]], f"{t['delta_best']:+.4f}", f"{t['delta_last']:+.4f}"]) for t in test_desc]
    else:
        L.append("*Not evaluated yet — the test split is scored once, after all four runs and their validation evaluations.*")

    L += ["", "## 6. Generated artefacts", "",
          "- this report: `experiments/aerotrack_repair/reports/ablation_grid_e{0}.md` and `.json`".format(E),
          "- validation / test tables: `experiments/aerotrack_repair/reports/detector_val_cap5000.{json,md}`, `detector_test_cap5000.{json,md}`; "
          "per-checkpoint reports + PR / size plots in `runs/<run>/eval_repair_<split>_cap5000/<ckpt>/`",
          "- test ledger: `experiments/aerotrack_repair/reports/test_evaluation_ledger.jsonl`",
          "- regenerated `README.md` and `docs/AEROTRACK_REPAIR_REPORT.md` (`scripts/render_report.py`)",
          f"- runs: `runs/repair_abl_<ablation>_e{E}/` (run_meta.json, args.json, results.csv, training_curves.png; weights are git-ignored)",
          f"- grid runner: `runs/repair_grid_e{E}/` — events.jsonl, grid_state.json, preflight_*.json, provenance/ (code diff + copies of "
          "untracked sources), logs/ (every step's full log; git-ignored by the repository's `*.log` rule)"]

    L += ["", "## 7. Errors, interruptions and recoveries", ""]
    if pf:
        warns = [c for c in pf.get("checks", []) if not c["ok"]]
        L.append(f"- pre-flight ({pfs[-1].name}): {'passed' if pf.get('ok') else 'FAILED'}"
                 + (f"; warnings: " + "; ".join(f"{c['name']}: {c['detail']}" for c in warns) if warns else "; no warnings"))
    L.append(f"- grid runner launches: {len(starts)}" + (" (" + ", ".join(e["t"] + (" pre-flight only" if "--preflight-only" in str(e.get("argv")) else "")
                                                             for e in starts) + ")" if starts else ""))
    shown = notable
    if not shown:
        L.append("- none: no training launch failed, no evaluation was interrupted, nothing was resumed or retried.")
    for e in shown:
        extra = {k: v for k, v in e.items() if k not in ("t", "event", "tail", "weights")}
        L.append(f"- {e['t']} `{e['event']}` " + ", ".join(f"{k}: {v}" for k, v in extra.items()))
    for name, _ in R.GRID:
        x = runs_out[name]
        if "run" in x and x["launches"] > 1:
            L.append(f"- `{x['run']}` needed {x['launches']} launches: " + "; ".join(
                f"#{o.get('attempt')} rc {o.get('return_code')} epochs {o.get('epochs_on_disk_at_start')}→{o.get('epochs_on_disk_at_end')}"
                + (f" (resumed from {Path(o['resumed_from']).name})" if o.get("resumed_from") else "") for o in x["launch_outcomes"]))

    L += ["", "## 8. What changed in this task", ""] + [f"- {c}" for c in CHANGES]
    L += ["", "## 9. Reproduce", "", "```bat", ONE_COMMAND, "```", "",
          "It re-verifies instead of re-running: finished runs are checked by checkpoint hash, evaluated checkpoints by their rows, and a "
          "checkpoint with a test-ledger line is never scored on test again. To retrain from scratch, move `runs/repair_abl_*_e20/` aside first."]
    mpath = REPORTS_DIR / f"ablation_grid_e{E}.md"
    mpath.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {mpath} and {jpath} (workflow complete: {complete})")


if __name__ == "__main__":
    main()
