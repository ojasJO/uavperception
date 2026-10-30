#!/usr/bin/env python
"""PHASE B — ADDITIONAL CONTROL: post-hoc absence rule on saved tracker outputs.

Single factor changed w.r.t. the Phase A protocol: the tracker declares the target ABSENT on a frame when its
top-1 detection score is below a threshold tau (Phase A: never absent, as GlobalTrack). Nothing is re-tracked:
the rule is applied to the prediction files written by evaluate.py (boxes + scores per frame).

Protocol (two explicit steps, test never used for selection):
    1. sweep:   python phaseb_absence_threshold.py --run dfsc_ir --sweep
                -> tau grid evaluated on the run's VAL predictions; tau* = argmax val mSA; written to
                   runs/<run>/phaseB_absence_thr/chosen_tau.json together with the full sweep table
    2. apply:   python phaseb_absence_threshold.py --run dfsc_ir --apply-test
                -> tau* applied ONCE to the run's saved TEST predictions; metrics + report + ledger entry
Outputs live under runs/<run>/phaseB_absence_thr/{val,test}/ and are picked up by aggregate_results.py as
"ADDITIONAL CONTROL" rows — never merged into the reproduction columns.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from anti_uav_repro.data.antiuav import load_sequences
from anti_uav_repro.eval.evaluate import evaluate_prediction_dicts, format_report, load_prediction
from anti_uav_repro.paths import REPORTS_DIR, RUNS_DIR
from anti_uav_repro.utils.repro import git_info, write_json

LEDGER = REPORTS_DIR / "test_evaluation_ledger.jsonl"
VARIANT = "absence_by_score_threshold"
TAU_GRID = [0.0] + [round(x, 3) for x in np.arange(0.05, 1.0, 0.05)] + [0.97, 0.99]


def apply_threshold(preds: dict, tau: float) -> dict:
    """score < tau -> absent ([0]) on tracked frames after the init frame; init frame and untracked frames untouched."""
    out = {}
    for name, p in preds.items():
        q = copy.deepcopy(p)
        start = q["start_frame"]
        for t, (done, sc) in enumerate(zip(q["processed"], q["scores"])):
            if done and t != start and sc < tau:
                q["res"][t] = [0]
        out[name] = q
    return out


def load_split(run_dir: Path, split: str, cfg: dict):
    ev = run_dir / f"eval_{split}_IR_last"
    if not (ev / "metrics.json").exists():
        sys.exit(f"missing full Phase A evaluation {ev} (run evaluate.py first)")
    meta = json.loads((ev / "metrics.json").read_text(encoding="utf-8"))["meta"]
    if meta.get("partial"):
        sys.exit("refusing to build a control on a PARTIAL evaluation")
    purpose = "final_eval" if split == "test" else "tune"
    seqs = load_sequences(cfg["data"]["root"], cfg["data"]["frames_root"], split, "IR", purpose=purpose)
    preds = {s.name: load_prediction(ev / "predictions", s.name) for s in seqs}
    return ev, meta, seqs, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under experiments/anti_uav/runs (e.g. dfsc_ir)")
    ap.add_argument("--sweep", action="store_true", help="step 1: tau sweep on VAL, choose tau*")
    ap.add_argument("--apply-test", action="store_true", help="step 2: apply the val-chosen tau* to TEST once")
    args = ap.parse_args()
    run_dir = RUNS_DIR / args.run
    import yaml
    cfg = yaml.safe_load((run_dir / "config.effective.yaml").read_text(encoding="utf-8"))
    out_root = run_dir / "phaseB_absence_thr"
    chosen_path = out_root / "chosen_tau.json"

    if args.sweep:
        ev, meta, seqs, preds = load_split(run_dir, "val", cfg)
        base = evaluate_prediction_dicts(preds, seqs)
        rows = []
        for tau in TAU_GRID:
            m = evaluate_prediction_dicts(apply_threshold(preds, tau), seqs)
            a = m["aggregate"]
            n_abs = sum(sum(1 for t, r in enumerate(p["res"]) if p["processed"][t] and len(r) != 4) for p in apply_threshold(preds, tau).values())
            rows.append({"tau": tau, "mSA_percent": a["mSA_percent"], "mSA_all_frames_percent": a["mSA_all_frames_percent"],
                         "precision_20px": a["precision_20px"], "success_auc": a["success_auc"], "frames_declared_absent": n_abs,
                         "by_attribute": {k: v["mSA_percent"] for k, v in a["by_attribute"].items()}})
        best = max(rows, key=lambda r: r["mSA_percent"])
        chosen = {"run": args.run, "variant": VARIANT, "selection_rule": "tau* = argmax over TAU_GRID of VAL mSA (frames from start frame)",
                  "tau": best["tau"], "val_mSA_percent_at_tau": best["mSA_percent"], "val_mSA_percent_phaseA": base["aggregate"]["mSA_percent"],
                  "source_val_eval": str(ev), "checkpoint_sha256": meta["checkpoint_sha256"], "git": git_info(),
                  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "grid": TAU_GRID}
        write_json({"rows": rows, "phaseA_val_mSA_percent": base["aggregate"]["mSA_percent"]}, out_root / "sweep_val.json")
        write_json(chosen, chosen_path)
        # full metrics of the chosen tau on val (same layout as a Phase A evaluation)
        mv = evaluate_prediction_dicts(apply_threshold(preds, best["tau"]), seqs)
        mv["meta"] = {**{k: meta[k] for k in ("split", "modality", "checkpoint", "checkpoint_sha256", "checkpoint_epoch", "train_run_name", "seed")},
                      "partial": False, "phase": "B_controlled_improvement", "variant": VARIANT, "tau": best["tau"], "git": git_info(),
                      "timestamp_start": chosen["timestamp"], "source_predictions": str(ev / "predictions")}
        write_json(mv, out_root / "val" / "metrics.json")
        (out_root / "val" / "report.md").write_text(format_report(mv, {"phase": "B (additional control)", "variant": VARIANT, "tau": best["tau"],
                                                                     "split": "val", "checkpoint_sha256": meta["checkpoint_sha256"]}), encoding="utf-8")
        print(f"{args.run} VAL: Phase A mSA {base['aggregate']['mSA_percent']:.2f} -> tau*={best['tau']} mSA {best['mSA_percent']:.2f} "
              f"(precision {100*best['precision_20px']:.2f}, success {100*best['success_auc']:.2f}, {best['frames_declared_absent']} frames declared absent)")
        print("tau   mSA    absent-frames")
        for r in rows:
            print(f"{r['tau']:<5} {r['mSA_percent']:6.2f} {r['frames_declared_absent']}")
        print(f"wrote {chosen_path}")

    if args.apply_test:
        if not chosen_path.exists():
            sys.exit("no chosen_tau.json — run --sweep (VAL) first; tau must never be chosen on TEST")
        if (out_root / "test" / "metrics.json").exists():
            sys.exit(f"{out_root / 'test'} already exists — the control was already applied to TEST once; not repeating")
        chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
        tau = float(chosen["tau"])
        ev, meta, seqs, preds = load_split(run_dir, "test", cfg)
        print("!" * 100 + f"\n!!  PHASE B control applied to TEST once: {VARIANT}, tau*={tau} (chosen on VAL)\n" + "!" * 100)
        mt = evaluate_prediction_dicts(apply_threshold(preds, tau), seqs)
        mt["meta"] = {**{k: meta[k] for k in ("split", "modality", "checkpoint", "checkpoint_sha256", "checkpoint_epoch", "train_run_name", "seed")},
                      "partial": False, "phase": "B_controlled_improvement", "variant": VARIANT, "tau": tau, "tau_selected_on": "val",
                      "git": git_info(), "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source_predictions": str(ev / "predictions")}
        write_json(mt, out_root / "test" / "metrics.json")
        (out_root / "test" / "report.md").write_text(format_report(mt, {"phase": "B (additional control)", "variant": VARIANT, "tau": tau,
                                                                      "split": "test", "checkpoint_sha256": meta["checkpoint_sha256"]}), encoding="utf-8")
        a = mt["aggregate"]
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": mt["meta"]["timestamp_start"], "checkpoint": meta["checkpoint"], "checkpoint_sha256": meta["checkpoint_sha256"],
                                "train_run_name": args.run, "modality": "IR", "mSA_percent": a["mSA_percent"], "out_dir": str(out_root / "test"),
                                "git_commit": git_info()["commit"], "phase": "B", "variant": VARIANT, "tau": tau, "note": "post-hoc rule on saved Phase A test predictions"}) + "\n")
        print(f"{args.run} TEST with tau*={tau}: mSA {a['mSA_percent']:.2f} | precision {100*a['precision_20px']:.2f} | success {100*a['success_auc']:.2f} "
              f"(Phase A test mSA was {json.loads((ev / 'metrics.json').read_text(encoding='utf-8'))['aggregate']['mSA_percent']:.2f}); ledger updated")


if __name__ == "__main__":
    main()
