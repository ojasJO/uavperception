#!/usr/bin/env python
"""Independent evaluation: checkpoint + split -> raw predictions -> metrics.json + report.md (+ provenance).

    # validation (allowed any time; VAL is the tuning/monitoring split)
    python experiments/anti_uav/scripts/evaluate.py --checkpoint experiments/anti_uav/runs/dfsc_ir/weights/last.pt --split val

    # FINAL test evaluation (once per final model; logged to reports/test_evaluation_ledger.jsonl)
    python experiments/anti_uav/scripts/evaluate.py --checkpoint ... --split test --final-eval

    # re-score existing predictions without running the model
    python experiments/anti_uav/scripts/evaluate.py --predictions <dir>/predictions --split val --checkpoint <same ckpt>

    # Phase B, hypothesis H4 only: same checkpoint, different test-time input scale (640 512 = native IR)
    python experiments/anti_uav/scripts/evaluate.py --checkpoint ... --split val --scale 640 512

The tracker never reads ground truth after the init frame; metrics are computed only from the saved
prediction files. The TEST split is refused without --final-eval, and every test evaluation is appended
to an append-only ledger so repeated peeking is visible.

A --scale that differs from the checkpoint's training scale is a Phase B additional control: its output goes to
<run>/phaseB_test_scale_<L>x<S>/<split>/ (never to an eval_* directory, which aggregate_results.py reads as Phase A) and
its metrics/ledger records carry the variant. A full-protocol evaluation never overwrites an existing result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from anti_uav_repro.config import validate_config
from anti_uav_repro.data.antiuav import (TestSplitAccessError, build_manifest, load_manifest, load_sequences,
                                        manifest_path, verify_against_manifest)
from anti_uav_repro.data.transforms import TestTransform
from anti_uav_repro.engine.track import GlobalTrackTracker
from anti_uav_repro.eval.evaluate import evaluate_predictions, format_report
from anti_uav_repro.models.qg_rcnn import QGRCNN
from anti_uav_repro.paths import MANIFESTS_DIR, REPORTS_DIR
from anti_uav_repro.utils.repro import file_sha256, run_metadata, set_seed, write_json

LEDGER = REPORTS_DIR / "test_evaluation_ledger.jsonl"


def main():
    ap = argparse.ArgumentParser(description="Evaluate a QG-RCNN/DFSC checkpoint on Anti-UAV (val or test)")
    ap.add_argument("--checkpoint", required=True, help="training checkpoint (.pt) — its embedded config defines the model")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--modality", default=None, choices=[None, "IR", "RGB"], help="default: the checkpoint's training modality")
    ap.add_argument("--final-eval", action="store_true", help="REQUIRED for --split test. Declares this a final evaluation.")
    ap.add_argument("--out-dir", default=None, help="default: <run_dir>/eval_<split>_<modality>_<ckpt>")
    ap.add_argument("--predictions", default=None, help="score these existing predictions instead of running the tracker")
    ap.add_argument("--frame-stride", type=int, default=1, help="1 = full protocol. >1 marks the result PARTIAL (debug only)")
    ap.add_argument("--limit-sequences", type=int, default=None, help="debug only; marks the result PARTIAL")
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, nargs=2, default=None, metavar=("LONG", "SHORT"),
                    help="Phase B (H4) only: test-time rescale target instead of the checkpoint's training scale "
                         "(GlobalTrack rule: long side <= LONG, short side <= SHORT; 640 512 = native IR, no resize)")
    args = ap.parse_args()

    if args.split == "test" and not args.final_eval:
        sys.exit("Refusing: --split test requires --final-eval. TEST is for the final evaluation of a finished model only.")
    partial = args.frame_stride != 1 or args.limit_sequences is not None
    if args.split == "test" and partial:
        sys.exit("Refusing: partial (strided / limited) evaluation on TEST is not allowed.")

    ckpt_path = Path(args.checkpoint)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    validate_config(cfg)
    modality = args.modality or cfg["data"]["modality"]
    ckpt_sha = file_sha256(ckpt_path)
    run_dir = ckpt_path.parent.parent
    train_scale = tuple(int(v) for v in cfg["data"]["transforms"]["scale"])
    test_scale = tuple(args.scale) if args.scale else train_scale
    variant = None if test_scale == train_scale else f"test_scale_{test_scale[0]}x{test_scale[1]}"
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if variant and out_dir.name.startswith("eval_"):
            sys.exit(f"Refusing: a {variant} result in an eval_* directory would be read as a Phase A reproduction")
    elif variant:
        out_dir = run_dir / f"phaseB_{variant}" / f"{args.split}{'_PARTIAL' if partial else ''}"
    else:
        out_dir = run_dir / f"eval_{args.split}_{modality}_{ckpt_path.stem}{'_PARTIAL' if partial else ''}"
    if not partial and (out_dir / "metrics.json").exists():
        sys.exit(f"Refusing: {out_dir} already holds a full-protocol result; earlier evaluations are never overwritten "
                 f"(choose another --out-dir)")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(args.predictions) if args.predictions else out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    purpose = "final_eval" if args.split == "test" else "tune"
    try:
        seqs = load_sequences(cfg["data"]["root"], cfg["data"]["frames_root"], args.split, modality, purpose=purpose)
    except TestSplitAccessError as e:
        sys.exit(str(e))
    if args.limit_sequences:
        seqs = seqs[: args.limit_sequences]

    # ---- split identity: compare with the frozen manifest if it exists
    mpath = manifest_path(MANIFESTS_DIR, args.split, modality)
    manifest_info = {"path": str(mpath), "exists": mpath.exists()}
    if mpath.exists() and not args.limit_sequences:
        frozen = load_manifest(mpath)
        problems = verify_against_manifest(seqs, frozen)
        manifest_info["sha256"] = frozen["manifest_sha256"]
        manifest_info["verified"] = not problems
        if problems:
            sys.exit("Refusing: split differs from the frozen manifest:\n  " + "\n  ".join(problems[:20]))
    else:
        live = build_manifest(seqs, check_frames=False)
        manifest_info["sha256_live"] = live["manifest_sha256"]

    if args.split == "test":
        print("!" * 100)
        print("!!  FINAL TEST EVALUATION — this event is recorded in", LEDGER)
        print("!" * 100)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = run_metadata(cfg, extra={
        "kind": "evaluation", "split": args.split, "modality": modality, "purpose": purpose, "partial": partial,
        "frame_stride": args.frame_stride, "limit_sequences": args.limit_sequences,
        "checkpoint": str(ckpt_path), "checkpoint_sha256": ckpt_sha, "checkpoint_epoch": ck.get("epoch"),
        "train_run_name": ck.get("run_name"), "manifest": manifest_info, "out_dir": str(out_dir),
        "predictions_dir": str(pred_dir), "amp": args.amp, "device": str(device),
        "train_scale": list(train_scale), "test_scale": list(test_scale), "variant": variant,
        **({"phase": "B_additional_control"} if variant else {}),   # else keep the run config's phase
    })
    write_json(meta, out_dir / "eval_meta.json")

    # ---- tracking (unless scoring existing predictions)
    if not args.predictions:
        # build the architecture without any pretrained download, then load the checkpoint weights strictly
        model = QGRCNN({**cfg["model"], "init": "none"})
        model.load_state_dict(ck["model"], strict=True)
        model.to(device).eval()
        tracker = GlobalTrackTracker(model, device, TestTransform(test_scale, int(cfg["data"]["transforms"]["pad_divisor"])),
                                     amp=args.amp)
        t0 = time.time()
        total_frames = sum(s.n_frames for s in seqs)
        done = 0
        for k, s in enumerate(seqs, 1):
            out = tracker.track_sequence(s, frame_stride=args.frame_stride)
            with open(pred_dir / f"{s.name}.json", "w", encoding="utf-8") as f:
                json.dump(out, f)
            done += s.n_frames
            el = time.time() - t0
            eta = (total_frames - done) / max(done / max(el, 1e-6), 1e-9)
            print(f"[{k:03d}/{len(seqs)}] {s.name:24s} {modality} frames {s.n_frames:4d} start {out['start_frame']:3d} "
                  f"fps {out['fps'] or 0:.1f} | elapsed {el/60:.1f} min ETA {eta/60:.1f} min", flush=True)
        meta["tracking_seconds"] = round(time.time() - t0, 1)

    # ---- scoring (from files only)
    metrics = evaluate_predictions(pred_dir, seqs, only_processed_frames=partial)
    metrics["meta"] = {k: meta[k] for k in ("split", "modality", "partial", "checkpoint", "checkpoint_sha256", "checkpoint_epoch",
                                            "train_run_name", "timestamp_start", "git", "seed", "train_scale", "test_scale",
                                            "variant", "phase")}
    metrics["meta"]["manifest"] = manifest_info
    write_json(metrics, out_dir / "metrics.json")
    header = {"split": args.split, "modality": modality, "checkpoint": ckpt_path, "checkpoint_sha256": ckpt_sha,
              "checkpoint_epoch": ck.get("epoch"), "train_run": ck.get("run_name"), "git_commit": meta["git"]["commit"],
              "partial": partial, "manifest_sha256": manifest_info.get("sha256", manifest_info.get("sha256_live"))}
    if variant:
        header["variant (Phase B additional control)"] = f"{variant} (trained at {train_scale[0]}x{train_scale[1]} rule)"
    paper_ref = None
    try:
        with open(REPORTS_DIR / "paper_reported_results.json", "r", encoding="utf-8") as f:
            pr = json.load(f)
        col = pr["attribute_columns"].index("ALL_IR" if modality == "IR" else "ALL_VIS")
        rows = pr["table_V_protocol_II"][args.split]
        paper_ref = {f"Table V {args.split} {modality} {k} mSA%": v[col] for k, v in rows.items()}
    except Exception:  # noqa: BLE001 — reference block is optional
        paper_ref = None
    (out_dir / "report.md").write_text(format_report(metrics, header, paper_ref), encoding="utf-8")
    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["mSA_percent"] = metrics["aggregate"]["mSA_percent"]
    write_json(meta, out_dir / "eval_meta.json")

    a = metrics["aggregate"]
    print("-" * 100)
    print(f"{args.split.upper()} {modality} | {a['n_sequences']} seqs | mSA {a['mSA_percent']:.2f}% | precision@20 {100*a['precision_20px']:.2f}% | "
          f"success {100*a['success_auc']:.2f}% | {a['scoring']}")
    print(f"metrics: {out_dir / 'metrics.json'}\nreport : {out_dir / 'report.md'}")
    if args.split == "test":
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        rec = {"timestamp": meta["timestamp_end"], "checkpoint": str(ckpt_path), "checkpoint_sha256": ckpt_sha,
               "train_run_name": ck.get("run_name"), "modality": modality, "mSA_percent": a["mSA_percent"],
               "out_dir": str(out_dir), "git_commit": meta["git"]["commit"]}
        if variant:
            rec.update(phase="B", variant=variant, test_scale=list(test_scale),
                       note="Phase B additional control on a frozen checkpoint; selects nothing")
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"test evaluation recorded in ledger: {LEDGER}")


if __name__ == "__main__":
    main()
