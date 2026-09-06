#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ############################################################################
# ##   Requires a trained checkpoint + a CUDA GPU. Run on the OMEN 16 after  ##
# ##   train.py has produced runs/<name>/weights/best.pt.                    ##
# ############################################################################
"""
val.py  --  AeroTrack-Net rigorous evaluation
=============================================
Runs the full §7.2 suite from `evaluation.py` over a SEQUENCE-DISJOINT split:

    python c_model/val.py --weights runs/aerotrack_spd_v1/weights/best.pt
    python c_model/val.py --split test --confirm-test        # once. ever.

Why `--confirm-test` exists
---------------------------
`cst[test]` is 72,175 frames of the only footage in this corpus that is NOT
gimbal-tracked. Anti-UAV is 90% centre-concentrated because the camera actively
chases the drone; a model trained on it can learn "look in the middle" and still
score well on Anti-UAV val. CST-test is the one measurement that catches that.
Every time it is used to pick a checkpoint, an epoch count or a threshold, it
stops being a test set. So the flag is deliberately annoying: touch it once, at
the very end, and report whatever it says.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "b_pipeline"))

from stacked_dataset import (AeroTrackDataset, aerotrack_collate,
                             build_split_samples, load_label_cache)
from aerotrack_trainer import load_checkpoint
import evaluation as E
from train import (MANIFEST, SPLITS, LABEL_CACHE_PATH, require_gpu, subsample,
                   _worker_init, plan_workers)

DEFAULT_WEIGHTS = ROOT / "runs" / "aerotrack_spd_v1" / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--confirm-test", action="store_true",
                    help="required to evaluate on the held-out test split")
    ap.add_argument("--datasets", default=None,
                    help="comma list to restrict, e.g. 'cst' for the "
                         "cross-domain generalisation probe")
    ap.add_argument("--cap", type=int, default=0,
                    help="max images per dataset (0 = all)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=30)
    ap.add_argument("--report-conf", type=float, default=0.25,
                    help="operating threshold for the FP-rate figures")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_test:
        raise SystemExit(
            "\n[REFUSED] --split test needs --confirm-test.\n"
            "  The test split (notably cst[test]) is the project's only honest "
            "cross-domain\n  measurement. Spend it once, at the end, on the "
            "final model — not while tuning.\n")

    device = require_gpu()
    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"\n[ABORT] weights not found: {w}\n"
                         f"        Run c_model/train.py first.\n")

    datasets = args.datasets.split(",") if args.datasets else None
    samples = build_split_samples(MANIFEST, SPLITS, args.split, datasets=datasets)
    if args.cap:
        samples = subsample(samples, {d: args.cap for d in
                                      {s["dataset"] for s in samples}})
    if not samples:
        raise SystemExit(f"[ABORT] no samples for split={args.split} "
                         f"datasets={datasets}")

    ds = AeroTrackDataset(samples, img_size=(args.imgsz, args.imgsz),
                          augment=None, letterbox=True, out_dtype="uint8",
                          return_meta=True,
                          label_cache=load_label_cache(LABEL_CACHE_PATH))
    # Windows spawn multiprocess requires commit headroom; default to 2 workers on NT
    safe_workers = min(args.workers, 2) if os.name == "nt" else args.workers
    val_workers = plan_workers(safe_workers, reserve_gb=6.0, verbose=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=val_workers, pin_memory=True,
                    collate_fn=aerotrack_collate, worker_init_fn=_worker_init)

    model, ck = load_checkpoint(w, device, prefer_ema=not args.no_ema)
    print(f"[val] {w}  (epoch {ck.get('epoch')}, "
          f"fitness {ck.get('best_fitness')}, "
          f"{'EMA' if not args.no_ema and ck.get('ema') is not None else 'raw'} weights)")
    from collections import Counter
    print(f"[val] split={args.split}  {len(ds):,} images  "
          f"{dict(Counter(s['dataset'] for s in samples))}")

    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    try:
        recs = E.collect_records(model, dl, device, imgsz=args.imgsz, conf=args.conf,
                                 iou=args.iou, max_det=args.max_det, amp_dtype=amp)
    except RuntimeError as e:
        if "DataLoader worker" in str(e) and val_workers > 0:
            print(f"\n[val] Notice: Windows DataLoader worker pool issue detected ({e}).")
            print("[val] Switching seamlessly to robust in-process loader (workers=0)...")
            dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=True,
                            collate_fn=aerotrack_collate)
            recs = E.collect_records(model, dl, device, imgsz=args.imgsz, conf=args.conf,
                                     iou=args.iou, max_det=args.max_det, amp_dtype=amp)
        else:
            raise

    out = Path(args.out) if args.out else (w.parent.parent / f"eval_{args.split}")
    rep = E.full_report(recs, save_dir=out, conf_thres=args.report_conf)
    rep["_meta"] = {"weights": str(w), "split": args.split,
                    "datasets": datasets, "epoch": ck.get("epoch"),
                    "conf": args.conf, "iou": args.iou, "max_det": args.max_det}
    (out / "evaluation_report.json").write_text(json.dumps(rep, indent=2),
                                                encoding="utf-8")
    E.print_report(rep)
    print(f"  artifacts -> {out}")


if __name__ == "__main__":
    main()
