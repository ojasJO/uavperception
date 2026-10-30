#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_detector.py -- apples-to-apples detector benchmark + inference-time ablations
===================================================================================
Evaluates every checkpoint on the SAME frozen sample set (same split, same cap,
same seed) so the numbers are directly comparable. Reuses the project's own
`evaluation.py` so the metric definitions are identical to the reported ones.

Input-ablation modes (applied to the [B,9,H,W] tensor before the model):
    normal    [t-1, t, t+1]         the trained configuration
    static    [t,   t, t  ]         temporal context removed -> if the score
                                    barely moves, the 9-ch stack is decoration
    reversed  [t+1, t, t-1]         motion direction flipped
    zero_nb   [0,   t, 0  ]         neighbours blanked

Usage:
    python scripts/audit/bench_detector.py --split val --cap 1000 \
        --weights results/runs/aerotrack_spd_v1/weights/best.pt results/runs/aerotrack_spd_finetuned/weights/best.pt \
        --modes normal static
"""
import os, sys, json, time, argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/model"))
sys.path.insert(0, str(ROOT / "src/pipeline"))

from stacked_dataset import (AeroTrackDataset, aerotrack_collate,
                             build_split_samples, load_label_cache)
from aerotrack_trainer import load_checkpoint
import evaluation as E
from train import MANIFEST, SPLITS, LABEL_CACHE_PATH, subsample


class InputAblation(nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        self.model, self.mode = model, mode

    def forward(self, x):
        if self.mode == "static":
            cur = x[:, 3:6]
            x = torch.cat([cur, cur, cur], 1)
        elif self.mode == "reversed":
            x = torch.cat([x[:, 6:9], x[:, 3:6], x[:, 0:3]], 1)
        elif self.mode == "zero_nb":
            z = torch.zeros_like(x[:, 0:3])
            x = torch.cat([z, x[:, 3:6], z], 1)
        return self.model(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--cap", type=int, default=1000, help="images per dataset (0=all)")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--modes", nargs="+", default=["normal"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=30)
    ap.add_argument("--report-conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "results/runs" / "audit"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = a.datasets.split(",") if a.datasets else None
    samples = build_split_samples(MANIFEST, SPLITS, a.split, datasets=datasets)
    if a.cap:
        samples = subsample(samples, {d: a.cap for d in {s["dataset"] for s in samples}})
    print(f"[bench] split={a.split} cap={a.cap} -> {len(samples):,} images "
          f"{dict(Counter(s['dataset'] for s in samples))}", flush=True)

    ds = AeroTrackDataset(samples, img_size=(640, 640), augment=None, letterbox=True,
                          out_dtype="uint8", return_meta=True,
                          label_cache=load_label_cache(LABEL_CACHE_PATH))
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                    pin_memory=(device.type == "cuda"), collate_fn=aerotrack_collate,
                    persistent_workers=a.workers > 0)
    amp = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for w in a.weights:
        w = Path(w)
        model, ck = load_checkpoint(w, device, prefer_ema=True)
        run_name = w.parent.parent.name
        for mode in a.modes:
            m = InputAblation(model, mode).to(device).eval()
            t0 = time.time()
            recs = E.collect_records(m, dl, device, imgsz=640, conf=a.conf, iou=a.iou,
                                     max_det=a.max_det, amp_dtype=amp, progress=False)
            el = time.time() - t0
            rep = E.full_report(recs, save_dir=None, conf_thres=a.report_conf)
            rep["_meta"] = {"weights": str(w), "run": run_name, "epoch": ck.get("epoch"),
                            "split": a.split, "cap": a.cap, "mode": mode,
                            "n_images": len(ds), "seconds": round(el, 1),
                            "conf": a.conf, "iou": a.iou, "max_det": a.max_det}
            name = f"{a.split}_cap{a.cap}{('_' + a.tag) if a.tag else ''}__{run_name}__{mode}.json"
            (out_dir / name).write_text(json.dumps(rep, indent=2), encoding="utf-8")
            o, bs, bd = rep["overall"], rep["by_size"], rep["by_dataset"]
            row = {"run": run_name, "mode": mode, "n_images": len(ds),
                   "mAP50": o["map50"], "mAP50-95": o["map5095"],
                   "drone_AP50": o["per_class_ap50"]["uav_drone"],
                   "bird_AP50": o["per_class_ap50"]["bird_hard_negative"],
                   "micro_AP50": bs["micro"]["map50"], "micro_n_gt": bs["micro"]["n_gt"],
                   "small_AP50": bs["small"]["map50"], "std_AP50": bs["standard"]["map50"],
                   "anti_uav_AP50": bd.get("anti_uav", {}).get("map50"),
                   "cst_AP50": bd.get("cst", {}).get("map50"),
                   "det_fly_AP50": bd.get("det_fly", {}).get("map50"),
                   "bird_img_FP/img@.25": rep.get("bird_fp", {}).get("fp_per_image"),
                   "empty_FP/img@.25": rep.get("empty_frame_fp", {}).get("fp_per_image"),
                   "sec": round(el, 1)}
            summary.append(row)
            print(f"[bench] {run_name:<26} {mode:<9} mAP50 {o['map50']:.4f} "
                  f"mAP50-95 {o['map5095']:.4f} drone {o['per_class_ap50']['uav_drone']:.4f} "
                  f"bird {o['per_class_ap50']['bird_hard_negative']:.4f} "
                  f"micro {bs['micro']['map50']:.4f} (n={bs['micro']['n_gt']}) "
                  f"cst {row['cst_AP50']} ({el:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()

    sname = f"summary_{a.split}_cap{a.cap}{('_' + a.tag) if a.tag else ''}.json"
    (out_dir / sname).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[bench] wrote {out_dir / sname}")


if __name__ == "__main__":
    main()
