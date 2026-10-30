#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_bird_confusion.py -- the bird->drone false-alarm rate, measured correctly
==============================================================================
`evaluation.py::fp_rate` counts EVERY prediction on a bird-only image as a false
positive -- including correct bird-class detections -- so the reported `bird_fp`
number is not the bird->drone confusion rate. This script computes what the
"hard negative" story actually needs, on the full Det-Fly val split:

  * bird-only images: drone-class (cls 0) predictions per image at conf .25/.5
  * fraction of bird-only images that raise >=1 drone alarm ("cries wolf" rate)
  * drone-only images: bird-class predictions per image (the reverse confusion)
  * confusion at IoU>=.5: GT bird matched by a drone-class box, and vice versa
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/model")); sys.path.insert(0, str(ROOT / "src/pipeline"))
from stacked_dataset import AeroTrackDataset, aerotrack_collate, build_split_samples, load_label_cache
from aerotrack_trainer import load_checkpoint
import evaluation as E
from train import MANIFEST, SPLITS, LABEL_CACHE_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    samples = build_split_samples(MANIFEST, SPLITS, a.split, datasets=["det_fly"])
    ds = AeroTrackDataset(samples, img_size=(640, 640), augment=None, letterbox=True, out_dtype="uint8",
                          return_meta=True, label_cache=load_label_cache(LABEL_CACHE_PATH))
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, collate_fn=aerotrack_collate)
    out = {}
    for w in a.weights:
        run = Path(w).parent.parent.name
        model, _ = load_checkpoint(w, dev, prefer_ema=True)
        recs = E.collect_records(model, dl, dev, conf=0.001, iou=0.7, max_det=30, amp_dtype=torch.bfloat16, progress=False)
        res = {"n_images": len(recs)}
        for thr in (0.25, 0.5):
            bird_only = [r for r in recs if len(r["gt"]) and not (r["gt"][:, 0] == 0).any()]
            drone_only = [r for r in recs if len(r["gt"]) and not (r["gt"][:, 0] == 1).any()]
            def cls_preds(r, c): p = r["pred"]; return p[(p[:, 5] == c) & (p[:, 4] >= thr)] if len(p) else p
            d_on_bird = [len(cls_preds(r, 0)) for r in bird_only]
            b_on_drone = [len(cls_preds(r, 1)) for r in drone_only]
            # box-level confusion: GT bird whose best-IoU prediction (>=.5) is drone-class, and vice versa
            def confused(rs, gt_cls, wrong_cls):
                n_gt = n_conf = 0
                for r in rs:
                    g = r["gt"][r["gt"][:, 0] == gt_cls]; p = r["pred"]; p = p[p[:, 4] >= thr] if len(p) else p
                    n_gt += len(g)
                    if not len(g) or not len(p): continue
                    iou = E.box_iou_np(g[:, 1:5], p[:, :4])
                    for gi in range(len(g)):
                        j = int(np.argmax(iou[gi]))
                        if iou[gi, j] >= 0.5 and int(p[j, 5]) == wrong_cls: n_conf += 1
                return n_gt, n_conf
            nb, nb_as_drone = confused(recs, 1, 0); nd, nd_as_bird = confused(recs, 0, 1)
            res[f"conf>={thr}"] = {
                "bird_only_images": len(bird_only),
                "drone_alarms_per_bird_only_image": round(float(np.mean(d_on_bird)), 3),
                "bird_only_images_with_>=1_drone_alarm_pct": round(100 * float(np.mean([x > 0 for x in d_on_bird])), 1),
                "drone_only_images": len(drone_only),
                "bird_preds_per_drone_only_image": round(float(np.mean(b_on_drone)), 3),
                "gt_birds": nb, "gt_birds_labelled_drone_pct": round(100 * nb_as_drone / max(nb, 1), 1),
                "gt_drones": nd, "gt_drones_labelled_bird_pct": round(100 * nd_as_bird / max(nd, 1), 1),
            }
        out[run] = res
        print(run, json.dumps(res, indent=1), flush=True)
        del model; torch.cuda.empty_cache()
    (ROOT / "results/runs/audit/bird_confusion.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
