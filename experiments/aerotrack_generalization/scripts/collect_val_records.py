#!/usr/bin/env python
"""D3 — per-image VAL predictions of the 8 grid checkpoints, same protocol as experiments/aerotrack_repair/scripts/
evaluate_detector.py (registered set 6fb41d44d45b, conf 0.001, IoU 0.7, max_det 30, bf16, each run on its own input).

    venv\\Scripts\\python.exe experiments\\aerotrack_generalization\\scripts\\collect_val_records.py

VAL ONLY: the test split is never read. Cache: runs/aerotrack_generalization/_cache/val_records/<run>__<ckpt>.pkl (git-ignored);
existing files are kept. The mAP50 printed per checkpoint must equal reports/detector_val_cap5000.json of the repair series."""
import pickle
import time

import _series as S
import torch
from torch.utils.data import DataLoader

import evaluation as E
from aerotrack_trainer import load_checkpoint
from evaluate_detector import eval_set, input_mode_for, single_frame_samples
from stacked_dataset import AeroTrackDataset, aerotrack_collate, load_label_cache
from train import LABEL_CACHE_PATH

from anti_uav_repro.utils.repro import file_sha256

GRID = ("baseline_ctrl", "no_spd", "single_frame", "stock_yolo11n")
OUT = S.CACHE / "val_records"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    weights = [S.REPO / "runs" / f"repair_abl_{a}_e20" / "weights" / f"{c}.pt" for a in GRID for c in ("best", "last")]
    samples, setinfo = eval_set("val", 5000, None)
    assert setinfo["sha256"].startswith("6fb41d44d45b"), setinfo["sha256"]
    lc = load_label_cache(LABEL_CACHE_PATH)
    dev = torch.device("cuda:0")
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    loaders, done = {}, {}
    for w in weights:
        mode, _ = input_mode_for(w)
        out = OUT / f"{w.parent.parent.name}__{w.stem}.pkl"
        if out.exists():
            done[out.name] = "cached"
            continue
        if mode not in loaders:
            ss = single_frame_samples(samples) if mode == "single_frame" else samples
            ds = AeroTrackDataset(ss, img_size=(640, 640), augment=None, letterbox=True, out_dtype="uint8", return_meta=True, label_cache=lc)
            loaders[mode] = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, collate_fn=aerotrack_collate)
        model, ck = load_checkpoint(w, dev, prefer_ema=True)
        t0 = time.time()
        recs = E.collect_records(model, loaders[mode], dev, imgsz=640, conf=0.001, iou=0.7, max_det=30, amp_dtype=amp, progress=False)
        slim = [{k: r[k] for k in ("pred", "gt", "gt_area", "dataset", "seq_key")} for r in recs]
        pickle.dump({"weights": str(w), "weights_sha256": file_sha256(w), "epoch": ck.get("epoch"), "input": mode, "set": setinfo["sha256"],
                     "records": slim}, open(out, "wb"))
        done[out.name] = round(E.score(recs)["map50"], 4)
        print(f"{out.name}: {len(recs)} images, mAP50 {done[out.name]} ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    S.ledger({"id": "D3", "type": "diagnostic", "what": "per-image VAL predictions of the 8 grid checkpoints (registered protocol)",
              "script": "scripts/collect_val_records.py", "val_set": setinfo["sha256"], "mAP50_check": done,
              "output": str(OUT.relative_to(S.REPO)), "test_usage": "none"})


if __name__ == "__main__":
    main()
