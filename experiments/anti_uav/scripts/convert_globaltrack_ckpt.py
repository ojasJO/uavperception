#!/usr/bin/env python
"""Track A gap A2, step 2: convert the official GlobalTrack checkpoint (mmdetection 0.6.0) into our torchvision QG-RCNN.

    python experiments/anti_uav/scripts/convert_globaltrack_ckpt.py --ckpt <qg_rcnn_r50_fpn_coco_got10k_lasot.pth> --out <converted.pt>

Mapping: inspect_globaltrack_ckpt.RULES (321/322 tensors, every shape equal; BN `num_batches_tracked` counters dropped).
The result loads with strict=True into QGRCNN(init='none', frozen_bn_eps=1e-5) -- 1e-5 is mmcv's BatchNorm eps, the value
these weights were trained with. Output: {"model": state_dict, "meta": {...}} as engine/train.py expects for init='checkpoint'.

Known convention differences between mmdetection 0.6.0 and torchvision (recorded, NOT corrected -- the paper fine-tunes from
these weights, which absorbs systematic offsets; quantified by the zero-shot check):
  C1 anchor centres: mmdet (stride-1)/2 inside each cell, torchvision on the grid points (offset 1.5 px at stride 4 ... 31.5 px at 64)
  C2 legacy box width/height +1 in mmdet's delta coding and RoI coordinates
  C3 identical: 2 classes with background at index 0, class-specific regression, RCNN stds (0.1,0.1,0.2,0.2) = torchvision weights
     (10,10,5,5), RPN stds 1, anchor scale 8 x ratios (0.5,1,2) with ratio = h/w in both, RoIAlign 7x7 sampling 2, FPN levels,
     ImageNet RGB normalisation (mean 123.675/116.28/103.53, std 58.395/57.12/57.375), test scale 1333x800.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from anti_uav_repro.config import load_config
from anti_uav_repro.models.qg_rcnn import QGRCNN
from inspect_globaltrack_ckpt import RULES, _PickleShim, map_key

HERE = Path(__file__).resolve().parent


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(HERE.parent / "configs" / "normal_ir.yaml"))
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, pickle_module=_PickleShim)
    src = ck["state_dict"]
    sd = {}
    for k, v in src.items():
        if k.endswith("num_batches_tracked"):
            continue
        t = map_key(k)
        if t is None:
            raise SystemExit(f"unmapped source key {k}")
        sd[t] = v.float().clone()
    cfg = load_config(a.config)
    model = QGRCNN({**cfg["model"], "init": "none", "frozen_bn_eps": 1e-5})
    sd["_frozen_bn_eps"] = torch.tensor(1e-5)
    model.load_state_dict(sd, strict=True)                       # raises on any missing / unexpected / shape-mismatched key
    back = model.state_dict()
    exact = all(torch.equal(back[k], sd[k]) for k in sd)
    meta = {"source": Path(a.ckpt).name, "source_sha256": sha256(a.ckpt), "source_mmdet_version": ck["meta"].get("mmdet_version"),
            "source_epoch": ck["meta"].get("epoch"), "source_iter": ck["meta"].get("iter"), "n_tensors": len(sd),
            "frozen_bn_eps": 1e-5, "round_trip_exact": exact, "mapping_rules": RULES,
            "convention_differences": ["C1 anchor centres (stride-1)/2 vs grid points", "C2 legacy +1 box convention"],
            "licence": "none stated in huanglianghua/GlobalTrack; research reproduction only; not redistributed"}
    tmp = Path(a.out).with_suffix(".tmp")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, tmp)
    tmp.replace(a.out)
    meta["out_sha256"] = sha256(a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps({k: v for k, v in meta.items() if k != "mapping_rules"}, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "mapping_rules"}, indent=1))


if __name__ == "__main__":
    main()
