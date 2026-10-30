#!/usr/bin/env python
"""Track A gap A2, step 3: functional check of the converted GlobalTrack weights (EXPLORATORY; TRAIN split only).

Zero-shot tracking (no fine-tuning) on a fixed, evenly spaced subset of Anti-UAV TRAIN sequences of one modality, with
  (a) the converted GlobalTrack checkpoint (COCO + GOT-10k + LaSOT), and
  (b) the COCO Faster R-CNN init with untrained modulators (the IR runs' starting point) as the "no tracking knowledge" control.
Pass criterion for using (a) as the visible-row init: mSA(a) > mSA(b) + 10 on this subset (a converted tracker that works must
beat an untrained one by a wide margin; a broken conversion -- wrong anchors / boxes / classes -- would not).
Never touches VAL or TEST.
    python experiments/anti_uav/scripts/zeroshot_globaltrack.py --converted <pt> --modality RGB --n 12 --out <json>
"""
import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from anti_uav_repro.config import load_config
from anti_uav_repro.data.antiuav import load_sequences
from anti_uav_repro.data.transforms import TestTransform
from anti_uav_repro.engine.track import GlobalTrackTracker
from anti_uav_repro.eval.evaluate import evaluate_prediction_dicts
from anti_uav_repro.paths import resolve

HERE = Path(__file__).resolve().parent


def run(model, seqs, device, scale, pad):
    model.to(device).eval()
    tr = GlobalTrackTracker(model, device, TestTransform(scale, pad), amp="none")
    return {s.name: tr.track_sequence(s) for s in seqs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--converted", required=True); ap.add_argument("--modality", default="RGB", choices=["IR", "RGB"])
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from anti_uav_repro.models.qg_rcnn import QGRCNN
    cfg = load_config(HERE.parent / "configs" / "normal_ir.yaml")
    d = cfg["data"]
    seqs = load_sequences(resolve(d["root"]), resolve(d["frames_root"]), "train", a.modality, purpose="tune")
    step = max(1, len(seqs) // a.n)
    seqs = seqs[::step][:a.n]
    device = torch.device("cuda")
    scale, pad = tuple(d["transforms"]["scale"]), int(d["transforms"]["pad_divisor"])
    out = {"split": "train", "modality": a.modality, "sequences": [s.name for s in seqs], "class": "EXPLORATORY"}
    t0 = time.time()
    m_gt = QGRCNN({**cfg["model"], "init": "none", "frozen_bn_eps": 1e-5})
    ck = torch.load(a.converted, map_location="cpu", weights_only=False)
    m_gt.load_state_dict(ck["model"], strict=True)
    out["converted"] = {"sha256": ck["meta"].get("out_sha256"), **evaluate_prediction_dicts(run(m_gt, seqs, device, scale, pad), seqs)["aggregate"]}
    m_coco = QGRCNN(dict(cfg["model"]))                         # COCO init, modulators at their init (no tracking training)
    out["coco_init_control"] = evaluate_prediction_dicts(run(m_coco, seqs, device, scale, pad), seqs)["aggregate"]
    out["pass"] = out["converted"]["mSA_percent"] > out["coco_init_control"]["mSA_percent"] + 10
    out["seconds"] = round(time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
