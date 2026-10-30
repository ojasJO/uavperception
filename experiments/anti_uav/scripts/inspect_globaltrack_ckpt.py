#!/usr/bin/env python
"""Track A gap A2 (visible / large-scale rows), step 1: inventory of the official GlobalTrack checkpoint and name/shape coverage
of a key mapping onto our torchvision QG-RCNN. Read-only; converts nothing yet.

    python experiments/anti_uav/scripts/inspect_globaltrack_ckpt.py --ckpt <qg_rcnn_r50_fpn_coco_got10k_lasot.pth> --out <json>

Checkpoint: GlobalTrack (Huang et al., AAAI 2020) pretrained on COCO + GOT-10k + LaSOT, Google Drive id
1ouTJDS5NACLukd810uiKCvmLxJfga48x (repository huanglianghua/GlobalTrack; no licence stated -> research reproduction only,
never redistributed). mmcv objects in its `meta` are unpickled as inert placeholders (mmcv is not installed).
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from anti_uav_repro.config import load_config
from anti_uav_repro.models.qg_rcnn import QGRCNN

HERE = Path(__file__).resolve().parent


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.state = s


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub


class _PickleShim:
    Unpickler = _TolerantUnpickler
    load = staticmethod(lambda f, **k: _TolerantUnpickler(f, **k).load())


# mmdetection (v1) name -> our QGRCNN name. Applied in order; first match wins.
RULES = [
    (r"^backbone\.(conv1|bn1|layer[1-4])\.(.*)$", r"backbone.body.\1.\2"),
    (r"^neck\.lateral_convs\.(\d)\.conv\.(weight|bias)$", r"backbone.fpn.inner_blocks.\1.0.\2"),
    (r"^neck\.fpn_convs\.(\d)\.conv\.(weight|bias)$", r"backbone.fpn.layer_blocks.\1.0.\2"),
    (r"^rpn_head\.rpn_conv\.(weight|bias)$", r"rpn.head.conv.0.0.\1"),
    (r"^rpn_head\.rpn_cls\.(weight|bias)$", r"rpn.head.cls_logits.\1"),
    (r"^rpn_head\.rpn_reg\.(weight|bias)$", r"rpn.head.bbox_pred.\1"),
    (r"^bbox_head\.shared_fcs\.0\.(weight|bias)$", r"roi_heads.box_head.fc6.\1"),
    (r"^bbox_head\.shared_fcs\.1\.(weight|bias)$", r"roi_heads.box_head.fc7.\1"),
    (r"^bbox_head\.fc_cls\.(weight|bias)$", r"roi_heads.box_predictor.cls_score.\1"),
    (r"^bbox_head\.fc_reg\.(weight|bias)$", r"roi_heads.box_predictor.bbox_pred.\1"),
    (r"^rpn_modulator\.(.*)$", r"rpn_modulator.\1"),
    (r"^rcnn_modulator\.(.*)$", r"rcnn_modulator.\1"),
]


def map_key(k):
    for pat, rep in RULES:
        if re.match(pat, k):
            return re.sub(pat, rep, k)
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(HERE.parent / "configs" / "normal_ir.yaml"))
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, pickle_module=_PickleShim)
    sd = ck.get("state_dict", ck)
    meta = ck.get("meta", {})
    cfg = load_config(a.config)
    ours = QGRCNN({**cfg["model"], "init": "none"}).state_dict()
    mapped, unmapped_src, shape_bad = {}, [], []
    for k, v in sd.items():
        t = map_key(k)
        if t is None or t not in ours:
            unmapped_src.append([k, list(v.shape)])
        elif tuple(ours[t].shape) != tuple(v.shape):
            shape_bad.append([k, t, list(v.shape), list(ours[t].shape)])
        else:
            mapped[k] = t
    ours_missing = [[k, list(v.shape)] for k, v in ours.items() if k not in set(mapped.values())
                    and not k.endswith("num_batches_tracked")]
    rep = {"ckpt": Path(a.ckpt).name, "n_src": len(sd), "n_ours": len(ours), "mapped": len(mapped),
           "unmapped_src": unmapped_src, "shape_mismatch": shape_bad, "ours_without_source": ours_missing,
           "meta_keys": sorted(map(str, meta)) if isinstance(meta, dict) else str(type(meta)),
           "src_key_prefixes": sorted({k.split(".")[0] for k in sd})}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items()}, indent=1))
    for k in ("unmapped_src", "shape_mismatch", "ours_without_source"):
        print(k, rep[k][:12])


if __name__ == "__main__":
    sys.exit(main())
