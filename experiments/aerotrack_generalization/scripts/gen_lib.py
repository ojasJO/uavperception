"""Shared definitions for the generalization series: configurations, initialisation, fold samples, metrics.

The recipe is the repair grid's (`src/model/train.py` STAGES["baseline"], 20 epochs x 25,000 samples, batch 32 x acc 2); a
configuration changes exactly one factor relative to `stock`, and every configuration starts from the SAME function
(COCO YOLO11n on frame t) thanks to the function-preserving initialisation (verified in reports/diag_training.json)."""
import hashlib
import json
from collections import Counter

import _series as S
import numpy as np
import torch

from stacked_dataset import build_split_samples
from train import MANIFEST, PRETRAINED, SPLITS, VAL_CAPS, subsample

import evaluation as E

CONFIGS = {
    # name: layout (model yaml), temporal gap (0 = t,t,t), question
    "stock": {"layout": "stride", "gap": 0, "question": "reference: stock YOLO11n layout, single frame, COCO"},
    "spd_fp": {"layout": "spd", "gap": 0, "question": "H1: SPD-Conv once it starts from the pretrained function"},
    "temporal_fp_g1": {"layout": "stride", "gap": 1, "question": "H1b: the original t-1,t,t+1 stack with a centre-frame init"},
    "temporal_fp_g4": {"layout": "stride", "gap": 4, "question": "H2: a motion-matched gap (g=4 from CST's 0.53 px/frame)"},
}
YAML = {"stride": "yolo11-stride.yaml", "spd": "yolo11-spd.yaml"}
DOWN = ("0", "1", "3", "5", "7")
RMAP = {0: (1, 0), 1: (0, 1), 2: (1, 1)}               # stride tap -> (sub-pixel phase, SPD tap)
BLK = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}     # (row phase, col phase) -> SPDConv.space_to_depth block
DEV_CAP = 5000                                         # images per dataset in a held-out fold's dev set (evenly over sequences)


def load_folds():
    f = json.loads((S.SERIES / "folds.json").read_text(encoding="utf-8"))
    return f


def run_name(config, fold, epochs, suffix=""):
    return f"gen_{config}_{'full' if fold == 'full' else f'f{fold}'}_e{epochs}{suffix}"


# ------------------------------------------------------------------------------------------------ initialisation
def spd_from_stride(W):
    """[c2, c1, 3, 3] stride-2 conv kernel -> [c2, 4*c1, 3, 3] SPDConv kernel computing the same function."""
    c2, c1 = W.shape[:2]
    V = torch.zeros(c2, 4 * c1, 3, 3, dtype=W.dtype)
    for a in range(3):
        rp, c = RMAP[a]
        for b in range(3):
            cp, d = RMAP[b]
            k = BLK[(rp, cp)]
            V[:, k * c1:(k + 1) * c1, c, d] = W[:, :, a, b]
    return V


def coco_state():
    import ultralytics.nn.tasks as UT
    out = UT.load_checkpoint(str(PRETRAINED))
    return (out[0] if isinstance(out, tuple) else out).float().state_dict()


def apply_fp_init(model, layout):
    """After `transfer_pretrained`: COCO stem on the centre frame (channels 3:6), zeros on t-1 / t+1; for the SPD layout each
    SPDConv := the pretrained stride-2 conv mapped by `spd_from_stride`, with that layer's BatchNorm. Returns a check dict."""
    coco = coco_state()
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    stem = torch.zeros(coco["model.0.conv.weight"].shape[0], 9, 3, 3)
    stem[:, 3:6] = coco["model.0.conv.weight"]
    if layout == "stride":
        sd["model.0.conv.weight"] = stem
    else:
        for i in DOWN:
            W = stem if i == "0" else coco[f"model.{i}.conv.weight"]
            sd[f"model.{i}.conv.conv.weight"] = spd_from_stride(W)
            for t in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                sd[f"model.{i}.conv.bn.{t}"] = coco[f"model.{i}.bn.{t}"].clone()
    model.load_state_dict(sd)
    return equivalence_check(model, layout)


def equivalence_check(model, layout):
    """Runtime proof on this very model: it equals a stride model carrying the same weights (SPD layout), and it ignores
    t-1 / t+1 at initialisation (both layouts)."""
    from aerotrack_trainer import build_model
    was = model.training
    model.eval()
    x = torch.rand(1, 9, 256, 256)
    t = x[:, 3:6]
    dev = next(model.parameters()).device
    with torch.no_grad():
        y = model(x.to(dev))[0].cpu()
        y_ttt = model(torch.cat([t, t, t], 1).to(dev))[0].cpu()
        out = {"ignores_neighbours_max_abs_diff": float((y - y_ttt).abs().max())}
        if layout == "spd":
            ref = build_model(S.REPO / "src/model" / YAML["stride"], ch=9, nc=2)
            rs = ref.state_dict()
            ms = model.state_dict()
            for k in rs:
                if k in ms and ms[k].shape == rs[k].shape:
                    rs[k] = ms[k].detach().cpu().clone()
            for i in DOWN:
                # invert the mapping for the check: read the stride kernel back from the SPD kernel's populated slots
                V = ms[f"model.{i}.conv.conv.weight"].detach().cpu()
                c1 = V.shape[1] // 4
                W = torch.zeros(V.shape[0], c1, 3, 3)
                for a in range(3):
                    rp, c = RMAP[a]
                    for b in range(3):
                        cp, d = RMAP[b]
                        k = BLK[(rp, cp)]
                        W[:, :, a, b] = V[:, k * c1:(k + 1) * c1, c, d]
                rs[f"model.{i}.conv.weight"] = W
                for tt in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    rs[f"model.{i}.bn.{tt}"] = ms[f"model.{i}.conv.bn.{tt}"].detach().cpu().clone()
            ref.load_state_dict(rs)
            ref.eval()
            yr = ref(x)[0]
            out["spd_vs_stride_max_abs_diff"] = float((y - yr).abs().max())
            out["output_max_abs"] = float(yr.abs().max())
    model.train(was)
    tol = 1e-5 * max(out.get("output_max_abs", 1.0), 1.0) + 1e-2
    out["ok"] = out["ignores_neighbours_max_abs_diff"] <= 1e-3 and out.get("spd_vs_stride_max_abs_diff", 0.0) <= tol
    return out


# ------------------------------------------------------------------------------------------------ samples
def _with_gap(samples, gap):
    if gap == 0:
        return [dict(s, prev_path=s["cur_path"], next_path=s["cur_path"]) for s in samples]
    return samples


def fold_samples(folds, fold_ids, gap):
    """TRAIN-split samples of the given folds with t-g / t / t+g neighbours (gap 0: t,t,t)."""
    keep = {sk for sk, k in folds["seq_fold"].items() if k in set(fold_ids)}
    base = build_split_samples(MANIFEST, SPLITS, "train", temporal_gap=max(gap, 1))
    return _with_gap([s for s in base if s["seq_key"] in keep], gap)


def set_identity(samples, split, cap):
    ident = sorted((s["dataset"], s["seq_key"], int(s["frame_idx"])) for s in samples)
    return {"split": split, "cap": cap, "n_images": len(samples), "per_dataset": dict(Counter(s["dataset"] for s in samples)),
            "n_sequences": len({s["seq_key"] for s in samples}),
            "sha256": hashlib.sha256(json.dumps(ident, separators=(",", ":")).encode()).hexdigest()}


def dev_set(folds, fold, gap, cap=DEV_CAP):
    """The held-out fold as an evaluation set: every dataset capped at `cap` images, evenly over its sequences."""
    ss = subsample(fold_samples(folds, [fold], gap), {d: cap for d in ("anti_uav", "cst", "det_fly")})
    return ss, set_identity(ss, f"dev_fold{fold}", cap)


def official_set(split, gap, cap=5000):
    """The registered VAL (or, final step only, TEST) image set, built with this configuration's input."""
    base = build_split_samples(MANIFEST, SPLITS, split, temporal_gap=max(gap, 1))
    ss = _with_gap(subsample(base, {d: cap for d in {s["dataset"] for s in base}}), gap)
    return ss, set_identity(ss, split, cap)


def monitor_val(folds, fold, gap):
    """Per-epoch monitoring set of a fold run (held-out fold, capped like the recipe's VAL_CAPS). Checkpoint = last.pt,
    so this set selects nothing."""
    if fold == "full":
        base = build_split_samples(MANIFEST, SPLITS, "val", temporal_gap=max(gap, 1))
        return _with_gap(subsample(base, VAL_CAPS), gap)
    return subsample(fold_samples(folds, [fold], gap), VAL_CAPS)


# ------------------------------------------------------------------------------------------------ metrics
def drone_metrics(recs):
    """Drone-class AP50 overall and per dataset, CST drone-micro AP50, plus the repair series' four registered metrics."""
    def ap(sub, **kw):
        r = E.score(sub, **kw) if sub else None
        return None if r is None else r["map50"]
    by = {d: [r for r in recs if r["dataset"] == d] for d in ("anti_uav", "cst", "det_fly")}
    out = {"drone_AP50": ap(recs, classes=(0,)),
           **{f"{d}_drone_AP50": ap(v, classes=(0,)) for d, v in by.items()},
           "cst_drone_micro_AP50": ap(by["cst"], size_bin="micro", classes=(0,)),
           # registered metrics of the repair series (two-class means where the repair defined them so)
           "reg_micro_AP50": ap(recs, size_bin="micro"), "reg_cst_AP50": ap(by["cst"]),
           "reg_cst_micro_AP50": ap(by["cst"], size_bin="micro"), "mAP50": ap(recs)}
    out["n_images"] = len(recs)
    return out
