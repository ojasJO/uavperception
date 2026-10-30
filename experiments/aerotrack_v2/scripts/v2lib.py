"""Configurations, temporal windows, evaluation sets and metrics of the AeroTrack v2 series.

Every configuration is YOLO11n (stride layout, 9-channel input) started from the SAME function -- COCO YOLO11n on the
labelled frame -- by the generalization series' verified function-preserving initialisation (gen_lib.apply_fp_init):
COCO stem on the centre channels 3:6, zeros on the side channels. A configuration differs from `stock` in the frames it
puts in the side channels only:

    stock      [t,    t, t   ]   single frame (side channels repeat t)
    sym_g4     [t-4,  t, t+4 ]   the generalization series' supported candidate; needs 4 future frames (+160 ms @ 25 fps)
    causal_g4  [t-4,  t, t-8 ]   same motion baseline, past frames only: zero look-ahead

The labelled frame is always in the centre slot (where the pretrained stem looks); the order of the two side slots is a
fixed convention per configuration, identical in training, evaluation and the deployed runtime.
"""
import hashlib
import json
from collections import Counter

import v2series as S  # noqa: F401  (pins AEROTRACK_CORPUS=data_unified_v2 before the pipeline imports below)

import evaluation as E
from stacked_dataset import build_split_samples
from train import MANIFEST, SPLITS, subsample

CONFIGS = {
    "stock": {"layout": "stride", "gap": 0, "window": "ttt"},
    "sym_g4": {"layout": "stride", "gap": 4, "window": "symmetric"},
    "causal_g4": {"layout": "stride", "gap": 4, "window": "causal"},
    # Track C experiment C2 (experiments/aerotrack_temporal/preregistration/C2_causal_windows.json): the causal gap sweep
    "causal_g1": {"layout": "stride", "gap": 1, "window": "causal"},
    "causal_g2": {"layout": "stride", "gap": 2, "window": "causal"},
    "causal_g8": {"layout": "stride", "gap": 8, "window": "causal"},
}
EVAL_CAP = 5000                       # images per dataset in an evaluation set, evenly over its sequences (repair protocol)
CST_CATEGORIES = ("building", "cn_mountains", "cn_sky", "jungle", "urban-areas", "water")


def run_name(config, seed, epochs=20):
    return f"v2_{config}_s{seed}_e{epochs}"


def window_samples(split, window, gap):
    """All samples of a split of corpus v2 with the configuration's 9-channel window."""
    if window == "ttt" or gap == 0:
        base = build_split_samples(MANIFEST, SPLITS, split, temporal_gap=1)
        return [dict(s, prev_path=s["cur_path"], next_path=s["cur_path"]) for s in base]
    a = build_split_samples(MANIFEST, SPLITS, split, temporal_gap=gap)
    if window == "symmetric":
        return a                                                   # prev = t-g, next = t+g
    if window != "causal":
        raise ValueError(window)
    b = build_split_samples(MANIFEST, SPLITS, split, temporal_gap=2 * gap)
    if [(s["seq_key"], s["frame_idx"]) for s in a] != [(s["seq_key"], s["frame_idx"]) for s in b]:
        raise RuntimeError("gap-g and gap-2g sample lists are not aligned")
    return [dict(x, next_path=y["prev_path"]) for x, y in zip(a, b)]   # [t-g, t, t-2g]: no future frame anywhere


def identity(samples, name):
    ident = sorted((s["dataset"], s["seq_key"], int(s["frame_idx"])) for s in samples)
    return {"set": name, "n_images": len(samples), "per_dataset": dict(Counter(s["dataset"] for s in samples)),
            "n_sequences": len({s["seq_key"] for s in samples}),
            "sha256": hashlib.sha256(json.dumps(ident, separators=(",", ":")).encode()).hexdigest()}


def eval_samples(split, window, gap, cap=EVAL_CAP):
    """Registered-style evaluation set: every dataset capped at `cap` images, evenly over its sequences. The FRAME set is
    the same for every configuration (subsample picks frames by position, the window only changes the side channels)."""
    ss = subsample(window_samples(split, window, gap), {d: cap for d in ("anti_uav", "cst", "det_fly")})
    return ss, identity(ss, f"v2_{split}_cap{cap}")


def cst_category(seq_key):
    """cst_<split>_<category>_<n> -> category."""
    rest = seq_key.split("_", 2)[2]
    return rest.rsplit("_", 1)[0]


def metrics(recs):
    """Drone AP50 overall / per dataset / CST micro / CST per scene category, + mAP50 (two classes)."""
    def ap(sub, **kw):
        r = E.score(sub, **kw) if sub else None
        return None if r is None else r["map50"]
    by = {d: [r for r in recs if r["dataset"] == d] for d in ("anti_uav", "cst", "det_fly")}
    out = {"drone_AP50": ap(recs, classes=(0,)),
           **{f"{d}_drone_AP50": ap(v, classes=(0,)) for d, v in by.items()},
           "cst_drone_micro_AP50": ap(by["cst"], size_bin="micro", classes=(0,)),
           "mAP50": ap(recs)}
    for c in CST_CATEGORIES:
        sub = [r for r in by["cst"] if cst_category(r["seq_key"]) == c]
        out[f"cst_{c}_drone_AP50"] = ap(sub, classes=(0,)) if sub else None
        out[f"cst_{c}_n_images"] = len(sub)
    out["n_images"] = len(recs)
    return out
