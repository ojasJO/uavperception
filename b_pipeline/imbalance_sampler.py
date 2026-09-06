#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imbalance_sampler.py  --  Hardness-aware weighted sampling for AeroTrack-Net
============================================================================
The unified corpus is severely imbalanced along the axes that actually matter
for anti-UAV detection:

  * easy, large, centre-frame RGB drones      -> abundant (Anti-UAV visible)
  * micro drones (<0.03 % of frame area)      -> the make-or-break tail
  * Bird HARD NEGATIVES (Det-Fly class 1)     -> rare but critical to suppress
                                                 false positives

Uniform sampling lets the backbone coast on the easy majority. This module scans
each frame's target label and assigns a per-sample weight so a
`WeightedRandomSampler` draws the hard cases far more often, forcing the YOLO-SPD
backbone to learn the edge cases instead of overfitting easy targets.

Priority (a frame takes the weight of the hardest thing it contains):

    bird hard-negative  >  micro (<0.03%)  >  small (<1%)  >  standard  >  empty

The scan is cached to disk (keyed by label path) so it is a one-time cost.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

MICRO_THR = 0.0003        # 0.03 % of frame area
SMALL_THR = 0.01          # 1 % of frame area
CLASS_BIRD = 1            # unified scheme: 0=uav_drone, 1=bird_hard_negative

# Default relative sampling weights (tune freely). Higher => sampled more often.
DEFAULT_WEIGHTS = {
    "bird": 8.0,          # hard negatives — suppress false positives
    "micro": 6.0,         # sub-0.03% drones — the hardest positives
    "small": 3.0,         # 0.03%–1% drones
    "standard": 1.0,      # easy, large drones
    "empty": 0.5,         # keep some pure-background frames (real negatives)
}
CATEGORIES = ["bird", "micro", "small", "standard", "empty"]

_CACHE = Path(__file__).resolve().parent / "data_unified" / "sample_hardness.csv"


# --------------------------------------------------------------------------- #
def scan_label(label_path):
    """Scan one YOLO .txt -> (n_boxes, has_bird, has_micro, has_small)."""
    n = 0
    has_bird = False
    min_area = 1.0
    p = Path(label_path)
    if p.exists() and p.stat().st_size > 0:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                a = ln.split()
                if len(a) < 5:
                    continue
                n += 1
                cls = int(float(a[0]))
                area = float(a[3]) * float(a[4])          # w_norm * h_norm
                if cls == CLASS_BIRD:
                    has_bird = True
                if area < min_area:
                    min_area = area
    has_micro = n > 0 and min_area < MICRO_THR
    has_small = n > 0 and min_area < SMALL_THR
    return n, int(has_bird), int(has_micro), int(has_small)


def category_of(n, has_bird, has_micro, has_small):
    if has_bird:
        return "bird"
    if has_micro:
        return "micro"
    if has_small:
        return "small"
    if n > 0:
        return "standard"
    return "empty"


def hardness_from_index(index):
    """Vectorised hardness straight off a `SampleIndex`'s in-memory label cache.

    The label files were already parsed once by `3_prepare_training.py`, so this
    is pure arithmetic -- no file walk at all. On the full 394k-sample training
    split the legacy path below is a 30-minute disk scan *inside epoch 1*
    (00_TRAINING_ROADMAP.md §3.5); this is milliseconds."""
    n = len(index)
    counts = index.box_counts()
    out = np.zeros((n, 4), dtype=np.int32)
    out[:, 0] = counts
    if len(index.lbl_box):
        area = index.lbl_box[:, 3] * index.lbl_box[:, 4]
        is_bird = (index.lbl_box[:, 0].astype(np.int32) == CLASS_BIRD).astype(np.int8)
        nz = np.flatnonzero(counts > 0)
        starts = index.lbl_off[nz]
        out[nz, 1] = np.maximum.reduceat(is_bird, starts)
        min_area = np.minimum.reduceat(area, starts)
        out[nz, 2] = (min_area < MICRO_THR).astype(np.int32)
        out[nz, 3] = (min_area < SMALL_THR).astype(np.int32)
    return out


def compute_hardness(samples, cache_path=_CACHE, use_cache=True, verbose=True):
    """Return an (N,4) int array [n_boxes, bird, micro, small] per sample.
    Results are cached to `cache_path`, keyed by label_path, so repeat runs and
    the training entry point pay the label-scan cost only once."""
    if hasattr(samples, "lbl_off"):                 # SampleIndex -> fast path
        h = hardness_from_index(samples)
        if verbose:
            print(f"[hardness] vectorised from label cache: {len(h):,} samples")
        return h
    cache = {}
    cp = Path(cache_path) if cache_path else None
    if use_cache and cp and cp.exists():
        with open(cp, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) == 5:
                    cache[row[0]] = tuple(int(x) for x in row[1:])
        if verbose:
            print(f"[hardness] loaded cache: {len(cache):,} entries")

    out = np.zeros((len(samples), 4), dtype=np.int32)
    new = 0
    for i, s in enumerate(samples):
        lp = s["label_path"]
        h = cache.get(lp)
        if h is None:
            h = scan_label(lp)
            cache[lp] = h
            new += 1
        out[i] = h
        if verbose and (i + 1) % 100000 == 0:
            print(f"[hardness] scanned {i+1:,}/{len(samples):,}")

    if use_cache and cp and new:
        cp.parent.mkdir(parents=True, exist_ok=True)
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for k, v in cache.items():
                w.writerow([k, *v])
        if verbose:
            print(f"[hardness] cache updated (+{new:,} new -> {len(cache):,} total)")
    return out


DEFAULT_DOMAIN_SHARES = {
    "anti_uav": 0.40,
    "cst": 0.35,
    "det_fly": 0.25,
}

FINETUNE_WEIGHTS = {
    "bird": 35.0,         # aggressive bird hard-negative suppression
    "micro": 8.0,         # hardest micro-targets
    "small": 4.0,         # small targets
    "standard": 1.0,      # standard targets
    "empty": 0.8,         # background frames to suppress hallucinations
}


def hardness_to_weights(hardness, weights=DEFAULT_WEIGHTS):
    """(N,4) hardness array -> (N,) float sampling weights + category labels."""
    w = np.empty(len(hardness), dtype=np.float64)
    cats = np.empty(len(hardness), dtype=object)
    for i, (n, bird, micro, small) in enumerate(hardness):
        c = category_of(n, bird, micro, small)
        cats[i] = c
        w[i] = weights[c]
    return w, cats


def domain_balanced_weights(samples, hardness, weights=DEFAULT_WEIGHTS,
                            domain_shares=DEFAULT_DOMAIN_SHARES):
    """Compute joint domain-balanced and hardness-aware sampling weights.

    Ensures that underrepresented datasets (e.g. Det-Fly at 1.2% of corpus)
    receive their target share (e.g. 25%) while still prioritizing hard cases
    (birds, micro-targets) within each domain.
    """
    n = len(hardness)
    w = np.empty(n, dtype=np.float64)
    cats = np.empty(n, dtype=object)

    for i, (cnt, bird, micro, small) in enumerate(hardness):
        cats[i] = category_of(cnt, bird, micro, small)

    # Extract dataset identifier per sample
    if hasattr(samples, "ds_code") and hasattr(samples, "ds_names"):
        ds_codes = samples.ds_code
        ds_names = list(samples.ds_names)
    elif isinstance(samples, (list, tuple)) and len(samples) > 0 and isinstance(samples[0], dict):
        ds_names_set = sorted({s.get("dataset", "unknown") for s in samples})
        ds_map = {name: idx for idx, name in enumerate(ds_names_set)}
        ds_codes = np.array([ds_map.get(s.get("dataset", "unknown"), 0) for s in samples], dtype=np.int16)
        ds_names = ds_names_set
    else:
        # Fallback to standard hardness if dataset metadata is absent
        for i, c in enumerate(cats):
            w[i] = weights[c]
        return w, cats

    # Apply within-domain category weighting and domain allocation
    total_target = sum(domain_shares.get(name, 1.0) for name in ds_names)
    normalized_shares = {name: domain_shares.get(name, 1.0) / total_target for name in ds_names}

    for ds_idx, ds_name in enumerate(ds_names):
        mask = (ds_codes == ds_idx)
        if not np.any(mask):
            continue
        target_share = normalized_shares.get(ds_name, 1.0 / len(ds_names))
        raw_cat_weights = np.array([weights.get(c, 1.0) for c in cats[mask]], dtype=np.float64)
        sum_raw = raw_cat_weights.sum()
        if sum_raw > 0:
            w[mask] = raw_cat_weights * (target_share / sum_raw)
        else:
            w[mask] = target_share / len(raw_cat_weights)

    sum_w = w.sum()
    if sum_w > 0:
        w /= sum_w

    return w, cats


def build_weighted_sampler(samples, weights=DEFAULT_WEIGHTS, num_samples=None,
                           replacement=True, cache_path=_CACHE, verbose=True,
                           generator=None, domain_balanced=False,
                           domain_shares=DEFAULT_DOMAIN_SHARES):
    """Return (sampler, weights_array, categories, hardness) ready for a
    DataLoader(sampler=...).

    `num_samples` is the ITERATION BUDGET (00_TRAINING_ROADMAP.md §3.2). Because
    the sampler draws with replacement, an "epoch" need not mean one pass over
    394k samples -- setting e.g. 25,000 gives 15x more frequent checkpoints, LR
    steps and validation signal while still traversing the corpus in expectation.
    That single change takes an epoch from hours to minutes."""
    hardness = compute_hardness(samples, cache_path=cache_path, verbose=verbose)
    if domain_balanced:
        w, cats = domain_balanced_weights(samples, hardness, weights, domain_shares)
    else:
        w, cats = hardness_to_weights(hardness, weights)

    kw = {"generator": generator} if generator is not None else {}
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=int(num_samples or len(samples)),
        replacement=replacement,
        **kw,
    )
    if verbose:
        from collections import Counter
        base = Counter(cats)
        exp = {c: (w[cats == c].sum() / w.sum()) for c in CATEGORIES if (cats == c).any()}
        print("[sampler] corpus category mix (uniform) :",
              {c: f"{base.get(c,0)/len(cats)*100:.2f}%" for c in CATEGORIES})
        print("[sampler] expected sampled mix (weighted):",
              {c: f"{exp.get(c,0)*100:.2f}%" for c in CATEGORIES})
        if domain_balanced and hasattr(samples, "ds_names") and hasattr(samples, "ds_code"):
            d_exp = {samples.ds_names[i]: (w[samples.ds_code == i].sum() / w.sum()) * 100
                     for i in range(len(samples.ds_names))}
            print("[sampler] domain-balanced mix (weighted):",
                  {k: f"{v:.1f}%" for k, v in d_exp.items()})
    return sampler, w, cats, hardness



# --------------------------------------------------------------------------- #
#  Self-test / demonstration (CPU-safe, no training)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import sys
    from collections import Counter

    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))
    from stacked_dataset import build_samples_from_manifest

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset-cap", type=int, default=6000,
                    help="max samples per dataset to keep the demo fast")
    ap.add_argument("--draws", type=int, default=40000)
    args = ap.parse_args()

    MANIFEST = HERE / "data_unified" / "manifest.csv"
    print("building sample list (anti_uav + cst + det_fly)...")
    all_samples = build_samples_from_manifest(
        MANIFEST, datasets=["anti_uav", "cst", "det_fly"], require_image=True)

    # cap per dataset so the label scan in this DEMO stays fast (deterministic)
    rng = np.random.default_rng(0)
    by_ds = {}
    for s in all_samples:
        by_ds.setdefault(s["dataset"], []).append(s)
    demo = []
    for ds, lst in by_ds.items():
        idx = rng.permutation(len(lst))[:args.per_dataset_cap]
        demo.extend(lst[i] for i in idx)
    print(f"demo subset: {len(demo):,} samples "
          f"({ {k: min(len(v), args.per_dataset_cap) for k, v in by_ds.items()} })")

    sampler, w, cats, hard = build_weighted_sampler(
        demo, num_samples=args.draws, cache_path=None, verbose=True)

    # empirically draw and measure the realised distribution shift
    drawn = list(iter(sampler))
    drawn_cats = Counter(cats[i] for i in drawn)
    uniform_cats = Counter(cats)
    print("\n---- REALISED SAMPLING (drew %d) ----" % args.draws)
    print(f"{'category':10s} {'corpus%':>9s} {'sampled%':>9s} {'x-fold':>7s}")
    for c in CATEGORIES:
        u = uniform_cats.get(c, 0) / len(cats) * 100
        d = drawn_cats.get(c, 0) / args.draws * 100
        fold = (d / u) if u > 0 else float("nan")
        print(f"{c:10s} {u:8.2f}% {d:8.2f}% {fold:6.2f}x")
    print("\nHard cases (bird+micro) share:",
          f"corpus={(uniform_cats.get('bird',0)+uniform_cats.get('micro',0))/len(cats)*100:.1f}%",
          f"-> sampled={(drawn_cats.get('bird',0)+drawn_cats.get('micro',0))/args.draws*100:.1f}%")
    print("Sampler OK.")
