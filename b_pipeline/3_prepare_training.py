#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_prepare_training.py  --  Phase B, Step 5: one-time training preparation
=========================================================================
Everything expensive that must happen *before* epoch 1 and must never happen
*inside* it. Run once after `1_standardize_annotations.py`; re-run only if the
manifest changes.

It produces three artifacts in `b_pipeline/data_unified/`:

  label_cache.npz      Every YOLO label file in the manifest, parsed once, into
                       (offsets[N+1], boxes[M,5]) indexed by MANIFEST ROW. The
                       dataset then does three image reads and zero label reads
                       per sample -- a 25% cut in random I/O.
                       (00_TRAINING_ROADMAP.md §3.5: the label scan is a 393k-file
                       walk; paying it inside the first epoch is 30 min of dead
                       GPU time.)

  splits.json          The sequence-level train/val/test specification, incl. the
                       CST validation holdout carved out of cst[train] with a
                       fixed seed. CST ships no official val split, and cst[test]
                       must stay pristine -- given Anti-UAV's 90% centre bias it
                       is the only honest measure of off-gimbal generalisation.
                       (§2.3)

  sample_hardness.csv  Per-label hardness tuple (n_boxes, bird, micro, small) in
                       the legacy `imbalance_sampler` cache format, so the
                       weighted sampler starts instantly.

A hard leakage assertion (train seq_keys  n  val seq_keys == 0) runs before
anything is written. This is roadmap test T4, and it is the difference between a
meaningful mAP and a meaningless one.
"""

import sys
import json
import csv
import time
import argparse
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UNIFIED = HERE / "data_unified"
MANIFEST = UNIFIED / "manifest.csv"
LOGS = HERE / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

LABEL_CACHE = UNIFIED / "label_cache.npz"
SPLITS_JSON = UNIFIED / "splits.json"
HARDNESS_CSV = UNIFIED / "sample_hardness.csv"

MICRO_THR = 0.0003
SMALL_THR = 0.01
CLASS_BIRD = 1

CST_VAL_FRACTION = 0.16          # ~14 of 84 CST train sequences
CST_HOLDOUT_SEED = 1337

_log = []


def log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    _log.append(line)


# --------------------------------------------------------------------------- #
#  1. Label cache
# --------------------------------------------------------------------------- #
def _parse_label(path):
    """Parse one YOLO .txt -> list of (cls,xc,yc,w,h). Missing/empty -> []."""
    out = []
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return out
    if not data:
        return out
    for ln in data.split(b"\n"):
        a = ln.split()
        if len(a) >= 5:
            try:
                out.append((float(a[0]), float(a[1]), float(a[2]),
                            float(a[3]), float(a[4])))
            except ValueError:
                continue
    return out


def build_label_cache(label_paths, workers=16):
    """Read every label file once (threaded; these are I/O-bound tiny reads)."""
    n = len(label_paths)
    t0 = time.time()
    results = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_parse_label, label_paths, chunksize=512)):
            results[i] = res
            if (i + 1) % 100000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                log(f"  [labels] {i+1:,}/{n:,}  ({rate:,.0f} files/s)")
    offsets = np.zeros(n + 1, dtype=np.int64)
    counts = np.fromiter((len(r) for r in results), dtype=np.int64, count=n)
    np.cumsum(counts, out=offsets[1:])
    boxes = np.empty((int(offsets[-1]), 5), dtype=np.float32)
    for i, r in enumerate(results):
        if r:
            boxes[offsets[i]:offsets[i + 1]] = r
    log(f"  [labels] parsed {n:,} files -> {len(boxes):,} boxes "
        f"in {time.time()-t0:,.1f}s")
    return offsets, boxes


# --------------------------------------------------------------------------- #
#  2. Hardness (vectorised, straight off the cache -- no second file walk)
# --------------------------------------------------------------------------- #
def compute_hardness(offsets, boxes):
    """-> (n_boxes, has_bird, has_micro, has_small) arrays, one row per label."""
    n = len(offsets) - 1
    counts = np.diff(offsets).astype(np.int32)
    n_box = counts
    has_bird = np.zeros(n, dtype=np.int8)
    has_micro = np.zeros(n, dtype=np.int8)
    has_small = np.zeros(n, dtype=np.int8)
    if len(boxes):
        area = boxes[:, 3] * boxes[:, 4]
        is_bird = boxes[:, 0].astype(np.int32) == CLASS_BIRD
        # np.add.reduceat needs non-empty groups; mask them out afterwards.
        nz = np.flatnonzero(counts > 0)
        starts = offsets[nz]
        bird_any = np.maximum.reduceat(is_bird.astype(np.int8), starts)
        min_area = np.minimum.reduceat(area, starts)
        has_bird[nz] = bird_any
        has_micro[nz] = (min_area < MICRO_THR).astype(np.int8)
        has_small[nz] = (min_area < SMALL_THR).astype(np.int8)
    return n_box, has_bird, has_micro, has_small


def category_of(n, bird, micro, small):
    if bird:
        return "bird"
    if micro:
        return "micro"
    if small:
        return "small"
    if n > 0:
        return "standard"
    return "empty"


# --------------------------------------------------------------------------- #
#  3. Sequence-level splits
# --------------------------------------------------------------------------- #
def build_splits(df):
    """Return the split spec dict + the CST validation holdout sequence list."""
    seq = (df.groupby(["dataset", "seq_key", "split"], as_index=False)
             .agg(frames=("frame_idx", "size"),
                  materialized=("materialized", "sum")))

    # -- sanity: no seq_key may span two splits --------------------------- #
    spans = seq.groupby("seq_key")["split"].nunique()
    bad = spans[spans > 1]
    if len(bad):
        raise SystemExit(f"[ABORT] {len(bad)} seq_keys span multiple splits: "
                         f"{list(bad.index[:5])}")

    # -- CST has no official val split -> carve one out of cst[train] ----- #
    cst_train = seq[(seq.dataset == "cst") & (seq.split == "train")]
    cst_train = cst_train.sort_values("seq_key").reset_index(drop=True)
    rng = np.random.default_rng(CST_HOLDOUT_SEED)
    order = rng.permutation(len(cst_train))
    target = CST_VAL_FRACTION * cst_train["materialized"].sum()
    holdout, acc = [], 0
    for j in order:
        if acc >= target:
            break
        holdout.append(str(cst_train.at[int(j), "seq_key"]))
        acc += int(cst_train.at[int(j), "materialized"])
    holdout = sorted(holdout)

    spec = {
        "generated": datetime.datetime.now().isoformat(),
        "seed": CST_HOLDOUT_SEED,
        "rationale": (
            "Sequence-level splits. Consecutive video frames are near-identical "
            "(median inter-frame motion 5.10 px Anti-UAV / 0.64 px CST), so a "
            "frame-level split leaks. CST ships no val split; a fixed-seed "
            "holdout is carved from cst[train]. cst[test] is NEVER used for "
            "model selection -- it is the cross-domain generalisation probe."),
        "train": {"anti_uav": ["train"], "cst": ["train"], "det_fly": ["train"]},
        "val": {"anti_uav": ["val"], "cst": ["train"], "det_fly": ["val"]},
        "test": {"anti_uav": ["test"], "cst": ["test"], "det_fly": ["test"]},
        "cst_val_holdout_seqs": holdout,
        "notes": ("`val.cst` reads the *train* split but is restricted to "
                  "cst_val_holdout_seqs; `train.cst` excludes exactly those "
                  "sequences. Apply both filters or the split leaks."),
    }

    # -- realised counts, for the record ---------------------------------- #
    counts = {}
    for name, sp in (("train", spec["train"]), ("val", spec["val"]),
                     ("test", spec["test"])):
        rows = []
        for dk, sps in sp.items():
            g = seq[(seq.dataset == dk) & (seq.split.isin(sps))]
            if dk == "cst":
                g = (g[g.seq_key.isin(holdout)] if name == "val"
                     else g[~g.seq_key.isin(holdout)] if name == "train" else g)
            rows.append((dk, len(g), int(g.frames.sum()), int(g.materialized.sum())))
        counts[name] = {dk: {"sequences": ns, "frames": nf, "materialized": nm}
                        for dk, ns, nf, nm in rows}
    spec["counts"] = counts
    return spec, seq


def seq_keys_for(seq_table, spec, phase):
    """Resolve a phase of the spec to the concrete set of seq_keys."""
    keys = set()
    holdout = set(spec["cst_val_holdout_seqs"])
    for dk, sps in spec[phase].items():
        g = seq_table[(seq_table.dataset == dk) & (seq_table.split.isin(sps))]
        k = set(g.seq_key.astype(str))
        if dk == "cst":
            if phase == "val":
                k &= holdout
            elif phase == "train":
                k -= holdout
        keys |= k
    return keys


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16,
                    help="threads for the one-time label scan")
    ap.add_argument("--force", action="store_true",
                    help="rebuild label_cache.npz even if it is present")
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"[ABORT] {MANIFEST} not found — run "
                         f"1_standardize_annotations.py first.")

    log("=" * 78)
    log("Phase B / Step 5: training preparation (label cache · splits · hardness)")
    log("=" * 78)

    df = pd.read_csv(MANIFEST, low_memory=False,
                     dtype={"image_path": str, "label_path": str, "seq_key": str,
                            "split": str, "modality": str})
    log(f"manifest rows: {len(df):,}")

    # ---- 1. label cache -------------------------------------------------- #
    if LABEL_CACHE.exists() and not args.force:
        z = np.load(LABEL_CACHE)
        offsets, boxes = z["offsets"].astype(np.int64), z["boxes"].astype(np.float32)
        if len(offsets) - 1 != len(df):
            log("  [labels] cache is stale (row count mismatch) -> rebuilding")
            offsets, boxes = build_label_cache(df["label_path"].tolist(), args.workers)
            np.savez_compressed(LABEL_CACHE, offsets=offsets, boxes=boxes)
        else:
            log(f"  [labels] reusing cache: {len(boxes):,} boxes")
    else:
        offsets, boxes = build_label_cache(df["label_path"].tolist(), args.workers)
        np.savez_compressed(LABEL_CACHE, offsets=offsets, boxes=boxes)
        log(f"  [labels] wrote {LABEL_CACHE.name} "
            f"({LABEL_CACHE.stat().st_size/1024**2:.1f} MB)")

    # ---- 1b. integrity audit (roadmap T3) -------------------------------- #
    if len(boxes):
        oob = int(((boxes[:, 1:] < 0.0) | (boxes[:, 1:] > 1.0)).any(1).sum())
        degen = int(((boxes[:, 3] <= 0) | (boxes[:, 4] <= 0)).sum())
        log(f"  [audit] {len(boxes):,} boxes | out-of-bounds={oob} | degenerate={degen}")
        if oob or degen:
            raise SystemExit("[ABORT] label integrity violated — fix Phase B first.")

    # ---- 2. hardness ----------------------------------------------------- #
    n_box, bird, micro, small = compute_hardness(offsets, boxes)
    cats = np.array([category_of(*t) for t in zip(n_box, bird, micro, small)])
    mat = df["materialized"].to_numpy() == 1
    from collections import Counter
    mix = Counter(cats[mat])
    tot = max(int(mat.sum()), 1)
    log("  [hardness] materialised-frame category mix: " +
        ", ".join(f"{k}={v/tot*100:.2f}%" for k, v in sorted(mix.items())))
    with open(HARDNESS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for p, a, b, c, d in zip(df["label_path"].tolist(), n_box, bird, micro, small):
            w.writerow([p, int(a), int(b), int(c), int(d)])
    log(f"  [hardness] wrote {HARDNESS_CSV.name} ({len(df):,} rows)")
    np.savez_compressed(UNIFIED / "hardness.npz", n_box=n_box, bird=bird,
                        micro=micro, small=small)

    # ---- 3. splits ------------------------------------------------------- #
    spec, seq_table = build_splits(df)
    tr = seq_keys_for(seq_table, spec, "train")
    va = seq_keys_for(seq_table, spec, "val")
    te = seq_keys_for(seq_table, spec, "test")

    # ---- T4: THE mandatory leakage assertion ----------------------------- #
    assert not (tr & va), f"SPLIT LEAKAGE train n val: {sorted(tr & va)[:5]}"
    assert not (tr & te), f"SPLIT LEAKAGE train n test: {sorted(tr & te)[:5]}"
    assert not (va & te), f"SPLIT LEAKAGE val n test: {sorted(va & te)[:5]}"
    log(f"  [T4] PASS — train({len(tr):,}) / val({len(va):,}) / test({len(te):,}) "
        f"sequence sets are pairwise disjoint")

    SPLITS_JSON.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    log(f"  [splits] wrote {SPLITS_JSON.name} "
        f"(CST val holdout = {len(spec['cst_val_holdout_seqs'])} sequences)")
    for phase in ("train", "val", "test"):
        for dk, c in spec["counts"][phase].items():
            log(f"    {phase:<5s} {dk:<9s} seqs={c['sequences']:>6,} "
                f"frames={c['frames']:>9,} images={c['materialized']:>9,}")

    (LOGS / "3_prepare_training.log").write_text("\n".join(_log), encoding="utf-8")
    log("Preparation complete.")


if __name__ == "__main__":
    main()
