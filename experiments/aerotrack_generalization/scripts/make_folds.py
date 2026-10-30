#!/usr/bin/env python
"""Grouped development folds carved from the TRAIN split only (the official VAL and TEST are not used).

    venv\\Scripts\\python.exe experiments\\aerotrack_generalization\\scripts\\make_folds.py

Every held-out fold is novel at the level that matters for generalisation (D1):
  * Anti-UAV  — grouped by recording SESSION (date_time of the clip name; IR and RGB clips of a session stay together),
  * CST       — grouped by SEQUENCE, stratified by scene category (each category spread over the folds),
  * Det-Fly   — grouped by FLIGHT (leading digits of the original frame id), so no adjacent frames straddle folds.
Assignment: groups sorted by size (seeded tie-break), each placed on the fold with the fewest frames of that dataset
(within the category for CST). Written once to ../folds.json with a hash; refuses to overwrite a different assignment."""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import _series as S
import numpy as np

from stacked_dataset import build_split_samples
from train import MANIFEST, SPLITS

from anti_uav_repro.utils.repro import file_sha256

K = 3
SEED = 20260926
OUT = S.SERIES / "folds.json"


def detfly_groups(samples):
    """Det-Fly = two drone flights (ids 1xxxxx / 2xxxxx) + web bird photos. Flight frames: split each flight into K contiguous
    chunks of its sorted frame ids (one chunk per fold, so only chunk-boundary frames are adjacent across folds). Everything
    else: grouped by base name (merges the export's duplicated copies of one photo)."""
    out, per_flight = {}, defaultdict(list)
    for s in samples:
        if s["dataset"] != "det_fly":
            continue
        name = Path(s["cur_path"]).name
        m = re.match(r"(\d+)_jpg", name)
        fid = int(m.group(1)) if m else None
        if fid is not None and fid // 100000 in (1, 2):
            per_flight[fid // 100000].append((fid, s["seq_key"]))
        else:
            out[s["seq_key"]] = "df:img:" + name.split(".rf.")[0]
    for f, lst in per_flight.items():
        lst.sort()
        for rank, (_, sk) in enumerate(lst):
            out[sk] = f"df:flight{f}:chunk{rank * K // len(lst)}"
    return out


def group_of(s, dfg):
    d, k = s["dataset"], s["seq_key"]
    if d == "anti_uav":
        return "au:" + re.match(r"antiuav_[a-z]+_(\d{8}_\d{6})_", k).group(1)
    if d == "cst":
        return "cst:" + k
    return dfg[k]


def stratum_of(s, g):
    if s["dataset"] == "cst":
        return "cst:" + re.match(r"cst_[a-z]+_(.+)_(\d+)$", s["seq_key"]).group(1)
    if g.startswith("df:flight"):
        return g.rsplit(":", 1)[0]                     # one stratum per flight: its K chunks go to K different folds
    return s["dataset"]


def main():
    samples = build_split_samples(MANIFEST, SPLITS, "train")
    dfg = detfly_groups(samples)
    frames, seqs, strat, ds_of = Counter(), defaultdict(set), {}, {}
    for s in samples:
        g = group_of(s, dfg)
        frames[g] += 1
        seqs[g].add(s["seq_key"])
        strat[g] = stratum_of(s, g)
        ds_of[g] = s["dataset"]
    rng = np.random.default_rng(SEED)
    fold_of = {}
    load = defaultdict(lambda: np.zeros(K))            # per dataset
    by_stratum = defaultdict(list)
    for g in frames:
        by_stratum[strat[g]].append(g)
    for st in sorted(by_stratum):
        gs = by_stratum[st]
        tie = {g: rng.random() for g in gs}
        within = st.startswith(("cst:", "df:flight"))   # balance inside the category / flight first
        local = np.zeros(K)
        for g in sorted(gs, key=lambda x: (-frames[x], tie[x])):
            d = ds_of[g]
            key = local if within else load[d]
            cand = np.flatnonzero(key == key.min())
            k = int(cand[np.argmin(load[d][cand])])
            fold_of[g] = k
            local[k] += frames[g]
            load[d][k] += frames[g]
    seq_fold = {sk: fold_of[g] for g in fold_of for sk in seqs[g]}
    stats = {k: {d: {"groups": sum(1 for g in fold_of if fold_of[g] == k and ds_of[g] == d),
                     "sequences": sum(len(seqs[g]) for g in fold_of if fold_of[g] == k and ds_of[g] == d),
                     "frames": sum(frames[g] for g in fold_of if fold_of[g] == k and ds_of[g] == d)} for d in ("anti_uav", "cst", "det_fly")}
             for k in range(K)}
    cst_cats = {k: dict(Counter(strat[g].split(":")[1] for g in fold_of if fold_of[g] == k and ds_of[g] == "cst")) for k in range(K)}
    digest = hashlib.sha256(json.dumps(seq_fold, sort_keys=True).encode()).hexdigest()
    out = {"k": K, "seed": SEED, "source": "TRAIN split of b_pipeline/data_unified/splits.json only",
           "grouping": {"anti_uav": "recording session", "cst": "sequence, stratified by scene category", "det_fly": "flight"},
           "inputs": {"manifest_sha256": file_sha256(MANIFEST), "splits_sha256": file_sha256(SPLITS)},
           "stats": stats, "cst_categories_per_fold": cst_cats, "group_fold": fold_of, "seq_fold": seq_fold, "sha256": digest}
    if OUT.exists():
        old = json.loads(OUT.read_text(encoding="utf-8"))
        if old.get("sha256") != digest:
            sys.exit(f"{OUT} exists with a different assignment ({old.get('sha256', '')[:12]}); refusing to overwrite")
        print(f"{OUT.name} already exists with the same assignment ({digest[:12]})")
    else:
        OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
        S.ledger({"id": "F0", "type": "protocol", "what": "grouped 3-fold development split of TRAIN", "script": "scripts/make_folds.py",
                  "output": str(OUT.relative_to(S.REPO)), "folds_sha256": digest, "test_usage": "none"})
    print(json.dumps({"sha256": digest[:12], "stats": stats, "cst_categories_per_fold": cst_cats}, indent=1))


if __name__ == "__main__":
    main()
