#!/usr/bin/env python
"""Dataset card of a derived corpus (mission §7): every dataset x split, generated from the manifest + label cache.

    venv\\Scripts\\python.exe scripts\\dataset_card.py                       (corpus v1, data/derived/data_unified)
    set AEROTRACK_CORPUS=data_unified_v2 && venv\\Scripts\\python.exe scripts\\dataset_card.py

Per (dataset, modality) x split: sequences, frames, frames with a target, boxes per class, source resolution, target size
(box diagonal in SOURCE pixels, binned as in the CST paper: tiny <= 10, small 10-30, normal 30-50, large > 50 px; plus the
evaluation's relative-area bins micro < 0.03 %, small 0.03-1 %, standard > 1 %), median inter-frame target motion (px, consecutive
labelled frames of a sequence), invisible-target runs (frames without a box inside a sequence: count, mean / max length) and, for
CST in a corpus that has them, frame-level attribute frequencies (cst_attributes.csv). Plus provenance (source archives, terms,
known defects). Output: results/artifacts/dataset_card_<corpus>.json and docs/DATASETS_<corpus>.md. Never edit by hand.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/pipeline"))
import corpus  # noqa: E402

RES = {("anti_uav", "IR"): (640, 512), ("anti_uav", "RGB"): (1920, 1080), ("cst", "IR"): (640, 512), ("det_fly", "RGB"): (640, 640)}
SOURCES = {
    "anti_uav": {"archive": "datasets/1_anti-uav/Anti-UAV-RGBT.zip", "source": "https://github.com/ZhaoJ9014/Anti-UAV (Jiang et al., arXiv:2101.08466)",
                 "terms": "academic research use per the Anti-UAV benchmark; no licence file inside the archive (verify before redistribution)",
                 "notes": "IR 640x512 + RGB 1920x1080 per clip; official 160/67/91 sequences per modality; released box count ~2 % below the paper's Table I; VAL shares all 26 recording sessions with TRAIN"},
    "cst": {"archive": "datasets/2_cst-antiuav (z01-z07) + datasets/2_cst-antiuav_part2 (z08-z10, .zip)", "source": "https://github.com/PCwenyue/CST-Anti-UAV (Xie et al., ICCVW 2025, arXiv:2507.23473)",
            "terms": "academic and non-commercial research (dataset README / DATASETS_TERMS)",
            "notes": "corpus v1 was built from parts z01-z07 only (84/120 train, 0/40 val, 60/60 test sequences); corpus v2 from all 11 parts (CRC-verified). Official release 120/40/60."},
    "det_fly": {"archive": "datasets/3_det-fly/Det-Fly.v5i.yolov11.zip", "source": "Roboflow export of https://github.com/Jake-WU/Det-Fly (Zheng et al., IEEE RA-L 2021)",
                "terms": "CC BY 4.0 (Roboflow export), research use only per the export disclaimer; cite the original",
                "notes": "random image-level export split (VAL frames adjacent to TRAIN frames: 1,224 of 1,469); images stretched to 640x640; the only source of birds"},
}
CLASSES = {0: "drone", 1: "bird"}


def runs_of_zeros(mask):
    """Lengths of maximal runs of False in a boolean array (invisible-target runs)."""
    out, n = [], 0
    for v in mask:
        if not v:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def main():
    t0 = time.time()
    m = pd.read_csv(corpus.MANIFEST, low_memory=False, dtype={"seq_key": str, "split": str, "modality": str, "dataset": str})
    z = np.load(corpus.LABEL_CACHE)
    off, boxes = z["offsets"], z["boxes"]
    if len(off) - 1 != len(m):
        sys.exit("label cache does not match the manifest")
    m["_row"] = np.arange(len(m))
    attrs = pd.read_csv(corpus.UNIFIED / "cst_attributes.csv") if (corpus.UNIFIED / "cst_attributes.csv").exists() else None
    groups = {}
    for (ds, mod, sp), g in m.groupby(["dataset", "modality", "split"]):
        W, H = RES[(ds, mod)]
        rows = g["_row"].to_numpy()
        nb = off[rows + 1] - off[rows]
        idx = np.concatenate([np.arange(off[r], off[r + 1]) for r in rows[nb > 0]]) if (nb > 0).any() else np.zeros(0, int)
        b = boxes[idx]
        w_px, h_px = b[:, 3] * W, b[:, 4] * H
        diag = np.hypot(w_px, h_px)
        area = b[:, 3] * b[:, 4]
        rec = {"sequences": int(g["seq_key"].nunique()), "frames": int(len(g)), "frames_with_target": int((nb > 0).sum()),
               "boxes": {CLASSES[c]: int((b[:, 0] == c).sum()) for c in (0, 1)}, "source_resolution": [W, H],
               "box_diag_px": {"median": float(np.median(diag)) if len(diag) else None, "p05": float(np.percentile(diag, 5)) if len(diag) else None,
                               "p95": float(np.percentile(diag, 95)) if len(diag) else None},
               "size_bins_diag_share": {k: float(((diag > lo) & (diag <= hi)).mean()) if len(diag) else None
                                        for k, lo, hi in (("tiny", 0, 10), ("small", 10, 30), ("normal", 30, 50), ("large", 50, np.inf))},
               "area_bins_share": {"micro": float((area < 0.0003).mean()) if len(area) else None,
                                   "small": float(((area >= 0.0003) & (area < 0.01)).mean()) if len(area) else None,
                                   "standard": float((area >= 0.01).mean()) if len(area) else None}}
        if ds != "det_fly":
            motion, inv = [], []
            for _, s in g.sort_values(["seq_key", "frame_idx"]).groupby("seq_key", sort=False):
                r = s["_row"].to_numpy()
                has = off[r + 1] > off[r]
                inv += runs_of_zeros(has)
                first = off[r]
                c = np.full((len(r), 2), np.nan)
                c[has, 0] = boxes[first[has], 1] * W
                c[has, 1] = boxes[first[has], 2] * H
                d = np.linalg.norm(np.diff(c, axis=0), axis=1)
                motion += list(d[np.isfinite(d)])
            rec["motion_px_per_frame"] = {"median": float(np.median(motion)) if motion else None, "p95": float(np.percentile(motion, 95)) if motion else None}
            rec["invisible_runs"] = {"count": len(inv), "mean_len": float(np.mean(inv)) if inv else None, "max_len": int(max(inv)) if inv else None}
        if ds == "cst" and attrs is not None:
            a = attrs[attrs["split"] == sp]
            rec["cst_frame_attributes_share"] = {k: float(pd.to_numeric(a[k], errors="coerce").fillna(0).astype(bool).mean())
                                                 for k in ("OC", "OV", "SV", "TC", "DBC", "CDB", "FM") if k in a}
            rec["cst_categories"] = {k: int(v) for k, v in g.groupby(g["seq_key"].str.split("_", n=2).str[2].str.rsplit("_", n=1).str[0])["seq_key"].nunique().items()}
        groups[f"{ds}/{mod}/{sp}"] = rec
    card = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "corpus": corpus.NAME, "manifest": str(corpus.MANIFEST.relative_to(REPO)),
            "class_mapping": {"0": "uav_drone (Anti-UAV, CST, Det-Fly 'Drone')", "1": "bird_hard_negative (Det-Fly 'Bird')"},
            "preprocessing": "YOLO-normalised labels; frames read from the source extraction (Anti-UAV: JPEG decodes of the MP4s); training letterboxes to 640",
            "sources": SOURCES, "groups": groups, "seconds": round(time.time() - t0, 1)}
    (REPO / "results/artifacts").mkdir(exist_ok=True)
    (REPO / "results/artifacts" / f"dataset_card_{corpus.NAME}.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    f = lambda v, d=1: "—" if v is None else f"{v:.{d}f}"      # noqa: E731
    pc = lambda v: "—" if v is None else f"{100 * v:.1f} %"   # noqa: E731
    L = [f"# Dataset card — corpus `{corpus.NAME}`", "", f"*Generated {card['generated']} by `scripts/dataset_card.py` from `{card['manifest']}` and its "
         "label cache; never edit by hand.*", "", f"Class mapping: {card['class_mapping']}. Preprocessing: {card['preprocessing']}.", "",
         "| dataset / modality / split | seq | frames | with target | drone boxes | bird boxes | res | box diag median (p05–p95) px | tiny / small / normal / large | micro area | motion px/frame median (p95) | invisible runs (mean / max len) |",
         "|---|---:|---:|---:|---:|---:|---|---|---|---:|---|---|"]
    for k, r in groups.items():
        sb = r["size_bins_diag_share"]
        mo, iv = r.get("motion_px_per_frame"), r.get("invisible_runs")
        L.append(f"| {k} | {r['sequences']:,} | {r['frames']:,} | {r['frames_with_target']:,} | {r['boxes']['drone']:,} | {r['boxes']['bird']:,} | "
                 f"{r['source_resolution'][0]}×{r['source_resolution'][1]} | {f(r['box_diag_px']['median'])} ({f(r['box_diag_px']['p05'])}–{f(r['box_diag_px']['p95'])}) | "
                 f"{pc(sb['tiny'])} / {pc(sb['small'])} / {pc(sb['normal'])} / {pc(sb['large'])} | {pc(r['area_bins_share']['micro'])} | "
                 f"{'—' if not mo else f(mo['median'], 2) + ' (' + f(mo['p95'], 1) + ')'} | {'—' if not iv else str(iv['count']) + ' (' + f(iv['mean_len']) + ' / ' + str(iv['max_len']) + ')'} |")
    cst_rows = {k: r for k, r in groups.items() if k.startswith("cst/") and r.get("cst_categories")}
    if cst_rows:
        L += ["", "CST scene categories (sequences) and frame-level attribute shares:", ""]
        for k, r in cst_rows.items():
            L.append(f"- {k}: " + ", ".join(f"{c} {n}" for c, n in sorted(r["cst_categories"].items())) +
                     (" | attributes: " + ", ".join(f"{a} {pc(v)}" for a, v in r.get("cst_frame_attributes_share", {}).items()) if r.get("cst_frame_attributes_share") else ""))
    L += ["", "## Sources, terms, known defects", ""]
    for ds, s in SOURCES.items():
        L.append(f"- **{ds}** — archive `{s['archive']}`; source {s['source']}; terms: {s['terms']}. {s['notes']}")
    (REPO / "docs" / f"DATASETS_{corpus.NAME}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote docs/DATASETS_{corpus.NAME}.md and results/artifacts/dataset_card_{corpus.NAME}.json ({card['seconds']} s)")


if __name__ == "__main__":
    main()
