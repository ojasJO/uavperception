#!/usr/bin/env python
"""Tracker comparison with the trivial single-target baselines it must beat (repair item 6).

    python experiments/aerotrack_repair/scripts/tracker_repair.py --weights runs/aerotrack_spd_finetuned/weights/best.pt

Same detections (conf 0.05 so ByteTrack's low band exists), same VAL sequences (rule: the `--sequences` longest
sequences per dataset — they carry the occlusion events; `track_eval.pick_sequences`), same metric code
(`bytetrack.evaluate_tracking`, CLEAR-MOT + IDF1 + id switches at occlusion re-entry), for:

    bytetrack          the project's tracker (track .5 / low .1 / buffer 90)
    sort               same code, second stage disabled (the original control)
    argmax@thr         per frame: highest-confidence drone box with conf >= thr, always id 1 (no association at all)
    greedy1@thr        one persistent id, IoU-nearest to the previous box, coasts up to 90 frames

Every sequence in this corpus holds ONE drone, so a multi-object tracker must at least beat "take the best box".
Pre-registered rule: ByteTrack stays in the pipeline/README only if it beats argmax@0.25 on BOTH MOTA and IDF1
on both datasets. Detections are cached in runs/<run>/tracking_repair/dets_val.npz.
(argmax/greedy implementations from audit/bench_tracker.py.)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from aerotrack_trainer import load_checkpoint
from bytetrack import evaluate_tracking, iou_batch
from stacked_dataset import build_split_samples, load_label_cache
from track_eval import _accumulate, detect_sequence, pick_sequences, run_tracker
from train import LABEL_CACHE_PATH, MANIFEST, SPLITS

from anti_uav_repro.utils.repro import env_snapshot, file_sha256, git_info, write_json

from _bootstrap import REPO_ROOT, REPORTS_DIR


def argmax_tracker(dets, thr):
    out = {}
    for f, d in dets.items():
        d = d[d[:, 4] >= thr] if len(d) else d
        if len(d):
            b = d[np.argmax(d[:, 4])]
            out[f] = np.array([[b[0], b[1], b[2], b[3], 1.0]], np.float32)
        else:
            out[f] = np.zeros((0, 5), np.float32)
    return out


def greedy1_tracker(dets, thr, buffer=90):
    out, last, age = {}, None, 0
    for f in sorted(dets):
        d = dets[f]
        d = d[d[:, 4] >= thr] if len(d) else d
        if len(d):
            if last is not None and age <= buffer:
                iou = iou_batch(last[None, :4], d[:, :4])[0]
                j = int(np.argmax(iou)) if iou.max() > 0 else int(np.argmax(d[:, 4]))
            else:
                j = int(np.argmax(d[:, 4]))
            last, age = d[j, :4].copy(), 0
            out[f] = np.array([[*last, 1.0]], np.float32)
        else:
            age += 1
            out[f] = np.zeros((0, 5), np.float32)
    return out


TRACKERS = {
    "bytetrack": lambda d: run_tracker(d, use_low_stage=True, track_thresh=.5, low_thresh=.1, track_buffer=90)[0],
    "sort": lambda d: run_tracker(d, use_low_stage=False, track_thresh=.5, low_thresh=.1, track_buffer=90)[0],
    "argmax@0.5": lambda d: argmax_tracker(d, 0.5),
    "argmax@0.25": lambda d: argmax_tracker(d, 0.25),
    "argmax@0.1": lambda d: argmax_tracker(d, 0.1),
    "greedy1@0.25": lambda d: greedy1_tracker(d, 0.25),
    "greedy1@0.1": lambda d: greedy1_tracker(d, 0.1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--datasets", default="anti_uav,cst")
    ap.add_argument("--sequences", type=int, default=5, help="longest sequences per dataset (val)")
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--conf", type=float, default=0.05)
    a = ap.parse_args()
    w = Path(a.weights)
    run = w.parent.parent.name
    out_dir = REPO_ROOT / "runs" / run / "tracking_repair"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_p = out_dir / "dets_val.npz"
    datasets = a.datasets.split(",")
    samples = build_split_samples(MANIFEST, SPLITS, "val", datasets=datasets)
    seqs = pick_sequences(samples, a.sequences)
    if cache_p.exists():
        store = np.load(cache_p, allow_pickle=True)["store"].item()
        print(f"[trk] loaded cached detections {cache_p}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, _ = load_checkpoint(w, device, prefer_ema=True)
        amp = torch.bfloat16 if device.type == "cuda" else None
        cache = load_label_cache(LABEL_CACHE_PATH)
        store = {}
        for (ds, seq), frames in sorted(seqs.items()):
            frames = frames[:a.max_frames] if a.max_frames else frames
            dets, gt, occ = detect_sequence(model, frames, device, conf=a.conf, batch=16, workers=2, amp_dtype=amp, label_cache=cache)
            store[(ds, seq)] = (dets, gt, sorted(occ))
            print(f"[trk] detected {seq} ({len(frames)} frames, {len(occ)} occluded)", flush=True)
        np.savez(cache_p, store=np.array(store, dtype=object))
    rows, per_seq = defaultdict(list), []
    for (ds, seq), (dets, gt, occ) in sorted(store.items()):
        e = {"dataset": ds, "sequence": seq, "frames": len(gt), "occluded": len(occ)}
        for name, fn in TRACKERS.items():
            m = evaluate_tracking(fn(dets), gt, occlusion_frames=set(occ))
            rows[name].append(m)
            e[name] = m
        per_seq.append(e)
    agg = {k: _accumulate(v) for k, v in rows.items()}
    per_ds = {dsn: {k: _accumulate([e[k] for e in per_seq if e["dataset"] == dsn]) for k in TRACKERS} for dsn in datasets}
    bt, ax = agg["bytetrack"], agg["argmax@0.25"]
    keep = all(per_ds[d]["bytetrack"]["MOTA"] > per_ds[d]["argmax@0.25"]["MOTA"] and
               per_ds[d]["bytetrack"]["IDF1_mean_per_sequence"] > per_ds[d]["argmax@0.25"]["IDF1_mean_per_sequence"] for d in datasets)
    rep = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_info(), "env": env_snapshot(), "command": " ".join(sys.argv),
           "weights": str(w), "weights_sha256": file_sha256(w), "split": "val", "datasets": datasets,
           "sequence_rule": f"{a.sequences} longest sequences per dataset (track_eval.pick_sequences), first {a.max_frames} frames",
           "sequences": [{"dataset": e["dataset"], "sequence": e["sequence"], "frames": e["frames"], "occluded": e["occluded"]} for e in per_seq],
           "conf": a.conf, "aggregate": agg, "per_dataset": per_ds, "per_sequence": per_seq,
           "rule": "ByteTrack is kept only if it beats argmax@0.25 on MOTA and IDF1 on every dataset",
           "verdict": "ByteTrack KEPT" if keep else "ByteTrack RETIRED (a no-association argmax baseline is at least as good)"}
    write_json(rep, out_dir / "tracker_repair.json")
    md = render_md(rep)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"tracker_val_{run}.md").write_text(md, encoding="utf-8")
    write_json(rep, REPORTS_DIR / f"tracker_val_{run}.json")
    print(md)


def render_md(rep):
    L = [f"# Tracker comparison — {Path(rep['weights']).parent.parent.name} — VAL ({rep['sequence_rule']})", "",
         f"Generated {rep['generated']}; weights sha256 `{rep['weights_sha256'][:12]}`; detections at conf {rep['conf']}; "
         f"{len(rep['sequences'])} sequences, {sum(s['frames'] for s in rep['sequences'])} frames.", "",
         "| tracker | MOTA | IDF1 | ID switches | ID sw @ occlusion re-entry | FP | FN | predicted IDs (GT: 1/seq) |", "|---|---|---|---|---|---|---|---|"]
    for k, v in rep["aggregate"].items():
        L.append(f"| {k} | {v['MOTA']:.4f} | {v['IDF1_mean_per_sequence']:.4f} | {v['id_switches']} | {v['id_switches_at_occlusion_reentry']} | "
                 f"{v['false_positives']} | {v['false_negatives']} | {v['n_pred_ids']} |")
    L += ["", "## Per dataset (MOTA / IDF1)", "", "| dataset | " + " | ".join(rep["aggregate"]) + " |", "|---|" + "---|" * len(rep["aggregate"])]
    for d, sub in rep["per_dataset"].items():
        L.append(f"| {d} | " + " | ".join(f"{sub[k]['MOTA']:.3f} / {sub[k]['IDF1_mean_per_sequence']:.3f}" for k in rep["aggregate"]) + " |")
    L += ["", f"**Rule:** {rep['rule']}.  **Verdict:** {rep['verdict']}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
