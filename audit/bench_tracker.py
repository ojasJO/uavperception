#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_tracker.py -- ByteTrack vs SORT vs TRIVIAL single-target trackers
=======================================================================
Every sequence in this corpus contains exactly ONE drone. The reported
comparison (ByteTrack vs SORT) never asks whether a multi-object tracker is
warranted at all. Same detections, same sequences, same metric code
(`bytetrack.evaluate_tracking`), four more trackers:

    argmax@thr      per frame: highest-confidence drone box with conf>=thr -> id 1.
                    No association, no Kalman, no memory.
    greedy1@thr     one persistent id; prefer the det with highest IoU to the
                    previous box (fallback: argmax conf); coast up to `buffer`
                    frames with no output when nothing clears thr.
    bytetrack/sort  the project's own, unchanged (track_thresh .5 / low .1 / buffer 90)

Detections are cached to runs/audit/track_dets_<run>.npz so reruns are free.
"""
import sys, json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "c_model")); sys.path.insert(0, str(ROOT / "b_pipeline"))
from track_eval import pick_sequences, detect_sequence, run_tracker, _accumulate
from bytetrack import evaluate_tracking, iou_batch
from stacked_dataset import build_split_samples, load_label_cache
from aerotrack_trainer import load_checkpoint
from train import MANIFEST, SPLITS, LABEL_CACHE_PATH


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
        d = dets[f]; d = d[d[:, 4] >= thr] if len(d) else d
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/aerotrack_spd_finetuned/weights/best.pt"))
    ap.add_argument("--datasets", default="anti_uav,cst")
    ap.add_argument("--sequences", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--out", default=str(ROOT / "runs/audit"))
    a = ap.parse_args()

    run = Path(a.weights).parent.parent.name
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    cache_p = out_dir / f"track_dets_{run}.npz"

    datasets = a.datasets.split(",")
    samples = build_split_samples(MANIFEST, SPLITS, "val", datasets=datasets)
    seqs = pick_sequences(samples, a.sequences)

    if cache_p.exists():
        z = np.load(cache_p, allow_pickle=True); store = z["store"].item()
        print(f"[trk] loaded cached detections {cache_p.name}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, ck = load_checkpoint(a.weights, device, prefer_ema=True)
        amp = torch.bfloat16 if device.type == "cuda" else None
        cache = load_label_cache(LABEL_CACHE_PATH)
        store = {}
        for (ds, seq), frames in sorted(seqs.items()):
            frames = frames[:a.max_frames] if a.max_frames else frames
            dets, gt, occ = detect_sequence(model, frames, device, conf=a.conf, batch=16,
                                            workers=2, amp_dtype=amp, label_cache=cache)
            store[(ds, seq)] = (dets, gt, sorted(occ))
            print(f"[trk] detected {seq} ({len(frames)} frames, {len(occ)} occluded)", flush=True)
        np.savez(cache_p, store=np.array(store, dtype=object))

    trackers = {
        "bytetrack": lambda d: run_tracker(d, use_low_stage=True, track_thresh=.5, low_thresh=.1, track_buffer=90)[0],
        "sort":      lambda d: run_tracker(d, use_low_stage=False, track_thresh=.5, low_thresh=.1, track_buffer=90)[0],
        "argmax@0.5":  lambda d: argmax_tracker(d, 0.5),
        "argmax@0.25": lambda d: argmax_tracker(d, 0.25),
        "argmax@0.1":  lambda d: argmax_tracker(d, 0.1),
        "greedy1@0.25": lambda d: greedy1_tracker(d, 0.25),
        "greedy1@0.1":  lambda d: greedy1_tracker(d, 0.1),
    }
    rows = defaultdict(list); per_seq = []
    for (ds, seq), (dets, gt, occ) in sorted(store.items()):
        e = {"dataset": ds, "sequence": seq, "frames": len(gt), "occluded": len(occ)}
        for name, fn in trackers.items():
            m = evaluate_tracking(fn(dets), gt, occlusion_frames=set(occ))
            rows[name].append(m); e[name] = m
        per_seq.append(e)

    agg = {k: _accumulate(v) for k, v in rows.items()}
    hdr = f"{'tracker':<14}{'MOTA':>8}{'IDF1':>8}{'IDsw':>6}{'IDsw@occ':>9}{'FP':>7}{'FN':>7}{'predIDs':>8}"
    print("\n" + hdr); print("-" * len(hdr))
    for k, v in agg.items():
        print(f"{k:<14}{v['MOTA']:>8.4f}{v['IDF1_mean_per_sequence']:>8.4f}{v['id_switches']:>6}"
              f"{v['id_switches_at_occlusion_reentry']:>9}{v['false_positives']:>7}{v['false_negatives']:>7}{v['n_pred_ids']:>8}")
    print("\nper-dataset MOTA / IDF1:")
    for dsn in datasets:
        sub = {k: _accumulate([e[k] for e in per_seq if e['dataset'] == dsn]) for k in trackers}
        print(f"  {dsn:<10}" + "  ".join(f"{k}: {v['MOTA']:.3f}/{v['IDF1_mean_per_sequence']:.3f}" for k, v in sub.items()))
    (out_dir / f"tracker_benchmark_{run}.json").write_text(
        json.dumps({"weights": a.weights, "aggregate": agg, "per_sequence": per_seq}, indent=2), encoding="utf-8")
    print(f"[trk] wrote {out_dir / f'tracker_benchmark_{run}.json'}")


if __name__ == "__main__":
    main()
