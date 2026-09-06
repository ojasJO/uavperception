#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ############################################################################
# ##   Requires a trained checkpoint + a CUDA GPU. Run after train.py.       ##
# ############################################################################
"""
track_eval.py  --  detector + ByteTrack, measured end to end
============================================================
Runs the trained detector over WHOLE SEQUENCES in frame order, feeds every
detection to `bytetrack.BYTETracker`, and reports the CLEAR-MOT + IDF1 suite
from 00_TRAINING_ROADMAP.md §7.3.

The comparison that matters
---------------------------
Reporting MOTA for one tracker configuration proves nothing about the paper
choice. So this runs the SAME detections through two trackers:

  * **bytetrack** -- two-stage association; low-confidence boxes may sustain an
    existing tracklet.
  * **sort**      -- identical code with the second stage disabled, i.e. low
    boxes are discarded. This is the control.

and reports both, plus **ID switches restricted to occlusion re-entry frames**,
which is the number the ByteTrack decision actually rests on. Review 1 measured
494 out-of-view events in CST with a mean length of 55.6 frames; if associating
the low boxes does not reduce switches on re-entry, the citation is decoration
and should be dropped.

Ground truth and occlusion
--------------------------
Every sequence in this corpus tracks exactly one target, so the GT id is
constant and an id switch is unambiguous. Frames the pipeline stored as a 0-byte
label are `exist = 0` -- the target is genuinely out of view. Those frames carry
no GT box (a detection there is a false positive) and their indices form the
occlusion set.

Usage
-----
    python c_model/track_eval.py --weights runs/aerotrack_spd_v1/weights/best.pt
    python c_model/track_eval.py --datasets cst --sequences 12
    python c_model/track_eval.py --split test --confirm-test
"""

import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "b_pipeline"))

from stacked_dataset import (AeroTrackDataset, aerotrack_collate,
                             build_split_samples, load_label_cache)
from aerotrack_trainer import load_checkpoint
from bytetrack import BYTETracker, evaluate_tracking
from train import MANIFEST, SPLITS, LABEL_CACHE_PATH, require_gpu, _worker_init

DEFAULT_WEIGHTS = ROOT / "runs" / "aerotrack_spd_v1" / "weights" / "best.pt"
DRONE_CLS = 0


# --------------------------------------------------------------------------- #
def pick_sequences(samples, n_per_dataset, seed=0):
    """Choose whole sequences (never partial ones) and keep them frame-ordered.

    Tracking metrics are only meaningful over a contiguous sequence: a sampled
    subset of frames would manufacture apparent occlusions and inflate the id
    switch count for every tracker equally, which looks like a result and is an
    artifact."""
    by_seq = defaultdict(list)
    for s in samples:
        by_seq[(s["dataset"], s["seq_key"])].append(s)
    rng = np.random.default_rng(seed)
    chosen = {}
    by_ds = defaultdict(list)
    for k in by_seq:
        by_ds[k[0]].append(k)
    for ds, keys in by_ds.items():
        keys = sorted(keys)
        if n_per_dataset and len(keys) > n_per_dataset:
            # longest sequences first: they carry the occlusion events
            keys = sorted(keys, key=lambda k: -len(by_seq[k]))[:n_per_dataset]
        for k in keys:
            chosen[k] = sorted(by_seq[k], key=lambda s: int(s["frame_idx"]))
    return chosen


@torch.no_grad()
def detect_sequence(model, samples, device, imgsz=640, conf=0.05, iou=0.7,
                    max_det=30, batch=16, workers=2, amp_dtype=None,
                    label_cache=None):
    """-> (dets_by_frame, gt_by_frame, occlusion_frames) for one sequence.

    `conf` is deliberately LOW (0.05, not the 0.25 used for reporting FP rates):
    ByteTrack's whole premise is that the boxes between `low_thresh` and
    `track_thresh` are the occluded target. Threshold them away here and the
    second association stage has nothing to associate, and the experiment
    silently measures SORT twice.
    """
    from ultralytics.utils.nms import non_max_suppression
    import inspect
    nms_kw = ({"max_time_img": 2.0}
              if "max_time_img" in inspect.signature(non_max_suppression).parameters
              else {})

    ds = AeroTrackDataset(samples, img_size=(imgsz, imgsz), augment=None,
                          letterbox=True, out_dtype="uint8", return_meta=True,
                          label_cache=label_cache)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=True, collate_fn=aerotrack_collate,
                    worker_init_fn=_worker_init)
    dets_by_frame, gt_by_frame, occl = {}, {}, set()
    i = 0
    for imgs, targets, metas in dl:
        imgs = imgs.to(device, non_blocking=True)
        if imgs.dtype == torch.uint8:
            imgs = imgs.float().div_(255.0)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                preds = model(imgs)
        else:
            preds = model(imgs)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        out = non_max_suppression(preds.float(), conf, iou, max_det=max_det,
                                  **nms_kw)
        for k, det in enumerate(out):
            f = i + k
            d = det.detach().cpu().numpy().astype(np.float32) if len(det) else \
                np.zeros((0, 6), dtype=np.float32)
            # tracking is single-class here: birds are a detector-level hard
            # negative, not a thing to keep an identity for
            d = d[d[:, 5] == DRONE_CLS] if len(d) else d
            dets_by_frame[f] = d[:, :6]

            t = targets[k].numpy()
            t = t[t[:, 0] == DRONE_CLS] if len(t) else t
            if len(t):
                b = np.empty((len(t), 5), dtype=np.float32)
                b[:, 0] = (t[:, 1] - t[:, 3] / 2) * imgsz
                b[:, 1] = (t[:, 2] - t[:, 4] / 2) * imgsz
                b[:, 2] = (t[:, 1] + t[:, 3] / 2) * imgsz
                b[:, 3] = (t[:, 2] + t[:, 4] / 2) * imgsz
                b[:, 4] = 1                       # one target per sequence
                gt_by_frame[f] = b
            else:
                gt_by_frame[f] = np.zeros((0, 5), dtype=np.float32)
                occl.add(f)                       # exist = 0
        i += len(out)
    return dets_by_frame, gt_by_frame, occl


def run_tracker(dets_by_frame, use_low_stage=True, **kw):
    """Feed one sequence's detections through the tracker, in frame order."""
    # Disabling the second stage is expressed as low_thresh == track_thresh:
    # the low band becomes empty, so stage 2 has nothing to match and the
    # tracker degenerates to SORT. Same code path, same Kalman filter, same
    # gates -- the ONLY difference is the thing under test.
    tk = BYTETracker(**kw)
    if not use_low_stage:
        tk.low_thresh = tk.track_thresh
    pred = {}
    for f in sorted(dets_by_frame):
        out = tk.update(dets_by_frame[f])
        pred[f] = out[:, :5] if len(out) else np.zeros((0, 5), dtype=np.float32)
    return pred, dict(tk.stats)


def _accumulate(rows):
    """Sum the count-like metrics over sequences, then recompute the rates.

    Averaging per-sequence MOTA would weight a 54-frame sequence the same as a
    1,998-frame one. CLEAR-MOT is defined over pooled counts; pool them."""
    tot = defaultdict(int)
    for r in rows:
        for k in ("id_switches", "id_switches_at_occlusion_reentry",
                  "fragmentations", "false_positives", "false_negatives",
                  "n_gt_boxes", "n_gt_ids", "n_pred_ids"):
            tot[k] += r[k]
    n_gt = max(tot["n_gt_boxes"], 1)
    idf1 = float(np.mean([r["IDF1"] for r in rows])) if rows else 0.0
    return {
        "MOTA": round(1.0 - (tot["false_negatives"] + tot["false_positives"]
                             + tot["id_switches"]) / n_gt, 5),
        "IDF1_mean_per_sequence": round(idf1, 5),
        **{k: int(v) for k, v in tot.items()},
        "n_sequences": len(rows),
    }


def main():
    ap = argparse.ArgumentParser(description="AeroTrack-Net tracking evaluation")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--datasets", default="anti_uav,cst")
    ap.add_argument("--sequences", type=int, default=8,
                    help="sequences per dataset (longest first); 0 = all")
    ap.add_argument("--max-frames", type=int, default=1200,
                    help="cap per sequence, 0 = whole sequence")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--track-thresh", type=float, default=0.5)
    ap.add_argument("--low-thresh", type=float, default=0.1)
    ap.add_argument("--track-buffer", type=int, default=90)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.split == "test" and not a.confirm_test:
        raise SystemExit(
            "\n[REFUSED] --split test needs --confirm-test. cst[test] is the "
            "project's only\n  honest cross-domain measurement; spend it once.\n")

    device = require_gpu()
    w = Path(a.weights)
    if not w.exists():
        raise SystemExit(f"[ABORT] weights not found: {w}")
    model, ck = load_checkpoint(w, device, prefer_ema=True)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    cache = load_label_cache(LABEL_CACHE_PATH)
    print(f"[track] {w} (epoch {ck.get('epoch')})")

    datasets = [d for d in a.datasets.split(",") if d]
    samples = build_split_samples(MANIFEST, SPLITS, a.split, datasets=datasets)
    seqs = pick_sequences(samples, a.sequences)
    print(f"[track] {len(seqs)} sequences from split={a.split} {datasets}")

    tk_kw = dict(track_thresh=a.track_thresh, low_thresh=a.low_thresh,
                 track_buffer=a.track_buffer)
    rows = {"bytetrack": [], "sort": []}
    per_seq = []
    for (ds, seq), frames in sorted(seqs.items()):
        if a.max_frames:
            frames = frames[:a.max_frames]
        dets, gt, occ = detect_sequence(model, frames, device, imgsz=a.imgsz,
                                        conf=a.conf, batch=a.batch,
                                        workers=a.workers, amp_dtype=amp,
                                        label_cache=cache)
        entry = {"dataset": ds, "sequence": seq, "frames": len(frames),
                 "occluded_frames": len(occ)}
        for name, low in (("bytetrack", True), ("sort", False)):
            pred, stats = run_tracker(dets, use_low_stage=low, **tk_kw)
            m = evaluate_tracking(pred, gt, occlusion_frames=occ)
            rows[name].append(m)
            entry[name] = {**m, "assoc_stats": stats}
        per_seq.append(entry)
        b, s = entry["bytetrack"], entry["sort"]
        print(f"  {ds:<9} {seq[-38:]:<38} {len(frames):>5} fr "
              f"({len(occ):>4} occluded) | IDsw byte {b['id_switches']:>3} "
              f"vs sort {s['id_switches']:>3} | MOTA {b['MOTA']:.3f} / "
              f"{s['MOTA']:.3f}", flush=True)

    report = {"weights": str(w), "split": a.split, "datasets": datasets,
              "conf": a.conf, "tracker": tk_kw,
              "bytetrack": _accumulate(rows["bytetrack"]),
              "sort": _accumulate(rows["sort"]),
              "per_sequence": per_seq}

    b, s = report["bytetrack"], report["sort"]
    print("\n" + "=" * 78)
    print("Tracking — ByteTrack (two-stage) vs SORT (high-confidence only)")
    print("=" * 78)
    hdr = f"  {'metric':<38}{'ByteTrack':>12}{'SORT':>12}{'delta':>12}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for k in ("MOTA", "IDF1_mean_per_sequence", "id_switches",
              "id_switches_at_occlusion_reentry", "fragmentations",
              "false_negatives", "false_positives", "n_pred_ids"):
        bv, sv = b[k], s[k]
        d = bv - sv
        fmt = "{:>12.4f}" if isinstance(bv, float) else "{:>12d}"
        print(f"  {k:<38}" + fmt.format(bv) + fmt.format(sv) +
              (f"{d:>+12.4f}" if isinstance(bv, float) else f"{d:>+12d}"))
    print("=" * 78)
    gain = s["id_switches_at_occlusion_reentry"] - b["id_switches_at_occlusion_reentry"]
    print(f"  Verdict: associating low-confidence boxes changed id switches at "
          f"occlusion re-entry by {-gain:+d}.")
    print("  That is the number the ByteTrack citation rests on — Review 1 "
          "measured 494 out-of-view\n  events in CST with a mean length of 55.6 "
          "frames.")

    out = Path(a.out) if a.out else (w.parent.parent / "eval_tracking")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tracking_report.json").write_text(json.dumps(report, indent=2),
                                              encoding="utf-8")
    print(f"\n  artifacts -> {out}")


if __name__ == "__main__":
    main()
