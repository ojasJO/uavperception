#!/usr/bin/env python
"""Run a detector over EVERY frame of the VAL sequences and cache its detections, for tracker / state-estimator development.

    set AEROTRACK_CORPUS=data_unified_v2   (optional; default corpus v1)
    venv\\Scripts\\python.exe experiments\\systems\\scripts\\cache_detections.py --run gen_temporal_fp_g4_full_e20 [--limit-seqs 3]

Per sequence (Anti-UAV IR + RGB and CST; Det-Fly has no sequences): every frame in order, the run's own 9-channel window
(run_meta: symmetric / causal / t,t,t; neighbours clamped inside the sequence exactly as in training), letterbox 640, bf16
(the evaluated protocol), NMS conf 0.01 / IoU 0.7 / max_det 10, boxes mapped back to ORIGINAL image pixels (xyxy).
Ground truth is NOT stored here: the tracker evaluation reads the official annotations itself, so nothing can leak.
Output: runs/systems/_cache/<run>/<corpus>/<split>/<seq_key>.npz  (+ index.json with set identity, run, weights sha256).
VAL only; refuses TEST.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "c_model", REPO / "b_pipeline", REPO / "experiments" / "anti_uav" / "src"):
    sys.path.insert(0, str(p))


def windows(samples, window, gap):
    """samples of ONE sequence, sorted by frame -> list of (prev, cur, next) paths for the run's window."""
    paths = [s["cur_path"] for s in samples]
    n = len(paths)
    out = []
    for i in range(n):
        if window == "ttt" or gap == 0:
            out.append((paths[i], paths[i], paths[i]))
        elif window == "symmetric":
            out.append((paths[max(i - gap, 0)], paths[i], paths[min(i + gap, n - 1)]))
        elif window == "causal":
            out.append((paths[max(i - gap, 0)], paths[i], paths[max(i - 2 * gap, 0)]))
        else:
            raise ValueError(window)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val"])
    ap.add_argument("--datasets", nargs="+", default=["anti_uav", "cst"])
    ap.add_argument("--limit-seqs", type=int, default=None, help="development only: first N sequences per dataset")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--corpus", default=None, help="'auto': corpus v2 if verified (cst_completion/reports/corpus_v2.json OK), "
                                                   "else v1; a name: that corpus; default: AEROTRACK_CORPUS / v1")
    a = ap.parse_args()
    if a.corpus == "auto":
        rep = REPO / "experiments" / "cst_completion" / "reports" / "corpus_v2.json"
        ok = rep.exists() and json.loads(rep.read_text(encoding="utf-8")).get("verdict") == "OK"
        os.environ["AEROTRACK_CORPUS"] = "data_unified_v2" if ok else "data_unified"
    elif a.corpus:
        os.environ["AEROTRACK_CORPUS"] = a.corpus

    cache_split(a.run, a.split, a.datasets, a.limit_seqs, a.batch)


def cache_split(run, split, datasets=("anti_uav", "cst"), limit_seqs=None, batch=16):
    """Cache every frame of `split` for `run` (corpus = AEROTRACK_CORPUS of this process). The CLI only allows VAL; the
    single pre-registered TEST pass calls this from experiments/aerotrack_v2/scripts/final_eval.py behind its own guards."""
    import cv2
    import torch
    from ultralytics.utils.nms import non_max_suppression

    import corpus
    from aerotrack_trainer import load_checkpoint
    from anti_uav_repro.utils.repro import file_sha256
    from stacked_dataset import build_split_samples, letterbox_image, letterbox_params
    from train import MANIFEST, SPLITS

    run_dir = REPO / "runs" / run
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    gap = int(meta.get("gap", meta.get("temporal_gap", 1)))
    window = meta.get("window") or ("ttt" if gap == 0 else "symmetric")
    w = run_dir / "weights" / "last.pt"
    wsha = file_sha256(w)
    out_dir = REPO / "runs" / "systems" / "_cache" / run / corpus.NAME / split
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    from precision import amp_dtype
    _AMP = amp_dtype(dev)        # bf16 on sm>=8.0, fp16 on the V100 (D-2026-10-06-04)
    model, ck = load_checkpoint(w, dev, prefer_ema=True)
    samples = build_split_samples(MANIFEST, SPLITS, split, temporal_gap=1, datasets=datasets)
    by = {}
    for s in samples:
        by.setdefault(s["seq_key"], []).append(s)
    keys = sorted(by)
    if limit_seqs:
        keep = []
        for d in datasets:
            keep += [k for k in keys if by[k][0]["dataset"] == d][: limit_seqs]
        keys = keep
    t0, n_frames = time.time(), 0
    index = {"run": run, "weights_sha256": wsha, "window": window, "gap": gap, "corpus": corpus.NAME, "split": split,
             "nms": {"conf": 0.01, "iou": 0.7, "max_det": 10}, "precision": f"{str(_AMP).replace('torch.', '')} autocast" if _AMP is not None else "fp32", "sequences": {}}
    for k in keys:
        seq = sorted(by[k], key=lambda s: s["frame_idx"])
        wins = windows(seq, window, gap)
        img0 = cv2.imread(seq[0]["cur_path"], cv2.IMREAD_COLOR)
        H, W = img0.shape[:2]
        r, dw, dh, nw, nh = letterbox_params(H, W, 640, 640)
        cache = {}

        def frame(p):
            if p not in cache:
                im = cv2.imread(p, cv2.IMREAD_COLOR)
                lb = letterbox_image(im, 640, 640, r, dw, dh, nw, nh)
                cache[p] = torch.from_numpy(np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))).permute(2, 0, 1)
                if len(cache) > 64:                     # windows only look back/forward a few frames
                    cache.pop(next(iter(cache)))
            return cache[p]
        boxes, scores, classes, offs = [], [], [], [0]
        for i in range(0, len(wins), batch):
            chunk = wins[i:i + batch]
            x = torch.stack([torch.cat([frame(p) for p in trip], 0) for trip in chunk]).to(dev).float().div_(255.0)
            with torch.no_grad(), torch.autocast("cuda", dtype=_AMP, enabled=_AMP is not None):
                y = model(x)
            y = (y[0] if isinstance(y, (list, tuple)) else y).float().cpu()
            for det in non_max_suppression(y, 0.01, 0.7, max_det=10):
                d = det.numpy().astype(np.float32)
                d[:, [0, 2]] = (d[:, [0, 2]] - dw) / r          # undo the letterbox -> original pixels
                d[:, [1, 3]] = (d[:, [1, 3]] - dh) / r
                d[:, [0, 2]] = d[:, [0, 2]].clip(0, W)
                d[:, [1, 3]] = d[:, [1, 3]].clip(0, H)
                boxes.append(d[:, :4]); scores.append(d[:, 4]); classes.append(d[:, 5].astype(np.int8))
                offs.append(offs[-1] + len(d))
        np.savez_compressed(out_dir / f"{k}.npz", boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
                            scores=np.concatenate(scores) if scores else np.zeros(0, np.float32),
                            classes=np.concatenate(classes) if classes else np.zeros(0, np.int8), offsets=np.array(offs, np.int64),
                            frame_idx=np.array([s["frame_idx"] for s in seq], np.int64), image_size=np.array([W, H]))
        index["sequences"][k] = {"dataset": seq[0]["dataset"], "modality": seq[0]["modality"], "n_frames": len(seq),
                                 "first_frame_path": seq[0]["cur_path"], "image_size": [W, H]}
        n_frames += len(seq)
        print(f"[cache] {k}: {len(seq)} frames ({n_frames / (time.time() - t0):.0f} fps overall)", flush=True)
    ident = sorted((k, v["n_frames"]) for k, v in index["sequences"].items())
    index["set_sha256"] = hashlib.sha256(json.dumps(ident).encode()).hexdigest()
    index["frames"], index["seconds"] = n_frames, round(time.time() - t0, 1)
    index["limited"] = limit_seqs
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"[cache] {len(keys)} sequences, {n_frames:,} frames in {index['seconds'] / 60:.1f} min -> {out_dir}")
    return out_dir, index


if __name__ == "__main__":
    main()
