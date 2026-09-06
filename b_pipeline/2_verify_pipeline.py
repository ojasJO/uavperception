#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_verify_pipeline.py  --  Phase B, Step 4: Pipeline Verification & Visualizer
=============================================================================
Validates the temporal-stacking data pipeline before AeroTrack-Net training:

  1. Instantiate AeroTrackDataset on Anti-UAV + CST samples via a DataLoader
     (batch_size = 4).
  2. Assert the batch tensor is strictly (4, 9, 640, 640), float32, in [0,1];
     additionally verify the t-1/t/t+1 channel layout and boundary duplication.
  3. Unstack one (9,640,640) sample back into three 3-channel frames.
  4. Save a 300-DPI side-by-side plot (t-1 | t + GT bbox | t+1) to
     stacked_sample_verification.png.
  5. Print standardization counts, label integrity, and DataLoader throughput
     (frames processed / second).
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stacked_dataset import (AeroTrackDataset, build_samples_from_manifest,
                             make_dataloader)

UNIFIED = HERE / "data_unified"
MANI = UNIFIED / "manifest.csv"
OUT_PNG = HERE / "stacked_sample_verification.png"


def chw_to_img(t):
    """[3,H,W] float [0,1] -> HxWx3 uint8 RGB."""
    return (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def draw_boxes(img_rgb, boxes, color=(0, 200, 0)):
    """Draw normalized YOLO boxes [N,5]=(cls,xc,yc,w,h) on an HxWx3 RGB image."""
    H, W = img_rgb.shape[:2]
    out = img_rgb.copy()
    for b in boxes:
        _, xc, yc, w, h = [float(v) for v in b[:5]]
        x1 = int((xc - w / 2) * W); y1 = int((yc - h / 2) * H)
        x2 = int((xc + w / 2) * W); y2 = int((yc + h / 2) * H)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def main():
    print("=" * 74)
    print("Phase B / Step 4: Pipeline verification")
    print("=" * 74)
    if not MANI.exists():
        print("ERROR: manifest.csv missing — run 1_standardize_annotations.py first")
        sys.exit(1)

    # ---- 1. build samples (materialized Anti-UAV + CST) ------------------ #
    samples = build_samples_from_manifest(MANI, datasets=["anti_uav", "cst"],
                                          require_image=True)
    ds_counts = Counter(s["dataset"] for s in samples)
    print(f"[samples] total={len(samples):,}  per-dataset={dict(ds_counts)}")
    if len(samples) < 4:
        print("ERROR: not enough materialized samples to form a batch.")
        sys.exit(1)

    # ---- 2. DataLoader + strict shape assertions ------------------------- #
    ds, dl = make_dataloader(samples, batch_size=4, img_size=(640, 640),
                             shuffle=True, num_workers=0, return_meta=True)
    imgs, targets, meta = next(iter(dl))
    assert tuple(imgs.shape) == (4, 9, 640, 640), f"bad batch shape {tuple(imgs.shape)}"
    assert imgs.dtype == torch.float32, f"bad dtype {imgs.dtype}"
    assert float(imgs.min()) >= 0.0 and float(imgs.max()) <= 1.0, "pixels not in [0,1]"
    print(f"[PASS] batch shape = {tuple(imgs.shape)} | dtype={imgs.dtype} | "
          f"range=[{float(imgs.min()):.3f}, {float(imgs.max()):.3f}]")

    # boundary-duplication + channel-layout checks
    def first_with(fidx_pred):
        for i, s in enumerate(samples):
            if fidx_pred(s):
                return i
        return None

    i0 = first_with(lambda s: s["frame_idx"] == 0)
    if i0 is not None:
        x0, _, _ = ds[i0]
        assert torch.equal(x0[0:3], x0[3:6]), "t=0 boundary: t-1 should duplicate t"
        print("[PASS] boundary t=0: channels[0:3] (t-1) == channels[3:6] (t)")
    iN = first_with(lambda s: s["frame_idx"] == s["seq_len"] - 1 and s["seq_len"] > 1)
    if iN is not None:
        xN, _, _ = ds[iN]
        assert torch.equal(xN[6:9], xN[3:6]), "t=N-1 boundary: t+1 should duplicate t"
        print("[PASS] boundary t=N-1: channels[6:9] (t+1) == channels[3:6] (t)")

    # ---- 3/4. unstack a target-bearing sample & visualize ---------------- #
    viz = None
    for s in samples:
        lp = Path(s["label_path"])
        if lp.exists() and lp.stat().st_size > 0:
            viz = s
            break
    if viz is None:
        viz = samples[len(samples) // 2]
    vds = AeroTrackDataset([viz], img_size=(640, 640), return_meta=True)
    stacked, tb, vmeta = vds[0]
    f_prev, f_cur, f_next = stacked[0:3], stacked[3:6], stacked[6:9]
    img_prev, img_cur, img_next = chw_to_img(f_prev), chw_to_img(f_cur), chw_to_img(f_next)
    img_cur_box = draw_boxes(img_cur, tb.numpy())

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, im, title in zip(
            axes,
            [img_prev, img_cur_box, img_next],
            [f"t-1  (frame {max(viz['frame_idx']-1,0)})",
             f"t  (frame {viz['frame_idx']}) + GT bbox [{tb.shape[0]} box]",
             f"t+1  (frame {min(viz['frame_idx']+1, viz['seq_len']-1)})"]):
        ax.imshow(im)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"AeroTrack 9-channel stack — {vmeta['dataset']} / {vmeta['seq_key']} "
                 f"(seq_len={viz['seq_len']})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {OUT_PNG.name} (sample: {vmeta['dataset']}/{vmeta['seq_key']} "
          f"frame {viz['frame_idx']}, {tb.shape[0]} target box)")

    # ---- 5a. throughput -------------------------------------------------- #
    t0 = time.time(); n_batches = n_samp = 0
    for b in dl:
        n_batches += 1
        n_samp += b[0].shape[0]
        if n_batches >= 25:
            break
    dt = time.time() - t0
    sps = n_samp / dt if dt else 0
    print(f"[throughput] {n_samp} stacked samples in {dt:.2f}s "
          f"-> {sps:.1f} samples/s  ({sps*3:.1f} frame-reads/s)  [num_workers=0]")
    print("             (single-thread + cold-disk I/O bound; each sample = 3 JPEG "
          "decodes+resize. Scales ~linearly with DataLoader num_workers in training.)")

    # ---- 5b. standardization counts + label integrity -------------------- #
    summ = {}
    try:
        summ = json.load(open(UNIFIED / "standardization_summary.json"))
    except Exception:
        pass
    import pandas as pd
    m = pd.read_csv(MANI, low_memory=False,
                    dtype={"image_path": str, "label_path": str})
    print("\n---- STANDARDIZATION SUMMARY ----")
    for dk in ["anti_uav", "cst", "det_fly"]:
        g = m[m.dataset == dk]
        if len(g) == 0:
            continue
        mat = int((g["materialized"] == 1).sum())
        tgt = int((g["has_target"] == 1).sum())
        print(f"  {dk:9s}: frames={len(g):,}  with_target={tgt:,} "
              f"({tgt/len(g)*100:.1f}%)  materialized_imgs={mat:,}  "
              f"boxes={int(g['n_boxes'].sum()):,}")
    # label integrity: every materialized frame must have an existing label file
    mat_rows = m[m.materialized == 1]
    sample_check = mat_rows.sample(min(500, len(mat_rows)), random_state=0)
    missing = sum(0 if Path(p).exists() else 1 for p in sample_check["label_path"])
    print(f"  label-integrity: {len(sample_check)} materialized frames sampled, "
          f"{missing} missing label files")
    if summ.get("det_fly"):
        print(f"  det_fly class remap: {summ['det_fly'].get('class_remap')}")
    print("\nVerification complete.")


if __name__ == "__main__":
    main()
