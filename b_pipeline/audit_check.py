#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_check.py -- Phase A/B System Readiness Audit (runtime checks)
=====================================================================
Lightweight, read-only verification run inside `venv` ahead of Phase C:

  1. PyTorch / DataLoader assertions on AeroTrackDataset (batch_size=2):
     batch tensor shape strictly [2,9,640,640], target tensors non-NaN.
  2. Coordinate boundary sanity check: 1,000 random label .txt files
     across data_unified/, asserting 0.0 <= x,y,w,h <= 1.0.
  3. File-count summary: images vs matching .txt labels per dataset.

Prints a JSON blob to stdout so the caller can build a report from it.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stacked_dataset import AeroTrackDataset, build_samples_from_manifest, make_dataloader

UNIFIED = HERE / "data_unified"
MANIFEST = UNIFIED / "manifest.csv"

results = {"checks": [], "ok": True}


def check(name, passed, details=""):
    results["checks"].append({"name": name, "pass": bool(passed), "details": details})
    if not passed:
        results["ok"] = False
    print(f"[{'PASS' if passed else 'FAIL'}] {name} :: {details}")


# --------------------------------------------------------------------------- #
# 1. PyTorch & DataLoader assertions
# --------------------------------------------------------------------------- #
try:
    samples = build_samples_from_manifest(MANIFEST, datasets=["anti_uav", "cst"], require_image=True)
    check("manifest.csv loads", True, f"{len(samples):,} materialized samples (anti_uav+cst)")
except Exception as e:
    check("manifest.csv loads", False, str(e))
    samples = []

if len(samples) >= 2:
    ds, dl = make_dataloader(samples, batch_size=2, img_size=(640, 640), shuffle=True,
                              num_workers=0, return_meta=True)
    imgs, targets, meta = next(iter(dl))

    check("batch shape == torch.Size([2, 9, 640, 640])",
          tuple(imgs.shape) == (2, 9, 640, 640), f"got {tuple(imgs.shape)}")
    check("batch dtype == float32", imgs.dtype == torch.float32, str(imgs.dtype))
    check("pixel range within [0,1]",
          float(imgs.min()) >= 0.0 and float(imgs.max()) <= 1.0,
          f"[{float(imgs.min()):.4f}, {float(imgs.max()):.4f}]")
    check("batch tensor has no NaNs", not torch.isnan(imgs).any().item(), "")

    target_ok = True
    target_shapes = []
    for t in targets:
        target_shapes.append(tuple(t.shape))
        if t.numel() > 0 and torch.isnan(t).any().item():
            target_ok = False
        if t.numel() > 0 and t.shape[1] != 5:
            target_ok = False
    check("target bbox tensors well-formed [Ni,5], non-NaN", target_ok, f"shapes={target_shapes}")
else:
    check("batch shape == torch.Size([2, 9, 640, 640])", False, "fewer than 2 materialized samples available")
    check("target bbox tensors well-formed [Ni,5], non-NaN", False, "skipped — no batch")

# --------------------------------------------------------------------------- #
# 2. Coordinate boundary sanity check (1,000 random label files via manifest)
# --------------------------------------------------------------------------- #
random.seed(0)
try:
    m = pd.read_csv(MANIFEST, low_memory=False, dtype={"label_path": str, "dataset": str})
    all_label_paths = m["label_path"].dropna().tolist()
    n_target = min(1000, len(all_label_paths))
    sample_paths = random.sample(all_label_paths, n_target)

    oob = []
    missing = []
    parse_errors = []
    checked = 0
    for p in sample_paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except FileNotFoundError:
            missing.append(p)
            continue
        checked += 1
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                parse_errors.append((p, ln))
                continue
            try:
                cid = int(parts[0])
                x, y, w, h = (float(v) for v in parts[1:])
            except ValueError:
                parse_errors.append((p, ln))
                continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                oob.append((p, ln))

    check("1,000 random label files: zero out-of-bounds boxes",
          len(oob) == 0,
          f"sampled={n_target}, checked={checked}, missing_files={len(missing)}, "
          f"parse_errors={len(parse_errors)}, oob={len(oob)}")
except Exception as e:
    check("1,000 random label files: zero out-of-bounds boxes", False, str(e))

# --------------------------------------------------------------------------- #
# 3. File-count summary (images vs matching .txt labels), per dataset
# --------------------------------------------------------------------------- #
counts = {}
for ds_name in ["anti_uav", "cst", "det_fly"]:
    img_dir = UNIFIED / ds_name / "images"
    lbl_dir = UNIFIED / ds_name / "labels"
    n_img = n_lbl = 0
    if img_dir.is_dir():
        for _, _, files in os.walk(img_dir):
            n_img += sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if lbl_dir.is_dir():
        for _, _, files in os.walk(lbl_dir):
            n_lbl += sum(1 for f in files if f.lower().endswith(".txt"))
    counts[ds_name] = {"images": n_img, "labels": n_lbl}
    print(f"[COUNT] {ds_name:9s}: images={n_img:,}  labels={n_lbl:,}")

results["file_counts"] = counts

print("\n===JSON===")
print(json.dumps(results, indent=2, default=str))

sys.exit(0 if results["ok"] else 1)
