#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stacked_dataset.py  --  Phase B, Step 3: Temporal Frame-Stacking Dataset
========================================================================
`AeroTrackDataset` builds 9-channel spatio-temporal input tensors for
AeroTrack-Net by stacking three frames (t-g, t, t+g) of a sequence along the
channel axis.

For a target frame at position t in its sequence:
  * fetch frames t-g, t, t+g  (boundary: clamp to [0, N-1] -> duplication)
  * each frame: BGR->RGB, LETTERBOXED to 640x640 (aspect preserved), -> [3,H,W]
  * concatenate along dim=0  ->  [9,640,640]
  * targets = normalized YOLO boxes of frame t, mapped into letterbox space

Changes vs the Review-1 baseline (all justified in 00_TRAINING_ROADMAP.md)
--------------------------------------------------------------------------
1. LETTERBOX instead of stretch (§4.3). Stretching 640x512 IR and 1920x1080 RGB
   to a 1:1 canvas gives the *same physical drone* two different aspect ratios
   (1.33 vs 0.93). Letterboxing preserves scale and keeps the two streams
   geometrically consistent, which is a correctness issue, not a nicety.
2. SPLIT filtering (§2). `build_samples_from_manifest(..., splits=...)` filters
   the manifest's `split` column so train/val can be disjoint at SEQUENCE level.
   Consecutive video frames are near-identical (median 5.10 px/frame motion), so
   a frame-level split leaks; only a sequence-level split is honest.
3. TEMPORALLY-CONSISTENT AUGMENTATION (§4.2). One transform is sampled per
   *sample* and applied identically to t-g, t, t+g. Anything else injects fake
   motion and destroys the temporal signal the 9-channel design exists for.
   No mosaic, no mixup, no vertical flip (see `TemporalAugment` docstring).
4. COMPACT SAMPLE INDEX. Samples are held as parallel numpy arrays instead of a
   list of dicts. On Windows `num_workers>0` uses spawn, so the sample container
   is pickled into every worker; 394k dicts is ~275 MB *per worker*. The array
   form is ~15x smaller and pickles as a handful of buffers.
5. IN-MEMORY LABEL CACHE. Labels are pre-parsed once (see
   `3_prepare_training.py` -> `label_cache.npz`) so `__getitem__` performs three
   image reads and *zero* label-file reads -- a 25% cut in random I/O.
6. uint8 OUTPUT PATH. `out_dtype="uint8"` moves the /255 to the GPU, quartering
   host->device bandwidth. Float output remains the default for backwards
   compatibility with `2_verify_pipeline.py`.

`build_samples_from_manifest()` (list-of-dicts) is retained unchanged in
signature for the Review-1 scripts that call it.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# OpenCV spawns its own thread pool; inside DataLoader workers that oversubscribes
# the CPU badly (6 workers x 10 threads on a 10-core part). One thread each.
cv2.setNumThreads(0)

PAD_VALUE = 114                     # Ultralytics' letterbox grey
MICRO_THR = 0.0003                  # 0.03 % of frame area
SMALL_THR = 0.01                    # 1 % of frame area
CLASS_BIRD = 1


# --------------------------------------------------------------------------- #
#  Letterbox geometry
# --------------------------------------------------------------------------- #
def letterbox_params(src_h, src_w, dst_h, dst_w, scaleup=True):
    """Return (r, dw, dh, new_w, new_h) for an aspect-preserving centred fit."""
    r = min(dst_h / max(src_h, 1), dst_w / max(src_w, 1))
    if not scaleup:
        r = min(r, 1.0)
    new_w = int(round(src_w * r))
    new_h = int(round(src_h * r))
    dw = (dst_w - new_w) / 2.0
    dh = (dst_h - new_h) / 2.0
    return r, dw, dh, new_w, new_h


def letterbox_image(img, dst_h, dst_w, r, dw, dh, new_w, new_h,
                    pad_value=PAD_VALUE):
    """Resize + centre-pad an HxWxC uint8 image onto a dst_h x dst_w canvas."""
    if (img.shape[0], img.shape[1]) != (new_h, new_w):
        interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    top = int(round(dh - 0.1)); bottom = dst_h - new_h - top
    left = int(round(dw - 0.1)); right = dst_w - new_w - left
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT,
                                 value=(pad_value, pad_value, pad_value))
    return img


def letterbox_boxes(boxes, src_h, src_w, dst_h, dst_w, r, dw, dh):
    """Map normalised (cls,xc,yc,w,h) boxes from source space into letterbox space.

    Returns a NEW array; input is not modified. Empty in -> empty out."""
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    b = boxes.astype(np.float32, copy=True)
    b[:, 1] = (b[:, 1] * src_w * r + dw) / dst_w
    b[:, 2] = (b[:, 2] * src_h * r + dh) / dst_h
    b[:, 3] = (b[:, 3] * src_w * r) / dst_w
    b[:, 4] = (b[:, 4] * src_h * r) / dst_h
    return b


# --------------------------------------------------------------------------- #
#  Temporally-consistent augmentation
# --------------------------------------------------------------------------- #
class TemporalAugment:
    """One transform per sample, applied identically to t-g, t and t+g.

    What is deliberately ABSENT and why (measured, from a_inspection/):
      * Mosaic  -- tiles 4 images into one, shrinking every target ~2x. The
                   median CST target is 9x9 px; halving it is fatal.
      * MixUp   -- alpha-blends two scenes. Median target/background contrast on
                   thermal is 43.4; blending pushes micro targets under it.
      * V-flip  -- "sky above ground" is a real physical prior in every one of
                   the three corpora. Flipping it teaches a lie.
      * Per-frame independent jitter -- would inject motion that never happened
                   and poison the exact signal the 9-channel stack encodes.

    What is present and why:
      * H-flip            -- free 2x on data, no prior violated.
      * Affine (scale + translate) -- scale is biased ABOVE 1.0 so the sampler
                   never shrinks an already-micro target; translate directly
                   attacks the 90% centre-bias of gimbal-tracked Anti-UAV.
      * Brightness/contrast -- thermal crossover (target/background inversion
                   through the day) is a real IR failure mode.
      * Small-object copy-paste -- pastes an existing micro/small target, WITH
                   its own motion (the same source rect is lifted from all three
                   frames), at a new location. Feathered edges so the network
                   cannot key on a rectangular seam.
    """

    def __init__(self, hflip=0.5, scale=(0.90, 1.25), translate=0.10,
                 brightness=0.15, contrast=0.15, copy_paste=0.30,
                 copy_paste_max=3, noise_std=0.0, gray_p=0.0, seed=None):
        self.hflip = float(hflip)
        self.scale = tuple(scale)
        self.translate = float(translate)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.copy_paste = float(copy_paste)
        self.copy_paste_max = int(copy_paste_max)
        self.noise_std = float(noise_std)
        self.gray_p = float(gray_p)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed):
        """Give each DataLoader worker its own stream.

        Windows uses spawn, so every worker inherits a *copy* of this object --
        including its RNG state. Without a reseed all six workers would apply the
        identical augmentation sequence, silently cutting augmentation diversity
        by 6x while looking perfectly healthy."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        return self

    # -- geometry ---------------------------------------------------------- #
    def _affine(self, frames, boxes, H, W):
        s = self.rng.uniform(*self.scale)
        tx = self.rng.uniform(-self.translate, self.translate) * W
        ty = self.rng.uniform(-self.translate, self.translate) * H
        if abs(s - 1.0) < 1e-3 and abs(tx) < 0.5 and abs(ty) < 0.5:
            return frames, boxes
        cx, cy = W / 2.0, H / 2.0
        M = np.array([[s, 0.0, cx - s * cx + tx],
                      [0.0, s, cy - s * cy + ty]], dtype=np.float32)
        frames = [cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(PAD_VALUE,) * 3) for f in frames]
        if len(boxes):
            boxes = self._warp_boxes(boxes, M, H, W)
        return frames, boxes

    @staticmethod
    def _warp_boxes(boxes, M, H, W):
        """Affine-transform normalised xywh boxes; clip; drop over-cropped ones."""
        b = boxes.copy()
        x1 = (b[:, 1] - b[:, 3] / 2) * W; y1 = (b[:, 2] - b[:, 4] / 2) * H
        x2 = (b[:, 1] + b[:, 3] / 2) * W; y2 = (b[:, 2] + b[:, 4] / 2) * H
        corners = np.stack([x1, y1, x2, y1, x2, y2, x1, y2], 1).reshape(-1, 2)
        corners = corners @ M[:, :2].T + M[:, 2]
        corners = corners.reshape(-1, 4, 2)
        nx1 = corners[:, :, 0].min(1); nx2 = corners[:, :, 0].max(1)
        ny1 = corners[:, :, 1].min(1); ny2 = corners[:, :, 1].max(1)
        area_before = np.maximum(nx2 - nx1, 1e-6) * np.maximum(ny2 - ny1, 1e-6)
        cx1 = nx1.clip(0, W); cx2 = nx2.clip(0, W)
        cy1 = ny1.clip(0, H); cy2 = ny2.clip(0, H)
        cw = (cx2 - cx1).clip(min=0); ch = (cy2 - cy1).clip(min=0)
        keep = (cw > 1.0) & (ch > 1.0) & ((cw * ch) / area_before > 0.25)
        if not keep.any():
            return np.zeros((0, 5), dtype=np.float32)
        out = np.empty((int(keep.sum()), 5), dtype=np.float32)
        out[:, 0] = b[keep, 0]
        out[:, 1] = (cx1[keep] + cw[keep] / 2) / W
        out[:, 2] = (cy1[keep] + ch[keep] / 2) / H
        out[:, 3] = cw[keep] / W
        out[:, 4] = ch[keep] / H
        return out

    # -- small-object copy-paste ------------------------------------------- #
    def _copy_paste(self, frames, boxes, H, W):
        if not len(boxes):
            return frames, boxes
        area = boxes[:, 3] * boxes[:, 4]
        cand = np.flatnonzero(area < SMALL_THR)
        if not len(cand):
            return frames, boxes
        occupied = boxes[:, 1:5].copy()
        added = []
        frames = [f.copy() for f in frames]
        n = int(self.rng.integers(1, self.copy_paste_max + 1))
        for _ in range(n):
            src = boxes[self.rng.choice(cand)]
            pw = max(int(round(src[3] * W)), 4)
            ph = max(int(round(src[4] * H)), 4)
            sx = int(round(src[1] * W - pw / 2)); sy = int(round(src[2] * H - ph / 2))
            sx = min(max(sx, 0), W - pw); sy = min(max(sy, 0), H - ph)
            if pw >= W // 3 or ph >= H // 3:
                continue
            ok = False
            for _try in range(12):
                dx = int(self.rng.integers(0, W - pw))
                dy = int(self.rng.integers(0, H - ph))
                nb = np.array([(dx + pw / 2) / W, (dy + ph / 2) / H, pw / W, ph / H])
                if _iou_xywhn(nb, occupied).max(initial=0.0) < 1e-6:
                    ok = True
                    break
            if not ok:
                continue
            mask = _feather_mask(ph, pw)[..., None]
            for f in frames:
                patch = f[sy:sy + ph, sx:sx + pw].astype(np.float32)
                dst = f[dy:dy + ph, dx:dx + pw].astype(np.float32)
                f[dy:dy + ph, dx:dx + pw] = (patch * mask + dst * (1 - mask)).astype(np.uint8)
            new = np.array([src[0], (dx + pw / 2) / W, (dy + ph / 2) / H,
                            pw / W, ph / H], dtype=np.float32)
            added.append(new)
            occupied = np.vstack([occupied, new[1:5][None]])
        if added:
            boxes = np.vstack([boxes, np.stack(added)]).astype(np.float32)
        return frames, boxes

    # -- photometric ------------------------------------------------------- #
    def _photometric(self, frames):
        alpha = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
        beta = 255.0 * self.rng.uniform(-self.brightness, self.brightness)
        if abs(alpha - 1.0) < 1e-3 and abs(beta) < 1.0:
            return frames
        return [cv2.convertScaleAbs(f, alpha=alpha, beta=beta) for f in frames]

    def __call__(self, frames, boxes, H, W):
        frames, boxes = self._affine(frames, boxes, H, W)
        if self.rng.random() < self.hflip:
            frames = [f[:, ::-1] for f in frames]
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, 1] = 1.0 - boxes[:, 1]
        if self.rng.random() < self.copy_paste:
            frames, boxes = self._copy_paste(frames, boxes, H, W)
        frames = self._photometric(frames)
        if self.gray_p > 0 and self.rng.random() < self.gray_p:
            frames = [cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) for f in frames]
        if self.noise_std > 0:
            sig = self.noise_std * 255.0
            frames = [np.clip(f.astype(np.float32) +
                              self.rng.normal(0, sig, f.shape).astype(np.float32),
                              0, 255).astype(np.uint8) for f in frames]
        return frames, boxes


def _iou_xywhn(box, others):
    """IoU of one normalised xywh box against an [n,4] array (also xywh-norm)."""
    if others is None or len(others) == 0:
        return np.zeros(0, dtype=np.float32)
    bx1, by1 = box[0] - box[2] / 2, box[1] - box[3] / 2
    bx2, by2 = box[0] + box[2] / 2, box[1] + box[3] / 2
    ox1, oy1 = others[:, 0] - others[:, 2] / 2, others[:, 1] - others[:, 3] / 2
    ox2, oy2 = others[:, 0] + others[:, 2] / 2, others[:, 1] + others[:, 3] / 2
    iw = np.clip(np.minimum(bx2, ox2) - np.maximum(bx1, ox1), 0, None)
    ih = np.clip(np.minimum(by2, oy2) - np.maximum(by1, oy1), 0, None)
    inter = iw * ih
    union = box[2] * box[3] + others[:, 2] * others[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


_FEATHER_CACHE = {}


def _feather_mask(h, w, border=2):
    """Cosine-tapered alpha mask so a pasted patch has no rectangular seam."""
    key = (h, w, border)
    m = _FEATHER_CACHE.get(key)
    if m is None:
        my = np.ones(h, dtype=np.float32)
        mx = np.ones(w, dtype=np.float32)
        b = min(border, h // 2, w // 2)
        if b > 0:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, b + 2)[1:-1])
            my[:b] = ramp; my[-b:] = ramp[::-1]
            mx[:b] = ramp; mx[-b:] = ramp[::-1]
        m = np.outer(my, mx).astype(np.float32)
        if len(_FEATHER_CACHE) < 512:
            _FEATHER_CACHE[key] = m
    return m


# --------------------------------------------------------------------------- #
#  Sample construction (temporal windowing with boundary duplication)
# --------------------------------------------------------------------------- #
_MANIFEST_CACHE = {}


def load_manifest(manifest_csv):
    """Read manifest.csv once per process and memoise it.

    The manifest is ~200 MB / 764k rows; the split builder needs it three times
    (once per dataset) and the preflight suite several more. Re-parsing it each
    time costs minutes for nothing."""
    import pandas as pd
    p = Path(manifest_csv)
    key = (str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size)
    df = _MANIFEST_CACHE.get(key)
    if df is None:
        # Only the columns anything downstream reads. `n_frames` and `n_boxes`
        # are recomputed from the sequence grouping and the label cache, so
        # carrying them costs memory for nothing on 763,819 rows.
        use = ["dataset", "seq_key", "modality", "split", "frame_idx",
               "image_path", "label_path", "has_target", "materialized"]
        df = pd.read_csv(p, low_memory=False, usecols=use,
                         dtype={"image_path": str, "label_path": str,
                                "seq_key": str, "split": str, "modality": str})
        df["_row"] = np.arange(len(df), dtype=np.int64)
        _MANIFEST_CACHE.clear()          # only ever one manifest in play
        _MANIFEST_CACHE[key] = df
    return df


def build_split_samples(manifest_csv, splits_json, phase, datasets=None,
                        **kwargs):
    """Resolve one phase ("train"/"val"/"test") of `splits.json` to samples.

    This is the ONLY sanctioned way to build a training or validation sample
    list. It applies the per-dataset split map *and* the CST holdout filter, so
    the train and val sequence sets are disjoint by construction (roadmap T4).
    """
    spec = json.loads(Path(splits_json).read_text(encoding="utf-8"))
    if phase not in ("train", "val", "test"):
        raise ValueError(f"phase must be train/val/test, got {phase!r}")
    holdout = set(spec.get("cst_val_holdout_seqs", []))
    out = []
    for dk, sps in spec[phase].items():
        if datasets and dk not in datasets:
            continue
        kw = dict(kwargs)
        if dk == "cst" and holdout:
            if phase == "val":
                kw["seq_keys"] = holdout
            elif phase == "train":
                kw["exclude_seq_keys"] = holdout
        out.extend(build_samples_from_manifest(manifest_csv, datasets=[dk],
                                               splits=sps, **kw))
    return out


def assert_no_leakage(train_samples, val_samples, label="train/val"):
    """Roadmap T4. Mandatory before every run; cheap; catches the one bug that
    silently invalidates every number the project will ever report."""
    a = {s["seq_key"] for s in train_samples}
    b = {s["seq_key"] for s in val_samples}
    inter = a & b
    if inter:
        raise AssertionError(
            f"SPLIT LEAKAGE in {label}: {len(inter)} shared sequence(s), "
            f"e.g. {sorted(inter)[:5]}")
    return True


def build_samples_from_manifest(manifest_csv, datasets=None, splits=None,
                                require_image=True, max_per_seq=None,
                                frame_stride=1, temporal_gap=1,
                                seq_keys=None, exclude_seq_keys=None):
    """Read manifest.csv -> list of sample dicts with resolved t-g / t / t+g
    image paths and the label path for t.

    Neighbours are resolved *within a sequence* using position order, so the
    temporal window never crosses a sequence boundary.

    datasets        : keep only these `dataset` values (e.g. ["cst"])
    splits          : keep only these `split` values, OR a per-dataset dict
                      {"anti_uav": ["train"], "cst": ["train"]}. This is the
                      leakage fix -- see 00_TRAINING_ROADMAP.md §2.
    frame_stride    : keep every k-th CENTRE frame (neighbours are still the
                      true t+-g frames, so the temporal signal is untouched).
    temporal_gap    : g. 1 = the Review-1 t-1/t/t+1 stack. Larger g makes motion
                      visible on slow sequences (CST median is 0.64 px/frame).
    seq_keys        : optional explicit allow-list of sequence keys.
    exclude_seq_keys: optional deny-list (used to carve a CST val holdout).
    """
    df = load_manifest(manifest_csv)
    if datasets:
        df = df[df["dataset"].isin(datasets)]
    if splits:
        if isinstance(splits, dict):
            keep = np.zeros(len(df), dtype=bool)
            ds_vals = df["dataset"].to_numpy()
            sp_vals = df["split"].to_numpy()
            for dk, sps in splits.items():
                keep |= (ds_vals == dk) & np.isin(sp_vals, list(sps))
            df = df[keep]
        else:
            df = df[df["split"].isin(list(splits))]
    if seq_keys is not None:
        df = df[df["seq_key"].isin(set(seq_keys))]
    if exclude_seq_keys:
        df = df[~df["seq_key"].isin(set(exclude_seq_keys))]
    df["image_path"] = df["image_path"].fillna("").astype(str)
    if require_image:
        df = df[(df["materialized"] == 1) & (df["image_path"].str.len() > 0)]
    if len(df) == 0:
        return []
    df = df.sort_values(["dataset", "seq_key", "frame_idx"]).reset_index(drop=True)

    g = max(int(temporal_gap), 1)
    stride = max(int(frame_stride), 1)
    samples = []
    for seq_key, grp in df.groupby("seq_key", sort=False):
        grp = grp.sort_values("frame_idx")
        imgs = grp["image_path"].tolist()
        lbls = grp["label_path"].tolist()
        fidx = grp["frame_idx"].tolist()
        rows = grp["_row"].tolist()
        htgt = grp["has_target"].tolist()
        dset = grp["dataset"].iloc[0]
        split = str(grp["split"].iloc[0])
        modality = str(grp["modality"].iloc[0])
        M = len(imgs)
        positions = range(0, M, stride)
        if max_per_seq:
            positions = list(positions)[:max_per_seq]
        for pos in positions:
            samples.append({
                "dataset": dset,
                "seq_key": seq_key,
                "split": split,
                "modality": modality,
                "frame_idx": int(fidx[pos]),
                "seq_len": M,
                "prev_path": imgs[max(pos - g, 0)],           # t-g (dup at start)
                "cur_path": imgs[pos],                         # t
                "next_path": imgs[min(pos + g, M - 1)],       # t+g (dup at end)
                "label_path": lbls[pos],
                "manifest_row": int(rows[pos]),
                "has_target": int(htgt[pos]),
            })
    return samples


# --------------------------------------------------------------------------- #
#  Compact, spawn-friendly sample container
# --------------------------------------------------------------------------- #
class SampleIndex:
    """Parallel-array form of a sample list.

    Pickling a list of 394k dicts into six Windows spawn workers costs ~1.7 GB of
    duplicated RAM. This holds the same information in a handful of numpy buffers
    (~110 MB total) and is what the Dataset actually iterates.
    """

    def __init__(self, samples, label_cache=None):
        n = len(samples)
        uniq = {}
        prev_i = np.empty(n, dtype=np.int32)
        cur_i = np.empty(n, dtype=np.int32)
        next_i = np.empty(n, dtype=np.int32)

        def _idx(p):
            j = uniq.get(p)
            if j is None:
                j = len(uniq)
                uniq[p] = j
            return j

        ds_names, mod_names, split_names, seq_names = {}, {}, {}, {}
        ds_code = np.empty(n, dtype=np.int16)
        mod_code = np.empty(n, dtype=np.int16)
        split_code = np.empty(n, dtype=np.int16)
        seq_code = np.empty(n, dtype=np.int32)
        frame_idx = np.empty(n, dtype=np.int32)
        seq_len = np.empty(n, dtype=np.int32)
        rows = np.full(n, -1, dtype=np.int64)

        def _code(d, v):
            c = d.get(v)
            if c is None:
                c = len(d)
                d[v] = c
            return c

        for i, s in enumerate(samples):
            prev_i[i] = _idx(s["prev_path"])
            cur_i[i] = _idx(s["cur_path"])
            next_i[i] = _idx(s["next_path"])
            ds_code[i] = _code(ds_names, s.get("dataset", ""))
            mod_code[i] = _code(mod_names, s.get("modality", ""))
            split_code[i] = _code(split_names, s.get("split", ""))
            seq_code[i] = _code(seq_names, s.get("seq_key", ""))
            frame_idx[i] = int(s.get("frame_idx", 0))
            seq_len[i] = int(s.get("seq_len", 1))
            rows[i] = int(s.get("manifest_row", -1))

        self.paths = np.array(list(uniq.keys()), dtype=np.bytes_)
        self.prev_i, self.cur_i, self.next_i = prev_i, cur_i, next_i
        self.ds_names = _inv(ds_names); self.ds_code = ds_code
        self.mod_names = _inv(mod_names); self.mod_code = mod_code
        self.split_names = _inv(split_names); self.split_code = split_code
        self.seq_names = np.array(_inv(seq_names), dtype=np.bytes_)
        self.seq_code = seq_code
        self.frame_idx = frame_idx
        self.seq_len = seq_len
        self.manifest_row = rows

        # -- labels: either from the prebuilt cache, or read now -------------- #
        self.lbl_off, self.lbl_box = _resolve_labels(samples, rows, label_cache)

    def __len__(self):
        return len(self.cur_i)

    def path(self, i, which="cur"):
        arr = {"prev": self.prev_i, "cur": self.cur_i, "next": self.next_i}[which]
        return self.paths[arr[i]].decode("utf-8", "replace")

    def boxes(self, i):
        a, b = self.lbl_off[i], self.lbl_off[i + 1]
        return self.lbl_box[a:b]

    def seq_key(self, i):
        return self.seq_names[self.seq_code[i]].decode("utf-8", "replace")

    def dataset(self, i):
        return self.ds_names[self.ds_code[i]]

    def modality(self, i):
        return self.mod_names[self.mod_code[i]]

    # -- derived, vectorised statistics used by the sampler & evaluator ----- #
    def box_counts(self):
        return np.diff(self.lbl_off).astype(np.int32)

    def min_areas(self):
        """Smallest normalised box area per sample (1.0 when the frame is empty)."""
        out = np.ones(len(self), dtype=np.float32)
        areas = self.lbl_box[:, 3] * self.lbl_box[:, 4]
        for i in range(len(self)):
            a, b = self.lbl_off[i], self.lbl_off[i + 1]
            if b > a:
                out[i] = areas[a:b].min()
        return out

    def has_class(self, cls):
        out = np.zeros(len(self), dtype=bool)
        cl = self.lbl_box[:, 0]
        for i in range(len(self)):
            a, b = self.lbl_off[i], self.lbl_off[i + 1]
            if b > a:
                out[i] = bool((cl[a:b] == cls).any())
        return out


def _inv(d):
    out = [None] * len(d)
    for k, v in d.items():
        out[v] = k
    return out


def _read_label_file(path):
    boxes = []
    p = Path(path)
    try:
        if p.stat().st_size == 0:
            return boxes
    except OSError:
        return boxes
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            a = ln.split()
            if len(a) >= 5:
                boxes.append((float(a[0]), float(a[1]), float(a[2]),
                              float(a[3]), float(a[4])))
    return boxes


def _resolve_labels(samples, rows, label_cache):
    """Return (offsets[n+1], boxes[m,5]) for the samples, from cache if possible.

    When no cache is passed, `data_unified/label_cache.npz` is looked up
    automatically. Without it the fallback reads every label file from disk --
    755,989 tiny random reads for the full corpus, which is ~10 minutes of dead
    time that silently turns a quick script into a stalled one."""
    n = len(samples)
    if label_cache is None:
        label_cache = load_label_cache()
    if label_cache is not None and len(rows) and (rows >= 0).all():
        c_off, c_box = label_cache
        if rows.max() < len(c_off) - 1:
            counts = (c_off[rows + 1] - c_off[rows]).astype(np.int64)
            off = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(counts, out=off[1:])
            box = np.empty((int(off[-1]), 5), dtype=np.float32)
            for i in range(n):
                k = counts[i]
                if k:
                    box[off[i]:off[i + 1]] = c_box[c_off[rows[i]]:c_off[rows[i] + 1]]
            return off, box
    # fall back: read every label file once
    off = np.zeros(n + 1, dtype=np.int64)
    chunks = []
    for i, s in enumerate(samples):
        b = _read_label_file(s["label_path"])
        off[i + 1] = off[i] + len(b)
        if b:
            chunks.append(np.asarray(b, dtype=np.float32))
    box = (np.concatenate(chunks, 0) if chunks
           else np.zeros((0, 5), dtype=np.float32))
    return off, box


DEFAULT_LABEL_CACHE = Path(__file__).resolve().parent / "data_unified" / "label_cache.npz"
_LABEL_CACHE_MEMO = {}


def load_label_cache(path=None):
    """Load `label_cache.npz` -> (offsets, boxes) or None if absent/invalid.

    Memoised per path: the cache is a few MB but every caller would otherwise
    decompress it again."""
    p = Path(path or DEFAULT_LABEL_CACHE)
    key = str(p)
    if key in _LABEL_CACHE_MEMO:
        return _LABEL_CACHE_MEMO[key]
    out = None
    if p.exists():
        try:
            z = np.load(p)
            out = (z["offsets"].astype(np.int64), z["boxes"].astype(np.float32))
        except Exception:
            out = None
    _LABEL_CACHE_MEMO[key] = out
    return out


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class AeroTrackDataset(Dataset):
    """Temporal 3-frame-stacking dataset -> (9x640x640 tensor, [N,5] targets)."""

    def __init__(self, samples, img_size=(640, 640), return_meta=False,
                 augment=None, letterbox=True, out_dtype="float32",
                 label_cache=None):
        """
        samples    : list of dicts from build_samples_from_manifest(), or a
                     ready-made SampleIndex.
        img_size   : (H, W) canvas.
        augment    : TemporalAugment instance, or None for eval.
        letterbox  : True (default) preserves aspect ratio; False reproduces the
                     Review-1 stretch behaviour, for the letterbox ablation.
        out_dtype  : "float32" -> tensor in [0,1] (Review-1 contract);
                     "uint8"   -> raw tensor, caller divides by 255 on the GPU.
        """
        self.index = (samples if isinstance(samples, SampleIndex)
                      else SampleIndex(samples, label_cache=label_cache))
        self.img_h, self.img_w = int(img_size[0]), int(img_size[1])
        self.return_meta = return_meta
        self.augment = augment
        self.letterbox = bool(letterbox)
        self.out_dtype = out_dtype
        assert out_dtype in ("float32", "uint8")

    # backwards-compatible view: `dataset.samples` used to be a list of dicts.
    @property
    def samples(self):
        return self.index

    def __len__(self):
        return len(self.index)

    # -- raw frame read (BGR uint8, native size) --------------------------- #
    def _imread(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:                                   # unreadable -> grey
            img = np.full((self.img_h, self.img_w, 3), PAD_VALUE, dtype=np.uint8)
        return img

    def __getitem__(self, i):
        idx = self.index
        cur = self._imread(idx.path(i, "cur"))
        prv = self._imread(idx.path(i, "prev"))
        nxt = self._imread(idx.path(i, "next"))
        src_h, src_w = cur.shape[:2]
        boxes = idx.boxes(i)
        # Area as a fraction of the ORIGINAL frame, before letterboxing. The
        # micro/small/standard thresholds were measured on source frames, and
        # letterboxing rescales normalised area (640x512 -> 640x640 multiplies
        # h_norm by 0.8), so stratifying on post-letterbox area would quietly
        # move every target one bin smaller.
        src_area = ((boxes[:, 3] * boxes[:, 4]).astype(np.float32)
                    if len(boxes) else np.zeros(0, dtype=np.float32))

        H, W = self.img_h, self.img_w
        if self.letterbox:
            r, dw, dh, nw, nh = letterbox_params(src_h, src_w, H, W)
            frames = []
            for f in (prv, cur, nxt):
                if f.shape[:2] != (src_h, src_w):
                    # Defensive: a neighbour that decoded at a different size
                    # (never observed in this corpus) gets its own geometry.
                    fr, fdw, fdh, fnw, fnh = letterbox_params(f.shape[0], f.shape[1], H, W)
                    frames.append(letterbox_image(f, H, W, fr, fdw, fdh, fnw, fnh))
                else:
                    frames.append(letterbox_image(f, H, W, r, dw, dh, nw, nh))
            boxes = letterbox_boxes(boxes, src_h, src_w, H, W, r, dw, dh)
        else:
            frames = [cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
                      if f.shape[:2] != (H, W) else f for f in (prv, cur, nxt)]
            boxes = boxes.astype(np.float32, copy=True)

        if self.augment is not None:
            frames, boxes = self.augment(frames, boxes, H, W)

        # BGR -> RGB and fuse into one HxWx9 buffer, then to CHW.
        stack = np.concatenate([f[:, :, ::-1] for f in frames], axis=2)
        stack = np.ascontiguousarray(stack.transpose(2, 0, 1))       # [9,H,W]
        t = torch.from_numpy(stack)
        if self.out_dtype == "float32":
            t = t.float().div_(255.0)

        targets = (torch.from_numpy(np.ascontiguousarray(boxes))
                   if len(boxes) else torch.zeros((0, 5), dtype=torch.float32))
        if self.return_meta:
            return t, targets, {"seq_key": idx.seq_key(i),
                                "frame_idx": int(idx.frame_idx[i]),
                                "dataset": idx.dataset(i),
                                "modality": idx.modality(i),
                                "src_area": src_area,
                                "index": i}
        return t, targets


# --------------------------------------------------------------------------- #
#  Collate (variable-length targets) + convenience loader
# --------------------------------------------------------------------------- #
def aerotrack_collate(batch):
    """Stack the fixed-shape 9ch tensors; keep targets as a list of [Ni,5]."""
    has_meta = len(batch[0]) == 3
    imgs = torch.stack([b[0] for b in batch], dim=0)      # [B,9,640,640]
    targets = [b[1] for b in batch]
    if has_meta:
        return imgs, targets, [b[2] for b in batch]
    return imgs, targets


def make_dataloader(samples, batch_size=4, img_size=(640, 640), shuffle=True,
                    num_workers=0, return_meta=False, **ds_kwargs):
    from torch.utils.data import DataLoader
    ds = AeroTrackDataset(samples, img_size=img_size, return_meta=return_meta,
                          **ds_kwargs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers, collate_fn=aerotrack_collate)
    return ds, dl


if __name__ == "__main__":
    mani = Path(__file__).resolve().parent / "data_unified" / "manifest.csv"
    if mani.exists():
        smp = build_samples_from_manifest(mani, datasets=["cst"], max_per_seq=3)
        print(f"built {len(smp)} sample records from CST")
        if smp:
            ds = AeroTrackDataset(smp[:8])
            x, y = ds[0]
            print("sample tensor:", tuple(x.shape), "targets:", tuple(y.shape))
            ds_a = AeroTrackDataset(smp[:8], augment=TemporalAugment(seed=0))
            xa, ya = ds_a[0]
            print("augmented    :", tuple(xa.shape), "targets:", tuple(ya.shape))
    else:
        print("manifest.csv not found yet — run 1_standardize_annotations.py first")
