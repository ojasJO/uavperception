#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py  --  AeroTrack-Net evaluation suite
=================================================
Plain mAP is not enough for this project, and saying so is the point.

Aggregate mAP is dominated by large, centred, easy targets -- exactly the
population Review 1 argued is *not* the hard part. A model that detects every
40 px gimbal-centred drone and zero 9 px ones can post a respectable mAP@50
while being useless as a counter-UAS system. So this module reports the
breakdown that actually tests the thesis (00_TRAINING_ROADMAP.md §7.2):

  1. Size-stratified AP   micro (<0.03% frame) / small (0.03-1%) / standard (>1%)
                          THE headline number. If SPD-Conv works, it shows here.
  2. Per-dataset AP       Anti-UAV / CST / Det-Fly separately. A large Anti-UAV
                          vs CST gap means the model learned "look in the middle"
                          from gimbal-tracked footage.
  3. Per-modality AP      IR vs RGB on the paired Anti-UAV sequences.
  4. Bird FP rate         false positives per image on bird-only Det-Fly images.
  5. Empty-frame FP rate  detections on `exist = 0` frames -- the model
                          hallucinating a target through an occlusion.
  6. PR curves + confusion matrix.

Everything is computed from one pass of stored per-image (pred, gt) records, so
every stratum is scored against exactly the same detections.

AP itself is implemented here (all-point interpolation, COCO convention) rather
than imported, so the numbers do not move when Ultralytics refactors its
metrics module -- and so stratified "ignore" semantics can be expressed
correctly, which the stock helper cannot do.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import torch

MICRO_THR = 0.0003
SMALL_THR = 0.01
NAMES = {0: "uav_drone", 1: "bird_hard_negative"}
SIZE_BINS = {
    "micro":    (0.0, MICRO_THR),
    "small":    (MICRO_THR, SMALL_THR),
    "standard": (SMALL_THR, 1.01),
}


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def xywhn_to_xyxy(boxes, w, h):
    """[n,4] normalised cx,cy,w,h -> [n,4] pixel x1,y1,x2,y2."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    b = np.empty((len(boxes), 4), dtype=np.float32)
    b[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    b[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    b[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    b[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
    return b


def box_iou_np(a, b):
    """[n,4] x [m,4] xyxy -> [n,m] IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


# --------------------------------------------------------------------------- #
#  AP (all-point interpolation, per class, over an IoU grid)
# --------------------------------------------------------------------------- #
def compute_ap(recall, precision):
    """All-point-interpolated AP. `recall` ascending."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    i = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])), mrec, mpre


def ap_per_class(tp, conf, pred_cls, target_cls, classes=(0, 1), eps=1e-12):
    """tp [n_pred, n_iou] bool -> dict with per-class AP arrays and PR curves."""
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    n_iou = tp.shape[1] if tp.ndim == 2 else 1
    out = {"classes": [], "ap": [], "p": [], "r": [], "f1": [], "curves": {},
           "n_gt": [], "n_pred": []}
    for c in classes:
        n_gt = int((target_cls == c).sum())
        m = pred_cls == c
        n_p = int(m.sum())
        out["classes"].append(int(c))
        out["n_gt"].append(n_gt)
        out["n_pred"].append(n_p)
        if n_gt == 0 or n_p == 0:
            out["ap"].append(np.zeros(n_iou))
            out["p"].append(0.0); out["r"].append(0.0); out["f1"].append(0.0)
            continue
        tpc = tp[m].cumsum(0)
        fpc = (1 - tp[m]).cumsum(0)
        recall = tpc / (n_gt + eps)
        precision = tpc / np.maximum(tpc + fpc, eps)
        aps = np.zeros(n_iou)
        for j in range(n_iou):
            aps[j], mrec, mpre = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                out["curves"][int(c)] = {"recall": mrec.tolist(),
                                         "precision": mpre.tolist(),
                                         "conf": conf[m].tolist()}
        out["ap"].append(aps)
        # P / R / F1 at the confidence that maximises F1 (IoU 0.5)
        f1 = 2 * precision[:, 0] * recall[:, 0] / np.maximum(precision[:, 0] + recall[:, 0], eps)
        k = int(np.argmax(f1))
        out["p"].append(float(precision[k, 0]))
        out["r"].append(float(recall[k, 0]))
        out["f1"].append(float(f1[k]))
    out["ap"] = np.stack(out["ap"]) if out["ap"] else np.zeros((0, n_iou))
    return out


# --------------------------------------------------------------------------- #
#  Per-image matching, with COCO "ignore" semantics for size strata
# --------------------------------------------------------------------------- #
def match_image(pred, gt, iouv, gt_keep=None):
    """Greedy IoU matching of one image's detections against its ground truth.

    pred     [n,6]  x1,y1,x2,y2,conf,cls   (pixel coords on the network canvas)
    gt       [m,5]  cls,x1,y1,x2,y2
    gt_keep  [m]    bool. GT outside the stratum are IGNORED, not counted as
                    misses: a detection that lands on an ignored GT is DROPPED
                    (neither TP nor FP), which is the COCO area-range rule. A
                    detection matching nothing is still a false positive.

    Returns (tp[n_kept, n_iou] bool, conf[n_kept], pred_cls[n_kept],
             target_cls[m_kept]).
    """
    n_iou = len(iouv)
    if gt_keep is None:
        gt_keep = np.ones(len(gt), dtype=bool)
    gt_in = gt[gt_keep]
    gt_out = gt[~gt_keep]
    tcls = gt_in[:, 0].copy() if len(gt_in) else np.zeros(0, dtype=np.float32)

    if len(pred) == 0:
        return (np.zeros((0, n_iou), dtype=bool), np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32), tcls)

    # drop detections that belong to an ignored GT (same class, IoU >= 0.5)
    if len(gt_out):
        iou_out = box_iou_np(pred[:, :4], gt_out[:, 1:5])
        same = pred[:, 5][:, None] == gt_out[:, 0][None, :]
        drop = ((iou_out >= iouv[0]) & same).any(1)
        pred = pred[~drop]
    if len(pred) == 0:
        return (np.zeros((0, n_iou), dtype=bool), np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32), tcls)

    tp = np.zeros((len(pred), n_iou), dtype=bool)
    if len(gt_in):
        iou = box_iou_np(pred[:, :4], gt_in[:, 1:5])
        same = pred[:, 5][:, None] == gt_in[:, 0][None, :]
        for j, thr in enumerate(iouv):
            ok = np.argwhere((iou >= thr) & same)
            if not len(ok):
                continue
            scores = iou[ok[:, 0], ok[:, 1]]
            ok = ok[np.argsort(-scores)]
            # one detection per GT, one GT per detection
            _, keep_gt = np.unique(ok[:, 1], return_index=True)
            ok = ok[np.sort(keep_gt)]
            _, keep_pd = np.unique(ok[:, 0], return_index=True)
            ok = ok[np.sort(keep_pd)]
            tp[ok[:, 0], j] = True
    return tp, pred[:, 4].astype(np.float32), pred[:, 5].astype(np.float32), tcls


# --------------------------------------------------------------------------- #
#  Inference pass -> per-image records
# --------------------------------------------------------------------------- #
def _nms_input(preds, topk=512, to_cpu=True):
    """[B, 4+nc, A] predictions -> the tensor NMS should actually see.

    Keeps the `topk` anchors with the highest class score per image and, by
    default, moves them to the CPU. Always returns a NEW tensor: Ultralytics'
    NMS converts boxes to xyxy in place, and `.float()` on an fp32 tensor is a
    no-op that would let it scribble on the model's own output."""
    x = preds.float()
    if topk and x.shape[2] > topk:
        cls = x[:, 4:, :].amax(1)                       # best class score/anchor
        idx = cls.topk(int(topk), dim=1).indices        # [B, topk]
        x = torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))
    elif not to_cpu:
        x = x.clone()
    return x.cpu() if to_cpu else x



@torch.no_grad()
def collect_records(model, loader, device, imgsz=640, conf=0.001, iou=0.7,
                    max_det=30, amp_dtype=None, progress=True, uint8_input=False,
                    nms_topk=512, nms_on_cpu=True):
    """Run the model over a meta-returning loader; keep raw preds + GT per image.

    Two decisions in here are MEASURED, not stylistic, because between them they
    are the difference between a 15-hour training run and a 45-hour one --
    validation, not training, was the bottleneck.

    **1. NMS runs on the CPU.** Ultralytics' `non_max_suppression` is a Python
    loop over the batch doing a dozen small indexing operations per image; on
    CUDA each one is a kernel launch and a synchronisation, and the launch
    overhead dwarfs the work. Measured on 400 real validation images:

        conf 0.001   GPU  3.8 img/s      CPU  20.5 img/s     (5.4x)
        conf 0.005   GPU  211  img/s     CPU  713  img/s     (3.4x)

    **2. Only the top `nms_topk` anchors per image enter NMS.** The cost scales
    with how many of the 8,400 anchors clear `conf`, and that is worst exactly
    when the model is worst: an untrained network puts ~2,000 anchors over
    0.001, a trained one a handful. `max_det` is 30 and this corpus averages
    0.95 boxes per image, so ranks past a few hundred cannot reach the output --
    and the measurement agrees, to six decimal places:

        conf 0.001, CPU:  no prefilter 20.5 img/s  mAP50 0.124377
                          top-512      21.5 img/s  mAP50 0.124377  (identical)
                          top-256      42.9 img/s  mAP50 0.124377  (identical)

    512 is the shipped default: it is twice the margin that was already exact,
    and the prefilter binds less as the model sharpens, so it gets safer with
    training rather than riskier. `nms_topk=0` disables it.

    Note that Ultralytics' NMS rewrites its input tensor IN PLACE (xywh ->
    xyxy). Both paths here hand it a fresh tensor, so the model's output is
    never silently mutated under a caller that wanted to reuse it.
    """
    from ultralytics.utils.nms import non_max_suppression
    # Ultralytics gives NMS a WALL-CLOCK budget of `2.0 + max_time_img * batch`
    # seconds and, when it expires, RETURNS EARLY -- every remaining image in
    # that batch silently comes back with zero detections. That is a metric bug,
    # not a slowdown: an early-epoch model puts most of its 8,400 anchors above
    # a 0.005 threshold, the budget blows, and the images at the end of each
    # batch are scored as if the detector had output nothing. Measured on a
    # 2-epoch checkpoint: 3 of 320 images lost every detection at batch 32.
    # Give it a budget it cannot hit; correctness first, and the flood subsides
    # on its own once the classifier separates.
    _sig = inspect.signature(non_max_suppression).parameters
    _nms_kw = {}
    if "max_time_img" in _sig:
        _nms_kw["max_time_img"] = 2.0
    model.eval()
    records = []
    n_done = 0
    for batch in loader:
        imgs, targets, metas = batch
        imgs = imgs.to(device, non_blocking=True)
        if uint8_input or imgs.dtype == torch.uint8:
            imgs = imgs.float().div_(255.0)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                preds = model(imgs)
        else:
            preds = model(imgs)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        dets = non_max_suppression(_nms_input(preds, nms_topk, nms_on_cpu),
                                   conf, iou, max_det=max_det, **_nms_kw)
        for k, det in enumerate(dets):
            d = det.detach().cpu().numpy().astype(np.float32) if len(det) else \
                np.zeros((0, 6), dtype=np.float32)
            t = targets[k].numpy()
            gt = np.zeros((len(t), 5), dtype=np.float32)
            if len(t):
                gt[:, 0] = t[:, 0]
                gt[:, 1:5] = xywhn_to_xyxy(t[:, 1:5], imgsz, imgsz)
            records.append({
                "pred": d, "gt": gt,
                # Prefer the SOURCE-frame area (carried in the meta) over the
                # letterboxed one: the micro/small thresholds were measured on
                # source frames in Review 1 and must mean the same thing here.
                "gt_area": np.asarray(
                    metas[k].get("src_area") if metas[k].get("src_area") is not None
                    and len(metas[k]["src_area"]) == len(t)
                    else ((t[:, 3] * t[:, 4]) if len(t) else np.zeros(0)),
                    dtype=np.float32),
                "dataset": metas[k]["dataset"],
                "modality": metas[k]["modality"],
                "seq_key": metas[k]["seq_key"],
                "frame_idx": metas[k]["frame_idx"],
            })
        n_done += len(dets)
        if progress and n_done % 5000 < len(dets):
            print(f"  [eval] {n_done:,} images", flush=True)
    return records


# --------------------------------------------------------------------------- #
#  Scoring
# --------------------------------------------------------------------------- #
def score(records, iouv=None, size_bin=None, classes=(0, 1)):
    """Score a set of records, optionally restricted to one GT size bin."""
    if iouv is None:
        iouv = np.linspace(0.5, 0.95, 10)
    lo, hi = SIZE_BINS[size_bin] if size_bin else (None, None)
    TP, CF, PC, TC = [], [], [], []
    for r in records:
        keep = None
        if size_bin:
            keep = (r["gt_area"] >= lo) & (r["gt_area"] < hi)
            if len(r["gt"]) and not keep.any() and len(r["pred"]) == 0:
                continue
        tp, cf, pc, tc = match_image(r["pred"], r["gt"], iouv, keep)
        TP.append(tp); CF.append(cf); PC.append(pc); TC.append(tc)
    if not TP:
        return None
    tp = np.concatenate(TP, 0) if len(TP) else np.zeros((0, len(iouv)), bool)
    res = ap_per_class(tp, np.concatenate(CF), np.concatenate(PC),
                       np.concatenate(TC), classes=classes)
    ap = res["ap"]
    present = [i for i, c in enumerate(res["classes"]) if res["n_gt"][i] > 0]
    res["map50"] = float(ap[present, 0].mean()) if present else 0.0
    res["map5095"] = float(ap[present].mean()) if present else 0.0
    res["n_images"] = len(TP)
    res["n_gt_total"] = int(sum(res["n_gt"]))
    return res


def _flat(res, keys=("map50", "map5095")):
    return {k: (round(res[k], 5) if res else None) for k in keys}


def fp_rate(records, predicate, conf_thres=0.25):
    """False positives per image over the subset selected by `predicate`.

    A detection on a frame whose ground truth contains no drone is a false
    positive by construction -- this is the number that decides whether the
    system cries wolf at a bird or hallucinates through an occlusion."""
    n_img = n_fp = 0
    for r in records:
        if not predicate(r):
            continue
        n_img += 1
        if len(r["pred"]):
            n_fp += int((r["pred"][:, 4] >= conf_thres).sum())
    return {"images": n_img, "false_positives": n_fp,
            "fp_per_image": (n_fp / n_img) if n_img else None,
            "conf_thres": conf_thres}


def full_report(records, save_dir=None, conf_thres=0.25, classes=(0, 1)):
    """The complete §7.2 suite as a JSON-serialisable dict."""
    rep = {"n_images": len(records)}
    overall = score(records, classes=classes)
    rep["overall"] = _flat(overall)
    rep["overall"]["per_class_ap50"] = {
        NAMES.get(c, c): round(float(overall["ap"][i, 0]), 5)
        for i, c in enumerate(overall["classes"])}
    rep["overall"]["per_class_ap5095"] = {
        NAMES.get(c, c): round(float(overall["ap"][i].mean()), 5)
        for i, c in enumerate(overall["classes"])}
    rep["overall"]["precision"] = dict(zip(
        [NAMES.get(c, c) for c in overall["classes"]],
        [round(v, 5) for v in overall["p"]]))
    rep["overall"]["recall"] = dict(zip(
        [NAMES.get(c, c) for c in overall["classes"]],
        [round(v, 5) for v in overall["r"]]))

    # 1. size-stratified -- the headline
    rep["by_size"] = {}
    for b in SIZE_BINS:
        r = score(records, size_bin=b, classes=classes)
        rep["by_size"][b] = _flat(r)
        rep["by_size"][b]["n_gt"] = r["n_gt_total"] if r else 0

    # 2. per dataset
    rep["by_dataset"] = {}
    for dk in sorted({r["dataset"] for r in records}):
        sub = [r for r in records if r["dataset"] == dk]
        r = score(sub, classes=classes)
        rep["by_dataset"][dk] = _flat(r)
        rep["by_dataset"][dk]["n_images"] = len(sub)
        rep["by_dataset"][dk]["micro_ap50"] = _flat(
            score(sub, size_bin="micro", classes=classes), ("map50",))["map50"]

    # 3. per modality (the paired IR<->RGB comparison lives in anti_uav)
    rep["by_modality"] = {}
    for m in sorted({r["modality"] for r in records if r["dataset"] == "anti_uav"}):
        sub = [r for r in records if r["dataset"] == "anti_uav" and r["modality"] == m]
        rep["by_modality"][m] = _flat(score(sub, classes=classes))
        rep["by_modality"][m]["n_images"] = len(sub)

    # 4. bird false positives: Det-Fly images whose GT holds only birds
    rep["bird_fp"] = fp_rate(
        records,
        lambda r: r["dataset"] == "det_fly" and len(r["gt"]) > 0
                  and not (r["gt"][:, 0] == 0).any(),
        conf_thres)

    # 5. empty-frame false positives: exist=0 frames (0-byte label files)
    rep["empty_frame_fp"] = fp_rate(records, lambda r: len(r["gt"]) == 0,
                                    conf_thres)

    # 6. curves
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        _plot_pr(overall, save_dir / "PR_curve.png")
        _plot_size_bars(rep["by_size"], save_dir / "AP_by_size.png")
        (save_dir / "evaluation_report.json").write_text(
            json.dumps(rep, indent=2), encoding="utf-8")
    return rep


# --------------------------------------------------------------------------- #
#  Plots
# --------------------------------------------------------------------------- #
def _plot_pr(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, c in enumerate(res["classes"]):
        cur = res["curves"].get(int(c))
        if not cur:
            continue
        ax.plot(cur["recall"], cur["precision"], lw=2.2,
                label=f"{NAMES.get(c, c)}  AP@50={res['ap'][i,0]:.3f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend()
    ax.set_title("AeroTrack-Net — Precision/Recall @ IoU 0.50", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _plot_size_bars(by_size, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = list(by_size.keys())
    v50 = [by_size[b]["map50"] or 0 for b in bins]
    v95 = [by_size[b]["map5095"] or 0 for b in bins]
    x = np.arange(len(bins)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - w / 2, v50, w, label="mAP@50", color="#2a78d6")
    ax.bar(x + w / 2, v95, w, label="mAP@50-95", color="#a3271a")
    for xi, (a, b) in enumerate(zip(v50, v95)):
        ax.text(xi - w / 2, a + .01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, b + .01, f"{b:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={by_size[b]['n_gt']:,})" for b in bins])
    ax.set_ylabel("AP"); ax.legend(); ax.grid(axis="y", alpha=.3)
    ax.set_title("AP by target size — the SPD-Conv thesis test", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# --------------------------------------------------------------------------- #
def print_report(rep):
    """Human-readable dump of full_report()."""
    p = print
    p("=" * 72)
    p(f"AeroTrack-Net evaluation — {rep['n_images']:,} images")
    p("=" * 72)
    o = rep["overall"]
    p(f"  mAP@50      {o['map50']:.4f}      mAP@50-95   {o['map5095']:.4f}")
    p(f"  per-class AP@50   {o['per_class_ap50']}")
    p(f"  precision         {o['precision']}")
    p(f"  recall            {o['recall']}")
    p("-" * 72)
    p("  BY TARGET SIZE  (the headline result)")
    for b, v in rep["by_size"].items():
        p(f"    {b:<9s} n_gt={v['n_gt']:>8,}  mAP@50={v['map50']}  "
          f"mAP@50-95={v['map5095']}")
    p("-" * 72)
    p("  BY DATASET")
    for d, v in rep["by_dataset"].items():
        p(f"    {d:<9s} n={v['n_images']:>7,}  mAP@50={v['map50']}  "
          f"micro AP@50={v['micro_ap50']}")
    p("  BY MODALITY (Anti-UAV paired scenes)")
    for m, v in rep["by_modality"].items():
        p(f"    {m:<9s} n={v['n_images']:>7,}  mAP@50={v['map50']}")
    p("-" * 72)
    b = rep["bird_fp"]; e = rep["empty_frame_fp"]
    p(f"  bird-only images   : {b['images']:,} imgs, {b['false_positives']:,} FP "
      f"-> {b['fp_per_image']} FP/img @conf {b['conf_thres']}")
    p(f"  empty (exist=0)    : {e['images']:,} imgs, {e['false_positives']:,} FP "
      f"-> {e['fp_per_image']} FP/img @conf {e['conf_thres']}")
    p("=" * 72)
