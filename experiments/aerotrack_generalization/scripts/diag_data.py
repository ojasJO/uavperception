#!/usr/bin/env python
"""D1 — data diagnostics: split structure and leakage risk, neighbour adjacency, drone size at the network input, per-frame
target motion vs camera motion, label flicker, Det-Fly near-duplicates train<->val.

    venv\\Scripts\\python.exe experiments\\aerotrack_generalization\\scripts\\diag_data.py

Reads train + val content. Test: sequence NAMES only (session / scene-category composition), never images or labels.
Output: reports/diag_data.json (+ one ledger line)."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import _series as S
import cv2
import numpy as np

from stacked_dataset import build_split_samples, load_label_cache, load_manifest
from train import LABEL_CACHE_PATH, MANIFEST, SPLITS, subsample

from anti_uav_repro.utils.repro import file_sha256


def main():
    out = {}
    rng = np.random.default_rng(0)
    spec = json.loads(Path(SPLITS).read_text(encoding="utf-8"))
    man = load_manifest(MANIFEST)
    hold = set(spec["cst_val_holdout_seqs"])

    # -- 1. split structure (names only for test)
    def phase_seqs(phase, ds):
        m = man[(man.dataset == ds) & (man.split.isin(spec[phase][ds]))]
        keys = set(m.seq_key.unique())
        if ds == "cst":
            keys = keys & hold if phase == "val" else (keys - hold if phase == "train" else keys)
        return keys
    seqs = {p: {d: phase_seqs(p, d) for d in ("anti_uav", "cst", "det_fly")} for p in ("train", "val", "test")}

    def au_session(k):
        m = re.match(r"antiuav_[a-z]+_(\d{8}_\d{6})_(\d+)_(\d+)__(\w+)", k)
        return m.group(1) if m else None
    sess = {p: {au_session(k) for k in seqs[p]["anti_uav"]} for p in seqs}
    out["anti_uav"] = {"sequences": {p: len(seqs[p]["anti_uav"]) for p in seqs}, "recording_sessions": {p: len(sess[p]) for p in sess},
                       "sessions_shared": {"train&val": len(sess["train"] & sess["val"]), "train&test": len(sess["train"] & sess["test"]),
                                           "val&test": len(sess["val"] & sess["test"])}}
    cat = lambda k: (re.match(r"cst_[a-z]+_(.+)_(\d+)$", k) or [None, k])[1]
    out["cst"] = {"sequences": {p: len(seqs[p]["cst"]) for p in seqs},
                  "scene_categories": {p: dict(Counter(cat(k) for k in seqs[p]["cst"])) for p in seqs}}
    out["cst"]["test_categories_absent_from_train"] = sorted(set(out["cst"]["scene_categories"]["test"]) - set(out["cst"]["scene_categories"]["train"]))

    # -- 2. adjacency, size, motion, flicker (train + val content)
    off, box = load_label_cache(LABEL_CACHE_PATH)
    res = {}
    for phase in ("train", "val"):
        samples = build_split_samples(MANIFEST, SPLITS, phase)
        for s in samples:
            key = (s["dataset"], s["modality"])
            if key not in res:
                im = cv2.imread(s["cur_path"], cv2.IMREAD_UNCHANGED)
                res[key] = (im.shape[1], im.shape[0])
        by_seq = defaultdict(list)
        for s in samples:
            by_seq[s["seq_key"]].append(s)
        sizes, motion, rel, gaps, flick, frames = defaultdict(list), defaultdict(list), defaultdict(list), Counter(), Counter(), Counter()
        for lst in by_seq.values():
            lst.sort(key=lambda s: s["frame_idx"])
            d, mod = lst[0]["dataset"], lst[0]["modality"]
            W, H = res[(d, mod)]
            r = min(640 / W, 640 / H)
            prev_c, prev_n, hts = None, None, []
            for i, s in enumerate(lst):
                b = box[off[s["manifest_row"]]:off[s["manifest_row"] + 1]]
                dr = b[b[:, 0] == 0] if len(b) else b
                hts.append(len(dr) > 0)
                sizes[(d, mod, phase)] += [max(bb[3] * W * r, bb[4] * H * r) for bb in dr]
                if i and d != "det_fly":
                    gaps[(d, s["frame_idx"] - lst[i - 1]["frame_idx"])] += 1
                if len(dr) == 1:
                    c = np.array([dr[0][1] * W * r, dr[0][2] * H * r])
                    sz = max(dr[0][3] * W * r, dr[0][4] * H * r)
                    if prev_n == 1 and i and s["frame_idx"] - lst[i - 1]["frame_idx"] == 1:
                        mv = float(np.linalg.norm(c - prev_c))
                        motion[(d, mod, phase)].append(mv)
                        rel[(d, mod, phase)].append(mv / max(sz, 1e-3))
                    prev_c = c
                prev_n = len(dr)
            if d != "det_fly" and len(hts) > 2:
                h = np.array(hts)
                flick[(d, phase)] += int(((h[1:-1] != h[:-2]) & (h[1:-1] != h[2:])).sum())
                frames[(d, phase)] += len(h) - 2
        for k, v in sizes.items():
            v = np.array(v)
            out.setdefault("drone_size_at_network_input_px", {})["/".join(k)] = {
                "n": int(len(v)), "median": round(float(np.median(v)), 1), "p10": round(float(np.percentile(v, 10)), 1),
                "frac_lt8px": round(float((v < 8).mean()), 3), "frac_lt16px": round(float((v < 16).mean()), 3)}
        for k, v in motion.items():
            v = np.array(v)
            out.setdefault("target_motion_per_frame_px", {})["/".join(k)] = {
                "n": int(len(v)), "median": round(float(np.median(v)), 2), "p90": round(float(np.percentile(v, 90)), 2),
                "median_motion_over_size": round(float(np.median(rel[k])), 3), "frac_lt_1px": round(float((v < 1).mean()), 3)}
        out.setdefault("neighbour_frame_gaps", {})[phase] = {f"{d}:{g}": n for (d, g), n in gaps.most_common(6)}
        out.setdefault("label_flicker_isolated_frames", {})[phase] = {d: f"{flick[(d, phase)]}/{frames[(d, phase)]}" for d in ("anti_uav", "cst")}
    out["native_resolution"] = {f"{d}/{m}": v for (d, m), v in res.items()}

    # -- 3. camera ego-motion (train pairs): global shift by phase correlation, network scale
    tr = build_split_samples(MANIFEST, SPLITS, "train", datasets=["anti_uav", "cst"])
    by = defaultdict(list)
    for s in tr:
        if s["prev_path"] != s["cur_path"]:
            by[(s["dataset"], s["modality"])].append(s)
    for key, lst in by.items():
        W, H = res[key]
        r = min(640 / W, 640 / H)
        shifts = []
        for s in (lst[i] for i in rng.choice(len(lst), min(250, len(lst)), replace=False)):
            a = cv2.imread(s["prev_path"], cv2.IMREAD_GRAYSCALE)
            b = cv2.imread(s["cur_path"], cv2.IMREAD_GRAYSCALE)
            if a is None or b is None or a.shape != b.shape:
                continue
            a = cv2.resize(a, None, fx=r, fy=r, interpolation=cv2.INTER_AREA).astype(np.float32)
            b = cv2.resize(b, None, fx=r, fy=r, interpolation=cv2.INTER_AREA).astype(np.float32)
            (dx, dy), _ = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
            shifts.append(float(np.hypot(dx, dy)))
        v = np.array(shifts)
        out.setdefault("camera_global_shift_per_frame_px", {})["/".join(key)] = {
            "n": int(len(v)), "median": round(float(np.median(v)), 2), "p90": round(float(np.percentile(v, 90)), 2)}

    # -- 4. Det-Fly: train<->val near-duplicates (perceptual hash + original frame ids), flights
    def dhash(p):
        g = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (9, 8), interpolation=cv2.INTER_AREA)
        return int("".join("1" if x else "0" for x in (g[:, 1:] > g[:, :-1]).flatten()), 2)
    dtr = build_split_samples(MANIFEST, SPLITS, "train", datasets=["det_fly"])
    dva = build_split_samples(MANIFEST, SPLITS, "val", datasets=["det_fly"])
    ht = np.array([dhash(s["cur_path"]) for s in dtr], dtype=np.uint64)
    bits = lambda x: np.array([bin(int(v)).count("1") for v in x])
    mind = np.array([bits(ht ^ np.uint64(dhash(s["cur_path"]))).min() for s in dva])
    fid = lambda s: int(m.group(1)) if (m := re.match(r"(\d+)_jpg", Path(s["cur_path"]).name)) else None
    tid = np.array(sorted(x for x in (fid(s) for s in dtr) if x is not None))
    vid = [x for x in (fid(s) for s in dva) if x is not None]
    dist = np.array([int(np.min(np.abs(tid[max(np.searchsorted(tid, k) - 1, 0):np.searchsorted(tid, k) + 1] - k))) for k in vid])
    out["det_fly"] = {"images": {"train": len(dtr), "val": len(dva)}, "source": "Roboflow export, stretched to 640x640, split by Roboflow",
                      "val_nearest_train_dhash_bits": {"<=2": int((mind <= 2).sum()), "<=6": int((mind <= 6).sum()), "median": float(np.median(mind))},
                      "val_frame_id_distance_to_nearest_train_frame": {"==1": int((dist == 1).sum()), "<=3": int((dist <= 3).sum()), "n": len(dist)},
                      "flights": {"train": len({k // 100000 for k in tid}), "val": len({k // 100000 for k in vid}),
                                  "shared": len({k // 100000 for k in tid} & {k // 100000 for k in vid})}}

    # -- 5. composition of the registered VAL set's micro GT (the pre-registered micro metrics)
    V = subsample(build_split_samples(MANIFEST, SPLITS, "val"), {d: 5000 for d in ("anti_uav", "cst", "det_fly")})
    micro, allg, bird_micro, mseq = Counter(), Counter(), 0, set()
    for s in V:
        for bb in box[off[s["manifest_row"]]:off[s["manifest_row"] + 1]]:
            is_micro = bb[3] * bb[4] < 0.0003
            if int(bb[0]) == 1:
                bird_micro += int(is_micro)
                continue
            allg[s["dataset"]] += 1
            if is_micro:
                micro[s["dataset"]] += 1
                mseq.add((s["dataset"], s["seq_key"]))
    out["registered_val_micro_gt"] = {"drone_gt": dict(allg), "drone_micro_gt": dict(micro), "bird_micro_gt": bird_micro,
                                      "sequences_with_drone_micro": dict(Counter(d for d, _ in mseq)),
                                      "note": "micro AP50 = mean over classes present of (drone-micro AP50, bird-micro AP50) -- evaluation.score"}
    p = S.write_report("diag_data.json", out)
    S.ledger({"id": "D1", "type": "diagnostic", "what": "data structure / leakage / motion / size", "script": "scripts/diag_data.py",
              "inputs": {"manifest_sha256": file_sha256(MANIFEST), "splits_sha256": file_sha256(SPLITS)}, "output": str(p.relative_to(S.REPO)),
              "test_usage": "sequence names only"})
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
