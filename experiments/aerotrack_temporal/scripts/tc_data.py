#!/usr/bin/env python
"""Track C trajectory data: ground-truth target trajectories per sequence, from the OFFICIAL annotations.

    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_data.py            (builds train + val)

Sources (TRAIN / VAL only -- TEST is sealed: `load_split("test")` raises unless purpose == "final_eval"):
  Anti-UAV IR / RGB   data/anti_uav/<split>/<seq>/{infrared,visible}.json through Track A's loader (exist + gt_rect, xywh px)
  CST Anti-UAV        the COMPLETE release, data/cst_anti_uav_full/CST-AntiUAV/CST-AntiUAV/<split>/<seq>/IR_label.json
                      (official 120 / 40 / 60 split; the truncated v1 archive is never read here)
Every sequence keeps its own pixel frame (IR 640x512, RGB 1920x1080, CST 640x512). `group` is the cross-fitting / dev-split unit
(Anti-UAV recording session -- IR and RGB of a clip together; CST sequence), as in experiments/systems/scripts/track_eval.py.
`dev` marks the TRAIN groups held out for early stopping / recipe choices (md5(group) % 10 == 0): VAL is never used to tune.

Output: results/runs/aerotrack_temporal/_data/traj_<split>.npz (concatenated arrays + offsets) and traj_<split>.json (per-sequence label
sha256, counts, the npz sha256). Loading checks the npz hash against the JSON.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "experiments" / "anti_uav" / "src"))

OUT = REPO / "results/runs" / "aerotrack_temporal" / "_data"
CST_ROOT = REPO / "data" / "cst_anti_uav_full" / "CST-AntiUAV" / "CST-AntiUAV"
ANTIUAV_ROOT = REPO / "data" / "anti_uav"
SIZES = {"antiuav_IR": (640, 512), "antiuav_RGB": (1920, 1080), "cst": (640, 512)}
DOMAINS = tuple(SIZES)


class SealedSplitError(PermissionError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def group_of(key):
    """Anti-UAV recording session (IR and RGB of a clip together), else the sequence itself (= track_eval.group_of)."""
    if key.startswith("antiuav_"):
        name = key.split("_", 2)[2].rsplit("__", 1)[0]
        return "antiuav:" + name.rsplit("_", 1)[0]
    return key


def is_dev(group):
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 10 == 0


def _boxes(exist, rects, n):
    box = np.full((n, 4), np.nan, np.float64)
    vis = np.zeros(n, bool)
    for i in range(n):
        r = rects[i] if i < len(rects) else None
        if exist[i] and isinstance(r, (list, tuple)) and len(r) == 4 and r[2] > 0 and r[3] > 0:
            box[i] = r
            vis[i] = True
    return vis, box


def collect(split, purpose):
    if split == "test" and purpose != "final_eval":
        raise SealedSplitError("Track C never reads TEST trajectories outside the registered final evaluation")
    from anti_uav_repro.data.antiuav import load_sequences
    seqs = []
    for mod in ("IR", "RGB"):
        for s in load_sequences(ANTIUAV_ROOT, REPO / "data" / "derived" / "data_unified" / "anti_uav" / "images", split, mod,
                                purpose={"train": "train", "val": "tune", "test": "final_eval"}[split], require_frames=False):
            vis, box = _boxes(s.exist, s.gt_rect, s.n_frames)
            key = f"antiuav_{split}_{s.name}__{mod}"
            seqs.append({"key": key, "domain": f"antiuav_{mod}", "split": split, "vis": vis, "box": box, "start": int(s.start_frame),
                         "label": str(s.label_path.relative_to(REPO)), "label_sha256": sha256(s.label_path)})
    labs = sorted((CST_ROOT / split).glob("*/IR_label.json"))
    want = {"train": 120, "val": 40, "test": 60}[split]
    if len(labs) != want:
        raise SystemExit(f"CST {split}: {len(labs)} label files, the complete release has {want}")
    for lab in labs:
        a = json.loads(lab.read_text(encoding="utf-8"))
        n = len(a["exist"])
        vis, box = _boxes([int(e) for e in a["exist"]], a["gt"], n)
        start = int(np.flatnonzero(vis)[0]) if vis.any() else 0
        seqs.append({"key": f"cst_{split}_{lab.parent.name}", "domain": "cst", "split": split, "vis": vis, "box": box, "start": start,
                     "label": str(lab.relative_to(REPO)), "label_sha256": sha256(lab)})
    for s in seqs:
        s["group"] = group_of(s["key"])
        s["dev"] = bool(split == "train" and is_dev(s["group"]))
        s["W"], s["H"] = SIZES[s["domain"]]
    return seqs


def save(split, seqs):
    OUT.mkdir(parents=True, exist_ok=True)
    off = np.cumsum([0] + [len(s["vis"]) for s in seqs])
    npz = OUT / f"traj_{split}.npz"
    np.savez_compressed(npz, vis=np.concatenate([s["vis"] for s in seqs]), box=np.concatenate([s["box"] for s in seqs]), offsets=off)
    meta = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "split": split, "npz_sha256": sha256(npz),
            "sources": {"anti_uav": str(ANTIUAV_ROOT.relative_to(REPO)), "cst": str(CST_ROOT.relative_to(REPO)) + " (complete release)"},
            "sequences": [{k: s[k] for k in ("key", "domain", "split", "group", "dev", "W", "H", "start", "label", "label_sha256")}
                          | {"n_frames": int(len(s["vis"])), "n_visible": int(s["vis"].sum())} for s in seqs]}
    by = {}
    for s in seqs:
        d = by.setdefault(s["domain"] + ("/dev" if s["dev"] else ""), {"sequences": 0, "frames": 0, "visible": 0})
        d["sequences"] += 1
        d["frames"] += len(s["vis"])
        d["visible"] += int(s["vis"].sum())
    meta["totals"] = by
    (OUT / f"traj_{split}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def load_split(split, purpose="develop"):
    """-> list of dicts (key, domain, group, dev, W, H, start, vis[T], box[T,4] xywh px with NaN where absent)."""
    if split == "test" and purpose != "final_eval":
        raise SealedSplitError("Track C never reads TEST trajectories outside the registered final evaluation")
    meta = json.loads((OUT / f"traj_{split}.json").read_text(encoding="utf-8"))
    npz = OUT / f"traj_{split}.npz"
    if sha256(npz) != meta["npz_sha256"]:
        raise SystemExit(f"{npz} does not match its recorded sha256; rebuild with tc_data.py")
    z = np.load(npz)
    off = z["offsets"]
    vis_all, box_all = z["vis"], z["box"]          # decompress ONCE: every z[...] access re-reads the whole array, and slicing a
    out = []                                        # fresh copy per sequence kept ~400 full copies alive (~6 GB; fixed 2026-10-05)
    for i, m in enumerate(meta["sequences"]):
        out.append(dict(m, vis=vis_all[off[i]:off[i + 1]], box=box_all[off[i]:off[i + 1]]))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for split in ("train", "val"):
        meta = save(split, collect(split, "develop"))
        print(split, json.dumps(meta["totals"]))


if __name__ == "__main__":
    main()
