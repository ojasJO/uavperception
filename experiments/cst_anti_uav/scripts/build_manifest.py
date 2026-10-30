#!/usr/bin/env python
"""Machine-readable manifest + validation of the local CST Anti-UAV release against the official benchmark repository.

    venv\\Scripts\\python.exe experiments\\cst_anti_uav\\scripts\\build_manifest.py

Per sequence: split, scene category, frame count on disk vs label length, image size(s), visible / invisible frames, first-frame
visibility (the OPE start frame), visible frames without a valid box, frame-attribute counts, size bins, label sha256, sha256 of the
sorted frame-name list, and agreement of the boxes / sequence attributes with the official repo's `annos/<split>/` files.
Split-level: totals, category counts, name overlap between splits, and the corpus-v2 split membership used by our training code.
Output: experiments/cst_anti_uav/reports/dataset_manifest.json (+ a printed summary). Frame bytes were CRC-verified when the
archive was extracted (experiments/cst_completion/reports/extraction.md); they are not re-hashed here.
"""
import hashlib
import io
import json
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = REPO / "data" / "cst_anti_uav_full" / "CST-AntiUAV" / "CST-AntiUAV"
ANNOS = REPO / "data" / "cst_anti_uav_official" / "toolkit" / "annos"
OUT = REPO / "experiments" / "cst_anti_uav" / "reports" / "dataset_manifest.json"
FRAME_ATTRS = ("OC", "OV", "SV", "TC", "DBC", "CDB", "FM")
SEQ_ATTRS = ("OC", "OV", "SV", "TC", "DBC", "CDB", "FM", "TS", "SS", "MS", "NS")   # toolkit att file column order


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def jpeg_size(p):
    """(width, height) from the JPEG SOF marker without decoding."""
    with open(p, "rb") as f:
        data = f.read(65536)
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return None


def main():
    t0 = time.time()
    seqs, problems = [], []
    for split in ("train", "val", "test"):
        for d in sorted(p for p in (ROOT / split).iterdir() if p.is_dir()):
            raw = (d / "IR_label.json").read_bytes()
            a = json.loads(raw)
            frames = sorted(p.name for p in d.glob("*.jpg"))
            n = len(a["exist"])
            sizes = Counter(jpeg_size(d / f) for f in (frames[0], frames[len(frames) // 2], frames[-1])) if frames else Counter()
            ex = np.asarray(a["exist"], dtype=bool)
            bad_vis = int(sum(1 for e, g in zip(a["exist"], a["gt"]) if e and (len(g) < 4 or sum(g) == 0 or g[2] <= 0 or g[3] <= 0)))
            diag = np.array([np.hypot(g[2], g[3]) if e and len(g) == 4 else np.nan for e, g in zip(a["exist"], a["gt"])])
            rec = {"split": split, "name": d.name, "category": d.name.rsplit("_", 1)[0], "n_frames_disk": len(frames),
                   "n_frames_label": n, "n_gt": len(a["gt"]), "image_size": [list(k) for k in sizes if k],
                   "n_visible": int(ex.sum()), "n_invisible": int((~ex).sum()), "first_frame_visible": bool(ex[0]) if n else False,
                   "visible_without_valid_box": bad_vis,
                   "frame_attr_counts": {k: int(np.sum(a[k])) for k in FRAME_ATTRS if k in a},
                   "size_bins": {"tiny": int(np.sum(diag <= 10)), "small": int(np.sum((diag > 10) & (diag <= 30))),
                                 "normal": int(np.sum((diag > 30) & (diag <= 50))), "large": int(np.sum(diag > 50))},
                   "size_keys": sorted(k for k in ("SIZE", "TS", "SS", "MS", "NS") if k in a),
                   "label_sha256": sha256_bytes(raw), "frame_names_sha256": sha256_bytes("\n".join(frames).encode())}
            # official repo annotations: annos/<split>/<seq>.txt (x,y,w,h per frame) and annos/<split>/att/<seq>.txt
            af = ANNOS / split / f"{d.name}.txt"
            if af.exists():
                txt = af.read_text(encoding="utf-8").strip().splitlines()
                rows = [[float(v) for v in ln.replace(",", " ").split()] for ln in txt]
                vis = [(r, g) for r, g, e in zip(rows, a["gt"], a["exist"]) if e and len(g) == 4]
                rec["official_anno"] = {"lines": len(rows),
                                        "max_abs_box_diff_visible": float(max((max(abs(x - y) for x, y in zip(r, g)) for r, g in vis
                                                                              if len(r) == 4), default=0.0))}
            else:
                rec["official_anno"] = None
            att = ANNOS / split / "att" / f"{d.name}.txt"
            if att.exists():
                v = np.loadtxt(io.StringIO(att.read_text(encoding="utf-8").replace(",", " ")))
                rec["sequence_attrs_official"] = {k: int(x) for k, x in zip(SEQ_ATTRS, np.atleast_1d(v))}
            for cond, msg in ((len(frames) != n, "frames on disk != label length"), (len(a["gt"]) != n, "len(gt) != len(exist)"),
                              (not rec["first_frame_visible"], "first frame invisible"), (bad_vis > 0, "visible frame without box"),
                              (rec["official_anno"] is None, "no official annotation file"),
                              (rec["official_anno"] and rec["official_anno"]["lines"] != n, "official annotation length differs"),
                              (rec["official_anno"] and rec["official_anno"]["max_abs_box_diff_visible"] > 1e-6,
                               "boxes differ from the official annotation file")):
                if cond:
                    problems.append(f"{split}/{d.name}: {msg}")
            seqs.append(rec)
    by = {s: [r for r in seqs if r["split"] == s] for s in ("train", "val", "test")}
    names = {s: {r["name"] for r in v} for s, v in by.items()}
    v2 = json.loads((REPO / "experiments" / "cst_completion" / "reports" / "corpus_v2.json").read_text(encoding="utf-8"))
    summary = {s: {"sequences": len(v), "frames": sum(r["n_frames_label"] for r in v), "visible": sum(r["n_visible"] for r in v),
                   "invisible": sum(r["n_invisible"] for r in v), "categories": dict(Counter(r["category"] for r in v)),
                   "image_sizes": dict(Counter(tuple(x) for r in v for x in r["image_size"]).most_common()),
                   "size_bins": {b: sum(r["size_bins"][b] for r in v) for b in ("tiny", "small", "normal", "large")},
                   "frame_attr_counts": {k: sum(r["frame_attr_counts"].get(k, 0) for r in v) for k in FRAME_ATTRS}}
               for s, v in by.items()}
    for s in summary:
        summary[s]["image_sizes"] = {f"{w}x{h}": c for (w, h), c in summary[s]["image_sizes"].items()}
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "root": str(ROOT.relative_to(REPO)),
           "official_repo_annos": str(ANNOS.relative_to(REPO)),
           "release_archive": "experiments/cst_completion/reports/volumes.json (11 parts, sha256) + extraction.md (246,131 members, "
                              "0 missing / size / CRC mismatches)",
           "totals": {"sequences": len(seqs), "frames": sum(r["n_frames_label"] for r in seqs),
                      "boxes_visible": sum(r["n_visible"] for r in seqs)},
           "splits": summary,
           "name_overlap": {"train_val": sorted(names["train"] & names["val"]), "train_test": sorted(names["train"] & names["test"]),
                            "val_test": sorted(names["val"] & names["test"])},
           "corpus_v2": {"verdict": v2.get("verdict"), "cst_val_is_official": v2.get("cst_val_official", v2.get("checks", {}))},
           "problems": problems, "sequences": seqs, "seconds": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("totals", "name_overlap", "problems")}, indent=1))
    for s, v in summary.items():
        print(s, json.dumps(v))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
