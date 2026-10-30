#!/usr/bin/env python
"""Build corpus v2 = corpus v1 with the COMPLETE CST Anti-UAV release, in b_pipeline/data_unified_v2/ (v1 untouched).

    venv\\Scripts\\python.exe experiments\\cst_completion\\scripts\\build_corpus_v2.py

Requires reports/extraction.json with verdict COMPLETE_AND_CONSISTENT (extract_and_verify.py).

* CST: the v1 conversion itself (`b_pipeline/1_standardize_annotations.py::standardize_cst`, same parser, same
  rect->YOLO rounding, same empty-label rule) is run against data/cst_anti_uav_full and writes labels + image junctions
  under data_unified_v2/cst/. Per-frame CST attribute annotations (exist, OC, OV, SV, TC, DBC, CDB, FM, SIZE), which v1
  ignored, are exported to data_unified_v2/cst_attributes.csv for robustness analysis.
* Anti-UAV and Det-Fly: their v1 manifest rows are copied verbatim (same label files, same images).
* Then `b_pipeline/3_prepare_training.py` runs with AEROTRACK_CORPUS=data_unified_v2: label cache, sampler hardness,
  and splits -- which now use the official CST val split (40 sequences) instead of v1's 15-sequence holdout.
* Verification: every CST TEST label file of v2 is byte-identical to v1's (the sealed test split is unchanged), every v1
  CST train sequence keeps identical labels, split counts, T4 leakage (inside 3_prepare_training).
Writes experiments/cst_completion/reports/corpus_v2.{json,md}. Refuses to overwrite an existing v2 manifest.
"""
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
V1 = REPO / "b_pipeline" / "data_unified"
V2 = REPO / "b_pipeline" / "data_unified_v2"
FULL = REPO / "data" / "cst_anti_uav_full"
REPORTS = REPO / "experiments" / "cst_completion" / "reports"
HEADER = ["dataset", "seq_key", "modality", "split", "frame_idx", "n_frames", "image_path", "label_path", "has_target",
          "n_boxes", "materialized"]
ATTR_KEYS = ["OC", "OV", "SV", "TC", "DBC", "CDB", "FM", "SIZE"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ext = json.loads((REPORTS / "extraction.json").read_text(encoding="utf-8"))
    if ext.get("verdict") != "COMPLETE_AND_CONSISTENT":
        sys.exit(f"extraction verdict is {ext.get('verdict')!r}; run extract_and_verify.py first")
    if (V2 / "manifest.csv").exists():
        sys.exit(f"{V2 / 'manifest.csv'} exists; corpus v2 is built once and never overwritten")
    t0 = time.time()
    V2.mkdir(parents=True, exist_ok=True)

    # ---- CST through the v1 conversion code, pointed at the complete extraction and the v2 directory
    sys.path.insert(0, str(REPO / "a_inspection"))
    import eda_common as C
    C.DATASET_DIRS["cst_anti_uav"] = FULL
    S = load_module("standardize_v1", REPO / "b_pipeline" / "1_standardize_annotations.py")
    S.UNIFIED = V2
    summary = {}
    tmp = V2 / "manifest.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        S.standardize_cst(w, summary)
        # Anti-UAV + Det-Fly rows: verbatim from v1 (identical label files and images)
        with open(V1 / "manifest.csv", newline="", encoding="utf-8") as g:
            r = csv.reader(g)
            if next(r) != HEADER:
                sys.exit("v1 manifest header differs from the expected schema")
            n_copied = Counter()
            for row in r:
                if row[0] in ("anti_uav", "det_fly"):
                    w.writerow(row)
                    n_copied[row[0]] += 1
    os.replace(tmp, V2 / "manifest.csv")

    # ---- CST per-frame attributes (ignored by v1)
    n_attr = 0
    with open(V2 / "cst_attributes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq_key", "split", "frame_idx", "exist"] + ATTR_KEYS)
        for split_dir in sorted(p for p in FULL.rglob("*") if p.is_dir() and p.name in ("train", "val", "test")):
            for sd in sorted(x for x in split_dir.iterdir() if x.is_dir()):
                j = sd / "IR_label.json"
                if not j.exists():
                    continue
                a = json.loads(j.read_text(encoding="utf-8"))
                n = len(a.get("exist", []))
                for i in range(n):
                    w.writerow([f"cst_{split_dir.name}_{sd.name}", split_dir.name, i, a["exist"][i]]
                               + [(a[k][i] if k in a and i < len(a[k]) else "") for k in ATTR_KEYS])
                    n_attr += 1

    # ---- label cache, hardness, splits (official CST val) via the v1 preparation script
    env = dict(os.environ, AEROTRACK_CORPUS=V2.name)
    rc = subprocess.call([sys.executable, str(REPO / "b_pipeline" / "3_prepare_training.py")], cwd=str(REPO), env=env)
    if rc != 0:
        sys.exit(f"3_prepare_training.py failed (rc {rc})")

    # ---- verification against v1
    import pandas as pd
    m1 = pd.read_csv(V1 / "manifest.csv", low_memory=False, dtype=str)
    m2 = pd.read_csv(V2 / "manifest.csv", low_memory=False, dtype=str)
    c1, c2 = m1[m1.dataset == "cst"], m2[m2.dataset == "cst"]

    def label_hashes(m):
        return {(r.seq_key, int(r.frame_idx)): hashlib.sha256(Path(r.label_path).read_bytes()).hexdigest()
                for r in m.itertuples()}
    t1, t2 = label_hashes(c1[c1.split == "test"]), label_hashes(c2[c2.split == "test"])
    test_identical = t1 == t2
    tr1 = label_hashes(c1[c1.split == "train"])
    tr2 = label_hashes(c2[c2.seq_key.isin(set(c1[c1.split == "train"].seq_key))])
    train_v1_identical = tr1 == tr2
    sp = json.loads((V2 / "splits.json").read_text(encoding="utf-8"))
    seqs = c2.groupby("split").seq_key.nunique().to_dict()
    cats = {s: dict(Counter(k.split("_", 2)[2].rsplit("_", 1)[0] for k in c2[c2.split == s].seq_key.unique()))
            for s in ("train", "val", "test")}
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "corpus": V2.name,
           "manifest_sha256": sha256(V2 / "manifest.csv"), "splits_sha256": sha256(V2 / "splits.json"),
           "v1_manifest_sha256": sha256(V1 / "manifest.csv"), "rows": len(m2), "rows_copied_from_v1": dict(n_copied),
           "cst": {"sequences_per_split": seqs, "categories_per_split": cats, "frames": summary.get("cst"),
                   "attributes_rows": n_attr},
           "checks": {"cst_test_labels_identical_to_v1": test_identical, "cst_test_frames_v1": len(t1), "cst_test_frames_v2": len(t2),
                      "v1_cst_train_labels_identical": train_v1_identical, "v1_cst_train_frames": len(tr1),
                      "official_val_used": sp["val"]["cst"] == ["val"] and not sp["cst_val_holdout_seqs"]},
           "splits_counts": sp["counts"], "seconds": round(time.time() - t0, 1)}
    out["verdict"] = "OK" if all(v for k, v in out["checks"].items() if isinstance(v, bool)) else "PROBLEMS"
    (REPORTS / "corpus_v2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    L = [f"# Corpus v2 (complete CST Anti-UAV)", "", f"*Generated {out['generated']} by `scripts/build_corpus_v2.py`. Verdict: "
         f"**{out['verdict']}**.*", "", f"- manifest sha256 `{out['manifest_sha256'][:16]}` ({out['rows']:,} rows); v1 manifest "
         f"`{out['v1_manifest_sha256'][:16]}` unchanged", f"- splits sha256 `{out['splits_sha256'][:16]}`; CST val = official split: "
         f"{out['checks']['official_val_used']}", f"- CST TEST labels identical to v1: {test_identical} ({len(t2):,} frames)",
         f"- v1 CST train sequences keep identical labels: {train_v1_identical} ({len(tr1):,} frames)", "",
         "| split | CST sequences | categories |", "|---|---:|---|"]
    for s in ("train", "val", "test"):
        L.append(f"| {s} | {seqs.get(s, 0)} | " + ", ".join(f"{k} {v}" for k, v in sorted(cats[s].items())) + " |")
    (REPORTS / "corpus_v2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if out["verdict"] != "OK":
        sys.exit(1)


if __name__ == "__main__":
    main()
