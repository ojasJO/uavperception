#!/usr/bin/env python
"""Score a series run's last.pt (the pre-registered checkpoint) on its held-out DEV fold and on the official VAL.

    python experiments/aerotrack_generalization/scripts/eval_gen.py --run gen_stock_f0_e20 --sets dev val
    python experiments/aerotrack_generalization/scripts/eval_gen.py --run gen_<cfg>_full_e20 --sets test --final-eval   (final step only)

Each set is built with the run's own input (gap 0: t,t,t; gap g: t-g,t,t+g). Official VAL/TEST are the registered image sets
(identity checked against the repair series' hashes). Protocol = the repair series': conf 0.001, NMS IoU 0.7, max_det 30, bf16.
Per-image records -> runs/aerotrack_generalization/_cache/gen_records/ (for the paired bootstrap); one row per (run, set) ->
reports/gen_results<tag>.json; TEST additionally needs --final-eval and is written once per checkpoint to reports/gen_test_ledger.jsonl."""
import argparse
import json
import os
import pickle
import sys
import time

import _series as S
import gen_lib as L
import torch
from torch.utils.data import DataLoader

import evaluation as E
from aerotrack_trainer import load_checkpoint
from evaluate_detector import eval_set
from stacked_dataset import AeroTrackDataset, aerotrack_collate, load_label_cache
from train import LABEL_CACHE_PATH

from anti_uav_repro.utils.repro import env_snapshot, file_sha256

REGISTERED = {"val": "6fb41d44d45b", "test": "37e944793773"}
TEST_LEDGER = S.REPORTS / "gen_test_ledger.jsonl"


def build(meta, which, cap):
    gap = int(meta["gap"])
    if which == "dev":
        folds = L.load_folds()
        return L.dev_set(folds, int(meta["fold"]), gap, cap)
    ss, ident = L.official_set(which, gap)
    reg_samples, reg = eval_set(which, 5000, None)      # the registered set and its identity (repair series' hash)
    assert reg["sha256"].startswith(REGISTERED[which]), reg["sha256"]
    frames = lambda xs: {(s["seq_key"], int(s["frame_idx"])) for s in xs}
    assert frames(ss) == frames(reg_samples), "image set differs from the registered one"
    ident["registered_sha256"] = reg["sha256"]
    return ss, ident


def write_row(tag, row):
    p = S.REPORTS / f"gen_results{tag}.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"rows": []}
    data["rows"] = [r for r in data["rows"] if (r["run"], r["set"]) != (row["run"], row["set"])] + [row]
    data["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = p.with_name(p.name + ".tmp")
    S.REPORTS.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sets", nargs="+", default=["dev", "val"], choices=["dev", "val", "test"])
    ap.add_argument("--cap", type=int, default=L.DEV_CAP, help="dev set: images per dataset")
    ap.add_argument("--tag", default="", help="results file suffix (smoke tests)")
    ap.add_argument("--final-eval", action="store_true")
    a = ap.parse_args()
    if "test" in a.sets and not a.final_eval:
        sys.exit("Refusing: TEST needs --final-eval (the final, pre-selected candidate only).")
    run_dir = S.REPO / "runs" / a.run
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if meta.get("return_code") != 0:
        sys.exit(f"{a.run} is not a finished run (return_code {meta.get('return_code')})")
    w = run_dir / "weights" / "last.pt"
    wsha = file_sha256(w)
    if "test" in a.sets and TEST_LEDGER.exists() and any(json.loads(x).get("weights_sha256") == wsha for x in TEST_LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()):
        sys.exit(f"Refusing: {w} already has its single TEST evaluation in {TEST_LEDGER.name}")
    dev = torch.device("cuda:0")
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    model, ck = load_checkpoint(w, dev, prefer_ema=True)
    lc = load_label_cache(LABEL_CACHE_PATH)
    out_dir = S.CACHE / "gen_records"
    out_dir.mkdir(parents=True, exist_ok=True)
    for which in a.sets:
        if which == "dev" and meta["fold"] == "full":
            continue
        samples, ident = build(meta, which, a.cap)
        ds = AeroTrackDataset(samples, img_size=(640, 640), augment=None, letterbox=True, out_dtype="uint8", return_meta=True, label_cache=lc)
        dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, collate_fn=aerotrack_collate)
        t0 = time.time()
        recs = E.collect_records(model, dl, dev, imgsz=640, conf=0.001, iou=0.7, max_det=30, amp_dtype=amp, progress=False)
        m = L.drone_metrics(recs)
        slim = [{k: r[k] for k in ("pred", "gt", "gt_area", "dataset", "seq_key")} for r in recs]
        pickle.dump({"run": a.run, "set": which, "identity": ident, "weights_sha256": wsha, "records": slim},
                    open(out_dir / f"{a.run}__{which}{a.tag}.pkl", "wb"))
        row = {"run": a.run, "set": which, "config": meta["config"], "fold": meta["fold"], "seed": meta["seed"], "epochs": meta["epochs"],
               "gap": meta["gap"], "checkpoint": "last", "epoch": ck.get("epoch"), "weights_sha256": wsha[:12], "set_identity": ident,
               "metrics": m, "seconds": round(time.time() - t0, 1), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        if which == "test":
            with open(TEST_LEDGER, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": row["timestamp"], "run": a.run, "weights_sha256": wsha, "set_sha256": ident["registered_sha256"],
                                    "metrics": m, "git": S.git()}) + "\n")
        write_row(a.tag, row)
        S.ledger({"id": f"EV:{a.run}:{which}{a.tag}", "type": "evaluation", "run": a.run, "set": which, "checkpoint": "last.pt",
                  "weights_sha256": wsha, "set_identity": ident, "metrics": m, "env": env_snapshot()["torch"]})
        print(f"[{a.run} / {which}] " + ", ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float)) + f" ({row['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
