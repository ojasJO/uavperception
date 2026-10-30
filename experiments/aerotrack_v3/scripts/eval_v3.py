#!/usr/bin/env python
"""v3 series: score a run's last.pt on corpus-v2 VAL with the v2 protocol (same frames, same metrics as eval_v2.py).

    venv\Scripts\python.exe experiments\aerotrack_v3\scripts\eval_v3.py --run v3_n_p2__causal_g4_s0_e20
    ... --set test --final-eval      (the single pre-registered final evaluation of the selected candidate; refused twice)

The input window comes from the run's run_meta.json (v2 runs: `window`; generalization runs: gap 0 -> t,t,t, gap g -> t-g,t,t+g).
Protocol = the repair / generalization series': letterbox 640, conf 0.001, NMS IoU 0.7, max_det 30, bf16, every dataset capped
at 5,000 images evenly over its sequences. Per-image records -> runs/aerotrack_v2/_cache/ (paired bootstrap); one row per
(run, set) -> experiments/aerotrack_v3/reports/v3_results.json; ledger entry per evaluation.
"""
import argparse
import json
import pickle
import sys
import time

import v3series as S
import v2lib as L          # evaluation sets + metrics shared with the v2 series
import torch
from torch.utils.data import DataLoader

import evaluation as E
from aerotrack_trainer import load_checkpoint
from stacked_dataset import AeroTrackDataset, aerotrack_collate, load_label_cache
from train import LABEL_CACHE_PATH

from anti_uav_repro.utils.repro import env_snapshot, file_sha256

RESULTS = S.REPORTS / "v3_results.json"


def run_window(meta):
    if "window" in meta:
        return meta["window"], int(meta["gap"])
    g = int(meta.get("gap", meta.get("temporal_gap", 1)))
    return ("ttt" if g == 0 else "symmetric"), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", default="val", choices=["val", "test"])
    ap.add_argument("--final-eval", action="store_true")
    ap.add_argument("--cap", type=int, default=L.EVAL_CAP)
    a = ap.parse_args()
    if a.set == "test" and not a.final_eval:
        sys.exit("Refusing: TEST needs --final-eval (the single evaluation of the pre-selected final candidate).")
    run_dir = S.REPO / "runs" / a.run
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if meta.get("return_code") != 0:
        sys.exit(f"{a.run} is not a finished run (return_code {meta.get('return_code')})")
    w = run_dir / "weights" / "last.pt"
    wsha = file_sha256(w)
    if meta.get("last_pt_sha256") and meta["last_pt_sha256"] != wsha:
        sys.exit(f"{w} does not match the sha256 recorded at the end of training")
    if a.set == "test" and S.TEST_LEDGER.exists() and any(
            json.loads(x).get("weights_sha256") == wsha for x in S.TEST_LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()):
        sys.exit(f"Refusing: {w} already has its single TEST evaluation in {S.TEST_LEDGER.name}")
    smoke = bool(meta.get("smoke"))
    if smoke and a.set == "test":
        sys.exit("Refusing: a smoke run is never scored on TEST")
    results, cache = (S.REPORTS / "v3_results_smoke.json", S.CACHE / "smoke") if smoke else (RESULTS, S.CACHE)
    window, gap = run_window(meta)
    samples, ident = L.eval_samples(a.set, window, gap, a.cap)
    dev = torch.device("cuda:0")
    from precision import amp_dtype as _amp_dtype
    amp = _amp_dtype()          # bf16 on sm>=8.0, fp16 on the V100 (D-2026-10-06-04)
    model, ck = load_checkpoint(w, dev, prefer_ema=True)
    ds = AeroTrackDataset(samples, img_size=(640, 640), augment=None, letterbox=True, out_dtype="uint8", return_meta=True,
                          label_cache=load_label_cache(LABEL_CACHE_PATH))
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, collate_fn=aerotrack_collate)
    t0 = time.time()
    recs = E.collect_records(model, dl, dev, imgsz=640, conf=0.001, iou=0.7, max_det=30, amp_dtype=amp, progress=False)
    m = L.metrics(recs)
    cache.mkdir(parents=True, exist_ok=True)
    slim = [{k: r[k] for k in ("pred", "gt", "gt_area", "dataset", "seq_key", "frame_idx")} for r in recs]
    with open(cache / f"{a.run}__{a.set}.pkl", "wb") as f:
        pickle.dump({"run": a.run, "set": a.set, "identity": ident, "weights_sha256": wsha, "window": window, "records": slim}, f)
    row = {"run": a.run, "set": a.set, "series": meta.get("series"), "config": meta.get("config"), "seed": meta.get("seed"),
           "window": window, "gap": gap, "train_corpus": (meta.get("data") or {}).get("corpus", "data_unified (v1)"),
           "checkpoint": "last", "epoch": ck.get("epoch"), "weights_sha256": wsha, "set_identity": ident, "metrics": m,
           "seconds": round(time.time() - t0, 1), "timestamp": S.now()}
    data = json.loads(results.read_text(encoding="utf-8")) if results.exists() else {"rows": []}
    data["rows"] = [r for r in data["rows"] if (r["run"], r["set"]) != (a.run, a.set)] + [row]
    data["generated"] = S.now()
    S.write_json_atomic(data, results)
    if a.set == "test":
        with open(S.TEST_LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": row["timestamp"], "run": a.run, "weights_sha256": wsha, "set_sha256": ident["sha256"],
                                "metrics": m, "git": S.git()}) + "\n")
    S.ledger({"id": f"EV:{a.run}:{a.set}", "type": "evaluation", "run": a.run, "set": a.set, "checkpoint": "last.pt", "smoke": smoke,
              "weights_sha256": wsha, "set_identity": ident, "metrics": m, "torch": env_snapshot().get("torch")})
    print(f"[{a.run} / {a.set}] " + ", ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float)) + f" ({row['seconds']:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
