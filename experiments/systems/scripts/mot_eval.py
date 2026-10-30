#!/usr/bin/env python
"""Established trackers vs the single-target layers, on the full VAL detection cache, with CLEAR-MOT / IDF1.

    venv\\Scripts\\python.exe experiments\\systems\\scripts\\mot_eval.py --run <run> [--corpus auto]

Same detections for every tracker (cached; drone class, conf >= 0.05 so ByteTrack's low band exists), every VAL sequence
(Anti-UAV IR / RGB, CST), the repair series' metric code (`c_model/bytetrack.evaluate_tracking` + pooled CLEAR-MOT
`c_model/track_eval._accumulate`, id switches also at re-entry after invisible frames), MOT protocol (no init box):

    bytetrack     c_model/bytetrack.BYTETracker defaults (track .5 / low .1 / buffer 90)
    sort          the same code with the low-score stage disabled
    argmax@0.25   the repair series' registered single-target baseline
    argmax@cf     argmax with the threshold cross-fitted in track_eval.py (each half: the other half's choice)
    ste@cf        the STE with its cross-fitted parameters, no init box (pure detection-based)
Output: experiments/systems/reports/mot_<run>_<corpus>.{json,md}. VAL only.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
for p in (REPO / "c_model", REPO / "b_pipeline"):
    sys.path.insert(1, str(p))
import ste as T  # noqa: E402
import track_eval as TE  # noqa: E402  (systems harness: cache + annotation loader)

_spec = importlib.util.spec_from_file_location("c_model_track_eval", REPO / "c_model" / "track_eval.py")
CM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CM)                       # the repair series' MOT helpers (run_tracker, _accumulate)
from bytetrack import evaluate_tracking  # noqa: E402

CONF = 0.05


def mot_inputs(item):
    s, dets = item["seq"], item["dets"]
    d_by, g_by, occ = {}, {}, set()
    for t in range(s.n_frames):
        b, sc, cl = dets[t]
        keep = (cl == 0) & (sc >= CONF)
        d_by[t] = np.concatenate([b[keep], sc[keep, None]], 1).astype(np.float32) if keep.any() else np.zeros((0, 5), np.float32)
        g = s.box_xywh(t)
        if g is not None:
            g_by[t] = np.array([[g[0], g[1], g[0] + g[2], g[1] + g[3], 1]], np.float32)
        else:
            occ.add(t)
    return d_by, g_by, occ


def single_id(boxes_by_frame):
    return {t: (np.array([[*b, 1]], np.float32) if b is not None else np.zeros((0, 5), np.float32)) for t, b in boxes_by_frame.items()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", default="auto")
    a = ap.parse_args()
    if a.corpus == "auto":
        base = REPO / "runs" / "systems" / "_cache" / a.run
        a.corpus = "data_unified_v2" if (base / "data_unified_v2" / "val" / "index.json").exists() else "data_unified"
    t0 = time.time()
    index, seqs = TE.load(a.run, a.corpus)
    rep = REPO / "experiments" / "systems" / "reports" / f"track_{a.run}_{a.corpus}.json"
    cf = json.loads(rep.read_text(encoding="utf-8"))["trackers"] if rep.exists() else None
    rows = {k: {} for k in ("bytetrack", "sort", "argmax@0.25", "argmax@cf", "ste@cf")}
    for key in sorted(seqs):
        it = seqs[key]
        d_by, g_by, occ = mot_inputs(it)
        half = str(it["half"])
        preds = {"bytetrack": CM.run_tracker(d_by, use_low_stage=True)[0], "sort": CM.run_tracker(d_by, use_low_stage=False)[0]}
        b, sc, cl = zip(*it["dets"])
        preds["argmax@0.25"] = single_id({t: T.argmax_step(b[t], sc[t], cl[t], 0.25) for t in range(it["seq"].n_frames)})
        if cf:
            thr = cf["argmax"]["params_per_half"][half]["thr"]
            preds["argmax@cf"] = single_id({t: T.argmax_step(b[t], sc[t], cl[t], thr) for t in range(it["seq"].n_frames)})
            trk = T.STE(T.STEParams(**{**T.STEParams().as_dict(), **cf["ste"]["params_per_half"][half]}))
            trk.reset(None)
            preds["ste@cf"] = single_id({t: trk.step(b[t], sc[t], cl[t])[0] for t in range(it["seq"].n_frames)})
        for name, pr in preds.items():
            r = evaluate_tracking(pr, g_by, occlusion_frames=occ)
            rows[name].setdefault(it["group"], []).append(r)
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": a.run, "corpus": a.corpus, "cache_set_sha256": index["set_sha256"],
           "protocol": "MOT (no init box); detections conf >= 0.05, drone class; pooled CLEAR-MOT, IDF1 mean per sequence",
           "cross_fitted_source": rep.name if cf else None,
           "results": {name: {g: CM._accumulate(r) for g, r in by.items()} for name, by in rows.items() if by},
           "seconds": round(time.time() - t0, 1)}
    path = REPO / "experiments" / "systems" / "reports" / f"mot_{a.run}_{a.corpus}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = [f"# Established trackers vs single-target layers (MOT protocol) — {a.run} ({a.corpus})", "",
          f"*Generated {out['generated']} by `scripts/mot_eval.py`; {out['protocol']}.*", "",
          "| tracker | group | sequences | MOTA | IDF1 | ID switches | at re-entry | predicted IDs | FP | FN |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, by in out["results"].items():
        for g, v in sorted(by.items()):
            md.append(f"| {name} | {g} | {v['n_sequences']} | {v['MOTA']:.4f} | {v['IDF1_mean_per_sequence']:.4f} | {v['id_switches']} | "
                      f"{v['id_switches_at_occlusion_reentry']} | {v['n_pred_ids']} | {v['false_positives']} | {v['false_negatives']} |")
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
