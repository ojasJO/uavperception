#!/usr/bin/env python
"""Track C tracking evaluation: identity persistence, continuity and recovery for every tracker on a VAL detection cache.

    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_track_eval.py --run <detector run> [--corpus auto]
                              [--groups anti_uav_IR,anti_uav_RGB,cst_IR]

MOT protocol (no init box), same detections for every tracker (drone class, conf >= 0.05), as experiments/systems/scripts/mot_eval.py:
    bytetrack, sort            c_model/bytetrack (repair series)
    argmax@cf                  per-frame best detection, threshold cross-fitted (track_eval report)
    ste@cf                     the STE, cross-fitted parameters, ONE identity (single-target semantics)
    ste@cf/segments            the same outputs, a NEW identity at every track start (MOT semantics: what a loss costs)
Metrics per group: HOTA / DetA / AssA / LocA (tc_track_metrics, TrackEval algorithm), CLEAR-MOT / IDF1 / ID switches /
fragmentation (repair-series code), MT / PT / ML, longest-run continuity, recovery time after loss and after re-entry (frames;
25 = 1 s at 25 fps), ids per target. VAL only. Output: experiments/aerotrack_temporal/reports/tracking_<run>_<corpus>.{json,md}.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "systems" / "scripts"))
import mot_eval as ME  # noqa: E402  (mot_inputs, single_id, CM = repair-series helpers, evaluate_tracking)
import ste as T  # noqa: E402
import tc_track_metrics as TM  # noqa: E402
import track_eval as TE  # noqa: E402


def to_tc(by_frame, T_):
    """{t: [n,5] xyxy+id} -> {t: (ids, boxes)}"""
    out = {}
    for t in range(T_):
        a = by_frame.get(t)
        if a is not None and len(a):
            out[t] = (np.asarray(a)[:, 4].astype(int), np.asarray(a)[:, :4].astype(np.float64))
    return out


def ste_outputs(it, params):
    s = it["seq"]
    b, sc, cl = zip(*it["dets"])
    trk = T.STE(T.STEParams(**{**T.STEParams().as_dict(), **params}))
    trk.reset(None)
    single, seg, sid = {}, {}, 0
    for t in range(s.n_frames):
        box, mode, _ = trk.step(b[t], sc[t], cl[t])
        if mode == "init":
            sid += 1
        if box is not None:
            single[t] = np.array([[*box, 1]], np.float32)
            seg[t] = np.array([[*box, sid]], np.float32)
        else:
            single[t] = seg[t] = np.zeros((0, 5), np.float32)
    return single, seg


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", default="auto")
    ap.add_argument("--groups", default=None, help="comma list of groups to keep (e.g. exclude the v1 CST holdout)")
    a = ap.parse_args()
    if a.corpus == "auto":
        base = REPO / "runs" / "systems" / "_cache" / a.run
        a.corpus = "data_unified_v2" if (base / "data_unified_v2" / "val" / "index.json").exists() else "data_unified"
    keep = set(a.groups.split(",")) if a.groups else None
    t0 = time.time()
    index, seqs = TE.load(a.run, a.corpus)
    rep = REPO / "experiments" / "systems" / "reports" / f"track_{a.run}_{a.corpus}.json"
    if not rep.exists():
        raise SystemExit(f"{rep} missing: run experiments/systems/scripts/track_eval.py first (cross-fitted parameters)")
    cf = json.loads(rep.read_text(encoding="utf-8"))["trackers"]
    names = ("bytetrack", "sort", "argmax@cf", "ste@cf", "ste@cf/segments")
    acc = {n: {} for n in names}
    for key in sorted(seqs):
        it = seqs[key]
        if keep and it["group"] not in keep:
            continue
        s = it["seq"]
        d_by, g_by, occ = ME.mot_inputs(it)
        half = str(it["half"])
        preds = {"bytetrack": ME.CM.run_tracker(d_by, use_low_stage=True)[0], "sort": ME.CM.run_tracker(d_by, use_low_stage=False)[0]}
        b, sc, cl = zip(*it["dets"])
        thr = cf["argmax"]["params_per_half"][half]["thr"]
        preds["argmax@cf"] = ME.single_id({t: T.argmax_step(b[t], sc[t], cl[t], thr) for t in range(s.n_frames)})
        preds["ste@cf"], preds["ste@cf/segments"] = ste_outputs(it, cf["ste"]["params_per_half"][half])
        gt = to_tc(g_by, s.n_frames)
        for n, pr in preds.items():
            r = acc[n].setdefault(it["group"], {"clear": [], "hota": [], "cont": []})
            r["clear"].append(ME.evaluate_tracking(pr, g_by, occlusion_frames=occ))
            p = to_tc(pr, s.n_frames)
            r["hota"].append(TM.hota_sequence(gt, p, s.n_frames))
            r["cont"] += TM.continuity(gt, p, s.n_frames)
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": a.run, "corpus": a.corpus, "groups_kept": sorted(keep) if keep else "all",
           "cache_set_sha256": index["set_sha256"], "cross_fitted_source": rep.name,
           "protocol": "MOT (no init box); drone detections conf >= 0.05; one GT identity per sequence", "results": {}}
    for n, by in acc.items():
        for g, r in by.items():
            out["results"].setdefault(n, {})[g] = {"hota": TM.hota_summary(TM.hota_combine(r["hota"])),
                                                    "clear": ME.CM._accumulate(r["clear"]),
                                                    "continuity": TM.continuity_summary(r["cont"])}
    out["seconds"] = round(time.time() - t0, 1)
    rdir = REPO / "experiments" / "aerotrack_temporal" / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / f"tracking_{a.run}_{a.corpus}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = [f"# Tracking: identity persistence, continuity, recovery — {a.run} ({a.corpus}, VAL)", "",
          f"*Generated {out['generated']} by `scripts/tc_track_eval.py`; {out['protocol']}; groups: {out['groups_kept']}; cross-fitted "
          f"parameters from `{rep.name}`. Recovery in frames (25 = 1 s at 25 fps); 'never' = not recovered before the sequence ends.*", "",
          "| tracker | group | HOTA | DetA | AssA | MOTA | IDF1 | ID sw | Frag | MT / ML | longest run | ids / target | loss → recovered: median / p90 / never | re-entry → recovered: median / p90 / never |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|"]
    f = lambda v, d=1: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    for n, by in out["results"].items():
        for g, v in sorted(by.items()):
            h, c, k = v["hota"], v["clear"], v["continuity"]
            rl, rr = k["recovery_loss"], k["recovery_reentry"]
            md.append(f"| {n} | {g} | {h['HOTA']:.3f} | {h['DetA']:.3f} | {h['AssA']:.3f} | {c['MOTA']:.3f} | {c['IDF1_mean_per_sequence']:.3f} | "
                      f"{c['id_switches']} | {k['fragments']} | {k['MT']:.2f} / {k['ML']:.2f} | {k['longest_run_frac_mean']:.2f} | "
                      f"{k['ids_per_target_mean']:.1f} | {f(rl['median_frames'], 0)} / {f(rl['p90_frames'], 0)} / {rl['never']} of {rl['events']} | "
                      f"{f(rr['median_frames'], 0)} / {f(rr['p90_frames'], 0)} / {rr['never']} of {rr['events']} |")
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
