#!/usr/bin/env python
"""Track C, experiment C3: tracking with domain / learned motion models (preregistration/C3_tracking_motion_models.json).

    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_c3.py --stage prep                   TRAIN only
    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_c3.py --stage run --run v2_causal_g4_s0_e20 [--workers 4]

prep (TRAIN, non-dev groups, oracle histories): per domain, the per-horizon / per-axis mean squared residual of every classical
forecaster at its TRAIN-tuned parameters (the variance table used by the forecaster-driven STE); ONNX export + parity of the C1
model (seed-0 checkpoint of the dev-chosen family, abs inputs). Output: runs/aerotrack_temporal/c3/prep.json.
run (VAL caches of a causal detector, corpus v2): argmax / ste_cv (track_eval's grids) and fste_classical / fste_learned (the
registered 252-setting grid) scored with Track A's SA code, cross-fitted over the two VAL halves exactly as track_eval.py (objective
= mean of the group mSAs of the tuning half); fste_learned only in domains where C1 adopts the learned forecaster. Then paired,
per-group sequence bootstrap of the registered comparisons, MOT-protocol identity / continuity metrics at the cross-fitted
settings, report. Refuses unless the registration is committed and unmodified and C1's VAL report exists.
Output: experiments/aerotrack_temporal/reports/C3_tracking_motion_models_<run>.{json,md}.
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "systems" / "scripts"))
import tc_data as D  # noqa: E402
import tc_forecast as F  # noqa: E402
import tc_ste as FS  # noqa: E402

RUN = REPO / "runs" / "aerotrack_temporal" / "c3"
C1RUN = REPO / "runs" / "aerotrack_temporal" / "c1"
REPORTS = REPO / "experiments" / "aerotrack_temporal" / "reports"
PRE = REPO / "experiments" / "aerotrack_temporal" / "preregistration" / "C3_tracking_motion_models.json"
GROUP_DOMAIN = {"anti_uav_IR": "antiuav_IR", "anti_uav_RGB": "antiuav_RGB", "cst_IR": "cst"}
GRID_FSTE = [dict(det_thr=a, init_thr=b, reacq_thr=c, max_coast=d, coast_output=e, gate_chi2=g)
             for a, b, c, d, e, g in itertools.product((0.05, 0.15, 0.3), (0.3, 0.5), (0.5, 0.7, 0.9), (0, 5, 20, 60), (True, False),
                                                       (9.2103, 25.0)) if not (d == 0 and not e)]
W_ = {}                                                              # worker globals


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ----------------------------------------------------------------------------------------------------------------- prep
def stage_prep():
    tune = json.loads((C1RUN / "classical_tuning.json").read_text(encoding="utf-8"))["domains"]
    tr = [s for s in D.load_split("train") if not s["dev"]]
    out = {"generated": now(), "split": "train (non-dev groups), oracle histories", "variance_tables": {}}
    for dom in D.DOMAINS:
        seqs = [s for s in tr if s["domain"] == dom]
        S = F.make_samples(seqs)
        tables = {}
        for name in ("cp", "cv", "mv2", "mv4", "mv8", "mv15", "ca", "dcv", "ou", "oudcv"):
            prm = tune[dom][name]["chosen"] if name in ("dcv", "ou", "oudcv") else None
            mean = F.forecast_classical(S, name, prm)
            r2 = np.where(S["fut_m"][..., None], (mean - S["fut_c"]) ** 2, np.nan)
            tables[name] = {"params": prm, "var": np.nanmean(r2, axis=0).tolist()}         # [H, 2] pixel^2
        out["variance_tables"][dom] = {"reference": tune[dom]["reference"], "tables": tables}
        del S
    book = json.loads((C1RUN / "learned_registry.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(HERE))
    import tc_c1_val
    arch = tc_c1_val.chosen_arch(book)[0]
    name = f"c1_{arch}_abs_s0"
    rc = subprocess.call([sys.executable, str(HERE / "tc_export.py"), "--name", name], cwd=str(REPO))
    exp = json.loads((REPORTS / "export_forecasters.json").read_text(encoding="utf-8")).get(name, {})
    out["learned"] = {"name": name, "arch": arch, "export_rc": rc, "onnx": exp.get("onnx"), "onnx_sha256": exp.get("onnx_sha256"),
                      "parity_ok": exp.get("parity_ok"), "checkpoint_sha256": book[name]["sha256"]}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "prep.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({d: v["reference"] for d, v in out["variance_tables"].items()}), json.dumps(out["learned"]))


# ----------------------------------------------------------------------------------------------------------------- run
def learned_onnx(name):
    """ONNX of a C1 / C1c checkpoint (exported with parity check if missing)."""
    path = C1RUN / "onnx" / f"{name}.onnx"
    if not path.exists():
        rc = subprocess.call([sys.executable, str(HERE / "tc_export.py"), "--name", name], cwd=str(REPO))
        if rc != 0 or not path.exists():
            raise SystemExit(f"ONNX export of {name} failed (rc {rc})")
    return path


def forecasters(prep, adopted):
    """adopted: {domain: tc_adoption.offered(domain)} for domains with a learned forecaster (amendment 2026-10-05)."""
    import tc_learned as M
    out = {}
    for grp, dom in GROUP_DOMAIN.items():
        v = prep["variance_tables"][dom]
        ref = v["reference"]
        cf = FS.ClassicalForecaster(ref, v["tables"][ref]["params"], v["tables"][ref]["var"])
        out[("fste_classical", grp)] = cf
        if dom in adopted:
            o = adopted[dom]
            onehot = [1.0 if dom == d else 0.0 for d in M.DOMAIN_TOKENS] if o["dom_tok"] else None
            out[("fste_learned", grp)] = FS.LearnedForecaster(learned_onnx(o["names"][0]), True, cf, dom_onehot=onehot)
    return out


def init_worker(run, corpus, prep, adopted):
    import track_eval as TE
    W_["TE"] = TE
    W_["seqs"] = TE.load(run, corpus)[1]
    W_["fc"] = forecasters(prep, adopted)


def run_fste(kind, params, it, sot=True):
    TE = W_["TE"]
    s, dets = it["seq"], it["dets"]
    dom = GROUP_DOMAIN[it["group"]]
    W, H = D.SIZES[dom]
    trk = FS.FSTE(FS.FSTEParams(**params), W_["fc"][(kind, it["group"])], W, H)
    res = [[] for _ in range(s.n_frames)]
    if sot:
        start = s.start_frame
        init = s.box_xywh(start)
        res[start] = [float(v) for v in init]
        trk.reset(np.array([init[0], init[1], init[0] + init[2], init[1] + init[3]], dtype=np.float64))
        frames = range(start + 1, s.n_frames)
    else:
        start = 0
        trk.reset(None)
        frames = range(s.n_frames)
    for t in frames:
        b, sc, cl = dets[t]
        box = trk.step(b, sc, cl)[0]
        if box is not None:
            res[t] = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
    return {"sequence": s.name, "n_frames": s.n_frames, "start_frame": int(start), "res": res,
            "processed": [t >= start for t in range(s.n_frames)], "n_fallback": 0, "fps": None}


def score(args):
    kind, params = args
    TE = W_["TE"]
    per = {}
    for k, it in W_["seqs"].items():
        if kind.startswith("fste") and (kind, it["group"]) not in W_["fc"]:
            continue
        pred = run_fste(kind, params, it) if kind.startswith("fste") else TE.run_tracker(kind, params, it)
        mask = np.arange(it["seq"].n_frames) >= pred["start_frame"]
        per[k] = {"SA": TE.sequence_metrics(pred["res"], it["seq"].exist, it["seq"].gt_rect, mask=mask)["SA"],
                  **TE.extra(pred, it["seq"], it.get("attrs"))}
    return per


def mot_block(kind, params_per_half, seqs):
    """MOT protocol (no init box) at the cross-fitted settings: HOTA / CLEAR / continuity per group (single identity per tracker)."""
    import mot_eval as ME
    import ste as T
    import tc_track_eval as TT
    import tc_track_metrics as TM
    acc = {}
    for k, it in sorted(seqs.items()):
        if kind.startswith("fste") and (kind, it["group"]) not in W_["fc"]:
            continue
        p = params_per_half[it["half"]]
        s = it["seq"]
        _, g_by, occ = ME.mot_inputs(it)
        b, sc, cl = zip(*it["dets"])
        if kind == "argmax":
            pr = ME.single_id({t: T.argmax_step(b[t], sc[t], cl[t], p["thr"]) for t in range(s.n_frames)})
        elif kind == "ste":
            trk = T.STE(T.STEParams(**{**T.STEParams().as_dict(), **p}))
            trk.reset(None)
            pr = ME.single_id({t: trk.step(b[t], sc[t], cl[t])[0] for t in range(s.n_frames)})
        else:
            res = run_fste(kind, p, it, sot=False)["res"]
            pr = ME.single_id({t: (np.array([r[0], r[1], r[0] + r[2], r[1] + r[3]]) if len(r) == 4 else None) for t, r in enumerate(res)})
        gt = TT.to_tc(g_by, s.n_frames)
        pt = TT.to_tc(pr, s.n_frames)
        a = acc.setdefault(it["group"], {"clear": [], "hota": [], "cont": []})
        a["clear"].append(ME.evaluate_tracking(pr, g_by, occlusion_frames=occ))
        a["hota"].append(TM.hota_sequence(gt, pt, s.n_frames))
        a["cont"] += TM.continuity(gt, pt, s.n_frames)
    return {g: {"hota": TM.hota_summary(TM.hota_combine(a["hota"])), "clear": ME.CM._accumulate(a["clear"]),
                "continuity": TM.continuity_summary(a["cont"])} for g, a in acc.items()}


def boot(a, b, keys, B=2000):
    d = np.array([a[k] - b[k] for k in keys]) * 100
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return {"n": len(d), "mean_delta": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}


def verdict(r):
    if r["mean_delta"] > 0.5 and r["ci95"][0] > 0:
        return "SUPPORTED"
    if r["ci95"][1] < 0:
        return "HARMFUL"
    return "NO_DIFFERENCE"


def stage_run(run, corpus, workers):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PRE)], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(PRE)], cwd=REPO).returncode == 0
    c1 = REPORTS / "C1_trajectory_forecasting.json"
    if not (tracked and clean):
        raise SystemExit("C3 registration must be committed and unmodified")
    if not c1.exists():
        raise SystemExit("C1's VAL report is required first (adoption decision for fste_learned)")
    import tc_adoption
    adopted = {d: o for d in D.DOMAINS if (o := tc_adoption.offered(d)) is not None}       # C1 / C1c rule (amendment)
    prep = json.loads((RUN / "prep.json").read_text(encoding="utf-8"))
    import track_eval as TE
    index, seqs = TE.load(run, corpus)
    t0 = time.time()
    jobs = ([("argmax", p) for p in TE.GRID_ARGMAX] + [("ste", p) for p in TE.GRID_STE] + [("fste_classical", p) for p in GRID_FSTE] +
            ([("fste_learned", p) for p in GRID_FSTE] if adopted else []))
    with Pool(workers, initializer=init_worker, initargs=(run, corpus, prep, adopted)) as pool:
        results = pool.map(score, jobs, chunksize=2)
    allk = sorted(seqs)
    halves = {h: [k for k in allk if seqs[k]["half"] == h] for h in (0, 1)}
    rep = {"generated": now(), "run": run, "corpus": corpus, "cache_set_sha256": index["set_sha256"], "window": index["window"],
           "adopted_learned": {d: {k: o[k] for k in ("source", "arm", "names")} for d, o in adopted.items()}, "prep": {"learned": prep["learned"],
                                                        "references": {d: v["reference"] for d, v in prep["variance_tables"].items()}},
           "grid_sizes": {"argmax": len(TE.GRID_ARGMAX), "ste": len(TE.GRID_STE), "fste": len(GRID_FSTE)}, "trackers": {}}
    cfsa = {}
    for kind in ("argmax", "ste", "fste_classical", "fste_learned"):
        idx = [i for i, (k, _) in enumerate(jobs) if k == kind]
        if not idx:
            continue
        cross, chosen = {}, {}
        for h in (0, 1):
            tune = [k for k in halves[1 - h] if k in results[idx[0]]]
            test = [k for k in halves[h] if k in results[idx[0]]]
            best = max(idx, key=lambda i: TE.objective(results[i], tune, seqs))
            chosen[h] = jobs[best][1]
            for k in test:
                cross[k] = results[best][k]
        keys = sorted(cross)
        rep["trackers"][kind] = {"cross_fitted": TE.summarize(cross, keys, seqs), "params_per_half": chosen}
        cfsa[kind] = {k: cross[k]["SA"] for k in keys}
    comps = {"H1_temporal_state": ("ste", "argmax"), "H2_domain_motion_model": ("fste_classical", "ste"),
             "H3_learned_motion_model": ("fste_learned", "fste_classical")}
    rep["hypotheses"] = {}
    for hid, (a, b) in comps.items():
        rep["hypotheses"][hid] = {}
        for grp in GROUP_DOMAIN:
            if a not in cfsa or b not in cfsa:
                rep["hypotheses"][hid][grp] = {"verdict": "NOT RUN"}
                continue
            keys = [k for k in cfsa[a] if seqs[k]["group"] == grp and k in cfsa[b]]
            if not keys:
                rep["hypotheses"][hid][grp] = {"verdict": "NOT RUN (no learned forecaster adopted by C1 / C1c)" if a == "fste_learned" else "NOT RUN"}
                continue
            r = boot(cfsa[a], cfsa[b], keys)
            rep["hypotheses"][hid][grp] = dict(r, verdict=verdict(r))
    W_["TE"], W_["fc"] = TE, forecasters(prep, adopted)               # MOT-protocol pass in the main process
    for kind, rec in rep["trackers"].items():
        rec["mot_protocol"] = mot_block(kind, rec["params_per_half"], seqs)
    rep["seconds"] = round(time.time() - t0, 1)
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"C3_tracking_motion_models_{run}.json"
    path.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    md = [f"# C3 — tracking with domain / learned motion models ({run}, {corpus}, VAL)", "",
          f"*Generated {rep['generated']} by `scripts/tc_c3.py` under `preregistration/C3_tracking_motion_models.json`; SOT protocol, mSA by "
          f"Track A's code; every tracker cross-fitted over the two VAL halves. Learned forecaster (C1 / C1c rule): "
          f"{ {d: o['source'] + ':' + o['names'][0] for d, o in adopted.items()} or 'none'}.*", "",
          "| tracker | group | n | mSA | located | false alarms | recover median (frames) |", "|---|---|---:|---:|---:|---:|---:|"]
    for kind, rec in rep["trackers"].items():
        for g, v in rec["cross_fitted"].items():
            fa = "—" if v["false_alarm_rate"] is None else f"{100 * v['false_alarm_rate']:.1f} %"
            rc = "—" if v["recover_frames_median"] is None else f"{v['recover_frames_median']:.0f}"
            md.append(f"| {kind} | {g} | {v['n_sequences']} | {v['mSA']:.2f} | {100 * v['located_rate']:.1f} % | {fa} | {rc} |")
    md += ["", "MOT protocol (no init box) at the same settings:", "",
           "| tracker | group | HOTA | AssA | IDF1 | MOTA | Frag | MT / ML | loss → recovered median / never | re-entry → recovered median / never |",
           "|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    for kind, rec in rep["trackers"].items():
        for g, v in sorted(rec["mot_protocol"].items()):
            h, c, k = v["hota"], v["clear"], v["continuity"]
            rl, rr = k["recovery_loss"], k["recovery_reentry"]
            fmt = lambda x: "—" if x is None else f"{x:.0f}"  # noqa: E731
            md.append(f"| {kind} | {g} | {h['HOTA']:.3f} | {h['AssA']:.3f} | {c['IDF1_mean_per_sequence']:.3f} | {c['MOTA']:.3f} | "
                      f"{k['fragments']} | {k['MT']:.2f} / {k['ML']:.2f} | {fmt(rl['median_frames'])} / {rl['never']} of {rl['events']} | "
                      f"{fmt(rr['median_frames'])} / {rr['never']} of {rr['events']} |")
    md += ["", "| hypothesis | group | Δ mSA [95 % CI] | wins / losses | verdict |", "|---|---|---|---|---|"]
    for hid, by in rep["hypotheses"].items():
        for g, r in by.items():
            if "ci95" in r:
                md.append(f"| {hid} | {g} | {r['mean_delta']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {r['wins']} / {r['losses']} | **{r['verdict']}** |")
            else:
                md.append(f"| {hid} | {g} | — | — | {r['verdict']} |")
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("prep", "run"))
    ap.add_argument("--run", default="v2_causal_g4_s0_e20")
    ap.add_argument("--corpus", default="data_unified_v2")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "prep":
        stage_prep()
    else:
        stage_run(a.run, a.corpus, a.workers)


if __name__ == "__main__":
    main()
