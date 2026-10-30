#!/usr/bin/env python
"""Track C, C1b: forecasting on TRACKER histories (preregistration/C1b_forecasting_on_tracker_histories.json).

    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_c1b.py --run v2_causal_g4_s0_e20 [--corpus data_unified_v2]

Histories: ste.py under the SOT protocol with C3's cross-fitted settings for this cache (each VAL half uses the other half's
choice); a frame is a history point only when the tracker associated a detection there (init / track / reacquired; the official
start-frame box counts, as the tracker is given it). Anchors additionally need the target visible at t in the ground truth.
Forecasters, metrics and rules as registered (C1's TRAIN-tuned classical forecasters, C1's learned checkpoints by sha256).
Output: experiments/aerotrack_temporal/reports/C1b_forecasting_on_tracker_histories_<run>.{json,md}.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "systems" / "scripts"))
import tc_c1 as C  # noqa: E402
import tc_c1_val as V  # noqa: E402
import tc_data as D  # noqa: E402
import tc_forecast as F  # noqa: E402

PRE = REPO / "experiments" / "aerotrack_temporal" / "preregistration" / "C1b_forecasting_on_tracker_histories.json"
GROUP_DOMAIN = {"anti_uav_IR": "antiuav_IR", "anti_uav_RGB": "antiuav_RGB", "cst_IR": "cst"}


def tracker_history(it, params):
    """SOT run of the STE -> (vis[T] associations, box[T,4] xywh px)."""
    import ste as T
    s, dets = it["seq"], it["dets"]
    T_ = s.n_frames
    vis = np.zeros(T_, bool)
    box = np.full((T_, 4), np.nan)
    start = s.start_frame
    init = s.box_xywh(start)
    vis[start], box[start] = True, init
    trk = T.STE(T.STEParams(**{**T.STEParams().as_dict(), **params}))
    trk.reset(np.array([init[0], init[1], init[0] + init[2], init[1] + init[3]], dtype=np.float64))
    for t in range(start + 1, T_):
        b, mode, _ = trk.step(*dets[t])
        if b is not None and mode in ("init", "track", "reacquired"):
            vis[t] = True
            box[t] = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
    return vis, box


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_causal_g4_s0_e20")
    ap.add_argument("--corpus", default="data_unified_v2")
    a = ap.parse_args()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PRE)], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(PRE)], cwd=REPO).returncode == 0
    c3 = C.REPORTS / f"C3_tracking_motion_models_{a.run}.json"
    if not (tracked and clean):
        raise SystemExit("C1b registration must be committed and unmodified")
    if not c3.exists():
        raise SystemExit(f"{c3.name} missing: C1b uses C3's cross-fitted tracker settings for this cache")
    import tc_learned as M
    import track_eval as TE
    halves = json.loads(c3.read_text(encoding="utf-8"))["trackers"]["ste"]["params_per_half"]
    tune = json.loads((C.RUN / "classical_tuning.json").read_text(encoding="utf-8"))["domains"]
    book = json.loads((C.RUN / "learned_registry.json").read_text(encoding="utf-8"))
    arch = V.chosen_arch(book)[0]
    models = {}
    for variant in ("abs", "aug"):
        for seed in C.SEEDS:
            name = f"c1_{arch}_{variant}_s{seed}"
            path = C.RUN / f"{name}.pt"
            if hashlib.sha256(path.read_bytes()).hexdigest() != book[name]["sha256"]:
                raise SystemExit(f"{path} does not match its recorded sha256")
            models[name] = M.load(path)
    import tc_adoption                                          # amendment 2026-10-05: learned arm per domain = C1 / C1c rule
    c1c_book = json.loads((C.RUN / "learned_registry_c1c.json").read_text(encoding="utf-8")) if (C.RUN / "learned_registry_c1c.json").exists() else {}
    offered = {d: o for d in GROUP_DOMAIN.values() if (o := tc_adoption.offered(d)) is not None}
    for o in offered.values():
        if o["source"] == "C1c":
            for name in o["names"]:
                if name not in models:
                    path = C.RUN / f"{name}.pt"
                    if hashlib.sha256(path.read_bytes()).hexdigest() != c1c_book[name]["sha256"]:
                        raise SystemExit(f"{path} does not match its recorded sha256")
                    models[name] = M.load(path)
    index, cache = TE.load(a.run, a.corpus)
    hist = {k: tracker_history(it, halves[str(it["half"])] if str(it["half"]) in halves else halves[it["half"]]) for k, it in cache.items()}
    val = {s["key"]: s for s in D.load_split("val")}
    out = {"generated": C.now(), "experiment": "C1b_forecasting_on_tracker_histories", "run": a.run, "corpus": a.corpus,
           "cache_set_sha256": index["set_sha256"], "tracker": "ste_cv, C3 cross-fitted settings", "chosen_architecture": arch, "domains": {}}
    for grp, dom in GROUP_DOMAIN.items():
        keys = sorted(k for k, it in cache.items() if it["group"] == grp and k in val)
        if not keys:
            continue
        seqs = [val[k] for k in keys]
        S = F.make_samples(seqs, hist_source={k: hist[k] for k in keys})
        gvis = np.array([seqs[q]["vis"][t] for q, t in zip(S["seq"], S["t"])])
        S = {k: v[gvis] for k, v in S.items()}                     # target visible at the anchor in the ground truth
        t_ = tune[dom]
        preds = {"kalman": F.kalman_forecast(seqs, S, F.KalmanBank(**t_["kalman"]["chosen"]), hist_source={k: hist[k] for k in keys})}
        if t_["reference"] != "kalman":
            ref_params = t_[t_["reference"]]["chosen"] if t_["reference"] in ("dcv", "ou", "oudcv") else None
            preds[t_["reference"]] = (F.forecast_classical(S, t_["reference"], ref_params), None)
        clean_names = offered[dom]["names"] if dom in offered else [f"c1_{arch}_abs_s{s}" for s in C.SEEDS]
        for name, (model, meta) in models.items():
            if name.startswith("c1c_") and name not in clean_names:
                continue                                                  # a C1c model of another domain / arm
            preds[name] = M.predict(model, S, abs_pos=meta["abs_pos"], dom_tok=meta.get("dom_tok", False))
        summ, sums = {}, {}
        for nm, (mean, var) in preds.items():
            err = F.point_errors(S, mean)
            d2, nll = F.gauss_terms(S, mean, var) if var is not None else (None, None)
            summ[nm] = {"all": F.summarize(S, err, d2, nll)}
            sums[nm] = F.seq_sums(S, err, nll)
        sd = {int(q): dom for q in np.unique(S["seq"])}
        ref = t_["reference"]
        clean_ = clean_names
        h1 = [F.paired_bootstrap(sums[m], sums[ref], sd) for m in clean_]
        h2 = [F.paired_bootstrap(sums[m], sums["kalman"], sd, stat="nll") for m in clean_]
        h4 = [F.paired_bootstrap(sums[f"c1_{arch}_aug_s{s}"], sums[f"c1_{arch}_abs_s{s}"], sd) for s in C.SEEDS]
        cal_l = [summ[m]["all"]["calibration_error"] for m in clean_]
        out["domains"][dom] = {"n_sequences": len(keys), "n_samples": int(len(S["t"])), "reference": ref, "summaries": summ,
                               "learned_arm": offered[dom]["source"] + ":" + offered[dom]["arm"] if dom in offered else "C1:pooled (no learned forecaster adopted)",
                               "H1b_point": {"pairs": h1, "verdict": V.verdict_h1(h1)},
                               "H2b_uncertainty": {"pairs": h2, "calibration_error_learned": cal_l,
                                                   "calibration_error_kalman": summ["kalman"]["all"]["calibration_error"],
                                                   "verdict": V.verdict_h2(h2, cal_l, summ["kalman"]["all"]["calibration_error"])},
                               "H4_augmentation": {"pairs": h4, "verdict": V.verdict_h3(h4)}}
    path = C.REPORTS / f"C1b_forecasting_on_tracker_histories_{a.run}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = [f"# C1b — forecasting on tracker histories ({a.run}, {a.corpus}, VAL)", "",
          f"*Generated {out['generated']} by `scripts/tc_c1b.py`; histories = ste_cv outputs with C3's cross-fitted settings; targets = ground "
          f"truth. Learned family: {arch}.*", ""]
    for dom, d in out["domains"].items():
        md += [f"## {dom} ({d['n_sequences']} sequences, {d['n_samples']:,} anchors; classical reference `{d['reference']}`)", "",
               f"H1b **{d['H1b_point']['verdict']}** · H2b **{d['H2b_uncertainty']['verdict']}** · H4 (aug) **{d['H4_augmentation']['verdict'].replace('REVERSED', 'HARMFUL')}**", "",
               "| forecaster | ADE px | FDE@24 px | NLL | 95 % coverage | calib. error |", "|---|---:|---:|---:|---:|---:|"]
        for nm, s in d["summaries"].items():
            x = s["all"]
            f = lambda v, k=1: "—" if v is None else f"{v:.{k}f}"  # noqa: E731
            md.append(f"| {nm} | {f(x['ade_px'])} | {f(x['fde_px'])} | {f(x.get('nll'), 2)} | {f(x.get('coverage', {}).get('0.95'), 3)} | "
                      f"{f(x.get('calibration_error'), 3)} |")
        md.append("")
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
