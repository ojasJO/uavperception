#!/usr/bin/env python
"""Track C, CF: the complete causal system (preregistration/CF_final_causal_system.json + amendment CF-A2).

    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_cf.py --stage freeze [--workers 8]     VAL only
    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_cf.py --stage test --final-eval        ONE pass

freeze (VAL only): every component by its registered rule, read from the committed reports --
  detector  C2 'selected_causal_window' + F1 status -> v2_causal_g4_s0_e20 (as-built network)
  tracker   C3's adoption rule per group on the CF detector's cache (C3 report hypotheses; where H1 is not SUPPORTED, the
            'fste_classical beats argmax' clause is evaluated like C3's comparisons from C3's recorded per-half settings)
  settings  each adopted tracker type re-tuned on ALL VAL with the C3 grids, objective = mean group mSA over the groups deploying
            that type; the CF-c argmax baseline tuned the same way
  forecaster C1b's adoption rule per domain (learned family seed 0 where adopted, else the classical reference + TRAIN variance)
  -> experiments/aerotrack_temporal/frozen_cf.json (commit before the pass).
test (--final-eval; refuses unless the registration and frozen_cf.json are committed and unmodified, and refuses a second pass):
  SOT per group with the frozen trackers on the verified TEST detection cache (mSA by Track A's code, located / false alarms /
  recovery, CST size bins + frame attributes), CF-a (Anti-UAV IR vs Track A normal_ir), CF-c (vs argmax at its frozen threshold),
  MOT protocol (HOTA / CLEAR / continuity), forecasting on the CF tracker's own TEST histories (CF-b: adopted forecaster vs
  Kalman and constant position, C1 rules), CF-d descriptive. -> reports/CF_final_causal_system.{json,md}, reports/cf_test_ledger.jsonl.
Deployment latency (RTX 5050, VAL inputs) is measured separately by experiments/deployment/scripts/stack_gate.py.
"""
import argparse
import hashlib
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
import tc_c3 as C3  # noqa: E402
import tc_data as D  # noqa: E402
import tc_forecast as F  # noqa: E402

REPORTS = REPO / "experiments" / "aerotrack_temporal" / "reports"
PRE = REPO / "experiments" / "aerotrack_temporal" / "preregistration" / "CF_final_causal_system.json"
FROZEN = REPO / "experiments" / "aerotrack_temporal" / "frozen_cf.json"
LEDGER = REPORTS / "cf_test_ledger.jsonl"
CORPUS = "data_unified_v2"
GROUPS = ("anti_uav_IR", "anti_uav_RGB", "cst_IR")
GROUP_DOMAIN = C3.GROUP_DOMAIN
TRACK_A_IR = REPO / "experiments" / "anti_uav" / "runs" / "normal_ir" / "eval_test_IR_last" / "metrics.json"
W = {}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def blob(p):
    """git blob hash (line-ending independent across the OMEN / HPC checkouts)."""
    return subprocess.run(["git", "hash-object", str(p)], cwd=REPO, capture_output=True, text=True).stdout.strip()


def rd(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def committed_clean(p):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(p)], cwd=REPO).returncode == 0
    return tracked and clean


# ----------------------------------------------------------------------------------------------- tracker execution (shared)
def run_kind(kind, params, it, sot=True):
    """-> tracker prediction dict (Track A format) for one cached sequence."""
    if kind.startswith("fste"):
        return C3.run_fste(kind, params, it, sot=sot)
    return W["TE"].run_tracker(kind, params, it)


def seq_sa(kind, params, it):
    TE = W["TE"]
    pred = run_kind(kind, params, it)
    mask = np.arange(it["seq"].n_frames) >= pred["start_frame"]
    return {"SA": TE.sequence_metrics(pred["res"], it["seq"].exist, it["seq"].gt_rect, mask=mask)["SA"],
            **TE.extra(pred, it["seq"], it.get("attrs"))}


def init_worker(run, split, prep):
    import track_eval as TE
    W["TE"] = TE
    W["seqs"] = TE.load(run, CORPUS, split=split)[1]
    C3.W_["TE"], C3.W_["seqs"] = TE, W["seqs"]
    C3.W_["fc"] = C3.forecasters(prep, {}) if prep is not None else {}


def score_setting(args):
    kind, params, keys = args
    return {k: seq_sa(kind, params, W["seqs"][k]) for k in keys}


# ----------------------------------------------------------------------------------------------- freeze (VAL)
def stage_freeze(workers, smoke=0, out_path=None):
    """smoke > 0: development check only -- first `smoke` sequences per group and grid settings, written to out_path (never FROZEN)."""
    target = Path(out_path) if smoke else FROZEN
    if smoke and (out_path is None or Path(out_path).resolve() == FROZEN.resolve()):
        raise SystemExit("--smoke needs --out outside experiments/aerotrack_temporal/frozen_cf.json")
    if not smoke and FROZEN.exists():
        raise SystemExit(f"Refusing: {FROZEN.relative_to(REPO)} exists (frozen settings are written once; inspect it, do not overwrite)")
    c2 = rd(REPORTS / "C2_causal_windows.json")
    f1 = rd(REPO / "experiments" / "aerotrack_v3" / "reports" / "F1_standard_yolo11n.json")
    window = c2["selected_causal_window"]
    if f1["status"] != "NOT SUPPORTED":
        raise SystemExit("F1 is SUPPORTED: CF would use the n_std run -- not implemented (the registered branch is not this one)")
    run = f"v2_{window}_s0_e20"
    meta = rd(REPO / "runs" / "hpc" / run / "run_meta.json")
    c3p = REPORTS / f"C3_tracking_motion_models_{run}.json"
    c1bp = REPORTS / f"C1b_forecasting_on_tracker_histories_{run}.json"
    c3, c1b = rd(c3p), rd(c1bp)
    prep = rd(C3.RUN / "prep.json")
    init_worker(run, "val", prep)
    TE, seqs = W["TE"], W["seqs"]
    idx = rd(REPO / "runs" / "systems" / "_cache" / run / CORPUS / "val" / "index.json")
    if idx["weights_sha256"] != meta["last_pt_sha256"]:
        raise SystemExit("VAL cache was made with other weights than the run's recorded last.pt")
    by_group = {g: sorted(k for k, it in seqs.items() if it["group"] == g)[:smoke or None] for g in GROUPS}

    # tracker type per group (C3 adoption rule)
    hyp = c3["hypotheses"]
    trackers, checks = {}, {}
    for g in GROUPS:
        v = lambda h: hyp[h].get(g, {}).get("verdict")  # noqa: E731
        if v("H1_temporal_state") == "SUPPORTED":
            t = "ste"
            if v("H2_domain_motion_model") == "SUPPORTED":
                t = "fste_classical"
        else:
            t = "argmax"
            pph = {k2: c3["trackers"][k2]["params_per_half"] for k2 in ("argmax", "fste_classical")}
            sa = {k2: {k: seq_sa(k2, pph[k2][str(seqs[k]["half"])], seqs[k])["SA"] for k in by_group[g]} for k2 in pph}
            r = C3.boot(sa["fste_classical"], sa["argmax"], by_group[g])
            r["verdict"] = C3.verdict(r)
            checks[g] = {"fste_classical_minus_argmax_cross_fitted": r, "source": "C3 params_per_half; C3 boot / verdict"}
            if r["verdict"] == "SUPPORTED":
                t = "fste_classical"
        if t == "fste_classical" and v("H3_learned_motion_model") == "SUPPORTED":
            t = "fste_learned"
        trackers[g] = t
    if any(t == "fste_learned" for t in trackers.values()):
        raise SystemExit("fste_learned adopted somewhere: not implemented in this freeze (no such adoption is expected)")

    # settings: per adopted type, tuned on ALL VAL over the groups deploying it (CF-A2); CF-c argmax baseline the same way
    grids = {"argmax": TE.GRID_ARGMAX, "ste": TE.GRID_STE, "fste_classical": C3.GRID_FSTE}
    if smoke:
        grids = {k: v[:smoke] for k, v in grids.items()}
    tune_sets = {}
    for g, t in trackers.items():
        tune_sets.setdefault(t, []).append(g)
    jobs = [(t, tuple(gs)) for t, gs in tune_sets.items()] + [("argmax", tuple(gs)) for t, gs in tune_sets.items() if t != "argmax"]
    tuned = {}
    with Pool(workers, initializer=init_worker, initargs=(run, "val", prep)) as pool:
        for kind, gs in jobs:
            keys = [k for g in gs for k in by_group[g]]
            res = pool.map(score_setting, [(kind, p, keys) for p in grids[kind]], chunksize=1)
            obj = [float(np.mean([np.mean([r[k]["SA"] for k in by_group[g]]) for g in gs])) for r in res]
            best = int(np.argmax(obj))
            tuned[(kind, gs)] = {"params": grids[kind][best], "objective_mean_group_mSA_pct": 100 * obj[best], "grid_size": len(grids[kind]),
                                 "groups": list(gs), "group_mSA_pct": {g: 100 * float(np.mean([res[best][k]["SA"] for k in by_group[g]])) for g in gs}}
    per_group = {}
    for g, t in trackers.items():
        gs = tuple(tune_sets[t])
        per_group[g] = {"tracker": t, "params": tuned[(t, gs)]["params"], "tuning": tuned[(t, gs)],
                        "argmax_baseline": (tuned[("argmax", gs)] if ("argmax", gs) in tuned else tuned[(t, gs)])}

    # forecaster per domain (C1b adoption rule)
    fc = {}
    for g in GROUPS:
        dom = GROUP_DOMAIN[g]
        d = c1b["domains"][dom]
        h1, h2, h4 = d["H1b_point"]["verdict"], d["H2b_uncertainty"]["verdict"], d["H4_augmentation"]["verdict"]
        if h1 == "SUPPORTED" and h2 != "LOSES":
            fam = "aug" if h4 == "SUPPORTED" else "abs"
            src = d["learned_arm"]
            arch = c1b["chosen_architecture"]
            names = [f"c1_{arch}_{fam}_s{s}" for s in (0, 1, 2)] if src.startswith("C1:") else None
            if names is None:
                raise SystemExit(f"{dom}: C1c-sourced learned arm in C1b ({src}) -- not implemented")
            book = rd(REPO / "runs" / "aerotrack_temporal" / "c1" / "learned_registry.json")
            fc[dom] = {"kind": "learned", "deployed": names[0], "seeds_reported": names, "variant": fam, "source": src,
                       "checkpoint_sha256": {n: book[n]["sha256"] for n in names}, "rule": {"H1b": h1, "H2b": h2, "H4": h4}}
        else:
            ref = prep["variance_tables"][dom]["reference"]
            fc[dom] = {"kind": "classical", "deployed": ref, "params": prep["variance_tables"][dom]["tables"][ref]["params"],
                       "variance": "TRAIN residual table (runs/aerotrack_temporal/c3/prep.json)", "rule": {"H1b": h1, "H2b": h2, "H4": h4}}
    git = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
           "tc_cf_committed_clean": committed_clean(Path(__file__).resolve())}
    out = {"registration": str(PRE.relative_to(REPO).as_posix()), "amendment": "CF-A2", "written": now(), "git": git,
           "split_used": "val (VAL cache + VAL-only C2 / C3 / C1b / F1 reports; no TEST input)", "smoke": smoke or None,
           "detector": {"run": run, "window": window, "network": f1["adoption"], "weights_sha256": meta["last_pt_sha256"],
                        "val_cache_set_sha256": idx["set_sha256"]},
           "groups": per_group, "adoption_checks": checks, "forecasters": fc,
           "sources_git_blob": {p.name: blob(p) for p in (REPORTS / "C2_causal_windows.json", c3p, c1bp, C3.RUN / "prep.json",
                                                REPO / "experiments" / "aerotrack_v3" / "reports" / "F1_standard_yolo11n.json")}}
    target.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({g: {k: v[k] for k in ("tracker", "params")} | {"tuned": v["tuning"]["group_mSA_pct"],
                                                                       "argmax": v["argmax_baseline"]["params"]}
                      for g, v in per_group.items()}, indent=1))
    print(json.dumps(checks, indent=1, default=str))
    print(json.dumps({d: {k: v[k] for k in ("kind", "deployed")} for d, v in fc.items()}, indent=1))


# ----------------------------------------------------------------------------------------------- test (ONE pass)
def history_of(kind, params, it):
    """CF tracker's own SOT history for forecasting: (vis[T], box[T,4] xywh)."""
    if kind == "ste":
        import tc_c1b as B
        return B.tracker_history(it, params)
    pred = run_kind(kind, params, it)
    T_ = it["seq"].n_frames
    vis = np.zeros(T_, bool)
    box = np.full((T_, 4), np.nan)
    for t, r in enumerate(pred["res"]):
        if t >= pred["start_frame"] and len(r) == 4:
            vis[t], box[t] = True, r
    return vis, box


def forecast_block(fz, prep, seqs, test_traj):
    import tc_c1_val as V
    import tc_learned as M
    tune = rd(REPO / "runs" / "aerotrack_temporal" / "c1" / "classical_tuning.json")["domains"]
    out = {}
    for g in GROUPS:
        dom = GROUP_DOMAIN[g]
        gz = fz["groups"][g]
        keys = sorted(k for k, it in seqs.items() if it["group"] == g and k in test_traj)
        if len(keys) != sum(1 for it in seqs.values() if it["group"] == g):
            raise SystemExit(f"{g}: TEST trajectories and TEST cache do not cover the same sequences")
        hist = {k: history_of(gz["tracker"], gz["params"], seqs[k]) for k in keys}
        sq = [test_traj[k] for k in keys]
        S = F.make_samples(sq, hist_source=hist)
        gvis = np.array([sq[q]["vis"][t] for q, t in zip(S["seq"], S["t"])])
        S = {k: v[gvis] for k, v in S.items()}
        preds = {"kalman": F.kalman_forecast(sq, S, F.KalmanBank(**tune[dom]["kalman"]["chosen"]), hist_source=hist),
                 "cp": (F.forecast_classical(S, "cp", None), None)}
        f = fz["forecasters"][dom]
        if f["kind"] == "learned":
            for n in f["seeds_reported"]:
                path = REPO / "runs" / "aerotrack_temporal" / "c1" / f"{n}.pt"
                if sha(path) != f["checkpoint_sha256"][n]:
                    raise SystemExit(f"{path} does not match its frozen sha256")
                model, meta = M.load(path)
                preds[n] = M.predict(model, S, abs_pos=meta["abs_pos"], dom_tok=meta.get("dom_tok", False))
            adopted = f["seeds_reported"]
        else:
            ref = f["deployed"]
            mean = F.forecast_classical(S, ref, f["params"])
            var = np.broadcast_to(np.asarray(prep["variance_tables"][dom]["tables"][ref]["var"])[None], mean.shape).copy()
            preds[ref] = (mean, var)
            adopted = [ref]
        summ, sums = {}, {}
        for nm, (mean, var) in preds.items():
            err = F.point_errors(S, mean)
            d2, nll = F.gauss_terms(S, mean, var) if var is not None else (None, None)
            summ[nm] = F.summarize(S, err, d2, nll)
            sums[nm] = F.seq_sums(S, err, nll)
        sd = {int(q): dom for q in np.unique(S["seq"])}
        vs_k = [F.paired_bootstrap(sums[m], sums["kalman"], sd) for m in adopted]
        vs_cp = [F.paired_bootstrap(sums[m], sums["cp"], sd) for m in adopted]
        nll_k = [F.paired_bootstrap(sums[m], sums["kalman"], sd, stat="nll") for m in adopted]
        out[dom] = {"n_sequences": len(keys), "n_anchors": int(len(S["t"])), "tracker_histories": gz["tracker"], "adopted": adopted,
                    "deployed": f["deployed"], "summaries": summ,
                    "CF_b_point_vs_kalman": {"pairs": vs_k, "verdict": V.verdict_h1(vs_k)},
                    "CF_b_point_vs_constant_position": {"pairs": vs_cp, "verdict": V.verdict_h1(vs_cp)},
                    "CF_b_nll_vs_kalman": {"pairs": nll_k, "verdict": V.verdict_h2(nll_k, [summ[m]["calibration_error"] for m in adopted],
                                                                                   summ["kalman"]["calibration_error"])}}
    return out


def stage_test(final_eval, dry_val=False, frozen_path=None, out_dir=None):
    """dry_val: the identical code path on VAL (no guards, no ledger, outputs to out_dir) -- a crash check before the single pass."""
    split = "val" if dry_val else "test"
    fpath = Path(frozen_path) if dry_val else FROZEN
    odir = Path(out_dir) if dry_val else REPORTS
    if dry_val:
        if out_dir is None or Path(out_dir).resolve() == REPORTS.resolve():
            raise SystemExit("--dry-run-val needs --out outside experiments/aerotrack_temporal/reports")
        odir.mkdir(parents=True, exist_ok=True)
    else:
        if not final_eval:
            raise SystemExit("Refusing: CF's TEST pass needs --final-eval")
        for p in (PRE, FROZEN):
            if not committed_clean(p):
                raise SystemExit(f"Refusing: {p.relative_to(REPO)} must be committed and unmodified before the pass")
        if LEDGER.exists() and LEDGER.read_text(encoding="utf-8").strip():
            raise SystemExit("Refusing: CF already has its single TEST pass (cf_test_ledger.jsonl)")
    fz = rd(fpath)
    run = fz["detector"]["run"]
    prep = rd(C3.RUN / "prep.json")
    t0 = time.time()
    init_worker(run, split, prep)
    TE, seqs = W["TE"], W["seqs"]
    idx = rd(REPO / "runs" / "systems" / "_cache" / run / CORPUS / split / "index.json")
    if idx["weights_sha256"] != fz["detector"]["weights_sha256"]:
        raise SystemExit(f"{split} cache was made with other weights than the frozen detector")
    want = {"anti_uav_IR": 67, "anti_uav_RGB": 67, "cst_IR": 40} if dry_val else {"anti_uav_IR": 91, "anti_uav_RGB": 91, "cst_IR": 60}
    got = {g: sum(1 for it in seqs.values() if it["group"] == g) for g in GROUPS}
    if got != want:
        raise SystemExit(f"{split} cache groups {got} != {want}")
    sot, base, per = {}, {}, {}
    for g in GROUPS:
        gz = fz["groups"][g]
        keys = sorted(k for k, it in seqs.items() if it["group"] == g)
        per[g] = {k: seq_sa(gz["tracker"], gz["params"], seqs[k]) for k in keys}
        b = {k: seq_sa("argmax", gz["argmax_baseline"]["params"], seqs[k]) for k in keys}
        sot[g] = TE.summarize(per[g], keys, seqs)[g]
        base[g] = {"summary": TE.summarize(b, keys, seqs)[g],
                   "CF_c_tracker_minus_argmax": C3.boot({k: per[g][k]["SA"] for k in keys}, {k: b[k]["SA"] for k in keys}, keys)}
    # CF-a: Anti-UAV IR vs Track A normal_ir seed 0 (same sequences; dry run: Track A's VAL evaluation)
    ta_path = TRACK_A_IR.parent.parent / "eval_val_IR_last" / "metrics.json" if dry_val else TRACK_A_IR
    ta = rd(ta_path)["per_sequence"]
    ir = {k.split("_", 2)[2].rsplit("__", 1)[0]: v["SA"] for k, v in per["anti_uav_IR"].items()}
    if set(ir) != set(ta):
        raise SystemExit(f"CF-a: Track A normal_ir sequences ({ta_path.name}) differ from the CF Anti-UAV IR {split} sequences")
    names = sorted(ir)
    cfa = C3.boot(ir, {n: ta[n]["SA"] for n in names}, names)
    cfa["claim"] = "better" if cfa["ci95"][0] > 0 else "worse" if cfa["ci95"][1] < 0 else "no detectable difference"
    cfa["TrackA_mSA_pct"] = 100 * float(np.mean([ta[n]["SA"] for n in names]))
    cfa["TrackA_source"] = str(ta_path.relative_to(REPO).as_posix())
    # MOT protocol
    mot = {}
    for g in GROUPS:
        gz = fz["groups"][g]
        sub = {k: it for k, it in seqs.items() if it["group"] == g}
        mot[g] = C3.mot_block(gz["tracker"], {0: gz["params"], 1: gz["params"]}, sub).get(g)
    # forecasting on the CF tracker's own histories
    traj = D.load_split("val") if dry_val else D.collect("test", "final_eval")
    fcst = forecast_block(fz, prep, seqs, {s["key"]: s for s in traj})
    v2f = REPO / "experiments" / "aerotrack_v2" / "hpc" / "reports" / "final_test.json"
    rep = {"generated": now(), "split": split, "dry_run": bool(dry_val), "registration": fz["registration"],
           "frozen": str(fpath.relative_to(REPO).as_posix()) if fpath.resolve().is_relative_to(REPO) else str(fpath),
           "frozen_sha256": sha(fpath), "detector": fz["detector"], "cache_set_sha256": idx["set_sha256"],
           "sot": {g: {"tracker": fz["groups"][g]["tracker"], "params": fz["groups"][g]["params"], "summary": sot[g],
                       "per_sequence_SA": {k: 100 * v["SA"] for k, v in per[g].items()}} for g in GROUPS},
           "CF_a": cfa, "CF_c": base, "mot": mot, "forecasting": fcst,
           "CF_d": {"note": "not applicable: the v2 candidate v2_causal_g4_s0_e20 is causal (CF-A2); its v2 final TEST pass, descriptively",
                    "v2_final_test": (rd(v2f).get("sot") if v2f.exists() and not dry_val else None)},
           "seconds": round(time.time() - t0, 1)}
    (odir / "CF_final_causal_system.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    if not dry_val:
        with open(LEDGER, "a", encoding="utf-8") as f:            # written on completion (as X1's ledger): a crash may resume
            f.write(json.dumps({"timestamp": now(), "event": "pass", "frozen_sha256": rep["frozen_sha256"],
                                "test_cache_set_sha256": idx["set_sha256"], "mSA": {g: sot[g]["mSA"] for g in GROUPS},
                                "CF_a": cfa["claim"]}) + "\n")
    write_md(rep, odir)


def write_md(rep, odir):
    L = [f"# CF — the complete causal system ({'DRY RUN on VAL -- not a result' if rep['dry_run'] else 'single registered TEST pass'})", "",
         f"*Generated {rep['generated']} by `scripts/tc_cf.py --stage test` ({rep['split'].upper()}) under `{rep['registration']}` (+ CF-A2); "
         f"frozen `{rep['frozen']}` (sha256 {rep['frozen_sha256'][:12]}…); detector `{rep['detector']['run']}`.*", "",
         "| group | tracker | sequences | mSA | located | false alarms | recover median | argmax (frozen thr) mSA | CF-c Δ [95 % CI] |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for g, s in rep["sot"].items():
        v, b = s["summary"], rep["CF_c"][g]
        c = b["CF_c_tracker_minus_argmax"]
        fa = "—" if v["false_alarm_rate"] is None else f"{100 * v['false_alarm_rate']:.1f} %"
        rc = "—" if v["recover_frames_median"] is None else f"{v['recover_frames_median']:.0f}"
        L.append(f"| {g} | {s['tracker']} | {v['n_sequences']} | {v['mSA']:.2f} | {100 * v['located_rate']:.1f} % | {fa} | {rc} | "
                 f"{b['summary']['mSA']:.2f} | {c['mean_delta']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] |")
    a = rep["CF_a"]
    L += ["", f"**CF-a** (Anti-UAV IR, {a['n']} sequences) CF − Track A GlobalTrack normal_ir ({a['TrackA_mSA_pct']:.2f}): "
          f"{a['mean_delta']:+.2f} SA [{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}], wins {a['wins']} / losses {a['losses']} → **{a['claim']}** "
          "(caveat: training data differ).", "", "MOT protocol (no init box):", "",
          "| group | HOTA | AssA | IDF1 | MOTA | Frag |", "|---|---:|---:|---:|---:|---:|"]
    for g, v in rep["mot"].items():
        if v:
            L.append(f"| {g} | {v['hota']['HOTA']:.3f} | {v['hota']['AssA']:.3f} | {v['clear']['IDF1_mean_per_sequence']:.3f} | "
                     f"{v['clear']['MOTA']:.3f} | {v['continuity']['fragments']} |")
    L += ["", f"Forecasting on the CF tracker's own {rep['split'].upper()} histories (CF-b):", "",
          "| domain | anchors | deployed | ADE px (deployed / Kalman / CP) | FDE@24 px | NLL (deployed / Kalman) | vs Kalman | vs CP |",
          "|---|---:|---|---|---|---|---|---|"]
    for dom, d in rep["forecasting"].items():
        s, k, c = d["summaries"][d["deployed"]], d["summaries"]["kalman"], d["summaries"]["cp"]
        nll = "—" if s.get("nll") is None else f"{s['nll']:.2f} / {k['nll']:.2f}"
        L.append(f"| {dom} | {d['n_anchors']:,} | {d['deployed']} | {s['ade_px']:.1f} / {k['ade_px']:.1f} / {c['ade_px']:.1f} | "
                 f"{s['fde_px']:.1f} / {k['fde_px']:.1f} / {c['fde_px']:.1f} | {nll} | {d['CF_b_point_vs_kalman']['verdict']} | "
                 f"{d['CF_b_point_vs_constant_position']['verdict']} |")
    L += ["", f"CF-d: {rep['CF_d']['note']}."]
    (odir / "CF_final_causal_system.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("freeze", "test"))
    ap.add_argument("--final-eval", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", type=int, default=0, help="development check: N sequences per group and N grid settings; needs --out")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run-val", action="store_true", help="test stage's code path on VAL with --frozen <file> --out <dir>")
    ap.add_argument("--frozen", default=None)
    a = ap.parse_args()
    if a.stage == "freeze":
        stage_freeze(a.workers, a.smoke, a.out)
    else:
        stage_test(a.final_eval, a.dry_run_val, a.frozen, a.out)


if __name__ == "__main__":
    main()
