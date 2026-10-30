#!/usr/bin/env python
"""Official scoring + registered paired comparisons of a CST TEST pass (runs on the OMEN after the pass's files are pulled).

    venv\\Scripts\\python.exe experiments\\cst_anti_uav\\scripts\\analyze_test.py --x1
    venv\\Scripts\\python.exe experiments\\cst_anti_uav\\scripts\\analyze_test.py --x2

X1 (amendment X1-A1 of experiments/aerotrack_v3/preregistration_X1_cst_protocol_matched.json): the exported STE / argmax outputs
(runs/aerotrack_v3/x1_test_predictions/) are scored by the unmodified official toolkit; the toolkit's mSA must equal the pass's Track A
value; X1-a2 = paired per-sequence comparison with the released CST-trained GlobalTrack (and SiamDT, descriptive).
X2 (experiments/cst_anti_uav/preregistration/X2_seeds_single_frame.json): every TEST run of the X2 pass (+ X1's seed 0) scored the
same way; T1 per-seed exceedance of 35.92, T2 seed-mean paired vs GlobalTrack, T3 temporal vs single-frame (seed-mean, paired).

Statistics (registered): sequence bootstrap, B = 10000, numpy default_rng(0), percentile 95 % CI; two-sided Wilcoxon signed-rank
(scipy, zero differences dropped) reported alongside; wins / losses = sequences with a positive / negative difference.
Output: experiments/cst_anti_uav/reports/{X1,X2}_test_official.{json,md}.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REPORTS = REPO / "experiments" / "cst_anti_uav" / "reports"
PUBLISHED = REPORTS / "official_reproduction_test" / "official_scores.json"
B = 10000
BEST_PUBLISHED = 35.92


def paired(a, b, names):
    """a, b: name -> SA (percent). Bootstrap CI of mean(a - b) over sequences."""
    from scipy.stats import wilcoxon
    d = np.array([a[n] - b[n] for n in names])
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 1 else None
    return {"n": len(d), "mean_delta": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()), "wilcoxon_p_two_sided": p}


def mean_ci(v):
    v = np.asarray(v, float)
    rng = np.random.default_rng(0)
    boots = v[rng.integers(0, len(v), (B, len(v)))].mean(axis=1)
    return float(v.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def official(trackers, out):
    """trackers: name -> predictions dir. Runs official_eval.py once (fresh output dir) and returns its tracker records."""
    if not (out / "official_scores.json").exists():
        cmd = [sys.executable, str(HERE / "official_eval.py"), "--subset", "test", "--out", str(out)]
        for n, d in trackers.items():
            cmd += ["--tracker", f"{n}={d}"]
        subprocess.run(cmd, check=True, cwd=str(REPO))
    return json.loads((out / "official_scores.json").read_text(encoding="utf-8"))["trackers"]


def decompose(pred_dir, names):
    """Descriptive split of Eq. 1 per sequence: mean IoU over visible frames, accuracy over absent frames (official loader,
    Track A per-frame scores). The released trackers never output 'absent' (verified on all 21), so on absent frames they all
    score 0; the visible-frame IoU isolates localisation from absence reporting."""
    import official_eval as OE
    from anti_uav_repro.eval.metrics import per_frame_scores
    out = {}
    for n in names:
        lab = json.loads((OE.DATA_ROOT / "test" / n / "IR_label.json").read_text(encoding="utf-8"))
        res = OE._load(Path(pred_dir) / f"{n}.txt")
        res = [list(r) if len(r) == 4 else [] for r in res]
        f = per_frame_scores(res, lab["exist"], lab["gt"])
        ex = f["exist"]
        out[n] = {"visible_iou": 100 * float(f["sa"][ex].mean()), "absent_acc": 100 * float(f["sa"][~ex].mean()) if (~ex).any() else None,
                  "n_vis": int(ex.sum()), "n_abs": int((~ex).sum())}
    return out


def decomposition_block(ours_dir, names):
    pub_root = REPO / "data" / "cst_anti_uav_official" / "tracking_results" / "TrainedCST"
    g = json.loads(PUBLISHED.read_text(encoding="utf-8"))["trackers"]["GlobalTrack"]["path"]
    ours, glob = decompose(ours_dir, names), decompose(REPO / g if not Path(g).is_absolute() else g, names)
    vis = paired({n: ours[n]["visible_iou"] for n in names}, {n: glob[n]["visible_iou"] for n in names}, names)
    fr = lambda d, k: sum(d[n][k] * d[n]["n_vis" if k == "visible_iou" else "n_abs"] for n in names if d[n][k] is not None) / \
        sum(d[n]["n_vis" if k == "visible_iou" else "n_abs"] for n in names if d[n][k] is not None)
    return {"ours_visible_iou_frame_mean": fr(ours, "visible_iou"), "ours_absent_acc_frame_mean": fr(ours, "absent_acc"),
            "GlobalTrack_visible_iou_frame_mean": fr(glob, "visible_iou"), "GlobalTrack_absent_acc_frame_mean": fr(glob, "absent_acc"),
            "visible_iou_vs_GlobalTrack_paired": vis, "results_root_checked": str(pub_root.relative_to(REPO)),
            "note": "descriptive (not a registered claim); sequence-level visible-frame mean IoU, paired over the 60 sequences"}


def fmt_p(p):
    return "n/a (all differences zero)" if p is None else f"{p:.2g}"


def verdict(ci, ref=0.0):
    return "exceeds" if ci[0] > ref else "below" if ci[1] < ref else "not distinguishable"


def x1():
    fin = json.loads((REPO / "experiments" / "aerotrack_v3" / "reports" / "x1_final_test.json").read_text(encoding="utf-8"))
    pred = REPO / fin["predictions_dir"]
    off = official({"AeroTrack-X1-STE": pred / "ste", "AeroTrack-X1-argmax": pred / "argmax"}, REPORTS / "X1_test_official")
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8"))["trackers"]
    names = sorted(pub["GlobalTrack"]["per_sequence_SA"])
    res = {"source_pass": "experiments/aerotrack_v3/reports/x1_final_test.json", "candidate": fin["candidate"],
           "weights_sha256": fin["weights_sha256"], "trackers": {}}
    for kind, n in (("ste", "AeroTrack-X1-STE"), ("argmax", "AeroTrack-X1-argmax")):
        o = off[n]
        ta = fin["per_sequence_SA"][kind]
        mism = max(abs(o["per_sequence_SA"][s] - ta[s]) for s in names)
        m, ci = mean_ci([o["per_sequence_SA"][s] for s in names])
        res["trackers"][kind] = {
            "official_mSA": o["mSA"], "track_a_mSA_from_pass": float(np.mean([ta[s] for s in names])), "max_abs_seq_diff_official_vs_pass": mism,
            "success_auc": o["success_auc"], "precision_20px": o["precision_20px"], "ci95_mSA": ci,
            "X1_a_rule": verdict(ci, BEST_PUBLISHED), "frame_attr_SA": o["frame_attr_SA_corrected"],
            "frame_attr_SA_official_toolkit": o["frame_attr_SA_official"], "sequence_attr_SA": o["sequence_attr_SA"],
            "vs_GlobalTrack_paired": paired(o["per_sequence_SA"], pub["GlobalTrack"]["per_sequence_SA"], names),
            "vs_SiamDT_paired": paired(o["per_sequence_SA"], pub["SiamDT"]["per_sequence_SA"], names),
            "per_sequence_SA": o["per_sequence_SA"]}
    s = res["trackers"]["ste"]
    s["X1_a2"] = "exceeds GlobalTrack (paired)" if s["vs_GlobalTrack_paired"]["ci95"][0] > 0 else \
        "below GlobalTrack (paired)" if s["vs_GlobalTrack_paired"]["ci95"][1] < 0 else "not distinguishable from GlobalTrack (paired)"
    res["published_reference"] = {n: {"mSA": pub[n]["mSA"], "tiny": pub[n]["frame_attr_SA_corrected"]["tiny"]["SA"]}
                                  for n in ("GlobalTrack", "SiamDT", "PrDiMP")}
    res["tracker_ms_per_frame"] = fin.get("tracker_ms_per_frame")
    res["decomposition_ste"] = decomposition_block(pred / "ste", names)
    write(res, "X1_test_official", md_x1(res))


def md_x1(r):
    s, a = r["trackers"]["ste"], r["trackers"]["argmax"]
    g, sd = s["vs_GlobalTrack_paired"], s["vs_SiamDT_paired"]
    p = r["published_reference"]
    L = ["# X1 on CST TEST — official toolkit scoring and paired comparison (amendment X1-A1)", "",
         f"Candidate `{r['candidate']}` (weights `{r['weights_sha256'][:12]}…`), single registered pass `{r['source_pass']}`.", "",
         "| tracker | official mSA | 95 % CI | success AUC | P@20px | tiny SA | rule X1-a (vs 35.92) |", "|---|---:|---|---:|---:|---:|---|"]
    for n, t in (("AeroTrack-X1 STE", s), ("AeroTrack-X1 argmax", a)):
        L.append(f"| {n} | **{t['official_mSA']:.2f}** | [{t['ci95_mSA'][0]:.2f}, {t['ci95_mSA'][1]:.2f}] | {t['success_auc']:.2f} | "
                 f"{t['precision_20px']:.2f} | {t['frame_attr_SA']['tiny']['SA']:.2f} | {t['X1_a_rule']} |")
    for n in ("GlobalTrack", "SiamDT", "PrDiMP"):
        L.append(f"| {n} (released, CST-trained) | {p[n]['mSA']:.2f} | — | | | {p[n]['tiny']:.2f} | |")
    L += ["", f"Official toolkit vs the pass's Track A values: max per-sequence |Δ| = {s['max_abs_seq_diff_official_vs_pass']:.1e} (STE), "
          f"{a['max_abs_seq_diff_official_vs_pass']:.1e} (argmax).", "",
          f"**X1-a2 (paired, 60 sequences):** STE − GlobalTrack = {g['mean_delta']:+.2f} SA, 95 % CI [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}], "
          f"wins {g['wins']} / losses {g['losses']}, Wilcoxon p = {g['wilcoxon_p_two_sided']:.2g} → **{s['X1_a2']}**.",
          f"Descriptive: STE − SiamDT = {sd['mean_delta']:+.2f} [{sd['ci95'][0]:+.2f}, {sd['ci95'][1]:+.2f}], wins {sd['wins']} / losses {sd['losses']}.",
          "", f"Tracker cost (from cached detections, HPC CPU): {r.get('tracker_ms_per_frame')} ms / frame."]
    dc = r["decomposition_ste"]
    v = dc["visible_iou_vs_GlobalTrack_paired"]
    L += ["", "**Decomposition (descriptive).** Visible frames: mean IoU STE "
          f"{dc['ours_visible_iou_frame_mean']:.2f} vs GlobalTrack {dc['GlobalTrack_visible_iou_frame_mean']:.2f}; absent frames: correct "
          f"'absent' STE {dc['ours_absent_acc_frame_mean']:.2f} % vs GlobalTrack {dc['GlobalTrack_absent_acc_frame_mean']:.2f} % (no released "
          f"tracker ever reports absence). Sequence-level visible-frame IoU, STE − GlobalTrack (paired): {v['mean_delta']:+.2f} "
          f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}], wins {v['wins']} / losses {v['losses']}."]
    return L


def write(res, stem, md):
    (REPORTS / f"{stem}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REPORTS / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def x2():
    import x2lib
    fin = json.loads((REPO / x2lib.FINAL_JSON).read_text(encoding="utf-8"))
    x1f = json.loads((REPO / "experiments" / "aerotrack_v3" / "reports" / "x1_final_test.json").read_text(encoding="utf-8"))
    preds = {r: REPO / d for r, d in fin["predictions_dirs"].items()}
    preds[x2lib.X1_RUN] = REPO / x1f["predictions_dir"]
    trackers = {f"{r}-{k}": d / k for r, d in preds.items() for k in ("ste", "argmax")}
    off = official(trackers, REPORTS / "X2_test_official")
    pass_sa = {r: v["per_sequence_SA"] for r, v in fin["runs"].items()}
    pass_sa[x2lib.X1_RUN] = x1f["per_sequence_SA"]
    agree = {}
    for r in preds:                    # registration: the official toolkit must agree with each pass's own Track A values
        for k in ("ste", "argmax"):
            o = off[f"{r}-{k}"]["per_sequence_SA"]
            agree[f"{r}-{k}"] = max(abs(o[s] - pass_sa[r][k][s]) for s in pass_sa[r][k])
            if set(o) != set(pass_sa[r][k]) or agree[f"{r}-{k}"] > 1e-6:
                raise RuntimeError(f"{r} {k}: official toolkit disagrees with the pass (max |d| {agree[f'{r}-{k}']})")
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8"))["trackers"]
    names = sorted(pub["GlobalTrack"]["per_sequence_SA"])
    runs = {}
    for r in preds:
        o = off[f"{r}-ste"]
        m, ci = mean_ci([o["per_sequence_SA"][s] for s in names])
        runs[r] = {"official_mSA": o["mSA"], "ci95": ci, "rule_vs_35_92": verdict(ci, BEST_PUBLISHED), "success_auc": o["success_auc"],
                   "precision_20px": o["precision_20px"], "tiny_SA": o["frame_attr_SA_corrected"]["tiny"]["SA"],
                   "frame_attr_SA": o["frame_attr_SA_corrected"], "frame_attr_SA_official_toolkit": o["frame_attr_SA_official"],
                   "sequence_attr_SA": o["sequence_attr_SA"], "per_sequence_SA": o["per_sequence_SA"],
                   "argmax_official_mSA": off[f"{r}-argmax"]["mSA"], "argmax_per_sequence_SA": off[f"{r}-argmax"]["per_sequence_SA"]}
    arms = {arm: [x2lib.run_name(arm, s) for s in x2lib.SEEDS] for arm in x2lib.ARMS}
    seedmean = {arm: {n: float(np.mean([runs[r]["per_sequence_SA"][n] for r in rs])) for n in names} for arm, rs in arms.items()}
    res = {"runs": runs, "arms": arms, "official_vs_pass_max_abs_seq_diff": agree,
           "T1": {arm: {"per_seed": {r: runs[r]["rule_vs_35_92"] for r in rs},
                        "all_seeds_exceed": all(runs[r]["rule_vs_35_92"] == "exceeds" for r in rs),
                        "seed_mean_mSA": float(np.mean([runs[r]["official_mSA"] for r in rs])),
                        "seed_sd_mSA": float(np.std([runs[r]["official_mSA"] for r in rs], ddof=1))} for arm, rs in arms.items()},
           "T2": {arm: paired(seedmean[arm], pub["GlobalTrack"]["per_sequence_SA"], names) for arm in arms},
           "T3": {"temporal_minus_single_frame_seed_mean": paired(seedmean["causal_g4"], seedmean["ttt"], names),
                  "per_seed": {s: paired(runs[x2lib.run_name("causal_g4", s)]["per_sequence_SA"],
                                         runs[x2lib.run_name("ttt", s)]["per_sequence_SA"], names) for s in x2lib.SEEDS}}}
    for arm in arms:
        res["T2"][arm]["verdict"] = verdict(res["T2"][arm]["ci95"])
    res["T3"]["verdict"] = {"exceeds": "SUPPORTED", "below": "OPPOSITE", "not distinguishable": "NOT DETECTED"}[   # registered labels
        verdict(res["T3"]["temporal_minus_single_frame_seed_mean"]["ci95"])]
    L = ["# X2 on CST TEST — seeds and the single-frame control (official toolkit)", "",
         "| run | arm | seed | official mSA | 95 % CI | vs 35.92 | S-AUC | P@20 | tiny SA |", "|---|---|---:|---:|---|---|---:|---:|---:|"]
    for arm, rs in arms.items():
        for s, r in zip(x2lib.SEEDS, rs):
            v = runs[r]
            L.append(f"| {r} | {arm} | {s} | {v['official_mSA']:.2f} | [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] | {v['rule_vs_35_92']} | "
                     f"{v['success_auc']:.2f} | {v['precision_20px']:.2f} | {v['tiny_SA']:.2f} |")
    L.append("")
    L.append(f"Official toolkit vs each pass's Track A values: max per-sequence |Δ| = {max(agree.values()):.1e} over {len(agree)} tracker outputs.")
    L.append("")
    for arm in arms:
        t1, t2 = res["T1"][arm], res["T2"][arm]
        L.append(f"- **T1 {arm}:** seed mean {t1['seed_mean_mSA']:.2f} (sd {t1['seed_sd_mSA']:.2f}); all seeds exceed 35.92: "
                 f"**{t1['all_seeds_exceed']}**. **T2** seed-mean − GlobalTrack (paired): {t2['mean_delta']:+.2f} "
                 f"[{t2['ci95'][0]:+.2f}, {t2['ci95'][1]:+.2f}], wins {t2['wins']} / losses {t2['losses']} → {t2['verdict']}")
    t3 = res["T3"]["temporal_minus_single_frame_seed_mean"]
    L.append(f"- **T3** causal_g4 − single frame (seed means, paired): {t3['mean_delta']:+.2f} [{t3['ci95'][0]:+.2f}, {t3['ci95'][1]:+.2f}], "
             f"wins {t3['wins']} / losses {t3['losses']}, Wilcoxon p = {fmt_p(t3['wilcoxon_p_two_sided'])} → **{res['T3']['verdict']}**; per seed: "
             + ", ".join(f"s{s} {v['mean_delta']:+.2f}" for s, v in res["T3"]["per_seed"].items()))
    write(res, "X2_test_official", L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--x1", action="store_true")
    g.add_argument("--x2", action="store_true")
    a = ap.parse_args()
    x1() if a.x1 else x2()
