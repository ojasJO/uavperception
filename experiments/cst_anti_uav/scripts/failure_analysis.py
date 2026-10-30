#!/usr/bin/env python
"""Descriptive failure / attribute analysis of a finished CST TEST pass (no tuning; reads the frozen predictions only).

    venv\\Scripts\\python.exe experiments\\cst_anti_uav\\scripts\\failure_analysis.py --x1
    venv\\Scripts\\python.exe experiments\\cst_anti_uav\\scripts\\failure_analysis.py --x2   (after the X2 pass -> X2_failure_analysis.json)

Compares AeroTrack-X1 STE with the released CST-trained GlobalTrack and SiamDT on the 60 TEST sequences:
  - frame outcomes: visible frames -> no box / box with IoU 0 / 0 < IoU < 0.5 / IoU >= 0.5; absent frames -> correct / false alarm
  - "absence-neutral" SA: Eq. 1 with every absent frame scored 0 for every tracker (what X1 would score if it were given no credit
    for reporting absence -- the released trackers' situation); paired vs GlobalTrack
  - per-scene mSA, per-size-bin outcomes, re-appearance recovery (frames until IoU >= 0.5 after an absent -> visible transition)
  - the worst sequences of X1 with their diagnostics
Statistics as in analyze_test.py (sequence bootstrap B = 10000, default_rng(0); Wilcoxon). Everything here is descriptive.
Output: experiments/cst_anti_uav/reports/X1_failure_analysis.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_test as AT          # noqa: E402
import official_eval as OE         # noqa: E402

REPO = AT.REPO
RECOVER_WITHIN = 25                # frames (1 s at 25 fps)


def scene(n):
    return n.rsplit("_", 1)[0]


def per_sequence(pred_dir, n):
    from anti_uav_repro.eval.metrics import per_frame_scores
    lab = json.loads((OE.DATA_ROOT / "test" / n / "IR_label.json").read_text(encoding="utf-8"))
    res = OE._load(Path(pred_dir) / f"{n}.txt")
    res = [list(r) if len(r) == 4 else [] for r in res]
    f = per_frame_scores(res, lab["exist"], lab["gt"])
    ex = f["exist"]
    has_box = np.array([len(r) == 4 for r in res])
    iou = np.where(ex, np.nan_to_num(f["iou"]), np.nan)
    gt = np.array([g if len(g) == 4 else [0, 0, 0, 0] for g in lab["gt"]], float)
    diag = np.hypot(gt[:, 2], gt[:, 3])
    size = np.full(len(ex), "", dtype=object)
    for name, lo, hi in OE.SIZE_BINS:
        size[ex & (diag > lo) & (diag <= hi)] = name                     # = official_eval.size_mask
    out = {"n": len(ex), "n_vis": int(ex.sum()), "n_abs": int((~ex).sum()),
           "SA": 100 * float(f["sa"].mean()), "absence_neutral_SA": 100 * float(np.where(ex, f["sa"], 0).mean()),
           "vis_no_box": int((ex & ~has_box).sum()), "vis_iou0": int((ex & has_box & (np.nan_to_num(iou) == 0)).sum()),
           "vis_low": int((ex & has_box & (iou > 0) & (iou < 0.5)).sum()), "vis_good": int((ex & (iou >= 0.5)).sum()),
           "abs_correct": int((~ex & ~has_box).sum()), "abs_false_alarm": int((~ex & has_box).sum()),
           "size": {b: {"frames": int((size == b).sum()), "good": int(((size == b) & (iou >= 0.5)).sum()),
                        "no_box": int(((size == b) & ~has_box).sum()), "iou_sum": float(np.nansum(np.where(size == b, iou, 0)))}
                    for b, _, _ in OE.SIZE_BINS}}
    rec = []
    for t in range(1, len(ex)):
        if ex[t] and not ex[t - 1]:
            k = next((j for j in range(t, min(len(ex), t + 10 * RECOVER_WITHIN)) if ex[j] and iou[j] >= 0.5), None)
            rec.append(None if k is None else k - t)
    out["reappear"] = rec
    return out


def summarize(seqs, names):
    tot = {k: sum(seqs[n][k] for n in names) for k in ("n", "n_vis", "n_abs", "vis_no_box", "vis_iou0", "vis_low", "vis_good",
                                                        "abs_correct", "abs_false_alarm")}
    pct = {k: 100 * tot[k] / tot["n_vis"] for k in ("vis_no_box", "vis_iou0", "vis_low", "vis_good")}
    pct.update({k: 100 * tot[k] / tot["n_abs"] for k in ("abs_correct", "abs_false_alarm")})
    size = {}
    for b, _, _ in OE.SIZE_BINS:
        fr = sum(seqs[n]["size"][b]["frames"] for n in names)
        size[b] = {"frames": fr, "SA": 100 * sum(seqs[n]["size"][b]["iou_sum"] for n in names) / max(fr, 1),
                   "iou_ge_0_5_pct": 100 * sum(seqs[n]["size"][b]["good"] for n in names) / max(fr, 1),
                   "no_box_pct": 100 * sum(seqs[n]["size"][b]["no_box"] for n in names) / max(fr, 1)}
    rec = [r for n in names for r in seqs[n]["reappear"]]
    got = [r for r in rec if r is not None]
    return {"frames": tot, "pct": pct, "size": size,
            "reappearance": {"events": len(rec), "recovered_within_1s_pct": 100 * sum(r <= RECOVER_WITHIN for r in got) / max(len(rec), 1),
                             "never_within_10s": sum(r is None for r in rec), "median_frames_when_recovered": float(np.median(got)) if got else None},
            "mSA": float(np.mean([seqs[n]["SA"] for n in names])),
            "absence_neutral_mSA": float(np.mean([seqs[n]["absence_neutral_SA"] for n in names])),
            "scene_mSA": {s: float(np.mean([seqs[n]["SA"] for n in names if scene(n) == s])) for s in sorted({scene(n) for n in names})},
            "scene_n": {s: sum(scene(n) == s for n in names) for s in sorted({scene(n) for n in names})}}


def x1():
    fin = json.loads((REPO / "experiments" / "aerotrack_v3" / "reports" / "x1_final_test.json").read_text(encoding="utf-8"))
    pub = json.loads(AT.PUBLISHED.read_text(encoding="utf-8"))["trackers"]
    names = sorted(pub["GlobalTrack"]["per_sequence_SA"])
    dirs = {"AeroTrack-X1-STE": REPO / fin["predictions_dir"] / "ste"}
    for t in ("GlobalTrack", "SiamDT"):
        p = Path(pub[t]["path"])
        dirs[t] = p if p.is_absolute() else REPO / p
    per = {t: {n: per_sequence(d, n) for n in names} for t, d in dirs.items()}
    res = {"source_pass": "experiments/aerotrack_v3/reports/x1_final_test.json", "note": "descriptive; no registered claim",
           "trackers": {t: summarize(per[t], names) for t in per}}
    ours, g = per["AeroTrack-X1-STE"], per["GlobalTrack"]
    if any(abs(per[k][n]["SA"] - pub[k]["per_sequence_SA"][n]) > 1e-6 for k in ("GlobalTrack", "SiamDT") for n in names):
        raise RuntimeError("per-frame recomputation differs from the official per-sequence SA")
    res["absence_neutral_vs_GlobalTrack_paired"] = AT.paired({n: ours[n]["absence_neutral_SA"] for n in names},
                                                             {n: g[n]["absence_neutral_SA"] for n in names}, names)
    res["scene_vs_GlobalTrack"] = {s: AT.paired({n: ours[n]["SA"] for n in names if scene(n) == s},
                                                 {n: g[n]["SA"] for n in names if scene(n) == s}, [n for n in names if scene(n) == s])
                                   for s in res["trackers"]["GlobalTrack"]["scene_mSA"]}
    worst = sorted(names, key=lambda n: ours[n]["SA"])[:10]
    res["worst_x1"] = [{"seq": n, "SA": ours[n]["SA"], "GlobalTrack_SA": g[n]["SA"], "frames": ours[n]["n"], "absent_frames": ours[n]["n_abs"],
                        "vis_no_box_pct": 100 * ours[n]["vis_no_box"] / max(ours[n]["n_vis"], 1),
                        "vis_iou0_pct": 100 * ours[n]["vis_iou0"] / max(ours[n]["n_vis"], 1),
                        "vis_low_pct": 100 * ours[n]["vis_low"] / max(ours[n]["n_vis"], 1),
                        "abs_false_alarm_pct": 100 * ours[n]["abs_false_alarm"] / max(ours[n]["n_abs"], 1) if ours[n]["n_abs"] else None,
                        "tiny_frames_pct": 100 * ours[n]["size"]["tiny"]["frames"] / max(ours[n]["n_vis"], 1)} for n in worst]
    res["losses_vs_GlobalTrack"] = [{"seq": n, "SA": ours[n]["SA"], "GlobalTrack_SA": g[n]["SA"]} for n in names if ours[n]["SA"] < g[n]["SA"]]
    out = AT.REPORTS / "X1_failure_analysis.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "trackers"}, indent=1))
    for t, v in res["trackers"].items():
        print(t, json.dumps({k: v[k] for k in ("mSA", "absence_neutral_mSA", "pct", "size", "reappearance", "scene_mSA")}, indent=1))


def size_ci(per, names, b):
    """Frame-level SA of one size bin with a sequence-bootstrap 95 % CI (ratio of sums, sequences resampled; B = 10000)."""
    num = np.array([per[n]["size"][b]["iou_sum"] for n in names])
    den = np.array([per[n]["size"][b]["frames"] for n in names], float)
    rng = np.random.default_rng(0)
    ix = rng.integers(0, len(names), (AT.B, len(names)))
    boots = 100 * num[ix].sum(1) / np.maximum(den[ix].sum(1), 1)
    return {"SA": 100 * float(num.sum() / max(den.sum(), 1)), "frames": int(den.sum()), "sequences": int((den > 0).sum()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def seq_row(d, n):
    v = d[n]
    return {"SA": v["SA"], "absence_neutral_SA": v["absence_neutral_SA"],
            "vis_no_box_pct": 100 * v["vis_no_box"] / max(v["n_vis"], 1), "vis_iou0_pct": 100 * v["vis_iou0"] / max(v["n_vis"], 1),
            "vis_good_pct": 100 * v["vis_good"] / max(v["n_vis"], 1),
            "abs_false_alarm_pct": 100 * v["abs_false_alarm"] / max(v["n_abs"], 1) if v["n_abs"] else None}


def large_by_seq(d, names, min_frames=30):
    return {n: 100 * d[n]["size"]["large"]["iou_sum"] / d[n]["size"]["large"]["frames"]
            for n in names if d[n]["size"]["large"]["frames"] >= min_frames}


def x2():
    """X2 (seeds 0-2 x causal_g4 / single frame): does any run or arm change X1's weak spots (large targets, jungle_18,
    absence-neutral localisation)? Descriptive; the registered X2 claims are T1-T3 in X2_test_official.json."""
    import x2lib
    fin = json.loads((REPO / x2lib.FINAL_JSON).read_text(encoding="utf-8"))
    x1f = json.loads((REPO / "experiments" / "aerotrack_v3" / "reports" / "x1_final_test.json").read_text(encoding="utf-8"))
    pub = json.loads(AT.PUBLISHED.read_text(encoding="utf-8"))["trackers"]
    names = sorted(pub["GlobalTrack"]["per_sequence_SA"])
    dirs = {r: REPO / d / "ste" for r, d in fin["predictions_dirs"].items()}
    dirs[x2lib.X1_RUN] = REPO / x1f["predictions_dir"] / "ste"
    gp = Path(pub["GlobalTrack"]["path"])
    per = {r: {n: per_sequence(d, n) for n in names} for r, d in dirs.items()}
    g = {n: per_sequence(gp if gp.is_absolute() else REPO / gp, n) for n in names}
    for r in per:
        want = fin["runs"][r]["per_sequence_SA"]["ste"] if r in fin["runs"] else x1f["per_sequence_SA"]["ste"]
        if any(abs(per[r][n]["SA"] - want[n]) > 1e-6 for n in names):
            raise RuntimeError(f"{r}: per-frame recomputation differs from the pass's per-sequence SA")
    arms = {arm: [x2lib.run_name(arm, s) for s in x2lib.SEEDS] for arm in x2lib.ARMS}
    x1 = per[x2lib.X1_RUN]
    g_an = {n: g[n]["absence_neutral_SA"] for n in names}
    res = {"source_passes": [x2lib.FINAL_JSON.as_posix(), "experiments/aerotrack_v3/reports/x1_final_test.json"],
           "note": "descriptive; X1 = causal_g4 seed 0 (its own single TEST pass, reused, not re-scored)", "runs": {}, "arms": {}}
    for r in per:
        sm = summarize(per[r], names)
        res["runs"][r] = {"mSA": sm["mSA"], "absence_neutral_mSA": sm["absence_neutral_mSA"], "pct": sm["pct"],
                          "reappearance": sm["reappearance"], "size": {b: size_ci(per[r], names, b) for b, _, _ in OE.SIZE_BINS},
                          "jungle_18": seq_row(per[r], "jungle_18"), "large_target_sequences": large_by_seq(per[r], names),
                          "absence_neutral_vs_GlobalTrack_paired": AT.paired({n: per[r][n]["absence_neutral_SA"] for n in names}, g_an, names),
                          "tracker_ms_per_frame": (fin["runs"][r] if r in fin["runs"] else x1f)["tracker_ms_per_frame"]}
        if r != x2lib.X1_RUN:
            res["runs"][r]["SA_minus_X1_paired"] = AT.paired({n: per[r][n]["SA"] for n in names}, {n: x1[n]["SA"] for n in names}, names)
    res["GlobalTrack"] = {"size": {b: size_ci(g, names, b) for b, _, _ in OE.SIZE_BINS}, "jungle_18": seq_row(g, "jungle_18"),
                          "large_target_sequences": large_by_seq(g, names)}
    for arm, rs in arms.items():
        sa = {n: float(np.mean([per[r][n]["SA"] for r in rs])) for n in names}
        an = {n: float(np.mean([per[r][n]["absence_neutral_SA"] for r in rs])) for n in names}
        res["arms"][arm] = {"runs": rs, "seed_mean_mSA": float(np.mean(list(sa.values()))),
                            "seed_mean_absence_neutral_mSA": float(np.mean(list(an.values()))),
                            "absence_neutral_vs_GlobalTrack_paired": AT.paired(an, g_an, names),
                            "seed_mean_minus_X1_paired": AT.paired(sa, {n: x1[n]["SA"] for n in names}, names),
                            "large_SA_per_seed": {r: res["runs"][r]["size"]["large"]["SA"] for r in rs},
                            "jungle_18_SA_per_seed": {r: per[r]["jungle_18"]["SA"] for r in rs}}
    out = AT.REPORTS / "X2_failure_analysis.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for r, v in res["runs"].items():
        lg = v["size"]["large"]
        print(f"{r:30s} mSA {v['mSA']:.2f}  absence-neutral {v['absence_neutral_mSA']:.2f}  large {lg['SA']:.2f} "
              f"[{lg['ci95'][0]:.2f}, {lg['ci95'][1]:.2f}]  jungle_18 {v['jungle_18']['SA']:.2f}")
    print("GlobalTrack large", res["GlobalTrack"]["size"]["large"], "jungle_18", res["GlobalTrack"]["jungle_18"]["SA"])
    for arm, v in res["arms"].items():
        print(arm, json.dumps({k: v[k] for k in v if k != "runs"}, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--x1", action="store_true")
    grp.add_argument("--x2", action="store_true")
    a = ap.parse_args()
    x1() if a.x1 else x2()
