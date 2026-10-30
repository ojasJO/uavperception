#!/usr/bin/env python
"""Score single-target trackers on cached VAL detections with the SOT benchmarks' own metric, honestly tuned.

    venv\\Scripts\\python.exe experiments\\systems\\scripts\\track_eval.py --run gen_temporal_fp_g4_full_e20 [--corpus data_unified]

Protocol (single-object tracking, as in the Anti-UAV / CST benchmarks): the tracker gets the ground-truth box of the official
start frame and must output, for every later frame, a box or "absent". Score = state accuracy SA (Eq. 1 of the Anti-UAV
paper: IoU on visible frames, 1 for a correct "absent", 0 otherwise), averaged per sequence -> mSA, computed with Track A's
own implementation (anti_uav_repro.eval) on the OFFICIAL annotations (Anti-UAV infrared.json / visible.json, CST IR_label.json),
so Anti-UAV IR numbers are directly comparable with our faithful GlobalTrack (reports of experiments/anti_uav).

Trackers: `argmax` (best drone detection >= thr, else absent; the repair series' strongest baseline) and `ste` (ste.py).
Hyper-parameters: a small grid, CROSS-FITTED -- VAL sequences are split into two halves by recording group (Anti-UAV IR/RGB
pairs and sessions stay together); each half is scored with the parameters that maximise mSA on the OTHER half. Also reported:
untuned defaults, and parameters tuned on all of VAL (optimistic; what a final test pass would use).
Extra metrics on frames >= start: located rate (IoU >= 0.5 on visible frames), false-alarm rate (box on invisible frames),
frames to recover (first IoU >= 0.5) after each invisible -> visible transition.
Output: experiments/systems/reports/track_<run>_<corpus>.{json,md}. VAL only.
"""
import argparse
import hashlib
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (HERE, REPO / "experiments" / "anti_uav" / "src"):
    sys.path.insert(0, str(p))
import ste as T  # noqa: E402
from anti_uav_repro.data.antiuav import Sequence, load_sequences  # noqa: E402
from anti_uav_repro.eval.metrics import iou_xywh, sequence_metrics  # noqa: E402

GRID_ARGMAX = [{"thr": t} for t in (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)]
GRID_STE = [dict(det_thr=a, init_thr=b, reacq_thr=c, max_coast=d, output=e, coast_output=f, gate_chi2=g)
            for a, b, c, d, e, f, g in itertools.product((0.05, 0.15, 0.3), (0.3, 0.5), (0.5, 0.7, 0.9), (0, 5, 20, 60),
                                                         ("det", "kf"), (True, False), (T.CHI2_2D_99, 25.0))
            if not (d == 0 and not f)]                                   # max_coast 0 never coasts: coast_output is moot
DEFAULT_STE = T.STEParams().as_dict()
CST_FRAME_ATTRS = ("OC", "OV", "SV", "TC", "DBC", "CDB", "FM")          # CST frame-level attribute flags (arXiv:2507.23473 Table 2)
CST_SIZE_BINS = (("tiny", 0, 10), ("small", 10, 30), ("normal", 30, 50), ("large", 50, np.inf))  # box diagonal, px
D = {}                                                                   # worker globals: sequences + detections


def group_of(seq_key):
    """Cross-fitting group: Anti-UAV recording session (IR and RGB of a clip together), CST sequence."""
    if seq_key.startswith("antiuav_"):
        name = seq_key.split("_", 2)[2].rsplit("__", 1)[0]               # 20190925_101846_1_4
        return "antiuav:" + name.rsplit("_", 1)[0]                        # session 20190925_101846_1
    return seq_key


def half_of(seq_key):
    return int(hashlib.md5(group_of(seq_key).encode()).hexdigest(), 16) % 2


def load(run, corpus, allow_limited=False, split="val"):
    """split="test" is used only by experiments/aerotrack_v2/scripts/final_eval.py (the single pre-registered pass)."""
    cdir = REPO / "runs" / "systems" / "_cache" / run / corpus / split
    index = json.loads((cdir / "index.json").read_text(encoding="utf-8"))
    if index.get("limited") and not allow_limited:
        raise SystemExit(f"{cdir} is a development cache (--limit-seqs {index['limited']}), not the full VAL split; refusing to report "
                         "on it (pass allow_limited=True for development)")
    import yaml
    cfg = yaml.safe_load((REPO / "experiments" / "anti_uav" / "configs" / "_base_ir.yaml").read_text(encoding="utf-8"))
    official = {}
    for mod in ("IR", "RGB"):
        for s in load_sequences(cfg["data"]["root"], cfg["data"]["frames_root"], split, mod,
                                purpose="final_eval" if split == "test" else "tune"):
            official[f"antiuav_{split}_{s.name}__{mod}"] = s
    seqs = {}
    for k, info in index["sequences"].items():
        attrs = None
        if info["dataset"] == "anti_uav":
            s = official[k]
        else:
            lab = Path(info["first_frame_path"]).parent / "IR_label.json"
            if not lab.exists():
                lab = REPO / "data" / "cst_anti_uav_full" / "CST-AntiUAV" / "CST-AntiUAV" / split / Path(info["first_frame_path"]).parent.name / "IR_label.json"
            a = json.loads(lab.read_text(encoding="utf-8"))
            s = Sequence(name=k, split=split, modality="IR", label_path=lab, frames_dir=lab.parent,
                         exist=[int(e) for e in a["exist"]], gt_rect=a["gt"])
            attrs = {kk: np.asarray(a[kk], dtype=bool) for kk in CST_FRAME_ATTRS if kk in a and len(a[kk]) == s.n_frames}
            diag = np.array([np.hypot(g[2], g[3]) if (e and len(g) == 4) else np.nan for e, g in zip(a["exist"], a["gt"])])
            for name, lo, hi in CST_SIZE_BINS:                              # tiny (0,10], small (10,30], normal (30,50], large > 50 px
                attrs[f"size_{name}"] = (diag > lo) & (diag <= hi)
        z = np.load(cdir / f"{k}.npz")
        if len(z["frame_idx"]) != s.n_frames:
            raise SystemExit(f"{k}: cache has {len(z['frame_idx'])} frames, annotations {s.n_frames}")
        off = z["offsets"]
        dets = [(z["boxes"][off[i]:off[i + 1]], z["scores"][off[i]:off[i + 1]], z["classes"][off[i]:off[i + 1]]) for i in range(s.n_frames)]
        group = f"{info['dataset']}_{info['modality']}"
        seqs[k] = {"seq": s, "dets": dets, "group": group, "half": half_of(k), "attrs": attrs if info["dataset"] == "cst" else None}
    return index, seqs


def run_tracker(kind, params, item):
    """-> Track A prediction record for one sequence."""
    s, dets = item["seq"], item["dets"]
    start = s.start_frame
    init = s.box_xywh(start)
    res = [[] for _ in range(s.n_frames)]
    res[start] = [float(v) for v in init]
    if kind == "ste":
        trk = T.STE(T.STEParams(**{**DEFAULT_STE, **params}))
        trk.reset(np.array([init[0], init[1], init[0] + init[2], init[1] + init[3]], dtype=np.float64))
    for t in range(start + 1, s.n_frames):
        b, sc, cl = dets[t]
        box = T.argmax_step(b, sc, cl, params["thr"]) if kind == "argmax" else trk.step(b, sc, cl)[0]
        if box is not None:
            res[t] = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
    return {"sequence": s.name, "n_frames": s.n_frames, "start_frame": int(start), "res": res,
            "processed": [t >= start for t in range(s.n_frames)], "n_fallback": 0, "fps": None}


def extra(pred, s, attrs=None):
    start = pred["start_frame"]
    ex = np.asarray(s.exist[start:], dtype=bool)
    out_box = np.array([len(r) == 4 for r in pred["res"][start:]])
    iou = np.array([iou_xywh(pred["res"][start + i], s.gt_rect[start + i]) if ex[i] and out_box[i] else 0.0 for i in range(len(ex))])
    hit = iou >= 0.5
    rec = []
    for i in range(1, len(ex)):
        if ex[i] and not ex[i - 1]:
            j = np.flatnonzero(hit[i:])
            rec.append(int(j[0]) if len(j) else len(ex) - i)
    out = {"n_vis": int(ex.sum()), "n_inv": int((~ex).sum()), "located": int(hit.sum()), "false_alarm": int((out_box & ~ex).sum()),
           "recover": rec}
    if attrs:                                   # frame-level SA (Eq. 1 per frame) summed over the frames carrying each attribute
        sa = np.where(ex, iou, (~out_box).astype(float))
        out["attr"] = {k: [float(sa[v[start:]].sum()), int(v[start:].sum())] for k, v in attrs.items()}
    return out


def score_config(args):
    kind, params = args
    per = {}
    for k, it in D.items():
        pred = run_tracker(kind, params, it)
        mask = np.arange(it["seq"].n_frames) >= pred["start_frame"]
        per[k] = {"SA": sequence_metrics(pred["res"], it["seq"].exist, it["seq"].gt_rect, mask=mask)["SA"],
                  **extra(pred, it["seq"], it.get("attrs"))}
    return per


def init_worker(run, corpus):
    global D
    D = load(run, corpus)[1]


def summarize(per, keys, seqs):
    groups = sorted({seqs[k]["group"] for k in keys})
    out = {}
    for g in groups:
        ks = [k for k in keys if seqs[k]["group"] == g]
        vis, inv = sum(per[k]["n_vis"] for k in ks), sum(per[k]["n_inv"] for k in ks)
        rec = [r for k in ks for r in per[k]["recover"]]
        out[g] = {"n_sequences": len(ks), "mSA": float(np.mean([per[k]["SA"] for k in ks]) * 100),
                  "located_rate": sum(per[k]["located"] for k in ks) / max(vis, 1),
                  "false_alarm_rate": sum(per[k]["false_alarm"] for k in ks) / max(inv, 1) if inv else None,
                  "n_invisible_frames": inv, "reappearances": len(rec),
                  "recover_frames_median": float(np.median(rec)) if rec else None, "recover_frames_mean": float(np.mean(rec)) if rec else None}
        att = {}
        for k in ks:
            for a_, (sm, n) in per[k].get("attr", {}).items():
                t_ = att.setdefault(a_, [0.0, 0])
                t_[0] += sm
                t_[1] += n
        if att:
            out[g]["frame_attr_SA"] = {a_: {"SA": 100 * sm / n if n else None, "n_frames": n} for a_, (sm, n) in sorted(att.items())}
    return out


def objective(per, keys, seqs):
    """Mean of the group mSAs (each dataset/modality weighs the same)."""
    s = summarize(per, keys, seqs)
    return float(np.mean([v["mSA"] for v in s.values()]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")            # Windows console default (cp1252) cannot print the report
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", default="data_unified")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.corpus == "auto":                              # the newest cache of this run: v2 if it exists, else v1
        base = REPO / "runs" / "systems" / "_cache" / a.run
        a.corpus = "data_unified_v2" if (base / "data_unified_v2" / "val" / "index.json").exists() else "data_unified"
    t0 = time.time()
    index, seqs = load(a.run, a.corpus)
    jobs = [("argmax", p) for p in GRID_ARGMAX] + [("ste", p) for p in GRID_STE] + [("ste", {})]
    with Pool(a.workers, initializer=init_worker, initargs=(a.run, a.corpus)) as pool:
        results = pool.map(score_config, jobs, chunksize=4)
    allk = sorted(seqs)
    halves = {h: [k for k in allk if seqs[k]["half"] == h] for h in (0, 1)}
    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": a.run, "corpus": a.corpus, "cache_set_sha256": index["set_sha256"],
              "window": index["window"], "n_sequences": len(allk), "halves": {h: len(v) for h, v in halves.items()},
              "grid_sizes": {"argmax": len(GRID_ARGMAX), "ste": len(GRID_STE)}, "trackers": {}}
    for kind in ("argmax", "ste"):
        idx = [i for i, (k, p) in enumerate(jobs) if k == kind and p]
        cross = {}
        chosen = {}
        for h in (0, 1):
            tune, test = halves[1 - h], halves[h]
            best = max(idx, key=lambda i: objective(results[i], tune, seqs)) if tune else idx[0]
            chosen[h] = jobs[best][1]
            for k in test:
                cross[k] = results[best][k]
        best_all = max(idx, key=lambda i: objective(results[i], allk, seqs))
        rec = {"cross_fitted": summarize(cross, allk, seqs), "params_per_half": chosen,
               "tuned_on_all_val": {"params": jobs[best_all][1], "summary": summarize(results[best_all], allk, seqs)}}
        if kind == "ste":
            dflt = next(i for i, (k, p) in enumerate(jobs) if k == "ste" and not p)
            rec["defaults_untuned"] = {"params": DEFAULT_STE, "summary": summarize(results[dflt], allk, seqs)}
        report["trackers"][kind] = rec
    # GlobalTrack (Track A, normal_ir seed 0) on the same Anti-UAV IR VAL sequences, per sequence, for a paired comparison
    gt_path = REPO / "experiments" / "anti_uav" / "runs" / "normal_ir" / "eval_val_IR_last" / "metrics.json"
    g = json.loads(gt_path.read_text(encoding="utf-8"))["per_sequence"]
    ir = [k for k in allk if seqs[k]["group"] == "anti_uav_IR"]
    names = {k: k.split("_", 2)[2].rsplit("__", 1)[0] for k in ir}
    if ir and all(names[k] in g for k in ir):
        glob = np.array([g[names[k]]["SA"] for k in ir]) * 100
        report["globaltrack_same_sequences"] = {"source": str(gt_path.relative_to(REPO)), "n_sequences": len(ir), "mSA": float(glob.mean())}
        for kind in ("argmax", "ste"):
            # cross-fitted per-sequence SA of this tracker, paired with GlobalTrack on the same sequences
            cf = {}
            for h in (0, 1):
                p = report["trackers"][kind]["params_per_half"][h]
                i = next(i for i, (kk, pp) in enumerate(jobs) if kk == kind and pp == p)
                cf.update({k: results[i][k]["SA"] * 100 for k in halves[h]})
            d = np.array([cf[k] for k in ir]) - glob
            rng = np.random.default_rng(0)
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            report["trackers"][kind]["vs_globaltrack_anti_uav_IR"] = {"mean_delta_mSA": float(d.mean()),
                                                                     "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                                                     "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    report["seconds"] = round(time.time() - t0, 1)
    out = REPO / "experiments" / "systems" / "reports" / f"track_{a.run}_{a.corpus}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    md = [f"# Single-target tracking on VAL — {a.run} ({a.corpus})", "",
          f"*Generated {report['generated']} by `scripts/track_eval.py` from cached detections (set `{index['set_sha256'][:12]}`, window "
          f"`{index['window']}`). SOT protocol: GT box at the official start frame, then box or 'absent' per frame; mSA = paper Eq. (1) "
          f"computed by Track A's code on the official annotations. Hyper-parameters cross-fitted over two halves of VAL (by recording "
          f"group).*", "", "| tracker | setting | group | n seq | mSA | located (IoU ≥ .5) | false alarms on invisible frames | "
          "recover median / mean (frames) |", "|---|---|---|---:|---:|---:|---:|---|"]
    for kind, rec in report["trackers"].items():
        for setting, summ in (("cross-fitted", rec["cross_fitted"]), ("tuned on all VAL", rec["tuned_on_all_val"]["summary"]),
                              ("defaults (untuned)", rec.get("defaults_untuned", {}).get("summary", {}))):
            for gname, v in summ.items():
                fa = "—" if v["false_alarm_rate"] is None else f"{100 * v['false_alarm_rate']:.1f} %"
                rc = "—" if v["recover_frames_median"] is None else f"{v['recover_frames_median']:.0f} / {v['recover_frames_mean']:.1f}"
                md.append(f"| {kind} | {setting} | {gname} | {v['n_sequences']} | {v['mSA']:.2f} | {100 * v['located_rate']:.1f} % | {fa} | {rc} |")
    if "globaltrack_same_sequences" in report:
        gg = report["globaltrack_same_sequences"]
        md += ["", f"GlobalTrack (Track A faithful reproduction, normal_ir seed 0) on the same {gg['n_sequences']} Anti-UAV IR VAL sequences: "
               f"mSA {gg['mSA']:.2f}."]
        for kind in ("argmax", "ste"):
            v = report["trackers"][kind].get("vs_globaltrack_anti_uav_IR")
            if v:
                md.append(f"- {kind} (cross-fitted) − GlobalTrack: {v['mean_delta_mSA']:+.2f} mSA, sequence-bootstrap 95 % CI "
                          f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}], wins {v['wins']} / losses {v['losses']}")
    md += ["", "Selected parameters (per half, then on all VAL):", ""]
    for kind, rec in report["trackers"].items():
        md.append(f"- {kind}: half0 ← {rec['params_per_half'][0]}; half1 ← {rec['params_per_half'][1]}; all ← {rec['tuned_on_all_val']['params']}")
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
