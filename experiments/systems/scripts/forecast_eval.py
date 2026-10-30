#!/usr/bin/env python
"""Short-horizon forecasting from the tracker's own state, against trivial and classical baselines, with uncertainty calibration.

    venv\\Scripts\\python.exe experiments\\systems\\scripts\\forecast_eval.py --run gen_temporal_fp_g4_full_e20 [--corpus auto]

At every frame where the STE has a fresh measurement (modes init / track / reacquired), the centre is forecast for h = 1..12
frames and compared with the ground-truth centre at t + h (frames where the target is visible). All forecasters see only the
tracker's OWN past outputs (no ground truth): this is the system-level question "given what the stack knows now, where will the
target be", not the oracle-trajectory benchmark of the repair series.

    kalman     the STE's constant-velocity Kalman forecast (mean) and its centre covariance
    zero_vel   the current output centre
    last_vel   current + h x (current - previous output centre)
    mean_vel4  current + h x mean displacement over the last 4 outputs
Metrics per dataset group and horizon: mean / median L2 error (px), error relative to the target size, miss rates (error >
2 x target size; error > 20 px), FDE (h = 12); for the Kalman forecast, coverage of its 68 % and 95 % ellipses (calibrated
uncertainty would give 0.68 / 0.95), split into all windows and 'sudden-jump' windows (GT displacement > 30 px in the horizon).
STE parameters: the cross-fitted choices of the matching track_eval report (each half uses the parameters tuned on the other
half), else ste.STEParams defaults. Output: experiments/systems/reports/forecast_<run>_<corpus>.{json,md}. VAL only.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import ste as T  # noqa: E402
import track_eval as TE  # noqa: E402

H = 12
CHI2_68, CHI2_95 = 2.2789, 5.9915


def centre(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def run_sequence(item, params):
    s, dets = item["seq"], item["dets"]
    start = s.start_frame
    init = s.box_xywh(start)
    trk = T.STE(T.STEParams(**{**T.STEParams().as_dict(), **params}))
    trk.reset(np.array([init[0], init[1], init[0] + init[2], init[1] + init[3]], dtype=np.float64))
    gt_c = [None] * s.n_frames
    gt_s = [None] * s.n_frames
    for t in range(s.n_frames):
        b = s.box_xywh(t)
        if b is not None:
            gt_c[t] = np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
            gt_s[t] = max(float(b[2]), float(b[3]), 1.0)
    hist = [centre([init[0], init[1], init[0] + init[2], init[1] + init[3]])]
    rows = []
    for t in range(start + 1, s.n_frames):
        b, sc, cl = dets[t]
        box, mode, _ = trk.step(b, sc, cl)
        if box is None:
            hist = []
            continue
        c = centre(box)
        hist.append(c)
        if mode not in ("init", "track", "reacquired") or trk.kf is None:
            continue
        km, kc = trk.kf.forecast(H)
        lv = hist[-1] - hist[-2] if len(hist) >= 2 else np.zeros(2)
        mv = (hist[-1] - hist[-5]) / 4 if len(hist) >= 5 else lv
        fut = [(h, t + h) for h in range(1, H + 1) if t + h < s.n_frames and gt_c[t + h] is not None]
        if not fut:
            continue
        g = np.array([gt_c[u] for _, u in fut])
        base = gt_c[t] if gt_c[t] is not None else g[0]           # ground truth only: no tracker error in the flag
        jump = bool(np.max(np.linalg.norm(np.diff(np.vstack([base, g]), axis=0), axis=1)) > 30)
        for (h, u), gc in zip(fut, g):
            e = gc - km[h - 1]
            d2 = float(e @ np.linalg.solve(kc[h - 1], e))
            rows.append((h, gt_s[u], jump, np.linalg.norm(e), np.linalg.norm(gc - c), np.linalg.norm(gc - (c + h * lv)),
                         np.linalg.norm(gc - (c + h * mv)), d2 <= CHI2_68, d2 <= CHI2_95))
    return rows


def summarize(rows):
    if not rows:
        return None
    a = np.array([r[:7] for r in rows], dtype=np.float64)
    cov68 = np.array([r[7] for r in rows])
    cov95 = np.array([r[8] for r in rows])
    out = {"n_points": int(len(a)), "by_horizon": {}}
    names = {3: "kalman", 4: "zero_vel", 5: "last_vel", 6: "mean_vel4"}
    for h in range(1, H + 1):
        m = a[:, 0] == h
        if not m.any():
            continue
        size = a[m, 1]
        rec = {"n": int(m.sum())}
        for col, nm in names.items():
            err = a[m, col]
            rec[nm] = {"mean_px": float(err.mean()), "median_px": float(np.median(err)), "rel_size_mean": float((err / size).mean()),
                       "miss_2x_size": float((err > 2 * size).mean()), "miss_20px": float((err > 20).mean())}
        rec["kalman_coverage68"] = float(cov68[m].mean())
        rec["kalman_coverage95"] = float(cov95[m].mean())
        out["by_horizon"][h] = rec
    out["ade_px"] = {nm: float(a[:, col].mean()) for col, nm in names.items()}
    out["fde_px_h12"] = {nm: out["by_horizon"].get(H, {}).get(nm, {}).get("mean_px") for nm in names.values()}
    out["kalman_coverage95_all"] = float(cov95.mean())
    out["kalman_coverage68_all"] = float(cov68.mean())
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", default="auto")
    a = ap.parse_args()
    if a.corpus == "auto":
        base = REPO / "results/runs" / "systems" / "_cache" / a.run
        a.corpus = "data_unified_v2" if (base / "data_unified_v2" / "val" / "index.json").exists() else "data_unified"
    t0 = time.time()
    index, seqs = TE.load(a.run, a.corpus)
    rep = REPO / "experiments" / "systems" / "reports" / f"track_{a.run}_{a.corpus}.json"
    halves = json.loads(rep.read_text(encoding="utf-8"))["trackers"]["ste"]["params_per_half"] if rep.exists() else None
    groups = defaultdict(list)
    for k, it in seqs.items():
        p = halves[str(it["half"])] if halves and str(it["half"]) in halves else (halves[it["half"]] if halves else {})
        rows = run_sequence(it, p)
        groups[it["group"]] += rows
        groups["all"] += rows
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": a.run, "corpus": a.corpus, "horizon": H,
           "ste_params": "cross-fitted per half from " + rep.name if halves else "STEParams defaults",
           "groups": {g: {"all": summarize(r), "sudden_jump": summarize([x for x in r if x[2]]), "smooth": summarize([x for x in r if not x[2]])}
                      for g, r in groups.items()}, "seconds": round(time.time() - t0, 1)}
    path = REPO / "experiments" / "systems" / "reports" / f"forecast_{a.run}_{a.corpus}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = [f"# Forecasting from the tracker's own state — {a.run} ({a.corpus})", "",
          f"*Generated {out['generated']} by `scripts/forecast_eval.py`; STE parameters: {out['ste_params']}. Errors in pixels at the "
          f"original resolution; every forecaster sees only the tracker's past outputs.*", "",
          "| group | windows | points | ADE kalman / zero / last-vel / mean-vel4 (px) | FDE@12 kalman / zero / last / mean4 | Kalman 95 % ellipse coverage (all / jumps) |",
          "|---|---|---:|---|---|---|"]
    for g, d in out["groups"].items():
        for part in ("all", "sudden_jump", "smooth"):
            s = d[part]
            if not s:
                continue
            ade, fde = s["ade_px"], s["fde_px_h12"]
            f = lambda v: "—" if v is None else f"{v:.1f}"      # noqa: E731
            cov = f"{s['kalman_coverage95_all']:.2f}"
            md.append(f"| {g} | {part} | {s['n_points']:,} | {f(ade['kalman'])} / {f(ade['zero_vel'])} / {f(ade['last_vel'])} / "
                      f"{f(ade['mean_vel4'])} | {f(fde['kalman'])} / {f(fde['zero_vel'])} / {f(fde['last_vel'])} / {f(fde['mean_vel4'])} | {cov} |")
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
