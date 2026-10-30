#!/usr/bin/env python
"""P1 (EXPLORATORY, VAL only): presence / output gate decoupled from STE association -- preregistration/P1_presence_gate_VAL.json.

    venv\\Scripts\\python.exe experiments\\cst_anti_uav\\scripts\\explore_p1.py

For each run and each CST VAL sequence the frozen STE runs ONCE per half-setting (track_eval's cross-fitted params_per_half); the
gate (variant x tau_out) is applied to that per-frame stream (mode, associated score, box). Cross-fitting as registered: the
pipeline for half h = STE params_per_half[h] + the gate setting that maximises mSA on half 1-h under the same STE params.
SA by Track A's code (track_eval.sequence_metrics). Output: reports/explore/P1_presence_gate_VAL.{json,md}. Never reads TEST.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import analyze_test as AT  # noqa: E402  (paired(): bootstrap B = 10000, default_rng(0), Wilcoxon)
import explore_presence as EP  # noqa: E402

CORPUS = "data_unified_v2"
RUNS = ["v3_n__causal_g4_cst_s0_e20", "v3_n__causal_g4_cst_s1_e20", "v3_n__causal_g4_cst_s2_e20"]
PRE = REPO / "experiments" / "cst_anti_uav" / "preregistration" / "P1_presence_gate_VAL.json"
GATES = [(v, t) for v in ("instant", "ema") for t in json.loads(PRE.read_text(encoding="utf-8"))["candidate"]["grid_tau_out"]]


def gated(stream, start, n, variant, tau):
    """stream: [(mode, score, box_xywh|None)] from the start frame -> Track A result list ([] = 'absent')."""
    res = [[] for _ in range(n)]
    g = 1.0
    for i, (mode, sc, box) in enumerate(stream):
        g = sc if variant == "instant" else (1.0 if mode == "init" else 0.5 * sc + 0.5 * g)
        if box is not None and (mode == "init" or tau <= 0 or g >= tau):
            res[start + i] = box
    return res


def counts(res, it):
    s = it["seq"]
    st = s.start_frame
    ex = np.asarray(s.exist[st:], bool)
    has = np.array([len(r) == 4 for r in res[st:]])
    return {"absent": int((~ex).sum()), "false_alarm": int((~ex & has).sum()), "visible": int(ex.sum()), "visible_no_box": int((ex & ~has).sum())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    TE = EP._sys_module("track_eval")
    T = EP._sys_module("ste")
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    out = {"registration": str(PRE.relative_to(REPO).as_posix()), "note": "EXPLORATORY, CST VAL only; no TEST access", "runs": {}}
    for run in RUNS:
        rep = json.loads((REPO / "experiments" / "systems" / "reports" / f"track_{run}_{CORPUS}.json").read_text(encoding="utf-8"))
        pph = {int(h): p for h, p in rep["trackers"]["ste"]["params_per_half"].items()}
        _, seqs = TE.load(run, CORPUS, split="val")
        keys = sorted(k for k, it in seqs.items() if it["group"] == "cst_IR")
        half = {k: seqs[k]["half"] for k in keys}
        sa = {}                                           # (ste_half, gate) -> {seq: SA}
        cnt = {}
        for h in (0, 1):
            for k in keys:
                it = seqs[k]
                st, stream = EP.run_frames(T, pph[h], it)
                n = it["seq"].n_frames
                mask = np.arange(n) >= st
                for gt_ in GATES:
                    res = gated(stream, st, n, *gt_)
                    sa.setdefault((h, gt_), {})[k] = 100 * TE.sequence_metrics(res, it["seq"].exist, it["seq"].gt_rect, mask=mask)["SA"]
                    if half[k] == h:
                        cnt.setdefault(gt_, {})[k] = counts(res, it)
        base = {k: sa[(half[k], ("instant", 0.0))][k] for k in keys}          # = the frozen STE, cross-fitted
        want = rep["trackers"]["ste"]["cross_fitted"]["cst_IR"]["mSA"]
        if abs(np.mean(list(base.values())) - want) > 1e-6:
            raise RuntimeError(f"{run}: baseline {np.mean(list(base.values()))} != track_eval cross-fitted {want}")
        chosen, cand, cgate = {}, {}, {}
        for h in (0, 1):
            tune = [k for k in keys if half[k] == 1 - h]
            best = max(GATES, key=lambda g_: (np.mean([sa[(h, g_)][k] for k in tune]), -GATES.index(g_)))
            chosen[h] = {"variant": best[0], "tau_out": best[1]}
            for k in keys:
                if half[k] == h:
                    cand[k], cgate[k] = sa[(h, best)][k], best
        pr = AT.paired(cand, base, keys)

        def rates(sel):
            c = [cnt[sel(k)][k] for k in keys]
            return {"false_alarm_rate": sum(x["false_alarm"] for x in c) / max(sum(x["absent"] for x in c), 1),
                    "visible_no_box_rate": sum(x["visible_no_box"] for x in c) / max(sum(x["visible"] for x in c), 1)}
        curve = {f"{v}_{t}": {"mSA_cross_fitted_ste": float(np.mean([sa[(half[k], (v, t))][k] for k in keys])), **rates(lambda k, g_=(v, t): g_)}
                 for v, t in GATES}                                            # descriptive: every gate under the cross-fitted STE
        out["runs"][run] = {"curve_descriptive": curve,
                            "baseline_cross_fitted_mSA": float(np.mean(list(base.values()))), "gated_cross_fitted_mSA": float(np.mean(list(cand.values()))),
                            "gated_minus_frozen_paired": pr, "chosen_gate_per_half": chosen,
                            "trade_off": {"frozen": rates(lambda k: ("instant", 0.0)), "gated": rates(lambda k: cgate[k])}}
    ds = [v["gated_minus_frozen_paired"] for v in out["runs"].values()]
    pos = sum(d["ci95"][0] > 0 for d in ds)
    neg = sum(d["ci95"][1] < 0 for d in ds)
    mean = float(np.mean([d["mean_delta"] for d in ds]))
    out["verdict"] = "PROMISING" if (pos >= 2 and mean > 0.3) else "HARMFUL" if neg >= 2 else "NO_GAIN"
    out["three_run_mean_delta"] = mean
    rd = REPO / "experiments" / "cst_anti_uav" / "reports" / "explore"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "P1_presence_gate_VAL.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    L = ["# P1 — presence gate decoupled from STE association (EXPLORATORY, CST VAL only)", "",
         f"*Registration `{out['registration']}` (committed before any P1 computation). Cross-fitted over track_eval's VAL halves; "
         "sequence bootstrap B = 10000. No TEST access; VAL shares source videos with TRAIN.*", "",
         "| run | frozen STE (cross-fitted) | gated | Δ [95 % CI] | wins / losses | gate chosen (half 0 / half 1) | false-alarm rate frozen → gated | visible miss rate frozen → gated |",
         "|---|---:|---:|---|---|---|---|---|"]
    for run, v in out["runs"].items():
        d, c, tr = v["gated_minus_frozen_paired"], v["chosen_gate_per_half"], v["trade_off"]
        L.append(f"| {run} | {v['baseline_cross_fitted_mSA']:.2f} | {v['gated_cross_fitted_mSA']:.2f} | {d['mean_delta']:+.2f} "
                 f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}] | {d['wins']} / {d['losses']} | "
                 f"{c[0]['variant']} {c[0]['tau_out']} / {c[1]['variant']} {c[1]['tau_out']} | "
                 f"{100 * tr['frozen']['false_alarm_rate']:.1f} → {100 * tr['gated']['false_alarm_rate']:.1f} % | "
                 f"{100 * tr['frozen']['visible_no_box_rate']:.1f} → {100 * tr['gated']['visible_no_box_rate']:.1f} % |")
    L += ["", f"**Verdict (registered rule): {out['verdict']}** (3-run mean Δ {mean:+.2f}; CI > 0 for {pos} / 3 runs, < 0 for {neg} / 3)."]
    (rd / "P1_presence_gate_VAL.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
