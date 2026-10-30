"""C1 VAL pass (called by tc_c1.py --stage val after the registration check): every forecaster once, the registered rules, a report.

Reads the TRAIN tuning (results/runs/aerotrack_temporal/c1/classical_tuning.json) and the learned checkpoints (learned_registry.json, sha256
verified); nothing here is tuned. Output: experiments/aerotrack_temporal/reports/C1_trajectory_forecasting.{json,md}.
"""
import hashlib
import json
import time

import numpy as np

import tc_c1 as C
import tc_data as D
import tc_forecast as F

JUMP_PX_AT_640 = 30.0          # sudden-jump subset: a frame-to-frame GT step > 30 px at 640 px width (scaled with the width)


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jump_mask(S):
    pts = np.concatenate([S["hist_c"][:, -1:, :], S["fut_c"]], 1)
    vis = np.concatenate([np.ones((len(pts), 1), bool), S["fut_m"]], 1)
    step = np.linalg.norm(np.diff(pts, axis=1), axis=-1)
    ok = vis[:, 1:] & vis[:, :-1]
    big = np.where(ok, step, 0).max(1)
    return big > JUMP_PX_AT_640 * S["W"] / 640.0


def chosen_arch(book):
    means = {}
    for a in C.ARCHS:
        v = [book[f"c1_{a}_abs_s{s}"]["dev_nll"] for s in C.SEEDS if f"c1_{a}_abs_s{s}" in book]
        if len(v) == len(C.SEEDS):
            means[a] = float(np.mean(v))
    return min(means, key=means.get), means


def verdict_h1(boots):
    if all(b["rel"] <= -0.05 and b["ci95"][1] < 0 for b in boots):
        return "SUPPORTED"
    if all(b["ci95"][0] > 0 for b in boots):
        return "LOSES"
    return "NOT_SUPPORTED"


def verdict_h2(boots, cal_l, cal_k):
    if all(b["ci95"][1] < 0 for b in boots) and all(c < cal_k for c in cal_l):
        return "SUPPORTED"
    if all(b["ci95"][0] > 0 for b in boots):
        return "LOSES"
    return "NOT_SUPPORTED"


def verdict_h3(boots):
    if all(b["rel"] <= -0.02 and b["ci95"][1] < 0 for b in boots):
        return "SUPPORTED"
    if all(b["ci95"][0] > 0 for b in boots):
        return "REVERSED"
    return "NOT_SUPPORTED"


def run():
    import tc_learned as M
    tune = json.loads((C.RUN / "classical_tuning.json").read_text(encoding="utf-8"))["domains"]
    book = json.loads((C.RUN / "learned_registry.json").read_text(encoding="utf-8"))
    arch, dev_means = chosen_arch(book)
    models = {}
    for variant in ("abs", "noabs", "aug"):
        for seed in C.SEEDS:
            name = f"c1_{arch}_{variant}_s{seed}"
            if name not in book:
                raise SystemExit(f"{name} missing: the registered arms must all be trained before the VAL pass")
            path = C.RUN / f"{name}.pt"
            if _sha(path) != book[name]["sha256"]:
                raise SystemExit(f"{path} does not match its recorded sha256")
            models[name] = M.load(path)
    val = D.load_split("val")
    out = {"generated": C.now(), "experiment": "C1_trajectory_forecasting", "condition": "C1a oracle histories", "split": "val",
           "chosen_architecture": arch, "dev_nll_seed_mean": dev_means, "domains": {}}
    for dom in D.DOMAINS:
        seqs = [s for s in val if s["domain"] == dom]
        S = F.make_samples(seqs)
        t = tune[dom]
        preds = {nm: (F.forecast_classical(S, nm), None) for nm in C.CLASSICAL}
        for nm in ("dcv", "ou", "oudcv"):
            preds[nm] = (F.forecast_classical(S, nm, t[nm]["chosen"]), None)
        kc = dict(t["kalman"]["chosen"])
        preds["kalman"] = F.kalman_forecast(seqs, S, F.KalmanBank(**kc))
        for name, (model, meta) in models.items():
            preds[name] = M.predict(model, S, abs_pos=meta["abs_pos"])
        jm = jump_mask(S)
        summ, sums = {}, {}
        for nm, (mean, var) in preds.items():
            err = F.point_errors(S, mean)
            d2, nll = F.gauss_terms(S, mean, var) if var is not None else (None, None)
            summ[nm] = {"all": F.summarize(S, err, d2, nll), "sudden_jump": F.summarize(S, err, d2, nll, jm),
                        "smooth": F.summarize(S, err, d2, nll, ~jm)}
            sums[nm] = F.seq_sums(S, err, nll)
        sd = {int(q): dom for q in np.unique(S["seq"])}
        ref = t["reference"]
        main = [f"c1_{arch}_abs_s{s}" for s in C.SEEDS]
        h1 = [F.paired_bootstrap(sums[m], sums[ref], sd) for m in main]
        h2 = [F.paired_bootstrap(sums[m], sums["kalman"], sd, stat="nll") for m in main]
        h3 = [F.paired_bootstrap(sums[f"c1_{arch}_abs_s{s}"], sums[f"c1_{arch}_noabs_s{s}"], sd) for s in C.SEEDS]
        cal_l = [summ[m]["all"]["calibration_error"] for m in main]
        cal_k = summ["kalman"]["all"]["calibration_error"]
        out["domains"][dom] = {
            "n_sequences": len(seqs), "n_samples": int(len(S["t"])), "n_jump_samples": int(jm.sum()), "reference": ref,
            "summaries": summ,
            "H1_point": {"pairs": h1, "verdict": verdict_h1(h1)},
            "H2_uncertainty": {"pairs": h2, "calibration_error_learned": cal_l, "calibration_error_kalman": cal_k,
                               "verdict": verdict_h2(h2, cal_l, cal_k)},
            "H3_absolute_position": {"pairs": h3, "verdict": verdict_h3(h3)},
            "H4_augmentation_descriptive": {s: F.paired_bootstrap(sums[f"c1_{arch}_aug_s{s}"], sums[f"c1_{arch}_abs_s{s}"], sd)
                                            for s in C.SEEDS}}
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    path = C.REPORTS / "C1_trajectory_forecasting.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    path.with_suffix(".md").write_text(render(out), encoding="utf-8")
    print(render(out))


def render(out):
    arch = out["chosen_architecture"]
    f = lambda v, d=1: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    md = ["# C1 — trajectory forecasting from causal histories (Track C, oracle histories, VAL)", "",
          f"*Generated {out['generated']} by `scripts/tc_c1.py --stage val` under `preregistration/C1_trajectory_forecasting.json`. "
          f"Learned family chosen on the dev split (seed-mean dev NLL): **{arch}** "
          f"({', '.join(f'{k} {v:.3f}' for k, v in out['dev_nll_seed_mean'].items())}). Pixels of each sequence's native frame; "
          "ADE over h = 1..24 on visible future frames.*", ""]
    for dom, d in out["domains"].items():
        md += [f"## {dom} ({d['n_sequences']} sequences, {d['n_samples']:,} anchors; classical reference chosen on TRAIN: `{d['reference']}`)", "",
               f"| verdict | H1 point (learned vs `{d['reference']}`) | H2 uncertainty (vs Kalman) | H3 absolute position |", "|---|---|---|---|",
               f"| | **{d['H1_point']['verdict']}** | **{d['H2_uncertainty']['verdict']}** | **{d['H3_absolute_position']['verdict']}** |", "",
               "| forecaster | ADE px | FDE@24 px | ADE / size | miss > 2x size | NLL | 95 % coverage | calib. error | ADE on jumps |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for nm, s in d["summaries"].items():
            a, j = s["all"], s["sudden_jump"]
            md.append(f"| {nm} | {f(a['ade_px'])} | {f(a['fde_px'])} | {f(a['ade_rel_size'], 2)} | {f(a['miss_2x_size'], 3)} | "
                      f"{f(a.get('nll'), 2)} | {f(a.get('coverage', {}).get('0.95'), 3)} | {f(a.get('calibration_error'), 3)} | "
                      f"{f(j['ade_px'] if j else None)} |")
        md += ["", "| pair | Δ ADE px [95 % CI] | relative |", "|---|---|---:|"]
        for lbl, key in (("H1", "H1_point"), ("H3", "H3_absolute_position")):
            for s, b in zip(("s0", "s1", "s2"), d[key]["pairs"]):
                md.append(f"| {lbl} {s} | {b['diff']:+.2f} [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}] | {100 * b['rel']:+.1f} % |")
        for s, b in zip(("s0", "s1", "s2"), d["H2_uncertainty"]["pairs"]):
            md.append(f"| H2 {s} (Δ NLL) | {b['diff']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | |")
        md.append("")
    return "\n".join(md) + "\n"
