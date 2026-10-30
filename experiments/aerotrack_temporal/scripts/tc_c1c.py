#!/usr/bin/env python
"""Track C, C1c: camera-mode-aware learned forecasters (preregistration/C1c_domain_aware_forecasting.json).

    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_c1c.py --stage train     TRAIN / dev only
    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_c1c.py --stage val       one VAL pass (registration and
                                                                                               learned_registry_c1c.json committed)
Arms: 'domtok' (pooled + 3-way camera-mode token) and 'perdom' (one model per domain), C1's transformer family and recipe, seeds
0 / 1 / 2. Per domain the arm with the lower seed-mean dev NLL (pixel-space Gaussian NLL on that domain's dev groups) is the C1c
model. Outputs: runs/aerotrack_temporal/c1/c1c_*.pt + learned_registry_c1c.json; reports/C1c_domain_aware_forecasting.{json,md}.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tc_c1 as C  # noqa: E402
import tc_c1_val as V  # noqa: E402
import tc_data as D  # noqa: E402
import tc_forecast as F  # noqa: E402

PRE = C.REPO / "experiments" / "aerotrack_temporal" / "preregistration" / "C1c_domain_aware_forecasting.json"
REG = C.RUN / "learned_registry_c1c.json"
ARCH = "tfm"


def committed_clean(p):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=C.REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(p)], cwd=C.REPO).returncode == 0
    return tracked and clean


def px_nll(model, S, dom_tok):
    import tc_learned as M
    mean, var = M.predict(model, S, abs_pos=True, dom_tok=dom_tok)
    return float(np.nanmean(F.gauss_terms(S, mean, var)[1])), float(np.nanmean(F.point_errors(S, mean)))


def stage_train(seeds):
    import tc_learned as M
    tr, dv = C.split_train()
    book = json.loads(REG.read_text(encoding="utf-8")) if REG.exists() else {}
    dev_by_dom = {d: F.make_samples([s for s in dv if s["domain"] == d]) for d in D.DOMAINS}

    def record(name, model, meta, hist):
        meta.update(best_epoch=int(np.argmin([h["dev_nll"] for h in hist]) + 1), L=F.L_HIST, H=F.H_FUT, epochs=12, arch=ARCH, abs_pos=True,
                    aug=False, trained=C.now())
        meta["dev_px"] = {d: dict(zip(("nll", "ade_px"), px_nll(model, dev_by_dom[d], meta["dom_tok"]))) for d in meta["domains"]}
        meta["sha256"] = M.save(model, C.RUN / f"{name}.pt", meta)
        book[name] = meta
        REG.write_text(json.dumps(book, indent=1), encoding="utf-8")
        print(name, json.dumps(meta["dev_px"]), flush=True)

    S_tr, S_dv = F.make_samples(tr), F.make_samples(dv)
    for seed in seeds:
        name = f"c1c_{ARCH}_domtok_s{seed}"
        if name not in book:
            model, hist = M.train(ARCH, S_tr, S_dv, seed, abs_pos=True, dom_tok=True)
            record(name, model, {"variant": "domtok", "seed": seed, "dom_tok": True, "domains": list(D.DOMAINS)}, hist)
    del S_tr, S_dv
    for dom in D.DOMAINS:
        S_tr_d = F.make_samples([s for s in tr if s["domain"] == dom])
        for seed in seeds:
            name = f"c1c_{ARCH}_perdom_{dom}_s{seed}"
            if name not in book:
                model, hist = M.train(ARCH, S_tr_d, dev_by_dom[dom], seed, abs_pos=True)
                record(name, model, {"variant": "perdom", "seed": seed, "dom_tok": False, "domains": [dom]}, hist)
    choice = {}
    for dom in D.DOMAINS:
        dt = np.mean([book[f"c1c_{ARCH}_domtok_s{s}"]["dev_px"][dom]["nll"] for s in seeds])
        pdm = np.mean([book[f"c1c_{ARCH}_perdom_{dom}_s{s}"]["dev_px"][dom]["nll"] for s in seeds])
        choice[dom] = {"arm": "domtok" if dt <= pdm else "perdom", "dev_nll_domtok": float(dt), "dev_nll_perdom": float(pdm)}
    book["_choice"] = choice
    REG.write_text(json.dumps(book, indent=1), encoding="utf-8")
    print("choice", json.dumps(choice))


def stage_val():
    import tc_learned as M
    if not committed_clean(PRE) or not committed_clean(REG):
        raise SystemExit("C1c registration and learned_registry_c1c.json must be committed and unmodified before VAL")
    book = json.loads(REG.read_text(encoding="utf-8"))
    c1book = json.loads((C.RUN / "learned_registry.json").read_text(encoding="utf-8"))
    tune = json.loads((C.RUN / "classical_tuning.json").read_text(encoding="utf-8"))["domains"]

    def load(name, registry):
        p = C.RUN / f"{name}.pt"
        if hashlib.sha256(p.read_bytes()).hexdigest() != registry[name]["sha256"]:
            raise SystemExit(f"{p} does not match its recorded sha256")
        return M.load(p)
    val = D.load_split("val")
    out = {"generated": C.now(), "experiment": "C1c_domain_aware_forecasting", "choice": book["_choice"], "domains": {}}
    for dom in D.DOMAINS:
        arm = book["_choice"][dom]["arm"]
        seqs = [s for s in val if s["domain"] == dom]
        S = F.make_samples(seqs)
        t = tune[dom]
        ref = t["reference"]
        preds = {"kalman": F.kalman_forecast(seqs, S, F.KalmanBank(**t["kalman"]["chosen"]))}
        if ref != "kalman":
            preds[ref] = (F.forecast_classical(S, ref, t[ref]["chosen"] if ref in ("dcv", "ou", "oudcv") else None), None)
        names = {}
        for s in C.SEEDS:
            c1c = f"c1c_{ARCH}_domtok_s{s}" if arm == "domtok" else f"c1c_{ARCH}_perdom_{dom}_s{s}"
            pooled = f"c1_{ARCH}_abs_s{s}"
            for nm, reg in ((c1c, book), (pooled, c1book)):
                model, meta = load(nm, reg)
                preds[nm] = M.predict(model, S, abs_pos=True, dom_tok=meta.get("dom_tok", False))
            names[s] = (c1c, pooled)
        summ, sums = {}, {}
        for nm, (mean, var) in preds.items():
            err = F.point_errors(S, mean)
            d2, nll = F.gauss_terms(S, mean, var) if var is not None else (None, None)
            summ[nm] = F.summarize(S, err, d2, nll)
            sums[nm] = F.seq_sums(S, err, nll)
        sd = {int(q): dom for q in np.unique(S["seq"])}
        h1 = [F.paired_bootstrap(sums[names[s][0]], sums[ref], sd) for s in C.SEEDS]
        h2 = [F.paired_bootstrap(sums[names[s][0]], sums[names[s][1]], sd) for s in C.SEEDS]
        h3 = [F.paired_bootstrap(sums[names[s][0]], sums["kalman"], sd, stat="nll") for s in C.SEEDS]
        cal = [summ[names[s][0]]["calibration_error"] for s in C.SEEDS]
        if all(b["rel"] <= -0.05 and b["ci95"][1] < 0 for b in h2):
            v2c = "SUPPORTED"
        elif all(b["ci95"][0] > 0 for b in h2):
            v2c = "WORSE"
        else:
            v2c = "NOT_SUPPORTED"
        out["domains"][dom] = {"arm": arm, "reference": ref, "summaries": summ,
                               "H1c_point": {"pairs": h1, "verdict": V.verdict_h1(h1)},
                               "H2c_vs_pooled": {"pairs": h2, "verdict": v2c},
                               "H3c_uncertainty": {"pairs": h3, "calibration_error": cal, "calibration_error_kalman": summ["kalman"]["calibration_error"],
                                                   "verdict": V.verdict_h2(h3, cal, summ["kalman"]["calibration_error"])}}
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    path = C.REPORTS / "C1c_domain_aware_forecasting.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# C1c — camera-mode-aware learned forecasters (Track C, oracle histories, VAL)", "",
          f"*Generated {out['generated']} by `scripts/tc_c1c.py --stage val` under `preregistration/C1c_domain_aware_forecasting.json`.*", "",
          "| domain | dev-chosen arm | H1c vs classical | H2c vs pooled C1 | H3c uncertainty | ADE: C1c s0/s1/s2 · pooled s0/s1/s2 · reference |",
          "|---|---|---|---|---|---|"]
    for dom, d in out["domains"].items():
        s = d["summaries"]
        c1c = [v["ade_px"] for k, v in s.items() if k.startswith("c1c_")]
        pooled = [v["ade_px"] for k, v in s.items() if k.startswith("c1_")]
        md.append(f"| {dom} | {d['arm']} | **{d['H1c_point']['verdict']}** | **{d['H2c_vs_pooled']['verdict']}** | **{d['H3c_uncertainty']['verdict']}** | "
                  f"{' / '.join(f'{x:.1f}' for x in c1c)} · {' / '.join(f'{x:.1f}' for x in pooled)} · {s[d['reference']]['ade_px']:.1f} ({d['reference']}) |")
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "val"))
    a = ap.parse_args()
    stage_train(list(C.SEEDS)) if a.stage == "train" else stage_val()


if __name__ == "__main__":
    main()
