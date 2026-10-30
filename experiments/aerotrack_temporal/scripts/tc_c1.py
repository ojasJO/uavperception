#!/usr/bin/env python
"""Track C, experiment C1: trajectory forecasting from causal histories (oracle ground-truth histories; tracker histories in C1b).

    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_c1.py --stage tune     TRAIN only: classical tuning
    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_c1.py --stage train    TRAIN only: learned models
    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_c1.py --stage val      VAL, once; needs the committed
                                                                                              preregistration/C1_trajectory_forecasting.json
Protocol and rules: experiments/aerotrack_temporal/preregistration/C1_trajectory_forecasting.json. TRAIN = non-dev TRAIN groups;
dev = held-out TRAIN groups (tc_data.is_dev) used only for the learned models' epoch choice and the architecture choice; VAL is
read only by --stage val; TEST is sealed (tc_data refuses it).
Outputs: runs/aerotrack_temporal/c1/ (tuning JSON, checkpoints + sha256, logs) and experiments/aerotrack_temporal/reports/C1_*.
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import tc_data as D  # noqa: E402
import tc_forecast as F  # noqa: E402

RUN = REPO / "runs" / "aerotrack_temporal" / "c1"
REPORTS = REPO / "experiments" / "aerotrack_temporal" / "reports"
PRE = REPO / "experiments" / "aerotrack_temporal" / "preregistration" / "C1_trajectory_forecasting.json"
CLASSICAL = ("cp", "cv", "mv2", "mv4", "mv8", "mv15", "ca")
G_GRID = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0)
K_GRID = (1, 2, 4, 8, 12, 15)                     # 15 = the longest look-back inside the 16-frame history
RHO_GRID = (0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 1.0)
DCV_GRID = [{"g": g, "k": k} for g, k in itertools.product(G_GRID, K_GRID)]
OU_GRID = [{"rho": r} for r in RHO_GRID]
OUDCV_GRID = [{"rho": r, "g": g, "k": k} for r, g, k in itertools.product(RHO_GRID, (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0), (1, 4, 8, 15))]
KF_GRID = [{"q_pos": a, "q_vel": b, "r_meas": c} for a, b, c in itertools.product((1 / 160, 1 / 80, 1 / 40, 1 / 20, 1 / 10, 1 / 5, 2 / 5, 4 / 5),
                                                                               (1 / 5120, 1 / 2560, 1 / 1280, 1 / 640, 1 / 320, 1 / 160, 1 / 80),
                                                                               (1 / 80, 1 / 40, 1 / 20, 1 / 10))]
COV_SCALES = tuple(float(x) for x in np.round(np.exp(np.linspace(np.log(0.01), np.log(500), 61)), 5))
# natural limits are not grid artefacts: k = 1 (shortest look-back), g = 0 / 1 (constant position / no damping), rho = 1 (no reversion),
# k = 15 (whole history); the CV-Kalman's constant-position limit (q_pos up, q_vel and r_meas down) is reported, not chased
NATURAL = {"k": (1, 15), "g": (0.0, 1.0), "rho": (1.0,)}
ARCHS = ("mlp", "gru", "tfm")
SEEDS = (0, 1, 2)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def split_train():
    seqs = D.load_split("train")
    tr = [s for s in seqs if not s["dev"]]
    dv = [s for s in seqs if s["dev"]]
    return tr, dv


def pooled_ade(S, mean):
    return float(np.nanmean(F.point_errors(S, mean)))


def _edge(grid, chosen):
    """Names of parameters whose chosen value lies on the edge of its grid (the grid would need extending)."""
    out = []
    for k, v in chosen.items():
        vals = sorted({g[k] for g in grid})
        if len(vals) > 1 and v in (vals[0], vals[-1]) and v not in NATURAL.get(k, ()):
            out.append(k)
    return out


def _search(S, name, grid):
    res = [(pooled_ade(S, F.forecast_classical(S, name, p)), p) for p in grid]
    best = min(res, key=lambda x: x[0])
    return {"chosen": best[1], "ade": best[0], "edge": _edge(grid, best[1]), "grid": [{"ade": a, **p} for a, p in res]}


def stage_tune():
    """Per domain, on TRAIN (non-dev groups): every classical forecaster's parameters by ADE, the Kalman covariance scale by NLL,
    and the best classical forecaster (lowest TRAIN ADE) = the domain's reference for C1-H1."""
    tr, _ = split_train()
    out = {"generated": now(), "split": "train (non-dev groups)", "domains": {}}
    for dom in D.DOMAINS:
        seqs = [s for s in tr if s["domain"] == dom]
        S = F.make_samples(seqs)
        r = {"n_sequences": len(seqs), "n_samples": int(len(S["t"])),
             "fixed": {nm: pooled_ade(S, F.forecast_classical(S, nm)) for nm in CLASSICAL}}
        r["dcv"] = _search(S, "dcv", DCV_GRID)
        r["ou"] = _search(S, "ou", OU_GRID)
        r["oudcv"] = _search(S, "oudcv", OUDCV_GRID)
        kf = []
        for p in KF_GRID:
            mean, _ = F.kalman_forecast(seqs, S, F.KalmanBank(**p))
            kf.append((pooled_ade(S, mean), p))
        kbest = min(kf, key=lambda x: x[0])
        mean, var = F.kalman_forecast(seqs, S, F.KalmanBank(**kbest[1]))
        nll = sorted((float(np.nanmean(F.gauss_terms(S, mean, var * c)[1])), c) for c in COV_SCALES)
        r["kalman"] = {"chosen": {**kbest[1], "cov_scale": nll[0][1]}, "ade": kbest[0], "nll": nll[0][0],
                       "edge": _edge(KF_GRID, kbest[1]) + (["cov_scale"] if nll[0][1] in (COV_SCALES[0], COV_SCALES[-1]) else []),
                       "grid": [{"ade": a, **p} for a, p in kf]}
        allc = dict(r["fixed"], dcv=r["dcv"]["ade"], ou=r["ou"]["ade"], oudcv=r["oudcv"]["ade"], kalman=r["kalman"]["ade"])
        r["train_ade"] = allc
        r["reference"] = min(allc, key=allc.get)
        out["domains"][dom] = r
        print(dom, json.dumps({"n": r["n_samples"], "ade": {k: round(v, 2) for k, v in allc.items()}, "reference": r["reference"],
                               "dcv": r["dcv"]["chosen"], "ou": r["ou"]["chosen"], "oudcv": r["oudcv"]["chosen"],
                               "kalman": r["kalman"]["chosen"], "edges": {k: r[k]["edge"] for k in ("dcv", "ou", "oudcv", "kalman")}}),
              flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "classical_tuning.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def stage_train(archs, seeds, variants, epochs):
    import tc_learned as M
    tr, dv = split_train()
    S_tr, S_dv = F.make_samples(tr), F.make_samples(dv)
    RUN.mkdir(parents=True, exist_ok=True)
    reg = RUN / "learned_registry.json"
    book = json.loads(reg.read_text(encoding="utf-8")) if reg.exists() else {}
    for variant in variants:
        abs_pos = "noabs" not in variant
        aug = "aug" in variant
        for arch in archs:
            for seed in seeds:
                name = f"c1_{arch}_{variant}_s{seed}"
                if name in book:
                    print("skip", name)
                    continue
                logf = open(RUN / f"{name}.log", "a", encoding="utf-8")
                model, hist = M.train(arch, S_tr, S_dv, seed, abs_pos=abs_pos, aug=aug, epochs=epochs,
                                      log=lambda s: (logf.write(s + "\n"), logf.flush()))
                logf.close()
                meta = {"arch": arch, "variant": variant, "seed": seed, "abs_pos": abs_pos, "aug": aug, "L": F.L_HIST, "H": F.H_FUT,
                        "epochs": epochs, "best_epoch": int(np.argmin([h["dev_nll"] for h in hist]) + 1),
                        "dev_nll": float(min(h["dev_nll"] for h in hist)), "train_seqs": len(tr), "dev_seqs": len(dv)}
                sha = M.save(model, RUN / f"{name}.pt", meta)
                mean, var = M.predict(model, S_dv, abs_pos=abs_pos)
                meta["dev_ade_px"] = pooled_ade(S_dv, mean)
                meta["sha256"] = sha
                meta["trained"] = now()
                book[name] = meta
                reg.write_text(json.dumps(book, indent=1), encoding="utf-8")
                print(name, json.dumps({k: meta[k] for k in ("best_epoch", "dev_nll", "dev_ade_px")}), flush=True)


def preregistered():
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(PRE)], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(PRE)], cwd=REPO).returncode == 0
    return tracked and clean


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("tune", "train", "val"))
    ap.add_argument("--archs", default=",".join(ARCHS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--variants", default="abs")
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    if a.stage == "tune":
        stage_tune()
    elif a.stage == "train":
        archs = a.archs.split(",")
        if archs == ["auto"]:                         # the registered architecture choice: lowest seed-mean dev NLL of the abs arms
            import tc_c1_val
            book = json.loads((RUN / "learned_registry.json").read_text(encoding="utf-8"))
            archs = [tc_c1_val.chosen_arch(book)[0]]
            print("architecture chosen on dev:", archs[0], flush=True)
        stage_train(archs, [int(s) for s in a.seeds.split(",")], a.variants.split(","), a.epochs)
    else:
        if not preregistered():
            raise SystemExit(f"{PRE.relative_to(REPO)} must be committed and unmodified before VAL is read")
        import tc_c1_val
        tc_c1_val.run()


if __name__ == "__main__":
    main()
