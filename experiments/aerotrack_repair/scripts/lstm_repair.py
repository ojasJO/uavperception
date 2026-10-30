#!/usr/bin/env python
"""Trajectory-forecaster repair (repair item 5): trivial baselines inside the report + a parameterisation
that cannot collapse to "no motion", per-domain, selection on VAL, one pass on TEST.

    python experiments/aerotrack_repair/scripts/lstm_repair.py [--epochs 30]

Audit finding: `c_model/lstm_forecaster.py` trains on MSE over cumulative positions of near-zero-mean
displacements; the trained model outputs ~0 regardless of input (ADE == zero-velocity baseline to 3 s.f.)
and its "+26 % over Kalman" is the margin by which *predicting no motion* beats constant velocity on
gimbal-jittered footage. This script keeps the model, data, windows and metric (`ade_fde`) of the original
and changes ONE thing per variant:

    original      TrajLSTM, output = displacement                      (the shipped model, retrained for the record)
    residual_lv   TrajLSTM, output = last observed displacement + f(x) (constant-velocity prior; f learns deviations)
    residual_mv4  TrajLSTM, output = mean of last 4 displacements + f(x)

Baselines (audit/bench_forecaster.py): zero_vel, last_vel, mean_vel_4/8/16, project Kalman, Kalman tuned on VAL.
Checkpoint selection: best VAL ADE per variant (as the original). The TEST table is produced once, at the end,
for the selected checkpoints and all baselines. Per-domain (Anti-UAV / CST) and sudden-jump / smooth subsets.
Outputs: runs/lstm_forecaster_repair/{forecaster_<variant>.pt, forecast_repair_report.json, report.md}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch import nn

from lstm_forecaster import JUMP_PX, TrajLSTM, ade_fde, kalman_forecast, load_trajectories, windows

from anti_uav_repro.utils.repro import env_snapshot, git_info, write_json

from _bootstrap import REPO_ROOT

OUT_DIR = REPO_ROOT / "runs" / "lstm_forecaster_repair"
VARIANTS = ("original", "residual_lv", "residual_mv4")


def zero_vel(X, pred):
    return np.zeros((len(X), pred, 2), np.float32)


def last_vel(X, pred):
    return np.repeat(X[:, -1:, :], pred, axis=1)


def mean_vel(X, pred, k):
    return np.repeat(X[:, -k:, :].mean(1, keepdims=True), pred, axis=1)


def prior(x: torch.Tensor, pred: int, variant: str) -> torch.Tensor:
    """Displacement prior the network adds to (zero for the original parameterisation)."""
    if variant == "original":
        return torch.zeros(x.shape[0], pred, 2, device=x.device, dtype=x.dtype)
    if variant == "residual_lv":
        return x[:, -1:, :].expand(-1, pred, -1)
    if variant == "residual_mv4":
        return x[:, -4:, :].mean(1, keepdim=True).expand(-1, pred, -1)
    raise ValueError(variant)


def predict(model, x, pred, variant, chunk=4096):
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            xb = x[i:i + chunk]
            outs.append(model(xb, pred) + prior(xb, pred, variant))
    return torch.cat(outs).cpu().numpy()


def train_variant(variant, Xtr, Ytr, Xva, Yva, Dva, obs, pred, hidden, layers, epochs, batch, lr, dev, seed, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TrajLSTM(hidden=hidden, layers=layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev)
    xv = torch.from_numpy(Xva).to(dev)
    n = len(xt)
    best = (float("inf"), None, -1)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = xt[idx]
            p = model(xb, pred) + prior(xb, pred, variant)
            loss = nn.functional.mse_loss(p.cumsum(1), yt[idx].cumsum(1))   # same objective as the original
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        pv = predict(model, xv, pred, variant)
        ade, fde = ade_fde(pv, Yva, Dva)
        history.append({"epoch": ep, "train_mse": tot / n, "val_ADE_px": ade, "val_FDE_px": fde})
        if ade < best[0]:
            best = (ade, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        if ep % max(epochs // 6, 1) == 0 or ep == epochs - 1:
            log(f"  [{variant}] ep {ep:>3d} train_mse {tot/n:.6f} val ADE {ade:7.3f} px FDE {fde:7.3f} px")
    model.load_state_dict(best[1])
    return model, history, best[2], best[0]


def score_table(preds: dict, Y, D, J, S):
    subsets = [("all", np.ones(len(Y), bool)), (f"sudden-jump (>{JUMP_PX:.0f}px in horizon)", J), ("smooth", ~J),
               ("Anti-UAV", S == "anti_uav"), ("CST Anti-UAV", S == "cst_anti_uav"),
               ("CST sudden-jump", (S == "cst_anti_uav") & J), ("Anti-UAV sudden-jump", (S == "anti_uav") & J)]
    table = {}
    for name, mask in subsets:
        table[name] = {"n": int(mask.sum())}
        if not mask.any():
            continue
        for k, p in preds.items():
            ade, fde = ade_fde(p[mask], Y[mask], D[mask])
            table[name][k] = {"ADE_px": round(ade, 3), "FDE_px": round(fde, 3)}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", type=int, default=16)
    ap.add_argument("--pred", type=int, default=12)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    trajs = load_trajectories(verbose=False)
    Xtr, Ytr, Dtr, Jtr, Str = windows(trajs, ["train"], a.obs, a.pred, a.stride)
    Xva, Yva, Dva, Jva, Sva = windows(trajs, ["val"], a.obs, a.pred, a.stride)
    Xte, Yte, Dte, Jte, Ste = windows(trajs, ["test"], a.obs, a.pred, a.stride)
    log(f"[data] windows train {len(Xtr):,} | val {len(Xva):,} | test {len(Xte):,} (obs {a.obs} -> pred {a.pred}); "
        f"sudden-jump share: val {100*Jva.mean():.1f} %, test {100*Jte.mean():.1f} %")

    # Kalman (q, r) chosen on VAL only
    grid = [(q, r) for q in (1e-4, 1e-3, 1e-2, 1e-1, 1.0) for r in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)]
    sub = np.random.default_rng(0).choice(len(Xva), size=min(6000, len(Xva)), replace=False)
    kbest = None
    for q, r in grid:
        ade, _ = ade_fde(kalman_forecast(Xva[sub], a.pred, q=q, r=r), Yva[sub], Dva[sub])
        if kbest is None or ade < kbest[0]:
            kbest = (ade, q, r)
    log(f"[kalman] val sweep best q={kbest[1]:g} r={kbest[2]:g} (val ADE {kbest[0]:.2f} px on a 6000-window subset)")

    def baselines(X):
        return {"zero_vel": zero_vel(X, a.pred), "last_vel": last_vel(X, a.pred), "mean_vel_4": mean_vel(X, a.pred, 4),
                "mean_vel_8": mean_vel(X, a.pred, 8), "mean_vel_16": mean_vel(X, a.pred, 16),
                "kalman_project(q=1e-2,r=1e-1)": kalman_forecast(X, a.pred, q=1e-2, r=1e-1),
                f"kalman_val_tuned(q={kbest[1]:g},r={kbest[2]:g})": kalman_forecast(X, a.pred, q=kbest[1], r=kbest[2])}

    val_preds, test_preds = baselines(Xva), baselines(Xte)
    models = {}
    for variant in a.variants:
        t0 = time.time()
        model, hist, best_ep, best_ade = train_variant(variant, Xtr, Ytr, Xva, Yva, Dva, a.obs, a.pred, a.hidden, a.layers,
                                                       a.epochs, a.batch, a.lr, dev, a.seed, log)
        log(f"[{variant}] selected epoch {best_ep} (val ADE {best_ade:.3f} px) in {time.time()-t0:.0f}s")
        torch.save({"model": model.state_dict(), "variant": variant,
                    "config": {"obs": a.obs, "pred": a.pred, "stride": a.stride, "hidden": a.hidden, "layers": a.layers,
                               "epochs": a.epochs, "lr": a.lr, "seed": a.seed, "selected_epoch": best_ep}},
                   OUT_DIR / f"forecaster_{variant}.pt")
        val_preds[f"lstm_{variant}"] = predict(model, torch.from_numpy(Xva).to(dev), a.pred, variant)
        test_preds[f"lstm_{variant}"] = predict(model, torch.from_numpy(Xte).to(dev), a.pred, variant)
        models[variant] = {"history": hist, "selected_epoch": best_ep, "val_ADE_px": best_ade,
                           "n_params": int(sum(p.numel() for p in model.parameters()))}

    # diagnostics: does each model actually move?
    diag = {}
    for k, p in test_preds.items():
        if k.startswith("lstm_"):
            diag[k] = {"mean_pred_step_norm": float(np.linalg.norm(p, axis=2).mean()),
                       "corr(step1, last_obs)": float(np.corrcoef(p[:, 0].ravel(), Xte[:, -1].ravel())[0, 1])}
    diag["true_mean_future_step_norm"] = float(np.linalg.norm(Yte, axis=2).mean())
    diag["mean_last_obs_step_norm"] = float(np.linalg.norm(Xte[:, -1], axis=1).mean())

    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_info(), "env": env_snapshot(), "command": " ".join(sys.argv),
              "config": vars(a), "windows": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
              "kalman_val_sweep_best": {"q": kbest[1], "r": kbest[2], "val_ADE_px": kbest[0]},
              "models": models, "val_table": score_table(val_preds, Yva, Dva, Jva, Sva),
              "test_table": score_table(test_preds, Yte, Dte, Jte, Ste), "diagnostics": diag,
              "selection_rule": "per variant: epoch with the lowest VAL ADE; Kalman (q,r) by VAL sweep; TEST scored once for all methods"}
    write_json(report, OUT_DIR / "forecast_repair_report.json")
    (OUT_DIR / "report.md").write_text(render_md(report), encoding="utf-8")
    (OUT_DIR / "train_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(render_md(report))
    print(f"wrote {OUT_DIR}")


def render_md(rep):
    L = ["# Trajectory forecasting — repaired comparison (ADE in px, lower is better)", "",
         f"Generated {rep['generated']} (git {rep['git'].get('commit', '')[:10]}). Windows: train {rep['windows']['train']:,}, "
         f"val {rep['windows']['val']:,}, test {rep['windows']['test']:,}. Selection: {rep['selection_rule']}.", ""]
    for split in ("val_table", "test_table"):
        tab = rep[split]
        methods = [k for k in next(iter(tab.values())) if k != "n"]
        L += [f"## {split.replace('_table', '').upper()}", "", "| subset | n | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
        for sname, row in tab.items():
            cells = [f"{row[m]['ADE_px']:.2f}" if m in row else "—" for m in methods]
            L.append(f"| {sname} | {row['n']} | " + " | ".join(cells) + " |")
        L.append("")
    d = rep["diagnostics"]
    L += ["## Does the model move? (test windows, normalised units)", "",
          f"- true mean future step: {d['true_mean_future_step_norm']:.5f}; mean last observed step: {d['mean_last_obs_step_norm']:.5f}"]
    for k, v in d.items():
        if isinstance(v, dict):
            L.append(f"- {k}: mean predicted step {v['mean_pred_step_norm']:.5f}, corr(step1, last obs) {v['corr(step1, last_obs)']:+.3f}")
    L += ["", "## Reading rule (pre-registered)", "",
          "The forecaster earns its place only if, on the TEST table, it beats the best trivial baseline (min over zero/last/mean-vel) "
          "AND the val-tuned Kalman on the CST sudden-jump subset — the population the citation rests on. Otherwise the LSTM is "
          "retired from the pipeline and the README says so.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
