#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_forecaster.py -- LSTM vs Kalman vs TRIVIAL baselines, same test windows
=============================================================================
The project reports LSTM ADE against a constant-velocity Kalman filter only.
A forecaster is only worth its citation if it beats the *cheapest* baselines:

    zero_vel     predict no motion (repeat position)
    last_vel     repeat the last observed displacement (naive constant velocity)
    mean_vel_k   repeat the mean of the last k displacements (k=4, 8, 16)
    kalman       the project's own CV Kalman (q=1e-2, r=1e-1, normalised units)
    kalman_tuned same filter, a handful of (q, r) settings swept on VAL only
    lstm         results/runs/lstm_forecaster/forecaster.pt

Everything is scored on the identical test windows with the project's own
`ade_fde` so the numbers are comparable with results/runs/lstm_forecaster/forecast_report.json.
"""
import sys, json, argparse, itertools
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/model"))
from lstm_forecaster import (load_trajectories, windows, TrajLSTM, kalman_forecast,
                             ade_fde, JUMP_PX)


def zero_vel(X, pred):      return np.zeros((len(X), pred, 2), np.float32)
def last_vel(X, pred):      return np.repeat(X[:, -1:, :], pred, axis=1)
def mean_vel(X, pred, k):   return np.repeat(X[:, -k:, :].mean(1, keepdims=True), pred, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "results/runs/lstm_forecaster/forecaster.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/runs/audit/forecaster_benchmark.json"))
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    obs, pred, stride = cfg["obs"], cfg["pred"], cfg["stride"]
    trajs = load_trajectories(verbose=False)
    Xva, Yva, Dva, Jva, Sva = windows(trajs, ["val"], obs, pred, stride)
    Xte, Yte, Dte, Jte, Ste = windows(trajs, ["test"], obs, pred, stride)
    print(f"[fc] obs {obs} pred {pred} | val {len(Xva):,} windows | test {len(Xte):,} windows")

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TrajLSTM(hidden=cfg["hidden"], layers=cfg["layers"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(Xte).to(dev)
        lstm_te = torch.cat([model(xt[i:i+4096], pred) for i in range(0, len(xt), 4096)]).cpu().numpy()

    # --- sweep Kalman (q, r) on VAL only, pick best, then apply to TEST ------- #
    grid = [(q, r) for q in (1e-4, 1e-3, 1e-2, 1e-1, 1.0) for r in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)]
    sub = np.random.default_rng(0).choice(len(Xva), size=min(6000, len(Xva)), replace=False)
    best = None
    for q, r in grid:
        ade, _ = ade_fde(kalman_forecast(Xva[sub], pred, q=q, r=r), Yva[sub], Dva[sub])
        if best is None or ade < best[0]:
            best = (ade, q, r)
    print(f"[fc] Kalman sweep on val: best q={best[1]:g} r={best[2]:g} (val ADE {best[0]:.2f} px); "
          f"project default q=1e-2 r=1e-1")

    preds = {
        "zero_vel":      zero_vel(Xte, pred),
        "last_vel":      last_vel(Xte, pred),
        "mean_vel_4":    mean_vel(Xte, pred, 4),
        "mean_vel_8":    mean_vel(Xte, pred, 8),
        "mean_vel_16":   mean_vel(Xte, pred, 16),
        "kalman_project(q=1e-2,r=1e-1)": kalman_forecast(Xte, pred, q=1e-2, r=1e-1),
        f"kalman_tuned(q={best[1]:g},r={best[2]:g})": kalman_forecast(Xte, pred, q=best[1], r=best[2]),
        "lstm": lstm_te,
    }
    subsets = [("all test windows", np.ones(len(Xte), bool)),
               (f"sudden-jump (>{JUMP_PX:.0f}px in horizon)", Jte),
               ("smooth", ~Jte),
               ("Anti-UAV", Ste == "anti_uav"),
               ("CST Anti-UAV", Ste == "cst_anti_uav")]
    table = {}
    for sname, mask in subsets:
        table[sname] = {"n": int(mask.sum())}
        for k, p in preds.items():
            ade, fde = ade_fde(p[mask], Yte[mask], Dte[mask])
            table[sname][k] = {"ADE_px": round(ade, 3), "FDE_px": round(fde, 3)}

    # what does the LSTM actually output? compare its mean predicted step to the inputs
    lstm_step = np.linalg.norm(lstm_te, axis=2).mean()
    true_step = np.linalg.norm(Yte, axis=2).mean()
    last_step = np.linalg.norm(Xte[:, -1], axis=1).mean()
    diag = {"lstm_mean_pred_step_norm": float(lstm_step), "true_mean_future_step_norm": float(true_step),
            "mean_last_obs_step_norm": float(last_step),
            "corr(lstm_step1, last_obs)": float(np.corrcoef(lstm_te[:, 0].ravel(), Xte[:, -1].ravel())[0, 1])}

    hdr = f"{'subset':<34}{'n':>7} " + "".join(f"{k[:18]:>20}" for k in preds)
    print("\n" + hdr); print("-" * len(hdr))
    for sname, row in table.items():
        print(f"{sname:<34}{row['n']:>7} " + "".join(f"{row[k]['ADE_px']:>20.2f}" for k in preds))
    print("\n(ADE in px; lower is better)")
    print("diag:", json.dumps(diag, indent=1))
    Path(a.out).write_text(json.dumps({"config": cfg, "kalman_val_sweep_best": {"q": best[1], "r": best[2], "val_ADE": best[0]},
                                       "ADE_FDE_by_subset": table, "diagnostics": diag}, indent=2), encoding="utf-8")
    print(f"[fc] wrote {a.out}")


if __name__ == "__main__":
    main()
