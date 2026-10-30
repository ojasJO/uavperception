#!/usr/bin/env python
"""Trajectory-forecast figure generated FROM MODEL OUTPUT (repair item 7; replaces the hand-typed plot of rev2 cell 10).

    python experiments/aerotrack_repair/scripts/plot_forecast_example.py

Example selection rule (fixed, label-based, no cherry-picking): among the VAL windows whose future contains a
>30 px jump, take the window at the MEDIAN LSTM ADE (i.e. a typical sudden-jump case, not the best one), plus the
window at the median ADE among smooth VAL windows. Curves: observed positions, ground-truth future, val-tuned Kalman,
mean-velocity(4) and the retrained LSTM (`results/runs/lstm_forecaster_repair/forecaster_original.pt`), all in pixels.
Writes docs/assets/repair_forecast_examples_val.png and the plotted coordinates as JSON next to it.
"""
from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lstm_forecaster import JUMP_PX, TrajLSTM, ade_fde, kalman_forecast, load_trajectories, windows

from anti_uav_repro.utils.repro import git_info, write_json

from _bootstrap import REPO_ROOT

RUN = REPO_ROOT / "results/runs" / "lstm_forecaster_repair"
OUT_PNG = REPO_ROOT / "docs/assets" / "repair_forecast_examples_val.png"


def main():
    rep = json.loads((RUN / "forecast_repair_report.json").read_text(encoding="utf-8"))
    cfg = rep["config"]
    obs, pred = cfg["obs"], cfg["pred"]
    kq, kr = rep["kalman_val_sweep_best"]["q"], rep["kalman_val_sweep_best"]["r"]
    ck = torch.load(RUN / "forecaster_original.pt", map_location="cpu", weights_only=False)
    model = TrajLSTM(hidden=cfg["hidden"], layers=cfg["layers"])
    model.load_state_dict(ck["model"])
    model.eval()
    trajs = load_trajectories(verbose=False)
    X, Y, D, J, S = windows(trajs, ["val"], obs, pred, cfg["stride"])
    with torch.no_grad():
        P = model(torch.from_numpy(X), pred).numpy()
    per_win = np.linalg.norm(np.cumsum(P, 1) - np.cumsum(Y, 1), axis=2).mean(1) * D   # per-window LSTM ADE in px
    picks = {}
    for name, mask in (("sudden-jump (median LSTM ADE)", J), ("smooth (median LSTM ADE)", ~J)):
        idx = np.where(mask)[0]
        order = idx[np.argsort(per_win[idx])]
        picks[name] = int(order[len(order) // 2])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    dump = {"rule": "median LSTM ADE among VAL sudden-jump windows / among VAL smooth windows", "git": git_info().get("commit"), "examples": {}}
    for ax, (name, i) in zip(axes, picks.items()):
        diag = D[i]
        obs_pos = np.vstack([[0, 0], np.cumsum(X[i], 0)]) * diag
        o = obs_pos[-1]
        gt = o + np.cumsum(Y[i], 0) * diag
        kf = o + np.cumsum(kalman_forecast(X[i:i + 1], pred, q=kq, r=kr)[0], 0) * diag
        mv = o + np.cumsum(np.repeat(X[i:i + 1, -4:].mean(1, keepdims=True), pred, 1)[0], 0) * diag
        ls = o + np.cumsum(P[i], 0) * diag
        for arr, lab, c, m in ((obs_pos, "observed (16 steps)", "#17549e", "o"), (np.vstack([o, gt]), "ground truth future", "#00631a", "s"),
                               (np.vstack([o, kf]), f"Kalman (val-tuned q={kq:g}, r={kr:g})", "#b85600", "^"),
                               (np.vstack([o, mv]), "mean-velocity(4)", "#7d5000", "v"), (np.vstack([o, ls]), "LSTM (retrained, original param.)", "#a3271a", "D")):
            ax.plot(arr[:, 0], arr[:, 1], color=c, marker=m, ms=4, lw=1.8, label=lab)
        ade = {lab: float(np.linalg.norm(a - gt, axis=1).mean()) for lab, a in (("kalman", kf), ("mean_vel_4", mv), ("lstm", ls))}
        ax.set_title(f"VAL {name}\nADE px — Kalman {ade['kalman']:.1f} | mean-vel4 {ade['mean_vel_4']:.1f} | LSTM {ade['lstm']:.1f}", fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("x (px, relative to first observed position)")
        ax.set_ylabel("y (px)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        dump["examples"][name] = {"window_index": i, "dataset": str(S[i]), "frame_diag_px": float(diag), "observed_px": obs_pos.tolist(),
                                  "gt_future_px": gt.tolist(), "kalman_px": kf.tolist(), "mean_vel4_px": mv.tolist(), "lstm_px": ls.tolist(), "ade_px": ade}
    fig.suptitle("Trajectory forecasting on VAL windows — every curve is computed from data or model output (see JSON next to this file)", fontsize=10)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    write_json(dump, OUT_PNG.with_suffix(".json"))
    print(f"wrote {OUT_PNG} and {OUT_PNG.with_suffix('.json')}; picks {picks}")


if __name__ == "__main__":
    main()
