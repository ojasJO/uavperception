#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lstm_forecaster.py  --  trajectory forecasting for AeroTrack-Net
================================================================
Alahi et al., *Social LSTM: Human Trajectory Prediction in Crowded Spaces*,
CVPR 2016 -- the fourth paper Review 1 adopted, and the one with the sharpest
measured justification in the whole project.

The number
----------
A Kalman filter assumes constant velocity. `tracking_metrics.json` records
**32,782 inter-frame displacements over 30 px** across 706,147 transitions, and
`xfactor_analysis.json` puts corr(speed, area) at **-0.043 / +0.064** -- i.e.
essentially zero. Speed is not predictable from scale, and the motion contains
tens of thousands of discontinuities. Those are dives and hard banking turns:
precisely the events a constant-velocity model extrapolates straight through,
and precisely the events a counter-UAS system must not lose.

So the claim under test is narrow and falsifiable: **an LSTM should beat a
Kalman filter on the sudden-jump subset specifically**, and may well tie with it
everywhere else, because most of the corpus really is near-linear motion (median
5.10 px/frame for Anti-UAV, 0.64 for CST). This module reports both, separately,
because a single averaged number would let the linear majority hide the result
either way.

What is faithful to the paper and what is not
---------------------------------------------
* **Faithful:** the encoder-decoder LSTM over *relative displacements* (which is
  what makes the model translation-invariant), an autoregressive decoder rather
  than a one-shot head, and evaluation by ADE / FDE.
* **Not applicable:** the *social* pooling layer. Social-LSTM's contribution is
  a shared hidden-state grid over neighbouring pedestrians. This corpus tracks
  **one** target per sequence, so there are no neighbours to pool over; keeping
  the layer would be citation theatre. The sequence-to-sequence formulation is
  what transfers, and that is what is implemented. Said plainly rather than
  glossed over.

Units
-----
Anti-UAV mixes IR (640x512) and RGB (1920x1080) in *raw pixels*, so a raw-px
model would learn that RGB drones are three times faster than IR ones -- a
resolution artifact, not motion. Every displacement is therefore divided by its
own frame diagonal (recovered per row as `speed_px / speed_norm`), trained
scale-free, and converted back to pixels only for reporting.

Splits
------
Sequence-level and taken from the sequence names themselves, so this is
disjoint in the same way the detector's splits are:
Anti-UAV `train/…|val/…|test/…`, CST `CST-AntiUAV/CST-AntiUAV/<split>/…`.

Usage
-----
    python c_model/lstm_forecaster.py                    # train + evaluate
    python c_model/lstm_forecaster.py --device cpu       # while the GPU trains
    python c_model/lstm_forecaster.py --obs 16 --pred 12
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "a_inspection" / "artifacts"
VELOCITIES = ART / "velocities.pkl"
JUMP_PX = 30.0                      # the roadmap's "sudden jump" definition


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def _split_of(dataset, sequence):
    """Recover the official split from a sequence name."""
    parts = sequence.split("/")
    if dataset == "anti_uav":
        return parts[0]
    if dataset == "cst_anti_uav":
        return parts[2] if len(parts) > 2 else "train"
    return "train"


def load_trajectories(path=VELOCITIES, verbose=True):
    """-> list of dicts: one per sequence, with scale-free displacements.

    Rows in `velocities.pkl` are consecutive within a sequence, so the array is
    already the trajectory -- there is nothing to re-derive from images.
    """
    with open(path, "rb") as f:
        v = pickle.load(f)
    out = []
    for (ds, seq), g in v.groupby(["dataset", "sequence"], sort=True):
        d = g[["dx", "dy"]].to_numpy(dtype=np.float32)
        sp = g["speed_px"].to_numpy(dtype=np.float64)
        sn = g["speed_norm"].to_numpy(dtype=np.float64)
        ok = (sn > 0) & (sp > 0)
        # diag = speed_px / speed_norm, constant per sequence by construction;
        # the median is just a robust way to read it off.
        diag = float(np.median(sp[ok] / sn[ok])) if ok.any() else 1.0
        if not np.isfinite(diag) or diag <= 1.0:
            diag = 1.0
        out.append({"dataset": ds, "sequence": seq,
                    "split": _split_of(ds, seq),
                    "diag": diag,
                    "modality": str(g["modality"].iloc[0]),
                    "disp": d / diag,           # scale-free
                    "disp_px": d})
    if verbose:
        from collections import Counter
        c = Counter((t["dataset"], t["split"]) for t in out)
        print(f"[traj] {len(out)} sequences, "
              f"{sum(len(t['disp']) for t in out):,} transitions")
        for k in sorted(c):
            print(f"       {k[0]:<14} {k[1]:<6} {c[k]:>4} sequences")
    return out


def windows(trajs, splits, obs=16, pred=12, stride=4):
    """Slice trajectories into (observed, future) displacement windows.

    Returns X [n,obs,2], Y [n,pred,2] (scale-free), plus the per-window frame
    diagonal and a boolean flag for windows whose FUTURE contains a jump over
    JUMP_PX. That flag is the whole experiment: it selects the population where
    the constant-velocity assumption is known to be wrong.
    """
    X, Y, D, J, S = [], [], [], [], []
    want = set(splits)
    need = obs + pred
    for t in trajs:
        if t["split"] not in want:
            continue
        d, dpx, diag = t["disp"], t["disp_px"], t["diag"]
        n = len(d)
        if n < need:
            continue
        for i in range(0, n - need + 1, stride):
            fut_px = dpx[i + obs:i + need]
            X.append(d[i:i + obs])
            Y.append(d[i + obs:i + need])
            D.append(diag)
            J.append(bool((np.hypot(fut_px[:, 0], fut_px[:, 1]) > JUMP_PX).any()))
            S.append(t["dataset"])
    if not X:
        raise SystemExit(f"no windows for splits={splits}")
    return (np.stack(X), np.stack(Y), np.asarray(D, dtype=np.float32),
            np.asarray(J), np.asarray(S))


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class TrajLSTM(nn.Module):
    """Encoder-decoder LSTM over displacements.

    The decoder is autoregressive -- it consumes its own previous output, the
    way Social-LSTM's does. A one-shot head that emits all `pred` steps from the
    encoder state trains faster and scores slightly better on ADE, and it is
    also not the model in the paper: it cannot represent a trajectory whose
    later steps depend on its own earlier ones, which is exactly what a dive is.
    """

    def __init__(self, hidden=128, layers=1, embed=32, dropout=0.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(2, embed), nn.ReLU())
        self.enc = nn.LSTM(embed, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.dec = nn.LSTM(embed, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 2)
        self.hidden, self.layers = hidden, layers

    def forward(self, x, pred_len):
        _, state = self.enc(self.embed(x))
        step = x[:, -1:, :]                       # last observed displacement
        outs = []
        for _ in range(pred_len):
            o, state = self.dec(self.embed(step), state)
            step = self.head(o)
            outs.append(step)
        return torch.cat(outs, 1)


# --------------------------------------------------------------------------- #
#  Kalman baseline (constant velocity) — the thing we have to beat
# --------------------------------------------------------------------------- #
def kalman_forecast(x_obs, pred_len, q=1e-2, r=1e-1):
    """Constant-velocity Kalman filter, run per window, in numpy.

    State (x, y, vx, vy) integrated over the observed displacements, then rolled
    forward `pred_len` steps with no measurements. This is the standard tracker
    baseline and it is genuinely strong on this corpus -- most motion is smooth.
    Returned as DISPLACEMENTS, to compare like with like.
    """
    n, obs, _ = x_obs.shape
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    Q = np.eye(4) * q
    R = np.eye(2) * r
    pos = np.cumsum(x_obs, axis=1)                       # positions rel. to t0
    out = np.zeros((n, pred_len, 2), dtype=np.float32)
    for i in range(n):
        m = np.array([pos[i, 0, 0], pos[i, 0, 1], 0.0, 0.0])
        P = np.eye(4)
        for t in range(1, obs):
            m = F @ m
            P = F @ P @ F.T + Q
            y = pos[i, t] - H @ m
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            m = m + K @ y
            P = (np.eye(4) - K @ H) @ P
        last = m[:2].copy()
        for t in range(pred_len):
            m = F @ m
            out[i, t] = m[:2] - last
            last = m[:2].copy()
    return out


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def ade_fde(pred_disp, true_disp, diag):
    """ADE / FDE in PIXELS from scale-free displacements.

    Displacements are cumulatively summed into positions first -- error must be
    measured where it accumulates. A model can have a small per-step error and a
    large final position error, and the final position is what a interceptor
    needs.
    """
    p = np.cumsum(np.asarray(pred_disp, dtype=np.float64), axis=1)
    t = np.cumsum(np.asarray(true_disp, dtype=np.float64), axis=1)
    err = np.linalg.norm(p - t, axis=2) * diag[:, None]
    return float(err.mean()), float(err[:, -1].mean())


# --------------------------------------------------------------------------- #
def train_forecaster(obs=16, pred=12, stride=4, hidden=128, layers=1,
                     epochs=30, batch=512, lr=1e-3, device=None, seed=0,
                     out_dir=None, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    trajs = load_trajectories(verbose=verbose)
    Xtr, Ytr, Dtr, Jtr, Str = windows(trajs, ["train"], obs, pred, stride)
    Xva, Yva, Dva, Jva, Sva = windows(trajs, ["val"], obs, pred, stride)
    Xte, Yte, Dte, Jte, Ste = windows(trajs, ["test"], obs, pred, stride)
    if verbose:
        print(f"[data] windows: train {len(Xtr):,} | val {len(Xva):,} | "
              f"test {len(Xte):,}  (obs {obs} -> pred {pred})")
        print(f"[data] windows whose FUTURE contains a >{JUMP_PX:.0f} px jump: "
              f"train {Jtr.sum():,} ({Jtr.mean()*100:.1f}%) | "
              f"test {Jte.sum():,} ({Jte.mean()*100:.1f}%)")

    model = TrajLSTM(hidden=hidden, layers=layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(Ytr).to(dev)
    xv = torch.from_numpy(Xva).to(dev)
    yv = torch.from_numpy(Yva).to(dev)

    n = len(xt)
    best = (float("inf"), None)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            p = model(xt[idx], pred)
            # loss on cumulative POSITIONS, so it optimises the ADE we report
            loss = nn.functional.mse_loss(p.cumsum(1), yt[idx].cumsum(1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(xv[i:i + 4096], pred)
                            for i in range(0, len(xv), 4096)]).cpu().numpy()
        ade, fde = ade_fde(pv, Yva, Dva)
        history.append({"epoch": ep, "train_mse": tot / n, "val_ADE_px": ade,
                        "val_FDE_px": fde})
        if ade < best[0]:
            best = (ade, {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()})
        if verbose and (ep % max(epochs // 10, 1) == 0 or ep == epochs - 1):
            print(f"  ep {ep:>3d}  train_mse {tot/n:.6f}  "
                  f"val ADE {ade:7.3f} px  FDE {fde:7.3f} px", flush=True)
    if best[1] is not None:
        model.load_state_dict(best[1])

    # ---- the comparison the paper choice rests on ------------------------- #
    model.eval()
    xte = torch.from_numpy(Xte).to(dev)
    with torch.no_grad():
        lstm_pred = torch.cat([model(xte[i:i + 4096], pred)
                               for i in range(0, len(xte), 4096)]).cpu().numpy()
    kf_pred = kalman_forecast(Xte, pred)

    def block(mask, label):
        if not mask.any():
            return {"subset": label, "n": 0}
        la, lf = ade_fde(lstm_pred[mask], Yte[mask], Dte[mask])
        ka, kf = ade_fde(kf_pred[mask], Yte[mask], Dte[mask])
        return {"subset": label, "n": int(mask.sum()),
                "LSTM_ADE_px": round(la, 4), "LSTM_FDE_px": round(lf, 4),
                "Kalman_ADE_px": round(ka, 4), "Kalman_FDE_px": round(kf, 4),
                "ADE_improvement_pct": round((ka - la) / max(ka, 1e-9) * 100, 2),
                "FDE_improvement_pct": round((kf - lf) / max(kf, 1e-9) * 100, 2)}

    report = {
        "config": {"obs": obs, "pred": pred, "stride": stride, "hidden": hidden,
                   "layers": layers, "epochs": epochs, "lr": lr,
                   "jump_px": JUMP_PX, "device": str(dev)},
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "windows": {"train": int(len(Xtr)), "val": int(len(Xva)),
                    "test": int(len(Xte))},
        "history": history,
        "results": [
            block(np.ones(len(Xte), dtype=bool), "all test windows"),
            block(Jte, f"SUDDEN-JUMP windows (>{JUMP_PX:.0f} px in horizon)"),
            block(~Jte, "smooth windows"),
            block(Ste == "anti_uav", "Anti-UAV"),
            block(Ste == "cst_anti_uav", "CST Anti-UAV"),
        ],
    }

    if verbose:
        print("\n" + "=" * 78)
        print("Trajectory forecasting — LSTM vs constant-velocity Kalman")
        print(f"  observe {obs} steps -> forecast {pred} steps, on the TEST "
              f"sequences ({len(Xte):,} windows)")
        print("=" * 78)
        hdr = f"  {'subset':<44}{'n':>8}  {'LSTM ADE':>9} {'KF ADE':>9} {'gain':>7}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in report["results"]:
            if not r.get("n"):
                continue
            print(f"  {r['subset']:<44}{r['n']:>8}  {r['LSTM_ADE_px']:>9.3f} "
                  f"{r['Kalman_ADE_px']:>9.3f} {r['ADE_improvement_pct']:>6.1f}%")
        print("=" * 78)
        j = report["results"][1]
        if j.get("n"):
            better = j["ADE_improvement_pct"] > 0
            print(f"  Verdict: on the sudden-jump subset the LSTM is "
                  f"{'BETTER' if better else 'NOT better'} than the Kalman "
                  f"baseline by {abs(j['ADE_improvement_pct']):.1f}% ADE.")
            print("  That subset, not the aggregate, is what the Social-LSTM "
                  "citation rests on.")

    out_dir = Path(out_dir or (ROOT / "runs" / "lstm_forecaster"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": report["config"]}, out_dir / "forecaster.pt")
    (out_dir / "forecast_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if verbose:
        print(f"\n  artifacts -> {out_dir}")
    return model, report


def main():
    ap = argparse.ArgumentParser(description="AeroTrack-Net trajectory forecaster")
    ap.add_argument("--obs", type=int, default=16)
    ap.add_argument("--pred", type=int, default=12)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None, help="cpu | cuda:0 (default: auto)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    train_forecaster(obs=a.obs, pred=a.pred, stride=a.stride, hidden=a.hidden,
                     layers=a.layers, epochs=a.epochs, batch=a.batch, lr=a.lr,
                     device=a.device, out_dir=a.out)


if __name__ == "__main__":
    main()
