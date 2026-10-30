"""Track C forecasting core: causal forecast samples, classical forecasters, metrics. No learning here (see tc_learned.py).

A forecast sample is an anchor frame t of one sequence at which the history source sees the target at t and t-1. Inputs are the
L most recent frames t-L+1..t (centre, size, visibility); targets are the ground-truth centres at t+1..t+H where visible.
Nothing at or after t+1 enters an input: `make_samples` builds histories from indices <= t only (tested).

Classical forecasters (all causal):
    cp          constant position: c_t
    cv          constant velocity from the last displacement: c_t + h (c_t - c_{t-1})
    mv{k}       mean velocity over the last k frames (earliest visible frame in [t-k, t-1])
    dcv         damped velocity: c_t + v_k sum_{i<=h} g^i  (k, g chosen on TRAIN)
    ca          constant acceleration through the last three frames (when t-2 is visible, else cv)
    ou          mean reversion to the image centre m: m + (c_t - m) rho^h (a gimbal-following camera re-centres the target)
    oudcv       ou plus a damped velocity term (rho, g, k chosen on TRAIN)
    kalman      constant-velocity Kalman filter per axis run over the WHOLE causal history (re-initialised after gaps longer than
                `max_gap`), noise proportional to the target size with a floor (STE convention); Gaussian forecasts. Its noise
                ratio is chosen on TRAIN for ADE and its covariance scale on TRAIN for NLL (scaling Q and R together leaves the
                mean unchanged), so the point forecast and the uncertainty are both tuned without VAL.
Metrics: per (sample, horizon) L2 error in native pixels, error / target size, misses (> 2 x size, > 20 px), Gaussian NLL and
ellipse coverage for probabilistic forecasters; aggregates per domain pooled over points; cluster (sequence) bootstrap for
paired differences.
"""
import numpy as np

L_HIST = 16
H_FUT = 24
REPORT_H = (1, 2, 4, 8, 12, 16, 24)
COVER_P = (0.5, 0.68, 0.9, 0.95)
CALIB_P = tuple(np.round(np.arange(0.1, 0.91, 0.1), 2))


def centres(box):
    return np.stack([box[:, 0] + box[:, 2] / 2, box[:, 1] + box[:, 3] / 2], 1)


def sizes(box):
    return np.maximum(box[:, 2], box[:, 3])


def make_samples(seqs, L=L_HIST, H=H_FUT, hist_source=None, stride=1):
    """seqs: tc_data.load_split records. hist_source: optional {key: (vis[T], box[T,4])} replacing the GT history (e.g. tracker
    outputs); targets are always the ground truth. -> dict of arrays (n samples)."""
    out = {k: [] for k in ("seq", "t", "hist_c", "hist_s", "hist_m", "fut_c", "fut_s", "fut_m", "W", "H", "domain")}
    for si, s in enumerate(seqs):
        gvis, gbox = s["vis"], s["box"]
        hvis, hbox = hist_source[s["key"]] if hist_source is not None else (gvis, gbox)
        T = len(gvis)
        gc, gs = centres(gbox), sizes(gbox)
        hc, hs = centres(hbox), sizes(hbox)
        t = np.arange(1, T, stride)
        fidx = t[:, None] + np.arange(1, H + 1)[None, :]
        fval = fidx < T
        fidx_c = np.minimum(fidx, T - 1)
        fm = fval & gvis[fidx_c]
        keep = hvis[t] & hvis[t - 1] & fm.any(1)
        t, fidx_c, fm = t[keep], fidx_c[keep], fm[keep]
        if not len(t):
            continue
        hidx = t[:, None] + np.arange(-L + 1, 1)[None, :]
        hval = hidx >= 0
        hidx_c = np.maximum(hidx, 0)
        assert (hidx_c <= t[:, None]).all()                      # causality: no history index after the anchor
        hm = hval & hvis[hidx_c]
        out["seq"].append(np.full(len(t), si))
        out["t"].append(t)
        out["hist_c"].append(np.where(hm[..., None], hc[hidx_c], 0.0))
        out["hist_s"].append(np.where(hm, hs[hidx_c], 0.0))
        out["hist_m"].append(hm)
        out["fut_c"].append(np.where(fm[..., None], gc[fidx_c], 0.0))
        out["fut_s"].append(np.where(fm, gs[fidx_c], 0.0))
        out["fut_m"].append(fm)
        out["W"].append(np.full(len(t), s["W"], np.float64))
        out["H"].append(np.full(len(t), s["H"], np.float64))
        out["domain"].append(np.array([s["domain"]] * len(t)))
    return {k: np.concatenate(v) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------------------- classical
def _vel_k(S, k):
    """Mean velocity over the last k frames: (c_t - c_j) / (t - j), j = earliest visible frame in [t-k, t-1]."""
    L = S["hist_m"].shape[1]
    win = S["hist_m"][:, L - 1 - k:L - 1]                     # frames t-k .. t-1
    first = np.argmax(win, axis=1)                            # earliest visible (t-1 is always visible -> exists)
    j = L - 1 - k + first
    lag = (L - 1 - j).astype(np.float64)
    cj = S["hist_c"][np.arange(len(j)), j]
    return (S["hist_c"][:, -1] - cj) / lag[:, None]


def forecast_classical(S, name, params=None):
    """-> mean [n, H, 2] (pixels)."""
    params = params or {}
    H = S["fut_c"].shape[1]
    h = np.arange(1, H + 1, dtype=np.float64)[None, :, None]
    c = S["hist_c"][:, -1][:, None, :]
    if name == "cp":
        return np.repeat(c, H, axis=1)
    if name == "cv":
        return c + h * (S["hist_c"][:, -1] - S["hist_c"][:, -2])[:, None, :]
    if name.startswith("mv"):
        return c + h * _vel_k(S, int(name[2:]))[:, None, :]
    if name == "dcv":
        g, k = params["g"], params["k"]
        w = np.cumsum(g ** np.arange(1, H + 1))[None, :, None]
        return c + w * _vel_k(S, k)[:, None, :]
    if name in ("ou", "oudcv"):                                   # mean reversion towards the image centre (gimbal re-centring)
        m = np.stack([S["W"] / 2, S["H"] / 2], 1)[:, None, :]
        rho = params["rho"] ** np.arange(1, H + 1)[None, :, None]
        base = m + (c - m) * rho
        if name == "ou":
            return base
        w = np.cumsum(params["g"] ** np.arange(1, H + 1))[None, :, None]
        return base + w * _vel_k(S, params["k"])[:, None, :]
    if name == "ca":
        c1, c2 = S["hist_c"][:, -2], S["hist_c"][:, -3]
        ok = S["hist_m"][:, -3][:, None]
        v = S["hist_c"][:, -1] - c1
        a = np.where(ok, S["hist_c"][:, -1] - 2 * c1 + c2, 0.0)
        return c + h * v[:, None, :] + (h * (h + 1) / 2) * a[:, None, :]
    raise ValueError(name)


class KalmanBank:
    """Constant-velocity Kalman filter per axis, vectorised over sequences; filtered state stored for every frame.

    Noise (STE convention): S = max(target size at the latest measurement, min_scale); Q = diag((q_pos S)^2, (q_vel S)^2),
    R = (r_meas S)^2; init P = diag((2 q_pos S)^2, (10 q_vel S)^2). A measurement after more than `max_gap` frames without one
    re-initialises the filter (velocity 0), as the STE drops a track after max_coast misses."""

    def __init__(self, q_pos=1 / 20, q_vel=1 / 160, r_meas=1 / 20, min_scale=10.0, max_gap=10, cov_scale=1.0):
        self.q_pos, self.q_vel, self.r_meas, self.min_scale, self.max_gap, self.cov_scale = q_pos, q_vel, r_meas, min_scale, max_gap, cov_scale

    def run(self, vis_list, box_list):
        """Filter every sequence; -> per sequence arrays p[T,2], v[T,2], P[T,2,3] (Pxx, Pxv, Pvv per axis), S[T]."""
        n = len(vis_list)
        Tm = max(len(v) for v in vis_list)
        Z = np.zeros((n, Tm, 2))
        M = np.zeros((n, Tm), bool)
        SZ = np.ones((n, Tm))
        for i, (v, b) in enumerate(zip(vis_list, box_list)):
            T = len(v)
            M[i, :T] = v
            cz = centres(np.nan_to_num(b))
            Z[i, :T] = cz
            SZ[i, :T] = np.where(v, sizes(np.nan_to_num(b)), 1.0)
        p = np.zeros((n, 2))
        vel = np.zeros((n, 2))
        Pxx = np.zeros((n, 2))
        Pxv = np.zeros((n, 2))
        Pvv = np.zeros((n, 2))
        S = np.full(n, self.min_scale)
        active = np.zeros(n, bool)
        gap = np.zeros(n, int)
        outp, outv, outP, outS = (np.zeros((n, Tm, 2)), np.zeros((n, Tm, 2)), np.zeros((n, Tm, 2, 3)), np.zeros((n, Tm)))
        qp, qv, rm = self.q_pos, self.q_vel, self.r_meas
        for t in range(Tm):
            # predict (active filters)
            a = active[:, None]
            qpp = (qp * S)[:, None] ** 2
            qvv = (qv * S)[:, None] ** 2
            p = np.where(a, p + vel, p)
            nPxx = Pxx + 2 * Pxv + Pvv + qpp
            nPxv = Pxv + Pvv
            nPvv = Pvv + qvv
            Pxx, Pxv, Pvv = np.where(a, nPxx, Pxx), np.where(a, nPxv, Pxv), np.where(a, nPvv, Pvv)
            gap = np.where(active, gap + 1, gap)
            m = M[:, t]
            S = np.where(m, np.maximum(SZ[:, t], self.min_scale), S)
            init = m & (~active | (gap > self.max_gap))
            upd = m & ~init
            # update
            r = (rm * S)[:, None] ** 2
            den = Pxx + r
            kx, kv = Pxx / den, Pxv / den
            innov = Z[:, t] - p
            u = upd[:, None]
            p = np.where(u, p + kx * innov, p)
            vel = np.where(u, vel + kv * innov, vel)
            uPxx, uPxv, uPvv = (1 - kx) * Pxx, (1 - kx) * Pxv, Pvv - kv * Pxv
            Pxx, Pxv, Pvv = np.where(u, uPxx, Pxx), np.where(u, uPxv, Pxv), np.where(u, uPvv, Pvv)
            # (re-)initialise
            i_ = init[:, None]
            p = np.where(i_, Z[:, t], p)
            vel = np.where(i_, 0.0, vel)
            Pxx = np.where(i_, ((2 * qp * S)[:, None]) ** 2, Pxx)
            Pxv = np.where(i_, 0.0, Pxv)
            Pvv = np.where(i_, ((10 * qv * S)[:, None]) ** 2, Pvv)
            active = active | init
            gap = np.where(m, 0, gap)
            outp[:, t], outv[:, t], outS[:, t] = p, vel, S
            outP[:, t, :, 0], outP[:, t, :, 1], outP[:, t, :, 2] = Pxx, Pxv, Pvv
        lens = [len(v) for v in vis_list]
        return [(outp[i, :T], outv[i, :T], outP[i, :T], outS[i, :T]) for i, T in enumerate(lens)]

    def forecast(self, states, S_, H=H_FUT):
        """states: per sample (p[2], v[2], P[2,3], S) gathered at the anchor -> mean [n,H,2], var [n,H,2] (diagonal: axes independent)."""
        p, v, P, S = states
        qpp = (self.q_pos * S)[:, None] ** 2
        qvv = (self.q_vel * S)[:, None] ** 2
        Pxx, Pxv, Pvv = P[..., 0].copy(), P[..., 1].copy(), P[..., 2].copy()
        mean = np.zeros((len(p), H, 2))
        var = np.zeros((len(p), H, 2))
        for h in range(H):
            Pxx, Pxv, Pvv = Pxx + 2 * Pxv + Pvv + qpp, Pxv + Pvv, Pvv + qvv
            mean[:, h] = p + (h + 1) * v
            var[:, h] = Pxx
        return mean, var * self.cov_scale


def kalman_forecast(seqs, S, kb, hist_source=None):
    vis = [hist_source[s["key"]][0] if hist_source else s["vis"] for s in seqs]
    box = [hist_source[s["key"]][1] if hist_source else s["box"] for s in seqs]
    runs = kb.run(vis, box)
    off = np.cumsum([0] + [len(r[3]) for r in runs])
    flat = off[S["seq"]] + S["t"]                                  # state AFTER processing the anchor frame (causal)
    p, v, P, Sc = (np.concatenate([r[j] for r in runs])[flat] for j in range(4))
    return kb.forecast((p, v, P, Sc), Sc, S["fut_c"].shape[1])


# ----------------------------------------------------------------------------------------------------------- metrics
def chi2_2(p):
    return -2.0 * np.log(1.0 - np.asarray(p, dtype=np.float64))


def point_errors(S, mean):
    """-> err [n,H] px (NaN where the target is not visible at t+h)."""
    e = np.linalg.norm(mean - S["fut_c"], axis=-1)
    return np.where(S["fut_m"], e, np.nan)


def gauss_terms(S, mean, var):
    """Diagonal Gaussian: Mahalanobis d2 [n,H] and NLL [n,H] (nats, per 2-D point), NaN where not visible."""
    var = np.maximum(var, 1e-6)
    d = mean - S["fut_c"]
    d2 = (d ** 2 / var).sum(-1)
    nll = 0.5 * (d2 + np.log(var).sum(-1) + 2 * np.log(2 * np.pi))
    return np.where(S["fut_m"], d2, np.nan), np.where(S["fut_m"], nll, np.nan)


def summarize(S, err, d2=None, nll=None, mask=None):
    """Pooled over visible (sample, horizon) points; mask [n] selects samples."""
    m = np.ones(len(err), bool) if mask is None else mask
    e, fm, fs = err[m], S["fut_m"][m], S["fut_s"][m]
    if not fm.any():
        return None
    out = {"n_samples": int(m.sum()), "n_points": int(fm.sum()), "ade_px": float(np.nanmean(e)),
           "ade_rel_size": float(np.nanmean(e / np.where(fm, np.maximum(fs, 1.0), np.nan))),
           "miss_2x_size": float(np.nanmean(np.where(fm, e > 2 * np.maximum(fs, 1.0), np.nan))),
           "miss_20px": float(np.nanmean(np.where(fm, e > 20, np.nan))), "by_h": {}}
    for h in REPORT_H:
        col = e[:, h - 1]
        if np.isfinite(col).any():
            out["by_h"][h] = {"mean_px": float(np.nanmean(col)), "median_px": float(np.nanmedian(col)), "n": int(np.isfinite(col).sum())}
    out["fde_px"] = out["by_h"].get(max(REPORT_H), {}).get("mean_px")
    if d2 is not None:
        dd, nn = d2[m], nll[m]
        out["nll"] = float(np.nanmean(nn))
        out["coverage"] = {str(p): float(np.nanmean(np.where(fm, dd <= chi2_2(p), np.nan))) for p in COVER_P}
        cal = [abs(float(np.nanmean(np.where(fm, dd <= chi2_2(p), np.nan))) - p) for p in CALIB_P]
        out["calibration_error"] = float(np.mean(cal))
    return out


def seq_sums(S, err, nll=None):
    """Per-sequence sums for the cluster bootstrap: {seq: (sum_err, n_points, sum_nll)}."""
    out = {}
    ok = np.isfinite(err)
    se = np.where(ok, err, 0).sum(1)
    n = ok.sum(1)
    sn = np.where(ok, nll, 0).sum(1) if nll is not None else np.zeros(len(err))
    for q in np.unique(S["seq"]):
        m = S["seq"] == q
        out[int(q)] = (float(se[m].sum()), int(n[m].sum()), float(sn[m].sum()))
    return out


def paired_bootstrap(sums_a, sums_b, seq_domain, n_boot=2000, seed=0, stat="ade"):
    """Cluster bootstrap over sequences, stratified by domain: pooled(a) - pooled(b) and the relative change; 95 % CI."""
    rng = np.random.default_rng(seed)
    keys = sorted(set(sums_a) & set(sums_b))
    idx = 0 if stat == "ade" else 2
    A = np.array([[sums_a[k][idx], sums_a[k][1]] for k in keys])
    B = np.array([[sums_b[k][idx], sums_b[k][1]] for k in keys])
    doms = np.array([seq_domain[k] for k in keys])
    strata = [np.flatnonzero(doms == d) for d in np.unique(doms)]

    def pooled(X, ix):
        return X[ix, 0].sum() / max(X[ix, 1].sum(), 1)

    full = np.arange(len(keys))
    da, db = pooled(A, full), pooled(B, full)
    diffs, rels = [], []
    for _ in range(n_boot):
        ix = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        a, b = pooled(A, ix), pooled(B, ix)
        diffs.append(a - b)
        rels.append((a - b) / b if b else 0.0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rlo, rhi = np.percentile(rels, [2.5, 97.5])
    return {"a": float(da), "b": float(db), "diff": float(da - db), "ci95": [float(lo), float(hi)],
            "rel": float((da - db) / db) if db else None, "rel_ci95": [float(rlo), float(rhi)], "n_sequences": len(keys)}
