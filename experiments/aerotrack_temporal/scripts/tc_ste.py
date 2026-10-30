"""Forecaster-driven single-target state estimator (FSTE) -- Track C, experiment C3.

Same decision logic as experiments/systems/scripts/ste.py (gate -> associate, else re-acquire, else coast, absent after max_coast
misses; init above init_thr), but the prediction used for gating and coasting comes from a FORECASTER fed with the tracker's own
history of associated detections (causal: frames <= the latest association), instead of a constant-velocity Kalman filter:

    k = frames since the latest association; prediction = that association's forecast at horizon k
        (beyond the forecast horizon: the last mean, variance x k / H)
    gate: d2 = sum_axes (c_det - mu_k)^2 / (var_k + (r_meas * max(size_det, min_scale))^2) <= gate_chi2
    output: the associated detection box; while coasting the predicted box (last associated size) if coast_output

Forecasters (callables hist_c[L,2], hist_s[L], hist_m[L], W, H -> mean[H,2], var[H,2] in pixels):
    ClassicalForecaster   tc_forecast's classical family evaluated on ONE history, robust to gaps (a velocity needs an
                          associated frame in its look-back window, else 0), with a fixed per-horizon variance table
                          (TRAIN residuals of that forecaster on oracle histories)
    LearnedForecaster     a C1 checkpoint exported to ONNX (CPU); used when the last two frames are associations (its training
                          condition), else its classical fallback
"""
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

L_HIST = 16
H_FUT = 24


@dataclass
class FSTEParams:
    det_thr: float = 0.10
    init_thr: float = 0.40
    reacq_thr: float = 0.60
    max_coast: int = 10
    coast_output: bool = True
    gate_chi2: float = 9.2103
    r_meas: float = 1 / 20
    min_scale: float = 10.0

    def as_dict(self):
        return asdict(self)


def _centre(b):
    b = np.asarray(b, dtype=np.float64)
    return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)


def _size(b):
    b = np.asarray(b, dtype=np.float64)
    return np.maximum(b[..., 2] - b[..., 0], b[..., 3] - b[..., 1])


def _vel(hc, hm, k):
    """Mean velocity over the last k frames from the earliest associated frame in [t-k, t-1]; 0 if none (gap-robust)."""
    L = len(hm)
    win = np.flatnonzero(hm[L - 1 - k:L - 1])
    if not len(win):
        return np.zeros(2)
    j = L - 1 - k + win[0]
    return (hc[-1] - hc[j]) / (L - 1 - j)


def classical_mean(name, params, hc, hm, W, H, horizon=H_FUT):
    """One history -> mean [horizon, 2]; equals tc_forecast.forecast_classical whenever t and t-1 are both associated."""
    params = params or {}
    h = np.arange(1, horizon + 1, dtype=np.float64)[:, None]
    c = hc[-1][None, :]
    if name == "cp":
        return np.repeat(c, horizon, 0)
    if name == "cv":
        return c + h * _vel(hc, hm, 1)[None, :]
    if name.startswith("mv"):
        return c + h * _vel(hc, hm, int(name[2:]))[None, :]
    if name in ("dcv", "ou", "oudcv"):
        if name == "dcv":
            base = c
        else:
            m = np.array([[W / 2, H / 2]])
            base = m + (c - m) * params["rho"] ** h
        if name == "ou":
            return base
        w = np.cumsum(params["g"] ** np.arange(1, horizon + 1))[:, None]
        return base + w * _vel(hc, hm, params["k"])[None, :]
    if name == "ca":
        if hm[-2] and hm[-3]:
            v = hc[-1] - hc[-2]
            a = hc[-1] - 2 * hc[-2] + hc[-3]
            return c + h * v[None, :] + (h * (h + 1) / 2) * a[None, :]
        return c + h * _vel(hc, hm, 1)[None, :]
    raise ValueError(name)


class ClassicalForecaster:
    def __init__(self, name, params, var_table):
        self.name, self.params = name, params
        self.var = np.asarray(var_table, dtype=np.float64)          # [H, 2] pixel^2

    def __call__(self, hc, hs, hm, W, H):
        return classical_mean(self.name, self.params, hc, hm, W, H, len(self.var)), self.var.copy()


class LearnedForecaster:
    """ONNX export of a tc_learned model (inputs seq [1, L, 4], glob [1, 3 | 1]); `fallback` when its input condition fails."""

    def __init__(self, onnx_path, abs_pos, fallback, dom_onehot=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.abs_pos, self.fallback = abs_pos, fallback
        self.dom_onehot = list(dom_onehot) if dom_onehot is not None else []        # C1c camera-mode token (tc_learned.DOMAIN_TOKENS)
        self.calls = self.fallbacks = 0

    def __call__(self, hc, hs, hm, W, H):
        if not (hm[-1] and hm[-2]):
            self.fallbacks += 1
            return self.fallback(hc, hs, hm, W, H)
        self.calls += 1
        scale = W / 100.0
        ct, st = hc[-1], max(float(hs[-1]), 1.0)
        m = hm.astype(np.float32)
        d = (hc - ct) / scale
        ls = np.log(np.maximum(hs, 1.0) / st)
        seq = np.concatenate([m[:, None], d * m[:, None], (ls * m)[:, None]], 1).astype(np.float32)[None]
        g = [np.log(st / scale)] + ([ct[0] / W - 0.5, ct[1] / H - 0.5] if self.abs_pos else []) + self.dom_onehot
        glob = np.array([g], dtype=np.float32)
        mu, logs = self.sess.run(None, {"seq": seq, "glob": glob})
        return ct[None, :] + mu[0].astype(np.float64) * scale, np.exp(2 * logs[0].astype(np.float64)) * scale ** 2


class FSTE:
    def __init__(self, params, forecaster, W, H, L=L_HIST):
        self.p, self.f, self.W, self.H, self.L = params, forecaster, W, H, L

    def reset(self, box_xyxy=None):
        self.t = 0
        self.hist = deque(maxlen=self.L)            # (t, centre, size)
        self.active = False
        self.misses = 0
        self.F = None
        if box_xyxy is not None:
            self._associate(np.asarray(box_xyxy, dtype=np.float64), restart=True)

    def _arrays(self):
        hc = np.zeros((self.L, 2))
        hs = np.zeros(self.L)
        hm = np.zeros(self.L, bool)
        for (tt, c, s) in self.hist:
            i = self.L - 1 - (self.t - tt)
            if i >= 0:
                hc[i], hs[i], hm[i] = c, s, True
        return hc, hs, hm

    def _associate(self, box, restart=False):
        if restart:
            self.hist.clear()
        self.hist.append((self.t, _centre(box), float(_size(box))))
        self.wh = (float(box[2] - box[0]), float(box[3] - box[1]))
        self.t_anchor = self.t
        self.active = True
        self.misses = 0
        self.F = self.f(*self._arrays(), self.W, self.H)

    def _prediction(self):
        k = self.t - self.t_anchor
        mean, var = self.F
        Hn = len(mean)
        if k <= Hn:
            return mean[k - 1], var[k - 1]
        return mean[-1], var[-1] * k / Hn

    def step(self, boxes, scores, classes):
        """One frame of detections (xyxy [n,4], scores [n], classes [n]; class 0 = drone) -> (box or None, mode, info)."""
        self.t += 1
        p = self.p
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        drone = np.asarray(classes).reshape(-1) == 0
        if not self.active:
            cand = np.flatnonzero(drone & (scores >= p.init_thr))
            if len(cand):
                i = cand[np.argmax(scores[cand])]
                self._associate(boxes[i], restart=True)
                return boxes[i], "init", {}
            return None, "absent", {}
        mu, var = self._prediction()
        cand = np.flatnonzero(drone & (scores >= p.det_thr))
        if len(cand):
            c = _centre(boxes[cand])
            r2 = (p.r_meas * np.maximum(_size(boxes[cand]), p.min_scale)) ** 2
            d2 = ((c - mu) ** 2 / (var[None, :] + r2[:, None])).sum(1)
            ok = cand[d2 <= p.gate_chi2]
            if len(ok):
                i = ok[np.argmax(scores[ok])]
                self._associate(boxes[i])
                return boxes[i], "track", {"d2": float(d2[list(cand).index(i)])}
        far = np.flatnonzero(drone & (scores >= p.reacq_thr))
        if len(far):
            i = far[np.argmax(scores[far])]
            self._associate(boxes[i], restart=True)
            return boxes[i], "reacquired", {}
        self.misses += 1
        if self.misses > p.max_coast:
            self.active = False
            self.hist.clear()
            return None, "absent", {}
        if p.coast_output:
            w, h = self.wh
            return np.array([mu[0] - w / 2, mu[1] - h / 2, mu[0] + w / 2, mu[1] + h / 2]), "coast", {}
        return None, "coast", {}
