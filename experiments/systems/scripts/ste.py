"""AeroTrack single-target state estimator (STE): per-frame detections -> one target state, its existence, and its future.

Why single-target: every Anti-UAV / CST sequence holds one UAV, and the repair series measured that a multi-object tracker
(ByteTrack) is beaten by a per-frame argmax (IDF1 0.31 vs 0.86) on this data. What a single-target layer can still add over
argmax is temporal reasoning about ONE state: reject detections that are implausible given the motion (gating), bridge
missed detections (coasting), recover after a loss (re-acquisition), decide existence, and forecast.

    state   constant-velocity Kalman filter on (cx, cy, w, h); process / measurement noise proportional to the target size
            (SORT / ByteTrack convention) with a floor `min_scale` px so few-pixel targets do not get sub-pixel noise
    gate    Mahalanobis distance of a detection centre to the predicted centre (2 dof); `gate_chi2` = 9.21 is the 99 % quantile
    per frame:
      track active : predict -> best gated detection (>= det_thr) updates the state;
                     no gated detection but one >= reacq_thr anywhere -> re-acquire there (jump);
                     else coast: misses += 1; after `max_coast` misses the target is declared absent and the track ends
      no track     : best detection >= init_thr starts a track
    output  the associated detection box (`output="det"`) or the filtered box (`"kf"`); while coasting the predicted box
            (`coast_output=True`) or "absent"
    forecast  h-step mean and covariance of the centre from the current state (for prediction / uncertainty)

All thresholds are hyper-parameters chosen on VAL (track_eval.py cross-fits them); nothing here reads ground truth.
"""
from dataclasses import asdict, dataclass

import numpy as np

CHI2_2D_99 = 9.2103


def xyxy_to_cxcywh(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], dtype=np.float64)


def cxcywh_to_xyxy(s):
    cx, cy, w, h = s[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


class KalmanCV:
    _F = np.eye(8)
    _F[:4, 4:] = np.eye(4)
    _H = np.eye(4, 8)

    def __init__(self, box_xyxy, q_pos=1 / 20, q_vel=1 / 160, r_meas=1 / 20, min_scale=10.0):
        self.q_pos, self.q_vel, self.r_meas, self.min_scale = q_pos, q_vel, r_meas, min_scale
        z = xyxy_to_cxcywh(box_xyxy)
        self.x = np.concatenate([z, np.zeros(4)])
        s = self._scale()
        self.P = np.diag(np.square([2 * q_pos * s] * 4 + [10 * q_vel * s] * 4))

    def _scale(self):
        return max(float(self.x[2]), float(self.x[3]), self.min_scale)

    def _R(self):
        return np.diag(np.square([self.r_meas * self._scale()] * 4))

    def predict(self):
        s = self._scale()
        Q = np.diag(np.square([self.q_pos * s] * 4 + [self.q_vel * s] * 4))
        self.x = self._F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.P = self._F @ self.P @ self._F.T + Q

    def gate_d2(self, centers):
        """Squared Mahalanobis distance of candidate centres [n, 2] to the predicted centre."""
        S = (self._H @ self.P @ self._H.T + self._R())[:2, :2]
        d = np.asarray(centers, dtype=np.float64) - self.x[:2]
        return np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d)

    def update(self, box_xyxy):
        z = xyxy_to_cxcywh(box_xyxy)
        S = self._H @ self.P @ self._H.T + self._R()
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self._H @ self.x)
        self.P = (np.eye(8) - K @ self._H) @ self.P

    def box(self):
        return cxcywh_to_xyxy(self.x)

    def forecast(self, horizon):
        """Centre mean [h, 2] and covariance [h, 2, 2] for steps 1..horizon (no measurement updates)."""
        x, P = self.x.copy(), self.P.copy()
        means, covs = [], []
        for _ in range(horizon):
            s = max(float(x[2]), float(x[3]), self.min_scale)
            Q = np.diag(np.square([self.q_pos * s] * 4 + [self.q_vel * s] * 4))
            x = self._F @ x
            P = self._F @ P @ self._F.T + Q
            means.append(x[:2].copy())
            covs.append(P[:2, :2].copy())
        return np.array(means), np.array(covs)


@dataclass
class STEParams:
    det_thr: float = 0.10
    init_thr: float = 0.40
    reacq_thr: float = 0.60
    gate_chi2: float = CHI2_2D_99
    max_coast: int = 10
    output: str = "det"            # "det" | "kf"
    coast_output: bool = True
    min_scale: float = 10.0
    q_pos: float = 1 / 20
    q_vel: float = 1 / 160
    r_meas: float = 1 / 20

    def as_dict(self):
        return asdict(self)


class STE:
    MODES = ("absent", "track", "coast", "reacquired", "init")

    def __init__(self, params: STEParams):
        self.p = params
        self.kf = None
        self.misses = 0
        self.n_reacq = 0

    def reset(self, init_box_xyxy=None):
        """Start a sequence; with an init box (single-object-tracking protocol) the state starts there."""
        self.kf = None if init_box_xyxy is None else self._new(init_box_xyxy)
        self.misses = 0

    def _new(self, box):
        p = self.p
        return KalmanCV(box, p.q_pos, p.q_vel, p.r_meas, p.min_scale)

    def step(self, boxes, scores, classes):
        """One frame. boxes [n, 4] xyxy (original pixels), scores [n], classes [n]. -> (box xyxy or None, mode, score)."""
        p = self.p
        keep = (classes == 0) & (scores >= p.det_thr)
        boxes, scores = boxes[keep], scores[keep]
        best = int(np.argmax(scores)) if len(scores) else -1
        if self.kf is None:
            if best >= 0 and scores[best] >= p.init_thr:
                self.kf = self._new(boxes[best])
                self.misses = 0
                return boxes[best].copy(), "init", float(scores[best])
            return None, "absent", 0.0
        self.kf.predict()
        if len(scores):
            centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
            inside = self.kf.gate_d2(centers) <= p.gate_chi2
            if inside.any():
                j = int(np.flatnonzero(inside)[np.argmax(scores[inside])])
                self.kf.update(boxes[j])
                self.misses = 0
                return (boxes[j].copy() if p.output == "det" else self.kf.box()), "track", float(scores[j])
            if scores[best] >= p.reacq_thr:
                self.kf = self._new(boxes[best])
                self.misses = 0
                self.n_reacq += 1
                return boxes[best].copy(), "reacquired", float(scores[best])
        self.misses += 1
        if self.misses > p.max_coast:
            self.kf = None
            return None, "absent", 0.0
        return (self.kf.box() if p.coast_output else None), "coast", 0.0


def argmax_step(boxes, scores, classes, thr):
    """The repair series' strongest 'tracker': the best drone detection of the frame, if it clears `thr`."""
    keep = (classes == 0) & (scores >= thr)
    if not keep.any():
        return None
    j = np.flatnonzero(keep)[np.argmax(scores[keep])]
    return boxes[j].copy()
