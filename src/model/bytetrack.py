#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bytetrack.py  --  ByteTrack association for AeroTrack-Net
=========================================================
Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection
Box*, ECCV 2022 -- the third of the four papers Review 1 adopted, and the one
that answers a number we measured rather than a number we read.

The number
----------
`xfactor_analysis.json` says **16.9 % of CST frames are out-of-view**, spread
over **494 occlusion events with a mean length of 55.6 frames and a maximum of
841**, and **78 % of CST sequences contain at least one**. A detector alone
cannot survive that: through the gap its confidence collapses, and a tracker
that keeps only high-confidence boxes drops the tracklet and issues a NEW id on
re-entry. Every one of those is an identity switch, and a counter-UAS system
that renumbers the threat mid-engagement has lost the engagement.

ByteTrack's claim is that the low-confidence boxes are not noise -- they are the
occluded object. So association runs twice:

  1. **High** boxes (score >= `track_thresh`) are matched to all active
     tracklets by IoU.
  2. **Low** boxes (`low_thresh` <= score < `track_thresh`) are matched to the
     tracklets that stage 1 left unmatched -- and *only* to those. A low box
     never starts a track; it can only sustain one that already exists.

That asymmetry is the whole idea, and it is why this file exists instead of a
call to a stock SORT: SORT thresholds once, at `track_thresh`, and throws the
rest away.

What is deliberately absent
---------------------------
* **Re-ID appearance embeddings.** The median CST target is a 9x9 px square with
  no colour (thermal). There is no appearance to embed; an embedding network
  would be fitting noise. Motion is the only usable cue, which is the same
  conclusion that put TrackNet in the detector.
* **Social pooling / interaction terms.** One drone per sequence in this corpus.

Also here: the CLEAR-MOT + IDF1 metric suite from 00_TRAINING_ROADMAP.md §7.3,
including **ID switches across occlusion events specifically**, because that is
the quantity ByteTrack is supposed to move and the aggregate MOTA can hide it.

Usage
-----
    from bytetrack import BYTETracker, evaluate_tracking

    tr = BYTETracker(frame_rate=30)
    for det in per_frame_detections:          # [n,5] xyxy+score, or [n,6] +cls
        tracks = tr.update(det)               # [m,6] xyxy + track_id

    print(evaluate_tracking(pred_by_frame, gt_by_frame))
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["KalmanBoxFilter", "Track", "BYTETracker", "evaluate_tracking",
           "iou_batch"]


# --------------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------------- #
def iou_batch(a, b):
    """[n,4] x [m,4] xyxy -> [n,m] IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def iou_buffered(a, b, buf=1.0):
    """IoU after expanding every box by `buf` x its own mean side.

    **This replaces plain IoU, and the reason is measured.** ByteTrack's stock
    association gate is IoU >= 0.8, which is sane for a 150 px pedestrian moving
    a few pixels between frames. Our median target is **9x9 px** and Anti-UAV's
    median inter-frame motion is **5.10 px**. Two consecutive 9 px boxes offset
    by 5 px overlap at IoU ~= 0.30 -- so a stock ByteTrack rejects the correct
    match on the *typical* frame of this corpus, drops the tracklet, and mints a
    new id. Measured on a synthetic 9 px target moving 4.3 px/frame: 200 frames
    of perfect detections produced 140 tracks and 198 false negatives.

    Buffering fixes it in the units that matter. Expanding both boxes by their
    own size makes the gate scale-relative: a 9 px target tolerates ~9 px of
    motion, a 90 px one ~90 px, with no second threshold. This is the same
    device as BIoU in the small-object tracking literature, and it is why the
    default thresholds here are nothing like the paper's.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    def _pad(x):
        x = np.asarray(x, dtype=np.float32).copy()
        m = buf * 0.5 * ((x[:, 2] - x[:, 0]) + (x[:, 3] - x[:, 1])) / 2.0
        x[:, 0] -= m; x[:, 1] -= m; x[:, 2] += m; x[:, 3] += m
        return x
    return iou_batch(_pad(a), _pad(b))


def _xyxy_to_xyah(box):
    """xyxy -> (centre x, centre y, aspect w/h, height). The Kalman state form."""
    x1, y1, x2, y2 = box[:4]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h], dtype=np.float64)


def _xyah_to_xyxy(s):
    cx, cy, a, h = s[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Kalman filter (constant velocity, 8-dim)
# --------------------------------------------------------------------------- #
class KalmanBoxFilter:
    """Constant-velocity Kalman filter on (cx, cy, aspect, h) + velocities.

    This is the SORT/ByteTrack filter, kept because it is the right tool for
    the ASSOCIATION gate -- one frame ahead, where constant velocity is a fine
    approximation and the covariance is what matters.

    It is emphatically NOT the right tool for multi-step FORECASTING on this
    corpus: `tracking_metrics.json` records **32,096 sudden jumps over 30
    px/frame**, and corr(speed, area) ~ 0, i.e. the motion is non-linear by
    measurement. That is what `lstm_forecaster.py` is for, and this class is
    the baseline it is measured against.

    Noise scales are position/velocity standard deviations expressed as a
    fraction of target height -- so a 9 px target gets a tight gate and a 90 px
    one a loose gate, without a second set of constants.
    """

    def __init__(self, std_weight_position=1.0 / 20, std_weight_velocity=1.0 / 160):
        ndim, dt = 4, 1.0
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim)
        self._sp = std_weight_position
        self._sv = std_weight_velocity

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros(4)]
        h = measurement[3]
        std = np.array([2 * self._sp * h, 2 * self._sp * h, 1e-2, 2 * self._sp * h,
                        10 * self._sv * h, 10 * self._sv * h, 1e-5, 10 * self._sv * h])
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = np.array([self._sp * h, self._sp * h, 1e-2, self._sp * h,
                        self._sv * h, self._sv * h, 1e-5, self._sv * h])
        Q = np.diag(np.square(std))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def update(self, mean, cov, measurement):
        h = mean[3]
        std = np.array([self._sp * h, self._sp * h, 1e-1, self._sp * h])
        R = np.diag(np.square(std))
        S = self._H @ cov @ self._H.T + R
        K = cov @ self._H.T @ np.linalg.inv(S)
        y = measurement - self._H @ mean
        mean = mean + K @ y
        cov = (np.eye(len(mean)) - K @ self._H) @ cov
        return mean, cov


# --------------------------------------------------------------------------- #
#  Tracklet
# --------------------------------------------------------------------------- #
NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


class Track:
    _count = 0

    def __init__(self, box, score, cls, kf, frame_id):
        Track._count += 1
        self.track_id = Track._count
        self.kf = kf
        self.mean, self.cov = kf.initiate(_xyxy_to_xyah(box))
        self.score = float(score)
        self.cls = int(cls)
        self.state = NEW
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.hits = 1
        self.age = 0
        # Frames sustained by a LOW-confidence detection only. This is the
        # quantity ByteTrack exists to make non-zero -- report it, do not just
        # trust that the mechanism fired.
        self.low_hits = 0
        self.history = [(frame_id, _xyah_to_xyxy(self.mean))]

    @property
    def tlbr(self):
        return _xyah_to_xyxy(self.mean)

    def predict(self):
        if self.state != TRACKED:
            self.mean[7] = 0.0          # a lost track does not grow or shrink
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1

    def update(self, box, score, frame_id, low=False):
        self.mean, self.cov = self.kf.update(self.mean, self.cov,
                                             _xyxy_to_xyah(box))
        self.state = TRACKED
        self.score = float(score)
        self.frame_id = frame_id
        self.hits += 1
        self.age = 0
        if low:
            self.low_hits += 1
        self.history.append((frame_id, _xyah_to_xyxy(self.mean)))

    def mark_lost(self):
        self.state = LOST

    def mark_removed(self):
        self.state = REMOVED


# --------------------------------------------------------------------------- #
#  The tracker
# --------------------------------------------------------------------------- #
def _match(cost, thresh):
    """Hungarian assignment on a cost matrix, rejecting pairs above `thresh`."""
    if cost.size == 0:
        return (np.empty((0, 2), dtype=int), np.arange(cost.shape[0]),
                np.arange(cost.shape[1]))
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] <= thresh
    pairs = np.stack([r[keep], c[keep]], 1) if keep.any() else np.empty((0, 2), int)
    un_a = np.setdiff1d(np.arange(cost.shape[0]), pairs[:, 0])
    un_b = np.setdiff1d(np.arange(cost.shape[1]), pairs[:, 1])
    return pairs, un_a, un_b


class BYTETracker:
    """ByteTrack, tuned against this corpus's measured occlusion statistics.

    `track_buffer` is the one parameter worth arguing about. The stock value is
    30 frames. CST's mean out-of-view event is **55.6 frames** and its longest
    is **841**, so 30 guarantees an id switch on the average occlusion -- the
    exact failure the tracker was added to prevent. The default here is 90
    frames (3 s at 30 fps), which covers the mean with margin; carrying the
    841-frame tail would mean holding a tracklet for 28 seconds, which in a
    counter-UAS context is a stale track, not a robust one. That trade-off is a
    choice, so it is a parameter and it is written down.
    """

    def __init__(self, track_thresh=0.5, low_thresh=0.1, match_thresh=0.35,
                 second_match_thresh=0.2, track_buffer=90, min_hits=2,
                 frame_rate=30, buffer_ratio=1.0, lost_buffer_ratio=2.0):
        self.track_thresh = float(track_thresh)
        self.low_thresh = float(low_thresh)
        # Gates are on BUFFERED IoU (see iou_buffered), so these numbers are not
        # comparable to the paper's 0.8 -- they are looser on purpose because
        # the buffering already did the size normalisation.
        self.match_thresh = float(match_thresh)
        self.second_match_thresh = float(second_match_thresh)
        self.buffer_ratio = float(buffer_ratio)
        # A track that has been coasting through an occlusion has an uncertain
        # position, so it gets a wider gate on re-entry. Growing it with age
        # would be better still; a flat two-step split is enough to reacquire
        # after CST's mean 55.6-frame gap without inviting spurious matches on
        # frame 2.
        self.lost_buffer_ratio = float(lost_buffer_ratio)
        self.max_age = int(track_buffer * frame_rate / 30.0)
        self.min_hits = int(min_hits)
        self.kf = KalmanBoxFilter()
        self.tracked, self.lost = [], []
        self.frame_id = 0
        self.stats = {"low_assoc": 0, "reactivations": 0, "new_tracks": 0}

    def reset(self):
        self.tracked, self.lost = [], []
        self.frame_id = 0
        self.stats = {k: 0 for k in self.stats}
        Track._count = 0

    def _gate(self, tracks, boxes):
        """Buffered-IoU affinity, with a wider buffer for coasting tracklets."""
        if not tracks or len(boxes) == 0:
            return np.zeros((len(tracks), len(boxes)), dtype=np.float32)
        tl = np.stack([t.tlbr for t in tracks])
        m = np.zeros((len(tracks), len(boxes)), dtype=np.float32)
        for ratio in (self.buffer_ratio, self.lost_buffer_ratio):
            sel = np.array([(t.state == LOST) == (ratio == self.lost_buffer_ratio)
                            for t in tracks])
            if sel.any():
                m[sel] = iou_buffered(tl[sel], boxes, ratio)
        return m

    def update(self, dets):
        """dets: [n,5] xyxy+score, or [n,6] xyxy+score+cls. -> [m,6] xyxy+id.

        An EMPTY array is a legitimate input, not a missing one: it is the
        `exist = 0` frame the pipeline preserves as a 0-byte label. The tracker
        must coast through it on prediction alone, and that is exactly what
        happens here -- every track is predicted, none is updated, and the age
        counter decides when to give up.
        """
        self.frame_id += 1
        dets = (np.zeros((0, 6), dtype=np.float32) if dets is None or len(dets) == 0
                else np.asarray(dets, dtype=np.float32))
        if dets.shape[1] == 5:
            dets = np.c_[dets, np.zeros(len(dets), dtype=np.float32)]

        scores = dets[:, 4]
        high = dets[scores >= self.track_thresh]
        low = dets[(scores < self.track_thresh) & (scores >= self.low_thresh)]

        pool = self.tracked + self.lost
        for t in pool:
            t.predict()

        # ---- stage 1: high-confidence boxes against every tracklet -------- #
        cost = 1.0 - self._gate(pool, high[:, :4])
        pairs, un_trk, un_det = _match(cost, 1.0 - self.match_thresh)
        for ti, di in pairs:
            t = pool[ti]
            if t.state == LOST:
                self.stats["reactivations"] += 1
            t.update(high[di, :4], high[di, 4], self.frame_id, low=False)

        # ---- stage 2: THE ByteTrack step ---------------------------------- #
        # Only the tracklets stage 1 could not match, and only against the LOW
        # boxes. A low box may sustain an existing identity; it may never mint a
        # new one, because at 0.1 confidence a new track is as likely to be a
        # cloud edge as a drone.
        rest = [pool[i] for i in un_trk]
        cost2 = 1.0 - self._gate(rest, low[:, :4])
        pairs2, un_trk2, _ = _match(cost2, 1.0 - self.second_match_thresh)
        for ti, di in pairs2:
            t = rest[ti]
            if t.state == LOST:
                self.stats["reactivations"] += 1
            t.update(low[di, :4], low[di, 4], self.frame_id, low=True)
            self.stats["low_assoc"] += 1

        still_unmatched = [rest[i] for i in un_trk2]

        # ---- births ------------------------------------------------------- #
        for di in un_det:
            if high[di, 4] < self.track_thresh:
                continue
            self.tracked.append(Track(high[di, :4], high[di, 4], high[di, 5],
                                      self.kf, self.frame_id))
            self.stats["new_tracks"] += 1

        # ---- deaths ------------------------------------------------------- #
        for t in still_unmatched:
            t.mark_lost()
            if t.age > self.max_age:
                t.mark_removed()

        alive = [t for t in (self.tracked + self.lost) if t.state != REMOVED]
        self.tracked = [t for t in alive if t.state in (NEW, TRACKED)]
        self.lost = [t for t in alive if t.state == LOST]

        out = []
        for t in self.tracked:
            if t.hits < self.min_hits and self.frame_id > self.min_hits:
                continue
            b = t.tlbr
            out.append([b[0], b[1], b[2], b[3], t.track_id, t.score])
        return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 6), np.float32)


# --------------------------------------------------------------------------- #
#  CLEAR-MOT + IDF1  (roadmap §7.3)
# --------------------------------------------------------------------------- #
def evaluate_tracking(pred_by_frame, gt_by_frame, iou_thresh=0.5,
                      occlusion_frames=None):
    """Tracking metrics over ONE sequence.

    pred_by_frame : {frame_idx: [n,5] xyxy + track_id}
    gt_by_frame   : {frame_idx: [m,5] xyxy + gt_id}   (absent/empty = exist 0)
    occlusion_frames : optional set of frame indices where the target was
                    out of view. ID switches are then reported BOTH overall and
                    restricted to the frames immediately following one of these
                    runs -- which is the number the ByteTrack decision actually
                    rests on. An aggregate MOTA can improve while re-entry
                    switches stay exactly as bad as they were.

    Returns MOTA, IDF1, id switches, fragmentation, and the usual counts.
    """
    frames = sorted(set(pred_by_frame) | set(gt_by_frame))
    n_gt = n_fp = n_fn = n_sw = 0
    prev_match = {}                      # gt_id -> pred_id on its last matched frame
    matched_pairs = {}                   # (gt_id, pred_id) -> count
    gt_count, pred_count = {}, {}
    frag = 0
    was_tracked = {}
    occ = set(occlusion_frames or ())
    reentry_sw = 0
    prev_in_occ = False

    for f in frames:
        gt = np.asarray(gt_by_frame.get(f, np.zeros((0, 5))), dtype=np.float32)
        pr = np.asarray(pred_by_frame.get(f, np.zeros((0, 5))), dtype=np.float32)
        n_gt += len(gt)
        for g in gt:
            gt_count[int(g[4])] = gt_count.get(int(g[4]), 0) + 1
        for p in pr:
            pred_count[int(p[4])] = pred_count.get(int(p[4]), 0) + 1

        iou = iou_batch(gt[:, :4], pr[:, :4])
        pairs, un_g, un_p = _match(1.0 - iou, 1.0 - iou_thresh)

        seen = set()
        for gi, pi in pairs:
            gid, pid = int(gt[gi, 4]), int(pr[pi, 4])
            key = (gid, pid)
            matched_pairs[key] = matched_pairs.get(key, 0) + 1
            if gid in prev_match and prev_match[gid] != pid:
                n_sw += 1
                if prev_in_occ or f - 1 in occ:
                    reentry_sw += 1
            prev_match[gid] = pid
            seen.add(gid)
            if not was_tracked.get(gid, False):
                was_tracked[gid] = True
        for gi in un_g:
            gid = int(gt[gi, 4])
            if was_tracked.get(gid, False):
                frag += 1
                was_tracked[gid] = False
        n_fn += len(un_g)
        n_fp += len(un_p)
        prev_in_occ = f in occ

    # ---- IDF1: one global id-to-id assignment, not a per-frame one -------- #
    ids_gt = sorted(gt_count)
    ids_pr = sorted(pred_count)
    idtp = 0
    if ids_gt and ids_pr:
        M = np.zeros((len(ids_gt), len(ids_pr)))
        for (g, p), c in matched_pairs.items():
            M[ids_gt.index(g), ids_pr.index(p)] = c
        r, c = linear_sum_assignment(-M)
        idtp = int(M[r, c].sum())
    idfp = sum(pred_count.values()) - idtp
    idfn = sum(gt_count.values()) - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1e-9)

    mota = 1.0 - (n_fn + n_fp + n_sw) / max(n_gt, 1)
    motp_pairs = sum(matched_pairs.values())
    return {
        "MOTA": round(float(mota), 5),
        "IDF1": round(float(idf1), 5),
        "id_switches": int(n_sw),
        "id_switches_at_occlusion_reentry": int(reentry_sw),
        "fragmentations": int(frag),
        "false_positives": int(n_fp),
        "false_negatives": int(n_fn),
        "n_gt_boxes": int(n_gt),
        "n_matched": int(motp_pairs),
        "n_gt_ids": len(ids_gt),
        "n_pred_ids": len(ids_pr),
    }
