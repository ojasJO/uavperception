"""Sequence-level tracker inference (GlobalTrack protocol) producing raw per-frame predictions.

Protocol (spec §5): initialise on the sequence's start frame with its GT box; for every later frame run
QG-RCNN on the whole image and output the top-scoring box. No motion model, no absence decision.
Frames before the start frame are recorded as absent ([0]); the start frame records the init box,
exactly like the challenge reference (`out_res.append(init_rect)`).

The tracker never reads ground truth after the init frame. Scores are stored alongside boxes so that
threshold-based absence rules can be studied later WITHOUT re-running the model (Phase B only).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..data.antiuav import Sequence
from ..data.transforms import TestTransform, read_image_rgb, xywh_to_xyxy


class GlobalTrackTracker:
    def __init__(self, model, device: torch.device, transform: TestTransform, amp: str = "none"):
        self.model = model.eval()
        self.device = device
        self.transform = transform
        self.amp = amp

    def _autocast(self):
        if self.amp == "bf16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if self.amp == "fp16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return torch.autocast("cpu", enabled=False)

    @torch.no_grad()
    def track_sequence(self, seq: Sequence, frame_stride: int = 1, max_frames: int | None = None) -> dict:
        n = seq.n_frames
        start = seq.start_frame
        res: list = [[0] for _ in range(n)]
        scores: list = [0.0 for _ in range(n)]
        processed = np.zeros(n, dtype=bool)

        init_box = seq.box_xywh(start)
        if init_box is None:
            raise ValueError(f"{seq.name}: start frame {start} has no valid GT box")
        img = read_image_rgb(seq.frame_path(start))
        t, meta = self.transform(img)
        box_xyxy = torch.from_numpy(xywh_to_xyxy(init_box) * meta["scale_factor"]).to(self.device)
        with self._autocast():
            self.model.track_init(t[None].to(self.device), box_xyxy, meta["img_shape"])
        res[start] = [float(v) for v in init_box]
        scores[start] = 1.0
        processed[start] = True

        last_box = res[start]
        n_fallback = 0
        t0 = time.time()
        frames = list(range(start + 1, n, frame_stride))
        if max_frames is not None:
            frames = frames[:max_frames]
        for f in frames:
            img = read_image_rgb(seq.frame_path(f))
            t, meta = self.transform(img)
            with self._autocast():
                boxes, sc = self.model.track_update(t[None].to(self.device), meta["img_shape"])
            if boxes.shape[0] == 0:
                # never happens with score_thr 0 in practice; keep the previous box and count it
                n_fallback += 1
                res[f] = list(last_box)
                scores[f] = 0.0
            else:
                b = (boxes[0].float() / meta["scale_factor"]).tolist()
                xywh = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                res[f] = xywh
                scores[f] = float(sc[0])
                last_box = xywh
            processed[f] = True
        elapsed = time.time() - t0
        return {
            "sequence": seq.name, "split": seq.split, "modality": seq.modality,
            "start_frame": int(start), "n_frames": int(n), "frame_stride": int(frame_stride),
            "res": res, "scores": scores, "processed": processed.tolist(),
            "n_processed": int(processed.sum()), "n_fallback": int(n_fallback),
            "fps": (len(frames) / elapsed) if elapsed > 0 else None,
        }
