"""Query/search pair sampling for training — port of GlobalTrack's `Seq2Pair` (neuron/data/datasets/structure.py).

Epoch definition (GlobalTrack): len = n_sequences * pairs_per_seq; item i uses sequence i % n_sequences.
Pair: two frames of the same sequence, query (z) before search (x), gap <= max_distance frames
(100 rejection tries, then any two frames). Frames must pass `globaltrack_frame_filter`.

Determinism: every item is drawn from `np.random.default_rng(seed, epoch, index)`, so a training run
is a pure function of (seed, config) up to cuDNN non-determinism. The epoch is encoded INTO the index
by the batch sampler (index = epoch * len(dataset) + local_index) so that DataLoader worker processes —
which hold stale copies of the dataset object, notably with persistent workers on Windows — never
re-sample the previous epoch's pairs.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .antiuav import Sequence
from .transforms import TrainPairTransform, read_image_rgb, xywh_to_xyxy


def globaltrack_frame_filter(boxes_xywh: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Boolean mask, faithful to Seq2Pair._filter (operates on original w, h in pixels)."""
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    areas = w * h
    conds = [
        areas > 20,
        (w >= 10) & (h >= 10),
        (w <= 960) & (h <= 960),
        (w / img_w >= 0.01) & (h / img_h >= 0.01),
        (w / img_w <= 0.75) & (h / img_h <= 0.75),
        w / np.maximum(1, h) >= 0.2,
        w / np.maximum(1, h) < 5,
    ]
    return np.logical_and.reduce(conds)


def trainable_frame_indices(seq: Sequence, frame_filter: str) -> np.ndarray:
    valid = seq.valid_frames()
    if len(valid) == 0:
        return valid
    boxes = np.stack([seq.box_xywh(int(i)) for i in valid]).astype(np.float32)
    if frame_filter == "globaltrack":
        w, h = seq.image_size()
        mask = globaltrack_frame_filter(boxes, w, h)
        return valid[mask]
    if frame_filter == "none":
        return valid
    raise ValueError(frame_filter)


class PairDataset(Dataset):
    def __init__(self, sequences: list[Sequence], pairs_per_seq: int, max_distance: int,
                 frame_filter: str, transform: TrainPairTransform, seed: int):
        if not sequences:
            raise ValueError("no sequences")
        self.sequences = sequences
        self.pairs_per_seq = int(pairs_per_seq)
        self.max_distance = int(max_distance)
        self.transform = transform
        self.seed = int(seed)
        self.epoch = 0
        # precompute per-sequence trainable frame indices once (deterministic, label-derived)
        self.frame_indices = [trainable_frame_indices(s, frame_filter) for s in sequences]
        self.usable = [i for i, idx in enumerate(self.frame_indices) if len(idx) >= 1]
        if not self.usable:
            raise RuntimeError("no sequence has a trainable frame after filtering")

    def set_epoch(self, epoch: int) -> None:
        """Kept for API symmetry; the epoch actually used for sampling is decoded from the index."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.sequences) * self.pairs_per_seq

    def _decode(self, index: int) -> tuple[int, int]:
        n = len(self)
        return index // n, index % n          # (epoch, local index)

    def _rng(self, index: int) -> np.random.Generator:
        epoch, local = self._decode(index)
        return np.random.default_rng([self.seed, epoch, local])

    def _sample_pair(self, indices: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
        n = len(indices)
        if n == 1:
            return int(indices[0]), int(indices[0])
        if n == 2:
            return int(indices[0]), int(indices[1])
        for _ in range(100):
            z, x = sorted(rng.choice(indices, 2, replace=False))
            if x - z <= self.max_distance:
                return int(z), int(x)
        z, x = rng.choice(indices, 2)
        return int(z), int(x)

    def __getitem__(self, index: int) -> dict:
        rng = self._rng(index)
        _, local = self._decode(index)
        seq_idx = local % len(self.sequences)
        indices = self.frame_indices[seq_idx]
        if len(indices) == 0:
            # upstream `_random_next`: fall back to a random usable sequence, deterministically
            seq_idx = int(rng.choice(self.usable))
            indices = self.frame_indices[seq_idx]
        seq = self.sequences[seq_idx]
        fz, fx = self._sample_pair(indices, rng)
        img_z = read_image_rgb(seq.frame_path(fz))
        img_x = read_image_rgb(seq.frame_path(fx))
        bz = xywh_to_xyxy(seq.box_xywh(fz))
        bx = xywh_to_xyxy(seq.box_xywh(fx))
        tz, bz_t, mz, tx, bx_t, mx = self.transform(img_z, bz, img_x, bx, rng)
        return {
            "img_z": tz, "img_x": tx, "boxes_z": bz_t, "boxes_x": bx_t,
            "meta_z": mz, "meta_x": mx,
            "seq_index": seq_idx, "seq_name": seq.name, "frame_z": fz, "frame_x": fx,
        }


class DistinctSequenceBatchSampler(Sampler[list[int]]):
    """Batches of `batch_size` items whose sequences are pairwise distinct.

    Each epoch is `pairs_per_seq` rounds; each round is a fresh permutation of all sequences chunked
    into batches, so every sequence contributes exactly `pairs_per_seq` items per epoch (as upstream)
    and no batch contains two pairs from the same sequence (a DFSC requirement, spec S5).
    """

    def __init__(self, n_sequences: int, pairs_per_seq: int, batch_size: int, seed: int, drop_last: bool = True):
        self.n = int(n_sequences)
        self.rounds = int(pairs_per_seq)
        self.bs = int(batch_size)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0
        if self.bs > self.n:
            raise ValueError("batch_size larger than number of sequences")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch, 987654321])
        offset = self.epoch * self.n * self.rounds     # epoch encoded into the index (see module docstring)
        for r in range(self.rounds):
            perm = rng.permutation(self.n)
            indices = perm + r * self.n + offset   # local index = round * n + seq (so local % n == seq)
            for start in range(0, self.n, self.bs):
                chunk = indices[start:start + self.bs]
                if len(chunk) < self.bs and self.drop_last:
                    continue
                yield [int(i) for i in chunk]

    def __len__(self) -> int:
        per_round = self.n // self.bs if self.drop_last else -(-self.n // self.bs)
        return per_round * self.rounds


def collate_pairs(batch: list[dict]) -> dict:
    """Pads images in the batch to a common size (per side) and stacks them; boxes stay as lists."""
    def stack(key):
        imgs = [b[key] for b in batch]
        H = max(i.shape[1] for i in imgs)
        W = max(i.shape[2] for i in imgs)
        out = imgs[0].new_zeros((len(imgs), 3, H, W))
        for k, im in enumerate(imgs):
            out[k, :, :im.shape[1], :im.shape[2]] = im
        return out
    return {
        "img_z": stack("img_z"), "img_x": stack("img_x"),
        "boxes_z": [b["boxes_z"] for b in batch], "boxes_x": [b["boxes_x"] for b in batch],
        "sizes_z": [tuple(b["meta_z"]["img_shape"]) for b in batch],
        "sizes_x": [tuple(b["meta_x"]["img_shape"]) for b in batch],
        "meta": [{k: b[k] for k in ("seq_index", "seq_name", "frame_z", "frame_x")} for b in batch],
    }
