"""Image transforms replicating GlobalTrack's `ExtraPairTransforms(with_photometric=True, with_expand=False,
with_crop=False)` for training and `BasicPairTransforms(train=False)` for tracking.

Source of truth: GlobalTrack `_submodules/neuron/neuron/data/transforms/pair_transforms/mmdet_transforms.py`
and `neuron/ops/transforms.py`. Differences from the original are limited to:
  * boxes use continuous [x1, y1, x2, y2] with x2 = x + w (torchvision convention) instead of the
    inclusive x2 = x + w - 1 of the original code (sub-pixel difference);
  * the query/search pair shares one RNG so the channel permutation is shared exactly as upstream.
All randomness comes from the `rng` argument (numpy Generator) so a pair is reproducible from
(seed, epoch, index).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)   # RGB, 0-255 scale
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_INTERPS = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]


def read_image_rgb(path) -> np.ndarray:
    """uint8 HxWx3 RGB (upstream uses PIL -> RGB; cv2 loads BGR so we flip channels)."""
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(im[:, :, ::-1])


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    out = b.copy()
    out[:, 2] = b[:, 0] + b[:, 2]
    out[:, 3] = b[:, 1] + b[:, 3]
    return out


def xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    out = b.copy()
    out[:, 2] = b[:, 2] - b[:, 0]
    out[:, 3] = b[:, 3] - b[:, 1]
    return out


def rescale_factor(h: int, w: int, scale: tuple[int, int]) -> float:
    long_edge, short_edge = max(scale), min(scale)
    return min(long_edge / max(h, w), short_edge / min(h, w))


def rescale_img(img: np.ndarray, scale: tuple[int, int], interp: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    f = rescale_factor(h, w, scale)
    out_w, out_h = int(w * f + 0.5), int(h * f + 0.5)
    return cv2.resize(img, (out_w, out_h), interpolation=interp), f


def photometric_distort(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Float32 RGB in/out. Faithful port of neuron.ops.photometric_distort (swap handled by caller)."""
    img = img.astype(np.float32)
    if rng.integers(2):
        img += rng.uniform(-32, 32)
    contrast_first = rng.integers(2)
    if contrast_first and rng.integers(2):
        img *= rng.uniform(0.5, 1.5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if rng.integers(2):
        img[:, :, 1] *= rng.uniform(0.5, 1.5)
    if rng.integers(2):
        img[:, :, 0] += rng.uniform(-18, 18)
        img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
        img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    if not contrast_first and rng.integers(2):
        img *= rng.uniform(0.5, 1.5)
    return img


def flip_img_boxes(img: np.ndarray, boxes: np.ndarray | None):
    img = np.ascontiguousarray(img[:, ::-1])
    if boxes is None:
        return img, None
    w = img.shape[1]
    out = boxes.copy()
    out[:, 0] = w - boxes[:, 2]
    out[:, 2] = w - boxes[:, 0]
    return img, out


def pad_to_divisor(img: np.ndarray, divisor: int) -> np.ndarray:
    h, w = img.shape[:2]
    oh, ow = int(np.ceil(h / divisor) * divisor), int(np.ceil(w / divisor) * divisor)
    if (oh, ow) == (h, w):
        return img
    out = np.zeros((oh, ow) + img.shape[2:], dtype=img.dtype)
    out[:h, :w] = img
    return out


def bound_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    out = boxes.copy()
    out[:, 0::2] = np.clip(out[:, 0::2], 0, w)
    out[:, 1::2] = np.clip(out[:, 1::2], 0, h)
    out[:, 2] = np.maximum(out[:, 2], out[:, 0])
    out[:, 3] = np.maximum(out[:, 3], out[:, 1])
    return out


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()


@dataclass
class TransformConfig:
    scale: tuple[int, int] = (1333, 800)
    photometric: bool = True
    channel_swap: bool = True
    flip_p: float = 0.5
    pad_divisor: int = 32


class TrainPairTransform:
    """Query/search pair augmentation (GlobalTrack 'extra_partial' + basic train transforms)."""

    def __init__(self, cfg: TransformConfig):
        self.cfg = cfg

    def _one(self, img: np.ndarray, boxes: np.ndarray, rng: np.random.Generator, swap_order):
        img = img.astype(np.float32)
        if self.cfg.photometric:
            img = photometric_distort(img, rng)
            if swap_order is not None:
                img = img[..., swap_order]
        interp = _INTERPS[int(rng.integers(len(_INTERPS)))]
        img, f = rescale_img(img, tuple(self.cfg.scale), interp)
        boxes = boxes * f
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        flipped = False
        if rng.random() < self.cfg.flip_p:
            img, boxes = flip_img_boxes(img, boxes)
            flipped = True
        h, w = img.shape[:2]
        img = pad_to_divisor(img, self.cfg.pad_divisor)
        boxes = bound_boxes(boxes, w, h)
        return to_tensor(img), torch.from_numpy(boxes).float(), {"scale_factor": f, "flip": flipped, "img_shape": (h, w), "pad_shape": img.shape[:2]}

    def __call__(self, img_z, boxes_z_xyxy, img_x, boxes_x_xyxy, rng: np.random.Generator):
        swap_order = None
        if self.cfg.photometric and self.cfg.channel_swap and rng.integers(2):
            swap_order = rng.permutation(3)
        tz, bz, mz = self._one(img_z, boxes_z_xyxy, rng, swap_order)
        tx, bx, mx = self._one(img_x, boxes_x_xyxy, rng, swap_order)
        return tz, bz, mz, tx, bx, mx


class TestTransform:
    """Deterministic transform for tracking (BasicPairTransforms(train=False)): bilinear rescale, normalise, pad."""

    def __init__(self, scale=(1333, 800), pad_divisor: int = 32):
        self.scale = tuple(scale)
        self.pad_divisor = pad_divisor

    def __call__(self, img: np.ndarray):
        img = img.astype(np.float32)
        img, f = rescale_img(img, self.scale, cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = pad_to_divisor(img, self.pad_divisor)
        return to_tensor(img), {"scale_factor": f, "img_shape": (h, w), "pad_shape": img.shape[:2]}
