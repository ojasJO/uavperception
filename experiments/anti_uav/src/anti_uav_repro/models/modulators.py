"""Query-guided modulators — port of GlobalTrack `modules/modulators.py` to torchvision primitives.

RPN_Modulator (Eq. 2 of the Anti-UAV paper, f_z = 7x7 conv, f_x = identity, f_out = 1x1 conv):
    z   = RoIAlign_7x7(query FPN features, query GT box)                     [1, C, 7, 7]
    t_k = proj_out[k]( proj_modulator[k](z) * x_k )   for each FPN level k    (channel-wise product)

RCNN_Modulator (Eq. 5, Hadamard product on RoI features):
    t   = proj_out( proj_x(x_rois) * proj_z(z_roi) )

Weight init: normal(std=0.01), zero bias — as upstream `normal_init(m, std=0.01)`.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor, nn
from torchvision.ops import MultiScaleRoIAlign


def _normal_init(m: nn.Conv2d, std: float = 0.01) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=std)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class RPNModulator(nn.Module):
    def __init__(self, channels: int = 256, roi_out_size: int = 7, roi_sampling_ratio: int = 2,
                 roi_featmap_names=("0", "1", "2", "3"), num_levels: int = 5):
        super().__init__()
        # upstream: SingleRoIExtractor(RoIAlign out 7, sample 2, strides [4, 8, 16, 32]) — FPN levels P2..P5
        self.roi_align = MultiScaleRoIAlign(list(roi_featmap_names), roi_out_size, roi_sampling_ratio)
        self.proj_modulator = nn.ModuleList(
            [nn.Conv2d(channels, channels, roi_out_size, padding=0) for _ in range(num_levels)])
        self.proj_out = nn.ModuleList([nn.Conv2d(channels, channels, 1, padding=0) for _ in range(num_levels)])
        self.init_weights()

    def init_weights(self) -> None:
        for m in list(self.proj_modulator) + list(self.proj_out):
            _normal_init(m)

    def learn(self, feats_z: "OrderedDict[str, Tensor]", boxes_z: list[Tensor], image_sizes_z: list[tuple[int, int]]) -> Tensor:
        """Query RoI features, one box per image -> [B, C, 7, 7]."""
        return self.roi_align(feats_z, boxes_z, image_sizes_z)

    def modulate(self, feats_x: "OrderedDict[str, Tensor]", z: Tensor) -> "OrderedDict[str, Tensor]":
        """Modulate every FPN level of ONE search image (batch dim 1) with ONE query feature z [1, C, 7, 7]."""
        assert z.shape[0] == 1, "one query instance at a time (as upstream)"
        out: "OrderedDict[str, Tensor]" = OrderedDict()
        for k, (name, x) in enumerate(feats_x.items()):
            kernel = self.proj_modulator[k](z)            # [1, C, 1, 1]
            out[name] = self.proj_out[k](kernel * x)      # broadcast channel-wise product, then 1x1 conv
        return out


class RCNNModulator(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.proj_z = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_x = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.init_weights()

    def init_weights(self) -> None:
        for m in (self.proj_z, self.proj_x, self.proj_out):
            _normal_init(m)

    def learn(self, z: Tensor) -> Tensor:
        assert z.shape[0] == 1, "one query instance at a time (as upstream)"
        return self.proj_z(z)

    def forward(self, z: Tensor, x: Tensor) -> Tensor:
        """z: query RoI feature [1, C, 7, 7]; x: proposal RoI features [N, C, 7, 7]."""
        return self.proj_out(self.proj_x(x) * self.learn(z))
