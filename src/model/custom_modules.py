#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
custom_modules.py  --  Phase C: custom SPD-Conv building block for AeroTrack-Net
================================================================================
`SPDConv` (Space-to-Depth Convolution) is a *lossless* downsampler that replaces
the stride-2 `Conv` blocks in a detection backbone.  A strided conv throws away
3/4 of the spatial samples; for the sub-16 px drones in our datasets that is
exactly the information we cannot afford to lose.  SPD-Conv instead folds every
2x2 spatial neighbourhood into the channel dimension (no pixel discarded), then
applies a *non-strided* convolution:

    (B, C, H, W)
      --space-to-depth-->  (B, 4C, H/2, W/2)     # 4 interleaved slices, no loss
      --Conv(4C->c2,s=1)-> (B, c2, H/2, W/2)      # learnable channel mixing

Reference: Sunkara & Luo, "No More Strided Convolutions or Pooling: A New CNN
Building Block for Low-Resolution Images and Small Objects" (2022).
"""

import torch
import torch.nn as nn

try:                                            # prefer the Ultralytics Conv block
    from ultralytics.nn.modules import Conv
except Exception:                               # standalone fallback (no ultralytics)
    Conv = None


def _autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class SPDConv(nn.Module):
    """Space-to-Depth + non-strided Conv downsampler.

    Signature mirrors Ultralytics ``Conv(c1, c2, k, s, ...)`` so the YAML parser
    can inject the input-channel count ``c1`` automatically.  The stride argument
    is accepted for API-compatibility but ignored: the spatial /2 is produced by
    the space-to-depth fold, never by striding.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.c1, self.c2 = c1, c2
        if Conv is not None:
            self.conv = Conv(4 * c1, c2, k=k, s=1, p=p, g=g, d=d, act=act)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(4 * c1, c2, k, 1, _autopad(k, p, d),
                          groups=g, dilation=d, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU() if act is True else (act if isinstance(act, nn.Module)
                                               else nn.Identity()),
            )

    @staticmethod
    def space_to_depth(x):
        """(B,C,H,W) -> (B,4C,H/2,W/2) by interleaved 2x2 sampling (lossless)."""
        return torch.cat(
            (x[..., 0::2, 0::2],       # top-left
             x[..., 1::2, 0::2],       # bottom-left
             x[..., 0::2, 1::2],       # top-right
             x[..., 1::2, 1::2]),      # bottom-right
            dim=1,
        )

    def forward(self, x):
        return self.conv(self.space_to_depth(x))

    def __repr__(self):
        return f"SPDConv(c1={self.c1}, c2={self.c2}) [space-to-depth /2 + Conv s=1]"


if __name__ == "__main__":
    # standalone shape check (no ultralytics needed thanks to the fallback)
    m = SPDConv(9, 64, k=3)
    x = torch.randn(1, 9, 640, 640)
    y = m(x)
    print("SPDConv:", tuple(x.shape), "->", tuple(y.shape),
          "(expect (1, 64, 320, 320))")
    assert tuple(y.shape) == (1, 64, 320, 320)
    print("OK")
