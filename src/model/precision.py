"""One place that decides the autocast dtype (D-2026-10-06-04).

Every AeroTrack run up to 2026-10-05 used bf16 autocast on the OMEN's RTX 5050 (sm_120). The HPC's Tesla V100 (sm_70) has no
native bf16 -- and ``torch.cuda.is_bf16_supported()`` still returns True there (emulation), so the old idiom silently selected a
slow, numerically different path. Rule: bf16 on compute capability >= 8.0, fp16 below; ``AEROTRACK_AMP=bf16|fp16|fp32``
overrides (fp32 -> no autocast). On the OMEN the result is bf16, exactly as before.
"""
import os

import torch

_NAMES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def amp_name(device=None):
    """'bf16' | 'fp16' | 'fp32' for this process."""
    env = os.environ.get("AEROTRACK_AMP", "").strip().lower()
    if env:
        if env not in _NAMES:
            raise ValueError(f"AEROTRACK_AMP must be one of {sorted(_NAMES)}, got {env!r}")
        return env
    if not torch.cuda.is_available() or (device is not None and torch.device(device).type != "cuda"):
        return "fp32"
    major, _ = torch.cuda.get_device_capability(device if device is not None else torch.cuda.current_device())
    return "bf16" if major >= 8 else "fp16"


def amp_dtype(device=None):
    """torch dtype for torch.autocast, or None for fp32."""
    return _NAMES[amp_name(device)]
