"""Shared helpers of the CST Anti-UAV study: official result format and the Table 5 size bins.

Official format (toolkit `experiments/cstanti_uav.py::report`): one file per sequence, `<seq>.txt`, read first as JSON
`{"res": [...]}` -- each entry a 4-number `[x, y, w, h]` box or `[]` for "absent" -- else as comma separated rows. We always write
the JSON form because rows cannot express "absent".
"""
import json
from pathlib import Path

import numpy as np

SIZE_BINS = (("tiny", 0, 10), ("small", 10, 30), ("normal", 30, 50), ("large", 50, np.inf))   # GT diagonal, px (Table 5)


def official_record(res):
    """Track A prediction list (box or []) -> the official JSON record; validates every entry."""
    out = []
    for i, r in enumerate(res):
        if r is None or len(r) == 0:
            out.append([])
        elif len(r) == 4 and all(np.isfinite(float(v)) for v in r):
            out.append([float(v) for v in r])
        else:
            raise ValueError(f"frame {i}: not a box or empty: {r!r}")
    return {"res": out}


def write_official(out_dir, seq_name, res):
    p = Path(out_dir) / f"{seq_name}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(official_record(res)), encoding="utf-8")
    return p


def size_code(gt, exist):
    """0 = absent / no box, 1..4 = tiny / small / normal / large by the GT diagonal (the release's SIZE codes)."""
    if not exist or len(gt) != 4:
        return 0
    d = float(np.hypot(gt[2], gt[3]))
    return next(i for i, (_, lo, hi) in enumerate(SIZE_BINS, 1) if lo < d <= hi)
