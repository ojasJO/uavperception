#!/usr/bin/env python
"""D5 — are CST micro drones missed, or found but mislocalised? For each CST micro drone GT in the D3 VAL records (last.pt):
best IoU among drone predictions at conf >= 0.25, and whether any prediction centre lies within max(3 px, half the GT size).
A diagnostic of the metric's sensitivity, not a new metric.

    venv\\Scripts\\python.exe experiments\\aerotrack_generalization\\scripts\\diag_localization.py
Output: reports/diag_localization.json."""
import pickle

import _series as S
import numpy as np

import evaluation as E

LO, HI = E.SIZE_BINS["micro"]


def main():
    out = {}
    for f in sorted((S.CACHE / "val_records").glob("*__last.pkl")):
        best, centre = [], []
        for r in pickle.load(open(f, "rb"))["records"]:
            if r["dataset"] != "cst" or not len(r["gt"]):
                continue
            m = (r["gt_area"] >= LO) & (r["gt_area"] < HI) & (r["gt"][:, 0] == 0)
            if not m.any():
                continue
            g = r["gt"][m]
            p = r["pred"][(r["pred"][:, 4] >= 0.25) & (r["pred"][:, 5] == 0)] if len(r["pred"]) else r["pred"]
            if not len(p):
                best += [0.0] * len(g)
                centre += [False] * len(g)
                continue
            best += list(E.box_iou_np(g[:, 1:5], p[:, :4]).max(1))
            gc, pc = (g[:, 1:3] + g[:, 3:5]) / 2, (p[:, :2] + p[:, 2:4]) / 2
            half = np.maximum(np.maximum(g[:, 3] - g[:, 1], g[:, 4] - g[:, 2]) / 2, 3.0)
            centre += list((np.linalg.norm(gc[:, None] - pc[None], axis=2) <= half[:, None]).any(1))
        b, c = np.array(best), np.array(centre)
        out[f.stem] = {"cst_micro_drone_gt": int(len(b)), "iou_ge_0.5": round(float((b >= 0.5).mean()), 3),
                       "iou_0.25_to_0.5_found_but_mislocalised": round(float(((b >= 0.25) & (b < 0.5)).mean()), 3),
                       "iou_lt_0.25_missed": round(float((b < 0.25).mean()), 3), "centre_hit": round(float(c.mean()), 3)}
    p = S.write_report("diag_localization.json", out)
    S.ledger({"id": "D5", "type": "diagnostic", "what": "CST micro: missed vs mislocalised (last.pt, conf>=0.25)",
              "script": "scripts/diag_localization.py", "output": str(p.relative_to(S.REPO)), "test_usage": "none"})
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
