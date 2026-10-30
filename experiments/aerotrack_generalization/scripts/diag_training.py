#!/usr/bin/env python
"""D2 — training dynamics of the completed grid + initialisation audit (CPU only; no training).

    venv\\Scripts\\python.exe experiments\\aerotrack_generalization\\scripts\\diag_training.py

* per-epoch curves of results/runs/repair_abl_*_e20/results.csv: early lead, final training loss, late-epoch volatility of the
  capped-val metrics that drive best.pt selection;
* how many PARAMETERS each layout receives from COCO under the registered transfer rule;
* a numerical check of a function-preserving initialisation (SPDConv mapped from the pretrained stride-2 conv; COCO stem on
  the centre frame, zeros on t-1/t+1): the SPD model then equals the stride model, and the 9-ch model equals YOLO11n on t.
Output: reports/diag_training.json (+ one ledger line)."""
import csv
import json
import statistics as st

import _series as S
import torch

from aerotrack_trainer import build_model, transfer_pretrained

RUNS = ("baseline_ctrl", "no_spd", "single_frame", "stock_yolo11n")
COCO = S.REPO / "src/model" / "yolo11n.pt"
DOWN = ("0", "1", "3", "5", "7")
RMAP = {0: (1, 0), 1: (0, 1), 2: (1, 1)}               # stride tap -> (sub-pixel phase, SPD tap)
BLK = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}     # (row phase, col phase) -> SPDConv.space_to_depth block


def spd_from_stride(W):
    """[c2, c1, 3, 3] stride-2 conv kernel -> [c2, 4*c1, 3, 3] SPDConv kernel computing the same function."""
    c2, c1 = W.shape[:2]
    V = torch.zeros(c2, 4 * c1, 3, 3, dtype=W.dtype)
    for a in range(3):
        rp, c = RMAP[a]
        for b in range(3):
            cp, d = RMAP[b]
            k = BLK[(rp, cp)]
            V[:, k * c1:(k + 1) * c1, c, d] = W[:, :, a, b]
    return V


def curves():
    out = {}
    for r in RUNS:
        rows = list(csv.DictReader(open(S.REPO / "results/runs" / f"repair_abl_{r}_e20" / "results.csv", encoding="utf-8")))
        f = lambda k: [float(x[k]) for x in rows]
        loss = [a + b + c for a, b, c in zip(f("box_loss"), f("cls_loss"), f("dfl_loss"))]
        mi, m50 = f("micro_AP50"), f("mAP50")
        out[r] = {"train_loss_epoch0": round(loss[0], 3), "train_loss_epoch19": round(loss[19], 3),
                  "train_loss_slope_last5_per_epoch": round((loss[19] - loss[14]) / 5, 4),
                  "capped_val_mAP50_epoch0": m50[0], "capped_val_mAP50_mean_ep0_4": round(st.mean(m50[:5]), 3),
                  "capped_val_mAP50_mean_ep15_19": round(st.mean(m50[15:]), 3),
                  "capped_val_micro_AP50_ep13_19": mi[13:], "capped_val_micro_AP50_sd_ep15_19": round(st.pstdev(mi[15:]), 3),
                  "best_fitness_epoch": int(max(rows, key=lambda x: float(x["fitness"]))["epoch"])}
    return out


def init_audit():
    import ultralytics.nn.tasks as UT
    coco = UT.load_checkpoint(str(COCO))
    coco = (coco[0] if isinstance(coco, tuple) else coco).float().state_dict()
    out = {}
    for cfg in ("yolo11-spd.yaml", "yolo11-stride.yaml"):
        sd = build_model(S.REPO / "src/model" / cfg, ch=9, nc=2).state_dict()
        tot = sum(v.numel() for v in sd.values() if v.dtype.is_floating_point)
        same = sum(v.numel() for k, v in sd.items() if v.dtype.is_floating_point and k in coco and coco[k].shape == v.shape)
        down = sum(v.numel() for k, v in sd.items() if v.dtype.is_floating_point and k.split(".")[1] in DOWN)
        out[cfg] = {"params_M": round(tot / 1e6, 3), "from_coco_shape_match_M": round(same / 1e6, 3),
                    "from_coco_pct": round(100 * same / tot, 1), "downsampling_layers_M": round(down / 1e6, 3)}
    # function-preserving init check
    torch.manual_seed(0)
    stride = build_model(S.REPO / "src/model" / "yolo11-stride.yaml", ch=9, nc=2)
    transfer_pretrained(stride, COCO, verbose=False)
    sd = {k: v.clone() for k, v in stride.state_dict().items()}
    tiled = sd["model.0.conv.weight"].clone()
    centre = torch.zeros_like(tiled)
    centre[:, 3:6] = tiled[:, 3:6] * 3.0
    sd["model.0.conv.weight"] = centre
    stride.load_state_dict(sd)
    spd = build_model(S.REPO / "src/model" / "yolo11-spd.yaml", ch=9, nc=2)
    sp = {k: v.clone() for k, v in spd.state_dict().items()}
    for k, v in sd.items():
        if k in sp and sp[k].shape == v.shape:
            sp[k] = v.clone()
    for i in DOWN:
        sp[f"model.{i}.conv.conv.weight"] = spd_from_stride(sd[f"model.{i}.conv.weight"])
        for t in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            sp[f"model.{i}.conv.bn.{t}"] = sd[f"model.{i}.bn.{t}"].clone()
    spd.load_state_dict(sp)
    stride.eval(); spd.eval()
    x = torch.rand(2, 9, 640, 640)
    t = torch.rand(2, 3, 640, 640)
    with torch.no_grad():
        ys, yp = stride(x)[0], spd(x)[0]
        a1 = stride(torch.cat([torch.rand_like(t), t, torch.rand_like(t)], 1))[0]
        a2 = stride(torch.cat([t, t, t], 1))[0]
        sd2 = {k: v.clone() for k, v in stride.state_dict().items()}
        sd2["model.0.conv.weight"] = tiled
        stride.load_state_dict(sd2)
        a3 = stride(torch.cat([torch.rand_like(t), t, torch.rand_like(t)], 1))[0]
    out["function_preserving_init_check"] = {
        "spd_mapped_vs_stride_max_abs_diff": float((ys - yp).abs().max()), "output_max_abs": float(ys.abs().max()),
        "centre_frame_init_ignores_neighbours_max_abs_diff": float((a1 - a2).abs().max()),
        "grid_tiled_stem_neighbours_vs_ttt_max_abs_diff": float((a3 - a2).abs().max()),
        "reading": "SPD with the mapped init computes the stride model's function (float32 rounding); the centre-frame stem makes the "
                   "9-ch model exactly YOLO11n on frame t; the grid's tiled /3 stem instead responds to the mean of the three frames"}
    return out


def main():
    out = {"curves": curves(), "init": init_audit()}
    p = S.write_report("diag_training.json", out)
    S.ledger({"id": "D2", "type": "diagnostic", "what": "training curves + COCO initialisation audit + function-preserving init check",
              "script": "scripts/diag_training.py", "output": str(p.relative_to(S.REPO)), "test_usage": "none"})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
