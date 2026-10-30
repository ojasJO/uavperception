"""Diagnose non-finite gradients of the AeroTrack detector under fp16 autocast on the V100 (env gate C3 failure, 2026-10-06).
Prints one JSON report: forward-output finiteness, loss, first parameters with non-finite gradients (no GradScaler), the same
in fp32, and eval-mode fp16 vs fp32 output agreement.  python tools/hpc/fp16_diag.py
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "c_model"))
from aerotrack_trainer import build_model, transfer_pretrained  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402


def finite_tree(t):
    if isinstance(t, (list, tuple)):
        return all(finite_tree(x) for x in t)
    if isinstance(t, dict):
        return all(finite_tree(x) for x in t.values())
    return bool(torch.isfinite(t).all()) if torch.is_tensor(t) else True


def step(m, x, batch, dtype):
    m.zero_grad(set_to_none=True)
    crit = v8DetectionLoss(m)
    with torch.autocast("cuda", dtype=dtype or torch.float16, enabled=dtype is not None):
        preds = m(x)
        loss, items = crit(preds, batch)
    out = {"preds_finite": finite_tree(preds), "loss": float(loss.sum()), "items": str(items)[:200]}
    loss.sum().backward()
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    out.update(n_bad_grads=len(bad), first_bad=bad[:6], last_bad=bad[-3:])
    # activations: hook every module output for non-finite values (forward only, fp16)
    return out


def act_scan(m, x, dtype):
    first = []
    hooks = []
    for name, mod in m.named_modules():
        if name.count(".") == 1:
            hooks.append(mod.register_forward_hook(
                lambda mod, i, o, name=name: first.append(name) if (torch.is_tensor(o) and not torch.isfinite(o).all()) else None))
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        m(x)
    for h in hooks:
        h.remove()
    return first[:5]


def main():
    torch.manual_seed(0)
    m = build_model(REPO / "c_model" / "yolo11-stride.yaml", ch=9, nc=2)
    transfer_pretrained(m, REPO / "c_model" / "yolo11n.pt", verbose=False)
    m = m.cuda().train()
    x = torch.rand(8, 9, 640, 640, device="cuda")
    batch = {"batch_idx": torch.tensor([0., 1., 2.], device="cuda"), "cls": torch.tensor([[0.], [0.], [1.]], device="cuda"),
             "bboxes": torch.tensor([[.5, .5, .05, .04], [.3, .6, .02, .02], [.7, .2, .1, .08]], device="cuda")}
    rep = {"fp16_train": step(m, x, batch, torch.float16), "fp32_train": step(m, x, batch, None),
           "fp16_train_nonfinite_activations": act_scan(m, x, torch.float16)}
    m.eval()
    with torch.no_grad():
        y32 = m(x)[0].float()
        with torch.autocast("cuda", dtype=torch.float16):
            y16 = m(x)[0].float()
    rep["eval_fp16_vs_fp32"] = {"finite16": bool(torch.isfinite(y16).all()), "max_abs_diff": float((y16 - y32).abs().max()),
                                "max_abs_fp32": float(y32.abs().max())}
    # same check on an Ultralytics stock 3-channel YOLO11n (does the problem follow the library or our model?)
    from ultralytics.nn.tasks import DetectionModel
    s = DetectionModel("yolo11n.yaml", ch=3, nc=2, verbose=False)
    from types import SimpleNamespace
    s.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    s = s.cuda().train()
    rep["stock3ch_fp16_train"] = step(s, x[:, :3].contiguous(), batch, torch.float16)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
