#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_cpu_dry_run.py  --  Phase C, Step 5: CPU dry-run architecture test
====================================================================
Proves that the 9-channel SPD-Conv AeroTrack-Net backbone is dimensionally
consistent and forward-passes end-to-end -- WITHOUT any training.

Steps:
  1. Register `SPDConv` with the Ultralytics YAML parser (`parse_model`) so the
     custom layer is recognised AND gets Conv-style channel bookkeeping.
  2. Build the DetectionModel from `yolo11-spd.yaml` with ch=9 on the CPU.
  3. Forward a dummy stacked-frame tensor  (1, 9, 640, 640).
  4. Print the multi-scale output shapes + a success banner.

This is a *mathematical* compile check only. NO GPU. NO optimiser. NO data.
"""

import sys
import inspect
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from custom_modules import SPDConv


def register_spdconv():
    """Make the Ultralytics YAML parser recognise SPDConv and treat it like a
    channel-changing Conv block (so c1 is injected and c2 is tracked)."""
    import ultralytics.nn.tasks as T

    # (a) name resolution: parse_model resolves YAML strings via globals()[name]
    T.SPDConv = SPDConv
    try:
        import ultralytics.nn.modules as M
        M.SPDConv = SPDConv
    except Exception:
        pass

    # (b) channel bookkeeping: parse_model only injects (c1, c2) for modules in
    #     its `base_modules` frozenset. Patch that single predicate so SPDConv
    #     takes the same branch as Conv. We recompile the *installed* source, so
    #     this stays correct across ultralytics versions.
    src = inspect.getsource(T.parse_model)
    if "or m is SPDConv" not in src:
        patched = src.replace("if m in base_modules:",
                              "if m in base_modules or m is SPDConv:", 1)
        if patched == src:
            raise RuntimeError("could not locate 'if m in base_modules:' to patch")
        g = dict(vars(T))
        g["SPDConv"] = SPDConv
        exec(compile(patched, "<patched_parse_model>", "exec"), g)
        T.parse_model = g["parse_model"]
    return T


def out_shapes(o):
    if torch.is_tensor(o):
        return tuple(o.shape)
    if isinstance(o, (list, tuple)):
        return [out_shapes(x) for x in o]
    if isinstance(o, dict):
        return {k: out_shapes(v) for k, v in o.items()}
    return type(o).__name__


def main():
    torch.manual_seed(0)
    print("=" * 72)
    print("Phase C / Step 5: CPU DRY-RUN architecture test  (NO TRAINING)")
    print("=" * 72)
    print(f"torch {torch.__version__} | device=CPU | cuda_available={torch.cuda.is_available()}")

    register_spdconv()
    print("[1] SPDConv registered with Ultralytics parse_model.")

    from ultralytics.nn.tasks import DetectionModel
    cfg = str(HERE / "yolo11-spd.yaml")
    model = DetectionModel(cfg=cfg, ch=9, nc=2, verbose=True).cpu().eval()

    n_spd = sum(1 for m in model.modules() if type(m).__name__ == "SPDConv")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] Model built on CPU | input ch=9 | nc=2 | "
          f"SPDConv layers={n_spd} | params={n_params:,}")
    assert n_spd == 5, f"expected 5 backbone SPDConv layers, found {n_spd}"

    dummy_input = torch.randn(1, 9, 640, 640)
    print(f"[3] dummy_input = {tuple(dummy_input.shape)}  (stacked t-1,t,t+1)")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"[4] forward pass OK -> output shapes: {out_shapes(output)}")
    # detection head (eval) returns (preds[B, 4+nc, N], [P3,P4,P5 feature maps])
    if isinstance(output, (list, tuple)) and torch.is_tensor(output[0]):
        print(f"    predictions tensor: {tuple(output[0].shape)}  "
              f"(= [batch, 4+nc, anchors])")

    print("\n" + "=" * 72)
    print("[SUCCESS] 9-channel SPD-Conv backbone compiles and forward-passes on "
          "CPU.\n          Tensor dimensions are consistent end-to-end. No "
          "training was run.")
    print("=" * 72)


if __name__ == "__main__":
    main()
