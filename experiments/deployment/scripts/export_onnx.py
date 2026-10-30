#!/usr/bin/env python
"""Export an AeroTrack detector checkpoint to ONNX and prove the graph computes the same function.

    venv\\Scripts\\python.exe experiments\\deployment\\scripts\\export_onnx.py --weights runs\\gen_temporal_fp_g4_full_e20\\weights\\last.pt

What is exported is exactly what the evaluation scores: the EMA weights (`load_checkpoint(prefer_ema=True)`), Conv+BN fused,
Ultralytics Detect head in export mode -> one output [B, 4 + nc, 8400] (xywh in input pixels, sigmoid class scores), the
tensor `c_model/evaluation.collect_records` hands to NMS. Input: [B, 9, 640, 640] float in [0, 1] (three letterboxed RGB
frames stacked as the run's temporal input). Pre/post-processing stay outside the graph, identical for every backend.

Checks (all recorded in <out>/export_record.json):
  * onnx.checker on the saved model; op-type inventory (custom ops would show up here — SPDConv lowers to Slice + Concat)
  * PyTorch fp32 vs ONNX Runtime CPU fp32 on REAL validation inputs (registered VAL set, first images per dataset),
    max |diff| of boxes (px) and class scores
  * parameter count, GFLOPs at 640 (thop), file sizes, SHA-256 of checkpoint and ONNX file
Output goes to artifacts/deploy/<run>/ (ignored by git; the record JSON is copied to experiments/deployment/reports/).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "c_model", REPO / "b_pipeline", REPO / "experiments" / "anti_uav" / "src"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run_gap(weights):
    """Temporal gap the run was trained with (run_meta.json), 0 = t,t,t."""
    meta = Path(weights).parent.parent / "run_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        for k in ("gap", "temporal_gap"):
            if k in m:
                return int(m[k])
    return 1


def run_window(weights):
    """The run's 9-channel window: run_meta `window` (v2 series), else t,t,t for gap 0 and symmetric otherwise."""
    meta = Path(weights).parent.parent / "run_meta.json"
    m = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    return m.get("window") or ("ttt" if run_gap(weights) == 0 else "symmetric")


def windowed_samples(split, gap, window):
    """All samples of a split with the run's window (same rule as aerotrack_v2/scripts/v2lib.window_samples)."""
    from stacked_dataset import build_split_samples
    from train import MANIFEST, SPLITS
    if window == "ttt" or gap == 0:
        base = build_split_samples(MANIFEST, SPLITS, split, temporal_gap=1)
        return [dict(s, prev_path=s["cur_path"], next_path=s["cur_path"]) for s in base]
    a = build_split_samples(MANIFEST, SPLITS, split, temporal_gap=gap)
    if window == "symmetric":
        return a
    b = build_split_samples(MANIFEST, SPLITS, split, temporal_gap=2 * gap)
    return [dict(x, next_path=y["prev_path"]) for x, y in zip(a, b)]          # causal: [t-g, t, t-2g]


def real_inputs(gap, n_per_dataset=4, window=None):
    """A few REAL letterboxed VAL inputs, built exactly as the evaluation builds them (registered VAL set order)."""
    from stacked_dataset import AeroTrackDataset
    from train import subsample
    window = window or ("ttt" if gap == 0 else "symmetric")
    base = windowed_samples("val", gap, window)
    ss = subsample(base, {d: n_per_dataset for d in ("anti_uav", "cst", "det_fly")})
    ds = AeroTrackDataset(ss, img_size=(640, 640), augment=None, letterbox=True, out_dtype="uint8", return_meta=True)
    xs = [ds[i][0] for i in range(len(ds))]
    return torch.stack(xs).float().div_(255.0), [s["seq_key"] for s in ss]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--out-dir", default=None, help="default artifacts/deploy/<run>")
    ap.add_argument("--dynamic-batch", action="store_true", help="export a dynamic batch axis (default: static batch 1)")
    ap.add_argument("--fp16", action="store_true", help="export a half-precision graph (TensorRT >= 11 builds FP16 engines only "
                                                         "from FP16-typed networks); traced on the GPU, parity vs PyTorch fp16")
    a = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from aerotrack_trainer import load_checkpoint
    from ultralytics.nn.modules import Detect

    w = Path(a.weights).resolve()
    run = w.parent.parent.name
    out = Path(a.out_dir) if a.out_dir else REPO / "artifacts" / "deploy" / run
    out.mkdir(parents=True, exist_ok=True)
    gap = run_gap(w)
    model, ck = load_checkpoint(w, torch.device("cpu"), prefer_ema=True, fuse=True)
    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True          # single tensor output, as the Ultralytics exporter does
            m.format = "onnx"
    # network input channels from the model definition -- NOT the first Conv2d, which in the SPD layout sees 4 x ch
    # channels after space-to-depth
    ch = int((getattr(model, "yaml", None) or {}).get("ch", 9))
    n_params = sum(p.numel() for p in model.parameters())
    try:
        import copy

        from thop import profile
        macs, _ = profile(copy.deepcopy(model), inputs=(torch.zeros(1, ch, 640, 640),), verbose=False)  # thop adds buffers
        gflops = 2 * macs / 1e9
    except Exception as e:  # noqa: BLE001
        gflops = None
        print(f"[warn] FLOPs not computed: {e}")

    onnx_path = out / f"{run}_{'dyn' if a.dynamic_batch else 'b1'}_op{a.opset}{'_fp16' if a.fp16 else ''}.onnx"
    dev = torch.device("cuda:0" if a.fp16 else "cpu")
    if a.fp16:
        model = model.to(dev).half()
    dummy = torch.zeros(1, ch, 640, 640, device=dev, dtype=torch.float16 if a.fp16 else torch.float32)
    t0 = time.time()
    try:
        torch.onnx.export(model, (dummy,), str(onnx_path), opset_version=a.opset, input_names=["images"], output_names=["preds"],
                          dynamic_axes={"images": {0: "batch"}, "preds": {0: "batch"}} if a.dynamic_batch else None,
                          do_constant_folding=True, dynamo=False)
        exporter = f"torch.onnx TorchScript exporter, torch {torch.__version__}"
    except Exception as e:  # noqa: BLE001 — legacy exporter removed/broken: use the dynamo exporter, and say so
        print(f"[export] TorchScript exporter failed ({type(e).__name__}: {str(e)[:200]}); using dynamo=True")
        torch.onnx.export(model, (dummy,), str(onnx_path), opset_version=a.opset, input_names=["images"], output_names=["preds"],
                          dynamic_shapes=({0: torch.export.Dim("batch")},) if a.dynamic_batch else None, dynamo=True)
        exporter = f"torch.onnx dynamo exporter, torch {torch.__version__}"
    export_s = time.time() - t0
    om = onnx.load(str(onnx_path))
    onnx.checker.check_model(om)
    ops = sorted({n.op_type for n in om.graph.node})

    # ---- parity on real inputs: PyTorch fp32 vs ONNX Runtime CPU fp32
    x, keys = real_inputs(gap, window=run_window(w))
    xd = x.to(dev).half() if a.fp16 else x
    with torch.no_grad():
        ref = torch.cat([model(xd[i:i + 1]) for i in range(len(x))]).float().cpu().numpy()
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if a.fp16 else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=prov)
    feed = (lambda i: x[i:i + 1].half().numpy()) if a.fp16 else (lambda i: x[i:i + 1].numpy())
    got = np.concatenate([sess.run(None, {"images": feed(i)})[0] for i in range(len(x))]).astype(np.float32)
    d_box_all = float(np.abs(ref[:, :4] - got[:, :4]).max())
    # boxes that matter = anchors the reference scores >= 0.05 (background anchors' box regression is ill-conditioned
    # in half precision and never reaches NMS output)
    live = ref[:, 4:].max(1) >= 0.05                                   # [n, anchors]
    d_box = float(np.abs(ref[:, :4] - got[:, :4]).transpose(0, 2, 1)[live].max()) if live.any() else 0.0
    d_cls = float(np.abs(ref[:, 4:] - got[:, 4:]).max())

    def top_box(a, i):
        j = a[i, 4:].max(0).argmax()
        cx, cy, w_, h_ = a[i, :4, j]
        return np.array([cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2])

    def iou(a, b):
        ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy + 1e-9)
    # downstream check: every image keeps the same top-scoring detection -- the same anchor (fp32) or, in fp16 where
    # near-tied neighbouring anchors can swap, a box on the same object (IoU >= 0.9)
    same_top = int(sum((ref[i, 4:].max(0).argmax() == got[i, 4:].max(0).argmax()) or
                       (a.fp16 and iou(top_box(ref, i), top_box(got, i)) >= 0.9) for i in range(len(x))))
    # tolerances: boxes are in input pixels (values up to 640), so fp32 rounding through ~100 layers gives ~1e-5 relative
    # = ~0.01 px; 1/20 px is far below anything NMS or IoU can see. Scores are probabilities. fp16: half-precision rounding.
    TOL_BOX_PX, TOL_SCORE = (1.0, 5e-3) if a.fp16 else (0.05, 1e-4)
    ok = d_box < TOL_BOX_PX and d_cls < TOL_SCORE and same_top == len(x)

    rec = {"run": run, "weights": str(w.relative_to(REPO)), "weights_sha256": sha256(w), "checkpoint_epoch": ck.get("epoch"),
           "temporal_gap": gap, "window": run_window(w), "input": {"name": "images", "shape": [1 if not a.dynamic_batch else "batch", ch, 640, 640],
                                          "dtype": "float32", "range": "[0, 1]", "layout": "RGB x3 frames (t-g, t, t+g)"},
           "output": {"name": "preds", "shape": list(ref.shape[1:]), "content": "xywh (input px) + sigmoid class scores"},
           "onnx": {"path": str(onnx_path.relative_to(REPO)), "sha256": sha256(onnx_path), "bytes": onnx_path.stat().st_size,
                    "opset": a.opset, "exporter": exporter, "onnx": onnx.__version__,
                    "op_types": ops, "export_seconds": round(export_s, 1)},
           "model": {"params": n_params, "gflops_640": gflops, "checkpoint_bytes": w.stat().st_size, "ema": ck.get("ema") is not None,
                     "fused": True},
           "precision": "fp16" if a.fp16 else "fp32",
           "parity_ort_cpu_fp32" if not a.fp16 else "parity_ort_cuda_fp16_vs_torch_fp16": {"n_inputs": len(x), "inputs": "registered VAL order, first 4 images per dataset",
                                   "max_abs_diff_box_px": d_box, "max_abs_diff_box_px_all_anchors": d_box_all,
                                   "box_diff_scope": "anchors with reference score >= 0.05", "max_abs_diff_score": d_cls,
                                   "same_top_detection": same_top,
                                   "tolerance": {"box_px": TOL_BOX_PX, "score": TOL_SCORE, "same_top_anchor": "all"},
                                   "onnxruntime": ort.__version__, "pass": ok},
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "export_record.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    rep = REPO / "experiments" / "deployment" / "reports" / f"export_{run}{'_fp16' if a.fp16 else ''}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k.startswith("parity") or k in ("run", "temporal_gap", "precision")}, indent=1))
    print(f"ops: {ops}\nparams {n_params:,}  GFLOPs {gflops}  onnx {rec['onnx']['bytes'] / 1e6:.1f} MB -> {onnx_path}")
    if not ok:
        sys.exit("PARITY FAILED")


if __name__ == "__main__":
    main()
