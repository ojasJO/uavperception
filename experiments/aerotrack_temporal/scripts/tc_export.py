#!/usr/bin/env python
"""Export a Track C forecaster checkpoint to ONNX (opset 18) and check parity with PyTorch (component gate: "exportable?").

    venv\\Scripts\\python.exe experiments\\aerotrack_temporal\\scripts\\tc_export.py --name c1_mlp_abs_s0 [--name ...]

Inputs of the exported graph: seq [n, 16, 4], glob [n, 3 | 1] (tc_learned.features); outputs: mean [n, 24, 2], log sigma [n, 24, 2]
in percent-of-width units. Parity on TRAIN-dev samples (no VAL), ONNX Runtime CPU; also times batch-1 CPU inference.
Output: results/runs/aerotrack_temporal/c1/onnx/<name>.onnx and experiments/aerotrack_temporal/reports/export_forecasters.json (appended).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import tc_data as D  # noqa: E402
import tc_forecast as F  # noqa: E402
import tc_learned as M  # noqa: E402

RUN = REPO / "results/runs" / "aerotrack_temporal" / "c1"
OUT = REPO / "experiments" / "aerotrack_temporal" / "reports" / "export_forecasters.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True)
    a = ap.parse_args()
    import onnxruntime as ort
    dev = [s for s in D.load_split("train") if s["dev"]][:8]
    S = F.make_samples(dev, stride=7)
    rep = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in a.name:
        model, meta = M.load(RUN / f"{name}.pt")
        model.eval()
        seq, glob, *_ = M.features(S, abs_pos=meta["abs_pos"], dom_tok=meta.get("dom_tok", False))
        path = RUN / "onnx" / f"{name}.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model, (seq[:2], glob[:2]), str(path), input_names=["seq", "glob"], output_names=["mean", "log_sigma"],
                          dynamic_axes={"seq": {0: "n"}, "glob": {0: "n"}, "mean": {0: "n"}, "log_sigma": {0: "n"}}, opset_version=18,
                          dynamo=False)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            tm, tl = (x.numpy() for x in model(seq, glob))
        om, ol = sess.run(None, {"seq": seq.numpy(), "glob": glob.numpy()})
        one = {"seq": seq[:1].numpy(), "glob": glob[:1].numpy()}
        for _ in range(20):
            sess.run(None, one)
        t0 = time.perf_counter()
        for _ in range(500):
            sess.run(None, one)
        ms = (time.perf_counter() - t0) / 500 * 1e3
        rec = {"checked": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "onnx": str(path.relative_to(REPO)),
               "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "opset": 18, "n_samples": int(len(seq)),
               "max_abs_diff_mean": float(np.abs(tm - om).max()), "max_abs_diff_log_sigma": float(np.abs(tl - ol).max()),
               "ort_cpu_batch1_ms": round(ms, 4), "params": int(sum(p.numel() for p in model.parameters())),
               "parity_ok": bool(np.abs(tm - om).max() < 1e-4 and np.abs(tl - ol).max() < 1e-4)}
        rep[name] = rec
        print(name, json.dumps(rec))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
