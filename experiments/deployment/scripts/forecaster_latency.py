#!/usr/bin/env python
"""Per-call CPU latency of a learned trajectory forecaster as the deployed tracker calls it (once per frame, batch 1).

    venv\\Scripts\\python.exe experiments\\deployment\\scripts\\forecaster_latency.py --name c1_tfm_abs_s0 [--iters 2000]

Builds the forecaster exactly as the runtime does (experiments/aerotrack_temporal/scripts/tc_ste.py::LearnedForecaster, same ONNX
Runtime session options, CPU provider) and times its session.run on a batch-1 input of the model's shape (the timing does not
depend on the values). Output: experiments/deployment/reports/forecaster_latency_<name>.json (p50 / p95 / p99 ms).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "aerotrack_temporal" / "scripts"))
import tc_ste as FS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    a = ap.parse_args()
    onnx = REPO / "runs" / "aerotrack_temporal" / "c1" / "onnx" / f"{a.name}.onnx"
    fc = FS.LearnedForecaster(onnx, True, None)
    inp = {i.name: np.random.default_rng(0).standard_normal([1] + list(i.shape[1:])).astype(np.float32) for i in fc.sess.get_inputs()}
    for _ in range(a.warmup):
        fc.sess.run(None, inp)
    t = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        fc.sess.run(None, inp)
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.asarray(t)
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "name": a.name, "onnx": str(onnx.relative_to(REPO).as_posix()),
           "host": platform.node(), "cpu": platform.processor(), "iters": a.iters, "batch": 1,
           "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)), "p99_ms": float(np.percentile(t, 99)),
           "mean_ms": float(t.mean())}
    path = REPO / "experiments" / "deployment" / "reports" / f"forecaster_latency_{a.name}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
