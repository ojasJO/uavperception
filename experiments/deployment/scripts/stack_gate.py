#!/usr/bin/env python
"""Deployment gate of the COMPLETE stack (detector + tracker + forecaster) on the RTX 5050: per-stage and end-to-end latency over
several real VAL sequences per domain, through the one-command runtime (aerotrack_stream.py, drawing disabled).

    venv\\Scripts\\python.exe experiments\\deployment\\scripts\\stack_gate.py --weights runs\\<run>\\weights\\last.pt --backend trt_fp16 ^
        [--tracker fste --forecaster <spec.json> --fste-params <p.json>] [--seqs 2] [--frames 300]

Run on an otherwise idle GPU. Sequences: the first --seqs VAL sequences (sorted) of Anti-UAV IR, Anti-UAV RGB and CST (corpus v2
frame folders), --frames frames each; per frame the runtime records read+preprocess / window / infer / NMS / tracker (incl. the
forecaster) and the total; warm-up frames are excluded by the runtime. Output: experiments/deployment/reports/
stack_gate_<run>_<tracker>.{json,md} with per-domain and pooled p50 / p95 / p99, FPS and the structural look-ahead. VAL only.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REPORTS = REPO / "experiments" / "deployment" / "reports"
CST_VAL = REPO / "data" / "cst_anti_uav_full" / "CST-AntiUAV" / "CST-AntiUAV" / "val"
AU_FRAMES = REPO / "b_pipeline" / "data_unified" / "anti_uav" / "images"


def sources(n):
    out = {"anti_uav_IR": sorted(AU_FRAMES.glob("antiuav_val_*__IR"))[:n], "anti_uav_RGB": sorted(AU_FRAMES.glob("antiuav_val_*__RGB"))[:n],
           "cst_IR": sorted(p for p in CST_VAL.iterdir() if p.is_dir())[:n]}
    return {g: [p for p in v if p.is_dir()] for g, v in out.items()}


def pct(a):
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(len(a)), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "mean_ms": float(a.mean()), "fps_from_mean": float(1000.0 / a.mean())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--backend", default="trt_fp16")
    ap.add_argument("--tracker", choices=("ste", "fste"), default="ste")
    ap.add_argument("--forecaster", default=None)
    ap.add_argument("--fste-params", default=None)
    ap.add_argument("--ste-params", default=None)
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    run = Path(a.weights).parent.parent.name
    if not (Path(a.weights).parent.parent / "run_meta.json").exists():   # the runtime would silently default the window
        raise SystemExit(f"run_meta.json missing next to {a.weights}: the stream runtime reads the temporal window from it")
    tmp = REPO / "runs" / "deploy_stream" / f"_stack_gate_{run}_{a.tracker}"
    tmp.mkdir(parents=True, exist_ok=True)
    per_group, summaries = {}, []
    for grp, dirs in sources(a.seqs).items():
        stages = {}
        for d in dirs:
            out = tmp / f"{grp}__{d.name}.mp4"
            cmd = [sys.executable, "-u", str(HERE / "aerotrack_stream.py"), "--source", str(d), "--weights", a.weights, "--backend", a.backend,
                   "--tracker", a.tracker, "--no-video", "--max-frames", str(a.frames), "--out", str(out)]
            for flag, v in (("--forecaster", a.forecaster), ("--fste-params", a.fste_params), ("--ste-params", a.ste_params)):
                if v:
                    cmd += [flag, v]
            rc = subprocess.call(cmd, cwd=str(REPO))
            if rc != 0:
                raise SystemExit(f"stream failed on {d} (rc {rc})")
            summ = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
            if summ["backend"] != a.backend:          # the runtime falls back silently (e.g. no engine for this run name)
                raise SystemExit(f"{d.name}: requested backend {a.backend} but the runtime used {summ['backend']}; "
                                 f"expected engines under artifacts/deploy/{run}/")
            summaries.append({"group": grp, "sequence": d.name, "backend": summ["backend"], "look_ahead_ms": summ["look_ahead_ms"],
                              "modes": summ["modes"]})
            for line in out.with_suffix(".frames.jsonl").read_text(encoding="utf-8").splitlines()[10:]:     # same warm-up rule
                for k, v in json.loads(line)["timings_ms"].items():
                    stages.setdefault(k, []).append(v)
        per_group[grp] = {k: pct(v) for k, v in stages.items()}
    allst = {}
    for grp, dirs in sources(a.seqs).items():
        for d in dirs:
            for line in (tmp / f"{grp}__{d.name}.frames.jsonl").read_text(encoding="utf-8").splitlines()[10:]:
                for k, v in json.loads(line)["timings_ms"].items():
                    allst.setdefault(k, []).append(v)
    pooled = {k: pct(v) for k, v in allst.items()}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    rep = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": run, "backend_requested": a.backend, "tracker": a.tracker,
           "forecaster": a.forecaster, "frames_per_sequence": a.frames, "sequences": summaries, "per_group": per_group, "pooled": pooled,
           "look_ahead_ms": summaries[0]["look_ahead_ms"] if summaries else None, "gpu_after": gpu}
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"stack_gate_{run}_{a.tracker}.json"
    path.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = [f"# Complete-stack deployment gate — {run} ({a.backend}, tracker {a.tracker})", "",
          f"*Generated {rep['generated']} by `scripts/stack_gate.py`: {len(summaries)} VAL sequences, {a.frames} frames each, drawing off; "
          f"structural look-ahead {rep['look_ahead_ms']:.0f} ms on top of the per-frame compute. GPU: {gpu}.*", "",
          "| group | stage | p50 ms | p95 ms | p99 ms |", "|---|---|---:|---:|---:|"]
    for grp, st in list(per_group.items()) + [("pooled", pooled)]:
        for k, v in st.items():
            md.append(f"| {grp} | {k} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} | {v['p99_ms']:.2f} |")
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
