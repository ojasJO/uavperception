#!/usr/bin/env python
"""Assemble the AeroTrack deployment package (M11) for a selected run, from artifacts only.

    venv\\Scripts\\python.exe tools\\deploy\\build_package.py --run v2_causal_g4_s0_e20 --weights runs\\hpc\\v2_causal_g4_s0_e20\\weights\\last.pt

Layout (deployment/<name>/): weights/ (last.pt, ONNX fp32/fp16, TensorRT engines built on this GPU), config/model_card.json,
tracker/ste_params.json (frozen on VAL before the TEST pass), benchmarks/ (deployment gate, smoke run), README.md (generated),
checksums.sha256, MANIFEST.json. Binaries are git-ignored; MANIFEST.json + checksums + README + json files are committed.
"""
import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    w = Path(a.weights)
    meta = json.loads((w.parent.parent / "run_meta.json").read_text(encoding="utf-8"))
    frozen = json.loads((REPO / "experiments/aerotrack_v2/frozen_ste_params.json").read_text(encoding="utf-8"))
    if frozen["candidate"] != a.run:
        raise SystemExit(f"frozen tracker parameters belong to {frozen['candidate']}, not {a.run}")
    if meta.get("last_pt_sha256") and meta["last_pt_sha256"] != sha(w):
        raise SystemExit("weights do not match the sha256 recorded at the end of training")
    name = a.name or f"aerotrack_{meta['config']}_v2"
    out = REPO / "deployment" / name
    for d in ("weights", "config", "tracker", "benchmarks"):
        (out / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(w, out / "weights" / "last.pt")
    shutil.copy2(w.parent.parent / "run_meta.json", out / "run_meta.json")   # the runtime reads the window from here
    art = REPO / "artifacts" / "deploy" / a.run
    for p in sorted(art.glob("*")):
        if p.suffix in (".onnx", ".engine"):
            shutil.copy2(p, out / "weights" / p.name)
    for p in sorted((REPO / "experiments/deployment/reports").glob(f"*{a.run}*")):
        shutil.copy2(p, out / "benchmarks" / p.name)
    (out / "tracker" / "ste_params.json").write_text(json.dumps(frozen["ste"], indent=1), encoding="utf-8")
    (out / "tracker" / "argmax_params.json").write_text(json.dumps(frozen["argmax"], indent=1), encoding="utf-8")
    card = {
        "model": a.run, "config": meta["config"], "seed": meta["seed"], "series": meta["series"],
        "architecture": "YOLO11n stride layout as built by Ultralytics 8.4.115 from c_model/yolo11-stride.yaml (C3k2 c3k=True; erratum "
                        "experiments/aerotrack_v2/ERRATUM_architecture.md), 9-channel input, 2 classes",
        "weights_sha256": sha(w), "trained_on": "HPC Tesla V100, fp16 autocast (amendment A1), corpus v2 TRAIN",
        "corpus": meta["data"], "temporal_window": {"kind": meta["window"], "gap": meta["gap"],
                                                     "frames": {"causal": "channels 0-2 = frame t-g, 3-5 = frame t, 6-8 = frame t-2g (experiments/aerotrack_v2/scripts/v2lib.window_samples)",
                                                                "symmetric": "channels 0-2 = t-g, 3-5 = t, 6-8 = t+g", "ttt": "t, t, t"}[meta["window"]],
                                                     "look_ahead_frames": 0 if meta["window"] in ("causal", "ttt") else meta["gap"],
                                                     "sequence_start": "missing past frames are replaced by the earliest available frame (training rule)"},
        "input": {"shape": [1, 9, 640, 640], "dtype": "float32 in [0, 1] (uint8 / 255)", "colour": "BGR as read by OpenCV, per frame",
                  "resize": "letterbox to 640 x 640 keeping aspect, pad value 114 (b_pipeline/stacked_dataset.letterbox_image)"},
        "classes": {"0": "uav_drone", "1": "bird_hard_negative"},
        "postprocess": {"nms_eval": {"conf": 0.001, "iou": 0.7, "max_det": 30}, "nms_stream": {"conf": 0.05, "iou": 0.7},
                        "boxes": "xyxy in letterboxed pixels -> undo letterbox to source pixels"},
        "tracker": {"kind": "STE (Kalman single-target state estimator)", "params": "tracker/ste_params.json",
                    "frozen": frozen["frozen"], "source_report": frozen["source_report"]},
        "evidence": {"val_decisions": "experiments/aerotrack_v2/hpc/reports/v2_decisions.json",
                     "test_single_pass": "experiments/aerotrack_v2/hpc/reports/final_test.json",
                     "deployment_gate": f"benchmarks/gate_{a.run}.json"},
        "runtime": {"recommended_backend": "trt_fp16 (TensorRT 11.3, engines are GPU-specific: rebuild on another GPU)",
                    "fallbacks": ["ort_cuda", "torch_fp16", "torch_fp32"], "command": f"python experiments/deployment/scripts/aerotrack_stream.py "
                    f"--source <video|folder> --weights deployment/{name}/weights/last.pt --backend trt_fp16 --ste-params deployment/{name}/tracker/ste_params.json"},
    }
    (out / "config" / "model_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("checksums.sha256", "MANIFEST.json", "README.md"))
    sums = {str(p.relative_to(out)).replace("\\", "/"): sha(p) for p in files}
    with open(out / "checksums.sha256", "w", encoding="utf-8", newline="\n") as fh:      # sha256sum -c needs LF
        fh.write("".join(f"{v}  {k}\n" for k, v in sums.items()))
    (out / "MANIFEST.json").write_text(json.dumps({"package": name, "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                   "host": platform.node(), "files": sums}, indent=1), encoding="utf-8")
    gate = json.loads((out / "benchmarks" / f"gate_{a.run}.json").read_text(encoding="utf-8"))
    rows = []
    for b, v in gate["backends"].items():
        if isinstance(v, dict) and "latency" in v:
            e = (v.get("e2e") or {}).get("per_frame_total", {})
            rows.append(f"| {b} | {v['latency']['p50_ms']:.2f} / {v['latency']['p99_ms']:.2f} | {e.get('p50_ms', float('nan')):.2f} / "
                        f"{e.get('p99_ms', float('nan')):.2f} |")
    readme = [f"# {name} — AeroTrack deployment package (generated by tools/deploy/build_package.py; do not edit)\n",
              f"Model `{a.run}` (sha256 `{card['weights_sha256'][:16]}…`), window **{meta['window']}** gap {meta['gap']} "
              f"(look-ahead {card['temporal_window']['look_ahead_frames']} frames), STE tracker with VAL-frozen parameters.\n",
              "## Run\n", f"```\n{card['runtime']['command']}\n```\n",
              "## Measured on the deployment GPU (RTX 5050 Laptop, batch 1; benchmarks/)\n",
              "| backend | model p50 / p99 ms | end-to-end p50 / p99 ms |", "|---|---:|---:|", *rows, "",
              "## Contract\n", "See `config/model_card.json` (input, preprocessing, class order, NMS, window, tracker). "
              "Verify files with `sha256sum -c checksums.sha256`.\n"]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"package {out.relative_to(REPO)}: {len(sums)} files")


if __name__ == "__main__":
    main()
