#!/usr/bin/env python
"""RTX 5050 deployment gate for one detector checkpoint: latency, memory and accuracy parity of every backend.

    venv\\Scripts\\python.exe experiments\\deployment\\scripts\\deploy_gate.py --weights runs\\gen_temporal_fp_g4_full_e20\\weights\\last.pt

Run on an otherwise idle GPU (the numbers are worthless next to a training job). Sections:
  latency   batch-1 model latency per backend on real VAL inputs: warm-up, then N timed calls, each bracketed by
            torch.cuda.synchronize(); mean / p50 / p95 / p99 / max, FPS = 1000 / mean, device memory used by the backend.
  e2e       the full per-frame loop on a real VAL sequence read from disk: JPEG decode -> letterbox -> ring buffer -> 9-ch window ->
            inference -> NMS (conf 0.05, IoU 0.7, max_det 30), timed as ONE loop (percentiles of the per-frame total, not sums of
            stage percentiles), plus the structural look-ahead the run's temporal window imposes.
  parity    the registered evaluation protocol (letterbox 640, conf 0.001, IoU 0.7, max_det 30) on a capped VAL set with each
            backend as the model -> drone AP50 per dataset; deltas vs torch_bf16 (the protocol the reported numbers used).
GPU clocks / temperature / power are recorded before and after. Output: experiments/deployment/reports/gate_<run>.{json,md}.
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
sys.path.insert(0, str(HERE))
import backends as B  # noqa: E402
import torch  # noqa: E402
from export_onnx import real_inputs, run_gap, windowed_samples  # noqa: E402

REPORTS = REPO / "experiments" / "deployment" / "reports"


def gpu_state():
    q = "name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,memory.total,pstate"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        return dict(zip(q.split(","), [v.strip() for v in out.strip().split(",")]))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def pct(a):
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(len(a)), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max()), "fps_from_mean": float(1000.0 / a.mean())}


def latency(backend, inputs, warmup, iters):
    for i in range(warmup):
        backend(inputs[i % len(inputs)])
    torch.cuda.synchronize()
    t = []
    for i in range(iters):
        x = inputs[i % len(inputs)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        backend(x)
        torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000.0)
    return pct(t)


def e2e(backend, gap, window, n_frames):
    """Per-frame streaming loop on the first Anti-UAV IR VAL sequence (sorted): the deployed computation, timed as one loop."""
    import cv2
    from ultralytics.utils.nms import non_max_suppression
    from stacked_dataset import letterbox_image, letterbox_params
    from train import MANIFEST, SPLITS
    from stacked_dataset import build_split_samples
    seqs = build_split_samples(MANIFEST, SPLITS, "val", temporal_gap=1, datasets=["anti_uav"])
    key = sorted({s["seq_key"] for s in seqs if s["modality"] == "IR"})[0]
    paths = [s["cur_path"] for s in sorted((s for s in seqs if s["seq_key"] == key), key=lambda s: s["frame_idx"])][:n_frames]
    span = 2 * gap if window == "causal" else gap            # frames that must be buffered behind t
    look_ahead = gap if window == "symmetric" else 0
    buf, stages, total = [], {"decode": [], "preprocess": [], "window": [], "infer": [], "nms": []}, []
    geom = None
    for k, p in enumerate(paths):
        t0 = time.perf_counter()
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        if geom is None:
            geom = letterbox_params(img.shape[0], img.shape[1], 640, 640)
        r, dw, dh, nw, nh = geom
        lb = letterbox_image(img, 640, 640, r, dw, dh, nw, nh)
        x = torch.from_numpy(np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))).cuda(non_blocking=True)
        x = x.permute(2, 0, 1).float().div_(255.0)
        buf.append(x)
        if len(buf) > span + look_ahead + 1:
            buf.pop(0)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        n = len(buf) - 1                                      # index of the newest frame
        if window == "causal":
            c = n
            stack = torch.cat([buf[max(c - gap, 0)], buf[c], buf[max(c - 2 * gap, 0)]], 0)
        elif window == "symmetric":
            c = max(n - look_ahead, 0)                        # the frame being detected lags the newest by `gap`
            stack = torch.cat([buf[max(c - gap, 0)], buf[c], buf[min(c + gap, n)]], 0)
        else:
            stack = torch.cat([buf[n]] * 3, 0)
        stack = stack.unsqueeze(0)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        y = backend(stack)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        non_max_suppression(y.detach().cpu(), 0.05, 0.7, max_det=30)
        t5 = time.perf_counter()
        if k >= 20:                                           # skip the warm-up frames
            for name, a, b in (("decode", t0, t1), ("preprocess", t1, t2), ("window", t2, t3), ("infer", t3, t4), ("nms", t4, t5)):
                stages[name].append((b - a) * 1000.0)
            total.append((t5 - t0) * 1000.0)
    return {"sequence": key, "frames_timed": len(total), "per_frame_total": pct(total), "stages": {k: pct(v) for k, v in stages.items()},
            "structural_look_ahead_frames": look_ahead, "structural_look_ahead_ms_at_25fps": look_ahead * 40.0}


def parity(backend, gap, cap, window):
    from torch.utils.data import DataLoader
    import evaluation as E
    from stacked_dataset import AeroTrackDataset, aerotrack_collate, build_split_samples, load_label_cache
    from train import LABEL_CACHE_PATH, MANIFEST, SPLITS, subsample
    base = windowed_samples("val", gap, window)                     # the run's own window (causal / symmetric / t,t,t)
    ss = subsample(base, {d: cap for d in ("anti_uav", "cst", "det_fly")})
    ds = AeroTrackDataset(ss, img_size=(640, 640), augment=None, letterbox=True, out_dtype="uint8", return_meta=True,
                          label_cache=load_label_cache(LABEL_CACHE_PATH))
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True, collate_fn=aerotrack_collate)
    recs = E.collect_records(backend, dl, torch.device("cuda:0"), imgsz=640, conf=0.001, iou=0.7, max_det=30, amp_dtype=None, progress=False)

    def ap(sub):
        r = E.score(sub, classes=(0,)) if sub else None
        return None if r is None else r["map50"]
    out = {"drone_AP50": ap(recs), "n_images": len(recs)}
    for d in ("anti_uav", "cst", "det_fly"):
        out[f"{d}_drone_AP50"] = ap([r for r in recs if r["dataset"] == d])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--onnx", default=None, help="default: artifacts/deploy/<run>/<run>_b1_op18.onnx")
    ap.add_argument("--backends", nargs="+", default=["torch_fp32", "torch_bf16", "torch_fp16", "ort_cuda", "trt_fp32", "trt_fp16"])
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--e2e-frames", type=int, default=400)
    ap.add_argument("--parity-cap", type=int, default=1000)
    ap.add_argument("--skip-parity", action="store_true")
    a = ap.parse_args()
    w = Path(a.weights).resolve()
    run = w.parent.parent.name
    onnx_path = Path(a.onnx) if a.onnx else REPO / "artifacts" / "deploy" / run / f"{run}_b1_op18.onnx"
    meta = json.loads((w.parent.parent / "run_meta.json").read_text(encoding="utf-8"))
    gap = run_gap(w)
    window = meta.get("window") or ("ttt" if gap == 0 else "symmetric")
    x, _ = real_inputs(gap, window=window)
    inputs = [x[i:i + 1].cuda() for i in range(len(x))]
    out = {"run": run, "weights": str(w.relative_to(REPO)), "onnx": str(onnx_path.relative_to(REPO)), "window": window, "gap": gap,
           "gpu_before": gpu_state(), "torch": torch.__version__, "backends": {}, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    for kind in a.backends:
        rec = {}
        torch.cuda.empty_cache()
        free0 = torch.cuda.mem_get_info()[0]
        t0 = time.perf_counter()
        try:
            be = B.make(kind, w, onnx_path)
            be(inputs[0])
            torch.cuda.synchronize()
        except B.BackendUnavailable as e:
            out["backends"][kind] = {"available": False, "reason": str(e)}
            print(f"[{kind}] not available: {e}", flush=True)
            continue
        rec["startup_s_incl_first_call"] = round(time.perf_counter() - t0, 2)
        rec["build_seconds"] = getattr(be, "build_seconds", None)
        torch.cuda.reset_peak_memory_stats()
        rec["latency"] = latency(be, inputs, a.warmup, a.iters)
        rec["device_memory_used_mb"] = round((free0 - torch.cuda.mem_get_info()[0]) / 2**20, 1)
        rec["torch_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        rec["e2e"] = e2e(be, gap, window, a.e2e_frames)
        if not a.skip_parity:
            t1 = time.time()
            rec["parity"] = parity(be, gap, a.parity_cap, window)
            rec["parity_seconds"] = round(time.time() - t1, 1)
        rec["available"] = True
        out["backends"][kind] = rec
        L = rec["latency"]
        print(f"[{kind}] model p50 {L['p50_ms']:.2f} / p95 {L['p95_ms']:.2f} / p99 {L['p99_ms']:.2f} ms | e2e p50 "
              f"{rec['e2e']['per_frame_total']['p50_ms']:.2f} p99 {rec['e2e']['per_frame_total']['p99_ms']:.2f} ms | "
              f"parity {rec.get('parity', {}).get('drone_AP50')}", flush=True)
        del be
    ref = out["backends"].get("torch_bf16", {}).get("parity")
    if ref:
        for k, v in out["backends"].items():
            if v.get("parity"):
                v["parity_delta_vs_torch_bf16"] = {m: (v["parity"][m] - ref[m]) if v["parity"].get(m) is not None and ref.get(m) is not None else None
                                                   for m in ref if m != "n_images"}
    out["gpu_after"] = gpu_state()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"gate_{run}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = [f"# Deployment gate — {run}", "", f"*Generated {out['generated']} by `scripts/deploy_gate.py` on {out['gpu_before'].get('name')} "
          f"(driver {out['gpu_before'].get('driver_version')}), window `{window}` (gap {gap}), batch 1, 640×640. GPU before: "
          f"{out['gpu_before'].get('temperature.gpu')} °C, SM {out['gpu_before'].get('clocks.sm')}; after: "
          f"{out['gpu_after'].get('temperature.gpu')} °C, SM {out['gpu_after'].get('clocks.sm')}.*", "",
          "| backend | model p50 / p95 / p99 ms | FPS (mean) | end-to-end p50 / p95 / p99 ms | device MB | startup s | drone AP50 (Δ vs bf16) |",
          "|---|---|---:|---|---:|---:|---|"]
    for k, v in out["backends"].items():
        if not v.get("available"):
            md.append(f"| {k} | not available: {v['reason'][:120]} | | | | | |")
            continue
        L, E2 = v["latency"], v["e2e"]["per_frame_total"]
        par = v.get("parity", {}).get("drone_AP50")
        dlt = (v.get("parity_delta_vs_torch_bf16") or {}).get("drone_AP50")
        md.append(f"| {k} | {L['p50_ms']:.2f} / {L['p95_ms']:.2f} / {L['p99_ms']:.2f} | {L['fps_from_mean']:.0f} | "
                  f"{E2['p50_ms']:.2f} / {E2['p95_ms']:.2f} / {E2['p99_ms']:.2f} | {v['device_memory_used_mb']:.0f} | "
                  f"{v['startup_s_incl_first_call']:.1f} | {'—' if par is None else f'{par:.4f}'}"
                  f"{'' if dlt is None else f' ({dlt:+.4f})'} |")
    any_e2e = next((v["e2e"] for v in out["backends"].values() if v.get("available")), None)
    if any_e2e:
        md += ["", f"Structural look-ahead of the `{window}` window: {any_e2e['structural_look_ahead_frames']} frames = "
               f"{any_e2e['structural_look_ahead_ms_at_25fps']:.0f} ms at 25 fps, on top of the per-frame compute above.",
               f"End-to-end loop: sequence `{any_e2e['sequence']}`, {any_e2e['frames_timed']} timed frames (decode from disk -> letterbox -> "
               "window -> inference -> NMS at conf 0.05)."]
    (REPORTS / f"gate_{run}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
