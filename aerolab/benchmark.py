"""Controlled local speed benchmark (protocol speed.bench.local.v1) -- separate from the live demonstration's observed FPS.

Method, recorded in every manifest:
- one model at a time; for each requested device the model is loaded fresh (init_s = adapter.load wall time, incl. SHA-256 check)
- the frames are a contiguous run of one VAL / TRAIN sequence from its start frame, processed by the adapter's stream() at batch 1
- the first frame's latency is reported separately (cold: cuDNN / allocator warm-up); the next `warmup` frames are discarded
- the remaining frames x `repeats` give the latency distribution; stage boundaries per frame:
  read_preprocess (JPEG decode + letterbox / transform + tensor) | model (forward, torch.cuda.synchronize() before the timer stops on
  CUDA) | postprocess (NMS / rescale) | tracker (STE / argmax step) | total = their sum. No drawing / display is included.
- memory: torch.cuda.max_memory_allocated after reset (GPU); process RSS (psutil) after the run; CPU time / wall time = average cores used
Percentiles are reported only with enough samples (p95 >= 20, p99 >= 100). PyTorch eager, not TensorRT: the deployment gate
(experiments/deployment, deploy.latency.v1) is a different setup and is never mixed with these numbers.
"""
import datetime as dt
import json
import time
import traceback
import uuid

from . import analysis, categories, data
from .adapters import get_adapter
from .adapters.base import AdapterError, pick_device
from .evaluation import environment_reference, git_state
from .paths import REPO, STUDIO_INDEX, STUDIO_RUNS
from .registry import Registry
from .test_lock import assert_split_allowed

SCHEMA = "aerolab/flightdeck_benchmark/1"
STAGES = ("read_preprocess", "model", "postprocess", "tracker", "total")


def build_config(dataset_id, split, modality, sequence, models, devices=("auto",), n_frames=100, warmup=10, repeats=1):
    return {"category": "speed", "mode": "benchmark", "dataset_id": dataset_id, "split": split, "modality": modality, "sequence": sequence,
            "models": [{"model_id": m["model_id"], "tracker": m.get("tracker")} for m in models], "devices": list(devices),
            "n_frames": int(n_frames), "warmup": int(warmup), "repeats": int(repeats)}


def validate(cfg, R):
    errs = []
    try:
        assert_split_allowed(cfg["split"])
    except PermissionError as e:
        return [str(e)]
    if not cfg["models"]:
        errs.append("select at least one model")
    for m in cfg["models"]:
        rec = R.by_id.get(m["model_id"])
        if rec is None:
            errs.append(f"unknown model {m['model_id']}")
            continue
        ok, why = categories.eligibility("speed", rec)
        if not ok:
            errs.append(f"{m['model_id']}: {why}")
    for d in cfg["devices"]:
        if d not in ("auto", "cpu", "cuda"):
            errs.append(f"unknown device {d!r}")
        elif d == "cuda":
            try:
                pick_device("cuda")
            except AdapterError as e:
                errs.append(f"{e}. Choose 'cpu'.")
    if cfg["n_frames"] < 5 or cfg["warmup"] < 0 or cfg["repeats"] < 1:
        errs.append("n_frames >= 5, warmup >= 0, repeats >= 1")
    if cfg["n_frames"] * cfg["repeats"] * len(cfg["models"]) * len(cfg["devices"]) > 20000:
        errs.append("more than 20,000 timed frames in total: reduce frames, repeats, models or devices")
    return errs


def run_benchmark(cfg, R=None, on_event=None, cancel=None):
    import psutil
    import torch
    R = R or Registry()
    errs = validate(cfg, R)
    if errs:
        raise ValueError("; ".join(errs))
    emit = on_event or (lambda e: None)
    eid = f"bench-{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = STUDIO_RUNS / "evals" / eid
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    emit({"type": "started", "experiment_id": eid, "dir": out.relative_to(REPO).as_posix()})
    head, dirty = git_state()
    seq = data.load_sequence(cfg["dataset_id"], cfg["split"], cfg["sequence"], cfg["modality"])
    end = min(seq.n_frames, seq.start_frame + 1 + cfg["warmup"] + cfg["n_frames"])
    proc = psutil.Process()
    results, raw, cancelled = [], {}, False
    for m in cfg["models"]:
        rec = R.model(m["model_id"])
        for dev in cfg["devices"]:
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            key = f"{rec['run_id']}@{rec['checkpoint_tag']}:{m.get('tracker')}:{dev}"
            r = {"model_id": rec["model_id"], "tracker": m.get("tracker"), "device_requested": dev, "status": "running", "errors": 0}
            a = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                cpu0, w0 = proc.cpu_times(), time.perf_counter()
                t0 = time.perf_counter()
                a = get_adapter(rec["adapter"], R).load(rec, dev)
                r["init_s"] = time.perf_counter() - t0
                r["device"] = a.device
                frames = []
                first = None
                for rep in range(cfg["repeats"]):
                    kw = {"end_frame": end, "dataset_id": cfg["dataset_id"], "modality": cfg["modality"]}
                    if m.get("tracker") in ("ste", "argmax"):
                        kw["tracker"] = m["tracker"]
                    for k, fr in enumerate(a.stream(seq, **kw)):
                        if rep == 0 and k == 0:
                            first = fr["timing_ms"]["total"]
                        if k > cfg["warmup"]:                 # frame 0 = init (cold) + warm-up frames are not timed
                            frames.append(fr["timing_ms"])
                        emit({"type": "frame", "slug": key, "model_id": rec["model_id"], "sequence": seq.name, "record": fr,
                              "frames_done": len(frames), "frames_total": cfg["n_frames"] * cfg["repeats"], "elapsed_s": time.perf_counter() - t0,
                              "device": a.device})
                        if cancel is not None and cancel.is_set():
                            cancelled = True
                            break
                    if cancelled:
                        break
                cpu1, w1 = proc.cpu_times(), time.perf_counter()
                r["first_frame_ms"] = first
                r["stages"] = {s: analysis.latency_stats([f[s] for f in frames]) for s in STAGES}
                r["n_timed_frames"] = len(frames)
                med = r["stages"]["total"].get("median")
                r["fps_from_median_total"] = 1000 / med if med else None
                r["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if (torch.cuda.is_available() and a.device == "cuda") else None
                r["rss_mb"] = proc.memory_info().rss / 2 ** 20
                r["avg_cpu_cores_used"] = ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) / max(w1 - w0, 1e-9)
                r["cuda_synchronized"] = a.device == "cuda"
                r["status"] = "cancelled" if cancelled else "completed"
                raw[key] = frames
            except Exception as e:  # noqa: BLE001
                r["status"], r["error"] = "failed", f"{type(e).__name__}: {e}"
                r["trace"] = traceback.format_exc()[-1500:]
            finally:
                if a is not None:
                    a.unload()
            results.append(r)
            emit({"type": "model_done", "slug": key, "status": r["status"]})
        if cancelled:
            break
    man = {"schema": SCHEMA, "experiment_id": eid, "category": "speed", "mode": "controlled benchmark", "status": "cancelled" if cancelled else
           ("completed" if all(r["status"] == "completed" for r in results) else "partial"), "evaluation_protocol_id": "speed.bench.local.v1",
           "dataset_id": cfg["dataset_id"], "split": cfg["split"], "modality": cfg["modality"], "sequence": seq.name,
           "frames": {"start": seq.start_frame, "end": end, "warmup": cfg["warmup"], "timed_per_repeat": max(0, end - seq.start_frame - 1 - cfg["warmup"]),
                      "repeats": cfg["repeats"]},
           "method": __doc__.split("Method, recorded in every manifest:")[1].strip(), "results": results, "git_commit": head, "git_dirty": dirty,
           "environment_reference": environment_reference(",".join(cfg["devices"])), "start_time": None, "completion_time":
           dt.datetime.now().astimezone().isoformat(timespec="seconds"), "outputs": {"dir": out.relative_to(REPO).as_posix()}, "test_pass": False}
    (out / "timing.json").write_text(json.dumps(raw), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    with open(STUDIO_INDEX, "a", encoding="utf-8") as f:
        for r in results:
            rec = R.model(r["model_id"])
            f.write(json.dumps({"schema": "aerolab/experiment/1", "experiment_id": f"{eid}/{rec['run_id']}:{r['tracker']}:{r['device_requested']}",
                                "experiment_group": eid, "direction": rec["direction"], "kind": "studio_evaluation", "provenance": "studio_run",
                                "model_ids": [rec["model_id"]], "checkpoint_sha256": rec["sha256"], "dataset_id": cfg["dataset_id"],
                                "split": cfg["split"], "modality": cfg["modality"], "protocol_id": "speed.bench.local.v1",
                                "metrics": {"median_ms": (r.get("stages") or {}).get("total", {}).get("median"),
                                            "model_median_ms": (r.get("stages") or {}).get("model", {}).get("median"),
                                            "p95_ms": (r.get("stages") or {}).get("total", {}).get("p95"), "init_s": r.get("init_s"),
                                            "first_ms": r.get("first_frame_ms"), "peak_gpu_mb": r.get("peak_gpu_mb"), "device": r.get("device")},
                                "per_sequence": None, "seed": rec.get("seed"), "git_commit": head, "status": r["status"], "registration": None,
                                "test_pass": False, "source_file": f"{out.relative_to(REPO).as_posix()}/manifest.json", "category": "speed",
                                "warnings": []}) + "\n")
    emit({"type": "finished", "experiment_id": eid, "status": man["status"]})
    return man


def list_benchmarks():
    out = []
    for p in sorted((STUDIO_RUNS / "evals").glob("bench-*/manifest.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return out
