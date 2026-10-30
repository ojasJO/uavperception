"""Shared evaluation service (TRAIN / VAL only): model id -> verified checkpoint -> adapter -> task metric -> manifest -> history.

    from aerolab.evaluation import evaluate_sot, evaluate_forecast
    man = evaluate_sot("aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last", "cst2025", "val", "IR", ["building_41"], max_frames=100)

Every run writes runs/studio/evals/<experiment_id>/{manifest.json, per_sequence.json, predictions/<seq>.json, log.txt} and appends one
experiment record (kind 'studio_evaluation', provenance 'studio_run') to runs/studio/index.jsonl, which the studio's history page reads.
TEST splits are refused by aerolab.test_lock before anything is loaded. Truncated runs (max_frames) use protocol studio.sot_smoke.v1,
which is never comparable with a full-sequence mSA.
"""
import datetime as dt
import json
import platform
import subprocess
import traceback
import uuid
from pathlib import Path

import numpy as np

from . import schemas
from .adapters import get_adapter
from .data import load_sequence, trackc_sequences
from .hashing import sha256_file
from .paths import REPO, STUDIO_INDEX, STUDIO_RUNS
from .registry import Registry
from .test_lock import assert_split_allowed

DATASET_MANIFEST = {
    ("cst2025", None): "experiments/cst_anti_uav/reports/dataset_manifest.json",
    ("antiuav2021", "IR"): "experiments/anti_uav/manifests/antiuav_IR_{split}.manifest.json",
    ("antiuav2021", "RGB"): "experiments/anti_uav/manifests/antiuav_RGB_{split}.manifest.json",
    ("trackc_trajectories", None): "runs/aerotrack_temporal/_data/traj_{split}.json",
}


def _now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True,
                                    timeout=30).stdout.strip())
        return head or None, dirty
    except OSError:
        return None, None


def environment_reference(device):
    ref = {"python": platform.python_version(), "platform": platform.platform(), "device": device,
           "venv_freeze_reference": "environments/venv_freeze_2026-10-05.txt"}
    try:
        import torch
        ref.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    return ref


def dataset_hash(dataset_id, split, modality):
    p = DATASET_MANIFEST.get((dataset_id, modality)) or DATASET_MANIFEST.get((dataset_id, None))
    if not p:
        return "not_recorded"
    f = REPO / p.format(split=split)
    return f"{f.relative_to(REPO).as_posix()} sha256:{sha256_file(f)}" if f.exists() else f"not_recorded ({f.relative_to(REPO).as_posix()} missing)"


def sot_sequence_metrics(pred: dict, seq) -> dict:
    """Eq. 1 SA etc. with Track A's sequence_metrics over the processed frames from the start frame onward."""
    import sys
    src = str(REPO / "experiments" / "anti_uav" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from anti_uav_repro.eval.metrics import is_box, sequence_metrics
    n = pred["n_frames_processed"]
    start = pred["start_frame"]
    proc = np.asarray(pred["processed"][:n], dtype=bool)
    mask = proc & (np.arange(n) >= start)
    res = [r if r is not None else [] for r in pred["res_xywh"][:n]]
    m = sequence_metrics(res, list(seq.exist[:n]), list(seq.gt_rect[:n]), mask=mask)
    ex = np.asarray(seq.exist[:n], dtype=bool)
    has = np.array([is_box(r) for r in res], dtype=bool)
    absent, visible = mask & ~ex, mask & ex
    return {"SA": 100 * m["SA"], "precision_20px": 100 * m["precision_20px"] if m["precision_20px"] == m["precision_20px"] else None,
            "success_auc": 100 * m["success_auc"] if m["success_auc"] == m["success_auc"] else None,
            "n_frames_scored": int(mask.sum()), "n_visible": int(visible.sum()), "n_absent": int(absent.sum()),
            "false_alarm_rate": float((has & absent).sum() / absent.sum()) if absent.any() else None,
            "located_rate": float((has & visible).sum() / visible.sum()) if visible.any() else None,
            "full_sequence": bool(n == seq.n_frames)}


class _Run:
    def __init__(self, kind, model, dataset_id, split, modality):
        self.id = f"studio-{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.dir = STUDIO_RUNS / "evals" / self.id
        (self.dir / "predictions").mkdir(parents=True, exist_ok=True)
        self.log = open(self.dir / "log.txt", "w", encoding="utf-8")
        self.start = _now()
        self.kind, self.model, self.dataset_id, self.split, self.modality = kind, model, dataset_id, split, modality

    def write(self, msg):
        self.log.write(f"[{_now()}] {msg}\n")
        self.log.flush()


def _finish(run, man, per, warnings, status, error=None):
    run.write(f"status {status}" + (f": {error}" if error else ""))
    run.log.close()
    (run.dir / "per_sequence.json").write_text(json.dumps(per, indent=1), encoding="utf-8")
    man.update(completion_time=_now(), status=status, warnings=warnings, error=error)
    errs = schemas.errors(man, schemas.STUDIO_MANIFEST_SCHEMA)
    if errs:
        man["warnings"].append("manifest schema: " + "; ".join(errs))
    (run.dir / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    rec = {"schema": "aerolab/experiment/1", "experiment_id": run.id, "direction": run.model["direction"], "kind": "studio_evaluation",
           "provenance": "studio_run", "model_ids": [run.model["model_id"]], "checkpoint_sha256": run.model["sha256"],
           "dataset_id": run.dataset_id, "split": run.split, "modality": run.modality, "protocol_id": man["evaluation_protocol_id"],
           "metrics": man["metrics"], "per_sequence": {k: v.get("SA", v.get("ade_px")) for k, v in per.items() if isinstance(v, dict)},
           "seed": man["seed"], "git_commit": man["git_commit"], "status": status, "registration": None, "test_pass": False,
           "source_file": (run.dir / "manifest.json").relative_to(REPO).as_posix(), "max_frames": man.get("max_frames"),
           "sequence_set": sorted(per), "start": man["start_time"], "end": man["completion_time"], "notes": man.get("note"),
           "warnings": man["warnings"]}
    STUDIO_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(STUDIO_INDEX, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return man


def _base_manifest(run, model, proto, protocol_registry, preprocessing, device):
    head, dirty = git_state()
    return {"schema": "aerolab/studio_manifest/1", "experiment_id": run.id, "research_direction": model["direction"],
            "model_id": model["model_id"], "checkpoint_path": model["primary_path"], "checkpoint_sha256": model["sha256"],
            "model_config_path": (model.get("config") or {}).get("path"), "model_config_hash": (model.get("config") or {}).get("sha256"),
            "dataset_id": run.dataset_id, "dataset_version_or_manifest_hash": dataset_hash(run.dataset_id, run.split, run.modality),
            "split": run.split, "modality": run.modality, "evaluation_protocol_id": proto,
            "metric_definition_version": protocol_registry.get(proto, {}).get("metric_definition_version", "unknown"),
            "preprocessing_config": preprocessing, "seed": model.get("seed"), "git_commit": head, "git_dirty": dirty,
            "environment_reference": environment_reference(device), "device": device, "start_time": run.start, "completion_time": None,
            "status": "partial", "metrics": {}, "per_sample_results_path": (run.dir / "per_sequence.json").relative_to(REPO).as_posix(),
            "predictions_dir": (run.dir / "predictions").relative_to(REPO).as_posix(), "logs_path": (run.dir / "log.txt").relative_to(REPO).as_posix(),
            "registration_reference": None, "warnings": []}


def evaluate_sot(model_id, dataset_id, split, modality, sequences, max_frames=None, tracker="ste", device="auto", registry=None,
                 adapter=None, progress=None):
    """Single-target tracking evaluation on chosen VAL/TRAIN sequences. Returns the manifest (also written to disk)."""
    assert_split_allowed(split)
    R = registry or Registry()
    model = R.model(model_id)
    run = _Run("sot", model, dataset_id, split, modality)
    proto = "studio.sot_smoke.v1" if max_frames else "sot.mSA.antiuav_eq1.v1"
    per, warnings, man = {}, [], None
    try:
        a = adapter or get_adapter(model["adapter"], R)
        if a.model is None or a.record is None or a.record["model_id"] != model_id:
            a.load(model, device)
        run.write(f"loaded {model_id} ({a.get_metadata()})")
        man = _base_manifest(run, model, proto, R.protocols, None, a.device)
        man["checkpoint_path"] = a.get_metadata()["checkpoint_path"]
        man["max_frames"] = max_frames
        man["tracker"] = tracker
        for i, name in enumerate(sequences):
            seq = load_sequence(dataset_id, split, name, modality)
            pred = a.predict(seq, dataset_id=dataset_id, split=split, modality=modality, max_frames=max_frames, tracker=tracker)
            met = sot_sequence_metrics(pred, seq)
            met["timing_ms"] = pred["timing_ms"]
            per[name] = met
            for w in pred["warnings"]:
                if w not in warnings:
                    warnings.append(w)
            man["preprocessing_config"] = pred.get("preprocessing")
            man["tracker_params"], man["tracker_params_source"] = pred.get("params"), pred.get("params_source")
            (run.dir / "predictions" / f"{name}.json").write_text(json.dumps(pred), encoding="utf-8")
            run.write(f"{name}: SA {met['SA']:.2f} over {met['n_frames_scored']} frames")
            if progress:
                progress((i + 1) / len(sequences), name)
        sa = [v["SA"] for v in per.values()]
        man["metrics"] = {"mSA" if not max_frames else "mSA_partial": float(np.mean(sa)) if sa else None, "n_sequences": len(sa),
                          "mean_false_alarm_rate": _nanmean([v["false_alarm_rate"] for v in per.values()]),
                          "mean_located_rate": _nanmean([v["located_rate"] for v in per.values()]),
                          "mean_precision_20px": _nanmean([v["precision_20px"] for v in per.values()])}
        if max_frames:
            warnings.append(f"truncated to {max_frames} frames after the start frame: protocol studio.sot_smoke.v1, not comparable with full mSA")
        return _finish(run, man, per, warnings, "completed")
    except Exception as e:  # noqa: BLE001 -- recorded as a failed evaluation, never swallowed silently
        run.write(traceback.format_exc())
        man = man or _base_manifest(run, model, proto, R.protocols, None, device)
        return _finish(run, man, per, warnings, "failed", f"{type(e).__name__}: {e}")


def evaluate_forecast(model_id, sequence_keys, split="val", max_anchors=None, registry=None, adapter=None, progress=None):
    assert_split_allowed(split)
    R = registry or Registry()
    model = R.model(model_id)
    run = _Run("forecast", model, "trackc_trajectories", split, None)
    per, warnings, man = {}, [], None
    try:
        a = adapter or get_adapter(model["adapter"], R)
        if a.model is None or a.record is None or a.record["model_id"] != model_id:
            a.load(model)
        man = _base_manifest(run, model, "forecast.ADE_NLL.trackc.v1", R.protocols,
                             {"history": "ground truth (C1 oracle)", "L": a.meta["L"], "H": a.meta["H"]}, "cpu")
        man["max_anchors"] = max_anchors
        seqs = trackc_sequences(split, set(sequence_keys))
        for i, s in enumerate(seqs):
            r = a.predict(s, max_anchors=max_anchors)
            per[s["key"]] = {"domain": s["domain"], "n_anchors": r["n_anchors"], "ade_px": r["metrics"]["learned"]["ade_px"],
                             "nll": r["metrics"]["learned"].get("nll"),
                             **{f"ade_px_{k}": (v or {}).get("ade_px") for k, v in r["metrics"].items() if k != "learned"}}
            (run.dir / "predictions" / f"{s['key']}.json").write_text(json.dumps(r), encoding="utf-8")
            for w in r["warnings"]:
                if w not in warnings:
                    warnings.append(w)
            if progress:
                progress((i + 1) / len(seqs), s["key"])
        keys = [k for k in next(iter(per.values()), {}) if k.startswith("ade_px")]
        man["metrics"] = {f"mean_{k}": _nanmean([v[k] for v in per.values()]) for k in keys}
        man["metrics"]["n_sequences"] = len(per)
        man["metrics"]["note"] = "unweighted mean of per-sequence ADE (Track C reports pool points; not identical)"
        return _finish(run, man, per, warnings, "completed")
    except Exception as e:  # noqa: BLE001
        run.write(traceback.format_exc())
        man = man or _base_manifest(run, model, "forecast.ADE_NLL.trackc.v1", R.protocols, None, "cpu")
        return _finish(run, man, per, warnings, "failed", f"{type(e).__name__}: {e}")


def _nanmean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return float(np.mean(xs)) if xs else None
