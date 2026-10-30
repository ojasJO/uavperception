"""Category F runner: Track C forecasters on VAL trajectories (protocol forecast.ADE_NLL.trackc.v1), 1-4 models on the same anchors.

Anchors come from tc_forecast.make_samples (deterministic for a given history length L and horizon H), so models with the same
L / H are evaluated on identical anchors; a model with different L / H is flagged non-equivalent. Classical references cp / cv /
dcv (TRAIN-tuned) are computed on the same anchors by the adapter. Metrics are pixels (lower is better) and NLL (lower is better);
they are never mixed with tracking mSA. Outputs: results/runs/studio/evals/fc-<id>/ (manifest, per_sequence, predictions/<slug>/<key>.json).
"""
import datetime as dt
import json
import traceback
import uuid

import numpy as np

from . import categories
from .adapters import get_adapter
from .data import trackc_sequences
from .evaluation import environment_reference, git_state
from .paths import REPO, STUDIO_INDEX, STUDIO_RUNS
from .registry import Registry
from .test_lock import assert_split_allowed

SCHEMA = "src/aerolab/flightdeck_forecast/1"


def build_config(sequence_keys, model_ids, max_anchors=500, keep_anchors=60, split="val"):
    return {"category": "forecasting", "dataset_id": "trackc_trajectories", "split": split, "sequences": list(sequence_keys),
            "models": [{"model_id": m, "tracker": None} for m in model_ids], "max_anchors": max_anchors, "keep_anchors": keep_anchors}


def validate(cfg, R):
    errs = []
    try:
        assert_split_allowed(cfg["split"])
    except PermissionError as e:
        return [str(e)]
    if not cfg["sequences"]:
        errs.append("select at least one trajectory sequence")
    if not 1 <= len(cfg["models"]) <= 4:
        errs.append("select 1-4 forecasters")
    for m in cfg["models"]:
        rec = R.by_id.get(m["model_id"])
        if rec is None:
            errs.append(f"unknown model {m['model_id']}")
            continue
        ok, why = categories.eligibility("forecasting", rec)
        if not ok:
            errs.append(f"{m['model_id']}: {why}")
    return errs


def run_forecast(cfg, R=None, on_event=None, cancel=None):
    R = R or Registry()
    errs = validate(cfg, R)
    if errs:
        raise ValueError("; ".join(errs))
    emit = on_event or (lambda e: None)
    eid = f"fc-{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = STUDIO_RUNS / "evals" / eid
    (out / "predictions").mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    emit({"type": "started", "experiment_id": eid})
    head, dirty = git_state()
    seqs = trackc_sequences(cfg["split"], set(cfg["sequences"]))
    states, per, cancelled = [], {}, False
    for i, m in enumerate(cfg["models"]):
        rec = R.model(m["model_id"])
        sl = f"m{i}_{rec['run_id']}"
        st = {"slug": sl, "model_id": rec["model_id"], "status": "running", "direction": rec["direction"]}
        states.append(st)
        per[sl] = {}
        a = None
        try:
            a = get_adapter(rec["adapter"], R).load(rec)
            st.update(device="cpu", load_seconds=a.load_s, checkpoint_sha256=rec["sha256"], L=a.meta["L"], H=a.meta["H"],
                      config={k: a.meta.get(k) for k in ("arch", "variant", "L", "H", "abs_pos", "dom_tok", "aug", "seed")})
            for s in seqs:
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                c = a.can_run(rec, "trackc_trajectories", domain=s["domain"])
                if not c.ok:
                    per[sl][s["key"]] = {"status": "skipped", "reason": c.reason}
                    continue
                r = a.predict(s, max_anchors=cfg["max_anchors"], keep_anchors=cfg["keep_anchors"])
                (out / "predictions" / sl).mkdir(exist_ok=True)
                (out / "predictions" / sl / f"{s['key']}.json").write_text(json.dumps(r), encoding="utf-8")
                mt = r["metrics"]
                per[sl][s["key"]] = {"status": "completed", "domain": s["domain"], "n_anchors": r["n_anchors"],
                                     **{f"{k}_{name}" if name != "learned" else k: (v or {}).get(k) for name, v in mt.items()
                                        for k in ("ade_px", "fde_px", "nll") if (v or {}).get(k) is not None},
                                     "anchor_t": r["anchor_t_all"][:5], "ms_per_anchor": r["timing_ms"]["forecaster_per_anchor"]}
                emit({"type": "sequence_done", "slug": sl, "sequence": s["key"], "ade_px": mt["learned"]["ade_px"]})
            st["status"] = "cancelled" if cancelled else "completed"
        except Exception as e:  # noqa: BLE001
            st["status"], st["error"] = "failed", f"{type(e).__name__}: {e}"
            st["trace"] = traceback.format_exc()[-1500:]
        finally:
            if a is not None:
                a.unload()
        if cancelled:
            break
    LH = {(s.get("L"), s.get("H")) for s in states if s["status"] == "completed"}
    metrics = {}
    for st in states:
        done = [v for v in per[st["slug"]].values() if v.get("status") == "completed"]
        metrics[st["slug"]] = {k: (float(np.mean([v[k] for v in done if v.get(k) is not None])) if any(v.get(k) is not None for v in done) else None)
                               for k in ("ade_px", "fde_px", "nll", "ade_px_cp", "ade_px_cv", "ade_px_dcv")}
        metrics[st["slug"]]["n_sequences"] = len(done)
    man = {"schema": SCHEMA, "experiment_id": eid, "category": "forecasting", "status": "cancelled" if cancelled else
           ("completed" if all(s["status"] == "completed" for s in states) else "partial"), "evaluation_protocol_id": "forecast.ADE_NLL.trackc.v1",
           "dataset_id": "trackc_trajectories", "split": cfg["split"], "sequences": cfg["sequences"], "models": states, "metrics": metrics,
           "anchors_identical": len(LH) <= 1, "comparability": "comparable (same L / H -> identical anchors)" if len(LH) <= 1 else
           f"NON-EQUIVALENT: different history / horizon {sorted(LH)}", "history_source": "ground truth (C1 oracle-history protocol)",
           "metric_note": "per-sequence ADE averaged without weighting (Track C pools points; values are not identical to its reports)",
           "git_commit": head, "git_dirty": dirty, "environment_reference": environment_reference("cpu"),
           "completion_time": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "outputs": {"dir": out.relative_to(REPO).as_posix()},
           "test_pass": False}
    (out / "per_sequence.json").write_text(json.dumps(per, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    with open(STUDIO_INDEX, "a", encoding="utf-8") as f:
        for st in states:
            rec = R.model(st["model_id"])
            f.write(json.dumps({"schema": "src/aerolab/experiment/1", "experiment_id": f"{eid}/{st['slug']}", "experiment_group": eid,
                                "direction": rec["direction"], "kind": "studio_evaluation", "provenance": "studio_run", "model_ids": [rec["model_id"]],
                                "checkpoint_sha256": rec["sha256"], "dataset_id": "trackc_trajectories", "split": cfg["split"], "modality": None,
                                "protocol_id": "forecast.ADE_NLL.trackc.v1", "metrics": metrics[st["slug"]],
                                "per_sequence": {k: v.get("ade_px") for k, v in per[st["slug"]].items() if v.get("status") == "completed"},
                                "seed": rec.get("seed"), "git_commit": head, "status": st["status"], "registration": None, "test_pass": False,
                                "source_file": f"{out.relative_to(REPO).as_posix()}/manifest.json", "category": "forecasting", "warnings": []}) + "\n")
    emit({"type": "finished", "experiment_id": eid, "status": man["status"]})
    return man


def list_forecasts():
    out = []
    for p in sorted((STUDIO_RUNS / "evals").glob("fc-*/manifest.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return out
