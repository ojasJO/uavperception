"""Flight Deck experiment runner: one experiment = one category, one dataset split, a sequence selection and 1-N models.

    cfg = build_config("tracking", "cst2025", "val", "IR", ["building_41"],
                       [{"model_id": "aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last", "tracker": "ste"},
                        {"model_id": "aerotrack.detector.cst_x:v3_n__ttt_cst_s0_e20@last", "tracker": "ste"}], max_frames=100)
    man = run(cfg)                    # or jobs.submit(cfg) from the UI (background thread, live frames, cancel)

Guarantees
- TEST is refused (aerolab.test_lock) before anything is loaded; categories decide which models are eligible.
- Every model runs on exactly the same frames: start frame (GT init) .. end frame, end = start + 1 + max_frames (or the whole
  sequence). The per-frame records come from the adapters' stream(), which reuse the evaluated code paths.
- Execution 'sequential' (default): one model in memory at a time, timings not disturbed by other models. 'lockstep': all models
  loaded, every frame processed by each model in turn (synchronised live view; more GPU memory, timings interleaved).
- Outputs (never overwriting anything): results/runs/studio/evals/<experiment_id>/ {config.json, manifest.json, per_sequence.json,
  timing.json, log.txt, predictions/<model slug>/<sequence>.json}; one record per model appended to results/runs/studio/index.jsonl.
- Cancellation (threading.Event) stops after the current frame; finished sequences keep their results, the interrupted one is
  saved with complete = false and no metric; the experiment status becomes 'cancelled'.
"""
import datetime as dt
import json
import re
import time
import traceback
import uuid

import numpy as np

from . import analysis, categories, data, sequences
from .adapters import compatibility, get_adapter
from .adapters.base import AdapterError, pick_device
from .compare import paired_delta
from .evaluation import dataset_hash, environment_reference, git_state
from .paths import REPO, STUDIO_INDEX, STUDIO_RUNS
from .registry import Registry
from .test_lock import assert_split_allowed

SCHEMA = "src/aerolab/flightdeck_experiment/1"
MAX_MODELS = 4
MAX_MODELS_EXTENDED = 6
TRACKERS = {"aerotrack_detector": ("ste", "argmax"), "globaltrack_repro": ("globaltrack",)}


class ConfigError(ValueError):
    pass


def _now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def slug(i, model_id, tracker):
    run = model_id.split(":", 1)[-1]
    return f"m{i}_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{run}_{tracker}")[:80]


def build_config(category, dataset_id, split, modality, sequences_, models, device="auto", execution="sequential", max_frames=None,
                 preset=None, direction=None, reference=0, note=None, allow_many=False, recorded_sources=None) -> dict:
    return {"category": category, "dataset_id": dataset_id, "split": split, "modality": modality, "sequences": list(sequences_),
            "models": [{"model_id": m["model_id"], "tracker": m.get("tracker")} for m in models], "device": device,
            "execution": execution, "max_frames": None if not max_frames else int(max_frames), "preset": preset, "direction": direction,
            "reference": int(reference), "note": note, "allow_many": bool(allow_many), "recorded_sources": list(recorded_sources or [])}


def validate(cfg: dict, R: Registry) -> tuple[list[str], list[str]]:
    """-> (errors, warnings). Errors block execution; warnings are recorded in the manifest."""
    errs, warns = [], []
    cat = categories.CATEGORIES.get(cfg.get("category"))
    if cat is None:
        return [f"unknown category {cfg.get('category')!r}"], []
    if cat["kind"] not in ("sot", "speed"):          # speed: the live demonstration mode (the benchmark has its own runner)
        errs.append(f"category '{cat['name']}' is not run by the tracking runner (use the {cat['kind']} runner)")
    try:
        assert_split_allowed(cfg["split"])
    except PermissionError as e:
        errs.append(str(e))
        return errs, warns
    if cfg["dataset_id"] not in cat["datasets"]:
        errs.append(f"dataset {cfg['dataset_id']} is not used by '{cat['name']}' (allowed: {', '.join(cat['datasets'])})")
    if cfg["modality"] not in data.SOT_DATASETS.get(cfg["dataset_id"], ()):
        errs.append(f"{cfg['dataset_id']} has no {cfg['modality']} modality")
    if not cfg["sequences"]:
        errs.append("select at least one sequence")
    else:
        try:
            have = set(data.list_sequences(cfg["dataset_id"], cfg["split"], cfg["modality"]))
            missing = [s for s in cfg["sequences"] if s not in have]
            if missing:
                errs.append(f"sequences not in {cfg['dataset_id']} {cfg['split']}: {', '.join(missing[:5])}")
        except (KeyError, OSError) as e:
            errs.append(f"cannot list sequences: {e}")
    ms = cfg["models"]
    cap = MAX_MODELS_EXTENDED if cfg.get("allow_many") else MAX_MODELS
    if not ms:
        errs.append("select at least one model")
    if len(ms) > cap:
        errs.append(f"{len(ms)} models selected; the limit is {cap}" + ("" if cfg.get("allow_many") else " (enable the extended comparison for up to 6)"))
    seen = set()
    for m in ms:
        key = (m["model_id"], m.get("tracker"))
        if key in seen:
            errs.append(f"{m['model_id']} with tracker {m.get('tracker')} is selected twice")
        seen.add(key)
        rec = R.by_id.get(m["model_id"])
        if rec is None:
            errs.append(f"unknown model {m['model_id']}")
            continue
        ok, why = categories.eligibility(cfg["category"], rec)
        if not ok:
            errs.append(f"{m['model_id']}: {why}")
            continue
        if m.get("tracker") not in TRACKERS[rec["adapter"]]:
            errs.append(f"{m['model_id']}: tracker {m.get('tracker')!r} not available (choose {', '.join(TRACKERS[rec['adapter']])})")
        c = compatibility(rec, cfg["dataset_id"], cfg["modality"], R)
        if not c.ok:
            errs.append(f"{m['model_id']}: {c.reason}")
        for w in c.warnings:
            warns.append(f"{rec['run_id']}: {w}")
    if cfg["device"] not in ("auto", "cpu", "cuda"):
        errs.append(f"device must be auto, cpu or cuda (got {cfg['device']!r})")
    elif cfg["device"] == "cuda":
        try:
            pick_device("cuda")
        except AdapterError as e:
            errs.append(f"{e}. Select device 'cpu' (or 'auto') or fix the CUDA installation.")
    if cfg["execution"] not in ("sequential", "lockstep"):
        errs.append("execution must be 'sequential' or 'lockstep'")
    elif cfg["execution"] == "lockstep" and len(ms) > 1:
        warns.append("lockstep: all models are loaded at once and process each frame in turn; per-model timings are interleaved "
                     "(use sequential execution or Category E for speed comparisons)")
    if cfg["max_frames"]:
        warns.append(f"truncated to the init frame + {cfg['max_frames']} frames per sequence: protocol studio.sot_smoke.v1, "
                     "not comparable with full-sequence mSA")
    return errs, warns


def protocol_of(cfg) -> str:
    if cfg["category"] == "presence":
        return "presence.frame.v1"
    return "studio.sot_smoke.v1" if cfg["max_frames"] else "sot.mSA.antiuav_eq1.v1"


class Experiment:
    """the on-disk record of one experiment; every write is incremental and nothing existing is ever overwritten"""

    def __init__(self, cfg, R, warnings, experiment_id=None):
        self.id = experiment_id or f"fd-{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.dir = STUDIO_RUNS / "evals" / self.id
        if self.dir.exists():
            raise FileExistsError(self.dir)
        (self.dir / "predictions").mkdir(parents=True)
        self.log_f = open(self.dir / "log.txt", "w", encoding="utf-8")
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        head, dirty = git_state()
        proto = protocol_of(cfg)
        self.man = {
            "schema": SCHEMA, "experiment_id": self.id, "category": cfg["category"], "category_name": categories.CATEGORIES[cfg["category"]]["name"],
            "research_directions": sorted({R.model(m["model_id"])["direction"] for m in cfg["models"]}), "status": "running",
            "dataset_id": cfg["dataset_id"], "split": cfg["split"], "modality": cfg["modality"],
            "dataset_version_or_manifest_hash": dataset_hash(cfg["dataset_id"], cfg["split"], cfg["modality"]),
            "sequences": cfg["sequences"], "preset": cfg.get("preset"), "max_frames": cfg["max_frames"],
            "frame_rule": "start frame (GT init) .. start + max_frames inclusive" if cfg["max_frames"] else "start frame .. last frame",
            "evaluation_protocol_id": proto, "metric_definition_version": (R.protocols.get(proto) or {}).get("metric_definition_version", "unknown"),
            "execution": {"mode": cfg["execution"], "device_requested": cfg["device"]}, "reference_model": cfg["reference"],
            "models": [], "metrics": {}, "comparison": {}, "git_commit": head, "git_dirty": dirty,
            "environment_reference": environment_reference(cfg["device"]), "start_time": _now(), "completion_time": None,
            "outputs": {"dir": self.dir.relative_to(REPO).as_posix(), "predictions_dir": (self.dir / "predictions").relative_to(REPO).as_posix(),
                        "per_sequence": (self.dir / "per_sequence.json").relative_to(REPO).as_posix(),
                        "timing": (self.dir / "timing.json").relative_to(REPO).as_posix(), "log": (self.dir / "log.txt").relative_to(REPO).as_posix()},
            "recorded_sources": cfg.get("recorded_sources") or [], "registration_reference": None, "test_pass": False,
            "note": cfg.get("note"), "warnings": list(warnings), "error": None}
        self.write_manifest()

    def log(self, msg):
        self.log_f.write(f"[{_now()}] {msg}\n")
        self.log_f.flush()

    def write_manifest(self):
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.man, indent=1), encoding="utf-8")
        tmp.replace(self.dir / "manifest.json")

    def save_prediction(self, mslug, payload):
        d = self.dir / "predictions" / mslug
        d.mkdir(exist_ok=True)
        (d / f"{payload['sequence']}.json").write_text(json.dumps(payload), encoding="utf-8")


def _end_frame(seq, max_frames):
    return seq.n_frames if not max_frames else min(seq.n_frames, seq.start_frame + 1 + int(max_frames))


def _payload(exp, m, a, meta, seq, cfg, recs, end, complete):
    n = seq.n_frames
    res = [[] for _ in range(end)]
    processed = [False] * end
    for r in recs:
        t = r["frame"]
        processed[t] = True
        b = r["box_xyxy"]
        res[t] = [] if b is None else [b[0], b[1], b[2] - b[0], b[3] - b[1]]
    return {"schema": "src/aerolab/flightdeck_prediction/1", "experiment_id": exp.id, "model_id": m["model_id"], "checkpoint_sha256": a.record["sha256"],
            "checkpoint_path": a.get_metadata()["checkpoint_path"], "tracker": meta["tracker"], "params": meta["params"],
            "params_source": meta["params_source"], "window": meta.get("window"), "preprocessing": meta.get("preprocessing"),
            "score_kind": meta.get("score_kind"), "dataset_id": cfg["dataset_id"], "split": cfg["split"], "modality": cfg["modality"],
            "sequence": seq.name, "n_frames_total": n, "start_frame": int(seq.start_frame), "end_frame": int(end), "complete": complete,
            "device": a.device, "res_xywh": res, "processed": processed, "frames": recs}


def _summarise(seq, payload, dataset_id):
    s = analysis.sequence_summary(payload["res_xywh"], seq.exist, seq.gt_rect, payload["start_frame"], payload["end_frame"], payload["processed"])
    s["attribute_SA"] = attribute_sa(seq, payload, dataset_id)
    t_all = [r["timing_ms"]["total"] for r in payload["frames"]]
    t_mod = [r["timing_ms"]["model"] for r in payload["frames"]]
    s["timing_ms"] = {"total": analysis.latency_stats(t_all[1:]), "model": analysis.latency_stats(t_mod[1:]), "first_frame_total": t_all[0] if t_all else None}
    return s


def attribute_sa(seq, payload, dataset_id) -> dict:
    """Eq. 1 frame-level SA averaged over the scored frames carrying each flag (official CST flags, absent, size bins, the 30 frames
    after a reappearance). The same 'corrected' convention as ATTRIBUTE_ANALYSIS.md (every flagged frame counts)."""
    flags = sequences.frame_flags(seq, dataset_id)
    rows = analysis.frame_table(payload["res_xywh"], seq.exist, seq.gt_rect, payload["start_frame"], payload["end_frame"], payload["processed"])
    sa = np.full(len(rows), np.nan)
    for r in rows:
        if r["status"] in ("not_processed",):
            continue
        sa[r["frame"]] = (r["iou"] or 0.0) if r["visible"] else (0.0 if r["has_box"] else 1.0)
    out = {}
    for k, mask in flags.items():
        m = mask[:len(sa)] & np.isfinite(sa)
        if m.any():
            out[k] = {"SA": float(100 * sa[m].mean()), "n_frames": int(m.sum())}
    return out


def run(cfg: dict, R: Registry | None = None, on_event=None, cancel=None, experiment_id=None) -> dict:
    """Execute (blocking). on_event(dict) receives live events; cancel is a threading.Event. Returns the final manifest."""
    assert_split_allowed(cfg["split"])
    R = R or Registry()
    errs, warns = validate(cfg, R)
    if errs:
        raise ConfigError("; ".join(errs))
    exp = Experiment(cfg, R, warns, experiment_id)
    emit = on_event or (lambda e: None)
    emit({"type": "started", "experiment_id": exp.id, "dir": exp.man["outputs"]["dir"]})
    seqs = {n: data.load_sequence(cfg["dataset_id"], cfg["split"], n, cfg["modality"]) for n in cfg["sequences"]}
    total_frames = sum(_end_frame(s, cfg["max_frames"]) - s.start_frame for s in seqs.values())
    states = [{"slug": slug(i, m["model_id"], m["tracker"]), "model_id": m["model_id"], "tracker": m["tracker"], "status": "pending",
               "direction": R.model(m["model_id"])["direction"], "frames_done": 0, "frames_total": total_frames, "per_sequence": {},
               "sequence_status": {}} for i, m in enumerate(cfg["models"])]
    exp.man["models"] = states
    exp.write_manifest()
    cancelled = False

    def load(i):
        st = states[i]
        rec = R.model(st["model_id"])
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        a = get_adapter(rec["adapter"], R).load(rec, cfg["device"])
        meta = a.stream_meta(st["tracker"])
        st.update(status="running", device=a.device, load_seconds=a.load_s, checkpoint_sha256=rec["sha256"],
                  checkpoint_path=a.get_metadata()["checkpoint_path"], seed=rec.get("seed"), aliases=rec.get("aliases") or [],
                  params=meta["params"], params_source=meta["params_source"], window=meta.get("window"),
                  preprocessing=meta.get("preprocessing"), score_kind=meta.get("score_kind"))
        for w in meta.get("warnings") or []:
            if w not in exp.man["warnings"]:
                exp.man["warnings"].append(f"{rec['run_id']}: {w}")
        exp.log(f"loaded {st['model_id']} on {a.device} in {a.load_s:.2f}s ({a.get_metadata()['checkpoint_path']})")
        emit({"type": "model_loaded", "slug": st["slug"], "model_id": st["model_id"], "device": a.device, "load_seconds": a.load_s})
        return a, meta

    def finish_model(i, a):
        st = states[i]
        try:
            import torch
            if torch.cuda.is_available() and a is not None and a.device == "cuda":
                st["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        except Exception:  # noqa: BLE001
            pass
        if a is not None:
            a.unload()

    def stream_kw(st, seq):
        return {"end_frame": _end_frame(seq, cfg["max_frames"]), "dataset_id": cfg["dataset_id"], "modality": cfg["modality"],
                **({"tracker": st["tracker"]} if st["tracker"] in ("ste", "argmax") else {})}

    def close_seq(i, a, meta, seq, recs, complete):
        st = states[i]
        end = _end_frame(seq, cfg["max_frames"]) if complete else (recs[-1]["frame"] + 1 if recs else seq.start_frame)
        payload = _payload(exp, cfg["models"][i], a, meta, seq, cfg, recs, end, complete)
        exp.save_prediction(st["slug"], payload)
        if complete:
            s = _summarise(seq, payload, cfg["dataset_id"])
            st["per_sequence"][seq.name] = s
            st["sequence_status"][seq.name] = "completed"
            exp.log(f"{st['slug']} {seq.name}: SA {s['SA']:.2f} over frames {payload['start_frame']}..{payload['end_frame'] - 1}")
            emit({"type": "sequence_done", "slug": st["slug"], "sequence": seq.name, "SA": s["SA"]})
        else:
            st["sequence_status"][seq.name] = "cancelled"
        exp.write_manifest()

    def event(i, seq, r, t0):
        st = states[i]
        st["frames_done"] += 1
        emit({"type": "frame", "slug": st["slug"], "model_id": st["model_id"], "sequence": seq.name, "record": r,
              "frames_done": st["frames_done"], "frames_total": st["frames_total"], "elapsed_s": time.perf_counter() - t0,
              "device": st.get("device")})

    try:
        if cfg["execution"] == "sequential" or len(states) == 1:
            for i, st in enumerate(states):
                if cancelled:
                    st["status"] = "not_started"
                    continue
                a = None
                try:
                    a, meta = load(i)
                    t0 = time.perf_counter()
                    for seq in seqs.values():
                        recs, complete = [], True
                        for r in a.stream(seq, **stream_kw(st, seq)):
                            recs.append(r)
                            event(i, seq, r, t0)
                            if cancel is not None and cancel.is_set():
                                complete = False
                                break
                        close_seq(i, a, meta, seq, recs, complete)
                        if not complete:
                            cancelled = True
                            break
                    st["status"] = "cancelled" if cancelled else "completed"
                except Exception as e:  # noqa: BLE001 -- one model failing must not hide the others' results
                    st["status"], st["error"] = "failed", f"{type(e).__name__}: {e}"
                    exp.log(traceback.format_exc())
                    emit({"type": "model_failed", "slug": st["slug"], "error": st["error"]})
                finally:
                    finish_model(i, a)
                    exp.write_manifest()
        else:
            loaded = []
            for i, st in enumerate(states):
                try:
                    loaded.append(load(i))
                except Exception as e:  # noqa: BLE001
                    st["status"], st["error"] = "failed", f"{type(e).__name__}: {e}"
                    exp.log(traceback.format_exc())
                    loaded.append((None, None))
            t0 = time.perf_counter()
            for seq in seqs.values():
                gens = [(i, a.stream(seq, **stream_kw(states[i], seq))) if a is not None else (i, None) for i, (a, _) in enumerate(loaded)]
                recs = {i: [] for i in range(len(states))}
                live = {i for i, g in gens if g is not None}
                while live and not cancelled:
                    for i, g in gens:
                        if i not in live:
                            continue
                        try:
                            r = next(g)
                        except StopIteration:
                            live.discard(i)
                            continue
                        except Exception as e:  # noqa: BLE001
                            states[i]["status"], states[i]["error"] = "failed", f"{type(e).__name__}: {e}"
                            exp.log(traceback.format_exc())
                            live.discard(i)
                            continue
                        recs[i].append(r)
                        event(i, seq, r, t0)
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                for i, (a, meta) in enumerate(loaded):
                    if a is not None and states[i]["status"] != "failed":
                        close_seq(i, a, meta, seq, recs[i], not cancelled)
                if cancelled:
                    break
            for i, (a, _) in enumerate(loaded):
                if states[i]["status"] != "failed":
                    states[i]["status"] = "cancelled" if cancelled else "completed"
                finish_model(i, a)
        return _finalize(exp, cfg, R, states, "cancelled" if cancelled else None, emit)
    except BaseException as e:  # includes Streamlit's script-stop exceptions: record, never leave 'running'
        if exp.man["status"] != "running":       # failed inside _finalize itself: the record is already final
            raise
        exp.log(traceback.format_exc())
        exp.man["error"] = f"{type(e).__name__}: {e}"
        _finalize(exp, cfg, R, states, "failed" if isinstance(e, Exception) else "cancelled", emit)
        raise


def _finalize(exp, cfg, R, states, forced, emit):
    per = {st["slug"]: st["per_sequence"] for st in states}
    timing = {}
    for st in states:
        ps = st["per_sequence"]
        sas = [v["SA"] for v in ps.values()]
        complete = st["status"] == "completed" and len(ps) == len(cfg["sequences"])
        key = "mSA" if not cfg["max_frames"] else "mSA_partial"
        st["metrics"] = {key: float(np.mean(sas)) if sas else None, "n_sequences": len(sas), "complete": complete,
                         "mean_false_alarm_rate": analysis_mean([v["false_alarm_rate"] for v in ps.values()]),
                         "mean_located_rate": analysis_mean([v["located_rate"] for v in ps.values()]),
                         "mean_precision_20px": analysis_mean([v["precision_20px"] for v in ps.values()]),
                         "mean_cle_px": analysis_mean([v["mean_cle_px"] for v in ps.values()]),
                         "loss_events": int(sum(v["episodes"]["loss_events"] for v in ps.values())),
                         "failure_intervals": int(sum(len(v["episodes"]["failure_intervals"]) for v in ps.values()))}
        frames = []
        for f in sorted((exp.dir / "predictions" / st["slug"]).glob("*.json")) if (exp.dir / "predictions" / st["slug"]).exists() else []:
            frames += json.loads(f.read_text(encoding="utf-8"))["frames"][1:]
        timing[st["slug"]] = {"observed_live": True, "device": st.get("device"), "load_seconds": st.get("load_seconds"),
                              "peak_gpu_mem_mb": st.get("peak_gpu_mem_mb"),
                              **{k: analysis.latency_stats([r["timing_ms"][k] for r in frames]) for k in
                                 ("total", "read_preprocess", "model", "postprocess", "tracker")},
                              "note": "observed while streaming (init frame excluded); not a controlled benchmark — see Category E"}
        exp.man["metrics"][st["slug"]] = st["metrics"]
    ref = states[exp.man["reference_model"]] if exp.man["reference_model"] < len(states) else states[0]
    comp = {"reference": ref["slug"], "pairs": {}}
    for st in states:
        if st is ref:
            continue
        reasons = []
        if not (st["metrics"]["complete"] and ref["metrics"]["complete"]):
            reasons.append("one side did not complete every sequence")
        if st.get("direction") != ref.get("direction"):
            reasons.append(f"different research directions ({ref.get('direction')} vs {st.get('direction')}): same frames and metric, "
                           "but different training data and paper provenance")
        a = {k: v["SA"] for k, v in st["per_sequence"].items()}
        b = {k: v["SA"] for k, v in ref["per_sequence"].items()}
        pd_ = paired_delta({"per_sequence": a}, {"per_sequence": b}, n_boot=10000) if a and set(a) == set(b) else None
        if pd_ and pd_["n"] < 5:
            pd_["ci95"] = None
            pd_["note"] = "fewer than 5 sequences: no bootstrap CI (per-sequence differences only)"
        comp["pairs"][st["slug"]] = {"verdict": "comparable" if not any("did not complete" in r for r in reasons) else "incomplete",
                                     "notes": reasons, "paired_delta_SA": pd_,
                                     "per_sequence_delta": {k: a[k] - b[k] for k in a if k in b}}
    exp.man["comparison"] = comp
    statuses = {st["status"] for st in states}
    exp.man["status"] = forced or ("completed" if statuses == {"completed"} else "failed" if statuses == {"failed"} else "partial")
    exp.man["completion_time"] = _now()
    (exp.dir / "per_sequence.json").write_text(json.dumps(per, indent=1), encoding="utf-8")
    (exp.dir / "timing.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    exp.write_manifest()
    STUDIO_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(STUDIO_INDEX, "a", encoding="utf-8") as f:
        for st in states:
            rec = R.model(st["model_id"])
            f.write(json.dumps({
                "schema": "src/aerolab/experiment/1", "experiment_id": f"{exp.id}/{st['slug']}", "experiment_group": exp.id,
                "direction": rec["direction"], "kind": "studio_evaluation", "provenance": "studio_run", "model_ids": [st["model_id"]],
                "checkpoint_sha256": rec["sha256"], "dataset_id": cfg["dataset_id"], "split": cfg["split"], "modality": cfg["modality"],
                "protocol_id": exp.man["evaluation_protocol_id"], "metrics": {k: v for k, v in st["metrics"].items() if k != "complete"},
                "complete": st["metrics"]["complete"], "per_sequence": {k: v["SA"] for k, v in st["per_sequence"].items()},
                "seed": rec.get("seed"), "git_commit": exp.man["git_commit"], "status": st["status"], "registration": None, "test_pass": False,
                "source_file": f"{exp.man['outputs']['dir']}/manifest.json", "max_frames": cfg["max_frames"], "sequence_set": sorted(cfg["sequences"]),
                "tracker": st["tracker"], "category": cfg["category"], "start": exp.man["start_time"], "end": exp.man["completion_time"],
                "notes": cfg.get("note"), "warnings": exp.man["warnings"]}) + "\n")
    exp.log(f"status {exp.man['status']}")
    exp.log_f.close()
    emit({"type": "finished", "experiment_id": exp.id, "status": exp.man["status"]})
    return exp.man


def analysis_mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and x == x]
    return float(np.mean(xs)) if xs else None


# ------------------------------------------------------------------------------------------------------------- reopening
def list_experiments() -> list[dict]:
    out = []
    for p in sorted((STUDIO_RUNS / "evals").glob("fd-*/manifest.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def load_experiment(experiment_id: str) -> dict:
    p = STUDIO_RUNS / "evals" / experiment_id / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_prediction(experiment_id: str, mslug: str, sequence: str) -> dict | None:
    p = STUDIO_RUNS / "evals" / experiment_id / "predictions" / mslug / f"{sequence}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
