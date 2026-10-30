"""Catalog of RECORDED per-frame predictions (read-only playback; no model is loaded).

Each source names the model / checkpoint identity, the tracker variant, the split, the registration or report that produced the
predictions, and where they are. TEST sources are the registered passes' own outputs (X1, X2 seeds and single-frame controls) and
the authors' released CST-trained trackers; they can be replayed and inspected, never re-run, and nothing computed while replaying
is a new registered result.
"""
import json
from functools import lru_cache
from pathlib import Path

from .paths import REPO

X2_RUNS = {"v3_n__causal_g4_cst_s1_e20": "X2 temporal seed 1", "v3_n__causal_g4_cst_s2_e20": "X2 temporal seed 2",
           "v3_n__ttt_cst_s0_e20": "X2 single-frame seed 0", "v3_n__ttt_cst_s1_e20": "X2 single-frame seed 1",
           "v3_n__ttt_cst_s2_e20": "X2 single-frame seed 2"}


def _jload(rel):
    p = REPO / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@lru_cache(maxsize=1)
def sources() -> list[dict]:
    out = []
    x1 = _jload("experiments/aerotrack_v3/reports/x1_final_test.json")
    for var in ("ste", "argmax"):
        out.append({"id": f"cst:test:x1:{var}", "label": f"X1 = X2 temporal seed 0 ({var.upper()})", "model_id": "aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last",
                    "checkpoint_sha256": x1.get("weights_sha256"), "direction": "aerotrack", "dataset_id": "cst2025", "split": "test",
                    "modality": "IR", "tracker": var, "dir": f"runs/aerotrack_v3/x1_test_predictions/{var}", "ext": ".txt", "format": "res_only",
                    "registration": "experiments/aerotrack_v3/preregistration_X1_cst_protocol_matched.json", "test_pass": True,
                    "record_id": f"cst:x1:{var}", "provenance": "experiments/aerotrack_v3/reports/x1_final_test.json", "seed": 0,
                    "arm": "temporal causal_g4"})
    for run, label in X2_RUNS.items():
        rr = _jload(f"experiments/cst_anti_uav/reports/x2_test_runs/{run}.json")
        for var in ("ste", "argmax"):
            out.append({"id": f"cst:test:{run}:{var}", "label": f"{label} ({var.upper()})", "model_id": f"aerotrack.detector.cst_x:{run}@last",
                        "checkpoint_sha256": rr.get("weights_sha256"), "direction": "aerotrack", "dataset_id": "cst2025", "split": "test",
                        "modality": "IR", "tracker": var, "dir": f"runs/cst_anti_uav/x2_test_predictions/{run}/{var}", "ext": ".txt",
                        "format": "res_only", "registration": "experiments/cst_anti_uav/preregistration/X2_seeds_single_frame.json",
                        "test_pass": True, "record_id": f"cst:x2:{run}:{var}", "provenance": f"experiments/cst_anti_uav/reports/x2_test_runs/{run}.json",
                        "seed": rr.get("seed"), "arm": "temporal causal_g4" if "causal" in run else "single-frame (ttt)"})
    base = _jload("experiments/cst_anti_uav/reports/official_reproduction_test/official_scores.json")
    for name, t in (base.get("trackers") or {}).items():
        p = Path(str(t.get("path", "")).replace("\\", "/"))
        out.append({"id": f"cst:test:released:{name}", "label": f"{name} (released CST-trained predictions)", "model_id": f"external:cst_released:{name}",
                    "checkpoint_sha256": None, "direction": "shared", "dataset_id": "cst2025", "split": "test", "modality": "IR",
                    "tracker": name, "dir": p.as_posix(), "ext": ".txt", "format": "res_only", "registration": None, "test_pass": True,
                    "record_id": f"cst:released:{name}", "provenance": "experiments/cst_anti_uav/reports/official_reproduction_test/official_scores.json",
                    "seed": None, "arm": "external baseline", "note": "authors' released predictions (TrainedCST.zip); checkpoints not released; never reports absence"})
    for var in ("ste", "argmax"):
        out.append({"id": f"cst:val:x1:{var}", "label": f"X1 = X2 temporal seed 0, VAL ({var.upper()}, settings tuned on all VAL)",
                    "model_id": "aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last", "checkpoint_sha256": x1.get("weights_sha256"),
                    "direction": "aerotrack", "dataset_id": "cst2025", "split": "val", "modality": "IR", "tracker": var,
                    "dir": f"runs/cst_anti_uav/val_predictions/v3_n__causal_g4_cst_s0_e20/{var}", "ext": ".txt", "format": "res_only",
                    "registration": None, "test_pass": False, "record_id": None,
                    "provenance": "runs/cst_anti_uav/val_predictions/v3_n__causal_g4_cst_s0_e20/track_a_scores.json", "seed": 0,
                    "arm": "temporal causal_g4", "note": "VAL settings were tuned on these sequences (optimistic)"})
    for run, lab in (("normal_ir", "Direction B normal (GlobalTrack fine-tuned)"), ("dfsc_ir", "Direction B DFSC")):
        for split in ("val", "test"):
            d = f"experiments/anti_uav/runs/{run}/eval_{split}_IR_last"
            out.append({"id": f"antiuav:{split}:{run}", "label": f"{lab}, {split.upper()}", "model_id": f"antiuav_repro.globaltrack:{run}@last",
                        "checkpoint_sha256": (_jload(f"{d}/metrics.json").get("meta") or {}).get("checkpoint_sha256"), "direction": "antiuav_repro",
                        "dataset_id": "antiuav2021", "split": split, "modality": "IR", "tracker": "globaltrack", "dir": f"{d}/predictions",
                        "ext": ".json", "format": "track_a", "registration": "experiments/anti_uav/reports/test_evaluation_ledger.jsonl" if split == "test" else None,
                        "test_pass": split == "test", "record_id": f"ledger:eval:anti_uav/repro/{run}/seed0/{split}", "provenance": f"{d}/metrics.json",
                        "seed": 0, "arm": run})
    for run in ("phaseB_normal_ir_seed1", "phaseB_dfsc_ir_seed1"):
        d = f"experiments/anti_uav/runs/{run}/eval_val_IR_last"
        out.append({"id": f"antiuav:val:{run}", "label": f"Direction B {run.replace('phaseB_', '')}, VAL", "model_id": f"antiuav_repro.globaltrack:{run}@last",
                    "checkpoint_sha256": (_jload(f"{d}/metrics.json").get("meta") or {}).get("checkpoint_sha256"), "direction": "antiuav_repro",
                    "dataset_id": "antiuav2021", "split": "val", "modality": "IR", "tracker": "globaltrack", "dir": f"{d}/predictions", "ext": ".json",
                    "format": "track_a", "registration": None, "test_pass": False, "record_id": f"ledger:eval:anti_uav/repl/{run}/val",
                    "provenance": f"{d}/metrics.json", "seed": 1, "arm": run})
    return [s for s in out if (REPO / s["dir"]).is_dir()]


def source(source_id: str) -> dict:
    for s in sources():
        if s["id"] == source_id:
            return s
    raise KeyError(source_id)


def sequences_of(src: dict) -> list[str]:
    return sorted(p.stem for p in (REPO / src["dir"]).glob(f"*{src['ext']}"))


def load(src: dict, sequence: str) -> dict:
    """-> {res_xywh, scores (or None), start_frame (or None = the dataset's), processed (or None)}. Absent = [] or a non-4 list."""
    p = REPO / src["dir"] / f"{sequence}{src['ext']}"
    txt = p.read_text(encoding="utf-8")
    try:                                     # the official toolkit's loader: JSON {"res": ...} first, else comma / space separated rows
        d = json.loads(txt)
        res = d["res"]
    except ValueError:
        import io

        import numpy as np
        arr = np.atleast_2d(np.loadtxt(io.StringIO(txt.replace(",", " "))))
        d, res = {}, [list(map(float, r)) for r in arr]
    return {"res_xywh": [r if isinstance(r, list) and len(r) == 4 else [] for r in res], "scores": d.get("scores"),
            "start_frame": d.get("start_frame"), "processed": d.get("processed"), "path": p.relative_to(REPO).as_posix()}


def compatible_sources(dataset_id: str, split: str, modality: str = "IR") -> list[dict]:
    return [s for s in sources() if s["dataset_id"] == dataset_id and s["split"] == split and s["modality"] == modality]
