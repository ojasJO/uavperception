"""The seven Flight Deck experiment categories: which models, datasets, presets, metrics and protocol each one uses.

Eligibility is decided here (and re-checked by aerolab.flightdeck before anything runs), so the UI can only offer what the
category can evaluate correctly. A model is runnable only if its family has an adapter AND scripts/verify_adapters.py passed it.
"""
import json

from .paths import CATALOG

SOT_ADAPTERS = ("aerotrack_detector", "globaltrack_repro")

CATEGORIES = {
    "tracking": {
        "letter": "A", "name": "Tracking performance",
        "question": "How accurately does each tracker follow its target through a sequence?",
        "adapters": SOT_ADAPTERS, "datasets": ("cst2025", "antiuav2021"), "kind": "sot",
        "presets": ("standard_quick", "standard", "target_absent", "occlusion_reappearance", "small_targets", "large_targets", "fast_motion",
                    "clutter", "thermal_crossover", "long_sequence", "confusing_objects", "difficult_cases", "custom"),
        "metrics": ("SA", "precision_20px", "success_auc", "mean_iou_visible", "mean_cle_px", "located_rate", "false_alarm_rate", "loss_events"),
        "views": ("video", "metrics", "timeline", "episodes"),
        "init": "the dataset's own start-frame ground-truth box (one-pass protocol); nothing is invented",
    },
    "presence": {
        "letter": "B", "name": "Target presence & detection",
        "question": "Can the model tell target-present from target-absent frames, and what does a threshold trade?",
        "adapters": ("aerotrack_detector",), "datasets": ("cst2025", "antiuav2021"), "kind": "sot",
        "presets": ("target_absent", "occlusion_reappearance", "standard_quick", "standard", "custom"),
        "metrics": ("false_alarm_rate", "miss_rate", "precision", "recall", "presence_auc"),
        "views": ("video", "presence"),
        "excluded": {"globaltrack_repro": "GlobalTrack outputs a box on every frame and its box score is not a presence signal; "
                                          "a presence analysis would be trivially 'always present'"},
        "init": "start-frame GT box; presence is analysed on the frames after it",
    },
    "temporal": {
        "letter": "C", "name": "Temporal & ablation",
        "question": "What does the temporal window contribute, beyond the single aggregate score?",
        "adapters": ("aerotrack_detector",), "datasets": ("cst2025",), "kind": "sot",
        "families": ("aerotrack.detector.cst_x", "aerotrack.detector.v2"),
        "presets": ("standard_quick", "standard", "confusing_objects", "occlusion_reappearance", "difficult_cases", "custom"),
        "metrics": ("SA", "located_rate", "false_alarm_rate", "wrong_object", "model_ms"),
        "views": ("video", "metrics", "seeds", "timeline"),
        "init": "start-frame GT box",
    },
    "robustness": {
        "letter": "D", "name": "Robustness & difficult scenarios",
        "question": "Where and why do the trackers fail under hard conditions?",
        "adapters": SOT_ADAPTERS, "datasets": ("cst2025", "antiuav2021"), "kind": "sot",
        "presets": ("occlusion_reappearance", "target_absent", "small_targets", "large_targets", "fast_motion", "clutter", "thermal_crossover",
                    "long_sequence", "confusing_objects", "difficult_cases", "custom"),
        "metrics": ("SA", "attribute_SA", "failure_intervals", "frames_to_recover"),
        "views": ("video", "scenario", "episodes"),
        "init": "start-frame GT box",
    },
    "speed": {
        "letter": "E", "name": "Speed & deployment",
        "question": "How fast is each model on this laptop, measured properly?",
        "adapters": SOT_ADAPTERS, "datasets": ("cst2025", "antiuav2021"), "kind": "speed",
        "presets": ("standard_quick", "long_sequence", "custom"),
        "metrics": ("init_s", "first_ms", "median_ms", "p95_ms", "fps", "peak_gpu_mb", "rss_mb"),
        "views": ("benchmark",),
        "init": "start-frame GT box",
    },
    "forecasting": {
        "letter": "F", "name": "Forecasting & trajectories",
        "question": "How well do forecasters predict the next 24 positions, with what uncertainty?",
        "adapters": ("trackc_forecaster",), "datasets": ("trackc_trajectories",), "kind": "forecast",
        "presets": (), "metrics": ("ade_px", "fde_px", "nll", "ade_px_cp", "ade_px_cv", "ade_px_dcv"),
        "views": ("trajectory", "metrics"),
        "init": "ground-truth histories (Track C C1 oracle-history protocol)",
    },
    "reproduction": {
        "letter": "G", "name": "Paper reproduction",
        "question": "Does the local implementation reproduce the paper, and where does it differ?",
        "adapters": ("globaltrack_repro",), "datasets": ("antiuav2021",), "kind": "sot",
        "presets": ("standard_quick", "standard", "occlusion_reappearance", "fast_motion", "thermal_crossover", "small_targets", "custom"),
        "metrics": ("SA", "precision_20px", "success_auc"),
        "views": ("paper", "video", "metrics"),
        "init": "the official start frame (Track A loader corrections)",
    },
}
ORDER = ("tracking", "presence", "temporal", "robustness", "speed", "forecasting", "reproduction")


def featured() -> list[dict]:
    p = CATALOG / "featured.json"
    return json.loads(p.read_text(encoding="utf-8")).get("featured", []) if p.exists() else []


def runnable(model: dict) -> tuple[bool, str]:
    if not model.get("adapter"):
        return False, "recorded results only: no inference adapter for this family"
    if model["status"] in ("smoke", "invalid"):
        return False, f"status '{model['status']}'"
    v = (model.get("verification") or {}).get("inference")
    if v != "passed":
        return False, f"local inference not verified ({v}); " + ("checkpoint only in the recovery backup" if not model.get("available_in_workspace") else "run scripts/verify_adapters.py")
    return True, ""


def eligibility(category: str, model: dict) -> tuple[bool, str]:
    c = CATEGORIES[category]
    ok, why = runnable(model)
    if not ok:
        return False, why
    if model["adapter"] not in c["adapters"]:
        return False, (c.get("excluded") or {}).get(model["adapter"]) or f"{model['task']} model: not part of '{c['name']}'"
    if c.get("families") and model["family_id"] not in c["families"]:
        return False, f"family {model['family_id']} is not a registered temporal-ablation family"
    return True, ""


EVAL_CHECKS = CATALOG.parent / "verification" / "evaluation_checks.json"
DEVICES = {"aerotrack_detector": ("cpu", "cuda"), "globaltrack_repro": ("cpu", "cuda"), "trackc_forecaster": ("cpu",)}


def evaluation_checks() -> dict:
    """{model_id: check} from scripts/verify_evaluations.py (fresh full-sequence run == a recorded per-sequence value)"""
    if not EVAL_CHECKS.exists():
        return {}
    return {c["model_id"]: c for c in json.loads(EVAL_CHECKS.read_text(encoding="utf-8")).get("checks", [])}


def status_labels(model: dict, registry=None, checks=None) -> list[str]:
    """the evidence ladder of one registry entry, weakest claim first; never stronger than its evidence"""
    v = model.get("verification") or {}
    out = []
    if model["status"] == "registered":
        out.append("registered")
    out.append("integrity-verified" if model.get("integrity_status") == "verified_against_record" else "hashed (no independent record)")
    if v.get("load") == "passed":
        out.append("load-verified")
    if v.get("inference") == "passed":
        out.append("inference-verified")
    chk = (checks if checks is not None else evaluation_checks()).get(model["model_id"])
    if chk and chk.get("status") == "passed":
        out.append("evaluation-verified")
    if registry is not None and registry.experiments_for(model["model_id"]):
        out.append("historical results")
    if not model.get("adapter"):
        out.append("recorded results only")
    ok, why = runnable(model)
    if not ok and model.get("adapter"):
        out.append("not runnable locally")
    if model["status"] in ("smoke", "invalid"):
        out.append(model["status"])
    return out


def eligible_models(category: str, registry, direction: str | None = None) -> list[dict]:
    out = []
    for m in registry.models:
        if direction and direction != "both" and m["direction"] != direction:
            continue
        ok, _ = eligibility(category, m)
        if ok:
            out.append(m)
    return out
