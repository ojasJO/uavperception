"""Turn saved tracker predictions into metrics + a human-readable report. Independent of training code.

Inputs : a predictions directory (one `<sequence>.json` per sequence, written by engine/track.py)
         + the split's sequences (labels, attributes) + the paper's reported numbers (optional).
Outputs: metrics.json (machine-readable), report.md (generated from metrics.json — no hand-typed numbers).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..data.antiuav import Sequence
from .metrics import ATTRIBUTE_ORDER, aggregate, sequence_metrics


def load_prediction(pred_dir: Path, seq_name: str) -> dict:
    p = Path(pred_dir) / f"{seq_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing prediction file {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_predictions(pred_dir: str | Path, sequences: list[Sequence], only_processed_frames: bool = False) -> dict:
    """Score saved predictions from a directory (see evaluate_prediction_dicts for the scoring rules)."""
    preds = {s.name: load_prediction(pred_dir, s.name) for s in sequences}
    return evaluate_prediction_dicts(preds, sequences, only_processed_frames)


def evaluate_prediction_dicts(preds: dict, sequences: list[Sequence], only_processed_frames: bool = False) -> dict:
    """Score prediction dicts (name -> prediction record as written by engine/track.py).

    Primary scoring (full protocol): every frame from the sequence's official START FRAME onward
    (the tracker is initialised there; earlier frames are never tracked). A secondary "all frames" variant,
    in which the untracked pre-start frames count as "absent predicted", is reported as well
    (spec §6; the two differ only for the few sequences whose start frame is > 0).
    only_processed_frames=True restricts scoring to frames the tracker actually processed (MONITOR mode,
    frame-strided subsets) and is never used for reported numbers."""
    per_seq: "OrderedDict[str, dict]" = OrderedDict()
    attributes = {}
    n_fallback = 0
    fps = []
    n_prestart = 0
    for s in sequences:
        pred = preds[s.name]
        if pred["n_frames"] != s.n_frames:
            raise ValueError(f"{s.name}: prediction has {pred['n_frames']} frames, labels have {s.n_frames}")
        start = int(pred["start_frame"])
        if only_processed_frames:
            mask = np.asarray(pred["processed"], dtype=bool)
        else:
            mask = np.arange(s.n_frames) >= start
        n_prestart += int(start)
        per_seq[s.name] = sequence_metrics(pred["res"], s.exist, s.gt_rect, mask=mask)
        per_seq[s.name]["SA_all_frames"] = sequence_metrics(pred["res"], s.exist, s.gt_rect, mask=None)["SA"]
        per_seq[s.name]["start_frame"] = start
        per_seq[s.name]["attributes"] = s.attributes
        attributes[s.name] = s.attributes
        n_fallback += int(pred.get("n_fallback", 0))
        if pred.get("fps"):
            fps.append(pred["fps"])
    agg = aggregate(per_seq, attributes)
    agg["mSA_all_frames_percent"] = float(100 * np.mean([d["SA_all_frames"] for d in per_seq.values()]))
    agg["prestart_frames_excluded"] = n_prestart
    agg["n_fallback_frames"] = n_fallback
    agg["tracker_fps_mean"] = float(np.mean(fps)) if fps else None
    agg["scoring"] = ("processed_frames_only (MONITOR — not comparable to the paper)" if only_processed_frames
                      else "frames_from_start_frame (primary; paper protocol with official start frames)")
    return {"aggregate": agg, "per_sequence": per_seq}


def format_report(metrics: dict, header: dict, paper: dict | None = None) -> str:
    """Markdown report. `header` carries identity info (checkpoint hash, split, config...)."""
    a = metrics["aggregate"]
    lines = ["# Anti-UAV tracking evaluation report", ""]
    lines.append("Generated automatically from `metrics.json`; every number below is computed from saved tracker outputs.")
    lines.append("")
    lines.append("## Identity")
    for k, v in header.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- sequences: {a['n_sequences']}  frames: {a['frames_total']}  (exist=1: {a['frames_exist']})")
    lines.append(f"- **mSA: {a['mSA_percent']:.2f} %**")
    lines.append(f"- precision@20px (exist frames): {100*a['precision_20px']:.2f} %")
    lines.append(f"- success AUC (exist frames): {100*a['success_auc']:.2f} %")
    lines.append(f"- scoring: {a['scoring']}; fallback frames (no detection): {a['n_fallback_frames']}; tracker fps: {a['tracker_fps_mean']}")
    if "mSA_all_frames_percent" in a:
        lines.append(f"- variant — mSA over ALL frames (pre-start frames counted as absent-predicted): {a['mSA_all_frames_percent']:.2f} % "
                     f"({a['prestart_frames_excluded']} pre-start frames excluded from the primary number)")
    lines.append("")
    lines.append("## mSA by attribute (%)")
    lines.append("| attribute | n_seq | mSA |")
    lines.append("|---|---|---|")
    for attr in ATTRIBUTE_ORDER + [k for k in a["by_attribute"] if k not in ATTRIBUTE_ORDER]:
        d = a["by_attribute"].get(attr)
        if d is None:
            continue
        v = "-" if d["mSA_percent"] is None else f"{d['mSA_percent']:.2f}"
        lines.append(f"| {attr} | {d['n_sequences']} | {v} |")
    lines.append("")
    if paper is not None:
        lines.append("## Paper reference values (transcribed from the paper, for orientation only)")
        for k, v in paper.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append("## Per-sequence SA (%)")
    lines.append("| sequence | start | frames | exist | SA | prec@20 | success |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, d in metrics["per_sequence"].items():
        lines.append(f"| {name} | {d['start_frame']} | {d['n_frames']} | {d['n_exist']} | {100*d['SA']:.2f} | "
                     f"{100*d['precision_20px']:.1f} | {100*d['success_auc']:.1f} |")
    return "\n".join(lines) + "\n"
