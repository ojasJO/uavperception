"""Sequence metadata index and grounded scenario presets (Flight Deck video selection).

Every preset states its BASIS, so a derived label is never shown as an annotation:
    official_annotation  -- a flag shipped with the dataset (CST per-frame OC OV SV TC DBC CDB FM; Anti-UAV sequence attributes
                            label_new/<split>.json; the `exist` flag = target present / absent)
    derived_from_gt      -- computed from the ground-truth boxes with the rule shown (size from the box diagonal = the CST SIZE
                            convention; motion from the GT-centre displacement)
    recorded_results     -- selected from recorded predictions of a named model (file shown)
    fixed_rule           -- a documented deterministic selection (standard sets)
A preset with no eligible sequence is returned with available = False and the reason; nothing is invented to fill it.

The index reads only label files and manifests (no frames). TEST may be indexed only for read-only replay of recorded predictions.
"""
import json
from pathlib import Path

import numpy as np

from . import data
from .paths import REPO, STUDIO_RUNS
from .test_lock import assert_split_allowed

CACHE = STUDIO_RUNS / "cache"
CST_FRAME_ATTRS = {"OC": "occlusion", "OV": "out of view", "SV": "scale variation", "TC": "thermal crossover",
                   "DBC": "dynamic background clutter", "CDB": "complex dynamic background", "FM": "fast motion"}
ANTIUAV_SEQ_ATTRS = {"OV": "out of view", "OC": "occlusion", "FM": "fast motion", "SV": "scale variation", "LI": "low illumination",
                     "TC": "thermal crossover", "LR": "low resolution"}
FPS = {"antiuav2021": (25.0, "25 FPS, paper-explicit (docs/ANTI_UAV_REPRODUCTION_SPEC.md)"),
       "cst2025": (None, "not reported by the CST paper or labels (experiments/cst_anti_uav/reports/PAPER_PROTOCOL.md)")}
TINY_PX, LARGE_PX = 10.0, 50.0           # CST SIZE convention (track_eval.CST_SIZE_BINS): tiny <= 10 px, large > 50 px (box diagonal)
RECORDED_VAL = {   # per-sequence SA of a recorded VAL evaluation, used by the 'difficult cases' preset (recorded_results basis)
    "cst2025": ("results/runs/cst_anti_uav/val_predictions/v3_n__causal_g4_cst_s0_e20/track_a_scores.json",
                "aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last (X1, STE, frozen VAL settings)"),
    "antiuav2021": ("experiments/anti_uav/runs/normal_ir/eval_val_IR_last/metrics.json",
                    "antiuav_repro.globaltrack:normal_ir@last (Direction B normal, recorded VAL evaluation)"),
}
RECORDED_VAL_PRED = {  # recorded per-frame VAL predictions for the 'confusing objects' preset
    "cst2025": ("results/runs/cst_anti_uav/val_predictions/v3_n__causal_g4_cst_s0_e20/ste", "X1 (STE) recorded VAL predictions"),
    "antiuav2021": ("experiments/anti_uav/runs/normal_ir/eval_val_IR_last/predictions", "Direction B normal_ir recorded VAL predictions"),
}


def _label_files(dataset_id, split, modality):
    if dataset_id == "cst2025":
        return sorted((data.CST_ROOT / split).glob("*/IR_label.json"))
    fn = "infrared.json" if modality == "IR" else "visible.json"
    return sorted((data.ANTIUAV_LABELS / split).glob(f"*/{fn}"))


def _fingerprint(files):
    return [len(files), sum(f.stat().st_size for f in files), max((f.stat().st_mtime for f in files), default=0)]


def _seq_attrs_antiuav(split):
    p = data.ANTIUAV_LABELS / "label_new" / f"{split}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _summarise(seq, dataset_id, seq_attrs=None):
    n = seq.n_frames
    ex = np.asarray(seq.exist[:n], dtype=bool)
    st = seq.start_frame
    g = np.array([b if (e and isinstance(b, (list, tuple)) and len(b) == 4) else [np.nan] * 4 for b, e in zip(seq.gt_rect[:n], ex)], float)
    diag = np.hypot(g[:, 2], g[:, 3])
    c = g[:, :2] + g[:, 2:] / 2
    speed = np.linalg.norm(np.diff(c, axis=0), axis=1)          # px / frame between consecutive visible frames (NaN otherwise)
    sc = ex[st:]
    reapp = np.flatnonzero((~ex[st:-1]) & ex[st + 1:]) + st + 1 if n - st > 1 else np.array([], int)
    episodes, run = [], 0
    for v in sc:
        if not v:
            run += 1
        elif run:
            episodes.append(run)
            run = 0
    if run:
        episodes.append(run)
    vis_diag = diag[ex]
    out = {"name": seq.name, "n_frames": int(n), "start_frame": int(st), "n_visible": int(sc.sum()), "n_absent": int((~sc).sum()),
           "absent_fraction": float((~sc).mean()) if len(sc) else 0.0, "n_absent_episodes": len(episodes),
           "longest_absence": int(max(episodes)) if episodes else 0, "reappearance_frames": [int(x) for x in reapp],
           "diag_median": float(np.nanmedian(vis_diag)) if np.isfinite(vis_diag).any() else None,
           "diag_min": float(np.nanmin(vis_diag)) if np.isfinite(vis_diag).any() else None,
           "diag_max": float(np.nanmax(vis_diag)) if np.isfinite(vis_diag).any() else None,
           "n_tiny": int(np.sum(vis_diag <= TINY_PX)), "n_large": int(np.sum(vis_diag > LARGE_PX)),
           "speed_p95": float(np.nanpercentile(speed, 95)) if np.isfinite(speed).any() else None,
           "speed_max": float(np.nanmax(speed)) if np.isfinite(speed).any() else None}
    if dataset_id == "cst2025":
        attrs = getattr(seq, "cst_attributes", {}) or {}
        out["frame_attrs"] = {k: int(np.sum(np.asarray(v[:n], dtype=float) > 0)) for k, v in attrs.items() if k in CST_FRAME_ATTRS}
        out["category"] = seq.name.rsplit("_", 1)[0]
        out["seq_attrs"] = []
    else:
        out["frame_attrs"] = {}
        out["seq_attrs"] = list((seq_attrs or {}).get(seq.name, []))
        out["category"] = None
    fps = FPS[dataset_id][0]
    out["duration_s"] = n / fps if fps else None
    return out


def sequence_index(dataset_id: str, split: str, modality: str = "IR", purpose: str = "studio") -> list[dict]:
    """per-sequence metadata (labels only; cached under results/runs/studio/cache keyed by the label files' size + mtime)."""
    assert_split_allowed(split, purpose)
    files = _label_files(dataset_id, split, modality)
    key = f"seqindex_{dataset_id}_{split}_{modality}.json"
    fp = _fingerprint(files)
    cp = CACHE / key
    if cp.exists():
        try:
            d = json.loads(cp.read_text(encoding="utf-8"))
            if d.get("fingerprint") == fp and d.get("version") == 2:
                return d["sequences"]
        except (OSError, ValueError):
            pass
    sa = _seq_attrs_antiuav(split) if dataset_id == "antiuav2021" else None
    seqs = []
    for f in files:
        s = data.load_sequence(dataset_id, split, f.parent.name, modality, purpose=purpose)
        seqs.append(_summarise(s, dataset_id, sa))
    CACHE.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"version": 2, "fingerprint": fp, "sequences": seqs}), encoding="utf-8")
    return seqs


def image_size(dataset_id, split, name, modality="IR", purpose="studio"):
    from PIL import Image
    s = data.load_sequence(dataset_id, split, name, modality, purpose=purpose)
    with Image.open(s.frame_path(s.start_frame)) as im:          # header only
        return im.size


# ------------------------------------------------------------------------------------------------------------------ presets
PRESETS = [
    ("standard", "Standard benchmark", "The complete official split (every sequence) — the set the recorded VAL results use."),
    ("standard_quick", "Standard quick set", "A fixed small representative subset for fast checks (rule shown)."),
    ("target_absent", "Target absent", "Sequences with target-absent frames after the start frame."),
    ("occlusion_reappearance", "Occlusion & reappearance", "Target leaves (occlusion / out of view) and returns."),
    ("small_targets", "Small targets", "Many visible frames with a tiny target (box diagonal <= 10 px)."),
    ("large_targets", "Large targets", "Visible frames with a large target (box diagonal > 50 px)."),
    ("fast_motion", "Fast motion", "Rapid target motion: official flags and / or the fastest sequences by GT-centre speed."),
    ("clutter", "Clutter & distractors", "Dynamic / complex background clutter (official flags)."),
    ("thermal_crossover", "Thermal crossover", "Target and background at similar temperature (official flags)."),
    ("long_sequence", "Long sequences", "Long sequences for prolonged-stability checks."),
    ("confusing_objects", "Confusing objects", "Recorded wrong-object locks: the recorded tracker box misses a visible target entirely."),
    ("difficult_cases", "Difficult cases", "Lowest recorded per-sequence SA of a named model (recorded results)."),
    ("custom", "Custom selection", "Any sequences chosen by hand."),
]
PRESET_NAMES = {p[0]: p[1] for p in PRESETS}


def _res(pid, basis, rule, source, items, reason=None):
    return {"id": pid, "name": PRESET_NAMES[pid], "basis": basis, "rule": rule, "source": source, "items": items,
            "available": bool(items), "reason": None if items else (reason or "no sequence of this split satisfies the rule")}


def _recorded_val_scores(dataset_id):
    p, who = RECORDED_VAL.get(dataset_id, (None, None))
    if not p or not (REPO / p).exists():
        return None, who, p
    d = json.loads((REPO / p).read_text(encoding="utf-8"))
    if dataset_id == "cst2025":
        per = (d.get("ste") or {}).get("per_sequence_SA") or {}
    else:
        per = {k: (v.get("SA") if isinstance(v, dict) else v) for k, v in (d.get("per_sequence") or {}).items()}
        per = {k: (100 * v if v is not None and v <= 1.0 else v) for k, v in per.items()}
    return per, who, p


def _wrong_object_counts(dataset_id, split, idx):
    p, who = RECORDED_VAL_PRED.get(dataset_id, (None, None))
    if split != "val" or not p or not (REPO / p).exists():
        return None, who, p
    out = {}
    for s in idx:
        f = REPO / p / (s["name"] + (".txt" if dataset_id == "cst2025" else ".json"))
        if not f.exists():
            continue
        res = json.loads(f.read_text(encoding="utf-8"))["res"]
        seq = data.load_sequence(dataset_id, split, s["name"], "IR")
        n, k = min(len(res), seq.n_frames), 0
        for t in range(seq.start_frame + 1, n):
            r, g = res[t], seq.box_xywh(t)
            if g is not None and isinstance(r, list) and len(r) == 4 and r[2] > 0:
                ix = max(0.0, min(r[0] + r[2], g[0] + g[2]) - max(r[0], g[0])) * max(0.0, min(r[1] + r[3], g[1] + g[3]) - max(r[1], g[1]))
                k += ix == 0
        out[s["name"]] = k
    return out, who, p


def resolve_preset(preset_id: str, dataset_id: str, split: str, modality: str = "IR", custom=None, purpose="studio") -> dict:
    idx = sequence_index(dataset_id, split, modality, purpose)
    by = {s["name"]: s for s in idx}
    cst = dataset_id == "cst2025"

    def item(s, why):
        return {"sequence": s["name"], "why": why, "n_frames": s["n_frames"]}

    if preset_id == "standard":
        return _res(preset_id, "fixed_rule", f"all {len(idx)} sequences of {dataset_id} {split} (official split)", "dataset labels",
                    [item(s, "member of the official split") for s in idx])
    if preset_id == "standard_quick":
        if cst:
            seen, items = set(), []
            for s in idx:
                if s["category"] not in seen:
                    seen.add(s["category"])
                    items.append(item(s, f"first sequence (sorted) of scene category '{s['category']}'"))
            rule = "the first sequence (sorted by name) of each CST scene category"
        else:
            pick = sorted({int(round(i)) for i in np.linspace(0, len(idx) - 1, min(8, len(idx)))})
            items = [item(idx[i], f"position {i} of the sorted split (8 evenly spaced)") for i in pick]
            rule = "8 evenly spaced sequences of the name-sorted split"
        return _res(preset_id, "fixed_rule", rule, "dataset labels", items)
    if preset_id == "target_absent":
        el = sorted([s for s in idx if s["n_absent"] > 0], key=lambda s: -s["absent_fraction"])
        return _res(preset_id, "official_annotation", "exist = 0 on at least one frame after the start frame; sorted by absent fraction",
                    "exist flags in the official labels",
                    [item(s, f"{s['n_absent']} absent frames ({100 * s['absent_fraction']:.0f} %), {s['n_absent_episodes']} episode(s)") for s in el])
    if preset_id == "occlusion_reappearance":
        el = []
        for s in idx:
            why = []
            if s["reappearance_frames"]:
                why.append(f"{len(s['reappearance_frames'])} reappearance(s) (exist 0 -> 1), first at frame {s['reappearance_frames'][0]}")
            if cst and (s["frame_attrs"].get("OC") or s["frame_attrs"].get("OV")):
                why.append(f"OC {s['frame_attrs'].get('OC', 0)} / OV {s['frame_attrs'].get('OV', 0)} flagged frames")
            if not cst and ({"OC", "OV"} & set(s["seq_attrs"])):
                why.append("sequence attribute " + "/".join(sorted({"OC", "OV"} & set(s["seq_attrs"]))))
            if why:
                el.append(item(s, "; ".join(why)))
        basis_note = ("CST per-frame OC / OV flags (in CST they mark the absent frames caused by occlusion / leaving the view) + exist transitions"
                      if cst else "Anti-UAV sequence attributes OC / OV (label_new) + exist transitions")
        return _res(preset_id, "official_annotation", "a reappearance (exist 0 -> 1 after the start frame) or an occlusion / out-of-view flag",
                    basis_note, el)
    if preset_id == "small_targets":
        el = sorted([s for s in idx if s["n_visible"] and s["n_tiny"] / max(s["n_visible"], 1) >= 0.25], key=lambda s: -s["n_tiny"])
        if not cst:
            lr = [s for s in idx if "LR" in s["seq_attrs"] and s not in el]
            el += lr
        return _res(preset_id, "derived_from_gt", "at least 25 % of visible frames have a GT box diagonal <= 10 px (CST 'tiny' bin)"
                    + ("" if cst else "; plus Anti-UAV sequences carrying the official LR (low resolution) attribute"),
                    "ground-truth boxes" + ("" if cst else " + label_new attributes"),
                    [item(s, f"{s['n_tiny']} tiny frames of {s['n_visible']} visible; median diagonal {s['diag_median'] or 0:.1f} px"
                          + (" ; LR attribute" if "LR" in s["seq_attrs"] else "")) for s in el])
    if preset_id == "large_targets":
        el = sorted([s for s in idx if s["n_large"] >= 20], key=lambda s: -s["n_large"])
        return _res(preset_id, "derived_from_gt", "at least 20 visible frames with a GT box diagonal > 50 px (CST 'large' bin)", "ground-truth boxes",
                    [item(s, f"{s['n_large']} large frames; max diagonal {s['diag_max']:.0f} px") for s in el],
                    "no sequence has 20 or more frames with a box diagonal > 50 px")
    if preset_id == "fast_motion":
        sp = [s["speed_p95"] for s in idx if s["speed_p95"] is not None]
        thr = float(np.percentile(sp, 75)) if sp else None
        el = []
        for s in idx:
            why = []
            if cst and s["frame_attrs"].get("FM"):
                why.append(f"official FM flag on {s['frame_attrs']['FM']} frames")
            if not cst and "FM" in s["seq_attrs"]:
                why.append("official FM sequence attribute")
            if thr is not None and s["speed_p95"] is not None and s["speed_p95"] >= thr:
                why.append(f"derived: 95th-percentile GT-centre speed {s['speed_p95']:.1f} px/frame >= {thr:.1f} (fastest quarter)")
            if why:
                el.append(item(s, "; ".join(why)))
        return _res(preset_id, "official_annotation+derived_from_gt",
                    "official fast-motion flag, OR in the fastest quarter of this split by 95th-percentile GT-centre displacement "
                    "(px / frame between consecutive visible frames; camera motion is not separated)", "labels + ground-truth boxes", el)
    if preset_id == "clutter":
        if not cst:
            return _res(preset_id, "unavailable", "—", "—", [], "Anti-UAV ships no clutter / distractor attribute (its attributes are "
                        "OV OC FM SV LI TC LR); use 'Confusing objects' (recorded wrong-object locks) instead")
        el = sorted([s for s in idx if s["frame_attrs"].get("DBC", 0) + s["frame_attrs"].get("CDB", 0) > 0],
                    key=lambda s: -(s["frame_attrs"].get("DBC", 0) + s["frame_attrs"].get("CDB", 0)))
        return _res(preset_id, "official_annotation", "official per-frame DBC (dynamic background clutter) or CDB (complex dynamic background) flags",
                    "CST IR_label.json", [item(s, f"DBC {s['frame_attrs'].get('DBC', 0)} / CDB {s['frame_attrs'].get('CDB', 0)} frames") for s in el])
    if preset_id == "thermal_crossover":
        if cst:
            el = sorted([s for s in idx if s["frame_attrs"].get("TC")], key=lambda s: -s["frame_attrs"]["TC"])
            return _res(preset_id, "official_annotation", "official per-frame TC flag", "CST IR_label.json",
                        [item(s, f"TC on {s['frame_attrs']['TC']} frames") for s in el])
        el = [s for s in idx if "TC" in s["seq_attrs"]]
        return _res(preset_id, "official_annotation", "official TC sequence attribute (with TC-EASY / MID / HARD level)", "data/anti_uav/label_new",
                    [item(s, "TC " + "/".join(a for a in s["seq_attrs"] if a.startswith("TC-"))) for s in el])
    if preset_id == "long_sequence":
        lens = np.array([s["n_frames"] for s in idx])
        q3, med = float(np.percentile(lens, 75)), float(np.median(lens))
        thr = max(q3, 1.2 * med)
        el = sorted([s for s in idx if s["n_frames"] >= thr], key=lambda s: -s["n_frames"])
        return _res(preset_id, "fixed_rule", f"n_frames >= max(Q3, 1.2 x median) = {thr:.0f} frames (Q3 {q3:.0f}, median {med:.0f})",
                    "dataset labels", [item(s, f"{s['n_frames']} frames") for s in el],
                    f"no sequence is long relative to this split (median {med:.0f}, max {lens.max()} frames)")
    if preset_id == "confusing_objects":
        cnt, who, src = _wrong_object_counts(dataset_id, split, idx)
        if cnt is None:
            return _res(preset_id, "unavailable", "—", src or "—", [], f"needs recorded per-frame VAL predictions ({who or 'none registered'}) — "
                        "available for VAL only")
        el = sorted([s for s in idx if cnt.get(s["name"], 0) >= 20], key=lambda s: -cnt[s["name"]])
        return _res(preset_id, "recorded_results", "at least 20 frames on which the recorded box has IoU 0 with a visible GT target "
                    f"(a lock on another object) — {who}", src,
                    [item(s, f"{cnt[s['name']]} wrong-object frames in the recorded predictions") for s in el])
    if preset_id == "difficult_cases":
        per, who, src = _recorded_val_scores(dataset_id)
        if split != "val" or not per:
            return _res(preset_id, "unavailable", "—", src or "—", [], "needs a recorded per-sequence VAL result; for TEST use the recorded "
                        "replay (Live Comparison -> recorded predictions), which shows the registered passes' own predictions")
        el = sorted([s for s in idx if s["name"] in per and per[s["name"]] is not None], key=lambda s: per[s["name"]])[:8]
        return _res(preset_id, "recorded_results", f"the 8 lowest per-sequence SA values of {who}", src,
                    [item(s, f"recorded SA {per[s['name']]:.1f}") for s in el])
    if preset_id == "custom":
        names = [n for n in (custom or []) if n in by]
        return _res(preset_id, "user_selection", "chosen by hand", "user", [item(by[n], "selected by hand") for n in names],
                    "choose sequences below")
    raise KeyError(f"unknown preset {preset_id!r}")


def frame_flags(seq, dataset_id: str) -> dict:
    """per-frame boolean masks used by scenario metrics: official CST flags, absent, size bins, reappearance windows (30 frames)."""
    n = seq.n_frames
    ex = np.asarray(seq.exist[:n], dtype=bool)
    g = np.array([b if (e and isinstance(b, (list, tuple)) and len(b) == 4) else [np.nan] * 4 for b, e in zip(seq.gt_rect[:n], ex)], float)
    diag = np.hypot(g[:, 2], g[:, 3])
    out = {"absent": ~ex, "visible": ex, "tiny": ex & (diag <= TINY_PX), "large": ex & (diag > LARGE_PX)}
    re = np.zeros(n, bool)
    for t in np.flatnonzero((~ex[:-1]) & ex[1:]) + 1:
        re[t:t + 30] = True
    out["after_reappearance_30f"] = re & ex
    for k, v in (getattr(seq, "cst_attributes", {}) or {}).items():
        if k in CST_FRAME_ATTRS:
            out[k] = np.asarray(v[:n], dtype=float) > 0
    return out
