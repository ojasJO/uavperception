"""Dataset access for the studio: sequence listing and loading for TRAIN / VAL only (aerolab.test_lock refuses TEST).

Uses the experiments' own loaders: Anti-UAV through `anti_uav_repro.data.antiuav.load_sequences` (official labels, start-frame
corrections, frame paths) and CST through the same `Sequence` class built from IR_label.json exactly as
experiments/systems/scripts/track_eval.py does. Track C trajectories through `tc_data.load_split` (which also seals TEST).
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

from .paths import REPO
from .test_lock import assert_split_allowed

ANTIUAV_LABELS = REPO / "data" / "anti_uav"
ANTIUAV_FRAMES = REPO / "b_pipeline" / "data_unified" / "anti_uav" / "images"
CST_ROOT = REPO / "data" / "cst_anti_uav_full" / "CST-AntiUAV" / "CST-AntiUAV"
TRACKC_SCRIPTS = REPO / "experiments" / "aerotrack_temporal" / "scripts"
SOT_DATASETS = {"antiuav2021": ("IR", "RGB"), "cst2025": ("IR",)}


def _src_path():
    p = str(REPO / "experiments" / "anti_uav" / "src")
    if p not in sys.path:
        sys.path.insert(0, p)


def list_sequences(dataset_id: str, split: str, modality: str = "IR", purpose: str = "studio") -> list[str]:
    """cheap: directory listing only"""
    assert_split_allowed(split, purpose)
    if dataset_id == "antiuav2021":
        d = ANTIUAV_LABELS / split
        fn = "infrared.json" if modality == "IR" else "visible.json"
        return sorted(x.name for x in d.iterdir() if (x / fn).exists()) if d.exists() else []
    if dataset_id == "cst2025":
        d = CST_ROOT / split
        return sorted(x.name for x in d.iterdir() if (x / "IR_label.json").exists()) if d.exists() else []
    if dataset_id == "trackc_trajectories":
        meta = json.loads((REPO / "runs" / "aerotrack_temporal" / "_data" / f"traj_{split}.json").read_text(encoding="utf-8"))
        return [s["key"] for s in meta["sequences"]]
    raise KeyError(f"no sequence access for dataset {dataset_id!r}")


@lru_cache(maxsize=6)
def _antiuav_all(split: str, modality: str):
    _src_path()
    from anti_uav_repro.data.antiuav import load_sequences
    # TEST is reachable here only after assert_split_allowed(..., REPLAY_PURPOSE): Track A's read-only 'verify' purpose (never used
    # for learning), so recorded predictions can be drawn over the official labels with Track A's start-frame handling.
    return {s.name: s for s in load_sequences(str(ANTIUAV_LABELS.relative_to(REPO)), str(ANTIUAV_FRAMES.relative_to(REPO)), split, modality,
                                              purpose="verify" if split == "test" else "tune")}


def load_sequence(dataset_id: str, split: str, name: str, modality: str = "IR", purpose: str = "studio"):
    """-> anti_uav_repro Sequence (exist, gt_rect xywh, frame_path(t), start_frame). TEST only with purpose=REPLAY_PURPOSE."""
    assert_split_allowed(split, purpose)
    _src_path()
    if dataset_id == "antiuav2021":
        seqs = _antiuav_all(split, modality)
        if name not in seqs:
            raise KeyError(f"{name} not in Anti-UAV {split}/{modality}")
        return seqs[name]
    if dataset_id == "cst2025":
        from anti_uav_repro.data.antiuav import Sequence
        lab = CST_ROOT / split / name / "IR_label.json"
        a = json.loads(lab.read_text(encoding="utf-8"))
        s = Sequence(name=name, split=split, modality="IR", label_path=lab, frames_dir=lab.parent,
                     exist=[int(e) for e in a["exist"]], gt_rect=a["gt"])
        s.cst_attributes = {k: a[k] for k in ("OC", "OV", "SV", "TC", "DBC", "CDB", "FM", "SIZE") if k in a and len(a[k]) == s.n_frames}
        return s
    raise KeyError(f"no sequence loader for dataset {dataset_id!r}")


def frame_files(seq) -> list[Path]:
    return [seq.frame_path(t) for t in range(seq.n_frames)]


def trackc_sequences(split: str, keys=None) -> list[dict]:
    assert_split_allowed(split)
    if str(TRACKC_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(TRACKC_SCRIPTS))
    import tc_data
    seqs = _trackc_split(split)
    return [s for s in seqs if keys is None or s["key"] in keys]


@lru_cache(maxsize=2)
def _trackc_split(split):
    import tc_data
    return tc_data.load_split(split, purpose="develop")


def dataset_examples(dataset_id: str, split: str, modality: str = "IR", n: int = 3) -> list[dict]:
    """a few (sequence, frame path, label) examples for the dataset browser, without loading the whole split"""
    out = []
    for name in list_sequences(dataset_id, split, modality)[:n]:
        s = load_sequence(dataset_id, split, name, modality)
        t = s.start_frame
        out.append({"sequence": name, "n_frames": s.n_frames, "n_exist": s.n_exist, "frame": t, "path": s.frame_path(t),
                    "box_xywh": None if s.box_xywh(t) is None else [float(v) for v in s.box_xywh(t)]})
    return out
