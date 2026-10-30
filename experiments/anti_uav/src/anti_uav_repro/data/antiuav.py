"""Anti-UAV sequence index, split access control, and dataset manifests.

Data layout consumed (see docs/ANTI_UAV_REPRODUCTION_SPEC.md §2):

    <data.root>/<split>/<seq>/{infrared,visible}.json        labels: {"exist": [0/1...], "gt_rect": [[x,y,w,h]...]}
    <data.root>/label_new/<split>.json                        {seq: [attribute, ...]}
    <data.frames_root>/antiuav_<split>_<seq>__{IR,RGB}/NNNNNN.jpg   decoded frames, 1-indexed

Split access is guarded: the TEST split can only be loaded with purpose="final_eval". Training,
tuning and monitoring code paths call this module with other purposes and therefore *cannot*
load test data, by construction.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from ..paths import resolve
from ..utils.repro import file_sha256, json_sha256

SPLITS = ("train", "val", "test")
MODALITY_FILE = {"IR": "infrared", "RGB": "visible"}
MODALITY_DIRTAG = {"IR": "IR", "RGB": "RGB"}

# Official start-frame corrections (github.com/ucas-vg/Anti-UAV README, "Data annotation correction").
# All other sequences start at the first frame with exist == 1.
START_FRAME_CORRECTIONS = {
    ("20190925_130434_1_2", "IR"): 16,
    ("20190926_134054_1_1", "RGB"): 150,
    ("20190925_194211_1_3", "RGB"): 150,
    ("20190925_194211_1_4", "RGB"): 1,
}

PURPOSES = ("train", "tune", "monitor", "final_eval", "verify")


class TestSplitAccessError(PermissionError):
    """Raised when a non-final-evaluation code path asks for the test split."""


@dataclass
class Sequence:
    name: str
    split: str
    modality: str
    label_path: Path
    frames_dir: Path
    exist: list[int]
    gt_rect: list[list[float]]           # [x, y, w, h] per frame (may be empty list / junk when exist==0)
    attributes: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None

    # ------------------------------------------------------------------ basic properties
    @property
    def n_frames(self) -> int:
        return len(self.exist)

    @property
    def n_exist(self) -> int:
        return int(sum(1 for e in self.exist if e))

    @property
    def start_frame(self) -> int:
        """Tracking init frame: official correction if any, else first frame with exist == 1."""
        key = (self.name, self.modality)
        if key in START_FRAME_CORRECTIONS:
            return START_FRAME_CORRECTIONS[key]
        for i, e in enumerate(self.exist):
            if e:
                return i
        raise ValueError(f"sequence {self.name}/{self.modality} has no frame with exist==1")

    def frame_path(self, idx: int) -> Path:
        return self.frames_dir / f"{idx + 1:06d}.jpg"

    def box_xywh(self, idx: int) -> np.ndarray | None:
        if not self.exist[idx]:
            return None
        r = self.gt_rect[idx]
        if not (isinstance(r, (list, tuple)) and len(r) == 4):
            return None
        return np.asarray(r, dtype=np.float32)

    def valid_frames(self) -> np.ndarray:
        """Indices with exist == 1 and a well-formed 4-number rect."""
        out = [i for i in range(self.n_frames) if self.box_xywh(i) is not None]
        return np.asarray(out, dtype=np.int64)

    def image_size(self) -> tuple[int, int]:
        """(width, height) read from the first frame on disk (cached)."""
        if self.width is None or self.height is None:
            import cv2
            im = cv2.imread(str(self.frame_path(0)), cv2.IMREAD_UNCHANGED)
            if im is None:
                raise FileNotFoundError(self.frame_path(0))
            self.height, self.width = im.shape[:2]
        return self.width, self.height


# ---------------------------------------------------------------------------------------- loading

def _attributes_for_split(root: Path, split: str) -> dict[str, list[str]]:
    p = root / "label_new" / f"{split}.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sequences(data_root: str | Path, frames_root: str | Path, split: str, modality: str,
                   purpose: str, require_frames: bool = True) -> list[Sequence]:
    """Load all sequences of one split/modality.

    purpose: 'train' | 'tune' | 'monitor' | 'final_eval' | 'verify'.
    The test split is only served for purpose == 'final_eval' (or 'verify', which never touches
    images for learning and is used by the dataset-verification script only).
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if modality not in MODALITY_FILE:
        raise ValueError(f"unknown modality {modality!r}")
    if purpose not in PURPOSES:
        raise ValueError(f"unknown purpose {purpose!r}; allowed {PURPOSES}")
    if split == "test" and purpose not in ("final_eval", "verify"):
        raise TestSplitAccessError(
            f"Refusing to load the TEST split for purpose={purpose!r}. "
            "TEST is reserved for the final evaluation (purpose='final_eval'); use 'val' for "
            "training-time monitoring, hyper-parameter or checkpoint decisions.")

    root = resolve(data_root)
    froot = resolve(frames_root)
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)
    attrs = _attributes_for_split(root, split)
    seqs: list[Sequence] = []
    for seq_name in sorted(os.listdir(split_dir)):
        seq_dir = split_dir / seq_name
        if not seq_dir.is_dir():
            continue
        label_path = seq_dir / f"{MODALITY_FILE[modality]}.json"
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        with open(label_path, "r", encoding="utf-8") as f:
            lab = json.load(f)
        exist = [int(bool(e)) for e in lab["exist"]]
        rects = lab["gt_rect"]
        if len(rects) != len(exist):
            raise ValueError(f"{label_path}: len(gt_rect)={len(rects)} != len(exist)={len(exist)}")
        frames_dir = froot / f"antiuav_{split}_{seq_name}__{MODALITY_DIRTAG[modality]}"
        if require_frames and not frames_dir.is_dir():
            raise FileNotFoundError(f"decoded frames missing: {frames_dir}")
        seqs.append(Sequence(name=seq_name, split=split, modality=modality, label_path=label_path,
                             frames_dir=frames_dir, exist=exist, gt_rect=rects,
                             attributes=list(attrs.get(seq_name, []))))
    if not seqs:
        raise RuntimeError(f"no sequences found under {split_dir}")
    return seqs


# --------------------------------------------------------------------------------------- manifest

def build_manifest(seqs: Iterable[Sequence], check_frames: bool = True) -> dict:
    """Per-sequence identity record: label-file hash, frame count on disk, start frame, boundary frame hashes.

    Hashing every frame (600k JPEGs) is impractical; the label hash + frame count + first/middle/last
    frame hashes pin the split identity tightly enough to detect substitution or truncation.
    """
    entries = []
    for s in seqs:
        n_disk = None
        f_hashes = {}
        if check_frames:
            n_disk = sum(1 for p in os.scandir(s.frames_dir) if p.name.endswith(".jpg"))
            for tag, idx in (("first", 0), ("middle", s.n_frames // 2), ("last", s.n_frames - 1)):
                fp = s.frame_path(idx)
                f_hashes[tag] = file_sha256(fp) if fp.exists() else None
        entries.append({
            "name": s.name, "split": s.split, "modality": s.modality,
            "n_frames": s.n_frames, "n_exist": s.n_exist, "start_frame": s.start_frame,
            "attributes": s.attributes,
            "label_sha256": file_sha256(s.label_path),
            "frames_on_disk": n_disk,
            "frame_sha256": f_hashes,
        })
    manifest = {
        "schema": "anti_uav_repro.manifest.v1",
        "split": entries[0]["split"] if entries else None,
        "modality": entries[0]["modality"] if entries else None,
        "n_sequences": len(entries),
        "n_frames": int(sum(e["n_frames"] for e in entries)),
        "n_exist": int(sum(e["n_exist"] for e in entries)),
        "sequences": entries,
    }
    manifest["manifest_sha256"] = json_sha256(manifest["sequences"])
    return manifest


def manifest_path(manifests_dir: Path, split: str, modality: str) -> Path:
    return Path(manifests_dir) / f"antiuav_{modality}_{split}.manifest.json"


def load_manifest(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_against_manifest(seqs: list[Sequence], manifest: dict) -> list[str]:
    """Return a list of human-readable mismatches (empty == identical split)."""
    problems = []
    by_name = {e["name"]: e for e in manifest["sequences"]}
    names = {s.name for s in seqs}
    if set(by_name) != names:
        problems.append(f"sequence set differs: +{sorted(names - set(by_name))[:5]} -{sorted(set(by_name) - names)[:5]}")
    for s in seqs:
        e = by_name.get(s.name)
        if e is None:
            continue
        if e["n_frames"] != s.n_frames:
            problems.append(f"{s.name}: n_frames {s.n_frames} != manifest {e['n_frames']}")
        if e["label_sha256"] != file_sha256(s.label_path):
            problems.append(f"{s.name}: label file hash changed")
    return problems
