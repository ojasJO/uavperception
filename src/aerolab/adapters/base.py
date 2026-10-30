"""Shared inference contract. Adapters are thin wrappers around the experiments' own code; they never re-implement a model.

Normalised single-target-tracking result (`kind = 'sot'`):
    frames[i] = {frame, present (bool | None), box_xyxy (px | None), score (float | None), mode (str | None), detections (list | None)}
    res_xywh / processed / start_frame  -- Track A's prediction format, scored by anti_uav_repro.eval.metrics
    model_id, checkpoint_sha256, checkpoint_path, dataset_id, split, modality, sequence, device, timing_ms, params, params_source, warnings
Forecasting result (`kind = 'forecast'`): see src/aerolab/adapters/forecaster.py.
"""
import gc
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..hashing import sha256_file
from ..paths import REPO, is_protected


class AdapterError(RuntimeError):
    """A model cannot be loaded / run with the requested inputs; the message says why and what to do."""


@dataclass
class Compatibility:
    ok: bool
    reason: str = ""
    warnings: list = field(default_factory=list)


def pick_device(device: str | None = None) -> str:
    import torch
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise AdapterError("CUDA requested but torch.cuda.is_available() is False -- choose device 'cpu'")
    return device


def verify_file(path: Path, sha256: str) -> None:
    got = sha256_file(path)
    if got != sha256:
        raise AdapterError(f"{path}: SHA-256 {got[:12]}... does not match the registry ({sha256[:12]}...) -- refusing to load")


class ModelAdapter:
    id = "base"
    tasks: tuple = ()

    def __init__(self):
        self.model = None
        self.record = None
        self.device = None
        self.load_s = None
        self.weights_path = None

    # -- contract ---------------------------------------------------------------------------------------------
    def can_run(self, model_record: dict, dataset_id: str, modality: str | None = None) -> Compatibility:
        raise NotImplementedError

    def load(self, model_record: dict, device: str | None = None):
        raise NotImplementedError

    def validate_inputs(self, dataset_id: str, modality: str | None, sample) -> None:
        c = self.can_run(self.record, dataset_id, modality)
        if not c.ok:
            raise AdapterError(c.reason)

    def predict(self, sample, **kw) -> dict:
        raise NotImplementedError

    def get_metadata(self) -> dict:
        import torch
        return {"adapter": self.id, "model_id": (self.record or {}).get("model_id"), "checkpoint_sha256": (self.record or {}).get("sha256"),
                "checkpoint_path": None if self.weights_path is None else Path(self.weights_path).relative_to(REPO).as_posix()
                if Path(self.weights_path).is_relative_to(REPO) else str(self.weights_path),
                "device": self.device, "load_seconds": self.load_s, "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "python": platform.python_version()}

    def unload(self):
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- helpers ----------------------------------------------------------------------------------------------
    def _resolve_workspace_path(self, record: dict, registry, need_sibling: str | None = None) -> Path:
        """first workspace path holding exactly this model's bytes (own location or a byte-identical model id), optionally
        requiring `need_sibling` (relative to the weights' run directory, e.g. 'run_meta.json'). Never a backup path."""
        tried = []
        for p in registry.same_bytes_paths(record) if registry is not None else [loc["path"] for loc in record["locations"]]:
            f = REPO / p
            if is_protected(f) and "hpc_recovery" in p:
                tried.append(f"{p} (recovery backup: read-only archive, not used for inference)")
                continue
            if not f.exists():
                tried.append(f"{p} (missing)")
                continue
            if need_sibling and not (f.parent.parent / need_sibling).exists():
                tried.append(f"{p} (no {need_sibling} next to it)")
                continue
            return f
        raise AdapterError(f"no usable workspace copy of {record['model_id']}" + (f" with {need_sibling}" if need_sibling else "")
                           + ": " + "; ".join(tried) + ". Restore it with scripts/restore_from_recovery.py if it exists only in the backup.")

    @staticmethod
    def _timer():
        return time.perf_counter()
