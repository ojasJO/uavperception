"""Reproducibility helpers: seeding, environment snapshot, git revision, file hashing."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..paths import REPO_ROOT


def set_seed(seed: int, cudnn_benchmark: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # GlobalTrack sets cudnn_benchmark=True; benchmark mode picks kernels by timing and is a known,
    # documented source of run-to-run non-determinism. Recorded in run_meta.
    torch.backends.cudnn.benchmark = cudnn_benchmark


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_sha256(obj: Any) -> str:
    return text_sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001 - best effort
            return ""
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit or None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(status),
        "dirty_files": [ln[3:] for ln in status.splitlines()][:200] if status else [],
    }


def env_snapshot() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:  # noqa: BLE001
        info["torchvision"] = None
    for mod in ("numpy", "cv2", "PIL", "yaml"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            info[mod] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": p.name, "total_memory_gb": round(p.total_memory / 1e9, 2),
                       "compute_capability": f"{p.major}.{p.minor}"}
        try:
            drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            info["gpu"]["driver"] = drv or None
        except Exception:  # noqa: BLE001
            info["gpu"]["driver"] = None
    return info


def run_metadata(cfg: dict, extra: dict | None = None) -> dict:
    meta = {
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
        "cwd": os.getcwd(),
        "seed": cfg.get("seed"),
        "run_name": cfg.get("run_name"),
        "phase": cfg.get("phase"),
        "config_source_files": cfg.get("_source_files", []),
        "git": git_info(),
        "env": env_snapshot(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
