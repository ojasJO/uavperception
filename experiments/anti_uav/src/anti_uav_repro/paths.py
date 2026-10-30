"""Repository-relative path resolution. No experiment-specific paths are hard-coded anywhere else."""
from __future__ import annotations

from pathlib import Path

# .../experiments/anti_uav/src/anti_uav_repro/paths.py -> repo root is 4 parents up
PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent.parent          # experiments/anti_uav
REPO_ROOT = EXPERIMENT_DIR.parent.parent            # A:/AEROTRACK_OMEN

CONFIGS_DIR = EXPERIMENT_DIR / "configs"
RUNS_DIR = EXPERIMENT_DIR / "runs"
MANIFESTS_DIR = EXPERIMENT_DIR / "manifests"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
LOGS_DIR = EXPERIMENT_DIR / "logs"


def resolve(path: str | Path) -> Path:
    """Resolve a config path: absolute paths are kept, relative ones are relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)
