"""Repository paths used by the workspace layer (never hard-code them elsewhere)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "results/registry"
CATALOG = REGISTRY / "catalog"
RECOVERY = REPO / "results/artifacts" / "hpc_recovery_20261013"
RECOVERY_ROOT = RECOVERY / "aerotrack_server"
STUDIO_RUNS = REPO / "results/runs" / "studio"          # git-ignored outputs of studio inference / evaluations
STUDIO_INDEX = STUDIO_RUNS / "index.jsonl"       # one line per studio evaluation (append-only)
PYTHON = REPO / "venv" / "Scripts" / "python.exe"

# directories the workspace tools never write into (pre-existing untracked work + the backup)
PROTECTED = (RECOVERY, REPO / "results/runs" / "_deploy", REPO / "results/runs" / "tmp_wheels", REPO / "results/runs" / "v2_causal_g4_s0_e20")


def rel(p) -> str:
    """repo-relative POSIX path (or the absolute path if outside the repo)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def is_protected(p) -> bool:
    p = Path(p).resolve()
    return any(p == d.resolve() or d.resolve() in p.parents for d in PROTECTED)
