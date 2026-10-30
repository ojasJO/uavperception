"""TEST-split lock.

Policy (docs/workspace/ARCHITECTURE.md §test lock): the studio and aerolab.evaluation NEVER evaluate a TEST split. Every TEST
split used in this repository has been exposed by at least one registered pass (ledgers below), so it is locked for model
development. Recorded TEST results may be displayed (read-only, with their registration); a new TEST evaluation requires its own
pre-registration and its own script, outside the studio.
"""
import json

from .paths import REPO

# every append-only TEST ledger in the repository (registration -> ledger file)
TEST_LEDGERS = {
    "Anti-UAV Protocol II (Direction B)": "experiments/anti_uav/reports/test_evaluation_ledger.jsonl",
    "AeroTrack repair grid": "experiments/aerotrack_repair/reports/test_evaluation_ledger.jsonl",
    "AeroTrack generalization selected detector": "experiments/aerotrack_generalization/reports/gen_test_ledger.jsonl",
    "AeroTrack v3 / F1": "experiments/aerotrack_v3/reports/v3_test_ledger.jsonl",
    "AeroTrack-X1 (CST)": "experiments/aerotrack_v3/reports/x1_test_ledger.jsonl",
    "AeroTrack-X2 (CST)": "experiments/cst_anti_uav/reports/x2_test_ledger.jsonl",
    "CF final causal system": "experiments/aerotrack_temporal/reports/cf_test_ledger.jsonl",
    "AeroTrack v2 final test": "experiments/aerotrack_v2/hpc/reports/v2_test_ledger.jsonl",
}
LOCKED_SPLITS = {"test"}
# The only purpose that may open a locked split: drawing RECORDED predictions of registered passes over their frames and GT.
# It never loads a model, never computes a new registered number and is not accepted by the evaluation service or the
# Flight Deck runner (they call assert_split_allowed with the default purpose).
REPLAY_PURPOSE = "replay_recorded"


class TestSplitLocked(PermissionError):
    """Raised when anything in the workspace layer is asked to evaluate a locked split."""


def assert_split_allowed(split: str, purpose: str = "studio"):
    if str(split).lower() in LOCKED_SPLITS and purpose != REPLAY_PURPOSE:
        raise TestSplitLocked(
            f"split '{split}' is locked: every TEST split in this repository has been exposed by registered passes "
            f"({', '.join(TEST_LEDGERS)}). The studio never runs TEST; recorded TEST results are read-only. "
            "A new final evaluation needs its own pre-registration and script.")


def consumed_registrations() -> list[dict]:
    """[{registration, ledger, exists, lines}] -- what has already used a TEST split."""
    out = []
    for name, p in TEST_LEDGERS.items():
        f = REPO / p
        lines = []
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except ValueError:
                        lines.append({"unparsed": line[:200]})
        out.append({"registration": name, "ledger": p, "exists": f.exists(), "n_lines": len(lines), "lines": lines})
    return out
