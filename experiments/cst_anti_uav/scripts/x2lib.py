"""Constants of registered experiment X2 (experiments/cst_anti_uav/preregistration/X2_seeds_single_frame.json)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REGISTRATION = Path("experiments/cst_anti_uav/preregistration/X2_seeds_single_frame.json")
FROZEN = Path("experiments/cst_anti_uav/frozen_x2_params.json")
VAL_REPORT = Path("experiments/cst_anti_uav/reports/X2_val")                 # .json / .md
FINAL_JSON = Path("experiments/cst_anti_uav/reports/x2_final_test.json")
TEST_LEDGER = Path("experiments/cst_anti_uav/reports/x2_test_ledger.jsonl")
PRED_ROOT = Path("results/runs/cst_anti_uav/x2_test_predictions")
ARMS = {"causal_g4": "n__causal_g4", "ttt": "n__ttt"}     # arm -> v3 config (same network 'n', same recipe; window only differs)
SEEDS = (0, 1, 2)
DOMAINS = "cst"
X1_RUN = "v3_n__causal_g4_cst_s0_e20"                      # X1's candidate = the causal_g4 seed-0 run; its TEST pass is X1's own


def run_name(arm, seed):
    return f"v3_{ARMS[arm]}_cst_s{seed}_e20"               # = v3lib.run_name(config + domain_tag(('cst',)), seed, 20)


def all_runs():
    return [run_name(a, s) for a in ARMS for s in SEEDS]


def new_runs():
    """Runs X2 trains (X1's run already exists)."""
    return [r for r in all_runs() if r != X1_RUN]
