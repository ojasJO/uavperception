"""The pre-registered plan of the generalization campaign (no heavy imports: the torch-free runner reads it too).

Frozen at the campaign's first launch: run_gen.py writes this dict and the sha256 of this file to ledger.jsonl (entry P1) and
refuses to continue if either changes afterwards. Nothing here may be edited after results exist."""

PLAN = {
    "reference": "stock",
    "candidates": ["spd_fp", "temporal_fp_g1", "temporal_fp_g4"],
    "stage_a_order": ["stock", "spd_fp", "temporal_fp_g1", "temporal_fp_g4"],
    "epochs": 20,
    "budget_h4_epochs": 40,
    "primary_metric": "cst_drone_AP50",          # held-out DEV fold, drone class only, last.pt
    "noninferiority_metric": "anti_uav_drone_AP50",
    "margin_ap50": 0.02,
    "rules": ["(1) fold-mean delta > +0.02", "(2) delta > 0 in each of the 3 folds",
              "(3) fold-pooled paired sequence bootstrap 95% CI of the mean delta excludes 0",
              "(4) fold-mean Anti-UAV drone AP50 delta > -0.02"],
    "futility": "a candidate with delta <= 0 on fold 0 is rejected (rule (2) cannot hold); folds 1-2 are run only for survivors",
    "checkpoint": "last.pt (no selection)",
    "seed": "fold index; 0 for the full-TRAIN final retrain",
    "dev_set": "held-out fold of folds.json, 5,000 images per dataset evenly over sequences",
    "secondary": "official VAL (registered set 6fb41d44d45b) with the run's own input; the repair series' four metrics",
    "h4_budget": "stock at 40 vs 20 epochs, same rule; fold 0 first, folds 1-2 only if ahead and time permits",
    "final": "only if >= 1 candidate is supported: winner = largest fold-mean delta; retrain on the full TRAIN split (seed 0, 20 epochs, "
             "last.pt); official VAL; TEST once (ledger-guarded). No candidate supported -> the test split is not touched",
    "time_budget_h": {"dev_runs_end_within": 24.0, "final_ends_within": 28.0},
    "estimated_hours": {"run_e20": 2.4, "run_e40": 4.7, "eval": 0.15},
}
