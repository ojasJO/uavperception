# AeroTrack generalization investigation — report (SMOKE TEST OF THE MACHINERY)

*Generated 2026-09-26 10:38 by `scripts/report_gen.py` from `reports/*.json` and `ledger_smoke.jsonl`. Labels: [fact] read from files · [measured] computed in this series · [rule] outcome of the pre-registered rule · [inference] interpretation.*

## 1. Original problem

[fact] The repair grid (`experiments/aerotrack_repair/reports/ablation_grid_e20.md`) found no factor credited: in all five pre-registered comparisons the ablated model scored higher on validation, and stock YOLO11n beat the full AeroTrack recipe. The question here: is there a change with a plausible mechanism that improves generalisation, verified without tuning to the validation or test sets?

## 2–4. Evidence, diagnosis, confounds

- [measured] COCO-initialised parameters: SPD layout 48.4 % of 4.085 M, stride layout 93.7 % of 2.63 M (D2).
- [measured] function-preserving init: SPD vs stride max |Δ| 2.38e-03 on outputs up to 639; centre-frame stem ignores t±1 (max |Δ| 0.0e+00); the grid's tiled stem responds to the frame mean (max |Δ| 404) (D2).
- [fact] grid training loss epoch 0 → 19: baseline_ctrl 6.932 → 2.261; no_spd 5.746 → 2.143; single_frame 6.536 → 2.279; stock_yolo11n 5.68 → 2.156 (D2).
- [measured] Anti-UAV VAL shares 26 of 26 recording sessions with TRAIN; TEST shares 0 (D1).
- [measured] CST TEST scene categories absent from TRAIN: urban-areas, water (D1).
- [measured] Det-Fly VAL frames whose adjacent frame (id ±1) is in TRAIN: 1224 of 1469 (D1).
- [measured] median target motion per frame at the network input: CST 0.53 px (68 % < 1 px), Anti-UAV IR 3.64 px, RGB 3.0 px (D1).
- [measured] grid VAL comparisons with sequence-bootstrap 95 % CIs (sampling noise only): SPD-Conv micro_AP50 best -0.016 [-0.045, +0.005]; SPD-Conv micro_AP50 last -0.078 [-0.114, -0.050]; SPD-Conv cst_micro_AP50 best -0.039 [-0.104, +0.009]; SPD-Conv cst_micro_AP50 last -0.102 [-0.155, -0.067]; temporal stack cst_AP50 best -0.045 [-0.080, -0.005]; temporal stack cst_AP50 last -0.015 [-0.034, +0.008]; temporal stack drone_AP50 best -0.027 [-0.041, -0.013]; temporal stack drone_AP50 last -0.003 [-0.012, +0.006]; full recipe vs stock YOLO11n drone_AP50 best -0.055 [-0.068, -0.040]; full recipe vs stock YOLO11n drone_AP50 last -0.035 [-0.044, -0.025]; [exploratory] temporal within the stride layout cst_AP50 best -0.021 [-0.040, -0.005]; [exploratory] temporal within the stride layout cst_AP50 last +0.029 [+0.008, +0.053] (D4).
- [measured] CST micro drones at last.pt, found-but-mislocalised vs missed: baseline_ctrl 0.070 / 0.210; no_spd 0.081 / 0.134; single_frame 0.087 / 0.189; stock_yolo11n 0.065 / 0.162 (D5).

## 5–6. Experiments and exact configurations

[fact] Pre-registered plan (ledger P1, `scripts/plan_gen.py`): reference `stock`; candidates spd_fp, temporal_fp_g1, temporal_fp_g4; 20 epochs, recipe `c_model/train.py` STAGES['baseline'] (25,000 weighted samples/epoch, batch 32 × acc 2, AdamW 1e-3 cosine); function-preserving init for every configuration; seed = fold index; checkpoint last.pt (no selection); primary cst_drone_AP50 on the held-out DEV fold; non-inferiority anti_uav_drone_AP50; margin 0.02; rules: (1) fold-mean delta > +0.02; (2) delta > 0 in each of the 3 folds; (3) fold-pooled paired sequence bootstrap 95% CI of the mean delta excludes 0; (4) fold-mean Anti-UAV drone AP50 delta > -0.02; futility: a candidate with delta <= 0 on fold 0 is rejected (rule (2) cannot hold); folds 1-2 are run only for survivors.

| run | config | fold | seed | epochs | launches | training time | last.pt sha | init check |
|---|---|---|---|---|---|---|---|---|
| `gen_stock_f0_e1_smoke` | stock | 0 | 0 | 1 | 1 | 0.02 h | 899e458da678 | ok |
| `gen_spd_fp_f0_e1_smoke` | spd_fp | 0 | 0 | 1 | 1 | 0.02 h | ba5d91a29f3d | ok |
| `gen_temporal_fp_g1_f0_e1_smoke` | temporal_fp_g1 | 0 | 0 | 1 | 1 | 0.02 h | 992258003091 | ok |
| `gen_temporal_fp_g4_f0_e1_smoke` | temporal_fp_g4 | 0 | 0 | 1 | 1 | 0.02 h | ff5e7012197a | ok |
| `gen_stock_f0_e2_smoke` | stock | 0 | 0 | 2 | 1 | 0.02 h | 56db1e25b4b2 | ok |

## 7–8. Development results (held-out DEV folds: unseen Anti-UAV sessions, unseen CST sequences, held-out Det-Fly flight chunks)

| run | set | CST drone AP50 (primary) | Anti-UAV drone AP50 | Det-Fly drone AP50 | CST drone-micro AP50 | pooled drone AP50 | images |
|---|---|---|---|---|---|---|---|
| `gen_spd_fp_f0_e1_smoke` | dev | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 180 |
| `gen_stock_f0_e1_smoke` | dev | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 180 |
| `gen_stock_f0_e2_smoke` | dev | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 180 |
| `gen_temporal_fp_g1_f0_e1_smoke` | dev | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 180 |
| `gen_temporal_fp_g4_f0_e1_smoke` | dev | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 180 |

## 9. Decisions and statistical uncertainty

| candidate | vs | fold deltas (primary) | fold-mean Δ | sd | bootstrap 95 % CI | Anti-UAV mean Δ | [rule] status |
|---|---|---|---|---|---|---|---|
| spd_fp | stock | f0: +0.0000 | — | — | — | — | rejected: not ahead on fold 0 (pre-registered futility; rule (2) cannot hold) |
| temporal_fp_g1 | stock | f0: +0.0000 | — | — | — | — | rejected: not ahead on fold 0 (pre-registered futility; rule (2) cannot hold) |
| temporal_fp_g4 | stock | f0: +0.0000 | — | — | — | — | rejected: not ahead on fold 0 (pre-registered futility; rule (2) cannot hold) |
| stock_e2 | stock | f0: +0.0000 | — | — | — | — | rejected: not ahead on fold 0 (pre-registered futility; rule (2) cannot hold) |

## 10. Final test

[fact] The test split was not read in this series (no candidate satisfied the pre-registered rule, or the final step has not run). The repair series' single test evaluation of stock YOLO11n remains the reference result.

## 11–13. What improved, what did not, credibility

- [rule] supported: none; not supported / rejected: spd_fp (rejected: not ahead on fold 0 (pre-registered futility; rule (2) cannot hold)), temporal_fp_g1 (rejected: not ahead on fold 0 (pre-registered futility; rule (2) cannot hold)), temporal_fp_g4 (rejected: not ahead on fold 0 (pre-registered futility; rule (2) cannot hold)).
- [rule] training budget (H4, 40 vs 20 epochs): rejected: not ahead on fold 0 (pre-registered futility; rule (2) cannot hold).
