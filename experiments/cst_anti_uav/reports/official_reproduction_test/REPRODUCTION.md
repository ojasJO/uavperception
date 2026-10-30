# Official-protocol reproduction of the CST Anti-UAV results (released TrainedCST predictions)

*Generated from `experiments\cst_anti_uav\reports\official_reproduction_test\official_scores.json` (2026-10-12T03:23:46+0530); toolkit commit `bd92b984`, unmodified; results archive sha256 `1c886465697b…`; subset `test`, 60 sequences.*

| tracker | paper mSA | ours mSA | Δ | paper P | ours P@20 | paper S | ours S-AUC | Track A mSA | max per-seq |Δ| official vs Track A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GlobalTrack | 35.92 | 35.92 | +0.00 | 58.72 | 58.72 | 35.38 | 35.38 | 35.92 | 0.0e+00 |
| SiamDT | 35.84 | 35.84 | +0.00 | 57.96 | 57.96 | 35.28 | 35.28 | 35.84 | 0.0e+00 |
| PrDiMP | 29.76 | 29.76 | -0.00 | 42.55 | 42.55 | 29.28 | 29.28 | 29.76 | 0.0e+00 |
| MixformerV2 | 29.39 | 29.39 | -0.00 | 52.37 | 52.37 | 29.09 | 29.09 | 29.39 | 0.0e+00 |
| AQATrack | 27.60 | 27.60 | -0.00 | 50.43 | 50.43 | 27.45 | 27.45 | 27.60 | 0.0e+00 |
| RomTrack | 27.43 | 27.43 | -0.00 | 48.20 | 48.20 | 27.23 | 27.23 | 27.43 | 0.0e+00 |
| GRM | 26.96 | 26.96 | -0.00 | 48.70 | 48.70 | 26.80 | 26.80 | 26.96 | 0.0e+00 |
| DropTrack | 26.89 | 26.89 | -0.00 | 49.08 | 49.08 | 26.65 | 26.65 | 26.89 | 0.0e+00 |
| AiATrack | 26.83 | 26.83 | +0.00 | 47.65 | 47.65 | 26.64 | 26.64 | 26.83 | 0.0e+00 |
| SeqTrack | 26.53 | 26.53 | -0.00 | 47.67 | 47.67 | 26.38 | 26.38 | 26.53 | 0.0e+00 |
| Refocus-TIR | 26.51 | 26.51 | +0.00 | 47.49 | 47.49 | 26.26 | 26.26 | 26.51 | 0.0e+00 |
| Mixformer | 26.46 | 26.46 | -0.00 | 47.14 | 47.14 | 26.19 | 26.19 | 26.46 | 0.0e+00 |
| OSTrack_256 | 26.34 | 26.34 | -0.00 | 46.95 | 46.95 | 26.16 | 26.16 | 26.34 | 0.0e+00 |
| KYS | 26.28 | 26.28 | +0.00 | 40.20 | 40.20 | 25.93 | 25.93 | 26.28 | 0.0e+00 |
| Stark_st | 25.63 | 25.63 | -0.00 | 44.99 | 44.99 | 25.44 | 25.44 | 25.63 | 0.0e+00 |
| HIPTrack | 25.62 | 25.62 | -0.00 | 47.48 | 47.48 | 25.42 | 25.42 | 25.62 | 0.0e+00 |
| OSTrack_384 | 25.25 | 25.25 | -0.00 | 45.99 | 45.99 | 25.10 | 25.10 | 25.25 | 0.0e+00 |
| ARTrack_384 | 25.19 | 25.19 | +0.00 | 46.56 | 46.56 | 25.04 | 25.04 | 25.19 | 0.0e+00 |
| ARTrack_256 | 25.14 | 25.14 | +0.00 | 45.39 | 45.39 | 24.96 | 24.96 | 25.14 | 0.0e+00 |
| Stark_s | 23.27 | 23.27 | +0.00 | 40.54 | 40.54 | 23.10 | 23.10 | 23.27 | 0.0e+00 |
| Tomp | 22.84 | 22.84 | +0.00 | 39.81 | 39.81 | 22.66 | 22.66 | 22.84 | 0.0e+00 |
| ARTrackV2 | 22.14 | not in the released archive | | 39.73 | | 22.03 | | | |

Largest |ours − paper| over mSA, P and S of the released trackers: **0.0050** points.

## Frame-level attribute SA (Table 5, CST-trained GlobalTrack)

| attribute | paper | official toolkit (`Evaluation_for_ALL_Frame_SA.py`, as-is) | all flagged frames (corrected) | frames |
|---|---:|---:|---:|---:|
| OC | 44.53 | 44.53 | 0.02 | 6469 |
| OV | 65.77 | 79.74 | 0.04 | 6310 |
| SV | 37.62 | 37.62 | 37.62 | 33445 |
| TC | 26.63 | 26.63 | 26.63 | 42202 |
| DBC | 17.26 | 17.26 | 17.26 | 8624 |
| CDB | 26.95 | 26.96 | 26.96 | 14381 |
| tiny | 15.83 | — | 15.83 | 10354 |
| small | 46.14 | — | 46.13 | 45812 |
| normal | 59.15 | — | 59.52 | 2600 |
| large | 34.52 | — | 34.52 | 484 |

Tiny-target SA of PrDiMP (paper's best tiny-size tracker, 28.2): 28.21.
Best tiny-target SA among the released trackers: PrDiMP 28.21.
