# CST Anti-UAV — baseline results

Registry of every method below: `baseline_registry.json`. Protocol: `PAPER_PROTOCOL.md`; compatibility: `PROTOCOL_COMPATIBILITY.md`.

## 1. Official-protocol baselines on CST TEST (released predictions, official toolkit)

Scored by the unmodified official toolkit from the authors' released per-frame predictions
(`official_reproduction_test/official_scores.json`, `REPRODUCTION.md`). Every value equals the paper's Table 3 to rounding.

| tracker (CST-trained) | mSA | success AUC | precision@20px | tiny-target SA (≤ 10 px) |
|---|---:|---:|---:|---:|
| **GlobalTrack** (best published) | **35.92** | 35.38 | 58.72 | 15.83 |
| SiamDT | 35.84 | 35.28 | 57.96 | 14.98 |
| PrDiMP | 29.76 | 29.28 | 42.55 | **28.21** (best tiny) |
| MixFormer-V2 | 29.39 | 29.09 | 52.37 | 19.36 |
| AQATrack | 27.60 | 27.45 | 50.43 | 20.37 |
| ROMTrack | 27.43 | 27.23 | 48.20 | 16.79 |
| GRM | 26.96 | 26.80 | 48.70 | 18.44 |
| DropTrack | 26.89 | 26.65 | 49.08 | 18.82 |
| AiATrack | 26.83 | 26.64 | 47.65 | 17.23 |
| SeqTrack | 26.53 | 26.38 | 47.67 | 17.93 |
| Refocus-TIR | 26.51 | 26.26 | 47.49 | 18.48 |
| MixFormer | 26.46 | 26.19 | 47.14 | 18.13 |
| OSTrack-256 | 26.34 | 26.16 | 46.95 | 15.39 |
| KYS | 26.28 | 25.93 | 40.20 | 23.81 |
| Stark-ST | 25.63 | 25.44 | 44.99 | 16.68 |
| HIPTrack | 25.62 | 25.42 | 47.48 | 20.17 |
| OSTrack-384 | 25.25 | 25.10 | 45.99 | 18.24 |
| ARTrack-384 | 25.19 | 25.04 | 46.56 | 17.96 |
| ARTrack-256 | 25.14 | 24.96 | 45.39 | 13.95 |
| Stark-S | 23.27 | 23.10 | 40.54 | 13.75 |
| ToMP | 22.84 | 22.66 | 39.81 | 9.82 |
| ARTrackV2 | 22.14 (paper; predictions not released) | 22.03 | 39.73 | — |

Observations from the released predictions (official metric):
- None of the released trackers ever reports "absent" on the 12,925 target-absent TEST frames (17.9 % of all frames), so each of
  them scores 0 there. That alone caps a tracker's mSA well below 100 and is one reason the CST scores are low.
- The two best trackers are global-search / re-detection designs (GlobalTrack, SiamDT), as the paper notes.

## 2. AeroTrack baselines on CST VAL (development evidence; TEST results in `FINAL_REPORT.md`, `ABLATION_RESULTS.md`)

CST VAL = the official 40 sequences. It shares source videos with CST TRAIN (paper §3.3), so these numbers are optimistic
relative to TEST. "Cross-fitted": settings chosen on one half of VAL, applied to the other half (the honest VAL estimate).
"Tuned on all VAL": the settings frozen for TEST.

| run | training data | protocol-matched? | CST AP50 | STE mSA cross-fitted | STE mSA tuned on all VAL | argmax cross-fitted |
|---|---|---|---:|---:|---:|---:|
| `v3_n__causal_g4_cst_s0_e20` (X1) | CST train only | **yes** | 0.694 | **63.61** | 64.03 | 62.60 |
| `v2_causal_g4_s0_e20` | CST + Anti-UAV + Det-Fly | no (extra data) | 0.676 | 63.02 | 63.19 | 62.18 |

- The X1 VAL predictions were exported in the official format and scored by the official toolkit on the `val` subset:
  STE 64.03 / argmax 63.42 — identical to Track A's values (max per-sequence |Δ| = 0), success AUC 62.67, precision@20px 82.08
  (`val_official/v3_n__causal_g4_cst_s0_e20/official_scores.json`). This validates the export used by the TEST passes end to end.
- The OMEN re-run of the X1 trackers from the HPC's VAL cache reproduces the HPC's numbers exactly (64.03 / 63.42).
- No released-tracker predictions exist for VAL, so there is no paired VAL comparison with the published methods.

## 3. Existing (not protocol-matched) CST TEST result

`v2_causal_g4_s0_e20` (all-domain detector), registered v2 final TEST pass: CST STE mSA **56.55**, argmax 56.28 (60 sequences),
located 67.1 %, false alarms on absent frames 33.1 % (`experiments/aerotrack_v2/hpc/reports/final_test.json`). This shows the
size of the VAL → TEST drop for this stack (VAL 63.02 → TEST 56.55 for the same run) but cannot be compared with Table 3,
because the detector saw Anti-UAV and Det-Fly training data. The protocol-matched TEST numbers are X1's (60.08) and X2's (causal_g4 seeds 60.08 / 60.26 / 60.60) — `FINAL_REPORT.md`.
