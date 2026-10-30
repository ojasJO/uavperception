# Anti-UAV paper benchmark — measured results

Generated from `benchmark_results.json` on 2026-09-23T03:39:16+0530. Do not edit by hand.

- Paper: Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking (Jiang et al., arXiv:2101.08466v3, 2021)
- Protocol: Protocol II (Anti-UAV train split used for fine-tuning), infrared modality

## Overall

| Run | Paper row | Split | Metric | Paper | Ours (live) | Delta | Verdict |
|---|---|---|---|---:|---:|---:|---|
| `normal_ir` | normal | val | mSA | 79.60 | 79.29 | -0.31 | reproduced |
| `normal_ir` | normal | test | mSA | 65.36 | 70.58 | +5.22 | not reproduced |
| `dfsc_ir` | DFSC | val | mSA | 80.09 | 79.26 | -0.83 | reproduced |
| `dfsc_ir` | DFSC | test | mSA | 66.04 | 70.29 | +4.25 | not reproduced |
| `normal_ir` | normal | val | precision | 87.61 | 99.56 | +11.95 | not reproduced |
| `normal_ir` | normal | val | success | 62.88 | 79.32 | +16.44 | not reproduced |
| `normal_ir` | normal | test | precision | 87.61 | 91.85 | +4.24 | not reproduced |
| `normal_ir` | normal | test | success | 62.88 | 71.01 | +8.13 | not reproduced |
| `dfsc_ir` | DFSC | val | precision | 87.77 | 99.57 | +11.80 | not reproduced |
| `dfsc_ir` | DFSC | val | success | 63.24 | 79.29 | +16.05 | not reproduced |
| `dfsc_ir` | DFSC | test | precision | 87.77 | 91.65 | +3.88 | not reproduced |
| `dfsc_ir` | DFSC | test | success | 63.24 | 70.72 | +7.48 | not reproduced |

## mSA by attribute

### normal_ir/val

| Attribute | n_seq | Paper | Ours (live) | Delta | Verdict |
|---|---:|---:|---:|---:|---|
| OV | 4 | 78.09 | 56.05 | -22.04 | not reproduced |
| OC | 5 | 81.42 | 78.26 | -3.16 | not reproduced |
| FM | 23 | 78.66 | 79.47 | +0.81 | reproduced |
| SV | 17 | 76.61 | 72.89 | -3.72 | not reproduced |
| LI | 29 | 79.95 | 79.94 | -0.01 | reproduced |
| TC-EASY | 17 | 81.23 | 81.62 | +0.39 | reproduced |
| TC-MID | 8 | 76.56 | 77.78 | +1.22 | approximately reproduced |
| TC-HARD | 7 | 74.04 | 76.71 | +2.67 | approximately reproduced |
| TC | 32 | 78.49 | 79.58 | +1.09 | approximately reproduced |
| LR | 10 | 73.55 | 76.94 | +3.39 | not reproduced |
| ALL | 67 | 79.60 | 79.29 | -0.31 | reproduced |

### normal_ir/test

| Attribute | n_seq | Paper | Ours (live) | Delta | Verdict |
|---|---:|---:|---:|---:|---|
| OV | 18 | 70.44 | 69.57 | -0.87 | reproduced |
| OC | 2 | 68.94 | 64.52 | -4.42 | not reproduced |
| FM | 31 | 60.66 | 70.43 | +9.77 | not reproduced |
| SV | 18 | 55.48 | 64.64 | +9.16 | not reproduced |
| LI | 18 | 59.78 | 75.51 | +15.73 | not reproduced |
| TC-EASY | 14 | 77.07 | 77.01 | -0.06 | reproduced |
| TC-MID | 31 | 64.64 | 72.31 | +7.67 | not reproduced |
| TC-HARD | 18 | 44.61 | 51.63 | +7.02 | not reproduced |
| TC | 63 | 61.68 | 67.45 | +5.77 | not reproduced |
| LR | 15 | 52.94 | 70.24 | +17.30 | not reproduced |
| ALL | 91 | 65.36 | 70.58 | +5.22 | not reproduced |

### dfsc_ir/val

| Attribute | n_seq | Paper | Ours (live) | Delta | Verdict |
|---|---:|---:|---:|---:|---|
| OV | 4 | 77.72 | 56.03 | -21.69 | not reproduced |
| OC | 5 | 82.70 | 78.24 | -4.46 | not reproduced |
| FM | 23 | 79.34 | 79.41 | +0.07 | reproduced |
| SV | 17 | 77.58 | 72.91 | -4.67 | not reproduced |
| LI | 29 | 80.31 | 79.88 | -0.43 | reproduced |
| TC-EASY | 17 | 81.48 | 81.68 | +0.20 | reproduced |
| TC-MID | 8 | 76.83 | 77.76 | +0.93 | reproduced |
| TC-HARD | 7 | 75.65 | 76.72 | +1.07 | approximately reproduced |
| TC | 32 | 79.04 | 79.61 | +0.57 | reproduced |
| LR | 10 | 74.33 | 76.94 | +2.61 | approximately reproduced |
| ALL | 67 | 80.09 | 79.26 | -0.83 | reproduced |

### dfsc_ir/test

| Attribute | n_seq | Paper | Ours (live) | Delta | Verdict |
|---|---:|---:|---:|---:|---|
| OV | 18 | 70.16 | 69.46 | -0.70 | reproduced |
| OC | 2 | 68.07 | 64.78 | -3.29 | not reproduced |
| FM | 31 | 60.95 | 69.77 | +8.82 | not reproduced |
| SV | 18 | 55.55 | 63.87 | +8.32 | not reproduced |
| LI | 18 | 60.13 | 74.50 | +14.37 | not reproduced |
| TC-EASY | 14 | 77.96 | 77.34 | -0.62 | reproduced |
| TC-MID | 31 | 65.85 | 71.99 | +6.14 | not reproduced |
| TC-HARD | 18 | 45.59 | 50.80 | +5.21 | not reproduced |
| TC | 63 | 62.75 | 67.13 | +4.38 | not reproduced |
| LR | 15 | 53.10 | 69.11 | +16.01 | not reproduced |
| ALL | 91 | 66.04 | 70.29 | +4.25 | not reproduced |

