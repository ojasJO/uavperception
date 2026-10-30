# CST Anti-UAV — attribute analysis (CST TEST)

Sources:
- AeroTrack-X1, the single registered TEST pass of `v3_n__causal_g4_cst_s0_e20`. Its predictions are scored by the unmodified
  official toolkit: `X1_test_official/official_scores.json`. Its frame-level breakdown is in `X1_failure_analysis.json`.
- The released CST-trained baselines, scored the same way: `official_reproduction_test/official_scores.json`.

Everything below is **descriptive**. The registered claims are X1-a, X1-a2 and X1-b (`FINAL_REPORT.md`). X2 adds seed and
single-frame evidence (`ABLATION_RESULTS.md`).

## 1. Frame-level attributes (Table 5 of the paper)

"Corrected" means frame-level SA (Eq. 1) averaged over every frame that carries the flag. The official script
`Evaluation_for_ALL_Frame_SA.py` gives OC and OV only from the last event's window of each sequence. That bug is documented in
`PAPER_PROTOCOL.md`, and the official script's values are shown separately below. The size bins use the ground-truth box
diagonal in pixels, which is identical to the release's `SIZE` codes.

| attribute | frames | X1 STE | X1 argmax | GlobalTrack | SiamDT | PrDiMP |
|---|---:|---:|---:|---:|---:|---:|
| OC occlusion (corrected) | 6,469 | **74.51** | 70.52 | 0.02 | 0.02 | 0.01 |
| OV out of view (corrected) | 6,310 | **67.28** | 64.81 | 0.04 | 0.02 | 0.05 |
| SV scale variation | 33,445 | 52.19 | **52.39** | 37.62 | 39.24 | 27.86 |
| TC thermal crossover | 42,202 | **51.47** | 51.41 | 26.63 | 27.23 | 16.99 |
| DBC dynamic background clutter | 8,624 | **65.40** | 61.85 | 17.26 | 18.00 | 11.50 |
| CDB complex dynamic background | 14,381 | **52.22** | 52.16 | 26.96 | 28.40 | 21.81 |
| FM fast motion | 93 | 31.76 | **40.29** | 34.50 | 36.53 | 3.49 |
| tiny (diagonal ≤ 10 px) | 10,354 | 39.88 | **40.43** | 15.83 | 14.98 | 28.21 |
| small (10–30 px] | 45,812 | 60.21 | **60.31** | 46.13 | 46.51 | 33.45 |
| normal (30–50 px] | 2,600 | 67.24 | **68.73** | 59.52 | 66.10 | 29.85 |
| large (> 50 px) | 484 | 23.38 | 22.34 | 34.52 | **38.35** | 12.94 |

Official script, as released (OC and OV come from the last event window only):

| | X1 STE | X1 argmax | GlobalTrack | SiamDT | PrDiMP |
|---|---:|---:|---:|---:|---:|
| OC | 43.99 | 43.99 | 44.53 | 39.31 | 51.25 |
| OV | 80.00 | 80.00 | 79.74 | 79.55 | 77.26 |

### Reading

- **OC and OV are almost entirely target-absent frames.** On TEST, 6,434 of the 6,469 OC frames (99.5 %) and 6,293 of the
  6,310 OV frames (99.7 %) are absent. The released trackers never report absence, so they score about 0
  there, while X1's absence reporting earns 67–75. The official script's OC and OV values hide this: every tracker gets about
  44 / 80, and X1's STE and argmax get identical values. These two rows from the official script should not be used to rank
  trackers.
- **Thermal crossover, scale variation and both clutter attributes favour X1 by 14–48 SA.** These frames are mostly
  visible-target frames, so this is a localisation gain and not an absence effect. The largest gap is dynamic background
  clutter: 65.40 vs 17.26.
- **Tiny targets: 39.88 vs GlobalTrack's 15.83.** The paper's best tiny-target tracker, PrDiMP, scores 28.21. Under X1, 46.1 %
  of tiny-target frames reach IoU ≥ 0.5, against 2.1 % for GlobalTrack (`X1_failure_analysis.json`). X1's 95 % CI on the
  tiny-target SA (sequence bootstrap in the pass) is [30.20, 49.02].
- **Large targets are X1's one weakness: 23.38 vs 34.52 / 38.35.** These 484 frames come from 8 sequences, and three of them
  hold 436 of the frames:
  - building_74: 159 frames, X1 28.9 vs GlobalTrack 61.0;
  - cn_mountains_32: 188 frames, X1 7.2 vs GlobalTrack 0.0;
  - urban-areas_38: 89 frames, X1 20.5 vs GlobalTrack 45.1.

  X1 outputs no box on 36.6 % of large-target frames. CST train contains only **6** large-target frames (0.01 % of
  its visible frames; TEST has 484, 0.82 %), and the detector was trained only on CST train. The two trackers that beat X1 here are template matchers, which do
  not depend on a learned size prior.
- **Fast motion covers only 93 frames**, too few for any conclusion. argmax beats STE there (40.29 vs 31.76), consistent with
  the motion gate occasionally rejecting a correct but distant detection.

## 2. Sequence-level attributes (official `Evaluation_for_ALL_Sequence_SA.py`)

The mean sequence SA over the sequences that carry each attribute:

| attribute | X1 STE | X1 argmax | GlobalTrack | SiamDT | PrDiMP |
|---|---:|---:|---:|---:|---:|
| OC | 58.34 | 58.09 | 31.38 | 31.79 | 23.12 |
| OV | 59.01 | 58.85 | 31.61 | 31.79 | 20.94 |
| SV | 60.08 | 59.70 | 35.92 | 35.84 | 29.76 |
| TC | 59.94 | 59.65 | 35.53 | 35.46 | 29.25 |
| DBC | 57.56 | 56.66 | 31.41 | 32.12 | 20.41 |
| CDB | 54.24 | 53.87 | 33.40 | 34.36 | 24.01 |
| FM | 56.68 | 56.38 | 29.64 | 31.28 | 14.86 |
| TS | 58.80 | 58.58 | 33.79 | 33.72 | 27.62 |
| SS | 60.08 | 59.70 | 35.92 | 35.84 | 29.76 |
| MS | 58.45 | 57.30 | 34.50 | 36.65 | 20.63 |
| NS | 64.15 | 63.07 | 40.07 | 42.35 | 26.95 |

SV and SS are carried by all 60 sequences, so those rows equal the overall mSA.

## 3. Scene categories (sequence-name prefix; paired vs GlobalTrack, B = 10000)

| scene | sequences | X1 STE mSA | GlobalTrack mSA | Δ [95 % CI] | wins / losses |
|---|---:|---:|---:|---|---|
| building | 9 | 48.34 | 23.52 | +24.82 [+17.36, +32.40] | 9 / 0 |
| cn_mountains | 9 | 68.44 | 33.42 | +35.02 [+28.39, +41.73] | 9 / 0 |
| cn_sky | 6 | 74.25 | 42.11 | +32.14 [+26.57, +37.91] | 6 / 0 |
| jungle | 13 | 57.26 | 45.70 | +11.55 [+7.61, +15.14] | 12 / 1 |
| urban-areas | 17 | 51.56 | 24.38 | +27.18 [+20.77, +34.10] | 17 / 0 |
| water | 6 | 81.26 | 63.62 | +17.64 [+11.91, +23.63] | 6 / 0 |

Jungle has the smallest margin and holds X1's only loss to GlobalTrack (jungle_18; see `FAILURE_ANALYSIS.md`). The CIs come
from only 6–17 sequences per scene and are descriptive.

## 4. X2: the attribute profile across seeds and the single-frame control

Frame-level attribute SA for all six X2 runs, scored by the same official toolkit, is in `X2_test_official.json` (`runs.<run>.frame_attr_SA`).
The seed-mean table is in `ABLATION_RESULTS.md` §3. What changes relative to §1:

- **X1's profile is representative of its configuration.** The causal_g4 seeds vary little on most attributes: SV 52.19–53.68,
  TC 50.22–51.47, tiny 39.88–42.45, small 60.21–61.39. OC varies most (60.60–74.51), because it is almost entirely absent frames,
  where each seed's false-alarm rate differs.
- **The temporal window helps on every attribute.** Seed means, causal_g4 vs single frame:

  | attribute | causal_g4 | single frame |
  |---|---:|---:|
  | SV | 52.81 | 40.29 |
  | TC | 50.89 | 39.88 |
  | CDB | 50.44 | 39.01 |
  | DBC | 63.85 | 56.02 |
  | tiny | 41.07 | 31.22 |
  | normal | 68.39 | 51.15 |
  | large | 25.48 | 14.72 |

  These attribute values are descriptive; the registered window test is T3.
- **Large targets stay the configuration's weakest bin on every seed** (23.4–27.1, vs GlobalTrack 34.5; `FAILURE_ANALYSIS.md` §7).
- Even the single-frame control beats GlobalTrack on tiny targets (31.2 vs 15.8). Its tiny-target SA is above the paper's best
  tiny-target tracker too (PrDiMP 28.2).
