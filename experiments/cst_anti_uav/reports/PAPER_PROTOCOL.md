# CST Anti-UAV — paper protocol, official code, and reproduction of the published result

Sources (both read in full for the parts below):
- paper: Xie et al., *CST Anti-UAV: A Thermal Infrared Benchmark for Tiny UAV Tracking in Complex Scenes*, ICCV 2025 Workshops,
  arXiv:2507.23473 **v2** (19 Nov 2025; PDF sha256 `af0025ab…`). v1: 31 Jul 2025.
- official toolkit: `github.com/PCwenyue/CST-Anti-UAV`, commit `bd92b984` (2026-09-06), local copy `data/cst_anti_uav_official/toolkit`
  (MIT licence; dataset academic / non-commercial, `DATASET_TERMS.md`).
- official results: README step 3, Google Drive file `TrainedCST.zip` (17,436,199 bytes, sha256 `1c886465697b…`), extracted to
  `data/cst_anti_uav_official/tracking_results/TrainedCST` — per-frame boxes of the CST-trained trackers on the 60 TEST sequences.

## 1. Benchmark facts (verified)

| item | paper | verified on the data / code |
|---|---|---|
| sequences | 220 | 220 |
| annotations | "over 240k" | 245,471 annotated frames, 208,221 of them with a visible box |
| split | 120 train / 40 val / 60 test (§3.3) | same; train and val are "non-overlapping clips of shared sequences", test independent |
| modality | thermal infrared, single target | 640×512 IR, one UAV per sequence |
| trackers | abstract: "20 trackers"; Table 3 lists 22 rows | released archive holds 21 CST-trained trackers (ARTrackV2 and STFTrack — listed in `utils/trackers.py` — are absent; `cmd-tir` is present but not listed) |
| training | "retrained 20 of the latest methods on both CST Anti-UAV and Anti-UAV410" with "the implementations, pre-trained models and default configurations provided by the official code"; 8× A100-40GB | — |
| best CST-trained tracker | GlobalTrack (AAAI 2020): mSA **35.92**, P 58.72, S 35.38 (Table 3); runner-up SiamDT 35.84 | reproduced exactly (§3) |
| 67.69 | SiamDT trained and tested on **Anti-UAV410** (Table 3, left block) | a different benchmark; never compared with CST numbers here |
| Table 4 | GlobalTrack trained on Anti-UAV410, tested on CST: 30.96 → 35.92 when retrained on CST (+16.02 %) | — |

## 2. Evaluation protocol (from the code, which is authoritative where the paper is terse)

**One-pass evaluation (OPE).** Each tracker is initialised with the ground-truth box of frame 0 and outputs, for every frame,
either a 4-number box `[x, y, w, h]` or "absent". Frame 0 is visible in all 220 sequences, so the start frame is always 0.
Results are read per sequence from `<seq>.txt` (comma / space separated rows) or JSON `{"res": [...]}` (an empty list = absent).

**State accuracy** (`experiments/cstanti_uav.py::eval`; paper Eq. 1, inherited from Anti-UAV, Jiang et al. 2021):

    SA(seq) = mean over frames t of  { IoU(pred_t, gt_t)          if exist_t = 1 and gt_t is a valid box
                                       1 if pred_t is empty else 0  if exist_t = 0 }
    mSA     = mean of SA(seq) over the 60 test sequences (unweighted)

- IoU uses the `x0 + w` convention (no +1) and is 0 when the overlap is non-positive.
- Visible frames whose GT is not a 4-vector or sums to 0 are **skipped** (none exist in CST — verified, `INITIAL_AUDIT.md`).
- A visible frame with no predicted box scores 0. An absent frame scores 1 **only** if the prediction is empty; any box scores 0.
- There is **no** extra penalty term (unlike the 2023 Anti-UAV challenge SA, which subtracts 0.2 × missed-absence rate).
- Text-file results always have 4 numbers per row, so trackers that never report absence score 0 on every absent frame.

**Success / precision** (`report`, `_calc_curves`, `utils/metrics.py`): computed on all frames after replacing frame 0 by the GT and
mapping absent predictions / GT to `[0,0,0,0]`; success AUC = mean over 21 IoU thresholds in [0, 1] of the fraction with IoU **>**
threshold; precision = fraction of frames with centre error ≤ 20 px (centre `x + (w−1)/2`); both averaged over sequences. Table 3's
"P(AUC)" column is this precision@20px value.

**Attributes.** Sequence-level: mean SA over sequences flagged in `annos/test/att/<seq>.txt` (OC OV SV TC DBC CDB FM TS SS MS NS).
Frame-level (Table 5, `experiments/cstanti_uav_frame_SA.py`): per-frame Eq. 1 scores of all 60 sequences concatenated and averaged
over the frames carrying each `IR_label.json` flag. Size bins TS / SS / NS / LS = GT diagonal ≤ 10 / (10, 30] / (30, 50] / > 50 px
(the release's `SIZE` codes 1–4; verified).

**Runtime.** The toolkit averages `1 / time` from optional per-sequence `times/<seq>_time.txt`; the released archive contains no
timing files and the paper reports no FPS for CST.

### Discrepancies and quirks found in the official code (documented, not "fixed" in any reported official number)
1. `ExperimentCSTAntiUAVF` frame-level **OC / OV**: inside the per-event loop the average is *overwritten* each iteration, so only
   the window around the **last** occlusion / out-of-view event (−1 … +2 or +90 frames, unflagged frames only) is scored. For
   CST-trained GlobalTrack this reproduces the paper's OC 44.53 exactly, but gives OV 79.74 against the paper's 65.77 (unexplained).
   Scoring all flagged frames gives OC 0.02 / OV 0.04, because OC / OV frames are mostly target-absent and GlobalTrack never
   reports absence. We therefore report OC / OV both ways and never use them for a claim.
2. The frame-level script reads labels with `os.walk` (filesystem order) but scores sequences in sorted order; on our NTFS copy the
   two orders coincide (verified via the matching SV / TC / DBC / CDB / size values below).
3. `not_exist()` in the frame-level script treats a 1-element prediction as "absent" (`len(pred) == 1`); the sequence-level script does
   not. Irrelevant for 4-number rows and for our JSON output (empty list = absent).
4. The size bins are not computed by any toolkit script (the TS / SS / MS / NS keys exist in one TEST sequence only); we compute them
   from the GT diagonal. GlobalTrack: tiny 15.83 / small 46.13 / large 34.52 match Table 5; normal 59.52 vs 59.15 (2,600 frames;
   unexplained 0.37-point difference).
5. `seq_wise[...]['sa']` in `performance.json` stores the *running list* of sequence SAs (a reference to a growing list), not the
   sequence's own value; our driver takes the per-sequence values from the final list in the toolkit's (sorted) sequence order.

## 3. Reproduction of the published numbers (exact)

`scripts/official_eval.py --published` imports the toolkit unmodified (two never-called imports, `wget` and `shapely`, are stubbed
because they are not installed) and scores the released predictions. Result (`official_reproduction_test/REPRODUCTION.md`):

- **All 21 released trackers reproduce Table 3 exactly** (mSA, precision@20px and success AUC; largest difference 0.005 = rounding).
  GlobalTrack 35.92 / 58.72 / 35.38; SiamDT 35.84 / 57.96 / 35.28; PrDiMP 29.76; MixFormer-V2 29.39.
- Tiny-target frame SA: GlobalTrack 15.83 (= Table 5); best released tracker on tiny targets PrDiMP 28.21 (paper 28.20).
- **Track A's SA code** (`anti_uav_repro.eval.metrics.sequence_metrics`, which scores every AeroTrack tracker) gives the same
  per-sequence SA as the official code on all 21 × 60 sequences (max |Δ| = 0). Our earlier CST numbers are therefore on the
  official scale; from now on our TEST predictions are additionally exported in the official format and scored by the toolkit itself.

This is an **exact reproduction of the published evaluation** (official code, official predictions, official annotations). It is
not a re-training of GlobalTrack on CST: the authors' CST-trained checkpoints are not released, so the per-frame predictions are the
verifiable artifact. The per-sequence SA of every released tracker (`official_scores.json`) enables **paired** comparisons on the
same 60 sequences.

## 4. Tests
`tests/test_cst_protocol.py` checks, on synthetic sequences, that Track A's SA equals the toolkit's `ExperimentCSTAntiUAV.eval`
(visible / absent / empty prediction / no overlap / degenerate GT), that the official JSON export round-trips, and that the
size-bin rule reproduces the `SIZE` codes.
