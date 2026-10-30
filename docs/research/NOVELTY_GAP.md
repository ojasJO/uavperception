# NOVELTY_GAP — where AeroTrack could make a defensible contribution

*First version 2026-10-04, written after the context audit and a targeted literature search (sources at the end). Numbers of
other works are as reported by their authors (not reproduced here) unless marked otherwise. This document separates what is
already known from what our evidence could add; a "contribution" below is a candidate until an experiment in
`results/artifacts/experiment_registry.json` supports it.*

## 1. Closest competing systems

| system (year) | task / architecture | data, modality | temporal mechanism | reported accuracy | speed / size / hardware | code, license | strength | weakness for our problem |
|---|---|---|---|---|---|---|---|---|
| **DFSC on GlobalTrack** — Anti-UAV benchmark paper (Jiang et al., arXiv:2101.08466, IEEE TMM) | SOT; query-guided Faster R-CNN R50-FPN, full-image search | Anti-UAV IR + RGB | none (re-detection each frame from the init query) | IR test mSA 66.04 (paper); **our faithful re-implementation 70.29 / 70.58 (normal)** | our measurement: ~10 fps at 1000×800 on RTX 5050; 58.9 M params (our torchvision re-implementation) | DFSC code never released; GlobalTrack Apache-2.0 (mmdet 1.x) | global re-detection survives out-of-view | heavy; no absence prediction (scores 0 on every `exist=0` frame — measured in Track A) |
| **SiamDT** — Anti-UAV410 (Huang et al., 2023/24) | SOT; dual-semantic Siamese tracker with background-clutter modelling | Anti-UAV410 IR (410 videos, 438 k boxes) | template matching over time | best on Anti-UAV410 (SA 67.69 per the CST paper) | not checked | public (BIT) | designed for drones in clutter | CST SA 35.84 (2nd); tiny-target SA 14.98 |
| **CST Anti-UAV benchmark** (Xie et al., ICCVW 2025, arXiv:2507.23473) | benchmark of 20 SOT trackers, **retrained on CST** | CST IR 640×512, 220 seq (120/40/60) | per tracker | best mSA **35.92** (GlobalTrack), SiamDT 35.84, MixFormer-V2 29.39; tiny-target SA ≤ 28.2 | — | dataset academic/non-commercial | the hardest public thermal tiny-UAV benchmark | SOT protocol only; no detection baselines |
| **YOLOMG** (2025, arXiv:2503.07115) | detection; YOLOv5s-derived, 4 heads (extra small-object head), RGB + motion-difference map, adaptive-weight fusion + CBAM | ARD100 air-to-air RGB 1080p (~0.01 % objects), NPS-Drones | 3-frame difference at interval **k (k = 2 best)**, **homography ego-motion compensation** (LK + RANSAC) | ARD100 AP50 0.78 (640) / 0.85 (1280) vs YOLOv9 0.64; NPS 0.95 | 133 FPS @640 / 35 FPS @1280 on RTX 2080 Ti | github Irisky123/YOLOMG, **CC BY-NC-SA 4.0** | explicit motion cue for sub-12-px drones | misses hovering / slow drones (authors); RGB only; motion map adds cost |
| **GL-YOMO** (2024, arXiv:2411.02582) | detection + tracking; YOLOv5 + Ghost + CPAM attention; **global → local (300×300 ROI) search**, template matching, Kalman verification | Drone-vs-Bird, fixed-wing (~0.01 % area), RGB | frame interval F(t−2)…F(t) optical flow | Drone-vs-Bird AP 0.763 | **21.6 FPS on Jetson Xavier NX (TensorRT, 640)**, 14.3 M params, 10.6 GFLOPs | paper CC BY 4.0; code not given | track-conditioned local search — resolution where the target is | moving cameras are a stated limitation |
| **TransVisDrone** (ICRA 2023, arXiv:2210.08423) | detection; CSPDarkNet-53 + VideoSwin | NPS, FL-Drones, AOT (RGB) | spatio-temporal transformer over clips | AP50 NPS 0.95, FL 0.75, AOT 0.80 | heavy (video transformer) | public | strong temporal modelling | ARD100 AP 0.15 (per YOLOMG); edge-unfriendly |
| **Multi-frame IR small-target detectors** — HyperTea, Tridos, DTUM, MOCID (2024–25) | detection / segmentation of point-like IR targets | DAUB, IRDST (IR) | hypergraph / memory / motion-aware temporal modules | HyperTea mAP50 95.6 (DAUB), 76.4 (IRDST) | mostly not edge-oriented | mixed | temporal modules for few-pixel IR targets | targets/datasets differ (no Anti-UAV / CST evaluation); not trackers |
| **Anti-UAV challenge entries** (CVPR 2023–2025 workshops), e.g. YOLOv12 + BoT-SORT-ReID (bronze, 2025) | tracking-by-detection, MOT style | challenge IR sets | tracker association | challenge leaderboards (not compared here) | — | public repos | strong engineering baselines | MOT machinery for a mostly single-target problem (our audit: ByteTrack < argmax baseline) |

## 2. What is already known (so it cannot be our contribution)

- Using **non-consecutive frames** to expose motion of small drones (YOLOMG k = 2; GL-YOMO F(t−2)); adaptive temporal strides driven by motion
  magnitude exist in the video literature.
- **Extra high-resolution (P2) heads** and small-object attention for tiny drones (YOLOMG has four heads).
- **SPD-Conv** as a "lossless" downsampler for small objects (Sunkara & Luo 2022) — widely adopted in tiny-object YOLO variants.
- **Global → local search** around a tracked target (GL-YOMO) and Kalman verification.
- **Tracking-by-detection** with YOLO + SORT/ByteTrack/BoT-SORT for anti-UAV challenges.

## 3. Gaps our evidence can address (candidate contributions, each with its falsifier)

| # | candidate contribution | why it is not already covered | evidence we have / need | falsified by |
|---|---|---|---|---|
| G1 | **Controlled evidence that SPD-Conv's benefit for tiny UAVs is an initialisation artefact.** With function-preserving init, SPD-Conv gives +0.014 CST AP50 (CI incl. 0) for +55 % params / +76 % FLOPs; the uncontrolled comparison showed −0.10 (penalty) — the sign of the "effect" depends on init. | SPD-Conv papers and follow-ups compare against stride convs whose pretrained weights are kept while SPD layers start random (or train from scratch); we found no work that equalises the starting function. | have (3 folds, 1 seed each); need: 3 seeds, a second dataset family | SPD with fair init beats stride by > 0.02 with CI excluding 0 on a pre-registered rule |
| G2 | **Motion-matched temporal sampling rule** for static-camera thermal tiny targets: choose the smallest gap g with median target displacement ≥ 2 px at network input; g = 1 gives nothing (+0.009), g = 4 gives +0.045 [+0.021, +0.072] on held-out CST; Anti-UAV (gimbal) unchanged. | prior art fixes k by tuning (YOLOMG) or uses a flow-driven stride for clips; none derives g from measured target-motion statistics or isolates it with a fair start | have for g ∈ {0, 1, 4}; **need the causal variant (t−8, t−4, t)** so the gain survives without 160 ms look-ahead, and seeds | causal variant loses the gain, or the gain vanishes with complete CST training data |
| G3 | **A real-time detection-based single-target stack evaluated with the SOT benchmarks' own metric (SA) against the strongest SOT tracker, on the same splits.** We own a faithful GlobalTrack (Track A: IR test mSA 70.58, ~10 fps, no absence prediction); CST's best tracker is GlobalTrack (35.92, tiny 15.83). A 2.6 M-param detector + state estimator with explicit absence prediction is a different operating point (> 60 fps). | anti-UAV SOT benchmarks report no detection-based baselines under SA; challenge MOT entries are not scored with SA against SOT trackers | **need**: tracker/state-estimator designed on VAL, one pre-registered test pass computing SA on Anti-UAV IR test (and CST test); training-data difference must be stated (AeroTrack also trains on CST, Det-Fly, RGB). **Protocol audit 2026-10-05**: CST's metric (Eq. 1 SA), one-pass first-frame initialisation and 60-sequence test split match ours, but its trackers are trained on CST train only -> registered arm X1 (`experiments/aerotrack_v3/preregistration_X1_cst_protocol_matched.json`): CST-only training, tracker frozen on CST val, one test pass, 'exceeds' only if the CI lower bound > 35.92 | stack's SA ≤ GlobalTrack's on Anti-UAV IR test; X1 'below' / 'not distinguishable' on CST |
| G4 | **Complete-data effect on tiny-target generalisation**: the "unseen scene" failure (CST urban-areas 0.242 AP50) coincides with a truncated training archive; retraining on the complete release isolates scene coverage from model capacity. | dataset-defect finding specific to our corpus; contribution is the controlled before/after | **need**: corpus v2 (download in progress), same recipe, same folds logic | no change on urban-areas / water after adding their training sequences |
| G5 | **Independent replication of DFSC** (Track A): validation reproduced within 1 pt, test 4–5 pt above the paper, DFSC gain not observed at 2 seeds; absence handling explains val attribute gaps. | no public DFSC code; no published replication | have; H4 running, H3 open | — (a replication is a contribution either way) |
| G6 | **Track-conditioned local re-detection** for 1080p RGB (Anti-UAV RGB is downscaled 3× to 640): a crop at native resolution around the predicted state recovers resolution for tiny RGB targets at little cost. | GL-YOMO does global/local switching for RGB fixed-wing data, without temporal-stack detectors or SA evaluation | **need**: measured RGB small-target failure first (tier-1 evidence), then an ablation | crop re-detection does not improve RGB small/micro AP or SA on VAL |

## 4. Positioning statement (to be confirmed by experiments)

The defensible story is not "a better YOLO". It is: *for tiny thermal UAVs, what actually moves accuracy under a fixed edge budget is
(i) honest initialisation and evaluation, (ii) giving the network motion it can resolve, (iii) data coverage of the deployment scenes,
and (iv) a cheap state estimator with absence prediction around a strong per-frame detector — and each of these claims is
measured against the obvious alternative (SPD-Conv, t±1 stacks, MOT trackers, learned forecasters, a heavy SOT tracker).*

## Sources

- [Anti-UAV benchmark, arXiv:2101.08466](https://arxiv.org/abs/2101.08466) (repository PDF `anti_uav_research_paper.pdf`)
- [CST Anti-UAV, arXiv:2507.23473](https://arxiv.org/html/2507.23473v2) · [ICCVW 2025](https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Xie_CST_Anti-UAV_A_Thermal_Infrared_Benchmark_for_Tiny_UAV_Tracking_ICCVW_2025_paper.html) · [dataset](https://github.com/PCwenyue/CST-Anti-UAV)
- [Anti-UAV410 / SiamDT](https://pure.bit.edu.cn/en/publications/anti-uav410-a-thermal-infrared-benchmark-and-customized-scheme-fo/)
- [YOLOMG, arXiv:2503.07115](https://arxiv.org/html/2503.07115v1)
- [GL-YOMO, arXiv:2411.02582](https://arxiv.org/html/2411.02582v1)
- [TransVisDrone, arXiv:2210.08423](https://arxiv.org/pdf/2210.08423)
- [HyperTea, arXiv:2508.10678](https://arxiv.org/pdf/2508.10678)
- [CVPR 2025 Anti-UAV workshop](https://openaccess.thecvf.com/CVPR2025_workshops/Anti-UAV) · [survey, arXiv:2504.11967](https://arxiv.org/abs/2504.11967)
- [YOLOv12 BoT-SORT-ReID (challenge entry)](https://oosmetrics.com/repo/wish44165/YOLOv12-BoT-SORT-ReID)
