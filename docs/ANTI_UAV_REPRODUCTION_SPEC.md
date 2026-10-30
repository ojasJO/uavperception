# Anti-UAV / DFSC — Reproduction Specification

Structured extraction of the paper shipped in this repository as `anti_uav_research_paper.pdf`,
reconciled with the official code it builds on. Every experimental parameter carries one tag:

| Tag | Meaning |
|---|---|
| `[PAPER-EXPLICIT]` | stated in the paper text/tables |
| `[REPO-DERIVED]` | taken from an official code base the paper defers to (GlobalTrack, or the Anti-UAV challenge scripts); source file named |
| `[INFERRED]` | our least-assumptive reading where paper + code leave a gap; reasoning given |
| `[UNSPECIFIED]` | not available from any official source; our choice is stated and is *not* a paper fact |
| `[DISCREPANCY]` | paper text disagrees with the released data/code; both versions recorded |

Nothing tagged `[INFERRED]`/`[UNSPECIFIED]` may be described as "the paper's setting" in any report.

---

## 0. Identity of the paper

| Field | Value | Tag |
|---|---|---|
| Title | *Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking* | `[PAPER-EXPLICIT]` |
| Authors | Nan Jiang†, Kuiran Wang†, Xiaoke Peng†, Xuehui Yu, Qiang Wang, Junliang Xing, Guorong Li, Jian Zhao‡, Guodong Guo, Zhenjun Han‡ (UCAS / CASIA / INEE / Baidu) | `[PAPER-EXPLICIT]` |
| Version in repo | arXiv:2101.08466 **v3**, 8 Feb 2021 (PDF metadata creation date 2021-02-09) | `[PAPER-EXPLICIT]` |
| Venue | arXiv preprint; journal version IEEE Transactions on Multimedia (ieeexplore document 9615243, linked from the challenge README as "Anti-UAV v1 paper") | `[REPO-DERIVED]` challenge README |
| Year | **2021** (preprint). The task brief called it "the Anti-UAV research published in 2025". The file in the repository is the 2021 paper; no 2025 paper is present. Flagged to the user; this spec follows the file that was provided. | `[DISCREPANCY]` |
| Official code | `https://github.com/ucas-vg/Anti-UAV` — contains only `README.md` + `LICENSE`, "Code coming". **DFSC training code was never released.** | `[REPO-DERIVED]` verified 2026-09-19 |
| Base tracker code | GlobalTrack (Huang, Zhao, Huang, AAAI 2020), `https://github.com/huanglianghua/GlobalTrack` — mmdetection 1.x, PyTorch 1.1, CUDA 9.0 (`environment.yml`) | `[REPO-DERIVED]` |
| Challenge reference metric | `https://github.com/ZhaoJ9014/Anti-UAV/blob/master/Eval/test.py` (1st Anti-UAV challenge, CVPR 2020 workshop) | `[REPO-DERIVED]` |

## 1. Task definition and problem statement

- Task: **single-object tracking (SOT)** of one UAV per sequence. The tracker is initialised with the
  ground-truth box of the start frame and must output, for every following frame, a bounding box and
  (implicitly) whether the target is present. `[PAPER-EXPLICIT]` §III-E, Eq. (1)
- Motivation: UAV surveillance; RGB trackers fail in low light, IR is low-resolution, hence a paired
  RGB + thermal-IR benchmark. `[PAPER-EXPLICIT]` §I
- The paper is **not** a detection paper and defines no detection metric (no mAP). The existing
  AeroTrack-Net code in this repository is a YOLO *detector* trained on Anti-UAV + CST + Det-Fly and
  is therefore *not* an implementation of this paper. `[REPO-DERIVED]` this repository
- Three evaluation protocols `[PAPER-EXPLICIT]` §III-E:
  - **Protocol I** — no UAV training data; evaluate off-the-shelf trackers (40+ trackers, Tables III/IV).
  - **Protocol II** — train/finetune on Anti-UAV train split (visible or infrared); the paper's own
    method (DFSC) is evaluated here (Tables V–VII). **This is the protocol we reproduce.**
  - **Protocol III** — multi-modal (unaligned RGB+IR); no results are reported in the paper.

## 2. Dataset

| Item | Value | Tag |
|---|---|---|
| Name | Anti-UAV (a.k.a. Anti-UAV300 / Anti-UAV-RGBT) | `[PAPER-EXPLICIT]` |
| Content | 318 RGB–thermal video **pairs** (not spatially aligned), one UAV per sequence; DJI/Parrot UAVs; day and night; backgrounds: buildings, cloud, trees | `[PAPER-EXPLICIT]` §III-A |
| Frame rate / container | 25 FPS, MP4 | `[PAPER-EXPLICIT]` |
| Resolution | IR 640×512, RGB 1920×1080 (paper only says "high-definition") | `[REPO-DERIVED]` measured on `data/anti_uav` |
| Sequence length | videos cut into sequences "every 1000 frames"; observed lengths 28–1000 frames | `[PAPER-EXPLICIT]` + `[REPO-DERIVED]` |
| Split sizes | train 160 / test 91 / val 67 (67 = "the rest") sequence pairs | `[PAPER-EXPLICIT]` §III-C, Table I |
| Split methodology | train and val come from **non-overlapping clips of the same videos**; test is from **independent videos** and is "much more complicated" | `[PAPER-EXPLICIT]` §III-C |
| Box counts (paper) | total 585.9k; train 294.4k; test 168.4k (val by subtraction 123.1k) | `[PAPER-EXPLICIT]` Table I |
| Box counts (repo data, `exist==1` frames) | IR 148,368 / 60,620 / 84,221; RGB 142,192 / 58,302 / 79,724 (train/val/test) = **573,427** total; train 290,560; test 163,945 | `[DISCREPANCY]` ~2 % fewer than the paper; cause unknown (annotation revision or counting of non-exist frames). We do not alter the data. |
| Frame counts (repo data) | 149,528 / 61,999 / 85,374 frames per modality (train/val/test); 593,802 total | `[REPO-DERIVED]` |
| Classes | one class: UAV | `[PAPER-EXPLICIT]` |
| Annotation format (paper) | per-frame `exist` ∈ {0,1}; `get rect` given as "[x1, y1, x2, y2]" | `[PAPER-EXPLICIT]` §III-B |
| Annotation format (data) | JSON keys `exist`, `gt_rect`; `gt_rect` is **[x, y, w, h]** (verified visually and by the challenge `iou()` which unpacks `x, y, w, h`) | `[DISCREPANCY]` — data wins |
| Attribute annotation | per IR sequence: OV, OC, FM (>60 px inter-frame motion), SV (ratio outside [0.66, 1.5]), LI, TC (thermal crossover; split into TC-easy/med/hard by SiamRPN++LT difficulty), LR (<400 px² box) | `[PAPER-EXPLICIT]` Table II; files `data/anti_uav/label_new/{split}.json` `[REPO-DERIVED]` |
| Annotation process | coarse every 25 frames → fine per frame for 30 pairs → inspection/correction | `[PAPER-EXPLICIT]` §III-B |
| Tracking start frame | "the first frame of the UAV existence (`exist`==1)", with 4 official corrections: `20190925_130434_1_2` IR → 16; `20190926_134054_1_1` visible → 150; `20190925_194211_1_3` visible → 150; `20190925_194211_1_4` visible → 1 | `[REPO-DERIVED]` ucas-vg/Anti-UAV README |
| Frames on disk | decoded JPEG (quality 95, sequential `cv2.VideoCapture` read) at `b_pipeline/data_unified/anti_uav/images/antiuav_<split>_<seq>__{IR,RGB}/NNNNNN.jpg` (1-indexed); frame count == label count for all 636 clips | `[REPO-DERIVED]` this repo; the paper reads MP4 directly — JPEG re-encoding is a preprocessing difference we document and accept |

Position/scale statistics (Fig. 3): boxes concentrate near the image centre, horizontal motion
dominates, mean target size √(wh) < 40 px in all splits. `[PAPER-EXPLICIT]`

## 3. Method: DFSC on GlobalTrack

The paper's contribution is a **training strategy**, not a new network. Inference is exactly
GlobalTrack ("DFSC only works in training, it will not affect ... inference"). `[PAPER-EXPLICIT]` §I, §IV

### 3.1 Base architecture (GlobalTrack QG-RCNN)

The paper names GlobalTrack as the baseline and gives no layer-level details; everything below is
`[REPO-DERIVED]` from `GlobalTrack/configs/qg_rcnn_r50_fpn.py` and `GlobalTrack/modules/*.py`.

| Component | Setting |
|---|---|
| Backbone | ResNet-50 (torchvision ImageNet init in GlobalTrack), 4 stages out, `frozen_stages=1` (stem + res2 frozen), BN in eval mode |
| Neck | FPN, in [256,512,1024,2048] → 256 ch, 5 outputs (P2–P6) |
| RPN head | 3×3 conv 256 → cls (sigmoid, 3 anchors) + reg (12); anchors: scale 8 × strides [4,8,16,32,64] → sizes 32…512, ratios [0.5,1,2]; target stds [1,1,1,1] |
| **QG-RPN modulator** (`RPN_Modulator`) | RoIAlign(7×7, sample 2) of the **query** GT box on FPN levels strides [4,8,16,32] → z ∈ ℝ^{256×7×7}; per FPN level k: `proj_modulator[k]` = Conv(256→256, k=7, pad 0) applied to z → 256×1×1 kernel, broadcast-multiplied with search features x_k; then `proj_out[k]` = Conv(256→256, k=1). Init normal(std 0.01). This realises Eq. (2) with f_z = 7×7 conv, f_x = identity, ⊗ = channel-wise product, f_out = 1×1 conv. |
| RoI extractor | RoIAlign 7×7, sample 2, on strides [4,8,16,32] |
| **QG-RCNN modulator** (`RCNN_Modulator`) | `proj_z` = Conv3×3(256→256) on query RoI feature; `proj_x` = Conv3×3(256→256) on each proposal RoI feature; Hadamard product; `proj_out` = Conv1×1(256→256). Init normal(std 0.01). Realises Eq. (5). |
| Box head | 2 shared FC (256·7·7 → 1024 → 1024), `num_classes=2` (bg + target), class-specific regression, target stds [0.1,0.1,0.2,0.2] |
| RPN train | MaxIoU assigner pos ≥ 0.7 / neg < 0.3, 256 anchors/img, pos fraction 0.5; proposals: nms_pre 2000, nms 0.7, post 2000 |
| RCNN train | assigner pos ≥ 0.5 / neg < 0.5, 512 RoIs/img, pos fraction 0.25, GT added as proposals |
| Losses | RPN: sigmoid CE + SmoothL1(β=1/9), weights 1/1. RCNN: softmax CE + SmoothL1(β=1), weights 1/1 |
| Test | RPN nms_pre/post 1000, nms 0.7; RCNN score_thr 0.0, nms 0.5, max_per_img 1000; **output = top-1 box by score**; no motion model, no online update, no absence decision |

The paper's own statement of the losses: "the classification and regression losses are respectively
cross-entropy and smooth l1" `[PAPER-EXPLICIT]` §V-B — consistent with the above.

### 3.2 DFSC training strategy `[PAPER-EXPLICIT]` §IV

Batch of n pairs (query zᵢ, search xᵢ) from n *different* sequences.

**CSM (class-level semantic modulation)** — Eq. (2)–(4):

- t̂ᵢⱼ = f_CSM(zᵢ, xⱼ): search features of sequence j modulated by the query RoI feature of sequence i,
  for all i, j ∈ [0, n). i = j → intra-sequence (the normal GlobalTrack term); i ≠ j → cross-sequence.
- L_CSM = Σ_{i=j} L_rpn(t̂ᵢⱼ) + α · Σ_{i≠j} L_rpn(t̂ᵢⱼ), with L_rpn = (1/N_cls) Σ L_cls + β (1/N_reg) Σ L_reg.
- The RPN target for t̂ᵢⱼ is the UAV box of search image j (any UAV counts as positive — "class-level").
  `[INFERRED]` — the paper says CSM aims to "retrieve anchors that may contain UAV as far as possible";
  no other target makes sense for a cross-sequence pair.

**ISM (instance-level semantic modulation)** — Eq. (5)–(7):

- Proposals selected by the CSM stage are modulated **only** by the query of their own sequence
  (Hadamard), then classified/regressed by the RCNN head.
- L_ISM = (1/N_pnum) Σ_k L_rcnn(t̂_k), L_rcnn = L'_cls + β L'_reg.
- Which proposals enter ISM: `[INFERRED]` those produced by the intra-sequence RPN branch (t̂ⱼⱼ),
  sampled exactly as in GlobalTrack (512 RoIs, 25 % positive). The paper does not say whether
  proposals from cross-modulated RPN outputs are also used; we take the minimal reading.
  Ablation "DFSC-all" (Table VI) = ISM also uses cross-sequence modulation — confirms that in DFSC
  proper the ISM modulation is intra-sequence only.

**Total loss** `[INFERRED]`: L = L_CSM + L_ISM (the paper trains "CSM and ISM" jointly and never
writes a total; GlobalTrack sums RPN and RCNN losses with unit weights).

| Hyper-parameter | Value | Tag |
|---|---|---|
| α (cross/same ratio) | **1** for the main "DFSC (Ours)" rows (Table V row == Table VI "DFSC-α1" row, all 12 numbers identical); 0.25 gives the best mSA in the ablation | `[PAPER-EXPLICIT]` |
| β (cls/reg balance) | not given → GlobalTrack `loss_weight=1.0` for both terms | `[UNSPECIFIED]` → `[REPO-DERIVED]` default |
| "normal" baseline | GlobalTrack finetuned on Anti-UAV without DFSC (= α = 0, no cross terms) | `[PAPER-EXPLICIT]` §V-D |
| "large-scale" baseline | the public GlobalTrack checkpoint (COCO + LaSOT + GOT-10k), no Anti-UAV training | `[PAPER-EXPLICIT]` §V-D |

### 3.3 Ablation variants `[PAPER-EXPLICIT]` Table VI (all IR/visible **test**, mSA)

`DFSC-all` (ISM uses class-level/cross modulation too), `DFSC-cls` (cross term = cls loss only),
`DFSC-reg` (cross term = reg loss only), `DFSC-α{0.25, 0.5, 1, 2}`.

## 4. Training configuration (Protocol II)

| Parameter | Infrared tracker | Visible tracker | Tag |
|---|---|---|---|
| Training data | Anti-UAV **infrared train** (160 seqs) | Anti-UAV **visible train** (160 seqs) | `[PAPER-EXPLICIT]` §V-B |
| Initialisation | "initial weights transferred from Faster RCNN" (COCO-pretrained Faster R-CNN R50-FPN, mmdet checkpoint `qg_rcnn_r50_fpn_2x_20181010-443129e1.pth` per GlobalTrack README) | GlobalTrack pretrained model (COCO + LaSOT + GOT-10k) | `[PAPER-EXPLICIT]`; checkpoint identity `[REPO-DERIVED]` |
| Epochs | 18 | 12 | `[PAPER-EXPLICIT]` |
| Initial LR | 0.02 | 0.02 | `[PAPER-EXPLICIT]` |
| LR schedule | ×0.1 at epoch 12 and 15 (0.002, 0.0002) | ×0.1 at epoch 8 and 11 | `[PAPER-EXPLICIT]` |
| Batch size | 2 per GPU | 2 per GPU | `[PAPER-EXPLICIT]` |
| Number of GPUs | not stated (GlobalTrack: 4–8; mmdet's LR 0.02 convention = 16 images total) | same | `[UNSPECIFIED]` |
| Optimizer | not stated → SGD, momentum 0.9, weight decay 1e-4, grad-clip L2 norm 35 | same | `[UNSPECIFIED]` → `[REPO-DERIVED]` GlobalTrack config |
| Warm-up | not stated → linear, 500 iters, start ratio 1/3 | same | `[UNSPECIFIED]` → `[REPO-DERIVED]` |
| Loss functions | CE + smooth-L1 (see §3.1) | same | `[PAPER-EXPLICIT]` |
| Epoch definition | not stated → GlobalTrack `Seq2Pair`: `pairs_per_seq=10` ⇒ **1,600 pairs/epoch** for 160 sequences (`PairWrapper.max_size=30000` is not reached) | same | `[UNSPECIFIED]` → `[REPO-DERIVED]` `neuron/data/datasets/structure.py` |
| Pair sampling | two frames of one sequence, z before x, gap ≤ 300 frames (`max_distance=300`), up to 100 rejection tries | same | `[REPO-DERIVED]` |
| Frame filter for training pairs | box area > 20 px², w,h ≥ 10 px, w,h ≤ 960 px, w/W and h/H ≥ 0.01 and ≤ 0.75, aspect w/h ∈ [0.2, 5) | same | `[REPO-DERIVED]` `Seq2Pair._filter` |
| Input scaling | rescale keeping aspect so long side ≤ 1333 and short side ≤ 800 → IR 1000×800, RGB 1333×750; pad to multiple of 32 | same | `[REPO-DERIVED]` `BasicPairTransforms(scale=(1333,800))` |
| Normalisation | ImageNet mean [123.675,116.28,103.53] / std [58.395,57.12,57.375] on RGB-ordered 0–255 pixels | same | `[REPO-DERIVED]` |
| Augmentation | `extra_partial`: photometric distortion (brightness ±32, contrast ×[0.5,1.5], saturation ×[0.5,1.5], hue ±18°, each p=0.5; channel permutation p=0.5 shared by query and search), horizontal flip p=0.5 (independent per image); **no** expand, **no** crop | same | `[UNSPECIFIED]` in paper → `[REPO-DERIVED]` GlobalTrack default training transform |
| Random seed | not stated | | `[UNSPECIFIED]` — we fix and record ours |
| Hardware | not stated | | `[UNSPECIFIED]` |
| Framework versions | not stated (GlobalTrack: PyTorch 1.1.0, mmdetection 1.x, CUDA 9.0) | | `[UNSPECIFIED]` / `[REPO-DERIVED]` |
| Validation during training | not mentioned; GlobalTrack reports the **final-epoch** checkpoint | | `[INFERRED]` final epoch, no checkpoint selection |

## 5. Inference / evaluation settings

| Item | Value | Tag |
|---|---|---|
| Init | GT box of the start frame (§2); query features computed once | `[REPO-DERIVED]` GlobalTrack `init()` |
| Per frame | full-image search, QG-RPN → 1000 proposals → QG-RCNN → NMS 0.5 → **argmax score box** | `[REPO-DERIVED]` GlobalTrack `update()` |
| Absence ("not exist") prediction | none — GlobalTrack always outputs a box ⇒ p_t = 0 for every frame; Protocol I trackers were "used without any modification" | `[INFERRED]` from `[PAPER-EXPLICIT]` §V-C + code |
| Confidence / NMS thresholds | RCNN score_thr 0.0, NMS IoU 0.5; RPN NMS 0.7 | `[REPO-DERIVED]` |
| Test-time augmentation | none (flip p = 0, bilinear resize) | `[REPO-DERIVED]` `BasicPairTransforms(train=False)` |
| Tracking parameters | none (no motion model, no window, no scale penalty) | `[REPO-DERIVED]` |

## 6. Metrics

**State accuracy** `[PAPER-EXPLICIT]` Eq. (1):

SA = (1/T) Σ_t [ IoU_t · δ(v_t > 0) + p_t · (1 − δ(v_t > 0)) ]

- v_t = GT `exist` flag; IoU_t between predicted and GT box; p_t = tracker's *absence* prediction
  (1 when the predicted box is empty, 0 otherwise — challenge README wording).
- SA is averaged over all frames of a sequence; **mSA** = mean SA over sequences = the headline number.
- Reference implementation: challenge `Eval/test.py::eval()`. Note `[REPO-DERIVED]`: its helper
  `not_exist(pred)` compares a list with an int (`pred == 0`) and therefore can never return True
  for a list prediction — a reference-code quirk; we implement the formula as written in the paper.
- Frame 0 (init frame): the reference appends the init box as the prediction (IoU = 1).
- Frames before the start frame (sequences with `exist[0] == 0` or an official correction; 1/4/3 IR
  sequences in train/val/test): `[UNSPECIFIED]` — the tracker is never run on them. **Primary number:
  SA over frames ≥ start frame** (the frames the tracker actually tracks). Secondary, also reported:
  SA over all frames with pre-start frames scored as "absent predicted" (p_t = 1). The two differ by
  ≤ 0.2 mSA on IR val; both are computed from the same saved predictions.
- Per-attribute mSA (Tables III–VI) = mean over the sequences carrying that attribute
  (attribute lists come from `label_new`). `[PAPER-EXPLICIT]` §III-D/V-C
- No revised penalty term (the −0.2·(Σ p_t δ(v_t>0))^0.3/T* term of later challenges) exists in this paper. `[PAPER-EXPLICIT]`

**Precision and success** "the same as [18]" (OTB) `[PAPER-EXPLICIT]`: precision = fraction of frames
with centre-location error ≤ 20 px; success = AUC of the IoU-threshold curve. Handling of `exist = 0`
frames in these two metrics is `[UNSPECIFIED]`; we compute them over frames with `exist = 1` and store
per-frame values so any convention can be recomputed.

## 7. Reported results (transcribed from the paper; machine-readable copy in `experiments/anti_uav/reports/paper_reported_results.json`)

### Table V — Protocol II, mSA (%)

Columns: OV, OC, FM, SV, LI, TC-easy, TC-med, TC-hard, TC-all, LR, **IR All**, **Visible All**

| Method | Split | OV | OC | FM | SV | LI | TCe | TCm | TCh | TCall | LR | IR All | Vis All |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| large-scale | val | 69.21 | 78.62 | 73.35 | 66.11 | 76.33 | 76.47 | 63.08 | 43.45 | 65.90 | 60.26 | 72.00 | 67.28 |
| normal | val | 78.09 | 81.42 | 78.66 | 76.61 | 79.95 | 81.23 | 76.56 | 74.04 | 78.49 | 73.55 | 79.60 | 73.25 |
| DFSC | val | 77.72 | 82.70 | 79.34 | 77.58 | 80.31 | 81.48 | 76.83 | 75.65 | 79.04 | 74.33 | **80.09** | **73.73** |
| large-scale | test | 68.98 | 79.47 | 63.42 | 57.34 | 67.78 | 74.38 | 60.24 | 43.02 | 58.46 | 58.48 | 63.86 | 66.24 |
| normal | test | 70.44 | 68.94 | 60.66 | 55.48 | 59.78 | 77.07 | 64.64 | 44.61 | 61.68 | 52.94 | 65.36 | 69.27 |
| DFSC | test | 70.16 | 68.07 | 60.95 | 55.55 | 60.13 | 77.96 | 65.85 | 45.59 | 62.75 | 53.10 | **66.04** | **69.84** |

DFSC − normal: IR val +0.49, IR test +0.68, visible val +0.48, visible test +0.57 mSA. `[PAPER-EXPLICIT]` §V-D

### Table VI — ablation, test set, mSA (%) (same columns)

| Variant | OV | OC | FM | SV | LI | TCe | TCm | TCh | TCall | LR | IR All | Vis All |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DFSC-all | 68.26 | 66.23 | 58.66 | 54.60 | 56.96 | 77.38 | 63.65 | 44.14 | 61.12 | 50.51 | 63.99 | 67.94 |
| DFSC-cls | 70.22 | 68.47 | 60.78 | 55.49 | 60.02 | 78.17 | 65.75 | 45.07 | 62.60 | 52.29 | 66.12 | 69.53 |
| DFSC-reg | 70.31 | 68.26 | 60.70 | 55.31 | 59.05 | 77.52 | 65.52 | 46.74 | 62.82 | 51.76 | 66.06 | 68.95 |
| DFSC (=α1) | 70.16 | 68.07 | 60.95 | 55.55 | 60.13 | 77.96 | 65.85 | 45.59 | 62.75 | 53.10 | 66.04 | 69.84 |
| DFSC-α0.25 | 70.63 | 69.18 | 60.80 | 55.74 | 59.61 | 78.56 | 65.70 | 46.26 | 63.00 | 52.75 | **66.33** | **70.31** |
| DFSC-α0.5 | 70.37 | 68.32 | 60.20 | 55.01 | 58.86 | 77.17 | 64.49 | 45.28 | 61.82 | 51.52 | 65.25 | 69.55 |
| DFSC-α2 | 69.76 | 68.19 | 60.46 | 55.20 | 59.53 | 77.57 | 65.43 | 45.42 | 62.41 | 52.85 | 65.61 | 69.98 |

### Table VII — precision / success (%) "on Anti-UAV" (split not stated; `[INFERRED]` test)

| Method | IR precision | IR success | Vis precision | Vis success |
|---|---|---|---|---|
| large-scale | 85.34 | 61.13 | 88.53 | 63.00 |
| normal | 87.61 | 62.88 | 93.30 | 65.22 |
| DFSC | 87.77 | 63.24 | 93.87 | 65.87 |
| DFSC-all | 86.97 | 61.59 | 91.87 | 64.20 |
| DFSC-cls | 87.47 | 63.14 | 93.72 | 65.56 |
| DFSC-reg | 87.87 | 63.35 | 93.26 | 65.24 |
| DFSC-α0.25 | 87.52 | 63.40 | 93.65 | 66.24 |
| DFSC-α0.5 | 87.08 | 62.53 | 93.84 | 65.66 |
| DFSC-α1 | 87.77 | 63.24 | 93.83 | 65.50 |
| DFSC-α2 | 87.83 | 62.81 | 93.74 | 65.94 |

(Table VII's "DFSC" and "DFSC-α1" visible columns differ slightly — 93.87/65.87 vs 93.83/65.50 —
an internal inconsistency of the paper; recorded, not resolved.)

### Protocol I highlights (Tables III/IV, Fig. 7) `[PAPER-EXPLICIT]`

- Best off-the-shelf tracker: SiamRCNN — IR test mSA 65.41, visible test 70.83; GlobalTrack 63.86 / 66.24.
- SiamRCNN IR val success 71.52 % / precision 95.70 %; IR test success 63.60 %; GlobalTrack IR test precision 87.13 %.
- Best correlation-filter tracker ECO-HC: IR test precision 60.78 %, success 41.54 %.
- Long-term trackers do better (out-of-view + re-detection); TC-hard is the hardest attribute.

## 8. Baselines required for the reproduction

| Role | Definition | Status |
|---|---|---|
| PAPER BASELINE "normal" | GlobalTrack finetuned on Anti-UAV (IR: from COCO Faster R-CNN, 18 ep) without DFSC | implemented as config `normal_ir.yaml` (α = 0) — **must be trained by us** |
| PAPER BASELINE "large-scale" | public GlobalTrack checkpoint, no Anti-UAV training | requires the mmdet checkpoint + weight conversion; **not runnable in this session**, recorded as not reproduced |
| OUR REPRODUCTION | DFSC α = 1, same recipe as "normal" | config `dfsc_ir.yaml` |
| Protocol I trackers (40+) | off-the-shelf | out of scope (not the paper's method) |

## 9. Limitations stated by the paper `[PAPER-EXPLICIT]`

- Train and val share source videos, so val gains overstate generalisation; test is independent and
  gains shrink there ("the tracker is overfitting in the training set to some extent").
- DFSC's gains are small (+0.5 to +0.7 mSA); the paper itself calls the improvement on most
  attributes "limited".
- RGB–IR pairs are unaligned; Protocol III is left as future work.

## 10. What the paper leaves unspecified (consolidated)

1. Number of GPUs / total batch size behind "LR 0.02, batch 2 per GPU".
2. Optimizer, momentum, weight decay, gradient clipping, warm-up.
3. Epoch definition (pairs per epoch), pair-sampling distance, frame filter.
4. Data augmentation.
5. β (cls/reg loss balance) and the total-loss composition.
6. Which proposals feed the ISM stage.
7. Whether an absence ("not exist") decision is made at inference, and with what threshold.
8. Handling of frames before the start frame / of `exist = 0` frames in precision/success.
9. Random seeds, hardware, software versions, run-to-run variance (single runs reported).
10. Which split Table VII refers to.
11. The exact mmdet checkpoint used as "Faster RCNN" init for the IR model.

## 11. Reconciliation statement — what our reproduction follows

The reproduction follows **the paper text** for every explicitly stated setting (§4 epochs, LR
values and steps, batch 2, losses, initialisation source, training data, α, metric), **GlobalTrack's
official code** for every setting the paper delegates to its baseline (§3.1, §4 `[REPO-DERIVED]` rows),
and applies the following **documented substitutions**, forced by the environment:

| # | Substitution | Why | Expected effect |
|---|---|---|---|
| S1 | Re-implementation on **torchvision 0.28 detection primitives** (`resnet50` + FPN, `RegionProposalNetwork`, `RoIHeads`, `MultiScaleRoIAlign`) instead of mmdetection 1.x | mmdet 1.x/mmcv CUDA ops do not build on Windows + PyTorch 2.13 + CUDA 13 (sm_120) | torchvision's Faster R-CNN mirrors Detectron/mmdet conventions (same anchors, IoU thresholds, samplers, box-coder stds, NMS); residual differences: anchor centre offset (+0.5 px legacy in mmdet), FrozenBN (freezes affine too) vs mmdet norm_eval, image resize rounding |
| S2 | IR init from **torchvision `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`** (box AP 37.0) instead of mmdet `faster_rcnn_r50_fpn_2x` (box AP 37.7) | the mmdet checkpoint is in mmdet's key layout; both are COCO-pretrained R50-FPN Faster R-CNN | same architecture and pretraining data; different optimisation trajectory |
| S3 | Single GPU, batch 2, **LR 0.0025** (= 0.02 × 2/16, the mmdet linear-scaling rule applied to the assumed 8-GPU × 2 setting); `accumulate_steps` exposed so an effective batch 16 / LR 0.02 run can be made if wanted | GPU count unspecified; literal LR 0.02 at batch 2 is 8× outside the convention that produced 0.02 | `[INFERRED]`; recorded in every run's config |
| S4 | Frames read from **JPEG q95 decodes** of the MP4s instead of the MP4s | random access needed for pair sampling; decode already materialised by the repo | negligible (both lossy) |
| S5 | Batch sampler guarantees the 2 pairs of a batch come from **different sequences** | DFSC's cross term is defined on different sequences; GlobalTrack's random sampler makes this true with p ≈ 1 − 1/160 | none in expectation |
| S6 | Visible-modality Protocol II runs are **deferred**: they require the GlobalTrack pretrained checkpoint (Google Drive, mmdet format) and a verified weight conversion | no official conversion exists | visible results reported as "not reproduced" until done |

Anything not listed here is implemented as the paper/GlobalTrack states it.
