"""QG-RCNN (GlobalTrack) with the DFSC training strategy, built on torchvision's Faster R-CNN parts.

Correspondence to GlobalTrack `modules/qg_rcnn.py` / `configs/qg_rcnn_r50_fpn.py`
(see docs/ANTI_UAV_REPRODUCTION_SPEC.md §3.1 and §11 for the residual differences):

    backbone + FPN      torchvision resnet50 + FPN (P2-P6), stem + res2 frozen, FrozenBN
    rpn_head            torchvision RPNHead (3x3 conv, 3 anchors/location: scale 8 x strides 4..64, ratios .5/1/2)
    rpn train/test      pos .7 / neg .3, 256 anchors @ .5 pos; pre/post NMS 2000 (train) 1000 (test), NMS .7
    roi extractor       MultiScaleRoIAlign 7x7, sampling 2, on P2-P5
    bbox head           TwoMLPHead (1024-1024) + FastRCNNPredictor(2 classes, class-specific reg)
    rcnn train/test     pos/neg .5, 512 RoIs @ .25 pos, GT added; score_thr 0, NMS .5, max 1000
    losses              RPN: BCE + SmoothL1(beta 1/9);  RCNN: CE + SmoothL1(beta 1.0)  (upstream values)

DFSC (paper §IV): with a batch of B pairs from B different sequences,
    L = (1/B) * [ sum_{i=j} L_rpn(t_ij) + alpha * sum_{i!=j} L_rpn(t_ij) + sum_j L_rcnn(ISM_j) ]
where t_ij = search features of pair j modulated by the query of pair i, and ISM_j uses the proposals
of the intra-sequence branch t_jj modulated by the query RoI feature of its own sequence j.
alpha = 0 reduces exactly to GlobalTrack fine-tuning (the paper's "normal" baseline).
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.ops.misc import FrozenBatchNorm2d

from .modulators import RCNNModulator, RPNModulator

COCO_INIT = "torchvision_coco_fasterrcnn_r50_fpn_v1"
IMAGENET_INIT = "imagenet_backbone_only"


def build_base_faster_rcnn(model_cfg: dict) -> tuple[FasterRCNN, float]:
    """Faster R-CNN R50-FPN whose ARCHITECTURE is independent of the weight source.

    torchvision's `fasterrcnn_resnet50_fpn` silently changes the architecture with the init: without
    pretrained weights it uses trainable `BatchNorm2d` and unfreezes every stage, and for COCO_V1 it sets
    FrozenBatchNorm eps to 0.0 (`overwrite_eps`). We always build FrozenBatchNorm2d (mmdet `norm_eval` analogue),
    always freeze stem + res2 (`trainable_backbone_layers=3` == mmdet `frozen_stages=1`), and pin the BN eps:
    0.0 when COCO_V1 weights are loaded (their training-time value), otherwise 1e-5, or the value recorded in
    the config (`frozen_bn_eps`) so a checkpoint is rebuilt exactly as it was trained.
    """
    init = model_cfg.get("init", COCO_INIT)
    rpn = model_cfg.get("rpn", {})
    roi = model_cfg.get("roi", {})
    body = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    if init == IMAGENET_INIT:
        body.load_state_dict(ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=False))
    backbone = _resnet_fpn_extractor(body, trainable_layers=int(model_cfg.get("trainable_backbone_layers", 3)))
    base = FasterRCNN(
        backbone, num_classes=91,
        rpn_pre_nms_top_n_train=rpn.get("pre_nms_top_n_train", 2000), rpn_post_nms_top_n_train=rpn.get("post_nms_top_n_train", 2000),
        rpn_pre_nms_top_n_test=rpn.get("pre_nms_top_n_test", 1000), rpn_post_nms_top_n_test=rpn.get("post_nms_top_n_test", 1000),
        rpn_nms_thresh=rpn.get("nms_thresh", 0.7),
        box_score_thresh=roi.get("score_thresh", 0.0), box_nms_thresh=roi.get("nms_thresh", 0.5),
        box_detections_per_img=roi.get("detections_per_img", 1000),
    )
    eps = model_cfg.get("frozen_bn_eps", None)
    if init == COCO_INIT:
        base.load_state_dict(FasterRCNN_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=False))
        eps = 0.0 if eps is None else float(eps)      # torchvision's rule for these weights
    eps = 1e-5 if eps is None else float(eps)
    overwrite_eps(base, eps)
    return base, eps


def fastrcnn_loss_beta(class_logits: Tensor, box_regression: Tensor, labels: list[Tensor],
                       regression_targets: list[Tensor], beta: float = 1.0) -> tuple[Tensor, Tensor]:
    """torchvision.models.detection.roi_heads.fastrcnn_loss with a configurable SmoothL1 beta.

    torchvision hard-codes beta = 1/9 for the box head; GlobalTrack/mmdet use beta = 1.0 (config
    `bbox_head.loss_bbox.beta=1.0`). Everything else is identical: CE over all sampled RoIs,
    class-specific regression on positives, normalised by the number of sampled RoIs.
    """
    labels_cat = torch.cat(labels, dim=0)
    regression_targets_cat = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels_cat)
    pos = torch.where(labels_cat > 0)[0]
    labels_pos = labels_cat[pos]
    n, _ = class_logits.shape
    box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)
    box_loss = F.smooth_l1_loss(box_regression[pos, labels_pos], regression_targets_cat[pos],
                                beta=beta, reduction="sum") / labels_cat.numel()
    return classification_loss, box_loss


class QGRCNN(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        base, self.frozen_bn_eps = build_base_faster_rcnn(model_cfg)
        self.init_source = model_cfg.get("init", COCO_INIT)
        self.backbone = base.backbone          # resnet50 + FPN -> OrderedDict('0','1','2','3','pool')
        self.rpn = base.rpn                    # RegionProposalNetwork (anchor gen + head + losses + proposal filtering)
        self.roi_heads = base.roi_heads        # box_roi_pool / box_head / (replaced) box_predictor / samplers
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        # 2 classes = background + target (GlobalTrack bbox_head num_classes=2); new random head as upstream
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, model_cfg.get("num_classes", 2))
        self.rpn_modulator = RPNModulator()
        self.rcnn_modulator = RCNNModulator()
        self.rcnn_reg_beta = float(model_cfg.get("rcnn_smooth_l1_beta", 1.0))
        # the FrozenBN eps travels inside the state_dict so a checkpoint rebuilds its exact architecture
        self.register_buffer("_frozen_bn_eps", torch.tensor(float(self.frozen_bn_eps)))
        # tracking state
        self._z_rpn: Tensor | None = None
        self._z_rcnn: Tensor | None = None

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if "_frozen_bn_eps" in state_dict:
            self.frozen_bn_eps = float(self._frozen_bn_eps)
            overwrite_eps(self, self.frozen_bn_eps)
        return result

    # ------------------------------------------------------------------------------ helpers
    @staticmethod
    def _slice_feats(feats: "OrderedDict[str, Tensor]", j: int) -> "OrderedDict[str, Tensor]":
        return OrderedDict((k, v[j:j + 1]) for k, v in feats.items())

    def _rcnn_stage(self, feats_xj, proposals: Tensor, boxes_xj: Tensor, size_xj, z_rcnn_q: Tensor):
        """ISM: sample RoIs against the GT of search image j, modulate with query RoI feature, box head, loss."""
        device = proposals.device
        targets = [{"boxes": boxes_xj, "labels": torch.ones((boxes_xj.shape[0],), dtype=torch.int64, device=device)}]
        props, _, labels, regression_targets = self.roi_heads.select_training_samples([proposals], targets)
        box_feats = self.roi_heads.box_roi_pool(feats_xj, props, [size_xj])
        box_feats = self.rcnn_modulator(z_rcnn_q, box_feats)
        box_feats = self.roi_heads.box_head(box_feats)
        class_logits, box_regression = self.roi_heads.box_predictor(box_feats)
        return fastrcnn_loss_beta(class_logits, box_regression, labels, regression_targets, beta=self.rcnn_reg_beta)

    # ------------------------------------------------------------------------------ training
    def forward_train(self, batch: dict, alpha: float, cross_terms=("cls", "reg"), ism_modulation: str = "same") -> dict:
        assert self.training
        device = next(self.parameters()).device
        img_z = batch["img_z"].to(device, non_blocking=True)
        img_x = batch["img_x"].to(device, non_blocking=True)
        boxes_z = [b.to(device) for b in batch["boxes_z"]]
        boxes_x = [b.to(device) for b in batch["boxes_x"]]
        sizes_z, sizes_x = batch["sizes_z"], batch["sizes_x"]
        B = img_z.shape[0]
        cross_terms = set(cross_terms)

        feats_z = self.backbone(img_z)
        feats_x = self.backbone(img_x)
        z_rpn = self.rpn_modulator.learn(feats_z, boxes_z, sizes_z)          # [B, C, 7, 7]
        z_rcnn = self.roi_heads.box_roi_pool(feats_z, boxes_z, sizes_z)      # [B, C, 7, 7]

        zero = img_z.new_zeros(())
        losses = {k: zero.clone() for k in ("rpn_cls_same", "rpn_reg_same", "rpn_cls_cross", "rpn_reg_cross", "rcnn_cls", "rcnn_reg")}
        n_cross = 0
        for j in range(B):                                   # search image j (targets = its own UAV box)
            feats_xj = self._slice_feats(feats_x, j)
            il = ImageList(img_x[j:j + 1], [tuple(sizes_x[j])])
            rpn_targets = [{"boxes": boxes_x[j]}]
            for i in range(B):                               # query i modulates search j
                if i != j and alpha <= 0.0:
                    continue
                mod = self.rpn_modulator.modulate(feats_xj, z_rpn[i:i + 1])
                proposals, rpn_losses = self.rpn(il, mod, rpn_targets)
                if i == j:
                    losses["rpn_cls_same"] = losses["rpn_cls_same"] + rpn_losses["loss_objectness"]
                    losses["rpn_reg_same"] = losses["rpn_reg_same"] + rpn_losses["loss_rpn_box_reg"]
                    lc, lr = self._rcnn_stage(feats_xj, proposals[0], boxes_x[j], tuple(sizes_x[j]), z_rcnn[j:j + 1])
                    losses["rcnn_cls"] = losses["rcnn_cls"] + lc
                    losses["rcnn_reg"] = losses["rcnn_reg"] + lr
                else:
                    n_cross += 1
                    if "cls" in cross_terms:
                        losses["rpn_cls_cross"] = losses["rpn_cls_cross"] + alpha * rpn_losses["loss_objectness"]
                    if "reg" in cross_terms:
                        losses["rpn_reg_cross"] = losses["rpn_reg_cross"] + alpha * rpn_losses["loss_rpn_box_reg"]
                    if ism_modulation == "cross":            # ablation "DFSC-all": ISM also uses cross-sequence queries
                        lc, lr = self._rcnn_stage(feats_xj, proposals[0], boxes_x[j], tuple(sizes_x[j]), z_rcnn[i:i + 1])
                        losses["rcnn_cls"] = losses["rcnn_cls"] + alpha * lc
                        losses["rcnn_reg"] = losses["rcnn_reg"] + alpha * lr
        # upstream averages over the number of query instances (B); Eq. (3)'s sums are kept, scaled by 1/B
        losses = {k: v / B for k, v in losses.items()}
        losses["total"] = sum(losses.values())
        losses["_n_cross_pairs"] = torch.tensor(float(n_cross))
        return losses

    # ------------------------------------------------------------------------------ tracking
    @torch.no_grad()
    def track_init(self, img_z: Tensor, box_z_xyxy: Tensor, size_z: tuple[int, int]) -> None:
        """img_z [1,3,H,W] normalised+padded; box_z_xyxy [1,4] in the same (rescaled) coordinates."""
        assert not self.training
        feats_z = self.backbone(img_z)
        self._z_rpn = self.rpn_modulator.learn(feats_z, [box_z_xyxy], [tuple(size_z)])
        self._z_rcnn = self.roi_heads.box_roi_pool(feats_z, [box_z_xyxy], [tuple(size_z)])

    @torch.no_grad()
    def track_update(self, img_x: Tensor, size_x: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """Returns (boxes [N,4] xyxy in rescaled coords, scores [N]) sorted by score desc, after NMS 0.5."""
        assert not self.training and self._z_rpn is not None, "call track_init first"
        feats_x = self.backbone(img_x)
        mod = self.rpn_modulator.modulate(feats_x, self._z_rpn)
        il = ImageList(img_x, [tuple(size_x)])
        proposals, _ = self.rpn(il, mod)
        box_feats = self.roi_heads.box_roi_pool(feats_x, proposals, [tuple(size_x)])
        box_feats = self.rcnn_modulator(self._z_rcnn, box_feats)
        box_feats = self.roi_heads.box_head(box_feats)
        class_logits, box_regression = self.roi_heads.box_predictor(box_feats)
        boxes, scores, _ = self.roi_heads.postprocess_detections(class_logits, box_regression, proposals, [tuple(size_x)])
        return boxes[0], scores[0]

    # ------------------------------------------------------------------------------ misc
    def parameter_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        groups = {}
        for name, p in self.named_parameters():
            g = name.split(".")[0]
            d = groups.setdefault(g, {"total": 0, "trainable": 0})
            d["total"] += p.numel()
            d["trainable"] += p.numel() if p.requires_grad else 0
        return {"total": total, "trainable": trainable, "by_module": groups, "init_source": self.init_source,
                "frozen_bn_eps": self.frozen_bn_eps}
