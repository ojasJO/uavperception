# Deferred — visible (RGB) Protocol II runs

The paper initialises the visible tracker from the **GlobalTrack pretrained checkpoint** (COCO + LaSOT + GOT-10k,
mmdetection format, Google Drive id `1ouTJDS5NACLukd810uiKCvmLxJfga48x`). Running these configs faithfully requires:

1. downloading that checkpoint,
2. a verified key-by-key conversion into this torchvision-based QG-RCNN (not written yet),
3. `model.init: checkpoint` + `model.init_checkpoint: <converted file>` in the config.

Until then the configs below use the COCO Faster R-CNN init and are therefore **NOT a faithful reproduction of
the visible rows of Table V** (they would count as an "ADDITIONAL CONTROL"). Paper visible schedule: 12 epochs,
LR x0.1 at epochs 8 and 11 (already set in `_base_rgb.yaml`).
