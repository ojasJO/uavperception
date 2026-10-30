# Erratum (2026-10-05): the "YOLO11n" of this series is not built as YOLO11n

Found during the 2026-10-05 repository knowledge audit; the registration (`preregistration.json`) is left untouched because the
campaign runs under it.

- **What the registration says:** "function-preserving initialisation (every configuration starts as COCO YOLO11n on frame t)".
- **What actually happens:** `c_model/yolo11-stride.yaml` has flat multipliers and no scale letter, so Ultralytics 8.4.115 resolves
  `scale = ''`; `parse_model`'s `if scale in "mlx"` is True for the empty string, and every C3k2 block is built with `c3k=True`
  (YOLO11 m/l/x's inner C3k) instead of YOLO11n's Bottleneck. The network has 2.613 M parameters and only **417 / 649 tensors
  (64.3 %)** of `yolo11n.pt` load into it; the C3k2 internals of layers 2, 4, 13, 16 and 19 start from random init. The same holds
  for every earlier AeroTrack run built from these YAMLs (repair and generalization series).
- **What remains valid:** every registered comparison of this series is between configurations built the same way, from identical
  starting weights except the zeroed neighbour-frame channels, so window verdicts are internally valid. "Starts as COCO YOLO11n" should
  read "starts as the same partially COCO-initialised network on frame t".
- **What is being done:** `c_model/yolo11-stride-std.yaml` (explicit `scales: n`) builds standard YOLO11n (2.591 M params, 448 / 499
  tensors = 89.8 % load). Its effect is measured by the registered experiment F1
  (`experiments/aerotrack_v3/preregistration_F1_standard_yolo11n.json`); X1 and the Track C final system adopt it only if F1 is
  SUPPORTED.
