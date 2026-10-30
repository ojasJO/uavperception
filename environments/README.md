# Environments

| environment | where | what | record |
|---|---|---|---|
| OMEN research + deployment venv | `venv/` (local only) | Python 3.11.9, torch 2.13.0+cu130, Ultralytics 8.4.115, RTX 5050 (sm_120), driver 592.19 | `venv_freeze_2026-10-05.txt` |
| remote HPC | not available (no access details on this machine) | — | — |

Changes made by the research program (purely additive; no existing package changed — verified with pip dry runs):

- 2026-10-04: `onnx 1.23.1`, `onnxruntime-gpu 1.30.0`, `onnxscript 0.7.2` (+ `onnx-ir`, `ml_dtypes`, `flatbuffers`) for the
  ONNX export / ONNX Runtime backends.
- 2026-10-05: `tensorrt_cu13_libs` + `tensorrt_cu13_bindings` 11.3.0.99 from the wheels in `environments/wheels/` (downloaded from
  pypi.nvidia.com, sha256 verified against the index; the `tensorrt` meta package is not installed, so the module is imported as
  `tensorrt_bindings` after `import tensorrt_libs` — `experiments/deployment/scripts/backends.py` handles both).
  TensorRT >= 11 has no FP16 builder flag: FP16 engines are built from an FP16-typed ONNX graph (`export_onnx.py --fp16`) as
  strongly typed networks.

Every run records its own environment snapshot (`run_meta.json` -> `env`).
