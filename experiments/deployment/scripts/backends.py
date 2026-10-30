"""Inference backends behind one interface: `backend(x)` with x a float32 CUDA tensor [B, 9, 640, 640] in [0, 1] returns the
raw prediction tensor [B, 4 + nc, 8400] on the same device -- exactly what `src/model/evaluation.collect_records` hands to NMS, so
every backend is scored by the identical post-processing and metric code.

    torch_fp32 / torch_bf16 / torch_fp16   the checkpoint in PyTorch (bf16 = autocast, the evaluation protocol)
    ort_cuda / ort_trt                     ONNX Runtime with the CUDA / TensorRT execution provider (if their DLLs load)
    trt_fp32 / trt_fp16                    a native TensorRT engine built from the exported ONNX (if `tensorrt` imports)

Every constructor raises BackendUnavailable with the reason when its runtime is missing, so a benchmark can report "not
available: <why>" instead of silently skipping.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src/model", REPO / "src/pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402


class BackendUnavailable(RuntimeError):
    pass


class TorchBackend:
    def __init__(self, weights, precision="fp32", device="cuda:0"):
        from aerotrack_trainer import load_checkpoint
        self.name = f"torch_{precision}"
        self.device = torch.device(device)
        self.model, self.ckpt = load_checkpoint(weights, self.device, prefer_ema=True, fuse=True)
        self.precision = precision
        if precision == "fp16":
            self.model = self.model.half()
        self.amp = torch.bfloat16 if precision == "bf16" else None

    def eval(self):
        return self

    @torch.no_grad()
    def __call__(self, x):
        if self.precision == "fp16":
            x = x.half()
        if self.amp is not None:
            with torch.autocast("cuda", dtype=self.amp):
                y = self.model(x)
        else:
            y = self.model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        return y.float()


class OrtBackend:
    def __init__(self, onnx_path, provider="cuda", device_id=0):
        import onnxruntime as ort
        self.name = f"ort_{provider}"
        prov = {"cuda": [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"],
                "trt": [("TensorrtExecutionProvider", {"device_id": device_id, "trt_fp16_enable": True}),
                        ("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"],
                "cpu": ["CPUExecutionProvider"]}[provider]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=prov)
        except Exception as e:  # noqa: BLE001
            raise BackendUnavailable(f"{self.name}: {type(e).__name__}: {str(e)[:300]}")
        want = {"cuda": "CUDAExecutionProvider", "trt": "TensorrtExecutionProvider", "cpu": "CPUExecutionProvider"}[provider]
        active = self.sess.get_providers()
        if active[0] != want:
            raise BackendUnavailable(f"{self.name}: requested {want}, ONNX Runtime fell back to {active} (missing CUDA/cuDNN/TensorRT DLLs)")
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
        self.out_shape = tuple(int(d) for d in self.sess.get_outputs()[0].shape)
        self.on_gpu = provider != "cpu"

    def eval(self):
        return self

    def __call__(self, x):
        import numpy as np
        outs = []
        for i in range(x.shape[0]):                      # the exported graph is batch-1
            xi = x[i:i + 1].float().contiguous()
            if not self.on_gpu:
                outs.append(torch.from_numpy(self.sess.run(None, {self.inp: xi.cpu().numpy()})[0]).to(x.device))
                continue
            # IO binding on the torch CUDA buffers: no host round trip (a fair comparison with the other GPU backends)
            torch.cuda.current_stream().synchronize()   # inputs written by torch kernels must be complete before ORT reads them
            y = torch.empty(self.out_shape, dtype=torch.float32, device=x.device)
            b = self.sess.io_binding()
            b.bind_input(name=self.inp, device_type="cuda", device_id=xi.device.index or 0, element_type=np.float32,
                         shape=tuple(xi.shape), buffer_ptr=xi.data_ptr())
            b.bind_output(name=self.out, device_type="cuda", device_id=y.device.index or 0, element_type=np.float32,
                          shape=self.out_shape, buffer_ptr=y.data_ptr())
            self.sess.run_with_iobinding(b)
            outs.append(y)
        return torch.cat(outs)


class TrtBackend:
    """Native TensorRT engine (built once from the ONNX file, cached next to it with the build settings in its name)."""

    def __init__(self, onnx_path, precision="fp16", workspace_gb=2):
        try:
            import tensorrt as trt
        except ImportError:
            try:                                          # TensorRT >= 11 wheels without the `tensorrt` meta package
                import tensorrt_libs  # noqa: F401        (registers the TensorRT DLL directory)
                import tensorrt_bindings as trt
            except Exception as e:  # noqa: BLE001
                raise BackendUnavailable(f"trt_{precision}: tensorrt not importable ({type(e).__name__}: {e})")
        self.name = f"trt_{precision}"
        self.trt = trt
        onnx_path = Path(onnx_path)
        if precision == "fp16":                           # TensorRT >= 11: precision comes from the graph's types
            onnx_path = onnx_path.with_name(onnx_path.stem + "_fp16.onnx")
            if not onnx_path.exists():
                raise BackendUnavailable(f"{self.name}: {onnx_path.name} missing (export_onnx.py --fp16)")
        engine_path = onnx_path.with_name(f"{onnx_path.stem}_trt{trt.__version__}_{precision}.engine")
        logger = trt.Logger(trt.Logger.WARNING)
        self.build_seconds = None
        if not engine_path.exists():
            t0 = time.time()
            builder = trt.Builder(logger)
            strongly = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
            net = builder.create_network(0 if strongly is None else 1 << int(strongly))
            parser = trt.OnnxParser(net, logger)
            if not parser.parse(onnx_path.read_bytes()):
                errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise BackendUnavailable(f"{self.name}: ONNX parse failed: {errs[:3]}")
            cfg = builder.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
            if precision == "fp16" and hasattr(trt.BuilderFlag, "FP16") and strongly is None:
                cfg.set_flag(trt.BuilderFlag.FP16)          # TensorRT < 10 weakly-typed path
            blob = builder.build_serialized_network(net, cfg)
            if blob is None:
                raise BackendUnavailable(f"{self.name}: engine build failed")
            engine_path.write_bytes(bytes(blob))
            self.build_seconds = round(time.time() - t0, 1)
        self.engine_path = engine_path
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.in_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        self.io_dtype = torch.float16 if self.engine.get_tensor_dtype(self.in_name) == trt.DataType.HALF else torch.float32
        self.out_dtype = torch.float16 if self.engine.get_tensor_dtype(self.out_name) == trt.DataType.HALF else torch.float32
        self.stream = torch.cuda.Stream()

    def eval(self):
        return self

    @torch.no_grad()
    def __call__(self, x):
        outs = []
        for i in range(x.shape[0]):
            xi = x[i:i + 1].to(self.io_dtype).contiguous()
            y = torch.empty(self.out_shape, dtype=self.out_dtype, device=x.device)
            self.ctx.set_tensor_address(self.in_name, xi.data_ptr())
            self.ctx.set_tensor_address(self.out_name, y.data_ptr())
            self.stream.wait_stream(torch.cuda.current_stream())
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            torch.cuda.current_stream().wait_stream(self.stream)
            outs.append(y.float())
        return torch.cat(outs)


def make(kind, weights, onnx_path):
    if kind.startswith("torch_"):
        return TorchBackend(weights, kind.split("_", 1)[1])
    if kind.startswith("ort_"):
        return OrtBackend(onnx_path, kind.split("_", 1)[1])
    if kind.startswith("trt_"):
        return TrtBackend(onnx_path, kind.split("_", 1)[1])
    raise ValueError(kind)
