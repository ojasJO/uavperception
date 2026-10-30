"""Direction A adapter: AeroTrack temporal-window detector + single-target tracker (STE or argmax), SOT protocol.

Composition of the existing evaluated code, nothing re-implemented:
  * detector loading      src/model/aerotrack_trainer.load_checkpoint(prefer_ema=True)            (as cache_detections.py)
  * temporal window       experiments/systems/scripts/cache_detections.windows (run_meta.json: causal / symmetric / ttt, gap)
  * preprocessing         src/pipeline/stacked_dataset.letterbox_params / letterbox_image, BGR->RGB, /255   (as cache_detections.py)
  * precision / NMS       src/model/precision.amp_dtype autocast on CUDA (fp32 on CPU); NMS conf 0.01, IoU 0.7, max_det 10
  * tracking              experiments/systems/scripts/track_eval.run_tracker (frame-0 GT init, STE / argmax)
  * tracker settings      frozen VAL selections: experiments/cst_anti_uav/frozen_x2_params.json (v3 runs),
                          experiments/aerotrack_v2/frozen_ste_params.json (v2_causal_g4_s0_e20); otherwise STE defaults (warned)
"""
import json
import sys

import numpy as np

from ..data import SOT_DATASETS
from ..paths import REPO
from .base import AdapterError, Compatibility, ModelAdapter, pick_device, verify_file

_PATHS = [REPO / "src/model", REPO / "src/pipeline", REPO / "experiments" / "systems" / "scripts", REPO / "experiments" / "anti_uav" / "src"]


def _sys_path():
    for p in _PATHS:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def frozen_tracker_params(run_id: str) -> tuple[dict, dict, str]:
    """-> (ste params, argmax params, source)"""
    fx = json.loads((REPO / "experiments" / "cst_anti_uav" / "frozen_x2_params.json").read_text(encoding="utf-8")).get("runs", {})
    if run_id in fx:
        return fx[run_id]["ste"], fx[run_id]["argmax"], "experiments/cst_anti_uav/frozen_x2_params.json (CST VAL selection)"
    v2 = REPO / "experiments" / "aerotrack_v2" / "frozen_ste_params.json"
    if v2.exists():
        d = json.loads(v2.read_text(encoding="utf-8"))
        if d.get("candidate") == run_id:
            return d["ste"], d["argmax"], "experiments/aerotrack_v2/frozen_ste_params.json (corpus-v2 VAL selection)"
    return {}, {"thr": 0.3}, "STE defaults (ste.STEParams) + argmax thr 0.3 -- NOT a VAL selection for this run"


class AeroTrackDetectorAdapter(ModelAdapter):
    id = "aerotrack_detector"
    tasks = ("detection+single_target_tracking",)

    def __init__(self, registry=None):
        super().__init__()
        self.registry = registry
        self.meta = {}
        self.amp = None

    def can_run(self, model_record, dataset_id, modality=None):
        if model_record.get("adapter") != self.id:
            return Compatibility(False, f"{model_record['model_id']} uses adapter {model_record.get('adapter')!r}, not {self.id!r}")
        if model_record["status"] in ("smoke", "invalid"):
            return Compatibility(False, f"status '{model_record['status']}': {model_record.get('role') or 'not a usable model'}")
        if dataset_id not in SOT_DATASETS:
            return Compatibility(False, f"dataset {dataset_id!r} has no single-target sequences (this adapter needs Anti-UAV or CST video)")
        if modality and modality not in SOT_DATASETS[dataset_id]:
            return Compatibility(False, f"{dataset_id} has no {modality} modality")
        w = []
        cfg = (model_record.get("config") or {}).get("summary") or {}
        doms = cfg.get("train_domains")
        if doms and dataset_id == "antiuav2021" and "anti_uav" not in doms:
            w.append(f"trained on {doms} only: Anti-UAV is outside its training distribution")
        if model_record.get("checkpoint_tag") == "best":
            w.append("best.pt is fitness-selected and was never the evaluated checkpoint (last.pt policy)")
        try:
            self._resolve_workspace_path(model_record, self.registry, "run_meta.json")
        except AdapterError as e:
            return Compatibility(False, str(e))
        return Compatibility(True, "", w)

    def load(self, model_record, device=None):
        self.record = model_record
        c = self.can_run(model_record, "cst2025")
        if not c.ok:
            raise AdapterError(c.reason)
        t0 = self._timer()
        w = self._resolve_workspace_path(model_record, self.registry, "run_meta.json")
        verify_file(w, model_record["sha256"])
        self.weights_path = w
        self.meta = json.loads((w.parent.parent / "run_meta.json").read_text(encoding="utf-8"))
        gap = int(self.meta.get("gap", self.meta.get("temporal_gap", 1)))
        self.window = self.meta.get("window") or ("ttt" if gap == 0 else "symmetric")
        self.gap = gap
        _sys_path()
        import torch
        from aerotrack_trainer import load_checkpoint
        from precision import amp_dtype
        self.device = pick_device(device)
        dev = torch.device("cuda:0" if self.device == "cuda" else self.device)
        self.model, _ = load_checkpoint(w, dev, prefer_ema=True)
        self.amp = amp_dtype(dev) if dev.type == "cuda" else None
        self._dev = dev
        self.load_s = self._timer() - t0
        return self

    # -- shared inference steps: detect() (batched) and stream() (one frame at a time) run exactly these operations ------
    def _geometry(self, seq):
        """-> (image size (W, H), letterbox params) from frame 0, as cache_detections does"""
        import cv2
        from stacked_dataset import letterbox_params
        img0 = cv2.imread(str(seq.frame_path(0)), cv2.IMREAD_COLOR)
        if img0 is None:
            raise AdapterError(f"cannot read frame {seq.frame_path(0)}")
        H, W = img0.shape[:2]
        return (W, H), letterbox_params(H, W, 640, 640)

    @staticmethod
    def _frame_tensor(path, geom):
        import cv2
        import torch
        from stacked_dataset import letterbox_image
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise AdapterError(f"cannot read frame {path}")
        r, dw, dh, nw, nh = geom
        lb = letterbox_image(im, 640, 640, r, dw, dh, nw, nh)
        return torch.from_numpy(np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))).permute(2, 0, 1)

    def _forward(self, x_uint8):
        """[B, 9, 640, 640] uint8 -> raw head output on the CPU (float)"""
        import torch
        x = x_uint8.to(self._dev).float().div_(255.0)
        with torch.no_grad(), torch.autocast(self._dev.type, dtype=self.amp, enabled=self.amp is not None):
            y = self.model(x)
        return (y[0] if isinstance(y, (list, tuple)) else y).float().cpu()

    @staticmethod
    def _postprocess(y, size, geom):
        """NMS (conf 0.01, IoU 0.7, max 10) and letterbox inverse -> [(boxes xyxy px, scores, classes)] per batch item"""
        from ultralytics.utils.nms import non_max_suppression
        W, H = size
        r, dw, dh, _, _ = geom
        out = []
        for det in non_max_suppression(y, 0.01, 0.7, max_det=10):
            d = det.numpy().astype(np.float32)
            d[:, [0, 2]] = ((d[:, [0, 2]] - dw) / r).clip(0, W)
            d[:, [1, 3]] = ((d[:, [1, 3]] - dh) / r).clip(0, H)
            out.append((d[:, :4].astype(np.float64), d[:, 4].astype(np.float64), d[:, 5].astype(np.int64)))
        return out

    def detect(self, seq, n_frames: int, batch: int = 16):
        """detections for frames 0..n_frames-1 (windows built over the whole sequence, exactly as cache_detections)."""
        import torch
        from cache_detections import windows
        paths = [str(seq.frame_path(t)) for t in range(seq.n_frames)]
        wins = windows([{"cur_path": p} for p in paths], self.window, self.gap)[:n_frames]
        size, geom = self._geometry(seq)
        cache, dets, ms = {}, [], []

        def frame(p):
            if p not in cache:
                cache[p] = self._frame_tensor(p, geom)
                if len(cache) > 64:
                    cache.pop(next(iter(cache)))
            return cache[p]
        for i in range(0, len(wins), batch):
            chunk = wins[i:i + batch]
            t0 = self._timer()
            y = self._forward(torch.stack([torch.cat([frame(p) for p in trip], 0) for trip in chunk]))
            dets += self._postprocess(y, size, geom)
            ms.append((self._timer() - t0) * 1000 / len(chunk))
        return dets, float(np.mean(ms)) if ms else None, size

    def stream_meta(self, tracker="ste", params=None) -> dict:
        """what a stream will use (recorded in the experiment manifest before the first frame)"""
        ste_p, arg_p, src = frozen_tracker_params(self.record["run_id"])
        p = params if params is not None else (ste_p if tracker == "ste" else arg_p)
        warn = [] if params is not None or "NOT a VAL" not in src else [f"tracker settings: {src}"]
        return {"tracker": tracker, "params": p, "params_source": src if params is None else "user override",
                "window": {"window": self.window, "gap": self.gap},
                "score_kind": "score = highest drone-class detection score of the frame; assoc_score = the score of the detection the "
                              "STE associated (argmax: none)",
                "preprocessing": {"letterbox": 640, "nms": {"conf": 0.01, "iou": 0.7, "max_det": 10}, "batch": 1,
                                  "precision": str(self.amp).replace("torch.", "") + " autocast" if self.amp is not None else "fp32"},
                "warnings": warn}

    def stream(self, seq, end_frame=None, tracker="ste", params=None, dataset_id="cst2025", modality="IR"):
        """Incremental single-target tracking, one frame at a time (batch 1), from the start frame to end_frame (exclusive).

        Same window, letterbox, autocast, NMS and STE / argmax calls as predict() (via _frame_tensor / _forward / _postprocess and
        ste.STE.step, initialised exactly as track_eval.run_tracker). Yields one frame record per processed frame with per-stage
        wall-clock timing; the model time is CUDA-synchronised. At the start frame the GT box is the output (one-pass protocol)."""
        if self.model is None:
            raise AdapterError("load() first")
        c = self.can_run(self.record, dataset_id, modality)
        if not c.ok:
            raise AdapterError(c.reason)
        if tracker not in ("ste", "argmax"):
            raise AdapterError(f"tracker must be 'ste' or 'argmax', not {tracker!r}")
        _sys_path()
        import torch
        import ste as T
        import track_eval as TE
        from cache_detections import windows
        p = self.stream_meta(tracker, params)["params"]
        end = seq.n_frames if end_frame is None else min(seq.n_frames, int(end_frame))
        start = seq.start_frame
        init = seq.box_xywh(start)
        if init is None:
            raise AdapterError(f"{seq.name}: start frame {start} has no ground-truth box to initialise from")
        paths = [str(seq.frame_path(t)) for t in range(seq.n_frames)]
        wins = windows([{"cur_path": q} for q in paths], self.window, self.gap)
        size, geom = self._geometry(seq)
        if tracker == "ste":
            trk = T.STE(T.STEParams(**{**TE.DEFAULT_STE, **p}))
            trk.reset(np.array([init[0], init[1], init[0] + init[2], init[1] + init[3]], dtype=np.float64))
        cache = {}
        sync = self._dev.type == "cuda"
        for t in range(start, end):
            t0 = self._timer()
            for q in wins[t]:
                if q not in cache:
                    cache[q] = self._frame_tensor(q, geom)
            while len(cache) > 32:
                cache.pop(next(iter(cache)))
            x = torch.cat([cache[q] for q in wins[t]], 0)[None]
            t1 = self._timer()
            y = self._forward(x)
            if sync:
                torch.cuda.synchronize()
            t2 = self._timer()
            b, sc, cl = self._postprocess(y, size, geom)[0]
            t3 = self._timer()
            if t == start:
                box, mode, assoc = [float(init[0]), float(init[1]), float(init[0] + init[2]), float(init[1] + init[3])], "init", None
            elif tracker == "argmax":
                bb = T.argmax_step(b, sc, cl, p["thr"])
                box, mode, assoc = (None, "absent", None) if bb is None else ([float(v) for v in bb], "argmax", None)
            else:
                bb, mode, assoc = trk.step(b, sc, cl)
                box = None if bb is None else [float(v) for v in bb]
            t4 = self._timer()
            keep = cl == 0
            top = [[*map(float, b[k]), float(sc[k])] for k in np.flatnonzero(keep)[np.argsort(-sc[keep])][:5]]
            yield {"frame": t, "present": box is not None, "box_xyxy": box, "score": float(sc[keep].max()) if keep.any() else None,
                   "assoc_score": assoc, "mode": mode, "detections": top,
                   "timing_ms": {"read_preprocess": (t1 - t0) * 1000, "model": (t2 - t1) * 1000, "postprocess": (t3 - t2) * 1000,
                                 "tracker": (t4 - t3) * 1000, "total": (t4 - t0) * 1000}}

    def predict(self, seq, dataset_id="cst2025", split="val", modality="IR", max_frames=None, tracker="ste", params=None):
        if self.model is None:
            raise AdapterError("load() first")
        c = self.can_run(self.record, dataset_id, modality)
        if not c.ok:
            raise AdapterError(c.reason)
        if tracker not in ("ste", "argmax"):
            raise AdapterError(f"tracker must be 'ste' or 'argmax', not {tracker!r}")
        _sys_path()
        import track_eval as TE
        from anti_uav_repro.data.antiuav import Sequence
        n = seq.n_frames if not max_frames else min(seq.n_frames, seq.start_frame + int(max_frames))
        sub = Sequence(name=seq.name, split=seq.split, modality=seq.modality, label_path=seq.label_path, frames_dir=seq.frames_dir,
                       exist=list(seq.exist[:n]), gt_rect=list(seq.gt_rect[:n]))
        dets, det_ms, size = self.detect(seq, n)
        ste_p, arg_p, src = frozen_tracker_params(self.record["run_id"])
        p = params if params is not None else (ste_p if tracker == "ste" else arg_p)
        warnings = list(c.warnings) + ([] if params is not None or "NOT a VAL" not in src else [f"tracker settings: {src}"])
        t0 = self._timer()
        pred = TE.run_tracker(tracker, p, {"seq": sub, "dets": dets})
        trk_ms = (self._timer() - t0) * 1000 / max(n, 1)
        frames = []
        for t in range(n):
            b = pred["res"][t]
            box = None if not (isinstance(b, list) and len(b) == 4) else [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            bx, sc, cl = dets[t]
            keep = cl == 0
            top = [[*map(float, bx[k]), float(sc[k])] for k in np.flatnonzero(keep)[np.argsort(-sc[keep])][:5]]
            frames.append({"frame": t, "present": None if t < sub.start_frame else box is not None, "box_xyxy": box,
                           "score": float(sc[keep].max()) if keep.any() else None, "mode": None, "detections": top})
        return {"kind": "sot", "adapter": self.id, "model_id": self.record["model_id"], "checkpoint_sha256": self.record["sha256"],
                "checkpoint_path": self.get_metadata()["checkpoint_path"], "dataset_id": dataset_id, "split": split, "modality": modality,
                "sequence": seq.name, "n_frames_total": seq.n_frames, "n_frames_processed": n, "start_frame": pred["start_frame"],
                "res_xywh": pred["res"], "processed": pred["processed"], "frames": frames, "image_size": list(size),
                "device": self.device, "timing_ms": {"detector_per_frame": det_ms, "tracker_per_frame": trk_ms},
                "score_kind": "highest drone-class detection score of the frame (the tracker's own association score is not exported by track_eval.run_tracker)",
                "tracker": tracker, "params": p, "params_source": src if params is None else "user override",
                "window": {"window": self.window, "gap": self.gap},
                "preprocessing": {"letterbox": 640, "nms": {"conf": 0.01, "iou": 0.7, "max_det": 10},
                                  "precision": str(self.amp).replace("torch.", "") + " autocast" if self.amp is not None else "fp32"},
                "warnings": warnings}
