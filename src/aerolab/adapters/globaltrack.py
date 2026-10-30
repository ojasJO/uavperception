"""Direction B adapter: GlobalTrack (QG-RCNN, torchvision re-implementation) fine-tuned on Anti-UAV -- the paper reproduction.

Exactly the reproduction's own tracker, as used by scripts/evaluate.py and the VAL demo (experiments/anti_uav/demo/demo_core.live_track):
QGRCNN({...model cfg, init: none}) + strict state-dict load + GlobalTrackTracker(TestTransform(scale, pad_divisor)).track_sequence.
The tracker never reports absence (a box every frame), which matters under Eq. 1 -- shown as a warning, never hidden.
"""
import sys

import yaml

from ..paths import REPO
from .base import AdapterError, Compatibility, ModelAdapter, pick_device, verify_file

SRC = REPO / "experiments" / "anti_uav" / "src"
BASE_CFG = REPO / "experiments" / "anti_uav" / "configs" / "_base_ir.yaml"


class GlobalTrackAdapter(ModelAdapter):
    id = "globaltrack_repro"
    tasks = ("single_target_tracking",)

    def __init__(self, registry=None):
        super().__init__()
        self.registry = registry
        self.cfg = None
        self.cfg_source = None

    def _config(self, weights):
        run_dir = weights.parent.parent
        p = run_dir / "config.effective.yaml"
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")), p.relative_to(REPO).as_posix()
        return yaml.safe_load(BASE_CFG.read_text(encoding="utf-8")), f"{BASE_CFG.relative_to(REPO).as_posix()} (run has no config.effective.yaml)"

    def can_run(self, model_record, dataset_id, modality=None):
        if model_record.get("adapter") != self.id:
            return Compatibility(False, f"{model_record['model_id']} uses adapter {model_record.get('adapter')!r}, not {self.id!r}")
        if model_record["status"] == "invalid":
            return Compatibility(False, f"status 'invalid': {model_record.get('role')}")
        if dataset_id not in ("antiuav2021", "cst2025"):
            return Compatibility(False, f"{dataset_id!r} has no single-target sequences")
        try:
            w = self._resolve_workspace_path(model_record, self.registry)
        except AdapterError as e:
            return Compatibility(False, str(e))
        cfg, _ = self._config(w)
        trained = (cfg.get("data") or {}).get("modality")
        warn = ["this tracker never reports 'absent': every target-absent frame scores 0 under Eq. 1"]
        if model_record["checkpoint_tag"].startswith("epoch_"):
            warn.append("intermediate epoch checkpoint (never evaluated; the reproduction uses the final checkpoint)")
        if dataset_id == "cst2025":
            warn.append("trained on Anti-UAV, applied to CST: cross-dataset, not a reproduction result")
        if modality and trained and modality != trained:
            warn.append(f"trained on {trained}, applied to {modality}: cross-modality, not a reproduction result")
        return Compatibility(True, "", warn)

    def load(self, model_record, device=None):
        self.record = model_record
        c = self.can_run(model_record, "antiuav2021")
        if not c.ok:
            raise AdapterError(c.reason)
        t0 = self._timer()
        w = self._resolve_workspace_path(model_record, self.registry)
        verify_file(w, model_record["sha256"])
        self.weights_path = w
        self.cfg, self.cfg_source = self._config(w)
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        import torch
        from anti_uav_repro.data.transforms import TestTransform
        from anti_uav_repro.engine.track import GlobalTrackTracker
        from anti_uav_repro.models.qg_rcnn import QGRCNN
        self.device = pick_device(device)
        dev = torch.device(self.device)
        ck = torch.load(w, map_location="cpu", weights_only=False)
        model = QGRCNN({**self.cfg["model"], "init": "none"})
        try:
            model.load_state_dict(ck["model"] if "model" in ck else ck, strict=True)
        except RuntimeError as e:
            raise AdapterError(f"state dict does not match QG-RCNN as configured by {self.cfg_source}: {str(e)[:300]}") from e
        model.to(dev).eval()
        tr = self.cfg["data"]["transforms"]
        self.model = model
        self.tracker = GlobalTrackTracker(model, dev, TestTransform(tuple(tr["scale"]), int(tr["pad_divisor"])))
        self.load_s = self._timer() - t0
        return self

    def stream_meta(self, tracker="globaltrack", params=None) -> dict:
        tr = self.cfg["data"]["transforms"]
        return {"tracker": "globaltrack", "params": {"frame_stride": 1}, "params_source": self.cfg_source, "window": None,
                "score_kind": "GlobalTrack top-1 box score (not a presence probability; the tracker never reports absent)",
                "preprocessing": {"transform": "TestTransform", "scale": tr["scale"], "pad_divisor": tr["pad_divisor"], "precision": "fp32"},
                "warnings": []}

    def stream(self, seq, end_frame=None, dataset_id="antiuav2021", modality="IR", **_):
        """Incremental tracking, one frame at a time: exactly the loop of GlobalTrackTracker.track_sequence (read_image_rgb,
        TestTransform, track_init at the start frame with the GT box, track_update per frame, top-1 box rescaled, previous box
        kept if the model returns none). Yields one frame record per processed frame with CUDA-synchronised timing."""
        if self.model is None:
            raise AdapterError("load() first")
        c = self.can_run(self.record, dataset_id, modality)
        if not c.ok:
            raise AdapterError(c.reason)
        import torch
        from anti_uav_repro.data.transforms import read_image_rgb, xywh_to_xyxy
        tr, model, dev = self.tracker, self.model, self.tracker.device
        sync = dev.type == "cuda"
        end = seq.n_frames if end_frame is None else min(seq.n_frames, int(end_frame))
        start = seq.start_frame
        init = seq.box_xywh(start)
        if init is None:
            raise AdapterError(f"{seq.name}: start frame {start} has no valid GT box")
        last = [float(v) for v in init]
        with torch.no_grad():
            for f in range(start, end):
                t0 = self._timer()
                img = read_image_rgb(seq.frame_path(f))
                x, meta = tr.transform(img)
                t1 = self._timer()
                fallback = False
                with tr._autocast():
                    if f == start:
                        box_xyxy = torch.from_numpy(xywh_to_xyxy(init) * meta["scale_factor"]).to(dev)
                        model.track_init(x[None].to(dev), box_xyxy, meta["img_shape"])
                    else:
                        boxes, sc = model.track_update(x[None].to(dev), meta["img_shape"])
                if sync:
                    torch.cuda.synchronize()
                t2 = self._timer()
                if f == start:
                    xywh, score, mode = last, 1.0, "init"
                elif boxes.shape[0] == 0:
                    xywh, score, mode, fallback = list(last), 0.0, "fallback (previous box)", True
                else:
                    b = (boxes[0].float() / meta["scale_factor"]).tolist()
                    xywh, score, mode = [b[0], b[1], b[2] - b[0], b[3] - b[1]], float(sc[0]), "track"
                last = xywh
                t3 = self._timer()
                yield {"frame": f, "present": True, "box_xyxy": [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]],
                       "score": score, "assoc_score": None, "mode": mode, "detections": None, "fallback": fallback,
                       "timing_ms": {"read_preprocess": (t1 - t0) * 1000, "model": (t2 - t1) * 1000, "postprocess": (t3 - t2) * 1000,
                                     "tracker": 0.0, "total": (t3 - t0) * 1000}}

    def predict(self, seq, dataset_id="antiuav2021", split="val", modality="IR", max_frames=None, **_):
        if self.model is None:
            raise AdapterError("load() first")
        c = self.can_run(self.record, dataset_id, modality)
        if not c.ok:
            raise AdapterError(c.reason)
        t0 = self._timer()
        out = self.tracker.track_sequence(seq, frame_stride=1, max_frames=max_frames)
        dt = (self._timer() - t0) * 1000
        n_proc = int(sum(out["processed"]))
        n = seq.start_frame + n_proc if max_frames else seq.n_frames
        frames = []
        for t in range(n):
            b = out["res"][t]
            ok = isinstance(b, list) and len(b) == 4 and bool(out["processed"][t])
            frames.append({"frame": t, "present": (True if ok else None) if t >= seq.start_frame else None,
                           "box_xyxy": [b[0], b[1], b[0] + b[2], b[1] + b[3]] if ok else None,
                           "score": float(out["scores"][t]) if ok else None, "mode": "track" if ok else None, "detections": None})
        return {"kind": "sot", "adapter": self.id, "model_id": self.record["model_id"], "checkpoint_sha256": self.record["sha256"],
                "checkpoint_path": self.get_metadata()["checkpoint_path"], "dataset_id": dataset_id, "split": split, "modality": modality,
                "sequence": seq.name, "n_frames_total": seq.n_frames, "n_frames_processed": n, "start_frame": int(out["start_frame"]),
                "res_xywh": out["res"][:n], "processed": [bool(x) for x in out["processed"][:n]], "frames": frames,
                "device": self.device, "timing_ms": {"tracker_per_frame": dt / max(n_proc, 1)},
                "score_kind": "GlobalTrack top-1 box score", "tracker": "globaltrack", "params": {"frame_stride": 1},
                "params_source": self.cfg_source, "window": None,
                "preprocessing": {"transform": "TestTransform", "scale": self.cfg["data"]["transforms"]["scale"],
                                  "pad_divisor": self.cfg["data"]["transforms"]["pad_divisor"]},
                "warnings": list(c.warnings)}
