#!/usr/bin/env python
"""AeroTrack streaming runtime: video or frame folder -> detector (run's own temporal window) -> NMS -> single-target state
estimator -> forecast with uncertainty -> annotated video + per-frame JSON + latency summary. One command:

    venv\\Scripts\\python.exe experiments\\deployment\\scripts\\aerotrack_stream.py --source <video.mp4 | frame folder> ^
        --weights results\\runs\\gen_temporal_fp_g4_full_e20\\weights\\last.pt [--backend trt_fp16] [--ste-params params.json] ^
        [--gt <IR_label.json | infrared.json>] [--out out.mp4] [--max-frames N]

* The temporal window is read from the run (run_meta.json): `causal` uses only past frames (output for frame t at time t),
  `symmetric` needs g future frames (output for frame t is emitted g frames later -- the structural look-ahead is reported),
  `ttt` is single-frame. Frames are letterboxed exactly as in training.
* Backends: torch_fp32 / torch_bf16 / torch_fp16 / ort_cuda / trt_fp32 / trt_fp16 (experiments/deployment/scripts/backends.py);
  an unavailable backend falls back to torch_fp32 with a warning; no CUDA -> CPU with a warning (slow, still correct).
* STE parameters: defaults of ste.STEParams unless --ste-params gives a JSON object (e.g. the cross-fitted choice from
  experiments/systems/reports/track_*.json).
* --tracker fste (Track C): the forecaster-driven STE (experiments/aerotrack_temporal/scripts/tc_ste.py) with --forecaster <spec.json>:
  {"kind": "classical", "name": "oudcv", "params": {...}, "var": [[vx, vy] x 24]} or {"kind": "learned", "onnx": "...", "abs_pos": true,
  "fallback": {classical spec}}; --fste-params gives FSTEParams fields. The tracker stage timing then includes the forecaster.
* Overlay: ground truth (if --gt), all drone detections >= det_thr (thin, with score), the STE output (thick; colour = mode),
  the track's trail, the 12-step forecast with its 95 % ellipse, and a HUD with per-stage latency, FPS and the look-ahead.
* Outputs next to --out: the video, <stem>.frames.jsonl (one record per frame: detections, state, forecast, timings) and
  <stem>.summary.json (p50 / p95 / p99 per stage and end-to-end, frames, mode counts, environment).
"""
import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (HERE, REPO / "experiments" / "systems" / "scripts", REPO / "src/model", REPO / "src/pipeline"):
    sys.path.insert(0, str(p))

MODE_COLOR = {"track": (0, 220, 0), "init": (0, 220, 0), "reacquired": (0, 200, 255), "coast": (0, 140, 255), "absent": (128, 128, 128)}


class FrameSource:
    def __init__(self, src):
        self.path = Path(src)
        if self.path.is_dir():
            self.files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
            if not self.files:
                raise SystemExit(f"[ERROR] no images in {src}")
            self.cap, self.fps = None, 25.0
        else:
            self.cap = cv2.VideoCapture(str(src))
            if not self.cap.isOpened():
                raise SystemExit(f"[ERROR] cannot open video {src}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.i = 0
        self.unreadable = 0

    def __iter__(self):
        while True:
            if self.cap is None:
                if self.i >= len(self.files):
                    return
                img = cv2.imread(str(self.files[self.i]), cv2.IMREAD_COLOR)
                self.i += 1
                if img is None:
                    self.unreadable += 1
                    continue
            else:
                ok, img = self.cap.read()
                if not ok:
                    return
            yield img


def load_gt(path):
    if not path:
        return None
    a = json.loads(Path(path).read_text(encoding="utf-8"))
    rect = a.get("gt_rect", a.get("gt"))
    return {"exist": a["exist"], "rect": rect}


def draw(frame, t, dets, out_box, mode, score, trail, fc_mean, fc_cov, gt, hud, det_thr):
    img = frame.copy()
    if gt is not None and t < len(gt["exist"]) and gt["exist"][t] and len(gt["rect"][t]) == 4:
        x, y, w, h = gt["rect"][t]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1)
    b, s, c = dets
    for k in range(len(s)):
        if c[k] == 0 and s[k] >= det_thr:
            x1, y1, x2, y2 = b[k].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 0), 1)
            cv2.putText(img, f"{s[k]:.2f}", (x1, max(y1 - 3, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1, cv2.LINE_AA)
    if len(trail) > 1:
        cv2.polylines(img, [np.array(trail, np.int32).reshape(-1, 1, 2)], False, (255, 0, 255), 1, cv2.LINE_AA)
    if fc_mean is not None and len(fc_mean):
        cv2.polylines(img, [fc_mean.astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 1, cv2.LINE_AA)
        vals, vecs = np.linalg.eigh(fc_cov[-1])
        axes = tuple(int(max(1, 2.4477 * np.sqrt(max(v, 1e-6)))) for v in vals[::-1])     # 95 % ellipse, 2 dof
        ang = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
        cv2.ellipse(img, (int(fc_mean[-1, 0]), int(fc_mean[-1, 1])), axes, ang, 0, 360, (0, 255, 255), 1, cv2.LINE_AA)
    if out_box is not None:
        x1, y1, x2, y2 = np.asarray(out_box).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), MODE_COLOR[mode], 2)
        cv2.putText(img, f"UAV-1 {mode} {score:.2f}", (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MODE_COLOR[mode], 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(img, hud[0], (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, hud[1], (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def pct(a):
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(len(a)), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())} if len(a) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--backend", default="torch_fp32")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--ste-params", default=None, help="JSON file with STEParams fields")
    ap.add_argument("--gt", default=None, help="annotation JSON (Anti-UAV infrared.json/visible.json or CST IR_label.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--forecast", type=int, default=12)
    ap.add_argument("--tracker", choices=("ste", "fste"), default="ste")
    ap.add_argument("--forecaster", default=None, help="forecaster spec JSON (required for --tracker fste)")
    ap.add_argument("--fste-params", default=None, help="JSON file with FSTEParams fields")
    ap.add_argument("--no-video", action="store_true", help="skip drawing / video writing (timing runs)")
    a = ap.parse_args()

    import torch
    from ultralytics.utils.nms import non_max_suppression

    import backends as B
    import ste as STE
    from stacked_dataset import letterbox_image, letterbox_params

    w = Path(a.weights)
    if not w.exists():
        raise SystemExit(f"[ERROR] weights not found: {w}")
    run = w.parent.parent.name
    meta_p = w.parent.parent / "run_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    gap = int(meta.get("gap", meta.get("temporal_gap", 1)))
    window = meta.get("window") or ("ttt" if gap == 0 else "symmetric")
    look = gap if window == "symmetric" else 0
    params = STE.STEParams(**json.loads(Path(a.ste_params).read_text(encoding="utf-8"))) if a.ste_params else STE.STEParams()
    cuda = torch.cuda.is_available()
    if not cuda:
        print("[WARN] CUDA not available: running on CPU (correct but slow)")
    onnx = Path(a.onnx) if a.onnx else REPO / "results/artifacts" / "deploy" / run / f"{run}_b1_op18.onnx"
    t_start = time.perf_counter()
    backend_name = a.backend
    try:
        if not cuda and not a.backend.startswith("torch"):
            raise B.BackendUnavailable("no CUDA")
        be = B.make(a.backend, w, onnx) if cuda else None
    except B.BackendUnavailable as e:
        print(f"[WARN] backend {a.backend} unavailable ({e}); falling back to torch_fp32")
        backend_name, be = "torch_fp32", None
    if be is None:
        be = B.TorchBackend(w, "fp32", device="cuda:0" if cuda else "cpu")
    dev = torch.device("cuda:0" if cuda else "cpu")
    src = FrameSource(a.source)
    gt = load_gt(a.gt)
    out = Path(a.out) if a.out else REPO / "results/runs" / "deploy_stream" / f"{run}__{Path(a.source).stem}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, jf = None, open(out.with_suffix(".frames.jsonl"), "w", encoding="utf-8")
    if a.tracker == "fste":
        sys.path.insert(0, str(REPO / "experiments" / "aerotrack_temporal" / "scripts"))
        import tc_ste as FS
        spec = json.loads(Path(a.forecaster).read_text(encoding="utf-8"))

        def classical(sp):
            return FS.ClassicalForecaster(sp["name"], sp.get("params"), np.asarray(sp["var"], dtype=np.float64))
        fc = (classical(spec) if spec["kind"] == "classical" else
              FS.LearnedForecaster(REPO / spec["onnx"], spec.get("abs_pos", True), classical(spec["fallback"])))
        params = FS.FSTEParams(**json.loads(Path(a.fste_params).read_text(encoding="utf-8"))) if a.fste_params else FS.FSTEParams()
        trk = None                                               # built at the first frame (needs the image size)
    else:
        trk = STE.STE(params)
        trk.reset(None)
    raw, lbs = deque(), deque()
    span = (2 * gap if window == "causal" else gap) + look + 1
    geom = None
    timings = {k: [] for k in ("read_preprocess", "window", "infer", "nms", "tracker", "total")}
    modes, trail = Counter(), deque(maxlen=60)
    startup_s = None
    n_in = n_out = 0

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def process(c_idx, t_read):
        """Run detector + STE for the frame at buffer index c_idx (its own frame number is n_out)."""
        nonlocal n_out, writer, startup_s, trk
        t0 = time.perf_counter()
        n = len(lbs) - 1
        if window == "causal":
            trip = [lbs[max(c_idx - gap, 0)], lbs[c_idx], lbs[max(c_idx - 2 * gap, 0)]]
        elif window == "symmetric":
            trip = [lbs[max(c_idx - gap, 0)], lbs[c_idx], lbs[min(c_idx + gap, n)]]
        else:
            trip = [lbs[c_idx]] * 3
        x = torch.cat(trip, 0).unsqueeze(0)
        sync()
        t1 = time.perf_counter()
        y = be(x)
        sync()
        t2 = time.perf_counter()
        det = non_max_suppression(y.detach().float().cpu(), max(min(params.det_thr, 0.25), 0.01), 0.7, max_det=10)[0].numpy()
        r, dw, dh, _, _ = geom
        H, W = raw[c_idx].shape[:2]
        if len(det):
            det[:, [0, 2]] = ((det[:, [0, 2]] - dw) / r).clip(0, W)
            det[:, [1, 3]] = ((det[:, [1, 3]] - dh) / r).clip(0, H)
        boxes, scores, classes = det[:, :4].astype(np.float64), det[:, 4], det[:, 5].astype(np.int64)
        if trk is None:                                          # FSTE needs the frame size for its forecaster inputs
            trk = FS.FSTE(params, fc, W, H)
            trk.reset(None)
        t3 = time.perf_counter()
        box, mode, score = trk.step(boxes, scores, classes)
        score = score if isinstance(score, (int, float)) or score is None else None
        fc_mean = fc_cov = None
        if a.tracker == "ste" and trk.kf is not None:
            fc_mean, fc_cov = trk.kf.forecast(a.forecast)
        elif a.tracker == "fste" and trk.active and trk.F is not None:
            k = trk.t - trk.t_anchor                                 # the anchor's forecast, re-based on the current frame
            m, v = trk.F
            fc_mean = m[k:k + a.forecast]
            fc_cov = np.stack([np.diag(x) for x in v[k:k + a.forecast]]) if len(fc_mean) else None
            fc_mean = fc_mean if len(fc_mean) else None
        t4 = time.perf_counter()
        if box is not None:
            trail.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        modes[mode] += 1
        if startup_s is None:
            startup_s = time.perf_counter() - t_start
        tim = {"read_preprocess": t_read, "window": (t1 - t0) * 1e3, "infer": (t2 - t1) * 1e3, "nms": (t3 - t2) * 1e3,
               "tracker": (t4 - t3) * 1e3}
        tim["total"] = sum(tim.values())
        if n_out >= 10:                                         # warm-up frames excluded from the statistics
            for k, v in tim.items():
                timings[k].append(v)
        fps = 1000.0 / np.mean(timings["total"][-30:]) if timings["total"] else 0.0
        hud = (f"{run} | {backend_name} | window {window} g={gap} | look-ahead {look} fr ({1000 * look / src.fps:.0f} ms)",
               f"frame {n_out} | infer {tim['infer']:.1f} ms | total {tim['total']:.1f} ms | {fps:.0f} FPS | {mode}")
        if not a.no_video:
            img = draw(raw[c_idx], n_out, (boxes, scores, classes), box, mode, score, list(trail), fc_mean, fc_cov, gt, hud, params.det_thr)
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (img.shape[1], img.shape[0]))
            writer.write(img)
        rec = {"frame": n_out, "mode": mode, "score": score, "box_xyxy": None if box is None else [float(v) for v in box],
               "detections": [[*map(float, boxes[k]), float(scores[k]), int(classes[k])] for k in range(len(scores))],
               "forecast_centers": None if fc_mean is None else fc_mean.round(2).tolist(), "timings_ms": {k: round(v, 3) for k, v in tim.items()}}
        jf.write(json.dumps(rec) + "\n")
        n_out += 1

    for img in src:
        if a.max_frames and n_in >= a.max_frames:
            break
        t0 = time.perf_counter()
        if geom is None:
            geom = letterbox_params(img.shape[0], img.shape[1], 640, 640)
        r, dw, dh, nw, nh = geom
        lb = letterbox_image(img, 640, 640, r, dw, dh, nw, nh)
        x = torch.from_numpy(np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB))).to(dev).permute(2, 0, 1).float().div_(255.0)
        sync()
        t_read = (time.perf_counter() - t0) * 1e3
        raw.append(img)
        lbs.append(x)
        while len(lbs) > span:
            raw.popleft()
            lbs.popleft()
        n_in += 1
        c = len(lbs) - 1 - look                                  # the frame that can be processed now
        if c >= 0 and n_out < n_in - look:
            process(c, t_read)
    while n_out < n_in:                                           # flush: the last `look` frames (future clamps to the newest)
        c = len(lbs) - 1 - (n_in - 1 - n_out)
        process(c, 0.0)
    if writer is not None:
        writer.release()
    jf.close()
    summary = {"run": run, "weights": str(w), "backend": backend_name, "requested_backend": a.backend, "window": window, "gap": gap,
               "look_ahead_frames": look, "look_ahead_ms": 1000.0 * look / src.fps, "source": str(a.source), "source_fps": src.fps,
               "frames_in": n_in, "frames_out": n_out, "unreadable_frames": src.unreadable, "startup_s_to_first_output": startup_s,
               "tracker": a.tracker, "forecaster": a.forecaster, "tracker_params": params.as_dict(), "modes": dict(modes),
               "n_reacquisitions": getattr(trk, "n_reacq", modes.get("reacquired", 0)),
               "latency": {k: pct(v) for k, v in timings.items()}, "cuda": cuda,
               "gpu": torch.cuda.get_device_name(0) if cuda else None, "torch": torch.__version__, "video": str(out)}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    L = summary["latency"]["total"]
    print(f"[done] {n_out} frames -> {out}")
    if L:
        print(f"       per-frame compute p50 {L['p50_ms']:.1f} / p95 {L['p95_ms']:.1f} / p99 {L['p99_ms']:.1f} ms "
              f"(+ structural look-ahead {summary['look_ahead_ms']:.0f} ms); modes {dict(modes)}")


if __name__ == "__main__":
    main()
