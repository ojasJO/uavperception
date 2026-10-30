"""Core logic of the VALIDATION-ONLY demo — no UI code here so it can be smoke-tested without Streamlit.

Everything is reused from the reproduction package: sequences (`data.antiuav.load_sequences`, split fixed to 'val',
purpose 'tune' — the loader refuses 'test' for this purpose), saved predictions written by `scripts/evaluate.py`,
metric functions (`eval.metrics.sequence_metrics`, `iou_xywh`, `center_error`), and the tracker
(`engine.track.GlobalTrackTracker`) for optional live tracking. Nothing here writes into a run directory: live
results and rendered videos go to experiments/anti_uav/demo/cache/ only.

Coordinate contract (verified in test_demo.py): label index t <-> frame file {t+1:06d}.jpg; GT `gt_rect[t]` and
prediction `res[t]` are both [x, y, w, h] in ORIGINAL image pixels (the tracker divides its boxes by the rescale
factor before saving), so boxes are drawn on the original frame without any conversion.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

DEMO_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = DEMO_DIR.parent
SRC = EXPERIMENT_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anti_uav_repro.data.antiuav import Sequence, load_sequences  # noqa: E402
from anti_uav_repro.eval.metrics import center_error, iou_xywh, is_box, sequence_metrics  # noqa: E402
from anti_uav_repro.paths import REPORTS_DIR, RUNS_DIR  # noqa: E402

SPLIT = "val"           # the demo is VALIDATION ONLY; never change to 'test'
PURPOSE = "tune"        # load_sequences raises TestSplitAccessError for split='test' with this purpose
MODALITY = "IR"
CACHE_DIR = DEMO_DIR / "cache"
GT_COLOR = (0, 255, 0)          # green  = ground truth
PRED_COLOR = (255, 128, 0)      # orange = prediction
OUTLINE = (0, 0, 0)
LOW_IOU = 0.5


@dataclass
class RunInfo:
    name: str
    run_dir: Path
    checkpoint: Path
    seed: int | None
    alpha: float | None
    phase: str | None
    method: str
    checkpoint_sha256: str | None
    val_mSA_percent: float | None
    val_precision: float | None
    val_success: float | None
    predictions_dir: Path | None
    eval_checkpoint_sha256: str | None
    per_sequence_SA: dict

    @property
    def label(self) -> str:
        return f"{self.name}  [{self.method}, seed {self.seed}]"

    @property
    def short(self) -> str:
        return f"{'DFSC' if self.alpha else 'normal'} s{self.seed} ({self.name})"

    @property
    def saved_predictions_match_checkpoint(self) -> bool | None:
        if self.predictions_dir is None or not self.eval_checkpoint_sha256 or not self.checkpoint_sha256:
            return None
        return self.eval_checkpoint_sha256 == self.checkpoint_sha256


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def discover_runs(runs_dir: Path = RUNS_DIR) -> list[RunInfo]:
    """Every run directory with a final checkpoint (smoke runs excluded)."""
    out = []
    for d in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if not d.is_dir() or d.name.startswith("_"):
            continue
        ck = d / "weights" / "last.pt"
        if not ck.exists():
            continue
        cfg = yaml.safe_load((d / "config.effective.yaml").read_text(encoding="utf-8")) if (d / "config.effective.yaml").exists() else {}
        meta = _load_json(d / "run_meta.json") or {}
        alpha = (cfg.get("dfsc") or {}).get("alpha")
        method = "DFSC (α=%g)" % alpha if alpha else "normal (α=0, GlobalTrack fine-tune)"
        ev = d / f"eval_{SPLIT}_{MODALITY}_last"
        m = _load_json(ev / "metrics.json")
        em = _load_json(ev / "eval_meta.json") or {}
        agg = (m or {}).get("aggregate", {})
        out.append(RunInfo(
            name=d.name, run_dir=d, checkpoint=ck, seed=cfg.get("seed"), alpha=alpha, phase=cfg.get("phase"), method=method,
            checkpoint_sha256=meta.get("final_checkpoint_sha256"),
            val_mSA_percent=agg.get("mSA_percent"), val_precision=agg.get("precision_20px"), val_success=agg.get("success_auc"),
            predictions_dir=(ev / "predictions") if (ev / "predictions").exists() else None,
            eval_checkpoint_sha256=em.get("checkpoint_sha256") or (m or {}).get("meta", {}).get("checkpoint_sha256"),
            per_sequence_SA={k: v.get("SA") for k, v in ((m or {}).get("per_sequence") or {}).items()},
        ))
    return out


def load_val_sequences(cfg_data: dict | None = None) -> list[Sequence]:
    """The 67 IR VAL sequences. Data paths come from any run config (all runs share them) or the defaults."""
    root = (cfg_data or {}).get("root", "data/anti_uav")
    frames_root = (cfg_data or {}).get("frames_root", "b_pipeline/data_unified/anti_uav/images")
    return load_sequences(root, frames_root, SPLIT, MODALITY, purpose=PURPOSE)


def data_cfg_of(run: RunInfo) -> dict:
    cfg = yaml.safe_load((run.run_dir / "config.effective.yaml").read_text(encoding="utf-8"))
    return cfg.get("data", {})


def paper_reference() -> dict:
    """Paper Table V (val, IR All) for the two methods — transcribed values, labelled as the paper's."""
    pr = _load_json(REPORTS_DIR / "paper_reported_results.json")
    if not pr:
        return {}
    col = pr["attribute_columns"].index("ALL_IR")
    rows = pr["table_V_protocol_II"]["val"]
    return {"normal": rows["normal"][col], "DFSC": rows["DFSC"][col], "source": "paper Table V, IR val, All"}


# ------------------------------------------------------------------------------------------ predictions
def load_saved_prediction(run: RunInfo, seq_name: str) -> dict | None:
    if run.predictions_dir is None:
        return None
    return _load_json(run.predictions_dir / f"{seq_name}.json")


def cached_live_prediction(run: RunInfo, seq_name: str) -> dict | None:
    return _load_json(CACHE_DIR / run.name / f"{seq_name}.json")


def live_track(run: RunInfo, seq: Sequence, max_frames: int | None = None, device: str | None = None) -> dict:
    """Run the reproduction's tracker on one VAL sequence (optionally only the first `max_frames` tracked frames).
    Uses the checkpoint read-only; writes the result to demo/cache/<run>/<seq>.json (never into the run directory)."""
    import torch

    from anti_uav_repro.data.transforms import TestTransform
    from anti_uav_repro.engine.track import GlobalTrackTracker
    from anti_uav_repro.models.qg_rcnn import QGRCNN

    cfg = yaml.safe_load((run.run_dir / "config.effective.yaml").read_text(encoding="utf-8"))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(run.checkpoint, map_location="cpu", weights_only=False)
    model = QGRCNN({**cfg["model"], "init": "none"})
    model.load_state_dict(ck["model"], strict=True)
    model.to(dev).eval()
    tracker = GlobalTrackTracker(model, dev, TestTransform(tuple(cfg["data"]["transforms"]["scale"]), int(cfg["data"]["transforms"]["pad_divisor"])))
    out = tracker.track_sequence(seq, frame_stride=1, max_frames=max_frames)
    out["source"] = "live (demo)"
    out["checkpoint_sha256"] = run.checkpoint_sha256
    out["max_frames"] = max_frames
    dst = CACHE_DIR / run.name / f"{seq.name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out), encoding="utf-8")
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------ metrics
def frame_metrics(pred: dict, seq: Sequence, t: int) -> dict:
    """Per-frame numbers computed with the same helpers as eval/metrics.py."""
    exist = bool(seq.exist[t])
    gt = seq.box_xywh(t)
    p = pred["res"][t] if t < len(pred["res"]) else [0]
    processed = bool(pred["processed"][t]) if t < len(pred.get("processed", [])) else False
    score = float(pred["scores"][t]) if t < len(pred.get("scores", [])) else None
    box = [float(v) for v in p] if is_box(p) else None
    iou = ce = None
    if exist and gt is not None and box is not None:
        iou = iou_xywh(box, gt)
        ce = center_error(box, gt)
    if not processed and t != pred["start_frame"]:
        status = "not tracked (before the start frame)" if t < pred["start_frame"] else "not tracked (beyond live range)"
    elif exist:
        status = "init frame (tracker given the GT box)" if t == pred["start_frame"] else ("tracked" if box is not None else "no output")
    else:
        status = "GT absent — " + ("tracker still outputs a box (scores 0)" if box is not None else "no box (scores 1)")
    return {"t": t, "exist": exist, "gt": None if gt is None else [float(v) for v in gt], "pred": box, "score": score,
            "iou": iou, "center_error_px": ce, "processed": processed, "status": status}


def per_frame_table(pred: dict, seq: Sequence) -> list[dict]:
    """IoU / CE / score for every frame (cheap: ~1 ms per 1000 frames) — feeds the low-IoU picker and the IoU strip."""
    rows = []
    for t in range(seq.n_frames):
        fm = frame_metrics(pred, seq, t)
        rows.append({"t": t, "exist": fm["exist"], "iou": fm["iou"], "ce": fm["center_error_px"], "score": fm["score"], "processed": fm["processed"]})
    return rows


def low_iou_frames(table: list[dict], thr: float = LOW_IOU) -> list[dict]:
    """Frames where the target exists, was tracked, and IoU < thr — the failures, never hidden."""
    return [r for r in table if r["exist"] and r["processed"] and r["iou"] is not None and r["iou"] < thr]


def sequence_summary(pred: dict, seq: Sequence) -> dict:
    """Sequence-level SA with the same convention as evaluate.py: frames from the start frame onward; for a live
    partial run only the processed frames (marked as partial)."""
    n = seq.n_frames
    start = int(pred["start_frame"])
    processed = np.asarray(pred.get("processed", [True] * n), dtype=bool)
    full = bool(processed[start:].all())
    mask = (np.arange(n) >= start) if full else processed
    m = sequence_metrics(pred["res"], seq.exist, seq.gt_rect, mask=mask)
    return {"SA_percent": 100 * m["SA"], "precision_20px": m["precision_20px"], "success_auc": m["success_auc"],
            "n_frames_scored": int(mask.sum()), "n_exist_scored": m["n_exist"], "start_frame": start,
            "partial": not full, "scoring": "frames from start frame (paper protocol)" if full else "processed frames only (PARTIAL live run)"}


# ------------------------------------------------------------------------------------------ rendering
def _font(size: int = 16):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rect(draw: ImageDraw.ImageDraw, b, color, width, pad=0):
    x, y, w, h = b
    draw.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], outline=color, width=width)


def draw_boxes(im: Image.Image, fm: dict, legend: bool = True) -> Image.Image:
    """GT = thick green, prediction = orange. Black halos are drawn FIRST for both boxes so that neither box's halo
    can hide the other when they (correctly) overlap; the thinner orange box sits inside the green one."""
    draw = ImageDraw.Draw(im)
    f = _font(15)
    if fm["gt"] is not None:
        _rect(draw, fm["gt"], OUTLINE, 7, pad=2)
    if fm["pred"] is not None:
        _rect(draw, fm["pred"], OUTLINE, 5, pad=1)
    if fm["gt"] is not None:
        _rect(draw, fm["gt"], GT_COLOR, 4, pad=1)
        x, y, w, h = fm["gt"]
        draw.text((x, max(2, y - 22)), "GT", fill=GT_COLOR, font=f, stroke_width=2, stroke_fill=OUTLINE)
    if fm["pred"] is not None:
        _rect(draw, fm["pred"], PRED_COLOR, 2, pad=0)
        x, y, w, h = fm["pred"]
        lab = f"PRED {fm['score']:.2f}" if fm["score"] is not None else "PRED"
        draw.text((x, y + h + 6), lab, fill=PRED_COLOR, font=f, stroke_width=2, stroke_fill=OUTLINE)
    if legend:
        draw.text((8, im.height - 24), "GREEN = ground truth   ORANGE = prediction", fill=(255, 255, 255), font=f, stroke_width=2, stroke_fill=OUTLINE)
    return im


def render_frame(seq: Sequence, t: int, fm: dict, legend: bool = True) -> Image.Image:
    """Original frame with thick boxes; nothing else drawn over the image except a legend line at the bottom."""
    im = Image.open(seq.frame_path(t)).convert("RGB")
    return draw_boxes(im, fm, legend=legend)


def render_inset(seq: Sequence, t: int, fm: dict, crop: int = 112, scale: int = 3) -> Image.Image | None:
    """3x zoom around the target (GT box centre, else prediction centre) so a 30-px drone is obvious."""
    ref = fm["gt"] or fm["pred"]
    if ref is None:
        return None
    im = Image.open(seq.frame_path(t)).convert("RGB")
    cx, cy = ref[0] + ref[2] / 2, ref[1] + ref[3] / 2
    half = max(crop // 2, int(max(ref[2], ref[3]) * 0.9))
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(im.width, cx + half)), int(min(im.height, cy + half))
    sub = im.crop((x0, y0, x1, y1)).resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.NEAREST)
    fm2 = dict(fm)
    for k in ("gt", "pred"):
        if fm[k] is not None:
            x, y, w, h = fm[k]
            fm2[k] = [(x - x0) * scale, (y - y0) * scale, w * scale, h * scale]
    return draw_boxes(sub, fm2, legend=False)


def render_iou_strip(table: list[dict], t: int, width: int = 640, height: int = 70) -> Image.Image:
    """IoU over the whole sequence with a red marker at the current frame (bad frames stay visible)."""
    im = Image.new("RGB", (width, height), (25, 25, 25))
    d = ImageDraw.Draw(im)
    n = max(len(table), 1)
    pts = []
    for r in table:
        x = int(r["t"] / max(n - 1, 1) * (width - 1))
        if r["iou"] is None:
            continue
        y = int((height - 12) - r["iou"] * (height - 20))
        pts.append((x, y))
    if len(pts) > 1:
        d.line(pts, fill=(80, 200, 80), width=1)
    ythr = int((height - 12) - LOW_IOU * (height - 20))
    d.line([(0, ythr), (width, ythr)], fill=(120, 120, 120), width=1)
    xt = int(t / max(n - 1, 1) * (width - 1))
    d.line([(xt, 0), (xt, height)], fill=(255, 60, 60), width=2)
    d.text((4, 2), "IoU over the sequence (grey line = 0.5, red = current frame)", fill=(200, 200, 200), font=_font(12))
    return im


def hud_text(fm: dict, n: int, sa_percent: float | None) -> str:
    parts = [f"frame {fm['t']}/{n-1}", fm["status"]]
    if fm["iou"] is not None:
        parts += [f"IoU {fm['iou']:.2f}", f"CE {fm['center_error_px']:.0f}px"]
    if fm["score"] is not None and fm["pred"] is not None:
        parts.append(f"conf {fm['score']:.2f}")
    if sa_percent is not None:
        parts.append(f"seq SA {sa_percent:.1f}%")
    return "  |  ".join(parts)


def render_video(runs: list[RunInfo], seq: Sequence, preds: list[dict], out_path: Path, fps: int = 25,
                 max_frames: int | None = None, start: int | None = None) -> Path:
    """Annotated H.264 MP4 (libx264, yuv420p, faststart) of the sequence from the start frame: one panel per run,
    boxes + HUD baked in. Cached by the caller under demo/cache/videos/."""
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    n = seq.n_frames
    s0 = int(preds[0]["start_frame"]) if start is None else int(start)
    frames = list(range(s0, n))
    if max_frames is not None:
        frames = frames[:max_frames]
    tables = [per_frame_table(p, seq) for p in preds]
    sas = [sequence_summary(p, seq)["SA_percent"] for p in preds]
    im0 = Image.open(seq.frame_path(s0))
    W, H = im0.size
    hud_h = 26
    panel_h = H + hud_h
    width, height = W * len(runs), panel_h
    width += width % 2
    height += height % 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "21",
           "-movflags", "+faststart", str(out_path)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    f = _font(15)
    for t in frames:
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        for k, (run, p) in enumerate(zip(runs, preds)):
            fm = frame_metrics(p, seq, t)
            panel = render_frame(seq, t, fm, legend=(k == 0))
            canvas.paste(panel, (k * W, 0))
            d = ImageDraw.Draw(canvas)
            d.text((k * W + 6, H + 5), f"{run.short}: " + hud_text(fm, n, sas[k]), fill=(240, 240, 240), font=f)
        pr.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
    pr.stdin.close()
    pr.wait()
    if pr.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {pr.returncode}")
    return out_path


def video_path(runs: list[RunInfo], seq: Sequence, fps: int) -> Path:
    tag = "__".join(r.name for r in runs)
    return CACHE_DIR / "videos" / f"{tag}__{seq.name}__{fps}fps.mp4"
