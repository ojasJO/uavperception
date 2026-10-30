"""Shared studio helpers: cached registry, labels, colours, drawing, the protocol banner. No page logic here."""
import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from aerolab.paths import REGISTRY  # noqa: E402
from aerolab.registry import Registry, RegistryMissing  # noqa: E402

DIR_LABEL = {"aerotrack": "A · AeroTrack system", "antiuav_repro": "B · Anti-UAV paper reproduction", "shared": "Shared / external"}
DIR_SHORT = {"aerotrack": "A", "antiuav_repro": "B", "shared": "S"}
DIR_COLOR = {"aerotrack": "#2b6cb0", "antiuav_repro": "#b7791f", "shared": "#4a5568"}
STATUS_HELP = {
    "registered": "used by a pre-registered pass (the identity of a reported result)",
    "validated": "trained and evaluated under a recorded protocol", "experimental": "trained; no controlled evidence or not the evaluated checkpoint",
    "legacy": "superseded series, kept as evidence", "retired": "evaluated and rejected by a pre-registered rule",
    "invalid": "known-invalid result (see role)", "smoke": "plumbing run, not a model", "external": "third-party weights",
}
METRIC_HELP = {
    "mSA": "mean state accuracy over sequences (%): IoU on visible frames, 1/0 for a correct/incorrect 'absent' (Anti-UAV Eq. 1)",
    "precision_20px": "% of visible frames whose centre error is at most 20 px", "success_auc": "area under the success (IoU threshold) curve, %",
    "ade_px": "average displacement error of forecast centres (pixels; lower is better)", "nll": "Gaussian negative log-likelihood (lower is better)",
    "drone_AP50": "drone-class average precision at IoU 0.5", "false_alarm_rate": "share of target-absent frames on which a box was reported",
}


def registry_mtime() -> float:
    p = REGISTRY / "models.json"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="loading the registry…")
def _registry(mtime: float):
    return Registry()


def get_registry() -> Registry:
    try:
        return _registry(registry_mtime())
    except RegistryMissing as e:
        st.error(str(e))
        st.stop()


def badge(direction: str) -> str:
    return f":{'blue' if direction == 'aerotrack' else 'orange' if direction == 'antiuav_repro' else 'gray'}-badge[{DIR_LABEL.get(direction, direction)}]"


def protocol_banner():
    st.caption("TEST splits are locked: every TEST split in this repository has been exposed by registered passes. "
               "The studio shows recorded TEST results read-only and runs inference / evaluations on TRAIN / VAL only.")


def fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return f"[{v[0]:.{nd}f}, {v[1]:.{nd}f}]"
    return str(v)


def result_context(e: dict) -> str:
    """the minimum context a number needs (task brief §7.1)"""
    return (f"{DIR_SHORT.get(e.get('direction'), '?')} · {e.get('dataset_id') or '—'} / {e.get('split') or '—'}"
            f"{' / ' + e['modality'] if e.get('modality') else ''} · protocol `{e.get('protocol_id') or '—'}` · {e.get('kind')}"
            f" · {e.get('provenance')}{' · TEST pass' if e.get('test_pass') else ''}")


def draw_frame(img_path, gt_xywh=None, pred_xyxy=None, dets=None, label=None, scale=1.0, gt_absent=False):
    """GT green, prediction orange (thick), other detections thin cyan. Returns a PIL image."""
    from PIL import Image, ImageDraw
    im = Image.open(img_path).convert("RGB")
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    d = ImageDraw.Draw(im)
    s = scale
    w = max(2, int(round(im.width / 320)))
    for b in dets or []:
        d.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=(0, 200, 255), width=1)
    if gt_xywh is not None:
        x, y, bw, bh = gt_xywh
        d.rectangle([x * s - 2, y * s - 2, (x + bw) * s + 2, (y + bh) * s + 2], outline=(0, 255, 0), width=w)
    if pred_xyxy is not None:
        d.rectangle([v * s for v in pred_xyxy], outline=(255, 128, 0), width=w)
    txt = label or ""
    if gt_absent:
        txt += "  [GT: target absent]"
    if txt:
        d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
        d.text((4, 3), txt, fill=(255, 255, 255))
    return im


def crop_zoom(img, box_xyxy, size=96, zoom=3):
    if box_xyxy is None:
        return None
    cx, cy = (box_xyxy[0] + box_xyxy[2]) / 2, (box_xyxy[1] + box_xyxy[3]) / 2
    h = size / 2
    c = img.crop((int(cx - h), int(cy - h), int(cx + h), int(cy + h)))
    return c.resize((size * zoom, size * zoom))


def model_label(m: dict) -> str:
    al = f" — {', '.join(m['aliases'])}" if m.get("aliases") else ""
    return f"{m['run_id']} [{m['checkpoint_tag']}]{al} · {m['status']}"
