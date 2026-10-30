"""Flight Deck UI components: colours, mode banners, multi-model frame rendering (side-by-side / overlay / difference), the
synchronised player and timeline charts. Pages compose these; no inference happens here."""
import json
import time

import numpy as np
import pandas as pd
import streamlit as st
from common import REPO, get_registry  # noqa: F401  (REPO import keeps sys.path set up)
from PIL import Image, ImageDraw, ImageFont

from aerolab import analysis, data, recorded
from aerolab import flightdeck as FD
from aerolab.categories import CATEGORIES, ORDER
from aerolab.sequences import FPS

PALETTE = ["#E69F00", "#56B4E9", "#CC79A7", "#D55E00", "#0072B2", "#F0E442"]   # Okabe-Ito without green (green = ground truth)
GT_COLOR = "#22DD44"
STATUS_COLOR = {"tracked": "#2f9e44", "poor": "#f08c00", "wrong_object": "#c92a2a", "miss": "#862e9c", "false_alarm": "#e8590c",
                "correct_absent": "#adb5bd", "init": "#1c7ed6", "not_processed": "#f1f3f5"}
STATUS_TEXT = {"tracked": "tracked (IoU ≥ 0.5)", "poor": "poor overlap", "wrong_object": "wrong object (IoU 0)", "miss": "missed (no box)",
               "false_alarm": "false alarm (target absent)", "correct_absent": "correctly absent", "init": "init (GT box)", "not_processed": "not processed"}
MODE = {"live": ("LIVE INFERENCE", "The selected models are running on this machine now; every box below is produced by the loaded checkpoint."),
        "saved": ("SAVED RUN — replay", "Predictions saved by a Flight Deck run on this machine; replayed from disk without running any model."),
        "recorded": ("RECORDED PREDICTIONS — replay", "Predictions recorded by earlier registered evaluations or released by paper authors; "
                                                         "read-only, no model is loaded and nothing here is a new result.")}


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mode_banner(mode, extra=""):
    title, text = MODE[mode]
    {"live": st.error, "saved": st.info, "recorded": st.warning}[mode](f"**{title}** — {text}{(' ' + extra) if extra else ''}")


def _font(size=14):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def crop_window(seq, t, boxes, zoom, W, H):
    """a square crop centred on the GT box (or the first predicted box) whose size is the image width / zoom; same for every panel"""
    if zoom <= 1:
        return None
    g = seq.box_xywh(t)
    if g is not None:
        cx, cy = g[0] + g[2] / 2, g[1] + g[3] / 2
    else:
        b = next((b for b in boxes if b is not None), None)
        if b is None:
            return None
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    side = max(48, int(W / zoom))
    x0 = int(min(max(cx - side / 2, 0), max(W - side, 0)))
    y0 = int(min(max(cy - side / 2, 0), max(H - side, 0)))
    return (x0, y0, x0 + side, y0 + min(side, H))


def render(seq, t, models, show_gt=True, zoom=1.0, label=None, width=None, show_dets=False):
    """models: [(name, colour, frame_record or None)] drawn on one image. Returns a PIL image (optionally cropped / zoomed)."""
    im = Image.open(seq.frame_path(t)).convert("RGB")
    W, H = im.size
    boxes = [(r or {}).get("box_xyxy") for _, _, r in models]
    win = crop_window(seq, t, boxes, zoom, W, H)
    sc = 1.0
    if win:
        im = im.crop(win)
        ox, oy = win[0], win[1]
        target = width or 640
        sc = target / im.width
        im = im.resize((int(im.width * sc), int(im.height * sc)), Image.NEAREST)
    else:
        ox = oy = 0
        if width and width != W:
            sc = width / W
            im = im.resize((int(W * sc), int(H * sc)))
    d = ImageDraw.Draw(im)
    lw = max(2, int(round(im.width / 300)))
    f = _font(max(12, im.width // 45))

    def tr(b):
        return [(b[0] - ox) * sc, (b[1] - oy) * sc, (b[2] - ox) * sc, (b[3] - oy) * sc]
    g = seq.box_xywh(t)
    if show_gt and g is not None:
        bb = tr([g[0], g[1], g[0] + g[2], g[1] + g[3]])
        d.rectangle([bb[0] - 2, bb[1] - 2, bb[2] + 2, bb[3] + 2], outline=hex_rgb(GT_COLOR), width=lw)
    for k, (name, col, r) in enumerate(models):
        if not r:
            continue
        if show_dets:
            for det in r.get("detections") or []:
                d.rectangle(tr(det[:4]), outline=(120, 200, 255), width=1)
        b = r.get("box_xyxy")
        if b is not None:
            bb = tr(b)
            d.rectangle(bb, outline=hex_rgb(col), width=lw)
            tag = name.split(" ")[0]
            tw = d.textlength(tag, font=f)
            ty = max(0, bb[1] - f.size - 4) if k % 2 == 0 else bb[3] + 2
            d.rectangle([bb[0], ty, bb[0] + tw + 6, ty + f.size + 3], fill=hex_rgb(col))
            d.text((bb[0] + 3, ty + 1), tag, fill=(0, 0, 0), font=f)
    top = (label or "") + ("   [GT: target absent]" if (show_gt and g is None) else "")
    if top:
        d.rectangle([0, 0, im.width, f.size + 8], fill=(0, 0, 0))
        d.text((6, 3), top, fill=(255, 255, 255), font=f)
    return im


def timestamp(dataset_id, t):
    fps = FPS.get(dataset_id, (None, ""))[0]
    return f"{t / fps:.2f} s" if fps else "n/a (frame rate not recorded)"


def model_frame_status(seq, t, rec, start):
    """the analysis.frame_table status rules for one frame of one model -> (status, IoU, centre error px)"""
    if rec is None:
        return "not_processed", None, None
    b = rec.get("box_xyxy")
    g = seq.box_xywh(t)
    g = None if g is None else [float(v) for v in g]
    iou = cle = None
    if b is not None and g is not None:
        iou = analysis.iou_xywh(_xywh(b), g)
        cle = float(np.hypot((b[0] + b[2]) / 2 - g[0] - g[2] / 2, (b[1] + b[3]) / 2 - g[1] - g[3] / 2))
    if t == start:
        return "init", iou, cle
    if g is not None:
        return ("miss" if b is None else "tracked" if iou >= 0.5 else "wrong_object" if iou == 0 else "poor"), iou, cle
    return ("false_alarm" if b is not None else "correct_absent"), iou, cle


def _xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


# ---------------------------------------------------------------------------------------------------- tracks for the player
class Track:
    """one model's predictions on one sequence, indexed by frame (saved run or recorded source)"""

    def __init__(self, name, colour, by_frame, start, end, model_id, provenance, tracker=None, extra=None):
        self.name, self.colour, self.by_frame, self.start, self.end = name, colour, by_frame, start, end
        self.model_id, self.provenance, self.tracker, self.extra = model_id, provenance, tracker, extra or {}

    def rec(self, t):
        return self.by_frame.get(t)

    def res_xywh(self, n):
        out = [[] for _ in range(n)]
        for t, r in self.by_frame.items():
            if t < n and r.get("box_xyxy") is not None:
                out[t] = _xywh(r["box_xyxy"])
        return out

    def processed(self, n):
        return [t in self.by_frame for t in range(n)]


def tracks_from_experiment(man, sequence):
    out = []
    for i, m in enumerate(man["models"]):
        p = FD.load_prediction(man["experiment_id"], m["slug"], sequence)
        if not p:
            continue
        name = f"M{i} {short_name(m['model_id'], m.get('tracker'))}"
        out.append(Track(name, PALETTE[i % len(PALETTE)], {r["frame"]: r for r in p["frames"]}, p["start_frame"], p["end_frame"], m["model_id"],
                         f"{man['experiment_id']} ({m.get('device')})", m.get("tracker"), {"complete": p["complete"], "device": p["device"]}))
    return out


def tracks_from_recorded(source_ids, seq):
    out = []
    for i, sid in enumerate(source_ids):
        src = recorded.source(sid)
        try:
            p = recorded.load(src, seq.name)
        except FileNotFoundError:
            continue
        start = p["start_frame"] if p["start_frame"] is not None else seq.start_frame
        by = {}
        for t, r in enumerate(p["res_xywh"][:seq.n_frames]):
            if t < start or (p["processed"] is not None and t < len(p["processed"]) and not p["processed"][t]):
                continue
            box = None if not r else [r[0], r[1], r[0] + r[2], r[1] + r[3]]
            sc = None if p["scores"] is None or t >= len(p["scores"]) else p["scores"][t]
            by[t] = {"frame": t, "box_xyxy": box, "present": box is not None, "score": sc, "mode": None, "timing_ms": None}
        out.append(Track(f"R{i} {src['label']}", PALETTE[i % len(PALETTE)], by, start, min(len(p["res_xywh"]), seq.n_frames), src["model_id"],
                         p["path"], src["tracker"], {"source": src}))
    return out


def recorded_sa(R, src, sequence):
    """the per-sequence SA (%) stored in the registered record behind a recorded source, or None. The record's scale (fraction or %)
    is decided from all its values, not from the one value."""
    if not src.get("record_id"):
        return None
    try:
        per = R.experiment(src["record_id"]).get("per_sequence") or {}
    except KeyError:
        return None
    v = per.get(sequence)
    if v is None:
        return None
    vals = [x for x in per.values() if isinstance(x, (int, float))]
    return 100 * v if vals and max(vals) <= 1.0 else v


def short_name(model_id, tracker=None):
    run = model_id.split(":", 1)[-1]
    al = {"v3_n__causal_g4_cst_s0_e20@last": "X1/X2-t s0", "v3_n__causal_g4_cst_s1_e20@last": "X2-t s1", "v3_n__causal_g4_cst_s2_e20@last": "X2-t s2",
          "v3_n__ttt_cst_s0_e20@last": "X2-sf s0", "v3_n__ttt_cst_s1_e20@last": "X2-sf s1", "v3_n__ttt_cst_s2_e20@last": "X2-sf s2",
          "v2_causal_g4_s0_e20@last": "CF", "normal_ir@last": "B-normal", "dfsc_ir@last": "B-DFSC",
          "phaseB_normal_ir_seed1@last": "B-normal s1", "phaseB_dfsc_ir_seed1@last": "B-DFSC s1"}.get(run, run.replace("@last", ""))
    return al + (f" ({tracker.upper()})" if tracker in ("ste", "argmax") else "")


def status_table(seq, tracks, n):
    rows = []
    for tr in tracks:
        ft = analysis.frame_table(tr.res_xywh(n), seq.exist, seq.gt_rect, tr.start, n, tr.processed(n))
        for r in ft:
            rows.append({"model": tr.name, "frame": r["frame"], "status": r["status"], "iou": r["iou"], "cle": r["cle"]})
    return pd.DataFrame(rows)


def disagreement_frames(df):
    if df.empty:
        return []
    piv = df[df.status != "not_processed"].pivot_table(index="frame", columns="model", values="status", aggfunc="first")
    piv = piv.dropna()
    if piv.shape[1] < 2:
        return []
    good = piv.isin(["tracked", "correct_absent", "init"])
    return [int(t) for t in piv.index[good.any(axis=1) & ~good.all(axis=1)]]


# ------------------------------------------------------------------------------------------------------------- the player
def player(key, seq, tracks, dataset_id, default_view="Side by side"):
    """synchronised multi-model player: every panel shows the same frame index; controls: play / pause, step, speed, jump to
    frame or time, GT / per-model toggles, zoom, view (side by side / overlay / difference)."""
    if not tracks:
        st.info("No predictions to show for this sequence.")
        return
    lo = min(tr.start for tr in tracks)
    hi = max(tr.end for tr in tracks) - 1
    tk, pk = f"{key}_t", f"{key}_play"
    if tk not in st.session_state or not lo <= st.session_state[tk] <= hi:
        st.session_state[tk] = lo
    st.session_state.setdefault(pk, False)
    n = hi + 1
    df = status_table(seq, tracks, n)

    c = st.columns([2.2, 1.2, 1.2, 1.4, 1.6])
    view = c[0].segmented_control("View", ["Side by side", "Overlay", "Difference"], default=default_view, key=f"{key}_view") or default_view
    zoom = c[1].select_slider("Zoom on target", options=[1, 2, 3, 4, 6, 8], value=4 if dataset_id == "cst2025" else 1, key=f"{key}_zoom",
                              help="crop around the GT box (or the first prediction when the target is absent); the same window for every panel")
    show_gt = c[2].toggle("Ground truth", value=True, key=f"{key}_gt")
    show_dets = c[3].toggle("Raw detections", value=False, key=f"{key}_dets", help="AeroTrack detector only: the top-5 drone detections (thin blue)")
    visible = c[4].multiselect("Models shown", [tr.name for tr in tracks], default=[tr.name for tr in tracks], key=f"{key}_vis")
    shown = [tr for tr in tracks if tr.name in visible] or tracks[:1]

    # transport buttons get a row of their own so their word labels are never squeezed to one letter
    b = st.columns(5)
    step = {"1x": 1, "2x": 2, "5x": 5, "10x": 10}
    if b[0].button("First", key=f"{key}_first", help="first frame", width="stretch"):
        st.session_state[tk] = lo
    if b[1].button("Prev", key=f"{key}_prev", help="previous frame", width="stretch"):
        st.session_state[tk] = max(lo, st.session_state[tk] - 1)
    # on_click flips the state before this rerun draws the label, so the button reads Pause while playing
    b[2].button("Pause" if st.session_state[pk] else "Play", key=f"{key}_playbtn", type="primary", width="stretch",
                on_click=lambda: st.session_state.update({pk: not st.session_state[pk]}))
    if b[3].button("Next", key=f"{key}_next", help="next frame", width="stretch"):
        st.session_state[tk] = min(hi, st.session_state[tk] + 1)
    if b[4].button("Last", key=f"{key}_last", help="last frame", width="stretch"):
        st.session_state[tk] = hi
    b = st.columns([1, 1.4, 2])
    speed = b[0].selectbox("Speed", list(step), key=f"{key}_speed", label_visibility="collapsed", help="frames advanced per tick (~5 ticks / s)")
    jump = b[1].number_input("Go to frame", lo, hi, value=None, key=f"{key}_jump", placeholder="go to frame…", label_visibility="collapsed")
    if jump is not None and jump != st.session_state.get(f"{key}_jlast"):
        st.session_state[tk] = int(jump)
        st.session_state[f"{key}_jlast"] = jump
    dis = disagreement_frames(df) if len(shown) > 1 else []
    nxt = [t for t in dis if t > st.session_state[tk]]
    if b[2].button(f"Next disagreement ({len(dis)})", key=f"{key}_dis", disabled=not nxt,
                   help="frames where at least one model is correct (tracked / correctly absent) and another is not"):
        st.session_state[tk] = nxt[0]
        st.session_state[pk] = False

    @st.fragment(run_every=0.2 if st.session_state[pk] else None)
    def body():
        if st.session_state[pk]:
            st.session_state[tk] = st.session_state[tk] + step[speed]
            if st.session_state[tk] > hi:
                st.session_state[tk] = hi
                st.session_state[pk] = False
                st.rerun()                       # whole page: stops the ticking and turns the button back to Play
        t = st.slider("Frame", lo, hi, key=tk, help="every panel shows this frame index")
        st.caption(f"frame **{t}** / {hi} · time {timestamp(dataset_id, t)} · GT: "
                   + ("target **visible**" if seq.box_xywh(t) is not None else "target **absent**"))
        if view == "Overlay":
            im = render(seq, t, [(tr.name, tr.colour, tr.rec(t)) for tr in shown], show_gt, zoom, f"frame {t}", 900, show_dets)
            st.image(im, width="stretch")
            legend(shown)
        elif view == "Difference":
            cc = st.columns([3, 2])
            im = render(seq, t, [(tr.name, tr.colour, tr.rec(t)) for tr in shown], show_gt, max(zoom, 2), f"frame {t}", 700, show_dets)
            cc[0].image(im, width="stretch")
            rows = []
            for tr in shown:
                s, iou, cle = model_frame_status(seq, t, tr.rec(t), tr.start)
                r = tr.rec(t) or {}
                rows.append({"model": tr.name, "status": STATUS_TEXT.get(s, s), "IoU": None if iou is None else round(iou, 3),
                             "centre err px": None if cle is None else round(cle, 1), "score": r.get("score"), "mode": r.get("mode")})
            cc[1].dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            ok = [r["status"] for r in rows]
            cc[1].caption("**Models disagree on this frame.**" if len(set(ok)) > 1 else "All shown models agree on this frame's outcome.")
            legend(shown, cc[1])
        else:
            cols = st.columns(min(len(shown), 4))
            for k, tr in enumerate(shown):
                s, iou, cle = model_frame_status(seq, t, tr.rec(t), tr.start)
                with cols[k % len(cols)]:
                    im = render(seq, t, [(tr.name, tr.colour, tr.rec(t))], show_gt, zoom, None, 480, show_dets)
                    st.image(im, width="stretch")
                    st.markdown(f"<span style='color:{tr.colour};font-weight:700'>■ {tr.name}</span> — {STATUS_TEXT.get(s, s)}"
                                + (f" · IoU {iou:.2f}" if iou is not None else ""), unsafe_allow_html=True)
            if len(shown) > 4:
                st.caption("Only four panels fit side by side; use Overlay for more models.")
        timeline(df[df.model.isin([tr.name for tr in shown])], shown, t)
    body()


def legend(tracks, where=None):
    where = where or st
    items = [f"<span style='color:{GT_COLOR};font-weight:700'>■ ground truth</span>"] + \
            [f"<span style='color:{tr.colour};font-weight:700'>■ {tr.name}</span>" for tr in tracks]
    where.markdown(" &nbsp; ".join(items), unsafe_allow_html=True)


def timeline(df, tracks, t):
    import altair as alt
    if df.empty:
        return
    d = df[df.status != "not_processed"].copy()
    d["iou_plot"] = d["iou"].fillna(0.0)
    d["frame_end"] = d["frame"] + 1                     # each status cell spans exactly one frame
    xs = alt.Scale(domain=[int(d["frame"].min()), int(d["frame"].max()) + 1], nice=False)
    colours = alt.Scale(domain=[tr.name for tr in tracks], range=[tr.colour for tr in tracks])
    line = alt.Chart(d).mark_line(strokeWidth=1.2).encode(
        x=alt.X("frame:Q", title=None, scale=xs), y=alt.Y("iou_plot:Q", title="IoU with GT (0 if no box)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=colours, legend=None), tooltip=["model", "frame", "status", "iou"])
    rule = alt.Chart(pd.DataFrame({"frame": [t]})).mark_rule(color="red", strokeWidth=2).encode(x=alt.X("frame:Q", scale=xs))
    st.altair_chart((line + rule).properties(height=150), width="stretch", height=170)
    st.image(status_strip(d, tracks, t), width="stretch")
    st.markdown(" ".join(f"<span style='color:{c}'>■</span> {STATUS_TEXT[s]}" for s, c in STATUS_COLOR.items() if s != "not_processed")
                + " · <span style='color:#ff2020'>│</span> current frame", unsafe_allow_html=True)


def status_strip(d, tracks, t, width=1300, row=24, label_w=330):
    """per-frame outcome of every model as one image row (one column per frame, coloured by status); red marker = current frame"""
    lo, hi = int(d["frame"].min()), int(d["frame"].max()) + 1
    n = max(1, hi - lo)
    W = label_w + width
    im = Image.new("RGB", (W, row * len(tracks) + 4), (14, 17, 23))
    dr = ImageDraw.Draw(im)
    f = _font(17)
    for k, tr in enumerate(tracks):
        y0 = 2 + k * row
        dr.text((4, y0 + 2), tr.name[:32], fill=hex_rgb(tr.colour), font=f)
        sub = d[d.model == tr.name]
        for fr, s in zip(sub["frame"], sub["status"]):
            x0 = label_w + int((fr - lo) * width / n)
            x1 = max(x0 + 1, label_w + int((fr - lo + 1) * width / n))
            dr.rectangle([x0, y0 + 2, x1 - 1, y0 + row - 2], fill=hex_rgb(STATUS_COLOR.get(s, "#ffffff")))
    xm = label_w + int((t - lo + 0.5) * width / n)
    dr.line([xm, 0, xm, im.height], fill=(255, 32, 32), width=2)
    return im


# ------------------------------------------------------------------------------------------------------------ live monitor
def live_monitor(job_id):
    """polls the background job (no inference in the UI thread); shows the latest real frame of each model and measured counters"""
    from aerolab import jobs
    j = jobs.manager().get(job_id)
    if j is None:
        st.warning("This job is not known to the running server (the app was restarted). Its saved results are under Results.")
        return None
    snap = j.snapshot()
    cfg = snap["cfg"]
    if snap["status"] in ("queued", "running"):
        mode_banner("live", f"Job `{snap['id']}` · experiment `{snap['experiment_id'] or '…'}`")
    c = st.columns([4, 1])
    c[0].markdown(f"**{CATEGORIES.get(cfg.get('category'), {}).get('name', cfg.get('category'))}** · {cfg.get('dataset_id')} / {cfg.get('split')}"
                  f" · status **{snap['status']}** · execution {cfg.get('execution', cfg.get('mode', '—'))}")
    if snap["status"] in ("queued", "running") and c[1].button("Cancel run", type="secondary", key=f"cancel_{job_id}"):
        jobs.manager().cancel(job_id)
        st.toast("Cancellation requested: the run stops after the current frame; finished sequences are kept.")
    live = snap["live"]
    models = cfg.get("models") or []
    if cfg.get("category") in ("forecasting",):
        st.write({k: v for k, v in snap["events"].items()})
        return snap
    cols = st.columns(min(max(len(models), 1), 4))
    for i, m in enumerate(models):
        sl_prefix = f"m{i}_"
        lv = next((v for k, v in live.items() if k.startswith(sl_prefix) or v.get("model_id") == m["model_id"]), None)
        with cols[i % len(cols)]:
            st.markdown(f"<span style='color:{PALETTE[i % len(PALETTE)]};font-weight:700'>■ M{i} {short_name(m['model_id'], m.get('tracker'))}</span>",
                        unsafe_allow_html=True)
            if lv is None:
                st.caption("pending (sequential execution: waits for the previous model)" if snap["status"] == "running" else "—")
                continue
            r = lv["record"]
            seq = data.load_sequence(cfg["dataset_id"], cfg["split"], lv["sequence"], cfg.get("modality", "IR"))
            im = render(seq, r["frame"], [(f"M{i}", PALETTE[i % len(PALETTE)], r)], True, 4 if cfg["dataset_id"] == "cst2025" else 1,
                        f"{lv['sequence']} · frame {r['frame']}", 420)
            st.image(im, width="stretch")
            st.progress(min(1.0, lv["frames_done"] / max(lv["frames_total"], 1)), text=f"{lv['frames_done']} / {lv['frames_total']} frames")
            tm = r.get("timing_ms") or {}
            k = st.columns(3)
            k[0].metric("latency ms", f"{tm.get('total', 0):.1f}", help="this frame, end-to-end (read + preprocess + model + post + tracker)")
            k[1].metric("rolling FPS", "—" if lv["rolling_fps"] is None else f"{lv['rolling_fps']:.1f}", help="measured over the last 2 s (observed, not a benchmark)")
            k[2].metric("elapsed s", f"{lv['elapsed_s']:.0f}")
            st.caption(f"device **{lv['device']}** · mode {r.get('mode')} · score {r.get('score') if r.get('score') is None else round(r['score'], 3)}")
    with st.expander("Run log"):
        for ts, e in snap["log"][-30:]:
            st.text(f"{ts} {json.dumps(e)[:300]}")
    return snap


def category_label(c):
    d = CATEGORIES[c]
    return f"{d['letter']} · {d['name']}"


def go_test_bench(category):
    st.session_state["tb_category"] = category
    st.switch_page("views/test_bench.py")


ORDERED = ORDER
_ = (time, np)
