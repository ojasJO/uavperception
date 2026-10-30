"""VALIDATION-ONLY browser demo for the Anti-UAV (DFSC / GlobalTrack) reproduction.

    venv\\Scripts\\python.exe -m streamlit run experiments\\anti_uav\\demo\\app.py

Two ways to watch the tracker on any of the 67 IR VAL sequences:
  * Frame inspector — thick GT (green) / prediction (orange) boxes, 3x zoom inset, per-frame IoU / centre error /
    confidence, sequence SA, prev / play / next, first-target / middle / last, low-IoU frame picker, IoU strip.
    Continuous playback uses st.fragment(run_every=...) so only the viewer re-renders, not the whole page.
  * Annotated video — the same frames rendered once into an H.264 MP4 and played by the browser's own player
    (play / pause / scrub / speed in the player controls); the most reliable way to show motion.
Predictions are the files written by scripts/evaluate.py --split val (read-only). The TEST split is never loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_core as C  # noqa: E402

st.set_page_config(page_title="Anti-UAV reproduction — VAL demo", layout="wide")
IMG_W_SINGLE, IMG_W_COMPARE = 700, 560


# ------------------------------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner=False)
def _runs():
    return C.discover_runs()


@st.cache_resource(show_spinner="loading the 67 validation sequences…")
def _sequences(data_cfg: dict):
    return {s.name: s for s in C.load_val_sequences(data_cfg)}


@st.cache_data(show_spinner=False)
def _saved_prediction(run_name: str, seq_name: str):
    run = next(r for r in _runs() if r.name == run_name)
    return C.load_saved_prediction(run, seq_name)


@st.cache_data(show_spinner=False)
def _table(run_name: str, seq_name: str, source: str):
    """per-frame IoU table; `source` is part of the key so live results are not confused with saved ones"""
    p = st.session_state.live.get((run_name, seq_name)) if source.startswith("live") else _saved_prediction(run_name, seq_name)
    seq = _sequences(C.data_cfg_of(next(r for r in _runs() if r.name == run_name)))[seq_name]
    return C.per_frame_table(p, seq)


def _prediction(run: C.RunInfo, seq_name: str):
    key = (run.name, seq_name)
    if key in st.session_state.live:
        return st.session_state.live[key], "live (this session)"
    p = _saved_prediction(run.name, seq_name)
    if p is not None:
        return p, "saved by scripts/evaluate.py --split val"
    c = C.cached_live_prediction(run, seq_name)
    if c is not None:
        st.session_state.live[key] = c
        return c, "live (demo cache)"
    return None, "none"


# ------------------------------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault("frame", 0)
ss.setdefault("playing", False)
ss.setdefault("skip_advance_once", False)
ss.setdefault("jump_to_start", True)
ss.setdefault("live", {})


def _on_seq_change():
    ss.playing = False
    ss.jump_to_start = True


def _pause_cb():
    ss.playing = False


def _lowiou_cb():
    ss.playing = False
    v = ss.get("lowiou")
    if v:
        ss.frame = int(v.split()[1])


# ------------------------------------------------------------------------------------------ sidebar
st.title("Anti-UAV reproduction (DFSC / GlobalTrack) — validation tracker demo")
st.error("**VALIDATION ONLY** — 67 Anti-UAV IR *val* sequences (split fixed in code; the loader refuses the test split). "
         "The tracker receives the ground-truth box only on the init frame; afterwards ground truth is used solely to score. "
         "Checkpoints and run directories are read-only.")
runs = _runs()
if not runs:
    st.stop()
with st.sidebar:
    st.header("1 · Sequence")
    seqs = _sequences(C.data_cfg_of(runs[0]))
    names = sorted(seqs)
    seq_name = st.selectbox("VAL sequence", names, format_func=lambda n: f"{n}  {seqs[n].attributes}", on_change=_on_seq_change)
    seq = seqs[seq_name]
    st.header("2 · Checkpoint")
    run = st.selectbox("checkpoint (run)", runs, format_func=lambda r: r.label, on_change=_on_seq_change)
    compare = st.toggle("compare with a second checkpoint (same frames)", value=False, on_change=_on_seq_change)
    run2 = None
    if compare and len(runs) > 1:
        run2 = st.selectbox("second checkpoint", [r for r in runs if r.name != run.name], format_func=lambda r: r.label, on_change=_on_seq_change)
    st.header("3 · Playback")
    fps = st.select_slider("inspector speed (frames/s)", options=[2, 5, 10, 15], value=5)
    stride = st.select_slider("frame stride", options=[1, 2, 5, 10], value=1)
    show_inset = st.toggle("3× zoom inset on the target", value=True)
    with st.expander("Live tracking (optional, GPU)"):
        live_n = st.number_input("frames to track live from the start frame", min_value=10, max_value=seq.n_frames, value=min(100, seq.n_frames), step=10)
        do_live = st.button("run tracker live now")
        st.caption("Runs the reproduction's tracker with the selected checkpoint(s); results cached under demo/cache/, never inside a run directory.")

targets = [run] + ([run2] if run2 else [])
if do_live:
    for r in targets:
        with st.spinner(f"tracking {seq.name} with {r.name} ({int(live_n)} frames)…"):
            ss.live[(r.name, seq.name)] = C.live_track(r, seq, max_frames=int(live_n))
    ss.jump_to_start = True

preds = {r.name: _prediction(r, seq.name) for r in targets}
missing = [r.name for r in targets if preds[r.name][0] is None]
if missing:
    st.warning(f"No saved predictions for {missing} on this sequence — use *run tracker live now* (sidebar) or run "
               "`scripts/evaluate.py --checkpoint <run>/weights/last.pt --split val` once.")
    st.stop()
tables = {r.name: _table(r.name, seq.name, preds[r.name][1]) for r in targets}
summaries = {r.name: C.sequence_summary(preds[r.name][0], seq) for r in targets}
n = seq.n_frames
start = int(preds[run.name][0]["start_frame"])
if ss.jump_to_start:
    ss.frame = min(start + 1, n - 1) if n > 1 else start     # first TRACKED frame (init frame is the GT box itself)
    ss.jump_to_start = False
ss.frame = int(min(max(ss.frame, 0), n - 1))
run_every_active = bool(ss.playing)

tab_frames, tab_video = st.tabs(["Frame inspector", "Annotated video (browser player)"])

# ------------------------------------------------------------------------------------------ frame inspector (fragment)
with tab_frames:
    @st.fragment(run_every=(1.0 / fps) if run_every_active else None)
    def inspector():
        if run_every_active and not ss.playing:      # paused inside the fragment -> drop the timer with one full rerun
            st.rerun(scope="app")
        if ss.playing:
            if ss.skip_advance_once:
                ss.skip_advance_once = False
            else:
                nxt = ss.frame + stride
                if nxt > n - 1:
                    ss.playing = False
                    ss.frame = n - 1
                    st.rerun(scope="app")
                ss.frame = nxt
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 1])
        if c1.button("⏮ Previous", width="stretch"):
            was = ss.playing
            ss.playing = False
            ss.frame = max(ss.frame - stride, 0)
            if was:
                st.rerun(scope="app")
        if ss.playing:
            if c2.button("⏸ PAUSE", width="stretch", type="primary"):
                ss.playing = False
                st.rerun(scope="app")
        else:
            if c2.button("▶ PLAY", width="stretch", type="primary"):
                ss.playing = True
                ss.skip_advance_once = True
                if ss.frame >= n - 1:
                    ss.frame = start
                st.rerun(scope="app")
        if c3.button("Next ⏭", width="stretch"):
            was = ss.playing
            ss.playing = False
            ss.frame = min(ss.frame + stride, n - 1)
            if was:
                st.rerun(scope="app")
        if c4.button("First target", width="stretch", help="the init frame (tracker is given the GT box here)"):
            ss.playing = False
            ss.frame = start
        if c5.button("Middle", width="stretch"):
            ss.playing = False
            ss.frame = (start + n - 1) // 2
        if c6.button("Last", width="stretch"):
            ss.playing = False
            ss.frame = n - 1
        low = C.low_iou_frames(tables[run.name])
        opts = [f"frame {r['t']}  —  IoU {r['iou']:.2f}, CE {r['ce']:.0f} px" for r in low]
        st.selectbox(f"Jump to a low-IoU frame ({len(low)} frames with IoU < {C.LOW_IOU} for {run.short})", opts, index=None,
                     placeholder="— pick a failure frame —" if opts else "no frame below the threshold", key="lowiou", on_change=_lowiou_cb, disabled=not opts)
        st.slider("Frame", 0, n - 1, key="frame", on_change=_pause_cb)
        t = int(ss.frame)

        cols = st.columns(len(targets))
        for col, r in zip(cols, targets):
            p, src = preds[r.name]
            fm = C.frame_metrics(p, seq, t)
            with col:
                if len(targets) > 1:
                    st.markdown(f"**{r.short}**")
                img_col, side_col = st.columns([3, 2]) if show_inset else (st.container(), None)
                img_col.image(C.render_frame(seq, t, fm), width=(IMG_W_SINGLE if len(targets) == 1 else IMG_W_COMPARE) if not show_inset else "stretch",
                              output_format="JPEG")
                if side_col is not None:
                    inset = C.render_inset(seq, t, fm)
                    if inset is not None:
                        side_col.image(inset, caption="3× zoom on the target", width="stretch", output_format="JPEG")
                    else:
                        side_col.info("no target and no box on this frame")
                s = summaries[r.name]
                iou_txt = "—" if fm["iou"] is None else f"{fm['iou']:.2f}"
                ce_txt = "—" if fm["center_error_px"] is None else f"{fm['center_error_px']:.0f} px"
                conf_txt = "—" if (fm["score"] is None or fm["pred"] is None) else f"{fm['score']:.2f}"
                sa_txt = f"{s['SA_percent']:.1f} %" + (" (PARTIAL live run)" if s["partial"] else "")
                gt_txt = "present" if fm["exist"] else "absent"
                st.markdown(
                    f"**Frame:** {t} / {n-1}  \n**IoU:** {iou_txt}  \n**Centre error:** {ce_txt}  \n"
                    f"**Confidence:** {conf_txt}  \n**Sequence SA:** {sa_txt}  \n"
                    f"**Status:** {fm['status']}  ·  GT {gt_txt}  ·  predictions: {src}")
                st.image(C.render_iou_strip(tables[r.name], t), width="stretch", output_format="PNG")

    inspector()

# ------------------------------------------------------------------------------------------ annotated video
with tab_video:
    vpath = C.video_path(targets, seq, fps=25)
    st.caption("The same frames and boxes rendered once into an MP4 (25 fps, from the start frame) and played by the browser's "
               "player — use its controls for play / pause / scrub / playback speed. Rendered files are cached under demo/cache/videos/.")
    if not vpath.exists():
        if st.button(f"Render annotated video for {seq.name} (≈10–20 s)", type="primary"):
            with st.spinner("rendering…"):
                C.render_video(targets, seq, [preds[r.name][0] for r in targets], vpath, fps=25)
            st.rerun()
    if vpath.exists():
        st.video(str(vpath), format="video/mp4")

# ------------------------------------------------------------------------------------------ summary
st.subheader("Checkpoint summary")
paper = C.paper_reference()
scols = st.columns(len(targets))
for col, r in zip(scols, targets):
    s = summaries[r.name]
    stored = r.per_sequence_SA.get(seq.name)
    key = "DFSC" if r.alpha else "normal"
    pv = paper.get(key)
    col.markdown(f"**{r.name}** — {r.method}, seed {r.seed}, phase `{r.phase}`, checkpoint `{(r.checkpoint_sha256 or '?')[:12]}`"
                 + ("" if r.saved_predictions_match_checkpoint in (None, True) else "  warning: saved predictions were made with a different checkpoint"))
    a, b, c = col.columns(3)
    a.metric("this sequence SA", f"{s['SA_percent']:.2f} %", help=s["scoring"])
    b.metric("run mSA over 67 val seqs", "—" if r.val_mSA_percent is None else f"{r.val_mSA_percent:.2f} %", help="eval_val_IR_last/metrics.json")
    c.metric(f"paper Table V ({key}, val IR)", "—" if pv is None else f"{pv:.2f} %",
             delta=None if (pv is None or r.val_mSA_percent is None) else f"{r.val_mSA_percent - pv:+.2f} ours − paper")
    col.caption(f"start frame {s['start_frame']} · frames scored {s['n_frames_scored']} (target present in {s['n_exist_scored']}) · "
                f"precision@20px {100*s['precision_20px']:.1f} % · success AUC {100*s['success_auc']:.1f} %"
                + (f" · stored in metrics.json: {100*stored:.2f} %" if stored is not None and not s["partial"] else ""))
st.caption(f"Sequence attributes: {seq.attributes or 'none'} · {seq.n_frames} frames, target present in {seq.n_exist}.")
