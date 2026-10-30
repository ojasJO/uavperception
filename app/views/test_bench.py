import json

import pandas as pd
import streamlit as st
from common import DIR_LABEL, get_registry, protocol_banner
from fd import PALETTE, category_label, render, short_name

from aerolab import categories, data, jobs, recorded, sequences
from aerolab import benchmark as BM
from aerolab import flightdeck as FD
from aerolab import forecasting as FC
from aerolab.categories import CATEGORIES, ORDER
from aerolab.paths import STUDIO_RUNS

R = get_registry()
st.title("Test Bench")
st.caption("Choose a category, a dataset and a scenario preset, pick sequences and compatible models, check the summary, then start. "
           "Runs execute in the background; the Live Comparison page shows their frames as they are produced.")
protocol_banner()

cat = st.segmented_control("Experiment category", list(ORDER), format_func=category_label, key="tb_category",
                           default=None if "tb_category" in st.session_state else "tracking") or "tracking"
C = CATEGORIES[cat]
with st.container(border=True):
    st.markdown(f"#### {category_label(cat)}")
    st.markdown(f"_{C['question']}_")
    st.caption(f"Initialisation: {C['init']} · metrics: {', '.join(C['metrics'])} · TRAIN / VAL only")

DEFAULT_DIR = {"reproduction": "antiuav_repro", "presence": "aerotrack", "temporal": "aerotrack", "forecasting": "aerotrack"}
_dk = f"tb_dir_{cat}"
direction = st.radio("Research direction", ["both", "aerotrack", "antiuav_repro"], horizontal=True, key=_dk,
                     **({} if _dk in st.session_state else {"index": ["both", "aerotrack", "antiuav_repro"].index(DEFAULT_DIR.get(cat, "both"))}),
                     format_func=lambda d: "Both (only where a fair comparison exists)" if d == "both" else DIR_LABEL[d])

FEATURED = categories.featured()
FEAT_OF = {v: f for f in FEATURED for v in f["versions"]}


def model_label(mid):
    m = R.model(mid)
    f = FEAT_OF.get(mid)
    star = "* " if f else ""
    return f"{star}[{'A' if m['direction'] == 'aerotrack' else 'B' if m['direction'] == 'antiuav_repro' else 'S'}] {short_name(mid)} — " \
           f"{m['run_id']} @{m['checkpoint_tag']} · {m['status']}" + (f" · {', '.join(m['aliases'])}" if m.get("aliases") else "")


def model_pool(category):
    el = categories.eligible_models(category, R, direction)
    feat = [v for f in FEATURED for v in f["versions"]]
    el.sort(key=lambda m: (m["model_id"] not in feat, feat.index(m["model_id"]) if m["model_id"] in feat else 0, m["family_id"], m["run_id"], m["checkpoint_tag"]))
    return el


def show_ineligible(category):
    rows = []
    for m in R.models:
        if direction != "both" and m["direction"] != direction:
            continue
        ok, why = categories.eligibility(category, m)
        if not ok:
            rows.append({"model": m["model_id"], "status": m["status"], "why not offered": why})
    with st.expander(f"Not offered for this category ({len(rows)} registry entries) — why"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=240)


def estimate_seconds(mid, device, n_frames):
    """median observed total ms / frame of earlier Flight Deck runs of this model on this device (None if never measured)"""
    vals = []
    for p in sorted((STUDIO_RUNS / "evals").glob("fd-*/timing.json"))[-60:]:
        try:
            tm = json.loads(p.read_text(encoding="utf-8"))
            man = json.loads((p.parent / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for m in man.get("models", []):
            t = tm.get(m["slug"]) or {}
            if m["model_id"] == mid and m.get("device") == device and (t.get("total") or {}).get("median"):
                vals.append(t["total"]["median"])
    return None if not vals else sorted(vals)[len(vals) // 2] * n_frames / 1000


def device_note():
    import torch
    ok = torch.cuda.is_available()
    return ok, (f"CUDA available: {torch.cuda.get_device_name(0)}" if ok else "CUDA not available in this process: GPU runs are refused; use CPU")


# ============================================================================================================ forecasting (F)
if cat == "forecasting":
    keys = data.list_sequences("trackc_trajectories", "val")
    dom = st.multiselect("Trajectory domains", ["antiuav_IR", "antiuav_RGB", "cst"], default=["cst"], key="tb_fc_dom")

    def dom_of(k):
        return "cst" if k.startswith("cst_") else "antiuav_" + k.rsplit("__", 1)[-1]
    pool = [k for k in keys if dom_of(k) in dom]
    seqs = st.multiselect(f"VAL trajectory sequences ({len(pool)} in the chosen domains)", pool, default=pool[:2], key="tb_fc_seqs")
    el = model_pool("forecasting")
    default = [m["model_id"] for m in el if m["model_id"] in ("aerotrack.forecaster.c1:c1_tfm_abs_s0@weights",)]
    mids = st.multiselect("Forecasters (1-4)", [m["model_id"] for m in el], default=default, format_func=model_label, max_selections=4, key="tb_fc_models")
    c = st.columns(2)
    max_anchors = c[0].number_input("Max anchors per sequence (evenly spaced)", 50, 5000, 400, 50, key="tb_fc_max")
    keep = c[1].number_input("Anchors kept for the trajectory browser", 5, 200, 40, 5, key="tb_fc_keep")
    show_ineligible("forecasting")
    cfg = FC.build_config(seqs, mids, int(max_anchors), int(keep))
    errs = FC.validate(cfg, R)
    with st.container(border=True):
        st.markdown("**Experiment summary**")
        st.markdown(f"- Category **F · Forecasting** · protocol `forecast.ADE_NLL.trackc.v1` · history = ground truth (C1 oracle protocol)\n"
                    f"- {len(seqs)} VAL sequence(s): {', '.join(seqs[:6])}{'…' if len(seqs) > 6 else ''}\n"
                    f"- Forecasters: {', '.join(short_name(m) for m in mids) or '—'} · CPU · classical cp / cv / dcv on the same anchors\n"
                    "- Metrics in pixels (lower is better) and NLL — never comparable with tracking mSA")
    for e in errs:
        st.error(e)
    if st.button("Start forecasting experiment", type="primary", disabled=bool(errs), key="tb_fc_start"):
        j = jobs.manager().submit(cfg, "forecast")
        st.session_state["live_job"] = j.id
        st.session_state["results_job"] = j.id
        st.switch_page("views/live.py")
    st.stop()

# ================================================================================================================= data
st.markdown("### 1 · Data")
c = st.columns(3)
dsets = list(C["datasets"])
if cat == "reproduction":
    dsets = ["antiuav2021"]
did = c[0].selectbox("Dataset", dsets, format_func=lambda i: R.dataset(i)["name"], key=f"tb_ds_{cat}")
split = c[1].selectbox("Split", ["val", "train"], key=f"tb_split_{cat}", help="TEST is locked; recorded TEST predictions can be replayed in Live Comparison")
mod = c[2].selectbox("Modality", data.SOT_DATASETS[did], key=f"tb_mod_{cat}")

pids = list(C["presets"])
preset_id = st.selectbox("Scenario preset", pids, format_func=lambda p: sequences.PRESET_NAMES[p], key=f"tb_preset_{cat}_{did}")
idx = sequences.sequence_index(did, split, mod)
by = {s["name"]: s for s in idx}
custom = st.session_state.get(f"tb_custom_{cat}_{did}_{split}", [])
P = sequences.resolve_preset(preset_id, did, split, mod, custom=custom)
BASIS = {"official_annotation": "official annotation", "derived_from_gt": "derived from ground truth", "recorded_results": "recorded results",
         "fixed_rule": "fixed rule", "official_annotation+derived_from_gt": "official annotation + derived from GT", "user_selection": "hand-picked",
         "unavailable": "unavailable"}
with st.container(border=True):
    st.markdown(f"**{P['name']}** · {BASIS.get(P['basis'], P['basis'])} · {len(P['items'])} eligible sequence(s)")
    st.caption(f"Rule: {P['rule']} · Source: {P['source']}")
    if not P["available"]:
        st.warning(f"Preset unavailable: {P['reason']}")
if preset_id == "custom":
    custom = st.multiselect("Choose sequences", [s["name"] for s in idx], default=custom, key=f"tb_custom_{cat}_{did}_{split}")
    P = sequences.resolve_preset(preset_id, did, split, mod, custom=custom)

items = P["items"]
if items:
    tab = pd.DataFrame([{"sequence": it["sequence"], "why it qualifies": it["why"], "frames": by[it["sequence"]]["n_frames"],
                         "absent %": round(100 * by[it["sequence"]]["absent_fraction"], 1), "reappearances": len(by[it["sequence"]]["reappearance_frames"]),
                         "median box diag px": None if by[it["sequence"]]["diag_median"] is None else round(by[it["sequence"]]["diag_median"], 1),
                         "attributes": ", ".join(f"{k}:{v}" for k, v in by[it["sequence"]]["frame_attrs"].items() if v) or ", ".join(by[it["sequence"]]["seq_attrs"])}
                        for it in items])
    st.dataframe(tab, hide_index=True, width="stretch", height=min(320, 36 + 35 * len(tab)))
names = [it["sequence"] for it in items]
default_n = 1 if cat in ("speed",) else 2
sel_key = f"tb_seqs_{cat}_{did}_{split}_{preset_id}"
if sel_key in st.session_state:
    st.session_state[sel_key] = [s for s in st.session_state[sel_key] if s in names]
sel = st.multiselect(f"Sequences to run ({len(names)} eligible)", names, default=None if sel_key in st.session_state else names[:default_n], key=sel_key,
                     max_selections=1 if cat == "speed" else None)

if sel:
    s0 = by[sel[0]]
    pc = st.columns([2, 3])
    seq = data.load_sequence(did, split, sel[0], mod)
    pc[0].image(render(seq, seq.start_frame, [], True, 4 if did == "cst2025" else 1, f"{sel[0]} · start frame {seq.start_frame} (GT init)", 420),
                width="stretch", caption="start frame with the GT box used for initialisation" + (" (4x zoom on the target)" if did == "cst2025" else ""))
    try:
        size = sequences.image_size(did, split, sel[0], mod)
    except OSError:
        size = None
    rec_srcs = [s["label"] for s in recorded.compatible_sources(did, split, mod) if sel[0] in recorded.sequences_of(s)]
    pc[1].markdown(f"**{sel[0]}**")
    pc[1].markdown(f"- {s0['n_frames']} frames · resolution {size[0]}×{size[1]} px" if size else f"- {s0['n_frames']} frames")
    fps, fps_src = sequences.FPS[did]
    rate = f"{fps:.0f} FPS → {s0['duration_s']:.1f} s" if fps else "not recorded"
    pc[1].markdown(f"- frame rate: {rate} ({fps_src})")
    pc[1].markdown(f"- target visible on {s0['n_visible']} scored frames, absent on {s0['n_absent']} ({s0['n_absent_episodes']} episode(s), longest {s0['longest_absence']})")
    pc[1].markdown(f"- reappearances at frames {s0['reappearance_frames'][:8]}" if s0["reappearance_frames"] else "- no reappearance")
    if s0["diag_median"] is not None:
        pc[1].markdown(f"- GT box diagonal: median {s0['diag_median']:.1f} px (min {s0['diag_min']:.1f}, max {s0['diag_max']:.1f}); tiny frames {s0['n_tiny']}, large {s0['n_large']}")
    if s0["frame_attrs"]:
        pc[1].markdown("- official per-frame flags: " + ", ".join(f"{k} {v}" for k, v in s0["frame_attrs"].items()))
    if s0["seq_attrs"]:
        pc[1].markdown("- official sequence attributes: " + ", ".join(s0["seq_attrs"]))
    pc[1].markdown("- recorded predictions available: " + ("; ".join(rec_srcs) if rec_srcs else "none"))

# ================================================================================================================ models
st.markdown("### 2 · Models")
el = model_pool(cat)
if not el:
    st.warning("No runnable model is eligible for this category and direction.")
DEFAULTS = {"tracking": {"cst2025": ["aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last", "aerotrack.detector.cst_x:v3_n__ttt_cst_s0_e20@last"],
                         "antiuav2021": ["antiuav_repro.globaltrack:normal_ir@last", "aerotrack.detector.v2:v2_causal_g4_s0_e20@last"]},
            "presence": {"cst2025": ["aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last", "aerotrack.detector.v2:v2_causal_g4_s0_e20@last"],
                         "antiuav2021": ["aerotrack.detector.v2:v2_causal_g4_s0_e20@last"]},
            "temporal": {"cst2025": ["aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last", "aerotrack.detector.cst_x:v3_n__ttt_cst_s0_e20@last"]},
            "robustness": {"cst2025": ["aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last", "aerotrack.detector.cst_x:v3_n__ttt_cst_s0_e20@last"],
                           "antiuav2021": ["antiuav_repro.globaltrack:normal_ir@last", "aerotrack.detector.v2:v2_causal_g4_s0_e20@last"]},
            "speed": {"cst2025": ["aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last"], "antiuav2021": ["antiuav_repro.globaltrack:normal_ir@last"]},
            "reproduction": {"antiuav2021": ["antiuav_repro.globaltrack:normal_ir@last", "antiuav_repro.globaltrack:dfsc_ir@last"]}}
ids = [m["model_id"] for m in el]
dflt = [m for m in DEFAULTS.get(cat, {}).get(did, []) if m in ids]
allow_many = st.toggle("Extended comparison (up to 6 models)", value=False, key=f"tb_many_{cat}", help="more GPU memory in lockstep; more time in sequential")
mids = st.multiselect("Models (* = featured)", ids, default=dflt, format_func=model_label, key=f"tb_models_{cat}_{did}_{direction}",
                      max_selections=FD.MAX_MODELS_EXTENDED if allow_many else FD.MAX_MODELS)
pairs = []
for mid in mids:
    m = R.model(mid)
    trackers = FD.TRACKERS[m["adapter"]]
    if len(trackers) > 1:
        tr = st.multiselect(f"Tracker(s) for {short_name(mid)}", trackers, default=["ste"], key=f"tb_tr_{cat}_{mid}",
                            help="STE = Kalman state estimator with an absent decision (frozen VAL settings); argmax = best detection above a threshold")
    else:
        tr = list(trackers)
    pairs += [{"model_id": mid, "tracker": t} for t in tr]
show_ineligible(cat)
ref = 0
if len(pairs) > 1:
    ref = st.selectbox("Reference model (differences are reported against it)", range(len(pairs)),
                       format_func=lambda i: f"M{i} {short_name(pairs[i]['model_id'], pairs[i]['tracker'])}", key=f"tb_ref_{cat}")

# =============================================================================================================== options
st.markdown("### 3 · Execution")
cuda_ok, dev_msg = device_note()
if cat == "speed":
    smode = st.segmented_control("Measurement", ["Controlled benchmark", "Live demonstration"], default="Controlled benchmark", key="tb_speed_mode")
    if smode == "Controlled benchmark":
        c = st.columns(4)
        devices = c[0].multiselect("Devices", ["cuda", "cpu"], default=["cuda"] if cuda_ok else ["cpu"], key="tb_bm_dev", help=dev_msg)
        n_frames = c[1].number_input("Timed frames per repeat", 5, 2000, 100, 5, key="tb_bm_n")
        warm = c[2].number_input("Warm-up frames (discarded)", 0, 200, 10, 1, key="tb_bm_w")
        reps = c[3].number_input("Repeats", 1, 10, 1, 1, key="tb_bm_r")
        st.caption(dev_msg)
        cfg = BM.build_config(did, split, mod, sel[0] if sel else None, pairs, devices, n_frames, warm, reps)
        errs = BM.validate(cfg, R) if sel else ["select one sequence"]
        with st.container(border=True):
            st.markdown("**Benchmark summary** — protocol `speed.bench.local.v1`")
            st.markdown(f"- {len(pairs)} model(s) × {len(devices)} device(s), one at a time, loaded fresh for each device\n"
                        f"- sequence `{sel[0] if sel else '—'}` from its start frame: 1 init frame (reported as cold first-frame latency) + {warm} warm-up "
                        f"frames (discarded) + {n_frames} timed frames, × {reps}\n"
                        "- stages: read+preprocess · model (CUDA-synchronised) · postprocess · tracker; no drawing/display; PyTorch eager, batch 1\n"
                        "- memory: peak GPU allocation (torch), process RSS; CPU time / wall time")
        for e in errs:
            st.error(e)
        if st.button("Run benchmark", type="primary", disabled=bool(errs), key="tb_bm_start"):
            j = jobs.manager().submit(cfg, "benchmark")
            st.session_state["live_job"] = j.id
            st.switch_page("views/live.py")
        st.stop()

c = st.columns(4)
full = c[0].toggle("Whole sequences", value=False, key=f"tb_full_{cat}", help="off = the init frame + N frames per sequence (protocol studio.sot_smoke.v1)")
max_frames = None if full else c[1].number_input("Frames after the init frame", 10, 5000, 150, 10, key=f"tb_n_{cat}")
device = c[2].selectbox("Device", ["auto", "cuda", "cpu"], key=f"tb_dev_{cat}", help="auto = CUDA when available. " + dev_msg)
execution = c[3].selectbox("Execution", ["sequential", "lockstep"], key=f"tb_exec_{cat}",
                           help="sequential: one model in memory at a time, undisturbed timings (default). lockstep: all models loaded, each frame by every "
                                "model in turn -> synchronised live panels; interleaved timings, more GPU memory")
st.caption(dev_msg)
rsrc = []
if cat in ("tracking", "reproduction", "robustness", "temporal") and sel:
    avail = [s for s in recorded.compatible_sources(did, split, mod) if all(q in recorded.sequences_of(s) for q in sel)]
    if avail:
        rsrc = st.multiselect("Also show recorded predictions in the replay (consistency check vs a recorded evaluation)", [s["id"] for s in avail],
                              format_func=lambda i: recorded.source(i)["label"], key=f"tb_rec_{cat}_{did}",
                              help="Recorded predictions are replayed next to the fresh run; they are labelled as recorded and are not re-run")
note = st.text_input("Note (stored in the manifest)", key=f"tb_note_{cat}")

cfg = FD.build_config(cat, did, split, mod, sel, pairs, device, execution, max_frames, preset={k: P[k] for k in ("id", "name", "basis", "rule", "source")},
                      direction=direction, reference=ref, note=note or None, allow_many=allow_many, recorded_sources=rsrc)
errs, warns = FD.validate(cfg, R)

# =============================================================================================================== summary
st.markdown("### 4 · Summary")
with st.container(border=True):
    nf = 0
    for q in sel:
        s = by[q]
        nf += (s["n_frames"] - s["start_frame"]) if full else min(s["n_frames"] - s["start_frame"], 1 + int(max_frames or 0))
    proto = FD.protocol_of(cfg)
    st.markdown(f"**{category_label(cat)}** · {R.dataset(did)['name']} / **{split.upper()}** / {mod} · preset **{P['name']}** ({BASIS.get(P['basis'], P['basis'])})")
    st.markdown(f"- {len(sel)} sequence(s): {', '.join(sel[:8])}{'…' if len(sel) > 8 else ''} — **{nf} frames per model** "
                f"({'whole sequences' if full else f'init + {max_frames} frames each'})")
    for i, p in enumerate(pairs):
        m = R.model(p["model_id"])
        est = estimate_seconds(p["model_id"], "cuda" if (device != "cpu" and cuda_ok) else "cpu", nf)
        st.markdown(f"- <span style='color:{PALETTE[i % len(PALETTE)]};font-weight:700'>M{i}</span> {short_name(p['model_id'], p['tracker'])} · "
                    f"`{m['sha256'][:12]}…` · {DIR_LABEL[m['direction']]}"
                    + (f" · ≈ {est:.0f} s (median of earlier runs on this machine)" if est else " · no earlier timing on this machine")
                    + (" · **reference**" if i == ref and len(pairs) > 1 else ""), unsafe_allow_html=True)
    st.markdown(f"- protocol `{proto}` · device **{device}** · execution **{execution}** · outputs → `results/runs/studio/evals/fd-…/`")
    if rsrc:
        st.markdown("- recorded predictions shown alongside: " + "; ".join(recorded.source(i)["label"] for i in rsrc))
for e in errs:
    st.error(e)
for w in warns:
    st.warning(w)
b = st.columns([1, 3])
if b[0].button("Start experiment", type="primary", disabled=bool(errs), key="tb_start", width="stretch"):
    j = jobs.manager().submit(cfg, "sot")
    st.session_state["live_job"] = j.id
    st.session_state["live_mode"] = "Live run"
    st.switch_page("views/live.py")
b[1].caption("The run continues in the background if you change page; cancel it from Live Comparison. Re-rendering this page never starts "
             "a run — only this button does.")
