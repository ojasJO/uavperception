import pandas as pd
import streamlit as st
from common import get_registry, protocol_banner
from fd import PALETTE, live_monitor, mode_banner, player, recorded_sa, short_name, tracks_from_experiment, tracks_from_recorded

from aerolab import analysis, data, jobs, recorded
from aerolab import flightdeck as FD
from aerolab.categories import CATEGORIES
from aerolab.test_lock import REPLAY_PURPOSE

R = get_registry()
st.title("Live Comparison")
protocol_banner()
M = jobs.manager()
active = M.active()
MODES = ["Live run", "Saved run", "Recorded predictions"]
if "live_mode" not in st.session_state:
    st.session_state["live_mode"] = "Live run" if (active or st.session_state.get("live_job")) else "Saved run"
mode = st.segmented_control("What to show", MODES, key="live_mode") or "Saved run"


def jump_buttons(key, frames, label):
    tk = f"{key}_t"
    cols = st.columns(min(8, max(1, len(frames))))
    for i, (f, txt) in enumerate(frames[:8]):
        cols[i % len(cols)].button(f"{label} {txt}", key=f"{key}_j{label}{i}", on_click=lambda f=f: st.session_state.update({tk: f, f"{key}_play": False}))


# ================================================================================================================== live run
if mode == "Live run":
    job_id = st.session_state.get("live_job") or (active[0].id if active else None)
    if job_id is None or M.get(job_id) is None:
        st.info("No run in progress in this server session. Start one from the **Test Bench**; finished runs are under *Saved run*.")
        if st.button("Open Test Bench"):
            st.switch_page("views/test_bench.py")
        st.stop()
    running = M.get(job_id).status in ("queued", "running")

    @st.fragment(run_every=0.5 if running else None)
    def monitor():
        snap = live_monitor(job_id)
        if snap and snap["status"] not in ("queued", "running"):
            if snap["status"] in ("completed", "partial", "cancelled"):
                st.success(f"Run finished — status **{snap['status']}** · experiment `{snap['experiment_id']}` (saved on disk).")
            else:
                st.error(f"Run failed: {snap.get('error')}")
            eid = snap["experiment_id"]
            if eid and eid.startswith("fd-"):
                if st.button("Open the synchronised replay of this run", type="primary", key="open_replay",
                             on_click=lambda: st.session_state.update({"live_mode": "Saved run", "live_experiment": eid})):
                    st.rerun(scope="app")            # a click inside a fragment reruns only the fragment; the page must switch mode
            elif eid:
                if st.button("Open results", type="primary", key="open_results"):
                    st.session_state["results_experiment"] = eid
                    st.switch_page("views/results.py")
            if running:
                st.rerun()
    monitor()
    st.stop()

# ================================================================================================================= saved run
if mode == "Saved run":
    exps = FD.list_experiments()
    if not exps:
        st.info("No saved Flight Deck run yet. Start one from the Test Bench.")
        st.stop()
    ids = [m["experiment_id"] for m in exps]
    want = st.session_state.pop("live_experiment", None)          # a request from another page / button, applied once
    if want in ids:
        st.session_state["live_saved_select"] = want
    elif st.session_state.get("live_saved_select") not in ids:
        st.session_state["live_saved_select"] = ids[0]
    eid = st.selectbox("Saved experiment", ids, key="live_saved_select",
                       format_func=lambda i: next(f"{i} · {CATEGORIES.get(m['category'], {}).get('name', m['category'])} · {m['dataset_id']}/{m['split']} · "
                                                  f"{len(m['models'])} model(s) · {m['status']}" for m in exps if m["experiment_id"] == i))
    man = FD.load_experiment(eid)
    mode_banner("saved", f"Experiment `{eid}` · {man['start_time']} · git `{(man.get('git_commit') or '')[:8]}`{' (dirty)' if man.get('git_dirty') else ''}")
    seqs = man["sequences"]
    c = st.columns([2, 3])
    name = c[0].selectbox(f"Sequence ({len(seqs)})", seqs, key=f"live_seq_{eid}")
    c[1].markdown(f"**{man['category_name']}** · {man['dataset_id']} / {man['split']} / {man['modality']} · preset "
                  f"*{(man.get('preset') or {}).get('name', '—')}* · protocol `{man['evaluation_protocol_id']}` · execution {man['execution']['mode']}")
    seq = data.load_sequence(man["dataset_id"], man["split"], name, man["modality"])
    tracks = tracks_from_experiment(man, name)
    if man.get("recorded_sources"):
        rt = tracks_from_recorded(man["recorded_sources"], seq)
        for k, tr in enumerate(rt):
            tr.name = f"R{k} recorded: {tr.name.split(' ', 1)[1]}"
            tr.colour = PALETTE[(len(tracks) + k) % len(PALETTE)]
        tracks += rt
        st.caption("R* tracks are **recorded** predictions replayed for comparison (not re-run); M* tracks were produced by this run.")
    if any(not tr.extra.get("complete", True) for tr in tracks):
        st.warning("Some predictions are incomplete (the run was cancelled on this sequence); no metric is reported for them.")
    key = f"pl_{eid}_{name}"
    player(key, seq, tracks, man["dataset_id"])

    st.markdown("#### Metrics on this sequence")
    rows = []
    for i, m in enumerate(man["models"]):
        s = (m.get("per_sequence") or {}).get(name)
        if not s:
            rows.append({"model": f"M{i} {short_name(m['model_id'], m.get('tracker'))}", "status": m.get("sequence_status", {}).get(name, m["status"])})
            continue
        tm = (s.get("timing_ms") or {}).get("total") or {}
        rows.append({"model": f"M{i} {short_name(m['model_id'], m.get('tracker'))}", "status": "completed", "SA %": round(s["SA"], 2),
                     "P@20px %": s["precision_20px"] and round(s["precision_20px"], 1), "success AUC %": s["success_auc"] and round(s["success_auc"], 1),
                     "mean IoU (visible)": s["mean_iou_visible"] and round(s["mean_iou_visible"], 3),
                     "mean centre error px": s["mean_cle_px"] and round(s["mean_cle_px"], 1),
                     "located %": None if s["located_rate"] is None else round(100 * s["located_rate"], 1),
                     "false alarms %": None if s["false_alarm_rate"] is None else round(100 * s["false_alarm_rate"], 1),
                     "loss events": s["episodes"]["loss_events"], "failure intervals": len(s["episodes"]["failure_intervals"]),
                     "frames": f"{s['n_scored']}", "observed ms/frame (median)": tm.get("median") and round(tm["median"], 1), "device": m.get("device")})
    for tr in tracks:
        if tr.name.startswith("R"):
            n = seq.n_frames
            s = analysis.sequence_summary(tr.res_xywh(n), seq.exist, seq.gt_rect, tr.start, None, tr.processed(n))
            recd = recorded_sa(R, tr.extra["source"], name)
            rows.append({"model": tr.name, "status": "recorded", "SA %": round(s["SA"], 2), "P@20px %": s["precision_20px"] and round(s["precision_20px"], 1),
                         "frames": f"{s['n_scored']} (whole sequence)", "recorded SA %": None if recd is None else round(recd, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Eq. 1 SA over the frames each model processed ({man['frame_rule']}). "
               + ("Truncated run → protocol studio.sot_smoke.v1: compare only within this experiment. " if man.get("max_frames") else "")
               + "Recorded rows cover the whole sequence and are therefore not comparable with a truncated run.")
    comp = man.get("comparison") or {}
    if comp.get("pairs"):
        st.markdown("#### Differences from the reference model (whole experiment)")
        ref = comp["reference"]
        rows = []
        for sl, p in comp["pairs"].items():
            pdl = p.get("paired_delta_SA") or {}
            rows.append({"model": sl, "vs reference": ref, "verdict": p["verdict"], "mean ΔSA (pp)": pdl.get("mean_delta") and round(pdl["mean_delta"], 2),
                         "95% CI": "—" if not pdl.get("ci95") else f"[{pdl['ci95'][0]:+.2f}, {pdl['ci95'][1]:+.2f}]",
                         "wins / losses": f"{pdl.get('wins', '—')} / {pdl.get('losses', '—')}", "notes": "; ".join(p["notes"] + ([pdl["note"]] if pdl.get("note") else []))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("ΔSA in percentage points (model − reference), sequence-paired; bootstrap CI only with ≥ 5 sequences. "
                   "A short demo run is a qualitative check, not a benchmark result.")
    st.markdown("#### Failure intervals on this sequence")
    for i, m in enumerate(man["models"]):
        s = (m.get("per_sequence") or {}).get(name)
        if not s:
            continue
        fi = s["episodes"]["failure_intervals"]
        st.markdown(f"<span style='color:{PALETTE[i % len(PALETTE)]};font-weight:700'>M{i} {short_name(m['model_id'], m.get('tracker'))}</span> — "
                    f"{len(fi)} interval(s), {s['episodes']['loss_events']} loss event(s)", unsafe_allow_html=True)
        if fi:
            jump_buttons(key, [(f["start"], f"{f['start']}–{f['end']} {f['onset']}/{f['dominant']}") for f in fi], f"M{i}")
    st.caption(analysis.episodes([])["rules"])
    with st.expander("Manifest (provenance: checkpoints, devices, settings, environment)"):
        st.json({k: man[k] for k in ("experiment_id", "status", "category", "dataset_id", "split", "modality", "dataset_version_or_manifest_hash",
                                      "evaluation_protocol_id", "metric_definition_version", "execution", "git_commit", "git_dirty", "environment_reference",
                                      "warnings", "outputs")})
        st.json([{k: m.get(k) for k in ("slug", "model_id", "tracker", "status", "device", "checkpoint_sha256", "checkpoint_path", "params_source",
                                         "params", "window", "preprocessing", "load_seconds", "peak_gpu_mem_mb", "error")} for m in man["models"]])
    st.stop()

# ===================================================================================================== recorded predictions
mode_banner("recorded")
c = st.columns([1.2, 1.2, 4])
did = c[0].selectbox("Dataset", ["cst2025", "antiuav2021"], format_func=lambda i: R.dataset(i)["name"], key="rec_ds")
split = c[1].selectbox("Split", ["test", "val"], key="rec_split", format_func=lambda s: "TEST (read-only replay)" if s == "test" else "VAL")
srcs = recorded.compatible_sources(did, split, "IR")
DEF = {("cst2025", "test"): ["cst:test:x1:ste", "cst:test:v3_n__ttt_cst_s0_e20:ste", "cst:test:released:GlobalTrack"],
       ("cst2025", "val"): ["cst:val:x1:ste", "cst:val:x1:argmax"],
       ("antiuav2021", "test"): ["antiuav:test:normal_ir", "antiuav:test:dfsc_ir"], ("antiuav2021", "val"): ["antiuav:val:normal_ir", "antiuav:val:dfsc_ir"]}
sids = c[2].multiselect("Recorded prediction sources (2-4 recommended)", [s["id"] for s in srcs], default=[i for i in DEF.get((did, split), []) if i in {s["id"] for s in srcs}],
                        format_func=lambda i: recorded.source(i)["label"], key=f"rec_srcs_{did}_{split}", max_selections=6)
if split == "test":
    st.caption("TEST frames are opened only to draw the registered passes' own recorded predictions (purpose `replay_recorded`). No model runs, "
               "no threshold or parameter can be changed here, and the numbers shown are recomputations of recorded results, checked against their records.")
if not sids:
    st.info("Choose at least one recorded source.")
    st.stop()
common = set.intersection(*[set(recorded.sequences_of(recorded.source(i))) for i in sids])
names = sorted(common)
if not names:
    st.warning("The chosen sources have no sequence in common.")
    st.stop()
default_seq = "jungle_18" if "jungle_18" in names else names[0]
name = st.selectbox(f"Sequence ({len(names)} shared by all sources)", names, index=names.index(default_seq), key=f"rec_seq_{did}_{split}",
                    help="jungle_18 (CST TEST): FINAL_REPORT §4 — X1's only loss to GlobalTrack; the distractor lock is seed-specific")
seq = data.load_sequence(did, split, name, "IR", purpose=REPLAY_PURPOSE)
tracks = tracks_from_recorded(sids, seq)
player(f"rec_{did}_{split}_{name}_{'_'.join(sids)}"[:180], seq, tracks, did)
rows = []
for tr in tracks:
    n = seq.n_frames
    s = analysis.sequence_summary(tr.res_xywh(n), seq.exist, seq.gt_rect, tr.start, None, tr.processed(n))
    src = tr.extra["source"]
    recd = recorded_sa(R, src, name)
    ep = s["episodes"]
    rows.append({"source": tr.name, "SA % (recomputed)": round(s["SA"], 2), "SA % (record)": None if recd is None else round(recd, 2),
                 "|Δ|": None if recd is None else round(abs(s["SA"] - recd), 4), "located %": s["located_rate"] and round(100 * s["located_rate"], 1),
                 "false alarms %": None if s["false_alarm_rate"] is None else round(100 * s["false_alarm_rate"], 1),
                 "wrong-object frames": s["status_counts"]["wrong_object"], "failure intervals": len(ep["failure_intervals"]),
                 "registration": src.get("registration") or "—", "file": src["dir"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("SA recomputed here with Track A's sequence_metrics from the recorded boxes; '|Δ|' checks it against the per-sequence value stored in the "
           "registered record (0 = the replay is the registered result). The official CST toolkit equals Track A per sequence (tests/test_cst_protocol.py).")
