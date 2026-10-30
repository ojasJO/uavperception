import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from common import get_registry, protocol_banner
from fd import PALETTE, short_name

from aerolab import analysis, data, evidence, recorded
from aerolab import benchmark as BM
from aerolab import flightdeck as FD
from aerolab import forecasting as FC
from aerolab.categories import CATEGORIES
from aerolab.paths import REPO, STUDIO_RUNS

R = get_registry()
st.title("Results & Failures")
st.caption("Your Flight Deck runs and the established (recorded) evidence, per category. Recorded numbers are read from their files at render "
           "time; recomputations say so. Numbers from different protocols are never ranked together.")
protocol_banner()
WANT = st.session_state.pop("results_experiment", None)        # a request from another page, applied once


def pick_run(category_ids, key, label="Run"):
    exps = [m for m in FD.list_experiments() if m["category"] in category_ids]
    if not exps:
        st.info("No saved run of this category yet — start one from the Test Bench.")
        return None
    ids = [m["experiment_id"] for m in exps]
    if WANT in ids:
        st.session_state[key] = WANT
    elif st.session_state.get(key) not in ids:
        st.session_state[key] = ids[0]
    eid = st.selectbox(label, ids, key=key,
                       format_func=lambda i: next(f"{i} · {m['dataset_id']}/{m['split']} · " + ", ".join(short_name(x["model_id"], x.get("tracker")) for x in m["models"])
                                                  + f" · {len(m['sequences'])} seq · {m['status']}" for m in exps if m["experiment_id"] == i))
    return FD.load_experiment(eid)


def open_replay(man, seq, frame=None):
    st.session_state["live_mode"] = "Saved run"
    st.session_state["live_experiment"] = man["experiment_id"]
    st.session_state[f"live_seq_{man['experiment_id']}"] = seq
    if frame is not None:
        st.session_state[f"pl_{man['experiment_id']}_{seq}_t"] = int(frame)


def model_table(man):
    rows = []
    for i, m in enumerate(man["models"]):
        mt = m.get("metrics") or {}
        tm = json.loads((REPO / man["outputs"]["timing"]).read_text(encoding="utf-8")).get(m["slug"], {}) if (REPO / man["outputs"]["timing"]).exists() else {}
        rows.append({"model": f"M{i} {short_name(m['model_id'], m.get('tracker'))}", "status": m["status"], "device": m.get("device"),
                     "mSA %" if not man.get("max_frames") else "mSA (partial) %": None if mt.get("mSA", mt.get("mSA_partial")) is None else round(mt.get("mSA", mt.get("mSA_partial")), 2),
                     "P@20 %": mt.get("mean_precision_20px") and round(mt["mean_precision_20px"], 1),
                     "centre err px": mt.get("mean_cle_px") and round(mt["mean_cle_px"], 1),
                     "located %": None if mt.get("mean_located_rate") is None else round(100 * mt["mean_located_rate"], 1),
                     "false alarms %": None if mt.get("mean_false_alarm_rate") is None else round(100 * mt["mean_false_alarm_rate"], 1),
                     "loss events": mt.get("loss_events"), "failure intervals": mt.get("failure_intervals"),
                     "observed ms/frame": ((tm.get("total") or {}).get("median") and round(tm["total"]["median"], 1)),
                     "checkpoint": (m.get("checkpoint_sha256") or "")[:12]})
    return pd.DataFrame(rows)


def per_seq_chart(man, metric="SA"):
    rows = []
    for i, m in enumerate(man["models"]):
        for s, v in (m.get("per_sequence") or {}).items():
            rows.append({"sequence": s, "model": f"M{i} {short_name(m['model_id'], m.get('tracker'))}", metric: v[metric]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    names = sorted(df.model.unique())
    st.altair_chart(alt.Chart(df).mark_bar().encode(x=alt.X("sequence:N", title=None), xOffset="model:N", y=alt.Y(f"{metric}:Q", title=f"{metric} (%)"),
                                                    color=alt.Color("model:N", scale=alt.Scale(domain=names, range=PALETTE[:len(names)])),
                                                    tooltip=["sequence", "model", metric]).properties(height=240), width="stretch")


def deltas(man):
    comp = man.get("comparison") or {}
    if not comp.get("pairs"):
        return
    rows = []
    for sl, p in comp["pairs"].items():
        pdl = p.get("paired_delta_SA") or {}
        rows.append({"model": sl, "reference": comp["reference"], "verdict": p["verdict"], "mean ΔSA (pp)": pdl.get("mean_delta") and round(pdl["mean_delta"], 2),
                     "95 % CI (pp)": "—" if not pdl.get("ci95") else f"[{pdl['ci95'][0]:+.2f}, {pdl['ci95'][1]:+.2f}]",
                     "wins / losses / ties": f"{pdl.get('wins', '—')} / {pdl.get('losses', '—')} / {pdl.get('ties', '—')}",
                     "notes": "; ".join(p["notes"] + ([pdl["note"]] if pdl.get("note") else []))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Percentage points (model − reference), paired by sequence; bootstrap B = 10,000 (default_rng(0)) only with ≥ 5 sequences. "
               "No significance claim is made from a short demonstration run.")


def failure_table(man, key):
    rows = []
    for i, m in enumerate(man["models"]):
        for s, v in (m.get("per_sequence") or {}).items():
            for f in v["episodes"]["failure_intervals"]:
                rows.append({"model": f"M{i} {short_name(m['model_id'], m.get('tracker'))}", "sequence": s, "start": f["start"], "end": f["end"],
                             "length": f["length"], "onset": f["onset"], "dominant": f["dominant"], "recovered after (frames)": f["frames_to_recover"]})
    if not rows:
        st.caption("No failure interval (≥ 5 consecutive visible frames with IoU < 0.5) in this run.")
        return
    df = pd.DataFrame(rows).sort_values("length", ascending=False)
    st.dataframe(df, hide_index=True, width="stretch", height=min(300, 36 + 35 * len(df)))
    pick = st.selectbox("Open an interval in the synchronised replay", range(len(df)), key=f"{key}_fi",
                        format_func=lambda i: f"{df.iloc[i]['model']} · {df.iloc[i]['sequence']} · frames {df.iloc[i]['start']}–{df.iloc[i]['end']} ({df.iloc[i]['dominant']})")
    if st.button("Open in Live Comparison", key=f"{key}_fi_open"):
        r = df.iloc[pick]
        open_replay(man, r["sequence"], max(0, int(r["start"]) - 5))
        st.switch_page("views/live.py")
    st.caption(analysis.episodes([])["rules"])


tabs = st.tabs(["A Tracking", "B Presence", "C Temporal", "D Robustness", "E Speed", "F Forecasting", "G Reproduction", "All runs"])

# ================================================================================================================ A tracking
with tabs[0]:
    st.markdown("#### Your runs")
    man = pick_run(("tracking", "robustness", "temporal", "reproduction", "speed"), "res_a")
    if man:
        st.caption(f"{man['category_name']} · {man['dataset_id']} / {man['split']} / {man['modality']} · protocol `{man['evaluation_protocol_id']}` · "
                   f"{len(man['sequences'])} sequence(s) · {man['frame_rule']}")
        st.dataframe(model_table(man), hide_index=True, width="stretch")
        deltas(man)
        per_seq_chart(man)
        st.markdown("##### Failure intervals (annotation-confirmed what, never why)")
        failure_table(man, "res_a")
    st.markdown("#### Established results (recorded, read-only)")
    rows = []
    for eid, lab in (("cst:x1:ste", "X1 (STE)"), ("cst:x2:v3_n__causal_g4_cst_s1_e20:ste", "X2 temporal s1"), ("cst:x2:v3_n__causal_g4_cst_s2_e20:ste", "X2 temporal s2"),
                     ("cst:x2:v3_n__ttt_cst_s0_e20:ste", "single-frame s0"), ("cst:released:GlobalTrack", "GlobalTrack (released)"),
                     ("cst:released:SiamDT", "SiamDT (released)"), ("cst:released:PrDiMP", "PrDiMP (released)")):
        try:
            e = R.experiment(eid)
        except KeyError:
            continue
        rows.append({"model": lab, "CST TEST mSA (official toolkit)": round(e["metrics"]["mSA"], 2), "success AUC": e["metrics"].get("success_auc") and round(e["metrics"]["success_auc"], 2),
                     "P@20px": e["metrics"].get("precision_20px") and round(e["metrics"]["precision_20px"], 2), "record": eid, "source": e["source_file"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Protocol sot.mSA.cst_official.v1 (60 CST TEST sequences, registered passes and released predictions). Not comparable with any studio run.")

# ================================================================================================================ B presence
with tabs[1]:
    st.markdown("#### P1 — the presence gate (recorded, CST VAL, exploratory)")
    h = evidence.p1_headline()
    p1 = evidence.p1()
    k = st.columns(4)
    k[0].metric("False alarms on absent frames", f"{100 * h['fa_to']:.1f} %", f"{100 * (h['fa_to'] - h['fa_from']):+.1f} pp from {100 * h['fa_from']:.1f} %", delta_color="inverse")
    k[1].metric("Missed visible targets", f"{100 * h['miss_to']:.1f} %", f"{100 * (h['miss_to'] - h['miss_from']):+.1f} pp from {100 * h['miss_from']:.1f} %", delta_color="inverse")
    k[2].metric("Cross-fitted mSA", f"{h['mSA_to']:.2f}", f"{h['mSA_to'] - h['mSA_from']:+.2f} from {h['mSA_from']:.2f}")
    k[3].metric("Registered verdict", p1["verdict"], f"3-run mean Δ {p1['three_run_mean_delta']:+.2f}", delta_color="off")
    st.caption(f"Run {h['run']}, {h['gate']} vs no gate. A **demonstrated trade-off**, not an improvement: suppressing low-score outputs removes false "
               f"alarms but loses visible targets, and the cross-fitted selection chose τ = 0 on every half. Source: `{h['source']}`.")
    cv = pd.DataFrame([p for r in p1["runs"].values() for p in r["curve"]])
    cv["run"] = cv["run"].str.replace("v3_n__causal_g4_cst_", "").str.replace("_e20", "")
    ch1 = alt.Chart(cv).mark_line(point=True).encode(x=alt.X("false_alarm_rate:Q", title="false alarms on absent frames", axis=alt.Axis(format="%")),
                                                     y=alt.Y("visible_miss_rate:Q", title="missed visible targets", axis=alt.Axis(format="%")),
                                                     color="run:N", strokeDash="variant:N", order="tau_out:Q",
                                                     tooltip=["run", "variant", "tau_out", alt.Tooltip("false_alarm_rate", format=".3f"), alt.Tooltip("visible_miss_rate", format=".3f"),
                                                              alt.Tooltip("mSA_cross_fitted", format=".2f")]).properties(height=280, title="each point = one gate threshold τ")
    ch2 = alt.Chart(cv).mark_line(point=True).encode(x=alt.X("tau_out:Q", title="gate threshold τ"), y=alt.Y("mSA_cross_fitted:Q", title="cross-fitted VAL mSA", scale=alt.Scale(zero=False)),
                                                     color="run:N", strokeDash="variant:N").properties(height=280, title="mSA is flat, then falls")
    st.altair_chart(alt.hconcat(ch1, ch2), width="stretch")

    st.markdown("#### Your runs — frame-level presence (TRAIN / VAL only)")
    man = pick_run(("presence", "tracking", "robustness", "temporal"), "res_b")
    if man:
        det = [(i, m) for i, m in enumerate(man["models"]) if R.model(m["model_id"])["adapter"] == "aerotrack_detector" and m.get("per_sequence")]
        if not det:
            st.info("This run has no AeroTrack detector output (GlobalTrack has no presence signal).")
        else:
            curves, P_all = [], {}
            for i, m in det:
                parts = []
                for s in man["sequences"]:
                    p = FD.load_prediction(man["experiment_id"], m["slug"], s)
                    if not p or not p["complete"]:
                        continue
                    q = data.load_sequence(man["dataset_id"], man["split"], s, man["modality"])
                    parts.append(analysis.presence_frames(p["frames"], q.exist, q.gt_rect, p["start_frame"], p["end_frame"]))
                if not parts:
                    continue
                P = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
                name = f"M{i} {short_name(m['model_id'], m.get('tracker'))}"
                P_all[name] = P
                for r in analysis.presence_curve(P):
                    curves.append({**r, "model": name})
            if not P_all:
                st.info("No completed predictions.")
            else:
                nab = {k: int((~v["visible"]).sum()) for k, v in P_all.items()}
                if not any(nab.values()):
                    st.warning("These sequences have no target-absent frame after the start frame: false-alarm rate and precision on absence are "
                               "undefined. Use the 'Target absent' preset.")
                thr = st.slider("Score threshold (exploration on this VAL / TRAIN run only)", 0.0, 1.0, 0.30, 0.01, key="res_b_thr")
                cdf = pd.DataFrame(curves)
                rows = []
                for name, P in P_all.items():
                    r = analysis.presence_curve(P, [thr])[0]
                    tp = analysis.tracker_presence(P)
                    rows.append({"model": name, "visible frames": int(P["visible"].sum()), "absent frames": nab[name],
                                 "TP (present, visible)": r["tp"], "FP (present, absent)": r["fp"], "FN (absent, visible)": r["fn"], "TN (absent, absent)": r["tn"],
                                 "false-alarm rate": r["false_alarm_rate"], "miss rate": r["miss_rate"], "precision": r["precision"],
                                 "correct localisation": r["correct_localisation_rate"], "score AUC": analysis.presence_auc(P),
                                 "tracker output: FA / miss": f"{tp['false_alarm_rate'] if tp['false_alarm_rate'] is None else round(tp['false_alarm_rate'], 3)} / "
                                                              f"{tp['miss_rate'] if tp['miss_rate'] is None else round(tp['miss_rate'], 3)}"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("Positive class = target visible. Score = highest drone-class detection score of the frame (not a calibrated probability); "
                           "'correct localisation' additionally needs the top box to overlap the GT (IoU ≥ 0.5). The tracker's own decision is a fixed point.")
                names = list(P_all)
                cs = alt.Scale(domain=names, range=PALETTE[:len(names)])
                long = cdf.melt(id_vars=["threshold", "model"], value_vars=["false_alarm_rate", "miss_rate"], var_name="rate", value_name="value")
                a1 = alt.Chart(long).mark_line().encode(x=alt.X("threshold:Q"), y=alt.Y("value:Q", title="rate", axis=alt.Axis(format="%")), color=alt.Color("model:N", scale=cs),
                                                        strokeDash="rate:N", tooltip=["model", "rate", "threshold", alt.Tooltip("value", format=".3f")])
                a1 = (a1 + alt.Chart(pd.DataFrame({"threshold": [thr]})).mark_rule(color="red").encode(x="threshold:Q")).properties(height=260, title="false alarms vs misses")
                a2 = alt.Chart(cdf.dropna(subset=["precision"])).mark_line().encode(x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1])),
                                                                                    color=alt.Color("model:N", scale=cs), order="threshold:Q").properties(height=260, title="precision–recall (presence)")
                hist = pd.DataFrame([{"model": n, "score": s, "GT": "visible" if v else "absent"} for n, P in P_all.items() for s, v in zip(P["score"], P["visible"])])
                a3 = alt.Chart(hist).mark_bar(opacity=0.6).encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=25)), y=alt.Y("count():Q", stack=None), color="GT:N",
                                                                  column=alt.Column("model:N", title=None)).properties(height=180, width=220)
                st.altair_chart(alt.hconcat(a1, a2), width="stretch")
                st.altair_chart(a3)

# ================================================================================================================ C temporal
with tabs[2]:
    st.markdown("#### Established: X2 temporal window vs single frame (CST TEST, registered, read-only)")
    tab = evidence.x2_seed_table(R)
    T = evidence.x2_tests()
    rows = []
    for arm in ("temporal causal_g4", "single-frame (ttt)"):
        s = tab[arm]["summary"]
        rows.append({"arm": arm, **{f"seed {k}": round(v, 2) for k, v in tab[arm]["per_seed"].items()}, "mean": round(s["mean"], 2), "SD": round(s["sd"], 2),
                     "range": f"{s['min']:.2f}–{s['max']:.2f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    t3 = T["T3"]["temporal_minus_single_frame_seed_mean"]
    dx = evidence.x2_minus_x1(R)
    k = st.columns(3)
    k[0].metric("T3 · temporal − single frame", f"{t3['mean_delta']:+.2f} pp", f"95 % CI [{t3['ci95'][0]:+.2f}, {t3['ci95'][1]:+.2f}] · {t3['wins']}/{t3['losses']}", delta_color="off")
    k[1].metric("T2 · temporal seed mean − GlobalTrack", f"{T['T2']['causal_g4']['mean_delta']:+.2f} pp",
                f"95 % CI [{T['T2']['causal_g4']['ci95'][0]:+.2f}, {T['T2']['causal_g4']['ci95'][1]:+.2f}] · {T['T2']['causal_g4']['wins']}/{T['T2']['causal_g4']['losses']}", delta_color="off")
    k[2].metric("X2 temporal seed mean − X1 (descriptive)", f"{dx['mean_delta']:+.2f} pp", f"95 % CI [{dx['ci95'][0]:+.2f}, {dx['ci95'][1]:+.2f}] · {dx['wins']}/{dx['losses']}", delta_color="off")
    st.caption("Sequence-paired bootstrap over the 60 CST TEST sequences (B = 10,000, default_rng(0), percentile CI) — the registered uncertainty. "
               "With three seeds no seed-level CI is shown (mean ± SD only). The window's benefit holds under this protocol (CST, causal t/t−4/t−8, "
               "STE); X2 does not demonstrably improve on X1. " + f"Sources: `{T['source']}`; X2 − X1 recomputed from the per-sequence records.")
    seqs = sorted(R.experiment("cst:x1:ste")["per_sequence"])
    t_mean = {s: np.mean([R.experiment(f"cst:x2:v3_n__causal_g4_cst_s{k}_e20:ste")["per_sequence"][s] for k in (0, 1, 2)]) for s in seqs}
    f_mean = {s: np.mean([R.experiment(f"cst:x2:v3_n__ttt_cst_s{k}_e20:ste")["per_sequence"][s] for k in (0, 1, 2)]) for s in seqs}
    dd = pd.DataFrame([{"sequence": s, "Δ SA (temporal − single frame), seed means": t_mean[s] - f_mean[s]} for s in seqs]).sort_values("Δ SA (temporal − single frame), seed means")
    st.altair_chart(alt.Chart(dd).mark_bar().encode(x=alt.X("sequence:N", sort=None, axis=alt.Axis(labels=False, title="60 CST TEST sequences (sorted)")),
                                                    y="Δ SA (temporal − single frame), seed means:Q",
                                                    color=alt.condition("datum['Δ SA (temporal − single frame), seed means'] > 0", alt.value("#2b8a3e"), alt.value("#c92a2a")),
                                                    tooltip=list(dd.columns)).properties(height=200), width="stretch")
    st.caption("Recorded TEST per-sequence differences — read-only evidence of the registered pass, never a basis for choosing settings.")
    vrows = []
    for arm, tag in (("temporal", "causal_g4"), ("single-frame", "ttt")):
        for s_ in (0, 1, 2):
            for trk in ("ste", "argmax"):
                try:
                    e = R.experiment(f"cst:x2val:v3_n__{tag}_cst_s{s_}_e20:{trk}")
                    vrows.append({"arm": arm, "seed": s_, "tracker": trk, "VAL mSA (cross-fitted)": round(e["metrics"]["mSA_cross_fitted"], 2)})
                except KeyError:
                    pass
    st.markdown("CST **VAL** (cross-fitted over two halves; VAL shares source videos with TRAIN → optimistic):")
    st.dataframe(pd.DataFrame(vrows).pivot_table(index=["arm", "tracker"], columns="seed", values="VAL mSA (cross-fitted)"), width="stretch")
    st.markdown("#### Your temporal runs")
    man = pick_run(("temporal", "tracking"), "res_c")
    if man:
        st.dataframe(model_table(man), hide_index=True, width="stretch")
        deltas(man)
        rows = []
        for i, m in enumerate(man["models"]):
            ps = m.get("per_sequence") or {}
            wo = sum(v["status_counts"]["wrong_object"] for v in ps.values())
            vis = sum(v["n_visible"] for v in ps.values())
            tm = json.loads((REPO / man["outputs"]["timing"]).read_text(encoding="utf-8")).get(m["slug"], {}) if (REPO / man["outputs"]["timing"]).exists() else {}
            rows.append({"model": f"M{i} {short_name(m['model_id'], m.get('tracker'))}", "window": json.dumps(m.get("window")),
                         "wrong-object frames % of visible": None if not vis else round(100 * wo / vis, 2),
                         "model ms (median, observed)": (tm.get("model") or {}).get("median") and round(tm["model"]["median"], 2),
                         "read+preprocess ms (median)": (tm.get("read_preprocess") or {}).get("median") and round(tm["read_preprocess"]["median"], 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("The temporal window reads 3 frames (2 of them usually cached) and the model sees 9 input channels; timings are observed during the run "
                   "(Category E measures them under controlled conditions).")

# =============================================================================================================== D robustness
with tabs[3]:
    st.markdown("#### Your scenario runs")
    man = pick_run(("robustness", "tracking", "temporal", "reproduction"), "res_d")
    if man:
        pr = man.get("preset") or {}
        st.markdown(f"Preset **{pr.get('name', '—')}** · basis `{pr.get('basis')}` · rule: {pr.get('rule')} · source: {pr.get('source')}")
        agg = {}
        for i, m in enumerate(man["models"]):
            name = f"M{i} {short_name(m['model_id'], m.get('tracker'))}"
            for s, v in (m.get("per_sequence") or {}).items():
                for k, a in (v.get("attribute_SA") or {}).items():
                    x = agg.setdefault((name, k), [0.0, 0])
                    x[0] += a["SA"] * a["n_frames"]
                    x[1] += a["n_frames"]
        if agg:
            df = pd.DataFrame([{"model": m, "condition": k, "SA %": v[0] / v[1], "frames": v[1]} for (m, k), v in agg.items() if v[1]])
            st.altair_chart(alt.Chart(df).mark_rect().encode(x=alt.X("condition:N", title="frames carrying the flag"), y=alt.Y("model:N", title=None),
                                                             color=alt.Color("SA %:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[0, 100])),
                                                             tooltip=["model", "condition", alt.Tooltip("SA %", format=".1f"), "frames"]).properties(height=40 + 30 * df.model.nunique())
                            + alt.Chart(df).mark_text(fontSize=11).encode(x="condition:N", y="model:N", text=alt.Text("SA %:Q", format=".0f")), width="stretch")
            st.caption("Frame-level Eq. 1 SA averaged over every scored frame carrying the flag (official CST flags OC OV SV TC DBC CDB FM; absent / visible; "
                       "size bins tiny ≤ 10 px, large > 50 px from the GT diagonal; 'after_reappearance_30f' = the 30 frames after the target returns). "
                       "Flags with few frames move a lot; frame counts are in the tooltip.")
        st.markdown("##### Failure intervals and re-acquisition")
        failure_table(man, "res_d")
        rec_rows = []
        for i, m in enumerate(man["models"]):
            ab = [e for v in (m.get("per_sequence") or {}).values() for e in v["episodes"]["absence_episodes"]]
            rq = [e["frames_to_reacquire"] for e in ab if e["frames_to_reacquire"] is not None]
            rec_rows.append({"model": f"M{i} {short_name(m['model_id'], m.get('tracker'))}", "absence episodes": len(ab),
                             "false-alarm share during absences": None if not ab else round(float(np.mean([e["false_alarm_share"] for e in ab])), 3),
                             "re-acquired (of returns)": f"{len(rq)} / {sum(1 for e in ab if e['target_returns_at'] is not None)}",
                             "frames to re-acquire (median)": None if not rq else float(np.median(rq))})
        st.dataframe(pd.DataFrame(rec_rows), hide_index=True, width="stretch")
    st.markdown("#### Established: CST TEST frame attributes (recorded, read-only)")
    d = json.loads((REPO / evidence.X2_OFFICIAL).read_text(encoding="utf-8"))["runs"]
    rows = []
    for run, r in d.items():
        for k, v in (r.get("frame_attr_SA") or {}).items():
            rows.append({"run": run.replace("v3_n__", "").replace("_cst", "").replace("_e20", ""), "attribute": k, "SA %": round(v["SA"], 2), "frames": v["n_frames"]})
    if rows:
        st.dataframe(pd.DataFrame(rows).pivot_table(index="run", columns="attribute", values="SA %"), width="stretch")
        st.caption(f"Corrected frame-level convention (every flagged frame); official toolkit scores of the registered X2 pass. Source: `{evidence.X2_OFFICIAL}`. "
                   "jungle_18 (X1's only loss to GlobalTrack) can be replayed frame by frame in Live Comparison → Recorded predictions.")

# ===================================================================================================================== E speed
with tabs[4]:
    st.markdown("#### Controlled benchmarks (protocol speed.bench.local.v1)")
    bms = BM.list_benchmarks()
    if not bms:
        st.info("No benchmark yet — Test Bench → E · Speed → Controlled benchmark.")
    else:
        ids = [b["experiment_id"] for b in bms]
        if WANT in ids or st.session_state.get("res_e") not in ids:
            st.session_state["res_e"] = WANT if WANT in ids else ids[0]
        bid = st.selectbox("Benchmark", ids, key="res_e",
                           format_func=lambda i: next(f"{i} · {b['sequence']} · {len(b['results'])} run(s) · {b['status']}" for b in bms if b["experiment_id"] == i))
        b = next(x for x in bms if x["experiment_id"] == bid)
        rows = []
        for r in b["results"]:
            stg = r.get("stages") or {}
            rows.append({"model": short_name(r["model_id"], r.get("tracker")), "device": r.get("device"), "status": r["status"], "init s": r.get("init_s") and round(r["init_s"], 2),
                         "first frame ms (cold)": r.get("first_frame_ms") and round(r["first_frame_ms"], 1),
                         "median ms": stg.get("total", {}).get("median") and round(stg["total"]["median"], 2), "p95 ms": stg.get("total", {}).get("p95") and round(stg["total"]["p95"], 2),
                         "p99 ms": stg.get("total", {}).get("p99") and round(stg["total"]["p99"], 2),
                         "model-only median ms": stg.get("model", {}).get("median") and round(stg["model"]["median"], 2),
                         "FPS (1000 / median)": r.get("fps_from_median_total") and round(r["fps_from_median_total"], 1),
                         "peak GPU MB": r.get("peak_gpu_mb") and round(r["peak_gpu_mb"], 0), "RSS MB": r.get("rss_mb") and round(r["rss_mb"], 0),
                         "CPU cores used": r.get("avg_cpu_cores_used") and round(r["avg_cpu_cores_used"], 1), "timed frames": r.get("n_timed_frames"),
                         "CUDA sync": r.get("cuda_synchronized")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        stg = pd.DataFrame([{"run": f"{short_name(r['model_id'], r.get('tracker'))} · {r.get('device')}", "stage": s, "median ms": (r.get("stages") or {}).get(s, {}).get("median")}
                            for r in b["results"] for s in ("read_preprocess", "model", "postprocess", "tracker") if r["status"] == "completed"])
        if not stg.empty:
            st.altair_chart(alt.Chart(stg).mark_bar().encode(y=alt.Y("run:N", title=None), x=alt.X("median ms:Q", stack="zero", title="median ms per frame"),
                                                             color="stage:N", tooltip=["run", "stage", alt.Tooltip("median ms", format=".2f")]).properties(height=50 + 40 * stg.run.nunique()),
                            width="stretch")
        raw = REPO / b["outputs"]["dir"] / "timing.json"
        if raw.exists():
            tj = json.loads(raw.read_text(encoding="utf-8"))
            dist = pd.DataFrame([{"run": k, "total ms": f["total"]} for k, fr in tj.items() for f in fr])
            if not dist.empty:
                st.altair_chart(alt.Chart(dist).mark_boxplot(extent="min-max").encode(x=alt.X("total ms:Q", scale=alt.Scale(zero=False)), y=alt.Y("run:N", title=None))
                                .properties(height=50 + 40 * dist.run.nunique(), title="latency distribution (timed frames)"), width="stretch")
        with st.expander("Method (recorded in the manifest)"):
            st.text(b["method"])
            st.json({"frames": b["frames"], "environment": b["environment_reference"], "git": b["git_commit"]})
    st.markdown("#### Historical deployment gate (TensorRT / ONNX, recorded — a different setup)")
    dr = evidence.deploy_records(R)
    if dr:
        st.dataframe(pd.DataFrame([{"record": e["experiment_id"], **{k: v for k, v in (e.get("metrics") or {}).items() if isinstance(v, (int, float))},
                                    "notes": (e.get("notes") or "")[:100], "source": e.get("source_file")} for e in dr]), hide_index=True, width="stretch")
        st.caption("deploy.latency.v1: TensorRT fp16 engines, batch 1, on this laptop's RTX 5050 — never mixed with the PyTorch-eager numbers above.")

# =============================================================================================================== F forecasting
with tabs[5]:
    st.markdown("#### Your forecasting runs (protocol forecast.ADE_NLL.trackc.v1, VAL)")
    fcs = FC.list_forecasts()
    if not fcs:
        st.info("No forecasting run yet — Test Bench → F · Forecasting.")
    else:
        ids = [f["experiment_id"] for f in fcs]
        if WANT in ids or st.session_state.get("res_f") not in ids:
            st.session_state["res_f"] = WANT if WANT in ids else ids[0]
        fid = st.selectbox("Forecasting run", ids, key="res_f",
                           format_func=lambda i: next(f"{i} · {len(f['sequences'])} seq · " + ", ".join(short_name(m["model_id"]) for m in f["models"]) for f in fcs if f["experiment_id"] == i))
        man = next(f for f in fcs if f["experiment_id"] == fid)
        st.caption(f"{man['comparability']} · history: {man['history_source']} · {man['metric_note']}")
        rows = [{"forecaster": sl, **{k: None if v is None else round(v, 3) for k, v in mt.items() if k != "n_sequences"}, "sequences": mt["n_sequences"]}
                for sl, mt in man["metrics"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("ADE / FDE in pixels over the 24-step horizon (lower is better); NLL per 2-D point (lower is better); cp / cv / dcv = classical "
                   "references on the same anchors (dcv TRAIN-tuned). Never comparable with tracking mSA.")
        c = st.columns([2, 3])
        key = c[0].selectbox("Sequence", man["sequences"], key=f"res_f_seq_{fid}")
        preds = {}
        for m in man["models"]:
            p = REPO / man["outputs"]["dir"] / "predictions" / m["slug"] / f"{key}.json"
            if p.exists():
                preds[m["slug"]] = json.loads(p.read_text(encoding="utf-8"))
        anyp = next(iter(preds.values()), None)
        if anyp and anyp.get("anchors"):
            A = anyp["anchors"]
            ai = c[1].slider("Anchor (frame t)", 0, len(A["t"]) - 1, len(A["t"]) // 2, key=f"res_f_anchor_{fid}_{key}")
            t0 = A["t"][ai]
            hm = np.array(A["hist_m"][ai], bool)
            hc, fc_, fm = np.array(A["hist_c"][ai]), np.array(A["fut_c"][ai]), np.array(A["fut_m"][ai], bool)
            rows = [{"x": x, "y": y, "series": "history (GT)", "h": i - len(hc), "sx": 0, "sy": 0} for i, (x, y) in enumerate(hc) if hm[i]]
            rows += [{"x": x, "y": y, "series": "future (GT)", "h": i + 1, "sx": 0, "sy": 0} for i, (x, y) in enumerate(fc_) if fm[i]]
            for k, (sl, p) in enumerate(preds.items()):
                a = p["anchors"]
                for i, ((x, y), (sx, sy)) in enumerate(zip(a["mean"][ai], a["std"][ai])):
                    rows.append({"x": x, "y": y, "series": f"{sl} mean ±2σ", "h": i + 1, "sx": 2 * sx, "sy": 2 * sy})
            for n, v in A["refs"].items():
                rows += [{"x": x, "y": y, "series": f"classical {n}", "h": i + 1, "sx": 0, "sy": 0} for i, (x, y) in enumerate(v[ai])]
            df = pd.DataFrame(rows)
            df["x0"], df["x1"], df["y0"], df["y1"] = df.x - df.sx, df.x + df.sx, df.y - df.sy, df.y + df.sy
            base = alt.Chart(df)
            lines = base.mark_line(point=True).encode(x=alt.X("x:Q", title="x (px)", scale=alt.Scale(zero=False)), y=alt.Y("y:Q", title="y (px, image down)", scale=alt.Scale(zero=False, reverse=True)),
                                                      color="series:N", order="h:Q", tooltip=["series", "h", alt.Tooltip("x", format=".1f"), alt.Tooltip("y", format=".1f")])
            ex = base.transform_filter("datum.sx > 0").mark_rule(opacity=0.35).encode(x="x0:Q", x2="x1:Q", y="y:Q", color="series:N")
            ey = base.transform_filter("datum.sy > 0").mark_rule(opacity=0.35).encode(y="y0:Q", y2="y1:Q", x="x:Q", color="series:N")
            st.altair_chart((lines + ex + ey).properties(height=380, title=f"{key} · anchor frame {t0} · horizon 1–24"), width="stretch")
            st.caption("Bars = ±2σ of the learned Gaussian per step (x and y separately). History and future are ground truth (oracle-history protocol).")
            ade = pd.DataFrame([{"anchor t": t, "ADE px": v, "series": sl} for sl, p in preds.items() for t, v in zip(p["anchors"]["t"], p["anchors"]["ade_px"])]
                               + [{"anchor t": t, "ADE px": v, "series": f"classical {n}"} for n, vs in A["refs_ade_px"].items() for t, v in zip(A["t"], vs)])
            st.altair_chart(alt.Chart(ade).mark_line(point=True).encode(x="anchor t:Q", y="ADE px:Q", color="series:N").properties(height=200, title="ADE per kept anchor"), width="stretch")
    st.markdown("#### Established Track C verdicts (recorded)")
    tv = evidence.trackc_verdicts(R)
    st.dataframe(pd.DataFrame([{"record": e["experiment_id"], "split": e.get("split"), "notes": (e.get("notes") or "")[:160],
                                "metrics": ", ".join(f"{k} {v:.3g}" for k, v in (e.get("metrics") or {}).items() if isinstance(v, (int, float)))[:160],
                                "source": e.get("source_file")} for e in tv]), hide_index=True, width="stretch", height=260)
    st.caption("Conclusion (TRACK_C_SUMMARY.md): on CST the learned forecaster did not justify replacing the classical one (pooled: negative transfer; "
               "per-domain: no detectable difference from dcv). Learned forecasters helped on Anti-UAV.")

# ============================================================================================================= G reproduction
with tabs[6]:
    st.markdown("#### G1 · Anti-UAV (Jiang et al. 2021, arXiv:2101.08466) — Protocol II, IR (Direction B)")
    rows = evidence.antiuav_reproduction(R)
    st.dataframe(pd.DataFrame([{"model": r["model"], "split": r["split"], "paper mSA": r["paper_mSA"], "reproduction mSA": round(r["ours_mSA"], 2),
                                "Δ (pp)": round(r["delta"], 2), "verdict (|Δ| ≤ 1 / ≤ 3)": r["verdict"], "TEST pass": "yes" if r["test_pass"] else "no",
                                "ours": r["ours_source"], "paper": r["paper_source"]} for r in rows]), hide_index=True, width="stretch")
    by = {(r["model"], r["split"]): r for r in rows}
    if all(k in by for k in (("normal", "val"), ("DFSC", "val"), ("normal", "test"), ("DFSC", "test"))):
        st.markdown(f"**DFSC effect (DFSC − normal):** paper {by[('DFSC', 'val')]['paper_mSA'] - by[('normal', 'val')]['paper_mSA']:+.2f} (val) / "
                    f"{by[('DFSC', 'test')]['paper_mSA'] - by[('normal', 'test')]['paper_mSA']:+.2f} (test) · reproduction "
                    f"{by[('DFSC', 'val')]['ours_mSA'] - by[('normal', 'val')]['ours_mSA']:+.2f} (val) / {by[('DFSC', 'test')]['ours_mSA'] - by[('normal', 'test')]['ours_mSA']:+.2f} (test). "
                    "The paper's DFSC gain was **not reproduced** (single seed each; this does not show DFSC is ineffective in general).")
    st.caption("Recorded registered results; both TEST passes are consumed and are never re-run here. The verdict rule is the project's pre-registered one.")
    with st.expander("Local configuration and its differences from the paper"):
        a = yaml.safe_load((REPO / "experiments/anti_uav/configs/normal_ir.yaml").read_text(encoding="utf-8"))
        b = yaml.safe_load((REPO / "experiments/anti_uav/configs/dfsc_ir.yaml").read_text(encoding="utf-8"))
        st.markdown(f"`normal_ir.yaml` vs `dfsc_ir.yaml` (both inherit `_base_ir.yaml`): differences → `{json.dumps({k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)})}`")
        spec = (REPO / "docs/ANTI_UAV_REPRODUCTION_SPEC.md").read_text(encoding="utf-8")
        i = spec.find("## 11")
        if i >= 0:
            j = spec.find("\n## ", i + 5)
            st.markdown(spec[i:j if j > 0 else None])
        else:
            st.caption("docs/ANTI_UAV_REPRODUCTION_SPEC.md: section 11 (substitutions S1–S6) not found")
        st.caption("Possible sources of the TEST gap (documented, not proven): checkpoint selection (final epoch), input scale (1333×800 rule), "
                   "the official start-frame handling, evaluation from JPEG frames — see FINAL_PROTOCOL_II_IR_BENCHMARK_REPORT.md.")
    st.markdown("##### Your reproduction runs — fresh VAL run vs the recorded VAL evaluation of the same checkpoint")
    man = pick_run(("reproduction", "tracking"), "res_g")
    if man:
        if man.get("max_frames"):
            st.warning("This run is truncated (studio.sot_smoke.v1); a consistency check needs whole sequences. Run the reproduction preset with 'Whole sequences'.")
        rows = []
        for i, m in enumerate(man["models"]):
            rid = {"antiuav_repro.globaltrack:normal_ir@last": "antiuav:val:normal_ir", "antiuav_repro.globaltrack:dfsc_ir@last": "antiuav:val:dfsc_ir",
                   "antiuav_repro.globaltrack:phaseB_normal_ir_seed1@last": "antiuav:val:phaseB_normal_ir_seed1",
                   "antiuav_repro.globaltrack:phaseB_dfsc_ir_seed1@last": "antiuav:val:phaseB_dfsc_ir_seed1"}.get(m["model_id"])
            for s, v in (m.get("per_sequence") or {}).items():
                recd = None
                if rid and man["split"] == "val" and not man.get("max_frames"):
                    src = recorded.source(rid)
                    q = data.load_sequence(man["dataset_id"], "val", s, man["modality"])
                    p = recorded.load(src, s)
                    recd = analysis.sequence_summary(p["res_xywh"], q.exist, q.gt_rect, p["start_frame"], None, p["processed"])["SA"]
                rows.append({"model": f"M{i} {short_name(m['model_id'])}", "sequence": s, "fresh SA %": round(v["SA"], 3),
                             "recorded SA %": None if recd is None else round(recd, 3), "Δ": None if recd is None else round(v["SA"] - recd, 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Δ = fresh − recorded on the same checkpoint and sequence (recorded: experiments/anti_uav/runs/<run>/eval_val_IR_last). 0 means the local "
                   "pipeline reproduces the recorded evaluation exactly.")
    st.markdown("#### G2 · CST benchmark (arXiv:2507.23473) — published baseline table reproduced from the authors' released predictions")
    br = evidence.cst_baseline_reproduction(R)
    st.dataframe(pd.DataFrame([{"tracker": r["tracker"], "paper Table 3 mSA": r["paper_mSA"], "released predictions → official toolkit": round(r["released_predictions_mSA"], 2),
                                "Δ": None if r["delta"] is None else round(r["delta"], 3)} for r in br]), hide_index=True, width="stretch")
    st.caption("Shared evidence of the CST study: every value equals the paper to rounding (BASELINE_RESULTS.md §1). These trackers never report "
               "absence. Their per-frame predictions can be replayed next to X1 / X2 in Live Comparison → Recorded predictions.")

# ================================================================================================================ all runs
with tabs[7]:
    allm = FD.list_experiments() + BM.list_benchmarks() + FC.list_forecasts()
    rows = [{"experiment": m["experiment_id"], "category": CATEGORIES.get(m.get("category"), {}).get("name", m.get("category")), "status": m["status"],
             "dataset / split": f"{m.get('dataset_id')} / {m.get('split')}",
             "models": ", ".join(short_name(x["model_id"], x.get("tracker")) for x in m.get("models", m.get("results", []))),
             "completed": m.get("completion_time"), "dir": (m.get("outputs") or {}).get("dir")} for m in allm]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"{len(rows)} saved Flight Deck runs under {STUDIO_RUNS.relative_to(REPO).as_posix()}/evals/ (fd- tracking, bench- speed, fc- forecasting). "
               "Each directory holds its config, manifest, per-sequence results, timings, log and predictions.")
