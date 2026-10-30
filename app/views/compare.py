import altair as alt
import pandas as pd
import streamlit as st
from common import DIR_LABEL, DIR_SHORT, METRIC_HELP, fmt, get_registry, model_label, protocol_banner

from aerolab import data
from aerolab.adapters import compatibility, get_adapter
from aerolab.compare import compatibility as proto_compat
from aerolab.compare import paired_delta
from aerolab.evaluation import evaluate_sot

R = get_registry()
st.title("Comparison laboratory")
st.caption("A number is ranked against another only when protocol, split and every comparability key match. Otherwise the pair is "
           "labelled NON-EQUIVALENT and shown side by side without a ranking.")
protocol_banner()
mode = st.radio("Mode", ["Recorded results", "Fresh VAL comparison (runs models now)"], horizontal=True)

if mode == "Recorded results":
    exps = [e for e in R.experiments() if e.get("protocol_id") and e.get("metrics")]
    c = st.columns(4)
    protos = sorted({e["protocol_id"] for e in exps})
    proto = c[0].selectbox("Protocol", protos, index=protos.index("sot.mSA.cst_official.v1") if "sot.mSA.cst_official.v1" in protos else 0, format_func=lambda p: f"{p} — {R.protocols.get(p, {}).get('name', '')}")
    pool = [e for e in exps if e["protocol_id"] == proto]
    splits = sorted({str(e.get("split")) for e in pool})
    split = c[1].selectbox("Split", splits, index=splits.index("test") if proto == "sot.mSA.cst_official.v1" and "test" in splits else 0)
    pool = [e for e in pool if str(e.get("split")) == split]
    dset = c[2].selectbox("Dataset", sorted({str(e.get("dataset_id")) for e in pool}))
    pool = [e for e in pool if str(e.get("dataset_id")) == dset]
    mods = sorted({str(e.get("modality")) for e in pool})
    mod = c[3].selectbox("Modality", mods)
    pool = [e for e in pool if str(e.get("modality")) == mod]
    if split == "test":
        st.warning("Recorded TEST results: read-only evidence of registered passes. Not for model or threshold selection.")
    P = R.protocols.get(proto, {})
    with st.expander("Protocol definition"):
        st.json(P)
    metrics = sorted({k for e in pool for k, v in (e.get("metrics") or {}).items() if isinstance(v, (int, float))})
    metric = st.selectbox("Metric", metrics, index=metrics.index("mSA") if "mSA" in metrics else 0, help=METRIC_HELP.get("mSA"))
    lab = {e["experiment_id"]: f"{DIR_SHORT[e['direction']]} · {e['experiment_id']} · {e.get('kind')}" for e in pool}
    pick = st.multiselect("Records (first = baseline)", list(lab), format_func=lab.get, max_selections=12,
                          default=[x for x in lab if x in ("cst:x1:ste", "cst:released:GlobalTrack")])
    sel = [next(e for e in pool if e["experiment_id"] == i) for i in pick]
    if len(sel) < 1:
        st.info("Select at least one record.")
        st.stop()
    base = sel[0]
    rows = []
    for e in sel:
        comp = proto_compat(e, base, R.protocols) if e is not base else {"verdict": "baseline", "reasons": []}
        pdlt = paired_delta(e, base) if e is not base and comp["verdict"] == "comparable" else None
        rows.append({"record": e["experiment_id"], "direction": DIR_LABEL[e["direction"]], "models": ", ".join(e.get("model_ids") or []),
                     metric: (e.get("metrics") or {}).get(metric), "n seq": len(e.get("per_sequence") or {}) or None,
                     "vs baseline": comp["verdict"], "paired Δ [95% CI]": f"{fmt(pdlt['mean_delta'])} {fmt(pdlt['ci95'])} ({pdlt['wins']}/{pdlt['losses']})" if pdlt else "—",
                     "why not comparable": "; ".join(comp["reasons"]), "provenance": e.get("provenance"), "source": e.get("source_file")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Paired Δ = sequence-paired mean difference (record − baseline), sequence bootstrap B = 10,000; wins/losses count sequences. "
               "Recomputed here from the recorded per-sequence values; registered reports remain the authority.")
    comparable = [e for e in sel if e is base or proto_compat(e, base, R.protocols)["verdict"] == "comparable"]
    long = [{"sequence": s, "value": v, "record": e["experiment_id"]} for e in comparable for s, v in (e.get("per_sequence") or {}).items()]
    if long and len(comparable) > 1:
        df = pd.DataFrame(long)
        st.altair_chart(alt.Chart(df).mark_point().encode(x=alt.X("sequence", sort=None, axis=alt.Axis(labels=False)), y="value",
                                                          color="record", tooltip=["sequence", "record", "value"]), width="stretch")
        wide = df.pivot(index="sequence", columns="record", values="value")
        st.dataframe(wide, width="stretch")
else:
    st.info("Runs each selected model on the same VAL sequences with the same settings (one model in memory at a time). "
            "Keep it small: a CST sequence is ~1,000 frames (~1 min on the RTX 5050 for a detector; GlobalTrack is slower).")
    c = st.columns(3)
    did = c[0].selectbox("Dataset", ["cst2025", "antiuav2021"], format_func=lambda i: R.dataset(i)["name"])
    mod = c[1].selectbox("Modality", data.SOT_DATASETS[did])
    max_frames = c[2].number_input("Frames after start (0 = whole sequences)", 0, 5000, 200, 50)
    names = data.list_sequences(did, "val", mod)
    seqs = st.multiselect("VAL sequences", names, default=names[:2], max_selections=10)
    cands = [m for m in R.models if m["task"] != "trajectory_forecasting" and m["verification"]["inference"] == "passed"]
    ok = {m["model_id"]: compatibility(m, did, mod, R) for m in cands}
    lab = {m["model_id"]: f"{DIR_SHORT[m['direction']]} · {model_label(m)}" for m in cands if ok[m["model_id"]].ok}
    pick = st.multiselect("Models (2-4)", list(lab), format_func=lab.get, max_selections=4)
    for mid in pick:
        for w in ok[mid].warnings:
            st.caption(f"{R.model(mid)['run_id']}: {w}")
    if st.button("Run comparison", type="primary", disabled=len(pick) < 2 or not seqs):
        out = []
        bar = st.progress(0.0)
        for i, mid in enumerate(pick):
            a = get_adapter(R.model(mid)["adapter"], R)
            man = evaluate_sot(mid, did, "val", mod, seqs, max_frames=max_frames or None, registry=R, adapter=a)
            a.unload()
            out.append(man)
            bar.progress((i + 1) / len(pick), text=f"{mid}: {man['status']}")
        st.session_state["fresh_compare"] = out
    out = st.session_state.get("fresh_compare")
    if out:
        exps = {e["experiment_id"]: e for e in R.studio_experiments()}
        recs = [exps.get(man["experiment_id"]) for man in out if man["experiment_id"] in exps]
        base = recs[0]
        rows = []
        for e in recs:
            comp = proto_compat(e, base, R.protocols) if e is not base else {"verdict": "baseline", "reasons": []}
            pdlt = paired_delta(e, base, n_boot=2000) if e is not base else None
            rows.append({"model": e["model_ids"][0], "status": e["status"], "SA": e["metrics"].get("mSA", e["metrics"].get("mSA_partial")),
                         "false alarms": e["metrics"].get("mean_false_alarm_rate"), "located": e["metrics"].get("mean_located_rate"),
                         "vs first": comp["verdict"], "paired Δ": fmt(pdlt and pdlt["mean_delta"]), "CI": fmt(pdlt and pdlt["ci95"]),
                         "record": e["experiment_id"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"{len(seqs)} sequence(s): a handful of sequences is a qualitative check, not a benchmark result.")
        long = [{"sequence": s, "SA": v, "model": e["model_ids"][0].split(":")[1]} for e in recs for s, v in (e.get("per_sequence") or {}).items()]
        if long:
            st.altair_chart(alt.Chart(pd.DataFrame(long)).mark_bar().encode(x="model", y="SA", column="sequence", color="model"))
