import altair as alt
import pandas as pd
import streamlit as st
from common import DIR_LABEL, fmt, get_registry, protocol_banner, result_context

from aerolab.test_lock import consumed_registrations

R = get_registry()
st.title("Experiment history")
st.caption("Historical records are imported verbatim from the master ledger and the report files (each names its `source_file`); studio "
           "evaluations are appended by the evaluation service. Historical values are never rewritten here.")
protocol_banner()
E = R.experiments()

with st.sidebar:
    st.markdown("**Filters**")
    dirs = st.multiselect("Direction", list(DIR_LABEL), format_func=DIR_LABEL.get, default=list(DIR_LABEL))
    kinds = st.multiselect("Kind", sorted({e["kind"] for e in E}))
    prov = st.multiselect("Provenance", sorted({e["provenance"] for e in E}))
    ds = st.multiselect("Dataset", sorted({str(e.get("dataset_id")) for e in E}))
    sp = st.multiselect("Split", sorted({str(e.get("split")) for e in E}))
    pr = st.multiselect("Protocol", sorted({str(e.get("protocol_id")) for e in E}))
    stt = st.multiselect("Status", sorted({str(e.get("status")) for e in E}))
    q = st.text_input("Search id / model / note")

def keep(e):
    return (e["direction"] in dirs and (not kinds or e["kind"] in kinds) and (not prov or e["provenance"] in prov)
            and (not ds or str(e.get("dataset_id")) in ds) and (not sp or str(e.get("split")) in sp) and (not pr or str(e.get("protocol_id")) in pr)
            and (not stt or str(e.get("status")) in stt)
            and (not q or q.lower() in (e["experiment_id"] + " ".join(e.get("model_ids") or []) + str(e.get("notes"))).lower()))
F = [e for e in E if keep(e)]

def headline(e):
    m = e.get("metrics") or {}
    for k in ("mSA", "mSA_partial", "mSA_cross_fitted", "frozen_mSA", "ade_px", "mean_ade_px", "drone_AP50", "mAP50"):
        if isinstance(m.get(k), (int, float)):
            return k, m[k]
    return None, None

df = pd.DataFrame([{"experiment_id": e["experiment_id"], "dir": DIR_LABEL[e["direction"]].split(" · ")[0], "kind": e["kind"],
                    "dataset": e.get("dataset_id"), "split": e.get("split"), "modality": e.get("modality"), "protocol": e.get("protocol_id"),
                    "metric": headline(e)[0], "value": headline(e)[1], "TEST pass": e.get("test_pass"), "status": e.get("status"),
                    "models": ", ".join(e.get("model_ids") or []), "date": e.get("end") or e.get("start")} for e in F])
st.write(f"**{len(F)}** of {len(E)} records")
sel = st.dataframe(df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", height=400)
rows = sel.selection.rows if sel and sel.selection else []
if rows:
    e = F[rows[0]]
    st.subheader(e["experiment_id"])
    st.caption(result_context(e))
    if e.get("test_pass"):
        st.warning("Recorded TEST result of a registered pass: read-only; not to be used for selection or tuning.")
    c1, c2 = st.columns(2)
    c1.markdown("**Metrics**")
    c1.json(e.get("metrics"), expanded=True)
    c2.markdown("**Provenance**")
    c2.json({k: e.get(k) for k in ("model_ids", "checkpoint_sha256", "seed", "git_commit", "registration", "source_file", "status", "start", "end",
                                   "notes", "warnings", "verdict", "variant", "experiment")}, expanded=True)
    ps = e.get("per_sequence")
    if isinstance(ps, dict) and ps:
        d = pd.DataFrame({"sequence": list(ps), "value": list(ps.values())}).sort_values("value")
        st.altair_chart(alt.Chart(d).mark_bar().encode(x=alt.X("sequence", sort=None, axis=alt.Axis(labelAngle=-60)), y="value",
                                                       tooltip=["sequence", "value"]), width="stretch")
        st.caption(f"Per-sequence values ({len(ps)}), sorted. Lowest: " + ", ".join(f"{k} {fmt(v)}" for k, v in d.head(5).values))
    if e.get("curve"):
        st.markdown("**Descriptive curve**")
        st.dataframe(pd.DataFrame(e["curve"]).T, width="stretch")
    if e.get("ledger_row"):
        with st.expander("Original ledger row (verbatim)"):
            st.json(e["ledger_row"])

with st.expander("Consumed TEST registrations (the lock)"):
    for x in consumed_registrations():
        st.write(f"- **{x['registration']}** — `{x['ledger']}` · {x['n_lines']} line(s)" + ("" if x["exists"] else " · ledger missing"))
