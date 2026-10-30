import pandas as pd
import streamlit as st
from common import DIR_LABEL, STATUS_HELP, badge, fmt, get_registry, model_label, result_context

R = get_registry()
st.title("Model registry")
st.caption("Every checkpoint identity in the workspace and the HPC recovery backup. Identity = SHA-256; one file name can hide "
           "several models (e.g. `@last~hpc` = an HPC re-run with different weights under the same run id).")

with st.sidebar:
    st.markdown("**Filters**")
    dirs = st.multiselect("Research direction", list(DIR_LABEL), format_func=DIR_LABEL.get, default=list(DIR_LABEL))
    fams = st.multiselect("Model family", sorted(R.families), format_func=lambda f: R.families[f]["name"])
    tasks = st.multiselect("Task", sorted({m["task"] for m in R.models}))
    stat = st.multiselect("Status", sorted({m["status"] for m in R.models}), help="\n".join(f"{k}: {v}" for k, v in STATUS_HELP.items()))
    integ = st.multiselect("Integrity", sorted({m["integrity_status"] for m in R.models}))
    runnable = st.selectbox("Runnable here", ["any", "load + inference verified", "recorded results only / not verified"])
    tags = st.multiselect("Checkpoint tag", sorted({m["checkpoint_tag"] for m in R.models}), help="last = the evaluated checkpoint for detectors / GlobalTrack")
    q = st.text_input("Search (run id, alias, role)")

ms = [m for m in R.models if m["direction"] in dirs and (not fams or m["family_id"] in fams) and (not tasks or m["task"] in tasks)
      and (not stat or m["status"] in stat) and (not integ or m["integrity_status"] in integ) and (not tags or m["checkpoint_tag"] in tags)]
if runnable != "any":
    ok = lambda m: m["verification"]["inference"] == "passed"  # noqa: E731
    ms = [m for m in ms if ok(m) == (runnable.startswith("load"))]
if q:
    ql = q.lower()
    ms = [m for m in ms if ql in m["model_id"].lower() or ql in " ".join(m.get("aliases", [])).lower() or ql in (m.get("role") or "").lower()]

ORDER = ["registered", "validated", "experimental", "external", "legacy", "retired", "invalid", "smoke"]
ms.sort(key=lambda m: (ORDER.index(m["status"]) if m["status"] in ORDER else 99, m["direction"], m["family_id"], m["run_id"], m["checkpoint_tag"]))
df = pd.DataFrame([{"direction": DIR_LABEL[m["direction"]].split(" · ")[0], "family": m["family_id"], "run": m["run_id"],
                    "tag": m["checkpoint_tag"], "aliases": ", ".join(m.get("aliases", [])), "status": m["status"], "seed": m.get("seed"),
                    "integrity": m["integrity_status"], "load": m["verification"]["load"], "inference": m["verification"]["inference"],
                    "results": len(m["evaluations"]), "in workspace": m.get("available_in_workspace"), "MB": round(m["size_bytes"] / 1e6, 1),
                    "sha256": m["sha256"][:12], "model_id": m["model_id"]} for m in ms])
st.write(f"**{len(ms)}** of {len(R.models)} model entries")
sel = st.dataframe(df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", height=380,
                   column_config={"model_id": None})
rows = sel.selection.rows if sel and sel.selection else []
default = st.session_state.get("selected_model")
visible = {m["model_id"] for m in ms}
mid = df.iloc[rows[0]]["model_id"] if rows else (default if default in visible else (ms[0]["model_id"] if ms else None))
if mid is None:
    st.info("No model matches the filters.")
    st.stop()
st.session_state["selected_model"] = mid
m = R.model(mid)
F = R.families[m["family_id"]]

st.divider()
st.subheader(model_label(m))
st.markdown(badge(m["direction"]) + f" · family **{F['name']}** · status **{m['status']}** — {STATUS_HELP.get(m['status'], '')}")
if m.get("role"):
    st.info(m["role"])
if m["direction"] == "antiuav_repro":
    st.caption("Direction B: a reproduction of arXiv:2101.08466 (DFSC on GlobalTrack). Deviations from the paper are tagged S1-S6 in "
               "docs/ANTI_UAV_REPRODUCTION_SPEC.md; never read its numbers as AeroTrack results.")
t1, t2, t3, t4, t5 = st.tabs(["Identity & integrity", "Contracts & config", "Provenance & lineage", "Versions", "Recorded results"])
with t1:
    st.write({"model_id": m["model_id"], "sha256": m["sha256"], "size_bytes": m["size_bytes"], "primary_path": m["primary_path"],
              "integrity_status": m["integrity_status"], "verification": m["verification"], "duplicate_of / same bytes as": m.get("same_bytes_as")})
    st.dataframe(pd.DataFrame(m["locations"]), hide_index=True, width="stretch")
    if m.get("recorded_hashes"):
        st.markdown("**Independent hash records**")
        st.dataframe(pd.DataFrame(m["recorded_hashes"]), hide_index=True, width="stretch")
    if m.get("verification_evidence"):
        st.markdown("**Adapter verification evidence**")
        st.json(m["verification_evidence"], expanded=False)
    if st.button("Re-hash this checkpoint now", help="reads every registered copy and compares with the registry (read-only)"):
        from aerolab.health import verify_model
        st.json(verify_model(m))
with t2:
    st.write({k: m.get(k) for k in ("task", "architecture", "framework", "adapter", "input_contract", "output_contract", "preprocessing", "training_data")})
    st.markdown("**Run configuration**")
    st.json(m.get("config") or {}, expanded=False)
    if F.get("selection_policy"):
        st.caption(f"Selection policy: {F['selection_policy']}")
with t3:
    for pid, how in m.get("papers") or []:
        p = R.papers.get(pid, {})
        st.markdown(f"- **{p.get('title', pid)}** ({p.get('identifier')}) — used as: *{how}*; citation status: {p.get('citation_status')}")
    if not m.get("papers"):
        st.caption("No published-paper component recorded for this family (own design or derived artifact).")
    st.write({"parent": m.get("parent"), "registrations": m.get("registrations"), "family verdict": m.get("family_verdict")})
    edges = [e for e in R.lineage.get("edges", []) if mid in (e["from"], e["to"])]
    if edges:
        st.dataframe(pd.DataFrame(edges), hide_index=True, width="stretch")
with t4:
    vs = R.versions(m["family_id"])
    st.caption(f"{len(vs)} versions in {F['name']}")
    lat = R.latest(m["family_id"])
    reg = R.demonstration_version(m["family_id"])
    st.write({"latest (by file time; says nothing about quality)": lat and lat["model_id"],
              "registered / demonstration version": reg and reg["model_id"]})
    st.dataframe(pd.DataFrame([{"version": v["version"], "status": v["status"], "seed": v.get("seed"), "aliases": ", ".join(v.get("aliases", [])),
                                "sha256": v["sha256"][:12], "results": len(v["evaluations"])} for v in vs]), hide_index=True, width="stretch")
    st.caption("'Best' is only defined per protocol and split — see Compare for a protocol-matched ranking.")
with t5:
    ex = R.experiments_for(mid)
    if not ex:
        st.info("No recorded result references this checkpoint.")
    for e in ex:
        with st.expander(f"{e['experiment_id']} — {e.get('kind')}"):
            st.caption(result_context(e))
            st.json({k: e.get(k) for k in ("metrics", "registration", "source_file", "notes", "warnings", "verdict")}, expanded=False)
st.page_link("views/inference.py", label="Run this model on a sample →")
