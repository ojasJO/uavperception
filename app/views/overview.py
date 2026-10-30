import pandas as pd
import streamlit as st
from common import DIR_LABEL, badge, fmt, get_registry, protocol_banner, result_context

R = get_registry()
st.title("AeroTrack Research Studio")
st.write("One workspace, two scientific identities. Results from the two directions are shown separately; a number is only "
         "compared with another when their protocols match (see **Compare**).")
protocol_banner()

summ = R.summary
c = st.columns(5)
c[0].metric("Model entries", len(R.models), help="one entry = one checkpoint identity (family, run, checkpoint tag); see Models")
c[1].metric("Checkpoint files", summ.get("files", {}).get("total"), help="physical files incl. backup copies and detection caches")
c[2].metric("Integrity verified", sum(m["integrity_status"] == "verified_against_record" for m in R.models),
            help="SHA-256 equals an independent record (run metadata, frozen config, registry, HPC hash)")
c[3].metric("Load + inference verified", sum(m["verification"]["inference"] == "passed" for m in R.models),
            help="loaded by its adapter and ran a tiny VAL inference (scripts/verify_adapters.py)")
c[4].metric("Result records", len(R.experiments()))

exps = {e["experiment_id"]: e for e in R.experiments()}
for d in ("aerotrack", "antiuav_repro"):
    D = R.directions[d]
    with st.container(border=True):
        st.subheader(D["name"])
        st.markdown(badge(d) + f"  \n**Question.** {D['question']}  \n*{D['nature']}*")
        ms = [m for m in R.models if m["direction"] == d]
        st.caption(f"{len(ms)} model entries in {len({m['family_id'] for m in ms})} families · "
                   f"{sum(m['status'] == 'registered' for m in ms)} registered · README `{D['readme']}`")
        for f in [f for f in R.findings.get("findings", []) if f["direction"] == d]:
            with st.expander(f"**{f['title']}** — {f['status']}"):
                st.write(f["statement"])
                rows = []
                for eid in f["evidence"]:
                    e = exps.get(eid)
                    if e is None:
                        rows.append({"record": eid, "value": "record not found", "context": ""})
                        continue
                    m = e.get("metrics") or {}
                    key = next((k for k in ("mSA", "ade_px", "frozen_mSA", "mSA_cross_fitted") if k in m), None)
                    rows.append({"record": eid, "model / source": ", ".join(e.get("model_ids") or []) or "—",
                                 "metric": key, "value": fmt(m.get(key)) if key else "—",
                                 "extra": fmt(m.get("mSA_ci95")) if m.get("mSA_ci95") else (f"Δ {fmt(m.get('delta'))} CI {fmt(m.get('delta_ci95'))}" if "delta" in m else ""),
                                 "context": result_context(e)})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption(f"Source: `{f['source']}` — values read from the records above, not typed.")
        lim = [x for x in R.findings.get("limitations", []) if x["direction"] == d]
        if lim:
            st.markdown("**Known limitations**  \n" + "  \n".join(f"- {x['text']} (`{x['source']}`)" for x in lim))

with st.container(border=True):
    st.subheader("Shared benchmarks, protocols and external baselines")
    st.write("The CST Anti-UAV paper (arXiv:2507.23473) is a **benchmark**: its protocol scores Direction A's X1 / X2 / CF, and its 21 "
             "released baselines are reproduced from the authors' predictions. It is not Direction B.")
    st.dataframe(pd.DataFrame([{"protocol": p["id"], "name": p["name"], "task": p["task"], "absent frames": p["absent_frame_convention"]}
                               for p in R.protocols.values()]), hide_index=True, width="stretch")

st.subheader("Recent studio evaluations")
recent = sorted(R.studio_experiments(), key=lambda e: e.get("start") or "", reverse=True)[:8]
if recent:
    st.dataframe(pd.DataFrame([{"id": e["experiment_id"], "model": e["model_ids"][0], "dataset": e["dataset_id"], "split": e["split"],
                                "protocol": e["protocol_id"], "status": e["status"],
                                "result": fmt((e["metrics"] or {}).get("mSA", (e["metrics"] or {}).get("mSA_partial", (e["metrics"] or {}).get("mean_ade_px"))))}
                               for e in recent]), hide_index=True, width="stretch")
else:
    st.info("No studio evaluations yet — run one from **Inference**.")
st.caption(f"Registry generated {R.models_doc.get('generated')} by `{R.models_doc.get('generator')}` at git {str(R.models_doc.get('git_head'))[:10]}.")
