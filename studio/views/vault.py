import pandas as pd
import streamlit as st
from common import DIR_LABEL, get_registry
from fd import short_name

from aerolab import categories
from aerolab.analysis import seed_summary
from aerolab.categories import CATEGORIES, DEVICES, evaluation_checks, featured, runnable, status_labels

R = get_registry()
CHECKS = evaluation_checks()
st.title("Model Vault")
st.caption("Featured models first; every registered checkpoint, seed, legacy run, export and baseline stays in the archive below. "
           "Labels never claim more than their evidence: integrity = SHA-256 equals an independent record; load / inference = "
           "tools/verify_adapters.py on this machine; evaluation = a fresh full-sequence run reproduces a recorded per-sequence value "
           "(tools/verify_evaluations.py).")
LADDER_HELP = {"registered": "the identity of a reported, pre-registered result", "integrity-verified": "SHA-256 equals an independent record",
               "hashed (no independent record)": "hashed now; nothing older to compare with", "load-verified": "loaded by its adapter on this machine",
               "inference-verified": "ran a small VAL inference here, deterministic", "evaluation-verified": "fresh full-sequence SA equals a recorded value",
               "historical results": "has imported result records", "recorded results only": "no inference adapter", "not runnable locally": "see the blocker"}

tab_f, tab_a = st.tabs(["★ Featured models", "Complete archive"])

with tab_f:
    FS = featured()
    ids = [f["id"] for f in FS]
    want = st.session_state.pop("vault_featured", None)
    if want in ids or st.session_state.get("vault_feat_radio") not in ids:
        st.session_state["vault_feat_radio"] = want if want in ids else ids[0]
    fid = st.radio("Featured model", ids, horizontal=True, key="vault_feat_radio",
                   format_func=lambda i: next(f["title"] for f in FS if f["id"] == i))
    f = next(x for x in FS if x["id"] == fid)
    st.markdown(f"## {f['title']}")
    st.markdown(f"{DIR_LABEL.get(f['direction'], f['direction'])} · used in: " + ", ".join(f"{CATEGORIES[c]['name']}" for c in f["categories"]))
    c = st.columns(3)
    c[0].markdown("**What it does**\n\n" + f["purpose"])
    c[1].markdown("**What distinguishes it**\n\n" + f["distinguishes"])
    c[2].markdown("**Why use it / what to compare**\n\n" + f["use_for"])
    if f["versions"]:
        st.markdown("#### Versions")
        rows = []
        for mid in f["versions"]:
            m = R.model(mid)
            ok, why = runnable(m)
            rows.append({"version": mid, "seed": m.get("seed"), "status": m["status"], "evidence": " · ".join(status_labels(m, R, CHECKS)),
                         "sha256": m["sha256"][:16] + "…", "size MB": round(m["size_bytes"] / 2 ** 20, 1), "devices": "/".join(DEVICES.get(m.get("adapter"), ())) or "—",
                         "runnable here": "yes" if ok else f"no — {why}", "aliases": ", ".join(m.get("aliases") or [])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if f["id"] == "x2_temporal":
            st.caption("Seed 0 is ONE checkpoint (sha256 e52a5645…) with two registered roles: AeroTrack-X1 and X2 temporal seed 0.")
    else:
        st.info("Recorded predictions only — no checkpoint was released, so this model cannot run live. Replay it in Live Comparison → Recorded predictions.")
    st.markdown("#### Evidence (read from the records)")
    ev = []
    for eid in f["evidence"]:
        try:
            e = R.experiment(eid)
        except KeyError:
            ev.append({"record": eid, "note": "record not found"})
            continue
        mt = e.get("metrics") or {}
        main = next((k for k in ("mSA", "ade_px", "mSA_cross_fitted") if mt.get(k) is not None), None)
        ev.append({"record": eid, "kind": e.get("kind"), "dataset / split": f"{e.get('dataset_id')} / {e.get('split')}", "protocol": e.get("protocol_id"),
                   "metric": main, "value": None if main is None else round(mt[main], 2), "TEST pass": "yes" if e.get("test_pass") else "no",
                   "source": e.get("source_file")})
    st.dataframe(pd.DataFrame(ev), hide_index=True, width="stretch")
    if len(f["versions"]) > 1 and f["kind"] == "tracking":
        per_seed = {}
        for mid in f["versions"]:
            m = R.model(mid)
            for e in R.experiments_for(mid):
                if e.get("protocol_id") == "sot.mSA.cst_official.v1" and e.get("split") == "test" and str(e["experiment_id"]).endswith(":ste") \
                        and e["experiment_id"].startswith("cst:x2:"):
                    per_seed[m.get("seed")] = e["metrics"]["mSA"]
        if per_seed:
            s = seed_summary(per_seed)
            st.markdown(f"**Seeds (CST TEST, STE, registered X2 pass):** " + " · ".join(f"seed {k}: {v:.2f}" for k, v in sorted(per_seed.items()))
                        + f" → mean **{s['mean']:.2f}**, SD {s['sd']:.2f}, range {s['min']:.2f}–{s['max']:.2f}. Individual seeds are kept; "
                        "no averaged checkpoint exists.")
    st.markdown("#### Known failure modes")
    for fm in f["failure_modes"]:
        st.markdown(f"- {fm['text']} — *{fm['source']}*")
    if f.get("compare_with"):
        st.markdown("**Natural comparisons:** " + ", ".join(next((x["title"] for x in FS if x["id"] == cid), cid) for cid in f["compare_with"]))
    b = st.columns(3)
    if f["versions"] and b[0].button(f"Use in Test Bench ({CATEGORIES[f['categories'][0]]['name']})", key=f"vault_tb_{fid}"):
        st.session_state["tb_category"] = f["categories"][0]
        st.switch_page("views/test_bench.py")
    if f.get("recorded_source") and b[1].button("Replay recorded predictions", key=f"vault_rec_{fid}"):
        st.session_state["live_mode"] = "Recorded predictions"
        st.switch_page("views/live.py")

with tab_a:
    st.caption(f"{len(R.models)} registry entries (one per checkpoint identity: family : run @ tag), from registry/models.json.")
    c = st.columns(6)
    q = c[0].text_input("Search", placeholder="run id, alias, sha prefix, role…", key="vault_q")
    dirs = c[1].multiselect("Direction", sorted({m["direction"] for m in R.models}), key="vault_dir", format_func=lambda d: DIR_LABEL.get(d, d))
    tasks = c[2].multiselect("Task", sorted({m["task"] for m in R.models}), key="vault_task")
    fams = c[3].multiselect("Family", sorted(R.families), key="vault_fam")
    stats = c[4].multiselect("Status", sorted({m["status"] for m in R.models}), key="vault_status")
    labs = c[5].multiselect("Evidence label", list(LADDER_HELP), key="vault_labels", help="\n".join(f"{k}: {v}" for k, v in LADDER_HELP.items()))
    c2 = st.columns(4)
    seeds = c2[0].multiselect("Seed", sorted({m.get("seed") for m in R.models if m.get("seed") is not None}), key="vault_seed")
    avail = c2[1].selectbox("Availability", ["any", "in the workspace", "backup only"], key="vault_avail")
    dev = c2[2].selectbox("Device support", ["any", "cuda", "cpu"], key="vault_dev")
    cat = c2[3].selectbox("Usable in category", ["any"] + list(categories.ORDER), key="vault_cat",
                          format_func=lambda x: x if x == "any" else f"{CATEGORIES[x]['letter']} · {CATEGORIES[x]['name']}")
    rows = []
    for m in R.models:
        labels = status_labels(m, R, CHECKS)
        text = " ".join([m["model_id"], m["run_id"], m["sha256"], m.get("role") or "", " ".join(m.get("aliases") or [])]).lower()
        if q and q.lower() not in text:
            continue
        if dirs and m["direction"] not in dirs or tasks and m["task"] not in tasks or fams and m["family_id"] not in fams:
            continue
        if stats and m["status"] not in stats or labs and not set(labs) <= set(labels) or seeds and m.get("seed") not in seeds:
            continue
        if avail == "in the workspace" and not m.get("available_in_workspace") or avail == "backup only" and m.get("available_in_workspace"):
            continue
        if dev != "any" and dev not in DEVICES.get(m.get("adapter"), ()):
            continue
        if cat != "any" and not categories.eligibility(cat, m)[0]:
            continue
        rows.append({"model_id": m["model_id"], "direction": m["direction"], "family": m["family_id"], "run": m["run_id"], "tag": m["checkpoint_tag"],
                     "seed": m.get("seed"), "status": m["status"], "evidence": " · ".join(labels), "sha256": m["sha256"][:12],
                     "MB": round(m["size_bytes"] / 2 ** 20, 1), "in workspace": m.get("available_in_workspace"),
                     "devices": "/".join(DEVICES.get(m.get("adapter"), ())) or "—"})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch", height=380)
    st.caption(f"{len(df)} of {len(R.models)} entries match.")
    if len(df):
        mid = st.selectbox("Details for", list(df["model_id"]), key="vault_detail", format_func=lambda i: f"{i} ({short_name(i)})")
        m = R.model(mid)
        ok, why = runnable(m)
        st.markdown(f"### `{mid}`")
        st.markdown(f"{DIR_LABEL.get(m['direction'], m['direction'])} · family **{R.families[m['family_id']]['name']}** · status **{m['status']}** · "
                    + " · ".join(f"`{x}`" for x in status_labels(m, R, CHECKS)))
        if m.get("role"):
            st.caption(m["role"])
        st.markdown(("runnable here" if ok else f"not runnable here — {why}") + f" · devices: {'/'.join(DEVICES.get(m.get('adapter'), ())) or '—'}")
        d = st.columns(2)
        d[0].json({k: m.get(k) for k in ("architecture", "framework", "task", "input_contract", "output_contract", "preprocessing", "training_data",
                                         "seed", "version", "papers", "registrations", "config")}, expanded=False)
        d[1].json({"sha256": m["sha256"], "size_bytes": m["size_bytes"], "integrity_status": m["integrity_status"], "recorded_hashes": m.get("recorded_hashes"),
                   "locations": m.get("locations"), "parent": m.get("parent"), "same_bytes_as": m.get("same_bytes_as"),
                   "duplicate_group": m.get("duplicate_group"), "verification": m.get("verification"),
                   "evaluation_check": CHECKS.get(mid)}, expanded=False)
        ex = R.experiments_for(mid)
        if ex:
            st.dataframe(pd.DataFrame([{"record": e["experiment_id"], "kind": e.get("kind"), "split": e.get("split"), "protocol": e.get("protocol_id"),
                                        "metrics": ", ".join(f"{k} {v:.2f}" for k, v in (e.get("metrics") or {}).items() if isinstance(v, (int, float)))[:120],
                                        "TEST pass": e.get("test_pass"), "source": e.get("source_file")} for e in ex]), hide_index=True, width="stretch")
        else:
            st.caption("No result record for this checkpoint.")
