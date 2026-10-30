import json

import numpy as np
import streamlit as st
from common import get_registry, protocol_banner
from fd import category_label, go_test_bench, short_name

from aerolab import benchmark, evidence, flightdeck, forecasting
from aerolab.categories import CATEGORIES, ORDER, featured
from aerolab.paths import REPO

R = get_registry()


def rec(eid):
    try:
        return R.experiment(eid)
    except KeyError:
        return None


def val(eid, k="mSA"):
    e = rec(eid)
    v = ((e or {}).get("metrics") or {}).get(k)
    return v


st.title("AeroTrack Flight Deck")
st.markdown("A local research laboratory for **two research directions**: choose an experiment, pick real sequences by scenario, run "
            "compatible models on this machine, watch their predictions frame by frame and compare them under one protocol.")
protocol_banner()

c = st.columns(2, gap="large")
with c[0], st.container(border=True):
    st.markdown("### Direction A — AeroTrack multi-source system")
    st.caption("Temporal-window YOLO11n detector + STE state estimator (absent-aware) · Track C forecasters · CF final causal system · "
               "scored under the CST Anti-UAV protocol (arXiv:2507.23473)")
    x2 = {s: val(f"cst:x2:v3_n__causal_g4_cst_s{s}_e20:ste") for s in (0, 1, 2)}
    sf = {s: val(f"cst:x2:v3_n__ttt_cst_s{s}_e20:ste") for s in (0, 1, 2)}
    k = st.columns(3)
    k[0].metric("X1 · CST TEST mSA", f"{val('cst:x1:ste'):.2f}", help="record cst:x1:ste (official toolkit, single registered pass)")
    k[1].metric("X2 temporal · seed mean", f"{np.mean(list(x2.values())):.2f}",
                help="records cst:x2:*causal_g4*:ste, seeds " + " / ".join(f"{v:.2f}" for v in x2.values()))
    k[2].metric("Single-frame control · mean", f"{np.mean(list(sf.values())):.2f}",
                help="records cst:x2:*ttt*:ste, seeds " + " / ".join(f"{v:.2f}" for v in sf.values()))
    t3 = json.loads((REPO / "experiments/cst_anti_uav/reports/X2_test_official.json").read_text(encoding="utf-8"))["T3"]["temporal_minus_single_frame_seed_mean"]
    st.markdown(f"- Temporal window − single frame (T3, paired over 60 TEST sequences): **{t3['mean_delta']:+.2f} SA** "
                f"[{t3['ci95'][0]:+.2f}, {t3['ci95'][1]:+.2f}] — `X2_test_official.json`")
    dx = evidence.x2_minus_x1(R)
    st.markdown(f"- Best published CST tracker (GlobalTrack, paper Table 3): **{val('cst:paper:GlobalTrack'):.2f}** — X2 does **not** improve on X1 "
                f"({dx['mean_delta']:+.2f} [{dx['ci95'][0]:+.2f}, {dx['ci95'][1]:+.2f}], {dx['wins']} / {dx['losses']} sequences; no detectable difference)")
    st.markdown(f"- CF final system (registered TEST): Anti-UAV IR **{val('cf:test:sot:anti_uav_IR'):.2f}** · RGB **{val('cf:test:sot:anti_uav_RGB'):.2f}**"
                f" · CST **{val('cf:test:sot:cst_IR'):.2f}** mSA")
with c[1], st.container(border=True):
    st.markdown("### Direction B — Anti-UAV (2021) paper reproduction")
    st.caption("Jiang et al., arXiv:2101.08466 · GlobalTrack (QG-RCNN) fine-tuned on Anti-UAV IR, normal vs DFSC · Protocol II, Eq. 1 mSA")
    rows = []
    for m, pm in (("normal_ir", "normal"), ("dfsc_ir", "DFSC")):
        for sp in ("val", "test"):
            ours = val(f"ledger:eval:anti_uav/repro/{m}/seed0/{sp}")
            paper = val(f"antiuav:paper:{pm}:{sp}:IR")
            d = ours - paper
            rows.append({"model": pm, "split": sp, "paper": round(paper, 2), "ours": round(ours, 2), "Δ": f"{d:+.2f}",
                         "verdict (|Δ| ≤ 1)": "reproduced" if abs(d) <= 1 else ("approximately" if abs(d) <= 3 else "not reproduced")})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Verdict rule pre-registered in experiments/anti_uav (|Δ| ≤ 1 reproduced, ≤ 3 approximately). TEST is 4–5 points *above* the "
               "paper, which the rule still calls 'not reproduced'. The paper's DFSC gain was not observed here.")
    st.caption("Also reproduced (shared, CST study): the paper's Table 3 baseline scores from the authors' released predictions — see Results → G2.")

st.markdown("### Run an experiment")
grid = st.columns(4)
for i, cid in enumerate(ORDER):
    d = CATEGORIES[cid]
    with grid[i % 4], st.container(border=True, height=205):
        st.markdown(f"**{category_label(cid)}**")
        st.caption(d["question"])
        if st.button("Open in Test Bench", key=f"cat_{cid}", width="stretch"):
            go_test_bench(cid)
with grid[3], st.container(border=True, height=205):
    st.markdown("**Replay recorded predictions**")
    st.caption("X1 / X2 / released trackers on CST TEST, Direction B on Anti-UAV — read-only.")
    if st.button("Open Live Comparison", key="cat_replay", width="stretch"):
        st.session_state["live_mode"] = "Recorded predictions"
        st.switch_page("views/live.py")

st.markdown("### Featured models")
fs = [f for f in featured()]
cols = st.columns(4)
for i, f in enumerate(fs):
    with cols[i % 4], st.container(border=True, height=250):
        st.markdown(f"**{f['title']}**")
        st.caption({"aerotrack": "Direction A", "antiuav_repro": "Direction B", "shared": "external baseline"}[f["direction"]]
                   + f" · {len(f['versions'])} version(s)" + ("" if f["versions"] else " · recorded only"))
        st.caption(f["purpose"][:140] + ("…" if len(f["purpose"]) > 140 else ""))
        if st.button("Details", key=f"feat_{f['id']}", width="stretch"):
            st.session_state["vault_featured"] = f["id"]
            st.switch_page("views/vault.py")

st.markdown("### Recent experiments")
recent = sorted(flightdeck.list_experiments()[:5] + benchmark.list_benchmarks()[:3] + forecasting.list_forecasts()[:3],
                key=lambda m: m.get("completion_time") or m.get("start_time") or "", reverse=True)[:4]
if not recent:
    st.caption("No Flight Deck experiment yet — start one from the Test Bench.")
for m in recent:
    with st.container(border=True):
        cc = st.columns([5, 1])
        models = ", ".join(short_name(x["model_id"], x.get("tracker")) for x in m.get("models", m.get("results", [])))
        cc[0].markdown(f"`{m['experiment_id']}` · **{CATEGORIES.get(m.get('category'), {}).get('name', m.get('category'))}** · "
                       f"{m.get('dataset_id')} / {m.get('split')} · {models} · status **{m['status']}**")
        if cc[1].button("Open", key=f"open_{m['experiment_id']}"):
            st.session_state["results_experiment"] = m["experiment_id"]
            if m["experiment_id"].startswith("fd-"):
                st.session_state["live_mode"] = "Saved run"
                st.session_state["live_experiment"] = m["experiment_id"]
                st.switch_page("views/live.py")
            st.switch_page("views/results.py")

with st.expander("Limitations to know before interpreting anything here"):
    st.markdown(
        "- **TEST is locked.** Every TEST split has been exposed by registered passes (X1, X2, CF, Anti-UAV Protocol II, …). Nothing here "
        "evaluates TEST, tunes on it or re-runs a registered pass; recorded TEST predictions can only be replayed.\n"
        "- **Live timing is observed, not benchmarked.** Use Category E for controlled measurements (warm-up, synchronisation, percentiles). "
        "PyTorch eager on this laptop; the TensorRT deployment numbers come from a different setup.\n"
        "- **GlobalTrack never reports 'absent'**, so every target-absent frame scores 0 under Eq. 1; the AeroTrack STE can report absence.\n"
        "- **CST has no recorded frame rate**: timestamps are shown only for Anti-UAV (25 FPS, paper-explicit).\n"
        "- **Short runs** (init + N frames) use protocol `studio.sot_smoke.v1` and are never comparable with full-sequence mSA.\n"
        "- Models whose checkpoints exist only in the recovery backup, ONNX / TensorRT exports and retired families are browsable in the Vault "
        "but not runnable here (the exact blocker is shown per model).")
