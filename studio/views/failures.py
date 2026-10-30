import altair as alt
import pandas as pd
import streamlit as st
from common import draw_frame, fmt, get_registry, protocol_banner

from aerolab import data
from aerolab import failures as FA

R = get_registry()
st.title("Failure analysis")
st.caption("Descriptive categories computed from ground truth and recorded per-frame predictions (VAL only). A category says what "
           "happened in a frame; it is not a causal explanation.")
protocol_banner()
tab1, tab2, tab3 = st.tabs(["Frame categories", "Same frame, several models", "Recorded per-sequence weak spots & P1 trade-off"])


@st.cache_data(show_spinner=False)
def sources():
    return [{k: (str(v) if k == "path" else v) for k, v in s.items()} for s in FA.prediction_sources()]


S = sources()
with tab1:
    if not S:
        st.info("No per-frame predictions yet: run something on the Inference page (Direction B saved VAL predictions appear automatically).")
    else:
        c = st.columns(3)
        src_names = sorted({s["source"] for s in S})
        src = c[0].selectbox("Prediction source", src_names)
        seqs = [s for s in S if s["source"] == src]
        sel = c[1].selectbox("Sequence", [s["sequence"] for s in seqs])
        s = next(x for x in seqs if x["sequence"] == sel)
        seq = data.load_sequence(s["dataset_id"], s["split"], s["sequence"], s["modality"] or "IR")
        p = FA.load_prediction(s)
        table = FA.frame_table(p["res_xywh"], p["scores"], seq.exist, seq.gt_rect, p["start_frame"], p["processed"])
        sm = FA.summary(table)
        st.caption(f"model `{s['model_id']}` · {s['dataset_id']} / {s['split']} / {s['modality']} · {sm['n_scored']} scored frames "
                   f"({sm['n_visible']} visible, {sm['n_absent']} absent)")
        st.dataframe(pd.DataFrame([{"category": k, "frames": sm[k]} for k in FA.CATEGORIES]), hide_index=True, width="content")
        st.dataframe(pd.DataFrame([{"size bin (GT diagonal)": k, **v} for k, v in sm.items() if k.startswith("size_")]), hide_index=True)
        cat = c[2].selectbox("Show examples of", FA.CATEGORIES + tuple(f"size_{n}" for n, _, _ in FA.SIZE_BINS))
        ex = [f for f in table if f["scored"] and f[cat]]
        st.write(f"{len(ex)} frame(s) in **{cat}**")
        if ex:
            step = max(1, len(ex) // 6)
            cols = st.columns(3)
            for i, f in enumerate(ex[::step][:6]):
                b = f["box"]
                img = draw_frame(seq.frame_path(f["frame"]), gt_xywh=seq.box_xywh(f["frame"]),
                                 pred_xyxy=None if b is None else [b[0], b[1], b[0] + b[2], b[1] + b[3]],
                                 label=f"frame {f['frame']} IoU {fmt(f['iou'])} score {fmt(f['score'])}", gt_absent=not f["exist"], scale=0.6)
                cols[i % 3].image(img, width="stretch")
with tab2:
    if S:
        keyset = sorted({(s["dataset_id"], s["split"], s["modality"], s["sequence"]) for s in S})
        k = st.selectbox("Sequence with predictions", keyset, format_func=lambda x: f"{x[0]} / {x[1]} / {x[2]} / {x[3]}")
        same = [s for s in S if (s["dataset_id"], s["split"], s["modality"], s["sequence"]) == k]
        pick = st.multiselect("Sources", [s["source"] for s in same], default=[s["source"] for s in same][:3], max_selections=4)
        if pick:
            seq = data.load_sequence(k[0], k[1], k[3], k[2] or "IR")
            preds = {src: FA.load_prediction(next(s for s in same if s["source"] == src)) for src in pick}
            n = min(len(p["res_xywh"]) for p in preds.values())
            t = st.slider("Frame", 0, n - 1, min(n - 1, max(p["start_frame"] for p in preds.values()) + 1))
            cols = st.columns(len(pick))
            for col, src in zip(cols, pick):
                r = preds[src]["res_xywh"][t]
                b = r if isinstance(r, list) and len(r) == 4 else None
                col.image(draw_frame(seq.frame_path(t), gt_xywh=seq.box_xywh(t), pred_xyxy=None if b is None else [b[0], b[1], b[0] + b[2], b[1] + b[3]],
                                     label=src.split(":", 1)[1][:40], gt_absent=seq.box_xywh(t) is None, scale=0.6), width="stretch")
            st.caption("Different sources may use different frame budgets (studio runs can be truncated).")
with tab3:
    st.markdown("**Weakest sequences of recorded results (read-only).** TEST records are shown as recorded evidence only.")
    recs = [e for e in R.experiments() if isinstance(e.get("per_sequence"), dict) and e["per_sequence"] and e.get("protocol_id", "").startswith("sot")]
    lab = {e["experiment_id"]: f"{e['experiment_id']} ({e.get('dataset_id')}/{e.get('split')})" for e in recs}
    pick = st.multiselect("Records", list(lab), format_func=lab.get,
                          default=[x for x in ("cst:x2:v3_n__causal_g4_cst_s0_e20:ste", "cst:x2:v3_n__causal_g4_cst_s1_e20:ste",
                                               "cst:x2:v3_n__causal_g4_cst_s2_e20:ste") if x in lab])
    if pick:
        d = pd.DataFrame({i: next(e for e in recs if e["experiment_id"] == i)["per_sequence"] for i in pick})
        d["mean"] = d.mean(axis=1)
        d["spread across records"] = d[pick].max(axis=1) - d[pick].min(axis=1)
        st.dataframe(d.sort_values("mean").head(15), width="stretch")
        st.caption("A sequence that is weak for one seed only (large spread) is a seed-specific failure (e.g. jungle_18 in X1), not a property of the method.")
    st.markdown("**P1 presence gate: false alarms vs misses (CST VAL, exploratory)**")
    p1 = [e for e in R.experiments() if e["experiment_id"].startswith("cst:p1:") and e.get("curve")]
    if p1:
        rows = []
        for e in p1:
            for gate, v in e["curve"].items():
                rows.append({"run": e["experiment_id"].split(":")[-1], "gate": gate, "variant": gate.split("_")[0], "tau": float(gate.split("_")[1]),
                             "false-alarm rate": v.get("false_alarm_rate"), "visible miss rate": v.get("visible_no_box_rate"),
                             "mSA (full-VAL curve)": v.get("mSA_cross_fitted_ste")})
        d = pd.DataFrame(rows)
        st.altair_chart(alt.Chart(d).mark_line(point=True).encode(x=alt.X("false-alarm rate", scale=alt.Scale(zero=False)),
                                                                  y=alt.Y("visible miss rate", scale=alt.Scale(zero=False)),
                                                                  color="run", strokeDash="variant", order="tau",
                                                                  tooltip=["run", "gate", "false-alarm rate", "visible miss rate", "mSA (full-VAL curve)"]),
                        width="stretch")
        st.caption(f"Registered verdict: {p1[0].get('verdict')} — the gate moves along a trade-off curve; it does not remove errors.")
