import json

import numpy as np
import pandas as pd
import streamlit as st
from common import DIR_LABEL, badge, crop_zoom, draw_frame, fmt, get_registry, model_label, protocol_banner

from aerolab import data
from aerolab.adapters import compatibility, get_adapter
from aerolab.evaluation import evaluate_forecast, evaluate_sot
from aerolab.paths import REPO

R = get_registry()
st.title("Interactive inference")
st.caption("Pick a model version, a compatible dataset and a VAL / TRAIN sequence; the run goes through the evaluation service, so every "
           "result is saved with a manifest (results/runs/studio/evals/) and appears under Experiments.")
protocol_banner()


@st.cache_resource(max_entries=1, show_spinner="loading the model (one model is kept in memory at a time)…")
def loaded_adapter(model_id: str, sha256: str, device: str):
    m = R.model(model_id)
    return get_adapter(m["adapter"], R).load(m, device)


c = st.columns([1, 2, 3])
direction = c[0].selectbox("Research direction", ["aerotrack", "antiuav_repro"], format_func=DIR_LABEL.get)
fams = [f for f in R.families.values() if f["direction"] == direction]
fam_id = c[1].selectbox("Model family", [f["id"] for f in fams], format_func=lambda i: R.families[i]["name"])
vs = [m for m in R.versions(fam_id) if m["status"] not in ("smoke",)]
runnable = [m for m in vs if m["verification"]["inference"] == "passed"]
show_all = c[2].toggle("Show non-runnable versions too", value=False, help="checkpoints without a verified adapter run (recorded results only)")
pool = vs if show_all else (runnable or vs)
default_i = next((i for i, m in enumerate(pool) if m["status"] == "registered"), 0)
mid = c[2].selectbox("Checkpoint version", [m["model_id"] for m in pool], index=default_i if pool else None,
                     format_func=lambda i: model_label(R.model(i)))
if not mid:
    st.info("This family has no checkpoint versions.")
    st.stop()
m = R.model(mid)
st.markdown(badge(m["direction"]) + f" · `{m['model_id']}` · sha256 `{m['sha256'][:16]}…` · {m['task']}")
if m.get("role"):
    st.caption(m["role"])

if not m.get("adapter"):
    st.warning("Recorded results only: this family has no inference adapter (e.g. retired / historical / export artifacts).")
    for e in R.experiments_for(mid):
        st.write(f"- `{e['experiment_id']}` — {e.get('kind')} · {e.get('dataset_id')} / {e.get('split')} · {json.dumps(e.get('metrics'))[:200]}")
    st.stop()

task = m["task"]
dsets = ["trackc_trajectories"] if task == "trajectory_forecasting" else ["cst2025", "antiuav2021"]
c = st.columns(4)
did = c[0].selectbox("Dataset", dsets, format_func=lambda i: R.dataset(i)["name"])
split = c[1].selectbox("Split", ["val", "train"], help="TEST is locked")
mods = data.SOT_DATASETS.get(did, ("IR",))
mod = c[2].selectbox("Modality", mods) if did in data.SOT_DATASETS else None
comp = compatibility(m, did, mod, R)
if not comp.ok:
    st.error(f"Not runnable on this selection: {comp.reason}")
    st.stop()
for w in comp.warnings:
    st.warning(w)
names = data.list_sequences(did, split, mod or "IR")
if task == "trajectory_forecasting":
    own = next((d for d in ("antiuav_IR", "antiuav_RGB", "cst") if f"perdom_{d}" in m["run_id"]), None)
    if own:
        names = [n for n in names if ("cst" if n.startswith("cst_") else "antiuav_" + n.rsplit("__", 1)[-1]) == own]
        st.caption(f"Domain-isolated model: only '{own}' trajectories are offered.")
name = c[3].selectbox(f"Sequence ({len(names)})", names)

o = st.columns(4)
device = o[0].selectbox("Device", ["auto", "cuda", "cpu"], help="auto = CUDA when available; forecasters always run on CPU")
if task == "trajectory_forecasting":
    max_anchors = o[1].number_input("Max anchors", 50, 5000, 500, 50)
else:
    full = o[1].toggle("Whole sequence", value=False, help="off = only the first N frames after the start frame (protocol studio.sot_smoke.v1)")
    max_frames = None if full else o[2].number_input("Frames after start", 10, 5000, 150, 10)
    tracker = o[3].selectbox("Tracker", ["ste", "argmax"]) if m["adapter"] == "aerotrack_detector" else "globaltrack"

key = f"run::{mid}::{did}::{split}::{mod}::{name}"
if st.button("Run inference", type="primary"):
    with st.status("running…", expanded=False) as s:
        dev = "cpu" if task == "trajectory_forecasting" else device
        a = loaded_adapter(mid, m["sha256"], dev)
        if task == "trajectory_forecasting":
            man = evaluate_forecast(mid, [name], split, max_anchors=int(max_anchors), registry=R, adapter=a)
        else:
            man = evaluate_sot(mid, did, split, mod, [name], max_frames=max_frames, tracker=tracker, device=dev, registry=R, adapter=a)
        s.update(label=f"{man['status']} — {man['experiment_id']}", state="complete" if man["status"] == "completed" else "error")
    st.session_state[key] = man
man = st.session_state.get(key)
if not man:
    st.stop()
if man["status"] != "completed":
    st.error(f"Evaluation failed: {man.get('error')} — log: {man['logs_path']}")
    st.stop()
pred = json.loads((REPO / man["predictions_dir"] / f"{name}.json").read_text(encoding="utf-8"))
st.success(f"Saved as **{man['experiment_id']}** · protocol `{man['evaluation_protocol_id']}` · device {man['device']} · "
           f"checkpoint `{man['checkpoint_path']}`")
for w in man["warnings"]:
    st.caption(f"{w}")

if pred["kind"] == "forecast":
    import altair as alt
    mt = pred["metrics"]
    st.dataframe(pd.DataFrame([{"forecaster": k, "ADE px": fmt((v or {}).get("ade_px")), "FDE px": fmt((v or {}).get("fde_px")),
                                "NLL": fmt((v or {}).get("nll")), "anchors": (v or {}).get("n_samples")} for k, v in mt.items()]),
                 hide_index=True, width="stretch")
    st.caption("Forecasting metrics (pixels, lower is better) on ground-truth histories — never comparable with tracking mSA. "
               f"Classical references: {pred['classical_params_source']}.")
    ex = pred["example"]
    hm = np.array(ex["hist_m"], bool)
    hc, fc, fm = np.array(ex["hist_c"]), np.array(ex["fut_c"]), np.array(ex["fut_m"], bool)
    rows = [{"x": x, "y": y, "series": "history (GT)", "h": i} for i, (x, y) in enumerate(hc[hm])]
    rows += [{"x": x, "y": y, "series": "future (GT)", "h": i} for i, (x, y) in enumerate(fc[fm])]
    rows += [{"x": x, "y": y, "series": "learned mean", "h": i} for i, (x, y) in enumerate(np.array(ex["mean"]))]
    for k, v in ex["refs"].items():
        rows += [{"x": x, "y": y, "series": k, "h": i} for i, (x, y) in enumerate(np.array(v))]
    ch = alt.Chart(pd.DataFrame(rows)).mark_line(point=True).encode(x=alt.X("x", title="x (px)"), y=alt.Y("y", title="y (px)", sort="descending"),
                                                                     color="series", order="h")
    st.altair_chart(ch, width="stretch")
    st.caption(f"One representative anchor (frame {ex['t']}) of {pred['sequence']}; y axis inverted like the image.")
    st.stop()

per = json.loads((REPO / man["per_sample_results_path"]).read_text(encoding="utf-8"))[name]
k = st.columns(6)
k[0].metric("SA (%)", fmt(per["SA"]), help="Eq. 1 over the scored frames" + (" — PARTIAL run" if not per["full_sequence"] else ""))
k[1].metric("P@20px (%)", fmt(per["precision_20px"]))
k[2].metric("Success AUC (%)", fmt(per["success_auc"]))
k[3].metric("False-alarm rate", fmt(per["false_alarm_rate"]), help="absent frames with a reported box / absent frames")
k[4].metric("Located rate", fmt(per["located_rate"]), help="visible frames with a box / visible frames")
k[5].metric("ms / frame", fmt(sum(v for v in pred["timing_ms"].values() if v)), help=json.dumps(pred["timing_ms"]))

seq = data.load_sequence(did, split, name, mod)
n = pred["n_frames_processed"]
rows = []
for t in range(n):
    f = pred["frames"][t]
    gt = seq.box_xywh(t)
    iou = None
    if gt is not None and f["box_xyxy"] is not None:
        b = f["box_xyxy"]
        g = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]
        ix = max(0, min(b[2], g[2]) - max(b[0], g[0])) * max(0, min(b[3], g[3]) - max(b[1], g[1]))
        iou = ix / ((b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - ix + 1e-9)
    status = ("before start" if t < pred["start_frame"] else "init (GT box)" if t == pred["start_frame"] else
              ("false alarm" if f["box_xyxy"] is not None else "correct absent") if gt is None else
              ("miss" if f["box_xyxy"] is None else ("poor localisation" if iou < 0.5 else "tracked")))
    rows.append({"frame": t, "gt_visible": gt is not None, "pred_box": f["box_xyxy"] is not None, "iou": iou, "score": f["score"], "status": status})
tab = pd.DataFrame(rows)
st.bar_chart(tab["status"].value_counts(), horizontal=True, height=180)
fr = st.columns([3, 1])
with fr[1]:
    pick = st.selectbox("Jump to", ["(slider)", "first false alarm", "first miss", "worst IoU"])
    target = {"first false alarm": tab.index[tab.status == "false alarm"], "first miss": tab.index[tab.status == "miss"],
              "worst IoU": tab["iou"].dropna().sort_values().index}.get(pick)
    t0 = int(target[0]) if target is not None and len(target) else pred["start_frame"]
t = fr[0].slider("Frame", 0, n - 1, t0)
f = pred["frames"][t]
gt = seq.box_xywh(t)
img = draw_frame(seq.frame_path(t), gt_xywh=gt, pred_xyxy=f["box_xyxy"], dets=[d[:4] for d in (f.get("detections") or [])],
                 label=f"{m['run_id']} [{m['checkpoint_tag']}]  frame {t}  {rows[t]['status']}", gt_absent=gt is None)
cc = st.columns([3, 1])
cc[0].image(img, width="stretch", caption="green = ground truth · orange = prediction · cyan = other detections")
z = crop_zoom(img, f["box_xyxy"] or (None if gt is None else [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]))
if z is not None:
    cc[1].image(z, caption="3x zoom")
cc[1].write({"IoU": fmt(rows[t]["iou"]), "score": fmt(f["score"], 3), "score kind": pred.get("score_kind"), "window": pred.get("window")})
st.line_chart(tab.set_index("frame")[["iou"]], height=150)
with st.expander("Per-frame table and run parameters"):
    st.dataframe(tab, hide_index=True, width="stretch")
    st.json({k: pred.get(k) for k in ("params", "params_source", "preprocessing", "timing_ms", "device", "checkpoint_path")})
