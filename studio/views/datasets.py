import pandas as pd
import streamlit as st
from common import draw_frame, get_registry, protocol_banner

from aerolab import data

R = get_registry()
st.title("Datasets")
st.caption("Registered datasets and splits. Nothing is copied: the registry points at the established locations and checks they exist.")
protocol_banner()

df = pd.DataFrame([{"id": d["id"], "name": d["name"], "modalities": ", ".join(d.get("modalities", [])),
                    "sequences per split": ", ".join(f"{k} {v}" for k, v in d.get("sequence_counts", {}).items()) or "—",
                    "roots present": all(d["status"]["roots"].values()) if d["status"]["roots"] else None,
                    "TEST status": d.get("test_split_status")} for d in R.datasets])
st.dataframe(df, hide_index=True, width="stretch")

did = st.selectbox("Dataset", [d["id"] for d in R.datasets], format_func=lambda i: R.dataset(i)["name"])
d = R.dataset(did)
c1, c2 = st.columns(2)
with c1:
    st.write({k: d.get(k) for k in ("provenance", "paper", "label_format", "image_size")})
    st.markdown("**Roots**")
    st.dataframe(pd.DataFrame([{"root": k, "path": v, "exists": d["status"]["roots"].get(k)} for k, v in d["roots"].items()]),
                 hide_index=True, width="stretch")
with c2:
    st.markdown("**Splits**")
    st.dataframe(pd.DataFrame([{"split": k, "path": v, "exists": d["status"]["splits"].get(k), "sequences": d.get("sequence_counts", {}).get(k)}
                               for k, v in d["splits"].items()]), hide_index=True, width="stretch")
    st.markdown("**Manifests**")
    st.dataframe(pd.DataFrame([{"manifest": k, "exists": v} for k, v in d["status"]["manifests"].items()]), hide_index=True, width="stretch")
for x in d.get("limitations", []):
    st.warning(x)
users = sorted({m for e in R.experiments() if e.get("dataset_id") == did for m in (e.get("model_ids") or [])})
st.caption(f"{len([e for e in R.experiments() if e.get('dataset_id') == did])} recorded results on this dataset; {len(users)} model ids.")

if did in data.SOT_DATASETS:
    st.subheader("Browse examples (TRAIN / VAL)")
    cc = st.columns(3)
    split = cc[0].selectbox("Split", ["val", "train"])
    mod = cc[1].selectbox("Modality", list(data.SOT_DATASETS[did]))
    names = data.list_sequences(did, split, mod)
    if names:
        name = cc[2].selectbox(f"Sequence ({len(names)})", names)
        seq = data.load_sequence(did, split, name, mod)
        t = st.slider("Frame", 0, seq.n_frames - 1, seq.start_frame)
        gt = seq.box_xywh(t)
        st.image(draw_frame(seq.frame_path(t), gt_xywh=gt, label=f"{name} frame {t}", gt_absent=gt is None), width=700)
        ex = pd.Series([int(e) for e in seq.exist])
        st.caption(f"{seq.n_frames} frames · target visible on {int(ex.sum())} · absent on {int((ex == 0).sum())} · start frame {seq.start_frame}"
                   + (f" · CST attributes: {', '.join(k for k, v in seq.cst_attributes.items() if any(v))}" if getattr(seq, "cst_attributes", None) else ""))
        st.line_chart(pd.DataFrame({"target visible": ex}), height=120)
    else:
        st.warning("No sequences found at the registered path.")
elif did == "trackc_trajectories":
    seqs = data.list_sequences(did, "val")
    st.caption(f"{len(seqs)} VAL trajectories (TRAIN also available; TEST sealed by tc_data).")
    st.dataframe(pd.DataFrame({"key": seqs[:200]}), hide_index=True, width="stretch")
else:
    st.info("Frame-level browsing is implemented for the sequence datasets (Anti-UAV, CST) and Track C trajectories.")
