import pandas as pd
import streamlit as st
from common import get_registry

from aerolab import health

R = get_registry()
st.title("Artifact & environment health")
st.caption("Read-only checks. Nothing on this page deletes, moves or rewrites an artifact; cleanup is a separate, deliberate process.")

env = health.environment()
c = st.columns(5)
c[0].metric("CUDA", "available" if env.get("cuda_available") else "not available", help=env.get("cuda_error"))
c[1].metric("GPU", env.get("gpu") or "—")
c[2].metric("GPU memory free", f"{env.get('gpu_mem_free_gb', '—')} / {env.get('gpu_mem_total_gb', '—')} GB")
c[3].metric("Disk free (A:)", f"{env['disk_free_gb']} / {env['disk_total_gb']} GB")
c[4].metric("Python", env["python"])
st.dataframe(pd.DataFrame([{"package": k, "version": v or "NOT INSTALLED"} for k, v in env["packages"].items()]), hide_index=True, width="content")
if not env.get("cuda_available"):
    st.warning("No CUDA: detector and GlobalTrack inference fall back to CPU (correct but slow); forecasters are CPU-only anyway.")
if not env["packages"].get("tensorrt"):
    st.caption("TensorRT is not importable in this venv: TensorRT engines (aerotrack.deploy) are registered but not runnable from the studio.")

res = health.checks(R)
st.subheader(f"Checks — {res['counts']['ok']} ok · {res['counts']['warn']} warnings · {res['counts']['fail']} failures")
for it in res["items"]:
    icon = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}[it["status"]]
    with st.expander(f"{icon} {it['check']} — {it['detail']}", expanded=it["status"] == "fail"):
        if it["rows"]:
            st.dataframe(pd.DataFrame(it["rows"]), hide_index=True, width="stretch")
        else:
            st.caption("nothing to list")

st.subheader("Recovery backup verification")
st.write("Re-hashes all 5,725 files of `results/artifacts/hpc_recovery_20261013/` against the HPC-side SHA-256 lists with the existing "
         "`scripts/hpc/recovery_manifest.py` (outputs go to `results/runs/studio/health/recovery_check/`; the backup is only read). About 1 minute.")
if st.button("Verify the recovery backup now"):
    with st.spinner("hashing 17 GB…"):
        r = health.recovery_check()
    (st.success if r["returncode"] == 0 else st.error)(f"exit {r['returncode']}")
    st.json({k: v for k, v in (r["summary"] or {}).items() if k != "registered_hash_matches"})
    st.code(r["stdout"][-1500:])
st.caption("Rebuild the registry after adding or changing checkpoints: `venv\\Scripts\\python.exe scripts\\build_workspace_registry.py`; "
           "verify adapters: `venv\\Scripts\\python.exe scripts\\verify_adapters.py`.")
