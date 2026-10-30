"""AeroTrack Flight Deck -- the local research laboratory for both research directions.

    studio.bat                                  (Windows launcher; uses venv\\; http://localhost:8501)
    venv\\Scripts\\python.exe -m streamlit run app\\app.py --server.address localhost

Primary: Flight Deck · Test Bench · Live Comparison · Results & Failures · Model Vault.
Workspace tools (secondary): the registry table, datasets, single-model inference, recorded-record comparison, experiment history,
failure tables, health. Writes only results/runs/studio/ (experiment outputs, caches). TEST splits are never evaluated; recorded TEST
predictions can be replayed read-only.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

st.set_page_config(page_title="AeroTrack Flight Deck", layout="wide")

PAGES = Path(__file__).resolve().parent / "views"   # not "pages/": that name triggers Streamlit's legacy page discovery on direct URLs
nav = st.navigation({
    "Flight Deck": [st.Page(PAGES / "flight_deck.py", title="Flight Deck", default=True),
                    st.Page(PAGES / "test_bench.py", title="Test Bench"),
                    st.Page(PAGES / "live.py", title="Live Comparison"),
                    st.Page(PAGES / "results.py", title="Results & Failures"),
                    st.Page(PAGES / "vault.py", title="Model Vault")],
    "Workspace tools": [st.Page(PAGES / "overview.py", title="Workspace overview"),
                        st.Page(PAGES / "models.py", title="Registry table"),
                        st.Page(PAGES / "datasets.py", title="Datasets"),
                        st.Page(PAGES / "inference.py", title="Single-model inference"),
                        st.Page(PAGES / "compare.py", title="Recorded comparison"),
                        st.Page(PAGES / "experiments.py", title="Experiment history"),
                        st.Page(PAGES / "failures.py", title="Failure tables"),
                        st.Page(PAGES / "health.py", title="Health")],
})
with st.sidebar:
    st.markdown("**AeroTrack Flight Deck**")
    st.caption("A · AeroTrack multi-source system  \nB · Anti-UAV (2021) paper reproduction")
    st.caption("TEST locked · recorded TEST predictions replay read-only")
    try:
        from aerolab import jobs
        act = jobs.manager().active()
        if act:
            st.caption(f"{len(act)} run(s) in progress — see Live Comparison")
    except Exception:  # noqa: BLE001
        pass
nav.run()
