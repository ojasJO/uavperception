"""The registered C1 / C1c adoption rule as code: which learned forecaster (if any) Track C's downstream experiments use per domain.

C1  (preregistration/C1_trajectory_forecasting.json):  adopt the pooled model in a domain iff H1 SUPPORTED and H2 not LOSES.
C1c (preregistration/C1c_domain_aware_forecasting.json): offer the domain's C1c model iff H1c SUPPORTED and (H2c SUPPORTED or
     C1's H1 was not SUPPORTED there); C1c takes precedence where it offers a model.
Used by tc_c3.py (fste_learned), tc_c1b.py (learned arm) and the CF final pass, as their pre-run amendments state. Reads only the
committed reports; returns None for a domain where neither rule offers a learned forecaster.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REPORTS = REPO / "experiments" / "aerotrack_temporal" / "reports"
RUN = REPO / "runs" / "aerotrack_temporal" / "c1"
SEEDS = (0, 1, 2)


def _rd(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def offered(dom, c1="load", c1c="load"):
    """-> None or {"source", "names" (3 seeds), "registry", "dom_tok", "domain"}. c1 / c1c: report dicts (tests) or "load"."""
    c1 = _rd(REPORTS / "C1_trajectory_forecasting.json") if c1 == "load" else c1
    c1c = _rd(REPORTS / "C1c_domain_aware_forecasting.json") if c1c == "load" else c1c
    c1d = (c1 or {}).get("domains", {}).get(dom)
    c1_h1 = c1d["H1_point"]["verdict"] if c1d else None
    if c1c and dom in c1c.get("domains", {}):
        d = c1c["domains"][dom]
        if d["H1c_point"]["verdict"] == "SUPPORTED" and (d["H2c_vs_pooled"]["verdict"] == "SUPPORTED" or c1_h1 != "SUPPORTED"):
            arm = d["arm"]
            names = [f"c1c_tfm_domtok_s{s}" if arm == "domtok" else f"c1c_tfm_perdom_{dom}_s{s}" for s in SEEDS]
            return {"source": "C1c", "arm": arm, "names": names, "registry": "learned_registry_c1c.json", "dom_tok": arm == "domtok", "domain": dom}
    if c1d and c1_h1 == "SUPPORTED" and c1d["H2_uncertainty"]["verdict"] != "LOSES":
        arch = c1["chosen_architecture"]
        return {"source": "C1", "arm": "pooled", "names": [f"c1_{arch}_abs_s{s}" for s in SEEDS], "registry": "learned_registry.json",
                "dom_tok": False, "domain": dom}
    return None


def summary(domains=("antiuav_IR", "antiuav_RGB", "cst")):
    return {d: (lambda o: None if o is None else {k: o[k] for k in ("source", "arm", "names")})(offered(d)) for d in domains}


if __name__ == "__main__":
    print(json.dumps(summary(), indent=1))
