"""Historical evidence used by the Flight Deck, read from the committed records at call time (never typed into the UI).

Each function returns values with the file they come from. Recomputations (paired deltas) use aerolab.compare.paired_delta, i.e.
the same sequence bootstrap (B = 10,000, default_rng(0)) as the reports; tests pin them to the reported values.
"""
import json

import numpy as np

from .analysis import seed_summary
from .compare import paired_delta
from .paths import REPO

X2_OFFICIAL = "experiments/cst_anti_uav/reports/X2_test_official.json"
P1_FILE = "experiments/cst_anti_uav/reports/explore/P1_presence_gate_VAL.json"


def _j(rel):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def _rec(R, eid):
    try:
        return R.experiment(eid)
    except KeyError:
        return None


def x2_seed_table(R, tracker="ste") -> dict:
    """per-seed CST TEST mSA (official toolkit) of the temporal and single-frame arms + seed summaries"""
    out = {}
    for arm, tag in (("temporal causal_g4", "causal_g4"), ("single-frame (ttt)", "ttt")):
        vals, ids = {}, {}
        for s in (0, 1, 2):
            e = _rec(R, f"cst:x2:v3_n__{tag}_cst_s{s}_e20:{tracker}")
            if e:
                vals[s] = e["metrics"]["mSA"]
                ids[s] = e["experiment_id"]
        out[arm] = {"per_seed": vals, "records": ids, "summary": seed_summary(vals)}
    out["source"] = "results/registry/experiments.jsonl <- experiments/cst_anti_uav/reports/X2_test_official/official_scores.json"
    return out


def x2_tests() -> dict:
    d = _j(X2_OFFICIAL)
    return {"T1": d["T1"], "T2": d["T2"], "T3": d["T3"], "source": X2_OFFICIAL}


def x2_minus_x1(R, tracker="ste") -> dict | None:
    """descriptive: per-sequence mean of the three temporal seeds minus X1 (FINAL_REPORT §2: +0.23 [-0.66, +1.14], 32 / 28)"""
    seeds = [_rec(R, f"cst:x2:v3_n__causal_g4_cst_s{s}_e20:{tracker}") for s in (0, 1, 2)]
    x1 = _rec(R, f"cst:x1:{tracker}")
    if not all(seeds) or not x1:
        return None
    keys = sorted(x1["per_sequence"])
    mean = {k: float(np.mean([s["per_sequence"][k] for s in seeds])) for k in keys}
    pd_ = paired_delta({"per_sequence": mean}, {"per_sequence": x1["per_sequence"]})
    pd_["source"] = "recomputed from the per-sequence values of cst:x2:*causal_g4*:ste and cst:x1:ste"
    return pd_


def p1() -> dict:
    """P1 presence gate (CST VAL, exploratory): trade-off curve per run and gate variant; the registered verdict"""
    d = _j(P1_FILE)
    runs = {}
    for run, r in d["runs"].items():
        pts = []
        for key, v in r["curve_descriptive"].items():
            variant, tau = key.rsplit("_", 1)
            pts.append({"run": run, "variant": variant, "tau_out": float(tau), "false_alarm_rate": v["false_alarm_rate"],
                        "visible_miss_rate": v["visible_no_box_rate"], "mSA_cross_fitted": v["mSA_cross_fitted_ste"]})
        runs[run] = {"curve": pts, "baseline_mSA": r["baseline_cross_fitted_mSA"], "gated_mSA": r["gated_cross_fitted_mSA"],
                     "paired": r["gated_minus_frozen_paired"], "chosen": r["chosen_gate_per_half"]}
    return {"verdict": d["verdict"], "three_run_mean_delta": d["three_run_mean_delta"], "registration": d["registration"], "note": d["note"],
            "runs": runs, "source": P1_FILE}


def p1_headline() -> dict:
    """the trade-off quoted in the P1 commit: run s0, instant gate, tau 0 vs 0.7"""
    r = _j(P1_FILE)["runs"]["v3_n__causal_g4_cst_s0_e20"]["curve_descriptive"]
    a, b = r["instant_0.0"], r["instant_0.7"]
    return {"fa_from": a["false_alarm_rate"], "fa_to": b["false_alarm_rate"], "miss_from": a["visible_no_box_rate"], "miss_to": b["visible_no_box_rate"],
            "mSA_from": a["mSA_cross_fitted_ste"], "mSA_to": b["mSA_cross_fitted_ste"], "run": "v3_n__causal_g4_cst_s0_e20", "gate": "instant, tau_out 0.7",
            "source": P1_FILE}


def antiuav_reproduction(R) -> list[dict]:
    rows = []
    for m, pm in (("normal_ir", "normal"), ("dfsc_ir", "DFSC")):
        for sp in ("val", "test"):
            ours = _rec(R, f"ledger:eval:anti_uav/repro/{m}/seed0/{sp}")
            paper = _rec(R, f"antiuav:paper:{pm}:{sp}:IR")
            if not ours or not paper:
                continue
            o, p = ours["metrics"]["mSA"], paper["metrics"]["mSA"]
            d = o - p
            rows.append({"model": pm, "run": m, "split": sp, "paper_mSA": p, "ours_mSA": o, "delta": d,
                         "verdict": "reproduced" if abs(d) <= 1 else ("approximately reproduced" if abs(d) <= 3 else "not reproduced"),
                         "ours_record": ours["experiment_id"], "ours_source": ours["source_file"], "paper_record": paper["experiment_id"],
                         "paper_source": paper["source_file"], "test_pass": ours.get("test_pass")})
    return rows


def cst_baseline_reproduction(R) -> list[dict]:
    rows = []
    for e in R.experiments():
        if e.get("kind") != "released_baseline":
            continue
        name = e["experiment_id"].split(":", 2)[2]
        paper = _rec(R, f"cst:paper:{name}")
        o = e["metrics"]["mSA"]
        rows.append({"tracker": name, "released_predictions_mSA": o, "paper_mSA": (paper or {}).get("metrics", {}).get("mSA"),
                     "delta": None if not paper else o - paper["metrics"]["mSA"], "record": e["experiment_id"], "source": e["source_file"]})
    return sorted(rows, key=lambda r: -(r["released_predictions_mSA"] or 0))


def trackc_verdicts(R) -> list[dict]:
    return [e for e in R.experiments() if e.get("kind") == "hypothesis_verdict" and e.get("protocol_id", "").startswith("forecast")]


def deploy_records(R) -> list[dict]:
    return [e for e in R.experiments() if e.get("protocol_id") == "deploy.latency.v1"]
