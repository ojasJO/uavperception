#!/usr/bin/env python
"""Host checks of amendment A1, computed from result files only -> experiments/hpc/reports/host_checks.json.

HC1  same OMEN-trained weights (v2_stock_s0_e20, and the v1 reference gen_temporal_fp_g4_full_e20) evaluated on the OMEN (bf16) and
     on the HPC (fp16): pass if |delta| <= 0.005 AP50 for CST, Anti-UAV, pooled.
HC2  descriptive: HPC-trained v2_stock_s0_e20 vs OMEN-trained (both evaluated on the HPC) = training host + run-to-run.
HC3  Track A normal_ir VAL mSA, V100 vs OMEN: pass if |delta| <= 0.05.
    venv\\Scripts\\python.exe tools\\hpc\\host_checks.py   (after tools/hpc/pull_results.py)
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KEYS = ("cst_drone_AP50", "anti_uav_drone_AP50", "drone_AP50")


def rows(rel):
    p = REPO / rel
    return {r["run"]: r for r in json.loads(p.read_text(encoding="utf-8"))["rows"] if r["set"] == "val"} if p.exists() else {}


def main():
    omen, hpc = rows("experiments/aerotrack_v2/reports/v2_results.json"), rows("experiments/aerotrack_v2/hpc/reports/v2_results.json")
    out = {"HC1": [], "HC2": None, "HC3": None}
    for o_name, h_name in (("v2_stock_s0_e20", "omen_v2_stock_s0_e20"), ("gen_temporal_fp_g4_full_e20", "gen_temporal_fp_g4_full_e20")):
        if o_name in omen and h_name in hpc:
            assert omen[o_name]["weights_sha256"] == hpc[h_name]["weights_sha256"], "HC1 must compare identical weights"
            d = {k: hpc[h_name]["metrics"][k] - omen[o_name]["metrics"][k] for k in KEYS}
            out["HC1"].append({"weights": omen[o_name]["weights_sha256"][:12], "omen_run": o_name, "hpc_run": h_name,
                               "omen": {k: omen[o_name]["metrics"][k] for k in KEYS}, "hpc": {k: hpc[h_name]["metrics"][k] for k in KEYS},
                               "delta": d, "pass": all(abs(v) <= 0.005 for v in d.values())})
    if "v2_stock_s0_e20" in hpc and "omen_v2_stock_s0_e20" in hpc:
        out["HC2"] = {"descriptive_only": True, "delta_hpc_trained_minus_omen_trained":
                      {k: hpc["v2_stock_s0_e20"]["metrics"][k] - hpc["omen_v2_stock_s0_e20"]["metrics"][k] for k in KEYS}}
    a = REPO / "experiments/anti_uav/runs/normal_ir/eval_val_IR_last_hpc_HC3/metrics.json"
    b = REPO / "experiments/anti_uav/runs/normal_ir/eval_val_IR_last/metrics.json"
    if a.exists() and b.exists():
        h, o = (json.loads(x.read_text(encoding="utf-8"))["aggregate"]["mSA_percent"] for x in (a, b))
        out["HC3"] = {"hpc": h, "omen": o, "delta": h - o, "pass": abs(h - o) <= 0.05}
    out["all_pass"] = bool(out["HC1"]) and all(x["pass"] for x in out["HC1"]) and bool(out["HC3"] and out["HC3"]["pass"])
    p = REPO / "experiments/hpc/reports/host_checks.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
