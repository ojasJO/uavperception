#!/usr/bin/env python
"""H3 analysis: apply the pre-registered rules (preregistration/H3_optimisation_regime.json) to saved metrics only.

Arms (8-GPU regime: effective batch 16, LR 0.02, 1,800 optimizer steps): phaseB_normal_ir_lr0.02_accum8,
phaseB_dfsc_ir_lr0.02_accum8. References: the Phase A runs (1 GPU x batch 2, LR 0.0025, 14,400 steps) normal_ir / dfsc_ir.
Writes reports/H3_optimisation_regime.{json,md}. Never runs a model.
"""
import json
import time

import _bootstrap  # noqa: F401
from anti_uav_repro.paths import REPORTS_DIR, RUNS_DIR

PREREG = REPORTS_DIR.parent / "preregistration" / "H3_optimisation_regime.json"
ARMS = {"normal": ("phaseB_normal_ir_lr0.02_accum8", "normal_ir"), "DFSC": ("phaseB_dfsc_ir_lr0.02_accum8", "dfsc_ir")}


def msa(run, split):
    p = RUNS_DIR / run / f"eval_{split}_IR_last" / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return None if m["meta"].get("partial") else m["aggregate"]["mSA_percent"]


def main():
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    paper = pre["paper_mSA"]
    rows = {}
    for arm, (run, ref) in ARMS.items():
        rows[arm] = {s: {"h3": msa(run, s), "ref": msa(ref, s), "paper": paper[arm][s]} for s in ("val", "test")}
    complete = all(rows[a][s]["h3"] is not None for a in rows for s in ("val", "test"))
    if not complete:
        v_a, why_a, v_b, why_b = "NOT RUN YET", "evaluations missing", "NOT RUN YET", "evaluations missing"
    else:
        val_ok = all(abs(rows[a]["val"]["h3"] - rows[a]["val"]["paper"]) <= 1.0 for a in rows)
        if val_ok and all(abs(rows[a]["test"]["h3"] - rows[a]["test"]["paper"]) <= 1.0 for a in rows):
            v_a, why_a = "SUPPORTED", "both arms reproduce the paper within 1 pt on val and test in the 8-GPU regime"
        elif val_ok and all((rows[a]["test"]["ref"] - rows[a]["test"]["h3"]) >= 0.5 * (rows[a]["test"]["ref"] - rows[a]["test"]["paper"])
                            for a in rows):
            v_a, why_a = "PARTIALLY_SUPPORTED", "val within 1 pt of the paper and at least half of the test surplus removed, both arms"
        else:
            v_a, why_a = "NOT_SUPPORTED", "the 8-GPU regime does not reconcile val and test with the paper for both arms"
        gain = {s: rows["DFSC"][s]["h3"] - rows["normal"][s]["h3"] for s in ("val", "test")}
        if all(g >= 0.4 for g in gain.values()):
            v_b, why_b = "OBSERVED", "DFSC - normal >= +0.4 on both val and test (single seed: screening result)"
        elif any(g <= 0.1 for g in gain.values()):
            v_b, why_b = "NOT_OBSERVED", "DFSC - normal <= +0.1 on at least one split"
        else:
            v_b, why_b = "INCONCLUSIVE", "gain between +0.1 and +0.4 on a split"
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "preregistration": str(PREREG.relative_to(REPORTS_DIR.parents[2])),
           "rows": rows, "H3a_test_surplus": {"verdict": v_a, "reason": why_a}, "H3b_dfsc_effect": {"verdict": v_b, "reason": why_b}}
    if complete:
        out["dfsc_gain"] = {s: rows["DFSC"][s]["h3"] - rows["normal"][s]["h3"] for s in ("val", "test")}
        out["dfsc_gain_reference_regime"] = {s: rows["DFSC"][s]["ref"] - rows["normal"][s]["ref"] for s in ("val", "test")}
    (REPORTS_DIR / "H3_optimisation_regime.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    f = lambda v, s=False: "—" if v is None else (f"{v:+.2f}" if s else f"{v:.2f}")   # noqa: E731
    L = ["# H3 — optimisation regime (8-GPU convention) and the test surplus / DFSC effect", "",
         f"*Generated {out['generated']} by `scripts/analyze_h3.py`; rules frozen in `{out['preregistration']}`. Do not edit numbers.*", "",
         f"**H3a (test surplus): {v_a}** — {why_a}.  ", f"**H3b (DFSC effect in the paper's regime): {v_b}** — {why_b}.", "",
         "| arm | split | paper | 1 GPU x b2, LR 0.0025 (Phase A) | 8-GPU regime (b16, LR 0.02) | regime − Phase A | regime − paper |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for a in rows:
        for s in ("val", "test"):
            r = rows[a][s]
            d1 = None if r["h3"] is None or r["ref"] is None else r["h3"] - r["ref"]
            d2 = None if r["h3"] is None else r["h3"] - r["paper"]
            L.append(f"| {a} | {s} | {f(r['paper'])} | {f(r['ref'])} | {f(r['h3'])} | {f(d1, True)} | {f(d2, True)} |")
    if complete:
        L += ["", f"DFSC − normal: 8-GPU regime val {f(out['dfsc_gain']['val'], True)}, test {f(out['dfsc_gain']['test'], True)}; "
              f"Phase A regime val {f(out['dfsc_gain_reference_regime']['val'], True)}, test {f(out['dfsc_gain_reference_regime']['test'], True)}; "
              f"paper val {f(paper['DFSC']['val'] - paper['normal']['val'], True)}, test {f(paper['DFSC']['test'] - paper['normal']['test'], True)}."]
    (REPORTS_DIR / "H3_optimisation_regime.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"H3a: {v_a} | H3b: {v_b}")


if __name__ == "__main__":
    main()
