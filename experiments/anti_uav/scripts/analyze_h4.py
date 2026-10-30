#!/usr/bin/env python
"""H4 analysis: apply the pre-registered rule (preregistration/H4_native_scale.json) to the native-scale evaluations.

Reads only saved metrics.json files: the four Phase A reference evaluations (eval_<split>_IR_last) and the four Phase B
native-scale evaluations (phaseB_test_scale_640x512/<split>). Writes reports/H4_native_scale.{json,md}. Never runs a model.

    python experiments/anti_uav/scripts/analyze_h4.py            (missing evaluations are reported as 'not run yet')
"""
import json
import sys
import time

import _bootstrap  # noqa: F401
from anti_uav_repro.paths import REPORTS_DIR, RUNS_DIR

PREREG = REPORTS_DIR.parent / "preregistration" / "H4_native_scale.json"
OUT_JSON = REPORTS_DIR / "H4_native_scale.json"
OUT_MD = REPORTS_DIR / "H4_native_scale.md"
VARIANT = "test_scale_640x512"
ATTRS = ["OV", "OC", "FM", "SV", "LI", "TC-EASY", "TC-MID", "TC-HARD", "TC", "LR"]


def load(path):
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    return None if m["meta"].get("partial") else m


def decide(models):
    """The rule exactly as registered. `models`: name -> {val/test: {ref, native, paper}}."""
    if any(models[n][s]["native"] is None for n in models for s in ("val", "test")):
        return "NOT RUN YET", "at least one of the four evaluations is missing"
    def r(n, s, k):
        return models[n][s][k]
    names = list(models)
    val_ok = all(abs(r(n, "val", "native") - r(n, "val", "paper")) <= 1.0 for n in names)
    if val_ok and all(abs(r(n, "test", "native") - r(n, "test", "paper")) <= 1.0 for n in names):
        return "SUPPORTED", "native evaluation reproduces the paper within 1 pt on both splits for both models"
    if val_ok and all((r(n, "test", "ref") - r(n, "test", "native")) >= 0.5 * (r(n, "test", "ref") - r(n, "test", "paper")) for n in names):
        return "PARTIALLY_SUPPORTED", "val still within 1 pt of the paper and native evaluation closes >= half of the test surplus for both models"
    if any(r(n, "val", "native") < r(n, "val", "paper") - 1.0 for n in names):
        return "NOT_SUPPORTED_VAL_BREAKS", "native evaluation no longer reproduces the paper's validation for at least one model"
    if all(abs(r(n, s, "native") - r(n, s, "ref")) <= 1.0 for n in names for s in ("val", "test")):
        return "NOT_SUPPORTED_INSENSITIVE", "|native - reference| <= 1 pt on both splits for both models"
    return "INCONCLUSIVE", "models disagree; reported descriptively"


def main():
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    models, attr_rows, provenance = {}, [], []
    for name, spec in pre["models"].items():
        models[name] = {}
        for split in ("val", "test"):
            ref = load(RUNS_DIR / name / f"eval_{split}_IR_last" / "metrics.json")
            nat = load(RUNS_DIR / name / f"phaseB_{VARIANT}" / split / "metrics.json")
            if nat is not None and nat["meta"].get("checkpoint_sha256") != spec["sha256"]:
                sys.exit(f"{name}/{split}: native evaluation used a different checkpoint than registered")
            if ref is not None and ref["meta"].get("checkpoint_sha256") != spec["sha256"]:
                sys.exit(f"{name}/{split}: reference evaluation used a different checkpoint than registered")
            models[name][split] = {"ref": ref["aggregate"]["mSA_percent"] if ref else None,
                                   "native": nat["aggregate"]["mSA_percent"] if nat else None,
                                   "paper": spec["paper_mSA"][split],
                                   "fps_ref": ref["aggregate"].get("tracker_fps_mean") if ref else None,
                                   "fps_native": nat["aggregate"].get("tracker_fps_mean") if nat else None}
            if ref and nat:
                for a in ATTRS:
                    ra, na = ref["aggregate"]["by_attribute"].get(a), nat["aggregate"]["by_attribute"].get(a)
                    if ra and na:
                        attr_rows.append({"model": name, "split": split, "attribute": a, "n_sequences": ra["n_sequences"],
                                          "ref": ra["mSA_percent"], "native": na["mSA_percent"], "delta": na["mSA_percent"] - ra["mSA_percent"]})
            if nat:
                provenance.append({"model": name, "split": split, "eval_dir": str((RUNS_DIR / name / f"phaseB_{VARIANT}" / split).relative_to(REPORTS_DIR.parents[2])),
                                   "checkpoint_sha256": nat["meta"]["checkpoint_sha256"], "git": nat["meta"].get("git"),
                                   "test_scale": nat["meta"].get("test_scale"), "timestamp_start": nat["meta"].get("timestamp_start")})
    verdict, why = decide(models)
    for n in models:
        for s in ("val", "test"):
            d = models[n][s]
            d["delta_native_minus_ref"] = None if d["native"] is None or d["ref"] is None else d["native"] - d["ref"]
            d["native_minus_paper"] = None if d["native"] is None else d["native"] - d["paper"]
        dv, dt = models[n]["val"]["delta_native_minus_ref"], models[n]["test"]["delta_native_minus_ref"]
        models[n]["asymmetry_delta_test_minus_delta_val"] = None if dv is None or dt is None else dt - dv
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "preregistration": str(PREREG.relative_to(REPORTS_DIR.parents[2])),
           "class": pre["class"], "verdict": verdict, "verdict_reason": why, "models": models, "by_attribute": attr_rows,
           "provenance": provenance, "known_confound": pre["known_confound"]}
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    f = lambda v: "—" if v is None else f"{v:.2f}"            # noqa: E731
    fs = lambda v: "—" if v is None else f"{v:+.2f}"          # noqa: E731
    L = [f"# H4 — does the test-time input scale explain the test surplus?",
         "",
         f"*Generated {out['generated']} by `scripts/analyze_h4.py` from saved `metrics.json` files; rule and predictions frozen in "
         f"`{out['preregistration']}` before any full run. Class: {pre['class']}. Do not edit numbers by hand.*",
         "",
         f"**Verdict (pre-registered rule): {verdict}** — {why}.",
         "",
         "Controlled variable: test-time rescale target only — reference = the Phase A evaluations (IR 640×512 rescaled to 1000×800 "
         "by GlobalTrack's 1333×800 rule), native = 640×512 without resizing. Same frozen final-epoch checkpoints, tracker, metric and "
         "split manifests.",
         "",
         "| model | split | paper | reference (1000×800) | native (640×512) | Δ native − reference | native − paper | fps ref → native |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
    for n in models:
        for s in ("val", "test"):
            d = models[n][s]
            fps = "—" if d["fps_native"] is None else f"{d['fps_ref'] or 0:.1f} → {d['fps_native']:.1f}"
            L.append(f"| {n} | {s} | {f(d['paper'])} | {f(d['ref'])} | {f(d['native'])} | {fs(d['delta_native_minus_ref'])} | "
                     f"{fs(d['native_minus_paper'])} | {fps} |")
    L += ["", "Val/test asymmetry of the scale effect (Δtest − Δval; the pre-registered trigger for a train-at-native follow-up is "
          "≤ −2 pt for both models): " + ", ".join(f"{n} {fs(models[n]['asymmetry_delta_test_minus_delta_val'])}" for n in models), ""]
    if attr_rows:
        L += ["## Per-attribute effect (Δ native − reference, mSA pt)", "",
              "| attribute | " + " | ".join(f"{n} {s} (n)" for n in models for s in ("val", "test")) + " |",
              "|---|" + "---:|" * (2 * len(models))]
        for a in ATTRS:
            cells = []
            for n in models:
                for s in ("val", "test"):
                    row = next((r for r in attr_rows if r["model"] == n and r["split"] == s and r["attribute"] == a), None)
                    cells.append("—" if row is None else f"{row['delta']:+.2f} ({row['n_sequences']})")
            L.append(f"| {a} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Known confound (stated before the runs)", "", pre["known_confound"], "",
          "## Provenance", "", "| model | split | eval dir | checkpoint sha256 | git | started |", "|---|---|---|---|---|---|"]
    for p in provenance:
        L.append(f"| {p['model']} | {p['split']} | `{p['eval_dir']}` | `{p['checkpoint_sha256'][:16]}` | "
                 f"`{(p['git'] or {}).get('commit', '')[:10]}`{' (dirty)' if (p['git'] or {}).get('dirty') else ''} | {p['timestamp_start']} |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"H4 verdict: {verdict} — {why}\nwrote {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
