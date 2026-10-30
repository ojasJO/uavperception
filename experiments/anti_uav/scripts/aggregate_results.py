#!/usr/bin/env python
"""Aggregate every completed evaluation into docs/ANTI_UAV_COMPARISON.md + reports/comparison.json.

    python experiments/anti_uav/scripts/aggregate_results.py

Sources (all machine-generated, nothing typed by hand):
  * experiments/anti_uav/reports/paper_reported_results.json      — the paper's numbers (transcribed, tagged)
  * experiments/anti_uav/runs/<run>/eval_<split>_<mod>_<ckpt>/metrics.json — our evaluations (full protocol only)
Run-to-paper mapping: run_name 'normal_ir' <-> paper 'normal'; 'dfsc_ir' <-> paper 'DFSC'. Phase B / deferred runs
are listed in their own section and never in the reproduction columns.

Verdict thresholds are PRE-REGISTERED here (before any result existed) so the label is not chosen after the fact:
  |delta mSA| <= 1.0 pt -> "reproduced"; <= 3.0 pt -> "approximately reproduced"; > 3.0 -> "not reproduced";
  paper number missing -> "not reproducible from available information"; no run -> "not yet run".
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from anti_uav_repro.paths import REPO_ROOT, REPORTS_DIR, RUNS_DIR

PAPER = REPORTS_DIR / "paper_reported_results.json"
OUT_MD = REPO_ROOT / "docs" / "ANTI_UAV_COMPARISON.md"
OUT_JSON = REPORTS_DIR / "comparison.json"
RUN_TO_PAPER = {"normal_ir": ("normal", "IR"), "dfsc_ir": ("DFSC", "IR")}
THRESH_REPRODUCED, THRESH_APPROX = 1.0, 3.0

PROTOCOL_NOTES = (
    "Same: paper epochs/LR steps/batch-per-GPU/losses/init source, GlobalTrack architecture & hyper-parameters, "
    "Eq.(1) metric, official start frames. Different (spec §11): torchvision re-implementation (S1), torchvision COCO "
    "Faster R-CNN init instead of the mmdet checkpoint (S2), 1 GPU x batch 2 with LR 0.0025 = linear-scaled 0.02 (S3), "
    "JPEG-decoded frames (S4), batches forced to distinct sequences (S5). Reproduction columns = seed 0; seed 1 runs are listed under seed variance."
)


def verdict(delta):
    if delta is None:
        return "not yet run"
    a = abs(delta)
    if a <= THRESH_REPRODUCED:
        return "reproduced"
    if a <= THRESH_APPROX:
        return "approximately reproduced"
    return "not reproduced"


def find_evaluations():
    evals = []
    if not RUNS_DIR.exists():
        return evals
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        cfg = None   # this run's config, read once and used by both loops below
        ceff = run_dir / "config.effective.yaml"
        if ceff.exists():
            import yaml
            cfg = yaml.safe_load(ceff.read_text(encoding="utf-8"))
        for ev in sorted(run_dir.glob("eval_*")):
            mfile = ev / "metrics.json"
            if not mfile.exists() or "PARTIAL" in ev.name:
                continue
            m = json.loads(mfile.read_text(encoding="utf-8"))
            if m.get("meta", {}).get("partial"):
                continue
            # an eval_* result is Phase A only if it ran the checkpoint's own protocol (no recorded variant)
            ev_variant = m["meta"].get("variant")
            evals.append({"run": run_dir.name, "eval_dir": str(ev.relative_to(REPO_ROOT)), "split": m["meta"]["split"],
                          "modality": m["meta"]["modality"], "checkpoint_epoch": m["meta"].get("checkpoint_epoch"),
                          "checkpoint_sha256": m["meta"].get("checkpoint_sha256"), "git_commit": (m["meta"].get("git") or {}).get("commit"),
                          "seed": m["meta"].get("seed"),
                          "phase": "B_additional_control" if ev_variant else (cfg or {}).get("phase"),
                          "alpha": ((cfg or {}).get("dfsc") or {}).get("alpha"),
                          "variant": ev_variant,
                          "mSA_percent": m["aggregate"]["mSA_percent"], "precision_20px": 100 * m["aggregate"]["precision_20px"],
                          "success_auc": 100 * m["aggregate"]["success_auc"], "by_attribute": m["aggregate"]["by_attribute"],
                          "n_sequences": m["aggregate"]["n_sequences"]})
        # Phase B post-hoc controls computed on saved predictions (phaseB_<variant>/<split>/metrics.json)
        for mfile in sorted(run_dir.glob("phaseB_*/*/metrics.json")):
            m = json.loads(mfile.read_text(encoding="utf-8"))
            meta = m.get("meta", {})
            if meta.get("partial"):
                continue
            evals.append({"run": run_dir.name, "eval_dir": str(mfile.parent.relative_to(REPO_ROOT)), "split": meta["split"],
                          "modality": meta["modality"], "checkpoint_epoch": meta.get("checkpoint_epoch"),
                          "checkpoint_sha256": meta.get("checkpoint_sha256"), "git_commit": (meta.get("git") or {}).get("commit"),
                          "seed": meta.get("seed"), "phase": "B_additional_control", "alpha": ((cfg or {}).get("dfsc") or {}).get("alpha"),
                          "variant": f"{meta.get('variant')}" + (f" tau={meta.get('tau')}" if meta.get("tau") is not None else ""),
                          "mSA_percent": m["aggregate"]["mSA_percent"], "precision_20px": 100 * m["aggregate"]["precision_20px"],
                          "success_auc": 100 * m["aggregate"]["success_auc"], "by_attribute": m["aggregate"]["by_attribute"],
                          "n_sequences": m["aggregate"]["n_sequences"]})
    return evals


def main():
    paper = json.loads(PAPER.read_text(encoding="utf-8"))
    cols = paper["attribute_columns"]
    evals = find_evaluations()
    rows = []
    for run, (pmethod, mod) in RUN_TO_PAPER.items():
        for split in ("val", "test"):
            pvals = paper["table_V_protocol_II"][split][pmethod]
            ours = [e for e in evals if e["run"] == run and e["split"] == split and e["modality"] == mod
                    and e["phase"] == "A_faithful_reproduction" and e["variant"] is None]
            ours = ours[-1] if ours else None
            for attr, pv in zip(cols, pvals):
                if attr == "ALL_VIS":
                    continue
                if attr == "ALL_IR":
                    ov = ours["mSA_percent"] if ours else None
                    label = "mSA All"
                else:
                    ov = (ours["by_attribute"].get(attr) or {}).get("mSA_percent") if ours else None
                    label = f"mSA {attr}"
                delta = None if (ov is None or pv is None) else ov - pv
                rows.append({"method": pmethod, "run": run, "split": split, "modality": mod, "metric": label, "paper": pv,
                             "ours": ov, "delta": delta, "verdict": verdict(delta) if pv is not None else "not reproducible from available information",
                             "eval_dir": ours["eval_dir"] if ours else None})
    # DFSC - normal gain
    gains = []
    for split in ("val", "test"):
        p_gain = paper["table_V_protocol_II"][split]["DFSC"][cols.index("ALL_IR")] - paper["table_V_protocol_II"][split]["normal"][cols.index("ALL_IR")]
        o_d = [r for r in rows if r["run"] == "dfsc_ir" and r["split"] == split and r["metric"] == "mSA All"]
        o_n = [r for r in rows if r["run"] == "normal_ir" and r["split"] == split and r["metric"] == "mSA All"]
        o_gain = None
        if o_d and o_n and o_d[0]["ours"] is not None and o_n[0]["ours"] is not None:
            o_gain = o_d[0]["ours"] - o_n[0]["ours"]
        gains.append({"split": split, "paper_gain": p_gain, "our_gain": o_gain,
                      "same_sign": None if o_gain is None else (o_gain > 0) == (p_gain > 0)})
    other = [e for e in evals if e["phase"] != "A_faithful_reproduction" or e["run"] not in RUN_TO_PAPER or e["variant"]]
    seeds = seed_table(evals)
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "thresholds": {"reproduced": THRESH_REPRODUCED, "approx": THRESH_APPROX},
           "rows": rows, "gains": gains, "other_evaluations": other, "all_evaluations": evals, "seeds": seeds}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render(out, paper), encoding="utf-8")
    print(f"wrote {OUT_MD} and {OUT_JSON}; {len(evals)} full evaluations found "
          f"({sum(1 for r in rows if r['ours'] is not None)}/{len(rows)} comparison cells filled)")


def seed_table(evals):
    """Per (method, split): mSA of every plain training run (seed-0 Phase A + seed-k Phase B), mean, spread, gain."""
    groups = {}
    for e in evals:
        if e["variant"] or e["modality"] != "IR" or e["alpha"] is None:
            continue
        method = "DFSC" if float(e["alpha"]) > 0 else "normal"
        groups.setdefault((method, e["split"]), []).append({"run": e["run"], "seed": e["seed"], "mSA": e["mSA_percent"]})
    out = {}
    for (method, split), items in groups.items():
        vals = [i["mSA"] for i in items]
        out.setdefault(split, {})[method] = {"runs": items, "n": len(vals), "mean": sum(vals) / len(vals),
                                             "min": min(vals), "max": max(vals), "spread": max(vals) - min(vals)}
    for split, d in out.items():
        if "DFSC" in d and "normal" in d:
            gains = [a["mSA"] - b["mSA"] for a in d["DFSC"]["runs"] for b in d["normal"]["runs"] if a["seed"] == b["seed"]]
            d["gain"] = {"mean_of_means": d["DFSC"]["mean"] - d["normal"]["mean"], "paired_by_seed": gains,
                         "n_pairs": len(gains), "min_pair": min(gains) if gains else None, "max_pair": max(gains) if gains else None}
    return out


def f(v):
    return "—" if v is None else f"{v:.2f}"


def fs(v):
    """signed difference"""
    return "—" if v is None else f"{v:+.2f}"


def render(out, paper):
    L = ["# Anti-UAV / DFSC — comparison with the paper", "",
         f"*Generated {out['generated']} by `experiments/anti_uav/scripts/aggregate_results.py`. Do not edit by hand: every "
         "number in the 'Paper' column is transcribed from the paper's tables (see `reports/paper_reported_results.json`, "
         "with table references), every number in 'Our reproduction' is read from a `metrics.json` produced by "
         "`scripts/evaluate.py` from saved tracker outputs. Empty cells = experiment not run yet.*", "",
         "## Verdict rules (pre-registered before any run)", "",
         f"- |Δ mSA| ≤ {out['thresholds']['reproduced']:.1f} pt → **reproduced**; ≤ {out['thresholds']['approx']:.1f} pt → "
         "**approximately reproduced**; larger → **not reproduced**; no paper value → *not reproducible from available information*; "
         "no run → *not yet run*.",
         "- The DFSC − normal *gain* counts as reproduced only if it has the paper's sign and |Δgain| ≤ 1.0 pt. The paper's gains are "
         "+0.49 (val) / +0.68 (test) mSA on IR — smaller than plausible seed-to-seed variance, so a single seed cannot settle this; "
         "Phase B seed runs are required before any claim.", "",
         "## Protocol differences (apply to every row)", "", PROTOCOL_NOTES, "",
         "## Table — mSA (%) on Anti-UAV infrared, Protocol II", "",
         "| Method | Split | Metric | Paper | Our reproduction | Difference | Verdict | Our evaluation dir |",
         "|---|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        L.append(f"| {r['method']} | {r['split']} | {r['metric']} | {f(r['paper'])} | {f(r['ours'])} | "
                 f"{fs(r['delta'])} | {r['verdict']} | {r['eval_dir'] or '—'} |")
    L += ["", "## DFSC − normal gain (IR, mSA All)", "", "| Split | Paper gain | Our gain | Same sign |", "|---|---|---|---|"]
    for g in out["gains"]:
        L.append(f"| {g['split']} | {fs(g['paper_gain'])} | {fs(g['our_gain'])} | {'—' if g['same_sign'] is None else g['same_sign']} |")
    L += ["", "## Seed variance (plain training runs only; α = 0 → normal, α = 1 → DFSC)", ""]
    if out["seeds"]:
        L += ["| Split | Method | runs (seed → mSA) | n | mean | min | max | spread |", "|---|---|---|---|---|---|---|---|"]
        for split in ("val", "test"):
            for method in ("normal", "DFSC"):
                d = out["seeds"].get(split, {}).get(method)
                if not d:
                    continue
                runs = ", ".join(f"{i['run']} (seed {i['seed']}) → {i['mSA']:.2f}" for i in sorted(d["runs"], key=lambda x: (x["seed"] or 0)))
                L.append(f"| {split} | {method} | {runs} | {d['n']} | {d['mean']:.2f} | {d['min']:.2f} | {d['max']:.2f} | {d['spread']:.2f} |")
        L += ["", "| Split | Paper gain | Gain of seed-means | Paired-by-seed gains | n pairs |", "|---|---|---|---|---|"]
        for split in ("val", "test"):
            g = out["seeds"].get(split, {}).get("gain")
            pg = [x["paper_gain"] for x in out["gains"] if x["split"] == split]
            if g:
                pairs = ", ".join(f"{x:+.2f}" for x in g["paired_by_seed"])
                L.append(f"| {split} | {fs(pg[0]) if pg else '—'} | {fs(g['mean_of_means'])} | {pairs or '—'} | {g['n_pairs']} |")
        L.append("")
        L.append("Reading rule: the paper's gain is 'observed' only if every paired-by-seed gain has the paper's sign and the gain of "
                 "seed-means is within 1 pt of the paper's value; otherwise the effect is reported as not observed at the available seeds.")
    else:
        L.append("- no runs yet")
    L += ["", "## Paper rows not reproducible here", "",
          "- **large-scale** (public GlobalTrack checkpoint, COCO+LaSOT+GOT-10k): needs the mmdet checkpoint + verified conversion → "
          "*not reproducible from available information* in this environment (spec S6).",
          "- **Visible (RGB) Protocol II rows**: same dependency (init from the GlobalTrack checkpoint) → deferred; COCO-init RGB runs, "
          "if made, are ADDITIONAL CONTROLS, not reproductions.",
          "- **Table VII precision/success**: split unspecified in the paper; ours are reported in each `report.md` for orientation only.", ""]
    L += ["## Other evaluations (Phase B / additional controls / deferred) — never merged into the reproduction columns", ""]
    if out["other_evaluations"]:
        L += ["| Run | Phase | α | Variant (single changed factor) | Split | mSA | Δ vs Phase A same run/split | prec@20 | success | Eval dir |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        phase_a = {(e["run"], e["split"]): e["mSA_percent"] for e in out["all_evaluations"] if e["phase"] == "A_faithful_reproduction" and e["variant"] is None}
        for e in out["other_evaluations"]:
            base = phase_a.get((e["run"], e["split"]))
            d = None if base is None else e["mSA_percent"] - base
            L.append(f"| {e['run']} | {e['phase']} | {e['alpha']} | {e['variant'] or '—'} | {e['split']} | "
                     f"{f(e['mSA_percent'])} | {fs(d)} | {f(e['precision_20px'])} | {f(e['success_auc'])} | {e['eval_dir']} |")
    else:
        L.append("- none yet")
    L += ["", "## Provenance of our numbers", ""]
    if out["all_evaluations"]:
        L += ["| Run | Split | Checkpoint epoch | Checkpoint sha256 | git commit | seed | sequences |", "|---|---|---|---|---|---|---|"]
        for e in out["all_evaluations"]:
            L.append(f"| {e['run']} | {e['split']} | {e['checkpoint_epoch']} | `{(e['checkpoint_sha256'] or '')[:16]}` | "
                     f"`{(e['git_commit'] or '')[:10]}` | {e['seed']} | {e['n_sequences']} |")
    else:
        L.append("- no full evaluations yet")
    L += ["", render_discrepancy(out, paper), ""]
    return "\n".join(L)


class _Missing(dict):
    def __missing__(self, key):
        return "—"


def render_discrepancy(out, paper) -> str:
    """Fill reports/discrepancy_analysis.md.tmpl with values from comparison.json (never typed by hand)."""
    tmpl_path = REPORTS_DIR / "discrepancy_analysis.md.tmpl"
    if not tmpl_path.exists():
        return "## Discrepancy analysis\n\n(to be written after the runs exist)"
    v = _Missing()
    for r in out["rows"]:
        key = f"{r['method']}_{r['split']}_{'all' if r['metric'] == 'mSA All' else r['metric'].split(' ', 1)[1]}"
        v[key + "_paper"] = f(r["paper"]); v[key + "_ours"] = f(r["ours"]); v[key + "_delta"] = fs(r["delta"]); v[key + "_verdict"] = r["verdict"]
    for g in out["gains"]:
        v[f"gain_{g['split']}_paper"] = fs(g["paper_gain"]); v[f"gain_{g['split']}_ours"] = fs(g["our_gain"])

    def val(method, split):
        r = [x for x in out["rows"] if x["method"] == method and x["split"] == split and x["metric"] == "mSA All"]
        return r[0] if r else None

    for method, tag in (("normal", "normal"), ("DFSC", "dfsc")):
        a, b = val(method, "val"), val(method, "test")
        if a and b:
            v[f"drop_paper_{tag}"] = f(a["paper"] - b["paper"])
            v[f"drop_ours_{tag}"] = f(a["ours"] - b["ours"]) if (a["ours"] is not None and b["ours"] is not None) else "—"
    phase_a = {(e["run"], e["split"]): e for e in out["all_evaluations"] if e["phase"] == "A_faithful_reproduction" and e["variant"] is None}
    for e in out["other_evaluations"]:
        if e["variant"] and e["variant"].startswith("absence_by_score_threshold"):
            base = phase_a.get((e["run"], e["split"]))
            tag = "dfsc" if e["run"] == "dfsc_ir" else "normal"
            v[f"absthr_{tag}_{e['split']}_mSA"] = f(e["mSA_percent"])
            v[f"absthr_{tag}_{e['split']}_delta"] = fs(e["mSA_percent"] - base["mSA_percent"]) if base else "—"
    t7 = paper.get("table_VII_precision_success", {})
    if "normal" in t7:
        v["tableVII_normal_prec"], v["tableVII_normal_succ"] = f(t7["normal"][0]), f(t7["normal"][1])
    e = phase_a.get(("normal_ir", "test"))
    if e:
        v["normal_test_prec_ours"], v["normal_test_succ_ours"] = f(e["precision_20px"]), f(e["success_auc"])
    for split, d in out.get("seeds", {}).items():
        for method in ("normal", "DFSC"):
            if method in d:
                v[f"seed_{split}_{method}_n"] = str(d[method]["n"]); v[f"seed_{split}_{method}_mean"] = f(d[method]["mean"])
                v[f"seed_{split}_{method}_spread"] = f(d[method]["spread"])
                v[f"seed_{split}_{method}_list"] = ", ".join(f"{i['mSA']:.2f}" for i in sorted(d[method]["runs"], key=lambda x: (x["seed"] or 0)))
        if "gain" in d:
            v[f"seed_{split}_gain_mean"] = fs(d["gain"]["mean_of_means"])
            v[f"seed_{split}_gain_pairs"] = ", ".join(f"{x:+.2f}" for x in d["gain"]["paired_by_seed"]) or "—"
    dv = REPORTS_DIR / "dataset_verification.json"
    if dv.exists():
        d = json.loads(dv.read_text(encoding="utf-8"))
        total = sum(d["splits"][sp][mod]["n_exist"] for sp in d["splits"] for mod in ("IR", "RGB"))
        v["boxes_total_ours"] = f"{total/1000:.1f}k"
    return tmpl_path.read_text(encoding="utf-8").format_map(v)


if __name__ == "__main__":
    main()
