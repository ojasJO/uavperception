#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_models.py — print the trained models of this project and where their weights are.

Read from the generated registry (`results/registry/models.json`, `results/registry/checkpoints.csv`,
`results/registry/build_summary.json`), so every identity and path is the one the registry
verified by SHA-256.

    venv\\Scripts\\python.exe scripts\\show_models.py
    venv\\Scripts\\python.exe scripts\\show_models.py --all     # every model, not just the headline ones
"""

import argparse
import csv
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "results/registry"

# The models that carry a published number. Everything else is in --all.
HEADLINE = [
    ("aerotrack.detector.v2", "v2_causal_g4_s0_e20",
     "selected detector: causal temporal window [t-8, t-4, t], CST+Anti-UAV+Det-Fly"),
    ("aerotrack.detector.v2", "v2_stock_s0_e20",
     "single-frame control for the window comparison"),
    ("aerotrack.detector.cst_x", "v3_n__causal_g4_cst_s0_e20",
     "CST-only detector behind the X1 / X2 benchmark result"),
    ("aerotrack.detector.gen", "gen_temporal_fp_g4_full_e20",
     "generalization-series detector (corpus v1)"),
    ("antiuav_repro.globaltrack", "normal_ir",
     "Anti-UAV 2021 reproduction, GlobalTrack baseline arm (IR)"),
    ("antiuav_repro.globaltrack", "dfsc_ir",
     "Anti-UAV 2021 reproduction, DFSC arm (IR)"),
    ("antiuav_repro.globaltrack", "normal_rgb",
     "Anti-UAV 2021 reproduction, visible arm"),
    ("shared.globaltrack_pretrained", None,
     "public GlobalTrack checkpoint, converted (the paper's initialisation)"),
]


def rule(c="=", w=100):
    print(c * w)


def human(n):
    n = float(n or 0)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:,.0f} {u}"
        n /= 1024.0
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="list every registered model")
    args = ap.parse_args()

    summary_f, ckpt_f = REG / "build_summary.json", REG / "checkpoints.csv"
    if not ckpt_f.exists():
        raise SystemExit(f"[MISSING] {ckpt_f} — run: venv\\Scripts\\python.exe scripts\\build_workspace_registry.py")

    summary = json.loads(summary_f.read_text(encoding="utf-8")) if summary_f.exists() else {}
    ckpts = list(csv.DictReader(ckpt_f.open(encoding="utf-8")))

    rule()
    print(" DRONE-Track / AeroTrack-Net  —  TRAINED MODELS AND CHECKPOINTS")
    rule()
    print(f" registry generated : {summary.get('generated', '?')}  (git {str(summary.get('git_head', ''))[:12]})")
    files, models = summary.get("files", {}), summary.get("models", {})
    print(f" checkpoint files   : {files.get('total', len(ckpts)):,}"
          f"  ({files.get('workspace', '?')} in the workspace, {files.get('recovery_backup', '?')} in the HPC backup)")
    print(f" model identities   : {models.get('total', '?')} across {len(models.get('by_family', {}))} families")
    rule()
    print()

    if models.get("by_family"):
        print(" models per family")
        for fam, n in sorted(models["by_family"].items(), key=lambda kv: -kv[1]):
            print(f"   {n:>4}  {fam}")
        print()
    if models.get("by_status"):
        print(" models per status : " + ", ".join(f"{k} {v}" for k, v in sorted(models["by_status"].items())))
        print()

    rule("-")
    print(" HEADLINE MODELS — the checkpoints every published number comes from")
    rule("-")
    print()
    for family, run, what in HEADLINE:
        rows = [r for r in ckpts
                if r.get("family_id") == family
                and (run is None or r.get("run_id") == run)
                and r.get("artifact_kind") == "model_weights"]
        label = f"{family}:{run}" if run else family
        print(f" {label}")
        print(f"   {what}")
        if not rows:
            print("   [not present in this checkout]")
            print()
            continue
        # one line per tag, workspace copy preferred over the backup copy
        seen = {}
        for r in rows:
            key = r.get("tag") or Path(r["physical_path"]).name
            if key not in seen or (r.get("location") == "workspace" and seen[key].get("location") != "workspace"):
                seen[key] = r
        # the per-epoch snapshots of the reproduction runs would bury the page: show the
        # checkpoints a result is reported from, then count the rest.
        shown = {t: r for t, r in seen.items() if t in ("best", "last", "") or not t.startswith("epoch_")}
        for tag, r in sorted(shown.items()):
            p = ROOT / r["physical_path"]
            print(f"   {tag or 'weights':<12} {'[ok]' if p.exists() else '[MISSING]':<10} {human(r.get('size_bytes')):>9}"
                  f"  sha256 {str(r.get('sha256', ''))[:12]}  {r['physical_path']}")
        rest = len(seen) - len(shown)
        if rest:
            print(f"   + {rest} per-epoch snapshots in the same weights/ directory")
        print()

    if args.all:
        rule("-")
        print(" EVERY REGISTERED CHECKPOINT")
        rule("-")
        for r in sorted(ckpts, key=lambda r: (r.get("family_id", ""), r.get("run_id", ""), r.get("tag", ""))):
            if r.get("artifact_kind") != "model_weights":
                continue
            print(f" {r.get('family_id', ''):<34} {r.get('run_id', ''):<30} {r.get('tag', ''):<12}"
                  f" {r.get('location', ''):<16} {r['physical_path']}")
        print()

    rule()
    print(" Full registry: results/registry/models.json, results/registry/checkpoints.csv, results/registry/README.md")
    print(" Browse interactively: studio.bat  ->  http://localhost:8501  ->  Models")
    rule()


if __name__ == "__main__":
    main()
