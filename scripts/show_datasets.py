#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_datasets.py — print the dataset directory layout of this project.

Every path and count below is read from the generated registry
(`results/registry/datasets.json`, built by `scripts/build_workspace_registry.py`) and
checked against the filesystem, so what it prints is what is actually on disk.

    venv\\Scripts\\python.exe scripts\\show_datasets.py
    venv\\Scripts\\python.exe scripts\\show_datasets.py --tree    # also show a sample sequence
"""

import argparse
import json
import sys
from pathlib import Path

# The registry text holds em-dashes; a cp1252 console would mangle them.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "results/registry" / "datasets.json"

OK = "[ok]"
NO = "[MISSING]"


def mark(p: Path) -> str:
    return OK if p.exists() else NO


def rule(char="=", width=100):
    print(char * width)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return ""


def dir_size(p: Path, cap=20000):
    """Apparent size of a directory, giving up after `cap` files (these trees are huge)."""
    total = files = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
            files += 1
            if files >= cap:
                return total, files, True
    return total, files, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", action="store_true", help="also list one sample sequence per dataset")
    ap.add_argument("--sizes", action="store_true", help="measure directory sizes (slow)")
    args = ap.parse_args()

    if not REGISTRY.exists():
        raise SystemExit(f"{NO} {REGISTRY} — run: venv\\Scripts\\python.exe scripts\\build_workspace_registry.py")

    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))

    rule()
    print(" DRONE-Track / AeroTrack-Net  —  DATASET DIRECTORY INFORMATION")
    rule()
    print(f" project root : {ROOT}")
    print(f" registry     : {REGISTRY.relative_to(ROOT)}  (generated {reg.get('generated', '?')})")
    print(f" git head     : {reg.get('git_head', '?')[:12]}")
    version = ROOT / "data" / "derived" / "data_unified_v2" / "DATA_VERSION"
    if not version.exists():
        version = ROOT / "DATA_VERSION"
    if version.exists():
        print(f" data version : {version.read_text(encoding='utf-8').strip()}")
    rule()
    print()

    for ds in reg.get("datasets", []):
        print(f"### {ds['id']}  —  {ds['name']}")
        print(f"    modalities : {', '.join(ds.get('modalities', []))}")
        print(f"    provenance : {ds.get('provenance', '-')}")
        print(f"    labels     : {ds.get('label_format', '-')}")
        print()

        print("    directories")
        roots = ds.get("roots", {})
        for key in ("labels", "frames"):
            if key not in roots:
                continue
            p = ROOT / roots[key]
            line = f"      {key:<9} {mark(p)}  {roots[key]}"
            if args.sizes and p.exists():
                size, files, capped = dir_size(p)
                line += f"   ({human(size)}{'+' if capped else ''} in {files:,}{'+' if capped else ''} files)"
            print(line)
        if "frame_dir_pattern" in roots:
            print(f"      pattern        {roots['frame_dir_pattern']}")
        print()

        splits = ds.get("splits", {})
        counts = ds.get("sequence_counts", {}) or {}
        if splits:
            print("    splits")
            print(f"      {'split':<8} {'sequences':>10}   path")
            for name, rel in splits.items():
                p = ROOT / rel
                n = counts.get(name)
                print(f"      {name:<8} {str(n) if n is not None else '-':>10}   {mark(p)} {rel}")
            print()

        if ds.get("image_size"):
            sizes = ", ".join(f"{k} {v[0]}x{v[1]}" for k, v in ds["image_size"].items())
            print(f"    frame size : {sizes}")
        if ds.get("test_split_status"):
            print(f"    TEST split : {ds['test_split_status']}")
        for lim in ds.get("limitations", []):
            print(f"    caveat     : {lim}")

        if args.tree:
            frames = roots.get("frames")
            if frames and (ROOT / frames).exists():
                base = ROOT / frames
                seqs = sorted(d for d in base.iterdir() if d.is_dir())
                # some releases nest sequences one level deeper, under split directories
                if seqs and not any(f.suffix.lower() == ".jpg" for f in seqs[0].iterdir() if f.is_file()):
                    nested = sorted(d for d in seqs[0].iterdir() if d.is_dir())
                    if nested:
                        base, seqs = seqs[0], nested
                if seqs:
                    sample = seqs[0]
                    jpgs = sorted(f.name for f in sample.iterdir() if f.is_file())
                    rel = sample.relative_to(ROOT).as_posix()
                    print(f"    sample     : {rel}/  ({len(jpgs):,} frames)")
                    for nm in jpgs[:3]:
                        print(f"                   {nm}")
                    if len(jpgs) > 3:
                        print(f"                   ... {len(jpgs) - 3:,} more")
                    print(f"    total      : {len(seqs):,} sequence directories under "
                          f"{base.relative_to(ROOT).as_posix()}")
        print()
        rule("-")
        print()

    # The unified YOLO corpus every detector trains on.
    print("### unified training corpus (built by src/pipeline/ from the three sources above)")
    for rel in ("data/derived/data_unified/splits.json",
                "data/derived/data_unified/manifest.csv",
                "data/derived/data_unified/label_cache.npz",
                "data/derived/data_unified/hardness.npz",
                "data/derived/data_unified_v2"):
        p = ROOT / rel
        print(f"      {mark(p)}  {rel}")
    splits_file = ROOT / "data" / "derived" / "data_unified" / "splits.json"
    if splits_file.exists():
        try:
            sp = json.loads(splits_file.read_text(encoding="utf-8"))
            print()
            print("      sequence-level splits (seed 1337, leakage-asserted)")
            for k, v in sp.items():
                if isinstance(v, dict):
                    inner = ", ".join(f"{kk} {len(vv):,}" for kk, vv in v.items() if hasattr(vv, "__len__"))
                    print(f"        {k:<12} {inner}")
                elif hasattr(v, "__len__"):
                    print(f"        {k:<12} {len(v):,} sequences")
        except (ValueError, TypeError):
            pass
    print()
    rule()
    print(" Full dataset documentation: docs/DATASETS_data_unified_v2.md, docs/DATASETS_data_unified.md")
    rule()


if __name__ == "__main__":
    main()
