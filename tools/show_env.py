#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_env.py — print the Python, PyTorch, CUDA, GPU and disk this machine will use.

Uses the project's own health probe (`aerolab.health.environment`), the same one the
Flight Deck shows, so the demo and the app never disagree about the machine.

    venv\\Scripts\\python.exe tools\\show_env.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from aerolab.health import environment  # noqa: E402


def main():
    d = environment()
    print("=" * 100)
    print(" DRONE-Track / AeroTrack-Net  —  ENVIRONMENT")
    print("=" * 100)
    print(f"  project root     {ROOT}")
    print(f"  python           {d.get('python')}")
    print(f"  torch            {d.get('packages', {}).get('torch')}")
    print(f"  CUDA available   {d.get('cuda_available')}")
    print(f"  GPU              {d.get('gpu') or 'CPU only (inference and evaluation still run)'}")
    print(f"  disk free        {d.get('disk_free_gb')} GB")
    print()
    pkgs = {k: v for k, v in d.get("packages", {}).items() if k != "torch"}
    if pkgs:
        print("  installed packages that matter to the results")
        for k, v in sorted(pkgs.items()):
            print(f"    {k:<20} {v}")
        print()
    for key in ("warnings", "notes"):
        for w in d.get(key, []) or []:
            print(f"  [{key[:-1]}] {w}")
    print("=" * 100)
    print(" Training needs a CUDA GPU. Everything the demo shows — inference, evaluation,")
    print(" the registries, the Flight Deck — also runs on CPU, only slower.")
    print("=" * 100)


if __name__ == "__main__":
    main()
