#!/usr/bin/env python
"""Wrap the converted GlobalTrack weights as the run `largescale_globaltrack` = the paper's "large-scale" baseline row (Table V:
public GlobalTrack checkpoint, NO Anti-UAV training), so the unchanged evaluate.py can score it (it needs an embedded config).

    python experiments/anti_uav/scripts/make_largescale_run.py --converted <globaltrack_qgrcnn_torchvision.pt>

Writes experiments/anti_uav/runs/largescale_globaltrack/{weights/last.pt, run_meta.json}. Refuses to overwrite. The same file is
the init checkpoint of the faithful visible runs (configs/phase_a_rgb/_base_rgb.yaml).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from anti_uav_repro.config import load_config, validate_config

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "runs" / "largescale_globaltrack"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--converted", required=True)
    a = ap.parse_args()
    w = RUN / "weights" / "last.pt"
    if w.exists():
        raise SystemExit(f"refusing: {w} exists")
    src = torch.load(a.converted, map_location="cpu", weights_only=False)
    cfg = load_config(HERE.parent / "configs" / "phase_a_rgb" / "_base_rgb.yaml")
    cfg["run_name"] = "largescale_globaltrack"
    cfg["model"]["init"] = "none"                    # weights come from this file itself
    cfg["model"].pop("init_checkpoint", None)
    validate_config(cfg)
    w.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": src["model"], "config": cfg, "epoch": 0, "run_name": "largescale_globaltrack",
                "source": src["meta"]}, w)
    meta = {"run_name": "largescale_globaltrack", "phase": "A_faithful_reproduction", "row": "Table V 'large-scale'",
            "training": "none (public GlobalTrack checkpoint, converted)", "converted_from": Path(a.converted).name,
            "converted_sha256": sha256(a.converted), "source": src["meta"], "final_checkpoint": str(w.relative_to(HERE.parents[2])),
            "final_checkpoint_sha256": sha256(w), "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (RUN / "run_meta.json").write_text(json.dumps({k: v for k, v in meta.items()}, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("final_checkpoint", "final_checkpoint_sha256", "converted_sha256")}, indent=1))


if __name__ == "__main__":
    main()
