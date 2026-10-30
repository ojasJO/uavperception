#!/usr/bin/env python
"""Training entry point (Protocol II, IR by default).

    python experiments/anti_uav/scripts/train.py --config experiments/anti_uav/configs/dfsc_ir.yaml
    python experiments/anti_uav/scripts/train.py --config ... --dry-run        # validate + build, no training
    python experiments/anti_uav/scripts/train.py --config ... --resume experiments/anti_uav/runs/dfsc_ir/weights/last.pt

This script can only see the TRAIN split (optimisation) and the VAL split (monitor). It has no code path
that loads TEST; `load_sequences` refuses it and `validate_config` rejects any config that names it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from anti_uav_repro.config import config_summary, load_config, validate_config
from anti_uav_repro.paths import RUNS_DIR


def main():
    ap = argparse.ArgumentParser(description="Train QG-RCNN / DFSC on Anti-UAV (train split only)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None, help="default: experiments/anti_uav/runs/<run_name>")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from (weights/last.pt)")
    ap.add_argument("--dry-run", action="store_true", help="validate config, build data + model, print summary, exit")
    ap.add_argument("--max-iters-per-epoch", type=int, default=None, help="SMOKE TESTS ONLY")
    ap.add_argument("--allow-existing", action="store_true", help="allow writing into a non-empty run dir (resume implies this)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / cfg["run_name"]
    has_files = run_dir.exists() and any(p.is_file() for p in run_dir.rglob("*"))
    if has_files and not (args.resume or args.allow_existing or args.dry_run):
        sys.exit(f"run dir {run_dir} already contains files. Use --resume <ckpt>, --allow-existing, or a new run_name. "
                 "Historical runs are never overwritten silently.")
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    print(config_summary(cfg))
    from anti_uav_repro.engine.train import Tee, Trainer
    if not args.dry_run:
        sys.stdout = Tee(run_dir / "train_log.txt")
        print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} launch: {' '.join(sys.argv)} =====")
    trainer = Trainer(cfg, run_dir, resume=args.resume, max_iters_per_epoch=args.max_iters_per_epoch)
    if args.dry_run:
        ps = trainer.model.parameter_summary()
        print(f"[dry-run] OK: {len(trainer.train_seqs)} train sequences ({cfg['data']['modality']}/{cfg['data']['train_split']}), "
              f"{len(trainer.dataset)} pairs/epoch, {len(trainer.sampler)} iterations/epoch, {cfg['train']['epochs']} epochs "
              f"-> {len(trainer.sampler) * cfg['train']['epochs']} iterations total")
        print(f"[dry-run] model {cfg['model']['name']} init={cfg['model']['init']} params {ps['total']/1e6:.2f}M "
              f"(trainable {ps['trainable']/1e6:.2f}M); device {trainer.device}; monitor sequences: "
              f"{[s.name for s in trainer.monitor_seqs]}")
        print(f"[dry-run] run dir would be {run_dir}. Nothing was trained.")
        return
    trainer.fit()


if __name__ == "__main__":
    main()
