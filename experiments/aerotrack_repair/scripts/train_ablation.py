#!/usr/bin/env python
"""Launch ONE run of the controlled ablation grid with recorded provenance (repair items 2, 3, 13).

    python experiments/aerotrack_repair/scripts/train_ablation.py --ablation baseline_ctrl --epochs 20
    python experiments/aerotrack_repair/scripts/train_ablation.py --ablation no_spd        --epochs 20
    python experiments/aerotrack_repair/scripts/train_ablation.py --ablation single_frame  --epochs 20
    python experiments/aerotrack_repair/scripts/train_ablation.py --ablation stock_yolo11n --epochs 20
    (--dry-run builds the job and prints the exact train.py command without training;
     --preflight-only runs train.py's T1-T8 pre-flight on that exact job under a scratch name, then removes it)

The four runs, their evaluations and the final test table run from ONE command: scripts/run_grid.py.

Grid rule: every run uses the historical `baseline` stage recipe (same data, sampler, augmentation, optimiser,
batch, samples/epoch) and the SAME reduced budget (--epochs, cosine schedule completes), differing in exactly
one factor named by --ablation (see c_model/train.py::ABLATIONS). Runs are named repair_abl_<ablation>_e<epochs>
and never overwrite an existing directory. `runs/<name>/run_meta.json` records git, environment, command, data
hashes (manifest.csv, splits.json, label_cache.npz), the ablation definition and, at the end, checkpoint hashes.
Checkpoint policy: both best.pt (fitness) and last.pt are kept and both are evaluated later.
Resume: --resume continues weights/last.pt (--resume-from names another checkpoint of the same run). The recipe
(ablation, epochs, batch) must match the recorded one; run_meta.json keeps the first start time and appends each
launch to `attempts`, so an interrupted-and-resumed run still has ONE provenance record.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import train as T

from anti_uav_repro.utils.repro import env_snapshot, file_sha256, git_info

from _bootstrap import REPO_ROOT

GRID = ("baseline_ctrl", "no_spd", "single_frame", "stock_yolo11n")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json_atomic(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _epochs_on_disk(run_dir):
    p = run_dir / "results.csv"
    if not p.exists():
        return 0
    with open(p, encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True, choices=GRID)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32, help="historical v1/v2 used 32")
    ap.add_argument("--workers", type=int, default=4, help="a ceiling: train.py plans the real count from free RAM/commit")
    ap.add_argument("--resume", action="store_true", help="resume from runs/<name>/weights/last.pt")
    ap.add_argument("--resume-from", default=None, help="resume from this checkpoint of the same run instead of last.pt")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preflight-only", action="store_true",
                    help="run train.py's T1-T8 pre-flight on this exact job (scratch run name), then remove it")
    a = ap.parse_args()
    if a.resume_from:
        a.resume = True
    name = f"repair_abl_{a.ablation}_e{a.epochs}"
    run_dir = REPO_ROOT / "runs" / name
    last = run_dir / "weights" / "last.pt"
    if run_dir.exists() and any(run_dir.rglob("*")) and not a.resume and not a.dry_run and not a.preflight_only:
        sys.exit(f"{run_dir} already exists — use --resume, or a different --epochs; historical runs are never overwritten")
    py = sys.executable
    base = [py, str(REPO_ROOT / "c_model" / "train.py"), "--stage", "baseline", "--ablation", a.ablation,
            "--epochs", str(a.epochs), "--batch", str(a.batch), "--workers", str(a.workers)]
    cmd = base + ["--name", name]
    resume_ckpt = Path(a.resume_from) if a.resume_from else (last if a.resume and last.exists() else None)
    if resume_ckpt is not None and not resume_ckpt.exists():
        sys.exit(f"--resume-from {resume_ckpt} does not exist")
    if resume_ckpt is not None:
        cmd += ["--resume", str(resume_ckpt)]
    abl = dict(T.ABLATIONS[a.ablation])
    meta = {
        "run_name": name, "ablation": a.ablation, "ablation_definition": abl, "grid": list(GRID),
        "stage": "baseline", "stage_recipe": {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
                                            for k, v in T.STAGES["baseline"].items()},
        "model_cfg": abl.get("model_cfg", T.MODEL_CFG.name), "temporal_gap": abl.get("temporal_gap", 1),
        "pretrained": str(T.PRETRAINED), "pretrained_sha256": file_sha256(T.PRETRAINED) if Path(T.PRETRAINED).exists() else None,
        "epochs": a.epochs, "batch": a.batch, "workers": a.workers,
        "data": {"manifest_csv_sha256": file_sha256(T.MANIFEST), "splits_json_sha256": file_sha256(T.SPLITS),
                 "label_cache_sha256": file_sha256(T.LABEL_CACHE_PATH), "val_caps": T.VAL_CAPS},
        "git": git_info(), "env": env_snapshot(), "command": " ".join(cmd), "launcher_command": " ".join(sys.argv),
        "timestamp_start": _now(),
        "checkpoint_policy": "best.pt = max fitness (0.30 mAP50 + 0.35 mAP50-95 + 0.35 micro AP50 on the capped val set); last.pt = final epoch; BOTH are evaluated by evaluate_detector.py",
        "split_policy": "train = splits.json train; per-epoch validation on the capped VAL set (selection only); TEST untouched until the final table",
    }
    print(json.dumps({k: meta[k] for k in ("run_name", "ablation", "ablation_definition", "model_cfg", "temporal_gap", "epochs", "batch")}, indent=2))
    print("train.py command:\n  " + meta["command"], flush=True)
    if a.preflight_only:
        # same command, scratch name: the trainer writes args.json + weights/ when it is built, and a check must
        # leave nothing behind. The `_preflight_` prefix is never a real run name.
        scratch = REPO_ROOT / "runs" / f"_preflight_{name}"
        if scratch.exists():
            if any(p.suffix == ".pt" for p in scratch.rglob("*")):
                sys.exit(f"{scratch} holds checkpoints; refusing to remove it")
            shutil.rmtree(scratch)
        pcmd = base + ["--name", scratch.name, "--preflight-only"]
        print("pre-flight command:\n  " + " ".join(pcmd), flush=True)
        rc = subprocess.call(pcmd, cwd=str(REPO_ROOT))
        if scratch.exists() and not any(p.suffix == ".pt" for p in scratch.rglob("*")):
            shutil.rmtree(scratch, ignore_errors=True)
        sys.exit(rc)
    if a.dry_run:
        # build the job (dataset + model) without training, to prove the command is valid
        device = T.require_gpu()
        trainer, ctx = T.build("baseline", ablation=a.ablation, device=device,
                               overrides={"epochs": a.epochs, "batch": a.batch, "workers": a.workers, "name": name}, verbose=True)
        print(f"[dry-run] OK: {ctx['n_train']:,} train samples / {ctx['n_val']:,} val samples; model {Path(ctx['model_cfg']).name} "
              f"{ctx['n_params']/1e6:.2f}M params, {ctx['n_spdconv']} SPDConv layers, temporal_gap={ctx['temporal_gap']}; "
              f"{len(ctx['train_loader']):,} iters/epoch; nothing trained; run dir removed again")
        T.release(trainer, ctx, free_manifest=True, verbose=False)
        # the trainer writes args.json + weights/ on construction; a dry run must leave nothing behind
        if run_dir.exists() and not any(p.suffix == ".pt" for p in run_dir.rglob("*")):
            shutil.rmtree(run_dir)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_meta.json"
    attempt = {"timestamp_start": meta["timestamp_start"], "command": meta["command"], "launcher_command": meta["launcher_command"],
               "resumed_from": str(resume_ckpt) if resume_ckpt else None, "epochs_on_disk_at_start": _epochs_on_disk(run_dir),
               "workers_requested": a.workers, "git": meta["git"], "env": meta["env"]}
    if a.resume and meta_path.exists():
        prior = json.loads(meta_path.read_text(encoding="utf-8"))
        changed = {k: (prior.get(k), meta[k]) for k in ("ablation", "epochs", "batch", "model_cfg", "temporal_gap")
                   if prior.get(k) != meta[k]}
        if changed:
            sys.exit(f"refusing to resume {name} with a different recipe: {changed}")
        attempts = prior.get("attempts") or [{k: prior.get(k) for k in ("timestamp_start", "timestamp_end", "command", "return_code")}]
        for p in attempts:
            if p.get("timestamp_end") is None:
                p["outcome"] = "no end record: the launcher was killed with train.py"
        meta = prior
        meta["attempts"] = attempts
    else:
        meta["attempts"] = []
    attempt["attempt"] = len(meta["attempts"]) + 1
    meta["attempts"].append(attempt)
    meta.pop("timestamp_end", None)
    meta.pop("return_code", None)
    _write_json_atomic(meta, meta_path)

    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    attempt.update({"timestamp_end": _now(), "seconds": round(time.time() - t0, 1), "return_code": rc,
                    "epochs_on_disk_at_end": _epochs_on_disk(run_dir)})
    try:
        attempt["workers_used"] = json.loads((run_dir / "args.json").read_text(encoding="utf-8")).get("workers")
    except Exception:  # noqa: BLE001
        attempt["workers_used"] = None
    meta["timestamp_end"] = attempt["timestamp_end"]
    meta["return_code"] = rc
    meta["train_seconds"] = round(sum(x.get("seconds") or 0 for x in meta["attempts"]), 1)
    for tag in ("best", "last"):
        p = run_dir / "weights" / f"{tag}.pt"
        meta[f"{tag}_pt_sha256"] = file_sha256(p) if p.exists() else None
    _write_json_atomic(meta, meta_path)
    print(f"[launcher] train.py exited with {rc} after {attempt['seconds']/3600:.2f} h (attempt {attempt['attempt']}, "
          f"{meta['train_seconds']/3600:.2f} h in total); provenance in {meta_path}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
