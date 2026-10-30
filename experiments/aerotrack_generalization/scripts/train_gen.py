#!/usr/bin/env python
"""Train ONE configuration of the generalization series on the grouped folds (or the full TRAIN split for the final step).

    python experiments/aerotrack_generalization/scripts/train_gen.py --config stock --fold 0 --epochs 20
    python experiments/aerotrack_generalization/scripts/train_gen.py --config spd_fp --fold 0 --epochs 20 --resume
    (--fold full: all three folds = the whole TRAIN split, monitored on the official VAL; --preflight-only: T1-T8 on the exact
     job under a scratch name, nothing trained)

Recipe = c_model/train.py STAGES["baseline"] (the repair grid's), identical for every configuration: 20 epochs x 25,000
weighted samples, batch 32 x accumulate 2, AdamW 1e-3 cosine, same sampler / augmentation / preflight. Differences allowed:
the configuration's single factor (gen_lib.CONFIGS), the fold's training sequences, seed = fold index (0 for 'full'),
function-preserving initialisation for every configuration, torch/numpy seeded before the model is built.

Launcher (default) writes runs/<name>/run_meta.json -- one record across resumes (attempts appended, recipe locked) -- and runs
the training in a `--_child` process, so a watchdog can stop training while the launcher still records the attempt.
Checkpoints: weights/last.pt every epoch (atomic), best.pt by the recipe's fitness (NOT used: the series scores last.pt)."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import _series as S
import gen_lib as L

from anti_uav_repro.utils.repro import env_snapshot, file_sha256


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json_atomic(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _epochs_on_disk(run_dir):
    p = run_dir / "results.csv"
    return sum(1 for _ in csv.DictReader(open(p, encoding="utf-8", newline=""))) if p.exists() else 0


def child(a):
    """Build the exact job, run the pre-flight, train (or stop after the pre-flight)."""
    import numpy as np
    import torch

    import preflight as P
    import train as T
    from aerotrack_trainer import AeroTrackTrainer, build_model, transfer_pretrained
    from stacked_dataset import assert_no_leakage, load_label_cache

    cfg_def = L.CONFIGS[a.config]
    gap, layout = cfg_def["gap"], cfg_def["layout"]
    fold = a.fold if a.fold == "full" else int(a.fold)
    T.set_process_priority("below_normal")
    T.limit_cpu_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = T.require_gpu()
    folds = L.load_folds()
    stage_cfg = dict(T.STAGES["baseline"])
    kw = {k: v for k, v in stage_cfg.items() if k in T.TrainConfig().to_dict()}
    kw.update(name=a.name, project=str(S.REPO / "runs"), epochs=a.epochs, batch=a.batch, workers=a.workers, seed=a.seed)
    if a.samples_per_epoch:
        kw["samples_per_epoch"] = a.samples_per_epoch
    cfg = T.TrainConfig(**kw)
    cache = load_label_cache(T.LABEL_CACHE_PATH)
    train_ids = [0, 1, 2] if fold == "full" else [k for k in (0, 1, 2) if k != fold]
    train_s = L.fold_samples(folds, train_ids, gap)
    val_s = L.monitor_val(folds, fold, gap)
    if a.monitor_cap:
        val_s = T.subsample(val_s, {d: a.monitor_cap for d in ("anti_uav", "cst", "det_fly")})
    assert_no_leakage(train_s, val_s)
    print(f"[gen] {a.config} fold {fold} gap {gap} layout {layout}: {len(train_s):,} train samples from folds {train_ids}, "
          f"{len(val_s):,} monitoring samples", flush=True)
    train_ds, val_ds, train_loader, val_loader = T.build_loaders(train_s, val_s, cfg, stage_cfg, label_cache=cache, augment_kwargs=None,
                                                                 sampler_weights=None, letterbox=True, verbose=True)
    model = build_model(S.REPO / "c_model" / L.YAML[layout], ch=9, nc=2, verbose=False, hyp=None, class_weights=None)
    transfer_pretrained(model, T.PRETRAINED, verbose=True)
    check = L.apply_fp_init(model, layout)
    print(f"[init] function-preserving: {json.dumps(check)}", flush=True)
    if not check["ok"]:
        raise SystemExit(f"[ABORT] function-preserving initialisation check failed: {check}")
    trainer = AeroTrackTrainer(model, train_loader, val_loader, device, cfg)
    (trainer.save_dir / "init_check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
    train_keys, val_keys = {s["seq_key"] for s in train_s}, {s["seq_key"] for s in val_s}
    P.run_all([
        P.check_environment(),
        P.check_dataset(val_ds, batch_size=min(4, cfg.batch)),
        P.check_boundary(val_ds),
        P.check_leakage(train_keys, val_keys),
        P.check_collate(trainer, val_ds, batch_size=min(4, cfg.batch)),
        P.check_checkpoint_roundtrip(trainer, device),
        P.check_throughput(train_loader),
        P.check_vram(trainer, train_ds, cfg.batch, device),
    ])
    if a.preflight_only:
        T.release(trainer, {"train_loader": train_loader, "val_loader": val_loader}, free_manifest=True, verbose=False)
        print("[preflight-only] every check passed on this exact job; nothing was trained", flush=True)
        return
    trainer.train(resume_from=a.resume_from)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(L.CONFIGS))
    ap.add_argument("--fold", required=True, help="0, 1, 2 (held-out fold) or 'full'")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None, help="default: the fold index (0 for full)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--name-suffix", default="", help="smoke tests only (e.g. _smoke); recorded")
    ap.add_argument("--samples-per-epoch", type=int, default=None, help="smoke tests only; recorded")
    ap.add_argument("--monitor-cap", type=int, default=None, help="smoke tests only; recorded")
    ap.add_argument("--name", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.fold not in ("full", "0", "1", "2"):
        sys.exit("--fold must be 0, 1, 2 or full")
    if a.seed is None:
        a.seed = 0 if a.fold == "full" else int(a.fold)
    if a._child:
        return child(a)

    name = L.run_name(a.config, a.fold, a.epochs, a.name_suffix)
    run_dir = S.REPO / "runs" / name
    last = run_dir / "weights" / "last.pt"
    base = [sys.executable, "-u", os.path.abspath(__file__), "--_child", "--config", a.config, "--fold", a.fold, "--epochs", str(a.epochs),
            "--batch", str(a.batch), "--workers", str(a.workers), "--seed", str(a.seed)]
    for flag, v in (("--samples-per-epoch", a.samples_per_epoch), ("--monitor-cap", a.monitor_cap)):
        if v:
            base += [flag, str(v)]
    if a.preflight_only:
        scratch = S.REPO / "runs" / f"_preflight_{name}"
        if scratch.exists():
            shutil.rmtree(scratch)
        rc = subprocess.call(base + ["--name", scratch.name, "--preflight-only"], cwd=str(S.REPO))
        if scratch.exists() and not any(p.suffix == ".pt" for p in scratch.rglob("*")):
            shutil.rmtree(scratch, ignore_errors=True)
        sys.exit(rc)
    if run_dir.exists() and any(run_dir.rglob("*")) and not a.resume:
        sys.exit(f"{run_dir} already exists — use --resume; earlier runs are never overwritten")
    resume_ckpt = a.resume_from or (str(last) if a.resume and last.exists() else None)
    cmd = base + ["--name", name] + (["--resume-from", resume_ckpt] if resume_ckpt else [])
    folds = L.load_folds()
    meta = {"run_name": name, "series": "aerotrack_generalization", "config": a.config, "config_definition": L.CONFIGS[a.config],
            "gap": L.CONFIGS[a.config]["gap"], "layout": L.CONFIGS[a.config]["layout"], "fold": a.fold, "seed": a.seed,
            "epochs": a.epochs, "batch": a.batch, "init": "function_preserving", "recipe": "c_model/train.py STAGES['baseline']",
            "folds_sha256": folds["sha256"], "smoke": bool(a.name_suffix or a.samples_per_epoch or a.monitor_cap),
            "overrides": {"samples_per_epoch": a.samples_per_epoch, "monitor_cap": a.monitor_cap},
            "data": {"manifest_sha256": file_sha256(S.REPO / "b_pipeline" / "data_unified" / "manifest.csv"),
                     "splits_sha256": file_sha256(S.REPO / "b_pipeline" / "data_unified" / "splits.json")},
            "checkpoint_policy": "last.pt is the scored checkpoint (fixed before any run); best.pt is kept but not used"}
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_meta.json"
    attempt = {"timestamp_start": _now(), "command": " ".join(cmd), "resumed_from": resume_ckpt, "epochs_on_disk_at_start": _epochs_on_disk(run_dir),
               "workers_requested": a.workers, "git": S.git(), "env": env_snapshot()}
    if meta_path.exists():
        prior = json.loads(meta_path.read_text(encoding="utf-8"))
        changed = {k: (prior.get(k), meta[k]) for k in ("config", "fold", "seed", "epochs", "batch", "folds_sha256", "overrides") if prior.get(k) != meta[k]}
        if changed:
            sys.exit(f"refusing to resume {name} with a different setup: {changed}")
        for p in prior.get("attempts", []):
            if p.get("timestamp_end") is None:
                p["outcome"] = "no end record: the launcher was killed"
        meta = prior
    meta.setdefault("attempts", []).append(attempt)
    attempt["attempt"] = len(meta["attempts"])
    meta.pop("timestamp_end", None)
    meta.pop("return_code", None)
    _write_json_atomic(meta, meta_path)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(S.REPO))
    attempt.update({"timestamp_end": _now(), "seconds": round(time.time() - t0, 1), "return_code": rc, "epochs_on_disk_at_end": _epochs_on_disk(run_dir)})
    try:
        attempt["workers_used"] = json.loads((run_dir / "args.json").read_text(encoding="utf-8")).get("workers")
    except Exception:  # noqa: BLE001
        attempt["workers_used"] = None
    meta.update(timestamp_end=attempt["timestamp_end"], return_code=rc, train_seconds=round(sum(x.get("seconds") or 0 for x in meta["attempts"]), 1))
    for tag in ("best", "last"):
        p = run_dir / "weights" / f"{tag}.pt"
        meta[f"{tag}_pt_sha256"] = file_sha256(p) if p.exists() else None
    _write_json_atomic(meta, meta_path)
    print(f"[launcher] {name}: exit {rc} after {attempt['seconds'] / 3600:.2f} h (attempt {attempt['attempt']})", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
