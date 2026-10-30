#!/usr/bin/env python
"""Train ONE configuration of the v3 series (model x window) on the full corpus-v2 TRAIN split.

    venv\\Scripts\\python.exe experiments\\aerotrack_v2\\scripts\\train_v2.py --config causal_g4 --seed 0
    ... --resume                      (continue an interrupted run; the recipe is locked in run_meta.json)
    ... --preflight-only              (T1-T8 on the exact job under a scratch name; nothing trained)
    ... --smoke                       (machinery test: 2 epochs x 512 samples, name suffix _smoke, flagged in run_meta)

Recipe = c_model/train.py STAGES["baseline"] exactly as in the repair grid and the generalization series: 20 epochs x 25,000
weighted samples, batch 32 x accumulate 2, AdamW 1e-3 cosine, same sampler / augmentation / letterbox / pre-flight.
Differences allowed: the configuration's window (v2lib.CONFIGS), the seed, the corpus (v2). Function-preserving init for every
configuration (runtime-checked; the run aborts if the check fails). Checkpoint scored: weights/last.pt (fixed before any run).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import v3series as S
import v3lib as L

from anti_uav_repro.utils.repro import env_snapshot, file_sha256


def _epochs_on_disk(run_dir):
    p = run_dir / "results.csv"
    return sum(1 for _ in csv.DictReader(open(p, encoding="utf-8", newline=""))) if p.exists() else 0


def child(a):
    import numpy as np
    import torch

    import preflight as P
    import train as T
    from aerotrack_trainer import AeroTrackTrainer
    from stacked_dataset import assert_no_leakage, load_label_cache

    if T.MANIFEST.parent.name != S.CORPUS:
        raise SystemExit(f"[ABORT] pipeline reads {T.MANIFEST}, not corpus {S.CORPUS}")
    model_key, window = L.parse(a.config)
    T.set_process_priority("below_normal")
    T.limit_cpu_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = T.require_gpu()
    stage_cfg = dict(T.STAGES["baseline"])
    kw = {k: v for k, v in stage_cfg.items() if k in T.TrainConfig().to_dict()}
    kw.update(name=a.name, project=str(S.REPO / "runs"), epochs=a.epochs, batch=a.batch, workers=a.workers, seed=a.seed)
    if a.samples_per_epoch:
        kw["samples_per_epoch"] = a.samples_per_epoch
    cfg = T.TrainConfig(**kw)
    cache = load_label_cache(T.LABEL_CACHE_PATH)
    doms = L.parse_domains(a.train_domains)
    train_s = L.filter_domains(L.window_samples("train", window), doms)       # cross-dataset runs: a training-domain subset
    val_s = T.subsample(L.window_samples("val", window), T.VAL_CAPS)      # monitoring only (selects nothing)
    if a.monitor_cap:
        val_s = T.subsample(val_s, {d: a.monitor_cap for d in ("anti_uav", "cst", "det_fly")})
    assert_no_leakage(train_s, val_s)
    print(f"[v3] {a.config} (model {model_key}, window {window}) seed {a.seed}: {len(train_s):,} train samples, "
          f"{len(val_s):,} monitoring samples, corpus {S.CORPUS}", flush=True)
    train_ds, val_ds, train_loader, val_loader = T.build_loaders(train_s, val_s, cfg, stage_cfg, label_cache=cache, augment_kwargs=None,
                                                                 sampler_weights=None, letterbox=True, verbose=True)
    model, check = L.build_initialised(model_key, S.REPO)
    print(f"[init] function-preserving: {json.dumps(check)}", flush=True)
    if not check["ok"]:
        raise SystemExit(f"[ABORT] function-preserving initialisation check failed: {check}")
    trainer = AeroTrackTrainer(model, train_loader, val_loader, device, cfg)
    (trainer.save_dir / "init_check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
    P.run_all([
        P.check_environment(),
        P.check_dataset(val_ds, batch_size=min(4, cfg.batch)),
        P.check_boundary(val_ds),
        P.check_leakage({s["seq_key"] for s in train_s}, {s["seq_key"] for s in val_s}),
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
    ap.add_argument("--config", required=True, help="<model>__<window>, e.g. n_p2__causal_g4")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("AEROTRACK_WORKERS", 4)),
                    help="loader workers (default 4; AEROTRACK_WORKERS overrides on the HPC, amendment A1)")
    ap.add_argument("--train-domains", default="all", help="'all' or a comma list of anti_uav_IR, anti_uav_RGB, cst, det_fly")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 512 samples, monitor cap 64, name suffix _smoke")
    ap.add_argument("--samples-per-epoch", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--monitor-cap", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--name", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.samples_per_epoch, a.monitor_cap = 2, 512, 64
    L.parse(a.config)
    if a._child:
        return child(a)

    doms = L.parse_domains(a.train_domains)
    name = L.run_name(a.config + L.domain_tag(doms), a.seed, a.epochs) + ("_smoke" if a.smoke else "")
    run_dir = S.REPO / "runs" / name
    last = run_dir / "weights" / "last.pt"
    base = [sys.executable, "-u", os.path.abspath(__file__), "--_child", "--config", a.config, "--seed", str(a.seed),
            "--train-domains", ",".join(doms),
            "--epochs", str(a.epochs), "--batch", str(a.batch), "--workers", str(a.workers)]
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
        sys.exit(f"{run_dir} already exists -- use --resume; earlier runs are never overwritten")
    resume_ckpt = a.resume_from or (str(last) if a.resume and last.exists() else None)
    cmd = base + ["--name", name] + (["--resume-from", resume_ckpt] if resume_ckpt else [])
    corpus_dir = S.REPO / "b_pipeline" / S.CORPUS
    mk, wn = L.parse(a.config)
    meta = {"run_name": name, "series": "aerotrack_v3", "config": a.config, "model": mk, "model_definition": L.MODELS[mk],
            "pretrained_sha256": file_sha256(S.REPO / "c_model" / L.MODELS[mk]["pretrained"]),
            "window": L.WINDOW_KIND[wn], "window_name": wn, "gap": L.WINDOWS[wn], "layout": L.MODELS[mk]["yaml"],
            "train_domains": list(doms),
            "seed": a.seed, "epochs": a.epochs, "batch": a.batch, "init": "function_preserving",
            "recipe": "c_model/train.py STAGES['baseline']", "smoke": bool(a.smoke),
            "overrides": {"samples_per_epoch": a.samples_per_epoch, "monitor_cap": a.monitor_cap},
            "data": {"corpus": S.CORPUS, "manifest_sha256": file_sha256(corpus_dir / "manifest.csv"),
                     "splits_sha256": file_sha256(corpus_dir / "splits.json"), "train_split": "corpus v2 TRAIN (all three datasets)"},
            "checkpoint_policy": "last.pt is the scored checkpoint (fixed before any run); best.pt is kept but not used"}
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_meta.json"
    attempt = {"timestamp_start": S.now(), "command": " ".join(cmd), "resumed_from": resume_ckpt,
               "epochs_on_disk_at_start": _epochs_on_disk(run_dir), "git": S.git(), "env": env_snapshot()}
    if meta_path.exists():
        prior = json.loads(meta_path.read_text(encoding="utf-8"))
        changed = {k: (prior.get(k), meta[k]) for k in ("config", "seed", "epochs", "batch", "overrides", "data", "train_domains")
                   if prior.get(k) != meta[k]}
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
    S.write_json_atomic(meta, meta_path)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(S.REPO))
    attempt.update({"timestamp_end": S.now(), "seconds": round(time.time() - t0, 1), "return_code": rc,
                    "epochs_on_disk_at_end": _epochs_on_disk(run_dir)})
    meta.update(timestamp_end=attempt["timestamp_end"], return_code=rc,
                train_seconds=round(sum(x.get("seconds") or 0 for x in meta["attempts"]), 1))
    for tag in ("best", "last"):
        p = run_dir / "weights" / f"{tag}.pt"
        meta[f"{tag}_pt_sha256"] = file_sha256(p) if p.exists() else None
    S.write_json_atomic(meta, meta_path)
    S.ledger({"id": f"TR:{name}", "type": "training", "run": name, "config": a.config, "seed": a.seed, "return_code": rc,
              "last_pt_sha256": meta.get("last_pt_sha256"), "smoke": bool(a.smoke)})
    print(f"[launcher] {name}: exit {rc} after {attempt['seconds'] / 3600:.2f} h (attempt {attempt['attempt']})", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
