#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py - AeroTrack-Net Staged Training Engine
==============================================
Orchestrates staged training runs across unified Anti-UAV, CST Anti-UAV,
and Det-Fly datasets.

Supported Stages:
  * overfit   : Single-batch sanity check to verify loss convergence.
  * smoke     : Small-scale plumbing validation (loaders, checkpointing, metrics).
  * pilot     : Domain-specific pilot run on CST Anti-UAV.
  * baseline  : Full-corpus baseline training (AeroTrack-Net v1).
  * finetune  : Cross-domain generalization & avian hard-negative rejection.
  * ablations : Model ablations (lossless SPD downsampling vs strided conv,
                temporal windowing vs single frame, weighted sampling).

Sequence-level disjoint splits are loaded from splits.json to prevent temporal leakage.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

# Enable expandable segments on platforms supporting it to reduce memory fragmentation
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "b_pipeline"))

from stacked_dataset import (AeroTrackDataset, TemporalAugment, aerotrack_collate,
                             build_split_samples, assert_no_leakage,
                             load_label_cache, SampleIndex)
from imbalance_sampler import (
    build_weighted_sampler, DEFAULT_WEIGHTS, DEFAULT_DOMAIN_SHARES, FINETUNE_WEIGHTS
)
from aerotrack_trainer import (AeroTrackTrainer, TrainConfig, build_model,
                               transfer_pretrained, LoaderWorkerDied)


UNIFIED = ROOT / "b_pipeline" / "data_unified"
MANIFEST = UNIFIED / "manifest.csv"
SPLITS = UNIFIED / "splits.json"
LABEL_CACHE_PATH = UNIFIED / "label_cache.npz"
MODEL_CFG = HERE / "yolo11-spd.yaml"
MODEL_CFG_P2 = HERE / "yolo11-spd-p2.yaml"
PRETRAINED = HERE / "yolo11n.pt"
NC = 2

# Validation caps, per dataset. Full val is ~140k frames; scoring all of it every
# epoch would cost far more than the epoch itself. The signal stays honest: all
# three datasets represented, spread across sequences, sequence-disjoint from
# train. The FULL sweep runs once at the end via c_model/val.py, which is where
# the reported numbers come from.
#
# Sized against MEASURED validation throughput, not against I/O. Validation is
# NMS-bound, not loader-bound, and the cost is worst exactly when the model is
# worst: on a 2-epoch checkpoint, conf 0.005 puts ~30 candidate boxes on every
# image and the whole pass runs at 7.3 img/s (44 img/s once the classifier
# separates). At 4,000 images that is a 9-minute validation against a 5-minute
# epoch -- the run would spend more time being measured than being trained.
# 2,000 images caps the worst case near 4.5 min and the steady state near 45 s.
VAL_CAPS = {"anti_uav": 1000, "cst": 700, "det_fly": 300}

STAGES = {
    "overfit": dict(  # S0
        epochs=200, samples_per_epoch=None, batch=8, augment=False,
        # EVERY dataset must appear here. `subsample` leaves a dataset that is
        # absent from `caps` UNCAPPED, so omitting det_fly silently turned "one
        # fixed batch" into 4,423 samples x 200 epochs -- a seven-hour run
        # masquerading as a two-minute smoke test.
        train_cap={"anti_uav": 8, "cst": 8, "det_fly": 0}, val_from_train=True,
        # nbs == batch -> accumulate 1. With a 16-image corpus an epoch is two
        # iterations, so an accumulation window of 8 (nbs 64 / batch 8) would
        # never close: `(i+1) % 8` is never 0 for i in {0,1}, and the optimiser
        # would stop stepping the moment warmup finished ramping it up.
        nbs=8,
        val_interval=20, patience=0, lr0=2e-3, warmup_epochs=0.5,
        name="s0_overfit", weighted_sampler=False, workers=0),
    "smoke": dict(    # S1
        epochs=3, samples_per_epoch=2000, batch=8, augment=True,
        val_caps={"anti_uav": 200, "cst": 200, "det_fly": 100},
        val_interval=1, patience=0, name="s1_smoke", workers=4),
    "pilot": dict(    # S2
        epochs=20, samples_per_epoch=15000, batch=8, augment=True,
        datasets=["cst"], val_caps={"cst": 2000},
        val_interval=1, patience=8, name="s2_pilot_cst"),
    "baseline": dict(  # S3
        epochs=150, samples_per_epoch=25000, batch=8, augment=True,
        val_interval=1, patience=30, name="aerotrack_spd_v1"),
    "finetune": dict(  # S4 - Cross-Domain Generalization & Bird Rejection Fine-Tuning
        pretrained=None,  # starts from trained AeroTrack checkpoint, not raw COCO
        epochs=25, samples_per_epoch=25000, batch=32, nbs=64,
        lr0=2e-4, lrf=0.02, warmup_epochs=2.0, patience=15,
        name="aerotrack_spd_finetuned",
        domain_balanced=True,
        domain_shares={"anti_uav": 0.40, "cst": 0.35, "det_fly": 0.25},
        sampler_weights=FINETUNE_WEIGHTS,
        class_weights=[1.0, 4.5],
        hyp=dict(cls=1.5, box=7.5, dfl=1.5),
        val_caps={"anti_uav": 1000, "cst": 700, "det_fly": 300},
        augment_kwargs=dict(translate=0.20, scale=(0.80, 1.30), gray_p=0.25, noise_std=0.02),
        reset_epochs=True,
        workers=4,
    ),

}


ABLATIONS = {
    "no_spd":      dict(model_cfg="yolo11-stride.yaml", name="abl_no_spd"),
    "single_frame": dict(temporal_gap=0, name="abl_single_frame"),
    "no_sampler":  dict(weighted_sampler=False, name="abl_no_sampler"),
    "no_letterbox": dict(letterbox=False, name="abl_no_letterbox"),
    "scratch":     dict(pretrained=None, name="abl_scratch"),
    "p2_head":     dict(model_cfg="yolo11-spd-p2.yaml", name="abl_p2_head"),
    "no_augment":  dict(augment=False, name="abl_no_augment"),
}


# --------------------------------------------------------------------------- #
#  Guards
# --------------------------------------------------------------------------- #
def require_gpu(require_sm120=True):
    if not torch.cuda.is_available():
        raise SystemExit(
            "\n[ABORT] No CUDA GPU detected. AeroTrack-Net training is staged for "
            "the OMEN 16 GPU and must not run on CPU-only hardware.\n"
            "        Install the CUDA build:  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu130\n")
    cap = torch.cuda.get_device_capability(0)
    want = f"sm_{cap[0]}{cap[1]}"
    arch = torch.cuda.get_arch_list()
    if require_sm120 and want not in arch:
        raise SystemExit(
            f"\n[ABORT] This PyTorch wheel has no {want} kernels "
            f"({torch.cuda.get_device_name(0)}).\n"
            f"        arch_list = {arch}\n"
            f"        `cuda.is_available()` is True but the first real kernel "
            f"launch would fail with 'no kernel image is available'.\n"
            f"        Fix: pip install --force-reinstall torch torchvision "
            f"--index-url https://download.pytorch.org/whl/cu130\n")
    return torch.device("cuda:0")


def require_corpus():
    missing = [p for p in (MANIFEST, SPLITS) if not p.exists()]
    if missing:
        raise SystemExit(
            "\n[ABORT] corpus not prepared: " + ", ".join(str(m) for m in missing) +
            "\n        Run, in order:\n"
            "          python a_inspection/1_uncompress_datasets.py\n"
            "          python b_pipeline/1_standardize_annotations.py --full-antiuav\n"
            "          python b_pipeline/3_prepare_training.py\n")


# --------------------------------------------------------------------------- #
#  Corpus assembly
# --------------------------------------------------------------------------- #
def subsample(samples, caps, seed=0):
    """Deterministically cap the number of samples per dataset.

    Sampling is spread evenly across each dataset's sample list (which is sorted
    by sequence then frame), so the cap keeps sequence diversity rather than
    taking the first N frames of the first few sequences.

    NOTE: a dataset that does not appear in `caps` is left UNCAPPED, not
    dropped. Every caller must therefore name all three datasets (use 0 to
    exclude one) -- see STAGES["overfit"]["train_cap"]."""
    if not caps:
        return samples
    by = {}
    for s in samples:
        by.setdefault(s["dataset"], []).append(s)
    out = []
    for dk, lst in by.items():
        cap = caps.get(dk)
        if cap is None or cap >= len(lst):
            out.extend(lst)
        else:
            idx = np.linspace(0, len(lst) - 1, cap).round().astype(int)
            out.extend(lst[i] for i in np.unique(idx))
    return out


def build_corpus(stage_cfg, temporal_gap=1, verbose=True):
    """-> (train_samples, val_samples, label_cache)."""
    require_corpus()
    cache = load_label_cache(LABEL_CACHE_PATH)
    if cache is None and verbose:
        print("[warn] label_cache.npz missing — labels will be read from disk "
              "(slow). Run b_pipeline/3_prepare_training.py.")
    datasets = stage_cfg.get("datasets")
    g = max(int(temporal_gap), 1)

    train = build_split_samples(MANIFEST, SPLITS, "train", datasets=datasets,
                                temporal_gap=g)
    if stage_cfg.get("val_from_train"):
        # S0 only: overfit a fixed batch, so val IS the train batch by design.
        # Restrict to frames that actually carry a target -- overfitting a batch
        # of empty frames drives the loss to zero while proving nothing.
        with_t = [s for s in train if s.get("has_target")]
        train = subsample(with_t or train, stage_cfg.get("train_cap"), seed=0)
        val = list(train)
    else:
        train = subsample(train, stage_cfg.get("train_cap"), seed=0)
        val = build_split_samples(MANIFEST, SPLITS, "val", datasets=datasets,
                                  temporal_gap=g)
        val = subsample(val, stage_cfg.get("val_caps", VAL_CAPS), seed=0)
        assert_no_leakage(train, val)            # roadmap T4 — non-negotiable
    if verbose:
        from collections import Counter
        print(f"[corpus] train {len(train):,} samples "
              f"{dict(Counter(s['dataset'] for s in train))}")
        print(f"[corpus] val   {len(val):,} samples "
              f"{dict(Counter(s['dataset'] for s in val))}")
        print(f"[corpus] train sequences {len({s['seq_key'] for s in train}):,} | "
              f"val sequences {len({s['seq_key'] for s in val}):,} | disjoint OK")
    return train, val, cache


def _worker_init(worker_id):
    """Per-worker RNG isolation for augmentation + single-threaded OpenCV."""
    import cv2
    cv2.setNumThreads(0)
    info = torch.utils.data.get_worker_info()
    if info is not None and getattr(info.dataset, "augment", None) is not None:
        base = int(torch.initial_seed()) % (2 ** 31)
        info.dataset.augment.reseed(base + worker_id * 7919)


def free_commit_gb():
    """Windows COMMIT headroom (RAM + pagefile), in GB. -1 if unavailable.

    `psutil.virtual_memory().available` is the wrong number on Windows: it
    reports free *physical* RAM, while what actually bounds process count here
    is the system COMMIT CHARGE against its limit. A machine can show 8 GB of
    free RAM and still refuse the next allocation."""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return -1.0
        return m.ullAvailPageFile / 1024 ** 3
    except Exception:
        return -1.0


# MEASURED on this machine, not estimated. A spawned DataLoader worker costs
# ~2.3 GB of COMMIT before it has queued a single batch -- 1.6 GB of that is
# just `import numpy` + `import torch` in a fresh interpreter (Windows spawn
# gives every worker its own), plus its copy of the 55 MB sample index and its
# prefetch queue. Probe, 4 workers on the real train loader:
#
#     system commit consumed by the loader   8.95 GB   -> 2.24 GB per worker
#     each child process reported            2.04-2.17 GB
#
# This is why `train_inflight_gb` alone could never bound the problem: it
# budgets the QUEUED TENSORS (~236 MB per worker), which are a tenth of the
# real cost. The interpreter is the cost.
COMMIT_PER_WORKER_GB = 2.4
COMMIT_RESERVE_GB = 4.0        # for the parent, the OS, and whatever else runs

# COMMIT is not the only ceiling. This box has a 35.6 GB commit limit against
# 23.6 GB of RAM, so a commit-only calculation happily authorises seven workers
# and then pays for three of them out of the pagefile -- the run survives, and
# every other application on the machine crawls. `phys_per_worker_gb` is the
# WORKING SET a worker actually keeps resident (measured lower than its commit
# charge, because most of a fresh interpreter's commit is reservation, not
# resident pages), and the physical-RAM ceiling is applied alongside the commit
# one. Whichever is tighter wins.
PHYS_PER_WORKER_GB = 1.3
# The trainer process itself, measured at the point where the worker count is
# decided: torch imported, manifest read, but the CUDA context and the model not
# yet allocated. Those land shortly afterwards, so this is what still has to be
# paid for out of the same free RAM the workers are being planned against.
PHYS_PER_PARENT_GB = 1.5
PARENT_COMMIT_GB = 2.5


def reserve_gb_default():
    """Host memory to leave untouched, in GB.

    Overridable from the environment so a detached background run inherits the
    notebook's budget: `AEROTRACK_RESERVE_GB=8` means "training may not plan for
    the last 8 GB — that is the user's browser and slide decks."
    """
    try:
        v = float(os.environ.get("AEROTRACK_RESERVE_GB", COMMIT_RESERVE_GB))
    except (TypeError, ValueError):
        return COMMIT_RESERVE_GB
    return max(v, 0.0)


def free_physical_gb():
    """Free physical RAM in GB, -1 if psutil is unavailable."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return -1.0


def plan_workers(requested, verbose=True, per_worker_gb=COMMIT_PER_WORKER_GB,
                 reserve_gb=None, phys_per_worker_gb=PHYS_PER_WORKER_GB,
                 phys_reserve_gb=None):
    """Cap the worker count at what COMMIT can actually pay for.

    This exists because the failure it prevents is silent and expensive: ask for
    six workers on a box with 17 GB of commit headroom and the run dies partway
    through an epoch with `DataLoader worker (pid(s) N) exited unexpectedly`, or
    a bare `numpy._core._exceptions._ArrayMemoryError` relayed through a
    `SystemError` -- neither of which names memory, a worker count, or a fix.
    Measured here: 6 workers needed ~17.8 GB against 17.6 GB free, and died at
    iteration 270 of epoch 0.

    The old knob was a number a human typed with no relationship to the limit it
    had to satisfy. This makes it self-limiting, so an over-ambitious config
    degrades to a slower run instead of a dead one.

    Two ceilings are applied, not one. COMMIT decides whether a worker can be
    CREATED; free physical RAM decides whether it can run without pushing
    somebody else's working set into the pagefile. On a machine that is also
    driving three slide decks and a browser, the second is usually the binding
    one, and ignoring it produces a run that technically survives while making
    the desktop unusable."""
    requested = int(requested)
    if requested <= 0:
        return 0
    if reserve_gb is None:
        reserve_gb = reserve_gb_default()
    if phys_reserve_gb is None:
        phys_reserve_gb = reserve_gb

    free = free_commit_gb()
    by_commit = (requested if free < 0
                 else int(max(free - reserve_gb, 0) // per_worker_gb))

    phys = free_physical_gb()
    by_phys = (requested if phys < 0
               else int(max(phys - phys_reserve_gb, 0) // phys_per_worker_gb))

    safe = max(0, min(requested, by_commit, by_phys))
    if verbose:
        note = "" if safe == requested else f"  <- CAPPED from {requested}"
        print(f"[commit] {free:.1f} GB free commit, reserve {reserve_gb:.1f} GB, "
              f"~{per_worker_gb:.1f} GB per worker -> {by_commit} workers")
        print(f"[ram]    {phys:.1f} GB free RAM, reserve {phys_reserve_gb:.1f} GB, "
              f"~{phys_per_worker_gb:.1f} GB per worker -> {by_phys} workers")
        print(f"[plan]   using {safe} workers{note}")
        if safe < requested:
            print(f"[plan]   {requested} workers would need "
                  f"~{requested * per_worker_gb + reserve_gb:.1f} GB commit / "
                  f"~{requested * phys_per_worker_gb + phys_reserve_gb:.1f} GB RAM. "
                  f"Lower AEROTRACK_RESERVE_GB, or close other apps, to use more.")
        if safe == 0:
            print("[plan]   falling back to workers=0 (in-process loading). "
                  "Slow but it cannot be killed by the OS.")
    return safe


def plan_loader_memory(cfg, workers, budget_gb, verbose=False, label=""):
    """Choose `prefetch_factor` from a BYTE budget instead of a batch count.

    This is the fix for "DataLoader worker exited unexpectedly".

    PyTorch's `prefetch_factor` counts BATCHES, and its default of 2-4 is tuned
    for 3-channel RGB. Our sample is nine channels of 640x640 uint8 =
    **3.69 MB**, three times a normal one, and the batch multiplies it again:

        prefetch_factor 4 x batch 32 x 3.69 MB = 472 MB  IN FLIGHT PER WORKER
        x 6 workers                            = 2.8 GB  on the train loader
        x2 because the val loader was configured identically and persistent
                                               = 5.7 GB, held for the whole run

    Windows commits that up front. With the kernel already holding the manifest
    and the sample index, the machine hit its commit limit and the OS killed a
    worker -- which surfaces as a queue timeout, not as an out-of-memory error,
    which is why the message is so unhelpful.

    So: pick the largest prefetch depth that fits `budget_gb` of in-flight
    tensor data across all workers, floor of 2 so the pipeline still overlaps.
    """
    if workers <= 0:
        return None
    per_sample = 9 * cfg.imgsz * cfg.imgsz            # uint8, 9 channels
    per_batch = per_sample * cfg.batch
    per_worker = (budget_gb * 1024 ** 3) / workers
    pf = max(2, min(int(cfg.prefetch_factor), int(per_worker // max(per_batch, 1))))
    if verbose:
        total = pf * workers * per_batch / 1024 ** 3
        print(f"[loader] {label}: {workers} workers x prefetch {pf} x batch "
              f"{cfg.batch} = {total:.2f} GB in flight "
              f"({per_batch/1024**2:.0f} MB per batch, budget {budget_gb} GB)")
    return pf


def build_loaders(train_samples, val_samples, cfg: TrainConfig, stage_cfg,
                  label_cache=None, augment_kwargs=None, letterbox=True,
                  sampler_weights=None, verbose=True):
    aug = None
    if stage_cfg.get("augment", True):
        aug = TemporalAugment(seed=cfg.seed, **(augment_kwargs or {}))

    train_ds = AeroTrackDataset(train_samples, img_size=(cfg.imgsz, cfg.imgsz),
                                augment=aug, letterbox=letterbox,
                                out_dtype="uint8", label_cache=label_cache)
    val_ds = AeroTrackDataset(val_samples, img_size=(cfg.imgsz, cfg.imgsz),
                              augment=None, letterbox=letterbox,
                              out_dtype="uint8", return_meta=True,
                              label_cache=label_cache)

    sampler = None
    if stage_cfg.get("weighted_sampler", True):
        n = cfg.samples_per_epoch or len(train_ds)
        domain_bal = stage_cfg.get("domain_balanced", False)
        domain_shares = stage_cfg.get("domain_shares", DEFAULT_DOMAIN_SHARES)
        sw = sampler_weights or stage_cfg.get("sampler_weights", DEFAULT_WEIGHTS)
        sampler, w, cats, _ = build_weighted_sampler(
            train_ds.index, weights=sw,
            num_samples=n, cache_path=None, verbose=verbose,
            domain_balanced=domain_bal, domain_shares=domain_shares)
    elif cfg.samples_per_epoch and cfg.samples_per_epoch < len(train_ds):

        from torch.utils.data import RandomSampler
        sampler = RandomSampler(train_ds, replacement=True,
                                num_samples=cfg.samples_per_epoch)

    # ---- train loader ---------------------------------------------------- #
    # The requested worker count is a CEILING, not an instruction: what the
    # machine can actually commit decides. See plan_workers().
    n_workers = plan_workers(
        cfg.workers, verbose=verbose,
        reserve_gb=reserve_gb_default() + PARENT_COMMIT_GB,
        phys_reserve_gb=reserve_gb_default() + PHYS_PER_PARENT_GB)
    if n_workers != cfg.workers:
        cfg.workers = n_workers
    tr_kw = dict(num_workers=n_workers, pin_memory=True,
                 collate_fn=aerotrack_collate, worker_init_fn=_worker_init)
    if n_workers > 0:
        tr_kw.update(persistent_workers=cfg.persistent_workers,
                     prefetch_factor=plan_loader_memory(
                         cfg, n_workers, cfg.train_inflight_gb, verbose, "train"))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch,
                              shuffle=(sampler is None), sampler=sampler,
                              drop_last=True, **tr_kw)

    # ---- val loader ------------------------------------------------------ #
    # Fewer workers, and NOT persistent. Validation runs for a couple of minutes
    # once per epoch; persistent val workers would sit idle for the other 95% of
    # the time still holding their prefetch queues and their copy of the sample
    # index. That doubled the process count and the committed memory for no
    # throughput at all. Respawning them each epoch costs a few seconds.
    # The val loader's workers are spawned WHILE the train loader's are alive
    # (persistent_workers keeps them resident across the validation pass), so
    # they are charged against the same commit budget -- with the train
    # workers' share already spent.
    val_workers = plan_workers(
        min(cfg.val_workers, cfg.workers), verbose=verbose,
        reserve_gb=reserve_gb_default() + n_workers * COMMIT_PER_WORKER_GB,
        phys_reserve_gb=reserve_gb_default() + n_workers * PHYS_PER_WORKER_GB)
    va_kw = dict(num_workers=val_workers, pin_memory=False,
                 collate_fn=aerotrack_collate, worker_init_fn=_worker_init)
    if val_workers > 0:
        va_kw.update(persistent_workers=False,
                     prefetch_factor=plan_loader_memory(
                         cfg, val_workers, cfg.val_inflight_gb, verbose, "val"))
    val_loader = DataLoader(val_ds, batch_size=max(cfg.batch, 8), shuffle=False,
                            drop_last=False, **va_kw)
    return train_ds, val_ds, train_loader, val_loader


def release(*objs, free_manifest=False, verbose=True):
    """Tear down trainers/contexts and actually give the memory back.

    `del trainer` is not enough. A DataLoader with `persistent_workers=True`
    keeps its worker PROCESSES alive for as long as the loader object exists,
    and each of those workers holds its own copy of the sample index plus its
    prefetch queue. If a notebook builds a job for the pre-flight and then
    builds another for training, the first job's workers are still resident --
    that is a full duplicate corpus and a second set of processes, and it is how
    a 24 GB machine runs out of commit and has a worker killed.

    Call this between stages.
    """
    import gc
    for o in objs:
        if o is None:
            continue
        holders = o.values() if isinstance(o, dict) else [o]
        for h in holders:
            it = getattr(h, "_iterator", None)      # live worker pool
            if it is not None:
                try:
                    it._shutdown_workers()
                except Exception:
                    pass
                try:
                    h._iterator = None
                except Exception:
                    pass
        if isinstance(o, dict):
            o.clear()
    if free_manifest:
        free_manifest_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if verbose:
        try:
            import psutil, os
            rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
            print(f"[release] workers shut down, caches dropped — "
                  f"process rss now {rss:.2f} GB")
        except Exception:
            print("[release] workers shut down, caches dropped")


def free_manifest_cache():
    """Drop the memoised manifest DataFrame (763,819 rows).

    Kept between builds it is pure overhead; re-reading it costs well under a
    minute and there are at most a handful of builds in a session."""
    from stacked_dataset import _MANIFEST_CACHE
    _MANIFEST_CACHE.clear()


# --------------------------------------------------------------------------- #
#  Assembly
# --------------------------------------------------------------------------- #
def build(stage="baseline", ablation=None, device=None, overrides=None,
          augment_kwargs=None, sampler_weights=None, verbose=True):
    """-> (trainer, ctx) fully wired and pre-flightable, but NOT started."""
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; pick one of {list(STAGES)}")
    stage_cfg = dict(STAGES[stage])
    abl = dict(ABLATIONS[ablation]) if ablation else {}
    if ablation and ablation not in ABLATIONS:
        raise SystemExit(f"unknown ablation {ablation!r}; pick from {list(ABLATIONS)}")
    stage_cfg.update({k: v for k, v in abl.items()
                      if k in ("name", "datasets", "weighted_sampler", "augment")})

    device = device or require_gpu()

    ov = dict(overrides or {})
    cfg_name = abl.get("model_cfg") or ov.pop("model_cfg", None) or MODEL_CFG.name
    model_cfg = Path(cfg_name)
    if not model_cfg.is_absolute():
        model_cfg = HERE / model_cfg
    if not model_cfg.exists():
        raise SystemExit(f"[ABORT] model cfg not found: {model_cfg}")
    letterbox = abl.get("letterbox", ov.pop("letterbox", True))
    temporal_gap = abl.get("temporal_gap", ov.pop("temporal_gap", 1))
    single_frame = temporal_gap == 0
    pretrained = abl["pretrained"] if "pretrained" in abl else ov.pop("pretrained", PRETRAINED)

    cfg_kwargs = {k: v for k, v in stage_cfg.items()
                  if k in TrainConfig().to_dict()}
    cfg_kwargs.setdefault("name", stage_cfg.get("name", stage))
    cfg_kwargs["project"] = str(ROOT / "runs")
    cfg_kwargs.update({k: v for k, v in ov.items() if k in TrainConfig().to_dict()})
    cfg = TrainConfig(**cfg_kwargs)

    # S0's corpus has to be sized in BATCHES, not in images. The train loader
    # uses drop_last=True, so a fixed 16-image cap yields ZERO batches the
    # moment the notebook's VRAM probe raises the batch to 32 -- and the
    # overfit test then reports "no batch carried a ground-truth box", which
    # names neither the cause nor the fix. Two batches, whatever the batch is.
    if stage_cfg.get("val_from_train"):
        stage_cfg["train_cap"] = {"anti_uav": cfg.batch, "cst": cfg.batch,
                                  "det_fly": 0}

    # ---- corpus ---------------------------------------------------------- #
    train_s, val_s, cache = build_corpus(stage_cfg, temporal_gap=max(temporal_gap, 1),
                                         verbose=verbose)
    if single_frame:
        # 3-channel ablation is expressed as g=0: t-1 == t == t+1, so the stack
        # carries no motion at all while the tensor shape stays [9,640,640] and
        # every other component is untouched. That isolates the TrackNet claim.
        for s in train_s + val_s:
            s["prev_path"] = s["next_path"] = s["cur_path"]

    aug_kw = augment_kwargs or stage_cfg.get("augment_kwargs", None)
    smp_w = sampler_weights or stage_cfg.get("sampler_weights", None)
    train_ds, val_ds, train_loader, val_loader = build_loaders(
        train_s, val_s, cfg, stage_cfg, label_cache=cache,
        augment_kwargs=aug_kw, sampler_weights=smp_w,
        letterbox=letterbox, verbose=verbose)

    # ---- model ----------------------------------------------------------- #
    hyp = stage_cfg.get("hyp", None)
    cw = stage_cfg.get("class_weights", getattr(cfg, "class_weights", None))
    model = build_model(model_cfg, ch=9, nc=NC, verbose=False, hyp=hyp, class_weights=cw)
    n_par = sum(p.numel() for p in model.parameters())
    n_spd = sum(1 for m in model.modules() if type(m).__name__ == "SPDConv")
    transferred = (0, n_par)
    if pretrained and Path(pretrained).exists():
        transferred = transfer_pretrained(model, pretrained, verbose=verbose)
    elif pretrained:
        print(f"[transfer] {Path(pretrained).name} not found — training from "
              f"scratch. Download it once with:\n"
              f"    python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\"")

    trainer = AeroTrackTrainer(model, train_loader, val_loader, device, cfg)


    # Keep the sequence-key SETS, not the sample lists. The training list is
    # 378,759 dicts of eleven keys -- a few hundred MB that nothing downstream
    # reads, since the dataset now holds the compact SampleIndex instead. The
    # leakage assertion only ever needed the key sets.
    train_keys = {s["seq_key"] for s in train_s}
    val_keys = {s["seq_key"] for s in val_s}
    ctx = {
        "stage": stage, "ablation": ablation, "cfg": cfg, "stage_cfg": stage_cfg,
        "model_cfg": str(model_cfg), "letterbox": letterbox,
        "temporal_gap": temporal_gap, "device": device,
        "train_seq_keys": train_keys, "val_seq_keys": val_keys,
        "n_train": len(train_s), "n_val": len(val_s),
        "val_samples": val_s,          # small (a few thousand), used for eval
        "train_ds": train_ds, "val_ds": val_ds,
        "train_loader": train_loader, "val_loader": val_loader,
        "n_params": n_par, "n_spdconv": n_spd, "transferred": transferred,
        "label_cache": cache,
    }
    del train_s
    import gc as _gc
    _gc.collect()
    if verbose:
        print(f"[model] {model_cfg.name} | {n_par/1e6:.2f} M params | "
              f"{n_spd} SPDConv layers | ch=9 nc={NC}")
        print(f"[engine] batch {cfg.batch} x accumulate {trainer.accumulate} "
              f"= effective {cfg.batch*trainer.accumulate} | "
              f"{len(train_loader):,} iters/epoch | "
              f"{'bf16' if trainer.amp_dtype is torch.bfloat16 else 'fp32'}")
    return trainer, ctx



# --------------------------------------------------------------------------- #
#  Running a stage that survives the night
# --------------------------------------------------------------------------- #
def _is_worker_death(exc):
    """Is this the OS killing a DataLoader worker, rather than a real bug?

    It arrives in three disguises. `LoaderWorkerDied` is our own translation of
    PyTorch's "worker exited unexpectedly". A worker that manages to RAISE
    instead of being killed reports the underlying allocation failure, and
    numpy's `_ArrayMemoryError` comes back across the process boundary wrapped
    as a bare `SystemError` -- the exception type is lost, only the text
    survives. Everything else must propagate: a real bug that silently retried
    three times with fewer workers would be much worse than a crash."""
    if isinstance(exc, (LoaderWorkerDied, MemoryError)):
        return True
    msg = f"{type(exc).__name__}: {exc}"
    return any(k in msg for k in ("exited unexpectedly", "Unable to allocate",
                                  "ArrayMemoryError", "paging file",
                                  "not enough memory"))


def train_stage(stage, device, overrides=None, augment_kwargs=None,
                sampler_weights=None, ablation=None, resume_from=None,
                max_attempts=3, verbose=True, on_built=None):
    """Build and run one stage, resuming itself if a worker is killed.

    On Windows the loaders are bounded by COMMIT (RAM + pagefile), and commit is
    a machine-wide resource -- a browser opening forty tabs at hour nine of a
    fifteen-hour run can push the total over the limit, and the OS then kills
    whichever process asks for memory next. That is usually a DataLoader worker,
    and PyTorch surfaces it as an unrecoverable error in the middle of an epoch.

    Nothing about the RUN is broken when that happens: `last.pt` was written at
    the end of the previous epoch, optimiser and EMA states included. So the
    correct response is not to lose fourteen hours -- it is to shut the workers
    down, halve the worker count (which halves the in-flight footprint), and
    resume. Each retry costs one epoch of repeated work.

    `on_built(trainer, ctx)` runs once, on the FIRST successful build only --
    that is where pre-flight belongs, so the checks share the build they are
    checking instead of paying for a second one. It is deliberately not re-run
    on a retry: the corpus and the model are identical, only the worker count
    changed, and re-running T1-T8 would add minutes to every recovery.

    Returns (trainer, ctx, history). The caller still owns `release()`.
    """
    ov = dict(overrides or {})
    last_err = None
    built_once = False
    for attempt in range(1, max_attempts + 1):
        trainer, ctx = build(stage, ablation=ablation, device=device,
                             overrides=ov, augment_kwargs=augment_kwargs,
                             sampler_weights=sampler_weights, verbose=verbose)
        if on_built is not None and not built_once:
            built_once = True
            on_built(trainer, ctx)
        ckpt = trainer.save_dir / "weights" / "last.pt"
        resume = resume_from if attempt == 1 else (str(ckpt) if ckpt.exists() else None)
        try:
            hist = trainer.train(resume_from=resume)
            return trainer, ctx, hist
        except BaseException as e:                       # noqa: BLE001
            if not _is_worker_death(e) or attempt == max_attempts:
                raise
            last_err = e
            workers = int(ov.get("workers", ctx["cfg"].workers))
            new_workers = max(workers // 2, 0)
            print()
            print(f"[recover] attempt {attempt}/{max_attempts}: a "
                  f"DataLoader worker was lost to host-memory pressure, "
                  f"not to a bug in the model.", flush=True)
            print(f"          {type(e).__name__}: "
                  f"{str(e).splitlines()[0]}", flush=True)
            print(f"          workers {workers} -> {new_workers}; resuming "
                  f"from {ckpt.name if ckpt.exists() else 'scratch'}.",
                  flush=True)
            release(trainer, ctx, free_manifest=True)
            del trainer, ctx
            ov["workers"] = new_workers
    raise last_err


# --------------------------------------------------------------------------- #
#  Running the whole staircase in one process
# --------------------------------------------------------------------------- #
def run_stages(stages, device=None, stage_overrides=None, augment_kwargs=None,
               sampler_weights=None, temporal_gap=None, preflight=True,
               resume_from=None, auto_resume=True, tag=None, verbose=True):
    """S0 -> S1 -> S3 in ONE process, releasing everything between stages.

    The notebook used to drive this loop itself, which meant the Jupyter kernel
    held the trainer, the loaders and their worker processes for the entire
    multi-hour run. Moving the loop here is what makes `--background` possible:
    the kernel launches this and exits, and the run no longer depends on a
    browser tab staying open.

    S0 is special-cased: it is not a training stage, it is the probe that asks
    whether the loss can fall at all on a fixed batch. A failed S0 aborts the
    whole sequence rather than burning hours on a pipeline that cannot learn.
    """
    import preflight as P

    device = device or require_gpu()
    stage_overrides = stage_overrides or {}
    results, done_preflight = {}, [not preflight]

    def _preflight(trainer, ctx):
        if done_preflight[0]:
            return
        done_preflight[0] = True
        print("\n[preflight] T1-T8 on the job that is about to run", flush=True)
        checks = [
            P.check_dataset(ctx["val_ds"], batch_size=min(4, ctx["cfg"].batch),
                            imgsz=ctx["cfg"].imgsz),
            P.check_boundary(ctx["val_ds"]),
            P.check_leakage(ctx["train_seq_keys"], ctx["val_seq_keys"]),
            P.check_collate(trainer, ctx["val_ds"], batch_size=min(4, ctx["cfg"].batch)),
            P.check_checkpoint_roundtrip(trainer, device, imgsz=ctx["cfg"].imgsz),
            P.check_throughput(ctx["train_loader"], n_batches=20),
            P.check_vram(trainer, ctx["train_ds"], ctx["cfg"].batch, device),
        ]
        allow = ("T4",) if ctx["stage_cfg"].get("val_from_train") else ()
        P.run_all(checks, raise_on_fail=True, allow_fail=allow)

    for stage in stages:
        ov = dict(stage_overrides.get(stage, {}))
        if temporal_gap is not None:
            ov.setdefault("temporal_gap", temporal_gap)
        print("\n" + "#" * 78)
        print(f"#  STAGE: {stage}   ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        print("#" * 78, flush=True)

        if stage == "overfit":
            iters = int(ov.pop("iters", 200))
            ov["workers"] = 0                     # 16 images; workers are pure cost
            trainer, ctx = build("overfit", device=device, overrides=ov,
                                 augment_kwargs=augment_kwargs,
                                 sampler_weights=sampler_weights, verbose=verbose)
            r = P.check_overfit(trainer, iters=iters)
            print("\n" + str(r), flush=True)
            ok = r.ok
            release(trainer, ctx, free_manifest=True)
            del trainer, ctx
            results["overfit"] = {"ok": bool(ok), "detail": r.detail}
            if not ok:
                raise SystemExit(
                    "\n[ABORT] S0 FAILED — the loss did not fall on a fixed batch.\n"
                    "        Do NOT start the long run. Look, in this order, at:\n"
                    "        (1) the label files for that batch, (2) the collate\n"
                    "        dict (T5), (3) whether the boxes survive the letterbox\n"
                    "        transform, (4) the assigner's topk on 9 px targets.\n")
            continue

        resume = resume_from if stage == stages[-1] else None
        # Auto-resume, final stage only. Losing an interrupted 15-hour run to a
        # relaunch that silently restarts at epoch 0 -- and then overwrites the
        # best.pt it was about to beat -- is the most expensive mistake
        # available here, and last.pt already carries the optimiser and EMA
        # state needed to avoid it. Earlier stages are minutes long and are
        # better re-run clean. To restart the final stage deliberately, pass a
        # new --name or delete its run directory.
        if stage == "finetune" and resume is None:
            _name = ov.get("name") or STAGES[stage]["name"]
            _last = ROOT / "runs" / str(_name) / "weights" / "last.pt"
            if _last.exists():
                resume = str(_last)
                print(f"[resume] {_last} exists -> continuing interrupted fine-tuning run.", flush=True)
            else:
                _base_best = ROOT / "runs" / "aerotrack_spd_v1" / "weights" / "best.pt"
                if _base_best.exists():
                    resume = str(_base_best)
                    print(f"[finetune] Initializing weights from baseline: {_base_best.relative_to(ROOT)}", flush=True)
        elif resume is None and auto_resume and stage == stages[-1]:
            _name = ov.get("name") or STAGES[stage]["name"]
            _last = ROOT / "runs" / str(_name) / "weights" / "last.pt"
            if _last.exists():
                resume = str(_last)
                print(f"[resume] {_last} exists -> continuing that run rather "
                      f"than overwriting it. Delete the run directory, or pass "
                      f"--name, to start over.", flush=True)


        tr, cx, hist = train_stage(
            stage, device=device, overrides=ov, augment_kwargs=augment_kwargs,
            sampler_weights=sampler_weights, resume_from=resume,
            verbose=verbose, on_built=_preflight)
        results[stage] = {
            "run_dir": str(tr.save_dir),
            "best": str(tr.save_dir / "weights" / "best.pt"),
            "best_fitness": float(tr.best_fitness),
            "best_epoch": int(tr.best_epoch),
            "epochs_run": len(hist),
        }
        release(tr, cx, free_manifest=True)
        del tr, cx

    # A completion record, so a relaunch -- a re-run of the notebook, say --
    # can tell "already finished" from "never started" and refuse to train from
    # scratch over the top of a model that already exists.
    if tag:
        BG_DIR.mkdir(parents=True, exist_ok=True)
        (BG_DIR / f"{tag}_results.json").write_text(
            json.dumps({"stages": list(stages),
                        "finished": time.strftime("%Y%m%d_%H%M%S"),
                        "results": results}, indent=2), encoding="utf-8")
    print("\nfinished stages:", list(results), flush=True)
    return results


# --------------------------------------------------------------------------- #
#  Detached background execution
# --------------------------------------------------------------------------- #
BG_DIR = ROOT / "runs" / "_bg"
LOG_DIR = ROOT / "logs"

# Windows process-creation flags. DETACHED_PROCESS is the one that matters: the
# child gets no console and no parent handle, so closing Jupyter, the terminal,
# or the VS Code window does not take the run with it.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000
_PRIORITY_FLAGS = {
    "idle": 0x00000040,           # IDLE_PRIORITY_CLASS
    "below_normal": 0x00004000,   # BELOW_NORMAL_PRIORITY_CLASS
    "normal": 0x00000020,         # NORMAL_PRIORITY_CLASS
}


def set_process_priority(level="below_normal", verbose=True):
    """Drop this process's CPU priority without touching its GPU share.

    CPU priority and GPU scheduling are separate: the GPU keeps running our
    kernels flat out, while Windows stops letting the loader's JPEG decoding
    preempt whatever is redrawing a slide. That is the whole trick behind
    "train in the background and keep working" -- the training job never
    competes for the CPU that the foreground application is using.
    """
    try:
        import psutil
        p = psutil.Process()
        p.nice({"idle": psutil.IDLE_PRIORITY_CLASS,
                "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                "normal": psutil.NORMAL_PRIORITY_CLASS}[level])
        if verbose:
            print(f"[priority] this process -> {level}", flush=True)
        return True
    except Exception as e:                                   # noqa: BLE001
        if verbose:
            print(f"[priority] could not set {level}: {e!r}", flush=True)
        return False


def limit_cpu_threads(n=4, verbose=True):
    """Cap the intra-op thread pool.

    Torch defaults to one thread per physical core (16 here). Almost nothing in
    this run is a CPU tensor op -- the loss and the model are on the GPU, and
    the loaders are separate processes with OpenCV already pinned to one thread
    -- so those threads buy no throughput and cost the foreground apps a lot of
    scheduler pressure."""
    try:
        torch.set_num_threads(int(n))
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(int(n)))
    if verbose:
        print(f"[cpu] torch intra-op threads -> {n}", flush=True)


def _state_path(tag="train"):
    return BG_DIR / f"{tag}.json"


def _pid_alive(pid, expect="train.py"):
    """Is `pid` still our training process? Guards against PID reuse."""
    try:
        import psutil
        p = psutil.Process(int(pid))
        if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
            return False
        return any(expect in str(a) for a in (p.cmdline() or []))
    except Exception:
        return False


def launch_background(stages=("overfit", "smoke", "baseline"), *, batch=None,
                      workers=None, epochs=None, samples_per_epoch=None,
                      lr0=None, patience=None, imgsz=None, name=None,
                      augment=None, sampler_weights=None, temporal_gap=None,
                      reserve_gb=None, priority="below_normal", preflight=True,
                      resume=None, tag="train", force=False, verbose=True):
    """Start the staged run as a DETACHED process and return immediately.

    Returns the state dict (pid, log path, command). The caller -- notebook cell
    or shell -- is free to exit; the run is not its child in any sense that
    matters, so it survives kernel restarts, closed terminals and logouts.

    Everything the run needs is on the command line or in `runs/_bg/<tag>.json`,
    so a monitor cell can attach to it later with no shared memory at all.
    """
    if _running_state(tag) is not None:
        raise SystemExit(
            f"[ABORT] a background run is already active (tag {tag!r}).\n"
            f"        Check it with train.bg_status(), or stop it with "
            f"train.bg_stop(), before starting another.\n"
            f"        Two trainers on one 8 GB card is an OOM, not a speedup.")

    done = BG_DIR / f"{tag}_results.json"
    if done.exists() and not force:
        _when = json.loads(done.read_text(encoding="utf-8")).get("finished")
        raise SystemExit(
            f"[ABORT] tag {tag!r} already has a FINISHED run ({_when}).\n"
            f"        Launching again would train from scratch over the top of "
            f"it.\n"
            f"        Continue that run:        resume=<run>/weights/last.pt\n"
            f"        Start a genuinely new one: pass a new name= or tag=\n"
            f"        Overwrite deliberately:    delete {done}\n")

    BG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"bg_{tag}_{stamp}.log"

    cmd = [sys.executable, "-u", str(HERE / "train.py"),
           "--stages", ",".join(stages), "--priority", priority]
    for flag, val in (("--batch", batch), ("--workers", workers),
                      ("--epochs", epochs), ("--samples-per-epoch", samples_per_epoch),
                      ("--lr0", lr0), ("--patience", patience), ("--imgsz", imgsz),
                      ("--name", name), ("--resume", resume)):
        if val is not None:
            cmd += [flag, str(val)]
    # A detached process shares no memory with its launcher, so the notebook's
    # AUGMENT / SAMPLER_WEIGHTS / TEMPORAL_GAP have to travel as data. They go
    # as a file rather than as command-line JSON: quoting a nested dict through
    # cmd.exe is a class of bug nobody should have to debug at 2 a.m.
    tuning = {k: v for k, v in (("augment", augment),
                                ("sampler_weights", sampler_weights),
                                ("temporal_gap", temporal_gap)) if v is not None}
    if tuning:
        tune_path = BG_DIR / f"{tag}_tuning.json"
        tune_path.write_text(json.dumps(tuning, indent=2), encoding="utf-8")
        cmd += ["--tuning", str(tune_path)]
    if not preflight:
        cmd.append("--no-preflight")
    cmd += ["--tag", tag]          # the child records completion under this tag

    env = dict(os.environ)
    if reserve_gb is not None:
        env["AEROTRACK_RESERVE_GB"] = str(float(reserve_gb))
    if os.name != "nt":       # see the note at the top of this file
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    flags = 0
    if os.name == "nt":
        flags = (_DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP |
                 _CREATE_NO_WINDOW | _PRIORITY_FLAGS.get(priority, 0))

    log = open(log_path, "w", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=flags, close_fds=True)
    finally:
        log.close()          # the child holds its own inherited handle

    # Record where each stage will write, so a monitor started days later (or
    # after a kernel restart) can find the results without re-deriving them.
    run_names = {s: STAGES[s]["name"] for s in stages if s in STAGES}
    if name and stages:
        run_names[stages[-1]] = name          # --name renames the FINAL stage only

    state = {"tag": tag, "pid": proc.pid, "log": str(log_path), "cmd": cmd,
             "stages": list(stages), "run_names": run_names, "priority": priority,
             "reserve_gb": reserve_gb, "started": stamp,
             "started_epoch_s": time.time()}
    _state_path(tag).write_text(json.dumps(state, indent=2), encoding="utf-8")

    if verbose:
        print(f"[bg] launched pid {proc.pid} ({priority} priority, detached)")
        print(f"[bg] stages : {' -> '.join(stages)}")
        print(f"[bg] log    : {log_path}")
        print(f"[bg] the run no longer depends on this kernel — you can close "
              f"Jupyter and it keeps going.")
    return state


def _running_state(tag="train"):
    """The state dict of a live run with this tag, or None."""
    p = _state_path(tag)
    if not p.exists():
        return None
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return st if _pid_alive(st.get("pid", -1)) else None


def bg_tail(path, n=40):
    """Last `n` lines of a log, read without loading the whole file."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block, data = 8192, b""
        while size > 0 and data.count(b"\n") <= n:
            step = min(block, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
    return data.decode("utf-8", "replace").splitlines()[-n:]


def bg_status(tag="train", lines=25, verbose=True):
    """-> dict describing the background run: alive, stage, log tail, progress."""
    p = _state_path(tag)
    if not p.exists():
        if verbose:
            print(f"[bg] no run has been launched with tag {tag!r}.")
        return {"exists": False, "alive": False}
    st = json.loads(p.read_text(encoding="utf-8"))
    alive = _pid_alive(st.get("pid", -1))
    tail = bg_tail(st.get("log", ""), lines)

    # Progress comes off results.csv, which the trainer rewrites every epoch --
    # a file, not a socket, so it is readable from any process at any time.
    run_names = st.get("run_names") or {s: STAGES[s]["name"]
                                        for s in st.get("stages", []) if s in STAGES}
    progress = {}
    for stage, name in run_names.items():
        run_dir = ROOT / "runs" / str(name)
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            progress[stage] = {"run_dir": str(run_dir), "csv": str(csv_path),
                               "epochs": _csv_rows(csv_path),
                               "mtime": csv_path.stat().st_mtime}
    out = {"exists": True, "alive": alive, "pid": st.get("pid"),
           "log": st.get("log"), "stages": st.get("stages"),
           "run_names": run_names, "started": st.get("started"),
           "tail": tail, "progress": progress}
    if verbose:
        el = (time.time() - st.get("started_epoch_s", time.time())) / 3600
        print(f"[bg] pid {st.get('pid')} — "
              f"{'RUNNING' if alive else 'not running (finished or stopped)'} — "
              f"{el:.2f} h since launch")
        print(f"[bg] log: {st.get('log')}")
        for stage, pr in progress.items():
            print(f"[bg] {stage}: {pr['epochs']} epochs recorded")
        if tail:
            print("-" * 78)
            for ln in tail:
                print(ln)
            print("-" * 78)
    return out


def _csv_rows(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


def bg_stop(tag="train", verbose=True):
    """Stop the background run. `last.pt` is intact; relaunch with resume."""
    st = _running_state(tag)
    if st is None:
        if verbose:
            print("[bg] nothing running to stop.")
        return False
    try:
        import psutil
        p = psutil.Process(st["pid"])
        for c in p.children(recursive=True):        # DataLoader workers
            try:
                c.terminate()
            except Exception:
                pass
        p.terminate()
        p.wait(timeout=30)
    except Exception as e:                                   # noqa: BLE001
        if verbose:
            print(f"[bg] terminate failed: {e!r}")
        return False
    if verbose:
        print(f"[bg] stopped pid {st['pid']}. The last completed epoch is in "
              f"weights/last.pt — relaunch with resume= to continue.")
    return True


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="AeroTrack-Net training",
        epilog="Typical use:\n"
               "  train.py --stages overfit,smoke,baseline --background\n"
               "  train.py --status          # how is it doing?\n"
               "  train.py --stop            # stop it; last.pt is intact",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default=None, choices=list(STAGES),
                    help="run exactly one stage (the old behaviour)")
    ap.add_argument("--stages", default=None,
                    help="comma-separated staircase, e.g. overfit,smoke,baseline")
    ap.add_argument("--ablation", default=None, choices=list(ABLATIONS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32,
                    help="batch size (default 32: max safe on 8GB VRAM)")
    ap.add_argument("--workers", type=int, default=4,
                    help="workers (default 4: safe for >=6GB host RAM buffer)")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--samples-per-epoch", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--tuning", default=None,
                    help="JSON file with {augment, sampler_weights, "
                         "temporal_gap} — how train.ipynb hands its config "
                         "cell to a detached run")
    # --- sharing the machine ------------------------------------------------
    ap.add_argument("--background", action="store_true",
                    help="detach: run in a separate process that survives this "
                         "shell, the Jupyter kernel and a closed terminal")
    ap.add_argument("--priority", default="below_normal",
                    choices=list(_PRIORITY_FLAGS),
                    help="CPU priority class (GPU share is unaffected)")
    ap.add_argument("--reserve-gb", type=float, default=None,
                    help="host RAM to leave for everything else, in GB. The "
                         "worker count is planned against what is left.")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--tag", default="train", help="background run identifier")
    ap.add_argument("--no-auto-resume", action="store_true",
                    help="start the final stage from scratch even if its "
                         "weights/last.pt exists (which it will overwrite)")
    ap.add_argument("--force", action="store_true",
                    help="launch even though this tag already has a finished run")
    ap.add_argument("--status", action="store_true", help="report and exit")
    ap.add_argument("--stop", action="store_true", help="stop a background run")
    args = ap.parse_args()

    if args.status:
        bg_status(args.tag)
        return
    if args.stop:
        bg_stop(args.tag)
        return

    stages = ([s.strip() for s in args.stages.split(",") if s.strip()]
              if args.stages else [args.stage or "baseline"])
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; pick from {list(STAGES)}")

    tuning = {}
    if args.tuning:
        tpath = Path(args.tuning)
        if not tpath.exists():
            raise SystemExit(f"[ABORT] --tuning file not found: {tpath}")
        tuning = json.loads(tpath.read_text(encoding="utf-8"))

    if args.background:
        launch_background(
            stages, batch=args.batch, workers=args.workers, epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch, lr0=args.lr0,
            patience=args.patience, imgsz=args.imgsz, name=args.name,
            augment=tuning.get("augment"),
            sampler_weights=tuning.get("sampler_weights"),
            temporal_gap=tuning.get("temporal_gap"),
            reserve_gb=args.reserve_gb, priority=args.priority,
            preflight=not args.no_preflight, resume=args.resume, tag=args.tag,
            force=args.force)
        return

    if args.reserve_gb is not None:
        os.environ["AEROTRACK_RESERVE_GB"] = str(args.reserve_gb)
    set_process_priority(args.priority)
    limit_cpu_threads(args.cpu_threads)

    # Two kinds of override, and conflating them is a real bug: `batch` and
    # `workers` describe the MACHINE and belong to every stage, while `epochs`,
    # `name` and `patience` describe the SCHEDULE and belong only to the final
    # one. Pushing --epochs 150 into "smoke" turns a ten-minute plumbing check
    # into the full run, under the smoke stage's name and directory.
    machine = {k: v for k, v in (
        ("batch", args.batch), ("workers", args.workers),
        ("imgsz", args.imgsz), ("lr0", args.lr0)) if v is not None}
    schedule = {k: v for k, v in (
        ("epochs", args.epochs), ("samples_per_epoch", args.samples_per_epoch),
        ("patience", args.patience), ("name", args.name)) if v is not None}
    over = {**machine, **schedule}

    device = require_gpu()

    # An ablation, or an explicit single --stage, keeps the original one-shot
    # path: build it, check it, train it. The staircase goes through run_stages.
    if args.ablation or (args.stage and not args.stages):
        trainer, ctx = build(args.stage or "baseline", args.ablation,
                             device=device, overrides=over,
                             augment_kwargs=tuning.get("augment"),
                             sampler_weights=tuning.get("sampler_weights"))
        if not args.no_preflight:
            import preflight as P
            P.run_all([
                P.check_environment(),
                P.check_dataset(ctx["val_ds"], batch_size=min(4, ctx["cfg"].batch)),
                P.check_boundary(ctx["val_ds"]),
                P.check_leakage(ctx["train_seq_keys"], ctx["val_seq_keys"]),
                P.check_collate(trainer, ctx["val_ds"], batch_size=min(4, ctx["cfg"].batch)),
                P.check_checkpoint_roundtrip(trainer, ctx["device"]),
                P.check_throughput(ctx["train_loader"]),
                P.check_vram(trainer, ctx["train_ds"], ctx["cfg"].batch, ctx["device"]),
            ], allow_fail=("T4",) if ctx["stage_cfg"].get("val_from_train") else ())
        trainer.train(resume_from=args.resume)
        return

    stage_overrides = {s: dict(machine) for s in stages}
    stage_overrides[stages[-1]].update(schedule)
    run_stages(stages, device=device, stage_overrides=stage_overrides,
               augment_kwargs=tuning.get("augment"),
               sampler_weights=tuning.get("sampler_weights"),
               temporal_gap=tuning.get("temporal_gap"),
               preflight=not args.no_preflight, resume_from=args.resume,
               auto_resume=not args.no_auto_resume, tag=args.tag)


if __name__ == "__main__":
    main()
