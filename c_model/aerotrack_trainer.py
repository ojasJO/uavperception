#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aerotrack_trainer.py - AeroTrack-Net Native Training Engine
===========================================================
A high-efficiency training loop engineered for 9-channel SPD-Conv architectures
with temporal frame stacking.

Key Capabilities:
  * 9-Channel Temporal Ingestion: Full native support for [B, 9, H, W] tensors.
  * Micro-Target Fitness Metric: Evaluates checkpoints with explicit weighting
    on sub-16px detection accuracy (micro-AP50).
  * Mixed Precision Stability: Native bfloat16 mixed precision execution.
  * Partial Weight Transfer: Transfers pre-trained backbone parameters while
    initializing SPD-Conv space-to-depth adapters.
  * Size-Stratified Validation: Automated performance breakdown across Micro,
    Small, and Standard target scales.
"""

import csv
import gc
import json
import math
import time
import inspect
import datetime
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  SPDConv registration (identical to the verified Phase-C dry-run path)
# --------------------------------------------------------------------------- #
def register_spdconv():
    """Teach Ultralytics' YAML parser about SPDConv.

    `parse_model` decides a module's output-channel bookkeeping from a hard-coded
    `base_modules` set. SPDConv is not in it, so we patch the one branch that
    matters and re-exec the function in a namespace that knows the symbol. This
    is the exact path proven by 1_cpu_dry_run.py."""
    from custom_modules import SPDConv
    import ultralytics.nn.tasks as T
    T.SPDConv = SPDConv
    try:
        import ultralytics.nn.modules as M
        M.SPDConv = SPDConv
    except Exception:
        pass
    # IDEMPOTENT. After the first call `T.parse_model` is a function compiled
    # from a string, so it has no source file and `inspect.getsource` raises
    # OSError("could not get source code"). Calling this twice in one process is
    # normal -- build a model, then load a checkpoint -- so the second call must
    # be a no-op rather than an exception.
    if getattr(T, "_AEROTRACK_SPD_PATCHED", False):
        return SPDConv
    try:
        src = inspect.getsource(T.parse_model)
    except (OSError, TypeError):
        T._AEROTRACK_SPD_PATCHED = True      # already replaced by our own copy
        return SPDConv
    if "or m is SPDConv" not in src:
        patched = src.replace("if m in base_modules:",
                              "if m in base_modules or m is SPDConv:", 1)
        g = dict(vars(T)); g["SPDConv"] = SPDConv
        exec(compile(patched, "<patched_parse_model>", "exec"), g)
        T.parse_model = g["parse_model"]
    T._AEROTRACK_SPD_PATCHED = True
    return SPDConv


# --------------------------------------------------------------------------- #
#  Model construction + partial COCO transfer
# --------------------------------------------------------------------------- #
DEFAULT_HYP = dict(
    box=7.5, cls=0.5, dfl=1.5,          # v8DetectionLoss gains (stock)
    label_smoothing=0.0, nbs=64,
)


def build_model(cfg, ch=9, nc=2, verbose=False, hyp=None, class_weights=None):
    """Construct the 9-channel SPD-Conv DetectionModel and attach loss hyps."""
    register_spdconv()
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel(cfg=str(cfg), ch=ch, nc=nc, verbose=verbose)
    h = dict(DEFAULT_HYP)
    if hyp:
        h.update(hyp)
    model.args = SimpleNamespace(**h)      # v8DetectionLoss reads model.args
    if class_weights is not None:
        model.class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
    return model



def transfer_pretrained(model, weights, verbose=True):
    """Copy every COCO tensor whose NAME and SHAPE both match (roadmap §4.1).

    Most of this network is architecturally stock yolo11n -- only the five
    downsampling convs became SPDConv and the head went 80 -> 2 classes. Those
    tensors are skipped automatically by the shape check; everything else (the
    C3k2 blocks, SPPF, C2PSA, the whole neck) transfers and starts from COCO
    features instead of noise."""
    import ultralytics.nn.tasks as UT
    sd_pre = None
    # Ultralytics renames its loader every couple of minor versions
    # (attempt_load_weights -> attempt_load_one_weight -> load_checkpoint), so
    # try what is present and fall back to a plain torch.load.
    for fn_name in ("load_checkpoint", "attempt_load_one_weight", "attempt_load_weights"):
        fn = getattr(UT, fn_name, None)
        if fn is None:
            continue
        try:
            out = fn(str(weights))
            pre = out[0] if isinstance(out, tuple) else out
            sd_pre = pre.float().state_dict()
            break
        except Exception:
            continue
    if sd_pre is None:
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        m = ckpt.get("ema") or ckpt.get("model")
        sd_pre = (m.float().state_dict() if hasattr(m, "state_dict") else m)
    sd_new = model.state_dict()
    copied = {k: v for k, v in sd_pre.items()
              if k in sd_new and sd_new[k].shape == v.shape}
    model.load_state_dict(copied, strict=False)
    if verbose:
        n_bb = sum(1 for k in copied if k.startswith("model.") and
                   int(k.split(".")[1]) < 11)
        print(f"[transfer] {len(copied):,}/{len(sd_new):,} tensors copied from "
              f"{Path(weights).name} ({len(copied)/max(len(sd_new),1)*100:.1f}%) "
              f"— {n_bb} in the backbone")
        skipped = [k for k in sd_new if k not in copied]
        print(f"[transfer] {len(skipped):,} left at init "
              f"(SPDConv stems + Detect head): e.g. {skipped[:3]}")
    return len(copied), len(sd_new)


# --------------------------------------------------------------------------- #
#  Optimiser / schedule
# --------------------------------------------------------------------------- #
def build_optimizer(model, name="AdamW", lr=1e-3, momentum=0.937, decay=5e-4):
    """Three parameter groups: weights get decay, BN and biases never do.

    Applying weight decay to BatchNorm scales fights the normalisation itself;
    on a 4 M-param network trained on tiny objects that is a measurable loss."""
    g_decay, g_no_decay, g_bias = [], [], []
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for module in model.modules():
        for pname, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if pname == "bias":
                g_bias.append(p)
            elif isinstance(module, bn):
                g_no_decay.append(p)
            else:
                g_decay.append(p)
    if name.lower() == "adamw":
        opt = torch.optim.AdamW(g_no_decay, lr=lr, betas=(momentum, 0.999),
                                weight_decay=0.0)
    elif name.lower() == "adam":
        opt = torch.optim.Adam(g_no_decay, lr=lr, betas=(momentum, 0.999),
                               weight_decay=0.0)
    else:
        opt = torch.optim.SGD(g_no_decay, lr=lr, momentum=momentum, nesterov=True)
    opt.add_param_group({"params": g_decay, "weight_decay": decay})
    opt.add_param_group({"params": g_bias, "weight_decay": 0.0})
    return opt


def cosine_lambda(epochs, lrf=0.01):
    """1 -> lrf over `epochs`, cosine. Returned as a plain callable on epoch."""
    return lambda e: ((1 - math.cos(min(e, epochs) * math.pi / epochs)) / 2) * (lrf - 1) + 1


def compute_loss(criterion, preds, batch):
    """Call `v8DetectionLoss` and normalise its return shape.

    Ultralytics has changed this contract more than once. In 8.4.x
    `criterion(preds, batch)` returns `(loss_vector[3] * batch_size,
    {"box_loss": ..., "cls_loss": ..., "dfl_loss": ...})` -- the first element
    is a VECTOR, not a scalar, so `loss.backward()` on it raises "grad can be
    implicitly created only for scalar outputs"; older versions returned a
    scalar and a detached tensor. This accepts either and always hands back
    `(scalar_for_backward, per_component_tensor_for_logging)`.
    """
    out = criterion(preds, batch)
    loss, items = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
    if loss.ndim > 0:
        loss = loss.sum()
    if isinstance(items, dict):
        items = torch.stack([v.detach().float().reshape(()) for v in items.values()])
    elif items is None:
        items = torch.zeros(3, device=loss.device)
    else:
        items = items.detach().float().flatten()
    return loss, items


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
class TrainConfig(SimpleNamespace):
    """Every knob in one place, printed into the run directory for the record."""

    def __init__(self, **kw):
        d = dict(
            # --- schedule ---------------------------------------------------
            epochs=150,
            iters_per_epoch=None,     # None -> full pass; else budgeted draws/batch
            samples_per_epoch=25_000, # the iteration budget (roadmap §3.2)
            warmup_epochs=3.0,
            lr0=1e-3, lrf=0.01, momentum=0.937, weight_decay=5e-4,
            optimizer="AdamW", cos_lr=True,
            patience=30,
            # --- batching ---------------------------------------------------
            batch=8, nbs=64,          # nbs = nominal batch -> grad accumulation
            workers=6, prefetch_factor=4, persistent_workers=True,
            # In-flight tensor budgets, in GB. `prefetch_factor` counts BATCHES,
            # and a 9-channel 640x640 uint8 batch of 32 is 118 MB -- three times
            # what the PyTorch defaults assume. These caps are what actually
            # bound the loaders' memory; see train.plan_loader_memory().
            train_inflight_gb=1.0, val_inflight_gb=0.35, val_workers=2,
            imgsz=640,
            # --- numerics ---------------------------------------------------
            amp=True, amp_dtype="bf16", grad_clip=10.0,
            # This card also drives the desktop. A CUDA OOM caused by a browser
            # or a slideshow briefly taking VRAM costs one batch; only a
            # sustained run of them means the batch size is genuinely too big.
            max_consecutive_oom=8,
            channels_last=True, cudnn_benchmark=True, compile=False,
            ema=True, ema_decay=0.9999, ema_tau=2000,
            # --- validation -------------------------------------------------
            val_interval=1, val_max_images=6000,
            # 0.005, not the 0.001 used for the FINAL report. At 0.001 an
            # early-epoch model floods NMS with 8,400 near-random candidates per
            # image and Ultralytics' NMS trips its own time limit; per-epoch
            # validation then costs more than the epoch. The final numbers in
            # val.py still use 0.001 — this threshold only steers checkpoint
            # selection, where the ranking matters and the last 0.4% of AP does not.
            val_conf=0.005, val_iou=0.7, val_max_det=30,
            fitness_weights=(0.30, 0.35, 0.35),   # (mAP50, mAP50-95, micro AP50)
            # --- domain & class weighting -----------------------------------
            class_weights=None,                   # e.g. [1.0, 4.5] for bird suppression
            reset_epochs=False,                   # True when fine-tuning from a checkpoint
            # --- bookkeeping ------------------------------------------------
            project="runs", name="aerotrack_spd_v1", seed=0,
            save_period=-1, log_interval=50,
        )
        d.update(kw)
        super().__init__(**d)

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in vars(self).items()}


class LoaderWorkerDied(RuntimeError):
    """A DataLoader worker was killed by the OS, with the reason spelled out."""


def _guard_loader(loader, cfg):
    """Iterate a DataLoader, translating a killed worker into a real diagnosis.

    PyTorch reports a killed worker as `RuntimeError: DataLoader worker (pid(s)
    N) exited unexpectedly`, raised from a bare `queue.Empty` -- the message says
    nothing at all about the cause. On Windows the cause is almost always that
    the system ran out of COMMIT (not RAM -- commit counts the pagefile too),
    because every worker holds `prefetch_factor x batch` samples of 3.69 MB each
    plus its own copy of the sample index. A worker that is KILLED cannot raise,
    so all the parent sees is a queue that stopped answering.
    """
    it = enumerate(loader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except RuntimeError as e:
            msg = str(e)
            if "worker" not in msg or "exited unexpectedly" not in msg:
                raise
            mb = 9 * cfg.imgsz * cfg.imgsz * cfg.batch / 1024 ** 2
            raise LoaderWorkerDied(
                msg + "\n\n"
                "  A worker was KILLED, not crashed: it never raised, it stopped\n"
                "  answering. On Windows that is the OS reclaiming committed\n"
                "  memory -- check FreeVirtualMemory, not free RAM.\n\n"
                f"  This run: batch {cfg.batch} x 9ch x {cfg.imgsz}^2 uint8 = "
                f"{mb:.0f} MB per batch; {cfg.workers} train workers; in-flight\n"
                f"  budget {cfg.train_inflight_gb} GB train + {cfg.val_inflight_gb}"
                " GB val.\n\n"
                "  Fix, in order of effect:\n"
                "    1. Restart the kernel. An earlier build's sample index and\n"
                "       its persistent workers are probably still resident.\n"
                "    2. Lower WORKERS (6 -> 4) or BATCH (32 -> 16) in the config\n"
                "       cell of train.ipynb.\n"
                "    3. Lower cfg.train_inflight_gb.\n\n"
                "  Headroom check (PowerShell):\n"
                "    Get-CimInstance Win32_OperatingSystem | fl FreeVirtualMemory"
            ) from e

# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
class AeroTrackTrainer:
    """Owns the loop: forward, loss, accumulate, step, EMA, validate, checkpoint."""

    def __init__(self, model, train_loader, val_loader, device, cfg: TrainConfig,
                 val_records_fn=None):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_records_fn = val_records_fn

        self.save_dir = Path(cfg.project) / cfg.name
        (self.save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "args.json").write_text(
            json.dumps(cfg.to_dict(), indent=2, default=str), encoding="utf-8")

        self.criterion = self.model.init_criterion()
        cw = getattr(self.model, "class_weights", None)
        if cw is None and getattr(cfg, "class_weights", None) is not None:
            cw = torch.as_tensor(cfg.class_weights, dtype=torch.float32, device=self.device)
            self.model.class_weights = cw
        if cw is not None:
            self.criterion.class_weights = cw.to(self.device).view(1, 1, -1)
        self.loss_names = tuple(getattr(self.criterion, "loss_names",
                                        ("box_loss", "cls_loss", "dfl_loss")))
        self.optimizer = build_optimizer(self.model, cfg.optimizer, cfg.lr0,
                                         cfg.momentum, cfg.weight_decay)
        self.lf = (cosine_lambda(cfg.epochs, cfg.lrf) if cfg.cos_lr
                   else (lambda e: 1 - (1 - cfg.lrf) * e / cfg.epochs))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lf)

        self.ema = None
        if cfg.ema:
            from ultralytics.utils.torch_utils import ModelEMA
            self.ema = ModelEMA(self.model, decay=cfg.ema_decay, tau=cfg.ema_tau)

        self.amp_dtype = None
        if cfg.amp and device.type == "cuda":
            self.amp_dtype = (torch.bfloat16 if cfg.amp_dtype == "bf16"
                              else torch.float16)
        self.scaler = (torch.amp.GradScaler("cuda")
                       if self.amp_dtype is torch.float16 else None)

        self.accumulate = max(1, round(cfg.nbs / cfg.batch))
        # Transient CUDA OOMs are tolerated (the desktop shares this card); a
        # sustained run of them is a real ceiling and must stop the run.
        self.n_oom = 0
        self.max_consecutive_oom = int(getattr(cfg, "max_consecutive_oom", 8))
        self.best_fitness = -1.0
        self.best_epoch = -1
        self.epoch = 0
        self.history = []
        self.csv_path = self.save_dir / "results.csv"
        self._t_start = None

        if cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # ---- one batch ------------------------------------------------------- #
    def _to_batch(self, imgs, targets):
        """Collated tensors -> the dict v8DetectionLoss consumes."""
        imgs = imgs.to(self.device, non_blocking=True)
        if imgs.dtype == torch.uint8:
            imgs = imgs.float().div_(255.0)
        if self.cfg.channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        # INVARIANT: rows are appended image by image, so batch_idx comes out
        # sorted and contiguous (0,0,1,1,1,2,...). v8DetectionLoss.preprocess
        # depends on exactly that -- it computes each box's within-image slot as
        # `arange(n) - offsets[batch_idx]`. Interleave the rows and the index
        # goes negative, which surfaces as a CUDA device-side assert that kills
        # the context rather than as an exception. Preflight T5 checks it.
        cls, box, bidx = [], [], []
        for i, t in enumerate(targets):
            if t.numel():
                cls.append(t[:, 0:1])
                box.append(t[:, 1:5])
                bidx.append(torch.full((t.shape[0],), i, dtype=torch.float32))
        return {
            "img": imgs,
            "cls": (torch.cat(cls, 0) if cls else torch.zeros(0, 1)).to(self.device),
            "bboxes": (torch.cat(box, 0) if box else torch.zeros(0, 4)).to(self.device),
            "batch_idx": (torch.cat(bidx, 0) if bidx else torch.zeros(0)).to(self.device),
        }

    # ---- warmup ---------------------------------------------------------- #
    def _warmup(self, ni, nw):
        """Linear LR + momentum warmup. Essential with a partially-pretrained
        backbone: the SPDConv stems are random while everything downstream is
        COCO-trained, so a cold high LR wrecks the transferred features."""
        xi = [0, nw]
        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.cfg.nbs / self.cfg.batch]).round()))
        for j, g in enumerate(self.optimizer.param_groups):
            warm_lr = 0.0 if j != 2 else 0.1 * self.cfg.lr0   # bias group starts hot
            g["lr"] = np.interp(ni, xi, [warm_lr, self.cfg.lr0 * self.lf(self.epoch)])
            if "momentum" in g:
                g["momentum"] = np.interp(ni, xi, [0.8, self.cfg.momentum])
            if "betas" in g:
                g["betas"] = (float(np.interp(ni, xi, [0.8, self.cfg.momentum])),
                              g["betas"][1])

    # ---- one epoch ------------------------------------------------------- #
    def train_one_epoch(self):
        cfg = self.cfg
        self.model.train()
        nb = len(self.train_loader)
        nw = max(round(cfg.warmup_epochs * nb), 100)
        tloss = torch.zeros(3, device=self.device)
        t0 = time.time()
        n_seen = 0
        self.optimizer.zero_grad(set_to_none=True)

        oom_run = 0                 # consecutive OOMs — see the handler below
        for i, batch in _guard_loader(self.train_loader, self.cfg):
            imgs, targets = batch[0], batch[1]
            ni = i + nb * self.epoch
            if ni <= nw:
                self._warmup(ni, nw)

            # VRAM on a laptop is a SHARED resource, and this run is expected to
            # sit alongside a browser and a slide deck on the same 8 GB card.
            # The desktop's share is not constant: a video starts playing, a
            # presentation enters slideshow mode, and 300 MB vanishes for a few
            # seconds. Letting that kill a fifteen-hour run would be absurd when
            # the correct response is to drop ONE batch of 24 out of 25,000 and
            # carry on. A sustained run of them is different -- that is a real
            # ceiling, not a transient -- so it still raises.
            try:
                b = self._to_batch(imgs, targets)
                if self.amp_dtype is not None:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        preds = self.model(b["img"])
                        loss, loss_items = compute_loss(self.criterion, preds, b)
                else:
                    preds = self.model(b["img"])
                    loss, loss_items = compute_loss(self.criterion, preds, b)

                if not torch.isfinite(loss):
                    print(f"[warn] non-finite loss at iter {i} — batch skipped "
                          f"(items={loss_items.tolist()})")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            except torch.cuda.OutOfMemoryError as e:
                oom_run += 1
                self.n_oom += 1
                self.optimizer.zero_grad(set_to_none=True)
                b = preds = loss = loss_items = None
                gc.collect()
                torch.cuda.empty_cache()
                free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
                print(f"[oom] iter {i}: batch skipped, {free_gb:.2f} GB free on "
                      f"the card ({self.n_oom} this run, {oom_run} in a row). "
                      f"Something else on the desktop is holding VRAM.",
                      flush=True)
                if oom_run >= self.max_consecutive_oom:
                    raise torch.cuda.OutOfMemoryError(
                        f"{oom_run} consecutive CUDA OOMs at batch "
                        f"{self.cfg.batch} — this is a real ceiling, not a "
                        f"transient. Lower BATCH (or RESERVE_VRAM_GB) in the "
                        f"config cell and resume from weights/last.pt.") from e
                continue
            oom_run = 0

            if (i + 1) % self.accumulate == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer); self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema:
                    self.ema.update(self.model)

            tloss = (tloss * i + loss_items.to(tloss.device)) / (i + 1)
            n_seen += imgs.shape[0]
            if cfg.log_interval and (i + 1) % cfg.log_interval == 0:
                el = time.time() - t0
                mem = (torch.cuda.max_memory_allocated() / 1024 ** 3
                       if self.device.type == "cuda" else 0)
                print(f"  ep{self.epoch:>3d} {i+1:>5d}/{nb}  "
                      f"box {tloss[0]:.4f} cls {tloss[1]:.4f} dfl {tloss[2]:.4f}  "
                      f"lr {self.optimizer.param_groups[0]['lr']:.2e}  "
                      f"{n_seen/max(el,1e-6):.1f} img/s  vram {mem:.2f} GB",
                      flush=True)
        return tloss.cpu().numpy(), n_seen / max(time.time() - t0, 1e-6)

    # ---- validation ------------------------------------------------------ #
    @torch.no_grad()
    def validate(self):
        """Size-stratified evaluation on the held-out sequences."""
        import evaluation as E
        model = (self.ema.ema if self.ema else self.model)
        model.eval()
        recs = E.collect_records(model, self.val_loader, self.device,
                                 imgsz=self.cfg.imgsz, conf=self.cfg.val_conf,
                                 iou=self.cfg.val_iou, max_det=self.cfg.val_max_det,
                                 amp_dtype=self.amp_dtype, progress=False)
        overall = E.score(recs)
        micro = E.score(recs, size_bin="micro")
        self.model.train()          # the EMA copy stays in eval, always
        m50 = overall["map50"] if overall else 0.0
        m5095 = overall["map5095"] if overall else 0.0
        mic = micro["map50"] if micro and micro["n_gt_total"] else 0.0
        w = self.cfg.fitness_weights
        fitness = w[0] * m50 + w[1] * m5095 + w[2] * mic
        return {"mAP50": m50, "mAP50-95": m5095, "micro_AP50": mic,
                "fitness": fitness,
                "n_val_images": overall["n_images"] if overall else 0,
                "records": recs}

    # ---- checkpoints ----------------------------------------------------- #
    def save_checkpoint(self, path, extra=None):
        """Ultralytics-compatible checkpoint.

        The whole module is pickled, so ch=9 survives the round trip -- the
        `.yaml` re-parse that would silently revert to ch=3 never happens.
        Roadmap T6 verifies this before the long run, not after it."""
        ck = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": _detached_copy(self.model).half(),
            "ema": _detached_copy(self.ema.ema).half() if self.ema else None,
            "updates": self.ema.updates if self.ema else 0,
            "optimizer": self.optimizer.state_dict(),
            "train_args": self.cfg.to_dict(),
            "date": datetime.datetime.now().isoformat(),
        }
        if extra:
            ck.update(extra)
        torch.save(ck, path)

    # ---- the loop -------------------------------------------------------- #
    def train(self, resume_from=None):
        cfg = self.cfg
        start = 0
        if resume_from and Path(resume_from).exists():
            start = self._resume(resume_from)
        if start >= cfg.epochs:
            # `for ep in range(start, cfg.epochs)` over an empty range does
            # nothing, prints nothing, and returns as if it had trained. Say so
            # instead: this is what a resume looks like when the checkpoint has
            # already met the budget, and it is a config question, not a bug.
            print(f"[resume] {resume_from} is already at epoch {start} of "
                  f"{cfg.epochs} — nothing left to train. Raise `epochs`, or "
                  f"start a new run with a different `name`.", flush=True)
            return self.history
        self._t_start = time.time()
        print("=" * 78)
        print(f"AeroTrack-Net training — {cfg.name}")
        print(f"  device      : {self.device} "
              f"({torch.cuda.get_device_name(0) if self.device.type=='cuda' else 'cpu'})")
        print(f"  epochs      : {cfg.epochs} x {len(self.train_loader)} iters "
              f"(batch {cfg.batch}, accumulate {self.accumulate} -> eff. "
              f"{cfg.batch*self.accumulate})")
        print(f"  precision   : {'bf16' if self.amp_dtype is torch.bfloat16 else ('fp16' if self.amp_dtype else 'fp32')}")
        print(f"  params      : {sum(p.numel() for p in self.model.parameters())/1e6:.2f} M")
        print(f"  save dir    : {self.save_dir}")
        print("=" * 78, flush=True)

        stop_counter = 0
        for ep in range(start, cfg.epochs):
            self.epoch = ep
            t_ep = time.time()
            tloss, ips = self.train_one_epoch()
            self.scheduler.step()

            row = {"epoch": ep, "box_loss": float(tloss[0]),
                   "cls_loss": float(tloss[1]), "dfl_loss": float(tloss[2]),
                   "lr": self.optimizer.param_groups[0]["lr"],
                   "img_per_s": round(ips, 2),
                   "epoch_time_s": round(time.time() - t_ep, 1)}

            if (ep + 1) % cfg.val_interval == 0 or ep == cfg.epochs - 1:
                v = self.validate()
                recs = v.pop("records")
                row.update({k: round(float(x), 5) for k, x in v.items()
                            if k != "n_val_images"})
                row["n_val_images"] = v["n_val_images"]
                fit = v["fitness"]
                if fit > self.best_fitness:
                    self.best_fitness, self.best_epoch = fit, ep
                    self.save_checkpoint(self.save_dir / "weights" / "best.pt")
                    stop_counter = 0
                    star = "  <-- best"
                else:
                    stop_counter += 1
                    star = ""
                print(f"[epoch {ep:>3d}] loss(box/cls/dfl) "
                      f"{tloss[0]:.4f}/{tloss[1]:.4f}/{tloss[2]:.4f} | "
                      f"mAP50 {v['mAP50']:.4f}  mAP50-95 {v['mAP50-95']:.4f}  "
                      f"microAP50 {v['micro_AP50']:.4f} | fitness {fit:.4f}"
                      f"{star}  ({row['epoch_time_s']}s)", flush=True)
            else:
                print(f"[epoch {ep:>3d}] loss {tloss.sum():.4f} "
                      f"({row['epoch_time_s']}s)", flush=True)

            self.save_checkpoint(self.save_dir / "weights" / "last.pt")
            if cfg.save_period > 0 and (ep + 1) % cfg.save_period == 0:
                self.save_checkpoint(self.save_dir / "weights" / f"epoch{ep}.pt")
            self.history.append(row)
            self._write_csv()

            if cfg.patience and stop_counter >= cfg.patience:
                print(f"[early-stop] no improvement for {cfg.patience} validations "
                      f"— best was epoch {self.best_epoch} "
                      f"(fitness {self.best_fitness:.4f})")
                break

        dt = (time.time() - self._t_start) / 3600
        print("=" * 78)
        print(f"Training finished in {dt:.2f} h | best fitness {self.best_fitness:.4f} "
              f"@ epoch {self.best_epoch}")
        print(f"  best  : {self.save_dir/'weights'/'best.pt'}")
        print(f"  last  : {self.save_dir/'weights'/'last.pt'}")
        print("=" * 78)
        self._plot_history()
        return self.history

    def _resume(self, path):
        reset_epochs = getattr(self.cfg, "reset_epochs", False)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"].float().state_dict())
        self.model.to(self.device)
        if self.ema and ck.get("ema") is not None:
            self.ema.ema.load_state_dict(ck["ema"].float().state_dict())
            self.ema.updates = ck.get("updates", 0)

        if reset_epochs:
            # Fine-tuning mode: retain model weights, start fresh schedule & epoch count
            self.best_fitness = ck.get("best_fitness", -1.0)
            print(f"[resume] Loaded checkpoint weights from {path} for fine-tuning.")
            print(f"         Epochs reset 0 -> {self.cfg.epochs}; fresh LR schedule ({self.cfg.lr0} -> {self.cfg.lrf}).")
            return 0

        if ck.get("optimizer"):
            self.optimizer.load_state_dict(ck["optimizer"])
        self.best_fitness = ck.get("best_fitness", -1.0)
        start = int(ck.get("epoch", -1)) + 1
        for _ in range(start):
            self.scheduler.step()


        # Re-adopt the epochs already on disk. `_write_csv` rewrites the whole
        # file from `self.history` after every epoch, so a resumed run that
        # started with an empty history would TRUNCATE results.csv to the epochs
        # it happened to run -- silently destroying the record of everything
        # before the interruption, which is also everything the training-curve
        # figure is drawn from. Reading them back keeps one continuous series
        # across any number of resumes.
        if self.csv_path.exists():
            try:
                with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                prior = []
                for r in rows:
                    if not r.get("epoch") or int(float(r["epoch"])) >= start:
                        continue
                    prior.append({k: (None if v in ("", None) else
                                      (float(v) if _isnum(v) else v))
                                  for k, v in r.items()})
                if prior:
                    self.history = prior
                    print(f"[resume] recovered {len(prior)} earlier epochs from "
                          f"{self.csv_path.name}")
            except Exception as e:                               # noqa: BLE001
                print(f"[resume] could not re-read {self.csv_path.name}: {e!r} "
                      f"— the curve will start at epoch {start}")

        print(f"[resume] continuing from epoch {start} "
              f"(best fitness {self.best_fitness:.4f})")
        return start

    def _write_csv(self):
        if not self.history:
            return
        keys = sorted({k for r in self.history for k in r})
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)

    def _plot_history(self):
        if not self.history:
            return
        try:
            # The OO API, deliberately: `matplotlib.use("Agg")` here would switch
            # the backend for the WHOLE process, and this runs inside
            # `train()`. In train.ipynb that silently disables `plt.show()` for
            # every cell AFTER training -- the training-curve figure, the one
            # the run exists to produce, renders nothing and reports no error.
            # A bare Figure needs no backend at all and saves identically.
            from matplotlib.figure import Figure
            h = self.history
            ep = [r["epoch"] for r in h]
            fig = Figure(figsize=(15, 4.2))
            ax = fig.subplots(1, 3)
            for k, c in (("box_loss", "#2a78d6"), ("cls_loss", "#008300"),
                         ("dfl_loss", "#eda100")):
                ax[0].plot(ep, [r[k] for r in h], label=k, color=c, lw=2)
            ax[0].set_title("training loss"); ax[0].legend(); ax[0].set_xlabel("epoch")
            vm = [(r["epoch"], r.get("mAP50"), r.get("mAP50-95"), r.get("micro_AP50"))
                  for r in h if r.get("mAP50") is not None]
            if vm:
                e2 = [v[0] for v in vm]
                ax[1].plot(e2, [v[1] for v in vm], label="mAP@50", color="#2a78d6", lw=2)
                ax[1].plot(e2, [v[2] for v in vm], label="mAP@50-95", color="#a3271a", lw=2)
                ax[1].plot(e2, [v[3] for v in vm], label="micro AP@50", color="#0f7a2e", lw=2.4)
                ax[1].set_title("validation"); ax[1].legend(); ax[1].set_xlabel("epoch")
            ax[2].plot(ep, [r["lr"] for r in h], color="#6b4fa0", lw=2)
            ax[2].set_title("learning rate"); ax[2].set_xlabel("epoch")
            for a in ax:
                a.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(self.save_dir / "training_curves.png", dpi=180)
        except Exception as e:
            print(f"[plot] skipped: {e!r}")


def _isnum(v):
    """Is this CSV field a number? Used to restore history across a resume."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _detached_copy(model):
    """A deep copy, detached from autograd, ready to pickle into a checkpoint.

    The copy comes FIRST. Stripping `requires_grad` off the live module and then
    copying it would freeze the model that is still training -- the run would
    continue, print falling-then-flat losses, and learn nothing after the first
    checkpoint."""
    m = deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


# --------------------------------------------------------------------------- #
#  Loading a trained checkpoint back (SPDConv must be registered first)
# --------------------------------------------------------------------------- #
def load_checkpoint(path, device, prefer_ema=True, fuse=False):
    register_spdconv()
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    m = ck.get("ema") if (prefer_ema and ck.get("ema") is not None) else ck["model"]
    m = m.float().to(device).eval()
    if fuse and hasattr(m, "fuse"):
        m = m.fuse()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck
