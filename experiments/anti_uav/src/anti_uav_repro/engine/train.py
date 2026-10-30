"""Training loop for QG-RCNN / DFSC on Anti-UAV (Protocol II).

Faithful to GlobalTrack's runner where the paper is silent (spec §4): SGD momentum 0.9, wd 1e-4, grad-clip
35, linear warm-up (500 iters, from lr/3), step decay at the paper's epochs, final-epoch checkpoint.

Everything a run needs to be reproduced is written into the run directory:
    config.effective.yaml   run_meta.json   metrics.jsonl   train_log.txt   weights/epoch_XX.pt, last.pt
The TEST split is unreachable from here: data are loaded with purpose='train'/'monitor'.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import config_summary, dump_config
from ..data.antiuav import build_manifest, load_sequences
from ..data.pairs import DistinctSequenceBatchSampler, PairDataset, collate_pairs
from ..data.transforms import TestTransform, TrainPairTransform, TransformConfig
from ..eval.metrics import aggregate, sequence_metrics
from ..models.qg_rcnn import QGRCNN
from ..utils.repro import file_sha256, run_metadata, set_seed, write_json
from .track import GlobalTrackTracker


class Tee:
    """Mirror stdout to a log file so the live terminal and the persisted log agree."""

    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def _fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def lr_at(base_lr: float, epoch: int, it_global: int, steps: list[int], gamma: float, warmup_iters: int, warmup_ratio: float) -> float:
    """mmdet 1.x StepLrUpdaterHook with linear warm-up (by iteration, epoch-indexed steps, 0-based epochs)."""
    regular = base_lr * (gamma ** sum(1 for s in steps if epoch >= s))
    if warmup_iters > 0 and it_global < warmup_iters:
        k = (1 - it_global / warmup_iters) * (1 - warmup_ratio)
        return regular * (1 - k)
    return regular


def build_monitor_sequences(seqs, num_sequences: int):
    """Deterministic rule: evenly spaced sequences of the (sorted) monitor split. Documented, not cherry-picked."""
    if num_sequences >= len(seqs):
        return list(seqs)
    idx = np.linspace(0, len(seqs) - 1, num_sequences).round().astype(int)
    return [seqs[i] for i in sorted(set(idx.tolist()))]


class Trainer:
    def __init__(self, cfg: dict, run_dir: Path, resume: str | None = None, max_iters_per_epoch: int | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.weights_dir = self.run_dir / "weights"     # created lazily by fit() so --dry-run leaves no trace
        self.max_iters_per_epoch = max_iters_per_epoch   # smoke tests only
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(int(cfg["seed"]), cudnn_benchmark=bool(cfg["train"].get("cudnn_benchmark", True)))

        d, t = cfg["data"], cfg["train"]
        # ---------------------------------------------------------------- data (TRAIN split only)
        self.train_seqs = load_sequences(d["root"], d["frames_root"], d["train_split"], d["modality"], purpose="train")
        tf_cfg = TransformConfig(scale=tuple(d["transforms"]["scale"]), photometric=bool(d["transforms"]["photometric"]),
                                 channel_swap=bool(d["transforms"]["channel_swap"]), flip_p=float(d["transforms"]["flip_p"]),
                                 pad_divisor=int(d["transforms"]["pad_divisor"]))
        self.dataset = PairDataset(self.train_seqs, d["pairs_per_seq"], d["max_distance"], d["frame_filter"],
                                   TrainPairTransform(tf_cfg), seed=int(cfg["seed"]))
        self.sampler = DistinctSequenceBatchSampler(len(self.train_seqs), d["pairs_per_seq"], t["batch_size"], seed=int(cfg["seed"]))
        self.loader = DataLoader(self.dataset, batch_sampler=self.sampler, collate_fn=collate_pairs,
                                 num_workers=int(t.get("workers", 4)), pin_memory=self.device.type == "cuda",
                                 persistent_workers=int(t.get("workers", 4)) > 0)
        # ---------------------------------------------------------------- monitor (VAL split, subset, strided)
        mon = t.get("monitor", {}) or {}
        self.monitor_enabled = bool(mon.get("enabled", False))
        self.monitor_seqs = []
        if self.monitor_enabled:
            msplit = d.get("monitor_split", "val")
            all_val = load_sequences(d["root"], d["frames_root"], msplit, d["modality"], purpose="monitor")
            self.monitor_seqs = build_monitor_sequences(all_val, int(mon.get("num_sequences", 8)))
            self.monitor_stride = int(mon.get("frame_stride", 10))
            self.monitor_every = int(mon.get("every_epochs", 3))
            self.monitor_max_frames = mon.get("max_frames_per_sequence", None)
        # ---------------------------------------------------------------- model / optimiser
        self.model = QGRCNN(cfg["model"]).to(self.device)
        # pin the architecture detail that depends on the weight source so evaluation rebuilds it identically
        cfg["model"]["frozen_bn_eps"] = float(self.model.frozen_bn_eps)
        if cfg["model"].get("init") == "checkpoint":
            sd = torch.load(cfg["model"]["init_checkpoint"], map_location="cpu", weights_only=False)
            self.model.load_state_dict(sd["model"] if "model" in sd else sd)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=float(t["lr"]), momentum=float(t["momentum"]), weight_decay=float(t["weight_decay"]))
        self.amp = t.get("amp", "none")
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp == "fp16" and self.device.type == "cuda"))
        self.accum = int(t.get("accumulate_steps", 1))
        self.epochs = int(t["epochs"])
        self.start_epoch = 0
        self.it_global = 0  # optimizer steps taken (for warm-up)
        if resume:
            self._load_resume(resume)

    # -------------------------------------------------------------------------------------- utils
    def _autocast(self):
        if self.amp in ("bf16", "fp16") and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16 if self.amp == "bf16" else torch.float16)
        return torch.autocast("cpu", enabled=False)

    def _log(self, msg: str):
        print(msg, flush=True)

    def _save_checkpoint(self, epoch: int, tag: str) -> Path:
        path = self.weights_dir / f"{tag}.pt"
        torch.save({
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "epoch": epoch + 1, "it_global": self.it_global, "config": self.cfg, "run_name": self.cfg["run_name"],
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, path)
        return path

    def _load_resume(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.start_epoch = int(ck["epoch"])
        self.it_global = int(ck.get("it_global", 0))
        self._log(f"[resume] loaded {path}: starting at epoch {self.start_epoch + 1}/{self.epochs}")

    # ------------------------------------------------------------------------------------ monitor
    @torch.no_grad()
    def run_monitor(self, epoch: int) -> dict | None:
        if not self.monitor_enabled or not self.monitor_seqs:
            return None
        self.model.eval()
        tracker = GlobalTrackTracker(self.model, self.device, TestTransform(tuple(self.cfg["data"]["transforms"]["scale"]),
                                                                              int(self.cfg["data"]["transforms"]["pad_divisor"])), amp=self.amp)
        per_seq, attrs = {}, {}
        t0 = time.time()
        for s in self.monitor_seqs:
            out = tracker.track_sequence(s, frame_stride=self.monitor_stride, max_frames=self.monitor_max_frames)
            per_seq[s.name] = sequence_metrics(out["res"], s.exist, s.gt_rect, mask=np.asarray(out["processed"], dtype=bool))
            attrs[s.name] = s.attributes
        agg = aggregate(per_seq, attrs)
        self.model.train()
        res = {"monitor_mSA_percent": agg["mSA_percent"], "monitor_precision_20px": agg["precision_20px"],
               "monitor_success_auc": agg["success_auc"], "monitor_n_sequences": len(self.monitor_seqs),
               "monitor_frame_stride": self.monitor_stride, "monitor_seconds": round(time.time() - t0, 1),
               "note": "MONITOR ONLY: val subset, frame-strided, processed frames scored; never used for checkpoint selection"}
        self._log(f"  [monitor] val-subset mSA {agg['mSA_percent']:.2f}% | prec@20 {100*agg['precision_20px']:.1f}% | "
                  f"success {100*agg['success_auc']:.1f}% ({len(self.monitor_seqs)} seqs, stride {self.monitor_stride}, {res['monitor_seconds']}s) "
                  f"— monitor only, not used for selection")
        return res

    # -------------------------------------------------------------------------------------- train
    def fit(self) -> dict:
        cfg, t = self.cfg, self.cfg["train"]
        f = cfg["dfsc"]
        alpha, cross_terms, ism = float(f.get("alpha", 0.0)), tuple(f.get("cross_terms", ["cls", "reg"])), f.get("ism_modulation", "same")
        base_lr, steps, gamma = float(t["lr"]), [int(s) for s in t["lr_steps"]], float(t["lr_gamma"])
        warm_it, warm_ratio = int(t["warmup_iters"]), float(t["warmup_ratio"])
        clip = float(t["grad_clip_norm"])
        log_every = int(t.get("log_interval", 20))

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        # ---- run metadata (written before training starts, completed at the end)
        train_manifest = build_manifest(self.train_seqs, check_frames=False)
        meta = run_metadata(cfg, extra={
            "run_dir": str(self.run_dir), "device": str(self.device),
            "train_split": {"split": cfg["data"]["train_split"], "modality": cfg["data"]["modality"],
                            "n_sequences": train_manifest["n_sequences"], "n_frames": train_manifest["n_frames"],
                            "manifest_sha256": train_manifest["manifest_sha256"]},
            "monitor": {"enabled": self.monitor_enabled, "sequences": [s.name for s in self.monitor_seqs]},
            "pairs_per_epoch": len(self.dataset), "iterations_per_epoch": len(self.sampler),
            "trainable_frames_per_sequence": {s.name: int(len(ix)) for s, ix in zip(self.train_seqs, self.dataset.frame_indices)},
            "model": self.model.parameter_summary(), "amp": self.amp, "accumulate_steps": self.accum,
            "test_split_touched": False,
        })
        write_json(meta, self.run_dir / "run_meta.json")
        write_json(train_manifest, self.run_dir / "train_manifest.json")
        dump_config(cfg, self.run_dir / "config.effective.yaml")
        self._log("=" * 100)
        self._log(config_summary(cfg))
        self._log(f"run_dir={self.run_dir}")
        self._log(f"train: {len(self.train_seqs)} sequences, {len(self.dataset)} pairs/epoch, {len(self.sampler)} iterations/epoch "
                  f"(batch {t['batch_size']} x accumulate {self.accum}); trainable params "
                  f"{meta['model']['trainable']/1e6:.2f}M / {meta['model']['total']/1e6:.2f}M; device {self.device}; amp {self.amp}")
        n_unusable = sum(1 for ix in self.dataset.frame_indices if len(ix) == 0)
        if n_unusable:
            self._log(f"WARNING: {n_unusable} training sequences have no frame passing the frame filter (resampled to others)")
        self._log("=" * 100)

        metrics_path = self.run_dir / "metrics.jsonl"
        total_iters = len(self.sampler) * self.epochs
        done_iters = len(self.sampler) * self.start_epoch
        t_start = time.time()
        self.model.train()
        last_ckpt = None
        for epoch in range(self.start_epoch, self.epochs):
            self.dataset.set_epoch(epoch)
            self.sampler.set_epoch(epoch)
            sums = {}
            n_it = 0
            t_ep = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            for it, batch in enumerate(self.loader):
                if self.max_iters_per_epoch is not None and it >= self.max_iters_per_epoch:
                    break
                lr = lr_at(base_lr, epoch, self.it_global, steps, gamma, warm_it, warm_ratio)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr
                with self._autocast():
                    losses = self.model.forward_train(batch, alpha=alpha, cross_terms=cross_terms, ism_modulation=ism)
                    loss = losses["total"] / self.accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch+1} iter {it}: { {k: float(v) for k, v in losses.items()} }")
                self.scaler.scale(loss).backward()
                if (it + 1) % self.accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.it_global += 1
                else:
                    gnorm = torch.tensor(float("nan"))
                for k, v in losses.items():
                    if k.startswith("_"):
                        continue
                    sums[k] = sums.get(k, 0.0) + float(v)
                n_it += 1
                done_iters += 1
                if (it + 1) % log_every == 0 or it == 0:
                    el = time.time() - t_start
                    rate = (done_iters - len(self.sampler) * self.start_epoch) / max(el, 1e-6)
                    eta_total = (total_iters - done_iters) / max(rate, 1e-9)
                    eta_epoch = (len(self.sampler) - (it + 1)) / max(rate, 1e-9)
                    mem = torch.cuda.max_memory_allocated() / 1e9 if self.device.type == "cuda" else 0.0
                    self._log(f"[ep {epoch+1:02d}/{self.epochs} it {it+1:04d}/{len(self.sampler)}] "
                              f"loss {sums['total']/n_it:.4f} | rpn same {sums['rpn_cls_same']/n_it:.3f}/{sums['rpn_reg_same']/n_it:.3f} "
                              f"cross {sums['rpn_cls_cross']/n_it:.3f}/{sums['rpn_reg_cross']/n_it:.3f} | "
                              f"rcnn {sums['rcnn_cls']/n_it:.3f}/{sums['rcnn_reg']/n_it:.3f} | lr {lr:.2e} | gnorm {float(gnorm):.1f} | "
                              f"{rate:.2f} it/s | mem {mem:.1f}G | ETA epoch {_fmt_time(eta_epoch)} total {_fmt_time(eta_total)}")
            # ---- end of epoch
            ep_time = time.time() - t_ep
            avg = {k: v / max(n_it, 1) for k, v in sums.items()}
            last_ckpt = self._save_checkpoint(epoch, f"epoch_{epoch+1:02d}")
            shutil.copyfile(last_ckpt, self.weights_dir / "last.pt")
            rec = {"epoch": epoch + 1, "iterations": n_it, "optimizer_steps_total": self.it_global, "lr_end": lr,
                   "epoch_seconds": round(ep_time, 1), "losses": avg, "checkpoint": str(last_ckpt),
                   "checkpoint_sha256": file_sha256(last_ckpt)}
            self._log(f"== epoch {epoch+1}/{self.epochs} done in {_fmt_time(ep_time)} | mean loss {avg.get('total', float('nan')):.4f} | "
                      f"saved {last_ckpt.name} (sha256 {rec['checkpoint_sha256'][:12]})")
            if self.monitor_enabled and ((epoch + 1) % self.monitor_every == 0 or epoch + 1 == self.epochs):
                mon = self.run_monitor(epoch)
                if mon:
                    rec["monitor"] = mon
            with open(metrics_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        # ---- finish
        meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        meta["train_seconds"] = round(time.time() - t_start, 1)
        meta["final_checkpoint"] = str(last_ckpt) if last_ckpt else None
        meta["final_checkpoint_sha256"] = file_sha256(last_ckpt) if last_ckpt else None
        meta["protocol_note"] = "final-epoch checkpoint is THE model (no checkpoint selection), as in GlobalTrack/the paper"
        write_json(meta, self.run_dir / "run_meta.json")
        self._log(f"training finished in {_fmt_time(meta['train_seconds'])}; final checkpoint {meta['final_checkpoint']}")
        return meta
