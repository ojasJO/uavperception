"""YAML experiment configuration: single source of truth per run.

A config may declare `_base_: <file>` (relative to its own directory); the child is deep-merged
over the base. The *effective* (merged) config is what gets validated, used, and dumped into the
run directory, so a run can always be re-created from `config.effective.yaml` alone.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = {
    "run_name": str,
    "phase": str,
    "seed": int,
    "data": dict,
    "model": dict,
    "dfsc": dict,
    "train": dict,
    "eval": dict,
}

ALLOWED_PHASES = {"A_faithful_reproduction", "B_controlled_improvement", "smoke"}
ALLOWED_MODALITIES = {"IR", "RGB"}
ALLOWED_INIT = {"torchvision_coco_fasterrcnn_r50_fpn_v1", "imagenet_backbone_only", "none", "checkpoint"}
ALLOWED_AMP = {"none", "bf16", "fp16"}


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)
    if base_ref is not None:
        base_cfg = load_config(path.parent / base_ref)
        cfg = _deep_merge(base_cfg, cfg)
    cfg.setdefault("_source_files", [])
    cfg["_source_files"] = [str(path)] + [s for s in cfg["_source_files"] if s != str(path)]
    return cfg


def validate_config(cfg: dict) -> None:
    for key, typ in REQUIRED_KEYS.items():
        if key not in cfg:
            raise ConfigError(f"missing required key '{key}'")
        if not isinstance(cfg[key], typ):
            raise ConfigError(f"key '{key}' must be {typ.__name__}, got {type(cfg[key]).__name__}")
    if cfg["phase"] not in ALLOWED_PHASES:
        raise ConfigError(f"phase must be one of {sorted(ALLOWED_PHASES)}")

    d = cfg["data"]
    for k in ("root", "frames_root", "modality", "train_split", "pairs_per_seq", "max_distance", "frame_filter", "transforms"):
        if k not in d:
            raise ConfigError(f"data.{k} missing")
    if d["modality"] not in ALLOWED_MODALITIES:
        raise ConfigError("data.modality must be IR or RGB")
    if d["train_split"] == "test":
        # This is the structural guarantee behind "training code cannot load TEST".
        raise ConfigError("data.train_split may never be 'test'")
    if d.get("monitor_split", "val") == "test":
        raise ConfigError("data.monitor_split may never be 'test'")
    if d["frame_filter"] not in {"globaltrack", "none"}:
        raise ConfigError("data.frame_filter must be 'globaltrack' or 'none'")

    m = cfg["model"]
    if m.get("init") not in ALLOWED_INIT:
        raise ConfigError(f"model.init must be one of {sorted(ALLOWED_INIT)}")
    if m.get("init") == "checkpoint" and not m.get("init_checkpoint"):
        raise ConfigError("model.init_checkpoint required when model.init == 'checkpoint'")

    f = cfg["dfsc"]
    if not (0.0 <= float(f.get("alpha", 0.0))):
        raise ConfigError("dfsc.alpha must be >= 0")
    if set(f.get("cross_terms", ["cls", "reg"])) - {"cls", "reg"}:
        raise ConfigError("dfsc.cross_terms may only contain 'cls' and 'reg'")
    if f.get("ism_modulation", "same") not in {"same", "cross"}:
        raise ConfigError("dfsc.ism_modulation must be 'same' or 'cross'")

    t = cfg["train"]
    for k in ("epochs", "batch_size", "lr", "lr_steps", "lr_gamma", "momentum", "weight_decay",
              "grad_clip_norm", "warmup_iters", "warmup_ratio", "amp"):
        if k not in t:
            raise ConfigError(f"train.{k} missing")
    if t["amp"] not in ALLOWED_AMP:
        raise ConfigError(f"train.amp must be one of {sorted(ALLOWED_AMP)}")
    if t["batch_size"] < 1:
        raise ConfigError("train.batch_size must be >= 1")
    if float(cfg["dfsc"].get("alpha", 0.0)) > 0 and t["batch_size"] < 2:
        raise ConfigError("DFSC cross-sequence terms need batch_size >= 2 (two different sequences per batch)")
    if any(s > t["epochs"] for s in t["lr_steps"]):
        raise ConfigError("train.lr_steps must be <= train.epochs")


def dump_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def config_summary(cfg: dict) -> str:
    d, m, f, t = cfg["data"], cfg["model"], cfg["dfsc"], cfg["train"]
    return (
        f"run={cfg['run_name']} phase={cfg['phase']} seed={cfg['seed']} | "
        f"data={d['modality']}/{d['train_split']} pairs_per_seq={d['pairs_per_seq']} | "
        f"model={m['name']} init={m['init']} | DFSC alpha={f.get('alpha')} cross_terms={f.get('cross_terms')} "
        f"ism={f.get('ism_modulation','same')} | epochs={t['epochs']} bs={t['batch_size']}x{t.get('accumulate_steps',1)} "
        f"lr={t['lr']} steps={t['lr_steps']} amp={t['amp']}"
    )
