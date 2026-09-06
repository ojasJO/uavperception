#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AeroTrack-Net — Unified Master CLI Runner
=========================================
Target: HP OMEN 16 · NVIDIA RTX 5050 (8 GB, Blackwell sm_120) · 24 GB RAM

Subcommands:
  train      Train or resume AeroTrack-Net with hardware-safe defaults
  evaluate   Run complete size-stratified evaluation on val or test splits
  track      Benchmark ByteTrack vs SORT on MOTA, IDF1 and occlusion re-entry
  forecast   Train and evaluate Social-LSTM vs Kalman filter on sudden jumps
  infer      Run live streaming video inference on any raw .mp4
  status     Inspect active background jobs and print latest training metrics
  stop       Safely stop a detached background training run
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

DEFAULT_BEST = ROOT / "runs" / "aerotrack_spd_v1" / "weights" / "best.pt"
DEFAULT_LAST = ROOT / "runs" / "aerotrack_spd_v1" / "weights" / "last.pt"
RESULTS_CSV = ROOT / "runs" / "aerotrack_spd_v1" / "results.csv"

FT_BEST = ROOT / "runs" / "aerotrack_spd_finetuned" / "weights" / "best.pt"


def relative_to_root(p):
    """Safely return path relative to ROOT without ValueError."""
    try:
        return Path(p).resolve().relative_to(ROOT.resolve())
    except (ValueError, Exception):
        return Path(p).name


def resolve_weights(weights_arg=None, default_to_baseline=False):
    """Resolve weights path to absolute, defaulting to finetuned best if available."""
    if weights_arg:
        return Path(weights_arg).resolve()
    if not default_to_baseline and FT_BEST.exists():
        return FT_BEST.resolve()
    return DEFAULT_BEST.resolve()


def check_python_environment():
    """Ensure we are running within the project's Blackwell-enabled venv."""
    venv_py = ROOT / "venv" / "Scripts" / "python.exe"
    if venv_py.exists() and Path(PYTHON).resolve() != venv_py.resolve():
        args = [str(venv_py), str(__file__)] + sys.argv[1:]
        sys.exit(subprocess.call(args))


def cmd_status(args):
    """Print background run status and latest results."""
    print("=" * 78)
    print("AeroTrack-Net — Status & Metric Summary")
    print("=" * 78)

    train_py = ROOT / "c_model" / "train.py"
    subprocess.call([PYTHON, str(train_py), "--status"])

    if RESULTS_CSV.exists():
        try:
            import pandas as pd
            df = pd.read_csv(RESULTS_CSV)
            total = len(df)
            cols = ["epoch", "box_loss", "cls_loss", "mAP50", "mAP50-95", "micro_AP50", "fitness"]
            avail_cols = [c for c in cols if c in df.columns]
            print("\nLatest Training Epochs (from results.csv):")
            print("-" * 78)
            print(df[avail_cols].tail(min(total, 10)).to_string(index=False))
            print("-" * 78)
            best_idx = df["fitness"].idxmax() if "fitness" in df else -1
            if best_idx >= 0:
                print(f"Peak Model: Epoch {int(df.at[best_idx, 'epoch'])} with Fitness = {df.at[best_idx, 'fitness']:.4f} "
                      f"| mAP@50 = {df.at[best_idx, 'mAP50']:.4f} | micro_AP50 = {df.at[best_idx, 'micro_AP50']:.4f}")
        except Exception as e:
            print(f"Could not parse results.csv: {e}")
    else:
        print("\nNo results.csv found yet under runs/aerotrack_spd_v1/")
    print("=" * 78)


def cmd_train(args):
    """Run model training with optimized max-safe hardware defaults."""
    train_py = ROOT / "c_model" / "train.py"
    cmd = [PYTHON, str(train_py)]

    cmd.extend(["--stages", args.stages])
    cmd.extend(["--batch", str(args.batch)])
    cmd.extend(["--workers", str(args.workers)])
    cmd.extend(["--imgsz", str(args.imgsz)])
    cmd.extend(["--epochs", str(args.epochs)])
    cmd.extend(["--patience", str(args.patience)])
    cmd.extend(["--reserve-gb", str(args.reserve_gb)])
    cmd.extend(["--priority", args.priority])

    if args.resume:
        cmd.extend(["--resume", args.resume])
    elif not args.no_resume and DEFAULT_LAST.exists():
        print(f"[run] Found existing checkpoint: {DEFAULT_LAST.relative_to(ROOT)}")
        print("[run] Automatically resuming from last completed epoch...")
        cmd.extend(["--resume", str(DEFAULT_LAST)])

    if args.background:
        cmd.append("--background")
        print("[run] Launching training in DETACHED BACKGROUND mode...")
    else:
        print("[run] Launching training in LIVE FOREGROUND mode...")
        print("[run] Press Ctrl+C anytime to pause training safely.")

    sys.exit(subprocess.call(cmd))


def cmd_finetune(args):
    """Run fine-tuning for cross-domain generalization & bird rejection."""
    train_py = ROOT / "c_model" / "train.py"
    weights = resolve_weights(args.weights, default_to_baseline=True)
    if not weights.exists():
        print(f"Error: Baseline checkpoint weights not found at {weights}")
        sys.exit(1)

    cmd = [PYTHON, str(train_py), "--stages", "finetune", "--resume", str(weights)]
    cmd.extend(["--batch", str(args.batch)])
    cmd.extend(["--workers", str(args.workers)])
    cmd.extend(["--reserve-gb", str(args.reserve_gb)])
    cmd.extend(["--priority", args.priority])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])

    if args.background:
        cmd.append("--background")
        print("[run] Launching fine-tuning in DETACHED BACKGROUND mode...")
    else:
        print("[run] Launching fine-tuning for Cross-Domain Generalization & Bird Rejection...")
        print(f"[run] Initializing from: {relative_to_root(weights)}")
        print("[run] Sampler : Domain-Balanced (40% Anti-UAV, 35% CST, 25% Det-Fly; ~8% birds)")
        print("[run] Loss    : Asymmetric Class-Weighted (1.0 drone, 4.5 bird; cls gain 1.5)")
        print("[run] Augment : Scale (0.8-1.3), Translate 0.20, Grayscale 25%, Noise std 0.02")
        print("[run] Press Ctrl+C anytime to pause safely.")

    sys.exit(subprocess.call(cmd))



def cmd_evaluate(args):
    """Run full evaluation suite on val or test splits."""
    val_py = ROOT / "c_model" / "val.py"
    weights = resolve_weights(args.weights)
    if not weights.exists():
        print(f"Error: Checkpoint weights not found at {weights}")
        sys.exit(1)

    cmd = [PYTHON, str(val_py), "--weights", str(weights), "--split", args.split]
    if args.split == "test":
        cmd.append("--confirm-test")
    if args.cap is not None:
        cmd.extend(["--cap", str(args.cap)])
    if args.batch is not None:
        cmd.extend(["--batch", str(args.batch)])
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])

    print(f"[run] Evaluating {weights.name} on split='{args.split}'...")
    sys.exit(subprocess.call(cmd))


def cmd_track(args):
    """Run detector + ByteTrack evaluation against SORT control."""
    track_py = ROOT / "c_model" / "track_eval.py"
    weights = resolve_weights(args.weights)
    if not weights.exists():
        print(f"Error: Checkpoint weights not found at {weights}")
        sys.exit(1)

    cmd = [PYTHON, str(track_py), "--weights", str(weights), "--split", args.split]
    if args.split == "test":
        cmd.append("--confirm-test")
    if args.sequences is not None:
        cmd.extend(["--sequences", str(args.sequences)])
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])

    print(f"[run] Running ByteTrack vs SORT evaluation on split='{args.split}'...")
    sys.exit(subprocess.call(cmd))


def cmd_forecast(args):
    """Run Social-LSTM trajectory forecasting training & evaluation."""
    forecaster_py = ROOT / "c_model" / "lstm_forecaster.py"
    cmd = [PYTHON, str(forecaster_py)]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.obs is not None:
        cmd.extend(["--obs", str(args.obs)])
    if args.pred is not None:
        cmd.extend(["--pred", str(args.pred)])
    if args.device is not None:
        cmd.extend(["--device", args.device])

    print("[run] Training & Evaluating Social-LSTM trajectory forecaster...")
    sys.exit(subprocess.call(cmd))


def cmd_infer(args):
    """Run video inference on a raw clip with 9-ch temporal stacking."""
    infer_py = ROOT / "c_model" / "inference.py"
    weights = resolve_weights(args.weights)
    if not weights.exists():
        print(f"Error: Checkpoint weights not found at {weights}")
        sys.exit(1)

    source = Path(args.source)
    if not source.exists():
        print(f"Error: Source video file not found at {source}")
        sys.exit(1)

    out = args.out or str(source.parent / f"{source.stem}_annotated.mp4")
    cmd = [PYTHON, str(infer_py), "--source", str(source), "--weights", str(weights),
           "--out", str(out), "--conf", str(args.conf), "--iou", str(args.iou)]

    print(f"[run] Running inference on {source.name} -> {out}...")
    sys.exit(subprocess.call(cmd))


def cmd_demo(args):
    """Run interactive visual window with live telemetry in terminal."""
    demo_py = ROOT / "c_model" / "visual_demo.py"
    weights = resolve_weights(args.weights)
    if not weights.exists():
        print(f"Error: Checkpoint weights not found at {weights}")
        sys.exit(1)

    cmd = [PYTHON, str(demo_py), "--weights", str(weights)]
    if args.source:
        cmd.extend(["--source", str(args.source)])
    if args.conf is not None:
        cmd.extend(["--conf", str(args.conf)])
    if args.iou is not None:
        cmd.extend(["--iou", str(args.iou)])
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if getattr(args, "rgb", False):
        cmd.append("--rgb")
    if getattr(args, "ir", False):
        cmd.append("--ir")
    if getattr(args, "modality", None):
        cmd.extend(["--modality", args.modality])
    if getattr(args, "list", False):
        cmd.append("--list")

    if not getattr(args, "list", False):
        print("[run] Launching interactive visual tracking window...")
        print("[run] Streaming live telemetry to terminal. Press [Space] to pause, [Q] to quit.")
    sys.exit(subprocess.call(cmd))


def cmd_stop(args):
    """Stop any active background training run."""
    train_py = ROOT / "c_model" / "train.py"
    cmd = [PYTHON, str(train_py), "--stop"]
    sys.exit(subprocess.call(cmd))


def build_parser():
    p = argparse.ArgumentParser(
        description="AeroTrack-Net Unified Master CLI Runner (RTX 5050 / OMEN 16)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="subcommand", help="Action to perform")

    sub.add_parser("status", help="Check active background jobs and view latest epoch metrics")
    sub.add_parser("stop", help="Safely stop a detached background training run")

    p_train = sub.add_parser("train", help="Train or resume the AeroTrack-Net detector")
    p_train.add_argument("--stages", default="baseline",
                         help="Staircase stages to run, e.g. 'baseline' or 'overfit,smoke,baseline' (default: baseline)")
    p_train.add_argument("--batch", type=int, default=32,
                         help="Batch size (default: 32, max safe for 8 GB VRAM)")
    p_train.add_argument("--workers", type=int, default=4,
                         help="DataLoader workers (default: 4, capped dynamically for >=6GB RAM buffer)")
    p_train.add_argument("--imgsz", type=int, default=640,
                         help="Input image resolution (default: 640)")
    p_train.add_argument("--epochs", type=int, default=150,
                         help="Total epochs (default: 150)")
    p_train.add_argument("--patience", type=int, default=30,
                         help="Early stopping patience (default: 30)")
    p_train.add_argument("--reserve-gb", type=float, default=6.0,
                         help="Host RAM buffer to keep free in GB (default: 6.0)")
    p_train.add_argument("--priority", default="normal", choices=["idle", "below_normal", "normal"],
                         help="Process CPU priority (default: normal)")
    p_train.add_argument("--resume", default=None,
                         help="Specific checkpoint path to resume from")
    p_train.add_argument("--no-resume", action="store_true",
                         help="Start from scratch even if a last.pt checkpoint exists")
    p_train.add_argument("--background", action="store_true",
                         help="Detach and run training in the background")
    p_train.set_defaults(func=cmd_train)

    p_fine = sub.add_parser("finetune", help="Fine-tune model for cross-domain generalization and bird rejection")
    p_fine.add_argument("--weights", default=None,
                        help="Path to starting checkpoint (default: runs/aerotrack_spd_v1/weights/best.pt)")
    p_fine.add_argument("--epochs", type=int, default=25,
                        help="Fine-tuning epochs (default: 25)")
    p_fine.add_argument("--batch", type=int, default=32,
                        help="Batch size (default: 32, max safe for 8 GB VRAM)")
    p_fine.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4, capped dynamically for >=6GB RAM buffer)")
    p_fine.add_argument("--reserve-gb", type=float, default=6.0,
                        help="Host RAM buffer to keep free in GB (default: 6.0)")
    p_fine.add_argument("--priority", default="normal", choices=["idle", "below_normal", "normal"],
                        help="Process CPU priority (default: normal)")
    p_fine.add_argument("--background", action="store_true",
                        help="Detach and run fine-tuning in the background")
    p_fine.set_defaults(func=cmd_finetune)

    p_eval = sub.add_parser("evaluate", aliases=["eval"], help="Run size-stratified evaluation on val or test split")

    p_eval.add_argument("--weights", default=None,
                        help="Path to checkpoint weights (default: runs/aerotrack_spd_v1/weights/best.pt)")
    p_eval.add_argument("--split", choices=["val", "test"], default="val",
                        help="Split to evaluate: 'val' or 'test' (default: val)")
    p_eval.add_argument("--cap", type=int, default=None,
                        help="Max images per dataset to evaluate (default: 0 = all)")
    p_eval.add_argument("--batch", type=int, default=32,
                        help="Batch size for evaluation (default: 32)")
    p_eval.add_argument("--workers", type=int, default=2,
                        help="Evaluation loader workers (default: 2, safe for Windows memory)")
    p_eval.set_defaults(func=cmd_evaluate)

    p_track = sub.add_parser("track", help="Run ByteTrack vs SORT evaluation on MOTA, IDF1 & occlusion switches")
    p_track.add_argument("--weights", default=None,
                         help="Path to checkpoint weights (default: best.pt)")
    p_track.add_argument("--split", choices=["val", "test"], default="val",
                         help="Split to evaluate (default: val)")
    p_track.add_argument("--sequences", type=int, default=10,
                         help="Number of sequences per dataset (default: 10, 0 = all)")
    p_track.add_argument("--max-frames", type=int, default=0,
                         help="Cap frames per sequence (default: 0 = whole sequence)")
    p_track.set_defaults(func=cmd_track)

    p_fore = sub.add_parser("forecast", help="Train & evaluate Social-LSTM trajectory forecaster")
    p_fore.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    p_fore.add_argument("--obs", type=int, default=16, help="Observation horizon in frames (default: 16)")
    p_fore.add_argument("--pred", type=int, default=12, help="Prediction horizon in frames (default: 12)")
    p_fore.add_argument("--device", default=None, help="Compute device: 'cuda:0' or 'cpu'")
    p_fore.set_defaults(func=cmd_forecast)

    p_infer = sub.add_parser("infer", help="Run streaming video inference on a clip with FPS overlay")
    p_infer.add_argument("--source", required=True, help="Path to input .mp4 video")
    p_infer.add_argument("--weights", default=None, help="Path to checkpoint weights (default: best.pt)")
    p_infer.add_argument("--out", default=None, help="Path to annotated output .mp4")
    p_infer.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    p_infer.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    p_infer.set_defaults(func=cmd_infer)

    p_demo = sub.add_parser("demo", help="Open real-time interactive visual tracking window with live HUD and terminal telemetry")
    p_demo.add_argument("--source", default=None, help="Sequence key name, keyword, or path")
    p_demo.add_argument("--weights", default=None, help="Path to checkpoint weights (default: best.pt)")
    p_demo.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    p_demo.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    p_demo.add_argument("--max-frames", type=int, default=0, help="Max frames to play (default: 0 = all)")
    p_demo.add_argument("--modality", choices=["rgb", "ir", "all"], default=None,
                        help="Filter playback by modality: 'rgb', 'ir', or 'all'")
    p_demo.add_argument("--rgb", action="store_true", help="Play only visible RGB daylight drone videos")
    p_demo.add_argument("--ir", action="store_true", help="Play only thermal IR drone videos")
    p_demo.add_argument("--list", action="store_true", help="List all available curated sequences and exit")
    p_demo.set_defaults(func=cmd_demo)

    return p


def main():
    check_python_environment()
    parser = build_parser()
    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    elif args.subcommand == "status":
        cmd_status(args)
    elif args.subcommand == "stop":
        cmd_stop(args)


if __name__ == "__main__":
    main()
