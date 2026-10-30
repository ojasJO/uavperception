#!/usr/bin/env python
"""Same-set detector evaluation for every AeroTrack-Net checkpoint (repair items 1, 9, 11, 12).

    python experiments/aerotrack_repair/scripts/evaluate_detector.py --split val \
        --weights results/runs/aerotrack_spd_v1/weights/best.pt results/runs/aerotrack_spd_v1/weights/last.pt \
                  results/runs/aerotrack_spd_finetuned/weights/best.pt results/runs/aerotrack_spd_finetuned/weights/last.pt

What it fixes
-------------
* ONE evaluation set for every checkpoint: the split's samples, deterministically sub-sampled per dataset
  (the project's own `subsample`, evenly spread over sequences) and identified by a SHA-256 over
  (dataset, sequence, frame, path). The hash is stored next to every result; a table only ever compares
  results carrying the same hash. `--cap 0` evaluates the full split.
* The project's own metric code (`src/model/evaluation.py`: size-stratified AP, per-dataset, per-modality,
  empty-frame FP) PLUS the corrected bird->drone confusion (drone-class predictions on bird-only images,
  GT birds boxed as drone at IoU>=.5) that `evaluation.fp_rate` did not measure.
* One size definition, stated in pixels per dataset: micro = box area < 0.03 % of the SOURCE frame,
  small = 0.03-1 %, standard > 1 % (the code's definition); the pixel equivalents are computed from the
  native resolutions of the evaluated images, not typed.
* `best.pt` and `last.pt` are evaluated side by side so checkpoint selection on the noisy fitness metric
  is visible.
* TEST needs --final-eval and is appended to reports/test_evaluation_ledger.jsonl.
* Input construction follows each model's own definition (grid runner repair): runs whose run_meta.json records
  temporal_gap 0 (the single_frame / stock_yolo11n ablations, trained AND validated on t-1 = t = t+1, see
  src/model/train.py::build) are scored on that input; every other checkpoint on the t-1/t/t+1 stack. The IMAGE set
  and its hash are the same for both. `--input-mode` overrides the automatic choice.
* The summary JSON/MD is updated (atomically) after EACH checkpoint and only then is the TEST ledger line written,
  so a ledger line always means "scored and recorded"; an interrupted invocation loses at most the checkpoint in
  flight.

Outputs: results/runs/<run>/eval_repair_<split>_<settag>/<ckpt>/evaluation_report.json (+ PR / size plots) and
experiments/aerotrack_repair/reports/detector_<split>_<settag>.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import evaluation as E
from aerotrack_trainer import load_checkpoint
from stacked_dataset import AeroTrackDataset, aerotrack_collate, build_split_samples, load_label_cache
from train import LABEL_CACHE_PATH, MANIFEST, SPLITS, subsample

from anti_uav_repro.utils.repro import env_snapshot, file_sha256, git_info, json_sha256

from _bootstrap import REPO_ROOT, REPORTS_DIR

LEDGER = REPORTS_DIR / "test_evaluation_ledger.jsonl"
INPUT_LABEL = {"temporal": "t-1,t,t+1", "single_frame": "t,t,t"}


def input_mode_for(weights: Path, requested: str = "auto"):
    """-> (mode, source). The run's own provenance decides; historical runs predate run_meta.json and were trained
    on the 3-frame stack (docs/historical/early_training_launch_records/train_tuning.json: temporal_gap 1)."""
    if requested != "auto":
        return requested, f"--input-mode {requested}"
    meta = weights.parent.parent / "run_meta.json"
    if meta.exists():
        g = json.loads(meta.read_text(encoding="utf-8")).get("temporal_gap", 1)
        return ("single_frame" if int(g) == 0 else "temporal"), f"results/runs/{weights.parent.parent.name}/run_meta.json: temporal_gap={g}"
    return "temporal", "no run_meta.json: historical run, trained on the t-1/t/t+1 stack"


def single_frame_samples(samples):
    """t-1 = t = t+1, exactly as src/model/train.py::build() rewrites the train and val samples of a temporal_gap-0 run."""
    return [dict(s, prev_path=s["cur_path"], next_path=s["cur_path"]) for s in samples]


def _write_text_atomic(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def merge_summary(summary_path: Path, setinfo, res_px, new_rows):
    """Replace rows for the same run/checkpoint, keep the others (all rows share the set hash by construction)."""
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"rows": []}
    keep = [r for r in existing.get("rows", []) if r.get("set_sha256") == setinfo["sha256"][:12]
            and (r["run"], r["checkpoint"]) not in {(x["run"], x["checkpoint"]) for x in new_rows}]
    out = {"eval_set": setinfo, "size_definition_px": res_px, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": keep + new_rows}
    _write_text_atomic(summary_path, json.dumps(out, indent=2, default=_np_default))
    _write_text_atomic(summary_path.with_suffix(".md"), render_md(out))
    return out


def eval_set(split: str, cap: int, datasets):
    samples = build_split_samples(MANIFEST, SPLITS, split, datasets=datasets)
    if cap:
        samples = subsample(samples, {d: cap for d in {s["dataset"] for s in samples}})
    ident = sorted((s["dataset"], s["seq_key"], int(s["frame_idx"]), str(Path(s["cur_path"]).as_posix())) for s in samples)
    return samples, {"split": split, "cap": cap, "datasets": sorted({s["dataset"] for s in samples}),
                     "n_images": len(samples), "per_dataset": dict(Counter(s["dataset"] for s in samples)),
                     "n_sequences": len({s["seq_key"] for s in samples}), "sha256": json_sha256(ident),
                     "manifest_csv_sha256": file_sha256(MANIFEST), "splits_json_sha256": file_sha256(SPLITS)}


def native_resolutions(samples):
    """(dataset, modality) -> (W, H) read from one image each; used to state the size bins in pixels."""
    seen = {}
    for s in samples:
        key = (s["dataset"], s.get("modality", "?"))
        if key in seen:
            continue
        im = cv2.imread(str(s["cur_path"]), cv2.IMREAD_UNCHANGED)
        if im is not None:
            seen[key] = (int(im.shape[1]), int(im.shape[0]))
    out = {}
    for (d, m), (w, h) in seen.items():
        out[f"{d}/{m}"] = {"width": w, "height": h,
                           "micro_max_side_px": round(float(np.sqrt(E.MICRO_THR * w * h)), 1),
                           "small_max_side_px": round(float(np.sqrt(E.SMALL_THR * w * h)), 1)}
    return out


def bird_confusion(recs, thresholds=(0.25, 0.5)):
    """Corrected bird<->drone confusion on Det-Fly records (from scripts/audit/bench_bird_confusion.py)."""
    recs = [r for r in recs if r["dataset"] == "det_fly"]
    out = {"det_fly_images": len(recs)}
    for thr in thresholds:
        bird_only = [r for r in recs if len(r["gt"]) and not (r["gt"][:, 0] == 0).any()]
        drone_only = [r for r in recs if len(r["gt"]) and not (r["gt"][:, 0] == 1).any()]

        def cls_preds(r, c):
            p = r["pred"]
            return p[(p[:, 5] == c) & (p[:, 4] >= thr)] if len(p) else p

        d_on_bird = [len(cls_preds(r, 0)) for r in bird_only]
        b_on_drone = [len(cls_preds(r, 1)) for r in drone_only]

        def confused(rs, gt_cls, wrong_cls):
            n_gt = n_conf = 0
            for r in rs:
                g = r["gt"][r["gt"][:, 0] == gt_cls]
                p = r["pred"]
                p = p[p[:, 4] >= thr] if len(p) else p
                n_gt += len(g)
                if not len(g) or not len(p):
                    continue
                iou = E.box_iou_np(g[:, 1:5], p[:, :4])
                for gi in range(len(g)):
                    j = int(np.argmax(iou[gi]))
                    if iou[gi, j] >= 0.5 and int(p[j, 5]) == wrong_cls:
                        n_conf += 1
            return n_gt, n_conf

        nb, nb_as_drone = confused(recs, 1, 0)
        nd, nd_as_bird = confused(recs, 0, 1)
        out[f"conf>={thr}"] = {
            "bird_only_images": len(bird_only),
            "drone_alarms_per_bird_only_image": float(np.mean(d_on_bird)) if d_on_bird else None,
            "bird_only_images_with_drone_alarm_pct": 100 * float(np.mean([x > 0 for x in d_on_bird])) if d_on_bird else None,
            "drone_only_images": len(drone_only),
            "bird_preds_per_drone_only_image": float(np.mean(b_on_drone)) if b_on_drone else None,
            "gt_birds": nb, "gt_birds_boxed_as_drone_pct": 100 * nb_as_drone / max(nb, 1),
            "gt_drones": nd, "gt_drones_boxed_as_bird_pct": 100 * nd_as_bird / max(nd, 1),
        }
    return out


def empty_frame_fp_by_class(recs, thr=0.25):
    empties = [r for r in recs if len(r["gt"]) == 0]
    n_drone = sum(int(((r["pred"][:, 5] == 0) & (r["pred"][:, 4] >= thr)).sum()) for r in empties if len(r["pred"]))
    n_any = sum(int((r["pred"][:, 4] >= thr).sum()) for r in empties if len(r["pred"]))
    return {"images": len(empties), "conf_thres": thr,
            "drone_fp_per_image": n_drone / len(empties) if empties else None,
            "any_class_fp_per_image": n_any / len(empties) if empties else None}


def main():
    ap = argparse.ArgumentParser(description="Same-set AeroTrack-Net detector evaluation")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final-eval", action="store_true", help="required for --split test")
    ap.add_argument("--cap", type=int, default=5000, help="images per dataset, evenly spread over sequences; 0 = full split")
    ap.add_argument("--datasets", default=None, help="comma list; default all")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=30)
    ap.add_argument("--report-conf", type=float, default=0.25)
    ap.add_argument("--tag", default=None, help="set tag; default cap<cap>")
    ap.add_argument("--input-mode", default="auto", choices=["auto", "temporal", "single_frame"],
                    help="auto: t-1 = t = t+1 for runs whose run_meta.json records temporal_gap 0, else the 3-frame stack")
    ap.add_argument("--smoke", type=int, default=0,
                    help="pipeline check: score only the first N images of the VAL set and write nothing")
    a = ap.parse_args()
    if a.split == "test" and not a.final_eval:
        sys.exit("Refusing: --split test requires --final-eval (final evaluation of finished models only).")
    if a.smoke and (a.split != "val" or a.final_eval):
        sys.exit("Refusing: --smoke is a VAL-only pipeline check.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = a.datasets.split(",") if a.datasets else None
    samples, setinfo = eval_set(a.split, a.cap, datasets)
    settag = a.tag or (f"cap{a.cap}" if a.cap else "full")
    print(f"[set] {a.split} {settag}: {setinfo['n_images']:,} images {setinfo['per_dataset']} from {setinfo['n_sequences']} sequences | sha256 {setinfo['sha256'][:12]}")
    if a.smoke:
        samples = samples[:a.smoke]
        print(f"[smoke] scoring the first {len(samples)} images only; nothing is written")
    res_px = native_resolutions(samples)
    label_cache = load_label_cache(LABEL_CACHE_PATH)
    amp = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    if a.split == "test":
        print("!" * 100 + "\n!!  FINAL TEST EVALUATION — recorded in " + str(LEDGER) + "\n" + "!" * 100)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / f"detector_{a.split}_{settag}.json"

    # one loader per input mode, built when first needed; checkpoints are grouped by mode (order kept within a
    # group) so at most one worker pool is alive at a time
    jobs = [(Path(w),) + input_mode_for(Path(w), a.input_mode) for w in a.weights]
    jobs = [j for m in ("temporal", "single_frame") for j in jobs if j[1] == m]
    loader_mode, dl = None, None

    def make_loader(mode):
        ss = single_frame_samples(samples) if mode == "single_frame" else samples
        ds = AeroTrackDataset(ss, img_size=(640, 640), augment=None, letterbox=True, out_dtype="uint8",
                              return_meta=True, label_cache=label_cache)
        return DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=(device.type == "cuda"),
                          collate_fn=aerotrack_collate, persistent_workers=a.workers > 0)

    rows = []
    for w, mode, mode_src in jobs:
        if mode != loader_mode:
            if dl is not None and getattr(dl, "_iterator", None) is not None:
                dl._iterator._shutdown_workers()
            dl, loader_mode = make_loader(mode), mode
            print(f"[input] {INPUT_LABEL[mode]} for the checkpoints below", flush=True)
        run_name = w.parent.parent.name
        out_dir = REPO_ROOT / "results/runs" / run_name / f"eval_repair_{a.split}_{settag}" / w.stem
        if not a.smoke:
            out_dir.mkdir(parents=True, exist_ok=True)
        model, ck = load_checkpoint(w, device, prefer_ema=True)
        model.eval()
        t0 = time.time()
        recs = E.collect_records(model, dl, device, imgsz=640, conf=a.conf, iou=a.iou, max_det=a.max_det, amp_dtype=amp, progress=False)
        el = time.time() - t0
        if a.smoke:
            o = E.score(recs)
            print(f"[smoke {run_name}/{w.stem} ep{ck.get('epoch')} input {INPUT_LABEL[mode]} ({mode_src})] {len(recs)} images, "
                  f"{sum(len(r['pred']) for r in recs)} predictions, mAP50 {o['map50'] if o else float('nan'):.4f} ({el:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
            continue
        rep = E.full_report(recs, save_dir=out_dir, conf_thres=a.report_conf)
        rep["bird_confusion_corrected"] = bird_confusion(recs)
        rep["empty_frame_fp_by_class"] = empty_frame_fp_by_class(recs, a.report_conf)
        rep["size_definition"] = {"rule": "relative area of the SOURCE frame: micro < 0.03 %, small 0.03-1 %, standard > 1 %",
                                  "pixel_equivalents_by_dataset": res_px}
        rep["_meta"] = {"weights": str(w), "weights_sha256": file_sha256(w), "run": run_name, "checkpoint": w.stem,
                        "epoch": ck.get("epoch"), "best_fitness": ck.get("best_fitness"),
                        "train_args": ck.get("train_args"), "eval_set": setinfo, "conf": a.conf, "iou": a.iou,
                        "max_det": a.max_det, "report_conf": a.report_conf, "seconds": round(el, 1),
                        "input_mode": mode, "input_frames": INPUT_LABEL[mode], "input_mode_source": mode_src,
                        "git": git_info(), "env": env_snapshot(), "command": " ".join(sys.argv),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        _write_text_atomic(out_dir / "evaluation_report.json", json.dumps(rep, indent=2))
        o, bs, bd = rep["overall"], rep["by_size"], rep["by_dataset"]
        bc = rep["bird_confusion_corrected"].get("conf>=0.25", {})
        row = {"run": run_name, "checkpoint": w.stem, "epoch": ck.get("epoch"), "weights_sha256": rep["_meta"]["weights_sha256"][:12],
               "set_sha256": setinfo["sha256"][:12], "n_images": len(recs),
               "mAP50": o["map50"], "mAP50-95": o["map5095"],
               "drone_AP50": o["per_class_ap50"]["uav_drone"], "bird_AP50": o["per_class_ap50"]["bird_hard_negative"],
               "micro_AP50": bs["micro"]["map50"], "micro_n_gt": bs["micro"]["n_gt"],
               "small_AP50": bs["small"]["map50"], "standard_AP50": bs["standard"]["map50"],
               "anti_uav_AP50": bd.get("anti_uav", {}).get("map50"), "cst_AP50": bd.get("cst", {}).get("map50"),
               "cst_micro_AP50": bd.get("cst", {}).get("micro_ap50"), "det_fly_AP50": bd.get("det_fly", {}).get("map50"),
               "bird_only_img_drone_alarm_pct@.25": bc.get("bird_only_images_with_drone_alarm_pct"),
               "gt_birds_boxed_as_drone_pct@.25": bc.get("gt_birds_boxed_as_drone_pct"),
               "empty_frame_drone_FP/img@.25": rep["empty_frame_fp_by_class"]["drone_fp_per_image"],
               "input": INPUT_LABEL[mode],
               "seconds": round(el, 1), "report": str(out_dir / "evaluation_report.json")}
        rows.append(row)
        print(f"[{run_name}/{w.stem} ep{ck.get('epoch')} input {INPUT_LABEL[mode]}] mAP50 {o['map50']:.4f} drone {row['drone_AP50']:.4f} bird {row['bird_AP50']:.4f} "
              f"micro {row['micro_AP50']:.4f} (n={row['micro_n_gt']}) cst {row['cst_AP50']} det_fly {row['det_fly_AP50']} "
              f"bird-alarm {row['bird_only_img_drone_alarm_pct@.25']} % ({el:.0f}s)", flush=True)
        # persist this checkpoint's row BEFORE the ledger line: a ledger line always means "scored and recorded"
        out = merge_summary(summary_path, setinfo, res_px, [row])
        if a.split == "test":
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            with open(LEDGER, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": rep["_meta"]["timestamp"], "weights": str(w), "weights_sha256": rep["_meta"]["weights_sha256"],
                                    "set_sha256": setinfo["sha256"], "mAP50": o["map50"], "input": INPUT_LABEL[mode],
                                    "git_commit": rep["_meta"]["git"]["commit"]}) + "\n")
                f.flush()
                os.fsync(f.fileno())
        del model
        torch.cuda.empty_cache()

    if dl is not None and getattr(dl, "_iterator", None) is not None:
        dl._iterator._shutdown_workers()
    if a.smoke:
        print(f"[smoke] OK: {len(jobs)} checkpoint(s) scored on {len(samples)} images; nothing written")
        return
    print(f"wrote {summary_path} ({len(out['rows'])} rows on set {setinfo['sha256'][:12]})")


def render_md(out):
    s = out["eval_set"]
    L = [f"# Detector comparison — {s['split']} set `{s['sha256'][:12]}` ({s['n_images']:,} images: {s['per_dataset']}, {s['n_sequences']} sequences)", "",
         "Every row is scored on exactly this image set (hash above) with conf 0.001 / IoU 0.7 / max_det 30; numbers come from "
         "`evaluation_report.json` files written by `evaluate_detector.py`. Size bins: micro < 0.03 % of source-frame area, "
         "small 0.03–1 %, standard > 1 %; pixel equivalents: " +
         ", ".join(f"{k} micro ≤ {v['micro_max_side_px']} px" for k, v in out["size_definition_px"].items()) + ". "
         "Input: each checkpoint is scored on its own training input (t,t,t for temporal_gap-0 runs; rows without the field predate "
         "input modes and are 3-frame t-1,t,t+1).", "",
         "| run | ckpt | epoch | input | mAP50 | mAP50-95 | drone AP50 | bird AP50 | micro AP50 (n) | small | standard | Anti-UAV | CST | CST micro | Det-Fly | bird-only imgs w/ drone alarm @.25 | GT birds boxed as drone @.25 | empty-frame drone FP/img @.25 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fm = lambda v: "—" if v is None else (f"{v:.4f}" if isinstance(v, float) and abs(v) <= 1.5 else f"{v:.1f}" if isinstance(v, float) else str(v))
    for r in sorted(out["rows"], key=lambda r: (r["run"], r["checkpoint"])):
        L.append(f"| {r['run']} | {r['checkpoint']} | {r['epoch']} | {r.get('input', 't-1,t,t+1')} | {fm(r['mAP50'])} | {fm(r['mAP50-95'])} | {fm(r['drone_AP50'])} | {fm(r['bird_AP50'])} | "
                 f"{fm(r['micro_AP50'])} ({r['micro_n_gt']}) | {fm(r['small_AP50'])} | {fm(r['standard_AP50'])} | {fm(r['anti_uav_AP50'])} | {fm(r['cst_AP50'])} | "
                 f"{fm(r['cst_micro_AP50'])} | {fm(r['det_fly_AP50'])} | {fm(r['bird_only_img_drone_alarm_pct@.25'])} % | {fm(r['gt_birds_boxed_as_drone_pct@.25'])} % | "
                 f"{fm(r['empty_frame_drone_FP/img@.25'])} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
