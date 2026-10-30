#!/usr/bin/env python
"""Dataset audit + manifest freeze for the Anti-UAV reproduction.

    python experiments/anti_uav/scripts/verify_dataset.py [--full-frame-check] [--force-rebuild]

Checks (spec §2 / task "dataset_integrity"): existence, sequence/frame/box counts per split and modality,
frame-vs-label alignment on disk, resolution per modality, malformed rects, RGB/IR pairing, duplicate label
files, cross-split sequence-name and frame-hash collisions, video-level overlap between splits (the paper's
train/val splits share source videos by design; test must not), object-size distributions, GlobalTrack
frame-filter pass rates, start-frame statistics, attribute counts.

Writes experiments/anti_uav/manifests/antiuav_<MOD>_<split>.manifest.json (frozen identity of each split; an
existing manifest is VERIFIED, not overwritten, unless --force-rebuild) and reports/dataset_verification.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from anti_uav_repro.data.antiuav import (SPLITS, build_manifest, load_manifest, load_sequences, manifest_path,
                                        verify_against_manifest)
from anti_uav_repro.data.pairs import globaltrack_frame_filter
from anti_uav_repro.paths import MANIFESTS_DIR, REPORTS_DIR, resolve
from anti_uav_repro.utils.repro import file_sha256, git_info, write_json

DATA_ROOT = "data/anti_uav"
FRAMES_ROOT = "data/derived/data_unified/anti_uav/images"
VIDEO_RE = re.compile(r"^(\d{8}_\d{6}_\d)_\d+$")   # <video id>_<clip index>


def pct(a, q):
    return [float(np.percentile(a, x)) for x in q] if len(a) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-frame-check", action="store_true", help="decode EVERY frame (slow: ~600k JPEGs)")
    ap.add_argument("--force-rebuild", action="store_true", help="overwrite existing manifests (changes split identity!)")
    args = ap.parse_args()
    t0 = time.time()
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_info(), "data_root": str(resolve(DATA_ROOT)),
                    "frames_root": str(resolve(FRAMES_ROOT)), "splits": {}, "problems": [], "manifests": {}}
    problems = report["problems"]
    label_hash_index = defaultdict(list)       # sha -> [(split, mod, seq)]
    first_frame_hash_index = defaultdict(list)
    videos_by_split = defaultdict(set)
    seqs_by_split = {}

    for split in SPLITS:
        report["splits"][split] = {}
        for mod in ("IR", "RGB"):
            seqs = load_sequences(DATA_ROOT, FRAMES_ROOT, split, mod, purpose="verify")
            seqs_by_split[(split, mod)] = seqs
            st = {"n_sequences": len(seqs), "n_frames": 0, "n_exist": 0, "n_absent": 0, "malformed_rect_on_exist": 0,
                  "nonempty_rect_on_absent": 0, "frames_on_disk_mismatch": [], "unreadable_frames": [], "resolutions": Counter(),
                  "start_frames_gt0": [], "seq_len_min": None, "seq_len_max": None, "attributes": Counter()}
            widths, heights, sizes, areas, keep = [], [], [], [], []
            for s in seqs:
                st["n_frames"] += s.n_frames
                st["n_exist"] += s.n_exist
                st["n_absent"] += s.n_frames - s.n_exist
                st["seq_len_min"] = s.n_frames if st["seq_len_min"] is None else min(st["seq_len_min"], s.n_frames)
                st["seq_len_max"] = s.n_frames if st["seq_len_max"] is None else max(st["seq_len_max"], s.n_frames)
                for a in s.attributes:
                    st["attributes"][a] += 1
                m = VIDEO_RE.match(s.name)
                if m:
                    videos_by_split[split].add(m.group(1))
                # rect sanity
                for i, e in enumerate(s.exist):
                    r = s.gt_rect[i]
                    ok = isinstance(r, (list, tuple)) and len(r) == 4
                    if e and not ok:
                        st["malformed_rect_on_exist"] += 1
                    if (not e) and ok and any(v != 0 for v in r):
                        st["nonempty_rect_on_absent"] += 1
                # frames on disk
                n_disk = sum(1 for p in os.scandir(s.frames_dir) if p.name.endswith(".jpg"))
                if n_disk != s.n_frames:
                    st["frames_on_disk_mismatch"].append((s.name, s.n_frames, n_disk))
                # readability + resolution (first / middle / last, or all with --full-frame-check)
                idxs = range(s.n_frames) if args.full_frame_check else (0, s.n_frames // 2, s.n_frames - 1)
                for i in idxs:
                    im = cv2.imread(str(s.frame_path(i)), cv2.IMREAD_UNCHANGED)
                    if im is None:
                        st["unreadable_frames"].append(str(s.frame_path(i)))
                    elif i == 0:
                        st["resolutions"][f"{im.shape[1]}x{im.shape[0]}"] += 1
                        s.height, s.width = im.shape[:2]
                if s.start_frame > 0:
                    st["start_frames_gt0"].append((s.name, s.start_frame))
                # sizes + filter
                valid = s.valid_frames()
                if len(valid):
                    b = np.stack([s.box_xywh(int(i)) for i in valid]).astype(np.float32)
                    widths.append(b[:, 2]); heights.append(b[:, 3])
                    sizes.append(np.sqrt(b[:, 2] * b[:, 3]))
                    W, H = s.image_size()
                    areas.append(b[:, 2] * b[:, 3] / float(W * H))
                    keep.append(globaltrack_frame_filter(b, W, H))
                # identity indices
                lh = file_sha256(s.label_path)
                label_hash_index[lh].append((split, mod, s.name))
                first_frame_hash_index[file_sha256(s.frame_path(0))].append((split, mod, s.name))
            w = np.concatenate(widths) if widths else np.array([])
            h = np.concatenate(heights) if heights else np.array([])
            sz = np.concatenate(sizes) if sizes else np.array([])
            ar = np.concatenate(areas) if areas else np.array([])
            kp = np.concatenate(keep) if keep else np.array([], dtype=bool)
            st["box_stats"] = {
                "n_boxes": int(len(sz)),
                "width_px_p5_50_95": pct(w, [5, 50, 95]), "height_px_p5_50_95": pct(h, [5, 50, 95]),
                "size_sqrt_wh_px_mean": float(sz.mean()) if len(sz) else None, "size_sqrt_wh_px_p5_50_95": pct(sz, [5, 50, 95]),
                "rel_area_percent_p5_50_95": [100 * v for v in pct(ar, [5, 50, 95])],
                "frac_LR_lt400px2": float((w * h < 400).mean()) if len(w) else None,
                "frac_side_lt10px": float(((w < 10) | (h < 10)).mean()) if len(w) else None,
                "globaltrack_filter_pass_rate": float(kp.mean()) if len(kp) else None,
                "sequences_with_zero_trainable_frames": int(sum(1 for k in keep if not k.any())),
            }
            st["resolutions"] = dict(st["resolutions"])
            st["attributes"] = dict(sorted(st["attributes"].items()))
            st["n_start_frames_gt0"] = len(st["start_frames_gt0"])
            report["splits"][split][mod] = st
            if st["frames_on_disk_mismatch"]:
                problems.append(f"{split}/{mod}: {len(st['frames_on_disk_mismatch'])} sequences with frame/label count mismatch")
            if st["unreadable_frames"]:
                problems.append(f"{split}/{mod}: {len(st['unreadable_frames'])} unreadable frames")
            if st["malformed_rect_on_exist"]:
                problems.append(f"{split}/{mod}: {st['malformed_rect_on_exist']} exist frames with malformed rect")
            if len(st["resolutions"]) != 1:
                problems.append(f"{split}/{mod}: mixed resolutions {st['resolutions']}")

            # ---- manifest freeze / verify
            mp = manifest_path(MANIFESTS_DIR, split, mod)
            if mp.exists() and not args.force_rebuild:
                frozen = load_manifest(mp)
                probs = verify_against_manifest(seqs, frozen)
                report["manifests"][mp.name] = {"status": "verified" if not probs else "MISMATCH", "sha256": frozen["manifest_sha256"], "problems": probs}
                if probs:
                    problems.append(f"manifest {mp.name}: {len(probs)} mismatches vs frozen manifest")
            else:
                man = build_manifest(seqs, check_frames=True)
                write_json(man, mp)
                report["manifests"][mp.name] = {"status": "created", "sha256": man["manifest_sha256"]}
            print(f"[{split:5s}/{mod:3s}] seqs {len(seqs):3d} frames {st['n_frames']:6d} exist {st['n_exist']:6d} "
                  f"res {st['resolutions']} filter-pass {st['box_stats']['globaltrack_filter_pass_rate']:.3f} "
                  f"start>0: {st['n_start_frames_gt0']}  ({time.time()-t0:.0f}s)", flush=True)

        # RGB/IR pairing
        ir_names = {s.name for s in seqs_by_split[(split, "IR")]}
        rgb_names = {s.name for s in seqs_by_split[(split, "RGB")]}
        if ir_names != rgb_names:
            problems.append(f"{split}: IR/RGB sequence sets differ")
        report["splits"][split]["pairing_ok"] = ir_names == rgb_names

    # ---- cross-split checks
    names = {sp: {s.name for s in seqs_by_split[(sp, "IR")]} for sp in SPLITS}
    PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))
    overlap = {f"{a}&{b}": sorted(names[a] & names[b]) for a, b in PAIRS}
    dup_labels = [v for v in label_hash_index.values() if len(v) > 1]
    dup_first = [v for v in first_frame_hash_index.values() if len(v) > 1]
    video_overlap = {f"{a}&{b}": sorted(videos_by_split[a] & videos_by_split[b]) for a, b in PAIRS}
    report["cross_split"] = {
        "sequence_name_overlap": {k: len(v) for k, v in overlap.items()},
        "duplicate_label_files": dup_labels,
        "duplicate_first_frames": dup_first,
        "video_id_overlap_counts": {k: len(v) for k, v in video_overlap.items()},
        "video_ids_per_split": {k: len(v) for k, v in videos_by_split.items()},
        "video_ids_shared_train_val": video_overlap["train&val"],
        "note": ("The paper states train and val come from non-overlapping clips of the SAME videos (expected video-id overlap), "
                 "while test is from independent videos (video-id overlap with train/val must be 0)."),
    }
    for k, v in overlap.items():
        if v:
            problems.append(f"sequence names shared between {k}: {len(v)}")
    if dup_labels:
        problems.append(f"{len(dup_labels)} label files are byte-identical across sequences")
    if dup_first:
        problems.append(f"{len(dup_first)} first frames are byte-identical across sequences")
    for k in ("train&test", "val&test"):
        if video_overlap[k]:
            problems.append(f"TEST shares source videos with {k.split('&')[0]}: {video_overlap[k]}")

    report["elapsed_seconds"] = round(time.time() - t0, 1)
    report["verdict"] = "OK" if not problems else "PROBLEMS"
    write_json(report, REPORTS_DIR / "dataset_verification.json")
    (REPORTS_DIR / "dataset_verification.md").write_text(render_md(report), encoding="utf-8")
    print("\n" + render_md(report))
    print(f"wrote {REPORTS_DIR / 'dataset_verification.json'} and .md")


def render_md(r: dict) -> str:
    L = ["# Anti-UAV dataset verification", "", f"Generated {r['generated']} (git {r['git'].get('commit')}, dirty={r['git'].get('dirty')}). "
         f"Verdict: **{r['verdict']}**", "", f"- data root: `{r['data_root']}`", f"- frames root: `{r['frames_root']}`", ""]
    L += ["## Counts", "", "| split | modality | sequences | frames | exist (boxes) | absent | seq len | resolution | start>0 | filter pass | zero-trainable seqs |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sp, mods in r["splits"].items():
        for mod in ("IR", "RGB"):
            s = mods[mod]
            b = s["box_stats"]
            L.append(f"| {sp} | {mod} | {s['n_sequences']} | {s['n_frames']} | {s['n_exist']} | {s['n_absent']} | {s['seq_len_min']}–{s['seq_len_max']} | "
                     f"{s['resolutions']} | {s['n_start_frames_gt0']} | {b['globaltrack_filter_pass_rate']:.3f} | {b['sequences_with_zero_trainable_frames']} |")
    L += ["", "Paper (Table I): 318 pairs, 585.9k boxes; train 160 seqs / 294.4k boxes; test 91 / 168.4k. "
          "Repo totals above are ~2% lower (documented discrepancy).", ""]
    L += ["## Object size (exist frames)", "", "| split | modality | boxes | √(wh) mean px | √(wh) p5/50/95 px | w p5/50/95 | h p5/50/95 | rel. area % p5/50/95 | LR (<400 px²) | side<10 px |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for sp, mods in r["splits"].items():
        for mod in ("IR", "RGB"):
            b = mods[mod]["box_stats"]
            f3 = lambda xs: "/".join(f"{x:.1f}" for x in xs)
            L.append(f"| {sp} | {mod} | {b['n_boxes']} | {b['size_sqrt_wh_px_mean']:.1f} | {f3(b['size_sqrt_wh_px_p5_50_95'])} | {f3(b['width_px_p5_50_95'])} | "
                     f"{f3(b['height_px_p5_50_95'])} | {'/'.join(f'{x:.3f}' for x in b['rel_area_percent_p5_50_95'])} | {100*b['frac_LR_lt400px2']:.1f}% | {100*b['frac_side_lt10px']:.1f}% |")
    L += ["", "## Attributes (sequences per attribute)", ""]
    for sp, mods in r["splits"].items():
        L.append(f"- {sp} (IR labels): {mods['IR']['attributes']}")
    L += ["", "## Start frames > 0 (tracker init frame = first exist frame + official corrections)", ""]
    for sp, mods in r["splits"].items():
        for mod in ("IR", "RGB"):
            L.append(f"- {sp}/{mod}: {mods[mod]['start_frames_gt0']}")
    c = r["cross_split"]
    L += ["", "## Cross-split integrity", "",
          f"- sequence-name overlap: {c['sequence_name_overlap']}",
          f"- byte-identical label files across sequences: {len(c['duplicate_label_files'])}",
          f"- byte-identical first frames across sequences: {len(c['duplicate_first_frames'])}",
          f"- source-video ids per split: {c['video_ids_per_split']}; overlap: {c['video_id_overlap_counts']}",
          f"- {c['note']}", "", "## Manifests", ""]
    for k, v in r["manifests"].items():
        L.append(f"- `{k}`: {v['status']} sha256 `{v['sha256'][:16]}…`" + (f" problems: {v['problems']}" if v.get("problems") else ""))
    L += ["", "## Problems", ""]
    L += [f"- {p}" for p in r["problems"]] or ["- none"]
    L += ["", f"Elapsed {r['elapsed_seconds']} s.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
