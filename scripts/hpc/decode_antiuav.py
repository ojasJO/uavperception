"""Re-create the decoded Anti-UAV frames on another machine with exactly the algorithm of
src/pipeline/1_standardize_annotations.py::standardize_antiuav (sequential cv2.VideoCapture.read, N = len(gt_rect) frames,
cv2.imwrite with default JPEG parameters, 1-indexed 6-digit names, directory antiuav_<split>_<seq>__<IR|RGB>).
Clip discovery and annotation parsing are the project's own (src/inspection/eda_common.py).

    python scripts/hpc/decode_antiuav.py --out <dir> [--only KEY ...] [--workers 8]

Identity is checked separately (scripts/hpc/frame_hashes.py check, Track A manifests' first/last frame sha256); this script
never overwrites an existing complete directory and writes each frame atomically (tmp + rename).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/inspection"))


def sanitize(seq):                                       # = 1_standardize_annotations.sanitize
    return seq.replace("\\", "/").strip("/").replace("/", "_")


def decode_clip(media, n, out_dir):
    import cv2
    out_dir = Path(out_dir)
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) >= n:
        return str(out_dir), n, "exists"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(media))
    written = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        p = out_dir / f"{i + 1:06d}.jpg"
        tmp = out_dir / f".{i + 1:06d}.tmp.jpg"
        cv2.imwrite(str(tmp), frame)
        os.replace(tmp, p)
        written += 1
    cap.release()
    return str(out_dir), written, "decoded"


def main():
    import eda_common as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="sequence keys antiuav_<split>_<seq>__<IR|RGB>")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = []
    for clip in C.discover_video_dataset("anti_uav"):
        exist, rects = C.parse_antiuav_ann(clip["ann_path"])
        if rects is None or not clip.get("media_path"):
            continue
        key = f"antiuav_{sanitize(clip['sequence'])}__{clip['modality']}"
        if a.only and key not in a.only:
            continue
        jobs.append((clip["media_path"], len(rects), Path(a.out) / key))
    print(f"[decode] {len(jobs)} clips -> {a.out}", flush=True)
    t0, done, frames = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(decode_clip, *j) for j in jobs]):
            d, n, how = f.result(); done += 1; frames += n
            if done % 25 == 0 or done == len(jobs):
                print(f"[decode] {done}/{len(jobs)} clips, {frames:,} frames, {time.time() - t0:.0f} s", flush=True)
    print(json.dumps({"clips": len(jobs), "frames": frames, "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
