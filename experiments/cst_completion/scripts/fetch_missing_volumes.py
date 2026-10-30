#!/usr/bin/env python
"""Fetch the CST Anti-UAV archive volumes the local copy is missing, and fingerprint every volume.

Finding (2026-10-04): `datasets/2_cst-antiuav/` holds CST-AntiUAV.z01..z07 only -- the content of the dataset's first
Google Drive folder. The official release (github.com/PCwenyue/CST-Anti-UAV, 120/40/60 train/val/test sequences) is an
11-part split zip; parts z08, z09, z10 and the terminal CST-AntiUAV.zip (central directory) are in the SECOND Drive folder.
The 2026-08-04 streaming recovery therefore stopped inside train/urban-areas_1: the local corpus has 84 of 120 train
sequences, none of the 40 val sequences, and no urban-areas / water training scenes.

This script
  * downloads ONLY the missing parts, into a NEW directory (`datasets/2_cst-antiuav_part2/`); the existing volumes are
    read, never written;
  * is resumable (gdown resume) and idempotent (a part whose size matches the Drive listing is not re-downloaded);
  * records source URL, Drive file id, byte size and SHA-256 of all 11 parts in `<dest>/volumes.json` and in the tracked
    provenance record `experiments/cst_completion/reports/volumes.json`;
  * writes RUNNING / DONE / FAILED sentinels and a log next to the parts, so it can run unattended.

Integrity beyond hashes (CRC-32 of every member) is checked by `verify_archive.py` once all parts are present.

    set PYTHONPATH=<dir with gdown>   (kept out of the research venv on purpose)
    venv\\Scripts\\python.exe experiments\\cst_completion\\scripts\\fetch_missing_volumes.py
"""
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OLD = REPO / "datasets" / "2_cst-antiuav"
DEST = REPO / "datasets" / "2_cst-antiuav_part2"
RECORD = REPO / "experiments" / "cst_completion" / "reports" / "volumes.json"
README = "https://github.com/PCwenyue/CST-Anti-UAV"
FOLDER1 = "https://drive.google.com/drive/folders/13DIRzt1pC2c4Dhyi6yFhHBeMFoKPQtxU"
FOLDER2 = "https://drive.google.com/drive/folders/132CBS47zNjQemhBXGKTEsHdTgwiGzMub"
# Drive file ids as listed by gdown.download_folder(skip_download=True) on 2026-10-04
PARTS = {
    "CST-AntiUAV.z01": ("12il79A75af_ahfQEakKtzrA3LEqWTn2G", FOLDER1),
    "CST-AntiUAV.z02": ("1bWscp-86dgeZ8mLUNQkoLCzLZGA7wvkt", FOLDER1),
    "CST-AntiUAV.z03": ("1V6jql4Kiq6ReTv61sEh3Rx8Yxi2ZYcJw", FOLDER1),
    "CST-AntiUAV.z04": ("1pq7XRB34DI3oMANx1HqU8M3Ah11VJESa", FOLDER1),
    "CST-AntiUAV.z05": ("1XRkXugzW7Aije4vy6dEZHs1fIy6-b5i9", FOLDER1),
    "CST-AntiUAV.z06": ("1PJgfgMbQpQ5kmYaGw51qwdOAUxc03Wt9", FOLDER1),
    "CST-AntiUAV.z07": ("1HMfUlGTOhinMPKRwmt9gYfPZq8OV28dl", FOLDER1),
    "CST-AntiUAV.z08": ("1YlL527rZ6SYgIj-8IN-Mb1vhqJe_kvG5", FOLDER2),
    "CST-AntiUAV.z09": ("1QvqlSLm_XuPBnpYDcFDBJxY7UG0MKoZ_", FOLDER2),
    "CST-AntiUAV.z10": ("1zCALY_qU2KI8NZF_pQ-kSNVP0nCNpR4l", FOLDER2),
    "CST-AntiUAV.zip": ("14zO4bcudHkY2DxmbLyiRKlHpUg6vIv0d", FOLDER2),
}
SPLIT_PART_BYTES = 2 * 1024 ** 3          # every non-terminal part of this split zip is exactly 2 GiB
MAX_ATTEMPTS = 60                         # per part; each retry resumes from the bytes already received


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(DEST / "fetch.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256(path, bufsize=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json_atomic(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    import gdown  # imported here so --help works without it

    DEST.mkdir(parents=True, exist_ok=True)
    for s in ("DONE", "FAILED"):
        (DEST / s).unlink(missing_ok=True)
    (DEST / "RUNNING").write_text(f"pid {os.getpid()} started {now()}\n", encoding="utf-8")
    try:
        missing = [p for p in PARTS if not (OLD / p).exists()]
        log(f"local volumes present in {OLD.name}: {sorted(set(PARTS) - set(missing))}; to fetch: {missing}")
        for name in missing:
            fid, folder = PARTS[name]
            out = DEST / name
            # gdown writes <name>_*.part and renames it only after a complete transfer, so a final-named file is whole;
            # a split part must additionally have the exact split size
            if out.exists() and (name == "CST-AntiUAV.zip" or out.stat().st_size == SPLIT_PART_BYTES):
                log(f"{name}: already complete ({out.stat().st_size} B), not re-downloaded")
                continue
            t0 = time.time()
            got = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                # the connection drops mid-transfer on this network (IncompleteRead); gdown keeps the received bytes in a
                # .part file and resume=True continues from them, so a retry costs only the reconnect
                log(f"{name}: downloading id {fid} -> {out} (attempt {attempt})")
                try:
                    got = gdown.download(id=fid, output=str(out), quiet=True, resume=True)
                    if got and out.exists():
                        break
                    log(f"{name}: gdown returned {got!r}")
                except Exception as e:  # noqa: BLE001 — network errors are retried, anything persistent ends the loop
                    log(f"{name}: attempt {attempt} interrupted: {type(e).__name__}: {str(e)[:160]}")
                time.sleep(min(60, 5 * attempt))
            if not got or not out.exists():
                raise RuntimeError(f"{name}: not complete after {MAX_ATTEMPTS} attempts")
            log(f"{name}: {out.stat().st_size} B in {time.time() - t0:.0f} s")
        rows = []
        for name, (fid, folder) in PARTS.items():
            path = OLD / name if (OLD / name).exists() else DEST / name
            size = path.stat().st_size
            if name != "CST-AntiUAV.zip" and size != SPLIT_PART_BYTES:
                raise RuntimeError(f"{name}: {size} B, expected exactly {SPLIT_PART_BYTES} B for a non-terminal split part")
            t0 = time.time()
            digest = sha256(path)
            log(f"{name}: sha256 {digest[:16]}... ({size} B, {time.time() - t0:.0f} s, {path.parent.name})")
            rows.append({"part": name, "bytes": size, "sha256": digest, "local_dir": str(path.parent.relative_to(REPO)),
                         "drive_file_id": fid, "drive_folder": folder,
                         "acquired": "2026-08-04 (original download)" if path.parent == OLD else now()})
        record = {"dataset": "CST Anti-UAV (Xie et al., ICCVW 2025, arXiv:2507.23473)", "source": README,
                  "terms": "academic and non-commercial research use (dataset README / DATASET_TERMS.md)",
                  "official_splits_sequences": {"train": 120, "val": 40, "test": 60},
                  "generated": now(), "parts": rows,
                  "note": "SHA-256 of the archive parts as received; per-member CRC-32 integrity is checked by verify_archive.py"}
        write_json_atomic(record, DEST / "volumes.json")
        write_json_atomic(record, RECORD)
        (DEST / "DONE").write_text(f"finished {now()}\n", encoding="utf-8")
        log("DONE")
    except BaseException as e:  # noqa: BLE001 - every exit path must leave a sentinel
        (DEST / "FAILED").write_text(f"{now()} {type(e).__name__}: {e}\n{traceback.format_exc()}", encoding="utf-8")
        log(f"FAILED: {type(e).__name__}: {e}")
        raise
    finally:
        (DEST / "RUNNING").unlink(missing_ok=True)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    main()
