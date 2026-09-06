#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_uncompress_datasets.py
========================
Automated decompression & structuring for three UAV tracking datasets.

  1. Anti-UAV (Anti-UAV-RGBT.zip)   -> valid ZIP64 single archive
  2. CST Anti-UAV (.z01 .. .z07)    -> INCOMPLETE split-zip set (final .zip volume missing)
  3. Det-Fly (Det-Fly.v5i.yolov11)  -> valid standard zip (Roboflow YOLO export)

Design notes
------------
* Anti-UAV and Det-Fly are extracted with the stdlib ``zipfile`` module
  (ZIP64 is handled transparently), with zip-slip path guarding.

* CST Anti-UAV is a *spanned* (multi-volume) zip whose terminal ``.zip`` volume
  -- the one that carries the End-Of-Central-Directory + central directory --
  was never delivered (only z01..z07 exist).  No conventional tool (unzip, 7z,
  Python zipfile) can open it, because they all seek to the EOCD first.

  We instead perform a **forward streaming recovery**: the seven volumes are
  chained into one logical byte stream and walked *local-file-header by
  local-file-header*.  Because the archive was written WITHOUT data descriptors
  (general-purpose bit 3 == 0, verified on inspection), every local header
  carries the authoritative compressed size, so each member can be extracted
  deterministically until the stream truncates inside the last (partial) member
  at the tail of z07.  CRC-32 is verified per member for integrity accounting.

All outputs are logged to a_inspection/logs and a machine-readable summary is
written to a_inspection/artifacts/extraction_summary.json .
"""

import os
import sys
import json
import time
import zlib
import struct
import zipfile
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # D:\Drone_Maxxing
DATASETS = ROOT / "datasets"
DATA = ROOT / "data"
LOG_DIR = ROOT / "a_inspection" / "logs"
ART_DIR = ROOT / "a_inspection" / "artifacts"
for d in (DATA, LOG_DIR, ART_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
ANN_EXTS = {".json", ".txt", ".xml", ".yaml", ".yml", ".csv"}

_LOG_LINES = []


def log(msg):
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    _LOG_LINES.append(line)


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024.0:
            return f"{nbytes:,.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:,.2f} PB"


# --------------------------------------------------------------------------- #
#  Standard (single-volume) zip extraction
# --------------------------------------------------------------------------- #
def extract_standard_zip(zip_path: Path, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted_ok"
    if marker.exists():
        log(f"[{label}] Already extracted (marker present) -> skipping.")
        return {"skipped": True}

    t0 = time.time()
    out_root = out_dir.resolve()
    n_total = n_files = n_dirs = 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        n_total = len(infos)
        log(f"[{label}] Opening {zip_path.name} : {n_total:,} entries "
            f"(zip64={zf._end_record if hasattr(zf,'_end_record') else 'n/a'})")
        for i, info in enumerate(infos, 1):
            # zip-slip guard
            target = (out_dir / info.filename).resolve()
            if not str(target).startswith(str(out_root)):
                log(f"[{label}]  !! skipping suspicious path: {info.filename}")
                continue
            zf.extract(info, out_dir)
            if info.is_dir():
                n_dirs += 1
            else:
                n_files += 1
            if i % 250 == 0 or i == n_total:
                log(f"[{label}]  extracted {i:,}/{n_total:,} entries")
    marker.write_text("ok")
    dt = time.time() - t0
    log(f"[{label}] Done: {n_files:,} files / {n_dirs:,} dirs in {dt:,.1f}s")
    return {"skipped": False, "entries": n_total, "files": n_files,
            "dirs": n_dirs, "seconds": round(dt, 1)}


# --------------------------------------------------------------------------- #
#  Split-zip streaming recovery (for the truncated CST volume set)
# --------------------------------------------------------------------------- #
class ChainedReader:
    """Read across a list of files as one continuous forward-only byte stream."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.sizes = [p.stat().st_size for p in self.paths]
        self.total = sum(self.sizes)
        self._idx = 0
        self._fh = open(self.paths[0], "rb")
        self.pos = 0

    def read(self, n):
        out = bytearray()
        while n > 0:
            chunk = self._fh.read(n)
            if not chunk:                       # current volume exhausted
                self._idx += 1
                if self._idx >= len(self.paths):
                    break                       # end of the whole chain
                self._fh.close()
                self._fh = open(self.paths[self._idx], "rb")
                continue
            out += chunk
            n -= len(chunk)
        self.pos += len(out)
        return bytes(out)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def _parse_zip64_extra(extra: bytes, comp, uncomp):
    """Pull true 64-bit sizes out of the ZIP64 extra field when the 32-bit
    fields are saturated (0xFFFFFFFF)."""
    pos = 0
    while pos + 4 <= len(extra):
        hid, dsize = struct.unpack_from("<HH", extra, pos)
        body = extra[pos + 4: pos + 4 + dsize]
        pos += 4 + dsize
        if hid == 0x0001:                       # ZIP64 extended information
            off = 0
            if uncomp == 0xFFFFFFFF and off + 8 <= len(body):
                uncomp = struct.unpack_from("<Q", body, off)[0]
                off += 8
            if comp == 0xFFFFFFFF and off + 8 <= len(body):
                comp = struct.unpack_from("<Q", body, off)[0]
                off += 8
            break
    return comp, uncomp


def _extract_member(reader: ChainedReader, method: int, comp_size: int, dst: Path):
    """Stream `comp_size` compressed bytes from `reader`, inflate if needed,
    write to `dst`, return (crc32, truncated_bool)."""
    remaining = comp_size
    crc = 0
    CHUNK = 1 << 20
    dec = zlib.decompressobj(-15) if method == 8 else None
    with open(dst, "wb") as out:
        while remaining > 0:
            chunk = reader.read(min(CHUNK, remaining))
            if not chunk:
                return crc & 0xFFFFFFFF, True   # stream truncated mid-member
            remaining -= len(chunk)
            if method == 0:                     # stored
                out.write(chunk)
                crc = zlib.crc32(chunk, crc)
            elif method == 8:                   # deflate
                data = dec.decompress(chunk)
                out.write(data)
                crc = zlib.crc32(data, crc)
            else:                               # unknown -> store raw
                out.write(chunk)
                crc = zlib.crc32(chunk, crc)
        if dec is not None:
            tail = dec.flush()
            out.write(tail)
            crc = zlib.crc32(tail, crc)
    return crc & 0xFFFFFFFF, False


def recover_split_zip(volumes, out_dir: Path, label: str, sample_flags=25):
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted_ok"
    if marker.exists():
        log(f"[{label}] Already recovered (marker present) -> skipping.")
        return {"skipped": True}

    reader = ChainedReader(volumes)
    log(f"[{label}] Chained {len(volumes)} volumes = {human(reader.total)} "
        f"(no central directory available -> streaming recovery)")

    # Skip the 4-byte spanning marker (PK\x07\x08) at the very start, if present.
    head = reader.read(4)
    if head != b"PK\x07\x08":
        log(f"[{label}]  note: first 4 bytes are {head!r}, not a spanning marker; "
            f"treating as start of data.")
        # We already consumed 4 bytes; those belong to the first signature.
        pending_sig = head
    else:
        pending_sig = None

    n_files = n_dirs = crc_ok = crc_bad = 0
    method_counts = {}
    flag_samples = []
    truncated_member = None
    ext_counts = {}
    t0 = time.time()

    while True:
        sig = pending_sig if pending_sig is not None else reader.read(4)
        pending_sig = None
        if len(sig) < 4:
            break                                       # clean end of stream

        if sig == b"PK\x03\x04":                        # local file header
            hdr = reader.read(26)
            if len(hdr) < 26:
                truncated_member = "<header>"
                break
            (ver, flag, method, mtime, mdate, crc32,
             comp, uncomp, name_len, extra_len) = struct.unpack("<HHHHHIIIHH", hdr)
            name = reader.read(name_len)
            extra = reader.read(extra_len)
            if len(name) < name_len or len(extra) < extra_len:
                truncated_member = "<name/extra>"
                break
            fname = name.decode("utf-8", "replace").replace("\\", "/")

            if len(flag_samples) < sample_flags:
                flag_samples.append({"name": fname, "flag": flag, "method": method,
                                     "comp": comp, "uncomp": uncomp})

            if comp == 0xFFFFFFFF or uncomp == 0xFFFFFFFF:
                comp, uncomp = _parse_zip64_extra(extra, comp, uncomp)

            # Data-descriptor safety: if bit 3 set and sizes are zero we cannot
            # know the member length from the header -> stop cleanly & report.
            if (flag & 0x08) and comp == 0:
                log(f"[{label}]  !! entry '{fname}' uses a data descriptor with "
                    f"zero header size; streaming recovery cannot bound it. Stopping.")
                truncated_member = fname
                break

            # zip-slip guard
            target = (out_dir / fname).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                log(f"[{label}]  !! skipping suspicious path: {fname}")
                # still must consume the data to stay aligned
                _extract_member(reader, method, comp, Path(os.devnull))
                continue

            if fname.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                n_dirs += 1
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            calc_crc, trunc = _extract_member(reader, method, comp, target)
            if trunc:
                truncated_member = fname
                log(f"[{label}]  stream truncated inside '{fname}' "
                    f"(expected {human(comp)}); removing partial file.")
                try:
                    target.unlink()
                except Exception:
                    pass
                break

            method_counts[method] = method_counts.get(method, 0) + 1
            ext = target.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            if crc32 != 0:
                if calc_crc == crc32:
                    crc_ok += 1
                else:
                    crc_bad += 1
            n_files += 1
            if n_files % 2000 == 0:
                log(f"[{label}]  recovered {n_files:,} files "
                    f"({human(reader.pos)} / {human(reader.total)} read)")

        elif sig == b"PK\x01\x02":              # central directory (unexpected here)
            log(f"[{label}]  reached central directory signature -> stop.")
            break
        elif sig in (b"PK\x05\x06", b"PK\x06\x06", b"PK\x06\x07"):
            break                               # EOCD / ZIP64 EOCD
        elif sig == b"PK\x07\x08":              # stray data descriptor -> skip 12 bytes
            reader.read(12)
        else:
            log(f"[{label}]  !! unexpected signature {sig!r} at byte {reader.pos:,}; "
                f"stopping recovery (stream likely misaligned or ended).")
            break

    reader.close()
    marker.write_text("ok")
    dt = time.time() - t0
    log(f"[{label}] Recovery done: {n_files:,} files / {n_dirs:,} dirs, "
        f"CRC ok={crc_ok:,} bad={crc_bad:,}, "
        f"read {human(reader.pos)}/{human(reader.total)} in {dt:,.1f}s")
    if truncated_member:
        log(f"[{label}] NOTE: last member truncated/absent at '{truncated_member}' "
            f"(consistent with the missing final .zip volume).")
    return {
        "skipped": False, "files": n_files, "dirs": n_dirs,
        "crc_ok": crc_ok, "crc_bad": crc_bad,
        "bytes_read": reader.pos, "bytes_total": reader.total,
        "seconds": round(dt, 1), "truncated_member": truncated_member,
        "method_counts": method_counts, "ext_counts": ext_counts,
        "flag_samples": flag_samples,
    }


# --------------------------------------------------------------------------- #
#  Post-extraction structure survey
# --------------------------------------------------------------------------- #
def survey_tree(out_dir: Path):
    n_img = n_vid = n_ann = n_other = 0
    total_bytes = 0
    ann_paths = []
    ext_hist = {}
    top_level = []
    for entry in sorted(out_dir.iterdir()) if out_dir.exists() else []:
        top_level.append(entry.name + ("/" if entry.is_dir() else ""))
    for p in out_dir.rglob("*"):
        if p.is_dir():
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            sz = 0
        total_bytes += sz
        ext = p.suffix.lower()
        ext_hist[ext] = ext_hist.get(ext, 0) + 1
        if ext in IMG_EXTS:
            n_img += 1
        elif ext in VID_EXTS:
            n_vid += 1
        elif ext in ANN_EXTS:
            n_ann += 1
            if len(ann_paths) < 40:
                ann_paths.append(str(p.relative_to(out_dir)))
        else:
            n_other += 1
    return {
        "top_level": top_level[:60],
        "n_images": n_img, "n_videos": n_vid, "n_annotations": n_ann,
        "n_other": n_other, "total_bytes": total_bytes,
        "total_human": human(total_bytes),
        "ext_histogram": dict(sorted(ext_hist.items(), key=lambda kv: -kv[1])),
        "sample_annotation_paths": ann_paths,
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    log("=" * 78)
    log("STEP 2 : Automated dataset decompression & structuring")
    log("=" * 78)

    summary = {"generated": datetime.datetime.now().isoformat(), "datasets": {}}

    # ---- 1. Anti-UAV (valid ZIP64) --------------------------------------- #
    try:
        antiuav_zip = DATASETS / "1_anti-uav" / "Anti-UAV-RGBT.zip"
        out = DATA / "anti_uav"
        if antiuav_zip.exists():
            res = extract_standard_zip(antiuav_zip, out, "Anti-UAV")
            res["survey"] = survey_tree(out)
            summary["datasets"]["anti_uav"] = res
            s = res["survey"]
            log(f"[Anti-UAV] images={s['n_images']:,} videos={s['n_videos']:,} "
                f"annotations={s['n_annotations']:,} size={s['total_human']}")
        else:
            log(f"[Anti-UAV] archive not found: {antiuav_zip}")
    except Exception as e:
        log(f"[Anti-UAV] ERROR: {e!r}")
        summary["datasets"]["anti_uav"] = {"error": repr(e)}

    # ---- 2. CST Anti-UAV (truncated split set -> recovery) --------------- #
    try:
        cst_dir = DATASETS / "2_cst-antiuav"
        volumes = sorted(cst_dir.glob("CST-AntiUAV.z0*"),
                         key=lambda p: p.suffix.lower())
        # ensure numeric order z01..z07 (suffix sort works: .z01 < .z02 ...)
        out = DATA / "cst_anti_uav"
        if volumes:
            res = recover_split_zip(volumes, out, "CST-AntiUAV")
            res["volumes"] = [v.name for v in volumes]
            res["survey"] = survey_tree(out)
            summary["datasets"]["cst_anti_uav"] = res
            s = res["survey"]
            log(f"[CST-AntiUAV] images={s['n_images']:,} videos={s['n_videos']:,} "
                f"annotations={s['n_annotations']:,} size={s['total_human']}")
        else:
            log(f"[CST-AntiUAV] no split volumes found in {cst_dir}")
    except Exception as e:
        import traceback
        log(f"[CST-AntiUAV] ERROR: {e!r}\n{traceback.format_exc()}")
        summary["datasets"]["cst_anti_uav"] = {"error": repr(e)}

    # ---- 3. Det-Fly (valid standard zip) --------------------------------- #
    try:
        detfly_zip = DATASETS / "3_det-fly" / "Det-Fly.v5i.yolov11.zip"
        out = DATA / "det_fly"
        if detfly_zip.exists():
            res = extract_standard_zip(detfly_zip, out, "Det-Fly")
            res["survey"] = survey_tree(out)
            summary["datasets"]["det_fly"] = res
            s = res["survey"]
            log(f"[Det-Fly] images={s['n_images']:,} annotations={s['n_annotations']:,} "
                f"size={s['total_human']}")
        else:
            log(f"[Det-Fly] archive not found: {detfly_zip}")
    except Exception as e:
        log(f"[Det-Fly] ERROR: {e!r}")
        summary["datasets"]["det_fly"] = {"error": repr(e)}

    # ---- persist --------------------------------------------------------- #
    (ART_DIR / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (LOG_DIR / "1_uncompress.log").write_text("\n".join(_LOG_LINES), encoding="utf-8")
    log("Wrote a_inspection/artifacts/extraction_summary.json")
    log("STEP 2 complete.")


if __name__ == "__main__":
    main()
