#!/usr/bin/env python
"""Extract the COMPLETE CST Anti-UAV archive (11 parts) into a NEW directory and prove it is complete and consistent.

    venv\\Scripts\\python.exe experiments\\cst_completion\\scripts\\extract_and_verify.py

1. Extraction: the repository's own CRC-checking streaming extractor (`a_inspection/1_uncompress_datasets.py::
   recover_split_zip`, the code that produced the truncated corpus v1 copy) is run over z01..z10 + .zip in order, into
   `data/cst_anti_uav_full/` -- the v1 extraction in `data/cst_anti_uav/` is never written.
2. Completeness, independently of step 1: the ZIP64 end-of-central-directory in the terminal part is parsed, every
   central-directory entry must exist in the new tree with the recorded size and CRC-32 (CRC recomputed from the file).
3. Consistency with v1: every file of the v1 extraction must exist in the new tree with identical CRC-32 (the v1 recovery
   stopped inside train/urban-areas_1, so that one partial frame was deleted there; nothing else may differ).
4. Inventory: sequences per split and scene category, frame counts, annotation keys.
Writes experiments/cst_completion/reports/extraction.{json,md}. Re-running skips step 1 if the extraction marker exists.
"""
import importlib.util
import json
import struct
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
P1 = REPO / "datasets" / "2_cst-antiuav"
P2 = REPO / "datasets" / "2_cst-antiuav_part2"
OUT = REPO / "data" / "cst_anti_uav_full"
V1 = REPO / "data" / "cst_anti_uav"
REPORTS = REPO / "experiments" / "cst_completion" / "reports"
PARTS = [f"CST-AntiUAV.z{i:02d}" for i in range(1, 11)] + ["CST-AntiUAV.zip"]


def volumes():
    vols = []
    for p in PARTS:
        cand = P1 / p if (P1 / p).exists() else P2 / p
        if not cand.exists():
            sys.exit(f"missing archive part {p} (looked in {P1} and {P2})")
        vols.append(cand)
    for v in vols[:-1]:
        if v.stat().st_size != 2 * 1024 ** 3:
            sys.exit(f"{v.name}: {v.stat().st_size} B, a non-terminal split part must be exactly 2 GiB")
    return vols


class Chain:
    """Random access over the concatenated parts (disk i = vols[i])."""

    def __init__(self, vols):
        self.vols, self.sizes = vols, [v.stat().st_size for v in vols]
        self.starts = [sum(self.sizes[:i]) for i in range(len(vols))]

    def read(self, pos, n):
        out = bytearray()
        for i, (s, z) in enumerate(zip(self.starts, self.sizes)):
            if pos < s + z and pos + n > s:
                a = max(pos, s) - s
                with open(self.vols[i], "rb") as f:
                    f.seek(a)
                    out += f.read(min(s + z, pos + n) - (s + a))
        return bytes(out)


def central_directory(vols):
    """Parse the (ZIP64) central directory of the split archive -> list of entries."""
    ch = Chain(vols)
    last = vols[-1]
    with open(last, "rb") as f:                       # EOCD (+ up to 64 KiB comment) sits at the very end
        f.seek(max(0, last.stat().st_size - ((1 << 16) + 22 + 20)))
        tail = f.read()
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        sys.exit("no end-of-central-directory record in the terminal part")
    (_, disk, cd_disk, n_disk, n_tot, cd_size, cd_off, _clen) = struct.unpack_from("<IHHHHIIH", tail, i)
    if 0xFFFF in (disk, cd_disk, n_disk, n_tot) or 0xFFFFFFFF in (cd_size, cd_off):
        j = tail.rfind(b"PK\x06\x07", 0, i)                               # ZIP64 EOCD locator
        _, z64_disk, z64_off, _ndisks = struct.unpack_from("<IIQI", tail, j)
        rec = ch.read(ch.starts[z64_disk] + z64_off, 56)
        if rec[:4] != b"PK\x06\x06":
            sys.exit("ZIP64 end-of-central-directory record not where the locator says")
        (_, _sz, _vm, _vn, disk, cd_disk, n_disk, n_tot, cd_size, cd_off) = struct.unpack_from("<IQHHIIQQQQ", rec, 0)
    raw = ch.read(ch.starts[cd_disk] + cd_off, cd_size)
    entries, p = [], 0
    while p < len(raw) and raw[p:p + 4] == b"PK\x01\x02":
        (_, _vm, _vn, flag, method, _t, _d, crc, comp, uncomp, nl, el, cl, dstart, _ia, _ea, loff) = \
            struct.unpack_from("<IHHHHHHIIIHHHHHII", raw, p)
        name = raw[p + 46:p + 46 + nl].decode("utf-8", "replace").replace("\\", "/")
        extra = raw[p + 46 + nl:p + 46 + nl + el]
        q = 0
        while q + 4 <= len(extra):                                       # ZIP64 extra: saturated fields, in order
            hid, dsz = struct.unpack_from("<HH", extra, q)
            body, q2 = extra[q + 4:q + 4 + dsz], 0
            if hid == 1:
                if uncomp == 0xFFFFFFFF:
                    uncomp = struct.unpack_from("<Q", body, q2)[0]; q2 += 8
                if comp == 0xFFFFFFFF:
                    comp = struct.unpack_from("<Q", body, q2)[0]; q2 += 8
            q += 4 + dsz
        entries.append({"name": name, "crc": crc, "size": uncomp, "dir": name.endswith("/")})
        p += 46 + nl + el + cl
    if len(entries) != n_tot:
        sys.exit(f"central directory parse: {len(entries)} entries, EOCD says {n_tot}")
    return entries


def crc_file(path):
    c = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def main():
    t0 = time.time()
    vols = volumes()
    spec = importlib.util.spec_from_file_location("uncompress", REPO / "a_inspection" / "1_uncompress_datasets.py")
    U = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(U)
    rec = U.recover_split_zip(vols, OUT, "CST-AntiUAV-complete")
    cd = central_directory(vols)
    files = [e for e in cd if not e["dir"]]
    missing, bad_size, bad_crc = [], [], []
    for k, e in enumerate(files):
        f = OUT / e["name"]
        if not f.exists():
            missing.append(e["name"])
        elif f.stat().st_size != e["size"]:
            bad_size.append(e["name"])
        elif crc_file(f) != e["crc"]:
            bad_crc.append(e["name"])
        if (k + 1) % 50000 == 0:
            print(f"[verify] {k + 1:,}/{len(files):,} central-directory entries checked", flush=True)
    # consistency with the v1 (truncated) extraction
    v1_files = [p for p in V1.rglob("*") if p.is_file() and p.name != ".extracted_ok"]
    v1_missing, v1_diff = [], []
    for p in v1_files:
        q = OUT / p.relative_to(V1)
        if not q.exists():
            v1_missing.append(str(p.relative_to(V1)))
        elif p.stat().st_size != q.stat().st_size or crc_file(p) != crc_file(q):
            v1_diff.append(str(p.relative_to(V1)))
    # inventory
    root = next(OUT.rglob("test")).parent if any(OUT.rglob("test")) else OUT
    inv, frames = {}, {}
    for split in ("train", "val", "test"):
        d = root / split
        seqs = sorted(x for x in d.iterdir() if x.is_dir()) if d.exists() else []
        inv[split] = dict(Counter(s.name.rsplit("_", 1)[0] for s in seqs))
        frames[split] = {"sequences": len(seqs), "frames": sum(len(list(s.glob("*.jpg"))) for s in seqs),
                         "with_IR_label_json": sum((s / "IR_label.json").exists() for s in seqs)}
    keys = Counter()
    for j in list(root.rglob("IR_label.json"))[:400]:
        keys.update(json.loads(j.read_text(encoding="utf-8")).keys())
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "volumes": [str(v.relative_to(REPO)) for v in vols],
           "extract_dir": str(OUT.relative_to(REPO)), "streaming_extraction": rec,
           "central_directory": {"entries": len(cd), "files": len(files), "missing": len(missing), "size_mismatch": len(bad_size),
                                 "crc_mismatch": len(bad_crc), "examples": (missing + bad_size + bad_crc)[:20]},
           "v1_consistency": {"v1_files": len(v1_files), "absent_in_complete": v1_missing[:50], "n_absent": len(v1_missing),
                              "different_bytes": v1_diff[:50], "n_different": len(v1_diff)},
           "inventory": {"root": str(root.relative_to(REPO)), "categories": inv, "counts": frames,
                         "annotation_keys_sample": dict(keys)},
           "seconds": round(time.time() - t0, 1)}
    out["verdict"] = ("COMPLETE_AND_CONSISTENT" if not (missing or bad_size or bad_crc or v1_diff or v1_missing)
                      and rec.get("crc_bad", 0) == 0 else "PROBLEMS")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "extraction.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    L = [f"# CST Anti-UAV complete archive — extraction and verification", "",
         f"*Generated {out['generated']} by `scripts/extract_and_verify.py`. Verdict: **{out['verdict']}**.*", "",
         f"- parts: {', '.join(Path(v).name for v in out['volumes'])}",
         f"- streaming extraction: {rec.get('files', '—')} files, CRC ok {rec.get('crc_ok', '—')}, bad {rec.get('crc_bad', '—')}"
         f"{' (skipped: marker present)' if rec.get('skipped') else ''}",
         f"- central directory: {len(files):,} files; missing {len(missing)}, size mismatch {len(bad_size)}, CRC mismatch {len(bad_crc)}",
         f"- v1 extraction: {len(v1_files):,} files; absent in the complete tree {len(v1_missing)}, different bytes {len(v1_diff)}", "",
         "| split | sequences | frames | per category |", "|---|---:|---:|---|"]
    for s in ("train", "val", "test"):
        L.append(f"| {s} | {frames[s]['sequences']} | {frames[s]['frames']:,} | "
                 + ", ".join(f"{k} {v}" for k, v in sorted(inv[s].items())) + " |")
    (REPORTS / "extraction.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if out["verdict"] != "COMPLETE_AND_CONSISTENT":
        sys.exit(1)


if __name__ == "__main__":
    main()
