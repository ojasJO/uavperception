"""Extract the complete CST Anti-UAV archive on the HPC with the project's own code, and prove it complete.

Re-uses experiments/cst_completion/scripts/extract_and_verify.py (volume order, ZIP64 central-directory parser, CRC-32) and
src/inspection/1_uncompress_datasets.py::recover_split_zip (the CRC-checking streaming extractor), with HPC paths:
  archives  ~/aerotrack_server/shared_data/source_archives/2_cst-antiuav/ (sha256-verified copies of the OMEN's 11 parts)
  output    <repo>/data/cst_anti_uav_full  (repo `data` -> shared_data/root/data)
Proof: every central-directory entry exists with the recorded size and a recomputed CRC-32, and the file count equals the
OMEN's verified extraction (246,131 files, experiments/cst_completion/reports/extraction.json). Step 3 of the original (consistency
with the truncated v1 copy) was done on the OMEN and is not repeated (no v1 copy exists here). Writes one JSON report.
    python scripts/hpc/extract_cst.py --archives <dir> --out <report.json>
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--archives", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X = load("extract_and_verify", REPO / "experiments" / "cst_completion" / "scripts" / "extract_and_verify.py")
    U = load("uncompress", REPO / "src/inspection" / "1_uncompress_datasets.py")
    X.P1 = X.P2 = Path(a.archives)
    out_dir = REPO / "data" / "cst_anti_uav_full"
    omen = json.loads((REPO / "experiments" / "cst_completion" / "reports" / "extraction.json").read_text(encoding="utf-8"))
    t0 = time.time()
    vols = X.volumes()
    rec = U.recover_split_zip(vols, out_dir, "CST-AntiUAV-complete")
    cd = X.central_directory(vols)
    files = [e for e in cd if not e["dir"]]
    missing, bad_size, bad_crc = [], [], []
    for k, e in enumerate(files):
        f = out_dir / e["name"]
        if not f.exists():
            missing.append(e["name"])
        elif f.stat().st_size != e["size"]:
            bad_size.append(e["name"])
        elif X.crc_file(f) != e["crc"]:
            bad_crc.append(e["name"])
        if (k + 1) % 50000 == 0:
            print(f"[verify] {k + 1:,}/{len(files):,}", flush=True)
    omen_n = omen["central_directory"]["files"]
    rep = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "volumes": [v.name for v in vols], "recover": str(rec)[:500],
           "central_directory_files": len(files), "missing": len(missing), "bad_size": len(bad_size), "bad_crc": len(bad_crc),
           "omen_report_files": omen_n, "examples": (missing + bad_size + bad_crc)[:10], "seconds": round(time.time() - t0)}
    rep["ok"] = not (missing or bad_size or bad_crc) and len(files) == omen_n
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, default=str))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
