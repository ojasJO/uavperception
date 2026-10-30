"""Build ARTIFACT_MANIFEST.csv for an HPC -> OMEN recovery backup (read-only on the backup).

Inputs (in <recovery_dir>/_verification/): remote_sha_weights.txt, remote_sha_other.txt (sha256sum
output relative to ~/aerotrack_server on the HPC). Every listed file is re-hashed locally under
<recovery_dir>/aerotrack_server/ and compared. Registered checkpoint hashes in the frozen configs /
registries are cross-checked against the backup.

usage: venv/Scripts/python.exe tools/hpc/recovery_manifest.py artifacts/hpc_recovery_20261013 [--out runs/studio/health/recovery_check]
(--out keeps the backup directory untouched: the check then only reads from it)
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEIGHT_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine", ".bin", ".pkl", ".npz", ".joblib"}
REGISTRIES = [
    "experiments/aerotrack_v3/reports/x1_final_test.json",
    "experiments/cst_anti_uav/frozen_x2_params.json",
    "experiments/cst_anti_uav/reports/x2_final_test.json",
    "experiments/aerotrack_temporal/frozen_cf.json",
    "runs/aerotrack_temporal/c1/learned_registry.json",
    "runs/aerotrack_temporal/c1/learned_registry_c1c.json",
]


def role_of(rel):
    """(artifact_type, experiment, role, seed, run_id) from a path relative to ~/aerotrack_server."""
    m = re.search(r"runs/([^/]+)/weights/(\w+)\.pt$", rel)
    seed = (re.search(r"_s(\d)(?:_|\.|$)", rel) or [None, ""])[1]
    if m:
        run, which = m.groups()
        if "anti_uav/runs" in rel:
            return "checkpoint", "Anti-UAV GlobalTrack/DFSC reproduction", f"{run} {which}", seed, run
        if run == "v3_n__causal_g4_cst_s0_e20" and which == "last":
            role = "X1 (STE + argmax) AND X2 temporal causal_g4 seed 0; registered TEST: X1 pass PBS 93019 + X2 pass PBS 93029"
        elif run.startswith("v3_n__causal_g4_cst") and which == "last":
            role = f"X2 temporal causal_g4 seed {seed}; registered TEST PBS 93029"
        elif run.startswith("v3_n__ttt_cst") and which == "last":
            role = f"X2 single-frame control seed {seed}; registered TEST PBS 93029"
        elif run == "v2_causal_g4_s0_e20" and which == "last":
            role = "CF detector (frozen_cf.json, CF-A2); v2 selected arm causal_g4 seed 0"
        elif run.startswith("v3_n_std"):
            role = f"F1 standard yolo11n {which}"
        elif run.startswith("v2_"):
            role = f"v2 campaign arm {which}"
        else:
            role = f"{which}"
        exp = "CST/AeroTrack X1/X2 (aerotrack_v3)" if run.startswith("v3_") else "AeroTrack v2 / generalization"
        return "checkpoint", exp, role, seed, run
    if "/c1/" in rel and rel.endswith(".pt"):
        name = Path(rel).stem
        exp = "Track C C1c forecaster" if name.startswith("c1c_") else "Track C C1 forecaster"
        return "forecaster", exp, name, seed, name
    if rel.endswith(".onnx"):
        return "onnx", "Track C C1 export", Path(rel).stem, seed, Path(rel).stem
    if rel.endswith(".pkl"):
        split = "TEST" if "__test" in rel else "VAL"
        return "detection_cache", "detector inference cache", f"{split} cache", seed, Path(rel).stem.split("__")[0]
    if rel.endswith(".npz"):
        return "array", "run outputs (npz)", "", seed, rel.split("/")[2] if rel.count("/") > 2 else ""
    if rel.endswith(".pt"):
        return "checkpoint", "base / reference weights", Path(rel).name, seed, ""
    return "file", rel.split("/")[0] if not rel.startswith("source/") else "/".join(rel.split("/")[:2]), "", "", ""


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(rec, out=None):
    rec = Path(rec).resolve()
    out = Path(out).resolve() if out else rec          # --out: write the outputs elsewhere (read-only check of the backup)
    out.mkdir(parents=True, exist_ok=True)
    root = rec / "aerotrack_server"
    ver = rec / "_verification"
    remote = {}
    for name in ("remote_sha_weights.txt", "remote_sha_other.txt"):
        for line in (ver / name).read_text(encoding="utf-8", errors="replace").splitlines():
            h, _, p = line.partition("  ")
            remote[p.removeprefix("./")] = h
    rows, stats = [], {"verified": 0, "mismatch": 0, "missing": 0, "bytes_verified": 0}
    by_hash = {}
    for rel, rh in sorted(remote.items()):
        lp = root / rel
        if not lp.is_file():
            status, lh, size = "MISSING_LOCAL", "", ""
            stats["missing"] += 1
        else:
            size = lp.stat().st_size
            lh = sha(lp)
            status = "VERIFIED_SHA256" if lh == rh else "SHA_MISMATCH"
            stats["verified" if lh == rh else "mismatch"] += 1
            if lh == rh:
                stats["bytes_verified"] += size
        is_model = Path(rel).suffix in WEIGHT_EXT
        a_type, exp, role, seed, run = role_of(rel)
        by_hash.setdefault(rh, []).append(rel)
        if is_model:
            rows.append(dict(artifact_type=a_type, experiment=exp, role=role, seed=seed, run_id=run,
                             remote_path=f"~/aerotrack_server/{rel}", local_path=str(lp.relative_to(REPO)).replace("\\", "/"),
                             size_bytes=size, sha256_remote=rh, sha256_local=lh, status=status, notes=""))
    # registered-hash cross-check
    reg = []
    for r in REGISTRIES:
        txt = (REPO / r).read_text(encoding="utf-8")
        for h in sorted(set(re.findall(r"\b[0-9a-f]{64}\b", txt))):
            hit = [p for p in by_hash.get(h, []) if Path(p).suffix in WEIGHT_EXT]
            if hit:
                reg.append({"registry": r, "sha256": h, "backup_files": hit})
                for row in rows:
                    if row["sha256_remote"] == h and r not in row["notes"]:
                        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + f"hash registered in {r}"
    with open(out / "ARTIFACT_MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "ALL_FILES_SHA256_VERIFIED.txt", "w", encoding="utf-8") as f:
        for rel, rh in sorted(remote.items()):
            f.write(f"{rh}  aerotrack_server/{rel}\n")
    stats.update(model_rows=len(rows), model_rows_verified=sum(r["status"] == "VERIFIED_SHA256" for r in rows),
                 files_listed=len(remote), registered_hash_matches=reg)
    ((out / "_verification") if out != rec else ver).mkdir(parents=True, exist_ok=True)
    ((out / "_verification") if out != rec else ver).joinpath("verification_summary.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "registered_hash_matches"}, indent=1))
    print("registered hashes found in backup:", len(reg))
    return 0 if stats["mismatch"] == 0 and stats["missing"] == 0 else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("recovery_dir")
    ap.add_argument("--out", default=None, help="write ARTIFACT_MANIFEST.csv / ALL_FILES_SHA256_VERIFIED.txt / verification_summary.json here instead of into the backup")
    a = ap.parse_args()
    sys.exit(main(a.recovery_dir, a.out))
