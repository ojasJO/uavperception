"""Copy (never move) selected artifacts from the verified HPC recovery backup back to their original repo-relative paths.

    venv\\Scripts\\python.exe scripts\\restore_from_recovery.py            # dry run: prints the plan
    venv\\Scripts\\python.exe scripts\\restore_from_recovery.py --apply    # copies, verifies, appends to the migration ledger

Rules (docs/workspace/ARCHITECTURE.md §artifact paths):
* the backup is read-only: files are copied out, nothing in results/artifacts/hpc_recovery_* is opened for writing;
* a destination that already exists is never overwritten: identical content -> `already_present_identical`, different content ->
  `conflict_kept_local` (reported, nothing copied);
* the pre-existing untracked directories (results/runs/_deploy, results/runs/tmp_wheels, results/runs/v2_causal_g4_s0_e20) are never written to;
* every copied file is re-hashed and must equal the SHA-256 recorded in the backup's ALL_FILES_SHA256_VERIFIED.txt.
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECOVERY = REPO / "results/artifacts" / "hpc_recovery_20261013"
LEDGER = REPO / "results/registry" / "migrations" / "restore_manifest.json"
PROTECTED = ("results/runs/_deploy/", "results/runs/tmp_wheels/", "results/runs/v2_causal_g4_s0_e20/")

# backup path prefix (relative to aerotrack_server/source/) -> why it is restored
SELECTION = {
    **{f"results/runs/{d}/": "HPC-trained detector run present only in the backup" for d in (
        "laptop_v2_stock_s0_e20", "v2_causal_g1_s0_e20", "v2_causal_g1_s1_e20", "v2_causal_g2_s0_e20", "v2_causal_g2_s1_e20",
        "v2_causal_g4_s1_e20", "v2_causal_g8_s0_e20", "v2_causal_g8_s1_e20", "v2_stock_s1_e20", "v2_sym_g4_s1_e20",
        "v3_n__causal_g4_cst_s0_e20", "v3_n__causal_g4_cst_s1_e20", "v3_n__causal_g4_cst_s2_e20",
        "v3_n__ttt_cst_s0_e20", "v3_n__ttt_cst_s1_e20", "v3_n__ttt_cst_s2_e20",
        "v3_n_std__causal_g4_s0_e20", "v3_n_std__causal_g4_s1_e20")},
    "results/runs/aerotrack_temporal/c1/c1c_": "Track C C1c forecaster present only in the backup",
    "results/runs/aerotrack_temporal/c1/onnx/": "Track C ONNX export",
    "experiments/anti_uav/runs/normal_rgb/weights/last.pt": "Direction B visible-modality final checkpoint (only in the backup)",
    "experiments/anti_uav/runs/dfsc_rgb/weights/last.pt": "Direction B visible-modality final checkpoint (only in the backup)",
    "experiments/anti_uav/runs/largescale_globaltrack/weights/last.pt": "GlobalTrack large-scale checkpoint (only in the backup)",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def plan():
    rows = []
    listing = RECOVERY / "ALL_FILES_SHA256_VERIFIED.txt"
    for line in listing.read_text(encoding="utf-8").splitlines():
        h, _, p = line.partition("  ")
        if not p.startswith("aerotrack_server/source/"):
            continue
        rel = p[len("aerotrack_server/source/"):]
        why = next((w for pre, w in SELECTION.items() if rel.startswith(pre)), None)
        if why is None or rel.startswith(PROTECTED):
            continue
        src, dst = RECOVERY / p, REPO / rel
        if dst.exists():
            status = "already_present_identical" if sha256(dst) == h else "conflict_kept_local"
        else:
            status = "to_copy"
        rows.append({"rel": rel, "src": str(src.relative_to(REPO)).replace("\\", "/"), "sha256": h, "bytes": src.stat().st_size,
                     "reason": why, "status": status})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    rows = plan()
    todo = [r for r in rows if r["status"] == "to_copy"]
    print(f"{len(rows)} selected files: {len(todo)} to copy ({sum(r['bytes'] for r in todo) / 1e9:.2f} GB), "
          f"{sum(r['status'] == 'already_present_identical' for r in rows)} already present, "
          f"{sum(r['status'] == 'conflict_kept_local' for r in rows)} conflicts (kept local)")
    if not a.apply:
        for r in rows[:15]:
            print(" ", r["status"], r["rel"])
        return 0
    failures = 0
    for r in todo:
        dst = REPO / r["rel"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".restoring")
        shutil.copyfile(REPO / r["src"], tmp)
        if sha256(tmp) != r["sha256"]:
            tmp.unlink()
            r["status"] = "copy_failed_hash"
            failures += 1
            continue
        tmp.replace(dst)
        r["status"] = "restored_verified"
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"events": []}
    old["events"].append({"at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "tool": "scripts/restore_from_recovery.py",
                          "source": "results/artifacts/hpc_recovery_20261013 (read-only)", "files": rows})
    LEDGER.write_text(json.dumps(old, indent=1), encoding="utf-8")
    print(f"restored {sum(r['status'] == 'restored_verified' for r in rows)}; failures {failures}; ledger {LEDGER.relative_to(REPO)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
