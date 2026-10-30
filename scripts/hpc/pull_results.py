"""Copy HPC results back to the OMEN (small files only; never weights), safely.

    venv\\Scripts\\python.exe scripts\\hpc\\pull_results.py [--dry-run]

Groups (remote path under ~/aerotrack_server -> local path):
  jobs/                                         -> experiments/hpc/jobs/            (job records)
  reports/hpc/                                  -> experiments/hpc/reports/         (gate / verification reports)
  docs/environments/at311_freeze.txt                 -> experiments/hpc/env/
  source/experiments/aerotrack_v2/{reports,ledger.jsonl}  -> experiments/aerotrack_v2/hpc/   (namespaced: amendment A1)
  source/runs/<run>/{run_meta,args,init_check}.json, results.csv, *.png  -> results/runs/hpc/<run>/   (v2 / v3 / C2 / omen_* runs)
  source/runs/aerotrack_v2/_cache/*.pkl         -> results/runs/hpc/aerotrack_v2/_cache/
  source/experiments/anti_uav/{runs,reports}    -> SAME paths (Track A continues on the HPC): a file that exists on the OMEN
                                                   may only be identical or an append-only extension (ledgers); anything else aborts.
"""
import argparse
import io
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GROUPS = [
    ("jobs", REPO / "experiments" / "hpc" / "jobs", ["--exclude=*.pt"], "namespaced"),
    ("reports/hpc", REPO / "experiments" / "hpc" / "reports", [], "namespaced"),
    ("docs/environments/at311_freeze.txt", REPO / "experiments" / "hpc" / "env", [], "namespaced"),
    ("source/experiments/aerotrack_v2/reports", REPO / "experiments" / "aerotrack_v2" / "hpc" / "reports", [], "namespaced"),
    ("source/experiments/aerotrack_v2/ledger.jsonl", REPO / "experiments" / "aerotrack_v2" / "hpc", [], "namespaced"),
    ("source/runs/aerotrack_v2/_cache", REPO / "results/runs" / "hpc" / "aerotrack_v2" / "_cache", [], "namespaced"),
    ("source/experiments/systems/reports", REPO / "experiments" / "systems", [], "inplace"),
    ("source/experiments/aerotrack_v2/frozen_ste_params.json", REPO / "experiments" / "aerotrack_v2", [], "inplace"),
    ("source/experiments/aerotrack_temporal/reports", REPO / "experiments" / "aerotrack_temporal", [], "inplace"),
    ("source/runs/aerotrack_temporal/c1/learned_registry_c1c.json", REPO / "results/runs" / "aerotrack_temporal" / "c1", [], "inplace"),
    ("source/experiments/aerotrack_v3/reports", REPO / "experiments" / "aerotrack_v3", [], "inplace"),
    ("source/experiments/anti_uav/runs", REPO / "experiments" / "anti_uav", ["--exclude=*.pt", "--exclude=predictions"], "inplace"),
    ("source/experiments/anti_uav/reports", REPO / "experiments" / "anti_uav", [], "inplace"),
]
RUN_GLOBS = ["v2_*", "v3_*", "x1_*", "f1_*", "m6bench_*"]   # omen_* / gen_* are OMEN runs re-hosted for evaluation
RUN_FILES = ["run_meta.json", "args.json", "init_check.json", "results.csv", "training_curves.png"]


def remote_tar(path, extra):
    parent, name = path.rsplit("/", 1) if "/" in path else ("", path)
    cmd = (f"cd ~/aerotrack_server/{parent} 2>/dev/null && [ -e {name} ] && tar -cf - {' '.join(extra)} {name} || true")
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "amritahpc", cmd], capture_output=True, timeout=1800)
    return out.stdout


def _lf(b):
    return b.replace(b"\r\n", b"\n")


OMEN_BASELINE = "4404e20"     # last commit before any HPC result (OMEN-era evidence, 2026-10-06)


def _tracked(path):
    """Protected = existed in the OMEN-era evidence (commit OMEN_BASELINE): such a file may only stay identical or grow
    append-only (ledgers). Files created later came from the HPC (possibly pulled mid-write) and are refreshed."""
    rel = Path(path).resolve().relative_to(REPO).as_posix()
    return subprocess.run(["git", "cat-file", "-e", f"{OMEN_BASELINE}:{rel}"], cwd=REPO, capture_output=True).returncode == 0


def apply(blob, dest, mode, dry):
    if not blob:
        return 0, []
    n, problems = 0, []
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            data = tf.extractfile(m).read()
            rel = m.name.split("/", 1)[1] if (mode == "namespaced" and "/" in m.name) else m.name
            target = dest / rel
            if target.exists():
                old = _lf(target.read_bytes())          # OMEN working copies are CRLF (core.autocrlf)
                if old == _lf(data):
                    continue
                if mode == "inplace" and not _lf(data).startswith(old) and _tracked(target):
                    problems.append(f"{target}: OMEN file differs and is not a prefix of the HPC file")
                    continue
            n += 1
            if not dry:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    return n, problems


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    total, problems = 0, []
    for path, dest, extra, mode in GROUPS:
        n, p = apply(remote_tar(path, extra), dest, mode, a.dry_run)
        total += n; problems += p
        print(f"{path:50s} -> {dest.relative_to(REPO)}: {n} files")
    inc = " ".join(f"--include='{g}/{f}'" for g in RUN_GLOBS for f in RUN_FILES)
    cmd = ("cd ~/aerotrack_server/source/runs && ls -d " + " ".join(RUN_GLOBS) + " 2>/dev/null | while read d; do "
           "for f in " + " ".join(RUN_FILES) + "; do [ -f \"$d/$f\" ] && echo \"$d/$f\"; done; done | tar -cf - -T - 2>/dev/null || true")
    blob = subprocess.run(["ssh", "-o", "BatchMode=yes", "amritahpc", cmd], capture_output=True, timeout=1800).stdout
    n, p = apply(blob, REPO / "results/runs" / "hpc", "overwrite", a.dry_run)   # HPC-owned namespace: always refreshed
    total += n; problems += p
    print(f"{'source/runs/<run>/<meta files>':50s} -> results/runs/hpc: {n} files")
    for x in problems:
        print("PROBLEM", x)
    print(f"{'(dry run) ' if a.dry_run else ''}{total} files {'would be ' if a.dry_run else ''}written; {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
