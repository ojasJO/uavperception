#!/usr/bin/env python
"""Generate docs/EXPERIMENT_LEDGER.jsonl + .csv (one row per training run and per evaluation) from machine-readable sources.

Sources (never edited by this script):
  artifacts/experiment_registry.json   (tools/build_registry.py: OMEN-era runs and evaluations)
  <run_dir>/run_meta.json              (start / end times, attempts, host, precision when recorded)
  experiments/hpc/jobs/<run_id>/       (HPC job records copied back by tools/hpc/pull_results.sh: events.jsonl, env.json, cmd.txt)
Re-run after every pull; the ledger is a view, the sources are the evidence.
    venv\\Scripts\\python.exe tools\\build_master_ledger.py
"""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "artifacts" / "experiment_registry.json"
HPC_JOBS = REPO / "experiments" / "hpc" / "jobs"
OUT_JSONL = REPO / "docs" / "EXPERIMENT_LEDGER.jsonl"
OUT_CSV = REPO / "docs" / "EXPERIMENT_LEDGER.csv"
FIELDS = ["kind", "run_id", "track", "experiment", "seed", "status", "host", "gpu", "environment", "git_commit", "git_dirty",
          "dataset_hash", "split_hash", "config", "config_hash", "start", "end", "checkpoint", "checkpoint_sha256", "split",
          "val_metrics", "test_metrics", "test_ledger_line", "notes", "source"]


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def run_meta_times(run_dir):
    m = load(REPO / run_dir / "run_meta.json") if run_dir else None
    if not m:
        return None, None, None, []
    att = m.get("attempts") or []
    start = (att[0].get("timestamp_start") if att else None) or m.get("timestamp_start")
    end = m.get("timestamp_end") or (att[-1].get("timestamp_end") if att else None)
    restarts = [{"start": a.get("timestamp_start"), "resumed_from": a.get("resumed_from"), "rc": a.get("return_code")}
                for a in att]
    return start, end, m.get("host") or (att[0].get("env", {}).get("platform") if att else None), restarts


def main():
    reg = load(REG) or {"runs": [], "evaluations": []}
    rows = []
    for r in reg["runs"]:
        start, end, host, restarts = run_meta_times(r.get("run_dir"))
        env = r.get("environment") or {}
        rows.append({
            "kind": "train", "run_id": r["id"], "track": r.get("track"), "experiment": r.get("series") or r.get("phase"),
            "seed": r.get("seed"), "status": r.get("status"), "host": host or env.get("platform"), "gpu": r.get("hardware"),
            "environment": {k: env.get(k) for k in ("python", "torch", "cuda_version")}, "git_commit": r.get("git_commit"),
            "git_dirty": r.get("git_dirty"), "dataset_hash": r.get("dataset_hash") or r.get("corpus"), "split_hash": r.get("split_hash"),
            "config": r.get("config"), "config_hash": r.get("config_hash"), "start": start, "end": end,
            "checkpoint": r.get("checkpoint"), "checkpoint_sha256": r.get("checkpoint_sha256"), "split": "train",
            "val_metrics": None, "test_metrics": None, "test_ledger_line": None,
            "notes": "; ".join(x for x in [r.get("note"), f"attempts={len(restarts)}" if len(restarts) > 1 else None,
                                            "smoke" if r.get("smoke") else None] if x),
            "source": "artifacts/experiment_registry.json (runs)"})
    for e in reg["evaluations"]:
        test = bool(e.get("test"))
        rows.append({
            "kind": "eval", "run_id": e["id"], "track": e.get("track"), "experiment": e.get("class"), "seed": None,
            "status": "completed", "host": None, "gpu": None, "environment": None, "git_commit": None, "git_dirty": None,
            "dataset_hash": None, "split_hash": None, "config": e.get("run"), "config_hash": None, "start": None, "end": None,
            "checkpoint": e.get("run"), "checkpoint_sha256": e.get("checkpoint_sha256"), "split": e.get("split"),
            "val_metrics": None if test else e.get("metrics"), "test_metrics": e.get("metrics") if test else None,
            "test_ledger_line": e.get("test_ledger_line"), "notes": "", "source": e.get("source")})
    if HPC_JOBS.exists():
        for d in sorted(p for p in HPC_JOBS.iterdir() if p.is_dir()):
            env = load(d / "env.json") or {}
            evs = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()] if (d / "events.jsonl").exists() else []
            sub = next((x for x in evs if x.get("kind") == "submitted"), {})
            starts = [x for x in evs if x.get("kind") == "start"]
            ends = [x for x in evs if x.get("kind") == "end"]
            status = ("completed" if (d / "DONE").exists() else "failed" if (d / "FAILED").exists() else
                      "gpu_busy" if (d / "GPU_BUSY").exists() else "running" if (d / "RUNNING").exists() else "unknown")
            rows.append({
                "kind": "hpc_job", "run_id": d.name, "track": None, "experiment": None, "seed": None, "status": status,
                "host": env.get("host") or sub.get("host"), "gpu": env.get("gpu"),
                "environment": {k: env.get(k) for k in ("python", "torch", "cuda_build")}, "git_commit": env.get("git_commit") or sub.get("commit"),
                "git_dirty": env.get("git_dirty"), "dataset_hash": None, "split_hash": None,
                "config": (d / "cmd.txt").read_text().strip() if (d / "cmd.txt").exists() else None, "config_hash": None,
                "start": starts[0]["t"] if starts else None, "end": ends[-1]["t"] if ends else None, "checkpoint": None,
                "checkpoint_sha256": None, "split": None, "val_metrics": None, "test_metrics": None, "test_ledger_line": None,
                "notes": f"pbs={sub.get('pbs_job') or env.get('pbs_job')}; attempts={len(starts)}; precision={(env.get('env') or {}).get('AEROTRACK_AMP')}",
                "source": str(d.relative_to(REPO)).replace("\\", "/")})
    OUT_JSONL.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    print(f"wrote {OUT_JSONL.name} / {OUT_CSV.name}: {len(rows)} rows "
          f"({sum(r['kind'] == 'train' for r in rows)} train, {sum(r['kind'] == 'eval' for r in rows)} eval, "
          f"{sum(r['kind'] == 'hpc_job' for r in rows)} hpc jobs)")


if __name__ == "__main__":
    main()
