"""Read-only artifact / environment health checks. Nothing here deletes, moves or rewrites an artifact.

checks() is fast (stat calls, no hashing). verify_model() re-hashes one checkpoint. recovery_check() runs the existing
scripts/hpc/recovery_manifest.py with --out under results/runs/studio/health/ so the backup itself is only read.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .hashing import sha256_file
from .paths import PYTHON, RECOVERY, REGISTRY, REPO, STUDIO_RUNS
from .test_lock import consumed_registrations


def environment() -> dict:
    out = {"python": sys.version.split()[0], "executable": sys.executable, "packages": {}}
    for m in ("torch", "torchvision", "ultralytics", "onnxruntime", "tensorrt", "streamlit", "numpy", "pandas", "cv2", "yaml", "jsonschema", "altair"):
        spec = importlib.util.find_spec(m)
        if spec is None:
            out["packages"][m] = None
            continue
        try:
            mod = importlib.import_module(m)
            out["packages"][m] = getattr(mod, "__version__", "installed")
        except Exception as e:  # noqa: BLE001
            out["packages"][m] = f"import error: {type(e).__name__}"
    try:
        import torch
        out["cuda_available"] = torch.cuda.is_available()
        out["cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_mem_free_gb"], out["gpu_mem_total_gb"] = round(free / 2**30, 2), round(total / 2**30, 2)
    except Exception as e:  # noqa: BLE001
        out["cuda_available"] = False
        out["cuda_error"] = str(e)[:200]
    du = shutil.disk_usage(REPO)
    out["disk_free_gb"], out["disk_total_gb"] = round(du.free / 1e9, 1), round(du.total / 1e9, 1)
    return out


def checks(R) -> dict:
    """-> {'items': [{check, status ok|warn|fail, detail, rows}], 'counts'}"""
    items = []

    def add(check, status, detail, rows=None):
        items.append({"check": check, "status": status, "detail": detail, "rows": rows or []})
    # registry freshness
    gen = R.models_doc.get("generated")
    newest = 0.0
    for m in R.models:
        for loc in m["locations"]:
            p = REPO / loc["path"]
            if loc["location"] == "workspace" and p.exists():
                newest = max(newest, p.stat().st_mtime)
    reg_t = (REGISTRY / "models.json").stat().st_mtime
    add("registry freshness", "ok" if newest <= reg_t else "warn",
        f"registry generated {gen}" + ("" if newest <= reg_t else " -- a checkpoint changed after the build: rerun scripts/build_workspace_registry.py"))
    # missing files
    missing = [{"model_id": m["model_id"], "path": loc["path"], "location": loc["location"]}
               for m in R.models for loc in m["locations"] if not (REPO / loc["path"]).exists()]
    add("registered files present", "ok" if not missing else "fail", f"{len(missing)} registered file(s) missing", missing)
    # size changes since the build (cheap proxy for modification; full proof = re-hash)
    changed = []
    for m in R.models:
        loc = next((x for x in m["locations"] if x["path"] == m["primary_path"]), None)
        p = REPO / m["primary_path"]
        if loc and p.exists() and p.stat().st_size != m["size_bytes"] and loc["location"] == "workspace":
            changed.append({"model_id": m["model_id"], "path": m["primary_path"], "registered_size": m["size_bytes"], "size_now": p.stat().st_size})
    add("checkpoint sizes unchanged", "ok" if not changed else "fail", f"{len(changed)} size change(s) since the registry build", changed)
    mism = [{"model_id": m["model_id"], "records": m.get("recorded_hashes")} for m in R.models if m["integrity_status"] == "mismatch"]
    add("SHA-256 vs independent records", "ok" if not mism else "fail", f"{len(mism)} mismatch(es); "
        f"{sum(m['integrity_status'] == 'verified_against_record' for m in R.models)} verified, "
        f"{sum(m['integrity_status'] == 'hashed_no_independent_record' for m in R.models)} hashed without an independent record", mism)
    add("duplicate content", "ok", f"{len(R.duplicates)} groups of byte-identical files (kept; listed, never deleted)",
        [{"sha256": d["sha256"][:16], "paths": ", ".join(d["logical_paths"])} for d in R.duplicates])
    backup_only = [{"model_id": m["model_id"], "path": m["primary_path"]} for m in R.models if not m.get("available_in_workspace")]
    add("models only in the recovery backup", "warn" if backup_only else "ok",
        f"{len(backup_only)} model(s) exist only in the backup (browsable; not runnable until restored with scripts/restore_from_recovery.py)", backup_only)
    nocfg = [{"model_id": m["model_id"]} for m in R.models if (m.get("config") or {}).get("path") == "not_recorded" and m["family_id"] not in
             ("shared.pretrained.yolo11", "aerotrack.deploy", "shared.globaltrack_pretrained")]
    add("configuration recorded", "ok" if not nocfg else "warn", f"{len(nocfg)} model(s) without a run config next to the checkpoint", nocfg)
    ds = [{"dataset": d["id"], "root": k, "exists": v} for d in R.datasets for k, v in d["status"]["roots"].items() if not v]
    add("dataset roots", "ok" if not ds else "warn", f"{len(ds)} missing dataset root(s)", ds)
    ver = R.models
    unver = [m["model_id"] for m in ver if m.get("adapter") and m["verification"]["load"] != "passed" and m["status"] not in ("smoke", "invalid")]
    add("adapter load verification", "ok" if not unver else "warn",
        f"{len(unver)} runnable-family model(s) not load-verified (run scripts/verify_adapters.py)", [{"model_id": x} for x in unver])
    bad = [{"experiment_id": e["experiment_id"], "status": e.get("status"), "error": (e.get("warnings") or [""])[-1][:200]}
           for e in R.studio_experiments() if e.get("status") != "completed"]
    add("studio evaluations complete", "ok" if not bad else "warn", f"{len(bad)} studio evaluation(s) failed or partial", bad)
    locks = consumed_registrations()
    add("TEST ledgers present", "ok" if all(x["exists"] for x in locks) else "warn",
        "; ".join(f"{x['registration']}: {x['n_lines']} line(s)" + ("" if x["exists"] else " (ledger missing)") for x in locks))
    counts = {s: sum(i["status"] == s for i in items) for s in ("ok", "warn", "fail")}
    return {"items": items, "counts": counts}


def verify_model(m) -> dict:
    out = []
    for loc in m["locations"]:
        p = REPO / loc["path"]
        if not p.exists():
            out.append({"path": loc["path"], "status": "missing"})
            continue
        h = sha256_file(p)
        out.append({"path": loc["path"], "status": "match" if h == m["sha256"] else "MISMATCH", "sha256": h})
    return {"model_id": m["model_id"], "registered_sha256": m["sha256"], "locations": out}


def recovery_check(timeout=1800) -> dict:
    """Re-hash the whole backup against the HPC-side hashes (about 45 s for 17 GB); outputs under results/runs/studio/health/."""
    out = STUDIO_RUNS / "health" / "recovery_check"
    r = subprocess.run([str(PYTHON), "-I", str(REPO / "scripts" / "hpc" / "recovery_manifest.py"), str(RECOVERY), "--out", str(out)],
                       cwd=REPO, capture_output=True, text=True, timeout=timeout)
    summ = out / "_verification" / "verification_summary.json"
    return {"returncode": r.returncode, "stdout": r.stdout[-3000:], "stderr": r.stderr[-2000:],
            "summary": json.loads(summ.read_text(encoding="utf-8")) if summ.exists() else None, "outputs": str(out.relative_to(REPO))}
