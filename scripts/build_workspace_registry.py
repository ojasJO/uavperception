"""Build the unified workspace registry (results/registry/*.json, *.csv, *.jsonl) from the files on disk + the curated catalog.

    venv\\Scripts\\python.exe scripts\\build_workspace_registry.py            # incremental (hash cache)
    venv\\Scripts\\python.exe scripts\\build_workspace_registry.py --rehash   # re-hash every workspace file

Read-only on everything except results/registry/. Every number in the outputs is copied from a result file named in its record
(`source_file`); nothing is typed by hand. Inputs:
  results/registry/catalog/{directions,papers,families,protocols,datasets}.json   (hand-curated provenance, no metrics)
  model files under results/runs/, experiments/, src/deployment/, src/model/, results/artifacts/deploy/  (hashed, cached)
  results/artifacts/hpc_recovery_20261013/ALL_FILES_SHA256_VERIFIED.txt           (backup copies; hashes verified 2026-10-13)
  recorded hashes: run_meta.json, frozen configs, learned registries, export records, ledgers, the backup manifest
  results: docs/EXPERIMENT_LEDGER.jsonl + CST / CF / P1 / Track C / paper-reported report files
Outputs:
  results/registry/models.json, results/registry/checkpoints.csv, results/registry/model_lineage.json, results/registry/duplicates.json,
  results/registry/datasets.json, results/registry/experiments.jsonl, results/registry/build_summary.json
"""
import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from aerolab import schemas  # noqa: E402
from aerolab.hashing import HashCache  # noqa: E402
from aerolab.paths import CATALOG, RECOVERY, REGISTRY  # noqa: E402

MODEL_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine", ".bin", ".joblib"}
CACHE_EXT = {".pkl"}
SCAN_ROOTS = ["results/runs", "experiments", "src/deployment", "src/model", "results/artifacts/deploy", "checkpoints"]
SKIP_DIRS = {"venv", "data", "datasets", ".git", "__pycache__", "tmp_wheels", "data_unified", "data_unified_v2", "node_modules"}
VERIFICATION_FILE = REGISTRY / "verification" / "adapter_checks.json"
WARN = []


def jload(p, default=None):
    try:
        return json.loads((REPO / p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except OSError:
        return None


# ----------------------------------------------------------------------------------------------- files
def scan_workspace(cache, rehash):
    rows = []
    for root in SCAN_ROOTS:
        base = REPO / root
        if not base.exists():
            continue
        stack = [base]
        while stack:
            d = stack.pop()
            try:
                entries = list(d.iterdir())
            except OSError:
                continue
            for e in entries:
                if e.is_dir():
                    if e.name in SKIP_DIRS or e.is_symlink():
                        continue
                    stack.append(e)
                elif e.suffix.lower() in MODEL_EXT | CACHE_EXT and e.is_file():
                    lp = e.relative_to(REPO).as_posix()
                    rows.append({"logical_path": lp, "physical_path": lp, "location": "workspace", "size_bytes": e.stat().st_size,
                                 "sha256": cache.get(e, verify=rehash)})
    return rows


def backup_logical(p):
    """aerotrack_server/source/X -> X ; aerotrack_server/checkpoints/X -> checkpoints/X ; other -> hpc/<rest>"""
    p = p[len("aerotrack_server/"):]
    if p.startswith("source/"):
        return p[len("source/"):]
    if p.startswith("checkpoints/"):
        return p
    return "hpc/" + p


def scan_backup():
    listing = RECOVERY / "ALL_FILES_SHA256_VERIFIED.txt"
    rows = []
    if not listing.exists():
        WARN.append("recovery listing missing: backup copies not registered")
        return rows
    for line in listing.read_text(encoding="utf-8").splitlines():
        h, _, p = line.partition("  ")
        if Path(p).suffix.lower() not in MODEL_EXT | CACHE_EXT:
            continue
        f = RECOVERY / p
        rows.append({"logical_path": backup_logical(p), "physical_path": f.relative_to(REPO).as_posix(), "location": "recovery_backup",
                     "size_bytes": f.stat().st_size if f.exists() else -1, "sha256": h, "backup_present": f.exists()})
    return rows


def classify(rows, families):
    comp = [(f, [re.compile(p) for p in f["patterns"]]) for f in families]
    for r in rows:
        lp = r["logical_path"]
        r.update(family_id=None, model_id=None, run_id=None, tag=None, seed=None)
        if Path(lp).suffix.lower() in CACHE_EXT:
            r["artifact_kind"] = "detection_cache"
            m = re.search(r"([^/]+?)__(val|test|train|dev)[^/]*\.pkl$", lp)
            r["run_id"] = m.group(1) if m else None
            r["cache_split"] = m.group(2) if m else None
            continue
        for fam, pats in comp:
            m = next((x for x in (p.match(lp) for p in pats) if x), None)
            if m:
                g = m.groupdict()
                run = g.get("run")
                tag = g.get("tag")
                if tag is None or tag == "":
                    tag = "onnx" if lp.endswith(".onnx") else "weights"
                r.update(family_id=fam["id"], run_id=run, tag=tag,
                         seed=int(g["seed"]) if g.get("seed") not in (None, "") else None,
                         artifact_kind="export" if fam["task"] == "export" or lp.endswith((".onnx", ".engine")) else "model_weights")
                r["model_id"] = f"{fam['id']}:{run}@{tag}"
                break
        else:
            r["artifact_kind"] = "unclassified"
    # same logical path, different bytes in the workspace and the backup = two different models sharing a run id
    # (e.g. a run id trained on the OMEN and re-trained on the HPC): the backup copy gets its own id '<id>~hpc'
    ws = {r["logical_path"]: r["sha256"] for r in rows if r["location"] == "workspace"}
    for r in rows:
        if r["location"] == "recovery_backup" and r["model_id"] and r["logical_path"] in ws and ws[r["logical_path"]] != r["sha256"]:
            r["model_id"] += "~hpc"
            r["variant_note"] = "HPC re-run of a run id that also exists on the OMEN; different weights (see the OMEN copy for the other model)"
    return rows


# ----------------------------------------------------------------------------------------------- recorded hashes
def recorded_hashes(ledger_rows):
    rec = defaultdict(list)          # logical path -> [(source, sha)]

    def add(path, sha, src):
        if path and isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha):
            rec[path].append({"source": src, "sha256": sha})
    for meta in list((REPO / "results/runs").glob("*/run_meta.json")):
        m = jload(meta.relative_to(REPO).as_posix(), {})
        run = meta.parent.name
        add(f"results/runs/{run}/weights/best.pt", m.get("best_pt_sha256"), f"results/runs/{run}/run_meta.json")
        add(f"results/runs/{run}/weights/last.pt", m.get("last_pt_sha256"), f"results/runs/{run}/run_meta.json")
    for meta in (REPO / "experiments" / "anti_uav" / "runs").glob("*/run_meta.json"):
        m = jload(meta.relative_to(REPO).as_posix(), {})
        add(f"experiments/anti_uav/runs/{meta.parent.name}/weights/last.pt", m.get("final_checkpoint_sha256"), meta.relative_to(REPO).as_posix())
    fx = jload("experiments/cst_anti_uav/frozen_x2_params.json", {})
    for run, v in (fx.get("runs") or {}).items():
        add(f"results/runs/{run}/weights/last.pt", v.get("weights_sha256"), "experiments/cst_anti_uav/frozen_x2_params.json")
    x1 = jload("experiments/aerotrack_v3/reports/x1_final_test.json", {})
    add(f"results/runs/{x1.get('candidate')}/weights/last.pt", x1.get("weights_sha256"), "experiments/aerotrack_v3/reports/x1_final_test.json")
    cf = jload("experiments/aerotrack_temporal/reports/CF_final_causal_system.json", {})
    det = cf.get("detector") or {}
    add(f"results/runs/{det.get('run')}/weights/last.pt", det.get("weights_sha256"), "experiments/aerotrack_temporal/reports/CF_final_causal_system.json")
    for reg in ("results/runs/aerotrack_temporal/c1/learned_registry.json", "results/runs/aerotrack_temporal/c1/learned_registry_c1c.json"):
        for name, v in (jload(reg, {}) or {}).items():
            if isinstance(v, dict):
                add(f"results/runs/aerotrack_temporal/c1/{name}.pt", v.get("sha256"), reg)
    for exp in (REPO / "results/artifacts" / "deploy").glob("*/export_record.json"):
        e = jload(exp.relative_to(REPO).as_posix(), {})
        run = exp.parent.name
        add(f"results/runs/{run}/weights/last.pt", e.get("weights_sha256"), exp.relative_to(REPO).as_posix())
    for ck in REPO.glob("src/deployment/*/checksums.sha256"):
        for line in ck.read_text(encoding="utf-8").splitlines():
            h, _, p = line.strip().partition("  ")
            add(f"{ck.parent.relative_to(REPO).as_posix()}/{p}", h, ck.relative_to(REPO).as_posix())
    for r in ledger_rows:
        c = r.get("checkpoint")
        if not c or not r.get("checkpoint_sha256"):
            continue
        p = c if c.endswith(".pt") else f"experiments/anti_uav/runs/{c}/weights/last.pt" if r.get("track") == "anti_uav" else None
        add(p.replace("\\", "/") if p else None, r["checkpoint_sha256"], "docs/EXPERIMENT_LEDGER.jsonl")
    return rec


# ----------------------------------------------------------------------------------------------- models
def run_dir_of(lp):
    m = re.match(r"^(results/runs/[^/]+|experiments/anti_uav/runs/[^/]+)/weights/", lp)
    return m.group(1) if m else None


def run_config(lp, cache):
    rd = run_dir_of(lp)
    if rd is None:
        if "/aerotrack_temporal/c1/" in lp:
            name = Path(lp).stem
            reg = "results/runs/aerotrack_temporal/c1/learned_registry_c1c.json" if name.startswith("c1c_") else "results/runs/aerotrack_temporal/c1/learned_registry.json"
            v = (jload(reg, {}) or {}).get(name, {})
            return {"path": reg + f"#{name}", "sha256": None, "summary": {k: v.get(k) for k in ("arch", "variant", "L", "H", "abs_pos", "dom_tok", "aug", "epochs", "best_epoch", "trained") if k in v}}
        return {"path": "not_recorded", "sha256": None, "summary": {}}
    for name in ("config.effective.yaml", "run_meta.json"):
        p = REPO / rd / name
        if p.exists():
            summ = {}
            if name == "run_meta.json":
                m = jload(f"{rd}/{name}", {})
                summ = {k: m.get(k) for k in ("window", "window_name", "gap", "temporal_gap", "train_domains", "epochs", "batch", "init", "config", "model", "series") if k in m}
                summ["corpus"] = (m.get("data") or {}).get("corpus")
                summ["pretrained_sha256"] = m.get("pretrained_sha256")
            return {"path": f"{rd}/{name}", "sha256": cache.get(p), "summary": summ}
    return {"path": "not_recorded", "sha256": None, "summary": {}}


def build_models(rows, families, catalog_roles, rec, cache, verification):
    fam = {f["id"]: f for f in families}
    by_model = defaultdict(list)
    for r in rows:
        if r["model_id"]:
            by_model[r["model_id"]].append(r)
    hash_to_models = defaultdict(set)
    for mid, rs in by_model.items():
        for r in rs:
            hash_to_models[r["sha256"]].add(mid)
    models = []
    for mid, rs in sorted(by_model.items()):
        f = fam[rs[0]["family_id"]]
        ws = [r for r in rs if r["location"] == "workspace"]
        primary = (ws or rs)[0]
        lp = primary["logical_path"]
        shas = {r["sha256"] for r in rs}
        run, tag = rs[0]["run_id"], rs[0]["tag"]
        hpc_variant = mid.endswith("~hpc")
        role_cfg = {} if hpc_variant else catalog_roles.get(f"{run}@{tag}", {})
        recs = [dict(x, match=(x["sha256"] == primary["sha256"])) for x in rec.get(lp, [])] if not hpc_variant else []
        if hpc_variant and run_dir_of(lp):
            bm = RECOVERY / "aerotrack_server" / "source" / run_dir_of(lp) / "run_meta.json"
            m_ = json.loads(bm.read_text(encoding="utf-8")) if bm.exists() else {}
            h_ = m_.get(f"{tag}_pt_sha256")
            if h_:
                recs.append({"source": bm.relative_to(REPO).as_posix(), "sha256": h_, "match": h_ == primary["sha256"]})
        # the backup listing is an independent (HPC-side) record for workspace copies
        for r in rs:
            if r["location"] == "recovery_backup" and primary["location"] == "workspace":
                recs.append({"source": "results/artifacts/hpc_recovery_20261013 (HPC-side SHA-256)", "sha256": r["sha256"], "match": r["sha256"] == primary["sha256"]})
        if len(shas) > 1 or any(not x["match"] for x in recs):
            integrity = "mismatch"
        elif any(x["match"] for x in recs):
            integrity = "verified_against_record"
        else:
            integrity = "hashed_no_independent_record"
        status = role_cfg.get("status") or f["default_status"]
        role = role_cfg.get("role") or (rs[0].get("variant_note") if hpc_variant else None)
        if "smoke" in (run or "") and not role_cfg:
            status, role = "smoke", "plumbing smoke run (1-3 epochs) -- not a model"
        if tag == "best" and f.get("selection_policy", "").startswith("last.pt") and not role:
            status, role = ("experimental" if status in ("validated", "registered") else status), "fitness-selected best.pt; NOT the evaluated checkpoint (last.pt policy)"
        if tag.startswith("epoch_") and not role:
            status, role = "experimental", "intermediate epoch checkpoint (never evaluated)"
        cfg = run_config(lp, cache) if not hpc_variant else {"path": f"{primary['physical_path'].rsplit('/weights/', 1)[0]}/run_meta.json", "sha256": None, "summary": {}}
        parent = None
        pre = (cfg["summary"] or {}).get("pretrained_sha256")
        if pre:
            parent = next(iter(sorted(hash_to_models.get(pre, []))), f"external:sha256:{pre[:12]}")
        elif f["direction"] == "antiuav_repro":
            parent = "external:torchvision_coco_fasterrcnn_r50_fpn_v1 (config model.init; spec substitution)"
        dup = [m for m in hash_to_models[primary["sha256"]] if m != mid]
        v = verification.get(mid, {})
        model = {
            "schema": "src/aerolab/model/1", "model_id": mid,
            "name": f"{f['name']} -- {run} [{tag}]", "direction": f["direction"], "family_id": f["id"], "run_id": run,
            "checkpoint_tag": tag, "version": f"{run}@{tag}", "seed": rs[0]["seed"] if rs[0]["seed"] is not None else (cfg["summary"] or {}).get("seed"),
            "status": status, "aliases": role_cfg.get("aliases", []), "role": role, "registrations": role_cfg.get("registrations", []),
            "sha256": primary["sha256"], "size_bytes": primary["size_bytes"],
            "locations": [{"path": r["physical_path"], "location": r["location"], "sha256": r["sha256"]} for r in rs],
            "primary_path": primary["physical_path"], "available_in_workspace": bool(ws),
            "integrity_status": integrity, "recorded_hashes": recs,
            "task": f["task"], "adapter": f.get("adapter"), "architecture": f["architecture"], "framework": f["framework"],
            "input_contract": f["input_contract"], "output_contract": f["output_contract"], "preprocessing": f["preprocessing"],
            "training_data": f["training_data"], "papers": f["papers"], "config": cfg, "parent": parent,
            "duplicate_group": primary["sha256"][:16] if dup else None, "same_bytes_as": sorted(dup),
            "verification": {"integrity": "passed" if integrity == "verified_against_record" else ("failed" if integrity == "mismatch" else "not_verified"),
                             "load": v.get("load", "not_verified"), "inference": v.get("inference", "not_verified"),
                             "evaluation": "not_verified"},
            "verification_evidence": v.get("evidence"), "evaluations": [], "family_verdict": f.get("verdict"),
        }
        # not applicable (refused by design, not merely unchecked): no adapter, smoke / invalid checkpoints, ONNX exports of
        # forecasters (the adapter loads the .pt of the same forecaster)
        if model["adapter"] is None or status in ("smoke", "invalid") or primary["physical_path"].endswith(".onnx"):
            for k in ("load", "inference"):
                if model["verification"][k] == "not_verified":
                    model["verification"][k] = "not_applicable"
        models.append(model)
    return models


# ----------------------------------------------------------------------------------------------- experiments
def model_lookup(models):
    by_sha = defaultdict(list)
    by_run = defaultdict(list)
    for m in models:
        by_sha[m["sha256"]].append(m["model_id"])
        by_run[m["run_id"]].append(m)
    def find(sha=None, run=None, tag="last"):
        if sha and by_sha.get(sha):
            ids = by_sha[sha]
            return sorted(ids, key=lambda x: (not x.endswith("@" + tag), x))[:1]
        if run and by_run.get(run):
            ms = sorted(by_run[run], key=lambda m: (m["checkpoint_tag"] != tag, m["model_id"]))
            return [ms[0]["model_id"]]
        return []
    return find


def rec(**kw):
    base = {"schema": "src/aerolab/experiment/1", "model_ids": [], "checkpoint_sha256": None, "dataset_id": None, "split": None, "modality": None,
            "protocol_id": None, "metrics": {}, "per_sequence": None, "seed": None, "git_commit": None, "status": "completed",
            "registration": None, "test_pass": False, "notes": None, "warnings": []}
    base.update(kw)
    return base


def import_ledger(ledger_rows, find):
    out = []
    seen = Counter()
    for r in ledger_rows:
        track = r.get("track")
        direction = "antiuav_repro" if track == "anti_uav" else "aerotrack"
        run = r.get("run_id", "")
        split = r.get("split")
        kind = {"train": "train", "hpc_job": "hpc_job"}.get(r["kind"], "registered_test_pass" if split == "test" else "evaluation")
        metrics = r.get("val_metrics") or r.get("test_metrics") or {}
        ds = proto = modality = None
        inferred = True
        per_seq = None
        if track == "anti_uav":
            ds, modality = "antiuav2021", ("RGB" if "rgb" in run.lower() else "IR")
            proto = "sot.mSA.antiuav_eq1.v1" if r["kind"] == "eval" else None
            src = r.get("source", "")
            if src.endswith("metrics.json"):
                m = jload(src, {})
                ps = m.get("per_sequence")
                if isinstance(ps, dict):
                    per_seq = {k: round(100 * v["SA"], 6) for k, v in ps.items() if isinstance(v, dict) and v.get("SA") is not None}
        elif run.startswith(("aerotrack_generalization/", "aerotrack_repair", "historical/")):
            ds, proto = "corpus_v1", "det.AP50.v1" if r["kind"] == "eval" else None
        elif run.startswith("aerotrack_v2/"):
            ds, proto = "corpus_v2", "det.AP50.v1" if r["kind"] == "eval" else None
        elif run.startswith("src/deployment/"):
            ds, proto = None, "deploy.latency.v1"
        elif run.startswith("systems/"):
            ds, proto = "corpus_v1", "sot.mSA.antiuav_eq1.v1"
        elif run.startswith("aerotrack_temporal/C1"):
            ds, proto = "trackc_trajectories", "forecast.ADE_NLL.trackc.v1"
        elif run.startswith("aerotrack_temporal/tracking"):
            ds, proto = "corpus_v1", "mot.HOTA.trackc.v1"
        elif run.startswith("aerotrack_temporal/"):
            ds = "trackc_trajectories"
        else:
            inferred = False
        name = run.split("/")[-1] if r["kind"] == "train" else None
        mids = find(sha=r.get("checkpoint_sha256"), run=name or (run.split("/")[1] if run.count("/") >= 1 else None))
        eid = f"ledger:{r['kind']}:{run}"
        seen[eid] += 1
        if seen[eid] > 1:
            eid += f"#{seen[eid]}"
        out.append(rec(experiment_id=eid, direction=direction, kind=kind, provenance="historical_import", model_ids=mids,
                       checkpoint_sha256=r.get("checkpoint_sha256"), dataset_id=ds, split=split, modality=modality, protocol_id=proto,
                       metrics=metrics, per_sequence=per_seq, seed=r.get("seed"), git_commit=r.get("git_commit"), status=r.get("status"),
                       test_pass=split == "test" and r["kind"] == "eval", source_file=r.get("source") or "docs/EXPERIMENT_LEDGER.jsonl",
                       notes=r.get("notes") or None, start=r.get("start"), end=r.get("end"), experiment=r.get("experiment"),
                       warnings=["dataset / protocol inferred from the ledger row's track and run id"] if inferred and ds else [],
                       ledger_row=r))
    return out


def import_cst(find):
    out = []
    x1 = jload("experiments/aerotrack_v3/reports/x1_final_test.json", {})
    x1o = jload("experiments/cst_anti_uav/reports/X1_test_official.json", {})
    for var, t in (x1o.get("trackers") or {}).items():
        out.append(rec(experiment_id=f"cst:x1:{var}", direction="aerotrack", kind="registered_test_pass", provenance="historical_import",
                       model_ids=find(sha=x1o.get("weights_sha256")), checkpoint_sha256=x1o.get("weights_sha256"), dataset_id="cst2025",
                       split="test", modality="IR", protocol_id="sot.mSA.cst_official.v1",
                       metrics={"mSA": t.get("official_mSA"), "mSA_ci95": t.get("ci95_mSA"), "success_auc": t.get("success_auc"),
                                "precision_20px": t.get("precision_20px")},
                       per_sequence=(x1.get("per_sequence_SA") or {}).get(var), seed=0,
                       registration="experiments/aerotrack_v3/preregistration_X1_cst_protocol_matched.json", test_pass=True,
                       source_file="experiments/cst_anti_uav/reports/X1_test_official.json", variant=var,
                       notes=f"AeroTrack-X1, tracker '{var}', single registered TEST pass PBS 93019; per-sequence SA from x1_final_test.json (Track A = official, |d| = 0)"))
    o = jload("experiments/cst_anti_uav/reports/X2_test_official/official_scores.json", {})
    fx = jload("experiments/cst_anti_uav/frozen_x2_params.json", {}).get("runs", {})
    for key, t in (o.get("trackers") or {}).items():
        run, var = key.rsplit("-", 1)
        arm = "temporal causal_g4" if "causal_g4" in run else "single-frame (ttt)"
        sha = (fx.get(run) or {}).get("weights_sha256")
        out.append(rec(experiment_id=f"cst:x2:{run}:{var}", direction="aerotrack", kind="registered_test_pass", provenance="historical_import",
                       model_ids=find(sha=sha, run=run), checkpoint_sha256=sha, dataset_id="cst2025", split="test", modality="IR",
                       protocol_id="sot.mSA.cst_official.v1",
                       metrics={k: t.get(k) for k in ("mSA", "success_auc", "precision_20px", "success_rate_iou_0_5")},
                       per_sequence=t.get("per_sequence_SA"), seed=(fx.get(run) or {}).get("seed"),
                       registration="experiments/cst_anti_uav/preregistration/X2_seeds_single_frame.json", test_pass=True,
                       source_file="experiments/cst_anti_uav/reports/X2_test_official/official_scores.json", variant=var,
                       notes=f"AeroTrack-X2 {arm}, tracker '{var}', PBS 93029" + (" (seed 0 = the X1 checkpoint and predictions)" if "cst_s0" in run and "causal" in run else "")))
        v = (fx.get(run) or {}).get("val_cross_fitted_mSA", {}).get(var)
        if v is not None and not any(e["experiment_id"] == f"cst:x2val:{run}:{var}" for e in out):
            out.append(rec(experiment_id=f"cst:x2val:{run}:{var}", direction="aerotrack", kind="evaluation", provenance="historical_import",
                           model_ids=find(sha=sha, run=run), checkpoint_sha256=sha, dataset_id="cst2025", split="val", modality="IR",
                           protocol_id="sot.mSA.antiuav_eq1.v1", metrics={"mSA_cross_fitted": v}, seed=(fx.get(run) or {}).get("seed"),
                           source_file="experiments/cst_anti_uav/frozen_x2_params.json", variant=var,
                           notes="CST VAL, tracker settings cross-fitted over the two VAL halves (selection evidence; VAL shares videos with TRAIN)"))
    base = jload("experiments/cst_anti_uav/reports/official_reproduction_test/official_scores.json", {})
    for name, t in (base.get("trackers") or {}).items():
        out.append(rec(experiment_id=f"cst:released:{name}", direction="shared", kind="released_baseline",
                       provenance="reproduced_from_released_predictions", model_ids=[f"external:cst_released:{name}"], dataset_id="cst2025",
                       split="test", modality="IR", protocol_id="sot.mSA.cst_official.v1",
                       metrics={k: t.get(k) for k in ("mSA", "success_auc", "precision_20px", "success_rate_iou_0_5")},
                       per_sequence=t.get("per_sequence_SA"), test_pass=True,
                       source_file="experiments/cst_anti_uav/reports/official_reproduction_test/official_scores.json",
                       notes=f"{name}: authors' released CST-trained predictions scored by the official toolkit (checkpoints not released; never reports absence)"))
    for name, v in ((x1.get("X1_a") or {}).get("published") or {}).items():
        out.append(rec(experiment_id=f"cst:paper:{name}", direction="shared", kind="paper_reported", provenance="paper_reported",
                       model_ids=[f"external:cst_released:{name}"], dataset_id="cst2025", split="test", modality="IR",
                       protocol_id="sot.mSA.cst_official.v1", metrics={"mSA": v}, test_pass=True,
                       source_file="experiments/aerotrack_v3/reports/x1_final_test.json", notes="value as printed in arXiv:2507.23473 Table 3"))
    return out


def import_antiuav_paper():
    out = []
    p = jload("experiments/anti_uav/reports/paper_reported_results.json", {})
    cols = p.get("attribute_columns", [])
    for split, rows in (p.get("table_V_protocol_II") or {}).items():
        for method, vals in rows.items():
            for col, mod in (("ALL_IR", "IR"), ("ALL_VIS", "RGB")):
                if col in cols and cols.index(col) < len(vals) and vals[cols.index(col)] is not None:
                    out.append(rec(experiment_id=f"antiuav:paper:{method}:{split}:{mod}", direction="antiuav_repro", kind="paper_reported",
                                   provenance="paper_reported", model_ids=[f"external:antiuav_paper:{method}"], dataset_id="antiuav2021", split=split,
                                   modality=mod, protocol_id="sot.mSA.antiuav_eq1.v1", metrics={"mSA": vals[cols.index(col)]},
                                   test_pass=split == "test", source_file="experiments/anti_uav/reports/paper_reported_results.json",
                                   notes=f"paper Table V, Protocol II, {method}, {col} (transcribed; the PAPER's number, not ours)"))
    return out


def import_cf(find):
    out = []
    cf = jload("experiments/aerotrack_temporal/reports/CF_final_causal_system.json", {})
    det = cf.get("detector") or {}
    dmids = find(sha=det.get("weights_sha256"), run=det.get("run"))
    dom = {"anti_uav_IR": ("antiuav2021", "IR", "sot.mSA.antiuav_eq1.v1"), "anti_uav_RGB": ("antiuav2021", "RGB", "sot.mSA.antiuav_eq1.v1"),
           "cst_IR": ("cst2025", "IR", "sot.mSA.antiuav_eq1.v1")}
    for d, v in (cf.get("sot") or {}).items():
        ds, mod, proto = dom.get(d, (None, None, None))
        s = v.get("summary") or {}
        out.append(rec(experiment_id=f"cf:test:sot:{d}", direction="aerotrack", kind="registered_test_pass", provenance="historical_import",
                       model_ids=dmids, checkpoint_sha256=det.get("weights_sha256"), dataset_id=ds, split="test", modality=mod, protocol_id=proto,
                       metrics={k: s.get(k) for k in ("mSA", "located_rate", "false_alarm_rate", "n_sequences")},
                       registration=cf.get("registration"), test_pass=True, variant=v.get("tracker"),
                       source_file="experiments/aerotrack_temporal/reports/CF_final_causal_system.json",
                       notes=f"CF final causal system, tracker {v.get('tracker')} {v.get('params')}; single registered pass (cf_test_ledger.jsonl)"))
    fdom = {"antiuav_IR": ("IR", "antiuav2021"), "antiuav_RGB": ("RGB", "antiuav2021"), "cst": ("IR", "cst2025")}
    for d, v in (cf.get("forecasting") or {}).items():
        for name, s in (v.get("summaries") or {}).items():
            mids = find(run=name, tag="weights") if name.startswith("c1") else []
            out.append(rec(experiment_id=f"cf:test:forecast:{d}:{name}", direction="aerotrack", kind="registered_test_pass",
                           provenance="historical_import", model_ids=mids, dataset_id=fdom.get(d, (None, None))[1], split="test",
                           modality=fdom.get(d, (None,))[0], protocol_id="forecast.ADE_NLL.trackc.v1",
                           metrics={k: s.get(k) for k in ("ade_px", "nll", "ade_rel_size", "miss_20px", "n_samples") if k in s},
                           registration=cf.get("registration"), test_pass=True, variant=name,
                           source_file="experiments/aerotrack_temporal/reports/CF_final_causal_system.json",
                           notes=f"CF forecasting on {v.get('tracker_histories')} tracker histories; forecaster {name}"))
    return out


def import_p1(find):
    out = []
    p = jload("experiments/cst_anti_uav/reports/explore/P1_presence_gate_VAL.json", {})
    for run, v in (p.get("runs") or {}).items():
        d = v.get("gated_minus_frozen_paired") or {}
        out.append(rec(experiment_id=f"cst:p1:{run}", direction="aerotrack", kind="exploratory_val", provenance="historical_import",
                       model_ids=find(run=run), dataset_id="cst2025", split="val", modality="IR", protocol_id="sot.mSA.antiuav_eq1.v1",
                       metrics={"frozen_mSA": v.get("baseline_cross_fitted_mSA"), "gated_mSA": v.get("gated_cross_fitted_mSA"),
                                "delta": d.get("mean_delta", d.get("mean")), "delta_ci95": d.get("ci95"), "trade_off": v.get("trade_off"),
                                "chosen_gate_per_half": v.get("chosen_gate_per_half")},
                       registration="experiments/cst_anti_uav/preregistration/P1_presence_gate_VAL.json", verdict=p.get("verdict"),
                       curve=v.get("curve_descriptive"), source_file="experiments/cst_anti_uav/reports/explore/P1_presence_gate_VAL.json",
                       notes=f"P1 presence gate (exploratory, VAL only); registered verdict {p.get('verdict')}"))
    return out


def import_trackc():
    out = []
    t = jload("experiments/aerotrack_temporal/reports/TRACK_C_SUMMARY.json", {})

    def walk(stage, scope, node, path):
        if isinstance(node, dict):
            if "verdict" in node:
                pair = (node.get("pairs") or [{}])[0] if isinstance(node.get("pairs"), list) else {}
                m = {k: node.get(k, pair.get(k)) for k in ("mean_delta", "diff", "ci95", "wins", "losses", "n", "n_sequences") if node.get(k, pair.get(k)) is not None}
                out.append(rec(experiment_id=f"trackc:{stage}:{scope}:{'/'.join(path)}", direction="aerotrack", kind="hypothesis_verdict",
                               provenance="historical_import", dataset_id="trackc_trajectories" if stage.startswith("C1") else "corpus_v2", split="val",
                               protocol_id="forecast.ADE_NLL.trackc.v1" if stage.startswith("C1") else "sot.mSA.antiuav_eq1.v1",
                               metrics=m, verdict=node["verdict"], source_file="experiments/aerotrack_temporal/reports/TRACK_C_SUMMARY.json",
                               notes=f"Track C {stage} ({scope}) {'/'.join(path)}: {node['verdict']} (VAL only)"))
                return
            for k, v in node.items():
                walk(stage, scope, v, path + [k])
    for stage in ("C1", "C1c", "C1b", "C3"):
        for scope, node in (t.get(stage) or {}).items():
            walk(stage, scope, node, [])
    return out


# ----------------------------------------------------------------------------------------------- datasets
def build_datasets(cat):
    out = []
    for d in cat["datasets"]:
        status = {"roots": {}, "splits": {}, "manifests": {}}
        for k, p in d.get("roots", {}).items():
            if k.endswith("pattern"):
                continue
            status["roots"][k] = (REPO / p).exists()
        counts = {}
        for s, p in d.get("splits", {}).items():
            f = REPO / p
            status["splits"][s] = f.exists()
            if f.is_dir():
                counts[s] = sum(1 for x in f.iterdir() if x.is_dir())
            elif f.suffix == ".json" and f.exists() and d["id"] == "trackc_trajectories":
                counts[s] = len((jload(p, {}) or {}).get("sequences", []))
        for p in d.get("manifests", []):
            f = REPO / p
            status["manifests"][p] = f.exists()
        out.append(dict(d, schema="src/aerolab/dataset/1", sequence_counts=counts, status=status))
    return out


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehash", action="store_true")
    a = ap.parse_args()
    t0 = dt.datetime.now().astimezone()
    families = jload("results/registry/catalog/families.json")
    cat_ds = jload("results/registry/catalog/datasets.json")
    ledger_rows = [json.loads(x) for x in (REPO / "docs" / "EXPERIMENT_LEDGER.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    verification = (jload("results/registry/verification/adapter_checks.json", {}) or {}).get("models", {})
    cache = HashCache()
    rows = scan_workspace(cache, a.rehash) + scan_backup()
    classify(rows, families["families"])
    rec_h = recorded_hashes(ledger_rows)
    models = build_models(rows, families["families"], families.get("run_roles", {}), rec_h, cache, verification)
    find = model_lookup(models)
    exps = (import_ledger(ledger_rows, find) + import_cst(find) + import_antiuav_paper() + import_cf(find) + import_p1(find) + import_trackc())
    by_model = defaultdict(list)
    for e in exps:
        for m in e["model_ids"]:
            by_model[m].append(e["experiment_id"])
    for m in models:
        m["evaluations"] = sorted(set(by_model.get(m["model_id"], [])))
    # duplicates (identical bytes at different logical paths)
    groups = defaultdict(set)
    for r in rows:
        groups[r["sha256"]].add(r["logical_path"])
    dups = [{"sha256": h, "logical_paths": sorted(p), "model_ids": sorted({r["model_id"] for r in rows if r["sha256"] == h and r["model_id"]}),
             "note": "identical bytes; NOT deleted. Different logical paths may still carry different roles."}
            for h, p in groups.items() if len(p) > 1]
    lineage = {"schema": "src/aerolab/lineage/1", "families": {}, "edges": []}
    for m in models:
        lineage["families"].setdefault(m["family_id"], []).append(m["model_id"])
        if m["parent"]:
            lineage["edges"].append({"from": m["parent"], "to": m["model_id"], "relation": "initialised_from"})
        for o in m["same_bytes_as"]:
            if o > m["model_id"]:
                lineage["edges"].append({"from": m["model_id"], "to": o, "relation": "same_bytes"})
    for exp in (REPO / "results/artifacts" / "deploy").glob("*/export_record.json"):
        e = jload(exp.relative_to(REPO).as_posix(), {})
        for m in models:
            if m["family_id"] == "aerotrack.deploy" and m["run_id"] == exp.parent.name:
                src = find(sha=e.get("weights_sha256"))
                if src:
                    lineage["edges"].append({"from": src[0], "to": m["model_id"], "relation": "exported_from"})
                    m["parent"] = m["parent"] or src[0]
    # validation
    errs = []
    for m in models:
        errs += [f"{m['model_id']}: {x}" for x in schemas.errors(m, schemas.MODEL_SCHEMA)]
    for e in exps:
        errs += [f"{e['experiment_id']}: {x}" for x in schemas.errors(e, schemas.EXPERIMENT_SCHEMA)]
    datasets = build_datasets(cat_ds)
    for d in datasets:
        errs += [f"{d['id']}: {x}" for x in schemas.errors(d, schemas.DATASET_SCHEMA)]
    if errs:
        print("SCHEMA ERRORS:\n  " + "\n  ".join(errs[:40]))
        return 1
    REGISTRY.mkdir(exist_ok=True)
    gen = {"generated": t0.isoformat(timespec="seconds"), "generator": "scripts/build_workspace_registry.py", "git_head": git_head()}
    (REGISTRY / "models.json").write_text(json.dumps({**gen, "schema": "src/aerolab/models/1", "models": models}, indent=1), encoding="utf-8")
    with open(REGISTRY / "checkpoints.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["logical_path", "physical_path", "location", "size_bytes", "sha256", "artifact_kind", "family_id", "model_id", "run_id", "tag", "seed"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["logical_path"], r["location"])))
    (REGISTRY / "model_lineage.json").write_text(json.dumps({**gen, **lineage}, indent=1), encoding="utf-8")
    (REGISTRY / "duplicates.json").write_text(json.dumps({**gen, "groups": sorted(dups, key=lambda d: d["logical_paths"])}, indent=1), encoding="utf-8")
    (REGISTRY / "datasets.json").write_text(json.dumps({**gen, "schema": "src/aerolab/datasets/1", "datasets": datasets}, indent=1), encoding="utf-8")
    with open(REGISTRY / "experiments.jsonl", "w", encoding="utf-8") as f:
        for e in exps:
            f.write(json.dumps(e) + "\n")
    cache.save()
    summ = {**gen,
            "files": {"total": len(rows), "workspace": sum(r["location"] == "workspace" for r in rows),
                      "recovery_backup": sum(r["location"] == "recovery_backup" for r in rows),
                      "by_kind": dict(Counter(r["artifact_kind"] for r in rows)),
                      "distinct_sha256": len({r["sha256"] for r in rows}),
                      "distinct_sha256_model_weights_and_exports": len({r["sha256"] for r in rows if r["artifact_kind"] in ("model_weights", "export")}),
                      "unclassified": sorted(r["logical_path"] for r in rows if r["artifact_kind"] == "unclassified")},
            "models": {"total": len(models), "by_family": dict(Counter(m["family_id"] for m in models)),
                       "by_status": dict(Counter(m["status"] for m in models)), "by_integrity": dict(Counter(m["integrity_status"] for m in models)),
                       "workspace_available": sum(m["available_in_workspace"] for m in models),
                       "backup_only": sorted(m["model_id"] for m in models if not m["available_in_workspace"])},
            "duplicate_groups": len(dups), "experiments": {"total": len(exps), "by_kind": dict(Counter(e["kind"] for e in exps))},
            "hash_cache": {"hits": cache.hits, "misses": cache.misses}, "warnings": WARN}
    (REGISTRY / "build_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("files", "models", "duplicate_groups", "experiments", "hash_cache")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
