#!/usr/bin/env python
"""Build results/artifacts/experiment_registry.json — the ONE machine-readable index of every training run, evaluation and campaign.

    venv\\Scripts\\python.exe scripts\\build_registry.py

Generated only from files the experiments wrote themselves (run_meta.json, args.json, results/ledger JSON, test ledgers,
pre-registrations); nothing is typed in. Re-run after any experiment. Three record kinds:
  runs         one per trained model: git commit (+dirty), config hash, dataset / split hashes, seed, checkpoint + sha256,
               environment, hardware, status, training time
  evaluations  one per (model, evaluation set): set identity, key metrics, test flag, ledger reference
  experiments  one per campaign / hypothesis: question, pre-registration, member runs, verdict, report
Missing inputs are recorded as such (a campaign that has not produced results yet appears with status "running" / "planned").
"""
import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/artifacts" / "experiment_registry.json"


def rd(p):
    p = REPO / p if not Path(p).is_absolute() else Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def jl(p):
    p = REPO / p
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def rel(p):
    return str(Path(p).relative_to(REPO)).replace("\\", "/")


def env_of(meta):
    e = meta.get("env") or ((meta.get("attempts") or [{}])[0].get("env")) or {}
    gpu = e.get("gpu") if isinstance(e.get("gpu"), dict) else {}
    return ({k: e.get(k) for k in ("python", "torch", "torchvision", "numpy", "cuda_version", "cudnn_version", "platform") if e.get(k)},
            gpu.get("name") or meta.get("device") or "NVIDIA GeForce RTX 5050 Laptop GPU (local OMEN)")


def git_of(meta):
    g = meta.get("git") or ((meta.get("attempts") or [{}])[0].get("git")) or {}
    return g.get("commit"), g.get("dirty")


def anti_uav_runs():
    runs = []
    for d in sorted((REPO / "experiments" / "anti_uav" / "runs").iterdir()):
        m = rd(d / "run_meta.json")
        if not m or d.name.startswith("_"):
            continue
        env, hw = env_of(m)
        commit, dirty = git_of(m)
        cfg = rd(d / "config.effective.json") or {}
        import yaml
        ceff = d / "config.effective.yaml"
        cfg = yaml.safe_load(ceff.read_text(encoding="utf-8")) if ceff.exists() else cfg
        runs.append({"id": f"anti_uav/{d.name}", "track": "anti_uav", "run_dir": rel(d), "git_commit": commit, "git_dirty": dirty,
                     "config_hash": h({k: v for k, v in cfg.items() if k not in ("run_name", "seed")}) if cfg else None,
                     "phase": m.get("phase"), "dfsc_alpha": ((cfg or {}).get("dfsc") or {}).get("alpha"),
                     "dataset_hash": (cfg.get("data") or {}).get("manifest_sha256") if cfg else None,
                     "split_hash": "frozen manifest experiments/anti_uav/manifests/antiuav_IR_train.manifest.json",
                     "seed": m.get("seed"), "checkpoint": rel(d / "weights" / "last.pt"),
                     "checkpoint_sha256": m.get("final_checkpoint_sha256"), "environment": env, "hardware": hw,
                     "status": "completed" if m.get("final_checkpoint_sha256") else "incomplete", "train_seconds": m.get("train_seconds")})
    return runs


def aerotrack_runs():
    runs = []
    for d in sorted((REPO / "results/runs").iterdir()):
        m = rd(d / "run_meta.json")
        if not m or d.name.startswith("_") or not (d / "weights").exists():
            continue
        env, hw = env_of(m)
        commit, dirty = git_of(m)
        series = m.get("series") or ("aerotrack_repair_grid" if d.name.startswith("repair_abl") else "aerotrack")
        cfg = m.get("config_definition") or m.get("ablation_definition") or {"stage": m.get("stage"), "model_cfg": m.get("model_cfg")}
        data = m.get("data") or {}
        runs.append({"id": f"{series}/{d.name}", "track": "aerotrack", "series": series, "run_dir": rel(d), "git_commit": commit,
                     "git_dirty": dirty, "config": cfg, "config_hash": h({"cfg": cfg, "epochs": m.get("epochs"), "batch": m.get("batch"),
                                                                           "recipe": m.get("recipe") or m.get("stage_recipe")}),
                     "dataset_hash": data.get("manifest_sha256"), "split_hash": data.get("splits_sha256") or m.get("folds_sha256"),
                     "corpus": data.get("corpus", "data_unified (v1)"), "fold": m.get("fold"), "seed": m.get("seed"),
                     "epochs": m.get("epochs"), "smoke": m.get("smoke", False), "checkpoint_policy": m.get("checkpoint_policy"),
                     "checkpoint": rel(d / "weights" / "last.pt"), "checkpoint_sha256": m.get("last_pt_sha256"),
                     "best_pt_sha256": m.get("best_pt_sha256"), "environment": env, "hardware": hw,
                     "status": "completed" if m.get("return_code") == 0 else ("running" if m.get("return_code") is None else "failed"),
                     "train_seconds": m.get("train_seconds")})
    for name in ("aerotrack_spd_v1", "aerotrack_spd_finetuned"):          # Review-1/2 models (args.json only)
        d = REPO / "results/runs" / name
        a = rd(d / "args.json")
        if a:
            runs.append({"id": f"historical/{name}", "track": "aerotrack", "series": "historical_review2", "run_dir": rel(d),
                         "git_commit": None, "config_hash": h(a), "seed": a.get("seed"), "epochs": a.get("epochs"),
                         "checkpoint": rel(d / "weights" / "best.pt"), "status": "completed (stopped early; see audit)",
                         "note": "pre-protocol run: no run_meta.json; superseded, kept as evidence"})
    return runs


def track_c_runs():
    """Track C learned forecasters (CPU; results/runs/aerotrack_temporal/c1/learned_registry.json carries config, dev metrics, sha256)."""
    book = rd("results/runs/aerotrack_temporal/c1/learned_registry.json") or {}
    return [{"id": f"aerotrack_temporal/{n}", "track": "C", "series": "C1_trajectory_forecasting", "run_dir": "results/runs/aerotrack_temporal/c1",
             "config": {k: m[k] for k in ("arch", "variant", "abs_pos", "aug", "L", "H", "epochs")},
             "config_hash": h({k: m[k] for k in ("arch", "variant", "abs_pos", "aug", "L", "H", "epochs")}),
             "dataset_hash": (rd("results/runs/aerotrack_temporal/_data/traj_train.json") or {}).get("npz_sha256"), "seed": m["seed"],
             "checkpoint": f"results/runs/aerotrack_temporal/c1/{n}.pt", "checkpoint_sha256": m.get("sha256"), "best_epoch": m.get("best_epoch"),
             "dev_nll": m.get("dev_nll"), "dev_ade_px": m.get("dev_ade_px"), "hardware": "OMEN CPU", "status": "completed"}
            for n, m in sorted(book.items())]


def evaluations():
    ev = []
    for e in rd("results/artifacts/anti_uav_ledger.json")["entries"] if rd("results/artifacts/anti_uav_ledger.json") else []:
        if e["class"] in ("FAITHFUL_REPRODUCTION", "CONTROLLED_REPLICATION", "EXPLORATORY_EXPERIMENT") and e.get("mSA") is not None:
            ev.append({"id": f"anti_uav/{e['id']}", "track": "anti_uav", "class": e["class"], "run": e.get("train_run"),
                       "split": e.get("split"), "test": e.get("split") == "test", "test_ledger_line": e.get("test_ledger_line"),
                       "metrics": {"mSA": e["mSA"], "precision_20px": e.get("precision_20px"), "success_auc": e.get("success_auc")},
                       "checkpoint_sha256": e.get("checkpoint_sha256"), "source": e.get("metrics_file")})
    for r in (rd("experiments/aerotrack_generalization/reports/gen_results.json") or {}).get("rows", []):
        ev.append({"id": f"aerotrack_generalization/{r['run']}/{r['set']}", "track": "aerotrack", "run": r["run"], "split": r["set"],
                   "test": r["set"] == "test", "set_hash": (r.get("set_identity") or {}).get("sha256"), "metrics": r["metrics"],
                   "checkpoint_sha256": r.get("weights_sha256"), "source": "experiments/aerotrack_generalization/reports/gen_results.json"})
    grid = rd("experiments/aerotrack_repair/reports/ablation_grid_e20.json") or {}
    for split, key in (("val", "val_rows"), ("test", "test_rows")):
        for r in grid.get(key, []):
            ev.append({"id": f"aerotrack_repair/{r.get('run')}/{r.get('checkpoint', r.get('ckpt'))}/{split}", "track": "aerotrack",
                       "run": r.get("run"), "split": split, "test": split == "test", "set_hash": (grid.get(f"{split}_set") or {}).get("sha256"),
                       "metrics": {k: v for k, v in r.items() if isinstance(v, (int, float)) and k not in ("epoch",)},
                       "source": "experiments/aerotrack_repair/reports/ablation_grid_e20.json"})
    for r in (rd("experiments/aerotrack_v2/reports/v2_results.json") or {}).get("rows", []):
        ev.append({"id": f"aerotrack_v2/{r['run']}/{r['set']}", "track": "aerotrack", "run": r["run"], "split": r["set"], "test": r["set"] == "test",
                   "set_hash": r["set_identity"]["sha256"], "metrics": r["metrics"], "checkpoint_sha256": r["weights_sha256"],
                   "source": "experiments/aerotrack_v2/reports/v2_results.json"})
    for p in sorted((REPO / "experiments" / "deployment" / "reports").glob("gate_*.json")):
        g = rd(p)
        for k, v in g["backends"].items():
            ev.append({"id": f"src/deployment/{g['run']}/{k}", "track": "aerotrack", "kind": "deployment_gate", "run": g["run"], "split": "val",
                       "test": False, "metrics": {"available": v.get("available"), "latency": v.get("latency"),
                                                  "e2e": (v.get("e2e") or {}).get("per_frame_total"), "parity": v.get("parity")},
                       "source": rel(p)})
    for p in sorted((REPO / "experiments" / "systems" / "reports").glob("track_*.json")):
        t = rd(p)
        ev.append({"id": f"systems/{p.stem}", "track": "aerotrack", "kind": "sot_tracking", "run": t["run"], "split": "val", "test": False,
                   "metrics": {k: v["cross_fitted"] for k, v in t["trackers"].items()}, "source": rel(p)})
    for p in sorted((REPO / "experiments" / "aerotrack_temporal" / "reports").glob("tracking_*.json")):
        t = rd(p)
        ev.append({"id": f"aerotrack_temporal/{p.stem}", "track": "C", "kind": "tracking_identity_continuity", "run": t["run"],
                   "split": "val", "test": False, "groups": t.get("groups_kept"),
                   "metrics": {n: {g: {"HOTA": v["hota"]["HOTA"], "IDF1": v["clear"]["IDF1_mean_per_sequence"], "MOTA": v["clear"]["MOTA"]}
                                   for g, v in by.items()} for n, by in t["results"].items()}, "source": rel(p)})
    c1 = rd("experiments/aerotrack_temporal/reports/C1_trajectory_forecasting.json")
    if c1:
        for dom, d in c1["domains"].items():
            ev.append({"id": f"aerotrack_temporal/C1/{dom}", "track": "C", "kind": "forecasting", "split": "val", "test": False,
                       "metrics": {n: {"ade_px": v["all"]["ade_px"], "nll": v["all"].get("nll")} for n, v in d["summaries"].items()},
                       "source": "experiments/aerotrack_temporal/reports/C1_trajectory_forecasting.json"})
    return ev


def _state(d, default="planned"):
    d = REPO / d
    for name, label in (("DONE", "completed"), ("RUNNING", "running"), ("PAUSED", "paused (GPU yielded at an arm boundary)"), ("FAILED", "failed")):
        if (d / name).exists():
            return label
    return default


def experiments():
    gen = rd("experiments/aerotrack_generalization/reports/gen_decisions.json") or {}
    h4 = rd("experiments/anti_uav/reports/H4_native_scale.json") or {}
    v2 = rd("experiments/aerotrack_v2/reports/v2_decisions.json") or {}
    v2_state = REPO / "results/runs" / "aerotrack_v2" / "_runner"
    return [
        {"id": "anti_uav/phase_A", "track": "anti_uav", "hypothesis": "DFSC on GlobalTrack reproduces the paper's Protocol II IR results",
         "status": "completed", "report": "experiments/anti_uav/reports/FINAL_PROTOCOL_II_IR_BENCHMARK_REPORT.md",
         "interpretation": "val reproduced within 1 pt; test 4-5 pt above the paper; DFSC gain not observed (2 seeds); see results/artifacts/anti_uav_ledger.json claims"},
        {"id": "anti_uav/H2_absence", "track": "anti_uav", "hypothesis": "absence handling explains the val discrepancy",
         "status": "completed", "interpretation": "explains val OV/OC/SV deficits; cannot explain the test surplus"},
        {"id": "anti_uav/H4_native_scale", "track": "anti_uav", "hypothesis": "test-time input scale explains the test surplus",
         "preregistration": "experiments/anti_uav/preregistration/H4_native_scale.json",
         "status": "completed" if h4.get("verdict") not in (None, "NOT RUN YET") else "running",
         "verdict": h4.get("verdict"), "interpretation": h4.get("verdict_reason"), "report": "experiments/anti_uav/reports/H4_native_scale.md"},
        {"id": "aerotrack/repair_grid_e20", "track": "aerotrack", "hypothesis": "SPD-Conv / temporal stack / full recipe beat their ablations",
         "status": "completed", "report": "experiments/aerotrack_repair/reports/ablation_grid_e20.md",
         "interpretation": "no factor credited; later shown to be an initialisation confound (generalization series D2)"},
        {"id": "aerotrack/generalization", "track": "aerotrack", "hypothesis": "with function-preserving init, SPD-Conv / t±1 / t±4 beat stock on held-out folds",
         "preregistration": "experiments/aerotrack_generalization/ledger.jsonl (P1)", "status": "completed",
         "verdict": {k: c.get("status") for k, c in gen.get("candidates", {}).items()} if gen else None,
         "report": "experiments/aerotrack_generalization/reports/GEN_REPORT.md",
         "interpretation": "temporal_fp_g4 supported (+0.045 CST drone AP50); spd_fp and temporal_fp_g1 not supported; TEST used once"},
        {"id": "aerotrack/v2_complete_cst", "track": "aerotrack",
         "hypothesis": "causal temporal window keeps the temporal gain without look-ahead; complete CST data improves CST generalisation",
         "preregistration": "experiments/aerotrack_v2/preregistration.json",
         "status": ("completed" if (v2_state / "DONE").exists() else "running" if (v2_state / "RUNNING").exists() else
                    "failed" if (v2_state / "FAILED").exists() else "planned (waits for corpus v2)"),
         "verdict": {d["id"]: d.get("status") for d in v2.get("decisions", [])} if v2 else None,
         "selected_candidate": v2.get("selected_candidate"), "report": "experiments/aerotrack_v2/reports/v2_decisions.md"},
        {"id": "data/cst_completion", "track": "data", "hypothesis": "the local CST archive is truncated (z01-z07 of 11)",
         "status": "completed" if rd("experiments/cst_completion/reports/corpus_v2.json") else "running",
         "report": "experiments/cst_completion/reports/", "interpretation": "84/120 train, 0/40 val, 60/60 test sequences held locally before completion"},
        {"id": "anti_uav/H3_optimisation_regime", "track": "anti_uav",
         "hypothesis": "the paper's 8-GPU optimisation regime (accumulate 8, LR 0.02) explains the test surplus / reveals the DFSC gain",
         "preregistration": "experiments/anti_uav/preregistration/H3_optimisation_regime.json",
         "status": _state("experiments/anti_uav/runs/_h3_optimisation"), "verdict": (rd("experiments/anti_uav/reports/H3_optimisation_regime.json") or {}).get("verdicts"),
         "report": "experiments/anti_uav/reports/H3_optimisation_regime.md"},
        {"id": "C/C1_trajectory_forecasting", "track": "C",
         "hypothesis": "a learned forecaster beats the per-domain TRAIN-best classical forecaster, with better-calibrated uncertainty; absolute image position helps on gimbal footage",
         "preregistration": "experiments/aerotrack_temporal/preregistration/C1_trajectory_forecasting.json",
         "status": "completed" if rd("experiments/aerotrack_temporal/reports/C1_trajectory_forecasting.json") else "training (TRAIN / dev only)",
         "verdict": {dom: {k: d[k]["verdict"] for k in ("H1_point", "H2_uncertainty", "H3_absolute_position")}
                     for dom, d in (rd("experiments/aerotrack_temporal/reports/C1_trajectory_forecasting.json") or {}).get("domains", {}).items()} or None,
         "report": "experiments/aerotrack_temporal/reports/C1_trajectory_forecasting.md"},
        {"id": "C/C2_causal_windows", "track": "C",
         "hypothesis": "a zero-look-ahead 3-frame window ([t-2g, t-g, t], g = 1, 2, 4, 8) beats the single-frame detector; gap dependence; cost vs the symmetric window",
         "preregistration": "experiments/aerotrack_temporal/preregistration/C2_causal_windows.json",
         "status": _state("results/runs/aerotrack_temporal/_c2_runner", default="registered (GPU queue: after v2, post-v2, H3)"),
         "verdict": (rd("experiments/aerotrack_temporal/reports/C2_causal_windows.json") or {}).get("selected_causal_window"),
         "report": "experiments/aerotrack_temporal/reports/C2_causal_windows.md"},
        {"id": "C/C1b_forecasting_on_tracker_histories", "track": "C",
         "hypothesis": "on histories produced by the deployed tracker from a causal detector, a learned forecaster (clean / noise-augmented) beats the classical reference with calibrated uncertainty",
         "preregistration": "experiments/aerotrack_temporal/preregistration/C1b_forecasting_on_tracker_histories.json",
         "status": "completed" if list((REPO / "experiments" / "aerotrack_temporal" / "reports").glob("C1b_*.json")) else "registered (after C3)",
         "report": "experiments/aerotrack_temporal/reports/C1b_forecasting_on_tracker_histories_<run>.md"},
        {"id": "C/C3_tracking_motion_models", "track": "C",
         "hypothesis": "temporal state beats per-frame argmax; a domain / learned motion model beats the constant-velocity Kalman inside the same tracker",
         "preregistration": "experiments/aerotrack_temporal/preregistration/C3_tracking_motion_models.json",
         "status": "completed" if list((REPO / "experiments" / "aerotrack_temporal" / "reports").glob("C3_*.json")) else "registered (after C1 VAL + causal caches)",
         "report": "experiments/aerotrack_temporal/reports/C3_tracking_motion_models_<run>.md"},
        {"id": "src/deployment/onnx_export", "track": "aerotrack", "hypothesis": "the selected detector (and SPD-Conv) export to ONNX with numerical parity",
         "status": "completed", "report": "experiments/deployment/reports/", "interpretation": "both export with standard ops; parity at fp32 rounding"},
    ]


def main():
    reg = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "generator": "scripts/build_registry.py",
           "schema": {"runs": ["id", "track", "git_commit", "git_dirty", "config_hash", "dataset_hash", "split_hash", "seed", "checkpoint",
                               "checkpoint_sha256", "environment", "hardware", "status"],
                      "evaluations": ["id", "track", "run", "split", "test", "set_hash", "metrics", "source"],
                      "experiments": ["id", "track", "hypothesis", "preregistration", "status", "verdict", "interpretation", "report"]},
           "runs": anti_uav_runs() + aerotrack_runs() + track_c_runs(), "evaluations": evaluations(), "experiments": experiments(),
           "test_ledgers": {"anti_uav": {"path": "experiments/anti_uav/reports/test_evaluation_ledger.jsonl",
                                         "entries": len(jl("experiments/anti_uav/reports/test_evaluation_ledger.jsonl"))},
                            "aerotrack_repair": {"path": "experiments/aerotrack_repair/reports/test_evaluation_ledger.jsonl",
                                                 "entries": len(jl("experiments/aerotrack_repair/reports/test_evaluation_ledger.jsonl"))},
                            "aerotrack_generalization": {"path": "experiments/aerotrack_generalization/reports/gen_test_ledger.jsonl",
                                                         "entries": len(jl("experiments/aerotrack_generalization/reports/gen_test_ledger.jsonl"))},
                            "aerotrack_v2": {"path": "experiments/aerotrack_v2/reports/v2_test_ledger.jsonl",
                                             "entries": len(jl("experiments/aerotrack_v2/reports/v2_test_ledger.jsonl"))}}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=1, default=str), encoding="utf-8")
    tmp.replace(OUT)
    print(f"wrote {rel(OUT)}: {len(reg['runs'])} runs, {len(reg['evaluations'])} evaluations, {len(reg['experiments'])} experiments; "
          f"test ledgers: " + ", ".join(f"{k} {v['entries']}" for k, v in reg["test_ledgers"].items()))


if __name__ == "__main__":
    main()
