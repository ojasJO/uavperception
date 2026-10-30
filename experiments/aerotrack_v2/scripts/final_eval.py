#!/usr/bin/env python
"""The single, pre-registered TEST pass of the v2 series' final candidate (protocol: experiments/aerotrack_v2/final_protocol.json).

    1. venv\\Scripts\\python.exe experiments\\aerotrack_v2\\scripts\\final_eval.py --freeze-from-val --candidate v2_<cfg>_s0_e20
       -> experiments/aerotrack_v2/frozen_ste_params.json (tracker parameters tuned on ALL of corpus-v2 VAL); COMMIT IT.
    2. venv\\Scripts\\python.exe experiments\\aerotrack_v2\\scripts\\final_eval.py --final-eval --candidate v2_<cfg>_s0_e20

Step 2 refuses unless: the candidate is the configuration selected by reports/v2_decisions.json (seed 0); final_protocol.json and
frozen_ste_params.json are committed and unmodified and the frozen parameters belong to this candidate; the candidate has no
entry in reports/v2_test_ledger.jsonl. It then computes, in one pass: detection AP (eval_v2.py --set test --final-eval), a
full-sequence detection cache of TEST, single-object-tracking mSA (Track A's code, official annotations) for the frozen STE and
argmax trackers, the paired comparisons F1 (vs our GlobalTrack on Anti-UAV IR test) and F3 (STE vs argmax), and forecasting
metrics. Output: reports/final_test.{json,md} and one systems record in the test ledger.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import v2series as S  # pins AEROTRACK_CORPUS=data_unified_v2

import numpy as np

SYS = S.REPO / "experiments" / "systems" / "scripts"
sys.path.insert(0, str(SYS))
FROZEN = S.SERIES / "frozen_ste_params.json"
PROTOCOL = S.SERIES / "final_protocol.json"


def _systems_module(name):
    """Import experiments/systems/scripts/<name>.py by path. c_model/ also contains a track_eval.py, and cache_detections puts
    c_model/ first on sys.path, so a plain `import track_eval` resolved to the wrong module (F-05, 2026-10-06)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"systems_{name}", SYS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert Path(m.__file__).resolve() == (SYS / f"{name}.py").resolve()
    return m


def committed_clean(path):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)], cwd=S.REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=S.REPO).returncode == 0
    return tracked and clean


def freeze(cand):
    from anti_uav_repro.utils.repro import file_sha256
    rep = S.REPO / "experiments" / "systems" / "reports" / f"track_{cand}_{S.CORPUS}.json"
    if not rep.exists():
        sys.exit(f"missing {rep}: cache VAL for the candidate (systems/scripts/cache_detections.py --corpus {S.CORPUS}) and run track_eval.py")
    r = json.loads(rep.read_text(encoding="utf-8"))
    out = {"candidate": cand, "source_report": str(rep.relative_to(S.REPO)), "source_report_sha256": file_sha256(rep),
           "cache_set_sha256": r["cache_set_sha256"], "selection": "maximum mean group mSA over ALL corpus-v2 VAL sequences",
           "ste": r["trackers"]["ste"]["tuned_on_all_val"]["params"], "argmax": r["trackers"]["argmax"]["tuned_on_all_val"]["params"],
           "val_cross_fitted": {k: v["cross_fitted"] for k, v in r["trackers"].items()}, "frozen": S.now()}
    S.write_json_atomic(out, FROZEN)
    print(f"wrote {FROZEN.relative_to(S.REPO)} -- commit it before --final-eval")


def paired(a, b, B=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    verdict = "better" if ci[0] > 0 else "worse" if ci[1] < 0 else "no detectable difference"
    return {"n": int(len(d)), "mean_delta": float(d.mean()), "ci95": ci, "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "verdict": verdict}


def final(cand, resume=False):
    from anti_uav_repro.utils.repro import file_sha256
    dpath = S.REPORTS / "v2_decisions.json"
    if not dpath.exists():
        sys.exit("Refusing: the v2 campaign has not selected a candidate yet (reports/v2_decisions.json missing)")
    dec = json.loads(dpath.read_text(encoding="utf-8"))
    if not dec.get("selected_candidate") or cand != f"v2_{dec['selected_candidate']}_s0_e20":
        sys.exit(f"Refusing: {cand} is not the selected candidate (v2_decisions.json: {dec.get('selected_candidate')!r}, seed 0)")
    for p in (PROTOCOL, FROZEN):
        if not committed_clean(p):
            sys.exit(f"Refusing: {p.relative_to(S.REPO)} must be committed and unmodified before the test pass")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen["candidate"] != cand:
        sys.exit(f"Refusing: frozen parameters belong to {frozen['candidate']}, not {cand}")
    w = S.REPO / "runs" / cand / "weights" / "last.pt"
    wsha = file_sha256(w)
    lines = [json.loads(x) for x in S.TEST_LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()] if S.TEST_LEDGER.exists() else []
    mine = [x for x in lines if x.get("weights_sha256") == wsha]
    t0 = time.time()
    out = {"candidate": cand, "weights_sha256": wsha, "protocol": str(PROTOCOL.relative_to(S.REPO)), "frozen": frozen, "started": S.now()}
    rows = json.loads((S.REPORTS / "v2_results.json").read_text(encoding="utf-8"))["rows"]
    if not resume:
        if mine:
            sys.exit("Refusing: this candidate already has its single TEST pass")
        # (a) detection AP, through the series' own guarded evaluator (writes the detection ledger record)
        rc = subprocess.call([sys.executable, "-u", str(S.SERIES / "scripts" / "eval_v2.py"), "--run", cand, "--set", "test", "--final-eval"],
                             cwd=str(S.REPO), env=dict(os.environ, AEROTRACK_CORPUS=S.CORPUS))
        if rc != 0:
            sys.exit(f"detection evaluation failed (rc {rc}); nothing else was run")
        rows = json.loads((S.REPORTS / "v2_results.json").read_text(encoding="utf-8"))["rows"]
    else:
        # --resume-after-crash: the SAME single pass crashed after (a) [and (b)]; finish only what was never computed, with the
        # parameters frozen before the pass. Allowed only if the ledger holds exactly this candidate's detection record and no
        # systems record (F-05 in docs/FAILURES_AND_RECOVERY.md).
        det = [x for x in mine if "kind" not in x]
        sysrec = [x for x in mine if str(x.get("kind", "")).startswith("systems")]
        if len(det) != 1 or sysrec:
            sys.exit(f"Refusing resume: ledger has {len(det)} detection and {len(sysrec)} systems records for this candidate")
        out["resumed_after_crash"] = {"detection_ledger_timestamp": det[0]["timestamp"], "resumed": S.now(),
                                      "reason": "systems part crashed (module shadowing, F-05); detection AP + TEST cache were complete"}
    out["detection"] = next(r for r in rows if r["run"] == cand and r["set"] == "test")["metrics"]
    # (b) full-sequence TEST cache (deterministic inference). On resume, a complete cache of these exact weights is reused.
    idx = S.REPO / "runs" / "systems" / "_cache" / cand / S.CORPUS / "test" / "index.json"
    reuse = False
    if resume and idx.exists():
        ix = json.loads(idx.read_text(encoding="utf-8"))
        reuse = ix.get("weights_sha256") == wsha and not ix.get("limited") and len(ix.get("sequences", {})) > 0
        out["resumed_after_crash"]["test_cache_reused"] = {"index": str(idx.relative_to(S.REPO)), "set_sha256": ix.get("set_sha256"),
                                                         "sequences": len(ix.get("sequences", {}))} if reuse else None
    if not reuse:
        import cache_detections as C
        C.cache_split(cand, "test", ("anti_uav", "cst"))
    # (c) SOT with the frozen trackers
    TE = _systems_module("track_eval")
    from anti_uav_repro.eval.evaluate import evaluate_prediction_dicts
    index, seqs = TE.load(cand, S.CORPUS, split="test")
    keys = sorted(seqs)
    per, preds = {}, {}
    for kind, params in (("ste", frozen["ste"]), ("argmax", frozen["argmax"])):
        per[kind], preds[kind] = {}, {}
        for k in keys:
            pred = TE.run_tracker(kind, params, seqs[k])
            preds[kind][k] = pred
            mask = np.arange(seqs[k]["seq"].n_frames) >= pred["start_frame"]
            from anti_uav_repro.eval.metrics import sequence_metrics
            per[kind][k] = {"SA": sequence_metrics(pred["res"], seqs[k]["seq"].exist, seqs[k]["seq"].gt_rect, mask=mask)["SA"],
                            **TE.extra(pred, seqs[k]["seq"], seqs[k].get("attrs"))}
    out["sot"] = {kind: TE.summarize(per[kind], keys, seqs) for kind in per}
    ir = [k for k in keys if seqs[k]["group"] == "anti_uav_IR"]
    names = {k: k.split("_", 2)[2].rsplit("__", 1)[0] for k in ir}
    ir_preds = {names[k]: dict(preds["ste"][k], sequence=names[k]) for k in ir}
    agg = evaluate_prediction_dicts(ir_preds, [seqs[k]["seq"] for k in ir])["aggregate"]
    out["sot"]["ste_anti_uav_IR_by_attribute"] = {a: v["mSA_percent"] for a, v in agg["by_attribute"].items()}
    gt = json.loads((S.REPO / "experiments" / "anti_uav" / "runs" / "normal_ir" / "eval_test_IR_last" / "metrics.json").read_text(encoding="utf-8"))
    g = gt["per_sequence"]
    out["F1_vs_globaltrack_anti_uav_IR_test"] = {**paired([100 * per["ste"][k]["SA"] for k in ir], [100 * g[names[k]]["SA"] for k in ir]),
                                                 "globaltrack_mSA": float(np.mean([100 * g[names[k]]["SA"] for k in ir])),
                                                 "source": "experiments/anti_uav/runs/normal_ir/eval_test_IR_last/metrics.json"}
    out["F3_ste_vs_argmax"] = {grp: paired([100 * per["ste"][k]["SA"] for k in keys if seqs[k]["group"] == grp],
                                           [100 * per["argmax"][k]["SA"] for k in keys if seqs[k]["group"] == grp])
                               for grp in sorted({seqs[k]["group"] for k in keys})}
    out["F2_cst_benchmark_reference"] = {"published_best_mSA": 35.92, "tracker": "GlobalTrack retrained on CST",
                                         "source": "arXiv:2507.23473 Table 3", "ours_ste_mSA": out["sot"]["ste"].get("cst_IR", {}).get("mSA"),
                                         "note": "descriptive only (pre-registered)"}
    # (f) forecasting on test windows
    FE = _systems_module("forecast_eval")
    rows_f = {}
    for k in keys:
        rows_f.setdefault(seqs[k]["group"], []).extend(FE.run_sequence(seqs[k], frozen["ste"]))
    out["forecast"] = {g_: FE.summarize(r) for g_, r in rows_f.items()}
    out["finished"], out["minutes"] = S.now(), round((time.time() - t0) / 60, 1)
    S.write_json_atomic(out, S.REPORTS / "final_test.json")
    with open(S.TEST_LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": S.now(), "run": cand, "weights_sha256": wsha, "kind": "systems (SOT + forecast) part of the single pass",
                            "resumed_after_crash": out.get("resumed_after_crash"),
                            "sot_ste": out["sot"]["ste"], "F1": out["F1_vs_globaltrack_anti_uav_IR_test"], "git": S.git()}) + "\n")
    S.ledger({"id": f"FINAL:{cand}", "type": "final_test", "run": cand, "weights_sha256": wsha, "output": "reports/final_test.json"})
    print(json.dumps({k: out[k] for k in ("detection", "F1_vs_globaltrack_anti_uav_IR_test")}, indent=1, default=float))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--freeze-from-val", action="store_true")
    ap.add_argument("--final-eval", action="store_true")
    ap.add_argument("--resume-after-crash", action="store_true", help="finish a crashed single pass (see final(); F-05)")
    a = ap.parse_args()
    if a.freeze_from_val == a.final_eval:
        sys.exit("choose exactly one of --freeze-from-val / --final-eval")
    freeze(a.candidate) if a.freeze_from_val else final(a.candidate, resume=a.resume_after_crash)


if __name__ == "__main__":
    main()
