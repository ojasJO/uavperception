#!/usr/bin/env python
"""X2's single registered TEST pass (preregistration/X2_seeds_single_frame.json): the five new CST-only runs on the 60 CST test sequences.

    python experiments/cst_anti_uav/scripts/x2_final.py --final-eval

Guards: the registration and experiments/cst_anti_uav/frozen_x2_params.json must be committed and unmodified; X1's own TEST pass
must exist (experiments/aerotrack_v3/reports/x1_test_ledger.jsonl); each run's weights must match the frozen sha256; a run already
in reports/x2_test_ledger.jsonl is not scored again (its saved per-run result is reused, so a crashed pass resumes without
re-scoring anything). Per run: CST detection AP (eval_v3 --set test --final-eval; skipped if those weights already have their v3 TEST
row), a full CST TEST detection cache, STE / argmax at the frozen settings, official-format export
(results/runs/cst_anti_uav/x2_test_predictions/<run>/<tracker>/), per-sequence SA (Track A) and tracker wall time.
Output: reports/x2_final_test.json. Official scoring and T1-T4: scripts/analyze_test.py --x2 (OMEN).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V3 = REPO / "experiments" / "aerotrack_v3" / "scripts"
for p in (HERE, V3, REPO / "experiments" / "systems" / "scripts"):
    sys.path.insert(0, str(p))
import cstlib  # noqa: E402
import x2lib as X  # noqa: E402
import v3series as S  # noqa: E402  (pins AEROTRACK_CORPUS=data_unified_v2)

PER_RUN = REPO / "experiments" / "cst_anti_uav" / "reports" / "x2_test_runs"


def _sys_module(name):
    """Explicit-path import of a systems module (src/model/track_eval.py would shadow systems/scripts/track_eval.py: F-05)."""
    import importlib.util
    p = REPO / "experiments" / "systems" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"systems_{name}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    assert Path(m.__file__).resolve() == p.resolve()
    return m


def committed_clean(p):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(p)], cwd=REPO).returncode == 0
    return tracked and clean


def ledger_shas(p):
    return {json.loads(x).get("weights_sha256") for x in p.read_text(encoding="utf-8").splitlines() if x.strip()} if p.exists() else set()


def score_run(run, fz, wsha):
    if wsha not in ledger_shas(S.TEST_LEDGER):
        rc = subprocess.call([sys.executable, "-u", str(V3 / "eval_v3.py"), "--run", run, "--set", "test", "--final-eval"],
                             cwd=str(REPO), env=dict(os.environ, AEROTRACK_CORPUS=S.CORPUS))
        if rc != 0:
            raise SystemExit(f"{run}: detection evaluation failed (rc {rc})")
    det = next(r for r in json.loads((S.REPORTS / "v3_results.json").read_text(encoding="utf-8"))["rows"]
               if r["run"] == run and r["set"] == "test")
    if det["weights_sha256"] != wsha:
        raise SystemExit(f"{run}: v3 TEST row belongs to other weights")
    C = _sys_module("cache_detections")
    TE = _sys_module("track_eval")
    from anti_uav_repro.eval.metrics import sequence_metrics
    C.cache_split(run, "test", ("cst",))
    index, seqs = TE.load(run, S.CORPUS, split="test")
    if index["weights_sha256"] != wsha:
        raise SystemExit(f"{run}: TEST cache was made with other weights")
    keys = sorted(k for k in seqs if seqs[k]["group"] == "cst_IR")
    if len(keys) != 60:
        raise SystemExit(f"{run}: expected the 60 CST test sequences, found {len(keys)}")
    official = {k: seqs[k]["seq"].label_path.parent.name for k in keys}
    pred_dir = REPO / X.PRED_ROOT / run
    per, wall = {}, {}
    for kind in ("ste", "argmax"):
        per[kind], t, n = {}, 0.0, 0
        for k in keys:
            tt = time.perf_counter()
            pred = TE.run_tracker(kind, fz[kind], seqs[k])
            t += time.perf_counter() - tt
            n += seqs[k]["seq"].n_frames
            cstlib.write_official(pred_dir / kind, official[k], pred["res"])
            mask = np.arange(seqs[k]["seq"].n_frames) >= pred["start_frame"]
            per[kind][k] = {"SA": sequence_metrics(pred["res"], seqs[k]["seq"].exist, seqs[k]["seq"].gt_rect, mask=mask)["SA"],
                            **TE.extra(pred, seqs[k]["seq"], seqs[k].get("attrs"))}
        wall[kind] = 1000 * t / max(n, 1)
    return {"run": run, "arm": fz["arm"], "seed": fz["seed"], "weights_sha256": wsha, "frozen": {k: fz[k] for k in ("ste", "argmax")},
            "detection": det["metrics"], "sot": {kind: TE.summarize(per[kind], keys, seqs) for kind in per},
            "per_sequence_SA": {kind: {official[k]: 100 * per[kind][k]["SA"] for k in keys} for kind in per},
            "predictions_dir": str((X.PRED_ROOT / run).as_posix()), "tracker_ms_per_frame": wall, "finished": S.now()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-eval", action="store_true")
    a = ap.parse_args()
    if not a.final_eval:
        sys.exit("Refusing: X2's TEST pass needs --final-eval")
    from anti_uav_repro.utils.repro import file_sha256
    for p in (X.REGISTRATION, X.FROZEN):
        if not committed_clean(REPO / p):
            sys.exit(f"Refusing: {p} must be committed and unmodified before the test pass")
    if not (REPO / "experiments" / "aerotrack_v3" / "reports" / "x1_test_ledger.jsonl").exists():
        sys.exit("Refusing: X1's own single TEST pass must come first")
    frozen = json.loads((REPO / X.FROZEN).read_text(encoding="utf-8"))
    ledger = REPO / X.TEST_LEDGER
    PER_RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out = {"registration": str(X.REGISTRATION.as_posix()), "frozen": str(X.FROZEN.as_posix()), "started": S.now(), "runs": {},
           "predictions_dirs": {}}
    for run in X.new_runs():
        fz = frozen["runs"][run]
        wsha = file_sha256(REPO / "results/runs" / run / "weights" / "last.pt")
        if wsha != fz["weights_sha256"]:
            sys.exit(f"Refusing: {run} weights differ from the frozen sha256")
        saved = PER_RUN / f"{run}.json"
        if wsha in ledger_shas(ledger):
            if not saved.exists():
                sys.exit(f"{run} is in the X2 TEST ledger but its result file is missing; investigate, do not re-score")
            r = json.loads(saved.read_text(encoding="utf-8"))
            print(f"{run}: already scored in this pass (ledger), reusing {saved.name}", flush=True)
        else:
            r = score_run(run, fz, wsha)
            S.write_json_atomic(r, saved)
            with open(ledger, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": S.now(), "run": run, "weights_sha256": wsha,
                                    "ste_mSA_track_a": r["sot"]["ste"]["cst_IR"]["mSA"], "git": S.git()}) + "\n")
            print(f"{run}: STE mSA {r['sot']['ste']['cst_IR']['mSA']:.2f}, argmax {r['sot']['argmax']['cst_IR']['mSA']:.2f}", flush=True)
        out["runs"][run] = {k: r[k] for k in ("arm", "seed", "weights_sha256", "frozen", "detection", "sot", "tracker_ms_per_frame")}
        out["runs"][run]["per_sequence_SA"] = r["per_sequence_SA"]
        out["predictions_dirs"][run] = r["predictions_dir"]
    out["finished"], out["minutes"] = S.now(), round((time.time() - t0) / 60, 1)
    S.write_json_atomic(out, REPO / X.FINAL_JSON)
    print(f"X2 TEST pass complete: {len(out['runs'])} runs, {out['minutes']} min -> {X.FINAL_JSON}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
