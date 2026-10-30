#!/usr/bin/env python
"""X1: the AeroTrack stack under the CST benchmark's own protocol (preregistration_X1_cst_protocol_matched.json) -- VAL design part.

    venv\\Scripts\\python.exe experiments\\aerotrack_v3\\scripts\\run_x1.py            (run by scripts/chains/x1_cst_only.json)

Steps (registered): window from the v2 decisions -> train v3 'n' on CST train only (seed 0, the v2 recipe; v3 runner's train /
evaluate, resumed once on failure) -> eval_v3 on corpus-v2 VAL -> CST VAL detection cache -> track_eval (cross-fitted + tuned on all
CST VAL) -> experiments/aerotrack_v3/frozen_x1_params.json (to be committed before the single TEST pass). Never touches TEST.
State: results/runs/aerotrack_v3/_x1_runner/ (events, logs); the chain that runs this script owns the RUNNING / DONE / FAILED sentinels.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v3 as R  # noqa: E402
import v3series as S  # noqa: E402

PRE = S.SERIES / "preregistration_X1_cst_protocol_matched.json"
WINDOW_OF = {"causal_g4": "causal_g4", "sym_g4": "sym_g4", "stock": "ttt"}
S.STATE = S.REPO / "results/runs" / "aerotrack_v3" / "_x1_runner"           # R.event / R.sh log here, not in the v3 campaign's dir


def committed_clean(p):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=S.REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(p)], cwd=S.REPO).returncode == 0
    return tracked and clean


def main():
    S.STATE.mkdir(parents=True, exist_ok=True)
    if not committed_clean(PRE):
        raise SystemExit("X1 registration must be committed and unmodified")
    dec = S.REPO / "experiments" / "aerotrack_v2" / "reports" / "v2_decisions.json"
    if not dec.exists():
        raise SystemExit("v2 decisions missing: X1 takes its window from the v2 campaign's registered selection")
    sel = json.loads(dec.read_text(encoding="utf-8")).get("selected_candidate")
    if sel not in WINDOW_OF:
        raise SystemExit(f"v2 selected_candidate {sel!r} not in {sorted(WINDOW_OF)}")
    f1 = S.REPORTS / "F1_standard_yolo11n.json"                     # registered amendment: the network F1 adopts
    f1_status = json.loads(f1.read_text(encoding="utf-8")).get("status") if f1.exists() else "MISSING"
    model = "n_std" if f1_status == "SUPPORTED" else "n"
    cfg = f"{model}__{WINDOW_OF[sel]}"
    R.event(kind="x1_start", v2_selected=sel, f1_status=f1_status, config=cfg, registration=str(PRE.relative_to(S.REPO)))
    name = R.train(cfg, 0, 32, domains="cst")
    R.evaluate(name)
    env = dict(os.environ, AEROTRACK_CORPUS="data_unified_v2")
    py = sys.executable
    for job, cmd in (("cache_val_cst", [py, "-u", str(S.REPO / "experiments/systems/scripts/cache_detections.py"), "--run", name,
                                        "--corpus", "data_unified_v2", "--datasets", "cst"]),
                     ("track_eval_cst", [py, "-u", str(S.REPO / "experiments/systems/scripts/track_eval.py"), "--run", name,
                                         "--corpus", "data_unified_v2", "--workers", "4"])):
        R.event(kind="start", job=job)
        t0 = time.time()
        with open(S.STATE / f"{job}.log", "a", encoding="utf-8") as f:
            rc = subprocess.call(cmd, cwd=str(S.REPO), stdout=f, stderr=subprocess.STDOUT, env=env)
        R.event(kind="end", job=job, rc=rc, minutes=round((time.time() - t0) / 60, 1))
        if rc != 0:
            raise RuntimeError(f"{job} failed (rc {rc})")
    rep = json.loads((S.REPO / "experiments" / "systems" / "reports" / f"track_{name}_data_unified_v2.json").read_text(encoding="utf-8"))
    ref = f"v2_{sel}_s0_e20" if model == "n" else f"v3_n_std__{WINDOW_OF[sel]}_s0_e20"   # X1-c VAL reference (same network)
    frozen = {"registration": str(PRE.relative_to(S.REPO)), "candidate": name, "v2_selected": sel, "network": model,
              "f1_status": f1_status, "x1c_val_reference": ref, "written": S.now(),
              "rule": "tuned on ALL CST VAL by systems/scripts/track_eval.py (maximum mSA); commit before the single TEST pass",
              "ste": rep["trackers"]["ste"]["tuned_on_all_val"]["params"], "argmax": rep["trackers"]["argmax"]["tuned_on_all_val"]["params"],
              "val_cross_fitted_mSA": {k: v["cross_fitted"] for k, v in rep["trackers"].items()}}
    (S.SERIES / "frozen_x1_params.json").write_text(json.dumps(frozen, indent=1, default=str), encoding="utf-8")
    R.event(kind="x1_val_design_done", candidate=name)


if __name__ == "__main__":
    main()
