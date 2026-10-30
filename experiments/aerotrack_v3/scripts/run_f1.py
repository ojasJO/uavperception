#!/usr/bin/env python
"""F1: standard YOLO11n blocks vs the as-built network (preregistration_F1_standard_yolo11n.json) -- run by scripts/chains/f1_std.json.

Window from the v2 decisions -> train v3 'n_std' seed 0 (v3 runner's train / evaluate, resumed once on failure) -> futility (seed-0
CST delta vs v2's own seed-0 run of the same configuration <= 0 stops here) -> seed 1 -> scripts/analyze_f1.py (registered rule).
State: results/runs/aerotrack_v3/_f1_runner/ (events, logs); the chain owns RUNNING / DONE / FAILED. VAL only.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v3 as R  # noqa: E402
import v3series as S  # noqa: E402

PRE = S.SERIES / "preregistration_F1_standard_yolo11n.json"
WINDOW_OF = {"causal_g4": "causal_g4", "sym_g4": "sym_g4", "stock": "ttt"}
S.STATE = S.REPO / "results/runs" / "aerotrack_v3" / "_f1_runner"


def committed_clean(p):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=S.REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(p)], cwd=S.REPO).returncode == 0
    return tracked and clean


def main():
    S.STATE.mkdir(parents=True, exist_ok=True)
    if not committed_clean(PRE):
        raise SystemExit("F1 registration must be committed and unmodified")
    dec = S.REPO / "experiments" / "aerotrack_v2" / "reports" / "v2_decisions.json"
    if not dec.exists():
        raise SystemExit("v2 decisions missing: F1 takes its window from the v2 campaign's registered selection")
    sel = json.loads(dec.read_text(encoding="utf-8")).get("selected_candidate")
    if sel not in WINDOW_OF:
        raise SystemExit(f"v2 selected_candidate {sel!r} not in {sorted(WINDOW_OF)}")
    cfg = f"n_std__{WINDOW_OF[sel]}"
    R.event(kind="f1_start", v2_selected=sel, config=cfg)
    s0 = R.train(cfg, 0, 32)
    R.evaluate(s0)
    d0 = R.cst(s0) - R.cst(f"v2_{sel}_s0_e20")
    R.event(kind="futility_check", seed0_cst_delta=d0)
    if d0 > 0:
        R.evaluate(R.train(cfg, 1, 32))
    else:
        R.event(kind="futility_stop", reason="seed-0 CST delta <= 0: F1 cannot be SUPPORTED; seed 1 not trained")
    rc = subprocess.call([sys.executable, "-u", str(HERE / "analyze_f1.py")], cwd=str(S.REPO))
    R.event(kind="analysis", rc=rc)
    if rc != 0:
        raise SystemExit(f"analyze_f1 failed (rc {rc})")


if __name__ == "__main__":
    main()
