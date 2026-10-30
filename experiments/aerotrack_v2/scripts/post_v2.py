#!/usr/bin/env python
"""VAL-side follow-up of the v2 campaign for the candidate its pre-registered rule selected (reports/v2_decisions.json).

    venv\\Scripts\\python.exe experiments\\aerotrack_v2\\scripts\\post_v2.py

Steps (each a separate process; a failed optional step is recorded and the rest continue):
  1 export_onnx.py            ONNX export + parity of the candidate
  2 deploy_gate.py            latency / memory / end-to-end / AP parity per backend (the GPU is idle after the campaign)
  3 cache_detections.py       every corpus-v2 VAL frame (Anti-UAV + CST), candidate's window
  4 track_eval.py             SOT mSA, cross-fitted trackers, paired vs our GlobalTrack
  5 forecast_eval.py          forecasting + uncertainty calibration
  6 failure_cases.py          deterministic failure analysis; mot_eval.py: ByteTrack / SORT / argmax / STE under CLEAR-MOT
  7 final_eval.py --freeze-from-val    writes frozen_ste_params.json (NOT committed here; the single TEST pass is run by hand)
Writes results/runs/aerotrack_v2/_post/summary.json. Never touches TEST.
"""
import json
import os
import subprocess
import sys
import time

import v2series as S

PY = sys.executable
D = S.REPO / "experiments"


def main():
    dec = json.loads((S.REPORTS / "v2_decisions.json").read_text(encoding="utf-8"))
    cfg = dec.get("selected_candidate")
    if not cfg:
        sys.exit("no selected candidate in v2_decisions.json (campaign incomplete?)")
    cand = f"v2_{cfg}_s0_e20"
    w = f"results/runs/{cand}/weights/last.pt"
    env = dict(os.environ, AEROTRACK_CORPUS=S.CORPUS)
    steps = [("export_onnx", [PY, "-u", str(D / "src/deployment/scripts/export_onnx.py"), "--weights", w], True),
             ("export_onnx_fp16", [PY, "-u", str(D / "src/deployment/scripts/export_onnx.py"), "--weights", w, "--fp16"], False),
             ("deploy_gate", [PY, "-u", str(D / "src/deployment/scripts/deploy_gate.py"), "--weights", w], False),
             ("cache_val", [PY, "-u", str(D / "systems/scripts/cache_detections.py"), "--run", cand, "--corpus", S.CORPUS], True),
             ("track_eval", [PY, "-u", str(D / "systems/scripts/track_eval.py"), "--run", cand, "--corpus", S.CORPUS, "--workers", "8"], True),
             ("forecast_eval", [PY, "-u", str(D / "systems/scripts/forecast_eval.py"), "--run", cand, "--corpus", S.CORPUS], False),
             ("failure_cases", [PY, "-u", str(D / "systems/scripts/failure_cases.py"), "--run", cand, "--corpus", S.CORPUS], False),
             ("mot_eval", [PY, "-u", str(D / "systems/scripts/mot_eval.py"), "--run", cand, "--corpus", S.CORPUS], False),
             ("freeze_ste", [PY, "-u", str(S.SERIES / "scripts/final_eval.py"), "--freeze-from-val", "--candidate", cand], True),
             ("report_v2", [PY, "-u", str(S.SERIES / "scripts/report_v2.py")], False)]
    # AEROTRACK_POST_SKIP="deploy_gate,..." (HPC, amendment A1): latency gates belong on the deployment GPU (RTX 5050), so the
    # HPC run skips them; they are run on the OMEN with the same weights. Skipped steps are recorded in summary.json.
    skip = {x.strip() for x in os.environ.get("AEROTRACK_POST_SKIP", "").split(",") if x.strip()}
    steps = [s for s in steps if s[0] not in skip]
    out_dir = S.REPO / "results/runs" / "aerotrack_v2" / "_post"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"candidate": cand, "started": S.now(), "skipped": sorted(skip), "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"), "steps": []}
    for name, cmd, required in steps:
        t0 = time.time()
        with open(out_dir / f"{name}.log", "a", encoding="utf-8") as f:
            rc = subprocess.call(cmd, cwd=str(S.REPO), stdout=f, stderr=subprocess.STDOUT, env=env)
        summary["steps"].append({"step": name, "rc": rc, "minutes": round((time.time() - t0) / 60, 1)})
        S.write_json_atomic(summary, out_dir / "summary.json")
        if rc != 0 and required:
            sys.exit(f"required step {name} failed (rc {rc})")
    summary["finished"] = S.now()
    S.write_json_atomic(summary, out_dir / "summary.json")
    S.ledger({"id": f"POST:{cand}", "type": "post_campaign_val_analysis", "candidate": cand, "steps": summary["steps"]})


if __name__ == "__main__":
    main()
