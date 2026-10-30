#!/usr/bin/env python
"""X1's single registered TEST pass (preregistration_X1_cst_protocol_matched.json): the CST-only candidate on the 60 CST test sequences.

    venv\\Scripts\\python.exe experiments\\aerotrack_v3\\scripts\\x1_final.py --final-eval

Guards: the registration and experiments/aerotrack_v3/frozen_x1_params.json (written by run_x1.py on VAL) must be committed and
unmodified; the candidate's weights must match; reports/x1_test_ledger.jsonl refuses a second pass. One pass computes: CST detection
AP (eval_v3.py --set test --final-eval), a full-sequence CST TEST detection cache, SOT with the frozen STE / argmax settings (Eq. 1
mSA by Track A's code; frame-level SA per size bin and CST attribute; located, false alarms, recovery), the registered comparisons
X1-a (bootstrap CI of our mSA vs the published 35.92), X1-b (tiny-target SA vs 15.83 / 28.20, descriptive) and X1-c (vs the v2
candidate's CST test result if its own pass exists, descriptive). Output: reports/x1_final_test.{json,md}.
Amendment X1-A1 (2026-10-12, before the pass): also writes the STE / argmax outputs in the official CST format to
results/runs/aerotrack_v3/x1_test_predictions/<tracker>/<official sequence>.txt, the per-sequence SA and the STE wall time; the official
toolkit scoring and the paired comparison with the released GlobalTrack predictions run afterwards on those files
(experiments/cst_anti_uav/scripts/analyze_test.py).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import v3series as S  # pins AEROTRACK_CORPUS=data_unified_v2

SYS = S.REPO / "experiments" / "systems" / "scripts"
sys.path.insert(0, str(SYS))
sys.path.insert(0, str(S.REPO / "experiments" / "cst_anti_uav" / "scripts"))
import cstlib  # noqa: E402  (official CST result format; amendment X1-A1)
REG = S.SERIES / "preregistration_X1_cst_protocol_matched.json"
FROZEN = S.SERIES / "frozen_x1_params.json"
LEDGER = S.REPORTS / "x1_test_ledger.jsonl"
PRED_DIR = S.REPO / "results/runs" / "aerotrack_v3" / "x1_test_predictions"
PUBLISHED = {"GlobalTrack": 35.92, "SiamDT": 35.84, "MixFormer-V2": 29.39}
PUBLISHED_TINY = {"GlobalTrack": 15.83, "PrDiMP (best tiny)": 28.20}


def _sys_module(name):
    """Explicit-path import of a systems module: after cache_detections is imported, src/model/ is first on sys.path and a plain
    `import track_eval` loads src/model/track_eval.py (F-05; reproduced for this script on 2026-10-12 before the pass)."""
    import importlib.util
    path = SYS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"systems_{name}", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    assert Path(m.__file__).resolve() == path.resolve()
    return m


def committed_clean(p):
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=S.REPO, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(p)], cwd=S.REPO).returncode == 0
    return tracked and clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-eval", action="store_true")
    a = ap.parse_args()
    if not a.final_eval:
        sys.exit("Refusing: X1's TEST pass needs --final-eval")
    from anti_uav_repro.utils.repro import file_sha256
    for p in (REG, FROZEN):
        if not committed_clean(p):
            sys.exit(f"Refusing: {p.relative_to(S.REPO)} must be committed and unmodified before the test pass")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    cand = frozen["candidate"]
    w = S.REPO / "results/runs" / cand / "weights" / "last.pt"
    wsha = file_sha256(w)
    if LEDGER.exists() and any(json.loads(x).get("weights_sha256") == wsha for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()):
        sys.exit("Refusing: X1 already has its single TEST pass")
    t0 = time.time()
    out = {"registration": str(REG.relative_to(S.REPO)), "candidate": cand, "weights_sha256": wsha, "frozen": frozen, "started": S.now()}
    rc = subprocess.call([sys.executable, "-u", str(S.SERIES / "scripts" / "eval_v3.py"), "--run", cand, "--set", "test", "--final-eval"],
                         cwd=str(S.REPO), env=dict(os.environ, AEROTRACK_CORPUS=S.CORPUS))
    if rc != 0:
        sys.exit(f"detection evaluation failed (rc {rc}); nothing else was run")
    rows = json.loads((S.REPORTS / "v3_results.json").read_text(encoding="utf-8"))["rows"]
    out["detection"] = next(r for r in rows if r["run"] == cand and r["set"] == "test")["metrics"]
    C = _sys_module("cache_detections")
    C.cache_split(cand, "test", ("cst",))
    TE = _sys_module("track_eval")
    from anti_uav_repro.eval.metrics import sequence_metrics
    index, seqs = TE.load(cand, S.CORPUS, split="test")
    keys = sorted(k for k in seqs if seqs[k]["group"] == "cst_IR")
    if len(keys) != 60:
        sys.exit(f"expected the 60 CST test sequences, found {len(keys)}")
    per, wall = {}, {}
    official = {k: seqs[k]["seq"].label_path.parent.name for k in keys}          # official CST sequence names
    for kind in ("ste", "argmax"):
        per[kind] = {}
        wall[kind] = [0.0, 0]
        for k in keys:
            tt = time.perf_counter()
            pred = TE.run_tracker(kind, frozen[kind], seqs[k])
            wall[kind][0] += time.perf_counter() - tt
            wall[kind][1] += seqs[k]["seq"].n_frames
            cstlib.write_official(PRED_DIR / kind, official[k], pred["res"])
            mask = np.arange(seqs[k]["seq"].n_frames) >= pred["start_frame"]
            per[kind][k] = {"SA": sequence_metrics(pred["res"], seqs[k]["seq"].exist, seqs[k]["seq"].gt_rect, mask=mask)["SA"],
                            **TE.extra(pred, seqs[k]["seq"], seqs[k].get("attrs"))}
    out["sot"] = {kind: TE.summarize(per[kind], keys, seqs) for kind in per}
    out["per_sequence_SA"] = {kind: {official[k]: 100 * per[kind][k]["SA"] for k in keys} for kind in per}
    out["predictions_dir"] = str(PRED_DIR.relative_to(S.REPO))
    out["tracker_ms_per_frame"] = {kind: 1000 * t / max(n, 1) for kind, (t, n) in wall.items()}   # tracker only, from cached detections
    rng = np.random.default_rng(0)
    sa = np.array([100 * per["ste"][k]["SA"] for k in keys])
    boots = [sa[rng.integers(0, len(sa), len(sa))].mean() for _ in range(2000)]
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    best = max(PUBLISHED.values())
    out["X1_a"] = {"ours_mSA": float(sa.mean()), "ci95": ci, "published": PUBLISHED, "reference": best,
                   "verdict": "exceeds the best published CST tracker" if ci[0] > best else "below the best published CST tracker"
                   if ci[1] < best else "not distinguishable from the best published CST tracker"}
    sums = np.array([per["ste"][k].get("attr", {}).get("size_tiny", [0.0, 0])[0] for k in keys])
    ns = np.array([per["ste"][k].get("attr", {}).get("size_tiny", [0.0, 0])[1] for k in keys])
    tb = []
    for _ in range(2000):
        ix = rng.integers(0, len(keys), len(keys))
        tb.append(100 * sums[ix].sum() / max(ns[ix].sum(), 1))
    out["X1_b_tiny"] = {"ours_frame_SA": float(100 * sums.sum() / max(ns.sum(), 1)), "frames": int(ns.sum()),
                        "ci95": [float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))], "published": PUBLISHED_TINY,
                        "note": "descriptive (frame-level SA on frames whose ground-truth diagonal <= 10 px)"}
    v2f = S.REPO / "experiments" / "aerotrack_v2" / "reports" / "final_test.json"
    if v2f.exists():
        v2 = json.loads(v2f.read_text(encoding="utf-8"))
        out["X1_c"] = {"v2_candidate": v2.get("candidate"), "v2_cst_ste_mSA": v2["sot"]["ste"].get("cst_IR", {}).get("mSA"),
                       "x1_cst_ste_mSA": out["sot"]["ste"]["cst_IR"]["mSA"], "note": "descriptive, unpaired across passes"}
    else:
        out["X1_c"] = {"note": "the v2 candidate's test pass does not exist yet; compare when it does (descriptive)"}
    out["finished"], out["minutes"] = S.now(), round((time.time() - t0) / 60, 1)
    S.write_json_atomic(out, S.REPORTS / "x1_final_test.json")
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": S.now(), "run": cand, "weights_sha256": wsha, "X1_a": out["X1_a"], "git": S.git()}) + "\n")
    md = [f"# X1 — the AeroTrack stack under the CST benchmark's protocol (CST test, single pass)", "",
          f"*Generated {out['finished']} by `scripts/x1_final.py` under `{out['registration']}`. Candidate `{cand}` (CST-only training, "
          f"window from the v2 decision), tracker settings frozen on CST VAL.*", "",
          f"- **X1-a:** STE mSA **{out['X1_a']['ours_mSA']:.2f}** [{ci[0]:.2f}, {ci[1]:.2f}] vs published best {best} "
          f"(GlobalTrack; SiamDT 35.84; MixFormer-V2 29.39) -> **{out['X1_a']['verdict']}**",
          f"- **X1-b:** tiny-target frame SA {out['X1_b_tiny']['ours_frame_SA']:.2f} [{out['X1_b_tiny']['ci95'][0]:.2f}, "
          f"{out['X1_b_tiny']['ci95'][1]:.2f}] over {out['X1_b_tiny']['frames']:,} frames (published: GlobalTrack 15.83, best PrDiMP 28.20)",
          f"- argmax mSA {out['sot']['argmax']['cst_IR']['mSA']:.2f}; CST drone AP50 {out['detection'].get('cst_drone_AP50')}",
          f"- X1-c: {out['X1_c']}"]
    (S.REPORTS / "x1_final_test.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
