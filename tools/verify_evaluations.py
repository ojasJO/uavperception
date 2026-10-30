"""Evaluation verification: does a fresh, whole-sequence Flight Deck run of a checkpoint reproduce a RECORDED per-sequence VAL result?

For every checkpoint with recorded per-frame VAL predictions (aerolab.recorded, VAL sources) the tool runs one Flight Deck experiment
(category tracking / reproduction, whole sequences, the same tracker) on N VAL sequences and compares the per-sequence Eq. 1 SA with
the recorded predictions' SA (Track A metrics on both sides). A model is 'evaluation-verified' when every |delta| <= tolerance.
Tolerances: GlobalTrack 0.05 SA (same fp32 code path); AeroTrack detector 0.5 SA (the recorded VAL predictions come from the HPC's
fp16 batched detection cache, the stream runs bf16 autocast at batch 1 on the OMEN, so boxes differ by numerics).

    venv\\Scripts\\python.exe tools\\verify_evaluations.py              # 1 sequence per model (~6 min with CUDA)
    venv\\Scripts\\python.exe tools\\verify_evaluations.py --n 2

Writes registry/verification/evaluation_checks.json (and one fd- experiment per model under runs/studio/evals/). VAL only.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aerolab import analysis, data, recorded  # noqa: E402
from aerolab import flightdeck as FD  # noqa: E402
from aerolab.registry import Registry  # noqa: E402

OUT = REPO / "registry" / "verification" / "evaluation_checks.json"
TOL = {"aerotrack_detector": 0.5, "globaltrack_repro": 0.05}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="VAL sequences per model")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    R = Registry()
    checks = []
    srcs = [s for s in recorded.sources() if s["split"] == "val" and s["tracker"] in ("ste", "globaltrack")]
    for src in srcs:
        rec = R.model(src["model_id"])
        names = recorded.sequences_of(src)[: a.n]
        cat = "reproduction" if rec["adapter"] == "globaltrack_repro" else "tracking"
        cfg = FD.build_config(cat, src["dataset_id"], "val", src["modality"], names, [{"model_id": rec["model_id"], "tracker": src["tracker"]}],
                              device=a.device, note=f"evaluation verification against {src['id']}")
        man = FD.run(cfg, R)
        st = man["models"][0]
        rows = []
        for n in names:
            q = data.load_sequence(src["dataset_id"], "val", n, src["modality"])
            p = recorded.load(src, n)
            rsa = analysis.sequence_summary(p["res_xywh"], q.exist, q.gt_rect, p["start_frame"] if p["start_frame"] is not None else q.start_frame,
                                            None, p["processed"])["SA"]
            fsa = st["per_sequence"][n]["SA"]
            rows.append({"sequence": n, "fresh_SA": fsa, "recorded_SA": rsa, "delta": fsa - rsa})
        tol = TOL[rec["adapter"]]
        ok = st["status"] == "completed" and all(abs(r["delta"]) <= tol for r in rows)
        checks.append({"model_id": rec["model_id"], "status": "passed" if ok else "failed", "tolerance_SA": tol, "recorded_source": src["id"],
                       "recorded_dir": src["dir"], "experiment_id": man["experiment_id"], "device": st.get("device"), "tracker": src["tracker"],
                       "sequences": rows, "checked": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
        print(f"{rec['model_id']:60s} {'PASS' if ok else 'FAIL'}  " + "  ".join(f"{r['sequence']} {r['fresh_SA']:.3f} vs {r['recorded_SA']:.3f}" for r in rows))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"schema": "aerolab/evaluation_checks/1", "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                               "method": __doc__.split("\n\n")[0] + " " + __doc__.split("\n\n")[1].split("\n")[0], "tolerances": TOL, "checks": checks},
                              indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}: {sum(c['status'] == 'passed' for c in checks)} / {len(checks)} passed")


if __name__ == "__main__":
    main()
