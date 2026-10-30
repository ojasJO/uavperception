#!/usr/bin/env python
"""Export a run's CST VAL tracking outputs in the official format (VAL only) so the official toolkit can score them.

    venv\\Scripts\\python.exe experiments\\cst_anti_uav\\scripts\\export_val.py --run <run> --params <frozen json> [--key ste]

Reads the run's full CST VAL detection cache (`runs/systems/_cache/<run>/data_unified_v2/val`), runs the STE and argmax trackers
with the given settings (the frozen settings tuned on all of VAL, e.g. experiments/aerotrack_v3/frozen_x1_params.json), writes
runs/cst_anti_uav/val_predictions/<run>/<tracker>/<official sequence>.txt and prints Track A's mSA. Then:
    official_eval.py --subset val --tracker <run>-ste=runs/cst_anti_uav/val_predictions/<run>/ste --out ...
must give the same mSA (end-to-end check of the export used by the TEST passes). TEST is refused here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (HERE, REPO / "experiments" / "systems" / "scripts", REPO / "experiments" / "anti_uav" / "src"):
    sys.path.insert(0, str(p))
import cstlib  # noqa: E402
import track_eval as TE  # noqa: E402
from anti_uav_repro.eval.metrics import sequence_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--params", required=True, help="JSON with 'ste' and 'argmax' settings")
    ap.add_argument("--corpus", default="data_unified_v2")
    a = ap.parse_args()
    params = json.loads(Path(a.params).read_text(encoding="utf-8"))
    _, seqs = TE.load(a.run, a.corpus, split="val")
    keys = sorted(k for k in seqs if seqs[k]["group"] == "cst_IR")
    if len(keys) != 40:
        raise SystemExit(f"expected the 40 official CST val sequences, found {len(keys)}")
    out = REPO / "runs" / "cst_anti_uav" / "val_predictions" / a.run
    res = {}
    for kind in ("ste", "argmax"):
        sa = {}
        for k in keys:
            it = seqs[k]
            pred = TE.run_tracker(kind, params[kind], it)
            name = it["seq"].label_path.parent.name
            cstlib.write_official(out / kind, name, pred["res"])
            mask = np.arange(it["seq"].n_frames) >= pred["start_frame"]
            sa[name] = 100 * sequence_metrics(pred["res"], it["seq"].exist, it["seq"].gt_rect, mask=mask)["SA"]
        res[kind] = {"mSA_track_a": float(np.mean(list(sa.values()))), "per_sequence_SA": sa}
        print(f"{a.run} {kind}: Track A mSA {res[kind]['mSA_track_a']:.4f} over {len(sa)} sequences -> {out / kind}")
    (out / "track_a_scores.json").write_text(json.dumps({"run": a.run, "params": params.get("ste") and
                                                         {k: params[k] for k in ("ste", "argmax")}, **res}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
