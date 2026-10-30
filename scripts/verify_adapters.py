"""Load + tiny VAL inference for every runnable registered model; records the evidence in results/registry/verification/adapter_checks.json.

    venv\\Scripts\\python.exe scripts\\verify_adapters.py              # default scope (all but intermediate epoch checkpoints)
    venv\\Scripts\\python.exe scripts\\verify_adapters.py --family aerotrack.forecaster.c1
    venv\\Scripts\\python.exe scripts\\verify_adapters.py --all        # also every intermediate epoch checkpoint (slow: 471 MB each)
then rebuild the registry (scripts/build_workspace_registry.py) so models.json shows the load / inference status.

Per model: SHA-256 checked by the adapter before loading; the same tiny sample is predicted twice (determinism); output structure
checked (one record per processed frame, boxes of 4 numbers, the model id / checksum carried through); resources released.
A model that fails is recorded as failed with the error -- it never stops the others.
"""
import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from aerolab import data  # noqa: E402
from aerolab.adapters import get_adapter  # noqa: E402
from aerolab.registry import Registry  # noqa: E402

OUT = REPO / "results/registry" / "verification" / "adapter_checks.json"


def sample_for(m):
    if m["adapter"] == "trackc_forecaster":
        dom = None
        for d in ("antiuav_IR", "antiuav_RGB", "cst"):
            if f"perdom_{d}" in m["run_id"]:
                dom = d
        seqs = data.trackc_sequences("val")
        s = next(x for x in seqs if dom is None or x["domain"] == dom)
        return {"kind": "forecast", "seq": s, "desc": f"trackc VAL {s['key']} (100 anchors)"}
    if m["adapter"] == "globaltrack_repro":
        mod = "RGB" if "rgb" in m["run_id"].lower() else "IR"
        name = data.list_sequences("antiuav2021", "val", mod)[0]
        return {"kind": "sot", "dataset": "antiuav2021", "modality": mod, "seq": data.load_sequence("antiuav2021", "val", name, mod),
                "max_frames": 5, "desc": f"Anti-UAV VAL {name} {mod}, 5 frames"}
    name = data.list_sequences("cst2025", "val")[0]
    return {"kind": "sot", "dataset": "cst2025", "modality": "IR", "seq": data.load_sequence("cst2025", "val", name), "max_frames": 8,
            "desc": f"CST VAL {name}, 8 frames"}


def run_once(a, m, smp):
    if smp["kind"] == "forecast":
        r = a.predict(smp["seq"], max_anchors=100)
        assert r["n_anchors"] > 0 and r["metrics"]["learned"]["ade_px"] > 0
        sig = np.round(np.asarray(r["example"]["mean"]), 4)
    else:
        r = a.predict(smp["seq"], dataset_id=smp["dataset"], split="val", modality=smp["modality"], max_frames=smp["max_frames"])
        assert len(r["frames"]) == r["n_frames_processed"] > 0
        for f in r["frames"]:
            assert f["box_xyxy"] is None or (len(f["box_xyxy"]) == 4 and all(np.isfinite(f["box_xyxy"])))
        sig = np.array([f["box_xyxy"] or [-1] * 4 for f in r["frames"]], dtype=float).round(3)
    assert r["model_id"] == m["model_id"] and r["checkpoint_sha256"] == m["sha256"]
    return r, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="auto")
    a_ = ap.parse_args()
    R = Registry()
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"models": {}}
    res = old.get("models", {})
    todo = [m for m in R.models if m.get("adapter") and m["status"] not in ("smoke", "invalid") and m.get("available_in_workspace")
            and (a_.family is None or m["family_id"] == a_.family) and (a_.all or not m["checkpoint_tag"].startswith("epoch_"))
            and m["primary_path"].endswith(".pt")]
    print(f"{len(todo)} models to verify")
    for i, m in enumerate(todo):
        t0 = dt.datetime.now()
        ev = {"checked": t0.astimezone().isoformat(timespec="seconds"), "sha256": m["sha256"]}
        ad = None
        try:
            smp = sample_for(m)
            ad = get_adapter(m["adapter"], R)
            ad.load(m, a_.device)
            ev.update(load="passed", device=ad.device, load_seconds=round(ad.load_s, 2), weights_path=ad.get_metadata()["checkpoint_path"])
            r1, s1 = run_once(ad, m, smp)
            r2, s2 = run_once(ad, m, smp)
            same = s1.shape == s2.shape and np.allclose(s1, s2, atol=1e-3)
            ev.update(inference="passed", sample=smp["desc"], deterministic=bool(same), warnings=r1.get("warnings", []))
            if not same:
                ev["inference_note"] = "two identical calls gave different outputs (non-deterministic GPU kernels?)"
        except Exception as e:  # noqa: BLE001
            ev.setdefault("load", "failed")
            ev["inference"] = "failed" if ev["load"] == "passed" else "not_verified"
            ev["error"] = f"{type(e).__name__}: {e}"[:600]
            ev["trace"] = traceback.format_exc()[-1500:]
        finally:
            if ad is not None:
                ad.unload()
        ev["seconds"] = round((dt.datetime.now() - t0).total_seconds(), 1)
        res[m["model_id"]] = ev
        print(f"[{i + 1}/{len(todo)}] {m['model_id']}: load {ev['load']} inference {ev.get('inference')} ({ev['seconds']} s)"
              + (f" ERROR {ev.get('error')}" if ev.get("error") else ""), flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                                   "generator": "scripts/verify_adapters.py", "models": res}, indent=1), encoding="utf-8")
    failed = [k for k in res if res[k].get("load") == "failed" or res[k].get("inference") == "failed"]
    print(f"done: {len(res)} recorded, {len(failed)} failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
