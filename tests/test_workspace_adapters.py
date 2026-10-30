"""Adapter / evaluation-service tests on real local artifacts (run: venv\\Scripts\\python.exe -m unittest tests.test_workspace_adapters -v).

Each adapter is tested on its own: checksum gate, load, small VAL inference, output structure, determinism, refusal of incompatible
inputs, and -- for the AeroTrack detector -- parity with the HPC-recorded VAL detection cache and SA fidelity against the tracker run
on those cached detections. Skipped (not faked) when the data or the checkpoints are absent. ~1-2 minutes with CUDA."""
import copy
import glob
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aerolab import data, schemas  # noqa: E402
from aerolab.adapters import AdapterError, compatibility, get_adapter  # noqa: E402
from aerolab.registry import Registry  # noqa: E402

R = Registry()
X1 = "aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last"
HAVE_CST = (REPO / "data" / "cst_anti_uav_full" / "CST-AntiUAV" / "CST-AntiUAV" / "val").exists()
HAVE_AU = (REPO / "data" / "anti_uav" / "val").exists()
CACHE = sorted(glob.glob(str(REPO / "runs/systems/_cache/v3_n__causal_g4_cst_s0_e20/*/val/index.json")))


def iou_xyxy(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix + 1e-9)


@unittest.skipUnless(HAVE_CST, "CST data absent")
class AeroTrackDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = get_adapter("aerotrack_detector", R).load(R.model(X1))
        cls.seq = data.load_sequence("cst2025", "val", "building_41")

    @classmethod
    def tearDownClass(cls):
        cls.a.unload()

    def test_metadata_and_window(self):
        md = self.a.get_metadata()
        self.assertEqual(md["checkpoint_sha256"], R.model(X1)["sha256"])
        self.assertEqual((self.a.window, self.a.gap), ("causal", 4))

    def test_prediction_structure_and_determinism(self):
        r1 = self.a.predict(self.seq, "cst2025", "val", "IR", max_frames=20)
        r2 = self.a.predict(self.seq, "cst2025", "val", "IR", max_frames=20)
        self.assertEqual(r1["n_frames_processed"], self.seq.start_frame + 20)
        self.assertEqual(len(r1["frames"]), r1["n_frames_processed"])
        for f in r1["frames"]:
            rec = {"frame": f["frame"], "present": f["present"], "box_xyxy": f["box_xyxy"], "score": f["score"], "mode": f["mode"],
                   "model_id": r1["model_id"], "checkpoint_sha256": r1["checkpoint_sha256"]}
            self.assertEqual(schemas.errors(rec, schemas.PREDICTION_SCHEMA), [])
        self.assertEqual([f["box_xyxy"] for f in r1["frames"]], [f["box_xyxy"] for f in r2["frames"]])
        self.assertIn("frozen_x2_params", r1["params_source"])

    @unittest.skipUnless(CACHE, "recorded VAL detection cache absent")
    def test_parity_with_recorded_cache_and_sa_fidelity(self):
        sys.path.insert(0, str(REPO / "experiments" / "systems" / "scripts"))
        import track_eval as TE
        from aerolab.adapters.aerotrack_detector import frozen_tracker_params
        from aerolab.evaluation import sot_sequence_metrics
        idx = json.loads(Path(CACHE[0]).read_text(encoding="utf-8"))
        key = next(k for k in idx["sequences"] if k.endswith("building_41"))
        z = np.load(Path(CACHE[0]).parent / f"{key}.npz")
        off = z["offsets"]
        n = 300
        dets, _, _ = self.a.detect(self.seq, n)
        ious = []
        for t in range(n):
            cs, cc = z["scores"][off[t]:off[t + 1]], z["classes"][off[t]:off[t + 1]]
            b, sc, cl = dets[t]
            if (cc == 0).any() and (cl == 0).any():
                i1 = int(np.argmax(np.where(cc == 0, cs, -1)))
                i2 = int(np.argmax(np.where(cl == 0, sc, -1)))
                ious.append(iou_xyxy(z["boxes"][off[t]:off[t + 1]][i1], b[i2]))
            else:
                self.assertEqual((cc == 0).any(), (cl == 0).any(), f"presence disagreement at frame {t}")
        self.assertGreater(np.median(ious), 0.98)                  # fp16 (V100 cache) vs bf16 (RTX 5050): close, not bit-equal
        from anti_uav_repro.data.antiuav import Sequence
        sub = Sequence(name="b41", split="val", modality="IR", label_path=self.seq.label_path, frames_dir=self.seq.frames_dir,
                       exist=list(self.seq.exist[:n]), gt_rect=list(self.seq.gt_rect[:n]))
        ste, _, _ = frozen_tracker_params("v3_n__causal_g4_cst_s0_e20")
        cached = [(z["boxes"][off[i]:off[i + 1]].astype(float), z["scores"][off[i]:off[i + 1]].astype(float), z["classes"][off[i]:off[i + 1]].astype(int))
                  for i in range(n)]
        ref = TE.run_tracker("ste", ste, {"seq": sub, "dets": cached})
        ref.update(n_frames_processed=n, res_xywh=ref["res"])
        live = self.a.predict(self.seq, "cst2025", "val", "IR", max_frames=n - self.seq.start_frame)
        self.assertAlmostEqual(sot_sequence_metrics(live, self.seq)["SA"], sot_sequence_metrics(ref, sub)["SA"], delta=0.5)

    def test_refusals(self):
        c = compatibility(R.model(X1), "trackc_trajectories", None, R)
        self.assertFalse(c.ok)
        smoke = next(m for m in R.models if m["status"] == "smoke" and m["adapter"] == "aerotrack_detector")
        self.assertFalse(compatibility(smoke, "cst2025", "IR", R).ok)
        self.assertFalse(compatibility(R.model(X1), "cst2025", "RGB", R).ok)

    def test_checksum_gate(self):
        bad = copy.deepcopy(R.model(X1))
        bad["sha256"] = "0" * 64
        with self.assertRaises(AdapterError):
            get_adapter("aerotrack_detector", R).load(bad)

    def test_cf_detector_resolves_copy_with_run_meta(self):
        m = R.model("aerotrack.detector.v2:v2_causal_g4_s0_e20@last")
        p = get_adapter("aerotrack_detector", R)._resolve_workspace_path(m, R, "run_meta.json")
        self.assertTrue((p.parent.parent / "run_meta.json").exists())
        self.assertNotIn("hpc_recovery", str(p))


@unittest.skipUnless(HAVE_AU, "Anti-UAV data absent")
class GlobalTrack(unittest.TestCase):
    def test_load_predict_unload(self):
        m = R.model("antiuav_repro.globaltrack:normal_ir@last")
        a = get_adapter("globaltrack_repro", R).load(m)
        seq = data.load_sequence("antiuav2021", "val", data.list_sequences("antiuav2021", "val", "IR")[0], "IR")
        r = a.predict(seq, "antiuav2021", "val", "IR", max_frames=5)
        self.assertEqual(r["checkpoint_sha256"], m["sha256"])
        self.assertTrue(all(f["box_xyxy"] is not None for f in r["frames"][seq.start_frame:]))
        self.assertTrue(any("never reports 'absent'" in w for w in r["warnings"]))
        a.unload()
        self.assertIsNone(a.model)

    def test_invalid_model_refused(self):
        m = R.model("antiuav_repro.globaltrack:phaseB_dfsc_ir_ism_cross@last")
        self.assertFalse(compatibility(m, "antiuav2021", "IR", R).ok)


class Forecaster(unittest.TestCase):
    def test_predict_and_domain_guard(self):
        m = R.model("aerotrack.forecaster.c1c:c1c_tfm_perdom_cst_s0@weights")
        a = get_adapter("trackc_forecaster", R).load(m)
        seqs = data.trackc_sequences("val")
        cst = next(s for s in seqs if s["domain"] == "cst")
        ir = next(s for s in seqs if s["domain"] == "antiuav_IR")
        r1, r2 = a.predict(cst, max_anchors=100), a.predict(cst, max_anchors=100)
        self.assertEqual(r1["example"]["mean"], r2["example"]["mean"])
        self.assertIn("dcv", r1["metrics"])
        with self.assertRaises(AdapterError):
            a.predict(ir, max_anchors=10)
        onnx = next(x for x in R.models if x["family_id"] == "aerotrack.forecaster.c1" and x["primary_path"].endswith(".onnx"))
        self.assertFalse(compatibility(onnx, "trackc_trajectories", None, R).ok)


@unittest.skipUnless(HAVE_CST, "CST data absent")
class EvaluationService(unittest.TestCase):
    def test_manifest_and_history(self):
        from aerolab.evaluation import evaluate_sot
        man = evaluate_sot(X1, "cst2025", "val", "IR", ["building_42"], max_frames=15)
        self.assertEqual(man["status"], "completed", man.get("error"))
        self.assertEqual(schemas.errors(man, schemas.STUDIO_MANIFEST_SCHEMA), [])
        self.assertEqual(man["evaluation_protocol_id"], "studio.sot_smoke.v1")
        self.assertTrue((REPO / man["per_sample_results_path"]).exists())
        self.assertIn(man["experiment_id"], {e["experiment_id"] for e in R.studio_experiments()})

    def test_failure_is_recorded_not_raised(self):
        from aerolab.evaluation import evaluate_sot
        man = evaluate_sot(X1, "cst2025", "val", "IR", ["no_such_sequence"], max_frames=5)
        self.assertEqual(man["status"], "failed")
        self.assertTrue(man["error"])


CPU_PROBE = """
import json, sys
sys.path.insert(0, %r)
import torch
from aerolab.adapters.base import AdapterError, pick_device
from aerolab.evaluation import evaluate_sot
out = {"cuda": torch.cuda.is_available(), "auto": pick_device("auto")}
try:
    pick_device("cuda"); out["cuda_refused"] = False
except AdapterError:
    out["cuda_refused"] = True
man = evaluate_sot(%r, "cst2025", "val", "IR", ["building_42"], max_frames=10)
out.update(status=man["status"], device=man["device"], error=man.get("error"))
print("PROBE" + json.dumps(out))
"""


@unittest.skipUnless(HAVE_CST, "CST data absent")
class CpuFallback(unittest.TestCase):
    def test_no_cuda_falls_back_to_cpu(self):
        # CUDA hidden in a child process; on Windows an empty CUDA_VISIBLE_DEVICES does not hide the GPU, -1 does
        import os
        import subprocess
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        p = subprocess.run([sys.executable, "-c", CPU_PROBE % (str(REPO), X1)], env=env, cwd=REPO, capture_output=True, text=True,
                           timeout=600)
        line = next((s for s in p.stdout.splitlines() if s.startswith("PROBE")), None)
        self.assertIsNotNone(line, p.stderr[-2000:])
        out = json.loads(line[5:])
        self.assertFalse(out["cuda"])
        self.assertEqual(out["auto"], "cpu")
        self.assertTrue(out["cuda_refused"])
        self.assertEqual(out["status"], "completed", out["error"])
        self.assertEqual(out["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
