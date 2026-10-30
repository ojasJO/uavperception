"""Flight Deck core tests: catalog, categories, presets, recorded replay, evidence, streaming adapters, the experiment runner (sequential,
lockstep, cancellation, failure, invalid selections), the benchmark, forecasting and the job manager.

    venv\\Scripts\\python.exe -m unittest tests.test_flight_deck -v          (~2-3 min with CUDA; uses local data, skips when absent)
"""
import json
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aerolab import analysis, categories, data, evidence, recorded, sequences  # noqa: E402
from aerolab import benchmark as BM  # noqa: E402
from aerolab import flightdeck as FD  # noqa: E402
from aerolab import forecasting as FC  # noqa: E402
from aerolab import jobs  # noqa: E402
from aerolab.adapters import get_adapter  # noqa: E402
from aerolab.paths import STUDIO_INDEX  # noqa: E402
from aerolab.registry import Registry  # noqa: E402
from aerolab.test_lock import REPLAY_PURPOSE, TestSplitLocked  # noqa: E402

R = Registry()
X1 = "aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last"
SF0 = "aerotrack.detector.cst_x:v3_n__ttt_cst_s0_e20@last"
GT_NORMAL = "antiuav_repro.globaltrack:normal_ir@last"
HAVE_CST = (data.CST_ROOT / "val").exists()
HAVE_AU = (data.ANTIUAV_LABELS / "val").exists()


def iou_xyxy(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


class Catalog(unittest.TestCase):
    def test_featured_ids_exist_and_provenance(self):
        fs = categories.featured()
        self.assertGreaterEqual(len(fs), 4)
        for f in fs:
            for v in f["versions"]:
                self.assertIn(v, R.by_id, v)
            if f["versions"]:
                self.assertIn(f["default"], f["versions"])
            for c in f["categories"]:
                self.assertIn(c, categories.CATEGORIES)
            for e in f["evidence"]:
                R.experiment(e)                                       # raises if a cited record is missing
            for fm in f["failure_modes"]:
                self.assertTrue(fm["source"])
        x2 = next(f for f in fs if f["id"] == "x2_temporal")
        m = R.model(x2["default"])
        self.assertEqual(m["sha256"][:8], "e52a5645")                  # X1 = X2 temporal seed 0: one checkpoint, two roles
        self.assertIn("AeroTrack-X1", m["aliases"])
        self.assertIn("X2 temporal seed 0", m["aliases"])
        fc = next(f for f in fs if f["kind"] == "forecasting")
        self.assertEqual(fc["categories"][0], "forecasting")          # forecasters never featured as trackers

    def test_category_eligibility(self):
        ok, why = categories.eligibility("presence", R.model(GT_NORMAL))
        self.assertFalse(ok)
        self.assertIn("presence", why)
        self.assertTrue(categories.eligibility("tracking", R.model(GT_NORMAL))[0])
        self.assertFalse(categories.eligibility("tracking", R.model("aerotrack.forecaster.c1:c1_tfm_abs_s0@weights"))[0])
        self.assertTrue(categories.eligibility("forecasting", R.model("aerotrack.forecaster.c1:c1_tfm_abs_s0@weights"))[0])
        self.assertFalse(categories.eligibility("temporal", R.model(GT_NORMAL))[0])
        no_adapter = next(m for m in R.models if not m.get("adapter"))
        self.assertFalse(categories.runnable(no_adapter)[0])
        backup_only = next(m for m in R.models if m["model_id"].endswith("~hpc") and m["status"] == "validated")
        ok, why = categories.runnable(backup_only)
        self.assertFalse(ok)
        self.assertIn("not verified", why)

    def test_status_labels_never_overclaim(self):
        lab = categories.status_labels(R.model(X1), R, {})
        for x in ("registered", "integrity-verified", "load-verified", "inference-verified", "historical results"):
            self.assertIn(x, lab)
        self.assertNotIn("evaluation-verified", lab)                  # no check supplied -> no claim
        hpc = next(m for m in R.models if m["model_id"].endswith("~hpc") and m["status"] == "validated")
        self.assertNotIn("inference-verified", categories.status_labels(hpc, R, {}))
        onnx = next(m for m in R.models if m["family_id"] == "aerotrack.deploy")
        self.assertIn("recorded results only", categories.status_labels(onnx, R, {}))


@unittest.skipUnless(HAVE_CST and HAVE_AU, "datasets absent")
class Presets(unittest.TestCase):
    def test_index_alignment_with_labels(self):
        idx = {s["name"]: s for s in sequences.sequence_index("cst2025", "val")}
        for name in ("building_41", "jungle_6"):
            lab = json.loads((data.CST_ROOT / "val" / name / "IR_label.json").read_text(encoding="utf-8"))
            s = data.load_sequence("cst2025", "val", name)
            self.assertEqual(idx[name]["n_frames"], len(lab["exist"]))
            self.assertEqual(s.n_frames, len(lab["exist"]))
            files = sorted(p.name for p in (data.CST_ROOT / "val" / name).glob("*.jpg"))
            self.assertEqual(len(files), s.n_frames)                   # one frame file per label row
            self.assertEqual(s.frame_path(0).name, files[0])           # frame order = sorted file names
            self.assertEqual(s.frame_path(s.n_frames - 1).name, files[-1])
            self.assertEqual(idx[name]["n_absent"], int(sum(1 for e in lab["exist"][s.start_frame:] if not e)))
            self.assertEqual(idx[name]["frame_attrs"].get("OC", 0), int(sum(1 for v in lab["OC"] if v)))

    def test_presets_are_grounded(self):
        idx = {s["name"]: s for s in sequences.sequence_index("cst2025", "val")}
        P = sequences.resolve_preset("target_absent", "cst2025", "val")
        self.assertEqual(P["basis"], "official_annotation")
        self.assertTrue(P["items"])
        self.assertTrue(all(idx[i["sequence"]]["n_absent"] > 0 for i in P["items"]))
        P = sequences.resolve_preset("small_targets", "cst2025", "val")
        self.assertEqual(P["basis"], "derived_from_gt")
        self.assertTrue(all(idx[i["sequence"]]["n_tiny"] / idx[i["sequence"]]["n_visible"] >= 0.25 for i in P["items"]))
        P = sequences.resolve_preset("large_targets", "cst2025", "val")
        self.assertTrue(all(idx[i["sequence"]]["n_large"] >= 20 for i in P["items"]))
        P = sequences.resolve_preset("occlusion_reappearance", "cst2025", "val")
        self.assertTrue(all(idx[i["sequence"]]["reappearance_frames"] or idx[i["sequence"]]["frame_attrs"].get("OC") for i in P["items"]))
        P = sequences.resolve_preset("difficult_cases", "cst2025", "val")
        self.assertEqual(P["basis"], "recorded_results")
        self.assertEqual(len(P["items"]), 8)

    def test_unavailable_presets_say_why(self):
        P = sequences.resolve_preset("clutter", "antiuav2021", "val")
        self.assertFalse(P["available"])
        self.assertIn("no clutter", P["reason"])
        P = sequences.resolve_preset("long_sequence", "antiuav2021", "val")
        self.assertFalse(P["available"])
        P = sequences.resolve_preset("custom", "cst2025", "val", custom=["building_41", "not_a_sequence"])
        self.assertEqual([i["sequence"] for i in P["items"]], ["building_41"])


class TestLock(unittest.TestCase):
    def test_test_split_refused_everywhere_but_replay(self):
        with self.assertRaises(TestSplitLocked):
            data.load_sequence("cst2025", "test", "jungle_18")
        with self.assertRaises(TestSplitLocked):
            FD.run(FD.build_config("tracking", "cst2025", "test", "IR", ["jungle_18"], [{"model_id": X1, "tracker": "ste"}]), R)
        errs, _ = FD.validate(FD.build_config("tracking", "cst2025", "test", "IR", ["jungle_18"], [{"model_id": X1, "tracker": "ste"}]), R)
        self.assertTrue(errs and "locked" in errs[0])
        self.assertTrue(BM.validate(BM.build_config("cst2025", "test", "IR", "jungle_18", [{"model_id": X1, "tracker": "ste"}]), R))
        self.assertTrue(FC.validate(FC.build_config(["x"], ["aerotrack.forecaster.c1:c1_tfm_abs_s0@weights"], split="test"), R))
        from aerolab.evaluation import evaluate_sot
        with self.assertRaises(TestSplitLocked):
            evaluate_sot(X1, "cst2025", "test", "IR", ["jungle_18"])

    @unittest.skipUnless((data.CST_ROOT / "test").exists(), "CST test frames absent")
    def test_replay_reads_test_without_a_model(self):
        s = data.load_sequence("cst2025", "test", "jungle_18", "IR", purpose=REPLAY_PURPOSE)
        self.assertGreater(s.n_frames, 0)


class Recorded(unittest.TestCase):
    def test_sources_match_registry_hashes(self):
        S = recorded.sources()
        self.assertGreaterEqual(len(S), 30)
        for s in S:
            if s["model_id"] in R.by_id and s["checkpoint_sha256"]:
                self.assertEqual(R.by_id[s["model_id"]]["sha256"], s["checkpoint_sha256"], s["id"])
            self.assertTrue(recorded.sequences_of(s), s["id"])

    @unittest.skipUnless((data.CST_ROOT / "test").exists(), "CST test frames absent")
    def test_replay_recomputes_the_registered_per_sequence_values(self):
        pairs = [("cst:test:x1:ste", "cst:x1:ste"), ("cst:test:v3_n__causal_g4_cst_s1_e20:ste", "cst:x2:v3_n__causal_g4_cst_s1_e20:ste")]
        pairs += [(s["id"], s["record_id"]) for s in recorded.sources() if s["id"].startswith("cst:test:released:")]   # JSON and text formats
        for sid, rid in pairs:
            src = recorded.source(sid)
            for name in ("jungle_18", "building_66"):
                q = data.load_sequence("cst2025", "test", name, "IR", purpose=REPLAY_PURPOSE)
                p = recorded.load(src, name)
                sa = analysis.sequence_summary(p["res_xywh"], q.exist, q.gt_rect, q.start_frame)["SA"]
                self.assertAlmostEqual(sa, R.experiment(rid)["per_sequence"][name], places=9, msg=f"{sid} {name}")

    @unittest.skipUnless(HAVE_AU, "Anti-UAV absent")
    def test_direction_b_val_replay(self):
        src = recorded.source("antiuav:val:normal_ir")
        per = json.loads((REPO / "experiments/anti_uav/runs/normal_ir/eval_val_IR_last/metrics.json").read_text(encoding="utf-8"))["per_sequence"]
        for name in recorded.sequences_of(src)[:5]:
            q = data.load_sequence("antiuav2021", "val", name)
            p = recorded.load(src, name)
            sa = analysis.sequence_summary(p["res_xywh"], q.exist, q.gt_rect, p["start_frame"], None, p["processed"])["SA"]
            self.assertAlmostEqual(sa, 100 * per[name]["SA"], places=9)


class Evidence(unittest.TestCase):
    def test_reported_values(self):
        d = evidence.x2_minus_x1(R)
        self.assertEqual(round(d["mean_delta"], 2), 0.23)
        self.assertEqual([round(x, 2) for x in d["ci95"]], [-0.66, 1.14])
        self.assertEqual((d["wins"], d["losses"]), (32, 28))
        t = evidence.x2_seed_table(R)
        self.assertEqual([round(v, 2) for v in t["temporal causal_g4"]["per_seed"].values()], [60.08, 60.26, 60.60])
        self.assertEqual([round(v, 2) for v in t["single-frame (ttt)"]["per_seed"].values()], [46.83, 49.81, 49.79])
        self.assertEqual(round(t["temporal causal_g4"]["summary"]["mean"], 2), 60.31)
        self.assertEqual(round(t["single-frame (ttt)"]["summary"]["mean"], 2), 48.81)
        T3 = evidence.x2_tests()["T3"]["temporal_minus_single_frame_seed_mean"]
        self.assertEqual(round(T3["mean_delta"], 2), 11.51)
        self.assertEqual(round(R.experiment("cst:x1:argmax")["metrics"]["mSA"], 2), 59.70)

    def test_p1_tradeoff_and_verdict(self):
        h = evidence.p1_headline()
        self.assertEqual((round(100 * h["fa_from"], 1), round(100 * h["fa_to"], 1)), (27.6, 4.7))
        self.assertEqual((round(100 * h["miss_from"], 1), round(100 * h["miss_to"], 1)), (12.1, 38.7))
        self.assertEqual(evidence.p1()["verdict"], "NO_GAIN")

    def test_reproduction_tables(self):
        rows = {(r["model"], r["split"]): r for r in evidence.antiuav_reproduction(R)}
        self.assertEqual(rows[("normal", "val")]["verdict"], "reproduced")
        self.assertEqual(rows[("normal", "test")]["verdict"], "not reproduced")
        self.assertEqual(round(rows[("DFSC", "test")]["delta"], 2), 4.25)
        br = evidence.cst_baseline_reproduction(R)
        self.assertEqual(len(br), 21)
        self.assertLess(max(abs(r["delta"]) for r in br if r["delta"] is not None), 0.01)


class Analysis(unittest.TestCase):
    def test_episode_rules_on_synthetic_tracks(self):
        n = 60
        gt = [[100.0, 100.0, 20.0, 20.0]] * n
        exist = [1] * n
        res = [list(g) for g in gt]
        for k, t in enumerate(range(20, 30)):                         # gradual drift: IoU falls step by step
            res[t] = [100.0 + 2.0 * (k + 1), 100.0, 20.0, 20.0]
        for t in range(30, 40):
            res[t] = [300.0, 300.0, 20.0, 20.0]                         # then a wrong-object lock (IoU 0)
        for t in range(45, 52):
            res[t] = []                                                # sudden miss after tracked frames
        s = analysis.sequence_summary(res, exist, gt, 0)
        fi = s["episodes"]["failure_intervals"]
        self.assertEqual(len(fi), 2)
        self.assertEqual(fi[1]["onset"], "sudden")
        self.assertEqual(fi[1]["dominant"], "no_output")
        self.assertIn(fi[0]["dominant"], ("wrong_target", "poor_overlap"))
        self.assertEqual(s["episodes"]["loss_events"], 2)

    def test_presence_curve_and_latency(self):
        P = {"visible": np.array([1, 1, 1, 0, 0], bool), "score": np.array([0.9, 0.6, 0.2, 0.5, 0.1]), "top_iou": np.array([0.8, 0.7, 0.0, 0, 0]),
             "tracker_box": np.array([1, 1, 0, 1, 0], bool)}
        c = {r["threshold"]: r for r in analysis.presence_curve(P, [0.0, 0.3, 0.55, 1.0])}
        self.assertEqual(c[0.0]["false_alarm_rate"], 1.0)
        self.assertAlmostEqual(c[0.55]["miss_rate"], 1 / 3)
        self.assertEqual(c[0.55]["false_alarm_rate"], 0.0)
        self.assertAlmostEqual(analysis.presence_auc(P), 5 / 6)
        self.assertEqual(analysis.tracker_presence(P)["false_alarm_rate"], 0.5)
        ls = analysis.latency_stats(list(range(1, 11)))
        self.assertNotIn("p95", ls)                                    # too few samples for a tail percentile
        self.assertIn("p95", analysis.latency_stats(list(range(1, 31))))


@unittest.skipUnless(HAVE_CST and HAVE_AU, "datasets absent")
class Streaming(unittest.TestCase):
    def test_globaltrack_stream_equals_predict(self):
        m = R.model(GT_NORMAL)
        a = get_adapter(m["adapter"], R).load(m, "auto")
        try:
            q = data.load_sequence("antiuav2021", "val", "20190925_101846_1_4")
            pr = a.predict(q, dataset_id="antiuav2021", max_frames=15)
            fr = list(a.stream(q, end_frame=q.start_frame + 16, dataset_id="antiuav2021"))
            self.assertEqual(len(fr), 16)
            for f in fr:
                self.assertEqual(f["box_xyxy"], pr["frames"][f["frame"]]["box_xyxy"])
                self.assertEqual(set(f["timing_ms"]), {"read_preprocess", "model", "postprocess", "tracker", "total"})
        finally:
            a.unload()

    def test_detector_stream_matches_batched_predict(self):
        m = R.model(X1)
        a = get_adapter(m["adapter"], R).load(m, "auto")
        try:
            q = data.load_sequence("cst2025", "val", "building_41")
            pr = a.predict(q, dataset_id="cst2025", max_frames=30)
            fr = list(a.stream(q, end_frame=30))
            self.assertEqual([f["frame"] for f in fr], list(range(30)))
            self.assertEqual(fr[0]["mode"], "init")
            agree = sum((f["box_xyxy"] is None) == (pr["frames"][f["frame"]]["box_xyxy"] is None) for f in fr)
            self.assertEqual(agree, 30)
            ious = [iou_xyxy(f["box_xyxy"], pr["frames"][f["frame"]]["box_xyxy"]) for f in fr if f["box_xyxy"] and pr["frames"][f["frame"]]["box_xyxy"]]
            self.assertGreater(min(ious), 0.95)                       # batch 1 vs batch 16 autocast numerics only
            self.assertEqual(a.device, json.loads(json.dumps(a.device)))
        finally:
            a.unload()


@unittest.skipUnless(HAVE_CST, "CST absent")
class Runner(unittest.TestCase):
    def cfg(self, **kw):
        base = dict(category="tracking", dataset_id="cst2025", split="val", modality="IR", sequences_=["building_41"],
                    models=[{"model_id": X1, "tracker": "ste"}, {"model_id": SF0, "tracker": "ste"}], max_frames=20)
        base.update(kw)
        return FD.build_config(base.pop("category"), base.pop("dataset_id"), base.pop("split"), base.pop("modality"), base.pop("sequences_"),
                               base.pop("models"), **base)

    def test_two_models_saved_reopened_identical(self):
        lines0 = len(STUDIO_INDEX.read_text(encoding="utf-8").splitlines()) if STUDIO_INDEX.exists() else 0
        seen = {}
        man = FD.run(self.cfg(), R, on_event=lambda e: seen.setdefault(e["slug"], []).append(e["record"]) if e["type"] == "frame" else None)
        self.assertEqual(man["status"], "completed")
        self.assertEqual(man["schema"], "aerolab/flightdeck_experiment/1")
        self.assertEqual(man["evaluation_protocol_id"], "studio.sot_smoke.v1")
        for m in man["models"]:
            self.assertEqual(m["status"], "completed")
            self.assertIn(m["device"], ("cuda", "cpu"))
            self.assertEqual(m["checkpoint_sha256"], R.model(m["model_id"])["sha256"])
            p = FD.load_prediction(man["experiment_id"], m["slug"], "building_41")
            self.assertTrue(p["complete"])
            self.assertEqual(p["end_frame"], 21)                        # init frame + 20
            self.assertEqual(p["frames"], json.loads(json.dumps(seen[m["slug"]])))   # replay == what was streamed live
            again = analysis.sequence_summary(p["res_xywh"], *(lambda q: (q.exist, q.gt_rect))(data.load_sequence("cst2025", "val", "building_41")),
                                              p["start_frame"], p["end_frame"], p["processed"])
            self.assertAlmostEqual(again["SA"], m["per_sequence"]["building_41"]["SA"], places=9)
        pair = next(iter(man["comparison"]["pairs"].values()))
        self.assertEqual(pair["verdict"], "comparable")
        self.assertIsNone(pair["paired_delta_SA"]["ci95"])             # 1 sequence -> no bootstrap CI
        lines1 = STUDIO_INDEX.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines1) - lines0, 2)
        recs = [json.loads(x) for x in lines1[-2:]]
        self.assertEqual({r["experiment_group"] for r in recs}, {man["experiment_id"]})
        self.assertTrue((REPO / man["outputs"]["dir"] / "config.json").exists())

    def test_lockstep_equals_sequential(self):
        a = FD.run(self.cfg(max_frames=12), R)
        b = FD.run(self.cfg(max_frames=12, execution="lockstep"), R)
        for i in range(2):
            pa = FD.load_prediction(a["experiment_id"], a["models"][i]["slug"], "building_41")
            pb = FD.load_prediction(b["experiment_id"], b["models"][i]["slug"], "building_41")
            self.assertEqual(pa["res_xywh"], pb["res_xywh"])

    def test_cancellation_keeps_a_consistent_record(self):
        ev = threading.Event()
        man = FD.run(self.cfg(max_frames=None, sequences_=["building_41", "building_42"]), R,
                     on_event=lambda e: ev.set() if e["type"] == "frame" and e["frames_done"] == 15 else None, cancel=ev)
        self.assertEqual(man["status"], "cancelled")
        m0, m1 = man["models"]
        self.assertEqual(m0["status"], "cancelled")
        self.assertEqual(m1["status"], "not_started")
        p = FD.load_prediction(man["experiment_id"], m0["slug"], "building_41")
        self.assertFalse(p["complete"])
        self.assertNotIn("building_41", m0["per_sequence"])            # no metric for an interrupted sequence
        self.assertEqual(json.loads((REPO / man["outputs"]["dir"] / "manifest.json").read_text(encoding="utf-8"))["status"], "cancelled")

    def test_one_failing_model_does_not_hide_the_other(self):
        from aerolab.adapters.globaltrack import GlobalTrackAdapter  # noqa: F401
        from aerolab.adapters import aerotrack_detector as AD
        orig = AD.AeroTrackDetectorAdapter.stream
        calls = {"n": 0}

        def flaky(self, *a, **k):
            calls["n"] += 1
            if self.record["model_id"] == SF0:
                raise RuntimeError("simulated decode failure")
            yield from orig(self, *a, **k)
        with mock.patch.object(AD.AeroTrackDetectorAdapter, "stream", flaky):
            man = FD.run(self.cfg(max_frames=8), R)
        self.assertEqual(man["status"], "partial")
        st = {m["model_id"]: m for m in man["models"]}
        self.assertEqual(st[X1]["status"], "completed")
        self.assertEqual(st[SF0]["status"], "failed")
        self.assertIn("simulated decode failure", st[SF0]["error"])

    def test_invalid_selections(self):
        cases = [(dict(models=[{"model_id": "nope:x@y", "tracker": "ste"}]), "unknown model"),
                 (dict(models=[{"model_id": X1, "tracker": "ste"}] * 2), "selected twice"),
                 (dict(models=[{"model_id": X1, "tracker": "globaltrack"}]), "not available"),
                 (dict(models=[{"model_id": X1, "tracker": t} for t in ("ste", "argmax")] + [{"model_id": SF0, "tracker": t} for t in ("ste", "argmax")]
                       + [{"model_id": GT_NORMAL, "tracker": "globaltrack"}]), "limit"),
                 (dict(sequences_=[]), "at least one sequence"),
                 (dict(sequences_=["no_such_seq"]), "not in"),
                 (dict(category="temporal", models=[{"model_id": GT_NORMAL, "tracker": "globaltrack"}]), "Temporal"),
                 (dict(category="forecasting"), "not run by the tracking runner"),
                 (dict(device="gpu"), "device must be")]
        for kw, needle in cases:
            errs, _ = FD.validate(self.cfg(**kw), R)
            self.assertTrue(any(needle in e for e in errs), f"{kw} -> {errs}")

    def test_explicit_cuda_refused_without_cuda(self):
        code = ("import sys; sys.path.insert(0, %r); from aerolab import flightdeck as FD; from aerolab.registry import Registry; "
                "e, _ = FD.validate(FD.build_config('tracking', 'cst2025', 'val', 'IR', ['building_41'], [{'model_id': %r, 'tracker': 'ste'}], device='cuda'), Registry()); "
                "print('ERR' + '|'.join(e))") % (str(REPO), X1)
        p = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), cwd=REPO, capture_output=True, text=True, timeout=300)
        line = next(x for x in p.stdout.splitlines() if x.startswith("ERR"))
        self.assertIn("CUDA requested but", line)
        self.assertIn("cpu", line)


@unittest.skipUnless(HAVE_CST, "CST absent")
class BenchmarkForecastJobs(unittest.TestCase):
    def test_benchmark_cpu(self):
        man = BM.run_benchmark(BM.build_config("cst2025", "val", "IR", "building_41", [{"model_id": X1, "tracker": "ste"}], ["cpu"], n_frames=6, warmup=2), R)
        r = man["results"][0]
        self.assertEqual(man["status"], "completed")
        self.assertEqual(r["device"], "cpu")
        self.assertFalse(r["cuda_synchronized"])
        self.assertEqual(r["n_timed_frames"], 6)
        self.assertIsNone(r["peak_gpu_mb"])
        self.assertGreater(r["first_frame_ms"], 0)
        self.assertEqual(set(r["stages"]), set(BM.STAGES))
        self.assertNotIn("p95", r["stages"]["total"])

    def test_forecast_same_anchors(self):
        keys = data.list_sequences("trackc_trajectories", "val")
        k = [x for x in keys if x.startswith("cst_")][:1]
        man = FC.run_forecast(FC.build_config(k, ["aerotrack.forecaster.c1:c1_tfm_abs_s0@weights", "aerotrack.forecaster.c1:c1_gru_abs_s0@weights",
                                                  "aerotrack.forecaster.c1c:c1c_tfm_perdom_antiuav_IR_s0@weights"], max_anchors=100, keep_anchors=10), R)
        self.assertTrue(man["anchors_identical"])
        a = json.loads((REPO / man["outputs"]["dir"] / "predictions" / man["models"][0]["slug"] / f"{k[0]}.json").read_text(encoding="utf-8"))
        b = json.loads((REPO / man["outputs"]["dir"] / "predictions" / man["models"][1]["slug"] / f"{k[0]}.json").read_text(encoding="utf-8"))
        self.assertEqual(a["anchor_t_all"], b["anchor_t_all"])
        self.assertEqual(len(a["anchors"]["t"]), 10)
        per = json.loads((REPO / man["outputs"]["dir"] / "per_sequence.json").read_text(encoding="utf-8"))
        self.assertEqual(per[man["models"][2]["slug"]][k[0]]["status"], "skipped")     # domain-isolated Anti-UAV model on CST

    def test_job_manager_runs_and_cancels(self):
        M = jobs.manager()
        j = M.submit(FD.build_config("tracking", "cst2025", "val", "IR", ["building_42"], [{"model_id": X1, "tracker": "ste"}], max_frames=15))
        q = M.submit(FD.build_config("tracking", "cst2025", "val", "IR", ["building_42"], [{"model_id": X1, "tracker": "ste"}], max_frames=15))
        M.cancel(q.id)
        jobs.wait(j, 300)
        jobs.wait(q, 60)
        self.assertEqual(j.status, "completed")
        self.assertEqual(q.status, "cancelled")
        s = j.snapshot()
        lv = next(iter(s["live"].values()))
        self.assertEqual(lv["frames_done"], 16)
        self.assertEqual(lv["frames_total"], 16)
        self.assertTrue(s["experiment_id"].startswith("fd-"))


if __name__ == "__main__":
    unittest.main()
