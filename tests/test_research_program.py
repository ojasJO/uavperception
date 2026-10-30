"""Regression tests for the research-program components (CPU only; run: venv\\Scripts\\python.exe -m unittest tests.test_research_program -v).

Each test pins a property that an experiment's validity depends on: tracker behaviour, causal windows without future frames,
exact function-preserving initialisations, the corpus switch, the CST-val split rule, the pre-registered H4 decision rule, and the
Ultralytics YAML-name hazard."""
import importlib.util
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "c_model", REPO / "b_pipeline", REPO / "experiments" / "systems" / "scripts",
          REPO / "experiments" / "aerotrack_generalization" / "scripts", REPO / "experiments" / "aerotrack_v2" / "scripts",
          REPO / "experiments" / "anti_uav" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def box(cx, cy, w=8.0, h=6.0):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def dets(*items):
    """items: (box, score, cls) -> arrays as the cache stores them."""
    if not items:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    b, s, c = zip(*items)
    return np.array(b, float), np.array(s, float), np.array(c, int)


class TestSTE(unittest.TestCase):
    def setUp(self):
        import ste
        self.T = ste

    def test_tracks_coasts_and_declares_absence(self):
        trk = self.T.STE(self.T.STEParams(max_coast=3, det_thr=0.1, init_thr=0.4, reacq_thr=0.9))
        trk.reset(box(100, 100))
        modes = []
        for t in range(1, 11):                                   # constant velocity 2 px/frame, detected
            modes.append(trk.step(*dets((box(100 + 2 * t, 100), 0.8, 0)))[1])
        self.assertTrue(all(m == "track" for m in modes))
        out = [trk.step(*dets()) for _ in range(3)]              # 3 misses with max_coast 3 -> coasting, box kept
        self.assertEqual([o[1] for o in out], ["coast"] * 3)
        self.assertIsNotNone(out[-1][0])
        self.assertEqual(trk.step(*dets())[1], "absent")         # 4th miss -> absent, track dropped
        self.assertIsNone(trk.kf)

    def test_gate_rejects_far_low_score_and_reacquires_high_score(self):
        trk = self.T.STE(self.T.STEParams(reacq_thr=0.7, det_thr=0.1, max_coast=10))
        trk.reset(box(100, 100))
        for t in range(1, 6):
            trk.step(*dets((box(100, 100), 0.9, 0)))
        self.assertEqual(trk.step(*dets((box(400, 300), 0.5, 0)))[1], "coast")      # far + below reacq -> ignored
        b, mode, _ = trk.step(*dets((box(400, 300), 0.95, 0)))                      # far + above reacq -> jump
        self.assertEqual(mode, "reacquired")
        np.testing.assert_allclose(b, box(400, 300))

    def test_birds_and_low_scores_ignored(self):
        trk = self.T.STE(self.T.STEParams(det_thr=0.2, init_thr=0.4))
        trk.reset(None)
        self.assertEqual(trk.step(*dets((box(50, 50), 0.9, 1)))[1], "absent")       # bird class never starts a track
        self.assertEqual(trk.step(*dets((box(50, 50), 0.15, 0)))[1], "absent")      # below det_thr
        self.assertEqual(trk.step(*dets((box(50, 50), 0.5, 0)))[1], "init")

    def test_argmax_baseline(self):
        b = self.T.argmax_step(*dets((box(10, 10), 0.3, 0), (box(20, 20), 0.6, 0), (box(30, 30), 0.99, 1)), 0.25)
        np.testing.assert_allclose(b, box(20, 20))
        self.assertIsNone(self.T.argmax_step(*dets((box(10, 10), 0.2, 0)), 0.25))

    def test_kalman_forecast_converges_on_constant_velocity(self):
        kf = self.T.KalmanCV(box(0, 0))
        for t in range(1, 40):
            kf.predict()
            kf.update(box(3 * t, -1 * t))
        mean, cov = kf.forecast(5)
        np.testing.assert_allclose(mean[-1], [3 * 44, -44], atol=0.5)
        self.assertTrue(np.all(np.linalg.eigvalsh(cov[-1]) > 0))


class TestCausalWindow(unittest.TestCase):
    def test_no_future_frame_and_offsets(self):
        from stacked_dataset import build_split_samples
        from train import MANIFEST, SPLITS
        a = build_split_samples(MANIFEST, SPLITS, "val", temporal_gap=4, datasets=["cst"])
        b = build_split_samples(MANIFEST, SPLITS, "val", temporal_gap=8, datasets=["cst"])
        self.assertEqual([(s["seq_key"], s["frame_idx"]) for s in a], [(s["seq_key"], s["frame_idx"]) for s in b])
        causal = [dict(x, next_path=y["prev_path"]) for x, y in zip(a, b)]          # v2lib.window_samples' causal rule
        num = lambda p: int(re.search(r"(\d+)\.jpg$", p).group(1))                   # noqa: E731
        self.assertFalse(any(num(s["prev_path"]) > num(s["cur_path"]) or num(s["next_path"]) > num(s["cur_path"]) for s in causal))
        interior = {(num(s["prev_path"]) - num(s["cur_path"]), num(s["next_path"]) - num(s["cur_path"])) for s in causal if s["frame_idx"] >= 8}
        self.assertEqual(interior, {(-4, -8)})


class TestPreflightWindow(unittest.TestCase):
    """c_model/preflight.check_boundary (T2) must accept single / symmetric / causal windows and reject a window that reaches a
    future frame while claiming causality (the 2026-10-05 v2 smoke failed on the old symmetric-only rule)."""

    @staticmethod
    def fake(offsets, n=40, seqs=2):
        import torch

        class Idx:
            pass
        paths, prev_i, cur_i, next_i, fidx, slen = [], [], [], [], [], []
        for q in range(seqs):
            base = len(paths)
            paths += [f"s{q}/{f:06d}.jpg" for f in range(n)]
            for f in range(n):
                dp, dn = offsets(f)
                prev_i.append(base + min(max(f + dp, 0), n - 1))
                cur_i.append(base + f)
                next_i.append(base + min(max(f + dn, 0), n - 1))
                fidx.append(f)
                slen.append(n)
        idx = Idx()
        idx.paths = np.array(paths, dtype=np.bytes_)
        idx.prev_i, idx.cur_i, idx.next_i = (np.array(a, dtype=np.int32) for a in (prev_i, cur_i, next_i))
        idx.frame_idx, idx.seq_len = np.array(fidx, dtype=np.int32), np.array(slen, dtype=np.int32)

        class DS:
            index = idx

            def __getitem__(self, i):
                frames = [int(idx.paths[a[i]].decode()[-10:-4]) for a in (idx.prev_i, idx.cur_i, idx.next_i)]
                return (torch.tensor([[[float(v)]] for v in frames for _ in range(3)]),)
        return DS()

    def test_t2_window_kinds(self):
        import preflight as P
        cases = {"single": (lambda f: (0, 0), True), "symmetric g4": (lambda f: (-4, 4), True),
                 "causal g4": (lambda f: (-4, -8), True), "causal g1": (lambda f: (-1, -2), True),
                 "claims causal, peeks +1": (lambda f: (-4, -8) if f % 7 else (-4, 1), False)}
        for name, (off, want) in cases.items():
            r = P.check_boundary(self.fake(off))
            ok = r[1] if isinstance(r, tuple) else getattr(r, "ok", r)
            self.assertEqual(bool(ok), want, f"{name}: {r}")


class TestInitialisations(unittest.TestCase):
    def test_p2fp_is_exactly_stock_with_inert_p2(self):
        import torch
        import gen_lib as G
        import p2fp
        from aerotrack_trainer import build_model, transfer_pretrained
        from train import PRETRAINED
        torch.manual_seed(0)
        m = build_model(REPO / "c_model" / "yolo11-stride-p2fp.yaml", ch=9, nc=2, verbose=False, hyp=None, class_weights=None)
        transfer_pretrained(m, PRETRAINED, verbose=False)
        chk = p2fp.init_p2fp(m, build_model, transfer_pretrained, G.apply_fp_init, REPO / "c_model" / "yolo11-stride.yaml", PRETRAINED)
        self.assertEqual(chk["stock_anchor_max_abs_diff"], 0.0)
        self.assertLess(chk["p2_max_class_score"], 1e-6)

    def test_scale_s_layout_and_centre_frame_init(self):
        import ultralytics.nn.tasks as UT
        import p2fp
        from aerotrack_trainer import build_model, transfer_pretrained
        self.assertEqual(UT.yaml_model_load(str(REPO / "c_model" / "yolo11-stride-scale-s.yaml"))["width_multiple"], 0.5)
        m = build_model(REPO / "c_model" / "yolo11-stride-scale-s.yaml", ch=9, nc=2, verbose=False, hyp=None, class_weights=None)
        transfer_pretrained(m, REPO / "c_model" / "yolo11s.pt", verbose=False)
        self.assertEqual(p2fp.apply_fp_init_from(m, REPO / "c_model" / "yolo11s.pt")["ignores_neighbours_max_abs_diff"], 0.0)

    def test_scale_resolution_c3k2_variant(self):
        """Ultralytics resolves scale '' for our flat-multiplier YAMLs and builds C3k2(c3k=True) (erratum 2026-10-05);
        yolo11-stride-std.yaml (explicit scales: n) builds standard YOLO11n, loads >= 89 % of yolo11n.pt, init exact."""
        import torch
        import gen_lib as G
        from aerotrack_trainer import build_model, transfer_pretrained
        from train import PRETRAINED
        built = build_model(REPO / "c_model" / "yolo11-stride.yaml", ch=9, nc=2, verbose=False, hyp=None, class_weights=None)
        self.assertEqual(type(built.model[2].m[0]).__name__, "C3k")
        std = build_model(REPO / "c_model" / "yolo11-stride-std.yaml", ch=9, nc=2, verbose=False, hyp=None, class_weights=None)
        self.assertEqual(type(std.model[2].m[0]).__name__, "Bottleneck")
        ck = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
        src = (ck.get("ema") or ck["model"]).float().state_dict()
        dst = std.state_dict()
        frac = sum(1 for k in dst if k in src and src[k].shape == dst[k].shape) / len(dst)
        self.assertGreater(frac, 0.89)
        transfer_pretrained(std, PRETRAINED, verbose=False)
        self.assertEqual(G.apply_fp_init(std, "stride")["ignores_neighbours_max_abs_diff"], 0.0)

    def test_ultralytics_name_hazard_is_real(self):
        """yolo11<scale>-<x>.yaml is silently mapped to yolo11-<x>.yaml; our files must never use that pattern."""
        offenders = [p.name for p in (REPO / "c_model").glob("*.yaml") if re.match(r"^yolo\d+[nslmx]-", p.name)]
        self.assertEqual(offenders, [])


class TestCorpusAndSplits(unittest.TestCase):
    def test_default_corpus_is_v1_and_switch_works(self):
        code = "import sys; sys.path.insert(0, r'%s'); import corpus; print(corpus.MANIFEST)" % (REPO / "b_pipeline")
        env = {k: v for k, v in os.environ.items() if k != "AEROTRACK_CORPUS"}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stdout.strip()
        self.assertEqual(Path(out), REPO / "b_pipeline" / "data_unified" / "manifest.csv")
        out2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(env, AEROTRACK_CORPUS="data_unified_v2")).stdout.strip()
        self.assertEqual(Path(out2), REPO / "b_pipeline" / "data_unified_v2" / "manifest.csv")

    def test_official_cst_val_replaces_holdout(self):
        import pandas as pd
        spec = importlib.util.spec_from_file_location("prep", REPO / "b_pipeline" / "3_prepare_training.py")
        P = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(P)
        rows = []
        for split, n in (("train", 6), ("val", 2), ("test", 2)):
            for i in range(n):
                rows += [{"dataset": "cst", "seq_key": f"cst_{split}_building_{i}", "split": split, "frame_idx": f, "materialized": 1}
                         for f in range(3)]
        rows += [{"dataset": "anti_uav", "seq_key": f"antiuav_{s}_x__IR", "split": s, "frame_idx": 0, "materialized": 1} for s in ("train", "val", "test")]
        spec_, seq = P.build_splits(pd.DataFrame(rows))
        self.assertEqual(spec_["val"]["cst"], ["val"])
        self.assertEqual(spec_["cst_val_holdout_seqs"], [])
        tr, va, te = (P.seq_keys_for(seq, spec_, ph) for ph in ("train", "val", "test"))
        self.assertTrue(va and not (tr & va) and not (tr & te) and not (va & te))


class TestTrackCForecasting(unittest.TestCase):
    """Track C (experiments/aerotrack_temporal): causal samples, exact classical forecasts, Kalman convergence, sealed TEST."""

    def setUp(self):
        sys.path.insert(0, str(REPO / "experiments" / "aerotrack_temporal" / "scripts"))
        import tc_forecast
        self.F = tc_forecast

    def synthetic(self, T=200, gap=(100, 105)):
        box = np.stack([10 + 2.0 * np.arange(T), 50 - 1.0 * np.arange(T), np.full(T, 8.0), np.full(T, 6.0)], 1)
        vis = np.ones(T, bool)
        vis[gap[0]:gap[1]] = False
        box[~vis] = np.nan
        return [{"key": "s", "domain": "x", "vis": vis, "box": box, "W": 640, "H": 512}]

    def test_histories_never_contain_future_frames(self):
        seqs = self.synthetic()
        # mark every frame's centre with its own index: any history value > anchor index would be a future leak
        seqs[0]["box"] = np.stack([np.arange(200) - 4.0, np.zeros(200), np.full(200, 8.0), np.full(200, 6.0)], 1)
        seqs[0]["box"][~seqs[0]["vis"]] = np.nan
        S = self.F.make_samples(seqs)
        hx = np.where(S["hist_m"], S["hist_c"][..., 0], -np.inf)
        self.assertTrue((hx.max(1) <= S["t"]).all())
        self.assertTrue((hx[:, -1] == S["t"]).all())
        fx = np.where(S["fut_m"], S["fut_c"][..., 0], np.inf)
        self.assertTrue((fx.min(1) > S["t"]).all())

    def test_classical_forecasts_exact_on_constant_velocity(self):
        seqs = self.synthetic()
        S = self.F.make_samples(seqs)
        for nm in ("cv", "mv4", "mv8", "ca"):
            self.assertLess(np.nanmax(self.F.point_errors(S, self.F.forecast_classical(S, nm))), 1e-9, nm)
        mean, var = self.F.kalman_forecast(seqs, S, self.F.KalmanBank())
        late = S["t"] > 150
        self.assertLess(np.nanmax(self.F.point_errors(S, mean)[late]), 1e-3)
        self.assertTrue((var > 0).all())

    def test_load_split_shares_one_array(self):
        """Every sequence is a view into ONE decompressed array (the per-access npz bug held ~400 full copies, ~6 GB)."""
        import tc_data
        if not (tc_data.OUT / "traj_val.json").exists():
            self.skipTest("trajectory data not built")
        va = tc_data.load_split("val")
        self.assertTrue(all(np.shares_memory(va[0]["box"].base, s["box"]) for s in va[1:20]))

    def test_test_split_is_sealed(self):
        import tc_data
        with self.assertRaises(tc_data.SealedSplitError):
            tc_data.load_split("test")
        with self.assertRaises(tc_data.SealedSplitError):
            tc_data.collect("test", "develop")


class TestFSTE(unittest.TestCase):
    """C3's forecaster-driven STE: classical forecasts identical to tc_forecast on eligible histories; STE decision semantics."""

    def setUp(self):
        sys.path.insert(0, str(REPO / "experiments" / "aerotrack_temporal" / "scripts"))
        import tc_forecast
        import tc_ste
        self.F, self.T = tc_forecast, tc_ste

    def test_single_history_forecasts_match_vectorised(self):
        rng = np.random.default_rng(1)
        T = 300
        c = np.cumsum(rng.normal(0, 2, (T, 2)), 0) + 300
        box = np.concatenate([c - 4, np.full((T, 2), 8.0)], 1)
        vis = rng.random(T) > 0.2
        box[~vis] = np.nan
        S = self.F.make_samples([{"key": "s", "domain": "x", "vis": vis, "box": box, "W": 640, "H": 512}])
        cases = [("cp", None), ("cv", None), ("mv4", None), ("mv15", None), ("ca", None), ("dcv", {"g": 0.9, "k": 4}),
                 ("ou", {"rho": 0.99}), ("oudcv", {"rho": 0.995, "g": 0.5, "k": 8})]
        for name, prm in cases:
            ref = self.F.forecast_classical(S, name, prm)
            for i in range(0, len(S["t"]), 17):
                one = self.T.classical_mean(name, prm, S["hist_c"][i], S["hist_m"][i], 640, 512)
                np.testing.assert_allclose(one, ref[i], atol=1e-9, err_msg=name)

    def test_tracks_coasts_and_declares_absence(self):
        fc = self.T.ClassicalForecaster("cv", None, np.full((24, 2), 4.0))
        trk = self.T.FSTE(self.T.FSTEParams(max_coast=3, det_thr=0.1, init_thr=0.4, reacq_thr=0.9), fc, 640, 512)
        trk.reset(box(100, 100))
        modes = [trk.step(*dets((box(100 + 2 * t, 100), 0.8, 0)))[1] for t in range(1, 11)]
        self.assertTrue(all(m == "track" for m in modes))
        out = [trk.step(*dets()) for _ in range(3)]
        self.assertEqual([o[1] for o in out], ["coast"] * 3)
        np.testing.assert_allclose(out[-1][0], box(100 + 2 * 13, 100), atol=1e-6)      # coasting follows the CV forecast
        self.assertEqual(trk.step(*dets())[1], "absent")

    def test_gate_and_reacquisition(self):
        fc = self.T.ClassicalForecaster("cp", None, np.full((24, 2), 4.0))
        trk = self.T.FSTE(self.T.FSTEParams(reacq_thr=0.7, det_thr=0.1, max_coast=10), fc, 640, 512)
        trk.reset(box(100, 100))
        for _ in range(5):
            trk.step(*dets((box(100, 100), 0.9, 0)))
        self.assertEqual(trk.step(*dets((box(400, 300), 0.5, 0)))[1], "coast")
        b, mode, _ = trk.step(*dets((box(400, 300), 0.95, 0)))
        self.assertEqual(mode, "reacquired")
        np.testing.assert_allclose(b, box(400, 300))
        self.assertEqual(trk.step(*dets((box(50, 50), 0.9, 1)))[1], "coast")          # bird class never associates


class TestAdoptionRule(unittest.TestCase):
    """The registered C1 / C1c adoption rule (tc_adoption.offered) on its four branches."""

    def test_branches(self):
        sys.path.insert(0, str(REPO / "experiments" / "aerotrack_temporal" / "scripts"))
        import tc_adoption as A

        def c1(h1, h2):
            return {"chosen_architecture": "tfm", "domains": {"cst": {"H1_point": {"verdict": h1}, "H2_uncertainty": {"verdict": h2}}}}

        def c1c(h1c, h2c, arm="perdom"):
            return {"domains": {"cst": {"arm": arm, "H1c_point": {"verdict": h1c}, "H2c_vs_pooled": {"verdict": h2c}}}}
        self.assertEqual(A.offered("cst", c1("SUPPORTED", "NOT_SUPPORTED"), None)["source"], "C1")
        self.assertIsNone(A.offered("cst", c1("SUPPORTED", "LOSES"), None))
        self.assertIsNone(A.offered("cst", c1("LOSES", "SUPPORTED"), None))
        o = A.offered("cst", c1("LOSES", "SUPPORTED"), c1c("SUPPORTED", "NOT_SUPPORTED"))
        self.assertEqual((o["source"], o["names"][0], o["dom_tok"]), ("C1c", "c1c_tfm_perdom_cst_s0", False))
        self.assertEqual(A.offered("cst", c1("SUPPORTED", "SUPPORTED"), c1c("SUPPORTED", "NOT_SUPPORTED"))["source"], "C1")
        self.assertTrue(A.offered("cst", c1("SUPPORTED", "SUPPORTED"), c1c("SUPPORTED", "SUPPORTED", "domtok"))["dom_tok"])


class TestHOTA(unittest.TestCase):
    """HOTA re-implementation (TrackEval algorithm) on cases with hand-computed values."""

    def setUp(self):
        sys.path.insert(0, str(REPO / "experiments" / "aerotrack_temporal" / "scripts"))
        import tc_track_metrics
        self.M = tc_track_metrics
        self.T = 10
        self.gt = {t: (np.array([7]), self.b(t)) for t in range(self.T)}

    @staticmethod
    def b(x, s=0.0):
        return np.array([[x + s, 0, x + s + 10, 10]], float)

    def hota(self, pr):
        return self.M.hota_summary(self.M.hota_sequence(self.gt, pr, self.T))

    def test_perfect_switch_half_and_localisation(self):
        self.assertAlmostEqual(self.hota({t: (np.array([3]), self.b(t)) for t in range(self.T)})["HOTA"], 1.0)
        sw = self.hota({t: (np.array([3 if t < 5 else 4]), self.b(t)) for t in range(self.T)})
        self.assertAlmostEqual(sw["AssA"], 0.5)
        self.assertAlmostEqual(sw["HOTA"], np.sqrt(0.5))
        half = self.hota({t: (np.array([3]), self.b(t)) for t in range(5)})
        self.assertAlmostEqual(half["DetA"], 0.5)
        self.assertAlmostEqual(half["AssA"], 0.5)
        s = 10 * (1 - 0.62) / (1 + 0.62)                                 # IoU 0.62 on every frame
        loc = self.hota({t: (np.array([3]), self.b(t, s)) for t in range(self.T)})
        self.assertAlmostEqual(loc["HOTA"], 12 / 19)                     # matched for the 12 alphas <= 0.6
        self.assertAlmostEqual(self.hota({})["HOTA"], 0.0)

    def test_combine_weights_by_tp(self):
        r1 = self.M.hota_sequence(self.gt, {t: (np.array([3]), self.b(t)) for t in range(self.T)}, self.T)
        r2 = self.M.hota_sequence(self.gt, {t: (np.array([3 if t < 5 else 4]), self.b(t)) for t in range(self.T)}, self.T)
        comb = self.M.hota_summary(self.M.hota_combine([r1, r2]))
        self.assertAlmostEqual(comb["AssA"], 0.75)
        self.assertAlmostEqual(comb["DetA"], 1.0)

    def test_continuity_and_recovery(self):
        gt = {t: (np.array([1]), self.b(t)) for t in range(20) if not 8 <= t < 11}      # invisible 8..10, re-enters at 11
        pr = {t: (np.array([5]), self.b(t)) for t in range(20) if t not in (4, 5, 11, 12)}  # lost 4-5, re-acquired 2 frames late
        c = self.M.continuity(gt, pr, 20)
        self.assertEqual(c[0]["fragments"], 2)
        ev = {(e["kind"], e["t"]): e["delay"] for e in c[0]["events"]}
        self.assertEqual(ev[("loss", 4)], 2)
        self.assertEqual(ev[("reentry", 11)], 2)
        self.assertEqual(c[0]["ids"], 1)


class TestPreregisteredRules(unittest.TestCase):
    def test_h4_decision_branches(self):
        # experiments/anti_uav/scripts has its own train.py: put it on the path only for this import (module-name clash)
        sys.path.insert(0, str(REPO / "experiments" / "anti_uav" / "scripts"))
        try:
            import analyze_h4 as A
        finally:
            sys.path.remove(str(REPO / "experiments" / "anti_uav" / "scripts"))

        def M(nv, nt, dv, dt):
            return {"normal_ir": {"val": {"ref": 79.29, "native": nv, "paper": 79.60}, "test": {"ref": 70.58, "native": nt, "paper": 65.36}},
                    "dfsc_ir": {"val": {"ref": 79.26, "native": dv, "paper": 80.09}, "test": {"ref": 70.29, "native": dt, "paper": 66.04}}}
        cases = {"SUPPORTED": M(79.5, 65.8, 79.8, 66.5), "PARTIALLY_SUPPORTED": M(79.0, 67.5, 79.4, 67.9),
                 "NOT_SUPPORTED_VAL_BREAKS": M(74.0, 64.0, 74.5, 65.0), "NOT_SUPPORTED_INSENSITIVE": M(79.0, 70.0, 79.5, 70.0),
                 "INCONCLUSIVE": M(79.3, 68.9, 79.3, 69.9), "NOT RUN YET": M(79.0, None, 79.0, 70.0)}
        for want, m in cases.items():
            self.assertEqual(A.decide(m)[0], want)


class TestPathMap(unittest.TestCase):
    """AEROTRACK_PATH_MAP rewrites manifest paths for the HPC and is a no-op when unset (D-2026-10-06-05)."""

    def _df(self):
        import pandas as pd
        return pd.DataFrame({"image_path": [r"A:\AEROTRACK_OMEN\data\cst\x\000001.jpg", "", r"C:\other\f.jpg"],
                             "label_path": [r"A:\AEROTRACK_OMEN\b_pipeline\data_unified_v2\cst\labels\x\000001.txt", "", ""]})

    def test_noop_when_unset(self):
        from stacked_dataset import remap_manifest_paths
        old = os.environ.pop("AEROTRACK_PATH_MAP", None)
        try:
            df = self._df(); ref = self._df()
            remap_manifest_paths(df)
            self.assertTrue(df.equals(ref))
        finally:
            if old is not None:
                os.environ["AEROTRACK_PATH_MAP"] = old

    def test_prefix_rewrite(self):
        from stacked_dataset import remap_manifest_paths
        os.environ["AEROTRACK_PATH_MAP"] = r"A:\AEROTRACK_OMEN=>/home/u/aerotrack_server/root/"
        try:
            df = remap_manifest_paths(self._df())
        finally:
            del os.environ["AEROTRACK_PATH_MAP"]
        self.assertEqual(df["image_path"][0], "/home/u/aerotrack_server/root/data/cst/x/000001.jpg")
        self.assertEqual(df["label_path"][0],
                         "/home/u/aerotrack_server/root/b_pipeline/data_unified_v2/cst/labels/x/000001.txt")
        self.assertEqual(df["image_path"][1], "")
        self.assertEqual(df["image_path"][2], r"C:\other\f.jpg")   # other prefixes untouched


if __name__ == "__main__":
    unittest.main(verbosity=2)
