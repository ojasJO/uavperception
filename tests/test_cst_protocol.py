"""CST Anti-UAV protocol tests (CPU only; run: venv\\Scripts\\python.exe -m unittest tests.test_cst_protocol -v).

Pins the properties the CST comparison depends on: Track A's SA (which scores every AeroTrack tracker) equals the official toolkit's
SA on synthetic sequences, the official JSON export round-trips through the toolkit's own loader, and the size-bin rule matches the
release's SIZE codes. Toolkit / data dependent tests are skipped when `data/` is absent (it is git-ignored)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "experiments" / "anti_uav" / "src", REPO / "experiments" / "cst_anti_uav" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cstlib  # noqa: E402
from anti_uav_repro.eval.metrics import sequence_metrics  # noqa: E402

TOOLKIT = REPO / "data" / "cst_anti_uav_official" / "toolkit"
DATA = REPO / "data" / "cst_anti_uav_full" / "CST-AntiUAV" / "CST-AntiUAV"


def official_experiment():
    import official_eval
    E = official_eval.import_toolkit()
    return E.ExperimentCSTAntiUAV.__new__(E.ExperimentCSTAntiUAV)   # eval() needs no dataset


# (predictions, exist, gt) cases; gt boxes x, y, w, h
G = [10.0, 10.0, 8.0, 6.0]
CASES = {
    "exact": ([G, G], [1, 1], [G, G]),
    "shifted": ([[12.0, 11.0, 8.0, 6.0], [10.0, 10.0, 4.0, 6.0]], [1, 1], [G, G]),
    "no_overlap": ([[30.0, 30.0, 5.0, 5.0]], [1], [G]),
    "touching_edge": ([[18.0, 10.0, 5.0, 6.0]], [1], [G]),           # x0 + w convention: 10 + 8 = 18 -> zero overlap
    "missed_visible": ([[]], [1], [G]),
    "correct_absent": ([[]], [0], [[]]),
    "false_alarm": ([G], [0], [[]]),
    "mixed": ([G, [], [], G, [11.0, 10.0, 8.0, 6.0]], [1, 1, 0, 0, 1], [G, G, [], [], G]),
}


class TestSAEquality(unittest.TestCase):
    @unittest.skipUnless((TOOLKIT / "experiments" / "cstanti_uav.py").exists(), "official toolkit not present")
    def test_track_a_equals_official(self):
        exp = official_experiment()
        for name, (pred, exist, gt) in CASES.items():
            with self.subTest(name):
                ours = sequence_metrics(pred, exist, gt)["SA"]
                theirs = float(exp.eval(pred, {"gt": gt, "exist": exist}))
                self.assertAlmostEqual(ours, theirs, places=12)

    def test_known_values(self):
        self.assertAlmostEqual(sequence_metrics(*CASES["exact"])["SA"], 1.0)
        self.assertAlmostEqual(sequence_metrics(*CASES["false_alarm"])["SA"], 0.0)
        self.assertAlmostEqual(sequence_metrics(*CASES["correct_absent"])["SA"], 1.0)
        self.assertAlmostEqual(sequence_metrics(*CASES["touching_edge"])["SA"], 0.0)
        iou = (7 * 6) / (2 * 48 - 42)                                      # shifted by 1 px in x: 7x6 overlap
        self.assertAlmostEqual(sequence_metrics(*CASES["mixed"])["SA"], (1 + 0 + 1 + 0 + iou) / 5)

    @unittest.skipUnless((TOOLKIT / "experiments" / "cstanti_uav.py").exists(), "official toolkit not present")
    def test_degenerate_gt_is_the_only_divergence(self):
        """The toolkit SKIPS visible frames with an invalid GT box; Track A scores them 0. CST has none (dataset_manifest.json)."""
        exp = official_experiment()
        pred, exist, gt = [G, G], [1, 1], [G, [0.0, 0.0, 0.0, 0.0]]
        self.assertAlmostEqual(float(exp.eval(pred, {"gt": gt, "exist": exist})), 1.0)
        self.assertAlmostEqual(sequence_metrics(pred, exist, gt)["SA"], 0.5)


class TestOfficialFormat(unittest.TestCase):
    def test_round_trip(self):
        import official_eval
        res = [[1.0, 2.0, 3.0, 4.0], [], [5.5, 6.5, 7.5, 8.5]]
        with tempfile.TemporaryDirectory() as d:
            p = cstlib.write_official(d, "seq", res)
            self.assertEqual(official_eval._load(p), res)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), {"res": res})

    def test_rejects_bad_entries(self):
        with self.assertRaises(ValueError):
            cstlib.official_record([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            cstlib.official_record([[1.0, 2.0, float("nan"), 4.0]])


class TestSizeBins(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(cstlib.size_code([0, 0, 6, 8], 1), 1)              # diagonal 10 -> tiny (<= 10)
        self.assertEqual(cstlib.size_code([0, 0, 6, 8.01], 1), 2)
        self.assertEqual(cstlib.size_code([0, 0, 30, 40], 1), 3)            # 50 -> normal
        self.assertEqual(cstlib.size_code([0, 0, 30, 40.1], 1), 4)
        self.assertEqual(cstlib.size_code([], 0), 0)

    @unittest.skipUnless(DATA.exists(), "CST data not present")
    def test_matches_release_size_codes(self):
        n = 0
        for lab in sorted((DATA / "val").glob("*/IR_label.json"))[:10]:
            a = json.loads(lab.read_text(encoding="utf-8"))
            if "SIZE" not in a:
                continue
            ours = [cstlib.size_code(g, e) for g, e in zip(a["gt"], a["exist"])]
            self.assertEqual(ours, [int(s) for s in a["SIZE"]], lab.parent.name)
            n += 1
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
