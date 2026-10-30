"""Headless UI tests of the studio with Streamlit's AppTest (run: venv\\Scripts\\python.exe -m unittest tests.test_studio_ui -v).

Every page must render without an exception; key interactions are exercised: search / filter, a real small inference run from the
Inference page, the incompatible / recorded-only path, the default recorded comparison (comparable verdict + paired delta), and
the TEST-lock banner. ~1-2 minutes with CUDA (one detector run of 10 frames)."""
import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]
APP = str(REPO / "app" / "app.py")
HAVE_CST = (REPO / "data" / "cst_anti_uav_full" / "CST-AntiUAV" / "CST-AntiUAV" / "val").exists()


def page(name, timeout=600):
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    if name != "overview":
        at.switch_page(f"views/{name}.py")
        at.run()
    return at


def texts(at):
    out = []
    for coll in (at.markdown, at.caption, at.title, at.subheader, at.info, at.warning, at.success, at.error):
        out += [str(x.value) for x in coll]
    return "\n".join(out)


def widget(coll, label):
    return next(w for w in coll if w.label == label)


class Pages(unittest.TestCase):
    def assertClean(self, at):
        self.assertEqual([e.value for e in at.exception], [])

    def test_every_page_renders(self):
        for p in ("overview", "models", "datasets", "inference", "compare", "experiments", "failures", "health"):
            with self.subTest(page=p):
                at = page(p)
                self.assertClean(at)

    def test_overview_shows_both_directions_and_lock(self):
        at = page("overview")
        t = texts(at)
        self.assertIn("Direction A", t)
        self.assertIn("Direction B", t)
        self.assertIn("TEST splits are locked", t)

    def test_models_search(self):
        at = page("models")
        widget(at.text_input, "Search (run id, alias, role)").set_value("X1").run()
        self.assertClean(at)
        self.assertIn("v3_n__causal_g4_cst_s0_e20", texts(at))

    def test_recorded_only_family(self):
        at = page("inference")
        widget(at.selectbox, "Model family").set_value("aerotrack.detector.repair").run()
        self.assertClean(at)
        self.assertIn("Recorded results only", texts(at))

    @unittest.skipUnless(HAVE_CST, "CST data absent")
    def test_inference_run_from_ui(self):
        at = page("inference")
        self.assertEqual(widget(at.selectbox, "Model family").value, "aerotrack.detector.cst_x")
        widget(at.number_input, "Frames after start").set_value(10).run()
        widget(at.button, "Run inference").click().run()
        self.assertClean(at)
        t = texts(at)
        self.assertIn("Saved as", t)
        self.assertIn("studio.sot_smoke.v1", t)
        self.assertTrue(any(m.label == "SA (%)" for m in at.metric))

    @unittest.skipUnless((REPO / "data" / "anti_uav" / "val").exists(), "Anti-UAV data absent")
    def test_direction_b_inference_from_ui(self):
        at = page("inference")
        widget(at.selectbox, "Research direction").set_value("antiuav_repro").run()
        self.assertTrue(widget(at.selectbox, "Checkpoint version").value.startswith("antiuav_repro.globaltrack:"))
        widget(at.selectbox, "Dataset").set_value("antiuav2021").run()
        widget(at.number_input, "Frames after start").set_value(10).run()
        widget(at.button, "Run inference").click().run()
        self.assertClean(at)
        t = texts(at)
        self.assertIn("Saved as", t)
        self.assertIn("never reports 'absent'", t)

    def test_forecaster_from_ui(self):
        at = page("inference")
        widget(at.selectbox, "Model family").set_value("aerotrack.forecaster.c1").run()
        widget(at.number_input, "Max anchors").set_value(100).run()
        widget(at.button, "Run inference").click().run()
        self.assertClean(at)
        self.assertIn("never comparable with tracking mSA", texts(at))
        self.assertTrue(len(at.get("arrow_vega_lite_chart")) >= 1 or len(at.get("vega_lite_chart")) >= 1)

    def test_failure_page_with_predictions(self):
        at = page("failures")
        self.assertClean(at)
        self.assertNotIn("No per-frame predictions yet", texts(at))

    @unittest.skipUnless(HAVE_CST, "CST data absent")
    def test_fresh_comparison_two_models(self):
        at = page("compare")
        widget(at.radio, "Mode").set_value("Fresh VAL comparison (runs models now)").run()
        widget(at.number_input, "Frames after start (0 = whole sequences)").set_value(10).run()
        widget(at.multiselect, "VAL sequences").set_value(["building_41"]).run()
        ms = widget(at.multiselect, "Models (2-4)")
        want = ["aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last", "aerotrack.detector.cst_x:v3_n__ttt_cst_s0_e20@last"]
        ms.set_value(want).run()
        widget(at.button, "Run comparison").click().run()
        self.assertClean(at)
        df = at.dataframe[0].value
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["status"]), ["completed", "completed"])
        self.assertEqual(df.iloc[1]["vs first"], "comparable")

    def test_compare_default_is_comparable_with_paired_delta(self):
        at = page("compare")
        self.assertClean(at)
        df = at.dataframe[0].value
        row = df[df["record"] == "cst:released:GlobalTrack"].iloc[0]
        self.assertEqual(row["vs baseline"], "comparable")
        self.assertTrue(row["paired Δ [95% CI]"].startswith("-24.1"))


if __name__ == "__main__":
    sys.exit(unittest.main())
