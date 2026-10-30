"""Flight Deck UI tests (Streamlit AppTest, headless): home, Test Bench per category, the full journey (configure -> start -> live ->
replay -> second model / synchronised views -> metrics), rerender safety, recorded TEST replay, results, vault, invalid input and
explicit-CUDA failure.

    venv\\Scripts\\python.exe -m unittest tests.test_flight_deck_ui -v      (~2-4 min with CUDA)
"""
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]
APP = str(REPO / "studio" / "app.py")
sys.path.insert(0, str(REPO))
from aerolab import data, jobs  # noqa: E402
from aerolab.categories import ORDER  # noqa: E402
from aerolab.paths import STUDIO_INDEX  # noqa: E402

HAVE_CST = (data.CST_ROOT / "val").exists()
HAVE_CST_TEST = (data.CST_ROOT / "test").exists()


def app(page=None, state=None, timeout=600):
    at = AppTest.from_file(APP, default_timeout=timeout)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run()
    if page:
        at.switch_page(f"views/{page}.py")
        at.run()
    return at


def texts(at):
    out = []
    for coll in (at.markdown, at.caption, at.title, at.subheader, at.info, at.warning, at.success, at.error):
        out += [str(x.value) for x in coll]
    return "\n".join(out)


def widget(coll, label):
    return next(w for w in coll if w.label == label)


def index_lines():
    return len(STUDIO_INDEX.read_text(encoding="utf-8").splitlines()) if STUDIO_INDEX.exists() else 0


class FlightDeckUI(unittest.TestCase):
    def clean(self, at):
        self.assertEqual([e.value for e in at.exception], [])

    def test_home(self):
        at = app()
        self.clean(at)
        t = texts(at)
        for s in ("Direction A", "Direction B", "TEST splits are locked", "Featured models", "Run an experiment", "TEST is locked"):
            self.assertIn(s, t)
        self.assertTrue(any(e.label.startswith("Limitations") for e in at.expander))
        self.assertEqual(sum(1 for b in at.button if b.label == "Open in Test Bench"), 7)

    def test_every_category_configures(self):
        for cat in ORDER:
            with self.subTest(category=cat):
                at = app("test_bench", {"tb_category": cat})
                self.clean(at)
                self.assertIn("summary", texts(at).lower())
                self.assertTrue(any("Start" in b.label or "Run benchmark" in b.label for b in at.button), cat)

    def test_presence_excludes_globaltrack_with_reason(self):
        at = app("test_bench", {"tb_category": "presence", "tb_dir_presence": "both"})
        self.clean(at)
        ids = widget(at.multiselect, "Models (* = featured)").options
        self.assertFalse(any("normal_ir" in o for o in ids))
        df = at.dataframe[-1].value
        row = df[df["model"] == "antiuav_repro.globaltrack:normal_ir@last"]
        self.assertIn("presence", row.iloc[0]["why not offered"])

    def test_invalid_selection_disables_start(self):
        at = app("test_bench", {"tb_category": "tracking"})
        ms = next(w for w in at.multiselect if w.label.startswith("Sequences to run"))
        ms.set_value([]).run()
        self.clean(at)
        self.assertIn("select at least one sequence", texts(at))
        self.assertTrue(widget(at.button, "Start experiment").disabled)

    @unittest.skipUnless(HAVE_CST, "CST absent")
    def test_full_journey_live_to_synchronised_replay(self):
        n0 = index_lines()
        at = app("test_bench", {"tb_category": "tracking"})
        self.assertEqual(len(widget(at.multiselect, "Models (* = featured)").value), 2)        # X1 + single-frame control by default
        widget(at.number_input, "Frames after the init frame").set_value(12).run()
        for _ in range(3):                                                                        # rerenders never start a run
            at.run()
        self.assertEqual(index_lines(), n0)
        widget(at.button, "Start experiment").click().run()
        self.clean(at)
        job_id = at.session_state["live_job"]
        at.switch_page("views/live.py")          # AppTest does not follow a script-initiated st.switch_page; the browser does
        at.run()
        self.clean(at)
        self.assertIn("LIVE INFERENCE", texts(at) + " ".join(str(e.value) for e in at.error))
        j = jobs.wait(jobs.manager().get(job_id), 600)
        self.assertEqual(j.status, "completed", j.error)
        at.run()
        self.clean(at)
        self.assertIn("Run finished", texts(at))
        widget(at.button, "Open the synchronised replay of this run").click().run()
        self.clean(at)
        self.assertIn("SAVED RUN", texts(at))
        self.assertEqual(len(at.get("image")), 3)                                                  # side by side: one panel per model + the status strip
        slider = widget(at.slider, "Frame")
        t0 = slider.value
        widget(at.button, "Next").click().run()
        self.assertEqual(widget(at.slider, "Frame").value, t0 + 1)                                 # synchronised frame index
        self.assertEqual(len(at.get("image")), 3)
        view = next(w for w in at.button_group if getattr(w, "label", "") == "View")
        view.set_value("Overlay").run()
        self.clean(at)
        self.assertEqual(len(at.get("image")), 2)                                                  # both models on one frame + the status strip
        view = next(w for w in at.button_group if getattr(w, "label", "") == "View")
        view.set_value("Difference").run()
        self.clean(at)
        metrics = [d.value for d in at.dataframe]
        self.assertTrue(any(len(df) == 2 and "SA %" in df.columns for df in metrics))
        self.assertEqual(index_lines(), n0 + 2)                                                   # exactly one record per model

    @unittest.skipUnless(HAVE_CST, "CST absent")
    def test_cancel_from_the_ui(self):
        at = app("test_bench", {"tb_category": "tracking"})
        widget(at.toggle, "Whole sequences").set_value(True).run()
        widget(at.button, "Start experiment").click().run()
        job_id = at.session_state["live_job"]
        at.switch_page("views/live.py")
        j = jobs.manager().get(job_id)
        t0 = time.time()
        while not j.live and time.time() - t0 < 120:
            time.sleep(0.2)
        at.run()
        widget(at.button, "Cancel run").click().run()
        jobs.wait(j, 300)
        self.assertEqual(j.status, "cancelled")
        at.run()
        self.clean(at)

    @unittest.skipUnless(HAVE_CST_TEST, "CST test frames absent")
    def test_recorded_test_replay_matches_records(self):
        at = app("live", {"live_mode": "Recorded predictions"})
        self.clean(at)
        self.assertIn("RECORDED PREDICTIONS", texts(at))
        self.assertEqual(widget(at.selectbox, "Sequence (60 shared by all sources)").value, "jungle_18")
        df = at.dataframe[-1].value
        self.assertEqual(len(df), 3)
        self.assertTrue((df["|Δ|"].fillna(0) < 1e-6).all())                                       # replay == registered per-sequence SA
        self.assertFalse(any(w.label.startswith("Score threshold") for w in at.slider))           # no tuning controls on TEST

    def test_results_and_vault(self):
        at = app("results")
        self.clean(at)
        t = texts(at)
        self.assertIn("NO_GAIN", [m.value for m in at.metric])
        self.assertIn("demonstrated trade-off", t)
        at = app("vault")
        self.clean(at)
        self.assertIn("one checkpoint", texts(at).lower())
        widget(at.text_input, "Search").set_value("ttt_cst").run()
        self.clean(at)
        arch = next(d.value for d in at.dataframe if "model_id" in d.value.columns and "evidence" in d.value.columns)
        self.assertGreater(len(arch), 0)
        self.assertTrue(all("ttt_cst" in m for m in arch["model_id"]))


@unittest.skipUnless(HAVE_CST, "CST absent")
class CudaHiddenUI(unittest.TestCase):
    def test_explicit_cuda_refused_and_cpu_default(self):
        code = f"""
import sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({APP!r}, default_timeout=300)
at.session_state["tb_category"] = "tracking"
at.run(); at.switch_page("views/test_bench.py"); at.run()
dev = next(w for w in at.selectbox if w.label == "Device")
dev.set_value("cuda").run()
errs = " | ".join(str(e.value) for e in at.error)
start = next(b for b in at.button if b.label == "Start experiment")
print("RESULT", "CUDA requested but" in errs, start.disabled, len(at.exception))
"""
        p = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), cwd=REPO, capture_output=True, text=True, timeout=600)
        line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT")), None)
        self.assertIsNotNone(line, p.stderr[-2000:])
        self.assertEqual(line.split()[1:], ["True", "True", "0"])


if __name__ == "__main__":
    unittest.main()
