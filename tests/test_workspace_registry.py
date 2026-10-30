"""Unified-workspace registry tests (CPU only, fast; run: venv\\Scripts\\python.exe -m unittest tests.test_workspace_registry -v).

Pins: schema validity, unique stable ids, the registered checkpoint identities (X1 = X2 seed 0, X2 seeds, single-frame seeds, CF
detector, 27 Track C forecasters) against their frozen records, preservation of historical metrics, duplicate / same-name handling,
protocol comparability, the TEST lock, and the protection of the recovery backup and pre-existing untracked directories."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from aerolab import schemas  # noqa: E402
from aerolab.compare import compatibility, paired_delta  # noqa: E402
from aerolab.paths import is_protected  # noqa: E402
from aerolab.registry import Registry  # noqa: E402
from aerolab.test_lock import TestSplitLocked, assert_split_allowed, consumed_registrations  # noqa: E402

R = Registry()
EXP = {e["experiment_id"]: e for e in R.historical_experiments}


def j(p):
    return json.loads((REPO / p).read_text(encoding="utf-8"))


class Schemas(unittest.TestCase):
    def test_models_valid_and_unique(self):
        ids = [m["model_id"] for m in R.models]
        self.assertEqual(len(ids), len(set(ids)))
        for m in R.models:
            self.assertEqual(schemas.errors(m, schemas.MODEL_SCHEMA), [], m["model_id"])

    def test_experiments_valid_and_unique(self):
        ids = [e["experiment_id"] for e in R.historical_experiments]
        self.assertEqual(len(ids), len(set(ids)))
        for e in R.historical_experiments:
            self.assertEqual(schemas.errors(e, schemas.EXPERIMENT_SCHEMA), [], e["experiment_id"])

    def test_datasets_valid(self):
        for d in R.datasets:
            self.assertEqual(schemas.errors(d, schemas.DATASET_SCHEMA), [], d["id"])

    def test_every_file_classified(self):
        self.assertEqual(R.summary["files"]["unclassified"], [])

    def test_catalog_references_exist(self):
        for p in R.papers.values():
            for ev in p.get("evidence", []):
                self.assertTrue((REPO / ev).exists(), f"paper {p['id']} evidence {ev}")
        for f in R.findings["findings"]:
            self.assertTrue((REPO / f["source"]).exists(), f["source"])
            for eid in f["evidence"]:
                self.assertIn(eid, EXP, f"finding {f['id']} cites unknown record {eid}")
        for x in R.findings["limitations"]:
            self.assertTrue((REPO / x["source"]).exists(), x["source"])
        for fam in R.families.values():
            self.assertIn(fam["direction"], R.directions)
            for pid, _ in fam["papers"]:
                self.assertIn(pid, R.papers)


class RegisteredIdentities(unittest.TestCase):
    def test_x1_is_x2_seed0_one_checkpoint(self):
        x1 = j("experiments/aerotrack_v3/reports/x1_final_test.json")
        m = R.model("aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last")
        self.assertEqual(m["sha256"], x1["weights_sha256"])
        self.assertEqual(m["sha256"], j("experiments/cst_anti_uav/frozen_x2_params.json")["runs"]["v3_n__causal_g4_cst_s0_e20"]["weights_sha256"])
        self.assertIn("AeroTrack-X1", m["aliases"])
        self.assertIn("X2 temporal seed 0", m["aliases"])
        self.assertEqual(m["status"], "registered")
        self.assertEqual(m["integrity_status"], "verified_against_record")

    def test_x2_seeds_match_frozen(self):
        fx = j("experiments/cst_anti_uav/frozen_x2_params.json")["runs"]
        for run, v in fx.items():
            m = R.model(f"aerotrack.detector.cst_x:{run}@last")
            self.assertEqual(m["sha256"], v["weights_sha256"], run)
            self.assertEqual(m["seed"], v["seed"], run)
            self.assertEqual(m["status"], "registered")

    def test_cf_detector(self):
        cf = j("experiments/aerotrack_temporal/frozen_cf.json")["detector"]
        m = R.model(f"aerotrack.detector.v2:{cf['run']}@last")
        self.assertEqual(m["sha256"], cf["weights_sha256"])
        self.assertIn("CF detector", m["aliases"])

    def test_trackc_27_forecasters(self):
        regs = {**j("results/runs/aerotrack_temporal/c1/learned_registry.json"), **j("results/runs/aerotrack_temporal/c1/learned_registry_c1c.json")}
        regs = {k: v for k, v in regs.items() if isinstance(v, dict) and "sha256" in v}
        self.assertEqual(len(regs), 27)
        for name, v in regs.items():
            fam = "aerotrack.forecaster.c1c" if name.startswith("c1c_") else "aerotrack.forecaster.c1"
            m = R.model(f"{fam}:{name}@weights")
            self.assertEqual(m["sha256"], v["sha256"], name)
            self.assertEqual(m["integrity_status"], "verified_against_record", name)

    def test_same_run_id_different_weights_split(self):
        a = R.model("aerotrack.detector.v2:v2_stock_s0_e20@last")
        b = R.model("aerotrack.detector.v2:v2_stock_s0_e20@last~hpc")
        self.assertNotEqual(a["sha256"], b["sha256"])
        self.assertEqual(b["integrity_status"], "verified_against_record")

    def test_no_integrity_mismatch(self):
        self.assertEqual([m["model_id"] for m in R.models if m["integrity_status"] == "mismatch"], [])

    def test_duplicates_recorded_not_merged(self):
        m = R.model("antiuav_repro.globaltrack:normal_ir@last")
        self.assertIn("antiuav_repro.globaltrack:normal_ir@epoch_18", m["same_bytes_as"])
        self.assertIn("antiuav_repro.globaltrack:normal_ir@epoch_18", R.by_id)

    def test_queries(self):
        b = R.best_validated("aerotrack.detector.cst_x", "sot.mSA.cst_official.v1", "mSA", split="test", variant="ste")
        self.assertEqual(b["model"]["run_id"], "v3_n__causal_g4_cst_s2_e20")
        self.assertIsNotNone(R.demonstration_version("aerotrack.detector.cst_x"))


class HistoricalMetricsPreserved(unittest.TestCase):
    def test_ledger_rows_verbatim(self):
        rows = [json.loads(x) for x in (REPO / "docs" / "EXPERIMENT_LEDGER.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        imported = [e for e in R.historical_experiments if e["experiment_id"].startswith("ledger:")]
        self.assertEqual(len(imported), len(rows))
        for e in imported:
            r = e["ledger_row"]
            self.assertEqual(e["metrics"], r.get("val_metrics") or r.get("test_metrics") or {})

    def test_cst_official_values(self):
        o = j("experiments/cst_anti_uav/reports/X2_test_official/official_scores.json")["trackers"]
        for key, t in o.items():
            run, var = key.rsplit("-", 1)
            self.assertEqual(EXP[f"cst:x2:{run}:{var}"]["metrics"]["mSA"], t["mSA"])
        self.assertAlmostEqual(EXP["cst:x1:ste"]["metrics"]["mSA"], 60.08125201735352, places=10)
        self.assertAlmostEqual(EXP["cst:x1:argmax"]["metrics"]["mSA"], 59.70, places=2)
        temporal = [EXP[f"cst:x2:v3_n__causal_g4_cst_s{s}_e20:ste"]["metrics"]["mSA"] for s in range(3)]
        single = [EXP[f"cst:x2:v3_n__ttt_cst_s{s}_e20:ste"]["metrics"]["mSA"] for s in range(3)]
        self.assertEqual([round(x, 2) for x in temporal], [60.08, 60.26, 60.60])
        self.assertAlmostEqual(sum(temporal) / 3, 60.31, places=2)
        self.assertEqual([round(x, 2) for x in single], [46.83, 49.81, 49.79])
        self.assertAlmostEqual(sum(single) / 3, 48.81, places=2)
        self.assertEqual(EXP["cst:paper:GlobalTrack"]["metrics"]["mSA"], 35.92)

    def test_paired_delta_reproduces_final_report(self):
        d = paired_delta(EXP["cst:x1:ste"], EXP["cst:released:GlobalTrack"])
        self.assertAlmostEqual(d["mean_delta"], 24.16, delta=0.01)                 # FINAL_REPORT: +24.16 [+20.78, +27.63], 59 / 1
        self.assertEqual((d["wins"], d["losses"]), (59, 1))
        self.assertAlmostEqual(d["ci95"][0], 20.78, delta=0.3)
        self.assertAlmostEqual(d["ci95"][1], 27.63, delta=0.3)

    def test_p1_verdict(self):
        e = EXP["cst:p1:v3_n__causal_g4_cst_s0_e20"]
        self.assertEqual(e["verdict"], "NO_GAIN")
        self.assertEqual(e["kind"], "exploratory_val")
        self.assertFalse(e["test_pass"])

    def test_paper_numbers_labelled(self):
        e = EXP["antiuav:paper:normal:val:IR"]
        self.assertEqual(e["provenance"], "paper_reported")
        self.assertEqual(e["metrics"]["mSA"], 79.6)


class Comparability(unittest.TestCase):
    def test_same_protocol_comparable(self):
        c = compatibility(EXP["cst:x2:v3_n__causal_g4_cst_s1_e20:ste"], EXP["cst:x1:ste"], R.protocols)
        self.assertTrue(c["valid"], c["reasons"])

    def test_different_split_or_protocol_not(self):
        c = compatibility(EXP["cst:x2val:v3_n__causal_g4_cst_s1_e20:ste"], EXP["cst:x1:ste"], R.protocols)
        self.assertFalse(c["valid"])
        c = compatibility(EXP["cf:test:sot:anti_uav_IR"], EXP["cst:x1:ste"], R.protocols)
        self.assertFalse(c["valid"])

    def test_paper_number_never_ranked(self):
        c = compatibility(EXP["cst:paper:GlobalTrack"], EXP["cst:released:GlobalTrack"], R.protocols)
        self.assertFalse(c["valid"])

    def test_forecasting_never_compared_with_mSA(self):
        c = compatibility(EXP["cf:test:forecast:cst:dcv"], EXP["cf:test:sot:cst_IR"], R.protocols)
        self.assertFalse(c["valid"])


class Locks(unittest.TestCase):
    def test_test_split_locked(self):
        with self.assertRaises(TestSplitLocked):
            assert_split_allowed("test")
        assert_split_allowed("val")

    def test_data_layer_refuses_test(self):
        from aerolab import data
        for ds in ("cst2025", "antiuav2021", "trackc_trajectories"):
            with self.assertRaises(TestSplitLocked):
                data.list_sequences(ds, "test")

    def test_evaluation_refuses_test_before_loading(self):
        from aerolab.evaluation import evaluate_sot
        with self.assertRaises(TestSplitLocked):
            evaluate_sot("aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last", "cst2025", "test", "IR", ["x"])

    def test_ledgers_known(self):
        locks = consumed_registrations()
        self.assertTrue(all(x["exists"] for x in locks), [x["ledger"] for x in locks if not x["exists"]])

    def test_protected_paths(self):
        self.assertTrue(is_protected(REPO / "results/artifacts" / "hpc_recovery_20261013" / "x"))
        self.assertTrue(is_protected(REPO / "results/runs" / "v2_causal_g4_s0_e20" / "weights" / "last.pt"))
        self.assertFalse(is_protected(REPO / "results/runs" / "studio" / "x"))

    def test_restore_tool_never_targets_protected(self):
        sys.path.insert(0, str(REPO / "scripts"))
        import restore_from_recovery as RF
        for r in RF.plan():
            self.assertFalse(r["rel"].startswith(RF.PROTECTED), r["rel"])
            self.assertNotEqual(r["status"], "conflict_kept_local", r["rel"])


class Docs(unittest.TestCase):
    DOCS = ("README.md", "docs/research/aero_multi_paper/README.md", "docs/research/anti_uav_paper_reproduction/README.md",
            "docs/workspace/ARCHITECTURE.md", "docs/workspace/OPERATING_GUIDE.md", "docs/workspace/INITIAL_AUDIT.md", "results/registry/README.md")

    def test_cited_paths_exist(self):
        import re
        missing = []
        for d in self.DOCS:
            txt = (REPO / d).read_text(encoding="utf-8")
            for p in re.findall(r"`([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-{}*<>]+)+/?)`", txt):
                # only repo-rooted paths (first segment exists at the root); schema ids like src/aerolab/model/1 are not paths
                if any(c in p for c in "{}*<>") or p.startswith(("results/runs/studio", "github.com")) or not (REPO / p.split("/")[0]).exists()                         or re.fullmatch(r"src/aerolab/\w+/\d+", p):
                    continue
                if not (REPO / p.split("#")[0].rstrip("/")).exists():
                    missing.append(f"{d}: {p}")
        self.assertEqual(missing, [])

    def test_generated_tables_current(self):
        import subprocess
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "render_workspace_docs.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
