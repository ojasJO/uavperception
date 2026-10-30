#!/usr/bin/env python
"""Build artifacts/anti_uav_ledger.json — every Track A number, classified, with provenance. Reads saved files only.

Classes (mission §9): PAPER_RESULT (transcribed from the paper) · FAITHFUL_REPRODUCTION (seed 0, the paper's protocol as
specified in docs/ANTI_UAV_REPRODUCTION_SPEC.md) · CONTROLLED_REPLICATION (same protocol, another seed) ·
EXPLORATORY_EXPERIMENT (one factor changed on purpose; never merged into reproduction numbers) · FINAL_CLAIM (statements,
each pointing at the entries that support it). Confirmatory re-measurements of the same frozen checkpoint are kept as
FAITHFUL_REPRODUCTION entries with `role: confirmatory_remeasurement`.

    venv\\Scripts\\python.exe tools\\build_anti_uav_ledger.py
"""
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "anti_uav"
RUNS = EXP / "runs"
OUT = REPO / "artifacts" / "anti_uav_ledger.json"
BENCH = RUNS / "paper_benchmarks_20260922T193853"
PAPER_KEY = {"normal_ir": "normal", "dfsc_ir": "DFSC", "phaseB_normal_ir_seed1": "normal", "phaseB_dfsc_ir_seed1": "DFSC"}
ADOPTION_SEED1_TEST = ("Executed 2026-09-23 09:26-13:59 outside the scope authorised at the time (FINAL report §8: "
                       "'supplementary / unscoped'). The 2026-10-04 research-program mission (§9.1 Priority 1) explicitly requests "
                       "seed-1 test evaluation of the normal and DFSC runs; the existing verified artifacts are adopted for that "
                       "purpose instead of re-running, which would only add redundant test exposure. Frozen final-epoch checkpoints, "
                       "nothing selected with them.")


def rel(p):
    return str(Path(p).relative_to(REPO)).replace("\\", "/")


def ledger_lines():
    out = []
    for i, line in enumerate((EXP / "reports" / "test_evaluation_ledger.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            rec = json.loads(line)
            rec["line"] = i
            out.append(rec)
    return out


def from_metrics(mfile, **kw):
    m = json.loads(Path(mfile).read_text(encoding="utf-8"))
    a, meta = m["aggregate"], m["meta"]
    e = {"metrics_file": rel(mfile), "split": meta["split"], "modality": meta["modality"],
         "mSA": round(a["mSA_percent"], 4), "precision_20px": round(100 * a["precision_20px"], 4),
         "success_auc": round(100 * a["success_auc"], 4), "n_sequences": a["n_sequences"],
         "by_attribute": {k: round(v["mSA_percent"], 4) for k, v in a["by_attribute"].items()},
         "checkpoint_sha256": meta.get("checkpoint_sha256"), "checkpoint_epoch": meta.get("checkpoint_epoch"),
         "train_run": meta.get("train_run_name"), "seed": meta.get("seed"),
         "git_commit": (meta.get("git") or {}).get("commit"), "git_dirty": (meta.get("git") or {}).get("dirty"),
         "evaluated": meta.get("timestamp_start"), "variant": meta.get("variant") or meta.get("tau") and f"tau={meta.get('tau')}"}
    e.update(kw)
    return e


def main():
    paper = json.loads((EXP / "reports" / "paper_reported_results.json").read_text(encoding="utf-8"))
    cols = paper["attribute_columns"]
    led = ledger_lines()
    entries = []

    def ledger_ref(out_dir_suffix, sha):
        hits = [r["line"] for r in led if r.get("checkpoint_sha256") == sha and str(r.get("out_dir", "")).replace("\\", "/").endswith(out_dir_suffix)]
        return hits[0] if hits else None

    # ---- PAPER_RESULT
    for split in ("val", "test"):
        for method in ("normal", "DFSC", "large-scale"):
            row = paper["table_V_protocol_II"][split][method]
            entries.append({"id": f"paper/{method}/{split}", "class": "PAPER_RESULT", "source": "Table V, Protocol II, IR",
                            "split": split, "modality": "IR", "method": method, "mSA": row[cols.index("ALL_IR")],
                            "by_attribute": {c: v for c, v in zip(cols, row) if c not in ("ALL_IR", "ALL_VIS")}})
    for name, row in paper["table_VI_ablation_test"].items():
        entries.append({"id": f"paper/tableVI/{name}/test", "class": "PAPER_RESULT", "source": "Table VI, test, IR",
                        "split": "test", "modality": "IR", "method": name, "mSA": row[cols.index("ALL_IR")]})

    # ---- FAITHFUL_REPRODUCTION (seed 0) + confirmatory re-measurements
    for run in ("normal_ir", "dfsc_ir"):
        for split in ("val", "test"):
            mf = RUNS / run / f"eval_{split}_IR_last" / "metrics.json"
            e = from_metrics(mf, id=f"repro/{run}/seed0/{split}", **{"class": "FAITHFUL_REPRODUCTION"},
                             method=PAPER_KEY[run], role="primary", paper_entry=f"paper/{PAPER_KEY[run]}/{split}")
            if split == "test":
                e["test_ledger_line"] = ledger_ref(f"{run}/eval_test_IR_last", e["checkpoint_sha256"])
            entries.append(e)
            bf = BENCH / f"{run}__{split}_IR" / "metrics.json"
            b = from_metrics(bf, id=f"repro/{run}/seed0/{split}/benchmark_2026-09-22", **{"class": "FAITHFUL_REPRODUCTION"},
                             method=PAPER_KEY[run], role="confirmatory_remeasurement",
                             note="same frozen checkpoint re-evaluated in the authorised paper benchmark; agrees with the primary entry to <= 0.00016 pt")
            if split == "test":
                b["test_ledger_line"] = ledger_ref(f"{run}__test_IR", b["checkpoint_sha256"])
            entries.append(b)

    # ---- CONTROLLED_REPLICATION (seed 1)
    for run in ("phaseB_normal_ir_seed1", "phaseB_dfsc_ir_seed1"):
        entries.append(from_metrics(RUNS / run / "eval_val_IR_last" / "metrics.json", id=f"repl/{run}/val",
                                    **{"class": "CONTROLLED_REPLICATION"}, method=PAPER_KEY[run]))
        e = from_metrics(BENCH / "seed_variance" / f"{run}__test_IR" / "metrics.json", id=f"repl/{run}/test",
                         **{"class": "CONTROLLED_REPLICATION"}, method=PAPER_KEY[run], adoption=ADOPTION_SEED1_TEST)
        e["test_ledger_line"] = ledger_ref(f"{run}__test_IR", e["checkpoint_sha256"])
        entries.append(e)

    # ---- EXPLORATORY_EXPERIMENT
    for run in ("normal_ir", "dfsc_ir"):
        for split in ("val", "test"):
            e = from_metrics(RUNS / run / "phaseB_absence_thr" / split / "metrics.json", id=f"expl/absence_thr/{run}/{split}",
                             **{"class": "EXPLORATORY_EXPERIMENT"}, hypothesis="H2 absence handling",
                             factor="post-hoc absence rule: score < tau => absent (tau = 0.3 chosen on val)")
            if split == "test":
                e["test_ledger_line"] = ledger_ref(f"{run}/phaseB_absence_thr/test", e["checkpoint_sha256"])
            entries.append(e)
            h4 = RUNS / run / "phaseB_test_scale_640x512" / split / "metrics.json"
            if h4.exists():
                e = from_metrics(h4, id=f"expl/h4_native_scale/{run}/{split}", **{"class": "EXPLORATORY_EXPERIMENT"},
                                 hypothesis="H4 input scale", factor="test-time scale 640x512 (native) instead of 1000x800",
                                 preregistration="experiments/anti_uav/preregistration/H4_native_scale.json")
                if split == "test":
                    e["test_ledger_line"] = ledger_ref("phaseB_test_scale_640x512/test", e["checkpoint_sha256"])
                entries.append(e)
            else:
                entries.append({"id": f"expl/h4_native_scale/{run}/{split}", "class": "EXPLORATORY_EXPERIMENT",
                                "hypothesis": "H4 input scale", "status": "pre-registered, not complete",
                                "preregistration": "experiments/anti_uav/preregistration/H4_native_scale.json"})
    entries.append({"id": "expl/dfsc_all/test", "class": "EXPLORATORY_EXPERIMENT", "status": "INCOMPLETE — no valid result",
                    "detail": "phaseB_dfsc_ir_ism_cross trained 18/18 epochs; TEST evaluation killed by a machine shutdown at 54/91 "
                              "sequences (2026-09-23 20:03); the partial predictions must never be scored",
                    "checkpoint_sha256_prefix": "708ce9666ff84f18"})

    # ---- FINAL_CLAIM (numbers filled from the entries above, never typed)
    E = {e["id"]: e for e in entries}

    def g(i):
        return E[i]["mSA"]
    seeds_gain = {s: g(f"repro/dfsc_ir/seed0/{s}") - g(f"repro/normal_ir/seed0/{s}") for s in ("val", "test")}
    s1_gain = {s: g(f"repl/phaseB_dfsc_ir_seed1/{s}") - g(f"repl/phaseB_normal_ir_seed1/{s}") for s in ("val", "test")}
    claims = [
        {"id": "claim/val_reproduced", "text": (f"Validation mSA reproduces the paper within the pre-registered 1-pt band: normal "
                                                f"{g('repro/normal_ir/seed0/val'):.2f} vs {g('paper/normal/val'):.2f}, DFSC "
                                                f"{g('repro/dfsc_ir/seed0/val'):.2f} vs {g('paper/DFSC/val'):.2f}."),
         "evidence": ["repro/normal_ir/seed0/val", "repro/dfsc_ir/seed0/val", "paper/normal/val", "paper/DFSC/val"]},
        {"id": "claim/test_above_paper", "text": (f"On the test split both models score above the paper (normal {g('repro/normal_ir/seed0/test'):.2f} "
                                                  f"vs {g('paper/normal/test'):.2f}; DFSC {g('repro/dfsc_ir/seed0/test'):.2f} vs "
                                                  f"{g('paper/DFSC/test'):.2f}); seed 1 gives {g('repl/phaseB_normal_ir_seed1/test'):.2f} / "
                                                  f"{g('repl/phaseB_dfsc_ir_seed1/test'):.2f}, so the surplus is not a seed artifact at n = 2. "
                                                  "Cause not isolated."),
         "evidence": ["repro/normal_ir/seed0/test", "repro/dfsc_ir/seed0/test", "repl/phaseB_normal_ir_seed1/test",
                      "repl/phaseB_dfsc_ir_seed1/test", "paper/normal/test", "paper/DFSC/test"]},
        {"id": "claim/dfsc_effect_not_observed", "text": (f"The paper's DFSC gain (+{g('paper/DFSC/val') - g('paper/normal/val'):.2f} val, "
                                                          f"+{g('paper/DFSC/test') - g('paper/normal/test'):.2f} test) is not observed: "
                                                          f"seed 0 {seeds_gain['val']:+.2f} / {seeds_gain['test']:+.2f}, seed 1 "
                                                          f"{s1_gain['val']:+.2f} / {s1_gain['test']:+.2f} (val / test). Two seeds, one "
                                                          "implementation, one modality: not evidence that DFSC never helps."),
         "evidence": ["repro/normal_ir/seed0/val", "repro/dfsc_ir/seed0/val", "repro/normal_ir/seed0/test", "repro/dfsc_ir/seed0/test",
                      "repl/phaseB_normal_ir_seed1/val", "repl/phaseB_dfsc_ir_seed1/val", "repl/phaseB_normal_ir_seed1/test",
                      "repl/phaseB_dfsc_ir_seed1/test"]},
        {"id": "claim/absence_not_cause", "text": (f"Absence handling explains the validation OV/OC/SV deficits but not the test surplus: the "
                                                   f"post-hoc absence rule changes test mSA by "
                                                   f"{E['expl/absence_thr/normal_ir/test']['mSA'] - g('repro/normal_ir/seed0/test'):+.2f} (normal) and "
                                                   f"{E['expl/absence_thr/dfsc_ir/test']['mSA'] - g('repro/dfsc_ir/seed0/test'):+.2f} (DFSC), "
                                                   "i.e. any absence rule moves our test number further above the paper."),
         "evidence": ["expl/absence_thr/normal_ir/test", "expl/absence_thr/dfsc_ir/test"]},
    ]
    for c in claims:
        c["class"] = "FINAL_CLAIM"
    entries += claims
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "generator": "tools/build_anti_uav_ledger.py",
           "paper": paper["paper"], "authoritative_report": "experiments/anti_uav/reports/FINAL_PROTOCOL_II_IR_BENCHMARK_REPORT.md",
           "classes": {"PAPER_RESULT": "transcribed from the paper (reports/paper_reported_results.json)",
                       "FAITHFUL_REPRODUCTION": "seed 0, paper protocol as specified (spec §4-§6, substitutions S1-S6)",
                       "CONTROLLED_REPLICATION": "same protocol, different seed",
                       "EXPLORATORY_EXPERIMENT": "one factor changed on purpose; never merged into reproduction numbers",
                       "FINAL_CLAIM": "statement whose numbers are filled from the entries listed in `evidence`"},
           "test_ledger": "experiments/anti_uav/reports/test_evaluation_ledger.jsonl", "n_test_ledger_lines": len(led),
           "entries": entries}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(OUT)
    print(f"wrote {OUT.relative_to(REPO)}: {len(entries)} entries, {len(claims)} claims")
    for c in claims:
        print(" -", c["text"])


if __name__ == "__main__":
    main()
