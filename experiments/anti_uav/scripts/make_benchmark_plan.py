#!/usr/bin/env python
"""Emit the paper -> implementation benchmark matrix (reports/benchmark_execution_plan.{json,md}).

The inventory is a transcription of every result-bearing experiment in anti_uav_research_paper.pdf
(arXiv:2101.08466v3), each classified against what this repository can actually execute. Classes:

    RUN                        runnable as-is on our trained checkpoint(s), and authorised for this task
    RUN_WITH_ADAPTATION        implementable here, but needs an artifact we would have to produce first
                               (e.g. a new training run); `executed` says whether it was actually run
    NOT_APPLICABLE             the protocol/table does not admit our model, for a stated technical reason
    BLOCKED_BY_MISSING_ARTIFACT  requires a checkpoint/model/implementation we do not possess

Nothing is omitted: every table and protocol in the paper appears exactly once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from anti_uav_repro.paths import REPORTS_DIR

BENCH_RUN = "experiments/anti_uav/runs/paper_benchmarks_20260922T193853"
PY = "venv\\Scripts\\python.exe"
EV = "experiments\\anti_uav\\scripts\\evaluate.py"

# measured on this machine from prior eval_meta.json / run_meta.json (RTX 5050 Laptop, 8 GB)
COST = {
    "val_eval_hours": 1.8,     # 61,763 IR val frames at 9.5-10 fps
    "test_eval_hours": 2.4,    # 85,312 IR test frames at 9.1-10 fps
    "train_hours": 2.5,        # 18 epochs x 800 iters at ~0.55-0.6 s/iter
}

# ---------------------------------------------------------------------------------------------
# Reasons, written once so they read identically everywhere they are cited.
# ---------------------------------------------------------------------------------------------
R_PROTOCOL_I = (
    "Protocol I is defined by the paper (SS III-E) as evaluation of trackers that used "
    "'any training set except for those with UAV objects'. Both of our checkpoints are fine-tuned on the "
    "Anti-UAV IR train split, so entering them into Protocol I would violate the protocol's own "
    "precondition. Separately, the 40+ comparison trackers of Tables III/IV are third-party methods whose "
    "implementations and weights are not present in this repository, so the ranking itself cannot be "
    "reproduced. Our model's numbers under the identical metric are reported under Protocol II instead."
)
R_LARGE_SCALE = (
    "The 'large-scale' row is the public GlobalTrack checkpoint (MS COCO + LaSOT + GOT-10k). It is "
    "distributed only in mmdetection-1.x key layout via Google Drive; this repository has no copy and no "
    "verified conversion to the torchvision-based QG-RCNN re-implementation (spec S1/S6). Producing it "
    "would require a weight conversion that no official source provides."
)
R_RGB = (
    "No visible-modality (RGB) model has been trained. Per the paper (SS V-B) the visible tracker is "
    "initialised from the GlobalTrack pretrained model, which is the same unavailable artifact as the "
    "'large-scale' row (spec S6). RGB configs exist but are parked in configs/deferred/ for exactly this "
    "reason. IR numbers are never reported as RGB numbers."
)
R_PROTOCOL_III = (
    "Protocol III (unaligned multi-modal RGB+IR) is proposed by the paper as future work: the paper "
    "reports no Protocol III result anywhere, so there is no paper value to compare against. Our model is "
    "also single-modality (IR only), and no multi-modal fusion is implemented."
)
R_TABLE_VI_NOT_RUN = (
    "Every Table VI variant is implementable with existing config switches (dfsc.alpha, dfsc.cross_terms, "
    "dfsc.ism_modulation) and needs no new code, but each requires a NEW 18-epoch training run before it "
    "can be evaluated. Training was explicitly excluded from the authorised scope of this benchmark task, "
    "so these are recorded as implementable-but-not-executed, with the exact command and measured cost."
)

ATTRS = ["OV", "OC", "FM", "SV", "LI", "TC-EASY", "TC-MID", "TC-HARD", "TC", "LR", "ALL"]

# ---------------------------------------------------------------------------------------------
# The models we are benchmarking
# ---------------------------------------------------------------------------------------------
MODELS = [
    {
        "our_name": "normal_ir",
        "paper_row": "normal",
        "role": "PAPER BASELINE 'normal' - GlobalTrack fine-tuned on Anti-UAV IR train without DFSC",
        "config": "experiments/anti_uav/configs/normal_ir.yaml",
        "dfsc": {"alpha": 0.0, "cross_terms": ["cls", "reg"], "ism_modulation": "same"},
        "checkpoint": "experiments/anti_uav/runs/normal_ir/weights/last.pt",
        "epoch": 18,
        "seed": 0,
        "modality": "IR",
    },
    {
        "our_name": "dfsc_ir",
        "paper_row": "DFSC (Ours) == DFSC-a1",
        "role": "OUR REPRODUCTION of the paper's method - DFSC alpha = 1",
        "config": "experiments/anti_uav/configs/dfsc_ir.yaml",
        "dfsc": {"alpha": 1.0, "cross_terms": ["cls", "reg"], "ism_modulation": "same"},
        "checkpoint": "experiments/anti_uav/runs/dfsc_ir/weights/last.pt",
        "epoch": 18,
        "seed": 0,
        "modality": "IR",
    },
]

NOT_BENCHMARKED = [
    {
        "our_name": "phaseB_normal_ir_seed1",
        "reason": "seed-1 replicate of normal_ir; Phase B. Explicitly excluded from the authorised scope "
                  "of this task (it answers a seed-variance question, not a paper-benchmark question).",
    },
    {
        "our_name": "phaseB_dfsc_ir_seed1",
        "reason": "seed-1 replicate of dfsc_ir; Phase B. Excluded from the authorised scope, as above.",
    },
    {
        "our_name": "runs/_smoke/*",
        "reason": "pipeline smoke test, not a trained model.",
    },
    {
        "our_name": "runs/aerotrack_spd_finetuned and every other runs/ entry at the repository root",
        "reason": "AeroTrack-Net is a YOLO DETECTOR with mAP-style metrics. It is not an implementation of "
                  "this paper and shares no metric with it. Never mixed into these tables.",
    },
]


def eval_cmd(run: str, split: str, final: bool) -> str:
    out = f"{BENCH_RUN}\\{run}__{split}_IR"
    c = (f"{PY} {EV} --checkpoint experiments\\anti_uav\\runs\\{run}\\weights\\last.pt "
         f"--split {split} --modality IR --out-dir {out}")
    return c + " --final-eval" if final else c


def train_cmd(cfg: str) -> str:
    return f"{PY} experiments\\anti_uav\\scripts\\train.py --config {cfg}"


def build_inventory() -> list[dict]:
    inv: list[dict] = []

    # -------------------------------------------------- dataset / definitional tables (no model result)
    inv.append({
        "id": "table_I",
        "paper_item": "Table I - dataset comparison (sequence/box counts of Anti-UAV vs other SOT datasets)",
        "kind": "dataset statistic",
        "metric": "counts", "split": "all", "modality": "both",
        "classification": "NOT_APPLICABLE",
        "applicable_to_our_model": False,
        "existing_implementation": "experiments/anti_uav/scripts/verify_dataset.py",
        "needs_new_implementation": False,
        "will_run": False,
        "executed": False,
        "reason": "Describes the dataset, not a tracker. Already verified independently in "
                  "reports/dataset_verification.json (which records a ~2 % box-count shortfall vs Table I, "
                  "carried into the discrepancy section of the final report).",
    })
    inv.append({
        "id": "table_II",
        "paper_item": "Table II - attribute definitions (OV, OC, FM, SV, LI, TC, LR)",
        "kind": "definition",
        "metric": "none", "split": "all", "modality": "IR",
        "classification": "NOT_APPLICABLE",
        "applicable_to_our_model": False,
        "existing_implementation": "eval/metrics.py ATTRIBUTE_ORDER + data/anti_uav/label_new/*.json",
        "needs_new_implementation": False,
        "will_run": False,
        "executed": False,
        "reason": "Definitional. Consumed by the attribute-wise evaluation below rather than reproduced.",
    })
    inv.append({
        "id": "fig_3",
        "paper_item": "Fig. 3 - dataset position/scale statistics",
        "kind": "dataset statistic",
        "metric": "distributions", "split": "all", "modality": "both",
        "classification": "NOT_APPLICABLE",
        "applicable_to_our_model": False,
        "existing_implementation": None, "needs_new_implementation": False,
        "will_run": False, "executed": False,
        "reason": "Dataset property, not a tracker result.",
    })
    inv.append({
        "id": "fig_8",
        "paper_item": "Fig. 8 - qualitative comparison of DFSC vs other methods",
        "kind": "qualitative figure",
        "metric": "none", "split": "test", "modality": "both",
        "classification": "NOT_APPLICABLE",
        "applicable_to_our_model": False,
        "existing_implementation": None, "needs_new_implementation": False,
        "will_run": False, "executed": False,
        "reason": "Carries no number to compare. Protocol SS5 additionally forbids rendering TEST "
                  "sequences for demonstration purposes.",
    })

    # -------------------------------------------------- Protocol I
    for tid, tbl, split in [("table_III", "Table III - mSA (%) of baseline trackers, per attribute",  "val"),
                            ("table_IV",  "Table IV - mSA (%) of baseline trackers, per attribute",   "test")]:
        inv.append({
            "id": tid,
            "paper_item": f"{tbl} [Protocol I]",
            "kind": "Protocol I benchmark",
            "metric": "mSA + attributes", "split": split, "modality": "IR + visible",
            "classification": "NOT_APPLICABLE",
            "applicable_to_our_model": False,
            "existing_implementation": "metric yes (eval/metrics.py); the 40+ trackers no",
            "needs_new_implementation": False,
            "will_run": False, "executed": False,
            "reason": R_PROTOCOL_I,
        })
    inv.append({
        "id": "fig_7",
        "paper_item": "Fig. 7 - Protocol I precision and success plots",
        "kind": "Protocol I benchmark",
        "metric": "precision / success curves", "split": "val + test", "modality": "IR + visible",
        "classification": "NOT_APPLICABLE",
        "applicable_to_our_model": False,
        "existing_implementation": "curves yes (per-sequence success_curve is stored); the trackers no",
        "needs_new_implementation": False,
        "will_run": False, "executed": False,
        "reason": R_PROTOCOL_I,
    })

    # -------------------------------------------------- Protocol II, Table V
    for m in MODELS:
        for split in ("val", "test"):
            inv.append({
                "id": f"table_V_IR_{split}_{m['our_name']}",
                "paper_item": f"Table V - Protocol II, mSA (%), IR, {split} split, row '{m['paper_row']}'",
                "kind": "Protocol II benchmark",
                "metric": "mSA overall + 10 attributes", "split": split, "modality": "IR",
                "classification": "RUN",
                "applicable_to_our_model": True,
                "existing_implementation": "experiments/anti_uav/scripts/evaluate.py + eval/metrics.py",
                "needs_new_implementation": False,
                "will_run": True, "executed": None,
                "our_model": m["our_name"],
                "checkpoint": m["checkpoint"],
                "command": eval_cmd(m["our_name"], split, final=(split == "test")),
                "estimated_hours": COST[f"{split}_eval_hours"],
                "reason": "Directly runnable: our checkpoint is exactly the Protocol II model this row "
                          "describes (GlobalTrack fine-tuned on the Anti-UAV IR train split), the metric is "
                          "implemented from paper Eq. (1), and the split is the paper's official one.",
            })
    for split in ("val", "test"):
        inv.append({
            "id": f"table_V_IR_{split}_large_scale",
            "paper_item": f"Table V - IR, {split} split, row 'large-scale'",
            "kind": "Protocol II benchmark",
            "metric": "mSA overall + 10 attributes", "split": split, "modality": "IR",
            "classification": "BLOCKED_BY_MISSING_ARTIFACT",
            "applicable_to_our_model": False,
            "existing_implementation": "evaluator yes; checkpoint no",
            "needs_new_implementation": False,
            "will_run": False, "executed": False,
            "missing_artifact": "public GlobalTrack checkpoint (mmdet 1.x layout) + verified conversion",
            "reason": R_LARGE_SCALE,
        })
    inv.append({
        "id": "table_V_visible",
        "paper_item": "Table V - Visible ('All') column, all three rows, val and test",
        "kind": "Protocol II benchmark",
        "metric": "mSA", "split": "val + test", "modality": "visible (RGB)",
        "classification": "BLOCKED_BY_MISSING_ARTIFACT",
        "applicable_to_our_model": False,
        "existing_implementation": "evaluator and RGB manifests yes; trained RGB model no",
        "needs_new_implementation": False,
        "will_run": False, "executed": False,
        "missing_artifact": "a trained visible-modality model (and the GlobalTrack init it requires)",
        "reason": R_RGB,
    })

    # -------------------------------------------------- Table VI ablations
    variants = [
        ("DFSC-all",    "configs/phase_b/dfsc_ir_ism_cross.yaml (TO BE CREATED from dfsc_ir.yaml)",
         {"alpha": 1.0, "cross_terms": ["cls", "reg"], "ism_modulation": "cross"}),
        ("DFSC-cls",    "configs/phase_b/dfsc_ir_cls_only.yaml (TO BE CREATED from dfsc_ir.yaml)",
         {"alpha": 1.0, "cross_terms": ["cls"], "ism_modulation": "same"}),
        ("DFSC-reg",    "configs/phase_b/dfsc_ir_reg_only.yaml (TO BE CREATED from dfsc_ir.yaml)",
         {"alpha": 1.0, "cross_terms": ["reg"], "ism_modulation": "same"}),
        ("DFSC-a0.25",  "configs/phase_b/dfsc_ir_alpha0.25.yaml (EXISTS)",
         {"alpha": 0.25, "cross_terms": ["cls", "reg"], "ism_modulation": "same"}),
        ("DFSC-a0.5",   "configs/phase_b/dfsc_ir_alpha0.5.yaml (TO BE CREATED from dfsc_ir.yaml)",
         {"alpha": 0.5, "cross_terms": ["cls", "reg"], "ism_modulation": "same"}),
        ("DFSC-a2",     "configs/phase_b/dfsc_ir_alpha2.yaml (TO BE CREATED from dfsc_ir.yaml)",
         {"alpha": 2.0, "cross_terms": ["cls", "reg"], "ism_modulation": "same"}),
    ]
    # DFSC-all was subsequently trained and test-evaluated (see EXECUTED_ABLATIONS); the rest were not.
    EXECUTED_ABLATIONS = {
        "DFSC-all": {
            "run_name": "phaseB_dfsc_ir_ism_cross",
            "config": "experiments/anti_uav/configs/phase_b/dfsc_ir_ism_cross.yaml",
            "out_dir": f"{BENCH_RUN}/ablation/phaseB_dfsc_ir_ism_cross__test_IR",
            "why": "the only Table VI variant whose paper-reported effect (-2.05 pt) exceeds plausible "
                   "run-to-run noise; the other five differ from DFSC by <= 0.3 pt.",
        },
    }
    for name, cfg, switches in variants:
        run_name = "phaseB_" + name.lower().replace("-", "_").replace(".", "")
        done = EXECUTED_ABLATIONS.get(name)
        if done:
            inv.append({
                "id": f"table_VI_{name}",
                "paper_item": f"Table VI - ablation on the TEST set, mSA (%), variant '{name}'",
                "kind": "Protocol II ablation",
                "metric": "mSA overall + 10 attributes", "split": "test", "modality": "IR",
                "classification": "RUN_WITH_ADAPTATION",
                "applicable_to_our_model": True,
                "existing_implementation": "training + evaluation implemented; config added for this run",
                "needs_new_implementation": False,
                "will_run": True,
                "executed": True,
                "config": done["config"],
                "config_switches": switches,
                "our_run": done["run_name"],
                "train_command": train_cmd(done["config"].replace("experiments/anti_uav/configs/",
                                                                  "experiments\\anti_uav\\configs\\").replace("/", "\\")),
                "eval_command": eval_cmd(done["run_name"], "test", final=True),
                "out_dir": done["out_dir"],
                "estimated_train_hours": COST["train_hours"],
                "estimated_eval_hours": COST["test_eval_hours"],
                "estimated_total_hours": COST["train_hours"] + COST["test_eval_hours"],
                "reason": "Trained and evaluated: " + done["why"],
            })
            continue
        inv.append({
            "id": f"table_VI_{name}",
            "paper_item": f"Table VI - ablation on the TEST set, mSA (%), variant '{name}'",
            "kind": "Protocol II ablation",
            "metric": "mSA overall + 10 attributes", "split": "test", "modality": "IR (+ visible column, blocked)",
            "classification": "RUN_WITH_ADAPTATION",
            "applicable_to_our_model": True,
            "existing_implementation": "training + evaluation both fully implemented; only the config is new",
            "needs_new_implementation": False,
            "will_run": False,
            "executed": False,
            "config": cfg,
            "config_switches": switches,
            "train_command": train_cmd(cfg.split(" ")[0].replace("configs/", "experiments\\anti_uav\\configs\\").replace("/", "\\")),
            "eval_command": eval_cmd(run_name, "test", final=True),
            "estimated_train_hours": COST["train_hours"],
            "estimated_eval_hours": COST["test_eval_hours"],
            "estimated_total_hours": COST["train_hours"] + COST["test_eval_hours"],
            "reason": R_TABLE_VI_NOT_RUN,
        })
    inv.append({
        "id": "table_VI_visible",
        "paper_item": "Table VI - Visible column of every ablation row",
        "kind": "Protocol II ablation",
        "metric": "mSA", "split": "test", "modality": "visible (RGB)",
        "classification": "BLOCKED_BY_MISSING_ARTIFACT",
        "applicable_to_our_model": False,
        "existing_implementation": "no trained RGB model",
        "needs_new_implementation": False,
        "will_run": False, "executed": False,
        "missing_artifact": "a trained visible-modality model",
        "reason": R_RGB,
    })

    # -------------------------------------------------- Table VII precision / success
    for m in MODELS:
        inv.append({
            "id": f"table_VII_IR_{m['our_name']}",
            "paper_item": f"Table VII - precision (%) and success (%) on Anti-UAV, IR, row '{m['paper_row']}'",
            "kind": "Protocol II benchmark",
            "metric": "OTB precision@20px + success AUC", "split": "test (inferred) and val (both reported)",
            "modality": "IR",
            "classification": "RUN",
            "applicable_to_our_model": True,
            "existing_implementation": "eval/metrics.py (computed in the same pass as mSA)",
            "needs_new_implementation": False,
            "will_run": True, "executed": None,
            "our_model": m["our_name"],
            "command": "(no extra run - produced by the Table V evaluations above)",
            "estimated_hours": 0.0,
            "reason": "Precision and success come out of the same saved predictions as mSA, so these rows "
                      "cost no additional GPU time. The paper does not state which split Table VII refers "
                      "to; we report BOTH val and test and label the ambiguity rather than picking one.",
        })
    inv.append({
        "id": "table_VII_other_rows",
        "paper_item": "Table VII - rows 'large-scale' and every ablation variant; all Visible columns",
        "kind": "Protocol II benchmark",
        "metric": "precision / success", "split": "test (inferred)", "modality": "IR + visible",
        "classification": "BLOCKED_BY_MISSING_ARTIFACT",
        "applicable_to_our_model": False,
        "existing_implementation": "metric yes; the models no",
        "needs_new_implementation": False,
        "will_run": False, "executed": False,
        "missing_artifact": "the GlobalTrack public checkpoint (large-scale), a trained RGB model (visible "
                            "columns), and the six un-trained ablation checkpoints (ablation rows)",
        "reason": R_LARGE_SCALE + " " + R_RGB + " Ablation rows: " + R_TABLE_VI_NOT_RUN,
    })

    # -------------------------------------------------- Protocol III
    inv.append({
        "id": "protocol_III",
        "paper_item": "Protocol III - unaligned multi-modal (RGB + IR) tracking",
        "kind": "Protocol III",
        "metric": "mSA on both modalities", "split": "n/a", "modality": "RGB + IR",
        "classification": "NOT_APPLICABLE",
        "applicable_to_our_model": False,
        "existing_implementation": None,
        "needs_new_implementation": True,
        "will_run": False, "executed": False,
        "reason": R_PROTOCOL_III,
    })

    return inv


def to_markdown(plan: dict) -> str:
    L: list[str] = []
    a = L.append
    a("# Anti-UAV paper -> implementation benchmark execution plan")
    a("")
    a(f"Generated by `experiments/anti_uav/scripts/make_benchmark_plan.py` on {plan['generated']}.")
    a("Machine-readable twin: `benchmark_execution_plan.json`. **Do not edit by hand.**")
    a("")
    a(f"Paper: {plan['paper']}")
    a(f"Benchmark run directory: `{plan['benchmark_run_dir']}`")
    a("")
    a("## 1. Models under benchmark")
    a("")
    a("| Our run | Paper row | Modality | alpha | cross_terms | ISM | Epoch | Seed | Checkpoint |")
    a("|---|---|---|---|---|---|---|---|---|")
    for m in plan["models_benchmarked"]:
        d = m["dfsc"]
        a(f"| `{m['our_name']}` | {m['paper_row']} | {m['modality']} | {d['alpha']} | "
          f"{'+'.join(d['cross_terms'])} | {d['ism_modulation']} | {m['epoch']} | {m['seed']} | `{m['checkpoint']}` |")
    a("")
    a("Both are the **final-epoch** checkpoint (`last.pt == epoch_18.pt`). No checkpoint selection was "
      "performed, as in GlobalTrack and the paper.")
    a("")
    a("### Trained artifacts deliberately NOT benchmarked here")
    a("")
    a("| Artifact | Why not |")
    a("|---|---|")
    for m in plan["models_not_benchmarked"]:
        a(f"| `{m['our_name']}` | {m['reason']} |")
    a("")
    a("## 2. Benchmark matrix")
    a("")
    a("| Paper experiment | Paper metric | Split | Modality | Applicable? | Existing impl.? | Needs new impl.? | Will run? | Class |")
    a("|---|---|---|---|---|---|---|---|---|")
    for it in plan["inventory"]:
        a(f"| {it['paper_item']} | {it['metric']} | {it['split']} | {it['modality']} | "
          f"{'yes' if it['applicable_to_our_model'] else 'no'} | "
          f"{'yes' if it['existing_implementation'] else 'no'} | "
          f"{'yes' if it['needs_new_implementation'] else 'no'} | "
          f"{'**YES**' if it['will_run'] else 'no'} | `{it['classification']}` |")
    a("")
    a("### Counts")
    for k, v in plan["summary"]["by_classification"].items():
        a(f"- `{k}`: {v}")
    a("")
    a("## 3. Authorised live executions")
    a("")
    a("| # | Model | Split | Modality | Estimated hours | Command |")
    a("|---|---|---|---|---|---|")
    for i, it in enumerate([x for x in plan["inventory"] if x["will_run"] and x.get("estimated_hours")], 1):
        a(f"| {i} | `{it['our_model']}` | {it['split']} | {it['modality']} | {it['estimated_hours']} | `{it['command']}` |")
    a("")
    a(f"Total estimated GPU time: **{plan['summary']['authorised_gpu_hours']} h** "
      f"({plan['summary']['authorised_runs']} evaluations).")
    a("")
    a("### Follow-up executions (added under a later authorisation)")
    a("")
    a("These are not paper table rows in their own right; they exist to make the table rows "
      "interpretable.")
    a("")
    a("| Work | Purpose | Training? | Est. hours | Script |")
    a("|---|---|---|---:|---|")
    for k, v in plan["follow_up_executions"].items():
        if k.startswith("_"):
            continue
        a(f"| {k} | {v['purpose']} | {'yes' if v['training_required'] else 'no'} | "
          f"{v['estimated_hours']} | `{v['script']}` |")
    a("")
    a("Precision/success (Table VII) and the attribute breakdown (Table V columns) are computed from the "
      "same saved predictions in the same pass, so they add no GPU time.")
    a("")
    a("## 4. Not executed: Table VI ablations")
    a("")
    a("All six variants are implementable with **existing code** - only a YAML config and a training run "
      "are missing. Training was excluded from this task's authorised scope, so each is recorded below "
      "with everything needed to launch it later. **None of these was executed.**")
    a("")
    a("| Variant | Config | Switches | Train cmd | Train h | Eval h | Total h | Executed |")
    a("|---|---|---|---|---|---|---|---|")
    for it in plan["inventory"]:
        if (it["id"].startswith("table_VI_") and it["classification"] == "RUN_WITH_ADAPTATION"
                and not it.get("executed")):
            s = it["config_switches"]
            sw = f"alpha={s['alpha']}, cross_terms={'+'.join(s['cross_terms'])}, ism={s['ism_modulation']}"
            variant_name = it["paper_item"].split("'")[1]
            a(f"| {variant_name} | `{it['config']}` | {sw} | `{it['train_command']}` | "
              f"{it['estimated_train_hours']} | {it['estimated_eval_hours']} | {it['estimated_total_hours']} | **NO** |")
    a("")
    a(f"Sum if all six were run: **{plan['summary']['table_VI_gpu_hours_if_run']} h** of GPU time.")
    a("")
    a("## 5. Blocked and not-applicable items, with the precise reason")
    a("")
    for it in plan["inventory"]:
        if it["classification"] in ("BLOCKED_BY_MISSING_ARTIFACT", "NOT_APPLICABLE"):
            a(f"**{it['paper_item']}** - `{it['classification']}`")
            if it.get("missing_artifact"):
                a(f"  - missing artifact: {it['missing_artifact']}")
            a(f"  - {it['reason']}")
            a("")
    a("## 6. Test-split rules observed")
    a("")
    a("- TEST is used **only** for the final evaluation of the two already-finished, frozen checkpoints. ")
    a("- No training, tuning, thresholding, checkpoint selection or debugging decision uses TEST.")
    a("- Both TEST evaluations go through `evaluate.py --final-eval` and append to the append-only ledger "
      "`reports/test_evaluation_ledger.jsonl`.")
    a("- Previous evaluations are **not** overwritten: all output of this task lands in "
      f"`{plan['benchmark_run_dir']}`.")
    a("- The two checkpoints were frozen before this task began; re-running the evaluator on them is a "
      "deterministic re-measurement, not a new selection event.")
    a("")
    return "\n".join(L) + "\n"


def main() -> int:
    inv = build_inventory()
    by_class: dict[str, int] = {}
    for it in inv:
        by_class[it["classification"]] = by_class.get(it["classification"], 0) + 1
    auth = [x for x in inv if x["will_run"] and x.get("estimated_hours")]
    # Table VI variants that remain UN-executed (DFSC-all was trained, so it drops out of this list)
    t6 = [x for x in inv if x["id"].startswith("table_VI_")
          and x["classification"] == "RUN_WITH_ADAPTATION" and not x.get("executed")]

    plan = {
        "generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "paper": "Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking "
                 "(Jiang et al., arXiv:2101.08466v3, 8 Feb 2021) - anti_uav_research_paper.pdf",
        "benchmark_run_dir": BENCH_RUN,
        "classification_scheme": {
            "RUN": "runnable as-is on our trained checkpoint; authorised and will be executed",
            "RUN_WITH_ADAPTATION": "implementable with existing code but needs a new artifact (a training "
                                   "run); `executed` records whether it was actually run",
            "NOT_APPLICABLE": "the protocol or table does not admit our model, for the stated reason",
            "BLOCKED_BY_MISSING_ARTIFACT": "requires a checkpoint/model we do not possess",
        },
        "attributes_reported": ATTRS,
        "models_benchmarked": MODELS,
        "models_not_benchmarked": NOT_BENCHMARKED,
        "measured_cost_basis": COST,
        "inventory": inv,
        "summary": {
            "inventory_items": len(inv),
            "by_classification": by_class,
            "items_executed_so_far": sum(1 for x in inv if x.get("executed") is True),
            "authorised_runs": len(auth),
            "authorised_gpu_hours": round(sum(x["estimated_hours"] for x in auth), 1),
            "table_VI_gpu_hours_if_run": round(sum(x["estimated_total_hours"] for x in t6), 1),
            "table_VI_variants_executed": [x["paper_item"].split("'")[1] for x in inv
                                           if x["id"].startswith("table_VI_") and x.get("executed")],
            "table_VI_variants_not_executed": [x["paper_item"].split("'")[1] for x in t6],
        },
        "follow_up_executions": {
            "_note": "Added after the initial four-run benchmark, under a later authorisation. These are "
                     "not paper table rows in their own right; they exist to make the table rows "
                     "interpretable.",
            "seed_variance": {
                "purpose": "a second seed for both Phase A configurations on TEST, so the DFSC effect can "
                           "be read against run-to-run noise",
                "runs": ["phaseB_normal_ir_seed1", "phaseB_dfsc_ir_seed1"],
                "training_required": False,
                "estimated_hours": 2 * COST["test_eval_hours"],
                "script": f"{BENCH_RUN}/run_seed_variance.sh",
            },
            "ablation_DFSC_all": {
                "purpose": "the one Table VI variant whose paper effect (-2.05 pt) exceeds plausible seed "
                           "noise; tests whether our DFSC implementation reproduces a measurable ablation",
                "runs": ["phaseB_dfsc_ir_ism_cross"],
                "training_required": True,
                "estimated_hours": COST["train_hours"] + COST["test_eval_hours"],
                "script": f"{BENCH_RUN}/run_ablation_dfsc_all.sh",
            },
            "metric_audit": {
                "purpose": "from-scratch re-implementation of Eq. (1)/precision/success to audit the "
                           "reported numbers; CPU only",
                "training_required": False,
                "estimated_hours": 0.0,
                "script": "experiments/anti_uav/scripts/verify_metrics_independently.py",
            },
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "benchmark_execution_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8")
    (REPORTS_DIR / "benchmark_execution_plan.md").write_text(to_markdown(plan), encoding="utf-8")
    print(f"inventory items: {len(inv)}")
    for k, v in sorted(by_class.items()):
        print(f"  {k:30s} {v}")
    print(f"authorised: {len(auth)} evaluations, ~{plan['summary']['authorised_gpu_hours']} GPU-h")
    print("wrote", REPORTS_DIR / "benchmark_execution_plan.json")
    print("wrote", REPORTS_DIR / "benchmark_execution_plan.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
