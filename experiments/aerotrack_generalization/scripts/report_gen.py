#!/usr/bin/env python
"""Final report of the generalization series, generated from the series' JSON files and ledger (no hand-typed numbers).

    python experiments/aerotrack_generalization/scripts/report_gen.py [--tag _smoke]

Sections follow the brief (problem -> evidence -> diagnosis -> confounds -> experiments -> configurations -> development results ->
generalization evidence -> uncertainty -> test -> improved / not improved -> credibility -> limitations -> next step). Statements are
labelled [fact] (read from files), [measured] (computed by this series), [rule] (outcome of the pre-registered decision rule) or
[inference]; free-text interpretation lives in ../REPORT_NOTES.md and is included verbatim under an [inference] heading.
Output: reports/GEN_REPORT<tag>.md"""
import argparse
import json
import time

import _series as S
from plan_gen import PLAN


def j(name):
    p = S.REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f4(v):
    return "—" if v is None else f"{v:.4f}"


def sgn(v):
    return "—" if v is None else f"{v:+.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t = a.tag
    res = j(f"gen_results{t}.json") or {"rows": []}
    dec = j(f"gen_decisions{t}.json")
    dd, dt, bv, dl = j("diag_data.json"), j("diag_training.json"), j("bootstrap_val.json"), j("diag_localization.json")
    ledger_file = S.SERIES / ("ledger_smoke.jsonl" if t else "ledger.jsonl")
    led = [json.loads(x) for x in ledger_file.read_text(encoding="utf-8").splitlines() if x.strip()] if ledger_file.exists() else []
    trained = [x for x in led if x.get("type") == "training"]
    test_led = S.REPORTS / "gen_test_ledger.jsonl"
    tests = [json.loads(x) for x in test_led.read_text(encoding="utf-8").splitlines() if x.strip()] if (test_led.exists() and not t) else []
    R = {(r["run"], r["set"]): r for r in res["rows"]}

    L = [f"# AeroTrack generalization investigation — report{' (SMOKE TEST OF THE MACHINERY)' if t else ''}", "",
         f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/report_gen.py` from `reports/*.json` and `{ledger_file.name}`. Labels: "
         "[fact] read from files · [measured] computed in this series · [rule] outcome of the pre-registered rule · [inference] interpretation.*", ""]

    L += ["## 1. Original problem", "",
          "[fact] The repair grid (`experiments/aerotrack_repair/reports/ablation_grid_e20.md`) found no factor credited: in all five "
          "pre-registered comparisons the ablated model scored higher on validation, and stock YOLO11n beat the full AeroTrack recipe. "
          "The question here: is there a change with a plausible mechanism that improves generalisation, verified without tuning to the "
          "validation or test sets?", ""]

    L += ["## 2–4. Evidence, diagnosis, confounds", ""]
    if dt:
        i = dt["init"]
        L.append(f"- [measured] COCO-initialised parameters: SPD layout {i['yolo11-spd.yaml']['from_coco_pct']} % of {i['yolo11-spd.yaml']['params_M']} M, "
                 f"stride layout {i['yolo11-stride.yaml']['from_coco_pct']} % of {i['yolo11-stride.yaml']['params_M']} M (D2).")
        fc = i["function_preserving_init_check"]
        L.append(f"- [measured] function-preserving init: SPD vs stride max |Δ| {fc['spd_mapped_vs_stride_max_abs_diff']:.2e} on outputs up to "
                 f"{fc['output_max_abs']:.0f}; centre-frame stem ignores t±1 (max |Δ| {fc['centre_frame_init_ignores_neighbours_max_abs_diff']:.1e}); "
                 f"the grid's tiled stem responds to the frame mean (max |Δ| {fc['grid_tiled_stem_neighbours_vs_ttt_max_abs_diff']:.0f}) (D2).")
        cu = dt["curves"]
        L.append("- [fact] grid training loss epoch 0 → 19: " + "; ".join(f"{k} {v['train_loss_epoch0']} → {v['train_loss_epoch19']}" for k, v in cu.items()) + " (D2).")
    if dd:
        au, cs, df = dd["anti_uav"], dd["cst"], dd["det_fly"]
        mo = dd["target_motion_per_frame_px"]
        L += [f"- [measured] Anti-UAV VAL shares {au['sessions_shared']['train&val']} of {au['recording_sessions']['val']} recording sessions with TRAIN; "
              f"TEST shares {au['sessions_shared']['train&test']} (D1).",
              f"- [measured] CST TEST scene categories absent from TRAIN: {', '.join(cs['test_categories_absent_from_train'])} (D1).",
              f"- [measured] Det-Fly VAL frames whose adjacent frame (id ±1) is in TRAIN: {df['val_frame_id_distance_to_nearest_train_frame']['==1']} of "
              f"{df['val_frame_id_distance_to_nearest_train_frame']['n']} (D1).",
              f"- [measured] median target motion per frame at the network input: CST {mo['cst/IR/train']['median']} px "
              f"({100 * mo['cst/IR/train']['frac_lt_1px']:.0f} % < 1 px), Anti-UAV IR {mo['anti_uav/IR/train']['median']} px, RGB {mo['anti_uav/RGB/train']['median']} px (D1)."]
    if bv:
        L.append("- [measured] grid VAL comparisons with sequence-bootstrap 95 % CIs (sampling noise only): " + "; ".join(
            f"{c['factor']} {c['metric']} {c['ablated'].split('__')[1]} {c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
            for c in bv["comparisons"]) + " (D4).")
    if dl:
        L.append("- [measured] CST micro drones at last.pt, found-but-mislocalised vs missed: " + "; ".join(
            f"{k.split('__')[0].replace('repair_abl_', '').replace('_e20', '')} {v['iou_0.25_to_0.5_found_but_mislocalised']:.3f} / {v['iou_lt_0.25_missed']:.3f}"
            for k, v in dl.items()) + " (D5).")
    L.append("")

    L += ["## 5–6. Experiments and exact configurations", "",
          f"[fact] Pre-registered plan (ledger P1, `scripts/plan_gen.py`): reference `{PLAN['reference']}`; candidates {', '.join(PLAN['candidates'])}; "
          f"{PLAN['epochs']} epochs, recipe `c_model/train.py` STAGES['baseline'] (25,000 weighted samples/epoch, batch 32 × acc 2, AdamW 1e-3 cosine); "
          f"function-preserving init for every configuration; seed = fold index; checkpoint {PLAN['checkpoint']}; primary {PLAN['primary_metric']} on the "
          f"held-out DEV fold; non-inferiority {PLAN['noninferiority_metric']}; margin {PLAN['margin_ap50']}; rules: {'; '.join(PLAN['rules'])}; futility: {PLAN['futility']}.", "",
          "| run | config | fold | seed | epochs | launches | training time | last.pt sha | init check |", "|---|---|---|---|---|---|---|---|---|"]
    for x in trained:
        ic = x.get("init_check") or {}
        L.append(f"| `{x['run']}` | {x['config']} | {x['fold']} | {x.get('seed')} | {x['epochs']} | {x.get('launches')} | "
                 f"{(x.get('train_seconds') or 0) / 3600:.2f} h | {str(x.get('last_pt_sha256'))[:12]} | {'ok' if ic.get('ok') else ic} |")
    L.append("")

    L += ["## 7–8. Development results (held-out DEV folds: unseen Anti-UAV sessions, unseen CST sequences, held-out Det-Fly flight chunks)", "",
          "| run | set | CST drone AP50 (primary) | Anti-UAV drone AP50 | Det-Fly drone AP50 | CST drone-micro AP50 | pooled drone AP50 | images |",
          "|---|---|---|---|---|---|---|---|"]
    for (run, s), r in sorted(R.items()):
        m = r["metrics"]
        L.append(f"| `{run}` | {s} | {f4(m.get('cst_drone_AP50'))} | {f4(m.get('anti_uav_drone_AP50'))} | {f4(m.get('det_fly_drone_AP50'))} | "
                 f"{f4(m.get('cst_drone_micro_AP50'))} | {f4(m.get('drone_AP50'))} | {m.get('n_images')} |")
    L.append("")

    fa9 = j("final_analysis.json")
    if fa9:
        L += ["[measured] secondary, official VAL (optimistic set, see §2–4), candidate − stock per fold, CST drone AP50: " + "; ".join(
            f"{c} " + " / ".join(f"{v[k]['cst_drone_AP50']:+.3f}" for k in ("0", "1", "2")) for c, v in fa9["val_secondary_deltas"].items()) + ".", "",
              "[measured] training curves (held-out-fold monitoring mAP50 and training loss, slope per epoch over the last 5 epochs): " + "; ".join(
            f"{k.replace('gen_', '').replace('_e20', '')} {v['monitor_mAP50_slope_last5']:+.4f} / {v['train_loss_slope_last5']:+.4f}" for k, v in fa9["training_curves"].items()) + ".", ""]
    L += ["## 9. Decisions and statistical uncertainty", ""]
    ev = S.REPO / "runs" / "aerotrack_generalization" / f"campaign{t}" / "events.jsonl"
    skipped = [json.loads(x)["run"] for x in (ev.read_text(encoding="utf-8").splitlines() if ev.exists() else [])
               if '"skipped_time_budget"' in x]
    if dec and dec["budget_h4"]["status"] == "pending" and any("_e40" in s for s in skipped):
        dec["budget_h4"]["status"] = "not run: skipped by the pre-registered 24 h development time guard"
    for v in (dec or {}).get("candidates", {}).values():                  # analyze_gen lists the rules that FAILED
        if v["status"].startswith("not supported: "):
            v["status"] = "not supported — failed rule " + v["status"][len("not supported: "):]
    if dec:
        L += ["| candidate | vs | fold deltas (primary) | fold-mean Δ | sd | bootstrap 95 % CI | Anti-UAV mean Δ | [rule] status |", "|---|---|---|---|---|---|---|---|"]
        for name, v in list(dec["candidates"].items()) + [("budget_h4", dec["budget_h4"])]:
            fd = ", ".join(f"f{k}: {sgn(f['delta'])} ({f['relative_pct']:+.1f} %)" if f.get("relative_pct") is not None else f"f{k}: {sgn(f['delta'])}"
                           for k, f in v["folds"].items())
            ci = v.get("ci95_mean_delta")
            L.append(f"| {v['candidate']} | {v['reference']} | {fd or '—'} | {sgn(v.get('mean_delta'))} | {f4(v.get('sd_delta'))} | "
                     f"{f'[{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else '—'} | {sgn(v.get('anti_uav_mean_delta'))} | {v['status']} |")
    else:
        L.append("*No decisions yet.*")
    L.append("")

    L += ["## 10. Final test", ""]
    if tests:
        L += ["| run | test set | CST drone AP50 | Anti-UAV drone AP50 | pooled drone AP50 | when |", "|---|---|---|---|---|---|"]
        for x in tests:
            m = x["metrics"]
            L.append(f"| `{x['run']}` | `{x['set_sha256'][:12]}` | {f4(m.get('cst_drone_AP50'))} | {f4(m.get('anti_uav_drone_AP50'))} | {f4(m.get('drone_AP50'))} | {x['timestamp']} |")
        L.append("[fact] scored once; the candidate was selected on the development folds before the test split was read.")
        fa = j("final_analysis.json")
        if fa:
            tb = fa["test_bootstrap"]
            L += ["", "[measured] uncertainty of that single test result (sequence-level bootstrap of its recorded predictions, B = 2000): " +
                  "; ".join(f"{k} {v['point']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] ({v['sequences']} sequences)" for k, v in tb.items()) + "."]
            sv = fa["test_cst_seen_vs_unseen"]
            L.append(f"[measured] CST test by scene type: categories seen in TRAIN {sv['seen_categories_AP50']:.4f} ({sv['seen_sequences']} sequences), "
                     f"categories absent from TRAIN {sv['unseen_categories_AP50']:.4f} ({sv['unseen_sequences']} sequences); per category: " +
                     ", ".join(f"{c} {v['drone_AP50']:.3f} ({v['sequences']}{'' if v['seen_in_train'] else ', unseen'})" for c, v in fa["test_cst_by_category"].items()) + ".")
            rf = fa["test_reference_stock_yolo11n"]
            L += ["", "[fact] descriptive reference — stock YOLO11n's single test evaluation (repair series, same image set, same protocol; a different "
                  "training run of the same recipe, so NOT a paired or seed-controlled comparison, and no per-image records exist for a paired CI):", "",
                  "| metric | selected (last.pt) | stock YOLO11n last | stock YOLO11n best | Δ vs stock last |", "|---|---|---|---|---|"]
            for k in rf["selected"]:
                L.append(f"| {k} | {f4(rf['selected'][k])} | {f4(rf['stock']['last'][k])} | {f4(rf['stock']['best'][k])} | {sgn(rf['delta_vs_stock_last'][k])} |")
    else:
        L.append("[fact] The test split was not read in this series (no candidate satisfied the pre-registered rule, or the final step has not run). "
                 "The repair series' single test evaluation of stock YOLO11n remains the reference result.")
    L.append("")

    L += ["## 11–13. What improved, what did not, credibility", ""]
    if dec:
        sup = [k for k, v in dec["candidates"].items() if v["status"] == "supported"]
        rej = [k for k, v in dec["candidates"].items() if v["status"] != "supported"]
        rej_txt = ", ".join(f"{k} ({dec['candidates'][k]['status']})" for k in rej) or "none"
        L.append(f"- [rule] supported: {', '.join(sup) if sup else 'none'}; not supported / rejected: {rej_txt}.")
        L.append(f"- [rule] training budget (H4, 40 vs 20 epochs): {dec['budget_h4']['status']}.")
    notes = S.SERIES / "REPORT_NOTES.md"
    if notes.exists() and not t:
        L += ["", "### [inference] Interpretation, limitations and recommended next step", "", notes.read_text(encoding="utf-8").strip()]
    out = S.REPORTS / f"GEN_REPORT{t}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
