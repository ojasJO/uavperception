#!/usr/bin/env python
"""Generate experiments/aerotrack_v2/reports/V2_REPORT.md from the series' own JSON (never type numbers into it).

    venv\\Scripts\\python.exe experiments\\aerotrack_v2\\scripts\\report_v2.py

Sources: preregistration.json, results/runs/<run>/run_meta.json + init_check.json, reports/v2_results.json, reports/v2_decisions.json,
experiments/cst_completion/reports/corpus_v2.json, and for the selected candidate experiments/deployment/reports/gate_<run>.json
and experiments/systems/reports/{track,forecast,mot,failures}_<run>_data_unified_v2.json when they exist. Sections whose inputs
do not exist yet say so. Interpretation is limited to restating the pre-registered rule outcomes.
"""
import json
import time

import v2series as S

R = S.REPORTS
SYS = S.REPO / "experiments" / "systems" / "reports"
DEP = S.REPO / "experiments" / "deployment" / "reports"


def rd(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(v, d=4, sign=False):
    return "—" if v is None else (f"{v:+.{d}f}" if sign else f"{v:.{d}f}")


def main():
    pre = rd(S.SERIES / "preregistration.json")
    res = (rd(R / "v2_results.json") or {}).get("rows", [])
    dec = rd(R / "v2_decisions.json")
    corp = rd(S.REPO / "experiments" / "cst_completion" / "reports" / "corpus_v2.json")
    L = ["# AeroTrack v2 series — complete CST corpus, causal temporal window", "",
         f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/report_v2.py` from the series' JSON; rules frozen in "
         "`preregistration.json` (committed before the corpus existed). No number here is typed by hand.*", "",
         "## 1. Questions (pre-registered)", ""]
    L += [f"- **{c['id']}** — {c['question']} (rule: {c['rule']}, metric: {c['metric']})" for c in pre["comparisons"]]
    L += ["", f"Configurations: " + "; ".join(f"`{k}` {v}" for k, v in pre["configurations"].items()), "",
          f"Held fixed: {pre['held_fixed']}", ""]
    L += ["## 2. Data: corpus v2", ""]
    if corp:
        sp = corp["cst"]["sequences_per_split"]
        L += [f"Corpus v2 (`{corp['corpus']}`, manifest `{corp['manifest_sha256'][:12]}`): CST sequences train {sp.get('train')}, val "
              f"{sp.get('val')}, test {sp.get('test')}; CST TEST labels identical to corpus v1: {corp['checks']['cst_test_labels_identical_to_v1']}; "
              f"official CST val split used: {corp['checks']['official_val_used']}.", "",
              "| split | CST categories |", "|---|---|"]
        for s_ in ("train", "val", "test"):
            L.append(f"| {s_} | " + ", ".join(f"{k} {v}" for k, v in sorted(corp["cst"]["categories_per_split"][s_].items())) + " |")
        L.append("")
    else:
        L += ["*corpus v2 report missing*", ""]
    L += ["## 3. Runs", "", "| run | config | seed | epochs | training h | init check | last.pt sha256 |", "|---|---|---|---:|---:|---|---|"]
    for r in sorted({x["run"] for x in res}):
        m = rd(S.REPO / "results/runs" / r / "run_meta.json") or {}
        ic = rd(S.REPO / "results/runs" / r / "init_check.json") or {}
        L.append(f"| `{r}` | {m.get('config')} | {m.get('seed')} | {m.get('epochs')} | {f((m.get('train_seconds') or 0) / 3600, 2)} | "
                 f"{'ok' if ic.get('ok') else ic.get('ok', '—')} | `{(m.get('last_pt_sha256') or '')[:12]}` |")
    L += ["", "## 4. VAL results (corpus v2, every dataset capped at 5,000 images, same frames for every run)", "",
          "| run | drone AP50 | CST | CST micro | Anti-UAV | Det-Fly | mAP50 | " + " | ".join(f"CST {c}" for c in
                                                                                  ("building", "cn_mountains", "cn_sky", "jungle", "urban-areas", "water")) + " |",
          "|---|" + "---:|" * 12]
    for r in sorted((x for x in res if x["set"] == "val"), key=lambda x: x["run"]):
        m = r["metrics"]
        L.append(f"| `{r['run']}` | {f(m['drone_AP50'])} | {f(m['cst_drone_AP50'])} | {f(m['cst_drone_micro_AP50'])} | {f(m['anti_uav_drone_AP50'])} | "
                 f"{f(m['det_fly_drone_AP50'])} | {f(m['mAP50'])} | " + " | ".join(f(m.get(f"cst_{c}_drone_AP50"))
                                                                                for c in ("building", "cn_mountains", "cn_sky", "jungle", "urban-areas", "water")) + " |")
    L += ["", "## 5. Decisions (pre-registered rules)", ""]
    if dec:
        L += ["| id | question | per seed | mean Δ | 95 % CI | Anti-UAV Δ | status |", "|---|---|---|---:|---|---:|---|"]
        for d in dec["decisions"]:
            if d.get("status") == "INCOMPLETE":
                L.append(f"| {d['id']} | {d['question']} | — | — | — | — | INCOMPLETE |")
                continue
            L.append(f"| {d['id']} | {d['question']} | {', '.join(f(x, 3, True) for x in d['per_pair'])} | {f(d['mean_delta'], 4, True)} | "
                     f"[{f(d['ci95'][0], 3, True)}, {f(d['ci95'][1], 3, True)}] | {f(d.get('anti_uav_mean_delta'), 4, True)} | {d['status']} |")
        L += ["", f"**Selected candidate (selection rule): `{dec.get('selected_candidate')}`**", "",
              "Per-category CST deltas (seed mean):", ""]
        for d in dec["decisions"]:
            if d.get("per_category"):
                L.append(f"- {d['id']}: " + ", ".join(f"{c} {f(v, 3, True)}" for c, v in d["per_category"].items() if v is not None))
        L.append("")
    else:
        L += ["*decisions not available yet*", ""]
    cand = f"v2_{dec['selected_candidate']}_s0_e20" if dec and dec.get("selected_candidate") else None
    L += ["## 6. Selected candidate as a system (VAL)", ""]
    if cand:
        gate = rd(DEP / f"gate_{cand}.json")
        if gate:
            L += [f"Deployment gate on {gate['gpu_before'].get('name')} (window `{gate['window']}`, batch 1):", "",
                  "| backend | model p50 / p95 / p99 ms | end-to-end p50 / p99 ms | device MB | drone AP50 (Δ vs bf16) |", "|---|---|---|---:|---|"]
            for k, v in gate["backends"].items():
                if v.get("available"):
                    la, e2 = v["latency"], v["e2e"]["per_frame_total"]
                    L.append(f"| {k} | {la['p50_ms']:.2f} / {la['p95_ms']:.2f} / {la['p99_ms']:.2f} | {e2['p50_ms']:.2f} / {e2['p99_ms']:.2f} | "
                             f"{v['device_memory_used_mb']:.0f} | {f((v.get('parity') or {}).get('drone_AP50'))} "
                             f"({f((v.get('parity_delta_vs_torch_bf16') or {}).get('drone_AP50'), 4, True)}) |")
                else:
                    L.append(f"| {k} | not available | | | |")
            L.append("")
        tr = rd(SYS / f"track_{cand}_data_unified_v2.json")
        if tr:
            L += ["Single-object tracking on VAL (cross-fitted parameters):", "", "| tracker | group | mSA | located | false alarms | recover median |",
                  "|---|---|---:|---:|---:|---:|"]
            for k, rec in tr["trackers"].items():
                for g, v in rec["cross_fitted"].items():
                    fa = "—" if v["false_alarm_rate"] is None else f"{100 * v['false_alarm_rate']:.1f} %"
                    rm = "—" if v["recover_frames_median"] is None else v["recover_frames_median"]
                    L.append(f"| {k} | {g} | {v['mSA']:.2f} | {100 * v['located_rate']:.1f} % | {fa} | {rm} |")
            gg = tr.get("globaltrack_same_sequences")
            if gg:
                v = tr["trackers"]["ste"].get("vs_globaltrack_anti_uav_IR", {})
                L += ["", f"GlobalTrack (Track A) on the same {gg['n_sequences']} Anti-UAV IR VAL sequences: mSA {gg['mSA']:.2f}; STE − GlobalTrack "
                      f"{f(v.get('mean_delta_mSA'), 2, True)} [{f((v.get('ci95') or [None])[0], 2, True)}, {f((v.get('ci95') or [None, None])[1], 2, True)}]."]
            L.append("")
        for name, title in (("mot", "Established trackers (MOT protocol)"), ("forecast", "Forecasting"), ("failures", "Failure analysis")):
            p = SYS / f"{name}_{cand}_data_unified_v2.md"
            L.append(f"- {title}: `{p.relative_to(S.REPO)}`" + ("" if p.exists() else " (not generated yet)"))
        L.append("")
    else:
        L += ["*no candidate selected yet*", ""]
    L += ["## 7. Limitations (structural)", "",
          f"- {len(pre['seeds'])} seeds per configuration; the bootstrap measures sampling noise over VAL sequences, seed noise enters only through the per-seed deltas.",
          "- Anti-UAV VAL shares all recording sessions with TRAIN (optimistic for Anti-UAV); CST val is the official split.",
          "- C4 (data effect) has one seed per arm.", "- TEST is untouched by this report; the candidate's single pass follows `final_protocol.json`."]
    R.mkdir(parents=True, exist_ok=True)
    (R / "V2_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {(R / 'V2_REPORT.md').relative_to(S.REPO)}")


if __name__ == "__main__":
    main()
