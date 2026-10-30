#!/usr/bin/env python
"""Generate docs/RESULTS.md from machine-readable artifacts only (never edit RESULTS.md by hand).

    venv\\Scripts\\python.exe tools\\render_results.py

Sources: artifacts/anti_uav_ledger.json (Track A, evidence classes), experiments/aerotrack_generalization/reports/
{gen_decisions.json, gen_test_ledger.jsonl} (selected detector), experiments/aerotrack_v2/hpc/reports/v2_results.json and
experiments/aerotrack_v2/hpc/reports/v2_decisions.json (v2 on the HPC), experiments/aerotrack_v2/reports/v2_results.json (OMEN, historical),
experiments/anti_uav/runs/normal_ir/eval_val_IR_last_hpc_HC3/metrics.json (host check HC3), experiments/deployment/reports/gate_*.json.
A missing source renders as "pending" -- nothing is filled in by hand.
"""
import json
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "RESULTS.md"


def load(rel):
    p = REPO / rel
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def jsonl(rel):
    p = REPO / rel
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def f(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def track_a(L):
    L.append("## 1. Track A — Anti-UAV reproduction (Jiang et al. 2021, arXiv:2101.08466v3; Protocol II, IR, mSA)\n")
    led = load("artifacts/anti_uav_ledger.json")
    if not led:
        L.append("pending (`artifacts/anti_uav_ledger.json` missing)\n"); return
    E = led["entries"]
    def pick(cls, method, split, seed=None, base_only=True):
        return [e for e in E if e["class"] == cls and e.get("method") == method and e.get("split") == split
                and (seed is None or e.get("seed") == seed) and (not base_only or not ("benchmark_" in e["id"] or "tableVI" in e["id"]))]
    L.append("PAPER = Table V; OURS — FAITHFUL = Phase A (seed 0, frozen protocol); OURS — CONTROLLED = seed-1 replication "
             "(same protocol). Columns are never mixed; differences are OURS − PAPER.\n")
    L.append("| method | split | PAPER | OURS — FAITHFUL (s0) | OURS — CONTROLLED (s1) | seed mean ± half-range | Δ mean vs paper |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for method in ("normal", "DFSC"):
        for split in ("val", "test"):
            p = pick("PAPER_RESULT", method, split)
            fa = pick("FAITHFUL_REPRODUCTION", method, split, 0)
            co = pick("CONTROLLED_REPLICATION", method, split, 1)
            pv = p[0]["mSA"] if p else None
            a = fa[0]["mSA"] if fa else None
            b = co[0]["mSA"] if co else None
            vals = [x for x in (a, b) if x is not None]
            m = statistics.mean(vals) if vals else None
            hr = (max(vals) - min(vals)) / 2 if len(vals) == 2 else None
            L.append(f"| {method} | {split} | {f(pv)} | {f(a)} | {f(b)} | {f(m)} ± {f(hr)} | "
                     f"{'—' if m is None or pv is None else f'{m - pv:+.2f}'} |")
    L.append(f"\nTEST ledger lines: {led.get('n_test_ledger_lines')} (`{led.get('test_ledger')}`). Seed-0 checkpoints have been scored on "
             "TEST four times (registered variants; see `docs/PROJECT_KNOWLEDGE.md` §A.4).\n")
    L.append("**Exploratory / Phase B variants (not reproduction columns)**\n")
    L.append("| id | split | mSA |")
    L.append("|---|---|---:|")
    for e in E:
        if e["class"] == "EXPLORATORY_EXPERIMENT" and e.get("mSA") is not None:
            L.append(f"| {e['id']} | {e.get('split')} | {f(e['mSA'])} |")
    L.append("\n**Claims (from the ledger)**: " + "; ".join(e["id"].split("/", 1)[1] for e in E if e["class"] == "FINAL_CLAIM") + "\n")
    hc3 = load("experiments/anti_uav/runs/normal_ir/eval_val_IR_last_hpc_HC3/metrics.json")
    omen = next((e["mSA"] for e in E if e["id"] == "repro/normal_ir/seed0/val"), None)
    L.append("**Host check HC3** (amendment A1: Phase A `normal_ir` VAL re-scored on a V100; pass if |Δ| ≤ 0.05): " +
             ("pending" if not hc3 else
              f"HPC {f(hc3['aggregate']['mSA_percent'], 5)} vs OMEN {f(omen, 5)} (Δ {hc3['aggregate']['mSA_percent'] - omen:+.5f}) -> {'PASS' if abs(hc3['aggregate']['mSA_percent'] - omen) <= 0.05 else 'FAIL'}") + "\n")
    L.append("**H3 (optimisation regime)**: " + ("pending" if not load("experiments/anti_uav/reports/H3_optimisation_regime.json")
                                                 else "see `experiments/anti_uav/reports/H3_optimisation_regime.md`") + "\n")


def track_b(L):
    L.append("## 2. Track B — AeroTrack detector\n")
    t = jsonl("experiments/aerotrack_generalization/reports/gen_test_ledger.jsonl")
    L.append("### 2.1 Selected detector of the generalization series (`gen_temporal_fp_g4_full_e20`, corpus v1, TEST spent once)\n")
    if t:
        m = t[0]["metrics"]
        L.append("| metric | TEST |\n|---|---:|")
        for k in ("cst_drone_AP50", "anti_uav_drone_AP50", "det_fly_drone_AP50", "drone_AP50", "cst_drone_micro_AP50", "mAP50"):
            L.append(f"| {k} | {f(m.get(k), 3)} |")
        L.append(f"\nweights sha256 `{t[0]['weights_sha256'][:12]}…`, set `{t[0]['set_sha256'][:12]}…` (corpus v1, CST train truncated "
                 "to 84/120 sequences — data defect, see AGENT_CONTEXT §B).\n")
    else:
        L.append("pending\n")
    L.append("### 2.2 v2 campaign on corpus v2 VAL (complete CST; HPC, fp16; amendment A1)\n")
    for label, rel in (("HPC (registered comparisons)", "experiments/aerotrack_v2/hpc/reports/v2_results.json"),
                       ("OMEN (historical, bf16; not part of any comparison)", "experiments/aerotrack_v2/reports/v2_results.json")):
        d = load(rel)
        L.append(f"**{label}** — `{rel}`\n")
        if not d:
            L.append("pending\n"); continue
        L.append("| run | config | seed | CST AP50 | Anti-UAV AP50 | Det-Fly AP50 | pooled AP50 | CST micro AP50 | CST urban-areas |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in d["rows"]:
            if r.get("set") != "val":
                continue
            m = r["metrics"]
            L.append(f"| {r['run']} | {r.get('config')} | {r.get('seed')} | {f(m.get('cst_drone_AP50'), 3)} | "
                     f"{f(m.get('anti_uav_drone_AP50'), 3)} | {f(m.get('det_fly_drone_AP50'), 3)} | {f(m.get('drone_AP50'), 3)} | "
                     f"{f(m.get('cst_drone_micro_AP50'), 3)} | {f(m.get('cst_urban-areas_drone_AP50'), 3)} |")
        L.append("")
    hc = load("experiments/hpc/reports/host_checks.json")
    if hc:
        L.append("**Host checks (amendment A1; `experiments/hpc/reports/host_checks.json`)**\n")
        L.append("| check | weights | Δ CST AP50 | Δ Anti-UAV AP50 | Δ pooled AP50 | verdict |\n|---|---|---:|---:|---:|---|")
        for x in hc.get("HC1", []):
            d = x["delta"]
            L.append(f"| HC1 eval host ({x['omen_run']}) | {x['weights']} | {d['cst_drone_AP50']:+.4f} | {d['anti_uav_drone_AP50']:+.4f} | "
                     f"{d['drone_AP50']:+.4f} | {'PASS' if x['pass'] else 'FAIL'} (|Δ| ≤ 0.005) |")
        if hc.get("HC2"):
            d = hc["HC2"]["delta_hpc_trained_minus_omen_trained"]
            L.append(f"| HC2 training host (descriptive) | v2_stock_s0 HPC − OMEN | {d['cst_drone_AP50']:+.4f} | {d['anti_uav_drone_AP50']:+.4f} | "
                     f"{d['drone_AP50']:+.4f} | one sample, no decision |")
        L.append("")
    dec = load("experiments/aerotrack_v2/hpc/reports/v2_decisions.json")
    if not dec:
        L.append("**Registered decisions (HPC)**: pending\n")
    else:
        L.append(f"**Registered decisions (HPC; `experiments/aerotrack_v2/hpc/reports/v2_decisions.json`, bootstrap B = {dec.get('B')})**\n")
        L.append("| id | question | per-seed Δ CST AP50 | mean Δ | 95 % CI | Anti-UAV Δ | status |")
        L.append("|---|---|---|---:|---|---:|---|")
        for c in dec["decisions"]:
            ci = c.get("ci95") or [float("nan")] * 2
            L.append(f"| {c['id']} | {c['question']} | {', '.join(f'{v:+.3f}' for v in c.get('per_pair', []))} | "
                     f"{c['mean_delta']:+.4f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {c.get('anti_uav_mean_delta', 0):+.4f} | {c['status']} |")
        L.append(f"\n**Selected candidate (registered selection rule)**: `{dec.get('selected_candidate')}`. Caveat: "
                 f"{dec.get('caveat')}; C4 compares an HPC-trained with an OMEN-trained model (training-host effect HC2 above).\n")


def final_test(L):
    L.append("### 2.3 v2 final TEST — the single registered pass (`experiments/aerotrack_v2/final_protocol.json`)\n")
    d = load("experiments/aerotrack_v2/hpc/reports/final_test.json")
    if not d:
        L.append("pending\n"); return
    m = d["detection"]
    L.append(f"Candidate `{d['candidate']}` (weights `{d['weights_sha256'][:12]}…`), tracker parameters frozen on VAL before the pass. "
             + ("**Note**: the pass crashed after its detection part and was finished with `--resume-after-crash` (F-05; nothing chosen "
                "after the pass)." if d.get("resumed_after_crash") else "") + "\n")
    L.append("| detection (TEST) | drone AP50 |\n|---|---:|")
    for k in ("cst_drone_AP50", "anti_uav_drone_AP50", "det_fly_drone_AP50", "drone_AP50", "cst_drone_micro_AP50", "cst_urban-areas_drone_AP50"):
        L.append(f"| {k} | {f(m.get(k), 3)} |")
    L.append("\n| tracker | group | n seq | mSA | located | false alarms (invisible frames) |\n|---|---|---:|---:|---:|---:|")
    for kind in ("ste", "argmax"):
        for grp, v in d["sot"][kind].items():
            L.append(f"| {kind} | {grp} | {v.get('n_sequences')} | {f(v.get('mSA'))} | {f(100 * v['located_rate'], 1)} % | "
                     f"{f(100 * v['false_alarm_rate'], 1)} % |")
    F1 = d["F1_vs_globaltrack_anti_uav_IR_test"]
    L.append(f"\n**F1** STE vs our faithful GlobalTrack on the 91 Anti-UAV IR TEST sequences (GlobalTrack {f(F1['globaltrack_mSA'])}): "
             f"Δ {F1['mean_delta']:+.2f} mSA, 95 % CI [{F1['ci95'][0]:+.2f}, {F1['ci95'][1]:+.2f}], wins {F1['wins']} / losses "
             f"{F1['losses']} → **{F1['verdict']}**.\n")
    L.append("**F3** STE vs argmax: " + "; ".join(f"{g} Δ {v['mean_delta']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] → {v['verdict']}"
                                                 for g, v in d["F3_ste_vs_argmax"].items()) + "\n")
    F2 = d["F2_cst_benchmark_reference"]
    L.append(f"**F2** (descriptive only): CST benchmark best published mSA {F2['published_best_mSA']} ({F2['tracker']}, {F2['source']}) vs ours "
             f"{f(F2['ours_ste_mSA'])} — different training data (4 domains vs CST only); the protocol-matched claim is X1's.\n")


def deployment(L):
    L.append("## 3. Deployment (OMEN RTX 5050, batch 1)\n")
    gates = sorted((REPO / "experiments" / "deployment" / "reports").glob("gate_*.json"))
    if not gates:
        L.append("pending\n"); return
    for g in gates:
        d = json.loads(g.read_text(encoding="utf-8"))
        L.append(f"**{d.get('run')}** (window {d.get('window')}, gap {d.get('gap')}) — `{g.relative_to(REPO).as_posix()}`\n")
        L.append("| backend | model p50 / p99 ms | end-to-end p50 / p99 ms | e2e FPS (mean) | device MB |")
        L.append("|---|---:|---:|---:|---:|")
        for name, b in (d.get("backends") or {}).items():
            if not isinstance(b, dict) or "latency" not in b:
                continue
            lat, e2e = b["latency"], (b.get("e2e") or {}).get("per_frame_total", {})
            L.append(f"| {name} | {f(lat.get('p50_ms'), 1)} / {f(lat.get('p99_ms'), 1)} | {f(e2e.get('p50_ms'), 1)} / "
                     f"{f(e2e.get('p99_ms'), 1)} | {f(e2e.get('fps_from_mean'), 1)} | {f(b.get('device_memory_used_mb'), 0)} |")
        L.append("")


def main():
    L = [f"# RESULTS (generated {time.strftime('%Y-%m-%d %H:%M %z')} by `tools/render_results.py` — do not edit)\n",
         "Every number below is read from the artifact named next to it. Status of running work: `docs/MASTER_STATE.md`.\n"]
    track_a(L); track_b(L); final_test(L); deployment(L)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(L)} lines)")


if __name__ == "__main__":
    main()
