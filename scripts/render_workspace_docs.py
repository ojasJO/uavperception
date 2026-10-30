"""Render the generated result tables of research/*/README.md from the registry (no hand-typed numbers).

    venv\\Scripts\\python.exe scripts\\render_workspace_docs.py          # rewrite the blocks between the GENERATED markers
    venv\\Scripts\\python.exe scripts\\render_workspace_docs.py --check  # exit 1 if a README is out of date (used by the tests)
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from aerolab.registry import Registry  # noqa: E402

BEGIN, END = "<!-- BEGIN GENERATED: scripts/render_workspace_docs.py -->", "<!-- END GENERATED -->"


def f2(v):
    return "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def ci(v):
    return f"[{v[0]:.2f}, {v[1]:.2f}]" if isinstance(v, list) and len(v) == 2 else ""


def aerotrack(R):
    E = {e["experiment_id"]: e for e in R.historical_experiments}
    rows = ["| record | model (checkpoint) | dataset / split | protocol | mSA | 95 % CI | kind |", "|---|---|---|---|---:|---|---|"]
    ids = (["cst:released:GlobalTrack", "cst:paper:GlobalTrack", "cst:x1:ste", "cst:x1:argmax"]
           + [f"cst:x2:v3_n__causal_g4_cst_s{s}_e20:ste" for s in range(3)] + [f"cst:x2:v3_n__ttt_cst_s{s}_e20:ste" for s in range(3)]
           + ["cf:test:sot:anti_uav_IR", "cf:test:sot:anti_uav_RGB", "cf:test:sot:cst_IR"])
    for i in ids:
        e = E.get(i)
        if not e:
            continue
        m = e["metrics"]
        rows.append(f"| `{i}` | {', '.join(x.split(':', 1)[-1] for x in e['model_ids']) or '—'} | {e['dataset_id']} / {e['split']} | "
                    f"`{e['protocol_id']}` | {f2(m.get('mSA'))} | {ci(m.get('mSA_ci95'))} | {e['kind']} ({e['provenance']}) |")
    t = [E[f"cst:x2:v3_n__causal_g4_cst_s{s}_e20:ste"]["metrics"]["mSA"] for s in range(3)]
    s = [E[f"cst:x2:v3_n__ttt_cst_s{s}_e20:ste"]["metrics"]["mSA"] for s in range(3)]
    out = "\n".join(rows)
    out += (f"\n\nSeed means: temporal causal_g4 {sum(t) / 3:.2f}, single frame {sum(s) / 3:.2f}. The registered paired statistics "
            "(X2 vs X1: no improvement, CI spans zero; temporal vs single frame: supported) are in "
            "`experiments/cst_anti_uav/reports/FINAL_REPORT.md`. All rows above are TEST passes (used up; read-only).\n\nP1 (VAL, exploratory):\n\n"
            "| run | frozen mSA | gated mSA | Δ | verdict |\n|---|---:|---:|---:|---|\n")
    for k, e in E.items():
        if k.startswith("cst:p1:"):
            m = e["metrics"]
            out += f"| {k.split(':')[-1]} | {f2(m.get('frozen_mSA'))} | {f2(m.get('gated_mSA'))} | {f2(m.get('delta'))} | {e.get('verdict')} |\n"
    return out


def antiuav(R):
    E = {e["experiment_id"]: e for e in R.historical_experiments}
    rows = ["| method | split | paper mSA (IR) | ours mSA (IR) | ours: record |", "|---|---|---:|---:|---|"]
    for method, run in (("normal", "normal_ir"), ("DFSC", "dfsc_ir")):
        for split in ("val", "test"):
            p = E.get(f"antiuav:paper:{method}:{split}:IR")
            o = E.get(f"ledger:eval:anti_uav/repro/{run}/seed0/{split}")
            rows.append(f"| {method} (seed 0) | {split} | {f2(p and p['metrics']['mSA'])} | {f2(o and o['metrics']['mSA'])} | `{o and o['experiment_id']}` |")
    for method, run in (("normal", "phaseB_normal_ir_seed1"), ("DFSC", "phaseB_dfsc_ir_seed1")):
        for split in ("val", "test"):
            o = E.get(f"ledger:eval:anti_uav/repl/{run}/{split}")
            if o:
                rows.append(f"| {method} (seed 1, replication) | {split} | — | {f2(o['metrics']['mSA'])} | `{o['experiment_id']}` |")
    return ("\n".join(rows) + "\n\nPaper numbers are transcribed from Table V (`experiments/anti_uav/reports/paper_reported_results.json`), labelled "
            "`paper_reported` in the registry and never ranked against ours in the studio (no per-sequence data). Authoritative report: "
            "`experiments/anti_uav/reports/FINAL_PROTOCOL_II_IR_BENCHMARK_REPORT.md`.")


TARGETS = {"docs/research/aero_multi_paper/README.md": aerotrack, "docs/research/anti_uav_paper_reproduction/README.md": antiuav}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    R = Registry()
    stale = []
    for rel, fn in TARGETS.items():
        p = REPO / rel
        txt = p.read_text(encoding="utf-8")
        new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: f"{BEGIN}\n{fn(R)}\n{END}", txt, flags=re.S)
        if new != txt:
            stale.append(rel)
            if not a.check:
                p.write_text(new, encoding="utf-8")
    print(("out of date: " if a.check else "rendered: ") + (", ".join(stale) or "none"))
    return 1 if (a.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
