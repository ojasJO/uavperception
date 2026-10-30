#!/usr/bin/env python
"""Compare the official-toolkit scores of the released TrainedCST results with the paper's Table 3 / Table 5.

    venv\\Scripts\\python.exe experiments\\cst_anti_uav\\scripts\\report_reproduction.py [<official_scores.json>]

Paper values: arXiv:2507.23473 v2 (sha256 af0025ab…), Table 3 "Trained with CST and test on CST" (mSA, P, S) and the GlobalTrack /
PrDiMP CST rows of Table 5, transcribed once here. Writes <dir>/REPRODUCTION.md next to the scores; exits non-zero if any released
tracker's mSA differs from the paper by more than 0.005 (rounding).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT = REPO / "experiments" / "cst_anti_uav" / "reports" / "official_reproduction_test" / "official_scores.json"
# Table 3, CST columns: mSA, P (precision@20px), S (success AUC). Keys = the toolkit's tracker names.
TABLE3 = {"GlobalTrack": (35.92, 58.72, 35.38), "KYS": (26.28, 40.20, 25.93), "PrDiMP": (29.76, 42.55, 29.28),
          "Stark_s": (23.27, 40.54, 23.10), "Stark_st": (25.63, 44.99, 25.44), "OSTrack_256": (26.34, 46.95, 26.16),
          "OSTrack_384": (25.25, 45.99, 25.10), "AiATrack": (26.83, 47.65, 26.64), "Tomp": (22.84, 39.81, 22.66),
          "Mixformer": (26.46, 47.14, 26.19), "MixformerV2": (29.39, 52.37, 29.09), "DropTrack": (26.89, 49.08, 26.65),
          "ARTrack_256": (25.14, 45.39, 24.96), "ARTrack_384": (25.19, 46.56, 25.04), "SeqTrack": (26.53, 47.67, 26.38),
          "GRM": (26.96, 48.70, 26.80), "RomTrack": (27.43, 48.20, 27.23), "HIPTrack": (25.62, 47.48, 25.42),
          "AQATrack": (27.60, 50.43, 27.45), "ARTrackV2": (22.14, 39.73, 22.03), "SiamDT": (35.84, 57.96, 35.28),
          "Refocus-TIR": (26.51, 47.49, 26.26)}
# Table 5, CST-trained rows (frame-level SA): OC OV SV TC DBC CDB TS SS NS LS
TABLE5 = {"GlobalTrack": dict(OC=44.53, OV=65.77, SV=37.62, TC=26.63, DBC=17.26, CDB=26.95, tiny=15.83, small=46.14, normal=59.15,
                              large=34.52)}
TINY_PRDIMP = 28.20


def main():
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    s = json.loads(p.read_text(encoding="utf-8"))
    tr = s["trackers"]
    worst = 0.0
    md = ["# Official-protocol reproduction of the CST Anti-UAV results (released TrainedCST predictions)", "",
          f"*Generated from `{p.relative_to(REPO)}` ({s['generated']}); toolkit commit `{s['toolkit']['commit'][:8]}`, unmodified; "
          f"results archive sha256 `{s.get('published_archive_sha256', '?')[:12]}…`; subset `{s['subset']}`, {s['n_sequences']} sequences.*",
          "", "| tracker | paper mSA | ours mSA | Δ | paper P | ours P@20 | paper S | ours S-AUC | Track A mSA | max per-seq |Δ| official vs Track A |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, (pm, pp, ps) in sorted(TABLE3.items(), key=lambda kv: -kv[1][0]):
        if n not in tr:
            md.append(f"| {n} | {pm:.2f} | not in the released archive | | {pp:.2f} | | {ps:.2f} | | | |")
            continue
        r = tr[n]
        d = r["mSA"] - pm
        worst = max(worst, abs(d), abs(r["precision_20px"] - pp), abs(r["success_auc"] - ps))
        md.append(f"| {n} | {pm:.2f} | {r['mSA']:.2f} | {d:+.2f} | {pp:.2f} | {r['precision_20px']:.2f} | {ps:.2f} | "
                  f"{r['success_auc']:.2f} | {r['track_a_crosscheck']['mSA']:.2f} | {r['track_a_crosscheck']['max_abs_seq_diff']:.1e} |")
    extra = sorted(set(tr) - set(TABLE3))
    if extra:
        md += ["", f"Evaluated but not in Table 3: {', '.join(extra)}"]
    md += ["", f"Largest |ours − paper| over mSA, P and S of the released trackers: **{worst:.4f}** points.", "",
           "## Frame-level attribute SA (Table 5, CST-trained GlobalTrack)", "",
           "| attribute | paper | official toolkit (`Evaluation_for_ALL_Frame_SA.py`, as-is) | all flagged frames (corrected) | frames |",
           "|---|---:|---:|---:|---:|"]
    g = tr["GlobalTrack"]
    for k, pv in TABLE5["GlobalTrack"].items():
        off = g["frame_attr_SA_official"].get(k)
        cor = g["frame_attr_SA_corrected"].get(k, {})
        cs = cor.get("SA")
        md.append(f"| {k} | {pv:.2f} | {'—' if off is None else f'{off:.2f}'} | "
                  f"{'—' if cs is None else f'{cs:.2f}'} | {cor.get('n_frames', '—')} |")
    pt = tr.get("PrDiMP", {}).get("frame_attr_SA_corrected", {}).get("tiny", {}).get("SA")
    md += ["", f"Tiny-target SA of PrDiMP (paper's best tiny-size tracker, {TINY_PRDIMP}): {pt:.2f}." if pt is not None else ""]
    best_tiny = max(((n, r["frame_attr_SA_corrected"]["tiny"]["SA"]) for n, r in tr.items()), key=lambda x: x[1])
    md.append(f"Best tiny-target SA among the released trackers: {best_tiny[0]} {best_tiny[1]:.2f}.")
    (p.parent / "REPRODUCTION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if worst > 0.005:
        sys.exit(f"reproduction mismatch: {worst:.4f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
