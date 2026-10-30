#!/usr/bin/env python
"""Score CST Anti-UAV tracking results with the OFFICIAL toolkit (github.com/PCwenyue/CST-Anti-UAV), unmodified.

    venv\\Scripts\\python.exe experiments\\cst_anti_uav\\scripts\\official_eval.py --published --subset test --out <dir>
    venv\\Scripts\\python.exe experiments\\cst_anti_uav\\scripts\\official_eval.py --tracker NAME=<results dir> --subset val --out <dir>

The toolkit (`data/cst_anti_uav_official/toolkit`, commit recorded in the output) is imported as-is: `ExperimentCSTAntiUAV.report`
computes per-sequence SA (paper Eq. 1), mSA, success AUC and precision@20px; `ExperimentCSTAntiUAVF.report` the frame-level
attribute SA (Table 5). Two of its imports are never used on the SA path (`wget` -- download only; `shapely` -- polygon IoU only);
when absent they are replaced by inert stubs so the project environment is not modified. No metric code is copied or changed.

`--published` evaluates the authors' released "TrainedCST" results (`data/cst_anti_uav_official/tracking_results/TrainedCST`,
mapped through the toolkit's own `utils/trackers.py` list). Every per-sequence SA is cross-checked against Track A's
implementation (`anti_uav_repro.eval.metrics.sequence_metrics`, the code that scores our trackers) on the same predictions.

Output (<out>/): official_scores.json (per tracker: mSA, success / precision, per-sequence SA, sequence-level attribute SA,
official frame-level attribute SA, corrected frame-level attribute SA, Track A cross-check, sha256 of every result file) and the
toolkit's own report files under <out>/toolkit_reports/.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TOOLKIT = REPO / "data" / "cst_anti_uav_official" / "toolkit"
PUBLISHED_ROOT = REPO / "data" / "cst_anti_uav_official" / "tracking_results" / "TrainedCST"
DATA_ROOT = REPO / "data" / "cst_anti_uav_full" / "CST-AntiUAV" / "CST-AntiUAV"
TOOLKIT_PREFIX = "D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/"   # absolute paths in the toolkit's utils/trackers.py
SEQ_ATTRS = ("OC", "OV", "SV", "TC", "DBC", "CDB", "FM", "TS", "SS", "MS", "NS")  # toolkit att_fig_name order (annos/<subset>/att)
FRAME_KEYS = ("OC", "OV", "SV", "TC", "DBC", "CDB", "FM", "TS", "SS", "MS", "NS")
SIZE_BINS = (("tiny", 0, 10), ("small", 10, 30), ("normal", 30, 50), ("large", 50, np.inf))   # Table 5 TS / SS / NS / LS

sys.path.insert(0, str(REPO / "experiments" / "anti_uav" / "src"))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def import_toolkit():
    for mod in ("wget", "shapely", "shapely.geometry"):
        try:
            __import__(mod)
        except ImportError:
            m = types.ModuleType(mod)
            if mod == "shapely.geometry":
                def _unused(*a, **k):
                    raise RuntimeError("shapely stub: polygon IoU is not used by the CST SA evaluation")
                m.box = m.Polygon = _unused
            sys.modules[mod] = m
    sys.path.insert(0, str(TOOLKIT))
    import experiments as E  # the toolkit's package (not this repo's experiments/ directory: TOOLKIT is first on sys.path)
    if not str(Path(E.__file__).resolve()).startswith(str(TOOLKIT.resolve())):
        raise SystemExit(f"imported the wrong 'experiments' package: {E.__file__}")
    # I/O robustness only: the frame-level script re-opens the same 11 label files once per sequence and tracker (~14k cycles);
    # on Windows an indexer / antivirus lock occasionally makes one open() fail with EINVAL / EACCES. Retry; metric code untouched.
    import builtins
    import importlib

    def _open(file, mode="r", *args, _o=builtins.open, **kw):
        for i in range(20):
            try:
                return _o(file, mode, *args, **kw)
            except OSError as e:
                if e.errno not in (13, 22) or i == 19:
                    raise
                time.sleep(0.05 * (i + 1))
    for m in ("experiments.cstanti_uav", "experiments.cstanti_uav_frame_SA"):
        importlib.import_module(m).open = _open
    return E


def published_trackers():
    tree = ast.parse((TOOLKIT / "utils" / "trackers.py").read_text(encoding="utf-8"))
    lst = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "CST_Trackers")
    out, missing = [], []
    for t in lst:
        p = PUBLISHED_ROOT / t["path"][len(TOOLKIT_PREFIX):]
        (out if p.is_dir() else missing).append({"name": t["name"], "path": str(p), "mode": t["mode"]})
    return out, [m["name"] for m in missing]


def track_a_sa(record_file, label):
    """Track A's SA on the same prediction file (cross-check of the code that scores our trackers)."""
    from anti_uav_repro.eval.metrics import sequence_metrics
    try:
        res = json.loads(Path(record_file).read_text(encoding="utf-8"))["res"]
    except Exception:
        arr = np.loadtxt(io.StringIO(Path(record_file).read_text(encoding="utf-8").replace(",", " ")))
        res = [list(map(float, r)) for r in np.atleast_2d(arr)]
    res = res[:len(label["exist"])] + [[]] * max(0, len(label["exist"]) - len(res))
    return sequence_metrics(res, label["exist"], label["gt"])["SA"]


def size_mask(label, name):
    """Paper Table 5 size bins by ground-truth box diagonal; identical to the release's SIZE codes 1-4 (verified on TEST: code 1
    spans 3.3-10.0 px, 2: 10.0-30.0, 3: 30.0-50.0, 4: 50.0-84.8; one TEST sequence carries TS/SS/MS/NS keys instead of SIZE)."""
    lo, hi = next((lo, hi) for n, lo, hi in SIZE_BINS if n == name)
    d = np.array([np.hypot(g[2], g[3]) if e and len(g) == 4 else np.nan for e, g in zip(label["exist"], label["gt"])])
    return ((d > lo) & (d <= hi)).astype(float)


def corrected_frame_attr(per_frame, labels):
    """Frame-level SA per attribute: mean of the Eq. 1 frame scores over all frames carrying the flag (every event counted;
    the toolkit's OC / OV branches keep only the last event's window -- see PAPER_PROTOCOL.md)."""
    out = {}
    for k in FRAME_KEYS[:7] + tuple(n for n, _, _ in SIZE_BINS):
        num = den = 0.0
        for seq, sc in per_frame.items():
            v = labels[seq].get(k) if k in FRAME_KEYS else size_mask(labels[seq], k)
            if v is None or len(v) != len(sc):
                continue
            v = np.asarray(v, dtype=float)
            num += float((np.asarray(sc) * v).sum())
            den += float(v.sum())
        out[k] = {"SA": 100 * num / den if den else None, "n_frames": int(den)}
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", action="store_true", help="the authors' released TrainedCST results")
    ap.add_argument("--tracker", action="append", default=[], help="NAME=DIR (DIR holds <seq>.txt; mode 1 = x,y,w,h)")
    ap.add_argument("--subset", default="test", choices=("train", "val", "test"))
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if (out / "official_scores.json").exists():
        raise SystemExit(f"{out} already holds a result; choose a new output directory")
    trackers, missing = published_trackers() if a.published else ([], [])
    for t in a.tracker:
        n, p = t.split("=", 1)
        trackers.append({"name": n, "path": str(Path(p).resolve()), "mode": 1})
    if not trackers:
        raise SystemExit("nothing to evaluate")
    E = import_toolkit()
    rep = out / "toolkit_reports"
    rep.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(TOOLKIT)                                            # the toolkit reads annos/<subset>/att/*.txt relative to the cwd
    try:
        t0 = time.time()
        exp = E.ExperimentCSTAntiUAV(root_dir=a.data_root, subset=a.subset, report_dir=str(rep))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            perf = exp.report(trackers, plot_curves=False, plot_attcurves=False)
        expf = E.ExperimentCSTAntiUAVF(root_dir=a.data_root, subset=a.subset, report_dir=str(rep / "frame"))
        with contextlib.redirect_stdout(buf):
            expf.report(trackers)
        names = exp.dataset.seq_names
        labels = {}
        for s in names:
            labels[s] = json.loads((Path(a.data_root) / a.subset / s / "IR_label.json").read_text(encoding="utf-8"))
        result = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "subset": a.subset, "n_sequences": len(names),
                  "toolkit": {"path": str(TOOLKIT.relative_to(REPO)),
                              "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=TOOLKIT, capture_output=True,
                                                       text=True).stdout.strip(),
                              "experiment_sha256": sha256(TOOLKIT / "experiments" / "cstanti_uav.py"),
                              "frame_experiment_sha256": sha256(TOOLKIT / "experiments" / "cstanti_uav_frame_SA.py")},
                  "published_missing_from_archive": missing, "trackers": {}}
        if a.published:
            result["published_archive_sha256"] = sha256(PUBLISHED_ROOT.parent / "TrainedCST.zip")
        for t in trackers:
            n = t["name"]
            p = perf[n]
            seq_sa = p["seq_wise"][names[-1]]["sa"]            # the toolkit stores the running list; the last one holds all
            assert len(seq_sa) == len(names)
            per_seq = {s: float(v) for s, v in zip(names, seq_sa)}
            files = {s: Path(t["path"]) / f"{s}.txt" for s in names}
            xa = {s: track_a_sa(files[s], labels[s]) for s in names}
            dmax = max(abs(per_seq[s] - xa[s]) for s in names)
            per_frame = {s: expf.eval(_load(files[s]), labels[s]) for s in names}   # the toolkit's own per-frame Eq. 1
            off = {}
            ff = Path(expf.report_dir) / f"{n}_frame_sa.txt"   # the toolkit writes to <report_dir>/CSTAntiUAV/<subset>/
            if ff.exists():
                for line in ff.read_text(encoding="utf-8").splitlines():
                    k, v = line.split(":")
                    off[k.strip()] = 100 * float(v)
            result["trackers"][n] = {
                "path": str(Path(t["path"]).relative_to(REPO)) if str(t["path"]).startswith(str(REPO)) else t["path"],
                "mSA": 100 * float(np.mean(seq_sa)),
                "success_auc": 100 * p["overall"]["success_score"], "precision_20px": 100 * p["overall"]["precision_score"],
                "success_rate_iou_0_5": 100 * p["overall"]["success_rate"],
                "sequence_attr_SA": {k: 100 * p[k]["avg_att_sa"] for k in SEQ_ATTRS},
                "frame_attr_SA_official": off,
                "frame_attr_SA_corrected": corrected_frame_attr(per_frame, labels),
                "per_sequence_SA": {s: 100 * v for s, v in per_seq.items()},
                "track_a_crosscheck": {"mSA": 100 * float(np.mean(list(xa.values()))), "max_abs_seq_diff": float(dmax)},
                "result_sha256": {s: sha256(f) for s, f in files.items()},
            }
        result["seconds"] = round(time.time() - t0, 1)
    finally:
        os.chdir(cwd)
    (out / "official_scores.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / "toolkit_stdout.txt").write_text(buf.getvalue(), encoding="utf-8")
    rows = sorted(result["trackers"].items(), key=lambda kv: -kv[1]["mSA"])
    print(f"{'tracker':16s} {'mSA':>7s} {'S-AUC':>7s} {'P@20':>7s} {'TS(off)':>8s} {'TS(cor)':>8s} {'TrackA':>7s} {'max|d|':>8s}")
    for n, r in rows:
        print(f"{n:16s} {r['mSA']:7.2f} {r['success_auc']:7.2f} {r['precision_20px']:7.2f} "
              f"{r['frame_attr_SA_official'].get('TS', float('nan')):8.2f} {r['frame_attr_SA_corrected']['tiny']['SA']:8.2f} "
              f"{r['track_a_crosscheck']['mSA']:7.2f} {r['track_a_crosscheck']['max_abs_seq_diff']:8.1e}")


def _load(record_file):
    """Prediction loader identical to the toolkit's report() (json 'res' first, else comma/space text)."""
    try:
        with open(record_file, "r") as f:
            return json.load(f)["res"]
    except Exception:
        with open(record_file, "r") as f:
            return np.loadtxt(io.StringIO(f.read().replace(",", " ")))


if __name__ == "__main__":
    main()
