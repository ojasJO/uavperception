#!/usr/bin/env python
r"""ONE command for the rest of the AeroTrack repair: the controlled ablation grid, its evaluations, the final
test table and every generated report -- sequential, one GPU, unattended, resumable.

    venv\Scripts\python.exe experiments\aerotrack_repair\scripts\run_grid.py              (foreground)
    venv\Scripts\python.exe experiments\aerotrack_repair\scripts\run_grid.py --detach     (background, survives the terminal)
    venv\Scripts\python.exe experiments\aerotrack_repair\scripts\run_grid.py --status     (progress of a running grid)
    venv\Scripts\python.exe experiments\aerotrack_repair\scripts\run_grid.py --stop       (stop it; re-run to resume)
    venv\Scripts\python.exe experiments\aerotrack_repair\scripts\run_grid.py --preflight-only

What it runs, in this order, and nothing else (plan: docs/AEROTRACK_REPAIR_PLAN.md items 2-4; rule: render_report.py):
  0. pre-flight: venv, packages, CUDA/GPU/VRAM, no other GPU job, data + registered val-set hashes, checkpoints, disk,
     RAM, scripts, an evaluation smoke test (VAL, nothing written), a training-DataLoader memory probe (a worker must not
     grow per batch) and train.py's own T1-T8 pre-flight on each of the four exact jobs (scratch names, nothing trained). Any failure stops here, before the first long run.
  1. for baseline_ctrl, no_spd, single_frame, stock_yolo11n -- strictly one after the other:
       train_ablation.py --ablation <x> --epochs 20 --batch 32   (the pre-registered recipe and budget)
       -> verify 20 epochs + loadable best.pt/last.pt
       -> evaluate_detector.py --split val --cap 5000 --weights best.pt last.pt   (the fixed, hash-identified val set)
       -> render_report.py
     A run that dies is resumed from weights/last.pt (atomic, so always a complete epoch); host-memory failures halve
     the DataLoader workers, VRAM contention waits, a stalled log (no output for --stall-minutes) is killed and resumed.
     Deterministic failures (pre-flight, config, code errors) stop the grid: nothing is skipped or substituted.
  2. only when all four runs are finished and evaluated: the TEST split, once per checkpoint, for every checkpoint
     of the validation table (v1 and v2 best/last + the grid's 8): evaluate_detector.py --split test --final-eval
     --cap 5000. reports/test_evaluation_ledger.jsonl is the commit record; a checkpoint already in it is never
     scored again, and a re-run only scores checkpoints that have no ledger line.
  3. render_report.py (README.md, docs/AEROTRACK_REPAIR_REPORT.md) + grid_report.py (reports/ablation_grid_e20.{json,md}).

Re-running the same command resumes: every step is checked against the files it produces (hashes), not against
memory. State, event log, pre-flight record, code snapshot and every step's log: runs/repair_grid_e<epochs>/.
The parent process imports no torch (it stays at a few tens of MB); all GPU work happens in child processes.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
REPORTS_DIR = REPO_ROOT / "experiments" / "aerotrack_repair" / "reports"
VENV_PY = REPO_ROOT / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

GRID = ("baseline_ctrl", "no_spd", "single_frame", "stock_yolo11n")
SINGLE_FRAME = {"single_frame", "stock_yolo11n"}          # temporal_gap 0 in c_model/train.py::ABLATIONS
HISTORICAL = [("aerotrack_spd_v1", "best"), ("aerotrack_spd_v1", "last"),
              ("aerotrack_spd_finetuned", "best"), ("aerotrack_spd_finetuned", "last")]
VAL_CAP = TEST_CAP = 5000                                 # the registered protocol: reports/detector_val_cap5000.*
REGISTERED_VAL_SET = "6fb41d44d45b"                       # sha256 prefix of that set (README / repair report)
TRAIN_ABLATION = HERE / "train_ablation.py"
EVALUATE = HERE / "evaluate_detector.py"
RENDER = HERE / "render_report.py"
GRID_REPORT = HERE / "grid_report.py"
DATA = REPO_ROOT / "b_pipeline" / "data_unified"
LEDGER = REPORTS_DIR / "test_evaluation_ledger.jsonl"

MEM_KEYS = ("LoaderWorkerDied", "exited unexpectedly", "Unable to allocate", "ArrayMemoryError", "paging file",
            "not enough memory", "MemoryError", "DefaultCPUAllocator")
VRAM_KEYS = ("OutOfMemoryError", "CUDA out of memory", "consecutive CUDA OOMs")
CUDA_KEYS = ("CUDA error", "cudaError", "CUBLAS_STATUS", "CUDNN_STATUS", "cuDNN error", "illegal memory access",
             "unspecified launch failure", "device-side assert", "driver shutting down")
FATAL_EXC = ("ImportError", "ModuleNotFoundError", "NameError", "SyntaxError", "AttributeError", "TypeError", "KeyError",
             "IndexError", "ValueError", "FileNotFoundError", "AssertionError", "NotImplementedError", "ZeroDivisionError",
             "JSONDecodeError", "UnboundLocalError", "PermissionError", "IsADirectoryError")
FATAL_KINDS = ("preflight", "config", "software")
TRAIN_PY_RX = r"c_model[\\/]train\.py"   # the process train_ablation.py launches

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CHILD_FLAGS = CREATE_NO_WINDOW if os.name == "nt" else 0


class GridFatal(RuntimeError):
    """A failure that retrying cannot fix: the grid stops and says why."""


class GridStopped(RuntimeError):
    """--stop, or Ctrl+C."""


# ============================================================================================ small utilities
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def jload(p):
    p = Path(p)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except json.JSONDecodeError:
        return None


def write_json_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def child_env():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def rel(p):
    """Repository-relative path for messages (absolute when outside the repository)."""
    p = Path(p)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def hours(s):
    return f"{s / 3600:.2f} h"


class Grid:
    """Paths + logging for one budget (runs/repair_grid_e<epochs>)."""

    def __init__(self, epochs):
        self.epochs = epochs
        self.dir = REPO_ROOT / "runs" / f"repair_grid_e{epochs}"
        self.logs = self.dir / "logs"
        self.state_path = self.dir / "grid_state.json"
        self.events_path = self.dir / "events.jsonl"
        self.driver_log = self.logs / "grid_driver.log"
        self.lock = self.dir / "grid.lock"
        self.stop_file = self.dir / "STOP"

    def mkdirs(self):
        self.logs.mkdir(parents=True, exist_ok=True)

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            with open(self.driver_log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def event(self, _event, **kw):
        rec = {"t": now(), "event": _event, **kw}
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        brief = ", ".join(f"{k}={v}" for k, v in kw.items() if k not in ("detail", "checks", "weights", "tail"))
        self.log(f"[event] {_event}" + (f": {brief}" if brief else ""))

    def run_dir(self, abl):
        return REPO_ROOT / "runs" / f"repair_abl_{abl}_e{self.epochs}"

    def state(self):
        return jload(self.state_path) or {}

    def update_state(self, **kw):
        st = self.state()
        st.update(kw)
        st["epochs"] = self.epochs
        st["updated"] = now()
        write_json_atomic(st, self.state_path)


# ============================================================================================ processes
def _psutil():
    import psutil
    return psutil


def pid_alive(pid, needle=None):
    ps = _psutil()
    try:
        p = ps.Process(int(pid))
        if not p.is_running() or p.status() == ps.STATUS_ZOMBIE:
            return False
        return needle is None or any(needle in str(a) for a in (p.cmdline() or []))
    except Exception:  # noqa: BLE001
        return False


def own_tree_pids():
    """This process, its descendants and its ancestors (on Windows the venv python.exe that was typed is a launcher
    whose child is the real interpreter, so our own command line also appears in the PARENT)."""
    ps = _psutil()
    me = ps.Process()
    pids = {me.pid} | {c.pid for c in me.children(recursive=True)}
    try:
        pids |= {p.pid for p in me.parents()}
    except (ps.NoSuchProcess, ps.AccessDenied):
        pass
    return pids


def kill_tree(pid, grid, grace_s=90, inner=None):
    """Stop the process tree under `pid`.

    With `inner` (a regex on the command line, e.g. c_model/train.py) the matching processes and their descendants
    (the DataLoader workers) go first, so the launcher above them (train_ablation.py) can record the attempt's end;
    it gets `grace_s` to exit by itself. Everything still alive after that is killed. (On Windows a venv python.exe
    is a launcher whose child is the real interpreter, so "all descendants" would include train_ablation.py itself.)"""
    ps = _psutil()
    try:
        root = ps.Process(pid)
    except ps.NoSuchProcess:
        return

    def _kill(procs):
        for c in procs:
            try:
                c.kill()
            except ps.NoSuchProcess:
                pass
        ps.wait_procs(procs, timeout=30)

    kids = root.children(recursive=True)
    if inner:
        rx = re.compile(inner)
        targets = {}
        for k in kids:
            try:
                if rx.search(" ".join(k.cmdline() or [])):
                    for t in [k] + k.children(recursive=True):
                        targets[t.pid] = t
            except (ps.NoSuchProcess, ps.AccessDenied):
                continue
        if targets:
            _kill(list(targets.values()))
            try:
                root.wait(timeout=grace_s)
                return
            except ps.TimeoutExpired:
                grid.log(f"[kill] launcher pid {pid} did not exit within {grace_s}s after its child was stopped")
    try:
        kids = root.children(recursive=True)
    except ps.NoSuchProcess:
        return
    _kill(list(reversed(kids)))
    try:
        root.wait(timeout=30 if inner else grace_s)
    except ps.TimeoutExpired:
        grid.log(f"[kill] pid {pid} did not exit; killing it")
        try:
            root.kill()
            root.wait(timeout=30)
        except ps.NoSuchProcess:
            pass


def run_logged(grid, cmd, log_path, stall_minutes, what, inner=None):
    """Run `cmd` with stdout+stderr in `log_path`. Watchdog: a log that stops growing for `stall_minutes` means a hung
    process (a dead-locked loader never raises) -> kill the tree. Honours the STOP file. -> (rc, stalled)"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as f:
        f.write(f"### {now()} {what}\n### cmd: {subprocess.list2cmdline([str(c) for c in cmd])}\n".encode("utf-8"))
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=child_env(), creationflags=CHILD_FLAGS)
    grid.log(f"[run] {what}: pid {p.pid}, log {rel(log_path)}")
    grid.update_state(current={"what": what, "pid": p.pid, "log": str(log_path), "started": now()})
    last_size, last_change = -1, time.time()
    stalled = False
    try:
        while True:
            try:
                rc = p.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                pass
            if grid.stop_file.exists():
                grid.log(f"[stop] STOP requested: stopping {what}")
                kill_tree(p.pid, grid, inner=inner)
                raise GridStopped(f"stopped during {what}")
            size = log_path.stat().st_size
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > stall_minutes * 60:
                grid.log(f"[watchdog] no output from {what} for {stall_minutes} min: killing the process tree")
                kill_tree(p.pid, grid, inner=inner)
                stalled = True
                rc = p.wait(timeout=120)
                break
    except KeyboardInterrupt:
        grid.log(f"[stop] Ctrl+C: stopping {what}")
        kill_tree(p.pid, grid, grace_s=30, inner=inner)
        raise GridStopped(f"interrupted during {what}")
    grid.update_state(current=None)
    return rc, stalled


def log_tail(path, n_bytes=60000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def classify(text, rc, stalled):
    """-> (kind, detail). Kinds in FATAL_KINDS stop the grid; the others are retried (bounded)."""
    if stalled:
        return "stall", "log stopped growing"
    m = re.search(r"PRE-FLIGHT FAILED: (.*)", text)
    if m:
        failed = [x.strip() for x in m.group(1).split(",")]
        if all(x.startswith("T8") for x in failed):
            return "vram", f"pre-flight VRAM headroom: {m.group(1)}"
        return "preflight", f"pre-flight failed: {m.group(1)}"
    if "refusing to resume" in text or "[ABORT]" in text or "already exists — use --resume" in text:
        line = next((ln for ln in text.splitlines() if "[ABORT]" in ln or "refusing" in ln or "already exists" in ln), "")
        return "config", line.strip()[:300]
    # VRAM before host memory: "OutOfMemoryError" contains "MemoryError"
    if any(k in text for k in VRAM_KEYS):
        return "vram", next((k for k in VRAM_KEYS if k in text), "")
    if any(k in text for k in MEM_KEYS):
        return "host_memory", next((k for k in MEM_KEYS if k in text), "")
    if any(k in text for k in CUDA_KEYS):
        return "cuda", next((k for k in CUDA_KEYS if k in text), "")
    if "Traceback (most recent call last)" in text:
        exc_lines = [ln for ln in text.splitlines() if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt)\b", ln)]
        last = exc_lines[-1] if exc_lines else ""
        name = last.split(":")[0].split(".")[-1]
        if name in FATAL_EXC:
            return "software", last[:300]
        return "exception", last[:300] or "traceback"
    return "killed", f"exit code {rc} without a Python traceback (external kill, crash or power event)"


# ============================================================================================ run inspection
def inspect_run(grid, abl):
    """Load best.pt / last.pt in a CHILD process (the parent never imports torch) -> dict."""
    out = subprocess.run([str(VENV_PY), str(Path(__file__).resolve()), "--_inspect", str(grid.run_dir(abl))],
                         cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900, env=child_env(),
                         creationflags=CHILD_FLAGS)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise GridFatal(f"could not inspect {grid.run_dir(abl).name}: {out.stderr[-2000:]}")


def _inspect_child(run_dir):
    run_dir = Path(run_dir)
    sys.path.insert(0, str(REPO_ROOT / "c_model"))
    sys.path.insert(0, str(REPO_ROOT / "b_pipeline"))
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from aerotrack_trainer import register_spdconv
    register_spdconv()
    res = {"exists": run_dir.exists(), "dir": str(run_dir)}
    for tag in ("last", "best"):
        p = run_dir / "weights" / f"{tag}.pt"
        if not p.exists():
            res[tag] = {"exists": False, "ok": False}
            continue
        try:
            ck = torch.load(str(p), map_location="cpu", weights_only=False)
            ok = all(k in ck for k in ("epoch", "model", "optimizer")) and ck["model"] is not None
            res[tag] = {"exists": True, "ok": bool(ok), "epoch": int(ck.get("epoch", -1)),
                        "best_fitness": float(ck.get("best_fitness", -1.0)), "bytes": p.stat().st_size}
        except Exception as e:  # noqa: BLE001
            res[tag] = {"exists": True, "ok": False, "error": repr(e)[:400]}
    rows = 0
    if (run_dir / "results.csv").exists():
        with open(run_dir / "results.csv", encoding="utf-8", newline="") as f:
            rows = sum(1 for _ in csv.DictReader(f))
    res["results_rows"] = rows
    meta = jload(run_dir / "run_meta.json") or {}
    res["meta"] = {k: meta.get(k) for k in ("return_code", "timestamp_end", "epochs", "train_seconds", "batch")}
    res["meta"]["launches"] = len(meta.get("attempts") or [])
    print(json.dumps(res))


def _loader_probe_child(workers=2, batches=40):
    """The TRAINING loader configuration (augmentation, pinned memory, persistent workers, prefetch 2, the project's
    collate) on 4,000 train samples: does a worker's private memory grow per batch? It did once -- by one whole batch
    per batch -- and the OS killed a worker minutes into the first run. Prints one JSON line."""
    sys.path.insert(0, str(REPO_ROOT / "c_model"))
    sys.path.insert(0, str(REPO_ROOT / "b_pipeline"))
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import psutil
    import torch
    from torch.utils.data import DataLoader, RandomSampler
    from stacked_dataset import AeroTrackDataset, TemporalAugment, aerotrack_collate, build_split_samples, load_label_cache
    from train import LABEL_CACHE_PATH, MANIFEST, SPLITS, _worker_init
    s = build_split_samples(MANIFEST, SPLITS, "train")
    pick = [s[i] for i in np.random.default_rng(0).choice(len(s), 4000, replace=False)]
    del s
    ds = AeroTrackDataset(pick, img_size=(640, 640), augment=TemporalAugment(seed=0), letterbox=True, out_dtype="uint8",
                          label_cache=load_label_cache(LABEL_CACHE_PATH))
    dl = DataLoader(ds, batch_size=32, sampler=RandomSampler(ds, replacement=True, num_samples=32 * batches), drop_last=True,
                    num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=True, prefetch_factor=2,
                    collate_fn=aerotrack_collate, worker_init_fn=_worker_init)
    me = psutil.Process()

    def wmem():
        return [k.memory_info().private / 2 ** 30 for k in me.children(recursive=True) if "spawn_main" in " ".join(k.cmdline())]
    first = last = None
    t0 = time.time()
    for i, _ in enumerate(dl):
        if i + 1 == 10:
            first = wmem()
    last = wmem()
    growth = max((b - a for a, b in zip(first, last)), default=0.0) if first else None
    print(json.dumps({"workers": workers, "batches": batches, "worker_private_gb_at_10": first, "worker_private_gb_at_end": last,
                      "max_growth_gb": growth, "img_per_s": round(32 * batches / (time.time() - t0), 1),
                      "ok": growth is not None and growth < 0.5}))


def run_complete(info, epochs):
    last, best, meta = info.get("last", {}), info.get("best", {}), info.get("meta", {})
    return (last.get("ok") and last.get("epoch") == epochs - 1 and best.get("ok") and info.get("results_rows", 0) >= epochs
            and meta.get("return_code") == 0 and bool(meta.get("timestamp_end")))


# ============================================================================================ evaluation bookkeeping
def summary_rows(split, cap):
    j = jload(REPORTS_DIR / f"detector_{split}_cap{cap}.json") or {}
    return j.get("rows", []), j.get("eval_set", {})


def expected_input(run_name):
    return "t,t,t" if any(run_name.startswith(f"repair_abl_{a}_e") for a in SINGLE_FRAME) else "t-1,t,t+1"


def row_for(rows, run, ck, sha12):
    return next((r for r in rows if r["run"] == run and r["checkpoint"] == ck and r["weights_sha256"] == sha12), None)


def val_row_ok(run, ck, w):
    rows, s = summary_rows("val", VAL_CAP)
    r = row_for(rows, run, ck, sha256(w)[:12])
    return bool(r and r["set_sha256"] == REGISTERED_VAL_SET and (s.get("sha256") or "").startswith(REGISTERED_VAL_SET)
                and r.get("input", "t-1,t,t+1") == expected_input(run))


def read_ledger():
    if not LEDGER.exists():
        return []
    out = []
    for ln in LEDGER.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def weights_path(run, ck):
    return REPO_ROOT / "runs" / run / "weights" / f"{ck}.pt"


# ============================================================================================ pre-flight
SOURCES = ["c_model/train.py", "c_model/aerotrack_trainer.py", "c_model/preflight.py", "c_model/evaluation.py",
           "c_model/custom_modules.py", "c_model/yolo11-spd.yaml", "c_model/yolo11-stride.yaml",
           "b_pipeline/stacked_dataset.py", "b_pipeline/imbalance_sampler.py",
           "experiments/aerotrack_repair/scripts/train_ablation.py", "experiments/aerotrack_repair/scripts/evaluate_detector.py",
           "experiments/aerotrack_repair/scripts/render_report.py", "experiments/aerotrack_repair/scripts/grid_report.py",
           "experiments/aerotrack_repair/scripts/run_grid.py", "experiments/aerotrack_repair/scripts/_bootstrap.py",
           "experiments/aerotrack_repair/README.template.md"]


def source_hashes():
    return {s: (sha256(REPO_ROOT / s) if (REPO_ROOT / s).exists() else None) for s in SOURCES}


def free_commit_gb():
    if os.name != "nt":
        return -1.0
    import ctypes

    class _MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = _MS()
    m.dwLength = ctypes.sizeof(_MS)
    return m.ullAvailPageFile / 1024 ** 3 if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) else -1.0


def foreign_gpu_jobs():
    """Python ML processes outside this tree, and python processes the driver lists as GPU clients."""
    ps = _psutil()
    mine = own_tree_pids()
    pat = re.compile(r"(c_model[\\/]train\.py|train_ablation\.py|evaluate_detector\.py|run_grid\.py|anti_uav.*[\\/](train|evaluate)\.py|"
                     r"audit[\\/]bench_|tracker_repair\.py|latency_repair\.py|lstm_repair\.py|make_demo\.py|val\.py|track_eval\.py)")
    found, kernels = [], []
    for p in ps.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in mine or "python" not in (p.info["name"] or "").lower():
                continue
            cl = " ".join(p.info["cmdline"] or [])
            if re.search(r"run_grid\.py.*(--status|--stop|--_inspect|--detach)", cl):
                continue                                   # read-only helpers, never GPU jobs
            if pat.search(cl):
                found.append({"pid": p.info["pid"], "cmdline": cl[:300]})
            elif "ipykernel" in cl:
                kernels.append({"pid": p.info["pid"], "cmdline": cl[:200]})
        except (ps.NoSuchProcess, ps.AccessDenied):
            continue
    gpu_py = []
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout
        for ln in q.splitlines():
            parts = [x.strip() for x in ln.split(",", 1)]
            if len(parts) == 2 and "python" in parts[1].lower() and parts[0].isdigit() and int(parts[0]) not in mine:
                gpu_py.append({"pid": int(parts[0]), "process": parts[1]})
    except Exception:  # noqa: BLE001
        pass
    return found, gpu_py, kernels


def parse_train_preflight(text):
    def g(rx, conv=str, grp=1, flags=0):
        m = re.search(rx, text, flags)
        return conv(m.group(grp).replace(",", "")) if m else None
    tr = None
    m = re.search(r"\[transfer\] ([\d,]+)/([\d,]+) tensors copied from \S+ \(([\d.]+)%\)", text)
    if m:
        tiled = re.search(r"stem tiled \(/3\): \[(.*?)\]", text)
        tr = {"copied": int(m.group(1).replace(",", "")), "total": int(m.group(2).replace(",", "")), "pct": float(m.group(3)),
              "tiled_stem": [t.strip(" '\"") for t in tiled.group(1).split(",")] if tiled else []}
    workers = re.findall(r"\[plan\]\s+using (\d+) workers", text)
    return {
        "passed": "[preflight-only] every check passed" in text and "PRE-FLIGHT FAILED" not in text,
        "transfer": tr,
        "model_cfg": g(r"\[model\] (\S+) \|"), "params_m": g(r"\| ([\d.]+) M params", float),
        "n_spdconv": g(r"\| (\d+) SPDConv layers", int),
        "batch": g(r"\[engine\] batch (\d+) x", int), "accumulate": g(r"x accumulate (\d+) =", int),
        "iters_per_epoch": g(r"\| ([\d,]+) iters/epoch", int), "precision": g(r"iters/epoch \| (\w+)"),
        "train_workers_planned": int(workers[0]) if workers else None,
        "train_samples": g(r"\[corpus\] train ([\d,]+) samples", int), "val_samples": g(r"\[corpus\] val\s+([\d,]+) samples", int),
        "t6_first_conv_in": g(r"first conv in_channels=(\d+)", int),
        "loader_samples_per_s": g(r"([\d,.]+) samples/s \(", float),
        "loader_epoch_min_estimate": g(r"would take ([\d.]+) min", float),
        "vram_peak_gb": g(r"peak ([\d.]+) GB at batch", float), "vram_headroom_gb": g(r"-> ([\d.]+) GB headroom", float),
        "checks": re.findall(r"\[(PASS|FAIL|EXEMPT)\] (\S+)", text),
    }


def preflight(grid, a):
    checks = []

    def add(name, ok, detail, fatal=True):
        checks.append({"name": name, "ok": bool(ok), "fatal": fatal, "detail": detail})
        grid.log(f"[preflight] {'PASS' if ok else ('FAIL' if fatal else 'WARN')} {name}: {detail}")

    rec = {"timestamp": now(), "epochs": a.epochs, "batch": a.batch, "workers": a.workers, "checks": checks}
    ps = _psutil()

    # P1 interpreter
    add("python/venv", Path(sys.executable).resolve() == VENV_PY.resolve(), f"{sys.executable} (Python {sys.version.split()[0]})")

    # P2+P3 packages, CUDA, GPU (child: the parent never imports torch)
    code = (
        "import json,importlib\n"
        "r={'versions':{}}\n"
        "for m in ('torch','torchvision','ultralytics','numpy','cv2','psutil','matplotlib','pandas','yaml','PIL'):\n"
        "    try: r['versions'][m]=getattr(importlib.import_module(m),'__version__','?')\n"
        "    except Exception as e: r['versions'][m]=None; r.setdefault('missing',[]).append(f'{m}: {e!r}')\n"
        "import torch\n"
        "r['cuda']=torch.cuda.is_available()\n"
        "if r['cuda']:\n"
        "    c=torch.cuda.get_device_capability(0); r['name']=torch.cuda.get_device_name(0); r['sm']=f'sm_{c[0]}{c[1]}'\n"
        "    r['arch_ok']=r['sm'] in torch.cuda.get_arch_list(); r['bf16']=torch.cuda.is_bf16_supported(); r['cuda_version']=torch.version.cuda\n"
        "    x=torch.randn(512,512,device='cuda'); r['kernel_ok']=bool((x@x).isfinite().all()); torch.cuda.synchronize()\n"
        "    f,t=torch.cuda.mem_get_info(); r['free_gb']=round(f/1024**3,2); r['total_gb']=round(t/1024**3,2)\n"
        "print(json.dumps(r))\n")
    out = subprocess.run([str(VENV_PY), "-c", code], capture_output=True, text=True, timeout=600, env=child_env(), creationflags=CHILD_FLAGS)
    try:
        env = json.loads(out.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        env = {"error": out.stderr[-1500:]}
    rec["environment"] = env
    add("packages", not env.get("missing") and "error" not in env,
        ", ".join(f"{k} {v}" for k, v in (env.get("versions") or {}).items()) or env.get("error", "?"))
    add("CUDA", env.get("cuda") and env.get("arch_ok") and env.get("kernel_ok") and env.get("bf16"),
        f"{env.get('name')} {env.get('sm')} (in wheel: {env.get('arch_ok')}), CUDA {env.get('cuda_version')}, kernel launch ok: "
        f"{env.get('kernel_ok')}, bf16: {env.get('bf16')}")
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,memory.used,memory.total,temperature.gpu,power.draw,pstate",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        drv = ""
    rec["nvidia_smi"] = drv
    # batch-32 peak of this recipe, measured with a pessimistic synthetic batch (8 boxes/img): 4.76 GB. Below 5 GB free
    # nothing fits; below ~6.3 GB T8 (real batches, >= 1.5 GB headroom) is likely to refuse -- T8 is the authority.
    free = env.get("free_gb") or 0
    add("VRAM", free >= 5.0, f"{free} GB free of {env.get('total_gb')} GB (batch-32 peak ~4.8 GB; T8 wants 1.5 GB headroom on "
        f"real batches); nvidia-smi: {drv}")
    if 5.0 <= free < 6.3:
        add("VRAM headroom", False, f"only {free} GB free: close GPU-heavy apps, or T8 may refuse the launch", fatal=False)

    # P4 nothing else on the GPU
    found, gpu_py, kernels = foreign_gpu_jobs()
    add("no other GPU job", not found and not gpu_py, "none" if not (found or gpu_py) else f"ML processes: {found}; python GPU clients: {gpu_py}")
    if kernels:
        add("jupyter kernels", True, f"{len(kernels)} running (not GPU clients now, but one could claim VRAM mid-run)", fatal=False)

    # P5 data = the data of the registered val set
    val = jload(REPORTS_DIR / "detector_val_cap5000.json") or {}
    vs = val.get("eval_set", {})
    man, spl, lab = DATA / "manifest.csv", DATA / "splits.json", DATA / "label_cache.npz"
    ok5 = man.exists() and spl.exists() and lab.exists()
    msha = sha256(man) if man.exists() else None
    ssha = sha256(spl) if spl.exists() else None
    rec["data"] = {"manifest_csv_sha256": msha, "splits_json_sha256": ssha, "label_cache_sha256": sha256(lab) if lab.exists() else None}
    add("data files", ok5 and msha == vs.get("manifest_csv_sha256") and ssha == vs.get("splits_json_sha256")
        and (vs.get("sha256") or "").startswith(REGISTERED_VAL_SET) and vs.get("cap") == VAL_CAP,
        f"manifest {str(msha)[:12]} / splits {str(ssha)[:12]} match the registered val set {str(vs.get('sha256'))[:12]} "
        f"(cap {vs.get('cap')}, {vs.get('n_images')} images)")
    # image files for every split the workflow reads, without opening a single label: every 997th manifest row
    missing, checked, per_split = [], 0, {}
    if man.exists():
        with open(man, encoding="utf-8", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i % 997 == 0:
                    checked += 1
                    per_split[row["split"]] = per_split.get(row["split"], 0) + 1
                    if not Path(row["image_path"]).exists():
                        missing.append(row["image_path"])
    add("image files", checked > 0 and not missing, f"{checked - len(missing)}/{checked} sampled image paths exist {per_split}"
        + (f"; missing e.g. {missing[:3]}" if missing else ""))

    # P6 model definitions + COCO weights
    cfgs = [REPO_ROOT / "c_model" / n for n in ("yolo11-spd.yaml", "yolo11-stride.yaml", "yolo11n.pt")]
    add("model files", all(p.exists() for p in cfgs), ", ".join(f"{p.name}" + (f" {sha256(p)[:12]}" if p.suffix == ".pt" and p.exists() else "")
                                                                  for p in cfgs))

    # P7 the historical checkpoints the final test table scores are the ones in the registered val table
    rows = val.get("rows", [])
    bad = []
    for run, ck in HISTORICAL:
        w = weights_path(run, ck)
        if not w.exists() or not row_for(rows, run, ck, sha256(w)[:12]):
            bad.append(f"{run}/{ck}")
    add("historical checkpoints", not bad, "v1 + v2 best/last present and hash-identical to their val rows" if not bad else f"mismatch: {bad}")

    # P8 disk
    du = shutil.disk_usage(REPO_ROOT)
    add("disk", du.free / 1024 ** 3 >= 20, f"{du.free / 1024 ** 3:.0f} GB free on {REPO_ROOT.anchor} (need >= 20; the grid writes < 2 GB)")
    try:
        grid.mkdirs()
        (grid.dir / ".write_test").write_text("ok", encoding="utf-8")
        (grid.dir / ".write_test").unlink()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        add("output dirs", True, f"{rel(grid.dir)} and {rel(REPORTS_DIR)} writable")
    except OSError as e:
        add("output dirs", False, repr(e))

    # P9 host memory: train.py plans its own worker count from these same two numbers
    vm = ps.virtual_memory()
    commit = free_commit_gb()
    reserve = float(os.environ.get("AEROTRACK_RESERVE_GB", 4.0))
    by_commit = int(max(commit - reserve - 2.5, 0) // 2.4) if commit >= 0 else a.workers
    by_phys = int(max(vm.available / 1024 ** 3 - reserve - 1.5, 0) // 1.3)
    est = max(0, min(a.workers, by_commit, by_phys))
    rec["host_memory"] = {"ram_total_gb": round(vm.total / 1024 ** 3, 1), "ram_available_gb": round(vm.available / 1024 ** 3, 1),
                          "commit_free_gb": round(commit, 1), "reserve_gb": reserve, "workers_estimate": est}
    add("host memory", est >= 1, f"{vm.available / 1024 ** 3:.1f} of {vm.total / 1024 ** 3:.1f} GB RAM free, {commit:.1f} GB commit free "
        f"-> train.py would plan {est} of {a.workers} loader workers now (it re-plans at every launch)")

    # P10 every script compiles and the report renders
    bad = []
    for s in SOURCES:
        if s.endswith(".py"):
            r = subprocess.run([str(VENV_PY), "-m", "py_compile", str(REPO_ROOT / s)], capture_output=True, text=True, env=child_env(),
                               creationflags=CHILD_FLAGS)
            if r.returncode:
                bad.append(f"{s}: {r.stderr.strip()[-300:]}")
    r = subprocess.run([str(VENV_PY), str(RENDER), "--check"], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600,
                       env=child_env(), creationflags=CHILD_FLAGS)
    if r.returncode:
        bad.append(f"render_report.py --check: {(r.stdout + r.stderr).strip()[-500:]}")
    add("scripts", not bad, "all compile; render_report.py --check OK" if not bad else "; ".join(bad))

    # P11 state of the grid and of the test ledger
    existing = {abl: grid.run_dir(abl).exists() for abl in GRID}
    led = read_ledger()
    rec["grid_on_disk"] = existing
    add("grid state", True, ", ".join(f"{k}: {'exists (will resume/verify)' if v else 'new'}" for k, v in existing.items())
        + f"; test ledger: {len(led)} entries", fatal=False)
    other = sorted(d.name for d in (REPO_ROOT / "runs").glob("repair_abl_*_e*") if not d.name.endswith(f"_e{a.epochs}"))
    if other:
        add("other budgets", True, f"runs at another budget exist and are left alone: {other}", fatal=False)

    # P12 power
    try:
        b = ps.sensors_battery()
        add("power", b is None or b.power_plugged, "on AC" if (b is None or b.power_plugged) else
            f"ON BATTERY ({b.percent} %) — a multi-day run needs AC; it will stop when the battery does and resume on re-run", fatal=False)
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        try:
            import winreg
            pend = []
            for k in (r"SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired",
                      r"SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"):
                try:
                    winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, k)
                    pend.append(k.split("\\")[-1])
                except OSError:
                    pass
            add("pending reboot", not pend, "none" if not pend else f"Windows reports {pend}: an automatic restart would interrupt "
                "the grid (re-running the command resumes it)", fatal=False)
        except Exception:  # noqa: BLE001
            pass

    cheap_ok = all(c["ok"] or not c["fatal"] for c in checks)
    rec["fingerprint"] = hashlib.sha256(json.dumps({"src": source_hashes(), "data": rec["data"], "epochs": a.epochs, "batch": a.batch,
                                                     "workers": a.workers}, sort_keys=True).encode()).hexdigest()
    # heavy checks: an earlier PASS with the same code + data + settings within 24 h is reused
    rec["ablations"] = {}
    reuse = None
    if cheap_ok and not a.fresh_preflight:
        for p in sorted(grid.dir.glob("preflight_*.json"), reverse=True):
            j = jload(p) or {}
            age_h = (time.time() - p.stat().st_mtime) / 3600
            if j.get("fingerprint") == rec["fingerprint"] and j.get("heavy_ok") and age_h < 24:
                reuse = p
                rec["ablations"] = j.get("ablations", {})
                rec["eval_smoke"] = j.get("eval_smoke")
                add("heavy checks", True, f"reused from {p.name} ({age_h:.1f} h old, same code/data/settings)")
                break
    if cheap_ok and reuse is None:
        # P13 evaluation pipeline, both input modes, 64 VAL images, nothing written
        smoke = {}
        w = weights_path("aerotrack_spd_v1", "last")
        for mode in ("temporal", "single_frame"):
            lp = grid.logs / f"preflight_eval_smoke_{mode}_{stamp()}.log"
            rc, _ = run_logged(grid, [VENV_PY, "-u", EVALUATE, "--split", "val", "--cap", str(VAL_CAP), "--smoke", "64",
                                      "--input-mode", mode, "--weights", w], lp, a.stall_minutes, f"eval smoke ({mode})")
            txt = log_tail(lp)
            smoke[mode] = {"rc": rc, "line": next((ln for ln in txt.splitlines() if ln.startswith("[smoke ")), None), "log": str(lp)}
            add(f"eval smoke ({mode})", rc == 0 and "[smoke] OK" in txt, smoke[mode]["line"] or f"rc {rc}, see {lp.name}")
        rec["eval_smoke"] = smoke
        # P13b the training DataLoader must not grow per batch (it once kept every batch in its workers)
        lp = grid.logs / f"preflight_loader_memory_{stamp()}.log"
        rc, _ = run_logged(grid, [VENV_PY, "-u", Path(__file__).resolve(), "--_loader_probe"], lp, a.stall_minutes, "loader memory probe")
        try:
            lm = json.loads([ln for ln in log_tail(lp).splitlines() if ln.startswith("{")][-1])
        except (IndexError, json.JSONDecodeError):
            lm = {"ok": False, "error": f"rc {rc}, no result (see {lp.name})"}
        rec["loader_memory"] = lm
        add("loader memory", rc == 0 and lm.get("ok"), f"train loader, {lm.get('workers')} workers x {lm.get('batches')} batches: worker private "
            f"{lm.get('worker_private_gb_at_10')} -> {lm.get('worker_private_gb_at_end')} GB, max growth {lm.get('max_growth_gb')} GB "
            f"(limit 0.5), {lm.get('img_per_s')} img/s" if "error" not in lm else lm["error"])
        # P14 train.py's own T1-T8 on each exact job
        for abl in GRID:
            lp = grid.logs / f"preflight_{abl}_{stamp()}.log"
            rc, _ = run_logged(grid, [VENV_PY, "-u", TRAIN_ABLATION, "--ablation", abl, "--epochs", str(a.epochs), "--batch", str(a.batch),
                                      "--workers", str(a.workers), "--preflight-only"], lp, a.stall_minutes, f"train.py pre-flight ({abl})")
            info = parse_train_preflight(log_tail(lp, 400000))
            info.update(rc=rc, log=str(lp))
            rec["ablations"][abl] = info
            tr = info.get("transfer") or {}
            add(f"T1-T8 {abl}", rc == 0 and info["passed"],
                f"{info.get('model_cfg')} {info.get('n_spdconv')} SPDConv, batch {info.get('batch')} x acc {info.get('accumulate')}, "
                f"{info.get('iters_per_epoch')} it/epoch, {info.get('train_workers_planned')} workers, COCO transfer {tr.get('copied')}/{tr.get('total')} "
                f"({tr.get('pct')} %), T6 first conv {info.get('t6_first_conv_in')} ch, VRAM peak {info.get('vram_peak_gb')} GB "
                f"(headroom {info.get('vram_headroom_gb')}), loader {info.get('loader_samples_per_s')} samples/s")
    rec["heavy_ok"] = all(c["ok"] for c in checks if c["name"].startswith(("eval smoke", "loader memory", "T1-T8", "heavy checks")))
    rec["ok"] = all(c["ok"] or not c["fatal"] for c in checks) and rec["heavy_ok"]
    path = grid.dir / f"preflight_{stamp()}.json"
    write_json_atomic(rec, path)
    rec["path"] = str(path)
    return rec


def print_preflight_summary(grid, rec, a):
    print("\n" + "=" * 100)
    print(f"PRE-FLIGHT {'PASSED' if rec['ok'] else 'FAILED'}  ({rel(rec['path'])})")
    print("=" * 100)
    for c in rec["checks"]:
        mark = "PASS" if c["ok"] else ("FAIL" if c["fatal"] else "WARN")
        print(f"  [{mark}] {c['name']}: {c['detail']}")
    ab = rec.get("ablations") or {}
    if ab:
        print("\n  per run (train.py T1-T8 on the exact job):")
        for k, v in ab.items():
            est = v.get("loader_epoch_min_estimate")
            print(f"    {k:<14} {v.get('model_cfg')}, transfer {(v.get('transfer') or {}).get('pct')} %, {v.get('iters_per_epoch')} it/epoch, "
                  f"workers {v.get('train_workers_planned')}, VRAM peak {v.get('vram_peak_gb')} GB, loader-only epoch estimate {est} min")
    print("=" * 100 + "\n", flush=True)


# ============================================================================================ steps
def snapshot_code(grid):
    d = grid.dir / "provenance"
    d.mkdir(parents=True, exist_ok=True)
    s = stamp()

    def git(*args):
        return subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120).stdout
    (d / f"code_{s}.patch").write_text(git("diff", "HEAD"), encoding="utf-8")
    new_files = [ln[3:] for ln in git("status", "--porcelain").splitlines() if ln.startswith("??")]
    # sources that git does not track yet (the new scripts) are copied verbatim, so the exact code of the run survives
    for src in SOURCES:
        if any(src == nf or src.startswith(nf.rstrip("/") + "/") for nf in new_files):
            dst = d / f"untracked_{s}" / src
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / src, dst)
    write_json_atomic({"commit": git("rev-parse", "HEAD").strip(), "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
                       "status_porcelain": git("status", "--porcelain").splitlines(), "untracked": new_files,
                       "source_sha256": source_hashes()}, d / f"code_{s}.json")
    grid.event("code_snapshot", patch=f"provenance/code_{s}.patch")


def wait_for_gpu(grid, what, max_wait_h=6.0):
    """Never start GPU work next to another GPU job: wait (bounded) until none is left."""
    t0 = time.time()
    announced = False
    while True:
        found, gpu_py, _ = foreign_gpu_jobs()
        if not found and not gpu_py:
            if announced:
                grid.event("gpu_free", waited_min=round((time.time() - t0) / 60, 1))
            return
        if not announced:
            grid.event("gpu_busy_waiting", before=what, processes=found + gpu_py)
            announced = True
        if time.time() - t0 > max_wait_h * 3600:
            raise GridFatal(f"another GPU job is still running after {max_wait_h} h: {found + gpu_py}")
        if grid.stop_file.exists():
            raise GridStopped("stopped while waiting for the GPU")
        time.sleep(120)


def train_run(grid, abl, a):
    rd = grid.run_dir(abl)
    info = inspect_run(grid, abl) if rd.exists() else {"exists": False}
    if info.get("exists") and run_complete(info, a.epochs):
        grid.log(f"[train] {rd.name}: already complete ({info['results_rows']} epochs) — verified, not re-run")
        return info
    workers = a.workers
    launches = failures = no_progress = vram_waits = 0
    while True:
        if grid.stop_file.exists():
            raise GridStopped("STOP requested")
        done_before = (info.get("last", {}).get("epoch", -1) + 1) if info.get("last", {}).get("ok") else 0
        cmd = [VENV_PY, "-u", TRAIN_ABLATION, "--ablation", abl, "--epochs", str(a.epochs), "--batch", str(a.batch), "--workers", str(workers)]
        resume_note = "fresh start"
        if rd.exists() and any(rd.iterdir()):
            last, best = info.get("last", {}), info.get("best", {})
            if last.get("exists") and not last.get("ok"):
                bad = rd / "weights" / f"last.unreadable_{stamp()}.pt"
                os.replace(rd / "weights" / "last.pt", bad)
                grid.event("checkpoint_unreadable", run=rd.name, file="last.pt", moved_to=bad.name, error=last.get("error"))
                if best.get("ok"):
                    cmd += ["--resume-from", str(rd / "weights" / "best.pt")]
                    resume_note = f"resume from best.pt (epoch {best.get('epoch')}; last.pt unreadable)"
                    done_before = best.get("epoch", -1) + 1
                else:
                    raise GridFatal(f"{rd.name}: neither last.pt nor best.pt is loadable; refusing to guess")
            else:
                cmd += ["--resume"]
                resume_note = f"resume from last.pt (epoch {last.get('epoch')})" if last.get("ok") else "restart in the existing dir (no checkpoint yet)"
        if failures >= a.max_launches:
            raise GridFatal(f"{rd.name}: {failures} failed launches without finishing; see {rel(grid.logs)}")
        wait_for_gpu(grid, f"train {abl}")
        launches += 1
        grid.update_state(phase=f"train:{abl}")
        grid.event("train_launch", run=rd.name, launch=launches, workers=workers, mode=resume_note)
        lp = grid.logs / f"train_{abl}_launch{launches}_{stamp()}.log"
        t0 = time.time()
        rc, stalled = run_logged(grid, cmd, lp, a.stall_minutes, f"train {abl} (launch {launches})", inner=TRAIN_PY_RX)
        info = inspect_run(grid, abl)
        done_after = (info.get("last", {}).get("epoch", -1) + 1) if info.get("last", {}).get("ok") else 0
        if rc == 0 and run_complete(info, a.epochs):
            grid.event("train_complete", run=rd.name, launch=launches, epochs=done_after, seconds=round(time.time() - t0),
                       train_seconds_total=info["meta"].get("train_seconds"), best_epoch=info["best"].get("epoch"))
            return info
        kind, detail = ("inconsistent", f"exit 0 but the run is not complete ({done_after}/{a.epochs} epochs)") if rc == 0 \
            else classify(log_tail(lp), rc, stalled)
        grid.event("train_failed", run=rd.name, launch=launches, rc=rc, kind=kind, detail=detail, epochs_before=done_before,
                   epochs_after=done_after, log=lp.name)
        if kind in FATAL_KINDS or kind == "inconsistent":
            raise GridFatal(f"{rd.name}: {kind}: {detail} (log {lp})")
        if kind == "vram" and done_after <= done_before:
            # the card is busy (another app holds VRAM): waiting is the fix, and it is not a failure of the run
            vram_waits += 1
            if vram_waits > 12:
                raise GridFatal(f"{rd.name}: VRAM too tight on {vram_waits} launches (~2 h) — something else holds the card")
            pause = 600
        else:
            failures += 1
            no_progress = 0 if done_after > done_before else no_progress + 1
            if no_progress >= 3:
                raise GridFatal(f"{rd.name}: three failed launches in a row finished no epoch (last: {kind}: {detail})")
            if kind == "host_memory":
                workers = max(1, workers // 2)
                pause = 180
            else:
                pause = 120
        grid.event("train_recover", run=rd.name, action=f"resume after {pause}s", workers=workers, kind=kind)
        time.sleep(pause)


def val_eval(grid, abl, a):
    rd = grid.run_dir(abl)
    ws = [(ck, rd / "weights" / f"{ck}.pt") for ck in ("best", "last")]
    need = [w for ck, w in ws if not val_row_ok(rd.name, ck, w)]
    if not need:
        grid.log(f"[val] {rd.name}: best.pt and last.pt already scored on set {REGISTERED_VAL_SET} — not re-run")
        return
    for attempt in range(1, 4):
        wait_for_gpu(grid, f"val eval {abl}")
        grid.update_state(phase=f"val:{abl}")
        grid.event("val_eval_start", run=rd.name, attempt=attempt, weights=[w.name for w in need])
        lp = grid.logs / f"val_{abl}_attempt{attempt}_{stamp()}.log"
        rc, stalled = run_logged(grid, [VENV_PY, "-u", EVALUATE, "--split", "val", "--cap", str(VAL_CAP), "--weights", *need],
                                 lp, a.stall_minutes, f"val eval {abl}")
        need = [w for ck, w in ws if not val_row_ok(rd.name, ck, w)]
        if not need:
            grid.event("val_eval_done", run=rd.name, attempt=attempt)
            return
        kind, detail = classify(log_tail(lp), rc, stalled) if rc else ("inconsistent", "exit 0 but rows missing / hash or input mismatch")
        grid.event("val_eval_failed", run=rd.name, attempt=attempt, rc=rc, kind=kind, detail=detail, log=lp.name)
        if kind in FATAL_KINDS or kind == "inconsistent":
            raise GridFatal(f"val evaluation of {rd.name}: {kind}: {detail} (log {lp})")
        time.sleep(120)
    raise GridFatal(f"val evaluation of {rd.name} failed three times")


def render(grid, final=False):
    lp = grid.logs / f"render_{stamp()}.log"
    rc, _ = run_logged(grid, [VENV_PY, "-u", RENDER], lp, 30, "render_report")
    if rc:
        grid.event("render_failed", rc=rc, log=lp.name, tail=log_tail(lp, 3000))
        if final:
            raise GridFatal(f"render_report.py failed (log {lp})")
    else:
        grid.event("rendered", files="README.md, docs/AEROTRACK_REPAIR_REPORT.md")


def test_weights(grid):
    return [(run, ck) for run, ck in HISTORICAL] + [(grid.run_dir(abl).name, ck) for abl in GRID for ck in ("best", "last")]


def test_todo(grid):
    """Checkpoints still to be scored on TEST. A ledger line is the commit record; a row scored and recorded but
    without its ledger line (a crash between the two writes) gets its ledger line back instead of a second scoring."""
    rows, tset = summary_rows("test", TEST_CAP)
    led = read_ledger()
    todo = []
    for run, ck in test_weights(grid):
        w = weights_path(run, ck)
        s = sha256(w)
        in_led = [x for x in led if x.get("weights_sha256") == s]
        row = row_for(rows, run, ck, s[:12])
        if in_led and row:
            continue
        if in_led and not row:
            raise GridFatal(f"{run}/{ck} is in the test ledger but not in detector_test_cap{TEST_CAP}.json — it was scored outside "
                            "this workflow; refusing to score the test split again. Inspect the ledger.")
        if row and not in_led:
            rep = jload(REPO_ROOT / "runs" / run / f"eval_repair_test_cap{TEST_CAP}" / ck / "evaluation_report.json") or {}
            m = rep.get("_meta", {})
            if m.get("weights_sha256") == s and (m.get("eval_set") or {}).get("sha256") == tset.get("sha256"):
                with open(LEDGER, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"timestamp": m.get("timestamp"), "weights": str(w), "weights_sha256": s, "set_sha256": tset["sha256"],
                                        "mAP50": rep["overall"]["map50"], "input": m.get("input_frames"),
                                        "git_commit": (m.get("git") or {}).get("commit"),
                                        "ledger_line_restored_by": "run_grid.py (scored and recorded; ledger write was interrupted)"}) + "\n")
                grid.event("test_ledger_restored", run=run, ckpt=ck)
                continue
            raise GridFatal(f"{run}/{ck}: test row present without a matching report/ledger line; refusing to guess")
        todo.append(w)
    return todo


def test_eval(grid, a):
    # preconditions: the whole grid finished and validated; the historical rows are the registered ones
    for abl in GRID:
        info = inspect_run(grid, abl)
        if not run_complete(info, a.epochs):
            raise GridFatal(f"test refused: {grid.run_dir(abl).name} is not complete")
        for ck in ("best", "last"):
            if not val_row_ok(grid.run_dir(abl).name, ck, grid.run_dir(abl) / "weights" / f"{ck}.pt"):
                raise GridFatal(f"test refused: {grid.run_dir(abl).name}/{ck} has no valid val row")
    todo = test_todo(grid)
    if not todo:
        grid.log("[test] every checkpoint already has its single test evaluation — not re-run")
        return
    for attempt in range(1, 4):
        wait_for_gpu(grid, "test eval")
        grid.update_state(phase="test")
        grid.event("test_eval_start", attempt=attempt, n=len(todo), weights=[rel(w) for w in todo])
        lp = grid.logs / f"test_attempt{attempt}_{stamp()}.log"
        rc, stalled = run_logged(grid, [VENV_PY, "-u", EVALUATE, "--split", "test", "--final-eval", "--cap", str(TEST_CAP), "--weights", *todo],
                                 lp, a.stall_minutes, f"TEST eval ({len(todo)} checkpoints)")
        todo = test_todo(grid)
        if not todo:
            grid.event("test_eval_done", attempt=attempt)
            return
        kind, detail = classify(log_tail(lp), rc, stalled) if rc else ("inconsistent", "exit 0 but checkpoints missing from the ledger")
        grid.event("test_eval_interrupted", attempt=attempt, rc=rc, kind=kind, detail=detail, remaining=len(todo), log=lp.name,
                   note="checkpoints with a ledger line are final; only the remaining ones are scored on the next attempt")
        if kind in FATAL_KINDS or kind == "inconsistent":
            raise GridFatal(f"test evaluation: {kind}: {detail} (log {lp})")
        time.sleep(120)
    raise GridFatal("test evaluation failed three times")


def final_report(grid, a):
    lp = grid.logs / f"grid_report_{stamp()}.log"
    rc, _ = run_logged(grid, [VENV_PY, "-u", GRID_REPORT, "--epochs", str(a.epochs)], lp, 30, "grid_report")
    if rc:
        raise GridFatal(f"grid_report.py failed (log {lp})")
    return REPORTS_DIR / f"ablation_grid_e{a.epochs}.md"


def verify_artifacts(a, out):
    """Never claim completion without the files: every report exists, the README links the grid report, and the test
    ledger holds exactly one line per checkpoint of the final table."""
    for p in (out, out.with_suffix(".json"), REPO_ROOT / "README.md", REPO_ROOT / "docs" / "AEROTRACK_REPAIR_REPORT.md",
              REPORTS_DIR / f"detector_test_cap{TEST_CAP}.json", REPORTS_DIR / f"detector_val_cap{VAL_CAP}.json", LEDGER):
        if not p.exists():
            raise GridFatal(f"expected artefact missing: {p}")
    if out.name not in (REPO_ROOT / "README.md").read_text(encoding="utf-8"):
        raise GridFatal("README.md does not link the grid report")
    shas = [x.get("weights_sha256") for x in read_ledger()]
    if len(shas) != len(set(shas)):
        raise GridFatal("the test ledger holds more than one line for a checkpoint")


# ============================================================================================ CLI modes
def keep_awake(on=True):
    """Ask Windows not to IDLE-sleep while this process runs (released on exit). No system setting is changed."""
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def acquire_lock(grid):
    grid.mkdirs()
    for _ in range(2):
        try:
            fd = os.open(str(grid.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "started": now(), "argv": sys.argv}, f)
            atexit.register(lambda: grid.lock.unlink(missing_ok=True))
            return
        except FileExistsError:
            info = jload(grid.lock) or {}
            if pid_alive(info.get("pid", -1), "run_grid.py"):
                raise SystemExit(f"[ABORT] a grid runner is already active (pid {info.get('pid')}, since {info.get('started')}). "
                                 f"Use --status, or --stop it first. Two trainers on one 8 GB card is an OOM, not a speed-up.")
            grid.lock.unlink(missing_ok=True)
    raise SystemExit("[ABORT] could not take the grid lock")


def status(grid, a):
    st = grid.state()
    info = jload(grid.lock) or {}
    alive = pid_alive(info.get("pid", -1), "run_grid.py")
    print(f"grid {rel(grid.dir)}: runner {'RUNNING pid ' + str(info.get('pid')) if alive else 'not running'}; "
          f"phase {st.get('phase')}; outcome {st.get('outcome')}; updated {st.get('updated')}")
    for abl in GRID:
        rd = grid.run_dir(abl)
        rows = 0
        if (rd / "results.csv").exists():
            with open(rd / "results.csv", encoding="utf-8", newline="") as f:
                rr = list(csv.DictReader(f))
            rows = len(rr)
            last = rr[-1] if rr else {}
            secs = [float(r["epoch_time_s"]) for r in rr if r.get("epoch_time_s")]
            extra = (f", last epoch {int(float(last['epoch']))}: mAP50 {last.get('mAP50')} micro {last.get('micro_AP50')} fitness {last.get('fitness')}, "
                     f"median train time/epoch {sorted(secs)[len(secs) // 2] / 60:.1f} min") if rr else ""
        else:
            extra = ""
        meta = jload(rd / "run_meta.json") or {}
        print(f"  {abl:<14} {'-' if not rd.exists() else f'{rows}/{a.epochs} epochs'}{extra}"
              + (f"; finished rc {meta.get('return_code')}" if meta.get("timestamp_end") else ""))
    cur = st.get("current") or {}
    if cur.get("log"):
        print(f"current step: {cur.get('what')} (pid {cur.get('pid')}, since {cur.get('started')})\n--- {cur['log']}")
        print("\n".join(log_tail(cur["log"], 4000).splitlines()[-12:]))
    if grid.events_path.exists():
        print("--- last events")
        print("\n".join(grid.events_path.read_text(encoding="utf-8").splitlines()[-8:]))


def stop(grid):
    info = jload(grid.lock) or {}
    if not pid_alive(info.get("pid", -1), "run_grid.py"):
        print("no grid runner is active")
        return
    grid.stop_file.write_text(now(), encoding="utf-8")
    print(f"STOP requested for pid {info['pid']}; it stops the current step within ~30 s (last.pt keeps the last finished epoch). "
          f"Re-run the command to resume.")


def detach(grid, argv):
    """Start the runner in the background, two steps removed from the calling shell: a short-lived spawner waits
    ~20 s (the calling shell has returned by then) and starts the runner, then exits. A runner started straight
    from an agent's shell tool was killed after ~2 h while its training child kept going; the runner is now never
    a process that the calling shell (or a tool that tracks it) knows about."""
    grid.mkdirs()
    log = grid.logs / f"driver_console_{stamp()}.log"
    rest = [x for x in argv if x != "--detach"]
    spawner = [str(VENV_PY), str(Path(__file__).resolve()), "--_spawn", str(log)] + rest
    _popen_detached(spawner, subprocess.DEVNULL)
    print(f"[detach] the grid runner starts in ~20 s, detached from this terminal\n"
          f"         timeline: {grid.driver_log}\n         console : {log}\n"
          f"         progress: {subprocess.list2cmdline([str(VENV_PY), str(Path(__file__).resolve()), '--status'])}")


def _popen_detached(cmd, out):
    base = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW if os.name == "nt" else 0
    kw = dict(cwd=str(REPO_ROOT), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=child_env(), close_fds=True)
    try:
        return subprocess.Popen(cmd, creationflags=base | (CREATE_BREAKAWAY_FROM_JOB if os.name == "nt" else 0), **kw)
    except OSError:
        return subprocess.Popen(cmd, creationflags=base, **kw)


def _spawn_runner(log, rest):
    time.sleep(20)
    with open(log, "ab") as f:
        _popen_detached([str(VENV_PY), "-u", str(Path(__file__).resolve())] + rest, f)


def grid_orphans(epochs):
    """Grid work still running from an earlier runner (killed or stopped): its training / evaluation / report children
    outlive it and finish their step on their own."""
    ps = _psutil()
    mine = own_tree_pids()
    rx = re.compile(rf"(train_ablation\.py.*--epochs {epochs}\b|c_model[\\/]train\.py.*repair_abl_\w+_e{epochs}\b|"
                    r"evaluate_detector\.py|render_report\.py|grid_report\.py)")
    out = []
    for p in ps.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in mine or "python" not in (p.info["name"] or "").lower():
                continue
            cl = " ".join(p.info["cmdline"] or [])
            if rx.search(cl):
                out.append({"pid": p.info["pid"], "cmdline": cl[:200]})
        except (ps.NoSuchProcess, ps.AccessDenied):
            continue
    return out


def adopt_orphans(grid, epochs, max_wait_h=10.0):
    """Let an orphaned step finish (its results land on disk and are verified like any other) before touching the GPU."""
    orphans = grid_orphans(epochs)
    if not orphans:
        return
    grid.event("orphans_adopted", n=len(orphans), processes=orphans,
               note="an earlier runner's step is still running; waiting for it to finish, then verifying its output")
    t0 = time.time()
    while grid_orphans(epochs):
        if grid.stop_file.exists():
            raise GridStopped("stopped while waiting for an orphaned step")
        if time.time() - t0 > max_wait_h * 3600:
            raise GridFatal(f"orphaned grid processes still running after {max_wait_h} h: {grid_orphans(epochs)}")
        time.sleep(60)
    grid.event("orphans_finished", waited_min=round((time.time() - t0) / 60, 1))


def main():
    ap = argparse.ArgumentParser(description="AeroTrack repair: the whole remaining workflow in one command",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--epochs", type=int, default=20, help="pre-registered grid budget (20)")
    ap.add_argument("--batch", type=int, default=32, help="pre-registered batch (historical v1/v2: 32)")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader worker ceiling; train.py plans the real number from free memory")
    ap.add_argument("--max-launches", type=int, default=8, help="launches per training run before giving up")
    ap.add_argument("--stall-minutes", type=float, default=60, help="kill + resume a step whose log is silent this long")
    ap.add_argument("--preflight-only", action="store_true", help="run the pre-flight, print the summary, exit")
    ap.add_argument("--fresh-preflight", action="store_true", help="do not reuse an earlier heavy pre-flight")
    ap.add_argument("--detach", action="store_true", help="start in the background and return")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--note", default=None, help="append an operator note (why something was done) to the event log")
    ap.add_argument("--_inspect", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_loader_probe", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_spawn", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._inspect:
        return _inspect_child(a._inspect)
    if a._loader_probe:
        return _loader_probe_child()
    if a._spawn:
        rest = sys.argv[1:]
        k = rest.index("--_spawn")
        return _spawn_runner(a._spawn, rest[:k] + rest[k + 2:])
    if Path(sys.executable).resolve() != VENV_PY.resolve() and VENV_PY.exists():
        # the project's venv holds the CUDA torch build; re-run under it whatever python was typed
        sys.exit(subprocess.call([str(VENV_PY)] + sys.argv, cwd=str(REPO_ROOT)))
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    grid = Grid(a.epochs)
    if a.status:
        return status(grid, a)
    if a.stop:
        return stop(grid)
    if a.note:
        grid.mkdirs()
        grid.event("operator_note", text=a.note)
        return 0
    if a.detach:
        return detach(grid, sys.argv[1:])

    acquire_lock(grid)
    if grid.stop_file.exists():
        grid.stop_file.unlink()
    try:
        _psutil().Process().nice(_psutil().BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001
        pass
    keep_awake(True)
    atexit.register(keep_awake, False)
    grid.update_state(phase="starting", outcome=None, current=None)
    grid.event("start", pid=os.getpid(), argv=" ".join(sys.argv), epochs=a.epochs, batch=a.batch, workers=a.workers)
    t_start = time.time()
    try:
        adopt_orphans(grid, a.epochs)
        pf = preflight(grid, a)
        print_preflight_summary(grid, pf, a)
        grid.event("preflight", ok=pf["ok"], record=Path(pf["path"]).name)
        if not pf["ok"]:
            grid.update_state(phase="preflight", outcome="preflight failed")
            return 2
        if a.preflight_only:
            return 0
        snapshot_code(grid)
        for abl in GRID:
            train_run(grid, abl, a)
            val_eval(grid, abl, a)
            render(grid)
            grid.update_state(**{f"done_{abl}": now()})
        test_eval(grid, a)
        out = final_report(grid, a)          # first, so the regenerated README can link to it
        render(grid, final=True)
        verify_artifacts(a, out)
        grid.update_state(phase="done", outcome="complete", final_report=str(out))
        grid.event("complete", hours=round((time.time() - t_start) / 3600, 2), report=rel(out))
        print(f"\nCOMPLETE. Final report: {out}\n", flush=True)
        return 0
    except GridStopped as e:
        grid.update_state(phase="stopped", outcome=str(e))
        grid.event("stopped", reason=str(e))
        return 3
    except GridFatal as e:
        grid.update_state(phase="failed", outcome=str(e))
        grid.event("fatal", reason=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
