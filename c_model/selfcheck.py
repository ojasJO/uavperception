#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selfcheck.py  --  whole-project consistency check
=================================================
Run this after ANY edit to the project's Python or to `train.ipynb`:

    venv\\Scripts\\python c_model/selfcheck.py            # static, ~5 seconds
    venv\\Scripts\\python c_model/selfcheck.py --full     # + functional, ~3 minutes

Why it exists
-------------
This project is a handful of modules that pass structured objects to each other
(`TrainConfig`, the `ctx` dict, the sample index) plus a notebook that calls into
all of them. Rename a field in one place and nothing complains until the code
path that reads it happens to run -- possibly twenty minutes into a build, or
only in the ablation branch nobody exercised. Python will not tell you; a test
suite over a GPU training loop is too slow to be the feedback loop.

So this checks the seams directly, by reading the source rather than running it:

  S1  every .py in the project byte-compiles
  S2  every module imports cleanly, in dependency order
  S3  every `cfg.<attr>` / `self.cfg.<attr>` access names a real TrainConfig field
      <- this is the check that catches "'TrainConfig' has no attribute 'val_workers'"
  S4  every `ctx["<key>"]` access names a key that build() actually returns
  S5  every `T.x` / `P.x` / `E.x` the notebook calls exists in that module
  S6  every `from X import a, b` resolves
  S7  the notebook's code cells all parse, and its config cell defines every
      name later cells read
  S8  no module shadows another's name, and nothing imports a module that would
      be stale under `importlib.reload` of a single module

`--full` then does the functional pass: build the corpus, run pre-flight T1-T8,
one forward/backward, one validation pass, and build every ablation config.
"""

import argparse
import ast
import io
import json
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "b_pipeline"))
sys.path.insert(0, str(ROOT / "a_inspection"))

PY_FILES = sorted(
    [p for p in (ROOT / "c_model").glob("*.py")] +
    [p for p in (ROOT / "b_pipeline").glob("*.py")] +
    [p for p in (ROOT / "a_inspection").glob("*.py")] +
    [ROOT / "rebuild_junctions.py"]
)
NOTEBOOK = ROOT / "train.ipynb"

# Modules that must be importable, in dependency order.
IMPORT_ORDER = ["custom_modules", "stacked_dataset", "imbalance_sampler",
                "aerotrack_trainer", "evaluation", "preflight", "train"]

_results = []


def record(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")


def _parse(path):
    src = io.open(path, encoding="utf-8").read()
    return src, ast.parse(src, filename=str(path))


def notebook_code_cells(nb_path=NOTEBOOK):
    """-> list of (cell_index, source) for code cells."""
    if not nb_path.exists():
        return []
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            out.append((i, "".join(c["source"])))
    return out


# --------------------------------------------------------------------------- #
#  S1 — compile
# --------------------------------------------------------------------------- #
def s1_compile():
    bad = []
    for p in PY_FILES:
        try:
            src = io.open(p, encoding="utf-8").read()
            compile(src, str(p), "exec")
        except SyntaxError as e:
            bad.append(f"{p.relative_to(ROOT)}:{e.lineno}: {e.msg}")
    record("S1 every .py compiles", not bad,
           "\n".join(bad) if bad else f"{len(PY_FILES)} files")


# --------------------------------------------------------------------------- #
#  S2 — import, in dependency order, FRESH
# --------------------------------------------------------------------------- #
def s2_imports():
    import importlib
    # Purge first so a stale sys.modules entry cannot mask a broken file.
    for name in list(sys.modules):
        if name in IMPORT_ORDER:
            del sys.modules[name]
    bad = []
    loaded = []
    for name in IMPORT_ORDER:
        try:
            importlib.import_module(name)
            loaded.append(name)
        except Exception as e:
            bad.append(f"{name}: {e!r}\n" + traceback.format_exc(limit=2))
    record("S2 modules import cleanly", not bad,
           "\n".join(bad) if bad else " -> ".join(loaded))


# --------------------------------------------------------------------------- #
#  S3 — TrainConfig field accesses
# --------------------------------------------------------------------------- #
def _cfg_attr_uses(tree, path):
    """Find every `cfg.X` and `self.cfg.X` attribute read."""
    uses = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        v = node.value
        is_cfg = (isinstance(v, ast.Name) and v.id in ("cfg", "config")) or \
                 (isinstance(v, ast.Attribute) and v.attr == "cfg" and
                  isinstance(v.value, ast.Name) and v.value.id == "self")
        if is_cfg:
            uses.append((node.attr, getattr(node, "lineno", 0), path))
    return uses


def s3_trainconfig():
    from aerotrack_trainer import TrainConfig
    fields = set(TrainConfig().to_dict()) | set(dir(TrainConfig))
    bad = []
    n = 0
    sources = [(p, _parse(p)[1]) for p in PY_FILES if p.suffix == ".py"]
    for p, tree in sources:
        for attr, lineno, _ in _cfg_attr_uses(tree, p):
            n += 1
            if attr not in fields:
                bad.append(f"{p.relative_to(ROOT)}:{lineno}: cfg.{attr} is not a "
                           f"TrainConfig field")
    for idx, src in notebook_code_cells():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for attr, lineno, _ in _cfg_attr_uses(tree, NOTEBOOK):
            n += 1
            if attr not in fields:
                bad.append(f"train.ipynb cell {idx}: cfg.{attr} is not a "
                           f"TrainConfig field")
    record("S3 cfg.<field> accesses all exist", not bad,
           "\n".join(bad) if bad else f"{n} accesses checked against "
                                      f"{len(TrainConfig().to_dict())} fields")


# --------------------------------------------------------------------------- #
#  S4 — ctx dict keys
# --------------------------------------------------------------------------- #
def _ctx_keys_produced():
    """Read build()'s `ctx = {...}` literal out of train.py."""
    src, tree = _parse(ROOT / "c_model" / "train.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id == "ctx":
                return {k.value for k in node.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return set()


def s4_ctx_keys():
    produced = _ctx_keys_produced()
    if not produced:
        record("S4 ctx keys", False, "could not locate the ctx literal in build()")
        return
    names = {"ctx", "cx", "s0_ctx"}
    bad = []
    n = 0

    def scan(tree, where):
        nonlocal n
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in names
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                n += 1
                if node.slice.value not in produced:
                    bad.append(f"{where}:{node.lineno}: "
                               f"ctx[{node.slice.value!r}] is never set by build()")

    for p in PY_FILES:
        scan(_parse(p)[1], p.relative_to(ROOT))
    for idx, src in notebook_code_cells():
        try:
            scan(ast.parse(src), f"train.ipynb cell {idx}")
        except SyntaxError:
            pass
    record("S4 ctx[...] keys all produced by build()", not bad,
           "\n".join(bad) if bad else
           f"{n} accesses checked against {len(produced)} keys: "
           f"{', '.join(sorted(produced))}")


# --------------------------------------------------------------------------- #
#  S5 — notebook -> module attribute calls
# --------------------------------------------------------------------------- #
NB_ALIASES = {"T": "train", "P": "preflight", "E": "evaluation"}


def s5_notebook_module_calls():
    import importlib
    mods = {}
    for alias, name in NB_ALIASES.items():
        try:
            mods[alias] = importlib.import_module(name)
        except Exception as e:
            record("S5 notebook module calls", False, f"cannot import {name}: {e!r}")
            return
    bad = []
    n = 0
    for idx, src in notebook_code_cells():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in mods):
                n += 1
                if not hasattr(mods[node.value.id], node.attr):
                    bad.append(f"train.ipynb cell {idx}: "
                               f"{node.value.id}.{node.attr} does not exist in "
                               f"{NB_ALIASES[node.value.id]}.py")
    record("S5 notebook's T./P./E. calls all resolve", not bad,
           "\n".join(bad) if bad else f"{n} attribute calls checked")


# --------------------------------------------------------------------------- #
#  S6 — from-imports resolve
# --------------------------------------------------------------------------- #
LOCAL_MODULES = set(IMPORT_ORDER) | {"eda_common"}


def s6_from_imports():
    import importlib
    bad = []
    n = 0
    for p in PY_FILES:
        tree = _parse(p)[1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in LOCAL_MODULES:
                try:
                    m = importlib.import_module(node.module)
                except Exception as e:
                    bad.append(f"{p.relative_to(ROOT)}:{node.lineno}: "
                               f"cannot import {node.module}: {e!r}")
                    continue
                for a in node.names:
                    n += 1
                    if a.name != "*" and not hasattr(m, a.name):
                        bad.append(f"{p.relative_to(ROOT)}:{node.lineno}: "
                                   f"{node.module} has no {a.name!r}")
    record("S6 local from-imports resolve", not bad,
           "\n".join(bad) if bad else f"{n} imported names checked")


# --------------------------------------------------------------------------- #
#  S7 — notebook internal consistency
# --------------------------------------------------------------------------- #
def s7_notebook():
    cells = notebook_code_cells()
    if not cells:
        record("S7 notebook", False, f"{NOTEBOOK} not found or has no code cells")
        return
    bad = []
    # every cell parses
    for idx, src in cells:
        try:
            ast.parse(src)
        except SyntaxError as e:
            bad.append(f"cell {idx}: {e.msg} (line {e.lineno})")
    # names assigned anywhere earlier must cover names read later at module level
    assigned = set(dir(__builtins__)) | {"__name__"}
    for idx, src in cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in tgts:
                    for nn in ast.walk(t):
                        if isinstance(nn, ast.Name):
                            assigned.add(nn.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    assigned.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                assigned.add(node.name)
                # ...and its PARAMETERS. `ast.walk` descends into the body, so a
                # helper's local reads are checked as if they were module-level.
                # Without this, `def table(df, note=None)` reports `df` and
                # `note` as undefined names in every notebook that defines a
                # formatting helper -- eleven false failures on this one, which
                # is enough noise to make the check get ignored.
                if not isinstance(node, ast.ClassDef):
                    a = node.args
                    for arg in (list(a.posonlyargs) + list(a.args) +
                                list(a.kwonlyargs) +
                                [x for x in (a.vararg, a.kwarg) if x]):
                        assigned.add(arg.arg)
            elif isinstance(node, ast.For):
                for nn in ast.walk(node.target):
                    if isinstance(nn, ast.Name):
                        assigned.add(nn.id)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for it in node.items:
                    if it.optional_vars is not None:
                        for nn in ast.walk(it.optional_vars):
                            if isinstance(nn, ast.Name):
                                assigned.add(nn.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                assigned.add(node.name)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                   ast.GeneratorExp)):
                for gen in node.generators:
                    for nn in ast.walk(gen.target):
                        if isinstance(nn, ast.Name):
                            assigned.add(nn.id)
            elif isinstance(node, ast.Lambda):
                for a in node.args.args:
                    assigned.add(a.arg)
            elif isinstance(node, ast.Global):
                assigned.update(node.names)
    import builtins
    known = assigned | set(dir(builtins))
    for idx, src in cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in known:
                    bad.append(f"cell {idx}: name {node.id!r} is read but never "
                               f"assigned in any cell")
    seen = set()
    bad = [b for b in bad if not (b in seen or seen.add(b))]
    record("S7 notebook cells parse and names resolve", not bad,
           "\n".join(bad) if bad else f"{len(cells)} code cells")


# --------------------------------------------------------------------------- #
#  S8 — the reload hazard
# --------------------------------------------------------------------------- #
def s8_reload_hazard():
    """A notebook that reloads ONE module while others hold stale copies of its
    objects is the source of `'TrainConfig' object has no attribute ...`.

    `importlib.reload(train)` re-executes train.py, but its
    `from aerotrack_trainer import TrainConfig` binds whatever is already in
    sys.modules -- the OLD class. New caller, old callee, and an AttributeError
    that points at a field which plainly exists on disk. The notebook must purge
    every project module before importing, not reload one."""
    bad = []
    joined = "\n".join(src for _, src in notebook_code_cells())
    if "importlib.reload" in joined and "sys.modules" not in joined:
        bad.append("train.ipynb calls importlib.reload() without purging "
                   "sys.modules first — a single-module reload leaves every "
                   "other project module holding stale classes.")
    if "_purge_project_modules" not in joined and "sys.modules" not in joined:
        bad.append("train.ipynb never purges project modules; edits made while "
                   "the kernel is alive will not take effect consistently.")
    record("S8 notebook reloads project modules safely", not bad,
           "\n".join(bad) if bad else
           "notebook purges project modules from sys.modules before importing")


# --------------------------------------------------------------------------- #
#  Functional pass
# --------------------------------------------------------------------------- #
def functional():
    import torch
    import train as T
    import preflight as P
    import evaluation as E
    from torch.utils.data import DataLoader
    from stacked_dataset import AeroTrackDataset, aerotrack_collate
    from aerotrack_trainer import build_model

    print("\n" + "=" * 78)
    print("FUNCTIONAL PASS")
    print("=" * 78)
    device = T.require_gpu()
    aug = dict(hflip=0.5, scale=(0.90, 1.25), translate=0.10, brightness=0.15,
               contrast=0.15, copy_paste=0.30, copy_paste_max=3)
    trainer, ctx = T.build("smoke", device=device,
                           overrides=dict(workers=2, batch=8, imgsz=640),
                           augment_kwargs=aug)
    checks = [
        P.check_environment(),
        P.check_dataset(ctx["val_ds"], batch_size=4),
        P.check_boundary(ctx["val_ds"]),
        P.check_leakage(ctx["train_seq_keys"], ctx["val_seq_keys"]),
        P.check_collate(trainer, ctx["val_ds"], batch_size=4),
        P.check_checkpoint_roundtrip(trainer, device),
        P.check_throughput(ctx["train_loader"], n_batches=10),
        P.check_vram(trainer, ctx["train_ds"], ctx["cfg"].batch, device),
    ]
    ok_pf = P.run_all(checks, raise_on_fail=False)
    record("F1 pre-flight T1-T8", ok_pf, "")

    # Eight steps, and the question is only "does the loss move in the right
    # direction" -- this is the seam test, not S0. The real S0 probe (200 iters,
    # 75% reduction) runs as the first stage of every training job.
    r = P.check_overfit(trainer, iters=8, min_reduction=0.05, verbose=False)
    record("F2 loss decreases on a fixed batch", r.ok, r.detail)

    small = T.subsample(ctx["val_samples"],
                        {"anti_uav": 12, "cst": 12, "det_fly": 8})
    ds = AeroTrackDataset(small, img_size=(640, 640), augment=None,
                          letterbox=True, out_dtype="uint8", return_meta=True,
                          label_cache=ctx["label_cache"])
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                    collate_fn=aerotrack_collate)
    recs = E.collect_records(trainer.ema.ema if trainer.ema else trainer.model,
                             dl, device, imgsz=640,
                             amp_dtype=trainer.amp_dtype, progress=False)
    rep = E.full_report(recs, save_dir=None)
    record("F3 evaluation suite runs", len(recs) > 0 and "by_size" in rep,
           f"{len(recs)} images, strata "
           f"{ {k: v['n_gt'] for k, v in rep['by_size'].items()} }")

    T.release(trainer, ctx)

    rows = []
    ok_abl = True
    for name, cfgname in (("baseline", "yolo11-spd.yaml"),
                          ("no_spd", "yolo11-stride.yaml"),
                          ("p2_head", "yolo11-spd-p2.yaml")):
        try:
            m = build_model(ROOT / "c_model" / cfgname, ch=9, nc=2,
                            verbose=False).eval()
            with torch.no_grad():
                out = m(torch.randn(1, 9, 640, 640))
            preds = out[0] if isinstance(out, (list, tuple)) else out
            rows.append(f"{name:<9s} {sum(p.numel() for p in m.parameters())/1e6:5.2f} M  "
                        f"preds {tuple(preds.shape)}")
            del m
        except Exception as e:
            ok_abl = False
            rows.append(f"{name}: {e!r}")
    record("F4 every model config builds", ok_abl, "\n".join(rows))

    for abl in T.ABLATIONS:
        cfgname = T.ABLATIONS[abl].get("model_cfg")
        if cfgname and not (ROOT / "c_model" / cfgname).exists():
            record(f"F5 ablation {abl} config present", False,
                   f"missing {cfgname}")
            return
    record("F5 every ablation resolves to a real config", True,
           ", ".join(T.ABLATIONS))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run the functional pass (needs the GPU + corpus)")
    args = ap.parse_args()

    print("=" * 78)
    print("AeroTrack-Net — whole-project self check")
    print("=" * 78)
    s1_compile()
    s2_imports()
    s3_trainconfig()
    s4_ctx_keys()
    s5_notebook_module_calls()
    s6_from_imports()
    s7_notebook()
    s8_reload_hazard()

    if args.full:
        try:
            functional()
        except Exception as e:
            record("functional pass", False,
                   f"{e!r}\n{traceback.format_exc(limit=4)}")

    print("=" * 78)
    failed = [n for n, ok, _ in _results if not ok]
    if failed:
        print(f"FAILED ({len(failed)}/{len(_results)}): " + ", ".join(failed))
        sys.exit(1)
    print(f"ALL {len(_results)} CHECKS PASSED")


if __name__ == "__main__":
    main()
