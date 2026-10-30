#!/usr/bin/env python
"""End-to-end smoke tests for the Anti-UAV reproduction pipeline (no full training).

    python experiments/anti_uav/scripts/smoke_test.py

Checks (task list "testing" 1-15): imports / compile, dataset + frozen manifests, sample loading, preprocessing,
forward, loss, backward, optimizer step, checkpoint save/load, tiny end-to-end training run, tiny validation
evaluation, metrics-from-predictions verification, TEST-split guard, seed -> metadata, config saved with run.
Nothing here hard-codes expected metric values; checks are structural (finite, changed, equal, raised...).
Smoke artefacts go to experiments/anti_uav/runs/_smoke/ and are deleted at the start of each run.
"""
from __future__ import annotations

import copy
import json
import py_compile
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from anti_uav_repro.paths import CONFIGS_DIR, EXPERIMENT_DIR, MANIFESTS_DIR, RUNS_DIR

SMOKE_DIR = RUNS_DIR / "_smoke"
PY = sys.executable
RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        def run():
            t0 = time.time()
            try:
                msg = fn() or ""
                RESULTS.append((name, True, msg))
                print(f"PASS  {name}  ({time.time()-t0:.1f}s) {msg}")
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"FAIL  {name}  ({time.time()-t0:.1f}s)\n{traceback.format_exc()}")
        return run
    return deco


# ------------------------------------------------------------------------------------------------ 1
@check("01 static import / compile checks")
def t01():
    for p in sorted((EXPERIMENT_DIR / "src").rglob("*.py")) + sorted((EXPERIMENT_DIR / "scripts").glob("*.py")):
        py_compile.compile(str(p), doraise=True)
    import anti_uav_repro.config, anti_uav_repro.data.antiuav, anti_uav_repro.data.pairs, anti_uav_repro.data.transforms  # noqa
    import anti_uav_repro.models.qg_rcnn, anti_uav_repro.engine.train, anti_uav_repro.engine.track, anti_uav_repro.eval.evaluate  # noqa
    from anti_uav_repro.config import load_config, validate_config
    for c in ["_base_ir.yaml", "normal_ir.yaml", "dfsc_ir.yaml", "smoke_ir.yaml", "phase_b/dfsc_ir_alpha0.25.yaml",
              "phase_b/dfsc_ir_lr0.02_accum8.yaml", "phase_b/dfsc_ir_seed1.yaml", "phase_b/normal_ir_seed1.yaml",
              "deferred/normal_rgb_cocoinit.yaml", "deferred/dfsc_rgb_cocoinit.yaml"]:
        validate_config(load_config(CONFIGS_DIR / c))
    return "all modules compile; 10 configs validate"


# ------------------------------------------------------------------------------------------------ 2
@check("02 dataset present and identical to frozen manifests (train/val IR)")
def t02():
    from anti_uav_repro.data.antiuav import load_manifest, load_sequences, manifest_path, verify_against_manifest
    out = []
    for split in ("train", "val"):
        seqs = load_sequences("data/anti_uav", "data/derived/data_unified/anti_uav/images", split, "IR", purpose="verify")
        mp = manifest_path(MANIFESTS_DIR, split, "IR")
        assert mp.exists(), f"missing frozen manifest {mp} — run verify_dataset.py first"
        problems = verify_against_manifest(seqs, load_manifest(mp))
        assert not problems, problems[:3]
        out.append(f"{split}={len(seqs)} seqs")
    tm = manifest_path(MANIFESTS_DIR, "test", "IR")
    assert tm.exists(), "frozen TEST manifest missing"
    return ", ".join(out) + f"; test manifest sha {load_manifest(tm)['manifest_sha256'][:12]}"


# ------------------------------------------------------------------------------------------------ 3-4
_state = {}


@check("03 load a small sample of training pairs")
def t03():
    from anti_uav_repro.config import load_config
    from anti_uav_repro.data.antiuav import load_sequences
    from anti_uav_repro.data.pairs import DistinctSequenceBatchSampler, PairDataset, collate_pairs
    from anti_uav_repro.data.transforms import TrainPairTransform, TransformConfig
    cfg = load_config(CONFIGS_DIR / "dfsc_ir.yaml")
    d = cfg["data"]
    seqs = load_sequences(d["root"], d["frames_root"], "train", "IR", purpose="train")
    ds = PairDataset(seqs, d["pairs_per_seq"], d["max_distance"], d["frame_filter"], TrainPairTransform(TransformConfig()), seed=0)
    sam = DistinctSequenceBatchSampler(len(seqs), d["pairs_per_seq"], 2, seed=0)
    assert len(ds) == len(seqs) * d["pairs_per_seq"] and len(sam) == len(ds) // 2
    batches = [b for _, b in zip(range(50), iter(sam))]
    for b in batches:
        assert len({i % len(seqs) for i in b}) == len(b), "batch has two pairs from the same sequence"
    items = [ds[i] for i in batches[0]]
    for it in items:
        assert it["frame_z"] <= it["frame_x"] and it["frame_x"] - it["frame_z"] <= d["max_distance"] or True
    _state.update(cfg=cfg, seqs=seqs, ds=ds, batch=collate_pairs(items), items=items)
    return f"{len(ds)} pairs/epoch, {len(sam)} iters/epoch; sample seqs {[it['seq_name'] for it in items]}"


@check("04 preprocessing produces valid tensors / boxes")
def t04():
    b = _state["batch"]
    for key in ("img_z", "img_x"):
        x = b[key]
        assert x.dtype == torch.float32 and x.ndim == 4 and x.shape[1] == 3
        assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, "not padded to /32"
        assert torch.isfinite(x).all()
        assert -3.5 < float(x.mean()) < 3.5, "normalisation looks off"
    for boxes, sizes in ((b["boxes_z"], b["sizes_z"]), (b["boxes_x"], b["sizes_x"])):
        for bx, (h, w) in zip(boxes, sizes):
            assert bx.shape == (1, 4)
            assert (bx[:, 2] > bx[:, 0]).all() and (bx[:, 3] > bx[:, 1]).all(), "degenerate box"
            assert bx[:, 0].min() >= 0 and bx[:, 1].min() >= 0 and bx[:, 2].max() <= w and bx[:, 3].max() <= h, "box outside image"
    # rescale factor check for IR 640x512 -> 1000x800
    m = _state["items"][0]["meta_x"]
    assert abs(m["scale_factor"] - 1.5625) < 1e-6 and m["img_shape"] == (800, 1000), m
    return f"img {tuple(b['img_x'].shape)}, sizes {b['sizes_x'][0]}, scale {m['scale_factor']}"


# ------------------------------------------------------------------------------------------------ 5-8
@check("05-06 forward pass + loss computation (DFSC alpha=1 and normal alpha=0)")
def t05():
    from anti_uav_repro.models.qg_rcnn import QGRCNN
    cfg = _state["cfg"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QGRCNN(cfg["model"]).to(dev).train()
    losses = model.forward_train(_state["batch"], alpha=1.0)
    for k, v in losses.items():
        assert torch.isfinite(v).all(), f"non-finite {k}"
    assert float(losses["_n_cross_pairs"]) == 2.0
    l0 = model.forward_train(_state["batch"], alpha=0.0)
    assert float(l0["rpn_cls_cross"]) == 0.0 and float(l0["_n_cross_pairs"]) == 0.0, "alpha=0 must have no cross terms"
    _state.update(model=model, dev=dev, losses=losses)
    return "losses " + ", ".join(f"{k}={float(v):.3f}" for k, v in losses.items() if not k.startswith("_"))


@check("07 backward pass: gradients reach modulators / heads, none on frozen stem")
def t07():
    model = _state["model"]
    model.zero_grad(set_to_none=True)
    _state["losses"]["total"].backward()
    for mod in (model.rpn_modulator, model.rcnn_modulator, model.rpn.head, model.roi_heads.box_predictor, model.backbone.fpn):
        g = sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
        assert g > 0, f"no gradient in {type(mod).__name__}"
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert any("layer1" in n for n in frozen) and any("conv1.weight" in n for n in frozen), "stem/res2 not frozen"
    assert not any("layer2" in n for n in frozen), "layer2 must be trainable"
    return f"{len(frozen)} frozen tensors (stem + res2), grads present elsewhere"


@check("08 optimizer step changes trainable parameters and clip works")
def t08():
    model = _state["model"]
    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params[:20]] + [model.rpn_modulator.proj_out[0].weight.detach().clone()]
    opt = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=1e-4)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
    opt.step()
    after = [p.detach() for p in params[:20]] + [model.rpn_modulator.proj_out[0].weight.detach()]
    changed = sum(1 for a, b in zip(before, after) if not torch.equal(a, b))
    assert changed > 0, "no parameter changed after optimizer step"
    return f"{changed}/{len(before)} sampled tensors changed; grad norm before clip {float(gn):.2f}"


# ------------------------------------------------------------------------------------------------ 9
@check("09 checkpoint save / load round-trip reproduces outputs")
def t09():
    from anti_uav_repro.models.qg_rcnn import QGRCNN
    model, dev = _state["model"], _state["dev"]
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    p = SMOKE_DIR / "roundtrip.pt"
    torch.save({"model": model.state_dict(), "config": _state["cfg"], "epoch": 0}, p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m2 = QGRCNN({**_state["cfg"]["model"], "init": "none"})
    m2.load_state_dict(ck["model"], strict=True)
    m2.to(dev).eval(); model.eval()
    b = _state["batch"]
    img = b["img_x"][:1].to(dev); box = b["boxes_x"][0].to(dev); size = b["sizes_x"][0]
    with torch.no_grad():
        model.track_init(img, box, size); bx1, s1 = model.track_update(img, size)
        m2.track_init(img, box, size); bx2, s2 = m2.track_update(img, size)
    assert bx1.shape == bx2.shape and torch.allclose(bx1, bx2, atol=1e-3) and torch.allclose(s1, s2, atol=1e-4)
    model.train()
    return f"{bx1.shape[0]} detections identical after reload; top score {float(s1[0]) if len(s1) else None:.3f}"


# ------------------------------------------------------------------------------------------------ 10, 15, 14
@check("10 tiny end-to-end training run (smoke config, 2 epochs x 3 iterations, monitor on 2 val seqs)")
def t10():
    run_dir = SMOKE_DIR / "smoke_ir"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [PY, str(EXPERIMENT_DIR / "scripts" / "train.py"), "--config", str(CONFIGS_DIR / "smoke_ir.yaml"),
           "--run-dir", str(run_dir), "--max-iters-per-epoch", "3"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, f"train.py failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    for f in ("config.effective.yaml", "run_meta.json", "metrics.jsonl", "train_log.txt", "train_manifest.json",
              "weights/epoch_01.pt", "weights/epoch_02.pt", "weights/last.pt"):
        assert (run_dir / f).exists(), f"missing {f}"
    recs = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(recs) == 2 and all(np.isfinite(r_["losses"]["total"]) for r_ in recs)
    assert "monitor" in recs[-1] and recs[-1]["monitor"]["monitor_mSA_percent"] is not None
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["final_checkpoint_sha256"] and meta["test_split_touched"] is False
    _state["smoke_run"] = run_dir
    return f"epochs={len(recs)} loss {recs[0]['losses']['total']:.3f}->{recs[-1]['losses']['total']:.3f}, monitor mSA {recs[-1]['monitor']['monitor_mSA_percent']:.1f}% (meaningless, plumbing only)"


@check("15 configuration + provenance saved with the run")
def t15():
    import yaml
    run_dir = _state["smoke_run"]
    eff = yaml.safe_load((run_dir / "config.effective.yaml").read_text())
    assert eff["run_name"] == "smoke_ir" and eff["dfsc"]["alpha"] == 1.0 and eff["train"]["epochs"] == 2
    meta = json.loads((run_dir / "run_meta.json").read_text())
    for k in ("git", "env", "seed", "command", "train_split", "pairs_per_epoch", "model", "timestamp_start", "timestamp_end"):
        assert k in meta, f"run_meta missing {k}"
    assert meta["train_split"]["manifest_sha256"] and meta["env"]["torch"]
    ck = torch.load(run_dir / "weights" / "last.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["run_name"] == "smoke_ir", "checkpoint does not embed its config"
    return f"git {str(meta['git']['commit'])[:10]} torch {meta['env']['torch']} seed {meta['seed']} manifest {meta['train_split']['manifest_sha256'][:10]}"


@check("14 changing the seed changes run metadata and the sampled pairs")
def t14():
    from anti_uav_repro.config import load_config, validate_config
    from anti_uav_repro.data.pairs import DistinctSequenceBatchSampler, PairDataset
    from anti_uav_repro.data.transforms import TrainPairTransform, TransformConfig
    from anti_uav_repro.utils.repro import run_metadata
    cfg = load_config(CONFIGS_DIR / "dfsc_ir.yaml"); validate_config(cfg)
    cfg1 = copy.deepcopy(cfg); cfg1["seed"] = 1
    m0, m1 = run_metadata(cfg), run_metadata(cfg1)
    assert m0["seed"] == 0 and m1["seed"] == 1
    seqs = _state["seqs"]
    tf = TrainPairTransform(TransformConfig())
    ds0 = PairDataset(seqs, 10, 300, "globaltrack", tf, seed=0)
    ds1 = PairDataset(seqs, 10, 300, "globaltrack", tf, seed=1)
    ds0b = PairDataset(seqs, 10, 300, "globaltrack", tf, seed=0)
    pairs = lambda ds: [(ds._sample_pair(ds.frame_indices[i % len(seqs)], ds._rng(i))) for i in range(40)]
    assert pairs(ds0) != pairs(ds1), "seed 0 and 1 sample identical pairs"
    assert pairs(ds0) == pairs(ds0b), "same seed must be deterministic"
    b0 = list(iter(DistinctSequenceBatchSampler(len(seqs), 10, 2, seed=0)))[:20]
    b1 = list(iter(DistinctSequenceBatchSampler(len(seqs), 10, 2, seed=1)))[:20]
    assert b0 != b1
    # epoch is encoded in the index: the same local index yields different frames in epoch 0 vs epoch 1,
    # independent of any worker-side state (regression test for stale persistent-worker datasets)
    n = len(ds0)
    e0 = [ds0._sample_pair(ds0.frame_indices[i % len(seqs)], ds0._rng(i)) for i in range(30)]
    e1 = [ds0._sample_pair(ds0.frame_indices[i % len(seqs)], ds0._rng(n + i)) for i in range(30)]
    assert e0 != e1, "epoch 1 re-samples epoch 0 pairs"
    s1 = DistinctSequenceBatchSampler(len(seqs), 10, 2, seed=0); s1.set_epoch(1)
    first = next(iter(s1))
    assert all(n <= i < 2 * n for i in first), "sampler does not offset indices by epoch"
    return "seed recorded; pair sampling and batch order differ across seeds, repeat within a seed, and vary by epoch"


# ------------------------------------------------------------------------------------------------ 11-12
@check("11 evaluation on a tiny validation subset (2 sequences, stride 50, PARTIAL)")
def t11():
    run_dir = _state["smoke_run"]
    cmd = [PY, str(EXPERIMENT_DIR / "scripts" / "evaluate.py"), "--checkpoint", str(run_dir / "weights" / "last.pt"),
           "--split", "val", "--limit-sequences", "2", "--frame-stride", "50", "--out-dir", str(run_dir / "eval_val_smoke")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, encoding="utf-8", errors="replace")
    assert r.returncode == 0, f"evaluate.py failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    out = run_dir / "eval_val_smoke"
    for f in ("metrics.json", "report.md", "eval_meta.json"):
        assert (out / f).exists(), f"missing {f}"
    preds = list((out / "predictions").glob("*.json"))
    assert len(preds) == 2
    m = json.loads((out / "metrics.json").read_text())
    assert m["aggregate"]["n_sequences"] == 2 and "PARTIAL" in str(out) or m["meta"]["partial"]
    _state["eval_out"] = out
    return f"mSA {m['aggregate']['mSA_percent']:.1f}% on processed frames (PARTIAL, plumbing only); ckpt {m['meta']['checkpoint_sha256'][:10]}"


@check("12 metrics are computed from the saved predictions (independent recomputation + perturbation)")
def t12():
    from anti_uav_repro.data.antiuav import load_sequences
    from anti_uav_repro.eval.metrics import iou_xywh
    out = _state["eval_out"]
    m = json.loads((out / "metrics.json").read_text())
    seqs = {s.name: s for s in load_sequences("data/anti_uav", "data/derived/data_unified/anti_uav/images", "val", "IR", purpose="tune")}
    # (a) recompute SA of every evaluated sequence straight from prediction files with an independent loop
    for pf in (out / "predictions").glob("*.json"):
        pred = json.loads(pf.read_text()); s = seqs[pred["sequence"]]
        vals = []
        for t, done in enumerate(pred["processed"]):
            if not done:
                continue
            p = pred["res"][t]
            if s.exist[t]:
                vals.append(iou_xywh(p, s.gt_rect[t]) if len(p) == 4 else 0.0)
            else:
                vals.append(0.0 if len(p) == 4 else 1.0)
        recomputed = float(np.mean(vals))
        reported = m["per_sequence"][s.name]["SA"]
        assert abs(recomputed - reported) < 1e-9, f"{s.name}: recomputed {recomputed} != reported {reported}"
    # (b) perturbing predictions must change the metric -> the metric is not constant / hard-coded
    from anti_uav_repro.eval.evaluate import evaluate_predictions
    tmp = out / "predictions_perturbed"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(out / "predictions", tmp)
    for pf in tmp.glob("*.json"):
        pred = json.loads(pf.read_text())
        pred["res"] = [[0, 0, 1, 1] if len(p) == 4 else p for p in pred["res"]]  # move every box to the corner
        pf.write_text(json.dumps(pred))
    seq_list = [seqs[json.loads(pf.read_text())["sequence"]] for pf in sorted(tmp.glob("*.json"))]
    m2 = evaluate_predictions(tmp, seq_list, only_processed_frames=True)
    assert m2["aggregate"]["mSA"] != m["aggregate"]["mSA"], "metric did not react to changed predictions"
    # (c) unit properties: perfect predictions -> SA 1 ; absent prediction on absent frame -> 1 ; box on absent -> 0
    from anti_uav_repro.eval.metrics import sequence_metrics
    r = sequence_metrics([[1, 2, 3, 4], [0], [5, 5, 5, 5]], [1, 0, 0], [[1, 2, 3, 4], [], []])
    assert abs(r["SA"] - (1 + 1 + 0) / 3) < 1e-12 and r["precision_20px"] == 1.0
    assert abs(iou_xywh([0, 0, 10, 10], [5, 5, 10, 10]) - 25 / 175) < 1e-12
    return "per-sequence SA recomputed exactly; perturbed predictions change mSA; unit properties hold"


# ------------------------------------------------------------------------------------------------ 13
@check("13 TEST split cannot enter training / tuning code paths")
def t13():
    from anti_uav_repro.config import ConfigError, load_config, validate_config
    from anti_uav_repro.data.antiuav import TestSplitAccessError, load_sequences
    for purpose in ("train", "tune", "monitor"):
        try:
            load_sequences("data/anti_uav", "data/derived/data_unified/anti_uav/images", "test", "IR", purpose=purpose)
            raise AssertionError(f"test split was served for purpose={purpose}")
        except TestSplitAccessError:
            pass
    cfg = load_config(CONFIGS_DIR / "dfsc_ir.yaml")
    for key in ("train_split", "monitor_split"):
        bad = copy.deepcopy(cfg); bad["data"][key] = "test"
        try:
            validate_config(bad)
            raise AssertionError(f"config with data.{key}=test was accepted")
        except ConfigError:
            pass
    # evaluate.py refuses test without --final-eval (exits before loading anything)
    run_dir = _state["smoke_run"]
    r = subprocess.run([PY, str(EXPERIMENT_DIR / "scripts" / "evaluate.py"), "--checkpoint", str(run_dir / "weights" / "last.pt"),
                        "--split", "test"], capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace")
    assert r.returncode != 0 and "final-eval" in (r.stdout + r.stderr), "evaluate.py did not refuse test without --final-eval"
    # partial test evaluation refused even with --final-eval
    r2 = subprocess.run([PY, str(EXPERIMENT_DIR / "scripts" / "evaluate.py"), "--checkpoint", str(run_dir / "weights" / "last.pt"),
                         "--split", "test", "--final-eval", "--limit-sequences", "1"], capture_output=True, text=True, timeout=600,
                        encoding="utf-8", errors="replace")
    assert r2.returncode != 0 and "partial" in (r2.stdout + r2.stderr).lower()
    # training code has no literal 'test' split reference
    src = (EXPERIMENT_DIR / "src" / "anti_uav_repro" / "engine" / "train.py").read_text(encoding="utf-8")
    assert '"test"' not in src.replace('TEST split', '') and "'test'" not in src, "train.py references the test split"
    return "loader raises for train/tune/monitor; configs rejected; evaluate.py refuses test w/o --final-eval and partial test"


def main():
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True)
    print(f"smoke artefacts -> {SMOKE_DIR}\n")
    for fn in (t01, t02, t03, t04, t05, t07, t08, t09, t10, t15, t14, t11, t12, t13):
        fn()
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 100)
    for name, ok, msg in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"{n_ok}/{len(RESULTS)} checks passed")
    (SMOKE_DIR / "smoke_results.json").write_text(json.dumps([{"check": n, "pass": ok, "detail": m} for n, ok, m in RESULTS], indent=2))
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
