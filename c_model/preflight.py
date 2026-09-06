#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preflight.py  --  the checks that run before every AeroTrack-Net training launch
================================================================================
A 20-hour run that produces an unloadable checkpoint, or a mAP computed on
frames the model was trained on, is a 20-hour loss. These are the eight tests
from 00_TRAINING_ROADMAP.md §7.1 plus the Blackwell environment gate from §1.1.
They take about a minute and they run every time.

  E0  environment  CUDA present, sm_120 in the arch list, a real kernel launches
  T1  dataset      batch is exactly [B,9,640,640] and pixel values are in range
  T2  boundary     frame_idx 0 -> channels[0:3] == channels[3:6] (t-1 duplicated)
  T3  labels       every box coordinate is inside [0,1]
  T4  LEAKAGE      train seq_keys n val seq_keys == 0        <- the important one
  T5  collate      the loss batch dict is well-formed; batch_idx < B
  T6  checkpoint   save -> load -> bit-identical forward on a fixed input
  T7  throughput   samples/s through the real loader
  T8  vram         peak allocation at the chosen batch, with headroom

E0 deserves its own paragraph. The RTX 5050 is Blackwell, compute capability
sm_120. A PyTorch wheel built before Blackwell support installs cleanly and
reports `cuda.is_available() == True`, then dies at the first kernel launch with
"no kernel image is available for execution on the device". `is_available()` is
NOT the test; `sm_120 in torch.cuda.get_arch_list()` is.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


class Result:
    def __init__(self, name, ok, detail=""):
        self.name, self.ok, self.detail = name, bool(ok), detail

    def __repr__(self):
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def _try(name, fn):
    try:
        ok, detail = fn()
        return Result(name, ok, detail)
    except Exception as e:
        return Result(name, False, f"{e!r}\n{traceback.format_exc(limit=3)}")


# --------------------------------------------------------------------------- #
def check_environment(require_sm120=True):
    """E0 — the Blackwell gate."""
    def _f():
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is False — this is a CPU wheel"
        arch = torch.cuda.get_arch_list()
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        want = f"sm_{cap[0]}{cap[1]}"
        has_arch = want in arch
        # a real kernel launch — the only proof that matters
        x = torch.randn(1024, 1024, device="cuda")
        finite = bool((x @ x).isfinite().all())
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        detail = (f"{name} cap={cap} ({want}) | torch {torch.__version__} "
                  f"cuda {torch.version.cuda} | arch_list={arch} | "
                  f"matmul finite={finite} | VRAM {total/1024**3:.1f} GB "
                  f"({free/1024**3:.1f} free) | bf16={torch.cuda.is_bf16_supported()}")
        ok = finite and (has_arch or not require_sm120)
        if not has_arch:
            detail += (f"\n     !! {want} is NOT in the wheel's arch list. Reinstall: "
                       f"pip install torch torchvision "
                       f"--index-url https://download.pytorch.org/whl/cu130")
        return ok, detail
    return _try("E0 environment", _f)


def check_dataset(dataset, batch_size=4, expect_channels=9, imgsz=640):
    """T1 + T3 — tensor contract and label integrity, straight off the dataset."""
    def _f():
        from torch.utils.data import DataLoader
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "b_pipeline"))
        from stacked_dataset import aerotrack_collate
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=aerotrack_collate)
        imgs, targets = next(iter(dl))[:2]
        shape_ok = tuple(imgs.shape) == (batch_size, expect_channels, imgsz, imgsz)
        if imgs.dtype == torch.uint8:
            rng_ok, rng = True, "uint8 0..255"
        else:
            lo, hi = float(imgs.min()), float(imgs.max())
            rng_ok = 0.0 <= lo and hi <= 1.0
            rng = f"[{lo:.3f}, {hi:.3f}]"
        bad = 0
        for t in targets:
            if t.numel():
                v = t[:, 1:5]
                bad += int(((v < 0) | (v > 1)).any(1).sum())
        return (shape_ok and rng_ok and bad == 0,
                f"shape={tuple(imgs.shape)} dtype={imgs.dtype} range={rng} "
                f"out-of-range boxes in first batch={bad}")
    return _try("T1/T3 dataset+labels", _f)


def check_boundary(dataset):
    """T2 — temporal boundary duplication is intact after letterbox/augment-off."""
    def _f():
        idx = dataset.index
        first = np.flatnonzero(idx.frame_idx == 0)
        last = np.flatnonzero((idx.frame_idx == idx.seq_len - 1) & (idx.seq_len > 1))
        msgs = []
        ok = True
        if len(first):
            x = dataset[int(first[0])][0]
            e = torch.equal(x[0:3], x[3:6])
            ok &= e
            msgs.append(f"t=0: ch[0:3]==ch[3:6] -> {e}")
        if len(last):
            x = dataset[int(last[0])][0]
            e = torch.equal(x[6:9], x[3:6])
            ok &= e
            msgs.append(f"t=N-1: ch[6:9]==ch[3:6] -> {e}")
        if not msgs:
            return True, "no boundary frames in this sample set (skipped)"
        return ok, " | ".join(msgs)
    return _try("T2 temporal boundary", _f)


def check_leakage(train_samples, val_samples, test_samples=None):
    """T4 — THE test. Sequence-level disjointness of the splits."""
    def _keys(x):
        """Accept either a list of sample dicts or a ready-made set of keys.

        Callers should prefer the set: holding 378k sample dicts alive purely so
        this check can recompute their keys is hundreds of MB for nothing."""
        if x is None:
            return None
        if isinstance(x, (set, frozenset)):
            return set(x)
        return {s["seq_key"] for s in x}

    def _f():
        a = _keys(train_samples)
        b = _keys(val_samples)
        pairs = [("train n val", a & b)]
        c = _keys(test_samples)
        if c is not None:
            pairs += [("train n test", a & c), ("val n test", b & c)]
        bad = {k: sorted(v)[:3] for k, v in pairs if v}
        return (not bad,
                f"train={len(a):,} seqs / val={len(b):,} seqs"
                + (f" | OVERLAP {bad}" if bad else " | disjoint"))
    return _try("T4 split leakage", _f)


def check_collate(trainer, dataset, batch_size=4):
    """T5 — the dict handed to v8DetectionLoss is well-formed."""
    def _f():
        from torch.utils.data import DataLoader
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "b_pipeline"))
        from stacked_dataset import aerotrack_collate
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=aerotrack_collate)
        imgs, targets = next(iter(dl))[:2]
        b = trainer._to_batch(imgs, targets)
        keys = {"img", "cls", "bboxes", "batch_idx"}
        have = keys.issubset(b)
        bi = b["batch_idx"]
        bi_ok = bi.numel() == 0 or int(bi.max()) < batch_size
        # Ordering invariant: v8DetectionLoss.preprocess derives each box's slot
        # from `arange(n) - offsets[batch_idx]`, which is only valid when every
        # image's rows are contiguous and in ascending order. Violating it does
        # not raise — it fires a CUDA device-side assert that kills the context.
        sorted_ok = bi.numel() == 0 or bool((bi[1:] >= bi[:-1]).all())
        n_ok = b["cls"].shape[0] == b["bboxes"].shape[0] == bi.shape[0]
        cls_ok = bi.numel() == 0 or (0 <= float(b["cls"].min()) and
                                     int(b["cls"].max()) < 2)
        return (have and bi_ok and sorted_ok and n_ok and cls_ok,
                f"keys={sorted(b)} img={tuple(b['img'].shape)} "
                f"boxes={tuple(b['bboxes'].shape)} "
                f"batch_idx max={'-' if bi.numel()==0 else int(bi.max())} < {batch_size} "
                f"| non-decreasing={sorted_ok} | class ids in range={cls_ok}")
    return _try("T5 collate -> loss dict", _f)


def check_checkpoint_roundtrip(trainer, device, tmp_dir=None, imgsz=640, ch=9,
                               score_tol=5e-2, box_tol_px=None):
    """T6 — save, reload, and prove the reloaded network IS the saved one.

    This is where the `ch=9` reload hazard would surface: if the round trip
    rebuilt the model from its YAML, `ch` would silently revert to 3 and the
    load would either explode or hand back a different network. A 20-hour run
    that produces an unloadable checkpoint is a 20-hour loss, so this runs
    before the run, not after it.

    Three separate claims are checked, because one loose numeric tolerance can
    hide all of them:

      1. WEIGHTS.  Every tensor equals the original cast to fp16 -- exactly.
         Checkpoints are stored half (Ultralytics convention, half the file), so
         fp16 rounding is expected; anything else is not. This is the real test:
         it is exact, and it cannot be passed by accident.
      2. GEOMETRY. The reloaded stem still takes 9 input channels.
      3. OUTPUT.   Class scores (already in [0,1]) match within `score_tol`;
         box channels are in PIXELS on a 640 canvas, so they get their own
         tolerance, derived from the coarsest stride. Comparing both against one
         absolute epsilon is how a healthy round trip gets reported as a failure
         -- a sub-pixel box shift and a catastrophic score shift look identical
         on that scale.

    On the box tolerance: the head regresses boxes as a DFL distribution over
    `reg_max` bins, one bin per stride unit, and takes its expectation. In an
    UNTRAINED network that distribution is nearly uniform, so fp16 rounding of
    the logits moves the expectation by a meaningful fraction of a bin -- and at
    P5/32 a bin is 32 px. Several pixels of movement is therefore the expected,
    healthy result here, not a defect; it shrinks once the distributions sharpen
    with training. Default tolerance is a quarter of the coarsest stride. The
    claim that actually proves the round trip is (1), and (1) is exact.
    """
    def _f():
        from aerotrack_trainer import load_checkpoint
        d = Path(tmp_dir or (trainer.save_dir / "weights"))
        d.mkdir(parents=True, exist_ok=True)
        p = d / "_preflight_roundtrip.pt"
        trainer.save_checkpoint(p)

        m0 = (trainer.ema.ema if trainer.ema else trainer.model)
        was_training = m0.training
        m0.eval()
        m1, ck = load_checkpoint(p, device, prefer_ema=trainer.ema is not None)

        # 1. weights, exactly
        sd0, sd1 = m0.state_dict(), m1.state_dict()
        missing = set(sd0) ^ set(sd1)
        bad = []
        for k, v in sd0.items():
            if k not in sd1:
                continue
            ref = v.detach().half().float() if v.is_floating_point() else v.detach()
            got = sd1[k].float() if sd1[k].is_floating_point() else sd1[k]
            if ref.shape != got.shape or not torch.equal(ref.to(got.device), got):
                bad.append(k)
        # 2. geometry: the first SPDConv folds each 2x2 into channels, so a
        #    9-channel input reaches its convolution as 36 channels. If the
        #    reload had rebuilt from YAML with the default ch=3, this reads 12.
        first_conv = next(m for m in m1.modules() if isinstance(m, torch.nn.Conv2d))
        ch_ok = first_conv.in_channels == 4 * ch

        # 3. forward
        x = torch.randn(1, ch, imgsz, imgsz, device=device)
        with torch.no_grad():
            y0 = m0(x)
            y1 = m1(x)
        a = (y0[0] if isinstance(y0, (list, tuple)) else y0).float()
        b = (y1[0] if isinstance(y1, (list, tuple)) else y1).float()
        shape_ok = a.shape == b.shape
        nc = getattr(m1.model[-1], "nc", 2)
        stride_max = float(getattr(m1, "stride", torch.tensor([32.0])).max())
        tol_px = box_tol_px if box_tol_px is not None else 0.25 * stride_max
        d_box = float((a[:, :4] - b[:, :4]).abs().max()) if shape_ok else float("inf")
        d_cls = float((a[:, 4:4 + nc] - b[:, 4:4 + nc]).abs().max()) if shape_ok else float("inf")

        p.unlink(missing_ok=True)
        if was_training:
            m0.train()
        ok = (not bad and not missing and ch_ok and shape_ok
              and d_cls <= score_tol and d_box <= tol_px)
        return (ok,
                f"weights: {len(sd0)} tensors, {len(bad)} mismatched, "
                f"{len(missing)} missing | first conv in_channels="
                f"{first_conv.in_channels} (expect {4*ch}) | out {tuple(a.shape)} "
                f"| max|dscore|={d_cls:.2e} (tol {score_tol}) "
                f"max|dbox|={d_box:.3f} px (tol {tol_px:.1f} = stride/4) "
                f"[weights are exact; the box delta is fp16 DFL sensitivity]")
    return _try("T6 checkpoint round-trip", _f)


def _cycled(loader, limit):
    """Yield up to `limit` batches, restarting the loader when it runs dry.

    S0's corpus is sixteen images -- two batches -- so a probe that wants 23 of
    them hits StopIteration and reports a hard FAIL for a loader that is
    perfectly healthy. Restart instead, and let the caller say so in its
    detail line."""
    n = 0
    passes = 0
    while n < limit:
        got = 0
        for b in loader:
            yield b
            got += 1
            n += 1
            if n >= limit:
                return
        passes += 1
        if got == 0 or passes > limit:      # an empty loader cannot be cycled
            return


def check_throughput(loader, n_batches=20, warmup=3):
    """T7 — is the loader fast enough to keep the GPU fed?"""
    def _f():
        n_have = len(loader)
        it = _cycled(loader, warmup + n_batches)
        for _ in range(warmup):
            next(it, None)
        t0 = time.time(); n = 0; nb = 0
        for b in it:
            n += b[0].shape[0]
            nb += 1
        dt = max(time.time() - t0, 1e-9)
        sps = n / dt
        short = ("  [loader holds only %d batches, so this re-reads them and the "
                 "OS page cache flatters the number]" % n_have
                 if n_have < warmup + n_batches else "")
        return (sps > 0,
                f"{sps:,.1f} samples/s ({sps*3:,.0f} frame reads/s) over {n} samples "
                f"— an epoch of 25,000 draws would take {25000/sps/60:.1f} min{short}")
    return _try("T7 loader throughput", _f)


def check_vram(trainer, dataset, batch_size, device, iters=3):
    """T8 — measure peak VRAM at the chosen batch and report the headroom."""
    def _f():
        if device.type != "cuda":
            return True, "cpu — skipped"
        from torch.utils.data import DataLoader
        from aerotrack_trainer import compute_loss
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "b_pipeline"))
        from stacked_dataset import aerotrack_collate
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=aerotrack_collate)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        trainer.model.train()
        it = _cycled(dl, iters)
        with _bn_eval(trainer.model):
            for batch in it:
                imgs, targets = batch[:2]
                b = trainer._to_batch(imgs, targets)
                if trainer.amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=trainer.amp_dtype):
                        loss, _ = compute_loss(trainer.criterion, trainer.model(b["img"]), b)
                else:
                    loss, _ = compute_loss(trainer.criterion, trainer.model(b["img"]), b)
                loss.backward()
                trainer.optimizer.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        free_b, _ = torch.cuda.mem_get_info()
        head = free_b / 1024 ** 3 + torch.cuda.memory_reserved() / 1024 ** 3 - peak
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        return (head >= 1.5,
                f"peak {peak:.2f} GB at batch {batch_size} on a {total:.1f} GB card "
                f"-> {head:.2f} GB headroom against what is actually free "
                f"(want >= 1.5)")
    return _try("T8 vram headroom", _f)


class _bn_eval:
    """Freeze BatchNorm running statistics for the duration of a probe.

    The VRAM and batch-size probes push RANDOM tensors through the network in
    train mode. Every such forward pass updates BatchNorm's running_mean and
    running_var -- so a "harmless" memory measurement would quietly poison the
    statistics of a freshly COCO-transferred backbone before epoch 0. Activation
    memory is identical either way, so there is no reason to pay that cost."""

    def __init__(self, model):
        self.model = model
        self.saved = []

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                self.saved.append((m, m.training))
                m.eval()
        return self

    def __exit__(self, *exc):
        for m, was in self.saved:
            m.train(was)
        return False


def probe_batch_size(model, criterion, device, imgsz=640, ch=9,
                     candidates=(4, 8, 12, 16, 24, 32), headroom_gb=1.5,
                     amp_dtype=None, boxes_per_img=8, verbose=True):
    """Find the largest batch that still leaves `headroom_gb` of VRAM.

    Ultralytics' autobatch measures a 3-channel model and gets this wrong here:
    SPD-Conv is *more* activation-hungry than strided conv in the early layers
    (the first block holds B x 36 x 320 x 320), and the input tensor is 3x
    fatter than an RGB one. Measure the real thing instead of estimating it.

    A synthetic forward+backward is used, so this costs seconds and does not
    touch the data pipeline.

    `boxes_per_img` defaults to 8, which is deliberately pessimistic: this
    corpus averages 0.95 boxes per image and its p99 is 1, but the tail runs to
    51 (a Det-Fly frame full of birds). The assigner allocates tensors shaped
    (batch, anchors, max_boxes_in_batch), so a single such frame is what decides
    the peak. Sizing the batch off the mean would pick a number that OOMs the
    first time one of those 282 frames is drawn.

    T8 (`check_vram`) then re-measures on REAL batches at the chosen size; that
    is the authoritative number, and this is what keeps it from being an OOM."""
    if device.type != "cuda":
        return candidates[0], {}
    model = model.to(device).train()
    torch.cuda.empty_cache()
    # Budget against FREE memory, not total. On a laptop the desktop compositor,
    # the browser and NVIDIA's own overlay are already holding 1-2 GB of the 8;
    # sizing the batch against `total_memory` hands back a number that OOMs the
    # moment someone opens a window.
    free_b, total_b = torch.cuda.mem_get_info()
    total = total_b / 1024 ** 3
    budget = free_b / 1024 ** 3 + torch.cuda.memory_reserved() / 1024 ** 3
    best, report = candidates[0], {}
    bn_guard = _bn_eval(model).__enter__()
    for b in candidates:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            img = torch.rand(b, ch, imgsz, imgsz, device=device)
            n = b * boxes_per_img
            # batch_idx MUST be sorted and contiguous per image. Ultralytics'
            # vectorised `preprocess` computes each box's within-image slot as
            # `arange(nl) - offsets[batch_idx]`, which only lands in range if all
            # of an image's rows sit together in ascending order. Interleaving
            # them (0,1,2,3,0,1,2,3) produces negative indices and a device-side
            # assert, not a Python exception — and it poisons the CUDA context.
            batch = {
                "img": img,
                "cls": torch.zeros(n, 1, device=device),
                "bboxes": torch.rand(n, 4, device=device) * 0.3 + 0.35,
                "batch_idx": torch.arange(n, device=device)
                                  .div(boxes_per_img, rounding_mode="floor").float(),
            }
            from aerotrack_trainer import compute_loss
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss, _ = compute_loss(criterion, model(img), batch)
            else:
                loss, _ = compute_loss(criterion, model(img), batch)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            head = budget - peak
            report[b] = {"peak_gb": round(peak, 2), "headroom_gb": round(head, 2)}
            if verbose:
                print(f"  [batch probe] b={b:<3d} peak {peak:5.2f} GB  "
                      f"headroom {head:5.2f} GB "
                      f"{'OK' if head >= headroom_gb else 'too tight'}")
            if head >= headroom_gb:
                best = b
            else:
                break
        except torch.cuda.OutOfMemoryError:
            report[b] = {"peak_gb": None, "headroom_gb": None, "oom": True}
            if verbose:
                print(f"  [batch probe] b={b:<3d} OOM")
            break
        finally:
            torch.cuda.empty_cache()
    bn_guard.__exit__()
    torch.cuda.reset_peak_memory_stats()
    if verbose:
        print(f"  [batch probe] -> batch {best} "
              f"(GPU {total:.1f} GB total, {budget:.1f} GB usable now, "
              f"headroom target {headroom_gb} GB)")
    return best, report


def check_overfit(trainer, iters=200, target_loss=0.15, min_reduction=0.75,
                  verbose=True):
    """S0 — can the loss learn AT ALL on one fixed batch?

    The single most valuable test in the roadmap (§6.1). If a 4 M-param network
    cannot drive the loss toward zero on 16 fixed images, the bug is in the
    labels, the collate or the assigner — and no amount of full-scale training
    will find it for you. Two minutes.
    """
    def _f():
        it = iter(trainer.train_loader)
        b, n_boxes = None, 0
        for _ in range(12):                       # a batch of pure empties proves nothing
            try:
                imgs, targets = next(it)[:2]
            except StopIteration:
                break
            cand = trainer._to_batch(imgs, targets)
            if cand["cls"].shape[0] > n_boxes:
                b, n_boxes = cand, int(cand["cls"].shape[0])
            if n_boxes >= imgs.shape[0]:          # ~one target per image is plenty
                break
        if b is None or n_boxes == 0:
            return False, ("no batch in the first 12 draws carried a single "
                           "ground-truth box — this test would be vacuous. Check "
                           "the label pipeline before going further.")
        from aerotrack_trainer import compute_loss
        first = last = None
        trainer.model.train()
        for i in range(iters):
            if trainer.amp_dtype is not None:
                with torch.autocast(device_type=trainer.device.type,
                                    dtype=trainer.amp_dtype):
                    loss, items = compute_loss(trainer.criterion,
                                               trainer.model(b["img"]), b)
            else:
                loss, items = compute_loss(trainer.criterion,
                                           trainer.model(b["img"]), b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 10.0)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            v = float(items.sum())
            if first is None:
                first = v
            last = v
            if verbose and (i + 1) % max(iters // 8, 1) == 0:
                print(f"    [S0] iter {i+1:>4d}/{iters}  loss {v:.4f}")
        # `min_reduction` has to match `iters`, and conflating the two is a
        # real trap: the S0 criterion (75% off the loss) is correct after 200
        # optimiser steps and impossible after 3, so a caller that shortens the
        # probe without loosening the bar gets a red FAIL on a perfectly healthy
        # network. A short probe asks a smaller question -- "does the gradient
        # path move the loss at all" -- and should say so.
        drop = 1 - last / max(first, 1e-9)
        ok = drop >= min_reduction or last < target_loss
        return (ok,
                f"{n_boxes} GT boxes | loss {first:.4f} -> {last:.4f} "
                f"({drop*100:.1f}% reduction over {iters} iters, "
                f"need {min_reduction*100:.0f}% or < {target_loss})")
    return _try("S0 overfit-one-batch", _f)


# --------------------------------------------------------------------------- #
def run_all(results, raise_on_fail=True, allow_fail=()):
    """Print a report; raise unless every non-exempt check passed."""
    print("=" * 78)
    print("PRE-FLIGHT")
    print("=" * 78)
    failed = []
    for r in results:
        code = r.name.split()[0]
        if not r.ok and code in allow_fail:
            mark = "EXEMPT"
            print(f"  [{mark}] {r.name} (exempt: val is train batch by design)")
        else:
            mark = "PASS" if r.ok else "FAIL"
            print(f"  [{mark}] {r.name}")
        for line in str(r.detail).splitlines():
            print(f"         {line}")
        if not r.ok and code not in allow_fail:
            failed.append(r.name)
    print("=" * 78)
    if failed:
        msg = "PRE-FLIGHT FAILED: " + ", ".join(failed)
        print(msg)
        if raise_on_fail:
            raise SystemExit(msg)
    else:
        print("All pre-flight checks passed — safe to launch.")
    return not failed
