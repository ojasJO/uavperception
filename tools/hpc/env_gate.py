"""M4 environment acceptance gate for an HPC node (docs/MASTER_PLAN.md). Writes one JSON report; exit 0 only if every check passes.

    python tools/hpc/env_gate.py --part core --out <report.json>      # GPU / model / numerics / checkpoint (no data needed)
    python tools/hpc/env_gate.py --part data --out <report.json>      # corpus v2 through the path map (needs M3 data)

core: CUDA kernel (fp32 + fp16 matmul vs CPU reference), arch in wheel, both detector builds with their registered COCO-transfer
counts (as-built 418/649 = 417 + the tiled 9-channel stem, as the trainer logged on the OMEN; standard 448/499), fp16 autocast forward + backward +
GradScaler step with a finite loss, checkpoint save/load round trip (bit-equal state dict), synthetic train-step throughput.
data: manifest/splits sha256 equal to the corpus v2 identity recorded by the v2 runs, every image path of TRAIN/VAL resolves,
200 random frames decode at the expected size, label cache aligned with the manifest, sequence-level leakage assert.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "c_model", REPO / "b_pipeline", REPO / "experiments" / "aerotrack_v2" / "scripts"):
    sys.path.insert(0, str(p))

MANIFEST_SHA = "f842d03909cf2534130f5852e5c9d6df2e498e8d03532fc9bb1f7ce9309c9347"     # runs/v2_*/run_meta.json (OMEN)
SPLITS_SHA = "30f0e35f35ca6dfd4ddcf89fa9c9acc907c37f9a846a976ff2300573d1bf8a6e"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(results, name, fn):
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as e:                       # a crash is a failed check, never a skipped one
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append({"check": name, "ok": bool(ok), "detail": detail, "seconds": round(time.time() - t0, 2)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def core(results):
    import torch
    from aerotrack_trainer import build_model, transfer_pretrained

    def kernel():
        cap = torch.cuda.get_device_capability(0)
        arch_ok = f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list()
        g = torch.Generator().manual_seed(0)
        a, b = torch.randn(512, 512, generator=g), torch.randn(512, 512, generator=g)
        ref = a @ b
        e32 = ((a.cuda() @ b.cuda()).cpu() - ref).abs().max().item()
        e16 = ((a.cuda().half() @ b.cuda().half()).float().cpu() - ref).abs().max().item()
        ok = arch_ok and e32 < 1e-2 and e16 < 1.0
        return ok, {"gpu": torch.cuda.get_device_name(0), "cap": cap, "arch_in_wheel": arch_ok, "torch": torch.__version__,
                    "cuda_build": torch.version.cuda, "max_err_fp32": e32, "max_err_fp16": e16,
                    "bf16_reported": torch.cuda.is_bf16_supported()}
    check(results, "C1 cuda kernel", kernel)

    models = {}

    def builds():
        out, ok = {}, True
        for tag, yaml, want in (("as_built", "yolo11-stride.yaml", (418, 649)), ("standard", "yolo11-stride-std.yaml", (448, 499))):
            m = build_model(REPO / "c_model" / yaml, ch=9, nc=2)
            r = transfer_pretrained(m, REPO / "c_model" / "yolo11n.pt", verbose=False)
            n_par = sum(p.numel() for p in m.parameters())
            models[tag] = m
            got = tuple(r)[:2] if isinstance(r, (tuple, list)) else None
            ok &= got == want
            out[tag] = {"yaml": yaml, "params": n_par, "transfer": got, "expected_loaded": want}
        return ok, out
    check(results, "C2 model builds + COCO transfer", builds)

    def fp16_step():
        # GradScaler starts at 2**16 and halves on overflow, skipping that step. With random-noise input the loss is ~300, so
        # the first ~10-16 steps overflow (fp16_diag.py: unscaled fp16 gradients are all finite). Pass = every loss finite and
        # the weights actually change within 40 steps.
        from ultralytics.utils.loss import v8DetectionLoss
        m = models["as_built"].cuda().train()
        crit = v8DetectionLoss(m)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        x = torch.rand(8, 9, 640, 640, device="cuda")
        batch = {"batch_idx": torch.tensor([0., 1., 2.], device="cuda"), "cls": torch.tensor([[0.], [0.], [1.]], device="cuda"),
                 "bboxes": torch.tensor([[.5, .5, .05, .04], [.3, .6, .02, .02], [.7, .2, .1, .08]], device="cuda")}
        w0 = [p.detach().clone() for p in m.parameters()]
        losses, scales = [], []
        for _ in range(40):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, _ = crit(m(x), batch)
            loss = loss.sum()
            losses.append(float(loss))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
            scaler.step(opt); scaler.update()
            scales.append(scaler.get_scale())
        moved = sum(int(not torch.equal(a, b.detach())) for a, b in zip(w0, m.parameters()))
        finite = all(map(lambda v: v == v and abs(v) != float("inf"), losses))
        return finite and moved > 0 and scales[-1] >= 1.0, {"losses": [round(v, 3) for v in losses[::5]], "scales": scales[::5] + scales[-1:],
                                                             "tensors_updated": moved}
    check(results, "C3 fp16 forward/backward/GradScaler", fp16_step)

    def roundtrip():
        m = models["as_built"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ck.pt"
            tmp = p.with_suffix(".tmp")
            torch.save({"model": m.state_dict()}, tmp); os.replace(tmp, p)        # atomic write, as the trainer does
            sd = torch.load(p, map_location="cuda", weights_only=False)["model"]
        bad = [k for k, v in m.state_dict().items() if not torch.equal(v, sd[k])]
        return not bad, {"tensors": len(sd), "mismatched": bad[:5]}
    check(results, "C4 checkpoint save/load round trip", roundtrip)

    def throughput():
        from ultralytics.utils.loss import v8DetectionLoss
        out = {}
        for prec in ("fp16", "fp32"):
            m = models["as_built"].cuda().train()
            crit = v8DetectionLoss(m)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
            scaler = torch.amp.GradScaler("cuda", enabled=prec == "fp16")
            x = torch.rand(32, 9, 640, 640, device="cuda")
            batch = {"batch_idx": torch.arange(32., device="cuda"), "cls": torch.zeros(32, 1, device="cuda"),
                     "bboxes": torch.tensor([[.5, .5, .03, .03]], device="cuda").repeat(32, 1)}
            for i in range(25):
                if i == 5:
                    torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast("cuda", dtype=torch.float16, enabled=prec == "fp16"):
                    loss, _ = crit(m(x), batch)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss.sum()).backward(); scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            out[prec] = {"img_per_s_gpu_only": round(20 * 32 / (time.time() - t0), 1),
                         "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            torch.cuda.reset_peak_memory_stats()
        return True, out
    check(results, "C5 synthetic train-step throughput (batch 32, 640)", throughput)


def data(results):
    import numpy as np
    import corpus
    import stacked_dataset as SD
    import cv2

    def identity():
        ms, ss = sha256(corpus.MANIFEST), sha256(corpus.SPLITS)
        return ms == MANIFEST_SHA and ss == SPLITS_SHA, {"manifest": ms, "splits": ss, "corpus": corpus.NAME,
                                                          "path_map": os.environ.get("AEROTRACK_PATH_MAP")}
    check(results, "D1 corpus identity (sha256 = OMEN)", identity)

    df = SD.load_manifest(corpus.MANIFEST)

    def paths():
        sp = json.load(open(corpus.SPLITS))
        keep = np.zeros(len(df), dtype=bool)
        for phase in ("train", "val"):
            for ds, splits in sp[phase].items():
                keep |= (df["dataset"].values == ds) & df["split"].isin(splits).values
        sub = df[keep & (df["materialized"].values == 1)]
        missing = [p for p in sub["image_path"].tolist() if not os.path.exists(p)]
        return not missing, {"train_val_images": len(sub), "missing": len(missing), "examples": missing[:3]}
    check(results, "D2 every TRAIN/VAL image path resolves", paths)

    def decode():
        rng = random.Random(0)
        rows = df[df["materialized"] == 1].sample(n=200, random_state=0)
        want = {("anti_uav", "IR"): (512, 640), ("anti_uav", "RGB"): (1080, 1920), ("cst", "IR"): (512, 640), ("det_fly", "RGB"): (640, 640)}
        bad = []
        for _, r in rows.iterrows():
            im = cv2.imread(r["image_path"])
            if im is None or im.shape[:2] != want[(r["dataset"], r["modality"])]:
                bad.append((r["image_path"], None if im is None else im.shape))
        return not bad, {"decoded": len(rows), "bad": bad[:3]}
    check(results, "D3 200 random frames decode at the expected size", decode)

    def labels():
        lc = SD.load_label_cache(corpus.LABEL_CACHE)
        offs, boxes = lc
        return len(offs) == len(df) + 1, {"manifest_rows": len(df), "cache_offsets": len(offs), "boxes": int(len(boxes))}
    check(results, "D4 label cache aligned with the manifest", labels)

    def leakage():
        tr = SD.build_split_samples(corpus.MANIFEST, corpus.SPLITS, "train")
        va = SD.build_split_samples(corpus.MANIFEST, corpus.SPLITS, "val")
        te = SD.build_split_samples(corpus.MANIFEST, corpus.SPLITS, "test", require_image=False)
        SD.assert_no_leakage(tr, va, "train/val")
        SD.assert_no_leakage(tr, te, "train/test")
        SD.assert_no_leakage(va, te, "val/test")
        return True, {"train": len(tr), "val": len(va), "test": len(te), "pairs": ["train/val", "train/test", "val/test"]}
    check(results, "D5 sequence-level leakage assert", leakage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["core", "data"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results = []
    (core if a.part == "core" else data)(results)
    rep = {"part": a.part, "host": platform.node(), "t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ok": all(r["ok"] for r in results),
           "checks": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print("GATE", "PASS" if rep["ok"] else "FAIL", a.out)
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
