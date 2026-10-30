"""Smoke tests for the VAL demo (no GPU training, no long evaluation): data contract, core logic, rendering,
a short video render, and a headless Streamlit run with button interactions.

    venv\\Scripts\\python.exe experiments\\anti_uav\\demo\\test_demo.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
import demo_core as C  # noqa: E402

from anti_uav_repro.data.antiuav import TestSplitAccessError, load_sequences  # noqa: E402


def main():
    t0 = time.time()
    runs = C.discover_runs()
    assert runs, "no runs with weights/last.pt found"
    print(f"[1] runs: {[r.name for r in runs]}")
    for r in runs:
        assert r.checkpoint.exists()
        print(f"    {r.label}: preds={'yes' if r.predictions_dir else 'no'} val mSA={r.val_mSA_percent} sha={(r.checkpoint_sha256 or '')[:12]} "
              f"eval-sha-matches={r.saved_predictions_match_checkpoint}")
    seqs = C.load_val_sequences(C.data_cfg_of(runs[0]))
    assert len(seqs) == 67, len(seqs)
    print(f"[2] {len(seqs)} VAL sequences; split constant = {C.SPLIT!r}")
    try:
        load_sequences("data/anti_uav", "b_pipeline/data_unified/anti_uav/images", "test", "IR", purpose=C.PURPOSE)
        raise AssertionError("test split was served with the demo's purpose")
    except TestSplitAccessError:
        print("[3] test split refused for the demo's purpose — OK")

    # ---- data contract on one sequence: files, labels, predictions, index alignment, coordinates
    run = next(r for r in runs if r.predictions_dir is not None)
    seq = seqs[0]
    assert sum(1 for _ in seq.frames_dir.glob("*.jpg")) == seq.n_frames
    assert seq.frame_path(0).name == "000001.jpg" and seq.frame_path(seq.n_frames - 1).exists()
    p = C.load_saved_prediction(run, seq.name)
    assert p is not None and p["n_frames"] == seq.n_frames == len(p["res"]) == len(p["scores"]) == len(p["processed"])
    tab = C.per_frame_table(p, seq)
    ious = np.array([r["iou"] for r in tab if r["iou"] is not None])
    shifted = np.array([C.iou_xywh(p["res"][t + 1], seq.box_xywh(t)) for t in range(1, 300) if seq.exist[t] and C.is_box(p["res"][t + 1])])
    assert ious.mean() > shifted.mean(), "predictions look misaligned with the labels"
    print(f"[4] {seq.name}: frames == labels == predictions ({seq.n_frames}); mean IoU {ious.mean():.3f} > shifted-by-one {shifted.mean():.3f} "
          f"-> indices aligned; boxes [x,y,w,h] in image pixels (GT {seq.box_xywh(0).tolist()}, pred {[round(v,1) for v in p['res'][0]]})")
    s = C.sequence_summary(p, seq)
    stored = run.per_sequence_SA.get(seq.name)
    assert stored is not None and abs(s["SA_percent"] - 100 * stored) < 1e-6
    print(f"[5] sequence SA recomputed {s['SA_percent']:.4f} % == metrics.json {100*stored:.4f} %")

    # ---- rendering
    t = p["start_frame"] + 5
    fm = C.frame_metrics(p, seq, t)
    im = C.render_frame(seq, t, fm)
    inset = C.render_inset(seq, t, fm)
    strip = C.render_iou_strip(tab, t)
    assert im.size == (640, 512) and inset is not None and inset.size[0] >= 300 and strip.size == (640, 70)
    arr = np.asarray(im)
    x, y, w, h = [int(v) for v in fm["gt"]]
    assert (arr[y, x:x + w, 1] > 200).mean() > 0.5, "green GT box not visible at the GT location"
    low = C.low_iou_frames(tab)
    print(f"[6] frame {t}: image {im.size}, inset {inset.size}, IoU strip {strip.size}; GT box pixels verified green; "
          f"{len(low)} low-IoU frames (first: {low[:2]})")
    out = C.DEMO_DIR / "cache" / "videos" / "_test_short.mp4"
    C.render_video([run], seq, [p], out, fps=25, max_frames=12)
    assert out.exists() and os.path.getsize(out) > 1000
    out.unlink()
    print("[7] 12-frame annotated MP4 rendered and removed")
    pr = C.paper_reference()
    assert "normal" in pr and "DFSC" in pr

    # ---- headless Streamlit run with interactions
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("VALIDATION ONLY" in (e.value or "") for e in at.error)
    f0 = at.slider[0].value
    labels = [b.label for b in at.button]
    nxt = next(i for i, l in enumerate(labels) if "Next" in l)
    prev = next(i for i, l in enumerate(labels) if "Previous" in l)
    last = next(i for i, l in enumerate(labels) if l == "Last")
    at.button[nxt].click().run()
    assert not at.exception and at.slider[0].value == f0 + 1, (f0, at.slider[0].value)
    at.button[prev].click().run()
    assert at.slider[0].value == f0
    at.button[last].click().run()
    assert at.slider[0].value == seqs[0].n_frames - 1
    md = " ".join(m.value for m in at.markdown)
    assert "**Frame:**" in md and "**IoU:**" in md and "**Sequence SA:**" in md
    print(f"[8] app: opens at frame {f0} (first tracked frame), Next -> {f0+1}, Previous -> {f0}, Last -> {seqs[0].n_frames-1}; metrics block present; no exceptions")
    at.toggle[0].set_value(True).run()
    assert not at.exception
    print(f"[9] compare mode: {len(at.markdown)} markdown blocks, {len(at.image)} images (two panels), no exceptions")
    print(f"all demo checks passed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
