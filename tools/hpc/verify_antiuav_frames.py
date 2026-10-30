"""M3 gate for the Anti-UAV frames re-decoded on the HPC: byte identity with the OMEN's frames.

Checks every sequence of the six frozen Track A manifests (experiments/anti_uav/manifests/antiuav_<MOD>_<split>.manifest.json):
frame count on disk, sha256 of the first / middle / last frame (index 0, n_frames // 2, n_frames - 1, as build_manifest
hashed them), label-file sha256 (verify_against_manifest); plus the random 2,000-frame sample hashed on the OMEN
(tools/hpc/frame_hashes.py). Read-only; writes one JSON report.
    python tools/hpc/verify_antiuav_frames.py --sample <omen_sample.json> --out <report.json>
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "anti_uav" / "src"))
from anti_uav_repro.data.antiuav import load_sequences, manifest_path, load_manifest, verify_against_manifest  # noqa: E402

DATA_ROOT, FRAMES_ROOT = REPO / "data" / "anti_uav", REPO / "b_pipeline" / "data_unified" / "anti_uav" / "images"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep, problems, n_seq, n_hash = {}, [], 0, 0
    for mod in ("IR", "RGB"):
        for split in ("train", "val", "test"):
            man = load_manifest(manifest_path(REPO / "experiments" / "anti_uav" / "manifests", split, mod))
            seqs = load_sequences(DATA_ROOT, FRAMES_ROOT, split, mod, purpose="verify")
            problems += [f"{mod}/{split}: {p}" for p in verify_against_manifest(seqs, man)]
            by = {s.name: s for s in seqs}
            for e in man["sequences"]:
                s = by.get(e["name"]); n_seq += 1
                if s is None:
                    continue
                n_disk = sum(1 for p in os.scandir(s.frames_dir) if p.name.endswith(".jpg"))
                if n_disk != e["frames_on_disk"]:
                    problems.append(f"{mod}/{split}/{e['name']}: frames_on_disk {n_disk} != {e['frames_on_disk']}")
                for tag, idx in (("first", 0), ("middle", s.n_frames // 2), ("last", s.n_frames - 1)):
                    want = e["frame_sha256"].get(tag); fp = s.frame_path(idx)
                    got = sha(fp) if fp.exists() else None
                    n_hash += 1
                    if got != want:
                        problems.append(f"{mod}/{split}/{e['name']}: {tag} frame hash differs")
    ref = json.load(open(a.sample)); root = REPO / ref["root_rel"]
    bad = [k for k, v in ref["hashes"].items() if not (root / k).exists() or sha(root / k) != v]
    problems += [f"sample: {k}" for k in bad]
    rep = {"ok": not problems, "sequences": n_seq, "boundary_hashes": n_hash, "sample_hashes": len(ref["hashes"]),
           "sample_mismatch": len(bad), "problems": problems[:50], "n_problems": len(problems)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "problems"}), problems[:5])
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
