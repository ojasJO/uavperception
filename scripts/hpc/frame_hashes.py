"""Reference hashes of decoded Anti-UAV frames (data/derived/data_unified/anti_uav/images) for the HPC decode-identity check.

    python scripts/hpc/frame_hashes.py sample --n 2000 --out <json>     # OMEN: random sample (seed 0) of frame paths + sha256
    python scripts/hpc/frame_hashes.py check  --ref <json> --root <dir> # HPC: re-hash the same relative paths under <dir>
"""
import argparse, hashlib, json, random, sys
from pathlib import Path

REL = Path("data/derived/data_unified/anti_uav/images")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["sample", "check"])
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--out"); ap.add_argument("--ref"); ap.add_argument("--root", default=".")
    a = ap.parse_args()
    if a.mode == "sample":
        base = Path(__file__).resolve().parents[2] / REL
        seqs = sorted(p.name for p in base.iterdir() if p.is_dir())
        rng = random.Random(0); out = {}
        while len(out) < a.n:
            s = rng.choice(seqs); frames = sorted((base / s).glob("*.jpg"))
            f = rng.choice(frames); out[f"{s}/{f.name}"] = sha(f)
        Path(a.out).write_text(json.dumps({"root_rel": str(REL).replace("\\", "/"), "seed": 0, "hashes": out}, indent=0))
        print(len(out), "hashes ->", a.out)
    else:
        ref = json.load(open(a.ref)); root = Path(a.root) / ref["root_rel"]
        bad = [k for k, v in ref["hashes"].items() if not (root / k).exists() or sha(root / k) != v]
        print(json.dumps({"checked": len(ref["hashes"]), "mismatch_or_missing": len(bad), "examples": bad[:5]}))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
