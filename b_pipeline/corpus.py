"""Which derived corpus the AeroTrack pipeline reads.

`b_pipeline/data_unified/` (corpus v1) is the corpus of every run up to 2026-10-04; its manifest hash is recorded in those
runs and it is never rewritten. A newer corpus lives in a sibling directory and is selected per process with

    set AEROTRACK_CORPUS=data_unified_v2

Every path the pipeline derives from the corpus (manifest, splits, label cache, sampler hardness cache) comes from here, so
one variable switches all of them together and a run can never mix files of two corpora.
"""
import os
from pathlib import Path

B_PIPELINE = Path(__file__).resolve().parent
NAME = os.environ.get("AEROTRACK_CORPUS", "data_unified")
if not NAME or any(sep in NAME for sep in ("/", "\\", "..")):
    raise ValueError(f"AEROTRACK_CORPUS must be a directory name under b_pipeline/, got {NAME!r}")
UNIFIED = B_PIPELINE / NAME
MANIFEST = UNIFIED / "manifest.csv"
SPLITS = UNIFIED / "splits.json"
LABEL_CACHE = UNIFIED / "label_cache.npz"
HARDNESS_CSV = UNIFIED / "sample_hardness.csv"
