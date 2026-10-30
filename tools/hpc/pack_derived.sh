#!/usr/bin/env bash
# Pack the derived corpus files the HPC needs (never the junctioned image directories) into one tarball + sha256.
#   tools/hpc/pack_derived.sh <out.tar.gz>
# Contents (repo-relative, so the tar unpacks into a repository checkout):
#   b_pipeline/data_unified_v2/{manifest.csv,splits.json,label_cache.npz,hardness.npz,sample_hardness.csv,cst_attributes.csv}
#   b_pipeline/data_unified_v2/cst/labels      (CST label txt, corpus v2)
#   b_pipeline/data_unified/anti_uav/labels    (Anti-UAV label txt; v2 manifest points here)
#   b_pipeline/data_unified/det_fly/labels     (Det-Fly label txt; v2 manifest points here)
#   runs/aerotrack_temporal/_data              (Track C trajectory files)
set -eu
OUT="$1"; cd "$(dirname "$0")/../.."
V2=b_pipeline/data_unified_v2
tar -czf "$OUT" $V2/manifest.csv $V2/splits.json $V2/label_cache.npz $V2/hardness.npz $V2/sample_hardness.csv \
    $V2/cst_attributes.csv $V2/cst/labels b_pipeline/data_unified/anti_uav/labels b_pipeline/data_unified/det_fly/labels \
    runs/aerotrack_temporal/_data
sha256sum "$OUT" > "$OUT.sha256"
