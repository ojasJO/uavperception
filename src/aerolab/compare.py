"""Protocol compatibility of result records (task brief §7.6).

A numerical comparison / ranking is VALID only if both records share the protocol id, the split and every comparability key of the
protocol, and the per-sequence sets match for paired statistics. Otherwise the comparison is 'non_equivalent' (side-by-side only,
no ranking) and the reasons are listed.
"""
import numpy as np


def _key(e: dict, k: str):
    if k == "sequence_set":
        ps = e.get("per_sequence")
        return tuple(sorted(ps)) if isinstance(ps, dict) and ps else e.get("sequence_set")
    return e.get(k)


def compatibility(a: dict, b: dict, protocols: dict) -> dict:
    reasons = []
    if a.get("protocol_id") != b.get("protocol_id"):
        reasons.append(f"different protocols: {a.get('protocol_id')} vs {b.get('protocol_id')}")
    if a.get("split") != b.get("split"):
        reasons.append(f"different splits: {a.get('split')} vs {b.get('split')}")
    proto = protocols.get(a.get("protocol_id")) or {}
    for k in proto.get("comparability_keys", []):
        va, vb = _key(a, k), _key(b, k)
        if va is None and vb is None:
            continue
        if va != vb:
            if k == "sequence_set" and isinstance(va, tuple) and isinstance(vb, tuple):
                reasons.append(f"different sequence sets ({len(va)} vs {len(vb)} sequences, {len(set(va) & set(vb))} shared)")
            else:
                reasons.append(f"{k} differs: {va} vs {vb}")
    if a.get("provenance") == "paper_reported" or b.get("provenance") == "paper_reported":
        reasons.append("one side is a number transcribed from a paper (no per-sequence data; different implementation)")
    return {"valid": not reasons, "verdict": "comparable" if not reasons else "non_equivalent", "reasons": reasons}


def paired_delta(a: dict, b: dict, n_boot: int = 10000, seed: int = 0) -> dict | None:
    """sequence-paired mean difference a - b with a bootstrap 95 % CI (needs identical sequence sets)."""
    pa, pb = a.get("per_sequence") or {}, b.get("per_sequence") or {}
    common = sorted(set(pa) & set(pb))
    if not common or set(pa) != set(pb):
        return None
    d = np.array([pa[k] - pb[k] for k in common], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(d), "mean_delta": float(d.mean()), "ci95": [float(lo), float(hi)], "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum()), "ties": int((d == 0).sum()), "method": f"sequence bootstrap B={n_boot} (default_rng({seed}))"}
