"""AeroTrack v3 series: capacity allocation around the v2 campaign's selected temporal window (corpus v2).

The question (mission §30): under an edge budget, where does extra capacity pay off for tiny UAVs -- in spatial resolution
(an extra stride-4 scale), in network size (YOLO11s), or in time (the temporal window)? Every model is started from a COCO
model on the labelled frame by a verified function-preserving initialisation, and trained with the v2 recipe on corpus v2:

    model            yaml                                   init                                                   params
    n                c_model/yolo11-stride.yaml             COCO yolo11n + centre-frame stem (gen_lib)             2.61 M
    n_p2             c_model/yolo11-stride-p2fp.yaml        = stock n + inert appended P2 scale (p2fp.init_p2fp)    2.66 M
    s                c_model/yolo11-stride-scale-s.yaml     COCO yolo11s + centre-frame stem (p2fp)                 9.52 M

A configuration = (model, window); windows are v2's (ttt / symmetric / causal, gap 4). The registered set and the comparisons
are fixed in experiments/aerotrack_v3/preregistration.json once the v2 winner is known; the v2 runs of the same window and
seeds are the paired references for the n model (same recipe, corpus, seeds).
"""
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[2] / "aerotrack_v2" / "scripts"
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))
import v2series as S  # noqa: E402,F401  (pins AEROTRACK_CORPUS=data_unified_v2)
import v2lib as V2L  # noqa: E402

MODELS = {
    "n": {"yaml": "yolo11-stride.yaml", "pretrained": "yolo11n.pt", "init": "fp_stride"},
    "n_p2": {"yaml": "yolo11-stride-p2fp.yaml", "pretrained": "yolo11n.pt", "init": "fp_p2"},
    "s": {"yaml": "yolo11-stride-scale-s.yaml", "pretrained": "yolo11s.pt", "init": "fp_from"},
    # standard YOLO11n blocks (explicit `scales: n`): the as-built YAMLs above resolve scale '' and get C3k2(c3k=True);
    # registration experiments/aerotrack_v3/preregistration_F1_standard_yolo11n.json
    "n_std": {"yaml": "yolo11-stride-std.yaml", "pretrained": "yolo11n.pt", "init": "fp_stride"},
}
WINDOWS = {"ttt": 0, "sym_g4": 4, "causal_g4": 4}            # name -> gap (window kinds as in v2lib)
WINDOW_KIND = {"ttt": "ttt", "sym_g4": "symmetric", "causal_g4": "causal"}


def config_name(model, window):
    return f"{model}__{window}"


def parse(config):
    model, window = config.split("__")
    if model not in MODELS or window not in WINDOWS:
        raise ValueError(f"unknown configuration {config!r}")
    return model, window


def run_name(config, seed, epochs=20):
    return f"v3_{config}_s{seed}_e{epochs}"


def build_initialised(model_key, repo):
    """Build + initialise one model exactly as registered; returns (model, init_check)."""
    import gen_lib as G
    import p2fp
    from aerotrack_trainer import build_model, transfer_pretrained
    m = MODELS[model_key]
    yaml, pre = repo / "c_model" / m["yaml"], repo / "c_model" / m["pretrained"]
    model = build_model(yaml, ch=9, nc=2, verbose=False, hyp=None, class_weights=None)
    transfer_pretrained(model, pre, verbose=True)
    if m["init"] == "fp_stride":
        check = G.apply_fp_init(model, "stride")
    elif m["init"] == "fp_p2":
        check = p2fp.init_p2fp(model, build_model, transfer_pretrained, G.apply_fp_init, repo / "c_model" / "yolo11-stride.yaml", pre)
    else:
        check = p2fp.apply_fp_init_from(model, pre)
    return model, check


def window_samples(split, window):
    return V2L.window_samples(split, WINDOW_KIND[window], WINDOWS[window])


# ---- cross-dataset generalisation (mission §32): training-domain subsets ------------------------------------------------
DOMAINS = {"anti_uav_IR": ("anti_uav", "IR"), "anti_uav_RGB": ("anti_uav", "RGB"), "cst": ("cst", "IR"), "det_fly": ("det_fly", "RGB")}


def parse_domains(spec):
    """'all' or a comma list of DOMAINS keys -> sorted tuple (validated)."""
    if spec in (None, "", "all"):
        return tuple(sorted(DOMAINS))
    doms = tuple(sorted(d.strip() for d in spec.split(",")))
    bad = [d for d in doms if d not in DOMAINS]
    if bad or not doms:
        raise ValueError(f"unknown training domains {bad or spec!r}; choose from {sorted(DOMAINS)}")
    return doms


def domain_tag(doms):
    return "" if doms == tuple(sorted(DOMAINS)) else "_" + "+".join(doms)


def filter_domains(samples, doms):
    keep = {DOMAINS[d] for d in doms}
    return [s for s in samples if (s["dataset"], s["modality"]) in keep]
