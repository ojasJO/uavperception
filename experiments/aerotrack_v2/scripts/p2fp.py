"""Function-preserving initialisation of the P2 layout (c_model/yolo11-stride-p2fp.yaml) and its runtime proof.

The P2 model starts as a copy of a stock stride model initialised by the series' own procedure (COCO transfer by name and
shape + gen_lib.apply_fp_init: COCO stem on the centre frame): layers 0-22 by identical name, the stock Detect branches
cv2/cv3 {0,1,2} + dfl -> the P2 model's Detect {0,1,2} (same widths because P3 is the first Detect input), and the new P2 class
branch bias set to -20. Hence at initialisation the first 8,400 output anchors (P3, P4, P5) equal the stock model's outputs and
the 25,600 P2 anchors score ~2e-9 (< the 0.001 evaluation threshold): any later difference is learned from data.
"""
import torch

P2_CLS_BIAS = -20.0


def init_p2fp(model_p2, build_model, transfer_pretrained, apply_fp_init, stride_yaml, pretrained, ch=9, nc=2):
    """Initialise `model_p2` in place; returns the equivalence check dict (raises if it fails)."""
    ref = build_model(stride_yaml, ch=ch, nc=nc, verbose=False, hyp=None, class_weights=None)
    transfer_pretrained(ref, pretrained, verbose=False)
    ref_check = apply_fp_init(ref, "stride")
    rs, ms = ref.state_dict(), model_p2.state_dict()
    det_ref = f"model.{len(ref.model) - 1}."
    det_p2 = f"model.{len(model_p2.model) - 1}."
    copied = 0
    for k, v in rs.items():
        if k.startswith(det_ref):
            tail = k[len(det_ref):]
            if tail.split(".")[0] in ("cv2", "cv3") and int(tail.split(".")[1]) > 2:
                continue
            nk = det_p2 + tail
        else:
            nk = k
        if nk not in ms or ms[nk].shape != v.shape:
            raise RuntimeError(f"P2fp init: {k} -> {nk} has no same-shape target")
        ms[nk] = v.clone()
        copied += 1
    model_p2.load_state_dict(ms)
    det = model_p2.model[-1]
    with torch.no_grad():
        det.cv3[3][-1].bias.fill_(P2_CLS_BIAS)
    check = equivalence(model_p2, ref)
    check.update(copied_tensors=copied, reference_check=ref_check)
    if not check["ok"]:
        raise RuntimeError(f"P2fp function-preserving check failed: {check}")
    return check


@torch.no_grad()
def equivalence(model_p2, ref, size=256):
    """Stock anchors identical to the reference model, P2 anchors inert. Runs on CPU on a random 9-channel input."""
    was = model_p2.training, ref.training
    model_p2.eval(), ref.eval()
    dev = next(model_p2.parameters()).device
    x = torch.rand(1, 9, size, size)
    y2 = model_p2(x.to(dev))[0].cpu()
    yr = ref(x)[0] if next(ref.parameters()).device.type == "cpu" else ref(x.to(dev))[0].cpu()
    n_stock = yr.shape[-1]
    d_stock = float((y2[..., :n_stock] - yr).abs().max())
    p2_score = float(y2[:, 4:, n_stock:].max())
    model_p2.train(was[0]), ref.train(was[1])
    scale = max(float(yr.abs().max()), 1.0)
    return {"stock_anchor_max_abs_diff": d_stock, "stock_output_max_abs": scale, "p2_max_class_score": p2_score,
            "n_stock_anchors": int(n_stock), "n_p2_anchors": int(y2.shape[-1] - n_stock),
            "ok": d_stock <= 1e-5 * scale + 1e-3 and p2_score < 1e-6}


def apply_fp_init_from(model, pretrained):
    """Centre-frame stem from ANY COCO checkpoint (gen_lib.apply_fp_init is tied to yolo11n.pt): COCO stem weights on the
    centre channels 3:6, zeros on t-g / t+g, so the 9-channel model computes the COCO model on frame t. Returns the check."""
    import ultralytics.nn.tasks as UT
    out = UT.load_checkpoint(str(pretrained))
    coco = (out[0] if isinstance(out, tuple) else out).float().state_dict()
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    w = coco["model.0.conv.weight"]
    stem = torch.zeros(w.shape[0], 9, *w.shape[2:])
    stem[:, 3:6] = w
    if sd["model.0.conv.weight"].shape != stem.shape:
        raise RuntimeError(f"stem shape {tuple(sd['model.0.conv.weight'].shape)} vs COCO-derived {tuple(stem.shape)}")
    sd["model.0.conv.weight"] = stem
    model.load_state_dict(sd)
    was = model.training
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = torch.rand(1, 9, 256, 256)
        t = x[:, 3:6]
        d = float((model(x.to(dev))[0] - model(torch.cat([t, t, t], 1).to(dev))[0]).abs().max())
    model.train(was)
    return {"ignores_neighbours_max_abs_diff": d, "ok": d <= 1e-3, "pretrained": str(pretrained)}
