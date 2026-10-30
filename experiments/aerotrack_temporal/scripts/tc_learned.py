"""Track C learned forecasters: history encoders with a per-horizon diagonal Gaussian head, trained with Gaussian NLL.

Inputs (all causal, relative to the anchor frame t; units = percent of the image width so IR 640 px and RGB 1920 px share a scale):
    per history frame j in t-L+1..t:  visible m_j, displacement (c_j - c_t), log size ratio log(s_j / s_t)   (zeros when not visible)
    global:                            log target size log(100 s_t / W), and (optional, `abs_pos`) the anchor position in the image
                                       (c_t / (W, H) - 0.5) -- the information a gimbal-following camera's re-centring depends on
Outputs: mean displacement (c_{t+h} - c_t) and log sigma per axis for h = 1..H.
Families (fixed sizes, chosen before any VAL use): `mlp` (2 x 256), `gru` (1 x 128), `tfm` (2-layer pre-norm transformer, d 64,
4 heads, the anchor token pooled). Augmentation `aug`: Gaussian jitter of every visible history centre (sigma = 0.1 x target size)
and dropout (p 0.1) of history frames before t-1 -- a fixed, a-priori stand-in for tracker noise (no VAL statistics used).
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

UNIT = 100.0                                   # displacement unit: percent of the image width
DOMAIN_TOKENS = ("antiuav_IR", "antiuav_RGB", "cst")   # C1c 'camera mode' token (gimbal IR / gimbal RGB / static IR)


def features(S, idx=None, aug=False, gen=None, abs_pos=True, dom_tok=False):
    """-> seq [n, L, 4], glob [n, 1|3], target [n, H, 2], tmask [n, H], scale [n] (px per unit), anchor centre [n, 2] (px, possibly
    jittered: forecasts are added to it)."""
    sl = slice(None) if idx is None else idx
    hc = torch.as_tensor(S["hist_c"][sl], dtype=torch.float32)
    hs = torch.as_tensor(S["hist_s"][sl], dtype=torch.float32)
    hm = torch.as_tensor(S["hist_m"][sl])
    W = torch.as_tensor(S["W"][sl], dtype=torch.float32)
    Hh = torch.as_tensor(S["H"][sl], dtype=torch.float32)
    if aug:
        noise = torch.randn(hc.shape, generator=gen) * (0.1 * hs.clamp(min=1.0))[..., None]
        hc = torch.where(hm[..., None], hc + noise, hc)
        drop = torch.rand(hm.shape, generator=gen) < 0.1
        drop[:, -2:] = False
        hm = hm & ~drop
    scale = W / UNIT
    ct = hc[:, -1]
    st = hs[:, -1].clamp(min=1.0)
    d = (hc - ct[:, None]) / scale[:, None, None]
    ls = torch.log(hs.clamp(min=1.0) / st[:, None])
    m = hm.float()
    seq = torch.cat([m[..., None], d * m[..., None], (ls * m)[..., None]], -1)
    g = [torch.log(st / scale)[:, None]]
    if abs_pos:
        g.append(torch.stack([ct[:, 0] / W - 0.5, ct[:, 1] / Hh - 0.5], 1))
    if dom_tok:
        dom = np.asarray(S["domain"][sl])
        g.append(torch.as_tensor(np.stack([(dom == d).astype(np.float32) for d in DOMAIN_TOKENS], 1)))
    glob = torch.cat(g, 1)
    fc = torch.as_tensor(S["fut_c"][sl], dtype=torch.float32)
    fm = torch.as_tensor(S["fut_m"][sl])
    target = (fc - ct[:, None]) / scale[:, None, None]
    return seq, glob, torch.where(fm[..., None], target, torch.zeros_like(target)), fm, scale, ct


class Head(nn.Module):
    def __init__(self, din, H):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Linear(256, H * 4))

    def forward(self, z):
        o = self.net(z).view(-1, self.H, 4)
        return o[..., :2], o[..., 2:].clamp(-7.0, 5.0)


class MLPF(nn.Module):
    def __init__(self, L, H, gdim):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(L * 4 + gdim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.head = Head(256, H)

    def forward(self, seq, glob):
        return self.head(self.body(torch.cat([seq.flatten(1), glob], 1)))


class GRUF(nn.Module):
    def __init__(self, L, H, gdim):
        super().__init__()
        self.inp = nn.Linear(4, 64)
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.head = Head(128 + gdim, H)

    def forward(self, seq, glob):
        _, h = self.gru(torch.nn.functional.gelu(self.inp(seq)))
        return self.head(torch.cat([h[-1], glob], 1))


class TFMF(nn.Module):
    def __init__(self, L, H, gdim, d=64):
        super().__init__()
        self.inp = nn.Linear(4, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = Head(d + gdim, H)

    def forward(self, seq, glob):
        z = self.enc(self.inp(seq) + self.pos)
        return self.head(torch.cat([z[:, -1], glob], 1))


ARCHS = {"mlp": MLPF, "gru": GRUF, "tfm": TFMF}


def nll_loss(mu, logs, target, mask):
    var = torch.exp(2 * logs)
    nll = 0.5 * ((target - mu) ** 2 / var).sum(-1) + logs.sum(-1) + math.log(2 * math.pi)
    return (nll * mask).sum() / mask.sum().clamp(min=1)


def build(arch, L, H, abs_pos, dom_tok=False):
    return ARCHS[arch](L, H, (3 if abs_pos else 1) + (len(DOMAIN_TOKENS) if dom_tok else 0))


@torch.no_grad()
def predict(model, S, abs_pos=True, batch=8192, dom_tok=False):
    """-> mean_px [n,H,2], var_px [n,H,2] (absolute image positions)."""
    model.eval()
    n = len(S["t"])
    means, vars_ = [], []
    for i in range(0, n, batch):
        idx = np.arange(i, min(n, i + batch))
        seq, glob, _, _, scale, ct = features(S, idx, abs_pos=abs_pos, dom_tok=dom_tok)
        mu, logs = model(seq, glob)
        means.append((ct[:, None] + mu * scale[:, None, None]).numpy().astype(np.float64))
        vars_.append((torch.exp(2 * logs) * scale[:, None, None] ** 2).numpy().astype(np.float64))
    return np.concatenate(means), np.concatenate(vars_)


def train(arch, S_tr, S_dev, seed, abs_pos=True, aug=False, epochs=12, batch=1024, lr=2e-3, wd=1e-4, threads=6, log=None, dom_tok=False):
    """Deterministic CPU training; keeps the epoch with the lowest dev NLL (dev = held-out TRAIN groups). -> model, history."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    L, H = S_tr["hist_c"].shape[1], S_tr["fut_c"].shape[1]
    model = build(arch, L, H, abs_pos, dom_tok)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(S_tr["t"])
    steps = epochs * math.ceil(n / batch)
    warm = max(1, int(0.05 * steps))                  # linear warm-up over 5 % of the steps, then cosine to 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
    gen = torch.Generator().manual_seed(seed)
    dev = features(S_dev, abs_pos=abs_pos, dom_tok=dom_tok)
    full = None if aug else features(S_tr, abs_pos=abs_pos, dom_tok=dom_tok)      # clean histories: features computed once
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).numpy()
        tot, cnt = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if full is None:
                seq, glob, tgt, m, _, _ = features(S_tr, idx, aug=True, gen=gen, abs_pos=abs_pos, dom_tok=dom_tok)
            else:
                ti = torch.as_tensor(idx)
                seq, glob, tgt, m = full[0][ti], full[1][ti], full[2][ti], full[3][ti]
            mu, logs = model(seq, glob)
            loss = nll_loss(mu, logs, tgt, m.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
        model.eval()
        with torch.no_grad():
            dn, dc = 0.0, 0
            for i in range(0, len(dev[0]), 8192):
                sl = slice(i, i + 8192)
                mu, logs = model(dev[0][sl], dev[1][sl])
                mm = dev[3][sl].float()
                dn += float(nll_loss(mu, logs, dev[2][sl], mm)) * float(mm.sum())
                dc += float(mm.sum())
        dev_nll = dn / max(dc, 1)
        rec = {"epoch": ep + 1, "train_nll": tot / cnt, "dev_nll": dev_nll, "seconds": round(time.time() - t0, 1)}
        hist.append(rec)
        if log:
            log(json.dumps(rec))
        if dev_nll < best:
            best = dev_nll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


def save(model, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    model = build(m["arch"], m["L"], m["H"], m["abs_pos"], m.get("dom_tok", False))
    model.load_state_dict(ck["state_dict"])
    return model, m
