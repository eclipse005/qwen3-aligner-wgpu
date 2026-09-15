"""Is the KV-cache max|delta| (0.25-0.40) a 1-ulp diff on an outlier dim, or a
real error?  Compare golden CUDA k_cache vs torch reference vs wgpu dump at the
worst element.
"""

import json
import os

import numpy as np
import torch
from safetensors.torch import load_file

F16 = torch.float16
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rmsnorm(x, w, eps, hs):
    xf = x.float()
    inv = torch.rsqrt((xf * xf).sum(-1, keepdim=True) / hs + eps)
    return (xf * inv * w.float()).to(F16)


def main():
    g = os.path.join(ROOT, "wgpu", "golden", "q06_15s_en")
    meta = json.load(open(os.path.join(g, "meta.json")))
    s, hs = meta["seq_len"], meta["hidden_size"]
    nqh, nkvh, hd = meta["num_attention_heads"], meta["num_key_value_heads"], meta["head_dim"]
    eps = meta["rms_norm_eps"]

    # torch reference k_new for layer 0
    t = load_file(os.path.join(ROOT, meta["model_dir"], "model.safetensors"))
    P = "thinker.model.layers.0."
    iln = t[P + "input_layernorm.weight"].to(F16)
    kn_w = t[P + "self_attn.k_norm.weight"].to(F16)
    wqkv = torch.cat([t[P + f"self_attn.{x}_proj.weight"] for x in "qkv"], 0).to(F16)
    h0 = np.fromfile(os.path.join(g, "hidden.bin"), dtype="<f2").reshape(s, hs)
    h = torch.from_numpy(h0).to(F16)
    cos = torch.from_numpy(np.fromfile(os.path.join(g, "cos.bin"), dtype="<f2")
                           .reshape(-1, hd)).float()
    sin = torch.from_numpy(np.fromfile(os.path.join(g, "sin.bin"), dtype="<f2")
                           .reshape(-1, hd)).float()
    norm1 = rmsnorm(h, iln, eps, hs)
    qkv = (norm1.float() @ wqkv.float().T).to(F16)
    q_dim = nqh * hd
    half = hd // 2
    j = torch.arange(hd)
    pj = torch.where(j < half, j + half, j - half)
    sign = torch.where(j < half, -1.0, 1.0)
    k_new = torch.empty(nkvh, s, hd, dtype=F16)
    for pos in range(s):
        c, si = cos[pos], sin[pos]
        for ih in range(nkvh):
            x = qkv[pos, q_dim + ih * hd:q_dim + (ih + 1) * hd].float()
            inv = torch.rsqrt((x * x).sum() / hd + eps)
            xn = x * inv * kn_w.float()
            k_new[ih, pos] = (xn * c + sign * xn[pj] * si).to(F16)

    # golden CUDA k_cache [nl][nkvh][s][hd]
    kc = np.fromfile(os.path.join(g, "k_cache.bin"), dtype="<f2")
    nl = len(kc) // (nkvh * s * hd)
    kc = kc.reshape(nl, nkvh, s, hd)
    print("layers in k_cache.bin:", nl)

    # wgpu dump k_rep = copy of L0 k cache rows [nqh][np][hd] -> k per nkvh
    raw = open(os.path.join(ROOT, "wgpu", "prefill_l0_debug.bin"), "rb").read()
    chunks, off = [], 0
    while off < len(raw):
        (n,) = np.frombuffer(raw, "<u4", 1, off)
        off += 4
        chunks.append(np.frombuffer(raw, "<f2", n, off).copy())
        off += 2 * n
    kr = torch.from_numpy(chunks[8]).float().reshape(nqh, -1, hd)[:, :s]  # [nqh,s,hd]

    kh = 4  # L00 worst head per prefill_check output
    kg = torch.from_numpy(np.ascontiguousarray(kc[0, kh])).float()  # CUDA golden
    kt = k_new[kh].float()          # torch
    kw = kr[kh // (nqh // nkvh)]    # wgpu (replicated head slot)
    d_gt = (kg - kt).abs()
    d_gw = (kg - kw).abs()
    i1 = d_gt.argmax()
    i2 = d_gw.argmax()
    p1, e1 = int(i1) // hd, int(i1) % hd
    p2, e2 = int(i2) // hd, int(i2) % hd
    print(f"\nL0 head {kh}:  golden-vs-torch max|d| = {d_gt.max():.4f} @ pos {p1} dim {e1}"
          f"  (golden {kg[p1, e1]:.3f}, torch {kt[p1, e1]:.3f})")
    print(f"L0 head {kh}:  golden-vs-wgpu  max|d| = {d_gw.max():.4f} @ pos {p2} dim {e2}"
          f"  (golden {kg[p2, e2]:.3f}, wgpu  {kw[p2, e2]:.3f}, torch {kt[p2, e2]:.3f})")

    # distribution of |K| and of diffs
    print(f"\n|K| quantiles (golden): 50%={kg.abs().quantile(0.5):.2f} "
          f"99%={kg.abs().quantile(0.99):.2f} 100%={kg.abs().max():.2f}")
    for thr in (0.03, 0.06, 0.125, 0.25):
        n = int((d_gw > thr).sum())
        print(f"  golden-vs-wgpu elems with |d|>{thr}: {n} / {d_gw.numel()}")
    big = (d_gw > 0.03).nonzero()[:10]
    for pp, ee in big.tolist():
        print(f"    pos {pp} dim {ee}: golden {kg[pp, ee]:8.3f}  wgpu {kw[pp, ee]:8.3f}"
              f"  torch {kt[pp, ee]:8.3f}  d={d_gw[pp, ee]:.4f}")


if __name__ == "__main__":
    main()
