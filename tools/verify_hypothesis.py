"""Verify hypothesis: AV GEMM prefetch store uses transb=0 half selector, so for
k-tiles >= 1 the B value fetched becomes V[k, (n & ~1) | (k & 1)] (the half is
chosen by the LOADING thread's tx = k mod 16 instead of the n-element parity).

Compare dump AV rows 16..24 against this exact model.
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
    mp = (s + 127) // 128 * 128
    cur16 = (s + 15) // 16 * 16

    raw = open(os.path.join(ROOT, "wgpu", "prefill_l0_debug.bin"), "rb").read()
    chunks, off = [], 0
    while off < len(raw):
        (n,) = np.frombuffer(raw, "<u4", 1, off)
        off += 4
        chunks.append(np.frombuffer(raw, "<f2", n, off).copy())
        off += 2 * n
    af = torch.from_numpy(chunks[3]).float().reshape(mp, nqh * hd)[:s]

    t = load_file(os.path.join(ROOT, meta["model_dir"], "model.safetensors"))
    P = "thinker.model.layers.0."
    iln = t[P + "input_layernorm.weight"].to(F16)
    qn = t[P + "self_attn.q_norm.weight"].to(F16)
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
    q_out = torch.empty(nqh, s, hd, dtype=F16)
    k_new = torch.empty(nkvh, s, hd, dtype=F16)
    for pos in range(s):
        c, si = cos[pos], sin[pos]
        for ih in range(nqh):
            x = qkv[pos, ih * hd:(ih + 1) * hd].float()
            inv = torch.rsqrt((x * x).sum() / hd + eps)
            xn = x * inv * qn.float()
            q_out[ih, pos] = (xn * c + sign * xn[pj] * si).to(F16)
        for ih in range(nkvh):
            x = qkv[pos, q_dim + ih * hd:q_dim + (ih + 1) * hd].float()
            inv = torch.rsqrt((x * x).sum() / hd + eps)
            xn = x * inv * kn_w.float()
            k_new[ih, pos] = (xn * c + sign * xn[pj] * si).to(F16)
    v_raw = qkv[:, q_dim + nkvh * hd:].reshape(s, nkvh, hd)

    scale = 1.0 / hd ** 0.5
    rep = nqh // nkvh
    pr_all = torch.empty(nqh, s, s)
    for ih in range(nqh):
        kh_i = ih // rep
        sc = (q_out[ih].float() @ k_new[kh_i].float().T).to(F16).float()
        for pos in range(s):
            row = sc[pos, :pos + 1] * scale
            pr_all[ih, pos] = torch.cat(
                [torch.softmax(row - row.max(), 0),
                 torch.zeros(s - pos - 1)])

    # hypothesis model: for k >= 16 the B tile came through the prefetch path
    # whose store picks the half by tx (= k & 15) parity instead of n parity
    v_bug = v_raw.clone()
    for kpos in range(16, s):
        nsel = (torch.arange(hd) & ~1) | (kpos & 1)
        v_bug[kpos] = v_raw[kpos][:, nsel]

    err_good, err_bug = [], []
    for pos in range(14, 26):
        av_good = torch.empty(nqh * hd)
        av_bug = torch.empty(nqh * hd)
        for ih in range(nqh):
            kh_i = ih // rep
            pr = pr_all[ih, pos, :pos + 1]
            av_good[ih * hd:(ih + 1) * hd] = (pr @ v_raw[:, kh_i][:pos + 1].float()).to(F16).float()
            av_bug[ih * hd:(ih + 1) * hd] = (pr @ v_bug[:, kh_i][:pos + 1].float()).to(F16).float()
        eg = (af[pos] - av_good).abs().max().item()
        eb = (af[pos] - av_bug).abs().max().item()
        err_good.append(round(eg, 6))
        err_bug.append(round(eb, 6))
        print(f"pos {pos:3d}: |dump-correct| = {eg:.6f}   |dump-hypOTHESIS| = {eb:.6f}")
    print("\nrows 14-15 (tile 0 only):", err_good[:2], err_bug[:2])
    print("rows 16-25 (need prefetch):", err_good[2:], err_bug[2:])


if __name__ == "__main__":
    main()
