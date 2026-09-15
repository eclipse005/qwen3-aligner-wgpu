"""Parse wgpu/prefill_l0_debug.bin (11 length-prefixed f16 chunks) and compare
every tensor against a torch reference of layer 0 (semantics mirror
ref_prefill.py: f16 storage, f32 accumulate, one rounding per op).

Usage (repo root):  python wgpu/tools/analyze_dump.py
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
    inter, nl = meta["intermediate_size"], meta["num_hidden_layers"]
    eps = meta["rms_norm_eps"]
    mp = (s + 127) // 128 * 128
    npad = mp
    cur = s
    cur16 = (cur + 15) // 16 * 16

    # ---- parse dump -------------------------------------------------------
    raw = open(os.path.join(ROOT, "wgpu", "prefill_l0_debug.bin"), "rb").read()
    chunks, off = [], 0
    while off < len(raw):
        (n,) = np.frombuffer(raw, "<u4", 1, off)
        off += 4
        chunks.append(np.frombuffer(raw, "<f2", n, off).copy())
        off += 2 * n
    names = ["h", "normed0", "qkv0", "attn_flat", "gu0", "act0",
             "q_out", "v_rep", "k_rep", "scores", "attn"]
    print("chunks:", [(names[i], len(c)) for i, c in enumerate(chunks)])
    d = {k: torch.from_numpy(v).float() for k, v in zip(names, chunks)}

    # ---- reference layer 0 ------------------------------------------------
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
    sc_ref = torch.empty(nqh, s, s, dtype=F16)
    sm_ref = torch.empty(nqh, s, s)
    av_ref = torch.empty(s, nqh * hd, dtype=F16)
    for ih in range(nqh):
        kh_i = ih // rep
        # scores stored UNSCALED f16 (the softmax applies scale in f32)
        sc = (q_out[ih].float() @ k_new[kh_i].float().T).to(F16).float()
        for pos in range(s):
            row = sc[pos, :pos + 1] * scale
            pr = torch.softmax(row - row.max(), 0)
            sc_ref[ih, pos] = sc[pos].to(F16)
            sm_ref[ih, pos, :pos + 1] = pr
            av_ref[pos, ih * hd:(ih + 1) * hd] = \
                (pr[None, :] @ v_raw[:, kh_i][:pos + 1].float())[0].to(F16)

    def rep_row_err(got, ref, rows, label, thresh):
        """per-row max|delta| for rows list; return first bad row"""
        first_bad = None
        worst = (0.0, -1)
        for r in rows:
            e = (got[r] - ref[r]).abs().max().item()
            if e > worst[0]:
                worst = (e, r)
            if e > thresh and first_bad is None:
                first_bad = r
        print(f"  {label}: rows 0-7 max|d| = "
              f"{[round((got[r]-ref[r]).abs().max().item(),5) for r in rows[:8]]}")
        print(f"    worst row {worst[1]}: {worst[0]:.5f}, first bad: {first_bad}")
        return first_bad

    print("\n== q_out [nqh,mp,hd] ==")
    qg = d["q_out"].reshape(nqh, mp, hd)[:, :s]
    print("  overall max|d| =", (qg - q_out.float()).abs().max().item())

    print("\n== v_rep / k_rep [nqh,np,hd] (rows<s are pure copies) ==")
    vr = d["v_rep"].reshape(nqh, npad, hd)[:, :s]
    vr_ref = torch.stack([v_raw[:, ih // rep] for ih in range(nqh)], 1).permute(1, 0, 2)
    print("  v_rep max|d| =", (vr - vr_ref.float()).abs().max().item())
    kr = d["k_rep"].reshape(nqh, npad, hd)[:, :s]
    kr_ref = torch.stack([k_new[ih // rep] for ih in range(nqh)], 1).permute(1, 0, 2)
    print("  k_rep max|d| =", (kr - kr_ref.float()).abs().max().item())

    print("\n== scores [nqh,mp,np] ==")
    sg = d["scores"].reshape(nqh, mp, npad)[:, :s, :s]
    print("  overall max|d| =", (sg - sc_ref.float()).abs().max().item())
    rep_row_err(sg.permute(1, 0, 2), sc_ref.float().permute(1, 0, 2),
                range(s), "scores", 0.05)

    print("\n== attn/softmax [nqh,mp,cur16] ==")
    ag = d["attn"].reshape(nqh, mp, cur16)[:, :s, :s]
    print("  overall max|d| =", (ag - sm_ref).abs().max().item())
    rep_row_err(ag.permute(1, 0, 2), sm_ref.permute(1, 0, 2),
                range(s), "softmax", 0.02)

    print("\n== attn_flat/AV [mp, nqh*hd] ==")
    af = d["attn_flat"].reshape(mp, nqh * hd)[:s]
    print("  overall max|d| =", (af - av_ref.float()).abs().max().item())
    rep_row_err(af, av_ref.float(), range(s), "AV", 0.02)

    print("\n== L0 h vs refp ==")
    href = np.load(os.path.join(g + "_refp", "L00_h.npy"))
    hg = d["h"].reshape(s, hs)
    print("  overall max|d| =", (hg - torch.from_numpy(href)).abs().max().item())
    rep_row_err(hg, torch.from_numpy(href), range(s), "h", 0.02)


if __name__ == "__main__":
    main()
