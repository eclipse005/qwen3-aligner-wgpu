"""PyTorch CPU reference for the FULL prefill (all layers), matching the CUDA
engine's op semantics (f16 storage, f32 accumulate, one rounding per op).
Dumps per-layer K caches + final h so the wgpu prefill can be bisected.

Usage (repo root):  python wgpu/tools/ref_prefill.py --tag q06_15s_en
"""

import argparse
import os
import numpy as np
import torch
import json
from safetensors.torch import load_file

F16 = torch.float16


def rmsnorm(x, w, eps, hs):
    xf = x.float()
    inv = torch.rsqrt((xf * xf).sum(-1, keepdim=True) / hs + eps)
    return (xf * inv * w.float()).to(F16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="q06_15s_en")
    args = ap.parse_args()
    g = os.path.join("wgpu", "golden", args.tag)
    meta = json.load(open(os.path.join(g, "meta.json")))
    s, hs = meta["seq_len"], meta["hidden_size"]
    nqh, nkvh, hd = meta["num_attention_heads"], meta["num_key_value_heads"], meta["head_dim"]
    inter, nl = meta["intermediate_size"], meta["num_hidden_layers"]
    eps = meta["rms_norm_eps"]

    t = load_file(meta["model_dir"] + "/model.safetensors")
    P = "thinker.model."

    hidden = np.fromfile(os.path.join(g, "hidden.bin"), dtype="<f2")
    h = torch.from_numpy(hidden).reshape(s, hs).to(F16)
    cos = np.fromfile(os.path.join(g, "cos.bin"), dtype="<f2").reshape(-1, hd)
    sin = np.fromfile(os.path.join(g, "sin.bin"), dtype="<f2").reshape(-1, hd)
    cos = torch.from_numpy(cos).float()
    sin = torch.from_numpy(sin).float()

    out_dir = g + "_refp"
    os.makedirs(out_dir, exist_ok=True)

    for l in range(nl):
        p = f"{P}layers.{l}."
        iln = t[p + "input_layernorm.weight"].to(F16)
        pln = t[p + "post_attention_layernorm.weight"].to(F16)
        qn = t[p + "self_attn.q_norm.weight"].to(F16)
        kn = t[p + "self_attn.k_norm.weight"].to(F16)
        wqkv = torch.cat([t[p + f"self_attn.{x}_proj.weight"] for x in "qkv"], 0).to(F16)
        wo = t[p + "self_attn.o_proj.weight"].to(F16)
        wgu = torch.cat([t[p + "mlp.gate_proj.weight"], t[p + "mlp.up_proj.weight"]], 0).to(F16)
        wd = t[p + "mlp.down_proj.weight"].to(F16)

        norm1 = rmsnorm(h, iln, eps, hs)
        qkv = (norm1.float() @ wqkv.float().T).to(F16)
        q_dim = nqh * hd

        # per-head norm + rotate-half RoPE at position p
        q_out = torch.empty(nqh, s, hd, dtype=F16)
        k_new = torch.empty(nkvh, s, hd, dtype=F16)
        half = hd // 2
        j = torch.arange(hd)
        pj = torch.where(j < half, j + half, j - half)
        sign = torch.where(j < half, -1.0, 1.0)
        for pos in range(s):
            c, si = cos[pos], sin[pos]
            for ih in range(nqh):
                x = qkv[pos, ih * hd:(ih + 1) * hd].float()
                inv = torch.rsqrt((x * x).sum() / hd + eps)
                xn = x * inv * qn.float()
                out = xn * c + sign * xn[pj] * si
                q_out[ih, pos] = out.to(F16)
            for ih in range(nkvh):
                x = qkv[pos, q_dim + ih * hd:q_dim + (ih + 1) * hd].float()
                inv = torch.rsqrt((x * x).sum() / hd + eps)
                xn = x * inv * kn.float()
                out = xn * c + sign * xn[pj] * si
                k_new[ih, pos] = out.to(F16)

        v_raw = qkv[:, q_dim + nkvh * hd:].reshape(s, nkvh, hd)
        rep = nqh // nkvh
        scale = 1.0 / hd ** 0.5
        attn_flat = torch.empty(s, nqh * hd, dtype=F16)
        for ih in range(nqh):
            kh_i = ih // rep
            # scores f16-stored (like the CUDA GEMM), then softmax in f32
            sc = (q_out[ih].float() @ k_new[kh_i].float().T).to(F16).float() * scale
            for pos in range(s):
                row = sc[pos, :pos + 1]
                row = row - row.max()
                pr = torch.softmax(row, 0)
                attn_flat[pos, ih * hd:(ih + 1) * hd] = (pr[None, :] @ v_raw[:, kh_i][:pos + 1].float())[0].to(F16)
        h = (h.float() + attn_flat.float() @ wo.float().T).to(F16)
        norm2 = rmsnorm(h, pln, eps, hs)
        gu = (norm2.float() @ wgu.float().T).to(F16)
        g_, u_ = gu[:, :inter].float(), gu[:, inter:].float()
        act = (g_ * torch.sigmoid(g_) * u_).to(F16)
        h = (h.float() + act.float() @ wd.float().T).to(F16)

        np.save(os.path.join(out_dir, f"L{l:02}_k.npy"), k_new.float().numpy())
        np.save(os.path.join(out_dir, f"L{l:02}_h.npy"), h.float().numpy())
        print(f"layer {l} done, h[:3] = {h[0][:3].float().tolist()}", flush=True)

    np.save(os.path.join(out_dir, "h_final.npy"), h.float().numpy())
    print("ref prefill complete")


if __name__ == "__main__":
    main()
