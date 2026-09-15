"""PyTorch CPU reference for one decode step of the Qwen3-ASR text decoder.

Mirrors the CUDA kernel semantics op-by-op: f16 storage between ops, f32
accumulation inside each op, one rounding to f16 at each op boundary.
Consumes the wgpu golden dump (KV prefix + token) and writes per-layer
intermediates so the wgpu engine can be bisected against it.

Usage (repo root):
    python wgpu/tools/ref_decode_step.py --tag q06_15s_en
"""

import argparse
import os
import numpy as np
import torch
from safetensors.torch import load_file

F16 = torch.float16


def rmsnorm(x_f32, w_f32, eps):
    # f32 sum of squares over f16-rounded input, one rounding at the end
    ss = (x_f32 * x_f32).sum()
    inv = torch.rsqrt(ss / x_f32.numel() + eps)
    return (x_f32 * inv * w_f32).to(F16)


def linear(x_f16, w_f16):
    # f32 GEMV on f16 inputs, one rounding at the end
    return (x_f16.float() @ w_f16.float().T).to(F16)


def linear_accum(h_f16, x_f16, w_f16):
    # residual fused into the GEMM epilogue: f32(h) + x@W^T, one rounding
    return (h_f16.float() + x_f16.float() @ w_f16.float().T).to(F16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="q06_15s_en")
    ap.add_argument("--golden", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    golden = args.golden or os.path.join("wgpu", "golden", args.tag)
    import json
    meta = json.load(open(os.path.join(golden, "meta.json")))
    model_dir = meta["model_dir"]
    out_dir = args.out or os.path.join("wgpu", "golden", args.tag + "_ref")
    os.makedirs(out_dir, exist_ok=True)

    hs = meta["hidden_size"]
    nqh = meta["num_attention_heads"]
    nkvh = meta["num_key_value_heads"]
    hd = meta["head_dim"]
    inter = meta["intermediate_size"]
    nl = meta["num_hidden_layers"]
    seq_len = meta["seq_len"]
    eps = meta["rms_norm_eps"]
    scale = 1.0 / (hd ** 0.5)

    t = load_file(os.path.join(model_dir, "model.safetensors"))
    P = "thinker.model."

    tokens = np.fromfile(os.path.join(golden, "tokens.bin"), dtype="<i4")
    tok0 = int(tokens[0])

    k_cache = np.fromfile(os.path.join(golden, "k_cache.bin"), dtype="<f2")
    v_cache = np.fromfile(os.path.join(golden, "v_cache.bin"), dtype="<f2")
    k_cache = torch.from_numpy(k_cache).reshape(nl, nkvh, seq_len, hd).float()
    v_cache = torch.from_numpy(v_cache).reshape(nl, nkvh, seq_len, hd).float()

    cos = np.fromfile(os.path.join(golden, "cos.bin"), dtype="<f2")
    sin = np.fromfile(os.path.join(golden, "sin.bin"), dtype="<f2")
    cos = torch.from_numpy(cos).reshape(-1, hd).float()
    sin = torch.from_numpy(sin).reshape(-1, hd).float()
    pos = seq_len  # decode step 0 writes at slot `pos`, ropes with row `pos`

    embed = t[P + "embed_tokens.weight"]  # bf16 [vocab, hs]

    h = embed[tok0].to(F16)  # [hs]
    print(f"token {tok0}, h[:4] = {h[:4].float().tolist()}")

    def head_norm_rotary(x, w, i):
        # x: [hd] f16 raw head slice; w: [hd] f16 norm weight; i: rope row
        xf = x.float()
        inv = torch.rsqrt((xf * xf).sum() / hd + eps)
        xn = xf * inv * w.float()  # f32, no rounding yet (kernel keeps f32 too)
        half = hd // 2
        j = torch.arange(hd)
        pj = torch.where(j < half, j + half, j - half)
        pair = torch.where(j < half, -xn[pj], xn[pj])
        c, s = cos[pos], sin[pos]
        return (xn * c + pair * s).to(F16)

    for l in range(nl):
        p = f"{P}layers.{l}."
        iln = t[p + "input_layernorm.weight"].to(F16)
        pln = t[p + "post_attention_layernorm.weight"].to(F16)
        qn = t[p + "self_attn.q_norm.weight"].to(F16)
        kn = t[p + "self_attn.k_norm.weight"].to(F16)
        wq = t[p + "self_attn.q_proj.weight"].to(F16)
        wk = t[p + "self_attn.k_proj.weight"].to(F16)
        wv = t[p + "self_attn.v_proj.weight"].to(F16)
        wo = t[p + "self_attn.o_proj.weight"].to(F16)
        wg = t[p + "mlp.gate_proj.weight"].to(F16)
        wu = t[p + "mlp.up_proj.weight"].to(F16)
        wd = t[p + "mlp.down_proj.weight"].to(F16)
        wqkv = torch.cat([wq, wk, wv], 0)
        wgu = torch.cat([wg, wu], 0)

        norm1 = rmsnorm(h.float(), iln, eps)
        qkv = linear(norm1, wqkv)  # [q_dim + 2*kv_dim]

        q_dim, kv_dim = nqh * hd, nkvh * hd
        q_out = torch.empty(nqh, hd, dtype=F16)
        k_new = torch.empty(nkvh, hd, dtype=F16)
        v_new = qkv[q_dim + kv_dim:].reshape(nkvh, hd).clone()
        for ih in range(nqh):
            q_out[ih] = head_norm_rotary(qkv[ih * hd:(ih + 1) * hd], qn, None)
        for ih in range(nkvh):
            k_new[ih] = head_norm_rotary(qkv[q_dim + ih * hd:q_dim + (ih + 1) * hd], kn, None)

        # attention per q-head (scores f32, softmax f32, output rounded once)
        attn = torch.empty(nqh, hd, dtype=F16)
        for ih in range(nqh):
            kh_i = ih // (nqh // nkvh)
            k_all = torch.cat([k_cache[l, kh_i], k_new[kh_i][None]], 0)  # [cur_len, hd]
            v_all = torch.cat([v_cache[l, kh_i], v_new[kh_i][None]], 0)
            sc = (k_all.float() @ q_out[ih].float()) * scale  # [cur_len]
            sc = sc - sc.max()
            p_ = torch.softmax(sc, 0)
            attn[ih] = (p_[None, :] @ v_all.float())[0].to(F16)
        attn_out = attn.reshape(-1)

        h = linear_accum(h, attn_out, wo)
        norm2 = rmsnorm(h.float(), pln, eps)
        gate_up = linear(norm2, wgu)
        g, u = gate_up[:inter].float(), gate_up[inter:].float()
        act = (g * torch.sigmoid(g) * u).to(F16)
        h = linear_accum(h, act, wd)

        np.save(os.path.join(out_dir, f"L{l:02d}_norm1.npy"), norm1.float().numpy())
        np.save(os.path.join(out_dir, f"L{l:02d}_qkv.npy"), qkv.float().numpy())
        np.save(os.path.join(out_dir, f"L{l:02d}_q_out.npy"), q_out.reshape(-1).float().numpy())
        np.save(os.path.join(out_dir, f"L{l:02d}_attn_out.npy"), attn_out.float().numpy())
        np.save(os.path.join(out_dir, f"L{l:02d}_norm2.npy"), norm2.float().numpy())
        np.save(os.path.join(out_dir, f"L{l:02d}_gate_up.npy"), gate_up.float().numpy())
        np.save(os.path.join(out_dir, f"L{l:02d}_activated.npy"), act.float().numpy())
        np.save(os.path.join(out_dir, f"L{l:02d}_h.npy"), h.float().numpy())
        if l % 7 == 0 or l == nl - 1:
            print(f"layer {l}: h[:4] = {h[:4].float().tolist()}")

    norm_w = t[P + "norm.weight"].to(F16)
    final_norm = rmsnorm(h.float(), norm_w, eps)
    logits = linear(final_norm, embed.to(F16))  # lm_head = embed^T
    tok = int(logits.argmax())
    print("final_norm[:4] =", final_norm[:4].float().tolist())
    print("argmax token   =", tok)
    np.save(os.path.join(out_dir, "final_norm.npy"), final_norm.float().numpy())
    np.save(os.path.join(out_dir, "logits.npy"), logits.float().numpy())

    # compare against CUDA golden step0 (layer 27 scratch) + step logits
    for name, mine in [
        ("norm1", norm1), ("qkv", qkv), ("q_out", q_out.reshape(-1)),
        ("attn_out", attn_out), ("norm2", norm2), ("gate_up", gate_up),
        ("activated", act), ("h_out", h), ("final_norm", final_norm),
    ]:
        ref = np.fromfile(os.path.join(golden, f"step0_{name}.bin"), dtype="<f2").astype(np.float32)
        d = np.abs(mine.float().numpy() - ref).max()
        print(f"step0 {name:<11} max|torch-CUDA| = {d:.4e}")
    ref_l = np.fromfile(os.path.join(golden, "step_logits.bin"), dtype="<f2").astype(np.float32)
    print(f"step0 logits   max|torch-CUDA| = {np.abs(logits.float().numpy() - ref_l[:len(logits)]).max():.4e}")
    print(f"torch next token = {tok}  (golden tokens[1] = {tokens[1] if len(tokens) > 1 else '?'})")


if __name__ == "__main__":
    main()
