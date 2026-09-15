//! CPU text decoder — the runtime for machines with no usable GPU adapter, and
//! the host-side oracle the GPU decoder can be diffed against.
//!
//! Design (mirrors the sibling CUDA port's `cpu_engine.rs`, which is the
//! reference implementation the user pointed at):
//!
//! * **Activations are f32, weights are f16.**  Every linear reads its weight
//!   matrix straight out of the mapped safetensors file and converts per
//!   element, so a decode step streams half the bytes.  Accumulation is f32.
//! * **Rayon parallelises every op** (rows of a linear, heads of an attention),
//!   and the `gemm` crate is *not* used: at m=1 its dispatch is single-threaded
//!   and a hand-rolled row-parallel dot product is both simpler and faster here.
//! * **The f16 rounding points are the contract.**  The GPU stores activations
//!   as f16 between ops (norm output, every linear output, the KV cache, the
//!   attention output); reproducing exactly those roundings is what makes the
//!   transcripts agree.  The *order* of the f32 accumulation inside a dot
//!   product is deliberately different — the reference's own CPU and CUDA paths
//!   differ there too, and both match the gold texts.
//!
//! Everything is verified by the same gate as the GPU path: run the fixtures
//! with `--cpu-dec` and compare against the frozen python-hf texts.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use half::f16;
use rayon::prelude::*;

use crate::decoder::TextConfig;
use crate::weights::{self, RawTensor};

/// A `[rows, cols]` weight matrix, row-major `[out_features, in_features]` —
/// the layout a GEMV wants to stream.
///
/// The file's f16 is widened to f32 **once, at load**: the per-element software
/// `f16 → f32` conversion (`half`'s, with its subnormal branches) cannot be
/// vectorised, and measured it cost 5.2 s of a 5.4 s prefill — the kernels were
/// conversion-bound at ~125 MB/s, not memory-bound.  As f32 the inner loop is a
/// plain FMA chain the compiler vectorises.
struct Mat {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Mat {
    /// f16 in the file, f32 in memory — the same values the GPU uploads, just
    /// widened before the arithmetic.
    fn load(w: &HashMap<String, RawTensor>, name: &str) -> Result<Self> {
        let (data, shape) = weights::get_f16(w, name)?;
        anyhow::ensure!(shape.len() == 2, "{name}: shape {shape:?}");
        Ok(Self {
            data: data.iter().map(|v| v.to_f32()).collect(),
            rows: shape[0],
            cols: shape[1],
        })
    }

    /// Batched over `rows` input vectors: `out[r][o] = Σ_k w[o,k]·x[r,k]`.
    ///
    /// Parallelised over **output rows** and accumulated in a transposed
    /// scratch (`acc[o][r]`), for two reasons that both showed up as measured
    /// slowdowns:
    ///
    /// * slicing by the batch leaves a decode step (batch 1) on a single core —
    ///   355 ms/token before this;
    /// * slicing by `(r, o)` makes every weight element be read once *per batch
    ///   row*, so a 210-token prefill streamed 139 GB instead of 0.66 GB —
    ///   5.5 s of prefill.
    ///
    /// With `o` outermost a weight row is streamed once (into registers/L1) and
    /// reused across the whole batch, while the activation block — the small
    /// tensor — is re-read from L2.  The f32 → f16 rounding happens once, as on
    /// the GPU.
    fn batch_dot(&self, x: &[f16], rows: usize, out: &mut [f16]) {
        debug_assert_eq!(x.len(), rows * self.cols);
        debug_assert_eq!(out.len(), rows * self.rows);
        if rows >= 2 {
            // Prefill: a real GEMM through the `gemm` crate (see
            // `gemm_row_major`).  Row-major in and out, so no transpose.
            let xs = widen_batch(x, rows, self.cols);
            let mut acc = vec![0.0f32; rows * self.rows];
            gemm_row_major(&mut acc, &xs, &self.data, rows, self.rows, self.cols, 0.0);
            out.par_iter_mut().enumerate().for_each(|(i, y)| *y = f16::from_f32(acc[i]));
            return;
        }
        let acc = self.dot_transposed(x, rows);
        let n_out = self.rows;
        out.par_iter_mut()
            .enumerate()
            .for_each(|(i, y)| *y = f16::from_f32(acc[(i % n_out) * rows + i / n_out]));
    }

    /// `h[r][o] += Σ_k w[o,k]·x[r,k]` — the residual-consumed projections
    /// (`o_proj`, `down_proj`), matching the GPU's accumulating GEMM.
    fn batch_dot_acc(&self, x: &[f16], rows: usize, h: &mut [f16]) {
        debug_assert_eq!(x.len(), rows * self.cols);
        debug_assert_eq!(h.len(), rows * self.rows);
        if rows >= 2 {
            let xs = widen_batch(x, rows, self.cols);
            let mut acc = vec![0.0f32; rows * self.rows];
            gemm_row_major(&mut acc, &xs, &self.data, rows, self.rows, self.cols, 0.0);
            h.par_iter_mut()
                .enumerate()
                .for_each(|(i, y)| *y = f16::from_f32(y.to_f32() + acc[i]));
            return;
        }
        let acc = self.dot_transposed(x, rows);
        let n_out = self.rows;
        h.par_iter_mut().enumerate().for_each(|(i, y)| {
            *y = f16::from_f32(y.to_f32() + acc[(i % n_out) * rows + i / n_out]);
        });
    }

    /// `acc[o][r] = Σ_k w[o,k]·x[r,k]`, parallel over blocks of `o`.
    ///
    /// A block (not one row) per task: with one row the chunk is `rows` floats —
    /// 840 bytes at a 210-token prefill — and rayon's per-chunk overhead swamps
    /// the dot product itself.
    fn dot_transposed(&self, x: &[f16], rows: usize) -> Vec<f32> {
        const OB: usize = 64;
        let (n_out, cols) = (self.rows, self.cols);
        // Widen the activations **once per op**, not once per output element:
        // the inner loop runs `n_out` times, so leaving `f16 → f32` inside it
        // repeats the (unvectorisable) conversion `n_out` times — measured as
        // the dominant cost after the weights were already f32.
        let mut xs = vec![0.0f32; rows * cols];
        xs.par_chunks_mut(cols)
            .zip(x.par_chunks(cols))
            .for_each(|(dst, src)| {
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = s.to_f32();
                }
            });
        let mut acc = vec![0.0f32; n_out * rows];
        acc.par_chunks_mut(OB * rows).enumerate().for_each(|(bi, block)| {
            let o0 = bi * OB;
            for (oi, orow) in block.chunks_mut(rows).enumerate() {
                let o = o0 + oi;
                let w_row = &self.data[o * cols..(o + 1) * cols];
                for (r, y) in orow.iter_mut().enumerate() {
                    let x_row = &xs[r * cols..(r + 1) * cols];
                    // Four accumulators: a single `s` serialises the FMA chain
                    // on its latency (~4 cycles), which measured 8x off this
                    // machine's f32 throughput.  This changes the summation
                    // order, which the CPU path is allowed to do — the gate is
                    // the transcript, not bit-equality (docs/PORTING.md §1.1).
                    let mut a = [0.0f32; 4];
                    let quads = cols / 4;
                    for q in 0..quads {
                        let k = q * 4;
                        a[0] += w_row[k] * x_row[k];
                        a[1] += w_row[k + 1] * x_row[k + 1];
                        a[2] += w_row[k + 2] * x_row[k + 2];
                        a[3] += w_row[k + 3] * x_row[k + 3];
                    }
                    let mut s = (a[0] + a[1]) + (a[2] + a[3]);
                    for k in quads * 4..cols {
                        s += w_row[k] * x_row[k];
                    }
                    *y = s;
                }
            }
        });
        acc
    }
}

struct Layer {
    iln: Vec<f32>,
    pln: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    q_proj: Mat,
    k_proj: Mat,
    v_proj: Mat,
    o_proj: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
}

impl Layer {
    fn load(w: &HashMap<String, RawTensor>, prefix: &str) -> Result<Self> {
        // The norm weights live in the file as f16 and the GPU consumes them as
        // f16; converting once here keeps the per-element arithmetic identical.
        let vec_f32 = |n: &str| -> Result<Vec<f32>> {
            Ok(weights::get_vector(w, &format!("{prefix}.{n}"))?
                .into_iter()
                .map(|v| v.to_f32())
                .collect())
        };
        Ok(Self {
            iln: vec_f32("input_layernorm.weight")?,
            pln: vec_f32("post_attention_layernorm.weight")?,
            qn: vec_f32("self_attn.q_norm.weight")?,
            kn: vec_f32("self_attn.k_norm.weight")?,
            q_proj: Mat::load(w, &format!("{prefix}.self_attn.q_proj.weight"))?,
            k_proj: Mat::load(w, &format!("{prefix}.self_attn.k_proj.weight"))?,
            v_proj: Mat::load(w, &format!("{prefix}.self_attn.v_proj.weight"))?,
            o_proj: Mat::load(w, &format!("{prefix}.self_attn.o_proj.weight"))?,
            gate: Mat::load(w, &format!("{prefix}.mlp.gate_proj.weight"))?,
            up: Mat::load(w, &format!("{prefix}.mlp.up_proj.weight"))?,
            down: Mat::load(w, &format!("{prefix}.mlp.down_proj.weight"))?,
        })
    }
}

/// The host-side decoder.  Its method set mirrors [`crate::decoder::WgpuTextDecoder`]
/// so the caller can switch between them with one enum.
pub struct CpuTextDecoder {
    cfg: TextConfig,
    embed: Vec<f16>,
    vocab: usize,
    hs: usize,
    layers: Vec<Layer>,
    norm_w: Vec<f32>,
    pub max_seq: usize,
    /// Positions already in the KV cache (the GPU decoder's `pos`).
    pub pos: usize,
    /// The token whose embedding the next [`Self::step`] consumes.
    last_token: u32,
    /// f16-rounded RoPE tables, `[position][head_dim]` — the same values the GPU
    /// uploads, so the rounding matches.
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// `[layer][nkvh][max_seq][hd]` — the GPU caches hold f16; these hold the
    /// **same** values already widened to f32, because both attention loops
    /// read every cached element once per query row and a per-element
    /// `f16 → f32` there is exactly the conversion-bound pathology that cost
    /// 18 s of a 34 s prefill.  Every store goes through `f16::from_f32` first,
    /// so the numbers are bit-identical to the GPU's caches.
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    /// Milliseconds per phase (norm, projections, rope, attention, mlp) —
    /// filled in by [`Self::forward`] and printed by `QASR_CPU_PROFILE=1`.
    pub profile: [f64; 5],
}

impl CpuTextDecoder {
    pub fn load(
        model_dir: &Path,
        prefix: &str,
        cfg: TextConfig,
        max_seq: usize,
        rope_positions: usize,
    ) -> Result<Self> {
        let w = weights::load_tensors(model_dir)?;
        let hs = cfg.hidden_size;
        let (embed, shape) = weights::get_f16(&w, &format!("{prefix}.embed_tokens.weight"))?;
        if shape.len() != 2 || shape[0] != cfg.vocab_size || shape[1] != hs {
            bail!("embed_tokens is {shape:?}, expected [{}, {}]", cfg.vocab_size, hs);
        }
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::load(&w, &format!("{prefix}.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let norm_w: Vec<f32> = weights::get_vector(&w, &format!("{prefix}.norm.weight"))?
            .into_iter()
            .map(|v| v.to_f32())
            .collect();
        let zeros = vec![0.0f32; rope_positions * cfg.head_dim];
        let kv = cfg.num_key_value_heads * max_seq * cfg.head_dim;
        Ok(Self {
            vocab: cfg.vocab_size,
            hs,
            embed,
            layers,
            norm_w,
            max_seq,
            pos: 0,
            last_token: 0,
            cos: zeros.clone(),
            sin: zeros,
            k_cache: vec![0.0; cfg.num_hidden_layers * kv],
            v_cache: vec![0.0; cfg.num_hidden_layers * kv],
            profile: [0.0; 5],
            cfg,
        })
    }

    /// The f16-rounded RoPE tables, exactly as the GPU consumes them.
    pub fn set_rope_tables(&mut self, cos: &[f16], sin: &[f16]) {
        let hd = self.cfg.head_dim;
        self.cos = cos.iter().map(|v| v.to_f32()).collect();
        self.sin = sin.iter().map(|v| v.to_f32()).collect();
        debug_assert_eq!(self.cos.len() % hd, 0);
    }

    fn rope_at(&self, pos: usize, j: usize) -> (f32, f32) {
        let hd = self.cfg.head_dim;
        (self.cos[pos * hd + j], self.sin[pos * hd + j])
    }

    fn embed_row(&self, token: u32, out: &mut [f16]) {
        let row = &self.embed[token as usize * self.hs..(token as usize + 1) * self.hs];
        out.copy_from_slice(row);
    }

    /// rms_norm over one row: f32 reduction, f16 output (the GPU writes f16).
    fn rms_norm_row(&self, x: &[f16], w: &[f32], out: &mut [f16]) {
        let n = x.len();
        let mut sum2 = 0.0f32;
        for v in x {
            let f = v.to_f32();
            sum2 += f * f;
        }
        let inv = 1.0f32 / (sum2 / n as f32 + self.cfg.rms_norm_eps).sqrt();
        for i in 0..n {
            out[i] = f16::from_f32(x[i].to_f32() * inv * w[i]);
        }
    }

    /// Norm + rotate-half RoPE for one head row, mirroring `shaders::qkv_extract`
    /// (`rope_elem`): `inv_rms` comes from the *raw* values without the norm
    /// weight, the weight is applied per element, then the halves rotate.
    fn norm_rope_head(&self, raw: &[f16], nw: &[f32], pos: usize, out: &mut [f16]) {
        let hd = self.cfg.head_dim;
        let half = hd / 2;
        let mut sum2 = 0.0f32;
        for v in raw {
            let f = v.to_f32();
            sum2 += f * f;
        }
        let inv = 1.0f32 / (sum2 / hd as f32 + self.cfg.rms_norm_eps).sqrt();
        for j in 0..hd {
            let xj = raw[j].to_f32() * inv * nw[j];
            let (pj, sign) = if j < half { (j + half, -1.0f32) } else { (j - half, 1.0f32) };
            let xp = raw[pj].to_f32() * inv * nw[pj];
            let (c, s) = self.rope_at(pos, j);
            out[j] = f16::from_f32(xj * c + sign * xp * s);
        }
    }

    /// One attention forward over `rows` query positions starting at `q0`
    /// (the position of the first row), attending to `0..=q` for each row —
    /// the causal mask.  `attn` is `[rows][nqh * hd]`, the layout the o_proj
    /// consumes.
    fn attention(&self, q: &[f16], layer: usize, q0: usize, attn: &mut [f16]) {
        let (nqh, nkvh, hd) = (self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let rep = nqh / nkvh;
        let kv_stride = self.max_seq * hd;
        let base = layer * nkvh * kv_stride;
        let _cur = (q0 + attn.len() / (nqh * hd)).min(self.max_seq);

        // The caches are already f32 (see the field docs), so no loop here
        // converts anything per element: that conversion, once per key per query
        // row, was 18 s of a 34 s prefill.
        //
        // One task per *row* (all heads), so the score/accumulator scratch is
        // allocated once per row instead of once per (row, head).
        // One task per *(row, head)*: a decode step has a single row, so
        // grouping the heads into one task would run the whole scan on one core
        // (measured 136 ms/token).  `map_init` gives each rayon worker its own
        // scratch, so this does not allocate per task either.
        let row_elems = nqh * hd;
        attn.par_chunks_mut(hd)
            .enumerate()
            .map_init(
                || {
                    (
                        vec![0.0f32; self.max_seq],
                        vec![0.0f32; hd],
                        vec![0.0f32; hd],
                    )
                },
                |(scores, acc, qrow), (hi, out)| {
            let (r, h) = (hi / nqh, hi % nqh);
            let pos = q0 + r;
            let valid = (pos + 1).min(self.max_seq);
            {
                let qh = &q[r * row_elems + h * hd..r * row_elems + (h + 1) * hd];
                // The query head is f16 (it comes out of the rope epilogue), so
                // it is widened once per (row, head) — not per key.
                for (d, s) in qrow.iter_mut().zip(qh) {
                    *d = s.to_f32();
                }
                let kh = h / rep;
                let kh_base = base + kh * kv_stride;
                let krow_all = &self.k_cache[kh_base..kh_base + valid * hd];
                let vrow_all = &self.v_cache[kh_base..kh_base + valid * hd];
                // Four accumulators per key row: a single `dot` serialises the
                // FMA chain on ~4-cycle latency, and the decode scan (2307 keys
                // x 128 dims per head) is latency-bound long before it is
                // bandwidth-bound — measured 147 ms/token with one accumulator.
                let mut best = f32::NEG_INFINITY;
                let quads = hd / 4;
                for t in 0..valid {
                    let krow = &krow_all[t * hd..(t + 1) * hd];
                    let mut a = [0.0f32; 4];
                    for q in 0..quads {
                        let j = q * 4;
                        a[0] += qrow[j] * krow[j];
                        a[1] += qrow[j + 1] * krow[j + 1];
                        a[2] += qrow[j + 2] * krow[j + 2];
                        a[3] += qrow[j + 3] * krow[j + 3];
                    }
                    let dot = (a[0] + a[1]) + (a[2] + a[3]);
                    let s = dot * scale;
                    scores[t] = s;
                    if s > best {
                        best = s;
                    }
                }
                let mut sum = 0.0f32;
                for s in scores[..valid].iter_mut() {
                    *s = (*s - best).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for a in acc.iter_mut() {
                    *a = 0.0;
                }
                for (t, sc) in scores[..valid].iter().enumerate() {
                    let w = sc * inv;
                    let vrow = &vrow_all[t * hd..(t + 1) * hd];
                    for j in 0..hd {
                        acc[j] += w * vrow[j];
                    }
                }
                for j in 0..hd {
                    out[j] = f16::from_f32(acc[j]);
                }
            }
        })
        .for_each(|_| {});
    }

    /// One forward over `rows` positions starting at `q0`; `x` is `[rows][hs]`
    /// f16 and is updated in place (residuals included).
    fn forward(&mut self, x: &mut [f16], q0: usize, rows: usize) {
        let hs = self.hs;
        let (nqh, nkvh, hd) = (self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim);
        let (q_dim, kv_dim) = (nqh * hd, nkvh * hd);
        let inter = self.cfg.intermediate_size;
        let kv_stride = self.max_seq * hd;

        let mut normed = vec![f16::from_f32(0.0); rows * hs];
        let mut q = vec![f16::from_f32(0.0); rows * q_dim];
        let mut k = vec![f16::from_f32(0.0); rows * kv_dim];
        let mut v = vec![f16::from_f32(0.0); rows * kv_dim];
        let mut attn = vec![f16::from_f32(0.0); rows * q_dim];
        let mut norm2 = vec![f16::from_f32(0.0); rows * hs];
        let mut gate = vec![f16::from_f32(0.0); rows * inter];
        let mut up = vec![f16::from_f32(0.0); rows * inter];

        let mut timings = [0.0f64; 5];
        let mut mark = std::time::Instant::now();
        for (li, layer) in self.layers.iter().enumerate() {
            // 1. input norm
            for r in 0..rows {
                let (src, dst) = (&x[r * hs..(r + 1) * hs], &mut normed[r * hs..(r + 1) * hs]);
                self.rms_norm_row(src, &layer.iln, dst);
            }
            timings[0] += mark.elapsed().as_secs_f64() * 1000.0;
            mark = std::time::Instant::now();
            // 2. q / k / v projections (separate matrices here; the GPU fuses
            //    them into one GEMM, which changes nothing but the launch count)
            layer.q_proj.batch_dot(&normed, rows, &mut q);
            layer.k_proj.batch_dot(&normed, rows, &mut k);
            layer.v_proj.batch_dot(&normed, rows, &mut v);
            timings[1] += mark.elapsed().as_secs_f64() * 1000.0;
            mark = std::time::Instant::now();

            // 3. q/k norm + rope, v straight into the cache
            let k_layer = li * nkvh * kv_stride;
            let v_off = li * nkvh * kv_stride;
            let mut raw = vec![f16::from_f32(0.0); hd];
            let mut tmp = vec![f16::from_f32(0.0); hd];
            for r in 0..rows {
                let pos = q0 + r;
                for h in 0..nqh {
                    let off = r * q_dim + h * hd;
                    raw.copy_from_slice(&q[off..off + hd]);
                    self.norm_rope_head(&raw, &layer.qn, pos, &mut tmp);
                    q[off..off + hd].copy_from_slice(&tmp);
                }
                for h in 0..nkvh {
                    let off = r * kv_dim + h * hd;
                    raw.copy_from_slice(&k[off..off + hd]);
                    self.norm_rope_head(&raw, &layer.kn, pos, &mut tmp);
                    let dst = k_layer + h * kv_stride + pos * hd;
                    for (d, t) in self.k_cache[dst..dst + hd].iter_mut().zip(&tmp) {
                        *d = t.to_f32(); // `tmp` is already f16-rounded
                    }
                    let dv = v_off + h * kv_stride + pos * hd;
                    for (d, s) in self.v_cache[dv..dv + hd].iter_mut().zip(&v[off..off + hd]) {
                        *d = s.to_f32();
                    }
                }
            }

            timings[2] += mark.elapsed().as_secs_f64() * 1000.0;
            mark = std::time::Instant::now();
            // 4. causal attention
            self.attention(&q, li, q0, &mut attn);
            timings[3] += mark.elapsed().as_secs_f64() * 1000.0;
            mark = std::time::Instant::now();

            // 5. o_proj + residual
            layer.o_proj.batch_dot_acc(&attn, rows, x);
            timings[1] += mark.elapsed().as_secs_f64() * 1000.0;
            mark = std::time::Instant::now();

            // 6. post-attention norm
            for r in 0..rows {
                let (src, dst) = (&x[r * hs..(r + 1) * hs], &mut norm2[r * hs..(r + 1) * hs]);
                self.rms_norm_row(src, &layer.pln, dst);
            }

            // 7-9. gate/up, silu, down + residual
            layer.gate.batch_dot(&norm2, rows, &mut gate);
            layer.up.batch_dot(&norm2, rows, &mut up);
            for (gr, ur) in gate.chunks_mut(inter).zip(up.chunks(inter)) {
                for i in 0..inter {
                    let g = gr[i].to_f32();
                    let silu = g / (1.0 + (-g).exp());
                    gr[i] = f16::from_f32(silu * ur[i].to_f32());
                }
            }
            layer.down.batch_dot_acc(&gate, rows, x);
            timings[4] += mark.elapsed().as_secs_f64() * 1000.0;
            mark = std::time::Instant::now();
        }
        for (a, b) in self.profile.iter_mut().zip(timings) {
            *a += b;
        }
    }

    fn final_norm_logits(&self, x: &[f16]) -> Vec<f32> {
        let hs = self.hs;
        let mut normed = vec![f16::from_f32(0.0); hs];
        self.rms_norm_row(&x[x.len() - hs..], &self.norm_w, &mut normed);
        let mut logits = vec![0.0f32; self.vocab];
        logits.par_iter_mut().enumerate().for_each(|(t, out)| {
            let row = &self.embed[t * hs..(t + 1) * hs];
            let mut acc = 0.0f32;
            for (wv, xv) in row.iter().zip(&normed) {
                acc += wv.to_f32() * xv.to_f32();
            }
            *out = acc;
        });
        logits
    }

    fn argmax(&self, logits: &[f32]) -> i32 {
        let mut best = 0usize;
        for (i, v) in logits.iter().enumerate() {
            if *v > logits[best] {
                best = i;
            }
        }
        best as i32
    }

    /// The GPU decoder's `prefill`: run `s` positions of f16 hidden states
    /// (little-endian words, `[s][hs]`) through every layer, then return the
    /// argmax of the last position.
    pub fn prefill(&mut self, hidden_words: &[u8], s: usize, kv_start: usize) -> Result<i32> {
        let hs = self.hs;
        anyhow::ensure!(hidden_words.len() >= s * hs * 2, "prefill hidden size mismatch");
        anyhow::ensure!(
            kv_start + s <= self.max_seq,
            "prefill needs {} positions, max_seq is {}",
            kv_start + s,
            self.max_seq
        );
        let mut x: Vec<f16> = hidden_words[..s * hs * 2]
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]))
            .collect();
        let t0 = std::time::Instant::now();
        self.forward(&mut x, kv_start, s);
        self.pos = kv_start + s;
        if std::env::var("QASR_CPU_PROFILE").is_ok() {
            let [norm, proj, rope, attn, mlp] = self.profile;
            eprintln!(
                "[cpu] prefill {} rows: norm {norm:.0} / proj {proj:.0} / rope {rope:.0} / attn {attn:.0} / mlp {mlp:.0} ms (forward {:.0} ms)",
                s,
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        let logits = self.final_norm_logits(&x);
        let tok = self.argmax(&logits);
        self.last_token = tok as u32;
        Ok(tok)
    }

    /// As [`Self::prefill`], but hands back the post-layer hidden states instead
    /// of running the head.
    ///
    /// The aligner reads its timestamps off *every* `<timestamp>` row, not just
    /// the last one, so it needs all `s` rows of `[s, hidden]` f16 — the same
    /// thing the GPU decoder exposes as `debug_prefill_h`.  `prefill` is left
    /// untouched: it is the path the ASR gate was verified on, and the layer
    /// loop it shares (`forward`) is the same code either way.
    pub fn prefill_hidden(
        &mut self,
        hidden_words: &[u8],
        s: usize,
        kv_start: usize,
    ) -> Result<Vec<f16>> {
        let hs = self.hs;
        anyhow::ensure!(hidden_words.len() >= s * hs * 2, "prefill hidden size mismatch");
        anyhow::ensure!(
            kv_start + s <= self.max_seq,
            "prefill needs {} positions, max_seq is {}",
            kv_start + s,
            self.max_seq
        );
        let mut x: Vec<f16> = hidden_words[..s * hs * 2]
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]))
            .collect();
        let t0 = std::time::Instant::now();
        self.forward(&mut x, kv_start, s);
        self.pos = kv_start + s;
        if std::env::var("QASR_CPU_PROFILE").is_ok() {
            let [norm, proj, rope, attn, mlp] = self.profile;
            eprintln!(
                "[cpu] prefill_hidden {s} rows: norm {norm:.0} / proj {proj:.0} / rope {rope:.0} / attn {attn:.0} / mlp {mlp:.0} ms (forward {:.0} ms)",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(x)
    }

    /// The GPU decoder's `step`: embed the previous token, run one position.
    pub fn step(&mut self) -> Result<i32> {
        let hs = self.hs;
        anyhow::ensure!(self.pos < self.max_seq, "decode past max_seq {}", self.max_seq);
        let mut x = vec![f16::from_f32(0.0); hs];
        self.embed_row(self.last_token, &mut x);
        let q0 = self.pos;
        self.forward(&mut x, q0, 1);
        self.pos += 1;
        let logits = self.final_norm_logits(&x);
        let tok = self.argmax(&logits);
        self.last_token = tok as u32;
        Ok(tok)
    }

    /// Tokens per second this decoder can be expected to reach — not measured,
    /// just a shape sanity check used by the CLI.
    pub fn describe(&self) -> String {
        format!(
            "cpu decoder: {} layers, hidden {}, {} heads ({} kv), max_seq {}, f16 weights",
            self.cfg.num_hidden_layers,
            self.hs,
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.max_seq,
        )
    }
}

/// f16 activations → f32, once per op (see [`Mat::dot_transposed`] for why the
/// conversion must not sit in an inner loop).
fn widen_batch(x: &[f16], rows: usize, cols: usize) -> Vec<f32> {
    let mut xs = vec![0.0f32; rows * cols];
    xs.par_chunks_mut(cols)
        .zip(x.par_chunks(cols))
        .for_each(|(dst, src)| {
            for (d, s) in dst.iter_mut().zip(src) {
                *d = s.to_f32();
            }
        });
    xs
}

/// `out[r][o] = Σ_k x[r][k]·w[o][k]` (row-major throughout, `w` = `[n, k]`), via
/// the `gemm` crate's microkernels with every core forced on.
///
/// Two traps, both measured: the two scalars before the strides are
/// **`(beta, alpha)`** (the crate's own test compares `gemm` against
/// `gemm_fallback` through the same wrapper, so a swapped pair passes there),
/// and the strides have to be verified at *production* shapes — small shapes
/// take a non-packing path, so the unit test below runs both.
fn gemm_row_major(out: &mut [f32], x: &[f32], w: &[f32], m: usize, n: usize, k: usize, beta: f32) {
    debug_assert_eq!(out.len(), m * n);
    debug_assert_eq!(x.len(), m * k);
    debug_assert_eq!(w.len(), n * k);
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,          // dst: column stride (row-major C)
            n as isize, // dst: row stride
            beta != 0.0,
            x.as_ptr(),
            1,          // lhs: column stride
            k as isize, // lhs: row stride
            w.as_ptr(),
            k as isize, // rhs: column stride (B is Wᵀ; j+1 advances by k in W)
            1,          // rhs: row stride (the k axis of W[n][k])
            beta,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the `gemm` wrapper against a hand-written triple loop — at a toy
    /// shape *and* the production ones, because the crate branches (packing,
    /// threading thresholds) differently there and a stride that is right for
    /// 3x5x4 was measured to produce garbage at 210x1280x1024.
    #[test]
    fn gemm_row_major_matches_naive() {
        for &(m, n, k) in &[
            (3usize, 5usize, 4usize),
            (8, 8, 8),
            (4, 64, 32),
            (210, 1280, 1024),
            (64, 3072, 1024),
            (64, 1024, 3072),
        ] {
            let x: Vec<f32> = (0..m * k).map(|i| ((i * 13 % 61) as f32) * 0.03 - 0.9).collect();
            let w: Vec<f32> = (0..n * k).map(|i| ((i * 7 % 53) as f32) * 0.02 - 0.5).collect();
            let mut got = vec![0.0f32; m * n];
            gemm_row_major(&mut got, &x, &w, m, n, k, 0.0);
            let mut worst = 0.0f32;
            for r in 0..m {
                for o in 0..n {
                    let mut s = 0.0f32;
                    for p in 0..k {
                        s += x[r * k + p] * w[o * k + p];
                    }
                    worst = worst.max((got[r * n + o] - s).abs());
                }
            }
            assert!(worst < 1e-3, "shape {m}x{n}x{k}: worst |Δ| = {worst}");
        }
    }
}

/// Reject a config this decoder cannot run, with a legible message rather than
/// an index panic later.  Called from the loader, so a bad config fails before
/// any allocation.
pub fn check_config(cfg: &TextConfig) -> Result<()> {
    if cfg.vocab_size == 0 || cfg.hidden_size == 0 || cfg.num_hidden_layers == 0 {
        return Err(anyhow!("degenerate text config {cfg:?}"));
    }
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
        return Err(anyhow!(
            "{} heads are not a multiple of {} kv heads",
            cfg.num_attention_heads,
            cfg.num_key_value_heads
        ));
    }
    Ok(())
}

