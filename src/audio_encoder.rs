#![allow(dead_code)]

//! CPU audio encoder for Qwen3-ASR — f32 GEMM (`gemm` crate + rayon), weights
//! converted from f16 once at load.  Architecture:
//!
//!   mel [T_mel, 128]  →  conv2d stem (3 × {im2col + gemm + bias + GELU})
//!   → conv_out (Linear + bias)  →  + sinusoidal PE
//!   → 18 × { LN + windowed attn + LN + FFN(GELU-erf) }
//!   → ln_post + proj1 + GELU + proj2  → [n_total, output_dim]
use anyhow::Result;
use gemm::{gemm, Parallelism};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::AudioEncoderConfig;
use crate::cpu_tensor::{linear, CpuTensor, CpuWeightF16};
use crate::weights::RawTensor;

// ─── Phase profiling (QWEN3_ENC_PROFILE=1) ─────────────────────────
//
// Buckets are filled only when `QWEN3_ENC_PROFILE` is set (`profiling()`), so
// the hot path pays one relaxed bool load per bucket.

static PROF_NS: [AtomicU64; 16] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];

pub(crate) const P_CONV_CHUNK: usize = 0;
pub(crate) const P_CONV1: usize = 1;
pub(crate) const P_CONV2: usize = 2;
pub(crate) const P_CONV3: usize = 3;
pub(crate) const P_PERM_CO: usize = 4;
pub(crate) const P_PACK: usize = 5;
pub(crate) const P_LN1: usize = 6;
pub(crate) const P_QKV: usize = 7;
pub(crate) const P_ATTN: usize = 8;
pub(crate) const P_OUTPROJ: usize = 9;
pub(crate) const P_LN2_FFN: usize = 10;
pub(crate) const P_FINAL: usize = 11;
pub(crate) const P_IM2COL: usize = 12;
pub(crate) const P_GEMM: usize = 13;
pub(crate) const P_EPILOGUE: usize = 14;

pub(crate) const PROF_NAMES: [&str; 15] = [
    "chunk-mel", "conv1", "conv2", "conv3", "perm+conv_out+pe",
    "pack-tokens", "attn-layernorm", "qkv-proj", "attention", "out-proj",
    "ffn(+layernorm)", "ln_post+proj1+proj2", "  └ im2col", "  └ conv gemm",
    "  └ conv bias+gelu",
];

#[inline]
pub(crate) fn profiling() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN3_ENC_PROFILE").map(|v| v != "0").unwrap_or(false))
}

#[inline]
pub(crate) fn prof_add(bucket: usize, t: std::time::Instant) {
    if profiling() {
        PROF_NS[bucket].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

pub(crate) fn prof_reset() {
    for a in PROF_NS.iter() {
        a.store(0, Ordering::Relaxed);
    }
}

/// Public hook for the end-to-end binary: print the last `forward`'s phase split.
pub fn profile_report() {
    prof_report();
}

pub fn profiling_enabled() -> bool {
    profiling()
}

pub(crate) fn prof_report() {
    let total: u64 = PROF_NS.iter().map(|a| a.load(Ordering::Relaxed)).sum();
    if total == 0 {
        return;
    }
    eprintln!("-- cpu audio encoder phase profile --");
    for (i, name) in PROF_NAMES.iter().enumerate() {
        let ns = PROF_NS[i].load(Ordering::Relaxed);
        eprintln!(
            "  {:<22} {:>9.1} ms  {:>5.1}%",
            name,
            ns as f64 / 1e6,
            100.0 * ns as f64 / total as f64
        );
    }
    eprintln!("  {:<22} {:>9.1} ms", "SUM(buckets)", total as f64 / 1e6);
}

// ─── Linear + LayerNorm primitives ─────────────────────────────────

pub(crate) struct CpuAudioLinear {
    pub w_f32: crate::cpu_tensor::CpuWeight,
    pub bias: Option<Vec<f32>>,
}

impl CpuAudioLinear {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
    ) -> Result<Self> {
        let (data, shape) = weights
            .get(&format!("{}.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("weight not found: {}.weight", prefix))?
            .as_f16()?;
        let rows = shape[0];
        let cols = shape[1];
        let bias = if weights.contains_key(&format!("{}.bias", prefix)) {
            let (b, _) = weights
                .get(&format!("{}.bias", prefix))
                .unwrap()
                .as_f32()?;
            Some(b)
        } else {
            None
        };
        let w_f16 = CpuWeightF16 { data, rows, cols };
        Ok(Self { w_f32: w_f16.to_f32(), bias })
    }

    /// x: [..., in_features] → [..., out_features]  (bias added if present)
    pub(crate) fn forward(&self, x: &CpuTensor) -> Result<CpuTensor> {
        let mut y = linear(x, &self.w_f32);
        if let Some(b) = &self.bias {
            let last = y.shape.last().unwrap();
            y.data.par_chunks_mut(*last).for_each(|row| {
                for j in 0..row.len() { row[j] += b[j]; }
            });
        }
        Ok(y)
    }
}

pub(crate) struct CpuAudioLayerNorm {
    pub w: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f32,
}

impl CpuAudioLayerNorm {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        eps: f32,
    ) -> Result<Self> {
        let (w, _) = weights
            .get(&format!("{}.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("ln weight not found: {}.weight", prefix))?
            .as_f32()?;
        let (bias, _) = weights
            .get(&format!("{}.bias", prefix))
            .ok_or_else(|| anyhow::anyhow!("ln bias not found: {}.bias", prefix))?
            .as_f32()?;
        Ok(Self { w, bias, eps })
    }

    /// x: [outer, d] → [outer, d]  (LayerNorm over last dim)
    pub(crate) fn forward(&self, x: &CpuTensor) -> CpuTensor {
        let d = *x.shape.last().unwrap();
        let w = &self.w;
        let bias = &self.bias;
        let eps = self.eps;
        let mut out = vec![0.0f32; x.data.len()];
        out.par_chunks_mut(d)
            .zip(x.data.par_chunks(d))
            .for_each(|(o_row, x_row)| {
                let mut mean = 0.0f32;
                for &v in x_row { mean += v; }
                mean /= d as f32;
                let mut var = 0.0f32;
                for &v in x_row { let d_ = v - mean; var += d_ * d_; }
                var /= d as f32;
                let inv_std = 1.0 / (var + eps).sqrt();
                for j in 0..d {
                    o_row[j] = (x_row[j] - mean) * inv_std * w[j] + bias[j];
                }
            });
        CpuTensor::new(out, x.shape.clone())
    }
}

/// GELU, erf form (not tanh).
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub(crate) fn gelu_inplace(x: &mut CpuTensor) {
    x.data.par_iter_mut().for_each(|v| *v = gelu(*v));
}

/// Standard im2col for conv2d(kernel=3, stride=2, pad=1).
/// Input x: [b, c_in, h, w]; output `[b*h_out*w_out, c_in*9]` row-major, with
/// `col[ic*9 + kh*3 + kw]` matching weight `[c_out, c_in, 3, 3]` flattened.
pub(crate) fn im2col_3x3_s2p1(x: &[f32], b: usize, c_in: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let h_out = (h + 2 - 3) / 2 + 1;
    let w_out = (w + 2 - 3) / 2 + 1;
    let col_count = b * h_out * w_out;
    let k = c_in * 9;
    let mut cols = vec![0.0f32; col_count * k];
    cols.par_chunks_mut(k)
        .enumerate()
        .for_each(|(col_idx, col)| {
            let ib = col_idx / (h_out * w_out);
            let rem = col_idx % (h_out * w_out);
            let ho = rem / w_out;
            let wo = rem % w_out;
            for ic in 0..c_in {
                for kh in 0..3 {
                    for kw in 0..3 {
                        let ih = (ho * 2 + kh) as isize - 1;
                        let iw = (wo * 2 + kw) as isize - 1;
                        let v = if ih < 0 || ih >= h as isize || iw < 0 || iw >= w as isize {
                            0.0
                        } else {
                            unsafe { *x.get_unchecked(((ib * c_in + ic) * h + ih as usize) * w + iw as usize) }
                        };
                        col[ic * 9 + kh * 3 + kw] = v;
                    }
                }
            }
        });
    (cols, h_out, w_out)
}

#[allow(dead_code)]
pub(crate) struct CpuConvStem {
    c1_w: crate::cpu_tensor::CpuWeight, c1_b: Vec<f32>,
    c2_w: crate::cpu_tensor::CpuWeight, c2_b: Vec<f32>,
    c3_w: crate::cpu_tensor::CpuWeight, c3_b: Vec<f32>,
    co: CpuAudioLinear,
    pe: Vec<f32>,        // [max_pos, d_model] f32
    d_model: usize,
    max_pos: usize,
}

impl CpuConvStem {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        config: &AudioEncoderConfig,
    ) -> Result<Self> {
        let c1_w = load_conv_weight(weights, &format!("{}.conv2d1.weight", prefix))?.to_f32();
        let c1_b = load_bias(weights, &format!("{}.conv2d1.bias", prefix))?;
        let c2_w = load_conv_weight(weights, &format!("{}.conv2d2.weight", prefix))?.to_f32();
        let c2_b = load_bias(weights, &format!("{}.conv2d2.bias", prefix))?;
        let c3_w = load_conv_weight(weights, &format!("{}.conv2d3.weight", prefix))?.to_f32();
        let c3_b = load_bias(weights, &format!("{}.conv2d3.bias", prefix))?;
        let co = CpuAudioLinear::load(weights, &format!("{}.conv_out", prefix))?;

        // Sinusoidal PE: sin in the first half of the channels, cos in the rest.
        let dm = config.d_model;
        let max_pos = config.max_source_positions;
        let half = dm / 2;
        let lt = (10000.0f64).ln() / (half as f64 - 1.0);
        let mut pe = vec![0.0f32; max_pos * dm];
        for p in 0..max_pos {
            for i in 0..half {
                let a = p as f64 * (-(i as f64) * lt).exp();
                pe[p * dm + i] = a.sin() as f32;
                pe[p * dm + half + i] = a.cos() as f32;
            }
        }
        Ok(Self { c1_w, c1_b, c2_w, c2_b, c3_w, c3_b, co, pe, d_model: dm, max_pos })
    }

    /// Run conv stem on chunked mel input, NCHW `[b_chunks, 1, n_mels, cs]`,
    /// flat `[b_chunks * n_mels * cs]` in (chunk, mel_bin, frame) order.
    /// Returns ([b_chunks * t2 * d_model], t2).
    pub(crate) fn forward(
        &self,
        mel_chunks: &[f32],
        b_chunks: usize,
        n_mels: usize,
        cs: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let (_, out, t2) = self.forward_stages(mel_chunks, b_chunks, n_mels, cs)?;
        Ok((out, t2))
    }

    /// As [`Self::forward`], also returning the `conv_out` operand — the permuted
    /// `[chunk·t2][c·f]` rows — without going through the projection.
    pub(crate) fn forward_stages(
        &self,
        mel_chunks: &[f32],
        b_chunks: usize,
        n_mels: usize,
        cs: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        let c1_out = self.c1_w.rows;
        let c2_out = self.c2_w.rows;

        let _t = std::time::Instant::now();
        let (x1, h1, w1) = self.conv_block(mel_chunks, b_chunks, 1, n_mels, cs, &self.c1_w, &self.c1_b)?;
        prof_add(P_CONV1, _t);
        let _t = std::time::Instant::now();
        let (x2, h2, w2) = self.conv_block(&x1, b_chunks, c1_out, h1, w1, &self.c2_w, &self.c2_b)?;
        prof_add(P_CONV2, _t);
        let _t = std::time::Instant::now();
        let (x3, h3, w3) = self.conv_block(&x2, b_chunks, c2_out, h2, w2, &self.c3_w, &self.c3_b)?;
        prof_add(P_CONV3, _t);
        // x3: [b_chunks, c3_out, h3, w3]
        let t2 = w3;
        let _t = std::time::Instant::now();

        // Permute [b, c, f, t] → [b, t, c, f] then reshape [b, t, c*f].
        let c_dim = self.c3_w.rows;
        let f_dim = h3;
        let row_len = c_dim * f_dim;
        let mut perm = vec![0.0f32; b_chunks * t2 * c_dim * f_dim];
        perm.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(idx, chunk)| {
                let ib = idx / t2;
                let it = idx % t2;
                for ic in 0..c_dim {
                    for f in 0..f_dim {
                        let src = ((ib * c_dim + ic) * f_dim + f) * t2 + it;
                        let dst = ic * f_dim + f;
                        chunk[dst] = x3[src];
                    }
                }
            });

        // ConvOut (Linear): [b, t2, c*f] → [b, t2, d_model]
        let perm_flat = perm.clone();
        let perm_t = CpuTensor::new(perm, vec![b_chunks, t2, c_dim * f_dim]);
        let co_out = self.co.forward(&perm_t)?;

        // Add PE (broadcast over batch).
        let mut out = co_out.data.clone();
        let dm = self.d_model;
        let pe = &self.pe;
        out.par_chunks_mut(dm)
            .enumerate()
            .for_each(|(idx, chunk)| {
                let it = idx % t2;
                let pe_base = it * dm;
                for j in 0..dm {
                    chunk[j] += pe[pe_base + j];
                }
            });
        prof_add(P_PERM_CO, _t);
        Ok((perm_flat, out, t2))
    }

    /// Single conv2d (3×3, stride 2, pad 1) + bias + GELU.
    /// Input x: [b, c_in, h, w]. Returns (flat [b, c_out, h_out, w_out], h_out, w_out).
    pub(crate) fn conv_block(
        &self,
        x: &[f32],
        b: usize, c_in: usize, h: usize, w: usize,
        w_w: &crate::cpu_tensor::CpuWeight, w_b: &[f32],
    ) -> Result<(Vec<f32>, usize, usize)> {
        let (_, act, ho, wo) = self.conv_block_stages(x, b, c_in, h, w, w_w, w_b, false)?;
        Ok((act, ho, wo))
    }

    /// As [`Self::conv_block`], also returning `raw` (the GEMM output
    /// `[c_out][n_chunks·plane]`, channel-major, pre-bias/GELU), `act` (the same,
    /// post-bias/GELU) and `cols` (the im2col operand `[n_chunks·plane][k]`,
    /// `k = ic·9 + kh·3 + kw`).
    #[allow(clippy::type_complexity)]
    pub(crate) fn conv_block_stages(
        &self,
        x: &[f32],
        b: usize, c_in: usize, h: usize, w: usize,
        w_w: &crate::cpu_tensor::CpuWeight, w_b: &[f32],
        want_raw: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
        let c_out = w_w.rows;
        assert_eq!(x.len(), b * c_in * h * w, "conv_block input size mismatch");
        let h_out = (h + 2 - 3) / 2 + 1;
        let w_out = (w + 2 - 3) / 2 + 1;
        let k = c_in * 9;
        assert_eq!(w_w.cols, k, "conv_block weight cols={} != c_in*9={}", w_w.cols, k);
        let plane = h_out * w_out;
        let in_plane = c_in * h * w;
        // `raw` is materialised only for the diagnostic path.
        let mut raw = if want_raw { vec![0.0f32; c_out * b * plane] } else { Vec::new() };
        let mut act = vec![0.0f32; b * c_out * plane];
        // Tile the batch so conv2/conv3's im2col stays cache-sized.
        let tile = CONV_TILE.load(Ordering::Relaxed).max(1) as usize;
        for b0 in (0..b).step_by(tile) {
            let nb = tile.min(b - b0);
            let _t = std::time::Instant::now();
            let (cols, ho, wo) = im2col_3x3_s2p1(
                &x[b0 * in_plane..(b0 + nb) * in_plane],
                nb, c_in, h, w,
            );
            prof_add(P_IM2COL, _t);
            debug_assert_eq!((ho, wo), (h_out, w_out));
            let col_count = nb * plane;
            // When `want_raw`, the GEMM writes straight into `raw`, laid out
            // `[c_out][b·plane]` (channel-major); otherwise it uses a tile-sized
            // scratch.
            let mut scratch = vec![0.0f32; if want_raw { 0 } else { c_out * col_count }];
            let (out_ptr, cstride) = if want_raw {
                (unsafe { raw.as_mut_ptr().add(b0 * plane) }, (b * plane) as isize)
            } else {
                (scratch.as_mut_ptr(), col_count as isize)
            };
            let _t = std::time::Instant::now();
            unsafe {
                gemm(
                    c_out, col_count, k,
                    out_ptr, 1, cstride,
                    false,
                    w_w.data.as_ptr(), 1, k as isize,
                    cols.as_ptr(), k as isize, 1,
                    0.0, 1.0, false, false, false,
                    Parallelism::Rayon(0),
                );
            }
            prof_add(P_GEMM, _t);
            let _t = std::time::Instant::now();
            let src_base = if want_raw { &raw[..] } else { &scratch[..] };
            let src_row_stride = if want_raw { b * plane } else { col_count };
            let dst = &mut act[b0 * c_out * plane..(b0 + nb) * c_out * plane];
            dst.par_chunks_mut(plane).enumerate().for_each(|(plane_idx, plane_dst)| {
                let ib = plane_idx / c_out;
                let oc = plane_idx % c_out;
                let bias = w_b[oc];
                // The `want_raw` source is the whole `[c_out][b·plane]` tensor,
                // so this tile's chunk is `b0 + ib`.
                let src_chunk = if want_raw { b0 + ib } else { ib };
                let src_row = oc * src_row_stride + src_chunk * plane;
                for i in 0..plane {
                    plane_dst[i] = gelu(src_base[src_row + i] + bias);
                }
            });
            prof_add(P_EPILOGUE, _t);
        }
        Ok((raw, act, h_out, w_out))
    }
}

/// Batch tile used by `conv_block` for the im2col buffer: `TILE * h_out * w_out
/// * c_in * 9` f32 elements.
pub static CONV_TILE: AtomicU64 = AtomicU64::new(8);

fn load_conv_weight(weights: &HashMap<String, RawTensor>, name: &str) -> Result<CpuWeightF16> {
    let (data, shape) = weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))?.as_f16()?;
    let c_out = shape[0];
    let k = shape[1..].iter().product::<usize>();
    Ok(CpuWeightF16 { data, rows: c_out, cols: k })
}

fn load_bias(weights: &HashMap<String, RawTensor>, name: &str) -> Result<Vec<f32>> {
    let (data, _) = weights.get(name).ok_or_else(|| anyhow::anyhow!("bias not found: {}", name))?.as_f32()?;
    Ok(data)
}

pub(crate) struct CpuAudioAttention {
    q_proj: CpuAudioLinear,
    k_proj: CpuAudioLinear,
    v_proj: CpuAudioLinear,
    out_proj: CpuAudioLinear,
    num_heads: usize,
    head_dim: usize,
}

impl CpuAudioAttention {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        nh: usize,
        d_model: usize,
    ) -> Result<Self> {
        let hd = d_model / nh;
        Ok(Self {
            q_proj: CpuAudioLinear::load(weights, &format!("{}.q_proj", prefix))?,
            k_proj: CpuAudioLinear::load(weights, &format!("{}.k_proj", prefix))?,
            v_proj: CpuAudioLinear::load(weights, &format!("{}.v_proj", prefix))?,
            out_proj: CpuAudioLinear::load(weights, &format!("{}.out_proj", prefix))?,
            num_heads: nh,
            head_dim: hd,
        })
    }

    /// Windowed attention flattened to `[b, s, nh·hd]` — everything `forward`
    /// does before `out_proj`.
    pub(crate) fn flat(
        &self,
        x: &CpuTensor,
        ws: Option<usize>,
    ) -> Result<CpuTensor> {
        let b = x.shape[0];
        let s = x.shape[1];
        let dm = x.shape[2];
        let nh = self.num_heads;
        let hd = self.head_dim;

        // Project Q, K, V.
        let _t = std::time::Instant::now();
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        prof_add(P_QKV, _t);

        let scale = 1.0f32 / (hd as f32).sqrt();
        let window = ws.filter(|&w| w > 0 && w < s);

        let _t = std::time::Instant::now();
        let attn_out = if let Some(w) = window {
            // Windowed: process chunks of `w` tokens, each chunk attends only to itself.
            let mut out = vec![0.0f32; b * nh * s * hd];
            for st in (0..s).step_by(w) {
                let len = w.min(s - st);
                let o = attention_window(&q.data, &k.data, &v.data, b, nh, s, dm, hd, len, st, scale);
                // Scatter: out[(ib, ih, st:st+len, hd)] = o[(ib, ih, 0:len, hd)]
                out.par_chunks_mut(s * hd)
                    .zip(o.par_chunks(len * hd))
                    .for_each(|(dst, src)| {
                        let dst_off = st * hd;
                        dst[dst_off..dst_off + len * hd].copy_from_slice(src);
                    });
            }
            out
        } else {
            // Full attention.
            attention_window(&q.data, &k.data, &v.data, b, nh, s, dm, hd, s, 0, scale)
        };

        // Reshape [b, nh, s, hd] → [b, s, nh*hd] (swap dims 1 and 2).
        let flat = {
            let mut out = vec![0.0f32; b * s * nh * hd];
            out.par_chunks_mut(nh * hd)
                .enumerate()
                .for_each(|(idx, chunk)| {
                    let ib = idx / s;
                    let is_ = idx % s;
                    for ih in 0..nh {
                        let src_off = ((ib * nh + ih) * s + is_) * hd;
                        let dst_off = ih * hd;
                        chunk[dst_off..dst_off + hd].copy_from_slice(&attn_out[src_off..src_off + hd]);
                    }
                });
            out
        };
        prof_add(P_ATTN, _t);
        Ok(CpuTensor::new(flat, vec![b, s, nh * hd]))
    }

    /// x [b, s, d_model] → [b, s, d_model]. ws: window size for attention.
    pub(crate) fn forward(
        &self,
        x: &CpuTensor,
        ws: Option<usize>,
    ) -> Result<CpuTensor> {
        let attn_flat = self.flat(x, ws)?;
        let _t = std::time::Instant::now();
        let r = self.out_proj.forward(&attn_flat);
        prof_add(P_OUTPROJ, _t);
        r
    }
}

/// Scalar attention with rayon parallelism across (batch, head) pairs.
/// q/k/v: [b, s, dm] flat. Processes positions [st, st+len).
/// Returns [b*nh, len*hd] where each chunk is one (batch, head) pair's output for the window.
fn attention_window(
    q: &[f32], k: &[f32], v: &[f32],
    b: usize, nh: usize, s: usize, dm: usize, hd: usize, len: usize, st: usize, scale: f32,
) -> Vec<f32> {
    let bn = b * nh;
    let chunk_len = len * hd;
    let mut out = vec![0.0f32; bn * chunk_len];
    out.par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(idx, out_chunk)| {
            let ib = idx / nh;
            let ih = idx % nh;
            let head_off = ih * hd;
            let mut scores = vec![0.0f32; len];
            for qi in 0..len {
                let q_pos = st + qi;
                let q_base = (ib * s + q_pos) * dm + head_off;
                let mut max_s = f32::NEG_INFINITY;
                for ki in 0..len {
                    let k_pos = st + ki;
                    let k_base = (ib * s + k_pos) * dm + head_off;
                    let mut dot = 0.0f32;
                    for jd in 0..hd {
                        unsafe {
                            dot += *q.get_unchecked(q_base + jd) * *k.get_unchecked(k_base + jd);
                        }
                    }
                    let s_ = dot * scale;
                    scores[ki] = s_;
                    if s_ > max_s { max_s = s_; }
                }
                // Softmax.
                let mut sum = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - max_s).exp();
                    sum += *sc;
                }
                let inv = 1.0 / sum;
                // Weighted sum of V.
                let out_base = qi * hd;
                for ki in 0..len {
                    let w = scores[ki] * inv;
                    let k_pos = st + ki;
                    let v_base = (ib * s + k_pos) * dm + head_off;
                    for jd in 0..hd {
                        unsafe {
                            *out_chunk.get_unchecked_mut(out_base + jd) += w * *v.get_unchecked(v_base + jd);
                        }
                    }
                }
            }
        });
    out
}

pub(crate) struct CpuAudioFfn {
    fc1: CpuAudioLinear,
    fc2: CpuAudioLinear,
}

impl CpuAudioFfn {
    pub(crate) fn load(weights: &HashMap<String, RawTensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: CpuAudioLinear::load(weights, &format!("{}.fc1", prefix))?,
            fc2: CpuAudioLinear::load(weights, &format!("{}.fc2", prefix))?,
        })
    }

    pub(crate) fn forward(&self, x: &CpuTensor) -> Result<CpuTensor> {
        let mut h = self.fc1.forward(x)?;
        gelu_inplace(&mut h);
        self.fc2.forward(&h)
    }
}

pub(crate) struct CpuAudioLayer {
    sln: CpuAudioLayerNorm,
    attn: CpuAudioAttention,
    fln: CpuAudioLayerNorm,
    ffn: CpuAudioFfn,
}

impl CpuAudioLayer {
    pub(crate) fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        nh: usize,
        d_model: usize,
    ) -> Result<Self> {
        Ok(Self {
            sln: CpuAudioLayerNorm::load(weights, &format!("{}.self_attn_layer_norm", prefix), 1e-5)?,
            attn: CpuAudioAttention::load(weights, &format!("{}.self_attn", prefix), nh, d_model)?,
            fln: CpuAudioLayerNorm::load(weights, &format!("{}.final_layer_norm", prefix), 1e-5)?,
            ffn: CpuAudioFfn::load(weights, prefix)?,
        })
    }

    /// x: [b, s, d_model] consumed; returns post-residual h of same shape.
    pub(crate) fn forward(&self, x: CpuTensor, ws: Option<usize>) -> Result<CpuTensor> {
        let shape = x.shape.clone();
        let _t = std::time::Instant::now();
        let normed = self.sln.forward(&x);
        prof_add(P_LN1, _t);
        let attn_out = self.attn.forward(&normed, ws)?;
        // Residual add in-place on x.data (no clone needed — x is consumed).
        let mut x_data = x.data;
        x_data.par_iter_mut().zip(&attn_out.data).for_each(|(a, b)| *a += *b);
        let x1 = CpuTensor::new(x_data, shape);
        let _t = std::time::Instant::now();
        let normed2 = self.fln.forward(&x1);
        let ffn_out = self.ffn.forward(&normed2)?;
        prof_add(P_LN2_FFN, _t);
        let mut x2_data = x1.data;
        x2_data.par_iter_mut().zip(&ffn_out.data).for_each(|(a, b)| *a += *b);
        Ok(CpuTensor::new(x2_data, x1.shape))
    }
}

/// `f` folded three times: the token count a chunk of `ifr` frames produces.
pub fn feo(ifr: usize) -> usize {
    let f = |l: usize| -> usize { (l - 1) / 2 + 1 };
    f(f(f(ifr)))
}

/// One conv level's tensors, in the GPU tower's layouts minus its padding, so a
/// stage comparison indexes both sides the same way.
pub struct CpuConvStage {
    /// GEMM output `[c_out][n_chunks·plane]`, pre-bias/GELU (`enc.cN_raw`).
    pub raw: Vec<f32>,
    /// The same, post-bias/GELU (`enc.cN_act`).
    pub act: Vec<f32>,
    /// im2col operand `[n_chunks·plane][k]` (`enc.cN_col` stores its transpose).
    pub cols: Vec<f32>,
    pub k: usize,
    pub c_out: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub plane: usize,
}

/// Every stage of the CPU conv stem, for the GPU tower's `--diag-enc`.
pub struct CpuConvTower {
    pub stages: Vec<CpuConvStage>,
    /// `conv_out` operand `[n_total][c·f]` (`enc.packed`), valid tokens packed.
    pub packed: Vec<f32>,
    /// Post-`conv_out` + PE tokens `[n_total][d_model]` (`enc.h`).
    pub h: Vec<f32>,
    pub n_total: usize,
    pub n_chunks: usize,
}

pub struct CpuAudioEncoder {
    conv_stem: CpuConvStem,
    layers: Vec<CpuAudioLayer>,
    ln_post: CpuAudioLayerNorm,
    proj1: CpuAudioLinear,
    proj2: CpuAudioLinear,
    config: AudioEncoderConfig,
}

impl CpuAudioEncoder {
    pub fn load(
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        config: &AudioEncoderConfig,
    ) -> Result<Self> {
        let dm = config.d_model;
        let nh = config.encoder_attention_heads;
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            layers.push(CpuAudioLayer::load(weights, &format!("{}.layers.{}", prefix, i), nh, dm)?);
        }
        let ln_post = CpuAudioLayerNorm::load(weights, &format!("{}.ln_post", prefix), 1e-5)?;
        let proj1 = CpuAudioLinear::load(weights, &format!("{}.proj1", prefix))?;
        let proj2 = CpuAudioLinear::load(weights, &format!("{}.proj2", prefix))?;
        let conv_stem = CpuConvStem::load(weights, prefix, config)?;
        Ok(Self { conv_stem, layers, ln_post, proj1, proj2, config: config.clone() })
    }

    /// mel: [n_mels * mel_len] flat (mel-bin-major, frame-minor). Returns [n_total, output_dim] flat.
    pub fn forward(&self, mel: &[f32], n_mels: usize, mel_len: usize) -> Result<Vec<f32>> {
        let _t_all = std::time::Instant::now();
        prof_reset();
        let cs = self.config.n_window * 2;
        let tpc = feo(cs);
        let nfull = mel_len / cs;
        let tail = mel_len % cs;
        let n_chunks = nfull + if tail > 0 { 1 } else { 0 };

        // Build chunked mel buffer [n_chunks * n_mels * cs], zero-padded.
        let _t = std::time::Instant::now();
        let mut chunked = vec![0.0f32; n_chunks * n_mels * cs];
        let mut chunk_tokens: Vec<usize> = Vec::with_capacity(n_chunks);
        for i in 0..nfull {
            let s = i * cs;
            for m in 0..n_mels {
                let dst_base = (i * n_mels + m) * cs;
                let src_base = m * mel_len + s;
                for j in 0..cs {
                    chunked[dst_base + j] = mel[src_base + j];
                }
            }
            chunk_tokens.push(tpc);
        }
        if tail > 0 {
            let s = nfull * cs;
            for m in 0..n_mels {
                let dst_base = (nfull * n_mels + m) * cs;
                let src_base = m * mel_len + s;
                for j in 0..tail {
                    chunked[dst_base + j] = mel[src_base + j];
                }
                // Rest is already zero-padded.
            }
            chunk_tokens.push(feo(tail));
        }

        // Conv stem on batched chunks.
        prof_add(P_CONV_CHUNK, _t);
        let (conv_data, t2) = self.conv_stem.forward(&chunked, n_chunks, n_mels, cs)?;
        let dm = self.config.d_model;
        let n_total: usize = chunk_tokens.iter().sum();

        // Pack valid tokens from each chunk into [1, n_total, d_model].
        let _t = std::time::Instant::now();
        let mut packed = Vec::with_capacity(n_total * dm);
        for (idx, &v) in chunk_tokens.iter().enumerate() {
            let base = idx * t2 * dm;
            packed.extend_from_slice(&conv_data[base..base + v * dm]);
        }
        prof_add(P_PACK, _t);

        // Transformer layers.
        let cs2 = self.config.n_window * 2;
        let tpc2 = feo(cs2);
        let cpw = self.config.n_window_infer / cs2;
        let ws = tpc2 * cpw;
        let mut h = CpuTensor::new(packed, vec![1, n_total, dm]);
        for layer in &self.layers {
            h = layer.forward(h, Some(ws))?;
        }

        let _t = std::time::Instant::now();
        let h = self.ln_post.forward(&h);
        let mut h = self.proj1.forward(&h)?;
        gelu_inplace(&mut h);
        let h = self.proj2.forward(&h)?;
        prof_add(P_FINAL, _t);
        if profiling() {
            eprintln!(
                "encoder: {} chunks, {} tokens, wall {:.1} ms",
                n_chunks,
                n_total,
                _t_all.elapsed().as_secs_f64() * 1000.0
            );
            prof_report();
        }
        Ok(h.data)
    }

    /// Run conv-stem on a single mel chunk of exactly `cs` frames.
    /// mel_chunk: [n_mels * cs] flat (mel-bin-major). cs must equal n_window * 2.
    /// Returns [tpc, d_model] tokens where tpc = feo(cs).
    pub(crate) fn run_conv_stem(
        &self, mel_chunk: &[f32], n_mels: usize, cs: usize,
    ) -> Result<Vec<f32>> {
        let (conv_data, _t2) = self.conv_stem.forward(mel_chunk, 1, n_mels, cs)?;
        // conv_data is [t2, d_model] — all tokens are valid for a full chunk.
        Ok(conv_data)
    }

    /// Run conv-stem on a partial (tail) mel chunk, zero-padded to `cs` frames.
    /// mel_chunk: [n_mels * actual_frames] flat. actual_frames < cs.
    /// Returns [feo(actual_frames), d_model] tokens.
    pub(crate) fn run_conv_stem_tail(
        &self, mel_chunk: &[f32], n_mels: usize, actual_frames: usize, cs: usize,
    ) -> Result<Vec<f32>> {
        let tpc = feo(actual_frames);
        // Zero-pad to cs frames.
        let mut padded = vec![0.0f32; n_mels * cs];
        for m in 0..n_mels {
            let dst_base = m * cs;
            let src_base = m * actual_frames;
            for j in 0..actual_frames {
                padded[dst_base + j] = mel_chunk[src_base + j];
            }
        }
        let (conv_data, _t2) = self.conv_stem.forward(&padded, 1, n_mels, cs)?;
        // Keep only tpc tokens (discard zero-padded extras).
        let dm = self.config.d_model;
        Ok(conv_data[..tpc * dm].to_vec())
    }

    /// Run transformer layers + final projection on packed tokens.
    /// tokens: [n_tokens, d_model] flat. Returns [n_tokens, output_dim] flat.
    pub(crate) fn run_transformer(&self, tokens: &[f32], n_tokens: usize) -> Result<Vec<f32>> {
        let dm = self.config.d_model;
        let cs2 = self.config.n_window * 2;
        let tpc2 = feo(cs2);
        let cpw = self.config.n_window_infer / cs2;
        let ws = tpc2 * cpw;

        let mut h = CpuTensor::new(tokens.to_vec(), vec![1, n_tokens, dm]);
        for layer in &self.layers {
            h = layer.forward(h, Some(ws))?;
        }
        // Final projections.
        let h = self.ln_post.forward(&h);
        let mut h = self.proj1.forward(&h)?;
        gelu_inplace(&mut h);
        let h = self.proj2.forward(&h)?;
        Ok(h.data)
    }

    /// Access config for streaming chunking parameters.
    pub(crate) fn config(&self) -> &AudioEncoderConfig {
        &self.config
    }

    /// Conv-stem geometry: `(c1_out, h1, w1, c2_out, h2, w2, c3_out, h3, w3)`
    /// for one full chunk of `n_window * 2` mel frames over `n_mels` bins.
    /// The GPU tower must reproduce exactly these numbers.
    pub fn conv_geometry(&self, n_mels: usize) -> Result<[usize; 9]> {
        let cs = self.config.n_window * 2;
        let dummy = vec![0.0f32; n_mels * cs];
        let c1 = self.conv_stem.conv_block(&dummy, 1, 1, n_mels, cs, &self.conv_stem.c1_w, &self.conv_stem.c1_b)?;
        eprintln!(
            "  conv1: in 1x{n_mels}x{cs} out {}x{}x{}",
            self.conv_stem.c1_w.rows, c1.1, c1.2
        );
        let c2 = self.conv_stem.conv_block(&c1.0, 1, self.conv_stem.c1_w.rows, c1.1, c1.2, &self.conv_stem.c2_w, &self.conv_stem.c2_b)?;
        eprintln!("  conv2: out {}x{}x{}", self.conv_stem.c2_w.rows, c2.1, c2.2);
        let c3 = self.conv_stem.conv_block(&c2.0, 1, self.conv_stem.c2_w.rows, c2.1, c2.2, &self.conv_stem.c3_w, &self.conv_stem.c3_b)?;
        eprintln!("  conv3: out {}x{}x{}", self.conv_stem.c3_w.rows, c3.1, c3.2);
        eprintln!("  conv_out: k={} n={}", self.conv_stem.co.w_f32.cols, self.conv_stem.co.w_f32.rows);
        Ok([
            self.conv_stem.c1_w.rows, c1.1, c1.2,
            self.conv_stem.c2_w.rows, c2.1, c2.2,
            self.conv_stem.c3_w.rows, c3.1, c3.2,
        ])
    }

    /// CPU check for the GPU tower's conv stem, returning
    /// `(c1_raw, c1_shape, packed, packed_shape)`: `c1_raw` in the GPU's
    /// `[chunk][c_out][pos]` layout before bias+GELU, and `packed`, the conv-stem
    /// output that feeds the transformer.
    pub fn conv_reference(
        &self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
    ) -> Result<(Vec<f32>, Vec<usize>, Vec<f32>, Vec<usize>)> {
        let (chunked, cs, tpc, nfull, tail, n_chunks, n_total) =
            self.chunk_mel(mel, n_mels, n_frames)?;
        let c1_out = self.conv_stem.c1_w.rows;
        let (x1, _h1, _w1) = self.conv_stem.conv_block(
            &chunked, n_chunks, 1, n_mels, cs, &self.conv_stem.c1_w, &self.conv_stem.c1_b,
        )?;

        let (conv_data, t2) = self.conv_stem.forward(&chunked, n_chunks, n_mels, cs)?;
        let dm = self.config.d_model;
        let mut packed = Vec::with_capacity(n_total * dm);
        for i in 0..n_chunks {
            let v = if i < nfull { tpc } else { feo(tail) };
            let base = i * t2 * dm;
            packed.extend_from_slice(&conv_data[base..base + v * dm]);
        }
        Ok((x1, vec![n_chunks * 400 * c1_out], packed, vec![n_total, dm]))
    }

    /// Chunk a mel exactly the way the encoder does.
    #[allow(clippy::type_complexity)]
    fn chunk_mel(
        &self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
    ) -> Result<(Vec<f32>, usize, usize, usize, usize, usize, usize)> {
        let cs = self.config.n_window * 2;
        let tpc = feo(cs);
        let nfull = n_frames / cs;
        let tail = n_frames % cs;
        let n_chunks = nfull + usize::from(tail > 0);
        let n_total: usize = (0..n_chunks)
            .map(|i| if i < nfull { tpc } else { feo(tail) })
            .sum();
        let mut chunked = vec![0.0f32; n_chunks * n_mels * cs];
        for i in 0..nfull {
            for m in 0..n_mels {
                let dst = (i * n_mels + m) * cs;
                let src = m * n_frames + i * cs;
                chunked[dst..dst + cs].copy_from_slice(&mel[src..src + cs]);
            }
        }
        if tail > 0 {
            let i = nfull;
            for m in 0..n_mels {
                let dst = (i * n_mels + m) * cs;
                let src = m * n_frames + i * cs;
                chunked[dst..dst + tail].copy_from_slice(&mel[src..src + tail]);
            }
        }
        Ok((chunked, cs, tpc, nfull, tail, n_chunks, n_total))
    }

    /// Raw c1 conv output only, in the GPU layout `[chunk][c_out][pos]`.
    pub fn conv_reference_c1(
        &self,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
    ) -> Result<(Vec<f32>, usize, usize, usize)> {
        let (chunked, cs, _tpc, _nfull, _tail, n_chunks, _n_total) =
            self.chunk_mel(mel, n_mels, n_frames)?;
        let (x1, h1, w1) = self.conv_stem.conv_block(
            &chunked, n_chunks, 1, n_mels, cs, &self.conv_stem.c1_w, &self.conv_stem.c1_b,
        )?;
        Ok((x1, self.conv_stem.c1_w.rows, h1 * w1, n_chunks))
    }

    /// Run the conv stem stage by stage, returning each level's operand, raw GEMM
    /// output and post-GELU activation in the GPU's layouts (diagnostic oracle).
    pub fn conv_tower(&self, mel: &[f32], n_mels: usize, n_frames: usize) -> Result<CpuConvTower> {
        let (chunked, cs, tpc, nfull, tail, n_chunks, n_total) =
            self.chunk_mel(mel, n_mels, n_frames)?;
        let cf = self.conv_stem.co.w_f32.cols;
        let dm = self.config.d_model;
        let cdim = self.conv_stem.c3_w.rows;
        let fdim = crate::audio_encoder_gpu::feo_positions(n_mels);
        anyhow::ensure!(cf == cdim * fdim, "conv_out k {cf} != {cdim}·{fdim}");

        let mut input = chunked.clone();
        let (mut c_in, mut h, mut w) = (1usize, n_mels, cs);
        let mut stages = Vec::new();
        let pairs = [
            (&self.conv_stem.c1_w, &self.conv_stem.c1_b),
            (&self.conv_stem.c2_w, &self.conv_stem.c2_b),
            (&self.conv_stem.c3_w, &self.conv_stem.c3_b),
        ];
        for (ww, wb) in pairs {
            let (cols, ho, wo) = im2col_3x3_s2p1(&input, n_chunks, c_in, h, w);
            let (raw, act, ho2, wo2) = self.conv_stem.conv_block_stages(&input, n_chunks, c_in, h, w, ww, wb, true)?;
            anyhow::ensure!((ho, wo) == (ho2, wo2), "im2col/conv geometry disagree");
            stages.push(CpuConvStage {
                raw,
                act: act.clone(),
                cols,
                k: c_in * 9,
                c_out: ww.rows,
                h_out: ho2,
                w_out: wo2,
                plane: ho2 * wo2,
            });
            input = act;
            c_in = ww.rows;
            h = ho2;
            w = wo2;
        }

        let (perm, out, t2) = self.conv_stem.forward_stages(&chunked, n_chunks, n_mels, cs)?;
        anyhow::ensure!(t2 == w, "conv_out time width {t2} != conv3 {w}");
        let mut packed = Vec::with_capacity(n_total * cf);
        let mut hh = Vec::with_capacity(n_total * dm);
        for i in 0..n_chunks {
            let v = if i < nfull { tpc } else { feo(tail) };
            packed.extend_from_slice(&perm[i * t2 * cf..(i * t2 + v) * cf]);
            hh.extend_from_slice(&out[i * t2 * dm..(i * t2 + v) * dm]);
        }
        Ok(CpuConvTower { stages, packed, h: hh, n_total, n_chunks })
    }

    /// One level's conv stages from an explicit input, so each level is a
    /// bit-for-bit comparison rather than a chained one that carries f16 rounding
    /// into deeper levels.
    ///
    /// `x` is `[n_chunks][c_in][h][w]` flattened, in the mel's `[mel_bin][frame]`
    /// form (`c_in == 1`) or an activation's channel-major form.
    pub fn conv_stages_from(
        &self,
        level: usize,
        x: &[f32],
        n_chunks: usize,
        c_in: usize,
        h: usize,
        w: usize,
    ) -> Result<CpuConvStage> {
        let (ww, wb) = match level {
            0 => (&self.conv_stem.c1_w, &self.conv_stem.c1_b),
            1 => (&self.conv_stem.c2_w, &self.conv_stem.c2_b),
            2 => (&self.conv_stem.c3_w, &self.conv_stem.c3_b),
            _ => anyhow::bail!("conv level {level} out of range"),
        };
        let (cols, ho, wo) = im2col_3x3_s2p1(x, n_chunks, c_in, h, w);
        let (raw, act, ho2, wo2) = self.conv_stem.conv_block_stages(x, n_chunks, c_in, h, w, ww, wb, true)?;
        anyhow::ensure!((ho, wo) == (ho2, wo2), "im2col/conv geometry disagree");
        Ok(CpuConvStage {
            raw,
            act,
            cols,
            k: c_in * 9,
            c_out: ww.rows,
            h_out: ho2,
            w_out: wo2,
            plane: ho2 * wo2,
        })
    }

    /// `conv_out` + PE from an **explicit** operand, so the projection can be
    /// checked without the CPU chain's own drift in front of it.
    pub fn conv_out_from(&self, packed: &[f32], n_tokens: usize) -> Result<Vec<f32>> {
        let dm = self.config.d_model;
        let cs = self.config.n_window * 2;
        let tpc = feo(cs);
        let cf = self.conv_stem.co.w_f32.cols;
        anyhow::ensure!(packed.len() >= n_tokens * cf, "packed too short");
        let t = CpuTensor::new(packed[..n_tokens * cf].to_vec(), vec![1, n_tokens, cf]);
        let co = self.conv_stem.co.forward(&t)?;
        let mut out = co.data;
        for tok in 0..n_tokens {
            let base = (tok % tpc) * dm;
            for j in 0..dm {
                out[tok * dm + j] += self.conv_stem.pe[base + j];
            }
        }
        Ok(out)
    }

    /// A conv level's bias (diagnostic).
    pub fn conv_bias(&self, level: usize) -> Vec<f32> {
        match level {
            0 => self.conv_stem.c1_b.clone(),
            1 => self.conv_stem.c2_b.clone(),
            _ => self.conv_stem.c3_b.clone(),
        }
    }

    /// The sinusoidal embedding row a token uses (`tok % tpc`).
    pub fn pe_row(&self, tok: usize) -> Vec<f32> {
        let dm = self.config.d_model;
        let tpc = feo(self.config.n_window * 2);
        let base = (tok % tpc) * dm;
        self.conv_stem.pe[base..base + dm].to_vec()
    }

    /// Run one transformer layer on explicit tokens (diagnostic oracle).
    pub fn layer_forward(&self, li: usize, tokens: &[f32], n_tokens: usize) -> Result<Vec<f32>> {
        let dm = self.config.d_model;
        let cs = self.config.n_window * 2;
        let ws = feo(cs) * (self.config.n_window_infer / cs);
        anyhow::ensure!(li < self.layers.len(), "layer {li} out of range");
        let t = CpuTensor::new(tokens[..n_tokens * dm].to_vec(), vec![1, n_tokens, dm]);
        Ok(self.layers[li].forward(t, Some(ws))?.data)
    }

    /// LayerNorm 1 of a transformer layer on explicit tokens (diagnostic).
    pub fn dbg_layer_norm(&self, li: usize, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let dm = self.config.d_model;
        let t = CpuTensor::new(x[..rows * dm].to_vec(), vec![1, rows, dm]);
        Ok(self.layers[li].sln.forward(&t).data)
    }

    /// The fused QKV projection (diagnostic; same layout as `enc.qkv`).
    pub fn dbg_qkv(&self, li: usize, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let dm = self.config.d_model;
        let t = CpuTensor::new(x[..rows * dm].to_vec(), vec![1, rows, dm]);
        let a = &self.layers[li].attn;
        let (q, k, v) = (a.q_proj.forward(&t)?, a.k_proj.forward(&t)?, a.v_proj.forward(&t)?);
        let mut out = vec![0.0f32; rows * 3 * dm];
        for r in 0..rows {
            out[r * 3 * dm..r * 3 * dm + dm].copy_from_slice(&q.data[r * dm..(r + 1) * dm]);
            out[r * 3 * dm + dm..r * 3 * dm + 2 * dm].copy_from_slice(&k.data[r * dm..(r + 1) * dm]);
            out[r * 3 * dm + 2 * dm..(r + 1) * 3 * dm].copy_from_slice(&v.data[r * dm..(r + 1) * dm]);
        }
        Ok(out)
    }

    /// A layer's attention block output (windowed attention + out_proj), i.e.
    /// what `enc.attn_flat` feeds the residual (diagnostic).
    pub fn dbg_attn(&self, li: usize, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let dm = self.config.d_model;
        let cs = self.config.n_window * 2;
        let ws = feo(cs) * (self.config.n_window_infer / cs);
        let t = CpuTensor::new(x[..rows * dm].to_vec(), vec![1, rows, dm]);
        Ok(self.layers[li].attn.forward(&t, Some(ws))?.data)
    }

    /// The same block *before* `out_proj` — the stage `enc.attn_flat` actually
    /// holds (diagnostic).
    pub fn dbg_attn_flat(&self, li: usize, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let dm = self.config.d_model;
        let cs = self.config.n_window * 2;
        let ws = feo(cs) * (self.config.n_window_infer / cs);
        let t = CpuTensor::new(x[..rows * dm].to_vec(), vec![1, rows, dm]);
        Ok(self.layers[li].attn.flat(&t, Some(ws))?.data)
    }

    /// `conv_out` input width = `c3_out * f3 * t3`.
    pub fn conv_out_in_features(&self) -> usize {
        self.conv_stem.co.w_f32.cols
    }

    /// `conv2d1` bias (diagnostic).
    pub fn conv_bias_c1(&self) -> Result<Vec<f32>> {
        Ok(self.conv_stem.c1_b.clone())
    }

    /// `conv2d1` weight as `(data, c_out, taps)`, row-major `[c_out, kh*3+kw]` —
    /// the exact layout the im2col k axis uses (diagnostic).
    pub fn conv_weight_c1(&self) -> Result<(Vec<f32>, usize, usize)> {
        let w = &self.conv_stem.c1_w;
        Ok((w.data.clone(), w.rows, w.cols))
    }
}
