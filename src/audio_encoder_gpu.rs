#![allow(dead_code)]

//! GPU audio encoder for Qwen3-ASR.
//!
//! ```text
//! mel [128, frames]  →  chunk to [chunks, 1, 128, 100]
//!   conv2d(3×3, s2, p1) + bias + GELU   ×3   (im2col → GEMM)
//!   → permute [c, f, t] → [t, c·f]  →  conv_out Linear + sinusoidal PE
//!   → 18 × { LayerNorm + windowed attn + LayerNorm + FFN(GELU-erf) }
//!   → ln_post + proj1 + GELU + proj2  → [n_total, 1024]
//! ```
//!
//! The mel is a single-channel `[128, frames]` image chunked into 100-frame
//! pieces; three `k=3, s=2, p=1` convs take it `(128,100) → (480,64,50) →
//! (480,32,25) → (480,16,13)`.  `conv_out` reads `(channel, freq)` as
//! `c·freq_bins + f`, not the other way round.  Attention is windowed, not
//! causal and not global: each `ceil(s/wlen)` window is an independent full
//! attention over `wlen = feo(100)·n_window_infer/100` tokens.  The norms are
//! LayerNorm with `eps = 1e-5`, not RMSNorm; conv planes are width-padded to an
//! even stride.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;

// f16 and bf16 are the same width and the same packing, so this file's host-side
// vectors are one type either way: `f16` here is [`crate::half16::H16`], which
// carries whichever format `--dtype` selected (bf16 by default).  The helper
// names keep their historical spelling — what they hold is the run's format.
use crate::half16::H16 as f16;

use crate::audio_encoder::feo;
use crate::config::AudioEncoderConfig;
use crate::gpu::{BulkUpload, Gpu};
use crate::shaders;
use crate::weights::{self, PackedWeight, RawTensor};

/// Tile edges of `shaders::prefill_gemm`, imported from the kernel's own
/// constants so the dispatch grids and buffer pads stay in sync with it.
const GEMM_BM: usize = shaders::PREFILL_GEMM_BM;
const GEMM_BN: usize = shaders::PREFILL_GEMM_BN;
const GEMM_BK: usize = shaders::PREFILL_GEMM_BK;
/// Conv taps: `3×3`, always one whole k-group of 16 per input channel.
const TAPS: usize = 9;
/// Token capacity of the preallocated pipeline (~22 min of audio).
const MAX_TOKENS: usize = 16384;
/// Largest attention window: `feo(100) · n_window_infer/100` = 1250.
const MAX_WINDOW: usize = 2048;
/// Uniform ring for per-dispatch `GDims` (256 B slots, dynamic offset).
/// 256 slots = 64 KiB, this stack's `max_uniform_buffer_binding_size`.
const MAX_GEMMS: usize = 256;
/// Chunks processed per conv-stem round.  The im2col operand is ~9× the
/// activation it produces, so tiling the chunk axis bounds the stem's buffers.
const CONV_TILE: usize = 8;

/// Layers per submit in the transformer stack.
///
/// A submit is a round trip to the GPU (`QALIGN_ENC_SKIP=<every group>` prices
/// the whole encode's 11 submits at 49.1 ms on 180s_zh, 3.5% of `enc`), so the
/// fewer the better -- but the stack cannot be one unbounded submit because
/// each `dispatch_gemm` claims a `GDims` slot out of `MAX_GEMMS`.  The stack
/// spends 6 slots per layer (qkv, scores, av, o, fc1, fc2), so 42 layers fill
/// the ring and this constant only has to stay under that: 12 keeps two submits
/// for the 24-layer tower and leaves the same headroom for a stack that grows.
const SUBMIT_EVERY: usize = 12;
/// `CONV_TILE`, published for the diagnostic's round bookkeeping.
pub fn conv_tile() -> usize {
    CONV_TILE
}

/// Out-of-plane tap sentinel, re-exported: the tap tables and the WGSL that
/// reads them must agree on it, or an out-of-bounds tap becomes a valid address.
const TAP_OOB: u32 = shaders::TAP_OOB;

// ─── Uniform layouts (byte-identical to the WGSL structs) ──────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GDims {
    m: u32,
    n: u32,
    k: u32,
    ldc: u32,
    bsa: u32,
    bsb: u32,
    bsc: u32,
    beta: u32,
    /// Key-tile row offset in the B operand; 0 for every tower GEMM.
    row0: u32,
    /// A row stride in elements; `k` for every tower GEMM.
    lda: u32,
}

/// Mirrors `shaders::audio_im2col`'s `Im2Cfg` field for field.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Im2Cfg {
    taps: u32,
    k: u32,
    k_pad: u32,
    plane: u32,
    plane_pad: u32,
    n_chunks: u32,
    n_all: u32,
    in_chunk: u32,
    in_ic: u32,
    chunk0: u32,
    bpc: u32,
    _a: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleCfg {
    /// Element count of the tensor being GELU'd.
    n: u32,
    /// Words per channel (channel-major) or per row (token-major).
    words: u32,
    /// 1 = index the bias by `i / words`, 0 = by `i % words`.
    mode: u32,
    /// x-axis grid size; `y` continues the flat index space beyond it.
    gx: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExCfg {
    n_tokens: u32,
    nh: u32,
    hd: u32,
    tpc: u32,
    row_stride: u32,
    attn_cols: u32,
    pe_stride: u32,
    dm: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SmCfg {
    s: u32,
    wlen: u32,
    /// Attention tile: `align(wlen, GEMM_BM)`; score blocks are `[wpad][wpad]`.
    wpad: u32,
    n_win: u32,
    scale: f32,
    _a: u32,
    _b: u32,
    _c: u32,
}

/// Mirrors `shaders::audio_win_pack`'s `WinCfg`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WinCfg {
    wlen: u32,
    wpad: u32,
    hd: u32,
    n_win: u32,
    s: u32,
    acols: u32,
    /// `align(hd, GEMM_BN)` — the AV GEMM's `n` and the `vp` row width.
    pad_n: u32,
    _a: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LnCfg {
    d: u32,
    eps: f32,
    _a: u32,
    _b: u32,
}

/// Mirrors `shaders::audio_permute_pe`'s `PmCfg`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PmCfg {
    c: u32,
    f: u32,
    t3: u32,
    s_pad: u32,
    n_all: u32,
    plane_pad: u32,
    tok0: u32,
    n_tokens: u32,
}

/// Mirrors `shaders::audio_add_pe`'s `PeCfg`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PeCfg {
    d: u32,
    tpc: u32,
    s_pad: u32,
    _a: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CpCfg {
    cols: u32,
    wlen: u32,
    wpad: u32,
    hd: u32,
    n_win: u32,
    rows: u32,
    _a: u32,
    _b: u32,
}

// ─── Weights ───────────────────────────────────────────────────────

struct GpuLinear {
    w: wgpu::Buffer,
    /// The bias, padded to `n_pad` with zeros: the GEMM's `bias = 1` variant
    /// adds it, and `bias_gelu` indexes it by channel or column.
    bias: wgpu::Buffer,
    n: usize,
    n_pad: usize,
    k: usize,
}

impl GpuLinear {
    fn finish(
        up: &mut BulkUpload,
        label: &str,
        w: PackedWeight,
        bias_host: Option<Vec<f16>>,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        let n_pad = align(n, GEMM_BM);
        // A missing bias is a zero bias (`conv_out` ships without one).
        let bias_host = bias_host.unwrap_or_else(|| vec![f16::ZERO; n]);
        anyhow::ensure!(bias_host.len() == n, "{label}: bias {} != n {n}", bias_host.len());
        anyhow::ensure!(w.rows == n_pad && w.cols == k, "{label}: weight {:?}", (w.rows, w.cols));
        Ok(Self {
            w: upload_words(up, &format!("{label}.w"), &w)?,
            bias: upload_bias(up, &format!("{label}.b"), n, n_pad, &bias_host)?,
            n,
            n_pad,
            k,
        })
    }

    fn load(
        up: &mut BulkUpload,
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        label: &str,
    ) -> Result<Self> {
        let w = weights::get_matrix(weights, &format!("{prefix}.weight"))?;
        let n = w.rows;
        let k = w.cols;
        let bias = weights::get_h16(weights, &format!("{prefix}.bias"))
            .ok()
            .map(|(b, _)| b);
        let w = pad_k(&pad_rows(&w, align(n, GEMM_BM))?)?;
        Self::finish(up, label, w, bias, n, pad_k_tile(k))
    }

    /// Conv weights `[c_out, c_in, 3, 3]`: row-major flattening of the trailing
    /// axes is exactly the im2col k order.
    fn load_conv(
        up: &mut BulkUpload,
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        label: &str,
    ) -> Result<Self> {
        let name = format!("{prefix}.weight");
        let t = weights.get(&name).ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        anyhow::ensure!(t.shape.len() >= 2, "{name}: shape {:?} not 2D+", t.shape);
        let n = t.shape[0];
        let k: usize = t.shape[1..].iter().product();
        anyhow::ensure!(
            t.shape.last() == Some(&3) && t.shape.get(t.shape.len() - 2) == Some(&3),
            "{name}: shape {:?} is not a 3×3 conv",
            t.shape
        );
        let (w, shape) = t.as_h16()?;
        anyhow::ensure!(shape[0] == n && w.len() == n * k);
        let (bias, bshape) = weights::get_h16(weights, &format!("{prefix}.bias"))?;
        anyhow::ensure!(bshape.len() == 1 && bshape[0] == n, "{prefix}.bias {bshape:?} != [{n}]");
        let packed = pad_k(&PackedWeight::from_h16(&w, n, k))?;
        let packed = pad_rows(&packed, align(n, GEMM_BM))?;
        Self::finish(up, label, packed, Some(bias), n, pad_k_tile(k))
    }

    fn load_fused(
        up: &mut BulkUpload,
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        parts: &[&str],
        label: &str,
    ) -> Result<Self> {
        let mut mats = Vec::with_capacity(parts.len());
        let mut bias: Vec<f16> = Vec::new();
        for p in parts {
            mats.push(weights::get_matrix(weights, &format!("{prefix}.{p}.weight"))?);
            let (b, _) = weights::get_h16(weights, &format!("{prefix}.{p}.bias"))?;
            bias.extend_from_slice(&b);
        }
        let k = mats[0].cols;
        for m in &mats {
            anyhow::ensure!(m.cols == k, "{prefix}: fused cols differ");
        }
        let n: usize = mats.iter().map(|m| m.rows).sum();
        let w = pad_rows(&PackedWeight::concat_rows(&mats, k)?, align(n, GEMM_BM))?;
        Self::finish(up, label, w, Some(bias), n, pad_k_tile(k))
    }
}

struct GpuLayerNorm {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
    eps: f32,
}

impl GpuLayerNorm {
    fn load(
        up: &mut BulkUpload,
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        label: &str,
        eps: f32,
    ) -> Result<Self> {
        let w = weights::get_h16_vector(weights, &format!("{prefix}.weight"))?;
        let b = weights::get_h16_vector(weights, &format!("{prefix}.bias"))?;
        anyhow::ensure!(w.len() == b.len(), "{prefix}: weight/bias len mismatch");
        Ok(Self {
            w: upload_f16(up, &format!("{label}.w"), &w)?,
            b: upload_f16(up, &format!("{label}.b"), &b)?,
            eps,
        })
    }
}

struct GpuLayer {
    sln: GpuLayerNorm,
    fln: GpuLayerNorm,
    qkv: GpuLinear,
    o: GpuLinear,
    fc1: GpuLinear,
    fc2: GpuLinear,
}

// ─── Pipelines ─────────────────────────────────────────────────────

struct Pipes {
    im2col: wgpu::ComputePipeline,
    gelu: wgpu::ComputePipeline,
    gemm: wgpu::ComputePipeline,
    gemm_t: wgpu::ComputePipeline,
    extract: wgpu::ComputePipeline,
    win_pack: wgpu::ComputePipeline,
    /// `prefill_gemm(false, true)`: `C += A·Wᵀ`, for the two residual adds.
    gemm_beta: wgpu::ComputePipeline,
    /// `prefill_gemm(false, false, true)`: the per-column bias add (`qkv`,
    /// `proj2`); `(false, true, true)` for residual+bias (`out_proj`, `fc2`).
    gemm_bias: wgpu::ComputePipeline,
    gemm_beta_bias: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    layernorm: wgpu::ComputePipeline,
    permute: wgpu::ComputePipeline,
    add_pe: wgpu::ComputePipeline,
    attn_flat: wgpu::ComputePipeline,
}

// ─── Conv geometry: the single source of truth ────────────────────

/// One conv round's tensors, in the buffers' own layouts (the same `ConvLevel`
/// fields the dispatches use).
pub struct ConvRound {
    pub chunk0: usize,
    pub n_chunks: usize,
    /// Per level (in chain order): the im2col operand `[k_pad][n_all]`, the raw
    /// GEMM output `[m_pad][n_all]`, the post-GELU activation.
    pub col: Vec<Vec<f16>>,
    pub raw: Vec<Vec<f16>>,
    pub act: Vec<Vec<f16>>,
}

/// Layer 0's attention tensors, read back verbatim so a host recomputation can
/// check them without any other oracle in the loop.
#[derive(Default)]
pub struct AttnShot {
    /// `[s_pad][acols]` projected Q/K/V.
    pub q: Vec<f16>,
    pub k: Vec<f16>,
    pub v: Vec<f16>,
    /// `[z][wpad][wpad]` raw scores / softmax probs; `[z][wpad][128]` AV output.
    pub scores: Vec<f16>,
    pub probs: Vec<f16>,
    pub attn_out: Vec<f16>,
    /// The per-`(head, window)` operands the two batched GEMMs read:
    /// `[z][wpad][hd]`, `[z][hd][wpad]` and `[z][wpad][hd_pad]`.
    pub qp: Vec<f16>,
    pub kt: Vec<f16>,
    pub vp: Vec<f16>,
    pub hd_pad: usize,
    /// `[n_total][d_model]` after layer 0's LN1, the fused QKV projection and
    /// the whole attention block.
    pub normed: Vec<f16>,
    pub qkv: Vec<f16>,
    pub attn_flat: Vec<f16>,
    /// `h` right after layer 0's attention residual, before its FFN.
    pub mid: Vec<f16>,
    pub s: usize,
    pub acols: usize,
    pub nh: usize,
    pub hd: usize,
    pub wlen: usize,
    pub wpad: usize,
    pub n_win: usize,
}

/// Optional host-side capture from [`GpuAudioEncoder::encode_capture`].  Empty
/// vectors mean "not captured".
#[derive(Default)]
pub struct Capture {
    pub rounds: Vec<ConvRound>,
    /// The first 64 mel halves and the first 72 taps of each level.
    pub mel_head: Vec<f16>,
    pub taps: Vec<Vec<u32>>,
    /// `[s_pad][conv_out.k]` before `conv_out`, and `[s_pad][d_model]` after.
    pub packed: Vec<f16>,
    pub h: Vec<f16>,
    /// `[s_pad][d_model]` after each transformer layer.
    pub layers: Vec<Vec<f16>>,
    /// Layer 0's attention internals, for a self-contained oracle.
    pub attn: Option<AttnShot>,
    pub embeds: Vec<f16>,
}

/// One conv level's geometry and weights (counts in f16 elements).
///
/// A `position` is one `(h_out, w_out)` cell, row-major (`p = ho·w_out + wo`);
/// `plane_pad = align(plane, 32)` is the per-chunk stride and
/// `n_all = align(CONV_TILE·plane_pad, GEMM_BN)` the round's position count (the
/// GEMM's `n` and the activation's channel stride, the same for every round).
///
/// The operand is `[k_pad][n_all]` (the GEMM's `B`, `transb = true`); the
/// activation is `[m_pad][n_all]` (the GEMM's `C`, channel-major).  The gather
/// reads tap `t` of `(p, k)` at `(chunk₀ + chunk)·in_chunk + ic·in_ic + tap`,
/// `ic = k/9`.
struct ConvLevel {
    /// `align(CONV_TILE·plane_pad, GEMM_BN)`: the GEMM's `n` and the channel
    /// stride of the activation, constant for every round.
    n_all: usize,
    k: usize,
    k_pad: usize,
    c_in: usize,
    c_out: usize,
    m_pad: usize,
    plane: usize,
    plane_pad: usize,
    h_in: usize,
    w_in: usize,
    w_stride: usize,
    in_chunk: usize,
    in_ic: usize,
    taps: wgpu::Buffer,
    w: GpuLinear,
}

impl ConvLevel {
    /// Build one level from the checkpoint weights and its input geometry:
    /// `h_in`/`w_in` is the input plane, `w_stride` the buffer row stride (`w0`
    /// for the mel), `in_chunk`/`in_ic` the gather strides and `n_all` the
    /// position count.
    #[allow(clippy::too_many_arguments)]
    fn build(
        up: &mut BulkUpload,
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        label: &str,
        h_in: usize,
        w_in: usize,
        w_stride: usize,
        c_in: usize,
        in_chunk: usize,
        in_ic: usize,
        n_all: usize,
    ) -> Result<Self> {
        let w = GpuLinear::load_conv(up, weights, prefix, label)?;
        let k = c_in * TAPS;
        let k_pad = pad_k_tile(k);
        anyhow::ensure!(
            w.k == k_pad,
            "{label}: weight k {} != c_in·9 = {k} padded to {k_pad}",
            w.k
        );
        let (offs, plane, h_out, w_out) = conv_taps(h_in, w_in, w_stride);
        anyhow::ensure!(
            (plane, h_out, w_out) == (h_out * w_out, plane / w_out, w_out),
            "{label}: tap table plane {plane} != {h_out}x{w_out}"
        );
        let taps = up.storage(&format!("{label}.taps"), (offs.len() * 4) as u64);
        up.upload(&taps, &u32_bytes(&offs))?;
        Ok(Self {
            n_all,
            k,
            k_pad,
            c_in,
            c_out: w.n,
            m_pad: w.n_pad,
            plane,
            plane_pad: align(plane, 32),
            h_in,
            w_in,
            w_stride,
            in_chunk,
            in_ic,
            taps,
            w,
        })
    }

    fn h_out(&self) -> usize {
        (self.h_in + 2 - 3) / 2 + 1
    }

    fn w_out(&self) -> usize {
        (self.w_in + 2 - 3) / 2 + 1
    }
}

/// A read-only view of one [`ConvLevel`] for the diagnostic comparator, built
/// from the same numbers the dispatches are.
#[derive(Clone, Copy, Debug)]
pub struct LevelView {
    /// Positions of the whole round per level (`align(CONV_TILE·plane_pad, BN)`).
    pub n_all: usize,
    pub k: usize,
    pub k_pad: usize,
    pub c_in: usize,
    pub c_out: usize,
    /// `align(c_out, GEMM_BM)`: the activation's row count.
    pub m_pad: usize,
    pub plane: usize,
    pub plane_pad: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub in_chunk: usize,
    pub in_ic: usize,
}

// ─── Encoder ───────────────────────────────────────────────────────

pub struct GpuAudioEncoder {
    pub d_model: usize,
    pub out_dim: usize,
    nh: usize,
    hd: usize,
    inter: usize,
    /// Tokens a full mel chunk produces (`feo(n_window·2)`).
    tpc: usize,
    cs: usize,
    n_mels: usize,
    pe_rows: usize,
    window_infer: usize,

    /// Conv-stem levels in chain order; `conv[2].plane` is `tpc`.
    conv: [ConvLevel; 3],
    conv_out: GpuLinear,
    layers: Vec<GpuLayer>,
    ln_post: GpuLayerNorm,
    proj1: GpuLinear,
    proj2: GpuLinear,
    pe: wgpu::Buffer,
    /// The final GELU runs on a token-major activation `[rows][proj1.n_pad]`.
    p: Pipes,

    /// Mel row stride (`cs` padded even), and the mel's per-chunk image size.
    w0: usize,
    mel_chunk: usize,
    /// `align(nh·hd, GEMM_BM)` — the attention operand's row stride.
    attn_cols: usize,
    /// Position count one conv round is laid out on, per level (`n_all`).
    n_all: [usize; 3],

    // ── conv buffers, sized for one `CONV_TILE` round and reused every round ──
    col: [wgpu::Buffer; 3],
    raw: [wgpu::Buffer; 3],
    act: [wgpu::Buffer; 3],
    /// Clip-sized tensors (`packed`, `mel_in`, activations) per `encode`.

    // uniforms
    u_gd: wgpu::Buffer,
    u_im: [wgpu::Buffer; 3],
    /// One `ScaleCfg` per distinct GELU shape: the three conv levels, the FFN
    /// and `proj1`.  A single shared uniform would be read by every dispatch as
    /// the last value written, since `queue.write_buffer` lands at the next
    /// submit.
    u_sc: [wgpu::Buffer; 5],
    u_ex: wgpu::Buffer,
    u_sm: wgpu::Buffer,
    u_ln: wgpu::Buffer,
    u_pm: wgpu::Buffer,
    u_pe: wgpu::Buffer,
    u_wp: wgpu::Buffer,
    u_cp: wgpu::Buffer,

    gd_slot: std::cell::Cell<usize>,
    /// Diagnostic: snapshot `h` inside layer 0 (between attention and FFN).
    mid_capture: std::cell::Cell<bool>,
    mid_h: std::cell::RefCell<Option<Vec<f16>>>,
}

#[derive(Clone, Copy, Default, Debug)]
struct Geom {
    n_chunks: usize,
    s: usize,
    wlen: usize,
    n_win: usize,
    s_win: usize,
}

/// `QALIGN_ENC_SKIP=<group>[,…]` — the audio tower's ablation hook, the
/// counterpart of the decoder's `QALIGN_SKIP` (see `decoder::prefill` for why
/// omitting a dispatch prices it better than adding one).  A skipped group's
/// operands go stale, so the *result* is wrong on purpose: the phase clock is
/// the measurement, and an unskipped run is what the gate judges.
///
/// Group names, one per dispatch site in [`GpuAudioEncoder::layer`] plus the
/// four after the stack: `ln`, `qkv`, `split`, `pack`, `scores`, `sm`, `av`,
/// `flat`, `o`, `fln`, `fc1`, `gelu`, `fc2`, `lnpost`, `proj1`, `p1gelu`,
/// `proj2`.
///
/// `hit` is the guard the decoder's version learned to need: without it a name
/// that matched nothing is indistinguishable from a group that costs nothing.
struct EncSkip {
    names: Vec<String>,
    /// One per `names` entry: did any dispatch site claim that group?
    hit: Vec<std::cell::Cell<bool>>,
    /// Dispatch sites actually omitted (a per-layer group counts 24 times).
    sites: std::cell::Cell<usize>,
}

impl EncSkip {
    /// Read the environment once per encode; inactive when unset.
    fn from_env() -> Self {
        let names: Vec<String> = std::env::var("QALIGN_ENC_SKIP")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        let hit = names.iter().map(|_| std::cell::Cell::new(false)).collect();
        Self { names, hit, sites: std::cell::Cell::new(0) }
    }

    /// Does the list name this group?  Records the hit and prices the site.
    fn matches(&self, name: &str) -> bool {
        let mut found = false;
        for (i, n) in self.names.iter().enumerate() {
            if n == name {
                self.hit[i].set(true);
                found = true;
            }
        }
        if found {
            self.sites.set(self.sites.get() + 1);
        }
        found
    }

    /// One line, printed only when the hook is in use.
    fn report(&self, blocks: usize, s: usize) {
        if self.names.is_empty() {
            return;
        }
        let unmatched: Vec<&str> = self
            .names
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.hit[*i].get())
            .map(|(_, n)| n.as_str())
            .collect();
        eprintln!(
            "audio encoder: QALIGN_ENC_SKIP={} skipped {} dispatch site(s) over {blocks} layer(s), \
             s={s}; unmatched: {}",
            self.names.join(","),
            self.sites.get(),
            if unmatched.is_empty() { "none".to_owned() } else { unmatched.join(",") },
        );
    }
}

/// Per-pass activation buffers, sized to this clip.
struct LayerCtx<'a> {
    geom: Geom,
    s_pad: usize,
    acols: usize,
    inter_pad: usize,
    h: &'a wgpu::Buffer,
    normed: &'a wgpu::Buffer,
    norm2: &'a wgpu::Buffer,
    qkv: &'a wgpu::Buffer,
    q: &'a wgpu::Buffer,
    k: &'a wgpu::Buffer,
    v: &'a wgpu::Buffer,
    scores: &'a wgpu::Buffer,
    attn: &'a wgpu::Buffer,
    attn_flat: &'a wgpu::Buffer,
    gu: &'a wgpu::Buffer,
    act: &'a wgpu::Buffer,
    /// Per-`(head, window)` attention blocks: Q, Kᵀ, V and the AV output.
    qp: &'a wgpu::Buffer,
    kt: &'a wgpu::Buffer,
    vp: &'a wgpu::Buffer,
    attn_out: &'a wgpu::Buffer,
    /// Ablation state — see [`EncSkip`].
    skip: &'a EncSkip,
}

impl GpuAudioEncoder {
    pub fn load(
        gpu: &Gpu,
        weights: &HashMap<String, RawTensor>,
        prefix: &str,
        cfg: &AudioEncoderConfig,
        window_infer: usize,
    ) -> Result<Self> {
        let dm = cfg.d_model;
        let nh = cfg.encoder_attention_heads;
        anyhow::ensure!(dm % nh == 0, "d_model {dm} not divisible by {nh} heads");
        let hd = dm / nh;
        let inter = cfg.encoder_ffn_dim;
        let out_dim = cfg.output_dim;
        let eps = 1e-5f32;
        let cs = cfg.n_window * 2;
        let n_mels = cfg.num_mel_bins;
        let tpc = feo(cs);

        let mut up = gpu.uploader();

        let c1 = GpuLinear::load_conv(&mut up, weights, &format!("{prefix}.conv2d1"), "enc.c1")?;
        anyhow::ensure!(c1.k == pad_k_tile(TAPS), "conv2d1 k {} != {}", c1.k, pad_k_tile(TAPS));
        let c2 = GpuLinear::load_conv(&mut up, weights, &format!("{prefix}.conv2d2"), "enc.c2")?;
        let c3 = GpuLinear::load_conv(&mut up, weights, &format!("{prefix}.conv2d3"), "enc.c3")?;
        anyhow::ensure!(c2.k == c1.n * TAPS, "conv2d2 k={} != {}*9", c2.k, c1.n);
        anyhow::ensure!(c3.k == c2.n * TAPS, "conv2d3 k={} != {}*9", c3.k, c2.n);
        let conv_out = GpuLinear::load(&mut up, weights, &format!("{prefix}.conv_out"), "enc.co")?;

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        // The layernorm stages one row in workgroup memory (`AUDIO_LN_WORDS`
        // 16-bit pairs), so the tower's width has to fit it.
        anyhow::ensure!(
            cfg.d_model as usize <= 2 * shaders::AUDIO_LN_WORDS,
            "audio d_model {} exceeds the layernorm staging ({} words)",
            cfg.d_model,
            shaders::AUDIO_LN_WORDS
        );
        for i in 0..cfg.encoder_layers {
            let p = format!("{prefix}.layers.{i}");
            let l = |s: &str| format!("enc.l{i}.{s}");
            let fc1 = GpuLinear::load(&mut up, weights, &format!("{p}.fc1"), &l("fc1"))?;
            layers.push(GpuLayer {
                sln: GpuLayerNorm::load(&mut up, weights, &format!("{p}.self_attn_layer_norm"), &l("sln"), eps)?,
                fln: GpuLayerNorm::load(&mut up, weights, &format!("{p}.final_layer_norm"), &l("fln"), eps)?,
                qkv: GpuLinear::load_fused(
                    &mut up,
                    weights,
                    &format!("{p}.self_attn"),
                    &["q_proj", "k_proj", "v_proj"],
                    &l("qkv"),
                )?,
                o: GpuLinear::load(&mut up, weights, &format!("{p}.self_attn.out_proj"), &l("o"))?,
                fc1,
                fc2: GpuLinear::load(&mut up, weights, &format!("{p}.fc2"), &l("fc2"))?,
            });
        }
        let ln_post = GpuLayerNorm::load(&mut up, weights, &format!("{prefix}.ln_post"), "enc.lnp", eps)?;
        let proj1 = GpuLinear::load(&mut up, weights, &format!("{prefix}.proj1"), "enc.p1")?;
        let proj2 = GpuLinear::load(&mut up, weights, &format!("{prefix}.proj2"), "enc.p2")?;

        // ── conv geometry: one chain, stated once ──
        // Each level's *output* plane is the next level's *input* plane, so the
        // tap tables are built from the input dims.  `conv_out_len` is one
        // `conv2d(3×3, s2, p1)` step; `feo` is three of them, i.e. the *token*
        // count, not any single level's width.
        let w0 = pad_even(cs);
        let h1 = conv_out_len(n_mels);
        let t1 = conv_out_len(cs);
        let h2 = conv_out_len(h1);
        let t2 = conv_out_len(t1);
        let h3 = conv_out_len(h2);
        let t3 = conv_out_len(t2);
        anyhow::ensure!(t3 == tpc, "conv chain: t3 {t3} != tpc feo({cs}) = {tpc}");
        anyhow::ensure!(
            conv_out.k == c3.n * h3,
            "conv_out k={} != c3_out·h3 = {}·{} = {}",
            conv_out.k,
            c3.n,
            h3,
            c3.n * h3
        );
        anyhow::ensure!(conv_out.n == dm, "conv_out n={} != d_model {dm}", conv_out.n);

        // Per-level position counts.  `plane_pad` (32-position blocks — the
        // im2col's thread granularity) is the per-chunk stride; `n_all` is the
        // round's position count and the activation's channel stride, constant
        // across rounds.
        let planes = [h1 * t1, h2 * t2, h3 * t3];
        let plane_pads = planes.map(|p| align(p, 32));
        let n_alls = plane_pads.map(|p| align(CONV_TILE * p, GEMM_BN));
        for (i, (pl, pp)) in planes.iter().zip(&plane_pads).enumerate() {
            anyhow::ensure!(pl % 2 == 0, "conv{} plane {pl} is odd", i + 1);
            anyhow::ensure!(pp % 32 == 0, "conv{} plane_pad {pp} not a 32-multiple", i + 1);
        }
        anyhow::ensure!(t3 == planes[2] / h3, "conv3 plane {} is not h3·t3", planes[2]);

        let mel_chunk = mel_elems(n_mels, w0);
        let conv = [
            ConvLevel::build(
                &mut up, weights, &format!("{prefix}.conv2d1"), "enc.c1",
                n_mels, cs, w0, 1, mel_chunk, 0, n_alls[0],
            )?,
            ConvLevel::build(
                &mut up, weights, &format!("{prefix}.conv2d2"), "enc.c2",
                h1, t1, t1, c1.n, plane_pads[0], n_alls[0], n_alls[1],
            )?,
            ConvLevel::build(
                &mut up, weights, &format!("{prefix}.conv2d3"), "enc.c3",
                h2, t2, t2, c2.n, plane_pads[1], n_alls[1], n_alls[2],
            )?,
        ];
        // The chain invariants the gather depends on: every level's input plane
        // is the previous level's *output* plane (so the tap offsets stay inside
        // one chunk's block) and its channel count is that level's real width.
        for i in 1..3 {
            anyhow::ensure!(
                (conv[i].h_in, conv[i].w_in, conv[i].c_in)
                    == (conv[i - 1].h_out(), conv[i - 1].w_out(), conv[i - 1].c_out),
                "conv{} input plane {}x{}x{} != conv{} output {}x{}x{}",
                i + 1,
                conv[i].h_in,
                conv[i].w_in,
                conv[i].c_in,
                i,
                conv[i - 1].h_out(),
                conv[i - 1].w_out(),
                conv[i - 1].c_out
            );
        }

        for (i, l) in layers.iter().enumerate() {
            anyhow::ensure!(l.qkv.k == dm && l.qkv.n == 3 * dm, "layer {i} qkv {:?}", (l.qkv.n, l.qkv.k));
            anyhow::ensure!(l.o.k == nh * hd && l.o.n == dm, "layer {i} out_proj {:?}", (l.o.n, l.o.k));
            anyhow::ensure!(l.fc1.k == dm && l.fc1.n == inter, "layer {i} fc1 {:?}", (l.fc1.n, l.fc1.k));
            anyhow::ensure!(l.fc2.k == inter && l.fc2.n == dm, "layer {i} fc2 {:?}", (l.fc2.n, l.fc2.k));
        }
        // `proj1` maps `d_model` to `output_dim`; `proj2` keeps that width.
        anyhow::ensure!(
            proj1.k == dm && proj2.k == proj1.n && proj2.n == out_dim,
            "projector chain {:?}/{:?} (d_model {dm}, out {out_dim})",
            (proj1.n, proj1.k),
            (proj2.n, proj2.k)
        );

        // ── sinusoidal PE ──
        // The *stride* is `max_position_embeddings`; a token's row is `tok % tpc`.
        let pe_rows = cfg.max_source_positions;
        let half = dm / 2;
        let lt = (10000.0f64).ln() / (half as f64 - 1.0);
        let mut pe = vec![f16::ZERO; pe_rows * dm];
        for p in 0..pe_rows {
            for i in 0..half {
                let a = p as f64 * (-(i as f64) * lt).exp();
                pe[p * dm + i] = f16::from_f32(a.sin() as f32);
                pe[p * dm + half + i] = f16::from_f32(a.cos() as f32);
            }
        }
        let pe_buf = upload_f16(&mut up, "enc.pe", &pe)?;

        let attn_cols = align(nh * hd, GEMM_BM);
        let wlen = tpc * (window_infer / cs);
        anyhow::ensure!(wlen <= MAX_WINDOW, "window {wlen} > {MAX_WINDOW}");

        eprintln!(
            "gpu audio encoder: conv {h1}x{t1} -> {h2}x{t2} -> {h3}x{t3} \
             [{} pos/chunk, {CONV_TILE}-chunk rounds], d_model {dm} / {nh}h x {hd}, \
             ffn {inter}, out {out_dim}, chunk {cs} -> {tpc} tok, wlen {wlen}",
            planes.iter().map(|p| p.to_string()).collect::<Vec<_>>().join("/"),
        );

        // Conv GEMMs read the weight as `A` and the im2col operand as `B`
        // (`transb`), so they share one pipeline; everything else uses the plain
        // `[m,k]`×`[n,k]` form.
        let gemm_layout = gemm_layout(gpu, "enc_gemm", 3, &[2]);
        // A bias needs a storage binding *after* the dynamic-offset uniform.
        let gemm_bias_layout = gemm_bias_layout(gpu, "enc_gemm_bias", &[2]);
        let p = Pipes {
            im2col: gpu.pipeline("enc_im2col", &shaders::audio_im2col(), "im2col", None)?,
            gelu: gpu.pipeline("enc_gelu", &shaders::audio_bias_gelu(), "bias_gelu", None)?,
            gemm: gpu.pipeline("enc_gemm", &shaders::prefill_gemm(false, false), "gemm", Some(&gemm_layout))?,
            gemm_t: gpu.pipeline("enc_gemm_t", &shaders::prefill_gemm(true, false), "gemm", Some(&gemm_layout))?,
            extract: gpu.pipeline("enc_extract", &shaders::audio_extract_qkv(), "extract", None)?,
            win_pack: gpu.pipeline("enc_win_pack", &shaders::audio_win_pack(), "win_pack", None)?,
            gemm_beta: gpu.pipeline("enc_gemm_beta", &shaders::prefill_gemm(false, true), "gemm", Some(&gemm_layout))?,
            gemm_bias: gpu.pipeline("enc_gemm_bias", &shaders::prefill_gemm_bias(false, false), "gemm", Some(&gemm_bias_layout))?,
            gemm_beta_bias: gpu.pipeline("enc_gemm_beta_bias", &shaders::prefill_gemm_bias(false, true), "gemm", Some(&gemm_bias_layout))?,
            softmax: gpu.pipeline("enc_softmax", &shaders::audio_window_softmax(), "softmax", None)?,
            layernorm: gpu.pipeline("enc_layernorm", &shaders::audio_layernorm(), "layernorm", None)?,
            permute: gpu.pipeline("enc_permute", &shaders::audio_permute_pe(), "permute_pe", None)?,
            add_pe: gpu.pipeline("enc_add_pe", &shaders::audio_add_pe(), "add_pe", None)?,
            attn_flat: gpu.pipeline("enc_attn_flat", &shaders::audio_attn_flat(), "attn_flat", None)?,
        };

        // ── conv buffers: one round's worth, allocated once ──
        // `col[i]` is the im2col operand `[k_pad][n_all]`, `raw[i]` the GEMM's
        // `C` (`[m_pad][n_all]`, pre-GELU) and `act[i]` the activation the next
        // level gathers from.
        let f16s = |n: usize| (n * 2) as u64;
        let col = [
            up.storage("enc.c1_col", f16s(conv[0].k_pad * conv[0].n_all)),
            up.storage("enc.c2_col", f16s(conv[1].k_pad * conv[1].n_all)),
            up.storage("enc.c3_col", f16s(conv[2].k_pad * conv[2].n_all)),
        ];
        let raw = [
            up.storage("enc.c1_raw", f16s(conv[0].m_pad * conv[0].n_all)),
            up.storage("enc.c2_raw", f16s(conv[1].m_pad * conv[1].n_all)),
            up.storage("enc.c3_raw", f16s(conv[2].m_pad * conv[2].n_all)),
        ];
        let act = [
            up.storage("enc.c1_act", f16s(conv[0].m_pad * conv[0].n_all)),
            up.storage("enc.c2_act", f16s(conv[1].m_pad * conv[1].n_all)),
            up.storage("enc.c3_act", f16s(conv[2].m_pad * conv[2].n_all)),
        ];

        let u_gd = up.uniform("enc.gd", (MAX_GEMMS * 256) as u64);
        let u_im = [
            up.uniform("enc.im0", 48),
            up.uniform("enc.im1", 48),
            up.uniform("enc.im2", 48),
        ];
        let u_sc = [
            up.uniform("enc.sc.c1", 32),
            up.uniform("enc.sc.c2", 32),
            up.uniform("enc.sc.c3", 32),
            up.uniform("enc.sc.ffn", 32),
            up.uniform("enc.sc.proj", 32),
        ];
        let u_ex = up.uniform("enc.ex", 32);
        let u_sm = up.uniform("enc.sm", 32);
        let u_ln = up.uniform("enc.ln", 32);
        let u_pm = up.uniform("enc.pm", 32);
        let u_pe = up.uniform("enc.pe_cfg", 32);
        let u_wp = up.uniform("enc.win_cfg", 32);
        let u_cp = up.uniform("enc.cp", 32);

        up.finish()?;

        Ok(Self {
            d_model: dm,
            out_dim,
            nh,
            hd,
            inter,
            tpc,
            cs,
            n_mels,
            pe_rows,
            window_infer,
            conv,
            conv_out,
            layers,
            ln_post,
            proj1,
            proj2,
            pe: pe_buf,
            p,
            w0,
            mel_chunk,
            attn_cols,
            n_all: n_alls,
            col,
            raw,
            act,
            u_gd,
            u_im,
            u_sc,
            u_ex,
            u_sm,
            u_ln,
            u_pm,
            u_pe,
            u_wp,
            u_cp,
            gd_slot: std::cell::Cell::new(0),
            mid_capture: std::cell::Cell::new(false),
            mid_h: std::cell::RefCell::new(None),
        })
    }

    /// Geometry of one conv level, as the dispatches see it.
    pub fn level(&self, i: usize) -> LevelView {
        let l = &self.conv[i];
        LevelView {
            n_all: l.n_all,
            k: l.k,
            k_pad: l.k_pad,
            c_in: l.c_in,
            c_out: l.c_out,
            m_pad: l.m_pad,
            plane: l.plane,
            plane_pad: l.plane_pad,
            h_in: l.h_in,
            w_in: l.w_in,
            h_out: l.h_out(),
            w_out: l.w_out(),
            in_chunk: l.in_chunk,
            in_ic: l.in_ic,
        }
    }

    /// Encode `[n_mels, n_frames]` (mel-bin major) into `[n_tokens, out_dim]`, in
    /// the run's 16-bit storage format.
    pub fn encode(
        &mut self,
        gpu: &Gpu,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
    ) -> Result<Vec<f16>> {
        Ok(self.encode_capture(gpu, mel, n_mels, n_frames, false)?.embeds)
    }

    /// As [`Self::encode`], optionally snapshotting each conv round's tensors on
    /// the host.  `capture` forces a submit+poll per readback, so it is a
    /// diagnostic path only.
    pub fn encode_capture(
        &mut self,
        gpu: &Gpu,
        mel: &[f32],
        n_mels: usize,
        n_frames: usize,
        capture: bool,
    ) -> Result<Capture> {
        let t_all = std::time::Instant::now();
        anyhow::ensure!(n_mels == self.n_mels, "mel bins {n_mels} != {}", self.n_mels);
        let (cs, tpc) = (self.cs, self.tpc);
        let nfull = n_frames / cs;
        let tail = n_frames % cs;
        let n_chunks = nfull + usize::from(tail > 0);
        anyhow::ensure!(n_chunks > 0, "empty mel");
        let n_total: usize = (0..n_chunks)
            .map(|i| if i < nfull { tpc } else { feo(tail) })
            .sum();
        anyhow::ensure!(n_total <= MAX_TOKENS, "mel needs {n_total} tokens > {MAX_TOKENS}");

        // ── host: one chunk's mel = `[mel_bin][frame]`, row-padded to `w0` ──
        let t_pack = std::time::Instant::now();
        let w0 = self.w0;
        let mut chunked = vec![0.0f32; n_chunks * self.mel_chunk];
        for i in 0..nfull + usize::from(tail > 0) {
            let len = if i < nfull { cs } else { tail };
            for m in 0..n_mels {
                let dst = (i * n_mels + m) * w0;
                let src = m * n_frames + i * cs;
                chunked[dst..dst + len].copy_from_slice(&mel[src..src + len]);
            }
        }
        let mel16: Vec<f16> = chunked.iter().map(|v| f16::from_f32(*v)).collect();
        let t_pack = t_pack.elapsed();

        // ── clip-sized buffers ──
        let (dm, nh, hd, inter, acols) = (self.d_model, self.nh, self.hd, self.inter, self.attn_cols);
        let out_dim = self.out_dim;
        let cf = self.conv_out.k;
        let s_pad = align(n_total, GEMM_BM);
        let qkv_cols = align(3 * dm, GEMM_BM);
        let inter_pad = align(inter, GEMM_BM);
        let gu_pad = align(2 * inter, GEMM_BM);
        let wlen = tpc * (self.window_infer / cs);
        let n_win = n_total.div_ceil(wlen);
        let geom = Geom { n_chunks, s: n_total, wlen, n_win, s_win: n_win * wlen };
        let words = |n: usize| (n / 2 * 4) as u64;

        let mel_buf = gpu.storage("enc.mel", (mel16.len() * 2) as u64);
        gpu.upload(&mel_buf, &weights::words_bytes(&mel16));
        let packed = gpu.storage("enc.packed", words(s_pad * cf));
        let h_raw = gpu.storage("enc.h_raw", words(s_pad * dm));
        let h_buf = gpu.storage("enc.h", words(s_pad * dm));
        let normed = gpu.storage("enc.normed", words(s_pad * dm));
        let norm2 = gpu.storage("enc.norm2", words(s_pad * dm));
        let qkv = gpu.storage("enc.qkv", words(s_pad * qkv_cols));
        let q_buf = gpu.storage("enc.q", words(s_pad * acols));
        let k_buf = gpu.storage("enc.k", words(s_pad * acols));
        let v_buf = gpu.storage("enc.v", words(s_pad * acols));
        let attn_flat = gpu.storage("enc.attn_flat", words(s_pad * acols));
        // Per-(head, window) attention blocks.  `wpad` is the GEMM tile
        // `wlen` is rounded up to, and `hd_pad` the `n` tile `hd` is.
        let wpad = align(wlen, GEMM_BM);
        let hd_pad = align(hd, GEMM_BN);
        let z_blocks = nh * n_win;
        let qp = gpu.storage("enc.qp", (z_blocks * wpad * hd * 2) as u64);
        let kt = gpu.storage("enc.kt", (z_blocks * hd * wpad * 2) as u64);
        let vp = gpu.storage("enc.vp", (z_blocks * wpad * hd_pad * 2) as u64);
        let scores = gpu.storage("enc.scores", (z_blocks * wpad * wpad * 2) as u64);
        let attn = gpu.storage("enc.attn", (z_blocks * wpad * wpad * 2) as u64);
        let attn_out = gpu.storage("enc.attn_out", (z_blocks * wpad * hd_pad * 2) as u64);
        let gu = gpu.storage("enc.gu", words(s_pad * gu_pad));
        let gact = gpu.storage("enc.gact", words(s_pad * inter_pad));
        let out_emb = gpu.storage("enc.out", words(s_pad * out_dim));

        // The FFN and projection GELU shapes depend on the clip's token count.
        for (i, (rows, lin, by_channel)) in [
            (s_pad, &self.layers[0].fc1, false),
            (s_pad, &self.proj1, false),
        ]
        .into_iter()
        .enumerate()
        {
            let words = lin.n_pad / 2;
            let n = rows * lin.n_pad;
            gpu.queue.write_buffer(
                &self.u_sc[3 + i],
                0,
                bytemuck::bytes_of(&ScaleCfg {
                    n: n as u32,
                    words: words as u32,
                    mode: u32::from(by_channel),
                    // The `bias_gelu` dispatch for these two sites splits over
                    // both grid axes (`grid_xy` in the dispatch below).
                    gx: crate::decoder::grid_xy(n.div_ceil(512)).0,
                }),
            );
        }

        let mut out = Capture::default();

        // Ablation hook (see [`EncSkip`]).  Parsed here so both the conv stem
        // and the transformer can be priced against the same list.
        let skip = EncSkip::from_env();

        // ── conv stem: `CONV_TILE` chunks per round, one submit each ──
        // Each round must be its own submit: `u_im`/`u_pm` carry per-round
        // values, and wgpu applies a `write_buffer` at the next submit.
        let mut ch0 = 0usize;
        while ch0 < n_chunks {
            let n = CONV_TILE.min(n_chunks - ch0);
            self.gd_slot.set(0);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            for level in 0..3 {
                let (input, cin0) = if level == 0 {
                    (&mel_buf, ch0)
                } else {
                    (&self.act[level - 1], 0)
                };
                if !skip.matches("im2col") {
                    self.im2col(gpu, &mut enc, level, input, cin0, n);
                }
                if !skip.matches("cgemm") {
                    self.gemm_conv(gpu, &mut enc, level);
                }
                if !skip.matches("cgelu") {
                    self.bias_gelu(gpu, &mut enc, level);
                }
            }
            if !skip.matches("permute") {
                self.permute(gpu, &mut enc, &packed, ch0, n_total, tpc, cf);
            }
            if capture {
                self.capture_round(gpu, &mut enc, &mut out, ch0, n, &mel_buf)?;
            } else {
                gpu.queue.submit([enc.finish()]);
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|e| anyhow::anyhow!("audio encoder: device lost in conv round {ch0}: {e:?}"))?;
            }
            ch0 += n;
        }

        // ── conv_out + PE ──
        // Its own submit: the transformer's first `out_proj` GEMM writes into
        // the residual buffer, so `h` must be read back before it runs.
        self.gd_slot.set(0);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        if !skip.matches("convout") {
            self.gemm(
                gpu, &mut enc, &packed, &self.conv_out.w, &h_raw,
                n_total, self.conv_out.n_pad, cf, self.conv_out.n_pad, None,
            );
        }
        if !skip.matches("pe") {
            self.add_pe(gpu, &mut enc, &h_raw, &h_buf, s_pad);
        }
        gpu.queue.submit([enc.finish()]);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow::anyhow!("audio encoder: device lost after conv_out: {e:?}"))?;
        if capture {
            out.h = read_f16_buf(gpu, &h_buf, s_pad * dm)?;
        }

        // ── transformer + projections ──
        self.mid_capture.set(capture);
        self.gd_slot.set(0);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let ctx = LayerCtx {
            geom,
            s_pad,
            acols,
            inter_pad,
            h: &h_buf,
            normed: &normed,
            norm2: &norm2,
            qkv: &qkv,
            q: &q_buf,
            k: &k_buf,
            v: &v_buf,
            scores: &scores,
            attn: &attn,
            attn_flat: &attn_flat,
            gu: &gu,
            act: &gact,
            qp: &qp,
            kt: &kt,
            vp: &vp,
            attn_out: &attn_out,
            skip: &skip,
        };
        for li in 0..self.layers.len() {
            self.layer(gpu, &mut enc, &ctx, li)?;
            if capture {
                // One submit per layer: a layer's output is overwritten by the
                // next one, and the oracle needs each in isolation.
                gpu.queue.submit([enc.finish()]);
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|e| anyhow::anyhow!("audio encoder: device lost after layer {li}: {e:?}"))?;
                enc = gpu.device.create_command_encoder(&Default::default());
                out.layers.push(read_f16_buf(gpu, &h_buf, s_pad * dm)?);
                if li == 0 {
                    let mid = self.mid_h.borrow_mut().take().unwrap_or_default();
                    let (wpad, hd_pad) = (align(wlen, GEMM_BM), align(hd, GEMM_BN));
                    out.attn = Some(AttnShot {
                        q: read_f16_buf(gpu, ctx.q, s_pad * acols)?,
                        k: read_f16_buf(gpu, ctx.k, s_pad * acols)?,
                        v: read_f16_buf(gpu, ctx.v, s_pad * acols)?,
                        scores: read_f16_buf(gpu, ctx.scores, z_blocks * wpad * wpad)?,
                        probs: read_f16_buf(gpu, ctx.attn, z_blocks * wpad * wpad)?,
                        attn_out: read_f16_buf(gpu, ctx.attn_out, z_blocks * wpad * hd_pad)?,
                        qp: read_f16_buf(gpu, ctx.qp, z_blocks * wpad * hd)?,
                        kt: read_f16_buf(gpu, ctx.kt, z_blocks * hd * wpad)?,
                        vp: read_f16_buf(gpu, ctx.vp, z_blocks * wpad * hd_pad)?,
                        hd_pad,
                        normed: read_f16_buf(gpu, ctx.normed, n_total * dm)?,
                        qkv: read_f16_buf(gpu, ctx.qkv, n_total * qkv_cols)?,
                        attn_flat: read_f16_buf(gpu, ctx.attn_flat, n_total * acols)?,
                        mid,
                        s: n_total,
                        acols,
                        nh,
                        hd,
                        wlen,
                        wpad,
                        n_win,
                    });
                }
            }
            // `SUBMIT_EVERY` layers per submit.  The reason a cadence exists at
            // all is `MAX_GEMMS` (`gd_slot`): every `dispatch_gemm` in a submit
            // takes its own 256-byte cfg slot, the stack spends 6 per layer
            // (qkv, scores, av, o, fc1, fc2), so 42 layers fit one submit and
            // the whole 24-layer stack fits with room to spare.  Each submit
            // costs a round trip to the GPU on this driver, which
            // `QALIGN_ENC_SKIP=<every group>` prices at 49.1 ms on 180s_zh.
            if !capture && (li + 1) < self.layers.len() && (li + 1) % SUBMIT_EVERY == 0 {
                gpu.queue.submit([enc.finish()]);
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|e| anyhow::anyhow!("audio encoder: device lost after layer {li}: {e:?}"))?;
                enc = gpu.device.create_command_encoder(&Default::default());
            }
        }
        // ── ln_post + projector ──
        if !skip.matches("lnpost") {
            self.layernorm(gpu, &mut enc, ctx.h, &self.ln_post, ctx.normed, geom.s);
        }
        if !skip.matches("proj1") {
            self.gemm(gpu, &mut enc, ctx.normed, &self.proj1.w, &gu, geom.s, self.proj1.n_pad, self.proj1.k, self.proj1.n_pad, None);
        }
        if !skip.matches("p1gelu") {
            self.bias_gelu_tensor(
                gpu, &mut enc, &gu, &self.proj1.bias, &gact, &self.u_sc[4],
                s_pad * self.proj1.n_pad, self.proj1.n_pad / 2, false,
            );
        }
        if !skip.matches("proj2") {
            self.gemm(
                gpu, &mut enc, &gact, &self.proj2.w, &out_emb, geom.s,
                self.proj2.n_pad, self.proj2.k, self.proj2.n_pad, Some(&self.proj2.bias),
            );
        }

        anyhow::ensure!(self.gd_slot.get() <= MAX_GEMMS, "{} GEMM slots > {MAX_GEMMS}", self.gd_slot.get());
        gpu.queue.submit([enc.finish()]);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow::anyhow!("audio encoder: device lost at readback: {e:?}"))?;

        let read_f16 = |buf: &wgpu::Buffer, n: usize| -> Result<Vec<f16>> {
            let bytes = gpu.readback(buf, (n * 2) as u64)?;
            Ok(bytes.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]])).collect())
        };
        if capture {
            out.packed = read_f16(&packed, s_pad * cf)?;
        }
        let mut embeds = read_f16(&out_emb, s_pad * out_dim)?;
        embeds.truncate(n_total * out_dim);
        out.embeds = embeds;

        ENC_MS.store((t_all.elapsed().as_secs_f64() * 1000.0) as u64, Ordering::Relaxed);
        PACK_MS.store((t_pack.as_secs_f64() * 1000.0) as u64, Ordering::Relaxed);
        skip.report(self.layers.len(), n_total);
        Ok(out)
    }

    /// Diagnostic-only: flush the round's dispatches and copy the conv tensors
    /// (operand, raw GEMM output, post-GELU activation per level) plus the two
    /// gather inputs back to the host.
    fn capture_round(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        out: &mut Capture,
        ch0: usize,
        n: usize,
        mel_buf: &wgpu::Buffer,
    ) -> Result<()> {
        let cb = std::mem::replace(enc, gpu.device.create_command_encoder(&Default::default()));
        gpu.queue.submit([cb.finish()]);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow::anyhow!("audio encoder: device lost during capture: {e:?}"))?;
        let mut r = ConvRound { chunk0: ch0, n_chunks: n, col: Vec::new(), raw: Vec::new(), act: Vec::new() };
        for level in 0..3 {
            let l = &self.conv[level];
            r.col.push(read_f16_buf(gpu, &self.col[level], l.k_pad * l.n_all)?);
            r.raw.push(read_f16_buf(gpu, &self.raw[level], l.m_pad * l.n_all)?);
            r.act.push(read_f16_buf(gpu, &self.act[level], l.m_pad * l.n_all)?);
        }
        if out.rounds.is_empty() {
            out.mel_head = read_f16_buf(gpu, mel_buf, 64)?;
            for level in 0..3 {
                let bytes = gpu.readback(&self.conv[level].taps, (TAPS * 8 * 4) as u64)?;
                out.taps.push(
                    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                );
            }
        }
        out.rounds.push(r);
        Ok(())
    }

    fn layer(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        ctx: &LayerCtx<'_>,
        li: usize,
    ) -> Result<()> {
        let l = &self.layers[li];
        let geom = ctx.geom;
        let s = geom.s;
        let dm = self.d_model;
        let (nh, hd) = (self.nh, self.hd);
        let wlen = geom.wlen;

        // See [`EncSkip`]: `skip!("…")` is true only for a group named in
        // `QALIGN_ENC_SKIP`, and an omitted group leaves its dispatch out.
        let sk = ctx.skip;
        macro_rules! skip {
            ($name:literal) => {
                sk.matches($name)
            };
        }

        // 1. self_attn_layer_norm
        if !skip!("ln") {
            self.layernorm(gpu, enc, ctx.h, &l.sln, ctx.normed, s);
        }
        // 2. fused QKV
        if !skip!("qkv") {
            self.gemm(
                gpu, enc, ctx.normed, &l.qkv.w, ctx.qkv, s, l.qkv.n_pad, l.qkv.k, l.qkv.n_pad,
                Some(&l.qkv.bias),
            );
        }
        // 3. split Q/K/V (PE was already added to the conv_out output by
        //    `add_pe`).
        if !skip!("split") {
            gpu.queue.write_buffer(
                &self.u_ex,
                0,
                bytemuck::bytes_of(&ExCfg {
                    n_tokens: s as u32,
                    nh: nh as u32,
                    hd: hd as u32,
                    tpc: self.tpc as u32,
                    row_stride: (3 * dm) as u32,
                    attn_cols: ctx.acols as u32,
                    pe_stride: 0,
                    dm: dm as u32,
                }),
            );
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("enc.extract"),
                layout: &self.p.extract.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ctx.qkv.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ctx.q.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.k.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ctx.v.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.u_ex.as_entire_binding() },
                ],
            });
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.p.extract);
            cp.set_bind_group(0, &bg, &[]);
            // One workgroup per token (`shaders::audio_extract_qkv`); `s` is
            // capped at `MAX_TOKENS`, well inside the per-dimension limit.
            cp.dispatch_workgroups(s as u32, 1, 1);
        }
        // 4. re-pack into the per-(head, window) blocks the GEMMs can read
        //    with their fixed operand row strides.
        let (wpad, hd_pad) = (align(wlen, GEMM_BM), align(hd, GEMM_BN));
        let n_blocks = nh * geom.n_win;
        if !skip!("pack") {
            gpu.queue.write_buffer(
                &self.u_wp,
                0,
                bytemuck::bytes_of(&WinCfg {
                    wlen: wlen as u32,
                    wpad: wpad as u32,
                    hd: hd as u32,
                    n_win: geom.n_win as u32,
                    s: s as u32,
                    acols: ctx.acols as u32,
                    pad_n: hd_pad as u32,
                    _a: 0,
                }),
            );
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("enc.win_pack"),
                layout: &self.p.win_pack.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ctx.q.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ctx.k.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.v.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ctx.qp.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: ctx.kt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: ctx.vp.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.u_wp.as_entire_binding() },
                ],
            });
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.p.win_pack);
            cp.set_bind_group(0, &bg, &[]);
            // x: 16 token pairs each; y: 16 head-dim words each.
            cp.dispatch_workgroups((wpad / 32) as u32, (hd_pad / 2 / 16) as u32, n_blocks as u32);
        }
        // 5. scores: `[wpad, wpad] = Qp · Ktᵀ` per (head, window) block.
        //    `kam` = hd, so the A row stride is `hd/2` words (Qp's rows) and the
        //    `transb` B row stride is `wpad/2` (Kt's rows) — both block strides
        //    are dense, one block per `z`.
        if !skip!("scores") {
            let bsa = wpad * hd / 2;
            let bsb = hd * wpad / 2;
            // `bsc` is in ELEMENTS: the epilogue divides the flat C index by 2.
            let bsc = wpad * wpad;
            self.gemm_batched(
                gpu, enc, &self.p.gemm_t, ctx.qp, ctx.kt, ctx.scores,
                wpad, wpad, hd, wpad, bsa, bsb, bsc, n_blocks, 1, 1, 0,
            );
        }
        // 6. windowed softmax (masks the short last window), scores -> probs
        if !skip!("sm") {
            gpu.queue.write_buffer(
                &self.u_sm,
                0,
                bytemuck::bytes_of(&SmCfg {
                    s: s as u32,
                    wlen: wlen as u32,
                    wpad: wpad as u32,
                    n_win: geom.n_win as u32,
                    scale: 1.0 / (hd as f32).sqrt(),
                    _a: 0,
                    _b: 0,
                    _c: 0,
                }),
            );
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("enc.sm"),
                layout: &self.p.softmax.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ctx.scores.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ctx.attn.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.u_sm.as_entire_binding() },
                ],
            });
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.p.softmax);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((wpad / 128) as u32, n_blocks as u32, 1);
        }
        // 7. AV: `[wpad, hd_pad] = P · Vp`, same block layout.
        if !skip!("av") {
            let bsa = wpad * wpad / 2;
            let bsb = wpad * hd_pad / 2;
            // Elements, like the scores GEMM's `bsc` above.
            let bsc = wpad * hd_pad;
            self.gemm_batched(
                gpu, enc, &self.p.gemm_t, ctx.attn, ctx.vp, ctx.attn_out,
                wpad, hd_pad, wpad, hd_pad, bsa, bsb, bsc, n_blocks, 1, 1, 0,
            );
        }
        // 8. compress the blocks into `[tok, nh·hd]`
        if !skip!("flat") {
            gpu.queue.write_buffer(
                &self.u_cp,
                0,
                bytemuck::bytes_of(&CpCfg {
                    cols: (nh * hd) as u32,
                    wlen: wlen as u32,
                    wpad: wpad as u32,
                    hd: hd as u32,
                    n_win: geom.n_win as u32,
                    rows: s as u32,
                    _a: 0,
                    _b: 0,
                }),
            );
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("enc.attn_flat"),
                layout: &self.p.attn_flat.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ctx.attn_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ctx.attn_flat.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.u_cp.as_entire_binding() },
                ],
            });
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.p.attn_flat);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((ctx.s_pad * ctx.acols / 2).div_ceil(256) as u32, 1, 1);
        }
        // 9. out_proj, added onto the residual (`beta = 1`)
        if !skip!("o") {
            self.gemm_beta(gpu, enc, ctx.attn_flat, &l.o.w, ctx.h, s, l.o.n_pad, l.o.k, dm, Some(&l.o.bias));
        }
        if self.mid_capture.get() && li == 0 {
            let cb = std::mem::replace(enc, gpu.device.create_command_encoder(&Default::default()));
            gpu.queue.submit([cb.finish()]);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| anyhow::anyhow!("audio encoder: device lost at mid capture: {e:?}"))?;
            *self.mid_h.borrow_mut() = Some(read_f16_buf(gpu, ctx.h, ctx.s_pad * dm)?);
        }
        // 10. final_layer_norm -> FFN (fc1, GELU, fc2) + residual
        if !skip!("fln") {
            self.layernorm(gpu, enc, ctx.h, &l.fln, ctx.norm2, s);
        }
        if !skip!("fc1") {
            self.gemm(gpu, enc, ctx.norm2, &l.fc1.w, ctx.gu, s, l.fc1.n_pad, l.fc1.k, l.fc1.n_pad, None);
        }
        if !skip!("gelu") {
            self.bias_gelu_tensor(
                gpu, enc, ctx.gu, &l.fc1.bias, ctx.act, &self.u_sc[3],
                ctx.s_pad * l.fc1.n_pad, l.fc1.n_pad / 2, false,
            );
        }
        if !skip!("fc2") {
            self.gemm_beta(gpu, enc, ctx.act, &l.fc2.w, ctx.h, s, l.fc2.n_pad, l.fc2.k, dm, Some(&l.fc2.bias));
        }
        Ok(())
    }

    // ── dispatch helpers ──

    /// `C += A·Wᵀ` — the residual form of [`Self::gemm`]; the transformer's two
    /// residual adds accumulate onto the residual buffer.
    #[allow(clippy::too_many_arguments)]
    fn gemm_beta(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
        bias: Option<&wgpu::Buffer>,
    ) {
        let pipe = if bias.is_some() { &self.p.gemm_beta_bias } else { &self.p.gemm_beta };
        self.dispatch_gemm(
            gpu, enc, pipe, a, w, c, m, n, k, ldc, 0, 0, 0, 1,
            (n / GEMM_BN) as u32, (align(m, GEMM_BM) / GEMM_BM) as u32, bias,
        );
    }

    /// As [`Self::gemm_bind`], with the per-column bias at binding 4.
    fn gemm_bind_bias(
        &self,
        gpu: &Gpu,
        pipe: &wgpu::ComputePipeline,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        bias: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("enc.gemm_bias"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.u_gd,
                        offset: 0,
                        size: std::num::NonZeroU64::new(64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 4, resource: bias.as_entire_binding() },
            ],
        })
    }

    fn gemm_bind(&self, gpu: &Gpu, pipe: &wgpu::ComputePipeline, a: &wgpu::Buffer, w: &wgpu::Buffer, c: &wgpu::Buffer) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("enc.gemm"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
                // The binding window must be ONE slot, not the whole ring: a
                // dynamic offset is only legal up to `binding_size - slot`.
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.u_gd,
                        offset: 0,
                        size: std::num::NonZeroU64::new(64),
                    }),
                },
            ],
        })
    }

    /// Single-tile GEMM at `prefill_gemm`'s `GEMM_BM × GEMM_BN` tile edges: `a`
    /// is `[m, k]`, `w` is `[n, k]` row-major, `c` is `[m, ldc]` — all packed
    /// f16, with `m` rounded up to `GEMM_BM` and `n` a multiple of `GEMM_BN`.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
        bias: Option<&wgpu::Buffer>,
    ) {
        let pipe = if bias.is_some() { &self.p.gemm_bias } else { &self.p.gemm };
        self.dispatch_gemm(
            gpu, enc, pipe, a, w, c, m, n, k, ldc, 0, 0, 0, 1,
            (n / GEMM_BN) as u32, (align(m, GEMM_BM) / GEMM_BM) as u32, bias,
        );
    }

    /// The conv GEMM for one level: `A` is the weight `[m_pad][k_pad]`, `B` the
    /// im2col operand read with `transb = 1` (`[k_pad][n_all]`, row stride
    /// `n_all/2` words).  Channel lands on `m` and position on `n`, so `C` is
    /// channel-major; `n_all == ldc`, one row per channel.
    fn gemm_conv(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, level: usize) {
        let l = &self.conv[level];
        self.dispatch_gemm(
            gpu, enc, &self.p.gemm_t, &l.w.w, &self.col[level], &self.raw[level],
            l.m_pad, l.n_all, l.k_pad, l.n_all, 0, 0, 0, 1,
            (l.n_all / GEMM_BN) as u32, (l.m_pad / GEMM_BM) as u32, None,
        );
    }

    /// The gather for one level: `cin0` is the global index of the round's first
    /// chunk in `input` (0 for round-local levels, `chunk0` for the mel).
    fn im2col(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        level: usize,
        input: &wgpu::Buffer,
        cin0: usize,
        n_chunks_round: usize,
    ) {
        let l = &self.conv[level];
        gpu.queue.write_buffer(
            &self.u_im[level],
            0,
            bytemuck::bytes_of(&Im2Cfg {
                taps: TAPS as u32,
                k: l.k as u32,
                k_pad: l.k_pad as u32,
                plane: l.plane as u32,
                plane_pad: l.plane_pad as u32,
                n_chunks: n_chunks_round as u32,
                n_all: l.n_all as u32,
                in_chunk: l.in_chunk as u32,
                in_ic: l.in_ic as u32,
                chunk0: cin0 as u32,
                bpc: (l.plane_pad / 32) as u32,
                _a: 0,
            }),
        );
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("enc.im2col"),
            layout: &self.p.im2col.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: l.taps.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.col[level].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.u_im[level].as_entire_binding() },
            ],
        });
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&self.p.im2col);
        cp.set_bind_group(0, &bg, &[]);
        // x: 32-position blocks (16 position pairs per 16-lane row), covering
        // the whole round's `n_all` columns; y: 16 k rows each.
        cp.dispatch_workgroups(
            (l.n_all / 32) as u32,
            (l.k_pad / 16) as u32,
            1,
        );
    }

    /// `dst[i] = gelu(src[i] + bias[i])` for the conv levels: the activation is
    /// channel-major, so the channel is `i / (n_all/2)` words in.
    fn bias_gelu(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, level: usize) {
        let l = &self.conv[level];
        self.bias_gelu_tensor(
            gpu,
            enc,
            &self.raw[level],
            &l.w.bias,
            &self.act[level],
            &self.u_sc[level],
            l.m_pad * l.n_all,
            l.n_all / 2,
            true,
        );
    }

    /// Gather the round's conv3 activation into `packed` — the `conv_out`
    /// operand, `[token][c·f]`, whose row stride is exactly `conv_out.k`
    /// (`transb` is off there, so the kernel's `kk = k/2` is the row stride).
    fn permute(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        packed: &wgpu::Buffer,
        ch0: usize,
        n_total: usize,
        tpc: usize,
        cf: usize,
    ) {
        let l = &self.conv[2];
        gpu.queue.write_buffer(
            &self.u_pm,
            0,
            bytemuck::bytes_of(&PmCfg {
                c: l.c_out as u32,
                f: (l.plane / tpc) as u32,
                t3: tpc as u32,
                s_pad: cf as u32,
                n_all: l.n_all as u32,
                plane_pad: l.plane_pad as u32,
                tok0: (ch0 * tpc) as u32,
                n_tokens: n_total as u32,
            }),
        );
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("enc.permute"),
            layout: &self.p.permute.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.act[2].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: packed.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.u_pm.as_entire_binding() },
            ],
        });
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&self.p.permute);
        cp.set_bind_group(0, &bg, &[]);
        // One row per token — the grid is padded so the `packed` rows past
        // `n_total` (which the conv_out GEMM still reads) are written zero.
        cp.dispatch_workgroups(align(n_total, GEMM_BM) as u32, (cf / 2 / 256) as u32, 1);
    }

    /// `h += PE[tok % tpc]` on the `conv_out` output.
    fn add_pe(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        rows: usize,
    ) {
        let dm = self.d_model;
        gpu.queue.write_buffer(
            &self.u_pe,
            0,
            bytemuck::bytes_of(&PeCfg { d: dm as u32, tpc: self.tpc as u32, s_pad: dm as u32, _a: 0 }),
        );
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("enc.add_pe"),
            layout: &self.p.add_pe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.pe.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.u_pe.as_entire_binding() },
            ],
        });
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&self.p.add_pe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(align(rows, GEMM_BM) as u32, (dm / 2).div_ceil(256) as u32, 1);
    }

    /// One `prefill_gemm` dispatch per `(head, window)` block; `bsa`/`bsb` are in
    /// words, `bsc` in elements — see [`crate::shaders::prefill_gemm`].
    #[allow(clippy::too_many_arguments)]
    fn gemm_batched(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        pipe: &wgpu::ComputePipeline,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
        bsa: usize,
        bsb: usize,
        bsc: usize,
        batch: usize,
        gx: u32,
        gy: u32,
        _reserved: usize,
    ) {
        self.dispatch_gemm(gpu, enc, pipe, a, w, c, m, n, k, ldc, bsa, bsb, bsc, batch, gx, gy, None);
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_gemm(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        pipe: &wgpu::ComputePipeline,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
        bsa: usize,
        bsb: usize,
        bsc: usize,
        batch: usize,
        gx: u32,
        gy: u32,
        bias: Option<&wgpu::Buffer>,
    ) {
        let slot = self.gd_slot.get();
        self.gd_slot.set(slot + 1);
        // Slots sit at distinct 256 B offsets and each dispatch reads only its
        // own; deferred `write_buffer`s land at submit start, so a single
        // reused slot would hand every dispatch the last-written values.
        gpu.queue.write_buffer(
            &self.u_gd,
            (slot * 256) as u64,
            bytemuck::bytes_of(&GDims {
                m: align(m, GEMM_BM) as u32,
                n: n as u32,
                k: k as u32,
                ldc: ldc as u32,
                bsa: bsa as u32,
                bsb: bsb as u32,
                bsc: bsc as u32,
                beta: 0,
                row0: 0,
                lda: k as u32,
            }),
        );
        let bg = match bias {
            Some(b) => self.gemm_bind_bias(gpu, pipe, a, w, c, b),
            None => self.gemm_bind(gpu, pipe, a, w, c),
        };
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(pipe);
        cp.set_bind_group(0, &bg, &[(slot * 256) as u32]);
        cp.dispatch_workgroups(gx, gy, batch.max(1) as u32);
    }

    /// `dst[i] = gelu(src[i] + bias[c(i)])` over `n` f16 elements, two per thread.
    ///
    /// `bias` is the per-channel vector, padded to the activation's channel
    /// count with zeros.  `by_channel` picks `i / words` (channel-major conv
    /// activation) or `i % words` (token-major GEMM output); `n` must be even so
    /// a thread's two halves never straddle the channel vector's padding.
    fn bias_gelu_tensor(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        sc: &wgpu::Buffer,
        n: usize,
        words: usize,
        by_channel: bool,
    ) {
        assert!(n % 2 == 0, "bias_gelu needs an even element count");
        // 256 threads × 2 halves per workgroup; `x` is capped at wgpu's 65535
        // per-dimension limit and `y` continues the index space (a long clip's
        // FFN activation needs >100k workgroups).
        let (gx, gy) = crate::decoder::grid_xy(n.div_ceil(512));
        gpu.queue.write_buffer(
            sc,
            0,
            bytemuck::bytes_of(&ScaleCfg {
                n: n as u32,
                words: words as u32,
                mode: u32::from(by_channel),
                gx,
            }),
        );
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("enc.gelu"),
            layout: &self.p.gelu.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bias.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sc.as_entire_binding() },
            ],
        });
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&self.p.gelu);
        cp.set_bind_group(0, &bg, &[]);
        // 256 threads × 2 halves = 512 elements per workgroup.
        cp.dispatch_workgroups(gx, gy, 1);
    }

    fn layernorm(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        ln: &GpuLayerNorm,
        dst: &wgpu::Buffer,
        rows: usize,
    ) {
        let mut u: [u8; 32] = [0; 32];
        u[0..4].copy_from_slice(&(self.d_model as u32).to_le_bytes());
        u[4..8].copy_from_slice(&ln.eps.to_le_bytes());
        gpu.upload(&self.u_ln, &u);
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("enc.ln"),
            layout: &self.p.layernorm.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: ln.w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ln.b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.u_ln.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dst.as_entire_binding() },
            ],
        });
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&self.p.layernorm);
        cp.set_bind_group(0, &bg, &[]);
        // `AUDIO_LN_ROWS` rows per workgroup.  The grid still covers the same
        // padded rows the row-per-workgroup form did: `align(rows, GEMM_BM)` is
        // a multiple of the row count, so the staging never leaves the buffer.
        cp.dispatch_workgroups(rows.div_ceil(shaders::AUDIO_LN_ROWS) as u32, 1, 1);
    }
}

// ─── timing hooks ──────────────────────────────────────────────────

static ENC_MS: AtomicU64 = AtomicU64::new(0);
static PACK_MS: AtomicU64 = AtomicU64::new(0);

pub fn last_encode_ms() -> f64 {
    ENC_MS.load(Ordering::Relaxed) as f64
}
pub fn last_pack_ms() -> f64 {
    PACK_MS.load(Ordering::Relaxed) as f64
}

// ─── helpers ───────────────────────────────────────────────────────

/// Explicit layout for the GEMM family: storage buffers `0..n_storage` plus a
/// dynamic-offset uniform, since every GEMM dispatch reads its own `GDims` slot
/// out of one ring buffer.
///
/// `read_write` names the bindings the shader declares `read_write` (only the
/// GEMM's `C`).
fn gemm_layout(gpu: &Gpu, label: &str, n_storage: u32, read_write: &[u32]) -> wgpu::PipelineLayout {
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n_storage)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: !read_write.contains(&binding),
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: n_storage,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    });
    let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_bgl")),
        entries: &entries,
    });
    gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pl")),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    })
}

/// GEMM layout with a per-column bias: storage 0..3, the dynamic-offset uniform
/// at 3 (as [`gemm_layout`]) and the bias as storage 4.
fn gemm_bias_layout(gpu: &Gpu, label: &str, read_write: &[u32]) -> wgpu::PipelineLayout {
    // Bindings 0..=2 storage, 3 the dynamic-offset uniform, 4 the bias — so the
    // range has to be 0..5 with the uniform overriding index 3.
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..5u32)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: !read_write.contains(&binding) },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    entries[3] = wgpu::BindGroupLayoutEntry {
        binding: 3,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    };
    let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}_bgl")),
        entries: &entries,
    });
    gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pl")),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    })
}

/// Read `n` packed 16-bit elements back as a vector in the run's format.
fn read_f16_buf(gpu: &Gpu, buf: &wgpu::Buffer, n: usize) -> Result<Vec<f16>> {
    let bytes = gpu.readback(buf, (n * 2) as u64)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// One `conv2d(3×3, stride 2, pad 1)` output length.  Three `conv_out_len` steps
/// are not the same as `feo` (`feo(100) = 13`, `conv_out_len³(100) = 25`), so the
/// single-step form is used per layer and `feo` only for the tail count.
#[inline]
/// Three `conv_out_len` steps: the *token* count a chunk produces, i.e. the
/// height of conv3's output plane (`feo(n_mels)` at the mel's resolution).
pub(crate) fn feo_positions(n_mels: usize) -> usize {
    let f = |l: usize| (l + 2 - 3) / 2 + 1;
    f(f(f(n_mels)))
}

pub(crate) fn conv_out_len(l: usize) -> usize {
    (l + 2 - 3) / 2 + 1
}

/// Elements of one mel "plane": all mel bins of one `cs`-frame chunk, padded to
/// `w0`.
#[inline]
fn mel_elems(n_mels: usize, w0: usize) -> usize {
    n_mels * w0
}

/// Zero-pad a `[rows, cols]` matrix's columns to the next even count: the GEMM
/// reads packed 16-bit pairs along k, so an odd k shears every row by half a
/// word.
fn pad_k(w: &PackedWeight) -> Result<PackedWeight> {
    let kw = pad_k_tile(w.cols);
    if kw == w.cols {
        return Ok(PackedWeight {
            data: w.data.clone(),
            rows: w.rows,
            cols: w.cols,
        });
    }
    let mut data = Vec::with_capacity(w.rows * kw * 2);
    for r in 0..w.rows {
        data.extend_from_slice(&w.data[r * w.cols * 2..(r + 1) * w.cols * 2]);
        data.resize(data.len() + (kw - w.cols) * 2, 0);
    }
    Ok(PackedWeight {
        data: data.into(),
        rows: w.rows,
        cols: kw,
    })
}

/// Zero-pad a `[rows, cols]` f16 matrix to `new_rows`, the n-tile width of
/// `prefill_gemm`, keeping every dispatch on a whole tile and every stride exact.
fn pad_rows(w: &PackedWeight, new_rows: usize) -> Result<PackedWeight> {
    anyhow::ensure!(new_rows >= w.rows, "pad_rows: shrink not supported");
    if new_rows == w.rows {
        return Ok(PackedWeight {
            data: w.data.clone(),
            rows: w.rows,
            cols: w.cols,
        });
    }
    let mut data = w.data.to_vec();
    data.resize(new_rows * w.cols * 2, 0);
    Ok(PackedWeight {
        data: data.into(),
        rows: new_rows,
        cols: w.cols,
    })
}

/// Tap-offset table for a `3×3/s2/p1` conv over an `h × w_in` plane.
///
/// `off[pos*9 + tap]` is that tap's source offset inside one input plane,
/// `TAP_OOB` outside it; positions are `(h_out, w_out)` row-major, the order the
/// GEMM's `n` axis uses.  `w_in` is the true width, `w_stride` the buffer's row
/// stride (`w0` for the mel, else `= w_in`).
fn conv_taps(h: usize, w_in: usize, w_stride: usize) -> (Vec<u32>, usize, usize, usize) {
    let h_out = (h + 2 - 3) / 2 + 1;
    let w_out = (w_in + 2 - 3) / 2 + 1;
    let mut off = vec![TAP_OOB; h_out * w_out * TAPS];
    for ho in 0..h_out {
        for wo in 0..w_out {
            for kh in 0..3 {
                for kw in 0..3 {
                    let ih = (ho * 2 + kh) as isize - 1;
                    let iw = (wo * 2 + kw) as isize - 1;
                    let v = if ih < 0 || ih >= h as isize || iw < 0 || iw >= w_in as isize {
                        TAP_OOB
                    } else {
                        (ih as usize * w_stride + iw as usize) as u32
                    };
                    off[(ho * w_out + wo) * TAPS + kh * 3 + kw] = v;
                }
            }
        }
    }
    (off, h_out * w_out, h_out, w_out)
}

/// A linear's bias padded to `n_pad` with zeros, so the GEMM's `bias = 1`
/// epilogue and `bias_gelu` can index it without a bound check.
fn upload_bias(up: &mut BulkUpload, label: &str, n: usize, n_pad: usize, host: &[f16]) -> Result<wgpu::Buffer> {
    let mut v = vec![f16::ZERO; n_pad.max(n)];
    v[..n].copy_from_slice(host);
    upload_f16(up, label, &v)
}

/// `conv_out.k` in 16-wide words for the permute kernel's y grid.
fn conv_out_cols_16(k: usize) -> u32 {
    (align(k, 32) / 32) as u32
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn align(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}

fn pad_even(v: usize) -> usize {
    if v % 2 == 1 {
        v + 1
    } else {
        v
    }
}

/// Round `k` up to a multiple of the GEMM's k-tile (`PREFILL_GEMM_BK`): `k` must
/// be even (the A operand is `array<u32>` with a `k/2` word row stride) and the
/// k loop's last whole 16-wide tile reads `align(k, 16)` values, so the padding
/// is zero on both sides only when `k` is a multiple of `BK`.
#[inline]
fn pad_k_tile(k: usize) -> usize {
    align(k, GEMM_BK)
}

fn upload_words(up: &mut BulkUpload, label: &str, w: &PackedWeight) -> Result<wgpu::Buffer> {
    let b = up.storage(label, w.data.len() as u64);
    up.upload(&b, &w.data)?;
    Ok(b)
}

fn upload_f16_now(gpu: &Gpu, label: &str, v: &[f16]) -> Result<wgpu::Buffer> {
    let b = gpu.storage(label, (v.len() * 2) as u64);
    gpu.upload(&b, &weights::words_bytes(v));
    Ok(b)
}

fn upload_f16(up: &mut BulkUpload, label: &str, v: &[f16]) -> Result<wgpu::Buffer> {
    let b = up.storage(label, (v.len() * 2) as u64);
    up.upload(&b, &weights::words_bytes(v))?;
    Ok(b)
}

























