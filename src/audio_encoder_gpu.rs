// Lifted from the ASR port `D:\qwen3-asr-wgpu` verbatim.  A few of its geometry
// fields and helpers are read only by that project's probe binaries, which this
// crate did not take; they are kept so the file remains a faithful copy of the
// verified original rather than a fork to re-diff on the next lift.
#![allow(dead_code)]

//! GPU audio encoder for Qwen3-ASR — wgpu port of the CUDA audio tower.
//!
//! Architecture (identical math to [`crate::audio_encoder`], the CPU reference):
//!
//! ```text
//! mel [128, frames]  →  chunk to [chunks, 1, 128, 100]
//!   conv2d(3×3, s2, p1) + bias + GELU   ×3   (im2col → GEMM)
//!   → permute [c, f, t] → [t, c·f]  →  conv_out Linear + sinusoidal PE
//!   → 18 × { LayerNorm + windowed attn + LayerNorm + FFN(GELU-erf) }
//!   → ln_post + proj1 + GELU + proj2  → [n_total, 1024]
//! ```
//!
//! Design notes that matter for correctness and speed:
//!
//! * **im2col is a gather driven by a precomputed tap table** (see
//!   [`shaders::audio_im2col`]) — no divisions or boundary tests in the inner
//!   loop, coalesced column stores.
//! * **Every GEMM is the text decoder's `prefill_gemm`** (64×64×16, f16
//!   operands, f32 accumulate, `pack2x16float` epilogue), so the audio tower
//!   does not invent a second accumulation order.
//! * **Attention is windowed, not causal.**  Each of the `ceil(s/wlen)` windows
//!   is an independent full attention over `wlen` tokens
//!   (`wlen = feo(100)·n_window_infer/100` = 104 at the shipped config, padded to
//!   the GEMM tile `wpad = align(wlen, 128)`).  `audio_win_pack` re-packs Q/K/V
//!   into one dense block per `(head, window)`, `z = head·n_win + win`
//!   ([`shaders::audio_win_pack`]), and the two batched GEMMs index those blocks
//!   with `wid.z` — see [`shaders::prefill_gemm`] for the batch-stride units.
//! * Conv planes are **width-padded to an even stride** (`w → w+1`) so the
//!   im2col's packed-pair read always lands in bounds; the pad column lives only
//!   in the tap table, never in the GEMM's `k` axis.
//! * Weights stay f16 (the checkpoint dtype); activations are f16 with f32
//!   accumulation, exactly like `WgpuTextDecoder::prefill`.
//!
//! This is the default tower (`transcribe` uses it unless `--cpu-enc` is given);
//! `--compare-enc` cross-checks the embedding tensor against the CPU reference,
//! and `--diag-enc` compares every stage of it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use half::f16;

use crate::audio_encoder::feo;
use crate::config::AudioEncoderConfig;
use crate::gpu::{BulkUpload, Gpu};
use crate::shaders;
use crate::weights::{self, PackedWeight, RawTensor};

/// Tile edges of `shaders::prefill_gemm`, taken **from the kernel's own
/// constants** rather than re-typed here.
///
/// These used to be hard-coded `64`, while the kernel's tiles are 128 wide
/// (`bm = 16 * tm`).  Nothing failed loudly: the dispatch grids and buffer pads
/// were derived from the wrong number, so half of each axis was simply never
/// computed.  Importing them makes that drift impossible.
const GEMM_BM: usize = shaders::PREFILL_GEMM_BM;
const GEMM_BN: usize = shaders::PREFILL_GEMM_BN;
const GEMM_BK: usize = shaders::PREFILL_GEMM_BK;
/// Conv taps: `3×3`, always one whole k-group of 16 per input channel.
const TAPS: usize = 9;
/// Token capacity of the preallocated pipeline (~22 min of audio at 12.5 tok/s).
const MAX_TOKENS: usize = 16384;
/// Largest attention window: `feo(100) · n_window_infer/100` = 1250 shipped.
const MAX_WINDOW: usize = 2048;
/// Uniform ring for per-dispatch `GDims` (256 B slots, dynamic offset).
/// Capped at 256 slots = 64 KiB, which is `max_uniform_buffer_binding_size`
/// on this stack (the text decoder's prefill ring uses the same budget).
const MAX_GEMMS: usize = 256;
/// Chunks processed per conv-stem round.
///
/// The conv is chunk-independent, but the *operands* are not small: a 3×3 conv
/// over 480 channels has `9 × 480 = 4320` k values per position, so one round's
/// im2col operand is ~9× the activation it produces.  At the whole-clip batch
/// (177 chunks on a 180 s clip) the c1 activation alone would be 580 MB; tiling
/// the chunk axis keeps the whole stem inside ~140 MB while leaving the GEMMs
/// ~100-400 workgroups.  Mirrors the CPU reference's `CONV_TILE`.
const CONV_TILE: usize = 8;
/// `CONV_TILE`, published for the diagnostic's round bookkeeping.
pub fn conv_tile() -> usize {
    CONV_TILE
}

/// Out-of-plane tap sentinel — **re-exported**, not redefined: the tap tables
/// and the WGSL that reads them must agree or every out-of-bounds tap becomes a
/// valid address (which is precisely what happened at `0xFFFF_FFFF` vs
/// `0xFFFF`).
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
    /// Key-tile row offset in the B operand; 0 for every tower GEMM (the field
    /// exists because the tower shares `shaders::prefill_gemm` with the decoder,
    /// where the slabbed attention uses it).
    row0: u32,
    /// A row stride in elements; `k` for every tower GEMM (same reason as
    /// `row0` — the decoder's slabbed AV gives its score slab a wider stride).
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
        let bias = weights::get_f16(weights, &format!("{prefix}.bias"))
            .ok()
            .map(|(b, _)| b);
        let w = pad_k(&pad_rows(&w, align(n, GEMM_BM))?)?;
        Self::finish(up, label, w, bias, n, pad_k_tile(k))
    }

    /// Conv weights arrive as `[c_out, c_in, 3, 3]` (or `[c_out, 1, 3, 3]`):
    /// row-major flattening of the trailing axes is exactly the im2col k order,
    /// so this is a plain 2-D view.
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
        let (w, shape) = t.as_f16()?;
        anyhow::ensure!(shape[0] == n && w.len() == n * k);
        let (bias, bshape) = weights::get_f16(weights, &format!("{prefix}.bias"))?;
        anyhow::ensure!(bshape.len() == 1 && bshape[0] == n, "{prefix}.bias {bshape:?} != [{n}]");
        let packed = pad_k(&PackedWeight::from_f16(&w, n, k))?;
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
            let (b, _) = weights::get_f16(weights, &format!("{prefix}.{p}.bias"))?;
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
        let w = weights::get_vector(weights, &format!("{prefix}.weight"))?;
        let b = weights::get_vector(weights, &format!("{prefix}.bias"))?;
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
    /// `proj2`), and `prefill_gemm(false, true, true)` for the residual+biased
    /// pair (`out_proj`, `fc2`).
    gemm_bias: wgpu::ComputePipeline,
    gemm_beta_bias: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    layernorm: wgpu::ComputePipeline,
    permute: wgpu::ComputePipeline,
    add_pe: wgpu::ComputePipeline,
    attn_flat: wgpu::ComputePipeline,
}

// ─── Conv geometry: the single source of truth ────────────────────

/// One conv round's tensors, in the buffers' own layouts — the comparator in
/// `inference.rs` indexes them with the same `ConvLevel` fields the dispatches
/// use, so a layout change cannot silently desynchronise the two.
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
    /// `[z][wpad][wpad]` raw scores and softmax probabilities, `[z][wpad][128]`
    /// AV output.
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
    /// `[s_pad][d_model]` after each transformer layer (capture mode submits
    /// once per layer, so this is what every layer actually produced).
    pub layers: Vec<Vec<f16>>,
    /// Layer 0's attention internals, for a self-contained oracle.
    pub attn: Option<AttnShot>,
    pub embeds: Vec<f16>,
}

/// One conv level's geometry and weights.
///
/// Terms (all counts are in f16 *elements*, not bytes):
///
/// * `position` — one `(h_out, w_out)` cell of this level's output plane,
///   flattened row-major (`p = ho·w_out + wo`).  A chunk produces `plane` of
///   them; `plane_pad = align(plane, 32)` is the per-chunk stride every buffer
///   and dispatch uses.
/// * `n_all = align(CONV_TILE·plane_pad, GEMM_BN)` — the position count the
///   whole round is laid out on: the GEMM's `n`, and the **channel stride** of
///   this level's activation.  It is deliberately the same for every round
///   (including the last, partial one) so the broadcast bias image has one
///   shape; the positions a short round does not have are zero in the operand.
/// * operand `[k_pad][n_all]` — the im2col output, i.e. the GEMM's `B` operand
///   read with `transb = true` (row stride `n_all/2` words).
/// * activation `[m_pad][n_all]` — the GEMM's `C`, channel-major with the
///   channel stride `n_all` and `m_pad = align(c_out, GEMM_BM)` rows.  This is
///   the layout `shaders::audio_permute_pe` and the next level's gather read,
///   and it is the CPU reference's `[c_out][pos]` layout padded on both axes.
///
/// The input side is what the gather addresses: tap `t` of `(p, k)` reads
/// `(chunk₀ + chunk)·in_chunk + ic·in_ic + tap`, `ic = k/9`, `tap` from the
/// `[plane_in·9]` table.  `in_chunk` is one chunk's image, `in_ic` the channel
/// stride (0 for the mel, where `c_in == 1` makes `ic` always 0).
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
    /// Build one level from the checkpoint weights and its **input** geometry.
    ///
    /// `h_in`/`w_in` are the true input plane the convolution runs over,
    /// `w_stride` the input buffer's row stride (`w0` for the mel, else
    /// `w_in`), `in_chunk`/`in_ic` what the gather strides read the input with
    /// (see the struct docs) and `n_all` this level's position count.
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

/// A read-only view of one [`ConvLevel`], for the diagnostic comparator: it
/// publishes exactly the numbers the dispatches are built from, so the two
/// cannot drift.
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
    /// `proj1` bias broadcast to `[rows][proj1.n_pad]` — the final GELU runs on
    /// a token-major activation, so its image is token-major too.
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
    /// Clip-sized tensors (`packed`, `mel_in`, the transformer's
    /// activations) are allocated per `encode`: at `MAX_TOKENS` rows the big
    /// ones are hundreds of MB each, and a clip rarely needs a third of that.

    // uniforms
    u_gd: wgpu::Buffer,
    u_im: [wgpu::Buffer; 3],
    /// One `ScaleCfg` per *distinct* GELU shape — the three conv levels (whose
    /// shapes are fixed), the FFN and `proj1` (sized per clip).  A single shared
    /// uniform is wrong however many times it is written: `queue.write_buffer`
    /// lands at the *next* submit, so every dispatch in one command buffer would
    /// read the last value written, not its own.
    u_sc: [wgpu::Buffer; 5],
    u_ex: wgpu::Buffer,
    u_sm: wgpu::Buffer,
    u_ln: wgpu::Buffer,
    u_pm: wgpu::Buffer,
    u_pe: wgpu::Buffer,
    u_wp: wgpu::Buffer,
    u_cp: wgpu::Buffer,

    gd_slot: std::cell::Cell<usize>,
    /// Diagnostic: snapshot `h` inside layer 0 (between the attention and the
    /// FFN) so a layer-level mismatch can be split in two.
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

/// Per-pass activation buffers, sized to this clip.  Grouped so the layer
/// dispatches take one argument instead of fifteen.
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
        // tap tables are built from the input dims.  `conv_taps(h_out, w_out)` is
        // a different (wrong) table: fed conv1's output dims it produced
        // 32×25 = 800 positions, which is conv2's output, and 56 for conv3.
        //
        // `conv_out_len` is one `conv2d(3×3, s2, p1)` step; `feo` is three of
        // them, i.e. the *token* count, not any single level's width.
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
        // whole round's position count and therefore the activation's channel
        // stride.  Both are constant across rounds, so the broadcast bias image
        // and the GEMM's `n` never move.
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
        // `proj1` maps `d_model` (896) → `output_dim` (1024) and `proj2` keeps
        // that width; the shipped 0.6B checkpoint is `[896, 896]`/`[1024, 896]`
        // because the audio tower's own `d_model` is the projection width.
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
        // level gathers from.  Each gets its own buffer: the old code wrote the
        // GELU result back over the operand the next gather still needed — in a
        // third layout again.
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

    /// Encode `[n_mels, n_frames]` (mel-bin major) into `[n_tokens, out_dim]` f16.
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
    /// the host.  `cap` forces a submit+poll per readback, so it is a
    /// diagnostic path only: the production call passes `None` and runs a whole
    /// round — and then the transformer — without a host round trip.
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
        let mel_f16: Vec<f16> = chunked.iter().map(|v| f16::from_f32(*v)).collect();
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

        let mel_buf = gpu.storage("enc.mel", (mel_f16.len() * 2) as u64);
        gpu.upload(&mel_buf, &weights::words_bytes(&mel_f16));
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

        // ── conv stem: `CONV_TILE` chunks per round, one submit each ──
        // The round is the unit that has to be submitted on its own: `u_im` /
        // `u_pm` carry per-round values, and wgpu applies a `write_buffer` at
        // the *next* submit, so two rounds in one submit would both read the
        // second round's config.
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
                self.im2col(gpu, &mut enc, level, input, cin0, n);
                self.gemm_conv(gpu, &mut enc, level);
                self.bias_gelu(gpu, &mut enc, level);
            }
            self.permute(gpu, &mut enc, &packed, ch0, n_total, tpc, cf);
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
        // the residual buffer, so `h` has to be read back before it runs (and
        // the debug capture needs the boundary anyway).
        self.gd_slot.set(0);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.gemm(
            gpu, &mut enc, &packed, &self.conv_out.w, &h_raw,
            n_total, self.conv_out.n_pad, cf, self.conv_out.n_pad, None,
        );
        self.add_pe(gpu, &mut enc, &h_raw, &h_buf, s_pad);
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
            if !capture && (li + 1) % 6 == 0 && li + 1 < self.layers.len() {
                gpu.queue.submit([enc.finish()]);
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|e| anyhow::anyhow!("audio encoder: device lost after layer {li}: {e:?}"))?;
                enc = gpu.device.create_command_encoder(&Default::default());
            }
        }
        self.layernorm(gpu, &mut enc, ctx.h, &self.ln_post, ctx.normed, geom.s);
        self.gemm(gpu, &mut enc, ctx.normed, &self.proj1.w, &gu, geom.s, self.proj1.n_pad, self.proj1.k, self.proj1.n_pad, None);
        self.bias_gelu_tensor(
            gpu, &mut enc, &gu, &self.proj1.bias, &gact, &self.u_sc[4],
            s_pad * self.proj1.n_pad, self.proj1.n_pad / 2, false,
        );
        self.gemm(
            gpu, &mut enc, &gact, &self.proj2.w, &out_emb, geom.s,
            self.proj2.n_pad, self.proj2.k, self.proj2.n_pad, Some(&self.proj2.bias),
        );

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

        // 1. self_attn_layer_norm
        self.layernorm(gpu, enc, ctx.h, &l.sln, ctx.normed, s);
        // 2. fused QKV
        self.gemm(
            gpu, enc, ctx.normed, &l.qkv.w, ctx.qkv, s, l.qkv.n_pad, l.qkv.k, l.qkv.n_pad,
            Some(&l.qkv.bias),
        );
        // 3. split Q/K/V.  No PE here: the reference adds the positional
        //    embedding to the *conv_out output*, which `add_pe` already did.
        {
            // The uniform was created, bound and never written: every field read
            // as 0, `n_tokens == 0` made the kernel return on its first guard, and
            // Q/K/V stayed the allocator's zeros.  Scores, softmax and AV then
            // compared equal to a host oracle that recomputed from those same
            // zeros — a self-consistent block with no signal in it at all.
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
            // (token, head, head-word) — `s_pad·nh` would overflow the 65535
            // per-dimension grid limit for audio longer than ~6 minutes.
            cp.dispatch_workgroups(ctx.s_pad as u32, nh as u32, (hd / 2) as u32);
        }
        // 4. re-pack into the per-(head, window) blocks the GEMMs can read
        //    with their fixed operand row strides.
        let (wpad, hd_pad) = (align(wlen, GEMM_BM), align(hd, GEMM_BN));
        let n_blocks = nh * geom.n_win;
        {
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
        {
            let bsa = wpad * hd / 2;
            let bsb = hd * wpad / 2;
            // `bsc` is in ELEMENTS: the epilogue divides the flat C index by 2.
            // Passing words here (as this did) put every batch block but the
            // first half a block early, so adjacent blocks overwrote each other
            // in whatever order the workgroups happened to run.
            let bsc = wpad * wpad;
            self.gemm_batched(
                gpu, enc, &self.p.gemm_t, ctx.qp, ctx.kt, ctx.scores,
                wpad, wpad, hd, wpad, bsa, bsb, bsc, n_blocks, 1, 1, 0,
            );
        }
        // 6. windowed softmax (masks the short last window), scores -> probs
        {
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
        {
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
        {
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
        self.gemm_beta(gpu, enc, ctx.attn_flat, &l.o.w, ctx.h, s, l.o.n_pad, l.o.k, dm, Some(&l.o.bias));
        if self.mid_capture.get() && li == 0 {
            let cb = std::mem::replace(enc, gpu.device.create_command_encoder(&Default::default()));
            gpu.queue.submit([cb.finish()]);
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| anyhow::anyhow!("audio encoder: device lost at mid capture: {e:?}"))?;
            *self.mid_h.borrow_mut() = Some(read_f16_buf(gpu, ctx.h, ctx.s_pad * dm)?);
        }
        // 9. final_layer_norm -> FFN (fc1, GELU, fc2) + residual
        self.layernorm(gpu, enc, ctx.h, &l.fln, ctx.norm2, s);
        self.gemm(gpu, enc, ctx.norm2, &l.fc1.w, ctx.gu, s, l.fc1.n_pad, l.fc1.k, l.fc1.n_pad, None);
        self.bias_gelu_tensor(
            gpu, enc, ctx.gu, &l.fc1.bias, ctx.act, &self.u_sc[3],
            ctx.s_pad * l.fc1.n_pad, l.fc1.n_pad / 2, false,
        );
        self.gemm_beta(gpu, enc, ctx.act, &l.fc2.w, ctx.h, s, l.fc2.n_pad, l.fc2.k, dm, Some(&l.fc2.bias));
        Ok(())
    }

    // ── dispatch helpers ──

    /// `C += A·Wᵀ` — the residual form of [`Self::gemm`].  The transformer's
    /// two residual adds must *accumulate* onto the residual buffer; running
    /// them as plain GEMMs (as this did) silently dropped every residual.
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

    /// Single-tile GEMM at `prefill_gemm`'s 64×64 tile edges: `a` is `[m, k]`,
    /// `w` is `[n, k]` row-major, `c` is `[m, ldc]` — all packed f16, with `m`
    /// rounded up to 64 (the A operand's allocation) and `n` a multiple of 64
    /// (the padded weight row count).
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

    /// The conv GEMM for one level.
    ///
    /// `A` is the **weight** `[m_pad][k_pad]` and `B` the im2col operand, read
    /// with `transb = 1` (i.e. as `[k_pad][n_all]`, row stride `n_all/2` words).
    /// That orientation is what puts the *channel* on the GEMM's `m` and the
    /// *position* on its `n`, so `C` comes out channel-major — the layout the
    /// reference uses, the permute expects and the next level's gather reads.
    ///
    /// `A`'s row stride is `k/2` words, which is exactly the weight's own
    /// layout, and `m_pad` is the weight's padded row count; `n_all` and `ldc`
    /// are the same number, so a row of `C` is one channel's `n_all` positions.
    fn gemm_conv(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, level: usize) {
        let l = &self.conv[level];
        self.dispatch_gemm(
            gpu, enc, &self.p.gemm_t, &l.w.w, &self.col[level], &self.raw[level],
            l.m_pad, l.n_all, l.k_pad, l.n_all, 0, 0, 0, 1,
            (l.n_all / GEMM_BN) as u32, (l.m_pad / GEMM_BM) as u32, None,
        );
    }

    /// The gather for one level: `input` is this level's input image and `cin0`
    /// the **global** index of the round's first chunk in it (0 for the levels
    /// that read the round-local activation, `chunk0` for the mel).
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

    /// One `prefill_gemm` dispatch per `(head, window)` block.
    ///
    /// `bsa`/`bsb` are in **words**, `bsc` in **elements** — see
    /// [`crate::shaders::prefill_gemm`].  With `batch = 1` neither unit matters
    /// (every base is 0), which is exactly how a wrong `bsc` survived here.
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

    /// `dst[i] = gelu(src[i] + bias[c(i)])` over `n` f16 elements, two per
    /// thread.
    ///
    /// `bias` is the *per-channel* vector (padded to the activation's channel
    /// count with zeros), not a broadcast image: the old code materialised a
    /// `[rows][n_pad]` image per layer, which at `MAX_TOKENS` rows was 117 MB
    /// **per FFN layer**.  The channel index comes from the tensor's own layout:
    /// `words` is the number of words per channel or per row, and `by_channel`
    /// picks `i / words` (channel-major conv activation, channel stride
    /// `n_all`) or `i % words` (token-major GEMM output, row stride `n_pad`).
    ///
    /// `n` is the *element* count; it must be even so a thread's two halves
    /// never straddle the real/padding boundary of the channel vector.
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
        // 256 threads × 2 halves = 512 elements per workgroup (the old divisor
        // 600 covered only 85 % of the tensor and left the tail un-GELU'd).
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
        cp.dispatch_workgroups(align(rows, GEMM_BM) as u32, 1, 1);
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
/// uniform with a dynamic offset.  wgpu's implicit layouts are
/// pipeline-exclusive and never enable dynamic offsets, and every GEMM dispatch
/// here reads its own `GDims` slot out of one ring buffer.
///
/// `read_write` names the bindings the shader declares `read_write` (only the
/// GEMM's `C`); a read-only buffer can bind where the shader is read-only, but
/// not the other way round.
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

/// Read `n` packed f16 elements back as a vector.
fn read_f16_buf(gpu: &Gpu, buf: &wgpu::Buffer, n: usize) -> Result<Vec<f16>> {
    let bytes = gpu.readback(buf, (n * 2) as u64)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// One `conv2d(3×3, stride 2, pad 1)` output length.  The reference's/// `feo`-style helper folds this three times — but note `feo` is *not* the same
/// as three `conv_out_len` steps for every input (e.g. `feo(100) = 13` while
/// `conv_out_len³(100) = 25`), so the tower uses the single-step form per layer
/// and only ever calls `feo` where the reference does (the tail-token count).
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

/// Elements of one mel "plane": all mel bins of one chunk, padded to `w0`.
/// The staging buffer is sized the same way the encoder is: `cs`-frame chunks.
#[inline]
fn mel_elems(n_mels: usize, w0: usize) -> usize {
    n_mels * w0
}

/// Zero-pad a `[rows, cols]` f16 matrix's columns to the next even count.  The
/// GEMM reads packed f16 pairs along k, so an odd k shears every row by one
/// half-word; the pad column is zero and the operand is zero there too.
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
        data,
        rows: w.rows,
        cols: kw,
    })
}

/// Zero-pad a `[rows, cols]` f16 matrix to `new_rows` — the n-tile width of
/// `prefill_gemm`.  The GEMM only ever reads the rows it is told to, but a
/// padded W keeps every dispatch on a whole tile and every stride exact.
fn pad_rows(w: &PackedWeight, new_rows: usize) -> Result<PackedWeight> {
    anyhow::ensure!(new_rows >= w.rows, "pad_rows: shrink not supported");
    if new_rows == w.rows {
        return Ok(PackedWeight {
            data: w.data.clone(),
            rows: w.rows,
            cols: w.cols,
        });
    }
    let mut data = w.data.clone();
    data.resize(new_rows * w.cols * 2, 0);
    Ok(PackedWeight {
        data,
        rows: new_rows,
        cols: w.cols,
    })
}

/// Tap-offset table for a `3×3/s2/p1` conv over an `h × w_in` plane.
///
/// `off[pos*9 + tap]` is the source offset of that tap *inside one input
/// plane*, `TAP_OOB` outside it.  Positions are `(h_out, w_out)` row-major, the
/// same order the GEMM's `n` axis uses, so `tap` doubles as a position index for
/// the next level's gather.
///
/// `w_in` is the true width and `w_stride` the buffer's row stride (`w0` for the
/// mel, else `= w_in`).  There is **one** sentinel: the old second one
/// (`TAP_WRAP`, for the packed-pair read that no longer exists) was compared
/// against a different constant in the shader than the table held, which made
/// every out-of-bounds tap a valid address.
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

/// A linear's bias as a standalone vector padded to `n_pad` with zeros, so
/// both the GEMM's `bias = 1` epilogue and `bias_gelu` can index it without a
/// bound check (an activation's padding channels/columns are zero and must stay
/// zero).
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

/// Round `k` up to a multiple of the GEMM's k-tile (`PREFILL_GEMM_BK`).
///
/// Two constraints, and the second is the one that bites:
///
/// * the A operand is indexed as `array<u32>` with a word row stride of `k/2`,
///   so `k` must be even (a multiple of 4 keeps every pair word-aligned);
/// * the k loop runs **whole 16-wide tiles** (`k0 += BK` while `k0 < gd.k`),
///   so the last tile covers `align(k, 16)` values.  With `k` merely padded to
///   4, those extra reads walk past the operand's row into the next one and
///   their products are **added** to every output.  The padding is zero on both
///   sides only when the operand row really is that wide, i.e. when `k` is a
///   multiple of `BK`.
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

























