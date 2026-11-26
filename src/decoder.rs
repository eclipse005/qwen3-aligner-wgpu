//! wgpu text decoder.
//!
//! Forced alignment has no generation loop, so this module is one causal pass
//! over the prompt (see [`WgpuTextDecoder::prefill`]) rather than a decode step:
//!
//! ```text
//! for each of 28 layers:
//!   rms_norm(h, input_layernorm)          -> normed
//!   gemm(normed, qkv_w)                   -> qkv            (4096)
//!   qkv_extract(qkv)                      -> q_out + shared K/V slots
//!   causal attention reads K/V by GQA head (slabbed once cur > SLAB_T)
//!                                         -> attn           (2048)
//!   gemm_acc(attn, o_w)                   -> h              (residual fused)
//!   rms_norm(h, post_attention_layernorm) -> norm2
//!   gemm(norm2, gate_up_w)                -> gate_up        (6144)
//!   silu_mul_split(gate_up)               -> activated      (3072)
//!   gemm_acc(activated, down_proj_w)      -> h
//! ```
//!
//! The whole pass is one command buffer and a handful of submits (batched every
//! few layers so a long prompt does not trip the driver's watchdog).  The last
//! layer's hidden states are what the caller reads back; the KV cache is sized
//! on demand ([`WgpuTextDecoder::ensure_capacity`]) rather than for the ceiling.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};

use crate::gpu::{BulkUpload, Gpu, PipeJob};
use crate::half16::H16;
use crate::shaders;
use crate::weights::{self, PackedWeight};

/// Text decoder hyper-parameters.
#[derive(Debug, Clone)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
}

impl TextConfig {
    /// Read text decoder dims from either the `thinker_config` layout
    /// or a Transformers-native `-hf` `config.json`.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let cfg = crate::config::AsrConfig::from_file(&dir.join("config.json"))?;
        let t = &cfg.thinker_config.text_config;
        Ok(Self {
            vocab_size: t.vocab_size,
            hidden_size: t.hidden_size,
            intermediate_size: t.intermediate_size,
            num_hidden_layers: t.num_hidden_layers,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            rms_norm_eps: t.rms_norm_eps as f32,
        })
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn fused_qkv_cols(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }
    fn scale(&self) -> f32 {
        1.0f32 / (self.head_dim as f32).sqrt()
    }
}

/// Reduction block size for a `last`-wide row: the smallest power of two ≥ `last`,
/// clamped to `[32, 1024]`.
fn block_for_reduction(last: usize) -> u32 {
    let mut bs: u32 = 32;
    let target = last as u32;
    while bs < target && bs < 1024 {
        bs *= 2;
    }
    bs.min(1024).max(32)
}

/// Reduction block width for the flat path's `softmax_causal`.
///
/// Deliberately not `block_for_reduction`: that helper sizes an `rms_norm` tree,
/// where a wide block is nearly free because each thread does one multiply-add
/// per element.  A causal softmax row is `cur` wide — 3712 at s=3651 — so a
/// 1024-wide block gives each thread three elements and then charges it two
/// ten-level trees, twenty barriers in all.  This card keeps two such blocks per
/// SM, so when one is at a barrier there is nothing else on that SM to run; with
/// 256-wide blocks there are eight, and their barriers interleave.
///
/// The width is not just a performance knob: it sets the *order* both reductions
/// sum in, so the f32 result differs in the last ulp.  The softmax writes 16-bit
/// values, which is what should absorb that — the gate decides, not this comment.
///
/// `SLAB_BS` above already picked 256 for the same reason on the slabbed path.
///
/// `QALIGN_SOFT_BS=<n>` overrides the choice, so two widths can be compared on one
/// binary (all six widths are precompiled).
///
/// **256, not 128, although 128 measures faster.**  Interleaved A/B on one binary,
/// three rounds each, prefill medians: 15s_en 181.5 -> 179.1 ms (-1.3%, every 128
/// run below every 256 run), 90s_ja 1030.6 -> 1020.1 (-1.0%), 180s_zh
/// 3360.9 -> 3329.0 (-0.9%).  That is a real ~1% and it is *rejected*, because it
/// costs alignment: against the f16 gold the raw stream goes 3620 exact / 2
/// excused to 3619 / 3, and items 3619 / 3 to 3617 / 5.  No new hard failure
/// appears -- every moved endpoint is one the reference itself answers differently
/// per dtype -- but the diff against the reference grows by two endpoints, and the
/// standing constraint on this work is that alignment does not move.  A per-row
/// tie-break the reference cannot decide is not ours to spend.
///
/// This is pinned by the gate rather than by a unit test: the numbers above are
/// what `scripts/bench.ps1` reports against the f16 gold, so a change that moves
/// the excused count is visible in every run.
fn softmax_bs(cur: usize) -> u32 {
    if let Ok(s) = std::env::var("QALIGN_SOFT_BS") {
        if let Ok(n) = s.parse::<u32>() {
            return n.clamp(32, 1024);
        }
    }
    block_for_reduction(cur).min(SOFTMAX_BS_CAP)
}

/// The flat softmax's block-width cap.  See [`softmax_bs`] for why 256 and not the
/// faster 128.
const SOFTMAX_BS_CAP: u32 = 256;

/// Reduction block width for `rms_norm`, by the same argument as [`softmax_bs`].
///
/// `rms_norm` is dispatched twice per layer over `s` rows of `hs = 1024`, so a
/// 1024-wide block gives each thread **one** element and then charges it a
/// ten-level tree.  In barrier terms that is worse than the softmax was: 56
/// dispatches x `s` rows is 257k workgroups at s=4589 against the softmax's 73k,
/// and every one of them pays its own tree.  The whole `rms_norm` share sits in
/// the 183.8 ms that `QALIGN_SKIP` leaves unattributed on 180s_en, which is what
/// bounds the prize at roughly 100 ms.
///
/// Same caveat as [`softmax_bs`]: the width sets the summation order, so the
/// variance moves in the last ulp and the 16-bit write is what absorbs it.  The
/// gate decides.
///
/// `QALIGN_RMS_BS=<n>` overrides, for A/B on one binary.
fn rms_bs(last: usize) -> u32 {
    if let Ok(s) = std::env::var("QALIGN_RMS_BS") {
        if let Ok(n) = s.parse::<u32>() {
            return n.clamp(32, 1024);
        }
    }
    block_for_reduction(last).min(256)
}

/// Split a flat workgroup count across two grid axes: `max_compute_workgroups_
/// per_dimension` caps at 65535, and wgpu rejects the whole command buffer rather
/// than clamping.
fn write_silu_rows(gpu: &Gpu, cfg: &TextConfig, u_silu: &wgpu::Buffer, rows: usize) -> (u32, u32) {
    let words = rows * cfg.intermediate_size / 2;
    let workgroups = words.div_ceil(256);
    let (gx, gy) = grid_xy(workgroups);
    gpu.upload(
        u_silu,
        bytemuck::bytes_of(&SiluCfg {
            inter2: (cfg.intermediate_size / 2) as u32,
            total2: words as u32,
            gx,
            _b: 0,
        }),
    );
    (gx, gy)
}

/// `QALIGN_TILE64=auto|on|off` override shared by both GPU towers.
/// `None` leaves the caller's automatic threshold in place.
pub(crate) fn tile64_mode() -> Option<bool> {
    match std::env::var("QALIGN_TILE64")
        .ok()
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("on" | "1" | "yes" | "true") => Some(true),
        Some("off" | "0" | "no" | "false") => Some(false),
        _ => None,
    }
}

pub(crate) fn grid_xy(workgroups: usize) -> (u32, u32) {
    let gx = workgroups.clamp(1, 65_535) as u32;
    let gy = workgroups.div_ceil(gx as usize).max(1) as u32;
    (gx, gy)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsCfg {
    eps: f32,
    _a: f32,
    _b: f32,
    _c: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QkvxCfg {
    max_seq: u32,
    start: u32,
    pos_offset: u32,
    /// positions in this dispatch (1 for decode, s for prefill)
    s: u32,
    eps: f32,
    _a: u32,
    _b: u32,
    _c: u32,
}

/// Prefill GEMM dimensions.
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
    /// Row offset into the B operand — the key-tile start for the slabbed
    /// attention (`transb=0`: K rows, `transb=1`: V rows).  Zero everywhere else.
    row0: u32,
    /// A row stride in elements — `k` everywhere except the slabbed AV, whose A
    /// operand (a score slab) is `SLAB_T` wide while its k sweep is narrower.
    lda: u32,
    /// Q heads per KV head: the attention GEMMs index the un-repeated K/V
    /// buffer by `wid.z / gqa_group`; 1 for every other GEMM.
    gqa_group: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluCfg {
    inter2: u32,
    total2: u32,
    /// x-axis grid size; `y` continues the flat index space beyond it.
    gx: u32,
    _b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxCfg {
    n_w: u32,
    n_x: u32,
    valid: u32,
    m: u32,
    mp: u32,
    scale: f32,
    /// x-axis grid size; `y` continues the row index beyond it.
    gx: u32,
    /// Column offset of the score block this dispatch covers: 0 on the flat path
    /// (the whole row), the slab's first key otherwise.  The row index stays
    /// absolute, so the causal bound is `pos + 1 − row0`.
    row0: u32,
}

/// Per-slab row statistics (`shaders::slab_stats`): one `(max, Σexp)` pair per
/// row per key slab.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SlabStatsCfg {
    /// score-row stride in words (slab width / 2)
    n_x: u32,
    /// columns of this slab that exist at all (≤ slab width, 16-aligned)
    valid: u32,
    /// rows in the head = `s`
    m: u32,
    /// padded rows per head = `mp`
    mp: u32,
    scale: f32,
    /// first key column of this slab
    row0: u32,
    /// x-axis grid size; `y` continues the row index beyond it
    gx: u32,
    /// total rows in the stats buffer (`nqh · mp`)
    rows: u32,
}

/// Merge weights (`shaders::slab_weights`): layer-independent, one slot.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SlabWeightsCfg {
    rows: u32,
    n_slab: u32,
    gx: u32,
    _p: u32,
}

/// Slab merge (`shaders::slab_merge`): layer-independent, one slot.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SlabMergeCfg {
    /// output rows (`mp`, padded positions)
    rows: u32,
    n_slab: u32,
    gx: u32,
    _p: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GqaCfg {
    cur_len: u32,
    max_seq: u32,
    scale: f32,
    _p: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitCfg {
    cur_len: u32,
    max_seq: u32,
    scale: f32,
    n_chunks: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MergeCfg {
    n_chunks: u32,
    _a: u32,
    _b: u32,
    _c: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxCfg {
    n: u32,
    slot: u32,
    _a: u32,
    _b: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EmbedCfg {
    slot: u32,
    d2: u32,
    _a: u32,
    _b: u32,
}

/// Buffers and uniforms reused across the prefill pass.
pub struct Scratch {
    /// RoPE tables, `[rope_positions, head_dim]`, in the run's storage format.
    pub cos: wgpu::Buffer,
    pub sin: wgpu::Buffer,
    u_rms: wgpu::Buffer,
    u_qkvx: wgpu::Buffer,
    u_silu: wgpu::Buffer,
}

/// One decoder layer's weights.  KV scratch is shared by all layers: forced
/// alignment runs one causal pass, and each layer consumes the K/V it writes
/// before the next one overwrites them.
struct Layer {
    iln_w: wgpu::Buffer,
    pln_w: wgpu::Buffer,
    qn_w: wgpu::Buffer,
    kn_w: wgpu::Buffer,
    qkv_w: wgpu::Buffer,
    o_w: wgpu::Buffer,
    gu_w: wgpu::Buffer,
    dp_w: wgpu::Buffer,
}

impl Pipes {
    /// `gemm`, `gemm_acc`, `gemm_av`, `gemm_causal`, `gemm_av_causal` by index;
    /// `small` selects the 64×64 sibling.
    fn pipe(&self, small: bool, i: usize) -> &wgpu::ComputePipeline {
        match (small, i) {
            (false, 0) => &self.gemm,
            (false, 1) => &self.gemm_acc,
            (false, 2) => &self.gemm_av,
            (false, 3) => &self.gemm_causal,
            (false, _) => &self.gemm_av_causal,
            (true, 0) => &self.gemm64,
            (true, 1) => &self.gemm64_acc,
            (true, 2) => &self.gemm64_av,
            (true, 3) => &self.gemm64_causal,
            (true, _) => &self.gemm64_av_causal,
        }
    }
}

struct Pipes {
    rms_norm: wgpu::ComputePipeline,
    extract: wgpu::ComputePipeline,
    silu: wgpu::ComputePipeline,
    /// Prefill GEMMs: plain and residual-accumulating (transb=0), plus the
    /// attention AV form (transb=1, batched).
    gemm: wgpu::ComputePipeline,
    gemm_acc: wgpu::ComputePipeline,
    gemm_av: wgpu::ComputePipeline,
    /// Plain GEMM with the causal score-tile skip (scores only, never read above
    /// the diagonal — see `shaders::prefill_gemm_causal`).
    gemm_causal: wgpu::ComputePipeline,
    /// AV GEMM with the causal k bound (skips the softmax's exact zeros).
    gemm_av_causal: wgpu::ComputePipeline,
    /// The 64×64 siblings of the five GEMMs above, for short shapes.
    gemm64: wgpu::ComputePipeline,
    gemm64_acc: wgpu::ComputePipeline,
    gemm64_av: wgpu::ComputePipeline,
    gemm64_causal: wgpu::ComputePipeline,
    gemm64_av_causal: wgpu::ComputePipeline,
    /// Causal softmax, one pipeline per block size (reduction tree depends on it).
    softmax: std::collections::HashMap<usize, wgpu::ComputePipeline>,
    /// Slabbed causal attention: per-slab `(max, Σexp)`, the per-row merge
    /// weights, and the weighted merge of the per-slab AV outputs.
    slab_stats: wgpu::ComputePipeline,
    slab_weights: wgpu::ComputePipeline,
    slab_merge: wgpu::ComputePipeline,
    /// Measurement only: the f16/bf16 -> f32 widening copy behind
    /// `QALIGN_DUMP_LAYERS` (see [`WgpuTextDecoder::prefill`]).  Built always —
    /// it costs one pipeline — but dispatched only when the env var is set.
    widen: wgpu::ComputePipeline,
}

/// KV slots allocated at load (112 KiB each, see [`WgpuTextDecoder::load`]).
///
/// `max_seq` is the ceiling, not the allocation: the cache grows from here on
/// demand ([`WgpuTextDecoder::ensure_capacity`]), so a 15-second clip does not
/// pay for the 8 192 slots a ten-minute one might need.
pub const KV_INITIAL_CAP: usize = 1024;

/// Capacity grows in multiples of this — a series of clips of one shape then
/// allocates once and pays nothing after that.
pub(crate) const KV_STEP: usize = 256;

/// Key-slab width for the tiled causal prefill attention: the key dimension is
/// processed in windows of this many positions rather than materialised whole.
const SLAB_T: usize = 1024;

/// Reduction block size for the slabbed softmax and its statistics.
///
/// Both run one workgroup per score row over a `SLAB_T`-wide slab and must agree
/// on the row sum to the bit (the merge weights are that sum), so the block size
/// is shared.  It is smaller than `SLAB_T` so fewer threads contend on the
/// barrier trees.
const SLAB_BS: usize = 256;

/// One 256 B slot per GEMM dispatch (dynamic-offset alignment).
const MAX_GEMMS: usize = 256;

/// Uniform slots reserved for the per-slab cfg blocks (softmax + stats) — the
/// most key slabs a prefill may tile into.  Only the slabbed path uses them.
const MAX_SLAB: usize = 16;

/// Whether prefill tiles its attention over the key axis, or materialises the
/// whole score matrix.  `QASR_SLAB=on|off` forces one path.
///
/// The flat path's scratch is `[nqh, mp, cur16]` f32 — O(s²) — and it is also
/// the faster of the two while it fits: on `180s_zh` (s = 4589) forcing the flat
/// path was **4.3% faster** (paired A/B over 3 rounds: −3.8 / −4.4 / −4.0%, both
/// arms passing the gate).
///
/// The gate used to be a flat `s > 4096`, which is far below what the hardware
/// allows — at s = 4589 the scratch is 679 MB against a 2047 MiB binding limit.
/// So it now tests the constraint that actually exists, and it is a *device*
/// property rather than a constant: `scores` and `attn` are two separate
/// buffers of the same size, so the working figure is twice the scratch against
/// the binding limit.  An adapter with a smaller limit (the integrated Intel
/// Vulkan device reports 1023 MiB) therefore tiles sooner than this card does,
/// and the caller's `ensure!` still states the per-buffer bound in full.
fn slab_path(s: usize, nqh: usize, bind_limit: u64) -> bool {
    match std::env::var("QASR_SLAB").unwrap_or_default().to_ascii_lowercase().as_str() {
        "1" | "on" | "yes" | "force" => true,
        "0" | "off" | "no" => false,
        _ => {
            let mp = s.div_ceil(128) * 128;
            let cur16 = s.div_ceil(16) * 16;
            let scratch = (nqh * mp * cur16 * 2) as u64; // f32, one buffer
            scratch.saturating_mul(2) > bind_limit
        }
    }
}

pub struct WgpuTextDecoder {
    pub gpu: Gpu,
    pub cfg: TextConfig,
    /// The ceiling on the sequence length — fixed at load.
    pub max_seq: usize,
    /// KV slots actually allocated: [`KV_INITIAL_CAP`] after load, grown on
    /// demand up to `max_seq`.  Every uniform that carries a KV stride uses
    /// this, not the ceiling.
    pub cap: usize,

    layers: Vec<Layer>,
    pipes: Pipes,
    pub scratch: Scratch,

    /// The last layer's hidden states, `[s, hs]` in the run's storage format —
    /// what the caller reads back after [`Self::prefill`].
    pub debug_prefill_h: Option<wgpu::Buffer>,

    /// One K and one V scratch buffer, shared by all layers.
    k_buf: wgpu::Buffer,
    v_buf: wgpu::Buffer,
    /// Rows per KV head actually allocated in `k_buf` / `v_buf` — the stride
    /// every KV-addressing uniform uses, including the slabbed tail.
    kv_cap: usize,

    /// Grow-only per-call prefill scratch, including the fixed uniform buffers.
    prefill_scratch: Option<PrefillScratch>,
}

impl WgpuTextDecoder {
    /// Build the decoder and upload `{prefix}.*` weights.
    ///
    /// `max_seq` is the ceiling on the sequence length; the KV cache starts at
    /// [`KV_INITIAL_CAP`] slots and grows up to it on demand.  `rope_positions`
    /// sizes the MRoPE tables.
    pub fn load(
        gpu: Gpu,
        model_dir: &Path,
        prefix: &str,
        cfg: TextConfig,
        max_seq: usize,
        rope_positions: usize,
    ) -> Result<Self> {
        let w = weights::load_tensors(model_dir)?;
        let hs = cfg.hidden_size;
        let nqh = cfg.num_attention_heads;
        let nkvh = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let nl = cfg.num_hidden_layers;

        let build =
            |label: &str, src: &str, entry: &str, pl: Option<&wgpu::PipelineLayout>| -> Result<wgpu::ComputePipeline> {
                gpu.pipeline(label, src, entry, pl)
                    .with_context(|| format!("build pipeline {label}"))
            };

        let gemm_pl = family_layout_dyn(
            &gpu,
            "prefill_gemm",
            &[(0, true), (1, true), (2, false)],
            3,
            true,
        );
        // dynamic: the slabbed path gives every key slab its own cfg slot
        let sm_pl = family_layout_dyn(&gpu, "softmax", &[(0, true), (1, false)], 2, true);
        // measurement dump: two storages + a per-dispatch uniform slot
        let wd_pl = family_layout_dyn(&gpu, "widen_dump", &[(0, true), (1, false)], 2, true);

        // ── pipelines ─────────────────────────────────────────────────────
        let rms_bs = rms_bs(hs) as usize;
        // 三个 slab 布局提出来：作业表要活得比生成它的表达式久。
        let slab_stats_pl =
            family_layout_dyn(&gpu, "slab_stats", &[(0, true), (1, false)], 2, true);
        let slab_weights_pl = family_layout(&gpu, "slab_weights", &[(0, true), (1, false)], 2);
        let slab_merge_pl =
            family_layout(&gpu, "slab_merge", &[(0, true), (1, true), (2, false)], 3);
        let use_vec4 = gpu.use_vec4_gemm();
        let use_f16 = gpu.use_f16_kernel();
        let jobs = [
            PipeJob::new("rms_norm", shaders::rms_norm(hs, rms_bs), "rms_norm", None),
            PipeJob::new("qkv_extract", shaders::qkv_extract(nqh, nkvh, hd), "qkv_extract", None),
            PipeJob::new("silu30", shaders::silu_mul_split(inter), "silu_mul_split", None),
            PipeJob::new("gemm", shaders::prefill_gemm_cfg(8, 8, false, false, false, false, false, use_vec4, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_acc", shaders::prefill_gemm_cfg(8, 8, false, true, false, false, false, use_vec4, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_av", shaders::prefill_gemm_cfg(8, 8, true, false, false, false, false, use_vec4, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_causal", shaders::prefill_gemm_cfg(8, 8, false, false, false, true, false, use_vec4, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm_av_causal", shaders::prefill_gemm_cfg(8, 8, true, false, false, false, true, use_vec4, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm64", shaders::prefill_gemm_cfg(4, 4, false, false, false, false, false, true, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm64_acc", shaders::prefill_gemm_cfg(4, 4, false, true, false, false, false, true, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm64_av", shaders::prefill_gemm_cfg(4, 4, true, false, false, false, false, true, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm64_causal", shaders::prefill_gemm_cfg(4, 4, false, false, false, true, false, true, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("gemm64_av_causal", shaders::prefill_gemm_cfg(4, 4, true, false, false, false, true, true, use_f16), "gemm", Some(&gemm_pl)),
            PipeJob::new("slab_stats", shaders::slab_stats(SLAB_BS, SLAB_T), "slab_stats", Some(&slab_stats_pl)),
            PipeJob::new("slab_weights", shaders::slab_weights(256), "slab_weights", Some(&slab_weights_pl)),
            PipeJob::new("slab_merge", shaders::slab_merge(nqh, hd), "slab_merge", Some(&slab_merge_pl)),
            PipeJob::new("widen_dump", shaders::widen_h16_f32(), "widen", Some(&wd_pl)),
        ];
        let mut built = gpu.pipelines(&jobs)?;
        let mut take = |name: &str| -> Result<wgpu::ComputePipeline> {
            built
                .remove(name)
                .with_context(|| format!("pipeline {name} was not built"))
        };
        let pipes = Pipes {
            rms_norm: take("rms_norm")?,
            extract: take("qkv_extract")?,
            silu: take("silu30")?,
            gemm: take("gemm")?,
            gemm_acc: take("gemm_acc")?,
            gemm_av: take("gemm_av")?,
            gemm_causal: take("gemm_causal")?,
            gemm_av_causal: take("gemm_av_causal")?,
            gemm64: take("gemm64")?,
            gemm64_acc: take("gemm64_acc")?,
            gemm64_av: take("gemm64_av")?,
            gemm64_causal: take("gemm64_causal")?,
            gemm64_av_causal: take("gemm64_av_causal")?,
            softmax: std::collections::HashMap::from([ (32, build("softmax32", &shaders::softmax_causal_mode(32, gpu.subgroup_mode()), "softmax", Some(&sm_pl))?), (64, build("softmax64", &shaders::softmax_causal_mode(64, gpu.subgroup_mode()), "softmax", Some(&sm_pl))?), (128, build("softmax128", &shaders::softmax_causal_mode(128, gpu.subgroup_mode()), "softmax", Some(&sm_pl))?), (256, build("softmax256", &shaders::softmax_causal_mode(256, gpu.subgroup_mode()), "softmax", Some(&sm_pl))?), (512, build("softmax512", &shaders::softmax_causal_mode(512, gpu.subgroup_mode()), "softmax", Some(&sm_pl))?), (1024, build("softmax1024", &shaders::softmax_causal_mode(1024, gpu.subgroup_mode()), "softmax", Some(&sm_pl))?),  ]),
            slab_stats: take("slab_stats")?,
            slab_weights: take("slab_weights")?,
            slab_merge: take("slab_merge")?,
            widen: take("widen_dump")?,
        };

        // ── scratch ───────────────────────────────────────────────────────
        let mut up = gpu.uploader();
        let scratch = Scratch {
            cos: up.storage("cos", (rope_positions * hd / 2 * 4) as u64),
            sin: up.storage("sin", (rope_positions * hd / 2 * 4) as u64),
            u_rms: up.uniform("u_rms", 16),
            u_qkvx: up.uniform("u_qkvx", 32),
            u_silu: up.uniform("u_silu", 16),
        };
        up.upload(
            &scratch.u_rms,
            bytemuck::bytes_of(&RmsCfg { eps: cfg.rms_norm_eps, _a: 0.0, _b: 0.0, _c: 0.0 }),
        )?;

        // ── per-layer ─────────────────────────────────────────────────────
        let cap = KV_INITIAL_CAP.min(max_seq);
        let kv_words = nkvh * cap * hd / 2;

        let mut layers = Vec::with_capacity(nl);
        let mut t_conv = std::time::Duration::ZERO;
        // Conversions run one layer ahead on another thread, so the narrowing of
        // layer i+1 happens while layer i crosses the bus — the two are the only
        // things this load does, and only one of them is on the critical path.
        // The channel bounds the lookahead to a couple of layers' host memory.
        std::thread::scope(|scope| -> Result<()> {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<LayerWeights>>(PREFETCH_LAYERS);
            scope.spawn(move || {
                for i in 0..nl {
                    let p = format!("{prefix}.layers.{i}");
                    if tx.send(convert_layer(&w, &p)).is_err() {
                        break; // consumer went away
                    }
                }
            });
            for i in 0..nl {
                let t = std::time::Instant::now();
                let lw = rx
                    .recv()
                    .map_err(|_| anyhow!("weight conversion thread died at layer {i}"))??;
                t_conv += t.elapsed();
                let LayerWeights {
                    qkv: qkv_p,
                    o: o_w,
                    gu: gu_p,
                    dp: dp_w,
                    iln: iln_v,
                    pln: pln_v,
                    qn: qn_v,
                    kn: kn_v,
                } = lw;

                let qkv = up.upload_pieces("qkv_w", &qkv_p)?;
                let o = upload_weight(&mut up, "o_w", &o_w)?;
                let gu = up.upload_pieces("gu_w", &gu_p)?;
                let dp = upload_weight(&mut up, "dp_w", &dp_w)?;
                let iln = upload_vec(&mut up, "iln_w", &iln_v)?;
                let pln = upload_vec(&mut up, "pln_w", &pln_v)?;
                let qn = upload_vec(&mut up, "qn_w", &qn_v)?;
                let kn = upload_vec(&mut up, "kn_w", &kn_v)?;

                layers.push(Layer {
                    iln_w: iln,
                    pln_w: pln,
                    qn_w: qn,
                    kn_w: kn,
                    qkv_w: qkv,
                    o_w: o,
                    gu_w: gu,
                    dp_w: dp,
                });
            }
            Ok(())
        })?;
        crate::load_trace::note_dur("decoder: convert wait (prefetched)", t_conv);

        up.finish()?;

        let k_buf = gpu.storage("k_cache", (kv_words * 4) as u64);
        let v_buf = gpu.storage("v_cache", (kv_words * 4) as u64);

        Ok(Self {
            gpu,
            cfg,
            max_seq,
            cap,
            kv_cap: cap,
            k_buf,
            v_buf,
            layers,
            pipes,
            scratch,
            prefill_scratch: None,
            debug_prefill_h: None,
        })
    }

    // ── KV cache / RoPE plumbing ─────────────────────────────────────────

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// Grow the KV cache so that it holds `need` positions.
    ///
    /// Grow-only and stepped ([`KV_STEP`]): a run of clips of one shape allocates
    /// once, and a request that already fits costs nothing but the comparison.
    /// `need` past the ceiling is clamped here — the caller's own check against
    /// [`Self::max_seq`] is what reports that.
    ///
    /// Swapping the two buffers is the whole job: [`Self::prefill`] builds its
    /// bind groups per call, so nothing else holds a view of the old ones.
    pub fn ensure_capacity(&mut self, need: usize) -> usize {
        let target = (need.div_ceil(KV_STEP) * KV_STEP).min(self.max_seq).max(self.cap);
        if target == self.cap {
            return self.cap;
        }
        self.grow_kv(target);
        self.cap = target;
        self.cap
    }

    /// Grow the shared K/V buffers; `prefill` also calls this directly when the
    /// slabbed path needs more rows than the stepped `cap`.
    fn grow_kv(&mut self, target: usize) {
        if target <= self.kv_cap {
            return;
        }
        let t0 = std::time::Instant::now();
        let prev = self.kv_cap;
        let nkvh = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim;
        let words = (nkvh * target * hd / 2) as u64;
        self.k_buf = self.gpu.storage("k_cache", words * 4);
        self.v_buf = self.gpu.storage("v_cache", words * 4);
        self.kv_cap = target;
        let mib = (2 * words * 4) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[kv] capacity {prev} -> {target} slots ({mib:.0} MiB) in {:.1} ms",
            t0.elapsed().as_secs_f64() * 1000.0,
        );
    }

    /// Upload MRoPE tables — `[rope_positions, head_dim]` in the run's 16-bit
    /// storage format, row-major.
    pub fn set_rope_tables(&self, cos: &[H16], sin: &[H16]) {
        self.gpu.upload(&self.scratch.cos, &weights::words_bytes(cos));
        self.gpu.upload(&self.scratch.sin, &weights::words_bytes(sin));
    }

    // ── the pass ─────────────────────────────────────────────────────────

    /// Read any scratch buffer back as 16-bit values in the run's format — the
    /// values the kernels actually round to, which is what the reference's own
    /// dtype holds.
    pub fn read_h16(&self, buf: &wgpu::Buffer, elems: usize) -> Result<Vec<H16>> {
        let bytes = self.gpu.readback(buf, (elems * 2) as u64)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| H16::from_le_bytes([c[0], c[1]]))
            .collect())
    }

}

/// Explicit pipeline layout for a kernel family whose variants share bind groups:
/// `storage` entries (binding, read_only) plus one uniform.
fn family_layout(
    gpu: &Gpu,
    label: &str,
    storage: &[(u32, bool)],
    uniform_binding: u32,
) -> wgpu::PipelineLayout {
    family_layout_dyn(gpu, label, storage, uniform_binding, false)
}

/// Same, with `uniform_dynamic` enabling per-dispatch dynamic offsets
/// (required when one uniform buffer carries per-dispatch values).
fn family_layout_dyn(
    gpu: &Gpu,
    label: &str,
    storage: &[(u32, bool)],
    uniform_binding: u32,
    uniform_dynamic: bool,
) -> wgpu::PipelineLayout {
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = storage
        .iter()
        .map(|&(binding, read_only)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: uniform_binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: uniform_dynamic,
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

fn upload_weight(up: &mut BulkUpload, label: &str, w: &PackedWeight) -> Result<wgpu::Buffer> {
    let b = up.storage(label, w.data.len() as u64);
    up.upload(&b, &w.data)?;
    Ok(b)
}

/// 16-bit vector in the run's format -> `array<u32>` binding.
fn upload_vec(up: &mut BulkUpload, label: &str, v: &[H16]) -> Result<wgpu::Buffer> {
    let b = up.storage(label, (v.len() * 2) as u64);
    up.upload(&b, &weights::words_bytes(v))?;
    Ok(b)
}

/// How many layers of converted weights the prefetch thread may run ahead with.
///
/// Two layers is a bounded amount of host memory, and enough that the narrowing
/// of the next layer always overlaps the transfer of the current one.
const PREFETCH_LAYERS: usize = 2;

/// One decoder layer's weights, converted and ready to be staged.
struct LayerWeights {
    qkv: Vec<(u64, bytes::Bytes)>,
    o: PackedWeight,
    gu: Vec<(u64, bytes::Bytes)>,
    dp: PackedWeight,
    iln: Vec<H16>,
    pln: Vec<H16>,
    qn: Vec<H16>,
    kn: Vec<H16>,
}

/// Read one layer's tensors out of the checkpoint in upload-ready form.
fn convert_layer(w: &HashMap<String, weights::RawTensor>, p: &str) -> Result<LayerWeights> {
    Ok(LayerWeights {
        o: weights::get_matrix(w, &format!("{p}.self_attn.o_proj.weight"))?,
        dp: weights::get_matrix(w, &format!("{p}.mlp.down_proj.weight"))?,
        qkv: fused_pieces(w, &format!("{p}.self_attn"), &["q_proj", "k_proj", "v_proj"])?,
        gu: fused_pieces(w, &format!("{p}.mlp"), &["gate_proj", "up_proj"])?,
        iln: weights::get_h16_vector(w, &format!("{p}.input_layernorm.weight"))?,
        pln: weights::get_h16_vector(w, &format!("{p}.post_attention_layernorm.weight"))?,
        qn: weights::get_h16_vector(w, &format!("{p}.self_attn.q_norm.weight"))?,
        kn: weights::get_h16_vector(w, &format!("{p}.self_attn.k_norm.weight"))?,
    })
}

/// Fuse `parts` into one row-major `[part0 | part1 | ...]` matrix, as
/// `(offset, bytes)` pieces ready for a single destination buffer.
///
/// The concatenation used to be a `PackedWeight` of its own: a full copy of
/// every fused matrix made purely so the upload could see one slice.  The
/// kernels only need the parts to be *contiguous*, so a checkpoint already in
/// the run's format contributes its own mapped bytes at their offsets (no copy
/// at all — the bf16 checkpoint in the default bf16 mode), and any other dtype
/// is narrowed straight into the fused layout, without the per-part buffers the
/// piecewise version needed.
fn fused_pieces(
    w: &HashMap<String, weights::RawTensor>,
    prefix: &str,
    parts: &[&str],
) -> Result<Vec<(u64, bytes::Bytes)>> {
    let tensors: Vec<&weights::RawTensor> = parts
        .iter()
        .map(|p| {
            let name = format!("{prefix}.{p}.weight");
            w.get(&name)
                .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))
        })
        .collect::<Result<_>>()?;

    if tensors.iter().all(|t| t.dtype == weights::storage_dtype()) {
        let mut pieces = Vec::with_capacity(tensors.len());
        let mut off = 0u64;
        for t in &tensors {
            pieces.push((off, t.data.clone()));
            off += t.data.len() as u64;
        }
        return Ok(pieces);
    }

    let mut out = Vec::with_capacity(fused_bytes(&tensors)?);
    for t in &tensors {
        let start = out.len();
        out.resize(start + t.data.len() / t.dtype.size() * 2, 0);
        t.narrow_h16_into(&mut out[start..])?;
    }
    Ok(vec![(0, out.into())])
}

fn fused_bytes(tensors: &[&weights::RawTensor]) -> Result<usize> {
    let mut total = 0usize;
    for t in tensors {
        if t.shape.len() != 2 {
            return Err(anyhow::anyhow!("expected 2D weight, got {:?}", t.shape));
        }
        total += t.shape[0] * t.shape[1] * 2;
    }
    Ok(total)
}

/// Grow-only buffers for one `prefill` call.  Uniforms have fixed size; every
/// token-sized buffer grows only when a longer clip needs it.  Held on the
/// decoder so consecutive clips reuse the allocations.
struct PrefillScratch {
    u_gd: wgpu::Buffer,
    u_sl: wgpu::Buffer,
    cap_mp: usize,
    h_buf: wgpu::Buffer,
    normed: wgpu::Buffer,
    norm2: wgpu::Buffer,
    qkv: wgpu::Buffer,
    q_out: wgpu::Buffer,
    attn_flat: wgpu::Buffer,
    gu: wgpu::Buffer,
    activated: wgpu::Buffer,
    cap_scores_bytes: u64,
    scores: wgpu::Buffer,
    cap_attn_bytes: u64,
    attn: wgpu::Buffer,
    slab: Option<SlabScratch>,
}

/// Slabbed-attention scratch, allocated only when the slabbed path is active.
struct SlabScratch {
    slab_part: wgpu::Buffer,
    slab_stats: wgpu::Buffer,
    slab_w: wgpu::Buffer,
    cap_part_bytes: u64,
    cap_stats_bytes: u64,
    cap_w_bytes: u64,
}

impl PrefillScratch {
    fn new(gpu: &Gpu) -> Self {
        Self {
            u_gd: gpu.uniform("p.gd", (MAX_GEMMS * 256) as u64),
            u_sl: gpu.uniform("p.sl", (2 * MAX_SLAB as u64 + 2) * 256),
            cap_mp: 0,
            h_buf: gpu.storage("p.h", 16),
            normed: gpu.storage("p.normed", 16),
            norm2: gpu.storage("p.norm2", 16),
            qkv: gpu.storage("p.qkv", 16),
            q_out: gpu.storage("p.q_out", 16),
            attn_flat: gpu.storage("p.attn_flat", 16),
            gu: gpu.storage("p.gu", 16),
            activated: gpu.storage("p.activated", 16),
            cap_scores_bytes: 0,
            scores: gpu.storage("p.scores", 16),
            cap_attn_bytes: 0,
            attn: gpu.storage("p.attn", 16),
            slab: None,
        }
    }

    fn storage(gpu: &Gpu, label: &str, buf: &mut wgpu::Buffer, cap: &mut u64, need: u64) {
        if need > *cap {
            *buf = gpu.storage(label, need);
            *cap = need;
        }
    }

    /// Grow every buffer that the next prefill needs, reusing those that fit.
    fn ensure(
        &mut self,
        gpu: &Gpu,
        words: &impl Fn(usize, usize) -> u64,
        (hs, fused, qh, hd, inter, mp, score_words, attn_words): (usize, usize, usize, usize, usize, usize, u64, u64),
        slab: Option<(u64, u64, u64)>,
    ) {
        if mp > self.cap_mp {
            let storages: [(&str, &mut wgpu::Buffer, u64); 8] = [
                ("p.h", &mut self.h_buf, words(mp, hs)),
                ("p.normed", &mut self.normed, words(mp, hs)),
                ("p.norm2", &mut self.norm2, words(mp, hs)),
                ("p.qkv", &mut self.qkv, words(mp, fused)),
                ("p.q_out", &mut self.q_out, words(qh * mp, hd)),
                ("p.attn_flat", &mut self.attn_flat, words(mp, qh * hd)),
                ("p.gu", &mut self.gu, words(mp, 2 * inter)),
                ("p.activated", &mut self.activated, words(mp, inter)),
            ];
            for (label, buf, need) in storages {
                *buf = gpu.storage(label, need);
            }
            self.cap_mp = mp;
        }
        Self::storage(gpu, "p.scores", &mut self.scores, &mut self.cap_scores_bytes, score_words);
        Self::storage(gpu, "p.attn", &mut self.attn, &mut self.cap_attn_bytes, attn_words);
        if let Some((part, stats, w)) = slab {
            let s = self.slab.get_or_insert_with(|| SlabScratch {
                slab_part: gpu.storage("p.slab_part", 16),
                slab_stats: gpu.storage("p.slab_stats", 16),
                slab_w: gpu.storage("p.slab_w", 16),
                cap_part_bytes: 0,
                cap_stats_bytes: 0,
                cap_w_bytes: 0,
            });
            Self::storage(gpu, "p.slab_part", &mut s.slab_part, &mut s.cap_part_bytes, part);
            Self::storage(gpu, "p.slab_stats", &mut s.slab_stats, &mut s.cap_stats_bytes, stats);
            Self::storage(gpu, "p.slab_w", &mut s.slab_w, &mut s.cap_w_bytes, w);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Prefill (stage 2)
// ═══════════════════════════════════════════════════════════════════════

impl WgpuTextDecoder {
    /// Prefill: run `s` input positions (hidden states `[s, hs]` as little-endian
    /// words in the run's 16-bit format) through every layer with causal
    /// attention, writing KV slots `kv_start..kv_start+s`, and leave the last
    /// layer's output in [`Self::debug_prefill_h`] for the caller to read back.
    ///
    /// This repo has no decode loop, so there is no first token to return and no
    /// LM head to run — the pass ends with the hidden states.
    ///
    /// f32 accumulation with one 16-bit rounding between ops; the accumulation
    /// order is this path's own.
    #[allow(clippy::too_many_lines)]
    pub fn prefill(&mut self, hidden_words: &[u8], s: usize, kv_start: usize) -> Result<()> {
        let cfg = self.cfg.clone();
        let hs = cfg.hidden_size;
        let nqh = cfg.num_attention_heads;
        let nkvh = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let cur = kv_start + s;
        // Modern f16-capable GPUs have high SM counts and lower shared-memory
        // pressure, so short sequences benefit from 64x64 wave occupancy;
        // older non-f16 GPUs keep 128x128 by default (`QALIGN_TILE64` overrides).
        let tile64 = tile64_mode().unwrap_or(self.gpu.has_f16() && s <= 512);
        let bm = if tile64 { 64 } else { 128 };
        let bn = if tile64 { 64 } else { 128 };
        let mp = s.div_ceil(128) * 128; // buffer padding stays 128-row aligned
        let np = cur.div_ceil(128) * 128;
        let cur16 = cur.div_ceil(16) * 16; // k-side zero pad for the AV GEMM
        anyhow::ensure!(hidden_words.len() >= s * hs * 2, "prefill hidden size mismatch");

        // Slabbed causal attention: tile the *key* dimension so the scratch is
        // `[nqh, mp, T]` instead of `[nqh, mp, cur]`.  The flat path's scratch is
        // O(s²), past the per-binding limit for long clips — and wgpu's failure
        // mode for that is garbage rather than an error, so the flat path refuses
        // explicitly.
        let slab_t = SLAB_T;
        let bind_limit = self.gpu.limits.max_storage_buffer_binding_size;
        let n_slab = if slab_path(s, nqh, bind_limit) { cur.div_ceil(slab_t) } else { 0 };
        let slabbed = n_slab > 0;
        // Attention reads whole key slabs directly from the shared K/V
        // scratch: rows must cover `n_slab · T`, which can exceed stepped
        // `cap` (4589 -> 4608 vs 5·1024).  Zero the tail below so exact-zero
        // softmax weights never multiply stale rows from a longer clip.
        let nkv_rows = if slabbed { n_slab * slab_t } else { np };
        let need_kv = nkv_rows.max(self.cap);
        if need_kv > self.kv_cap {
            self.grow_kv(need_kv);
        }
        let kv_stride = self.kv_cap;
        if slabbed {
            anyhow::ensure!(
                n_slab <= MAX_SLAB,
                "prefill: {n_slab} key slabs of {slab_t} exceed the {MAX_SLAB} uniform slots"
            );
            let part = (n_slab * mp * nqh * hd * 2) as u64;
            anyhow::ensure!(
                part <= bind_limit,
                "prefill slab scratch is {:.2} GiB for {s} tokens, over the {:.2} GiB per-binding limit",
                part as f64 / (1u64 << 30) as f64,
                bind_limit as f64 / (1u64 << 30) as f64,
            );
        } else {
            let scratch = (nqh * mp * cur16 * 2) as u64;
            anyhow::ensure!(
                scratch <= bind_limit,
                "prefill attention scratch is {:.2} GiB for {s} tokens ({:.1} min of audio), over the {:.2} GiB per-binding limit — \
                 the O(s²) prefill attention supports at most ~{} tokens",
                scratch as f64 / (1u64 << 30) as f64,
                s as f64 / 12.5 / 60.0,
                bind_limit as f64 / (1u64 << 30) as f64,
                ((bind_limit / (2 * nqh as u64)) as f64).sqrt() as usize,
            );
        }

        let words = |rows: usize, cols: usize| (rows * cols / 2 * 4) as u64;
        // `scores` is the score slab on the slabbed path (softmaxed in place) and
        // the whole `[nqh, mp, cur]` matrix otherwise; `attn` is its AV operand.
        let slab_cols = if slabbed { slab_t } else { np };
        let slab_bytes = slabbed.then(|| (
            words(n_slab * mp, nqh * hd),
            (n_slab * nqh * mp * 2 * 4) as u64,
            (n_slab * nqh * mp * 4) as u64,
        ));
        let gpu = &self.gpu;
        let ps = self.prefill_scratch.get_or_insert_with(|| PrefillScratch::new(gpu));
        ps.ensure(
            gpu, &words,
            (hs, cfg.fused_qkv_cols(), nqh, hd, inter, mp,
             words(nqh * mp, slab_cols),
             words(nqh * mp, if slabbed { slab_t } else { cur16 })),
            slab_bytes,
        );
        let PrefillScratch {
            u_gd, u_sl, h_buf, normed, norm2, qkv, q_out, scores, attn, attn_flat,
            gu, activated, slab: slab_scratch, ..
        } = &*ps;
        let (slab_part, slab_stats, slab_w) = match slab_scratch {
            Some(s) => (Some(&s.slab_part), Some(&s.slab_stats), Some(&s.slab_w)),
            None => (None, None, None),
        };
        // ── measurement dump (QALIGN_DUMP_LAYERS) ────────────────────────────
        // Off unless the env var names a file.  One f32 slab per layer index
        // holding the running hidden state (`h`, in the run's 16-bit format) at
        // each layer boundary, widened on the device, so a profile can be read
        // off against a torch reference layer by layer.  Nothing else in the
        // pass reads or writes these two buffers, so the arithmetic, the buffer
        // sizes and the dispatch order of every other kernel are untouched.
        let dump_path = std::env::var("QALIGN_DUMP_LAYERS")
            .ok()
            .filter(|p| !p.trim().is_empty());
        let mut up = gpu.uploader();
        let dump_h = dump_path
            .as_ref()
            .map(|_| up.storage("p.dump_h", (self.layers.len() + 1) as u64 * (mp * hs) as u64 * 4));
        let u_wd = dump_path
            .as_ref()
            .map(|_| up.uniform("p.wd", ((self.layers.len() + 1) * 256) as u64));
        up.upload(h_buf, hidden_words)?;
        up.finish()?;

        let dump_grid = dump_h
            .as_ref()
            .map(|_| grid_xy((mp * hs / 2).div_ceil(256)));
        if let (Some(u_wd), Some(grid)) = (u_wd.as_ref(), dump_grid) {
            // One cfg slot per dispatch: `write_buffer` copies are all retired at
            // submit start, so a single reused slot would give every dispatch the
            // last value written (the same reason `u_gd` carries 256 of them).
            for li in 0..=self.layers.len() {
                gpu.write_at(
                    u_wd,
                    (li * 256) as u64,
                    bytemuck::bytes_of(&WidenCfg {
                        words: (mp * hs / 2) as u32,
                        src0: 0,
                        dst0: (li * mp * hs) as u32,
                        gx: grid.0,
                    }),
                );
            }
        }
        // `coff` = element offset of C's batch 0 — non-zero only for the slabbed
        // AV GEMM, whose per-slab outputs all live in one buffer.
        let gemm_bg = |pipe: &wgpu::ComputePipeline,
                       a: &wgpu::Buffer,
                       w: &wgpu::Buffer,
                       c: &wgpu::Buffer,
                       coff: u64,
                       u_gd: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p.gemm"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: if coff == 0 {
                            c.as_entire_binding()
                        } else {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: c,
                                offset: coff,
                                size: None,
                            })
                        },
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: u_gd,
                            offset: 0,
                            size: std::num::NonZeroU64::new(64),
                        }),
                    },
                ],
            })
        };
        // Slab dispatches write their cfg into a slot of their own (the tile's),
        // so they never go through `gd_slot`; the first `2 · MAX_SLAB` slots are
        // reserved for them and the counter restarts there after every submit.
        let mut gd_slot = 2 * MAX_SLAB;
        macro_rules! gemm_slot {
            ($cp:expr, $pipe:expr, $a:expr, $w:expr, $c:expr, $coff:expr, $slot:expr,
             $gx:expr, $gy:expr, $gz:expr) => {{
                assert!($slot < MAX_GEMMS, "prefill: GEMM uniform slots exhausted");
                let off = (($slot) * 256) as u64;
                let bg = gemm_bg($pipe, $a, $w, $c, $coff, u_gd);
                $cp.set_pipeline($pipe);
                $cp.set_bind_group(0, &bg, &[off as u32]);
                $cp.dispatch_workgroups($gx, $gy, $gz);
            }};
        }
        macro_rules! gemm_at {
            ($cp:expr, $pipe:expr, $a:expr, $w:expr, $c:expr, $coff:expr, $slot:expr, $gdims:expr,
             $gx:expr, $gy:expr, $gz:expr) => {{
                assert!($slot < MAX_GEMMS, "prefill: GEMM uniform slots exhausted");
                let off = (($slot) * 256) as u64;
                gpu.queue.write_buffer(u_gd, off, bytemuck::bytes_of(&$gdims));
                gemm_slot!($cp, $pipe, $a, $w, $c, $coff, $slot, $gx, $gy, $gz);
            }};
        }
        macro_rules! gemm {
            ($cp:expr, $pipe:expr, $a:expr, $w:expr, $c:expr, $m:expr, $n:expr, $k:expr,
             $ldc:expr, $bsa:expr, $bsb:expr, $bsc:expr, $gqa:expr, $gx:expr, $gy:expr, $gz:expr) => {{
                let slot = gd_slot;
                gd_slot += 1;
                gemm_at!(
                    $cp, $pipe, $a, $w, $c, 0u64, slot,
                    GDims {
                        m: $m as u32,
                        n: $n as u32,
                        k: $k as u32,
                        ldc: $ldc as u32,
                        bsa: $bsa as u32,
                        bsb: $bsb as u32,
                        bsc: $bsc as u32,
                        beta: 0,
                        row0: 0,
                        lda: $k as u32,
                        gqa_group: $gqa,
                        _pad: 0,
                    },
                    $gx, $gy, $gz
                );
            }};
        }

        // per-call uniforms: multi-position extract + silu rows + softmax
        let silu_grid = write_silu_rows(gpu, &cfg, &self.scratch.u_silu, s);
        // One workgroup per (head, position) score row.
        let softmax_grid = grid_xy(nqh * s);
        gpu.upload(
            &self.scratch.u_qkvx,
            bytemuck::bytes_of(&QkvxCfg {
                max_seq: kv_stride as u32,
                start: kv_start as u32,
                pos_offset: kv_start as u32,
                // QOut row stride per head — mp (tile-padded), not s
                s: mp as u32,
                eps: cfg.rms_norm_eps,
                _a: 0,
                _b: 0,
                _c: 0,
            }),
        );
        // grids of the two layer-independent slab merges (used inside the loop):
        // the weights cover every stats row, the merge one head-dim band per head
        let w_grid = grid_xy((nqh * mp).div_ceil(256));
        let slab_m_grid = (mp * hd / 2).div_ceil(256) as u32;
        if slabbed {
            // One softmax + stats cfg per key slab; a slot's bytes are live for
            // the whole submit, so every tile needs its own.
            for t in 0..n_slab {
                let t0 = t * slab_t;
                let tl = (cur16 - t0).min(slab_t);
                gpu.write_at(
                    u_sl,
                    (t * 256) as u64,
                    bytemuck::bytes_of(&SoftmaxCfg {
                        n_w: (slab_t / 2) as u32,
                        n_x: (slab_t / 2) as u32,
                        valid: tl as u32,
                        m: s as u32,
                        mp: mp as u32,
                        scale: cfg.scale(),
                        gx: softmax_grid.0,
                        row0: t0 as u32,
                    }),
                );
                let sg = GDims {
                    m: s as u32,
                    n: slab_t as u32,
                    k: hd as u32,
                    ldc: slab_t as u32,
                    bsa: (mp * hd / 2) as u32,
                    bsb: (kv_stride * hd / 2) as u32,
                    bsc: (mp * slab_t) as u32,
                    beta: 0,
                    row0: t0 as u32,
                    lda: hd as u32,
                    gqa_group: (nqh / nkvh) as u32,
                    _pad: 0,
                };
                gpu.write_at(u_gd, (t * 256) as u64, bytemuck::bytes_of(&sg));
                let ag = GDims {
                    m: s as u32,
                    n: hd as u32,
                    k: tl as u32,
                    ldc: (nqh * hd) as u32,
                    bsa: (mp * slab_t / 2) as u32,
                    bsb: (kv_stride * hd / 2) as u32,
                    bsc: hd as u32,
                    beta: 0,
                    row0: t0 as u32,
                    lda: slab_t as u32,
                    gqa_group: (nqh / nkvh) as u32,
                    _pad: 0,
                };
                gpu.write_at(u_gd, ((n_slab + t) * 256) as u64, bytemuck::bytes_of(&ag));
                gpu.write_at(
                    u_sl,
                    ((MAX_SLAB + t) * 256) as u64,
                    bytemuck::bytes_of(&SlabStatsCfg {
                        n_x: (slab_t / 2) as u32,
                        valid: tl as u32,
                        m: s as u32,
                        mp: mp as u32,
                        scale: cfg.scale(),
                        row0: t0 as u32,
                        gx: softmax_grid.0,
                        rows: (nqh * mp) as u32,
                    }),
                );
            }
            gpu.write_at(
                u_sl,
                (2 * MAX_SLAB * 256) as u64,
                bytemuck::bytes_of(&SlabWeightsCfg {
                    rows: (nqh * mp) as u32,
                    n_slab: n_slab as u32,
                    gx: w_grid.0,
                    _p: 0,
                }),
            );
            gpu.write_at(
                u_sl,
                ((2 * MAX_SLAB + 1) * 256) as u64,
                bytemuck::bytes_of(&SlabMergeCfg {
                    rows: mp as u32,
                    n_slab: n_slab as u32,
                    gx: slab_m_grid,
                    _p: 0,
                }),
            );
        } else {
            gpu.write_at(
                u_sl,
                0,
                bytemuck::bytes_of(&SoftmaxCfg {
                    n_w: (cur16 / 2) as u32,
                    n_x: (np / 2) as u32,
                    valid: cur as u32,
                    m: s as u32,
                    mp: mp as u32,
                    scale: cfg.scale(),
                    gx: softmax_grid.0,
                    row0: 0,
                }),
            );
        }
        // The K/V scratch is reused across clips and read by `scores` up to
        // `nkv_rows` and by `av` up to `cur16`.  Zero rows `cur..nkv_rows` for
        // every KV head so an exact-zero softmax weight can never multiply a
        // stale value left by a longer previous clip (`0 * NaN` is NaN).
        let tail_rows = nkv_rows - cur;
        if tail_rows > 0 {
            let zero = vec![0u8; tail_rows * hd * 2];
            for ih in 0..nkvh {
                let off = ((ih * kv_stride + cur) * hd * 2) as u64;
                gpu.write_at(&self.k_buf, off, &zero);
                gpu.write_at(&self.v_buf, off, &zero);
            }
        }

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut cp = enc.begin_compute_pass(&Default::default());
        // Resolve the pipelines once: `self.pipe()` borrows `self`, which cannot
        // coexist with the mutable borrow of `self.debug_prefill_h` held by the
        // state below.
        let pipe0 = Pipes::pipe(&self.pipes, tile64, 0);
        let pipe1 = Pipes::pipe(&self.pipes, tile64, 1);
        let pipe2 = Pipes::pipe(&self.pipes, tile64, 2);
        let pipe3 = Pipes::pipe(&self.pipes, tile64, 3);
        let pipe4 = Pipes::pipe(&self.pipes, tile64, 4);

        // Ablation hook (see `encode_step`).
        let dup = std::env::var("QASR_DUP").unwrap_or_default();

        // The other ablation hook: `QALIGN_SKIP=a,b` leaves whole dispatch groups
        // out of the pass, which prices exactly what it omits.  Prefer it to
        // `QASR_DUP`, which pays for a dispatch *added* in the middle of a chain
        // and reads high on a GEMM because of the cache it disturbs on the way in.
        //
        // A skipped group's operands go stale, so the result is wrong on purpose:
        // the clock is the measurement, and the unskipped run is what the gate
        // judges.  `skipped` is printed at the end — a name that matched nothing
        // would otherwise be indistinguishable from a group that costs nothing.
        let skip_list: Vec<String> = std::env::var("QALIGN_SKIP")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        let mut skipped = 0usize;
        macro_rules! skip {
            ($name:literal) => {
                skip_list.iter().any(|s| s == $name)
            };
        }
        macro_rules! gemm_or_skip {
            ($name:literal, $($rest:tt)*) => {{
                if skip!($name) {
                    skipped += 1;
                } else {
                    gemm!($($rest)*);
                }
            }};
        }
        macro_rules! dispatch {
            ($name:literal, $cp:ident, $gx:expr, $gy:expr, $gz:expr) => {{
                if skip!($name) {
                    skipped += 1;
                } else {
                    $cp.dispatch_workgroups($gx, $gy, $gz);
                }
            }};
        }
        // Measurement dump: copy `h` (16-bit storage) into dump slab `slot`,
        // widened to f32.  A no-op when `QALIGN_DUMP_LAYERS` is unset, and never
        // on a `QALIGN_SKIP` list — it is instrumentation, not a GEMM.
        macro_rules! dump_layer {
            ($cp:expr, $slot:expr) => {{
                if let (Some(d), Some(u), Some(g)) = (dump_h.as_ref(), u_wd.as_ref(), dump_grid) {
                    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("p.widen"),
                        layout: &self.pipes.widen.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: h_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: d.as_entire_binding() },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: u,
                                    offset: 0,
                                    size: std::num::NonZeroU64::new(32),
                                }),
                            },
                        ],
                    });
                    $cp.set_pipeline(&self.pipes.widen);
                    $cp.set_bind_group(0, &bg, &[($slot * 256) as u32]);
                    $cp.dispatch_workgroups(g.0, g.1, 1);
                }
            }};
        }

        // Layer index 0 of the dump is the state the first layer's input norm
        // reads: the embedding rows with the audio frames scattered in, before
        // any layer has run.
        dump_layer!(cp, 0);

        for (li, layer) in self.layers.iter().enumerate() {
            // 1. rms_norm(h, iln) → normed   [s, hs]
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p.rms1"),
                layout: &self.pipes.rms_norm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: h_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: layer.iln_w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: normed.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.scratch.u_rms.as_entire_binding() },
                ],
            });
            cp.set_pipeline(&self.pipes.rms_norm);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(s as u32, 1, 1);

            // 2. qkv GEMM  [s, fused] = normed × qkv_wᵀ
            gemm_or_skip!(
                "qkv",
                &mut cp, pipe0, &normed, &layer.qkv_w, &qkv,
                s, cfg.fused_qkv_cols(), hs, cfg.fused_qkv_cols(), 0, 0, 0, 1,
                (cfg.fused_qkv_cols() / bn) as u32, (mp / bm) as u32, 1
            );

            // 3. extract Q/K/V for all positions (K/V land in the cache)
            let bg_ex = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p.extract"),
                layout: &self.pipes.extract.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: qkv.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: layer.qn_w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: layer.kn_w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.scratch.cos.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.scratch.sin.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: q_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.k_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: self.v_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: self.scratch.u_qkvx.as_entire_binding() },
                ],
            });
            cp.set_pipeline(&self.pipes.extract);
            cp.set_bind_group(0, &bg_ex, &[]);
            cp.dispatch_workgroups(s as u32, (nqh + nkvh) as u32, 1);
            if dup == "p_extract" {
                cp.dispatch_workgroups(s as u32, (nqh + nkvh) as u32, 1);
            }

            if slabbed {
                // 5-8 (slabbed): one key slab of scores at a time, three passes
                // over the slabs.  `slab_stats` records each slab's (max, Σexp)
                // while the slab is live, the softmax normalises the slab against
                // its own max, the AV GEMM turns it into that slab's output, and
                // `slab_weights` + `slab_merge` combine the slabs weighted by
                // `exp(m_t − M)·Σexp_t` — the exact per-slab softmax masses.
                let spart = slab_part.as_ref().unwrap();
                let sstats = slab_stats.as_ref().unwrap();
                let swt = slab_w.as_ref().unwrap();
                let sgrid = ((slab_t / bn) as u32, (mp / bm) as u32, nqh as u32);
                for t in 0..n_slab {
                    // score tile [mp, T] = q × K[t0..t0+T)ᵀ, tiles above the
                    // diagonal skipped (slab GDims pre-written in slots 0..n_slab)
                    gemm_slot!(
                        &mut cp, pipe3, &q_out, &self.k_buf, &scores,
                        0u64, t,
                        sgrid.0, sgrid.1, sgrid.2
                    );
                    // Ablation hook (as in the flat path): the GEMM overwrites its
                    // output, so re-dispatching leaves the token stream alone.
                    if dup == "p_scores" {
                        gemm_slot!(
                            &mut cp, pipe3, &q_out, &self.k_buf, &scores,
                            0u64, t,
                            sgrid.0, sgrid.1, sgrid.2
                        );
                    }

                    // per-slab (max, Σexp) for the merge weights
                    let bg_st = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("p.slab_stats"),
                        layout: &self.pipes.slab_stats.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: scores.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: sstats.as_entire_binding() },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    // the dynamic offset is the whole slot
                                    // address — an entry offset here would be
                                    // *added* to it (and silently read another
                                    // cfg: this dispatch reads the weights slot
                                    // if both are set)
                                    buffer: u_sl,
                                    offset: 0,
                                    size: std::num::NonZeroU64::new(32),
                                }),
                            },
                        ],
                    });
                    cp.set_pipeline(&self.pipes.slab_stats);
                    cp.set_bind_group(0, &bg_st, &[((MAX_SLAB + t) * 256) as u32]);
                    cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);

                    // causal softmax on the slab: the normalised slab lands in
                    // `attn` (the AV GEMM's A operand; the same buffer cannot be
                    // both operands of this dispatch)
                    let bg_sm = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("p.slab_sm"),
                        layout: &self.pipes.softmax[&slab_t].get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: scores.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: attn.as_entire_binding() },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    // slot address via the dynamic offset alone
                                    buffer: u_sl,
                                    offset: 0,
                                    size: std::num::NonZeroU64::new(32),
                                }),
                            },
                        ],
                    });
                    cp.set_pipeline(&self.pipes.softmax[&SLAB_BS]);
                    cp.set_bind_group(0, &bg_sm, &[(t * 256) as u32]);
                    cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);
                    if dup == "p_softmax" {
                        cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);
                    }

                    // this slab's AV output: [mp, nqh·hd], k bounded by the
                    // diagonal inside the slab (GDims pre-written in n_slab..2*n_slab)
                    gemm_slot!(
                        &mut cp, pipe4, &attn, &self.v_buf, spart,
                        (t * mp * nqh * hd * 2) as u64, n_slab + t,
                        (hd / bn) as u32, (mp / bm) as u32, nqh as u32
                    );
                    if dup == "p_av" {
                        gemm_slot!(
                            &mut cp, pipe4, &attn, &self.v_buf, spart,
                            (t * mp * nqh * hd * 2) as u64, n_slab + t,
                            (hd / bn) as u32, (mp / bm) as u32, nqh as u32
                        );
                    }
                }

                // per-row slab weights, then the weighted merge into attn_flat
                let bg_w = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("p.slab_w"),
                    layout: &self.pipes.slab_weights.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: sstats.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: swt.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: u_sl,
                                // static: the weights cfg is layer-independent
                                offset: (2 * MAX_SLAB * 256) as u64,
                                size: std::num::NonZeroU64::new(32),
                            }),
                        },
                    ],
                });
                cp.set_pipeline(&self.pipes.slab_weights);
                cp.set_bind_group(0, &bg_w, &[]);
                cp.dispatch_workgroups(w_grid.0, w_grid.1, 1);

                let bg_m = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("p.slab_merge"),
                    layout: &self.pipes.slab_merge.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: spart.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: swt.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: attn_flat.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: u_sl,
                                // static: one merge cfg for the whole prefill
                                offset: ((2 * MAX_SLAB + 1) * 256) as u64,
                                size: std::num::NonZeroU64::new(32),
                            }),
                        },
                    ],
                });
                cp.set_pipeline(&self.pipes.slab_merge);
                cp.set_bind_group(0, &bg_m, &[]);
                cp.dispatch_workgroups(slab_m_grid, nqh as u32, 1);
            } else {
                // 5. scores GEMM, batched over heads: [s, cur] = q × Kᵀ  (K is [cur, hd])
                // batch strides in WORDS (the shader indexes array<u32> directly);
                // bsc stays in ELEMENTS (the epilogue divides the sum by 2)
                gemm_or_skip!(
                    "scores",
                    &mut cp, pipe3, &q_out, &self.k_buf, &scores,
                    s, cur, hd, np, mp * hd / 2, kv_stride * hd / 2, mp * np, (nqh / nkvh) as u32,
                    (np / bn) as u32, (mp / bm) as u32, nqh as u32
                );
                // Ablation hook: re-dispatching is idempotent (the GEMM overwrites
                // its output), so the token stream is unchanged.
                if dup == "p_scores" {
                    gemm!(
                        &mut cp, pipe0, &q_out, &self.k_buf, &scores,
                        s, cur, hd, np, mp * hd / 2, kv_stride * hd / 2, mp * np, (nqh / nkvh) as u32,
                        (np / bn) as u32, (mp / bm) as u32, nqh as u32
                    );
                }

                // 6. causal softmax, in place on scores
                let bs = softmax_bs(cur) as usize;
                // softmax reads scores, writes straight into the AV input buffer
                let bg_sm = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("p.sm"),
                    layout: &self.pipes.softmax[&bs].get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: scores.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: attn.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: u_sl,
                                offset: 0,
                                size: std::num::NonZeroU64::new(32),
                            }),
                        },
                    ],
                });
                cp.set_pipeline(&self.pipes.softmax[&bs]);
                cp.set_bind_group(0, &bg_sm, &[0]);
                dispatch!("softmax", cp, softmax_grid.0, softmax_grid.1, 1);
                if dup == "p_softmax" {
                    cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);
                }

                // 7. AV GEMM, batched: attn_flat[s, nqh*hd] = attn × V  (V is [cur, hd])
                gemm_or_skip!(
                    "av",
                    &mut cp, pipe4, &attn, &self.v_buf, &attn_flat,
                    s, hd, cur16, nqh * hd, mp * cur16 / 2, kv_stride * hd / 2, hd, (nqh / nkvh) as u32,
                    (hd / bn) as u32, (mp / bm) as u32, nqh as u32
                );
                if dup == "p_av" {
                    gemm!(
                        &mut cp, pipe2, &attn, &self.v_buf, &attn_flat,
                        s, hd, cur16, nqh * hd, mp * cur16 / 2, kv_stride * hd / 2, hd, (nqh / nkvh) as u32,
                        (hd / bn) as u32, (mp / bm) as u32, nqh as u32
                    );
                }
            }

            // 8. o projection + residual:  h += attn_flat × o_wᵀ
            gemm_or_skip!(
                "o",
                &mut cp, pipe1, &attn_flat, &layer.o_w, &h_buf,
                s, hs, nqh * hd, hs, 0, 0, 0, 1,
                (hs / bn) as u32, (mp / bm) as u32, 1
            );

            // 9. rms_norm(h, pln) → norm2
            let bg_rms2 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p.rms2"),
                layout: &self.pipes.rms_norm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: h_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: layer.pln_w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: norm2.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.scratch.u_rms.as_entire_binding() },
                ],
            });
            cp.set_pipeline(&self.pipes.rms_norm);
            cp.set_bind_group(0, &bg_rms2, &[]);
            cp.dispatch_workgroups(s as u32, 1, 1);
            if dup == "p_rms" {
                cp.dispatch_workgroups(s as u32, 1, 1);
            }

            // 10. gate/up GEMM → silu → down GEMM + residual
            gemm_or_skip!(
                "gu",
                &mut cp, pipe0, &norm2, &layer.gu_w, &gu,
                s, 2 * inter, hs, 2 * inter, 0, 0, 0, 1,
                ((2 * inter) / bn) as u32, (mp / bm) as u32, 1
            );

            let bg_silu = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("p.silu"),
                layout: &self.pipes.silu.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gu.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: activated.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.scratch.u_silu.as_entire_binding() },
                ],
            });
            cp.set_pipeline(&self.pipes.silu);
            cp.set_bind_group(0, &bg_silu, &[]);
            cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
            if dup == "p_silu" {
                cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
            }

            gemm_or_skip!(
                "down",
                &mut cp, pipe1, &activated, &layer.dp_w, &h_buf,
                s, hs, inter, hs, 0, 0, 0, 1,
                (hs / bn) as u32, (mp / bm) as u32, 1
            );

            // Measurement dump: `h` now holds layer `li`'s output.
            dump_layer!(cp, li + 1);

            // Split long command buffers for WDDM's TDR watchdog without
            // blocking the CPU between batches: every GEMM keeps its own uniform
            // slot, so the driver can run submitted work while we keep encoding.
            let submit_every = if s >= 4096 { 4 } else { 7 };
            let per_layer_gemms = if dup.is_empty() { 6usize } else { 8usize };
            let wrap = gd_slot + per_layer_gemms > MAX_GEMMS;
            if (li + 1) < self.layers.len()
                && ((s >= 512 && (li + 1) % submit_every == 0) || wrap)
            {
                drop(cp);
                gpu.queue.submit([enc.finish()]);
                if wrap {
                    if let Err(e) = gpu.device.poll(wgpu::PollType::wait_indefinitely()) {
                        anyhow::bail!("prefill: device lost after layer {li}: {e:?}");
                    }
                }
                enc = gpu.device.create_command_encoder(&Default::default());
                cp = enc.begin_compute_pass(&Default::default());
                if wrap {
                    // Slab cfgs 0..2·n_slab are written once before the loop;
                    // only the per-layer slots are reused.
                    gd_slot = 2 * MAX_SLAB;
                }
            }
        }

        drop(cp);
        gpu.queue.submit([enc.finish()]);

        // The guard: a name that matched nothing must not look like a group that
        // costs nothing.
        if !skip_list.is_empty() {
            eprintln!(
                "prefill: QALIGN_SKIP={} skipped {skipped} dispatch(es) over {} layer(s), s={s} \
                 (0 means no name matched)",
                skip_list.join(","),
                self.layers.len(),
            );
        }

        // The caller reads the hidden states back out of this buffer; it stays
        // alive (stored in the struct) until the next prefill replaces it.
        self.debug_prefill_h = Some(ps.h_buf.clone());

        // ── measurement dump: hand the f32 layer stack to the comparison ──────
        if let (Some(path), Some(d)) = (dump_path.as_ref(), dump_h.as_ref()) {
            let n = self.layers.len() + 1;
            let bytes = self
                .gpu
                .readback(d, n as u64 * (mp * hs) as u64 * 4)
                .context("read back QALIGN_DUMP_LAYERS")?;
            write_layer_dump(path, &bytes, n, s, mp, hs)?;
        }
        Ok(())
    }
}

/// CFG for [`shaders::widen_h16_f32`] — word count plus the source word offset
/// and the destination *element* offset.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WidenCfg {
    words: u32,
    src0: u32,
    dst0: u32,
    gx: u32,
}

/// Write a measurement layer stack as raw little-endian f32 plus a `.json`
/// sidecar describing the shape (the task's "raw `.bin` + sidecar" option —
/// nothing here needs a container format, and numpy reads it with `fromfile`).
///
/// The buffer is the device's `[n, mp, hs]` tile layout, padding rows included:
/// the sidecar carries `rows_padded` so the reader can slice the first `seq`
/// rows without guessing, and `layer_stride_elems` is the padded stride.
fn write_layer_dump(
    path: &str,
    bytes: &[u8],
    layers: usize,
    seq: usize,
    mp: usize,
    hs: usize,
) -> Result<()> {
    std::fs::write(path, bytes).with_context(|| format!("write {path}"))?;
    let meta = serde_json::json!({
        "kind": "layer_hidden",
        "shape": [layers, seq, hs],
        "shape_padded": [layers, mp, hs],
        "dtype": "float32",
        "order": "layer-major, then row-major ([layer][row][hidden])",
        "row_stride_elems": hs,
        "layer_stride_elems": mp * hs,
        "rows_padded": mp,
        "half": crate::shaders::half().name(),
        "note": "hidden state at every layer boundary of the text decoder's prefill, in the run's 16-bit storage rounding, widened to f32 on the device through unpack_h (values, not bits); index 0 is the state the first layer's input norm reads (embeddings with audio rows scattered in); rows >= shape[1] are tile padding",
        "bytes": bytes.len(),
    });
    let side = format!("{path}.json");
    std::fs::write(&side, serde_json::to_string_pretty(&meta)?)
        .with_context(|| format!("write {side}"))?;
    eprintln!("[dump] QALIGN_DUMP_LAYERS: {layers} layers x {seq} rows ({mp} padded) -> {path} (+ .json)");
    Ok(())
}
