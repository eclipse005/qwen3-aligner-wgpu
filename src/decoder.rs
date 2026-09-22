//! wgpu text decoder.
//!
//! Forced alignment has no generation loop, so this module is one causal pass
//! over the prompt (see [`WgpuTextDecoder::prefill`]) rather than a decode step:
//!
//! ```text
//! for each of 28 layers:
//!   rms_norm(h, input_layernorm)          -> normed
//!   gemm(normed, qkv_w)                   -> qkv            (4096)
//!   qkv_extract(qkv)                      -> q_out, k_cache[i], v_cache[i]
//!   repeat_kv(k/v_cache[i])               -> k_rep, v_rep
//!   causal attention (slabbed once cur > SLAB_T)
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

use crate::gpu::{BulkUpload, Gpu};
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
struct RepeatKvCfg {
    nkvh: u32,
    max_seq: u32,
    cur: u32,
    hd: u32,
    npw: u32,
    /// x-axis grid size; `y` continues the flat index space beyond it.
    gx: u32,
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

/// Everything a single decoder layer needs, including its own KV cache slice.
///
/// Only the weights and the two caches live here: `prefill` builds its own bind
/// groups per call, so there is nothing KV-dependent to keep in sync.
struct Layer {
    k_cache: wgpu::Buffer,
    v_cache: wgpu::Buffer,
    iln_w: wgpu::Buffer,
    pln_w: wgpu::Buffer,
    qn_w: wgpu::Buffer,
    kn_w: wgpu::Buffer,
    qkv_w: wgpu::Buffer,
    o_w: wgpu::Buffer,
    gu_w: wgpu::Buffer,
    dp_w: wgpu::Buffer,
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
    /// Causal softmax, one pipeline per block size (reduction tree depends on it).
    softmax: std::collections::HashMap<usize, wgpu::ComputePipeline>,
    repeat_kv: wgpu::ComputePipeline,
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
        let rk_pl = family_layout(&gpu, "repeat_kv", &[(0, true), (1, false)], 2);
        // measurement dump: two storages + a per-dispatch uniform slot
        let wd_pl = family_layout_dyn(&gpu, "widen_dump", &[(0, true), (1, false)], 2, true);

        // ── pipelines ─────────────────────────────────────────────────────
        let rms_bs = rms_bs(hs) as usize;
        let pipes = Pipes {
            rms_norm: build("rms_norm", &shaders::rms_norm(hs, rms_bs), "rms_norm", None)?,
            extract: build("qkv_extract", &shaders::qkv_extract(nqh, nkvh, hd), "qkv_extract", None)?,
            silu: build("silu30", &shaders::silu_mul_split(inter), "silu_mul_split", None)?,
            gemm: build("gemm", &shaders::prefill_gemm(false, false), "gemm", Some(&gemm_pl))?,
            gemm_acc: build("gemm_acc", &shaders::prefill_gemm(false, true), "gemm", Some(&gemm_pl))?,
            gemm_av: build("gemm_av", &shaders::prefill_gemm(true, false), "gemm", Some(&gemm_pl))?,
            gemm_causal: build("gemm_causal", &shaders::prefill_gemm_causal(), "gemm", Some(&gemm_pl))?,
            gemm_av_causal: build("gemm_av_causal", &shaders::prefill_gemm_causal_av(), "gemm", Some(&gemm_pl))?,
            softmax: std::collections::HashMap::from([
                (32, build("softmax32", &shaders::softmax_causal(32, gpu.subgroup32()), "softmax", Some(&sm_pl))?),
                (64, build("softmax64", &shaders::softmax_causal(64, gpu.subgroup32()), "softmax", Some(&sm_pl))?),
                (128, build("softmax128", &shaders::softmax_causal(128, gpu.subgroup32()), "softmax", Some(&sm_pl))?),
                (256, build("softmax256", &shaders::softmax_causal(256, gpu.subgroup32()), "softmax", Some(&sm_pl))?),
                (512, build("softmax512", &shaders::softmax_causal(512, gpu.subgroup32()), "softmax", Some(&sm_pl))?),
                (1024, build("softmax1024", &shaders::softmax_causal(1024, gpu.subgroup32()), "softmax", Some(&sm_pl))?),
            ]),
            repeat_kv: build("repeat_kv", &shaders::repeat_kv(nqh / nkvh), "repeat_kv", Some(&rk_pl))?,
            slab_stats: build(
                "slab_stats",
                &shaders::slab_stats(SLAB_BS, SLAB_T),
                "slab_stats",
                Some(&family_layout_dyn(&gpu, "slab_stats", &[(0, true), (1, false)], 2, true)),
            )?,
            slab_weights: build(
                "slab_weights",
                &shaders::slab_weights(256),
                "slab_weights",
                Some(&family_layout(&gpu, "slab_weights", &[(0, true), (1, false)], 2)),
            )?,
            slab_merge: build(
                "slab_merge",
                &shaders::slab_merge(nqh, hd),
                "slab_merge",
                Some(&family_layout(&gpu, "slab_merge", &[(0, true), (1, true), (2, false)], 3)),
            )?,
            widen: build("widen_dump", &shaders::widen_h16_f32(), "widen", Some(&wd_pl))?,
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

                let k_cache = gpu.storage("k_cache", (kv_words * 4) as u64);
                let v_cache = gpu.storage("v_cache", (kv_words * 4) as u64);

                layers.push(Layer {
                    k_cache,
                    v_cache,
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

        Ok(Self {
            gpu,
            cfg,
            max_seq,
            cap,
            layers,
            pipes,
            scratch,
            debug_prefill_h: None,
        })
    }

    // ── KV cache / RoPE plumbing ─────────────────────────────────────────

    /// The device the decoder lives on — shared with the GPU audio tower.
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
        let t0 = std::time::Instant::now();
        let prev = self.cap;
        let kv_words = (self.cfg.num_key_value_heads * target * self.cfg.head_dim / 2) as u64;
        {
            let Self { gpu, layers, .. } = self;
            for layer in layers.iter_mut() {
                layer.k_cache = gpu.storage("k_cache", kv_words * 4);
                layer.v_cache = gpu.storage("v_cache", kv_words * 4);
            }
        }
        self.cap = target;
        let mib = (2 * self.cfg.num_hidden_layers as u64 * kv_words * 4) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[kv] capacity {prev} -> {target} slots ({mib:.0} MiB) in {:.1} ms",
            t0.elapsed().as_secs_f64() * 1000.0,
        );
        self.cap
    }

    /// Upload MRoPE tables — `[rope_positions, head_dim]` in the run's 16-bit
    /// storage format, row-major.
    pub fn set_rope_tables(&self, cos: &[H16], sin: &[H16]) {
        self.gpu.upload(&self.scratch.cos, &weights::words_bytes(cos));
        self.gpu.upload(&self.scratch.sin, &weights::words_bytes(sin));
    }

    // ── the pass ─────────────────────────────────────────────────────────

    /// SiLU row count: decode processes one `[gate|up]` row; prefill s rows.
    /// Returns the `(x, y)` grid that covers `rows · inter/2` words — wgpu caps
    /// each grid dimension at 65535, which a long prefill exceeds.
    fn write_silu_rows(&self, rows: usize) -> (u32, u32) {
        let words = rows * self.cfg.intermediate_size / 2;
        let workgroups = words.div_ceil(256);
        let (gx, gy) = grid_xy(workgroups);
        self.gpu.upload(
            &self.scratch.u_silu,
            bytemuck::bytes_of(&SiluCfg {
                inter2: (self.cfg.intermediate_size / 2) as u32,
                total2: words as u32,
                gx,
                _b: 0,
            }),
        );
        (gx, gy)
    }

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
        let mp = s.div_ceil(128) * 128; // m-side tile rows
        let np = cur.div_ceil(128) * 128; // n-side tile columns
        let cur16 = cur.div_ceil(16) * 16; // k-side zero pad for the AV GEMM
        anyhow::ensure!(hidden_words.len() >= s * hs * 2, "prefill hidden size mismatch");

        // Slabbed causal attention: tile the *key* dimension so the scratch is
        // `[nqh, mp, T]` instead of `[nqh, mp, cur]`.  The flat path's scratch is
        // O(s²), past the per-binding limit for long clips — and wgpu's failure
        // mode for that is garbage rather than an error, so the flat path refuses
        // explicitly.
        let slab_t = SLAB_T;
        let bind_limit = self.gpu.limits.max_storage_buffer_binding_size as u64;
        let n_slab = if slab_path(s, nqh, bind_limit) { cur.div_ceil(slab_t) } else { 0 };
        let slabbed = n_slab > 0;
        // k_rep/v_rep carry whole key slabs on the slabbed path, so their rows
        // cover `n_slab · T` (a little past `cur`); the tail is never *read*
        // meaningfully — the softmax zeroes the columns past the causal bound
        // before the AV GEMM sees them.
        let nkv_rows = if slabbed { n_slab * slab_t } else { np };
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
        let mut up = self.gpu.uploader();
        let h_buf = up.storage("p.h", words(mp, hs));
        let normed = up.storage("p.normed", words(mp, hs));
        let norm2 = up.storage("p.norm2", words(mp, hs));
        let qkv = up.storage("p.qkv", words(mp, cfg.fused_qkv_cols()));
        let q_out = up.storage("p.q_out", words(nqh * mp, hd));
        // `scores` is the score slab on the slabbed path (softmaxed in place) and
        // the whole `[nqh, mp, cur]` matrix otherwise; `attn` only exists flat.
        let slab_cols = if slabbed { slab_t } else { np };
        let scores = up.storage("p.scores", words(nqh * mp, slab_cols));
        let k_rep = up.storage("p.k_rep", words(nqh * nkv_rows, hd));
        let v_rep = up.storage("p.v_rep", words(nqh * nkv_rows, hd));
        // softmax output / AV operand: `cur16` columns flat, the slab width when
        // tiled.  Separate from `scores` on both paths — wgpu refuses to bind one
        // buffer as both read-only and read-write in a single dispatch, so the
        // softmax cannot normalise the slab in place.
        let attn = up.storage("p.attn", words(nqh * mp, if slabbed { slab_t } else { cur16 }));
        // `[n_slab, mp, nqh·hd]`: each slab's own (already normalised) AV output,
        // combined by `slab_merge` with the per-slab softmax weights
        let slab_part = slabbed.then(|| up.storage("p.slab_part", words(n_slab * mp, nqh * hd)));
        let slab_stats = slabbed.then(|| up.storage("p.slab_stats", (n_slab * nqh * mp * 2 * 4) as u64));
        let slab_w = slabbed.then(|| up.storage("p.slab_w", (n_slab * nqh * mp * 4) as u64));
        let attn_flat = up.storage("p.attn_flat", words(mp, nqh * hd));
        let gu = up.storage("p.gu", words(mp, 2 * inter));
        let activated = up.storage("p.activated", words(mp, inter));
        // one 256B slot per GEMM dispatch (dynamic-offset alignment); writes
        // land at submit start, so per-dispatch values MUST live in distinct
        // slots — a single reused uniform would give every dispatch the last
        // written value
        const MAX_GEMMS: u64 = 256;
        let u_gd = up.uniform("p.gd", MAX_GEMMS * 256);
        // slab cfgs, one 256B slot each: [0, MAX_SLAB) softmax, [MAX_SLAB, 2·MAX_SLAB)
        // slab_stats, then the layer-independent weights and merge cfgs
        let u_sl = up.uniform("p.sl", (2 * MAX_SLAB as u64 + 2) * 256);
        let u_rk = up.uniform("p.rk", 32);
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
        let dump_h = dump_path
            .as_ref()
            .map(|_| up.storage("p.dump_h", (self.layers.len() + 1) as u64 * (mp * hs) as u64 * 4));
        let u_wd = dump_path
            .as_ref()
            .map(|_| up.uniform("p.wd", ((self.layers.len() + 1) * 256) as u64));
        up.upload(&h_buf, hidden_words)?;
        up.finish()?;

        let gpu = &self.gpu;
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
        macro_rules! gemm_at {
            ($cp:expr, $pipe:expr, $a:expr, $w:expr, $c:expr, $coff:expr, $slot:expr, $gdims:expr,
             $gx:expr, $gy:expr, $gz:expr) => {{
                assert!($slot < MAX_GEMMS as usize, "prefill: GEMM uniform slots exhausted");
                let off = (($slot) * 256) as u64;
                gpu.queue.write_buffer(&u_gd, off, bytemuck::bytes_of(&$gdims));
                let bg = gemm_bg($pipe, $a, $w, $c, $coff, &u_gd);
                $cp.set_pipeline($pipe);
                $cp.set_bind_group(0, &bg, &[off as u32]);
                $cp.dispatch_workgroups($gx, $gy, $gz);
            }};
        }
        macro_rules! gemm {
            ($cp:expr, $pipe:expr, $a:expr, $w:expr, $c:expr, $m:expr, $n:expr, $k:expr,
             $ldc:expr, $bsa:expr, $bsb:expr, $bsc:expr, $gx:expr, $gy:expr, $gz:expr) => {{
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
                    },
                    $gx, $gy, $gz
                );
            }};
        }

        // per-call uniforms: multi-position extract + silu rows + softmax + repeat
        let silu_grid = self.write_silu_rows(s);
        // One workgroup per (head, position) score row.
        let softmax_grid = grid_xy(nqh * s);
        gpu.upload(
            &self.scratch.u_qkvx,
            bytemuck::bytes_of(&QkvxCfg {
                max_seq: self.cap as u32,
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
                    &u_sl,
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
                gpu.write_at(
                    &u_sl,
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
                &u_sl,
                (2 * MAX_SLAB * 256) as u64,
                bytemuck::bytes_of(&SlabWeightsCfg {
                    rows: (nqh * mp) as u32,
                    n_slab: n_slab as u32,
                    gx: w_grid.0,
                    _p: 0,
                }),
            );
            gpu.write_at(
                &u_sl,
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
                &u_sl,
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
        // `repeat_kv` writes rows `0..cur` only, but both attention GEMMs address
        // whole key slabs: the score GEMM reads K up to the slab end and the AV
        // GEMM reads V rows `cur..cur16`, whose weights the softmax has written as
        // exact zeros.  `0 · NaN` is NaN, so those rows have to be real zeros
        // rather than whatever the allocator last handed out.
        let tail_rows = nkv_rows - cur;
        if tail_rows > 0 {
            let zero = vec![0u8; tail_rows * hd * 2];
            for h in 0..nqh {
                let off = ((h * nkv_rows + cur) * hd * 2) as u64;
                gpu.write_at(&k_rep, off, &zero);
                gpu.write_at(&v_rep, off, &zero);
            }
        }
        let repeat_grid = grid_xy((nqh * cur * hd / 2).div_ceil(256));
        gpu.upload(
            &u_rk,
            bytemuck::bytes_of(&RepeatKvCfg {
                nkvh: nkvh as u32,
                max_seq: self.cap as u32,
                cur: cur as u32,
                hd: hd as u32,
                npw: (nkv_rows * hd / 2) as u32,
                gx: repeat_grid.0,
            }),
        );

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut cp = enc.begin_compute_pass(&Default::default());

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
                &mut cp, &self.pipes.gemm, &normed, &layer.qkv_w, &qkv,
                s, cfg.fused_qkv_cols(), hs, cfg.fused_qkv_cols(), 0, 0, 0,
                (cfg.fused_qkv_cols() / 128) as u32, (mp / 128) as u32, 1
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
                    wgpu::BindGroupEntry { binding: 6, resource: layer.k_cache.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: layer.v_cache.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: self.scratch.u_qkvx.as_entire_binding() },
                ],
            });
            cp.set_pipeline(&self.pipes.extract);
            cp.set_bind_group(0, &bg_ex, &[]);
            cp.dispatch_workgroups(s as u32, (nqh + nkvh) as u32, 1);
            if dup == "p_extract" {
                cp.dispatch_workgroups(s as u32, (nqh + nkvh) as u32, 1);
            }

            // 4. repeat_kv (K and V) — GQA head duplication
            for (cache, out) in [(&layer.k_cache, &k_rep), (&layer.v_cache, &v_rep)] {
                let bg_rk = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("p.rk"),
                    layout: &self.pipes.repeat_kv.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: cache.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: u_rk.as_entire_binding() },
                    ],
                });
                cp.set_pipeline(&self.pipes.repeat_kv);
                cp.set_bind_group(0, &bg_rk, &[]);
                cp.dispatch_workgroups(repeat_grid.0, repeat_grid.1, 1);
                // Ablation hook (see `encode_step`): re-dispatching is idempotent,
                // so the token stream is unchanged.
                if dup == "p_repeat" {
                    cp.dispatch_workgroups(repeat_grid.0, repeat_grid.1, 1);
                }
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
                for t in 0..n_slab {
                    let t0 = t * slab_t;
                    let tl = (cur16 - t0).min(slab_t); // 16-aligned columns in this slab
                    // score tile [mp, T] = q × K[t0..t0+T)ᵀ, tiles above the
                    // diagonal skipped (row0 shifts the diagonal to this slab)
                    let sg = GDims {
                        m: s as u32,
                        n: slab_t as u32,
                        k: hd as u32,
                        ldc: slab_t as u32,
                        bsa: (mp * hd / 2) as u32,
                        bsb: (nkv_rows * hd / 2) as u32,
                        bsc: (mp * slab_t) as u32,
                        beta: 0,
                        row0: t0 as u32,
                        lda: hd as u32,
                    };
                    let sgrid = ((slab_t / 128) as u32, (mp / 128) as u32, nqh as u32);
                    gemm_at!(
                        &mut cp, &self.pipes.gemm_causal, &q_out, &k_rep, &scores,
                        0u64, t, sg,
                        sgrid.0, sgrid.1, sgrid.2
                    );
                    // Ablation hook (as in the flat path): the GEMM overwrites its
                    // output, so re-dispatching leaves the token stream alone.
                    if dup == "p_scores" {
                        gemm_at!(
                            &mut cp, &self.pipes.gemm_causal, &q_out, &k_rep, &scores,
                            0u64, t, sg,
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
                                    buffer: &u_sl,
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
                                    buffer: &u_sl,
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
                    // diagonal inside the slab (row0 = t0)
                    let ag = GDims {
                        m: s as u32,
                        n: hd as u32,
                        k: tl as u32,
                        ldc: (nqh * hd) as u32,
                        bsa: (mp * slab_t / 2) as u32,
                        bsb: (nkv_rows * hd / 2) as u32,
                        bsc: hd as u32,
                        beta: 0,
                        row0: t0 as u32,
                        lda: slab_t as u32,
                    };
                    gemm_at!(
                        &mut cp, &self.pipes.gemm_av_causal, &attn, &v_rep, spart,
                        (t * mp * nqh * hd * 2) as u64, n_slab + t, ag,
                        1, (mp / 128) as u32, nqh as u32
                    );
                    if dup == "p_av" {
                        gemm_at!(
                            &mut cp, &self.pipes.gemm_av_causal, &attn, &v_rep, spart,
                            (t * mp * nqh * hd * 2) as u64, n_slab + t, ag,
                            1, (mp / 128) as u32, nqh as u32
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
                                buffer: &u_sl,
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
                                buffer: &u_sl,
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
                    &mut cp, &self.pipes.gemm_causal, &q_out, &k_rep, &scores,
                    s, cur, hd, np, mp * hd / 2, np * hd / 2, mp * np,
                    (np / 128) as u32, (mp / 128) as u32, nqh as u32
                );
                // Ablation hook: re-dispatching is idempotent (the GEMM overwrites
                // its output), so the token stream is unchanged.
                if dup == "p_scores" {
                    gemm!(
                        &mut cp, &self.pipes.gemm, &q_out, &k_rep, &scores,
                        s, cur, hd, np, mp * hd / 2, np * hd / 2, mp * np,
                        (np / 128) as u32, (mp / 128) as u32, nqh as u32
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
                                buffer: &u_sl,
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
                    &mut cp, &self.pipes.gemm_av_causal, &attn, &v_rep, &attn_flat,
                    s, hd, cur16, nqh * hd, mp * cur16 / 2, np * hd / 2, hd,
                    1, (mp / 128) as u32, nqh as u32
                );
                if dup == "p_av" {
                    gemm!(
                        &mut cp, &self.pipes.gemm_av, &attn, &v_rep, &attn_flat,
                        s, hd, cur16, nqh * hd, mp * cur16 / 2, np * hd / 2, hd,
                        1, (mp / 128) as u32, nqh as u32
                    );
                }
            }

            // 8. o projection + residual:  h += attn_flat × o_wᵀ
            gemm_or_skip!(
                "o",
                &mut cp, &self.pipes.gemm_acc, &attn_flat, &layer.o_w, &h_buf,
                s, hs, nqh * hd, hs, 0, 0, 0,
                (hs / 128) as u32, (mp / 128) as u32, 1
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
                &mut cp, &self.pipes.gemm, &norm2, &layer.gu_w, &gu,
                s, 2 * inter, hs, 2 * inter, 0, 0, 0,
                ((2 * inter) / 128) as u32, (mp / 128) as u32, 1
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
                &mut cp, &self.pipes.gemm_acc, &activated, &layer.dp_w, &h_buf,
                s, hs, inter, hs, 0, 0, 0,
                (hs / 128) as u32, (mp / 128) as u32, 1
            );

            // Measurement dump: `h` now holds layer `li`'s output.
            dump_layer!(cp, li + 1);

            // Long prefills (s>=512) must submit+poll or a hung submit hits the
            // WDDM TDR watchdog and device-loses.  The interval is in *layers* but
            // the driver times wall clock, so the slabbed path submits twice as
            // often.
            let submit_every = if s >= 4096 { 2 } else { 4 };
            if s >= 512 && (li + 1) % submit_every == 0 {
                drop(cp);
                gpu.queue.submit([enc.finish()]);
                if let Err(e) = gpu.device.poll(wgpu::PollType::wait_indefinitely()) {
                    anyhow::bail!("prefill: device lost after layer {li}: {e:?}");
                }
                enc = gpu.device.create_command_encoder(&Default::default());
                cp = enc.begin_compute_pass(&Default::default());
                // the submitted cfgs are retired, so the per-dispatch slots can
                // be handed out again (the slab slots are static, never reused)
                gd_slot = 2 * MAX_SLAB;
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
        self.debug_prefill_h = Some(h_buf);

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
