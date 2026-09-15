//! wgpu text decoder — a structural port of `src/cudarc_engine.rs`'s decode path.
//!
//! One decode step is **one command buffer, one submit** (~256 dispatches batched
//! together).  That discipline is not stylistic: measured on this machine,
//! batching costs 0.64 µs/dispatch while submitting per-op costs 32.8 µs — a 50×
//! difference that would add ~8 ms to every token.  See `wgpu/FEASIBILITY.md` §4.
//!
//! Operator order per step mirrors `GpuDecoderLayer::forward_decode` exactly:
//!
//! ```text
//! embed_lookup(token_buf[0], embed_table) -> h
//! for each of 28 layers:
//!   rms_norm(h, input_layernorm)          -> norm1
//!   gemv(norm1, qkv_w)                    -> qkv            (4096)
//!   qkv_extract(qkv)                      -> q_out, k_cache[i], v_cache[i]
//!   gqa_decode(q_out, k/v_cache[i])       -> attn_out       (2048)
//!   gemv(attn_out, o_w, accum)            -> h              (residual fused)
//!   rms_norm(h, post_attention_layernorm) -> norm2
//!   gemv(norm2, gate_up_w)                -> gate_up        (6144)
//!   silu_mul_split(gate_up)               -> activated      (3072)
//!   gemv(activated, down_proj_w, accum)   -> h
//! rms_norm(h, final_norm)                 -> final_norm
//! gemv(final_norm, embed_table)           -> logits         (151936)
//! argmax(logits)                          -> token_buf[0]
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};

use crate::gpu::{BulkUpload, Gpu};
use crate::shaders;
use crate::weights::{self, PackedWeight};

/// Text decoder hyper-parameters (mirrors `TextDecoderConfig`).
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
    /// Read text decoder dims from either the original `thinker_config` layout
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

/// `block_for_reduction` from `cudarc_engine.rs` — kept identical so the reduction
/// tree matches the CUDA kernels bit for bit.
fn block_for_reduction(last: usize) -> u32 {
    let mut bs: u32 = 32;
    let target = last as u32;
    while bs < target && bs < 1024 {
        bs *= 2;
    }
    bs.min(1024).max(32)
}

/// Split a flat workgroup count across two grid axes.  `max_compute_workgroups_
/// per_dimension` is 65535: a 15-minute clip's prefill needs 85k workgroups for
/// the SiLU and 194k rows for the causal softmax, and wgpu rejects the whole
/// command buffer rather than clamping.
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

/// Prefill GEMM dimensions — mirrors the bench kernel's uniform block.
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

/// Buffers reused by every layer (the CUDA `DecodeScratch`).
pub struct Scratch {
    pub h: wgpu::Buffer,
    pub norm1: wgpu::Buffer,
    pub qkv: wgpu::Buffer,
    pub q_out: wgpu::Buffer,
    pub attn_out: wgpu::Buffer,
    pub norm2: wgpu::Buffer,
    pub gate_up: wgpu::Buffer,
    pub activated: wgpu::Buffer,
    pub final_norm: wgpu::Buffer,
    pub logits: wgpu::Buffer,
    pub token: wgpu::Buffer,
    token_staging: wgpu::Buffer,
    pub cos: wgpu::Buffer,
    pub sin: wgpu::Buffer,
    u_rms: wgpu::Buffer,
    u_qkvx: wgpu::Buffer,
    u_gqa: wgpu::Buffer,
    u_gp1: wgpu::Buffer,
    u_gp2: wgpu::Buffer,
    u_silu: wgpu::Buffer,
    /// Split-K attention partials, `[nqh, max_chunks, …]` — f32, CUDA layout.
    pub split_part_out: wgpu::Buffer,
    pub split_part_max: wgpu::Buffer,
    pub split_part_sum: wgpu::Buffer,
}

/// Everything a single decoder layer needs, including its own KV cache slice.
struct Layer {
    k_cache: wgpu::Buffer,
    v_cache: wgpu::Buffer,
    /// weight buffers kept alive for prefill-time re-binding
    iln_w: wgpu::Buffer,
    pln_w: wgpu::Buffer,
    qn_w: wgpu::Buffer,
    kn_w: wgpu::Buffer,
    qkv_w: wgpu::Buffer,
    o_w: wgpu::Buffer,
    gu_w: wgpu::Buffer,
    dp_w: wgpu::Buffer,
    bg_gemv_qkv: wgpu::BindGroup,
    bg_gemv_o: wgpu::BindGroup,
    bg_gemv_gu: wgpu::BindGroup,
    bg_gemv_dp: wgpu::BindGroup,
    /// The two GEMVs whose input is a layer norm, with the norm folded in.
    bg_gemv_qkv_norm: wgpu::BindGroup,
    bg_gemv_gu_norm: wgpu::BindGroup,
    bg_rms1: wgpu::BindGroup,
    bg_rms2: wgpu::BindGroup,
    bg_extract: wgpu::BindGroup,
    bg_gqa: wgpu::BindGroup,
    bg_gqa_split: wgpu::BindGroup,
    bg_gqa_merge: wgpu::BindGroup,
}

struct Pipes {
    gemv_qkv: wgpu::ComputePipeline,
    gemv_o: wgpu::ComputePipeline,
    gemv_gu: wgpu::ComputePipeline,
    gemv_dp: wgpu::ComputePipeline,
    gemv_lm: wgpu::ComputePipeline,
    /// `gemv` + the RMSNorm that produces its input, for the three sites where
    /// the activation is a normed row (see [`shaders::gemv_norm`]).
    gemv_qkv_norm: wgpu::ComputePipeline,
    gemv_gu_norm: wgpu::ComputePipeline,
    gemv_lm_norm: wgpu::ComputePipeline,
    rms_norm: wgpu::ComputePipeline,
    extract: wgpu::ComputePipeline,
    gqa256: wgpu::ComputePipeline,
    gqa512: wgpu::ComputePipeline,
    gqa_split256: wgpu::ComputePipeline,
    gqa_split512: wgpu::ComputePipeline,
    gqa_merge: wgpu::ComputePipeline,
    silu: wgpu::ComputePipeline,
    argmax: wgpu::ComputePipeline,
    embed: wgpu::ComputePipeline,
    /// Prefill GEMMs (cuBLAS stand-ins): plain and residual-accumulating
    /// (transb=0), plus the attention AV form (transb=1, batched).
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
}

/// Capacity of the single-block attention path — CUDA's `SPLIT_THRESHOLD`.
pub const GQA_SINGLE_CAP: usize = 1024;

/// Key-slab width for the tiled causal prefill attention: the key dimension is
/// processed in windows of this many positions rather than materialised whole.
const SLAB_T: usize = 1024;

/// Reduction block size for the slabbed softmax and its statistics.
///
/// Both run one workgroup per score row over a `SLAB_T`-wide slab, and they
/// must agree on the row sum to the bit (the merge weights are that sum), so
/// this is shared.  It is deliberately smaller than `SLAB_T`: at 1024 threads a
/// thread owned a single column and the two barrier trees — not the arithmetic
/// or the traffic — dominated the pass, which measured as two thirds of a
/// 6-minute prefill's attention time.
const SLAB_BS: usize = 256;

/// Uniform slots reserved for the per-slab cfg blocks (softmax + stats), i.e.
/// the most key slabs a prefill may tile into.  Only the slabbed path uses
/// them, and it only runs past 4096 tokens (~5 minutes).
const MAX_SLAB: usize = 16;

/// Sequence length above which prefill tiles its attention.  Below it the flat
/// path runs — that is every fixture and every FLEURS clip, i.e. everything the
/// alignment gate measures.  `QASR_SLAB=on|off` forces one path for A/B work.
fn slab_path(s: usize) -> bool {
    match std::env::var("QASR_SLAB").unwrap_or_default().to_ascii_lowercase().as_str() {
        "1" | "on" | "yes" | "force" => true,
        "0" | "off" | "no" => false,
        _ => s > 4096,
    }
}

/// Split-K attention chunk size — mirrors CUDA's `fused_gqa_decode_split_into`
/// choice (`cur_len >= 2048` uses 512, else 256).  The P1 block size is always
/// 256 regardless of the chunk width.
fn gqa_split_chunk(cur_len: usize) -> usize {
    if cur_len >= 2048 {
        512
    } else {
        256
    }
}

pub struct WgpuTextDecoder {
    pub gpu: Gpu,
    pub cfg: TextConfig,
    pub max_seq: usize,
    /// Positions already in the KV cache — the next step attends at `cur_len = pos + 1`.
    pub pos: usize,
    /// Host-side time accumulated by [`Self::step`]: (uniform + encode + submit,
    /// token readback).  Purely diagnostic — what the batched-submission idea
    /// below would remove.
    pub host_submit_ms: f64,
    pub host_read_ms: f64,

    embed_table: wgpu::Buffer,
    layers: Vec<Layer>,
    pipes: Pipes,
    pub scratch: Scratch,

    bg_final_rms: wgpu::BindGroup,
    bg_gemv_lm: wgpu::BindGroup,
    /// Final norm folded into the LM head's GEMV (decode path).
    bg_gemv_lm_norm: wgpu::BindGroup,
    bg_silu: wgpu::BindGroup,
    bg_argmax: wgpu::BindGroup,
    bg_embed: wgpu::BindGroup,
    /// decoder-final norm weight, kept for prefill-time re-binding
    norm_buf: wgpu::Buffer,
    /// port-time scaffolding: the prefill output hidden states [s, hs] f16
    pub debug_prefill_h: Option<wgpu::Buffer>,
    /// port-time scaffolding: layer-0 row-0 hidden state after the layer
    pub debug_l0_h: Option<wgpu::Buffer>,
}

impl WgpuTextDecoder {
    /// Build the decoder and upload `{prefix}.*` weights.
    ///
    /// `max_seq` sizes the KV cache (`seq_len + max_new_tokens`, exactly as the
    /// CUDA backend does); `rope_positions` sizes the MRoPE tables.
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
        let q_dim = cfg.q_dim();
        let _kv_dim = cfg.kv_dim();
        let nqh = cfg.num_attention_heads;
        let nkvh = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let nl = cfg.num_hidden_layers;

        let build =
            |label: &str, src: &str, entry: &str, pl: Option<&wgpu::PipelineLayout>| -> Result<wgpu::ComputePipeline> {
                gpu.pipeline(label, src, entry, pl)
                    .with_context(|| format!("build pipeline {label}"))
            };

        // Single-block and split-K attention each come in two sibling variants
        // (workgroup 256/512, chunk 256/512) that *share* one bind group per
        // layer.  wgpu's implicit layouts are pipeline-exclusive, so each family
        // gets one explicit pipeline layout of its own.
        let gqa_pl = family_layout(
            &gpu,
            "gqa_single",
            &[(0, true), (1, true), (2, true), (3, false)],
            4,
        );
        let split_pl = family_layout(
            &gpu,
            "gqa_split",
            &[(0, true), (1, true), (2, true), (3, false), (4, false), (5, false)],
            6,
        );
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

        // ── pipelines ─────────────────────────────────────────────────────
        let rms_bs = block_for_reduction(hs) as usize;
        // `gemv`'s 32-lane xor butterfly can use `subgroupShuffleXor` instead of
        // 5 shared-memory rounds.  Both produce the SAME reduction tree, so the
        // results are bit-identical (A/B: 1.17-1.18x, 0 differing outputs on
        // 18992 rows).
        //
        // The capability bit alone is NOT enough: the butterfly folds lane xors
        // 16/8/4/2/1 and maps output rows with `warp = lid >> 5`, i.e. it assumes
        // exactly 32 lanes.  Intel's Vulkan driver reports SUBGROUP as a range
        // (8..32) and taking it produced a model's worth of garbage with no
        // error anywhere — the failure was visible only in the transcript.  Use
        // the shuffle path only when the adapter promises exactly 32 lanes.
        let subgroup = gpu.features.contains(wgpu::Features::SUBGROUP)
            && gpu.info.subgroup_min_size == 32
            && gpu.info.subgroup_max_size == 32;
        let pipes = Pipes {
            gemv_qkv: build("gemv_qkv", &shaders::gemv(cfg.fused_qkv_cols(), hs, false, subgroup), "gemv", None)?,
            gemv_o: build("gemv_o", &shaders::gemv(hs, q_dim, true, subgroup), "gemv", None)?,
            gemv_gu: build("gemv_gu", &shaders::gemv(2 * inter, hs, false, subgroup), "gemv", None)?,
            gemv_dp: build("gemv_dp", &shaders::gemv(hs, inter, true, subgroup), "gemv", None)?,
            gemv_lm: build("gemv_lm", &shaders::gemv(vocab, hs, false, subgroup), "gemv", None)?,
            // The three sites whose activation is a normed row: the norm is folded
            // into the GEMV prologue instead of being its own 1-workgroup dispatch.
            gemv_qkv_norm: build(
                "gemv_qkv_norm",
                &shaders::gemv_norm(cfg.fused_qkv_cols(), hs, false, subgroup, hs, rms_bs as usize, cfg.rms_norm_eps),
                "gemv",
                None,
            )?,
            gemv_gu_norm: build(
                "gemv_gu_norm",
                &shaders::gemv_norm(2 * inter, hs, false, subgroup, hs, rms_bs as usize, cfg.rms_norm_eps),
                "gemv",
                None,
            )?,
            gemv_lm_norm: build(
                "gemv_lm_norm",
                &shaders::gemv_norm(vocab, hs, false, subgroup, hs, rms_bs as usize, cfg.rms_norm_eps),
                "gemv",
                None,
            )?,
            rms_norm: build("rms_norm", &shaders::rms_norm(hs, rms_bs), "rms_norm", None)?,
            extract: build("qkv_extract", &shaders::qkv_extract(nqh, nkvh, hd), "qkv_extract", None)?,
            gqa256: build("gqa256", &shaders::gqa_decode_single(nqh, nkvh, hd, 256, GQA_SINGLE_CAP), "gqa", Some(&gqa_pl))?,
            gqa512: build("gqa512", &shaders::gqa_decode_single(nqh, nkvh, hd, 512, GQA_SINGLE_CAP), "gqa", Some(&gqa_pl))?,
            gqa_split256: build("gqa_split256", &shaders::gqa_decode_split_p1(nqh, nkvh, hd, 256), "gqa_split_p1", Some(&split_pl))?,
            gqa_split512: build("gqa_split512", &shaders::gqa_decode_split_p1(nqh, nkvh, hd, 512), "gqa_split_p1", Some(&split_pl))?,
            gqa_merge: build("gqa_merge", &shaders::gqa_split_merge(hd), "gqa_merge", None)?,
            silu: build("silu30", &shaders::silu_mul_split(inter), "silu_mul_split", None)?,
            argmax: build("argmax", &shaders::argmax_into_slot(), "argmax", None)?,
            embed: build("embed", &shaders::embed_lookup_single(), "embed", None)?,
            gemm: build("gemm", &shaders::prefill_gemm(false, false), "gemm", Some(&gemm_pl))?,
            gemm_acc: build("gemm_acc", &shaders::prefill_gemm(false, true), "gemm", Some(&gemm_pl))?,
            gemm_av: build("gemm_av", &shaders::prefill_gemm(true, false), "gemm", Some(&gemm_pl))?,
            gemm_causal: build("gemm_causal", &shaders::prefill_gemm_causal(), "gemm", Some(&gemm_pl))?,
            gemm_av_causal: build("gemm_av_causal", &shaders::prefill_gemm_causal_av(), "gemm", Some(&gemm_pl))?,
            softmax: std::collections::HashMap::from([
                (32, build("softmax32", &shaders::softmax_causal(32), "softmax", Some(&sm_pl))?),
                (64, build("softmax64", &shaders::softmax_causal(64), "softmax", Some(&sm_pl))?),
                (128, build("softmax128", &shaders::softmax_causal(128), "softmax", Some(&sm_pl))?),
                (256, build("softmax256", &shaders::softmax_causal(256), "softmax", Some(&sm_pl))?),
                (512, build("softmax512", &shaders::softmax_causal(512), "softmax", Some(&sm_pl))?),
                (1024, build("softmax1024", &shaders::softmax_causal(1024), "softmax", Some(&sm_pl))?),
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
        };

        // ── scratch ───────────────────────────────────────────────────────
        // Split partials are sized for the smallest chunk (256) — CUDA's
        // conservative max_chunks; the merge kernel only ever reads the
        // `n_chunks` slots P1 actually wrote for the launched chunk width.
        let max_chunks = max_seq.div_ceil(256);
        let mut up = gpu.uploader();
        let scratch = Scratch {
            h: up.storage("h", (hs / 2 * 4) as u64),
            norm1: up.storage("norm1", (hs / 2 * 4) as u64),
            qkv: up.storage("qkv", (cfg.fused_qkv_cols() / 2 * 4) as u64),
            q_out: up.storage("q_out", (q_dim / 2 * 4) as u64),
            attn_out: up.storage("attn_out", (q_dim / 2 * 4) as u64),
            norm2: up.storage("norm2", (hs / 2 * 4) as u64),
            gate_up: up.storage("gate_up", (inter * 4) as u64),
            activated: up.storage("activated", (inter / 2 * 4) as u64),
            final_norm: up.storage("final_norm", (hs / 2 * 4) as u64),
            logits: up.storage("logits", (vocab / 2 * 4) as u64),
            token: up.storage("token", 4),
            token_staging: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("token_staging"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cos: up.storage("cos", (rope_positions * hd / 2 * 4) as u64),
            sin: up.storage("sin", (rope_positions * hd / 2 * 4) as u64),
            u_rms: up.uniform("u_rms", 16),
            u_qkvx: up.uniform("u_qkvx", 32),
            u_silu: up.uniform("u_silu", 16),
            u_gqa: up.uniform("u_gqa", 16),
            u_gp1: up.uniform("u_gp1", 16),
            u_gp2: up.uniform("u_gp2", 16),
            split_part_out: up.storage("split_part_out", (nqh * max_chunks * hd * 4) as u64),
            split_part_max: up.storage("split_part_max", (nqh * max_chunks * 4) as u64),
            split_part_sum: up.storage("split_part_sum", (nqh * max_chunks * 4) as u64),
        };
        up.upload(
            &scratch.u_rms,
            bytemuck::bytes_of(&RmsCfg { eps: cfg.rms_norm_eps, _a: 0.0, _b: 0.0, _c: 0.0 }),
        )?;

        // ── global weights ────────────────────────────────────────────────
        let embed = weights::get_matrix(&w, &format!("{prefix}.embed_tokens.weight"))?;
        if embed.rows != vocab || embed.cols != hs {
            bail!(
                "embed_tokens is [{}x{}], expected [{vocab}x{hs}]",
                embed.rows,
                embed.cols
            );
        }
        let embed_table = upload_weight(&mut up, "embed_tokens", &embed)?;
        let norm_buf = upload_vec(
            &mut up,
            "final_norm_w",
            &weights::get_vector(&w, &format!("{prefix}.norm.weight"))?,
        )?;

        // ── per-layer ─────────────────────────────────────────────────────
        let kv_words = nkvh * max_seq * hd / 2;
        let u_argmax = up.uniform("u_argmax", 16);
        let u_embed = up.uniform("u_embed", 16);
        up.upload(
            &u_argmax,
            bytemuck::bytes_of(&ArgmaxCfg { n: vocab as u32, slot: 0, _a: 0, _b: 0 }),
        )?;
        up.upload(
            &u_embed,
            bytemuck::bytes_of(&EmbedCfg { slot: 0, d2: (hs / 2) as u32, _a: 0, _b: 0 }),
        )?;

        let gemv_bg = |pipe: &wgpu::ComputePipeline,
                       wt: &wgpu::Buffer,
                       x: &wgpu::Buffer,
                       y: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                ],
            })
        };
        // `gemv_norm`: weights, raw activation, output, and the layer-norm weight
        // the prologue applies.
        let gemv_norm_bg = |pipe: &wgpu::ComputePipeline,
                            wt: &wgpu::Buffer,
                            x: &wgpu::Buffer,
                            y: &wgpu::Buffer,
                            nw: &wgpu::Buffer|
         -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemv_norm"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: nw.as_entire_binding() },
                ],
            })
        };
        let rms_bg = |x: &wgpu::Buffer, wt: &wgpu::Buffer, out: &wgpu::Buffer| -> wgpu::BindGroup {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rms"),
                layout: &pipes.rms_norm.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: scratch.u_rms.as_entire_binding() },
                ],
            })
        };

        let mut layers = Vec::with_capacity(nl);
        for i in 0..nl {
            let p = format!("{prefix}.layers.{i}");
            let qkv = upload_weight(&mut up, "qkv_w", &fused_qkv(&w, &format!("{p}.self_attn"))?)?;
            let o = upload_weight(
                &mut up,
                "o_w",
                &weights::get_matrix(&w, &format!("{p}.self_attn.o_proj.weight"))?,
            )?;
            let gu = upload_weight(&mut up, "gu_w", &fused_gate_up(&w, &format!("{p}.mlp"))?)?;
            let dp = upload_weight(
                &mut up,
                "dp_w",
                &weights::get_matrix(&w, &format!("{p}.mlp.down_proj.weight"))?,
            )?;
            let iln = upload_vec(&mut up, "iln_w", &weights::get_vector(&w, &format!("{p}.input_layernorm.weight"))?)?;
            let pln = upload_vec(&mut up, "pln_w", &weights::get_vector(&w, &format!("{p}.post_attention_layernorm.weight"))?)?;
            let qn = upload_vec(&mut up, "qn_w", &weights::get_vector(&w, &format!("{p}.self_attn.q_norm.weight"))?)?;
            let kn = upload_vec(&mut up, "kn_w", &weights::get_vector(&w, &format!("{p}.self_attn.k_norm.weight"))?)?;

            let k_cache = gpu.storage("k_cache", (kv_words * 4) as u64);
            let v_cache = gpu.storage("v_cache", (kv_words * 4) as u64);

            let bg_gemv_qkv = gemv_bg(&pipes.gemv_qkv, &qkv, &scratch.norm1, &scratch.qkv);
            let bg_gemv_qkv_norm = gemv_norm_bg(&pipes.gemv_qkv_norm, &qkv, &scratch.h, &scratch.qkv, &iln);
            let bg_gemv_o = gemv_bg(&pipes.gemv_o, &o, &scratch.attn_out, &scratch.h);
            let bg_gemv_gu = gemv_bg(&pipes.gemv_gu, &gu, &scratch.norm2, &scratch.gate_up);
            let bg_gemv_gu_norm = gemv_norm_bg(&pipes.gemv_gu_norm, &gu, &scratch.h, &scratch.gate_up, &pln);
            let bg_gemv_dp = gemv_bg(&pipes.gemv_dp, &dp, &scratch.activated, &scratch.h);
            let bg_rms1 = rms_bg(&scratch.h, &iln, &scratch.norm1);
            let bg_rms2 = rms_bg(&scratch.h, &pln, &scratch.norm2);

            let bg_extract = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("qkv_extract"),
                layout: &pipes.extract.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: scratch.qkv.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: qn.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: kn.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: scratch.cos.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: scratch.sin.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: scratch.q_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: k_cache.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: v_cache.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: scratch.u_qkvx.as_entire_binding() },
                ],
            });
            let bg_gqa = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gqa"),
                layout: &pipes.gqa256.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: scratch.q_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: scratch.attn_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: scratch.u_gqa.as_entire_binding() },
                ],
            });
            // Split-K: both chunk-width pipelines share one binding shape, so the
            // group is built from the 256-wide layout and used for both (wgpu
            // dedupes equal layouts — the same trick as the single-path gqa256/512).
            let bg_gqa_split = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gqa_split"),
                layout: &pipes.gqa_split256.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: scratch.q_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: k_cache.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: v_cache.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: scratch.split_part_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: scratch.split_part_max.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: scratch.split_part_sum.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: scratch.u_gp1.as_entire_binding() },
                ],
            });
            let bg_gqa_merge = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gqa_merge"),
                layout: &pipes.gqa_merge.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: scratch.attn_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: scratch.split_part_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: scratch.split_part_max.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: scratch.split_part_sum.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: scratch.u_gp2.as_entire_binding() },
                ],
            });

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
                bg_gemv_qkv,
                bg_gemv_qkv_norm,
                bg_gemv_o,
                bg_gemv_gu,
                bg_gemv_gu_norm,
                bg_gemv_dp,
                bg_rms1,
                bg_rms2,
                bg_extract,
                bg_gqa,
                bg_gqa_split,
                bg_gqa_merge,
            });
        }

        let bg_final_rms = rms_bg(&scratch.h, &norm_buf, &scratch.final_norm);
        let bg_gemv_lm = gemv_bg(&pipes.gemv_lm, &embed_table, &scratch.final_norm, &scratch.logits);
        let bg_gemv_lm_norm = gemv_norm_bg(&pipes.gemv_lm_norm, &embed_table, &scratch.h, &scratch.logits, &norm_buf);
        let bg_silu = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("silu"),
            layout: &pipes.silu.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scratch.gate_up.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.activated.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scratch.u_silu.as_entire_binding() },
            ],
        });
        let bg_argmax = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("argmax"),
            layout: &pipes.argmax.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scratch.logits.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.token.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: u_argmax.as_entire_binding() },
            ],
        });
        let bg_embed = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embed"),
            layout: &pipes.embed.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: embed_table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.token.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scratch.h.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: u_embed.as_entire_binding() },
            ],
        });

        up.finish()?;

        Ok(Self {
            gpu,
            cfg,
            max_seq,
            pos: 0,
            host_submit_ms: 0.0,
            host_read_ms: 0.0,
            embed_table,
            layers,
            pipes,
            scratch,
            bg_final_rms,
            bg_gemv_lm,
            bg_gemv_lm_norm,
            bg_silu,
            bg_argmax,
            bg_embed,
            norm_buf,
            debug_prefill_h: None,
            debug_l0_h: None,
        })
    }

    /// Dispatches a single step issues at `cur_len` (the split-attention path
    /// adds one merge dispatch per layer).
    pub fn dispatches_per_step(&self, cur_len: usize) -> usize {
        let per_layer = if cur_len > GQA_SINGLE_CAP { 10 } else { 9 };
        1 + self.cfg.num_hidden_layers * per_layer + 3
    }

    // ── KV cache / RoPE plumbing ─────────────────────────────────────────

    /// The device the decoder lives on — shared with the GPU audio tower.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn k_cache(&self, layer: usize) -> &wgpu::Buffer {
        &self.layers[layer].k_cache
    }
    pub fn v_cache(&self, layer: usize) -> &wgpu::Buffer {
        &self.layers[layer].v_cache
    }

    /// Upload MRoPE tables — `[rope_positions, head_dim]` f16, row-major.
    pub fn set_rope_tables(&self, cos: &[half::f16], sin: &[half::f16]) {
        self.gpu.upload(&self.scratch.cos, &weights::words_bytes(cos));
        self.gpu.upload(&self.scratch.sin, &weights::words_bytes(sin));
    }

    /// Seed the token that the next `step()` will embed.
    pub fn set_input_token(&self, t: i32) {
        self.gpu.upload(&self.scratch.token, &t.to_le_bytes());
    }

    // ── the step ─────────────────────────────────────────────────────────

    fn write_step_uniforms(&self, pos: usize) {
        let cur_len = pos + 1;
        self.write_silu_rows(1);
        self.gpu.upload(
            &self.scratch.u_qkvx,
            bytemuck::bytes_of(&QkvxCfg {
                max_seq: self.max_seq as u32,
                start: pos as u32,
                pos_offset: pos as u32,
                s: 1,
                eps: self.cfg.rms_norm_eps,
                _a: 0,
                _b: 0,
                _c: 0,
            }),
        );
        self.gpu.upload(
            &self.scratch.u_gqa,
            bytemuck::bytes_of(&GqaCfg {
                cur_len: cur_len as u32,
                max_seq: self.max_seq as u32,
                scale: self.cfg.scale(),
                _p: 0.0,
            }),
        );
        let chunk = gqa_split_chunk(cur_len) as u32;
        let n_chunks = (cur_len as u32).div_ceil(chunk);
        self.gpu.upload(
            &self.scratch.u_gp1,
            bytemuck::bytes_of(&SplitCfg {
                cur_len: cur_len as u32,
                max_seq: self.max_seq as u32,
                scale: self.cfg.scale(),
                n_chunks,
            }),
        );
        self.gpu.upload(
            &self.scratch.u_gp2,
            bytemuck::bytes_of(&MergeCfg { n_chunks, _a: 0, _b: 0, _c: 0 }),
        );
    }

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

    pub fn encode_step(&self, enc: &mut wgpu::CommandEncoder, pos: usize) {
        let cfg = &self.cfg;
        let cur_len = pos + 1;
        let gemv_grid = |rows: usize| (rows / 8) as u32;
        // One row of `[gate|up]`: the grid is tiny here, the helper just keeps
        // the uniform's `gx` consistent with what the prefill writes.
        let silu_grid = grid_xy((cfg.intermediate_size / 2).div_ceil(256));

        // Ablation hook for RTFx work: `QASR_DUP=<op>` dispatches that op twice.
        // Every one of these ops is a pure function of its inputs writing the same
        // buffer, so the second dispatch recomputes the same values: the token
        // stream is unchanged and `t(dup) - t(base)` is that op's true cost —
        // including its launch bubble, which op-by-op profiling cannot separate.
        // Default: unset, zero effect.
        let dup = std::env::var("QASR_DUP").unwrap_or_default();
        let d = |name: &str| dup == name;

        let mut cp = enc.begin_compute_pass(&Default::default());

        cp.set_pipeline(&self.pipes.embed);
        cp.set_bind_group(0, &self.bg_embed, &[]);
        cp.dispatch_workgroups(1, 1, 1);

        for l in &self.layers {
            // No standalone `rms_norm` here: `gemv_qkv_norm` / `gemv_gu_norm` carry
            // the norm in their prologue (bit-identical tree, see `shaders::gemv_norm`).
            cp.set_pipeline(&self.pipes.gemv_qkv_norm);
            cp.set_bind_group(0, &l.bg_gemv_qkv_norm, &[]);
            cp.dispatch_workgroups(gemv_grid(cfg.fused_qkv_cols()), 1, 1);
            if d("qkv") {
                cp.dispatch_workgroups(gemv_grid(cfg.fused_qkv_cols()), 1, 1);
            }

            cp.set_pipeline(&self.pipes.extract);
            cp.set_bind_group(0, &l.bg_extract, &[]);
            cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1);
            if d("extract") {
                cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1);
            }

            self.encode_gqa(&mut cp, l, cur_len);
            if d("gqa") {
                self.encode_gqa(&mut cp, l, cur_len);
            }

            cp.set_pipeline(&self.pipes.gemv_o);
            cp.set_bind_group(0, &l.bg_gemv_o, &[]);
            cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            if d("o") {
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            }

            cp.set_pipeline(&self.pipes.gemv_gu_norm);
            cp.set_bind_group(0, &l.bg_gemv_gu_norm, &[]);
            cp.dispatch_workgroups(gemv_grid(2 * cfg.intermediate_size), 1, 1);
            if d("gu") {
                cp.dispatch_workgroups(gemv_grid(2 * cfg.intermediate_size), 1, 1);
            }

            cp.set_pipeline(&self.pipes.silu);
            cp.set_bind_group(0, &self.bg_silu, &[]);
            cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
            if d("silu") {
                cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
            }

            cp.set_pipeline(&self.pipes.gemv_dp);
            cp.set_bind_group(0, &l.bg_gemv_dp, &[]);
            cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            if d("dp") {
                cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
            }
        }

        cp.set_pipeline(&self.pipes.gemv_lm_norm);
        cp.set_bind_group(0, &self.bg_gemv_lm_norm, &[]);
        cp.dispatch_workgroups(gemv_grid(cfg.vocab_size), 1, 1);
        if d("lm") {
            cp.dispatch_workgroups(gemv_grid(cfg.vocab_size), 1, 1);
        }

        cp.set_pipeline(&self.pipes.argmax);
        cp.set_bind_group(0, &self.bg_argmax, &[]);
        cp.dispatch_workgroups(1, 1, 1);

        drop(cp);
        // D2H staging for the token — 4 bytes, one copy, no extra submit.
        enc.copy_buffer_to_buffer(&self.scratch.token, 0, &self.scratch.token_staging, 0, 4);
    }

    /// Attention dispatch for one layer: single-block kernel up to
    /// `GQA_SINGLE_CAP`, CUDA's split-K pair beyond (chunk 256/512 per
    /// `gqa_split_chunk`).
    fn encode_gqa<'pass>(&self, cp: &mut wgpu::ComputePass<'pass>, l: &Layer, cur_len: usize) {
        if cur_len <= GQA_SINGLE_CAP {
            let gqa = if cur_len > 512 { &self.pipes.gqa512 } else { &self.pipes.gqa256 };
            cp.set_pipeline(gqa);
            cp.set_bind_group(0, &l.bg_gqa, &[]);
            cp.dispatch_workgroups(self.cfg.num_attention_heads as u32, 1, 1);
            return;
        }
        let chunk = gqa_split_chunk(cur_len) as u32;
        let n_chunks = (cur_len as u32).div_ceil(chunk);
        let split = if chunk == 512 { &self.pipes.gqa_split512 } else { &self.pipes.gqa_split256 };
        let dup = std::env::var("QASR_DUP").unwrap_or_default();
        cp.set_pipeline(split);
        cp.set_bind_group(0, &l.bg_gqa_split, &[]);
        cp.dispatch_workgroups(self.cfg.num_attention_heads as u32, n_chunks, 1);
        if dup == "gqa_p1" {
            cp.dispatch_workgroups(self.cfg.num_attention_heads as u32, n_chunks, 1);
        }
        cp.set_pipeline(&self.pipes.gqa_merge);
        cp.set_bind_group(0, &l.bg_gqa_merge, &[]);
        cp.dispatch_workgroups(self.cfg.num_attention_heads as u32, 1, 1);
        if dup == "gqa_merge" {
            cp.dispatch_workgroups(self.cfg.num_attention_heads as u32, 1, 1);
        }
    }

    /// One decode step at `self.pos`, then `pos += 1`.  Returns the token argmax
    /// wrote into `token_buf[0]` (which the *next* step will embed).
    pub fn step(&mut self) -> Result<i32> {
        let pos = self.pos;
        let t_host = std::time::Instant::now();
        self.write_step_uniforms(pos);
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        self.encode_step(&mut enc, pos);
        self.gpu.queue.submit([enc.finish()]);
        self.host_submit_ms += t_host.elapsed().as_secs_f64() * 1000.0;
        let t_read = std::time::Instant::now();
        let v = self.read_token()?;
        self.host_read_ms += t_read.elapsed().as_secs_f64() * 1000.0;
        self.pos += 1;
        Ok(v)
    }

    // ── port-time debug scaffolding ──────────────────────────────────────
    // Used by `src/bin/probe_layers.rs` to run one decode step layer-by-layer
    // with host readbacks in between, localising the first divergent op.

    /// Dispatch only the embedding lookup.
    pub fn debug_encode_embed(&self, enc: &mut wgpu::CommandEncoder) {
        let mut cp = enc.begin_compute_pass(&Default::default());
        self.debug_dispatch_embed(&mut cp);
    }

    /// Embed dispatch into an already-open compute pass.
    pub fn debug_dispatch_embed(&self, cp: &mut wgpu::ComputePass) {
        cp.set_pipeline(&self.pipes.embed);
        cp.set_bind_group(0, &self.bg_embed, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }

    /// Per-step uniforms for `pos` (probe scaffolding).
    pub fn debug_write_step_uniforms(&self, pos: usize) {
        self.write_step_uniforms(pos);
    }

    /// Dispatch only layer `l`'s ops.  `h` must already hold the incoming
    /// residual; per-step uniforms are written for `pos`.
    pub fn debug_encode_layer(&self, enc: &mut wgpu::CommandEncoder, l: usize, pos: usize) {
        self.write_step_uniforms(pos);
        let cfg = &self.cfg;
        let cur_len = pos + 1;
        let gemv_grid = |rows: usize| (rows / 8) as u32;
        let silu_grid = grid_xy((cfg.intermediate_size / 2).div_ceil(256));
        let layer = &self.layers[l];
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&self.pipes.rms_norm);
        cp.set_bind_group(0, &layer.bg_rms1, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.pipes.gemv_qkv);
        cp.set_bind_group(0, &layer.bg_gemv_qkv, &[]);
        cp.dispatch_workgroups(gemv_grid(cfg.fused_qkv_cols()), 1, 1);
        cp.set_pipeline(&self.pipes.extract);
        cp.set_bind_group(0, &layer.bg_extract, &[]);
        cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1);
        self.encode_gqa(&mut cp, layer, cur_len);
        cp.set_pipeline(&self.pipes.gemv_o);
        cp.set_bind_group(0, &layer.bg_gemv_o, &[]);
        cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
        cp.set_pipeline(&self.pipes.rms_norm);
        cp.set_bind_group(0, &layer.bg_rms2, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.pipes.gemv_gu);
        cp.set_bind_group(0, &layer.bg_gemv_gu, &[]);
        cp.dispatch_workgroups(gemv_grid(2 * cfg.intermediate_size), 1, 1);
        cp.set_pipeline(&self.pipes.silu);
        cp.set_bind_group(0, &self.bg_silu, &[]);
        cp.dispatch_workgroups(silu_grid.0, silu_grid.1, 1);
        cp.set_pipeline(&self.pipes.gemv_dp);
        cp.set_bind_group(0, &layer.bg_gemv_dp, &[]);
        cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1);
    }

    /// Dispatch the decoder tail: final norm + lm_head + argmax.
    pub fn debug_encode_tail(&self, enc: &mut wgpu::CommandEncoder) {
        let mut cp = enc.begin_compute_pass(&Default::default());
        self.debug_dispatch_tail(&mut cp);
    }

    /// Tail dispatches into an already-open compute pass.
    pub fn debug_dispatch_tail(&self, cp: &mut wgpu::ComputePass) {
        cp.set_pipeline(&self.pipes.rms_norm);
        cp.set_bind_group(0, &self.bg_final_rms, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.pipes.gemv_lm);
        cp.set_bind_group(0, &self.bg_gemv_lm, &[]);
        cp.dispatch_workgroups((self.cfg.vocab_size / 8) as u32, 1, 1);
        cp.set_pipeline(&self.pipes.argmax);
        cp.set_bind_group(0, &self.bg_argmax, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }

    /// Read back the first bytes of the embed table (probe scaffolding).
    pub fn debug_embed_table_bytes(&self, n: u64) -> Result<Vec<u8>> {
        self.gpu.readback(&self.embed_table, n)
    }

    /// Dispatch a single one of layer `l`'s nine ops (probe scaffolding):
    /// 0=rms1, 1=gemv_qkv, 2=extract, 3=gqa, 4=gemv_o, 5=rms2, 6=gemv_gu,
    /// 7=silu, 8=gemv_dp.  Per-step uniforms are written for `pos`.
    pub fn debug_encode_op(&self, enc: &mut wgpu::CommandEncoder, l: usize, pos: usize, op: usize) {
        self.write_step_uniforms(pos);
        let mut cp = enc.begin_compute_pass(&Default::default());
        self.debug_dispatch_op(&mut cp, l, pos, op);
    }

    /// Single layer-op dispatch into an already-open compute pass.
    pub fn debug_dispatch_op(&self, cp: &mut wgpu::ComputePass, l: usize, pos: usize, op: usize) {
        let cfg = &self.cfg;
        let gemv_grid = |rows: usize| (rows / 8) as u32;
        let layer = &self.layers[l];
        match op {
            0 => { cp.set_pipeline(&self.pipes.rms_norm); cp.set_bind_group(0, &layer.bg_rms1, &[]); cp.dispatch_workgroups(1, 1, 1); }
            1 => { cp.set_pipeline(&self.pipes.gemv_qkv); cp.set_bind_group(0, &layer.bg_gemv_qkv, &[]); cp.dispatch_workgroups(gemv_grid(cfg.fused_qkv_cols()), 1, 1); }
            2 => { cp.set_pipeline(&self.pipes.extract); cp.set_bind_group(0, &layer.bg_extract, &[]); cp.dispatch_workgroups(1, (cfg.num_attention_heads + cfg.num_key_value_heads) as u32, 1); }
            3 => self.encode_gqa(cp, layer, pos + 1),
            4 => { cp.set_pipeline(&self.pipes.gemv_o); cp.set_bind_group(0, &layer.bg_gemv_o, &[]); cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1); }
            5 => { cp.set_pipeline(&self.pipes.rms_norm); cp.set_bind_group(0, &layer.bg_rms2, &[]); cp.dispatch_workgroups(1, 1, 1); }
            6 => { cp.set_pipeline(&self.pipes.gemv_gu); cp.set_bind_group(0, &layer.bg_gemv_gu, &[]); cp.dispatch_workgroups(gemv_grid(2 * cfg.intermediate_size), 1, 1); }
            7 => { cp.set_pipeline(&self.pipes.silu); cp.set_bind_group(0, &self.bg_silu, &[]); cp.dispatch_workgroups(((cfg.intermediate_size / 2).div_ceil(256)) as u32, 1, 1); }
            8 => { cp.set_pipeline(&self.pipes.gemv_dp); cp.set_bind_group(0, &layer.bg_gemv_dp, &[]); cp.dispatch_workgroups(gemv_grid(cfg.hidden_size), 1, 1); }
            _ => panic!("debug_encode_op: op {op} out of range"),
        }
    }

    fn read_token(&self) -> Result<i32> {
        let slice = self.scratch.token_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll token")?;
        rx.recv().context("map callback dropped")?.context("map token")?;
        let v = {
            let d = slice.get_mapped_range()?;
            i32::from_le_bytes([d[0], d[1], d[2], d[3]])
        };
        // `BufferSlice` is `Copy`; the map is released by `unmap`, not by a drop.
        self.scratch.token_staging.unmap();
        Ok(v)
    }

    /// Read any scratch buffer back as f16 values.
    pub fn read_f16(&self, buf: &wgpu::Buffer, elems: usize) -> Result<Vec<half::f16>> {
        let bytes = self.gpu.readback(buf, (elems * 2) as u64)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
            .collect())
    }

    /// Timed decode without per-step host synchronisation: the token stays on the
    /// GPU (embed reads `token_buf`), so this measures pure GPU-bound ms/token.
    pub fn bench_steps(&mut self, steps: usize, warmup: usize) -> Result<f64> {
        for _ in 0..warmup {
            let pos = self.pos;
            self.write_step_uniforms(pos);
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            self.encode_step(&mut enc, pos);
            self.gpu.queue.submit([enc.finish()]);
            self.pos += 1;
        }
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("warmup poll")?;

        let t0 = Instant::now();
        for _ in 0..steps {
            let pos = self.pos;
            self.write_step_uniforms(pos);
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            self.encode_step(&mut enc, pos);
            self.gpu.queue.submit([enc.finish()]);
            self.pos += 1;
        }
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("bench poll")?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        Ok(ms)
    }

    /// Bytes of weight traffic per decode step (for the bandwidth figure).
    pub fn step_weight_bytes(&self) -> u64 {
        let c = &self.cfg;
        let per_layer = (c.fused_qkv_cols() * c.hidden_size
            + c.hidden_size * c.q_dim()
            + 2 * c.intermediate_size * c.hidden_size
            + c.hidden_size * c.intermediate_size) as u64
            * 2;
        per_layer * c.num_hidden_layers as u64 + (c.vocab_size * c.hidden_size) as u64 * 2
    }

    pub fn embed_table(&self) -> &wgpu::Buffer {
        &self.embed_table
    }
}

/// Explicit pipeline layout for a kernel family whose sibling variants share
/// bind groups: `storage` entries (binding, read_only) plus one uniform.
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

/// f16 vector -> `array<u32>` binding.
fn upload_vec(up: &mut BulkUpload, label: &str, v: &[half::f16]) -> Result<wgpu::Buffer> {
    let b = up.storage(label, (v.len() * 2) as u64);
    up.upload(&b, &weights::words_bytes(v))?;
    Ok(b)
}

fn fused_qkv(w: &HashMap<String, weights::RawTensor>, prefix: &str) -> Result<PackedWeight> {
    let q = weights::get_matrix(w, &format!("{prefix}.q_proj.weight"))?;
    let k = weights::get_matrix(w, &format!("{prefix}.k_proj.weight"))?;
    let v = weights::get_matrix(w, &format!("{prefix}.v_proj.weight"))?;
    let cols = q.cols;
    PackedWeight::concat_rows(&[q, k, v], cols)
}

fn fused_gate_up(w: &HashMap<String, weights::RawTensor>, prefix: &str) -> Result<PackedWeight> {
    let g = weights::get_matrix(w, &format!("{prefix}.gate_proj.weight"))?;
    let u = weights::get_matrix(w, &format!("{prefix}.up_proj.weight"))?;
    let cols = g.cols;
    PackedWeight::concat_rows(&[g, u], cols)
}

// ═══════════════════════════════════════════════════════════════════════
//  Prefill (stage 2) — the cuBLAS stand-in chain
// ═══════════════════════════════════════════════════════════════════════

impl WgpuTextDecoder {
    /// Prefill: run `s` input positions (hidden states `[s, hs]` f16
    /// little-endian words) through every layer with causal attention, writing
    /// KV slots `kv_start..kv_start+s`, then final-norm + lm-head the last
    /// position.  Returns the argmax token (the first decode token) and leaves
    /// `self.pos` at `kv_start + s` so `step()` continues the sequence.
    ///
    /// This is the wgpu stand-in for the CUDA engine's cuBLAS-based prefill:
    /// f32 accumulate / f16 round like cuBLAS Hgemm, but with our own
    /// accumulation order — the KV cache and logits are validated against the
    /// CUDA golden at f16 tolerance, not bit-exactly.
    #[allow(clippy::too_many_lines)]
    pub fn prefill(&mut self, hidden_words: &[u8], s: usize, kv_start: usize) -> Result<i32> {
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

        // Slabbed causal attention: tile the *key*
        // dimension so the scratch is `[nqh, mp, T]` instead of `[nqh, mp, cur]`.
        // The flat path's scratch is O(s²) — 4.7 GiB per matrix at 15 minutes,
        // past both VRAM and the per-binding limit — and wgpu's failure mode for
        // that is garbage rather than an error, so the flat path refuses
        // explicitly.  Gated well above the 180 s fixtures: everything the
        // alignment gate covers stays on the path it was verified on.
        let slab_t = SLAB_T;
        let n_slab = if slab_path(s) { cur.div_ceil(slab_t) } else { 0 };
        let slabbed = n_slab > 0;
        // k_rep/v_rep carry whole key slabs on the slabbed path, so their rows
        // cover `n_slab · T` (a little past `cur`); the tail is never *read*
        // meaningfully — the softmax zeroes the columns past the causal bound
        // before the AV GEMM sees them.
        let nkv_rows = if slabbed { n_slab * slab_t } else { np };
        let bind_limit = self.gpu.limits.max_storage_buffer_binding_size as u64;
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
        up.upload(&h_buf, hidden_words)?;
        up.finish()?;

        let gpu = &self.gpu;
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
                max_seq: self.max_seq as u32,
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
        // rather than whatever the allocator last handed out — the slabbed path
        // reads a much longer tail than the flat one (a whole 1024-wide slab
        // against a 128-wide pad), which is what makes it worth pinning down.
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
                max_seq: self.max_seq as u32,
                cur: cur as u32,
                hd: hd as u32,
                npw: (nkv_rows * hd / 2) as u32,
                gx: repeat_grid.0,
            }),
        );

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut cp = enc.begin_compute_pass(&Default::default());

        // Ablation hook for prefill RTFx work (see `encode_step` for the scheme).
        let dup = std::env::var("QASR_DUP").unwrap_or_default();

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
            gemm!(
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
                // so the token stream is unchanged and the time delta is the cost.
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
                // `exp(m_t − M)·Σexp_t` — the exact per-slab softmax masses,
                // which is what makes the tiling a reassociation rather than an
                // approximation.
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
                gemm!(
                    &mut cp, &self.pipes.gemm_causal, &q_out, &k_rep, &scores,
                    s, cur, hd, np, mp * hd / 2, np * hd / 2, mp * np,
                    (np / 128) as u32, (mp / 128) as u32, nqh as u32
                );
                // Ablation hook: re-dispatching is idempotent (the GEMM overwrites its
                // output), so the token stream is unchanged and the delta is the cost.
                if dup == "p_scores" {
                    gemm!(
                        &mut cp, &self.pipes.gemm, &q_out, &k_rep, &scores,
                        s, cur, hd, np, mp * hd / 2, np * hd / 2, mp * np,
                        (np / 128) as u32, (mp / 128) as u32, nqh as u32
                    );
                }

                // 6. causal softmax, in place on scores
                let bs = block_for_reduction(cur) as usize;
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
                cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);
                if dup == "p_softmax" {
                    cp.dispatch_workgroups(softmax_grid.0, softmax_grid.1, 1);
                }

                // 7. AV GEMM, batched: attn_flat[s, nqh*hd] = attn × V  (V is [cur, hd])
                gemm!(
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
            gemm!(
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
            gemm!(
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

            gemm!(
                &mut cp, &self.pipes.gemm_acc, &activated, &layer.dp_w, &h_buf,
                s, hs, inter, hs, 0, 0, 0,
                (hs / 128) as u32, (mp / 128) as u32, 1
            );

            // Long prefills (s>=512) must submit+poll or Pascal/WDDM TDR
            // device-loses.  The interval is in *layers*, but what the driver
            // times is wall clock, and the slabbed path spends ~1 s per layer at
            // 8k tokens: four layers there is past the 2 s TDR timeout and the
            // device is lost mid-prefill.  Halve it once the slab path is in.
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

        // tail: final norm of the LAST row → lm head → argmax
        let last_off = ((s - 1) * hs * 2) as u64;
        let bg_fn = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("p.final_rms"),
            layout: &self.pipes.rms_norm.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &h_buf,
                        offset: last_off,
                        size: std::num::NonZeroU64::new((hs * 2) as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: self.norm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.scratch.final_norm.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.scratch.u_rms.as_entire_binding() },
            ],
        });
        cp.set_pipeline(&self.pipes.rms_norm);
        cp.set_bind_group(0, &bg_fn, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.pipes.gemv_lm);
        cp.set_bind_group(0, &self.bg_gemv_lm, &[]);
        cp.dispatch_workgroups((cfg.vocab_size / 8) as u32, 1, 1);
        cp.set_pipeline(&self.pipes.argmax);
        cp.set_bind_group(0, &self.bg_argmax, &[]);
        cp.dispatch_workgroups(1, 1, 1);
        drop(cp);
        enc.copy_buffer_to_buffer(&self.scratch.token, 0, &self.scratch.token_staging, 0, 4);
        gpu.queue.submit([enc.finish()]);

        // scaffolding: snapshot L0's row-0 output for offline bisection
        {
            let dbg = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("p.dbg_h0"),
                size: (hs * 2) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut denc = gpu.device.create_command_encoder(&Default::default());
            denc.copy_buffer_to_buffer(&h_buf, 0, &dbg, 0, (hs * 2) as u64);
            gpu.queue.submit([denc.finish()]);
            self.debug_l0_h = Some(dbg);
        }
        self.debug_prefill_h = Some(h_buf);
        let tok = self.read_token()?;
        self.pos = kv_start + s;
        Ok(tok)
    }
}
