//! WGSL sources for the decode chain.  f32 addition is not associative, so the
//! reduction orders below are load-bearing.
//!
//! Layout: activations are 16-bit floats packed as `array<u32>` (word `j` holds
//! elements `2j` / `2j+1`); weights are packed as `array<vec4<u32>>` (8 halves =
//! 16 B per element).  A plain `array<f16>` binding is never needed — Pascal
//! exposes 16-bit storage but not `shaderFloat16`.
//!
//! ## Why the storage format is a parameter
//!
//! The published checkpoint stores **bf16** and the reference runs it that way
//! (`AutoModelForTokenClassification.from_pretrained(..., dtype=torch.bfloat16)`),
//! so matching the reference means rounding the activations to bf16 at the same
//! boundaries.  f16 and bf16 are the same width and the same packing, so the
//! choice costs nothing in layout — it changes only how a packed word is
//! unpacked and how a result is packed, which is what [`Half`] supplies.
//!
//! Storing f16 instead (10 mantissa bits rather than bf16's 8) computes *more*
//! precisely, and lands on the fp32 reference instead — see `docs/perf.md` §1.1b.

#![allow(dead_code)]

use std::sync::atomic::{AtomicU8, Ordering};

pub const LOG2_E: f32 = std::f32::consts::LOG2_E;

/// The 16-bit storage format, as a shader prelude.
///
/// Both variants present the same interface to a kernel: `unpack_h(w) ->
/// vec2<f32>` (element 0 = `.x`, element 1 = `.y`) and `pack_h(v) -> u32`.
/// Swapping the format is therefore a rebuild of the shader source and nothing
/// else: same buffer sizes, same strides, same bindings.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Half {
    /// IEEE binary16.  What this implementation shipped with; more precise than
    /// the reference for the same footprint.
    F16,
    /// bfloat16 — the top 16 bits of an f32.  What the reference computes in.
    Bf16,
}

impl Half {
    pub const ALL: [Half; 2] = [Half::F16, Half::Bf16];

    pub fn name(self) -> &'static str {
        match self {
            Half::F16 => "f16",
            Half::Bf16 => "bf16",
        }
    }

    pub fn parse(s: &str) -> Option<Half> {
        match s.to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "half" => Some(Half::F16),
            "bf16" | "bfloat16" => Some(Half::Bf16),
            _ => None,
        }
    }

    /// The prelude a kernel needs in order to move values in and out of storage.
    pub fn prelude(self) -> &'static str {
        match self {
            Half::F16 => "\
fn unpack_h(w: u32) -> vec2<f32> { return unpack2x16float(w); }
fn pack_h(v: vec2<f32>) -> u32 { return pack2x16float(v); }
",
            Half::Bf16 => "\
fn unpack_h(w: u32) -> vec2<f32> {
    // bf16 is the top half of an f32, so each element is a bitcast with the
    // other half filled in: the low element shifts up, the high element is
    // already in place (its low bits are zero, which is a valid f32).
    return vec2<f32>(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xFFFF0000u));
}
fn rne_bf16(x: f32) -> u32 {
    // Round to nearest, ties to even, at bit 16 — `x + 0x7FFF + lsb` carries
    // into the exponent exactly when the discarded bits are past the halfway
    // point, or exactly at it with an odd keeper.  Finite operands only; the
    // rne_bf16_matches_half_crate test states where this stops holding.
    let u = bitcast<u32>(x);
    return u + 0x7FFFu + ((u >> 16u) & 1u);
}
fn pack_h(v: vec2<f32>) -> u32 {
    return (rne_bf16(v.x) >> 16u) | (rne_bf16(v.y) & 0xFFFF0000u);
}
",
        }
    }

    /// The byte the process-wide setting stores — the discriminant, so the
    /// `u8` and the enum cannot drift apart.
    fn from_byte(b: u8) -> Half {
        Half::ALL
            .into_iter()
            .find(|h| *h as u8 == b)
            .unwrap_or(DEFAULT_HALF)
    }
}

/// The run's storage format, resolved once.
///
/// The format follows from the checkpoint plus the caller's `--dtype`, so it is
/// a constant for the whole process: every kernel's prelude, every weight the
/// uploader narrows and every host-side conversion reads this one value.  It
/// must be fixed **before the first pipeline is built** — a pipeline compiled
/// for the other format keeps unpacking the other format while the weights come
/// from this one, and nothing about that is visible in the output except the
/// timestamps being wrong.
///
/// [`set_half`] therefore panics on a second, *different* value, and treats a
/// repeat of the same value as a no-op: the CLI and a library caller both
/// naming bf16 is not a conflict.
static HALF: AtomicU8 = AtomicU8::new(UNSET);

/// No format chosen yet.  Not a `Half` discriminant, so "unset" cannot be
/// mistaken for a choice.
const UNSET: u8 = u8::MAX;

/// The format a run gets unless it asks for another one: **f16**.
///
/// Not bf16, even though the checkpoint stores bf16 and the reference computes in
/// it.  Coarsening the activations to match was implemented and measured against
/// every gold, and it made the port a *worse* reproduction of the reference, not
/// a better one: against the bf16 gold, the f16 engine has **0** hard-failing
/// endpoints on both streams (3597/3622 raw, 3590/3622 repaired — the remainder
/// being endpoints the reference itself answers differently per dtype, with our
/// value equal to the fp32 run's at every one of them), while the bf16 engine has
/// 10 and 15.
///
/// The reason is that the reference's dtype governs its whole arithmetic — the
/// accumulation order and every place torch rounds — and this port fuses
/// differently on purpose, because that is where its speed comes from.  f16's
/// finer grid keeps those structural differences below the granularity that can
/// flip an argmax; bf16's coarser grid promotes them into visible flips.  See
/// `docs/perf.md` §1.1c for the cross-table.
///
/// `--dtype bf16` stays available to anyone who wants to see that directly.
pub const DEFAULT_HALF: Half = Half::F16;

/// Fix the 16-bit storage format for this process — before the first pipeline
/// exists, see [`HALF`].
pub fn set_half(h: Half) {
    match HALF.compare_exchange(UNSET, h as u8, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => {}
        Err(cur) if cur == h as u8 => {}
        Err(cur) => panic!(
            "shaders::set_half({}) after the format was already fixed to {}: the pipelines \
             built for the first format would go on unpacking it",
            h.name(),
            Half::from_byte(cur).name()
        ),
    }
}

/// The process-wide storage format — what [`set_half`] fixed, or
/// [`DEFAULT_HALF`] if nobody did.
pub fn half() -> Half {
    Half::from_byte(HALF.load(Ordering::Relaxed))
}

/// Read element `i` out of a packed 16-bit word (`v.x` = even index, `v.y` = odd).
///
/// Format-independent — it picks out of a pair `unpack_h` already widened — so it
/// is one string rather than a [`Half`] method.  Injected by the builders that
/// need it; the audio tower's shaders spell their own equivalent inline.
const HALF_AT: &str = "\
fn half_at(v: vec2<f32>, i: u32) -> f32 { return select(v.y, v.x, (i & 1u) == 0u); }
";

/// [`Half`]'s bf16 rounding, in Rust — the high 16 bits of the WGSL `rne_bf16`.
///
/// The shader cannot be executed from a unit test on this box, so the test below
/// pins *this* transcription against the `half` crate's `bf16::from_f32`, which
/// rounds the same way the reference's casts do.  A transcription can drift from
/// the WGSL it mirrors, and that risk is smaller than shipping a rounding rule
/// nothing checks (see `OPTIMIZATION_PLAYBOOK` §4.4).
pub fn rne_bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    (u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod bf16_round_tests {
    use super::*;

    /// The bf16 path's rounding has to be the reference's rule, or every layer
    /// it touches is off in a direction nobody chose.
    ///
    /// `half::bf16::from_f32` is round-to-nearest-even, the same rule as
    /// `torch.Tensor.to(torch.bfloat16)` and the cast cuBLAS applies to a bf16
    /// GEMM's f32 accumulator, so agreeing with it is the whole test.
    #[test]
    fn rne_bf16_matches_half_crate() {
        let mut checked = 0usize;
        let mut bad: Vec<(f32, u16, u16)> = Vec::new();

        let check = |x: f32, checked: &mut usize, bad: &mut Vec<(f32, u16, u16)>| {
            // NaN is excluded on purpose: this rule propagates the payload bits
            // instead of coercing to a quiet NaN like `half` does.  Nothing in
            // the model feeds NaN into a cast, and `nan_is_the_only_gap` states
            // the gap rather than hiding it.
            if !x.is_finite() {
                return;
            }
            *checked += 1;
            let ours = rne_bf16_bits(x);
            let want = half::bf16::from_f32(x).to_bits();
            if ours != want && bad.len() < 8 {
                bad.push((x, ours, want));
            }
        };

        // Halfway cases: the low 16 bits at exactly 0x8000 are ties, and
        // 0x7FFF / 0x8001 sit either side of one.  Cancellation inside a GEMM's
        // accumulator lands on exactly these far more often than chance, which
        // is why they get their own sweep rather than trusting a random one.
        for &e in &[0u32, 1, 2, 50, 63, 64, 100, 126, 127, 128, 200, 250, 253, 254] {
            for &m_top in &[0u32, 1, 2, 0x3F, 0x40, 0x7E, 0x7F] {
                for &low in &[0u32, 1, 0x7FFE, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF] {
                    for &sign in &[0u32, 1u32 << 31] {
                        check(
                            f32::from_bits(sign | (e << 23) | (m_top << 16) | low),
                            &mut checked,
                            &mut bad,
                        );
                    }
                }
            }
        }

        // And a dense pseudo-random sweep over the whole bit space, so the
        // exponent ranges the table above never reaches are still covered.
        let mut s = 0x1234_5678u32;
        for _ in 0..500_000 {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            check(f32::from_bits(s), &mut checked, &mut bad);
        }

        assert!(
            bad.is_empty(),
            "{checked} values checked; first mismatches (x, ours, half) = {bad:?}"
        );
        assert!(checked > 400_000, "sweep did not run: {checked} checked");
    }

    /// The one input class this rule does not reproduce, stated explicitly: a
    /// NaN keeps its payload here and becomes a quiet NaN in `half`.
    #[test]
    fn nan_is_the_only_gap() {
        let qnan = f32::from_bits(0x7FC0_0001);
        assert!(qnan.is_nan());
        assert!(half::bf16::from_f32(qnan).is_nan());
        // Ours is also a NaN -- it just may not be the same NaN.
        assert!(f32::from_bits((rne_bf16_bits(qnan) as u32) << 16).is_nan());
    }

    /// The setting starts unset and reads back as the checkpoint's own format,
    /// so a caller that names none still gets the reference's arithmetic.
    ///
    /// The conflict path is deliberately not exercised here: the setting is
    /// process-wide, and a test that trips the panic takes every other test in
    /// this binary with it.
    /// The default is f16, and that is a measured choice rather than a
    /// preference: it is the format with no unmatched endpoint against any of the
    /// three golds.  A test that pinned bf16 here would have been pinning the
    /// worse of the two, so what this asserts is the *reason* as much as the
    /// value -- see [`DEFAULT_HALF`].
    #[test]
    fn the_format_defaults_to_f16() {
        assert_eq!(DEFAULT_HALF, Half::F16);
        assert_eq!(half(), DEFAULT_HALF);
        assert_eq!(Half::from_byte(u8::MAX), DEFAULT_HALF);
        for h in Half::ALL {
            assert_eq!(Half::from_byte(h as u8), h, "{}", h.name());
        }
        // Naming the running format again is not a conflict.
        set_half(half());
        set_half(DEFAULT_HALF);
        assert_eq!(half(), DEFAULT_HALF);
    }

    /// f16 and bf16 must present the same interface *and* the same element
    /// order: word `j` holds element `2j` in its low half and `2j+1` in its high
    /// half.  Every kernel's indexing assumes that, and a swapped `.x`/`.y` in
    /// the bf16 arm is precisely what it would break.
    ///
    /// The order is pinned by the expressions that spell it out rather than by
    /// containment, because a containment check passes just as happily on the
    /// swapped version.  That makes this test sensitive to reformatting the
    /// shader — which is the right way round for the one convention that has no
    /// other check behind it.
    #[test]
    fn both_formats_agree_on_the_packing() {
        for h in Half::ALL {
            assert!(h.prelude().contains("fn unpack_h("), "{} unpack", h.name());
            assert!(h.prelude().contains("fn pack_h("), "{} pack", h.name());
        }

        let bf = Half::Bf16.prelude();
        // Unpack: element 0 from the low half, element 1 from the high half.
        assert!(
            bf.contains("bitcast<f32>(w << 16u)"),
            "bf16 element 0 must be taken from the low half"
        );
        assert!(
            bf.contains("bitcast<f32>(w & 0xFFFF0000u)"),
            "bf16 element 1 must be taken from the high half"
        );
        // Pack, the same convention in reverse.
        assert!(
            bf.contains("rne_bf16(v.x) >> 16u"),
            "bf16 pack must place element 0 in the low half"
        );
        assert!(
            bf.contains("rne_bf16(v.y) & 0xFFFF0000u"),
            "bf16 pack must place element 1 in the high half"
        );

        // f16 delegates to the builtins, which pack the same way; that the two
        // agree is the whole point of the switch.
        let f = Half::F16.prelude();
        assert!(f.contains("unpack2x16float(w)"));
        assert!(f.contains("pack2x16float(v)"));

        assert_eq!(Half::parse("bf16"), Some(Half::Bf16));
        assert_eq!(Half::parse("BFloat16"), Some(Half::Bf16));
        assert_eq!(Half::parse("f16"), Some(Half::F16));
        assert_eq!(Half::parse("fp32"), None);
    }
}

/// `rms_norm_f16` — one workgroup per row, block tree reduction over `LAST`.
///
/// Each step accumulates one packed word's `vx*vx + vy*vy`; `LAST` is even, so no
/// odd/even tail exists.  The `s`-halving tree and the two passes over `x` are
/// part of the arithmetic order.
pub fn rms_norm(last: usize, bs: usize) -> String {
    assert!(last % 2 == 0 && last >= bs, "rms_norm: last must be even and >= bs");
    format!(
        "{HALF_AT}
struct Cfg {{ eps: f32, _a: f32, _b: f32, _c: f32 }};

@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read>       Wt:  array<u32>;
@group(0) @binding(2) var<storage, read_write> Out: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: Cfg;

const LAST: u32 = {last}u;
const BS: u32 = {bs}u;
const LAST2: u32 = {last2}u;

var<workgroup> red: array<f32, {bs}>;

@compute @workgroup_size({bs})
fn rms_norm(@builtin(workgroup_id) wgid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {{
    // row = workgroup id: decode dispatches one row; prefill dispatches s rows
    // of a [s, LAST] activation.  The reduction order is row-independent.
    let row = wgid.x * LAST2;
    var local = 0.0;
    for (var j = lid.x; j < LAST2; j = j + BS) {{
        let v = unpack_h(X[row + j]);
        local = local + (v.x * v.x + v.y * v.y);
    }}
    red[lid.x] = local;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red[lid.x] = red[lid.x] + red[lid.x + s]; }}
        workgroupBarrier();
    }}
    let inv_rms = inverseSqrt(red[0] / f32(LAST) + cfg.eps);
    workgroupBarrier();
    for (var j = lid.x; j < LAST2; j = j + BS) {{
        let xv = unpack_h(X[row + j]);
        let wv = unpack_h(Wt[j]);
        Out[row + j] = pack_h(vec2<f32>(
            xv.x * inv_rms * wv.x,
            xv.y * inv_rms * wv.y));
    }}
}}
",
        last2 = last / 2
    )
}

/// [`gemv`] with a split-K dimension: `splits` workgroups (distinguished by
/// `wgid.z`) each own a contiguous range of the row's K granules and write an f32
/// partial, which [`gemv_merge`] combines.  Unlike [`gemv`] this is not
/// bit-identical: each per-lane accumulator covers a sub-range of K.
pub fn gemv_split(n: usize, k: usize, subgroup: bool, splits: usize) -> String {
    assert_eq!(n % 8, 0, "gemv_split: rows must be a multiple of 8");
    assert_eq!(k % 8, 0, "gemv_split: k must be a multiple of 8");
    let kg = k / 8;
    assert_eq!(kg % 32, 0, "gemv_split: k/8 must be a multiple of 32");
    let granules_per_lane = kg / 32;
    assert_eq!(granules_per_lane % splits, 0, "gemv_split: granules/lane must divide by splits");
    let gpt = granules_per_lane / splits;

    let subgroup_body = if subgroup {
        "    var t = v;
    t = t + subgroupShuffleXor(t, 16u);
    t = t + subgroupShuffleXor(t, 8u);
    t = t + subgroupShuffleXor(t, 4u);
    t = t + subgroupShuffleXor(t, 2u);
    t = t + subgroupShuffleXor(t, 1u);
    return t;"
    } else {
        "    let wb = lid & 0xFFFFFFE0u;
    bt0[lid] = v;
    workgroupBarrier();
    var t = bt0[lid] + bt0[wb + (lane ^ 16u)];
    bt1[lid] = t;
    workgroupBarrier();
    t = bt1[lid] + bt1[wb + (lane ^ 8u)];
    bt0[lid] = t;
    workgroupBarrier();
    t = bt0[lid] + bt0[wb + (lane ^ 4u)];
    bt1[lid] = t;
    workgroupBarrier();
    t = bt1[lid] + bt1[wb + (lane ^ 2u)];
    bt0[lid] = t;
    workgroupBarrier();
    t = bt0[lid] + bt0[wb + (lane ^ 1u)];
    return t;"
    };
    let bfly_scratch = if subgroup {
        ""
    } else {
        "var<workgroup> bt0: array<f32, 256>;
var<workgroup> bt1: array<f32, 256>;
"
    };
    // One granule = 4 u32 words = 8 f16 columns; the four words feed the four
    // accumulators.
    let mut body = String::new();
    for g in 0..gpt {
        body.push_str(&format!(
            "        let wv{g} = Wt[wbase + (sp * {gpt}u + {g}u) * 32u + lane];\n\
             \x20        let xv{g} = X[(sp * {gpt}u + {g}u) * 32u + lane];\n",
        ));
        for c in 0..4 {
            let sel = ["x", "y", "z", "w"][c];
            body.push_str(&format!(
                "        let w{g}{c} = unpack_h(wv{g}.{sel});\n\
                 \x20        let x{g}{c} = unpack_h(xv{g}.{sel});\n\
                 \x20        a{c} = fma(w{g}{c}.x, x{g}{c}.x, fma(w{g}{c}.y, x{g}{c}.y, a{c}));\n",
            ));
        }
    }
    format!(
        "@group(0) @binding(0) var<storage, read>       Wt: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       X:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> P:  array<f32>;

const KG: u32 = {kg}u;
const SPLITS: u32 = {splits}u;
const SUBGROUP: u32 = {subgroup_lit}u;

{bfly_scratch}var<workgroup> rows_out: array<f32, 8>;

fn bfly(v: f32, lid: u32, lane: u32) -> f32 {{
{subgroup_body}
}}

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row = workgroup_id.x * 8u + warp;
    let sp = workgroup_id.z;
    let wbase = row * KG;
    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    {{
{body}    }}
    let acc = (a0 + a1) + (a2 + a3);
    let r = bfly(acc, lid.x, lane);
    if (lane == 0u) {{ rows_out[warp] = r; }}
    workgroupBarrier();
    if (lid.x < 8u) {{
        P[(workgroup_id.x * 8u + lid.x) * SPLITS + sp] = rows_out[lid.x];
    }}
}}
",
        subgroup_lit = u32::from(subgroup),
    )
}

/// Combine the [`gemv_split`] partials: `Y[i] = sum_s P[i*SPLITS + s]`, plus the
/// residual word when `accum`.  One thread per packed f16 output word (two rows).
pub fn gemv_merge(accum: bool) -> String {
    format!(
        "struct Cfg {{ words: u32, splits: u32 }};

@group(0) @binding(0) var<storage, read>       P: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

const ACCUM: u32 = {accum}u;

@compute @workgroup_size(256)
fn gemv_merge(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= cfg.words) {{ return; }}
    var va = 0.0;
    var vb = 0.0;
    for (var s = 0u; s < cfg.splits; s = s + 1u) {{
        va = va + P[(2u * i) * cfg.splits + s];
        vb = vb + P[(2u * i + 1u) * cfg.splits + s];
    }}
    if (ACCUM == 1u) {{
        let old = unpack_h(Y[i]);
        va = va + old.x;
        vb = vb + old.y;
    }}
    Y[i] = pack_h(vec2<f32>(va, vb));
}}
",
        accum = u32::from(accum),
    )
}

/// `gemv_f16` -- warp-per-row, 8-half `vec4<u32>` lane-strided loads on both the
/// weight row and the activation vector, four independent f32 accumulators,
/// 5-round xor butterfly over 32-lane warps, residual add folded into the epilogue.
///
/// `n` must be a multiple of 8 so no partial workgroup exists; `k/8` must be a
/// multiple of 32 so the granule loop divides evenly.
pub fn gemv(n: usize, k: usize, accum: bool, subgroup: bool) -> String {
    assert_eq!(n % 8, 0, "gemv: rows must be a multiple of 8");
    let kg = k / 8;
    assert_eq!(kg % 32, 0, "gemv: k/8 must be a multiple of 32");
    assert_eq!(k % 8, 0, "gemv: k must be a multiple of 8");
    let tiles = kg / 32;
    assert_eq!(tiles % 4, 0, "gemv: k/256 must be a multiple of 4 (unrolled x4)");
    let accum_lit = if accum { 1u32 } else { 0u32 };
    let subgroup_lit = if subgroup { 1u32 } else { 0u32 };
    // The two bodies compute the same tree; see `bfly`'s doc comment.  The
    // shared-memory form alternates two buffers so each round's read cannot
    // observe another lane's write from the same round.
    let subgroup_body = if subgroup {
        "    var t = v;
    t = t + subgroupShuffleXor(t, 16u);
    t = t + subgroupShuffleXor(t, 8u);
    t = t + subgroupShuffleXor(t, 4u);
    t = t + subgroupShuffleXor(t, 2u);
    t = t + subgroupShuffleXor(t, 1u);
    return t;"
    } else {
        "    let wb = lid & 0xFFFFFFE0u;
    bt0[lid] = v;
    workgroupBarrier();
    var t = bt0[lid] + bt0[wb + (lane ^ 16u)];
    bt1[lid] = t;
    workgroupBarrier();
    t = bt1[lid] + bt1[wb + (lane ^ 8u)];
    bt0[lid] = t;
    workgroupBarrier();
    t = bt0[lid] + bt0[wb + (lane ^ 4u)];
    bt1[lid] = t;
    workgroupBarrier();
    t = bt1[lid] + bt1[wb + (lane ^ 2u)];
    bt0[lid] = t;
    workgroupBarrier();
    t = bt0[lid] + bt0[wb + (lane ^ 1u)];
    return t;"
    };
    // The shared-memory form still needs its staging buffers declared.
    let bfly_scratch = if subgroup {
        ""
    } else {
        "var<workgroup> bt0: array<f32, 256>;
var<workgroup> bt1: array<f32, 256>;
"
    };
    format!(
        "@group(0) @binding(0) var<storage, read>       Wt: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       X:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> Y:  array<u32>;

const KG: u32 = {kg}u;
const TILES: u32 = {tiles}u;
const ACCUM: u32 = {accum_lit}u;
const SUBGROUP: u32 = {subgroup_lit}u;

{bfly_scratch}var<workgroup> rows_out: array<f32, 8>;

/// 5-round xor butterfly over the 32 lanes of one warp, order ^16,^8,^4,^2,^1.
///
/// `SUBGROUP=1` emits `subgroupShuffleXor` (requires `Features::SUBGROUP`);
/// `SUBGROUP=0` runs the same tree through shared memory with 5 barriers.  Both
/// are bit-identical: every step is one f32 add of the same two operands.
fn bfly(v: f32, lid: u32, lane: u32) -> f32 {{
{subgroup_body}
}}

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row = wgid.x * 8u + warp;

    let wbase = row * KG;
    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    // Tiles unrolled x4 with prefetched loads: each warp keeps 4 (Wt, X) pairs
    // in flight to hide DRAM latency (rows == warps leaves the machine
    // underfilled on the small projections).  The fma chain still consumes
    // tiles in ascending order, so the reduction bits are unchanged.
    var i = lane;
    for (var g = 0u; g < TILES; g = g + 4u) {{
        let wv0 = Wt[wbase + i];
        let xv0 = X[i];
        let wv1 = Wt[wbase + i + 32u];
        let xv1 = X[i + 32u];
        let wv2 = Wt[wbase + i + 64u];
        let xv2 = X[i + 64u];
        let wv3 = Wt[wbase + i + 96u];
        let xv3 = X[i + 96u];
        let w00 = unpack_h(wv0.x); let x00 = unpack_h(xv0.x);
        let w01 = unpack_h(wv0.y); let x01 = unpack_h(xv0.y);
        let w02 = unpack_h(wv0.z); let x02 = unpack_h(xv0.z);
        let w03 = unpack_h(wv0.w); let x03 = unpack_h(xv0.w);
        a0 = fma(w00.x, x00.x, fma(w00.y, x00.y, a0));
        a1 = fma(w01.x, x01.x, fma(w01.y, x01.y, a1));
        a2 = fma(w02.x, x02.x, fma(w02.y, x02.y, a2));
        a3 = fma(w03.x, x03.x, fma(w03.y, x03.y, a3));
        let w10 = unpack_h(wv1.x); let x10 = unpack_h(xv1.x);
        let w11 = unpack_h(wv1.y); let x11 = unpack_h(xv1.y);
        let w12 = unpack_h(wv1.z); let x12 = unpack_h(xv1.z);
        let w13 = unpack_h(wv1.w); let x13 = unpack_h(xv1.w);
        a0 = fma(w10.x, x10.x, fma(w10.y, x10.y, a0));
        a1 = fma(w11.x, x11.x, fma(w11.y, x11.y, a1));
        a2 = fma(w12.x, x12.x, fma(w12.y, x12.y, a2));
        a3 = fma(w13.x, x13.x, fma(w13.y, x13.y, a3));
        let w20 = unpack_h(wv2.x); let x20 = unpack_h(xv2.x);
        let w21 = unpack_h(wv2.y); let x21 = unpack_h(xv2.y);
        let w22 = unpack_h(wv2.z); let x22 = unpack_h(xv2.z);
        let w23 = unpack_h(wv2.w); let x23 = unpack_h(xv2.w);
        a0 = fma(w20.x, x20.x, fma(w20.y, x20.y, a0));
        a1 = fma(w21.x, x21.x, fma(w21.y, x21.y, a1));
        a2 = fma(w22.x, x22.x, fma(w22.y, x22.y, a2));
        a3 = fma(w23.x, x23.x, fma(w23.y, x23.y, a3));
        let w30 = unpack_h(wv3.x); let x30 = unpack_h(xv3.x);
        let w31 = unpack_h(wv3.y); let x31 = unpack_h(xv3.y);
        let w32 = unpack_h(wv3.z); let x32 = unpack_h(xv3.z);
        let w33 = unpack_h(wv3.w); let x33 = unpack_h(xv3.w);
        a0 = fma(w30.x, x30.x, fma(w30.y, x30.y, a0));
        a1 = fma(w31.x, x31.x, fma(w31.y, x31.y, a1));
        a2 = fma(w32.x, x32.x, fma(w32.y, x32.y, a2));
        a3 = fma(w33.x, x33.x, fma(w33.y, x33.y, a3));
        i = i + 128u;
    }}
    let acc = (a0 + a1) + (a2 + a3);
    let r = bfly(acc, lid.x, lane);
    if (lane == 0u) {{ rows_out[warp] = r; }}
    workgroupBarrier();
    if (lid.x == 0u) {{
        let wordbase = (wgid.x * 8u) >> 1u;
        for (var w = 0u; w < 4u; w = w + 1u) {{
            var va = rows_out[2u * w];
            var vb = rows_out[2u * w + 1u];
            if (ACCUM == 1u) {{
                let old = unpack_h(Y[wordbase + w]);
                va = va + old.x;
                vb = vb + old.y;
            }}
            Y[wordbase + w] = pack_h(vec2<f32>(va, vb));
        }}
    }}
}}
",
        kg = kg,
        tiles = tiles,
        accum_lit = accum_lit,
        subgroup_lit = subgroup_lit,
        subgroup_body = subgroup_body,
        bfly_scratch = bfly_scratch,
    )
}

/// [`gemv`] with the RMSNorm that feeds it folded into the workgroup prologue,
/// removing the per-site 1-workgroup norm dispatches.
///
/// `bs = block_for_reduction(hs)` and the workgroup is 256, so each thread folds
/// `vc = bs/256` virtual partials with the same pairing and add order as the
/// norm's `red[t] += red[t+s]` tree.  The staging loop writes the f16
/// `pack_h(x * inv_rms * w)` the norm would have stored, not an f32
/// intermediate.  Writes `Y` exactly like [`gemv`].
pub fn gemv_norm(
    n: usize,
    k: usize,
    accum: bool,
    subgroup: bool,
    last: usize,
    bs: usize,
    eps: f32,
) -> String {
    assert_eq!(n % 8, 0, "gemv_norm: rows must be a multiple of 8");
    let kg = k / 8;
    assert_eq!(kg % 32, 0, "gemv_norm: k/8 must be a multiple of 32");
    assert_eq!(k % 8, 0, "gemv_norm: k must be a multiple of 8");
    let tiles = kg / 32;
    assert_eq!(tiles % 4, 0, "gemv_norm: k/256 must be a multiple of 4");
    let last2 = last / 2;
    assert_eq!(last2, kg * 4, "gemv_norm: activation row {last2} words != k/2 {kg_expected}", kg_expected = k / 2);
    let vc = bs / 256;
    assert!(
        [1usize, 2, 4].contains(&vc) && bs == vc * 256,
        "gemv_norm: block size {bs} is not 256/512/1024"
    );
    let accum_lit = u32::from(accum);
    let subgroup_lit = u32::from(subgroup);
    // The folded rounds pair up as the reduction tree does.
    let folded = match vc {
        1 => "l0".to_string(),
        2 => "l0 + l1".to_string(),
        _ => "(l0 + l2) + (l1 + l3)".to_string(),
    };
    let mut locals = String::new();
    for i in 0..vc {
        if i == 0 {
            locals.push_str("    var l0 = 0.0;\n");
        } else {
            locals.push_str(&format!("    var l{i} = 0.0;\n"));
        }
    }
    let mut sums = String::new();
    for i in 0..vc {
        let off = i * 256;
        sums.push_str(&format!(
            "    for (var j = lid.x + {off}u; j < LAST2; j = j + BS) {{\n\
             \x20       let v = unpack_h(Xr[j]);\n\
             \x20       l{i} = l{i} + (v.x * v.x + v.y * v.y);\n\
             \x20   }}\n"
        ));
    }
    let subgroup_body = if subgroup {
        "    var t = v;
    t = t + subgroupShuffleXor(t, 16u);
    t = t + subgroupShuffleXor(t, 8u);
    t = t + subgroupShuffleXor(t, 4u);
    t = t + subgroupShuffleXor(t, 2u);
    t = t + subgroupShuffleXor(t, 1u);
    return t;"
    } else {
        "    let wb = lid & 0xFFFFFFE0u;
    bt0[lid] = v;
    workgroupBarrier();
    var t = bt0[lid] + bt0[wb + (lane ^ 16u)];
    bt1[lid] = t;
    workgroupBarrier();
    t = bt1[lid] + bt1[wb + (lane ^ 8u)];
    bt0[lid] = t;
    workgroupBarrier();
    t = bt0[lid] + bt0[wb + (lane ^ 4u)];
    bt1[lid] = t;
    workgroupBarrier();
    t = bt1[lid] + bt1[wb + (lane ^ 2u)];
    bt0[lid] = t;
    workgroupBarrier();
    t = bt0[lid] + bt0[wb + (lane ^ 1u)];
    return t;"
    };
    let bfly_scratch = if subgroup {
        ""
    } else {
        "var<workgroup> bt0: array<f32, 256>;
var<workgroup> bt1: array<f32, 256>;
"
    };
    format!(
        "@group(0) @binding(0) var<storage, read>       Wt:  array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       Xr:  array<u32>;
@group(0) @binding(2) var<storage, read_write> Y:   array<u32>;
@group(0) @binding(3) var<storage, read>       NW:  array<u32>;

const KG: u32 = {kg}u;
const TILES: u32 = {tiles}u;
const ACCUM: u32 = {accum_lit}u;
const SUBGROUP: u32 = {subgroup_lit}u;
const LAST: u32 = {last}u;
const LAST2: u32 = {last2}u;
const BS: u32 = {bs}u;
const EPS: f32 = {eps:?}f;

{bfly_scratch}var<workgroup> rows_out: array<f32, 8>;
var<workgroup> red: array<f32, 256>;
/// The normalized activation row, f16-packed.
var<workgroup> xs: array<vec4<u32>, {kg}u>;

fn bfly(v: f32, lid: u32, lane: u32) -> f32 {{
{subgroup_body}
}}

/// Normalized output word `j`: the same two multiplies in the same order the
/// norm applies, so the f16 rounding matches.
fn norm_word(j: u32, inv_rms: f32) -> u32 {{
    let xv = unpack_h(Xr[j]);
    let wv = unpack_h(NW[j]);
    return pack_h(vec2<f32>(xv.x * inv_rms * wv.x, xv.y * inv_rms * wv.y));
}}

@compute @workgroup_size(256)
fn gemv(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let lane = lid.x & 31u;
    let warp = lid.x >> 5u;
    let row = wgid.x * 8u + warp;

    // ── prologue: the RMSNorm folded into this GEMV ──
{locals}{sums}    red[lid.x] = {folded};
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red[lid.x] = red[lid.x] + red[lid.x + s]; }}
        workgroupBarrier();
    }}
    let inv_rms = inverseSqrt(red[0] / f32(LAST) + EPS);
    workgroupBarrier();
    for (var w4 = lid.x; w4 < KG; w4 = w4 + 256u) {{
        let b0 = norm_word(4u * w4, inv_rms);
        let b1 = norm_word(4u * w4 + 1u, inv_rms);
        let b2 = norm_word(4u * w4 + 2u, inv_rms);
        let b3 = norm_word(4u * w4 + 3u, inv_rms);
        xs[w4] = vec4<u32>(b0, b1, b2, b3);
    }}
    workgroupBarrier();

    let wbase = row * KG;
    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    // Body identical to `gemv`; only the activation comes from `xs`.
    var i = lane;
    for (var g = 0u; g < TILES; g = g + 4u) {{
        let wv0 = Wt[wbase + i];
        let xv0 = xs[i];
        let wv1 = Wt[wbase + i + 32u];
        let xv1 = xs[i + 32u];
        let wv2 = Wt[wbase + i + 64u];
        let xv2 = xs[i + 64u];
        let wv3 = Wt[wbase + i + 96u];
        let xv3 = xs[i + 96u];
        let w00 = unpack_h(wv0.x); let x00 = unpack_h(xv0.x);
        let w01 = unpack_h(wv0.y); let x01 = unpack_h(xv0.y);
        let w02 = unpack_h(wv0.z); let x02 = unpack_h(xv0.z);
        let w03 = unpack_h(wv0.w); let x03 = unpack_h(xv0.w);
        a0 = fma(w00.x, x00.x, fma(w00.y, x00.y, a0));
        a1 = fma(w01.x, x01.x, fma(w01.y, x01.y, a1));
        a2 = fma(w02.x, x02.x, fma(w02.y, x02.y, a2));
        a3 = fma(w03.x, x03.x, fma(w03.y, x03.y, a3));
        let w10 = unpack_h(wv1.x); let x10 = unpack_h(xv1.x);
        let w11 = unpack_h(wv1.y); let x11 = unpack_h(xv1.y);
        let w12 = unpack_h(wv1.z); let x12 = unpack_h(xv1.z);
        let w13 = unpack_h(wv1.w); let x13 = unpack_h(xv1.w);
        a0 = fma(w10.x, x10.x, fma(w10.y, x10.y, a0));
        a1 = fma(w11.x, x11.x, fma(w11.y, x11.y, a1));
        a2 = fma(w12.x, x12.x, fma(w12.y, x12.y, a2));
        a3 = fma(w13.x, x13.x, fma(w13.y, x13.y, a3));
        let w20 = unpack_h(wv2.x); let x20 = unpack_h(xv2.x);
        let w21 = unpack_h(wv2.y); let x21 = unpack_h(xv2.y);
        let w22 = unpack_h(wv2.z); let x22 = unpack_h(xv2.z);
        let w23 = unpack_h(wv2.w); let x23 = unpack_h(xv2.w);
        a0 = fma(w20.x, x20.x, fma(w20.y, x20.y, a0));
        a1 = fma(w21.x, x21.x, fma(w21.y, x21.y, a1));
        a2 = fma(w22.x, x22.x, fma(w22.y, x22.y, a2));
        a3 = fma(w23.x, x23.x, fma(w23.y, x23.y, a3));
        let w30 = unpack_h(wv3.x); let x30 = unpack_h(xv3.x);
        let w31 = unpack_h(wv3.y); let x31 = unpack_h(xv3.y);
        let w32 = unpack_h(wv3.z); let x32 = unpack_h(xv3.z);
        let w33 = unpack_h(wv3.w); let x33 = unpack_h(xv3.w);
        a0 = fma(w30.x, x30.x, fma(w30.y, x30.y, a0));
        a1 = fma(w31.x, x31.x, fma(w31.y, x31.y, a1));
        a2 = fma(w32.x, x32.x, fma(w32.y, x32.y, a2));
        a3 = fma(w33.x, x33.x, fma(w33.y, x33.y, a3));
        i = i + 128u;
    }}
    let acc = (a0 + a1) + (a2 + a3);
    let r = bfly(acc, lid.x, lane);
    if (lane == 0u) {{ rows_out[warp] = r; }}
    workgroupBarrier();
    if (lid.x == 0u) {{
        let wordbase = (wgid.x * 8u) >> 1u;
        for (var w = 0u; w < 4u; w = w + 1u) {{
            var va = rows_out[2u * w];
            var vb = rows_out[2u * w + 1u];
            if (ACCUM == 1u) {{
                let old = unpack_h(Y[wordbase + w]);
                va = va + old.x;
                vb = vb + old.y;
            }}
            Y[wordbase + w] = pack_h(vec2<f32>(va, vb));
        }}
    }}
}}
",
        kg = kg,
        tiles = tiles,
        accum_lit = accum_lit,
        subgroup_lit = subgroup_lit,
        last = last,
        last2 = last2,
        bs = bs,
        eps = eps,
        locals = locals,
        sums = sums,
        folded = folded,
        subgroup_body = subgroup_body,
        bfly_scratch = bfly_scratch,
    )
}

/// One workgroup per head slot, `grid = (1, nqh + nkvh)`.  Q heads land in
/// `QOut`; K heads write the roped K into `KCache` and copy V through verbatim.
pub fn qkv_extract(nqh: usize, nkvh: usize, d: usize) -> String {
    assert_eq!(d % 2, 0);
    let total_cols = (nqh + 2 * nkvh) * d;
    let q_dim = nqh * d;
    let kv_dim = nkvh * d;
    format!(
        "{HALF_AT}
struct Cfg {{ max_seq: u32, start: u32, pos_offset: u32, s: u32, eps: f32, _a: u32, _b: u32, _c: u32 }};

@group(0) @binding(0) var<storage, read>       Qkv:    array<u32>;
@group(0) @binding(1) var<storage, read>       QnW:    array<u32>;
@group(0) @binding(2) var<storage, read>       KnW:    array<u32>;
@group(0) @binding(3) var<storage, read>       Cos:    array<u32>;
@group(0) @binding(4) var<storage, read>       Sin:    array<u32>;
@group(0) @binding(5) var<storage, read_write> QOut:   array<u32>;
@group(0) @binding(6) var<storage, read_write> KCache: array<u32>;
@group(0) @binding(7) var<storage, read_write> VCache: array<u32>;
@group(0) @binding(8) var<uniform>             cfg:    Cfg;

const D: u32 = {d}u;
const D2: u32 = {d2}u;
const HD: u32 = {hd}u;
const NQH: u32 = {nqh}u;
const NKVH: u32 = {nkvh}u;
const TOT2: u32 = {tot2}u;
const QD2: u32 = {qd2}u;
const KVD2: u32 = {kvd2}u;
const BS: u32 = 128u;

var<workgroup> red: array<f32, 128>;

/// Sum of squares + tree reduction + inverse RMS (one element per thread, `s`
/// halving from `bs/2`).
fn head_inv_rms(base: u32, lid: u32) -> f32 {{
    var local = 0.0;
    for (var j = lid; j < D; j = j + BS) {{
        let e = half_at(unpack_h(Qkv[base + (j >> 1u)]), j);
        local = local + e * e;
    }}
    red[lid] = local;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid < s) {{ red[lid] = red[lid] + red[lid + s]; }}
        workgroupBarrier();
    }}
    let r = inverseSqrt(red[0] / f32(D) + cfg.eps);
    workgroupBarrier();
    return r;
}}

/// One roped output element: norm over the head row, then rotate-half RoPE.
/// `use_q` picks the Q norm weights over the K norm weights.
fn rope_elem(use_q: bool, base: u32, csbase: u32, j: u32, inv_rms: f32) -> f32 {{
    let xj = half_at(unpack_h(Qkv[base + (j >> 1u)]), j);
    var wj = 0.0;
    if (use_q) {{
        wj = half_at(unpack_h(QnW[j >> 1u]), j);
    }} else {{
        wj = half_at(unpack_h(KnW[j >> 1u]), j);
    }}
    let x_val_j = xj * inv_rms * wj;
    let pj = select(j - HD, j + HD, j < HD);
    let xp = half_at(unpack_h(Qkv[base + (pj >> 1u)]), pj);
    var wp = 0.0;
    if (use_q) {{
        wp = half_at(unpack_h(QnW[pj >> 1u]), pj);
    }} else {{
        wp = half_at(unpack_h(KnW[pj >> 1u]), pj);
    }}
    let x_pair = xp * inv_rms * wp;
    let pair_val = select(x_pair, -x_pair, j < HD);
    let c = half_at(unpack_h(Cos[csbase + (j >> 1u)]), j);
    let si = half_at(unpack_h(Sin[csbase + (j >> 1u)]), j);
    return x_val_j * c + pair_val * si;
}}

@compute @workgroup_size(128)
fn qkv_extract(@builtin(workgroup_id) wgid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {{
    // wgid.x = position within the batch (0 for decode, 0..s for prefill)
    let is = wgid.x;
    let hy = wgid.y;
    // cos/sin are `array<u32>` (2 f16 per word): row `p` starts at word p*D2.
    let csbase = (cfg.pos_offset + is) * D2;

    if (hy < NQH) {{
        // Q head `hy` of position `is`: word is*TOT2 + hy*D2 (D2 words = D f16).
        let base = is * TOT2 + hy * D2;
        let inv_rms = head_inv_rms(base, lid.x);
        for (var w = lid.x; w < D2; w = w + BS) {{
            let j0 = w * 2u;
            QOut[(hy * cfg.s + is) * D2 + w] = pack_h(vec2<f32>(
                rope_elem(true, base, csbase, j0, inv_rms),
                rope_elem(true, base, csbase, j0 + 1u, inv_rms)));
        }}
    }} else {{
        let ih = hy - NQH;
        if (ih < NKVH) {{
            let kbase = is * TOT2 + QD2 + ih * D2;
            let vbase = is * TOT2 + QD2 + KVD2 + ih * D2;
            let cache = (ih * cfg.max_seq + cfg.start + is) * D2;
            let inv_rms = head_inv_rms(kbase, lid.x);
            for (var w = lid.x; w < D2; w = w + BS) {{
                let j0 = w * 2u;
                KCache[cache + w] = pack_h(vec2<f32>(
                    rope_elem(false, kbase, csbase, j0, inv_rms),
                    rope_elem(false, kbase, csbase, j0 + 1u, inv_rms)));
                VCache[cache + w] = Qkv[vbase + w];
            }}
        }}
    }}
}}
",
        d2 = d / 2,
        hd = d / 2,
        tot2 = total_cols / 2,
        qd2 = q_dim / 2,
        kvd2 = kv_dim / 2,
    )
}

/// Single-workgroup-per-q_head flash-style attention.  `bs` is the workgroup
/// size: 256 for `cur_len <= 512`, else 512.
// Correctly-rounded expf for the decode attention softmax, evaluated so the
// result is reproducible bit-for-bit.  The three correctly-rounded fmas and the
// final power-of-two scaling use pure u32 integer arithmetic, which is exact and
// associative, so driver transforms (FFMA contraction, reassociation, CSE) cannot
// perturb it.  WGSL's exp2 is `ex2.approx.ftz` bit-for-bit, including the
// subnormal output flush; the rest of the exactness is `scale_pow2`'s integer GRS.
const EXP_BT: &str = "
fn lead32(v0: u32) -> i32 {
    var n = 0i;
    var v = v0;
    if ((v & 0xFFFF0000u) != 0u) { n = n + 16; v = v >> 16u; }
    if ((v & 0xFF00u) != 0u)     { n = n + 8;  v = v >> 8u;  }
    if ((v & 0xF0u) != 0u)       { n = n + 4;  v = v >> 4u;  }
    if ((v & 0xCu) != 0u)        { n = n + 2;  v = v >> 2u;  }
    if ((v & 0x2u) != 0u)        { n = n + 1; }
    return n;
}
fn shr64(hi0: u32, lo0: u32, t0: i32) -> vec3<u32> {
    if (t0 <= 0) { return vec3<u32>(hi0, lo0, 0u); }
    if (t0 >= 64) { return vec3<u32>(0u, 0u, select(0u, 1u, (hi0 | lo0) != 0u)); }
    if (t0 >= 32) {
        let s = u32(t0) - 32u;
        let lost = select(0u, 1u, lo0 != 0u) | select(0u, 1u, (hi0 & ((1u << s) - 1u)) != 0u);
        return vec3<u32>(0u, hi0 >> s, lost);
    }
    let s = u32(t0);
    let lost = select(0u, 1u, (lo0 & ((1u << s) - 1u)) != 0u);
    return vec3<u32>(hi0 >> s, (lo0 >> s) | (hi0 << (32u - s)), lost);
}
fn shl64(hi0: u32, lo0: u32, t0: i32) -> vec2<u32> {
    if (t0 <= 0) { return vec2<u32>(hi0, lo0); }
    if (t0 >= 32) {
        let s = u32(t0) - 32u;
        return vec2<u32>(lo0 << s, 0u);
    }
    let s = u32(t0);
    return vec2<u32>((hi0 << s) | (lo0 >> (32u - s)), lo0 << s);
}
fn bit64(hi: u32, lo: u32, pos0: i32) -> u32 {
    if (pos0 >= 32) { return (hi >> u32(pos0 - 32)) & 1u; }
    return (lo >> u32(pos0)) & 1u;
}
fn stickyBelow(hi: u32, lo: u32, pos0: i32) -> u32 {
    if (pos0 <= 0) { return 0u; }
    if (pos0 >= 64) { return select(0u, 1u, (hi | lo) != 0u); }
    if (pos0 >= 32) {
        let s = u32(pos0) - 32u;
        return select(0u, 1u, lo != 0u) | select(0u, 1u, (hi & ((1u << s) - 1u)) != 0u);
    }
    let s = u32(pos0);
    return select(0u, 1u, (lo & ((1u << s) - 1u)) != 0u);
}
// correctly-rounded f32 fma: rn(a*b + c), pure integer
fn fma_int(a: f32, b: f32, c: f32) -> f32 {
    let ba = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let bc = bitcast<u32>(c);
    let ea0 = i32((ba >> 23u) & 0xFFu);
    let eb0 = i32((bb >> 23u) & 0xFFu);
    let ec0 = i32((bc >> 23u) & 0xFFu);
    if (ea0 == 255 || eb0 == 255 || ec0 == 255) { return a * b + c; }
    var ma = ba & 0x7FFFFFu;
    var mb = bb & 0x7FFFFFu;
    var mc = bc & 0x7FFFFFu;
    var ea = ea0; var eb = eb0; var ec = ec0;
    if (ea0 == 0) { ea = 1; } else { ma = ma | 0x800000u; }
    if (eb0 == 0) { eb = 1; } else { mb = mb | 0x800000u; }
    if (ec0 == 0) { ec = 1; } else { mc = mc | 0x800000u; }
    let sa = ba >> 31u;
    let sbS = bb >> 31u;
    let scS = bc >> 31u;
    if (ma == 0u || mb == 0u) { return c; }
    let a1 = ma >> 12u; let a0 = ma & 0xFFFu;
    let b1 = mb >> 12u; let b0 = mb & 0xFFFu;
    let t0 = a0 * b0;
    let t1 = a0 * b1 + a1 * b0;
    let t2 = a1 * b1;
    let c1w = (t0 >> 12u) + t1;
    let w0 = t0 & 0xFFFu;
    let w1 = c1w & 0xFFFu;
    let w2 = (c1w >> 12u) + t2;
    var pHi = w2 >> 8u;
    var pLo = ((w2 & 0xFFu) << 24u) | (w1 << 12u) | w0;
    var ep = ea + eb - 300;
    if (pHi < 0x8000u) {
        pHi = (pHi << 1u) | (pLo >> 31u);
        pLo = pLo << 1u;
        ep = ep - 1;
    }
    let EP = ep + 47;
    var E = EP + 4;
    var hasC = (mc != 0u);
    if (hasC) { E = max(E, ec - 127 + 4); }
    let shP = E - EP;
    var sticky = 0u;
    var Mp = shl64(pHi, pLo, 16 - shP);
    if (shP > 16) {
        let r = shr64(pHi, pLo, shP - 16);
        Mp = vec2<u32>(r.x, r.y);
        sticky = r.z;
    }
    var McHi = 0u; var McLo = 0u;
    if (hasC) {
        let shC = E - (ec - 127);
        if (shC <= 40) {
            let r = shl64(0u, mc, 40 - shC);
            McHi = r.x; McLo = r.y;
        } else {
            let r = shr64(0u, mc, shC - 40);
            McHi = r.x; McLo = r.y;
            sticky = sticky | r.z;
        }
    }
    let spSign = sa ^ sbS;
    var rHi = 0u; var rLo = 0u; var rSign = 0u;
    if (!hasC) {
        rHi = Mp.x; rLo = Mp.y; rSign = spSign;
    } else if (spSign == scS) {
        let lo = Mp.y + McLo;
        let carry = select(0u, 1u, lo < Mp.y);
        rHi = Mp.x + McHi + carry;
        rLo = lo;
        rSign = spSign;
    } else {
        var gt = (Mp.x > McHi) || (Mp.x == McHi && Mp.y > McLo);
        var eq = (Mp.x == McHi && Mp.y == McLo);
        if (eq) { return bitcast<f32>(0u); }
        var bigHi = Mp.x; var bigLo = Mp.y; var smlHi = McHi; var smlLo = McLo;
        rSign = spSign;
        if (!gt) {
            bigHi = McHi; bigLo = McLo; smlHi = Mp.x; smlLo = Mp.y;
            rSign = scS;
        }
        let borrow = select(0u, 1u, bigLo < smlLo);
        rHi = bigHi - smlHi - borrow;
        rLo = bigLo - smlLo;
    }
    if (rHi == 0u && rLo == 0u) { return bitcast<f32>(0u); }
    var lb: i32;
    if (rHi != 0u) { lb = lead32(rHi) + 32; } else { lb = lead32(rLo); }
    let F = E - 63;
    var eUnb = F + lb;
    let signBit = rSign << 31u;
    if (eUnb >= 128) { return bitcast<f32>(signBit | 0x7F800000u); }
    if (eUnb >= -126) {
        var mR: u32;
        let R = lb - 23;
        if (R <= 0) {
            let r = shl64(rHi, rLo, -R);
            mR = r.y;
        } else {
            if (R >= 32) { mR = rHi >> u32(R - 32); }
            else { mR = (rHi << u32(32 - R)) | (rLo >> u32(R)); }
            var g = 0u; var rr = 0u; var st = sticky;
            if (R >= 2) {
                g = bit64(rHi, rLo, R - 1);
                rr = bit64(rHi, rLo, R - 2);
                st = st | stickyBelow(rHi, rLo, R - 2);
            } else {
                g = bit64(rHi, rLo, 0);
            }
            if (g != 0u && (rr != 0u || st != 0u || (mR & 1u) == 1u)) {
                mR = mR + 1u;
                if (mR == 0x1000000u) { mR = 0x800000u; eUnb = eUnb + 1; }
                if (eUnb >= 128) { return bitcast<f32>(signBit | 0x7F800000u); }
            }
        }
        return bitcast<f32>(signBit | (u32(eUnb + 127) << 23u) | (mR & 0x7FFFFFu));
    }
    let shiftS = F + 149;
    var k: u32;
    if (shiftS >= 0) {
        let r = shl64(rHi, rLo, shiftS);
        k = r.y;
    } else {
        let t = -shiftS;
        if (t >= 64) { return bitcast<f32>(signBit); }
        let r = shr64(rHi, rLo, t);
        k = r.y;
        var g = 0u; var rr = 0u; var st = sticky;
        if (t >= 2) {
            g = bit64(rHi, rLo, t - 1);
            rr = bit64(rHi, rLo, t - 2);
            st = st | stickyBelow(rHi, rLo, t - 2);
        } else {
            g = bit64(rHi, rLo, 0);
        }
        if (g != 0u && (rr != 0u || st != 0u || (k & 1u) == 1u)) {
            k = k + 1u;
            if (k == 0x800000u) { return bitcast<f32>(signBit | 0x800000u); }
        }
    }
    if (k == 0u) { return bitcast<f32>(signBit); }
    return bitcast<f32>(signBit | k);
}
// rn(A * 2^(q-126)) for normal A > 0, q in [0,252]: power-of-two scaling with
// integer GRS rounding in the subnormal grid, because the driver's OpFMul
// flushes subnormal outputs and this must not.
fn scale_pow2(Abits: u32, q: u32) -> u32 {
    let sgn = Abits & 0x80000000u;
    if ((Abits & 0x7FFFFFFFu) == 0u) { return sgn; }
    let mA = (Abits & 0x7FFFFFu) | 0x800000u;
    let eA = i32((Abits >> 23u) & 0xFFu) - 127;
    let eRes = eA + i32(q) - 126;
    if (eRes >= 128) { return sgn | 0x7F800000u; }
    if (eRes >= -126) {
        return sgn | (u32(eRes + 127) << 23u) | (Abits & 0x7FFFFFu);
    }
    let s = -126 - eRes;
    if (s >= 25) { return sgn; }
    let k = mA >> u32(s);
    var kR = k;
    var g = 0u; var rr = 0u; var st = 0u;
    if (s >= 2) {
        g = (mA >> u32(s - 1)) & 1u;
        rr = (mA >> u32(s - 2)) & 1u;
        st = select(0u, 1u, (mA & ((1u << u32(s - 2)) - 1u)) != 0u);
    } else {
        g = (mA >> u32(s - 1)) & 1u;
    }
    if (g != 0u && (rr != 0u || st != 0u || (k & 1u) == 1u)) { kR = k + 1u; }
    if (kR >= 0x800000u) { return sgn | 0x800000u; }
    return sgn | kR;
}
fn expf_bt(x: f32) -> f32 {
    let f5 = clamp(fma_int(x, bitcast<f32>(0x3BBB989Du), 0.5), 0.0, 1.0);
    let b5 = bitcast<u32>(f5);
    var f8 = 12582913u;
    if ((b5 & 0x7FFFFFFFu) != 0u) {
        let raw_e = i32((b5 >> 23u) & 0xFFu) - 127;
        let m = select(b5 & 0x7FFFFFu, (b5 & 0x7FFFFFu) | 0x800000u, raw_e != -127);
        let ee = select(-126, raw_e, raw_e != -127);
        let pp = (m << 8u) - (m << 2u);
        let shift = u32(23 - ee);
        let q = select(0u, pp >> shift, shift < 32u);
        f8 = 12582913u + q;
    }
    let f8f = bitcast<f32>(0x4B000000u | (f8 - 0x800000u));
    let f10 = -(f8f + bitcast<f32>(0xCB40007Fu));
    let f12 = fma_int(x, bitcast<f32>(0x3FB8AA3Bu), f10);
    let f14 = fma_int(x, bitcast<f32>(0x32A57060u), f12);
    let q = f8 - 12582913u;
    return bitcast<f32>(scale_pow2(bitcast<u32>(exp2(f14)), q));
}
";

pub fn gqa_decode_single(nqh: usize, nkvh: usize, d: usize, bs: usize, cap: usize) -> String {
    let tchunks = (bs / d).max(1);
    assert_eq!(d % 2, 0);
    assert_eq!(bs % d, 0);
    format!(
        "{HALF_AT}
{EXP_BT}
struct Cfg {{ cur_len: u32, max_seq: u32, scale: f32, _p: f32 }};

@group(0) @binding(0) var<storage, read>       Q4:  array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       KC4: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       VC:  array<u32>;
@group(0) @binding(3) var<storage, read_write> Out: array<u32>;
@group(0) @binding(4) var<uniform>             cfg: Cfg;

const D: u32 = {d}u;
const D2: u32 = {d2}u;
const D4: u32 = {d4}u;
const BS: u32 = {bs}u;
const TCH: u32 = {tchunks}u;
const REP: u32 = {rep}u;
const CAP: u32 = {cap}u;

var<workgroup> sc:      array<f32, {cap}>;
var<workgroup> partial: array<f32, {d} * {tchunks}>;
var<workgroup> red_max: array<f32, {bs}>;
var<workgroup> red_sum: array<f32, {bs}>;

@compute @workgroup_size({bs})
fn gqa(@builtin(workgroup_id) wgid: vec3<u32>,
       @builtin(local_invocation_id) lid: vec3<u32>) {{
    let qh = wgid.x;
    let kh = qh / REP;
    let qbase = qh * D2;
    let kbase = kh * cfg.max_seq * D2;

    // Stage 1 — scores[t] = (Q . K[t]) * scale; four f16 words per `vec4` load.
    let q4 = qbase >> 2u;
    for (var t = lid.x; t < cfg.cur_len; t = t + BS) {{
        var dot = 0.0;
        let row4 = (kbase + t * D2) >> 2u;
        for (var j4 = 0u; j4 < D4; j4 = j4 + 1u) {{
            let qv = Q4[q4 + j4];
            let kv = KC4[row4 + j4];
            let q0 = unpack_h(qv.x);
            let k0 = unpack_h(kv.x);
            dot = dot + (q0.x * k0.x + q0.y * k0.y);
            let q1 = unpack_h(qv.y);
            let k1 = unpack_h(kv.y);
            dot = dot + (q1.x * k1.x + q1.y * k1.y);
            let q2 = unpack_h(qv.z);
            let k2 = unpack_h(kv.z);
            dot = dot + (q2.x * k2.x + q2.y * k2.y);
            let q3 = unpack_h(qv.w);
            let k3 = unpack_h(kv.w);
            dot = dot + (q3.x * k3.x + q3.y * k3.y);
        }}
        sc[t] = dot * cfg.scale;
    }}
    workgroupBarrier();

    // Stage 2 — row max
    var lmax = bitcast<f32>(0xFF800000u);
    for (var t = lid.x; t < cfg.cur_len; t = t + BS) {{
        if (sc[t] > lmax) {{ lmax = sc[t]; }}
    }}
    red_max[lid.x] = lmax;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + s]); }}
        workgroupBarrier();
    }}
    let row_max = red_max[0];
    workgroupBarrier();

    // Stage 3 — exp + sum
    var lsum = 0.0;
    for (var t = lid.x; t < cfg.cur_len; t = t + BS) {{
        let e = expf_bt(sc[t] - row_max);
        sc[t] = e;
        lsum = lsum + e;
    }}
    red_sum[lid.x] = lsum;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + s]; }}
        workgroupBarrier();
    }}
    let inv_sum = 1.0 / red_sum[0];
    workgroupBarrier();

    // Stage 4 — partial[t_idx][j] then cross-t_chunk merge.  One thread owns a
    // *pair* of dims (both halves of one f16 word), so one load and unpack feed
    // two FMAs; each dim walks the stride-`TCH` key set in order, so every
    // accumulator keeps its summation order.
    let jp = lid.x % D2;
    let t_idx = lid.x / D2;
    if (t_idx < TCH) {{
        var a0 = 0.0;
        var a1 = 0.0;
        for (var t = t_idx; t < cfg.cur_len; t = t + TCH) {{
            let row = kbase + t * D2;
            let v = unpack_h(VC[row + jp]);
            a0 = a0 + sc[t] * v.x;
            a1 = a1 + sc[t] * v.y;
        }}
        partial[t_idx * D + jp * 2u] = a0;
        partial[t_idx * D + jp * 2u + 1u] = a1;
    }}
    workgroupBarrier();

    // One thread per f16 output word; the per-element accumulation order is
    // unchanged.
    if (lid.x < D2) {{
        var a0 = 0.0;
        var a1 = 0.0;
        for (var ti = 0u; ti < TCH; ti = ti + 1u) {{
            a0 = a0 + partial[ti * D + lid.x * 2u];
            a1 = a1 + partial[ti * D + lid.x * 2u + 1u];
        }}
        Out[qbase + lid.x] = pack_h(vec2<f32>(a0 * inv_sum, a1 * inv_sum));
    }}
}}
",
        d2 = d / 2,
        d4 = d / 8,
        rep = nqh / nkvh,
    )
}

/// `gu` holds `[gate | up]` per row, output `up * gate * sigmoid(gate)`.  The
/// sigmoid is evaluated as `exp2(-g * log2(e))` rather than `exp(x)`.
pub fn silu_mul_split(inter: usize) -> String {
    assert!(inter % 2 == 0);
    let inter2 = inter / 2;
    let threads = 256usize;
    let grid = inter2.div_ceil(threads);
    let _ = grid;
    format!(
        "@group(0) @binding(0) var<storage, read>       Gu:  array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

struct Cfg {{ inter2: u32, total2: u32, gx: u32, _b: u32 }};

const INTER2: u32 = {inter2}u;
const LOG2E: f32 = {log2e};
const THREADS: u32 = {threads}u;

@compute @workgroup_size({threads})
fn silu_mul_split(@builtin(global_invocation_id) gid: vec3<u32>) {{
    // Two grid axes: `rows · inter/2` words overflows wgpu's 65535-per-dimension
    // limit for prefills longer than ~5.8 minutes, so x is capped and y continues
    // the flat index space (`gx · THREADS` words per row).
    let i = gid.x + gid.y * (cfg.gx * THREADS);
    if (i >= cfg.total2) {{ return; }}
    // rows of [row_gate | row_up]: decode runs one row, prefill s rows
    let row = i / cfg.inter2;
    let c = i - row * cfg.inter2;
    let base = row * cfg.inter2 * 2u;
    let gv = unpack_h(Gu[base + c]);
    let uv = unpack_h(Gu[base + cfg.inter2 + c]);
    let sx = 1.0 / (1.0 + exp2(-gv.x * LOG2E));
    let sy = 1.0 / (1.0 + exp2(-gv.y * LOG2E));
    Out[i] = pack_h(vec2<f32>(gv.x * sx * uv.x, gv.y * sy * uv.y));
}}
",
        log2e = format_f32(LOG2_E),
    )
}

/// Widen a run of the 16-bit storage buffer into an f32 buffer — the
/// measurement-only companion of `QALIGN_DUMP_LAYERS` (see
/// [`crate::decoder::WgpuTextDecoder::prefill`]).
///
/// Values, not bits: the copy goes through `unpack_h`, which is the one place
/// the storage format is read, so an f16 run and a bf16 run produce the same
/// kind of quantity and can be compared like for like.  Two grid axes for the
/// same reason `silu_mul_split` has them — a long prefill's word count overflows
/// one dimension's 65535 cap.
pub fn widen_h16_f32() -> String {
    "\
@group(0) @binding(0) var<storage, read>       Src: array<u32>;
@group(0) @binding(1) var<storage, read_write> Dst: array<f32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

struct Cfg { words: u32, src0: u32, dst0: u32, gx: u32 };

const THREADS: u32 = 256u;

@compute @workgroup_size(256)
fn widen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * (cfg.gx * THREADS);
    if (i >= cfg.words) { return; }
    let v = unpack_h(Src[cfg.src0 + i]);
    // element 2j is the low half of word j and 2j+1 the high half -- the packing
    // `pack_h` writes and `H16::to_f32` reads back
    Dst[cfg.dst0 + 2u * i] = v.x;
    Dst[cfg.dst0 + 2u * i + 1u] = v.y;
}
"
        .to_string()
}

/// Long-context (cur_len > 1024) split-K attention, phase 1: one workgroup per
/// (q_head, chunk).  Chunk-local scores → chunk max/sum via the shared
/// block-reduction tree → unnormalized partial numerator.  Block size is fixed
/// at 256; `t_split = 256/d` threads cooperate per output element.
///
/// Empty chunks (t_start >= cur_len) write max=-inf, sum=0, partial=0; the merge
/// kernel's `> -inf` guard depends on it.
pub fn gqa_decode_split_p1(nqh: usize, nkvh: usize, d: usize, chunk: usize) -> String {
    assert_eq!(d % 2, 0);
    let t_split = 256 / d;
    format!(
        "{HALF_AT}
{EXP_BT}
struct Cfg {{ cur_len: u32, max_seq: u32, scale: f32, n_chunks: u32 }};

@group(0) @binding(0) var<storage, read>       Q4:   array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       KC4:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       VC:   array<u32>;
@group(0) @binding(3) var<storage, read_write> POut: array<f32>;
@group(0) @binding(4) var<storage, read_write> PMax: array<f32>;
@group(0) @binding(5) var<storage, read_write> PSum: array<f32>;
@group(0) @binding(6) var<uniform>             cfg:  Cfg;

const D: u32 = {d}u;
const D2: u32 = {d2}u;
const D4: u32 = {d4}u;
const CHUNK: u32 = {chunk}u;
const BS: u32 = 256u;
const T_SPLIT: u32 = {t_split}u;
const REP: u32 = {rep}u;

var<workgroup> sc:      array<f32, {chunk}u>;
var<workgroup> partial: array<f32, {d} * {t_split}u>;
var<workgroup> red_max: array<f32, 256u>;
var<workgroup> red_sum: array<f32, 256u>;

@compute @workgroup_size(256)
fn gqa_split_p1(@builtin(workgroup_id) wgid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {{
    let qh = wgid.x;
    let by = wgid.y;
    let kh = qh / REP;
    let qbase = qh * D2;
    let kbase = kh * cfg.max_seq * D2;
    let t_start = by * CHUNK;
    let mi = qh * cfg.n_chunks + by;

    if (t_start >= cfg.cur_len) {{
        if (lid.x == 0u) {{
            PMax[mi] = bitcast<f32>(0xFF800000u);
            PSum[mi] = 0.0;
        }}
        if (lid.x < D) {{
            POut[mi * D + lid.x] = 0.0;
        }}
        return;
    }}
    let chunk_len = min(CHUNK, cfg.cur_len - t_start);

    // Stage 1 — scores[t] = (Q . K[t_start + t]) * scale, chunk-local t.
    // Read through `vec4<u32>` (four words, 16 B per lane); each word is unpacked
    // and accumulated in a fixed sequence, so the f32 reduction order is stable.
    let q4 = qbase >> 2u;
    for (var t = lid.x; t < chunk_len; t = t + BS) {{
        var dot = 0.0;
        let row4 = (kbase + (t_start + t) * D2) >> 2u;
        for (var j4 = 0u; j4 < D4; j4 = j4 + 1u) {{
            let qv = Q4[q4 + j4];
            let kv = KC4[row4 + j4];
            let q0 = unpack_h(qv.x);
            let k0 = unpack_h(kv.x);
            dot = dot + (q0.x * k0.x + q0.y * k0.y);
            let q1 = unpack_h(qv.y);
            let k1 = unpack_h(kv.y);
            dot = dot + (q1.x * k1.x + q1.y * k1.y);
            let q2 = unpack_h(qv.z);
            let k2 = unpack_h(kv.z);
            dot = dot + (q2.x * k2.x + q2.y * k2.y);
            let q3 = unpack_h(qv.w);
            let k3 = unpack_h(kv.w);
            dot = dot + (q3.x * k3.x + q3.y * k3.y);
        }}
        sc[t] = dot * cfg.scale;
    }}
    workgroupBarrier();

    // Stage 2 — chunk max
    var lmax = bitcast<f32>(0xFF800000u);
    for (var t = lid.x; t < chunk_len; t = t + BS) {{
        if (sc[t] > lmax) {{ lmax = sc[t]; }}
    }}
    red_max[lid.x] = lmax;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + s]); }}
        workgroupBarrier();
    }}
    let chunk_max = red_max[0];
    workgroupBarrier();

    // Stage 3 — exp + sum
    var lsum = 0.0;
    for (var t = lid.x; t < chunk_len; t = t + BS) {{
        let e = expf_bt(sc[t] - chunk_max);
        sc[t] = e;
        lsum = lsum + e;
    }}
    red_sum[lid.x] = lsum;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{ red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + s]; }}
        workgroupBarrier();
    }}
    let chunk_sum = red_sum[0];
    workgroupBarrier();

    // Stage 4 — partial numerator, unnormalized; V read at the global position.
    // One thread owns two dims sharing an f16 word, so one load and unpack feed
    // two FMAs; each dim keeps its stride-`T_SPLIT` key order.
    let jp = lid.x % D2;
    let t_idx = lid.x / D2;
    if (t_idx < T_SPLIT) {{
        var a0 = 0.0;
        var a1 = 0.0;
        for (var t = t_idx; t < chunk_len; t = t + T_SPLIT) {{
            let row = kbase + (t_start + t) * D2;
            let v = unpack_h(VC[row + jp]);
            a0 = a0 + sc[t] * v.x;
            a1 = a1 + sc[t] * v.y;
        }}
        partial[t_idx * D + jp * 2u] = a0;
        partial[t_idx * D + jp * 2u + 1u] = a1;
    }}
    workgroupBarrier();

    if (lid.x < D) {{
        var acc = 0.0;
        for (var ti = 0u; ti < T_SPLIT; ti = ti + 1u) {{
            acc = acc + partial[ti * D + lid.x];
        }}
        POut[mi * D + lid.x] = acc;
    }}
    if (lid.x == 0u) {{
        PMax[mi] = chunk_max;
        PSum[mi] = chunk_sum;
    }}
}}
",
        d2 = d / 2,
        d4 = d / 8,
        t_split = t_split,
        rep = nqh / nkvh,
    )
}

/// Merge phase: online-softmax correction across chunks.  One workgroup per
/// q_head, one thread per f16 word.
pub fn gqa_split_merge(d: usize) -> String {
    format!(
        "struct Cfg {{ n_chunks: u32, _a: u32, _b: u32, _c: u32 }};

{EXP_BT}
@group(0) @binding(0) var<storage, read_write> Out:  array<u32>;
@group(0) @binding(1) var<storage, read>       POut: array<f32>;
@group(0) @binding(2) var<storage, read>       PMax: array<f32>;
@group(0) @binding(3) var<storage, read>       PSum: array<f32>;
@group(0) @binding(4) var<uniform>             cfg:  Cfg;

const D: u32 = {d}u;
const D2: u32 = {d2}u;

@compute @workgroup_size({d})
fn gqa_merge(@builtin(workgroup_id) wgid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {{
    let qh = wgid.x;
    let n = cfg.n_chunks;
    let maxbase = qh * n;
    let neg_inf = bitcast<f32>(0xFF800000u);

    var g_max = neg_inf;
    for (var c = 0u; c < n; c = c + 1u) {{
        if (PMax[maxbase + c] > g_max) {{ g_max = PMax[maxbase + c]; }}
    }}
    var g_sum = 0.0;
    for (var c = 0u; c < n; c = c + 1u) {{
        if (PMax[maxbase + c] > neg_inf) {{
            g_sum = g_sum + PSum[maxbase + c] * expf_bt(PMax[maxbase + c] - g_max);
        }}
    }}
    let inv = 1.0 / g_sum;

    if (lid.x < D2) {{
        var a0 = 0.0;
        var a1 = 0.0;
        for (var c = 0u; c < n; c = c + 1u) {{
            if (PMax[maxbase + c] > neg_inf) {{
                let w = expf_bt(PMax[maxbase + c] - g_max) * inv;
                a0 = a0 + w * POut[(maxbase + c) * D + lid.x * 2u];
                a1 = a1 + w * POut[(maxbase + c) * D + lid.x * 2u + 1u];
            }}
        }}
        Out[qh * D2 + lid.x] = pack_h(vec2<f32>(a0, a1));
    }}
}}
",
        d2 = d / 2,
    )
}

/// Single block of 1024; strict `>` so the lowest index wins a tie.
pub fn argmax_into_slot() -> String {
    format!(
        "{HALF_AT}
struct Cfg {{ n: u32, slot: u32, _a: u32, _b: u32 }};

@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<i32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

const BS: u32 = 1024u;

var<workgroup> smax: array<f32, 1024>;
var<workgroup> sidx: array<i32, 1024>;

@compute @workgroup_size(1024)
fn argmax(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let n2 = cfg.n >> 1u;
    var lmax = bitcast<f32>(0xFF800000u);
    var lidx = 0;
    for (var i = lid.x; i < n2; i = i + BS) {{
        let v = unpack_h(X[i]);
        let ix = i * 2u;
        if (v.x > lmax) {{ lmax = v.x; lidx = i32(ix); }}
        if (v.y > lmax) {{ lmax = v.y; lidx = i32(ix + 1u); }}
    }}
    if ((cfg.n & 1u) == 1u && lid.x == 0u) {{
        let e = half_at(unpack_h(X[cfg.n >> 1u]), cfg.n - 1u);
        if (e > lmax) {{ lmax = e; lidx = i32(cfg.n - 1u); }}
    }}
    smax[lid.x] = lmax;
    sidx[lid.x] = lidx;
    workgroupBarrier();
    for (var s = BS >> 1u; s > 0u; s = s >> 1u) {{
        if (lid.x < s) {{
            if (smax[lid.x + s] > smax[lid.x]) {{
                smax[lid.x] = smax[lid.x + s];
                sidx[lid.x] = sidx[lid.x + s];
            }}
        }}
        workgroupBarrier();
    }}
    if (lid.x == 0u) {{ Out[cfg.slot] = sidx[0]; }}
}}
"
    )
}

/// Gather one embedding row using a token id that never left the GPU.
pub fn embed_lookup_single() -> String {
    format!(
        "struct Cfg {{ slot: u32, d2: u32, _a: u32, _b: u32 }};

@group(0) @binding(0) var<storage, read>       Table: array<u32>;
@group(0) @binding(1) var<storage, read>       Ids:   array<i32>;
@group(0) @binding(2) var<storage, read_write> Out:   array<u32>;
@group(0) @binding(3) var<uniform>             cfg:   Cfg;

const BS: u32 = 1024u;

@compute @workgroup_size(1024)
fn embed(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let id = u32(Ids[cfg.slot]);
    let base = id * cfg.d2;
    for (var w = lid.x; w < cfg.d2; w = w + BS) {{
        Out[w] = Table[base + w];
    }}
}}
"
    )
}

/// Format an f32 so WGSL parses back the exact same value (hex float, exact).
pub fn format_f32(v: f32) -> String {
    let bits = v.to_bits();
    let sign = if bits >> 31 == 1 { "-" } else { "" };
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x007F_FFFF;
    if exp == 0 && man == 0 {
        return format!("{sign}0.0");
    }
    // WGSL hex float literal: 0x1.<mantissa>p<exponent>.  Six hex digits are 24
    // fractional bits but the f32 mantissa is 23, so emit `man << 1` (always <
    // 2^24) — otherwise every non-dyadic constant comes out half-ish wrong
    // (1.4427 -> 1.2213, which silently bent every SiLU in the network).
    let man = man << 1;
    let e = exp - 127;
    format!("{sign}0x1.{man:06x}p{e}")
}


/// Prefill GEMM:  C[m,n] f16 = A[m,k] f16 × W[n,k]ᵀ (or × W[n,k] when
/// `transb`), f32 accumulate, one f16 rounding.  Tile 128×128 per workgroup
/// (16×16 threads, 8×8 micro-tile), BK=16, software-pipelined (next tile
/// prefetched into registers during compute).  `beta=1` folds the residual add
/// into the epilogue (acc + f16(C_in), one rounding).
///
/// The tile geometry, as a single source of truth.  The kernel's own
/// `BM`/`BN`/`BK` constants and every caller's padding and dispatch grids must
/// agree: passing 64 where the kernel uses 128 leaves half of each axis
/// uncomputed and the result is silently wrong rather than an error.
pub const PREFILL_GEMM_TM: usize = 8;
pub const PREFILL_GEMM_TN: usize = 8;
pub const PREFILL_GEMM_BK: usize = 16;
/// M tile width — `m` must be padded to a multiple of this.
pub const PREFILL_GEMM_BM: usize = 16 * PREFILL_GEMM_TM;
/// N tile width — `n` (the weight's row count) must be padded to a multiple.
pub const PREFILL_GEMM_BN: usize = 16 * PREFILL_GEMM_TN;

/// `bias = 1` adds the per-column `Bias` vector (binding 4) before the `beta`
/// residual.  `transb=1` reads W as a [k, n] f16 matrix instead of [n, k]
/// (attention AV: V is [cur, d]).
///
/// `ldc` = C row stride in elements; `bsa`/`bsb` = per-batch operand strides in
/// words (the shader indexes `array<u32>` directly); `bsc` = the per-batch C
/// stride in elements (the epilogue divides the flat C index by 2).  Batch index
/// is `wgid.z`.  `lda` is the A row stride in elements — `k` except the slabbed
/// AV, whose A operand (a score slab) is `T` wide while the tile it sweeps is
/// narrower.  `row0` shifts the B operand's rows (K or V rows = key positions)
/// and, on the causal variants, the diagonal they test against.
///
/// The tile geometry comes from the `PREFILL_GEMM_*` constants above; callers
/// must pad `m`, `n` and the operand row strides with those same values.
pub fn prefill_gemm(transb: bool, beta: bool) -> String {
    prefill_gemm_impl(transb, beta, false, false, false)
}

/// [`prefill_gemm`] with the causal-attention tile skip: a tile wholly above the
/// diagonal (`n0 + row0 > m0 + BM - 1`, `row0` = the key offset of the score
/// slab) is entirely masked by the causal softmax, which never reads columns
/// past `row + 1`, so the skip is bit-identical.
pub fn prefill_gemm_causal() -> String {
    prefill_gemm_impl(false, false, false, true, false)
}

/// [`prefill_gemm_bias`]-style AV form (`transb = 1`) with the causal *k* bound:
/// the A operand is the softmax output, whose columns past `row + 1` are exactly
/// zero, so a row block only has to sweep `k < m0 + BM - row0` (`row0` = the
/// slab's key offset; zero on the flat path).  Skipping exact zeros from a sum is
/// bit-identical.
pub fn prefill_gemm_causal_av() -> String {
    prefill_gemm_impl(true, false, false, false, true)
}

/// As [`prefill_gemm`], plus the per-column bias add (binding 4).  A separate
/// entry point rather than a flag on the shared one: the binding must be
/// *declared* only for the variants that read it, and a declared-but-unbound
/// binding fails pipeline validation even when the read is dead code.
pub fn prefill_gemm_bias(transb: bool, beta: bool) -> String {
    prefill_gemm_impl(transb, beta, true, false, false)
}

fn prefill_gemm_impl(
    transb: bool,
    beta: bool,
    bias: bool,
    causal_skip: bool,
    causal_k: bool,
) -> String {
    let tm = PREFILL_GEMM_TM;
    let tn = PREFILL_GEMM_TN;
    let bk = PREFILL_GEMM_BK;
    let bm = 16 * tm;
    let bn = 16 * tn;
    let pad = bk + 1;
    let n_as = bm * bk / 256;
    let n_bs = bn * bk / 256;

    let mut s = String::new();
    s.push_str(
        "struct GDims { m: u32, n: u32, k: u32, ldc: u32, bsa: u32, bsb: u32, bsc: u32, beta: u32, row0: u32, lda: u32 };\n\
         @group(0) @binding(0) var<storage, read>       A: array<u32>;\n\
         @group(0) @binding(1) var<storage, read>       W: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C: array<u32>;\n\
         @group(0) @binding(3) var<uniform>             gd: GDims;\n",
    );
    if bias {
        s.push_str("@group(0) @binding(4) var<storage, read> Bias: array<u32>;\n");
    }
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {bk}u;\n\
         const PAD: u32 = {pad}u;\nconst TM: u32 = {tm}u;\nconst TN: u32 = {tn}u;\n\
         const TRANSB: u32 = {}u;\nconst BETA: u32 = {}u;\nconst BIAS: u32 = {}u;\n\
         const CAUSAL: u32 = {}u;\nconst CAUSAL_K: u32 = {}u;\n",
        u32::from(transb),
        u32::from(beta),
        u32::from(bias),
        u32::from(causal_skip),
        u32::from(causal_k),
    ));
    s.push_str(&format!("var<workgroup> As: array<f32, {}>;\n", bm * pad));
    s.push_str(&format!("var<workgroup> Bs: array<f32, {}>;\n", bn * pad));
    s.push_str(
        "fn halve(w: u32, odd: bool) -> f32 {\n\
         \x20 let p = unpack_h(w);\n\
         \x20 return select(p.x, p.y, odd);\n\
         }\n",
    );

    // A element (m-row r, k-col c): word abase + r*kk + c/2, half c&1.
    // W element: transb=0 -> [n, k] f16: word wb + r*kk + c/2, half c&1;
    //   transb=1 -> [k, n] f16: element (r, c) at (k0+c)*(n/2) + (n0+r)/2,
    //   half (n0+r)&1 (n padded even, n0 multiple of BN).
    s.push_str(
        "@compute @workgroup_size(16, 16)\n\
         fn gemm(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.y * BM;\n let n0 = wid.x * BN;\n let kk = gd.k / 2u;\n\
         let alda = gd.lda / 2u;\n\
         let abase = wid.z * gd.bsa;\n let wb = wid.z * gd.bsb;\n\
         let cbase = wid.z * gd.bsc;\n\
         if (CAUSAL == 1u && n0 + gd.row0 > m0 + BM - 1u) { return; }\n\
         let klim = select(gd.k, min(gd.k, max(m0 + BM, gd.row0) - gd.row0), CAUSAL_K == 1u);\n",
    );

    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j} = 0.0;\n"));
        }
    }

    let load_as = |kx: &str| -> String {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!(
                "  As[(ty + {}u) * PAD + tx] = halve(A[abase + (m0 + ty + {}u) * alda + ({kx} + tx) / 2u], (tx & 1u) == 1u);\n",
                e * 16, e * 16
            ));
        }
        t
    };
    let load_bs = |kx: &str| -> String {
        let mut t = String::new();
        if !transb {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "  Bs[(ty + {}u) * PAD + tx] = halve(W[wb + (gd.row0 + n0 + ty + {}u) * kk + ({kx} + tx) / 2u], (tx & 1u) == 1u);\n",
                    e * 16, e * 16
                ));
            }
        } else {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "  Bs[(ty + {}u) * PAD + tx] = halve(W[wb + (gd.row0 + {kx} + tx) * (gd.n / 2u) + (n0 + ty + {}u) / 2u], ((n0 + ty + {}u) & 1u) == 1u);\n",
                    e * 16, e * 16, e * 16
                ));
            }
        }
        t
    };
    let pf_as = |kx: &str| -> String {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!(
                "   pfa{e} = A[abase + (m0 + ty + {}u) * alda + ({kx} + tx) / 2u];\n", e * 16
            ));
        }
        t
    };
    let pf_bs = |kx: &str| -> String {
        let mut t = String::new();
        if !transb {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "   pfb{e} = W[wb + (gd.row0 + n0 + ty + {}u) * kk + ({kx} + tx) / 2u];\n", e * 16
                ));
            }
        } else {
            for e in 0..n_bs {
                t.push_str(&format!(
                    "   pfb{e} = W[wb + (gd.row0 + {kx} + tx) * (gd.n / 2u) + (n0 + ty + {}u) / 2u];\n", e * 16
                ));
            }
        }
        t
    };
    let pf_decl = || {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!("  var pfa{e}: u32 = 0u;\n"));
        }
        for e in 0..n_bs {
            t.push_str(&format!("  var pfb{e}: u32 = 0u;\n"));
        }
        t
    };
    let store_as = || {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!(
                "   As[(ty + {}u) * PAD + tx] = halve(pfa{e}, (tx & 1u) == 1u);\n", e * 16
            ));
        }
        t
    };
    let store_bs = || {
        let mut t = String::new();
        for e in 0..n_bs {
            // the prefetched word's halves follow the SAME layout as load_bs:
            // transb=0 -> k parity (= tx parity); transb=1 -> n parity of the
            // n-element this word was fetched for.
            let sel = if transb {
                format!("((n0 + ty + {}u) & 1u) == 1u", e * 16)
            } else {
                "(tx & 1u) == 1u".to_string()
            };
            t.push_str(&format!(
                "   Bs[(ty + {}u) * PAD + tx] = halve(pfb{e}, {sel});\n",
                e * 16
            ));
        }
        t
    };
    let compute = || {
        let mut t = String::new();
        t.push_str("  var q: u32 = 0u;\n  loop {\n   if (q >= BK) { break; }\n");
        for i in 0..tm {
            t.push_str(&format!("   let a{i} = As[(ty * {tm}u + {i}u) * PAD + q];\n"));
        }
        for j in 0..tn {
            t.push_str(&format!("   let b{j} = Bs[(tx * {tn}u + {j}u) * PAD + q];\n"));
        }
        for i in 0..tm {
            for j in 0..tn {
                t.push_str(&format!("   c{i}{j} = c{i}{j} + a{i} * b{j};\n"));
            }
        }
        t.push_str("   q = q + 1u;\n  }\n");
        t
    };

    s.push_str(" var k0: u32 = 0u;\n");
    s.push_str(&load_as("k0"));
    s.push_str(&load_bs("k0"));
    s.push_str(" workgroupBarrier();\n");
    s.push_str(" loop {\n  if (k0 >= klim) { break; }\n");
    s.push_str("  let kn = k0 + BK;\n");
    s.push_str(&pf_decl());
    s.push_str("  if (kn < klim) {\n");
    s.push_str(&pf_as("kn"));
    s.push_str(&pf_bs("kn"));
    s.push_str("  }\n");
    s.push_str(&compute());
    s.push_str("  workgroupBarrier();\n");
    s.push_str("  if (kn < klim) {\n");
    s.push_str(&store_as());
    s.push_str(&store_bs());
    s.push_str("  }\n  workgroupBarrier();\n");
    s.push_str("  k0 = kn;\n }\n");

    // epilogue: f16 words along n; a thread owns TN consecutive columns so
    // pairs stay in-thread.  beta=1 adds the f16 C_in before the rounding.
    for i in 0..tm {
        s.push_str(&format!("let row{i} = m0 + ty * {tm}u + {i}u;\n"));
    }
    for i in 0..tm {
        for e in 0..tn / 2 {
            let je = 2 * e;
            let jo = je + 1;
            s.push_str(&format!(
                "let we{i}_{e} = (cbase + row{i} * gd.ldc + n0 + tx * {tn}u + {je}u) / 2u;\n"
            ));
            s.push_str(&format!(
                "var v{i}{e} = vec2<f32>(c{i}{je}, c{i}{jo});\n"
            ));
            // `Bias` is the per-column vector, two columns per word: the pair a
            // thread owns is exactly one word, so no half select is needed.
            if bias {
                s.push_str(&format!(
                    "v{i}{e} = v{i}{e} + unpack_h(Bias[(n0 + tx * {tn}u + {je}u) / 2u]);\n"
                ));
            }
            s.push_str(&format!(
                "if (BETA == 1u) {{\n  let old = unpack_h(C[we{i}_{e}]);\n  v{i}{e} = v{i}{e} + old;\n}}\n"
            ));
            s.push_str(&format!(
                "C[we{i}_{e}] = pack_h(v{i}{e});\n"
            ));
        }
    }
    s.push_str("}\n");
    s
}

/// Causal scaled softmax over prefill scores.  One workgroup per score row; block
/// size `bs` matches `block_for_reduction(n)` so the reduction trees line up.
/// Row `p` (of the head) attends `min(p + 1 - row0, valid)` positions; columns
/// `valid..n_w` are written zero so downstream GEMMs read zeros.
///
/// `row0` is the column offset of the score block this dispatch covers (0 = the
/// whole row): the row index is still absolute, so the causal bound is
/// `p + 1 - row0` clamped at zero.
pub fn softmax_causal(bs: usize) -> String {
    format!(
        "struct Cfg {{ n_w: u32, n_x: u32, valid: u32, m: u32, mp: u32, scale: f32, gx: u32, row0: u32 }};

@group(0) @binding(0) var<storage, read>       X:   array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: Cfg;

const BS: u32 = {bs}u;

{HALF_AT}
var<workgroup> red_max: array<f32, {bs}>;
var<workgroup> red_sum: array<f32, {bs}>;

@compute @workgroup_size({bs})
fn softmax(@builtin(workgroup_id) wgid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>) {{
    // X rows are n_x-strided (tile-padded scores, one row per head*pos);
    // Out uses the BATCHED [head][mp][cur16] layout the AV GEMM expects.
    // Two grid axes: `nqh · s` rows overflow the 65535-per-dimension limit for
    // prefills beyond ~4.2 minutes (x is capped, y continues the row index).
    let row = wgid.x + wgid.y * cfg.gx;
    let head = row / cfg.m;
    let pos = row % cfg.m;
    let base_x = (head * cfg.mp + pos) * cfg.n_x;
    let base_o = (head * cfg.mp + pos) * cfg.n_w;
    let row_in_head = pos;
    // causal bound inside this dispatch's column window; row0 == 0 leaves
    // `min(row_in_head + 1, valid)`
    let valid = min(cfg.valid, max(row_in_head + 1u, cfg.row0) - cfg.row0);
    let scale = cfg.scale;

    // A row left of the whole slab has no live column there, but the AV GEMM of
    // a straddling row block still reads its columns — they have to be zeros.
    // Only the reductions are skipped; the branch is workgroup-uniform (it
    // depends on the row alone), so it may skip the barriers.
    if (valid == 0u) {{
        for (var w = lid.x; w < cfg.n_w; w = w + BS) {{
            Out[base_o + w] = 0u;
        }}
        return;
    }}

    var lmax = bitcast<f32>(0xFF800000u);
    for (var j = lid.x; j < valid; j = j + BS) {{
        let v = half_at(unpack_h(X[base_x + (j >> 1u)]), j);
        let sc = v * scale;
        if (sc > lmax) {{ lmax = sc; }}
    }}
    red_max[lid.x] = lmax;
    workgroupBarrier();
    for (var sh = BS >> 1u; sh > 0u; sh = sh >> 1u) {{
        if (lid.x < sh) {{ red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + sh]); }}
        workgroupBarrier();
    }}
    let row_max = red_max[0];
    workgroupBarrier();

    var lsum = 0.0;
    for (var j = lid.x; j < valid; j = j + BS) {{
        let v = half_at(unpack_h(X[base_x + (j >> 1u)]), j);
        lsum = lsum + exp(v * scale - row_max);
    }}
    red_sum[lid.x] = lsum;
    workgroupBarrier();
    for (var sh = BS >> 1u; sh > 0u; sh = sh >> 1u) {{
        if (lid.x < sh) {{ red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + sh]; }}
        workgroupBarrier();
    }}
    let inv_sum = 1.0 / red_sum[0];
    workgroupBarrier();

    // one thread per f16 word — no read-modify-write races
    for (var w = lid.x; w < cfg.n_w; w = w + BS) {{
        let j0 = w * 2u;
        let j1 = j0 + 1u;
        var e0 = 0.0;
        var e1 = 0.0;
        if (j0 < valid) {{
            let v0 = half_at(unpack_h(X[base_x + w]), j0);
            e0 = exp(v0 * scale - row_max) * inv_sum;
        }}
        if (j1 < valid) {{
            let v1 = half_at(unpack_h(X[base_x + w]), j1);
            e1 = exp(v1 * scale - row_max) * inv_sum;
        }}
        Out[base_o + w] = pack_h(vec2<f32>(e0, e1));
    }}
}}
",
        bs = bs,
    )
}

/// Per-*slab* softmax statistics for the tiled prefill attention: one `(max,
/// Σexp)` pair per score row per key slab, `Stats[slab][row]` as `(max, sum)` f32.
///
/// The slab width is the workgroup size, so an element is read once and held in a
/// register across both reductions; the causal bound is `min(valid, row + 1 - row0)`.
/// The pair recorded here is the pair the slab it describes was normalized by.
///
/// `row0` doubles as the slab index (`row0 / T`), supplying the slot without a
/// second uniform field.
pub fn slab_stats(bs: usize, t: usize) -> String {
    assert!(t.is_power_of_two(), "slab_stats: the slab width must be a power of two");
    assert!(bs.is_power_of_two() && bs <= t, "slab_stats: block size");
    format!(
        "struct StatsCfg {{ n_x: u32, valid: u32, m: u32, mp: u32, scale: f32, row0: u32, gx: u32, rows: u32 }};

@group(0) @binding(0) var<storage, read>       X:     array<u32>;
@group(0) @binding(1) var<storage, read_write> Stats: array<f32>;
@group(0) @binding(2) var<uniform>             cfg:   StatsCfg;

const BS: u32 = {bs}u;
const T: u32 = {t}u;

{HALF_AT}
var<workgroup> red_max: array<f32, BS>;
var<workgroup> red_sum: array<f32, BS>;

@compute @workgroup_size(BS)
fn slab_stats(@builtin(workgroup_id) wgid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {{
    // Two grid axes: `nqh · mp` rows pass the 65535-per-dimension limit at a
    // ~5-minute prefill (the softmax carries the same pair).
    let row = wgid.x + wgid.y * cfg.gx;
    let head = row / cfg.m;
    let pos = row % cfg.m;
    let base_x = (head * cfg.mp + pos) * cfg.n_x;
    let valid = min(cfg.valid, max(pos + 1u, cfg.row0) - cfg.row0);
    // rows are the head-major `head·mp + pos` of the score buffer, not the
    // dispatch's flat row index (those differ: the grid spans `nqh · s`)
    let r = head * cfg.mp + pos;

    // Most rows of a given slab are entirely left of its first column and have
    // no live column.  The branch is workgroup-uniform (`valid` depends on the
    // row only), so it may skip the barriers.
    if (valid == 0u) {{
        if (lid.x == 0u) {{
            Stats[(cfg.row0 / T * cfg.rows + r) * 2u] = bitcast<f32>(0xFF800000u);
            Stats[(cfg.row0 / T * cfg.rows + r) * 2u + 1u] = 0.0;
        }}
        return;
    }}

    var lmax = bitcast<f32>(0xFF800000u);
    for (var j = lid.x; j < valid; j = j + BS) {{
        let sc = half_at(unpack_h(X[base_x + (j >> 1u)]), j) * cfg.scale;
        if (sc > lmax) {{ lmax = sc; }}
    }}
    red_max[lid.x] = lmax;
    workgroupBarrier();
    for (var sh = BS >> 1u; sh > 0u; sh = sh >> 1u) {{
        if (lid.x < sh) {{ red_max[lid.x] = max(red_max[lid.x], red_max[lid.x + sh]); }}
        workgroupBarrier();
    }}
    let row_max = red_max[0];
    workgroupBarrier();

    var lsum = 0.0;
    for (var j = lid.x; j < valid; j = j + BS) {{
        let sc = half_at(unpack_h(X[base_x + (j >> 1u)]), j) * cfg.scale;
        lsum = lsum + exp(sc - row_max);
    }}
    red_sum[lid.x] = lsum;
    workgroupBarrier();
    for (var sh = BS >> 1u; sh > 0u; sh = sh >> 1u) {{
        if (lid.x < sh) {{ red_sum[lid.x] = red_sum[lid.x] + red_sum[lid.x + sh]; }}
        workgroupBarrier();
    }}
    if (lid.x == 0u) {{
        Stats[(cfg.row0 / T * cfg.rows + r) * 2u] = row_max;
        Stats[(cfg.row0 / T * cfg.rows + r) * 2u + 1u] = red_sum[0];
    }}
}}
",
    )
}

/// Merge weights from the per-slab statistics: `w_t = exp(m_t − M) · Σexp_t`
/// normalised by the row's total, where `M` is the row's global max over slabs.
/// One thread per row — its own `n_slab` slots, read and written by nobody else.
///
/// A slab with no valid column for a row has `m_t = -inf, Σexp = 0`, so
/// `w_t = 0` with no NaN: `exp(-inf − M)` is 0 and `0 · 0` is 0.  At least one
/// slab (the first, `row0 = 0`) always has a valid column, so `M` is finite and
/// the total cannot be zero.
pub fn slab_weights(bs: usize) -> String {
    format!(
        "struct WCfg {{ rows: u32, n_slab: u32, gx: u32, _p: u32 }};

@group(0) @binding(0) var<storage, read>       Stats: array<f32>;
@group(0) @binding(1) var<storage, read_write> Wt:    array<f32>;
@group(0) @binding(2) var<uniform>             cfg:   WCfg;

@compute @workgroup_size({bs})
fn slab_weights(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x + gid.y * (cfg.gx * {bs}u);
    if (row >= cfg.rows) {{ return; }}

    var m = bitcast<f32>(0xFF800000u);
    for (var t = 0u; t < cfg.n_slab; t = t + 1u) {{
        m = max(m, Stats[(t * cfg.rows + row) * 2u]);
    }}
    var total = 0.0;
    for (var t = 0u; t < cfg.n_slab; t = t + 1u) {{
        total = total + exp(Stats[(t * cfg.rows + row) * 2u] - m) * Stats[(t * cfg.rows + row) * 2u + 1u];
    }}
    // rows past the last position have no statistics at all (the stats dispatch
    // only covers real positions); `total == 0` there, and 0 · (1/0) would be a
    // NaN the merge would then hand to the padded output rows
    let inv = select(0.0, 1.0 / total, total > 0.0);
    for (var t = 0u; t < cfg.n_slab; t = t + 1u) {{
        Wt[t * cfg.rows + row] =
            exp(Stats[(t * cfg.rows + row) * 2u] - m) * Stats[(t * cfg.rows + row) * 2u + 1u] * inv;
    }}
}}
",
        bs = bs,
    )
}

/// Weighted merge of the per-slab AV outputs into the attention output:
/// `out[head][row] = Σ_t w_t[head][row] · part[t][head][row]`, the weights
/// already normalised by [`slab_weights`].  Elementwise — `n_slab` f16 words
/// per output word, accumulated in f32 and rounded once.
///
/// One grid axis per (row, head-dim word) band with the head on `wgid.y`, so
/// neither index needs a division.  The two operands are laid out differently —
/// the statistics/weights are head-major (`head · mp + pos`, one entry per row),
/// the AV output is position-major (`pos · nqh·hd + head·hd + …`, the AV GEMM's
/// own C layout) — hence the two independent indices.
pub fn slab_merge(nqh: usize, hd: usize) -> String {
    let hd2 = hd / 2;
    format!(
        "struct MCfg {{ rows: u32, n_slab: u32, gx: u32, _p: u32 }};

@group(0) @binding(0) var<storage, read>       Part: array<u32>;
@group(0) @binding(1) var<storage, read>       Wt:   array<f32>;
@group(0) @binding(2) var<storage, read_write> Out:  array<u32>;
@group(0) @binding(3) var<uniform>             cfg:  MCfg;

const NQH: u32 = {nqh}u;
const HD2: u32 = {hd2}u;

@compute @workgroup_size(256)
fn slab_merge(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(workgroup_id) wgid: vec3<u32>) {{
    // `rows` = mp (padded positions); the stats/weights rows are `nqh · mp` with
    // the head-major index `head · rows + row`.
    let head = wgid.y;
    let i = gid.x + gid.z * (cfg.gx * 256u);
    if (i >= cfg.rows * HD2) {{ return; }}
    let row = i / HD2;
    let w = i - row * HD2;
    // weights/stats: [t][head · mp + pos]; AV output: [t][pos][head · hd + col]
    let wi = head * cfg.rows + row;
    let pi = row * (NQH * HD2) + head * HD2 + w;

    var acc = vec2<f32>(0.0, 0.0);
    for (var t = 0u; t < cfg.n_slab; t = t + 1u) {{
        acc = acc + Wt[t * (cfg.rows * NQH) + wi] * unpack_h(Part[t * (cfg.rows * NQH * HD2) + pi]);
    }}
    Out[pi] = pack_h(acc);
}}
",
        nqh = nqh,
        hd2 = hd2,
    )
}

/// repeat_kv:  K cache `[nkvh, max_seq, hd]` rows `0..cur` duplicated per GQA
/// group into `[nqh, cur, hd]`.
pub fn repeat_kv(nrep: usize) -> String {
    format!(
        "struct Cfg {{ nkvh: u32, max_seq: u32, cur: u32, hd: u32, npw: u32, gx: u32 }};

@group(0) @binding(0) var<storage, read>       Cache: array<u32>;
@group(0) @binding(1) var<storage, read_write> Out:   array<u32>;
@group(0) @binding(2) var<uniform>             cfg:   Cfg;

const NREP: u32 = {nrep}u;

@compute @workgroup_size(256)
fn repeat_kv(@builtin(global_invocation_id) gid: vec3<u32>) {{
    // Two grid axes: `nkvh · nrep · cur · hd/2` reaches 65535 words at a
    // ~19-minute prefill, so x is capped and y continues the index space.
    let i = gid.x + gid.y * (cfg.gx * 256u);
    let words_per_head_pos = cfg.hd / 2u;
    let per_head = cfg.cur * words_per_head_pos;
    let total = cfg.nkvh * NREP * per_head;
    if (i >= total) {{ return; }}
    let qh = i / per_head;
    let rem = i - qh * per_head;
    let p = rem / words_per_head_pos;
    let w = rem - p * words_per_head_pos;
    let kh = qh / NREP;
    // out rows are np-padded per head (the GEMM's n tiles read the padding)
    Out[qh * cfg.npw + p * words_per_head_pos + w] =
        Cache[(kh * cfg.max_seq + p) * words_per_head_pos + w];
}}
",
        nrep = nrep,
    )
}

// ═══════════════════════════════════════════════════════════════════════
//  GPU audio encoder
//
//  All kernels use the packed-f16 `array<u32>` convention (`Gpu::storage` pads
//  to 16 B, one `u32` = two halves) and reuse the `prefill_gemm` tile.
// ═══════════════════════════════════════════════════════════════════════

/// Out-of-plane tap sentinel.  **One definition for both sides** — it is
/// interpolated into the WGSL below *and* written into the tap table by the Rust
/// caller, so the two cannot drift.
pub const TAP_OOB: u32 = 0xFFFF_FFFF;

/// im2col for `conv2d(3×3, stride 2, pad 1)`: a gather driven by a precomputed
/// tap table, so the inner loop has no divisions and no boundary tests.
///
/// The operand is `[k_pad][n]` f16 (`n` the position axis) with row stride `n/2`
/// words — what `prefill_gemm(transb = true)` reads as its `B`, with `n` the
/// whole tile (every chunk's positions end to end, chunk `c` at
/// `[c*plane_pad, c*plane_pad + plane)`).  A thread owns one `k` and two adjacent
/// positions and writes one packed word `Cols[k*(n/2) + col/2]`, so consecutive
/// `tx` write consecutive words of one row.
///
/// `k = ic*9 + kh*3 + kw`; `Taps[p*9 + tap]` is that tap's source offset inside
/// one input channel plane, `TAP_OOB` outside it; the source is
/// `in_chunk0*in_chunk + ic*in_ic + tap`.  Pad rows (`k >= k_real`), chunks past
/// `n_chunks` and positions past `plane` are written zero, so the GEMM never
/// accumulates undefined bytes.
///
/// Bindings: 0 = input, 1 = taps, 2 = operand, 3 = `Im2Cfg`.
pub fn audio_im2col() -> String {
    format!(
        "struct Im2Cfg {{ taps: u32, k: u32, k_pad: u32, plane: u32, plane_pad: u32,
                     n_chunks: u32, n_all: u32, in_chunk: u32, in_ic: u32,
                     chunk0: u32, bpc: u32, _a: u32 }};

@group(0) @binding(0) var<storage, read>       Input: array<u32>;
@group(0) @binding(1) var<storage, read>       Taps:  array<u32>;
@group(0) @binding(2) var<storage, read_write> Cols:  array<u32>;
@group(0) @binding(3) var<uniform>             cfg:   Im2Cfg;

/// `TAP_OOB` injected from the Rust table builder — see `TAP_OOB`.
const OOB: u32 = {oob}u;

fn scalar(addr: u32) -> f32 {{
    let w = unpack_h(Input[addr / 2u]);
    return select(w.x, w.y, (addr & 1u) == 1u);
}}

@compute @workgroup_size(16, 16)
fn im2col(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {{
    // `bpc` = position-blocks per chunk, so `wid.x` splits into (chunk, position
    // block) with no division inside the element loop; blocks past `chunks*bpc`
    // are the tile's tail padding and write zeros.
    let chunk = wid.x / cfg.bpc;
    let p = (wid.x % cfg.bpc) * 32u + 2u * lid.x;
    let col = chunk * cfg.plane_pad + p;
    let k = wid.y * 16u + lid.y;
    if (k >= cfg.k_pad || col >= cfg.n_all) {{ return; }}
    let word = k * (cfg.n_all / 2u) + col / 2u;
    if (k >= cfg.k || chunk >= cfg.n_chunks || p >= cfg.plane) {{
        Cols[word] = 0u;
        return;
    }}
    let ic = k / cfg.taps;
    let base = p * cfg.taps + (k % cfg.taps);
    let src = (cfg.chunk0 + chunk) * cfg.in_chunk + ic * cfg.in_ic;
    let a = Taps[base];
    let b = Taps[base + cfg.taps];
    var x = vec2<f32>(0.0, 0.0);
    if (a != OOB) {{ x.x = scalar(src + a); }}
    if (b != OOB && p + 1u < cfg.plane) {{ x.y = scalar(src + b); }}
    Cols[word] = pack_h(x);
}}
",
        oob = TAP_OOB
    )
}

/// `dst[i] = gelu(src[i] + bias[c])` over packed f16, two elements per thread.
/// `bias` is the per-channel vector, packed plain f16 two channels per word, so
/// channel `c` reads half of `Bias[c/2]`.  `cfg.words` is words per channel or per
/// row and `cfg.mode` selects `i / words` (channel-major conv activation, both
/// halves the same channel) or `i % words` (token-major GEMM output, the halves
/// are adjacent columns of one row).
///
/// WGSL has no `erf`, so GELU uses the A&S 7.1.26 `tanh`-style rational
/// approximation (max abs error 1.5e-7, well below the f16 rounding of the result).
/// `n` must be even so a thread's pair never straddles the bias vector's
/// real/padding boundary.
///
/// Bindings: 0 = src, 1 = bias, 2 = dst, 3 = `ScaleCfg { n, words, mode }`.
pub fn audio_bias_gelu() -> String {
    "struct ScaleCfg { n: u32, words: u32, mode: u32, gx: u32 };

@group(0) @binding(0) var<storage, read>       Src:  array<u32>;
@group(0) @binding(1) var<storage, read>       Bias: array<u32>;
@group(0) @binding(2) var<storage, read_write> Dst:  array<u32>;
@group(0) @binding(3) var<uniform>             cfg:  ScaleCfg;

const FRAC_1_SQRT_2: f32 = 0.70710678;
// A&S 7.1.26's `p` = 1/(1 + p|x|).  It was `0.147`, which is not this
// approximation's constant: the resulting erf was off by ~5 %, i.e. an order of
// magnitude more than the f16 rounding of the operand it feeds.
const C_A: f32 = 0.3275911;
const C_2_SQRT_PI: f32 = 1.128379167;

/// Abramowitz & Stegun 7.1.26: max abs error 1.5e-7.
fn erf_approx(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + C_A * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                   - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return select(-y, y, x >= 0.0);
}

fn gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf_approx(x * FRAC_1_SQRT_2));
}

@compute @workgroup_size(256)
fn bias_gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Two grid axes: a long clip's FFN activation is `tokens · ffn/2` words,
    // which passes 65535 workgroups well before the 22-minute token cap.
    let i = gid.x + gid.y * (cfg.gx * 256u);
    if (i * 2u >= cfg.n) { return; }
    let s = unpack_h(Src[i]);
    var b = vec2<f32>(0.0, 0.0);
    if (cfg.mode == 1u) {
        // Channel-major: both halves of the word are the same channel, one
        // channel per `words` words, and `Bias` is packed two channels per word
        // — so the channel's element is one *half* of `Bias[c/2]`, not `Bias[c]`.
        let c = i / cfg.words;
        let w = unpack_h(Bias[c / 2u]);
        let b0 = select(w.x, w.y, (c & 1u) == 1u);
        b = vec2<f32>(b0, b0);
    } else {
        // Token-major: the halves are adjacent columns of one row, and the
        // word's own two halves are exactly those two biases.
        b = unpack_h(Bias[i % cfg.words]);
    }
    Dst[i] = pack_h(vec2<f32>(gelu(s.x + b.x), gelu(s.y + b.y)));
}
"
    .to_string()
}

/// `LayerNorm` over the last dim, one workgroup per row; a single lane runs the
/// serial two-pass reduction: `mean`, then `var`, then `(x - mean) * inv_std * w + b`.
///
/// Bindings: 0 = src, 1 = weight, 2 = bias, 3 = `LnCfg { d, eps, ... }`, 4 = dst.
/// (Uniform before storage: wgpu requires the uniform last, so `Dst` takes
/// binding 4 and the uniform stays at 3.)
pub fn audio_layernorm() -> String {
    "struct LnCfg { d: u32, eps: f32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read>       Src: array<u32>;
@group(0) @binding(1) var<storage, read>       Wgt: array<u32>;
@group(0) @binding(2) var<storage, read>       Bia: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: LnCfg;
@group(0) @binding(4) var<storage, read_write> Dst: array<u32>;

fn half_at(v: vec2<f32>, i: u32) -> f32 { return select(v.x, v.y, (i & 1u) == 1u); }

@compute @workgroup_size(1)
fn layernorm(@builtin(workgroup_id) wid: vec3<u32>) {
    let d = cfg.d;
    let base = wid.x * (d / 2u);
    var mean = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        mean = mean + half_at(unpack_h(Src[base + j / 2u]), j);
    }
    mean = mean / f32(d);
    var var_ = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        let x = half_at(unpack_h(Src[base + j / 2u]), j) - mean;
        var_ = var_ + x * x;
    }
    var_ = var_ / f32(d);
    let inv = 1.0 / sqrt(var_ + cfg.eps);
    for (var w: u32 = 0u; w < d / 2u; w = w + 1u) {
        let j = w * 2u;
        let s = unpack_h(Src[base + w]);
        let g = unpack_h(Wgt[w]);
        let b = unpack_h(Bia[w]);
        Dst[base + w] = pack_h(vec2<f32>(
            (s.x - mean) * inv * g.x + b.x,
            (s.y - mean) * inv * g.y + b.y,
        ));
    }
}
"
    .to_string()
}

/// Split the fused QKV projection into the attention layouts.
///
/// `Qkv` is `[tok, 3·d_model]`; Q/K/V are written as `[tok, nh, hd]` (row stride
/// `attn_cols`, head-major inside the row), the `[head][tok][hd]` view the
/// attention GEMMs index with a per-head batch stride.
///
/// Bindings: 0 = qkv, 1 = Q, 2 = K, 3 = V, 5 = `ExCfg`.
pub fn audio_extract_qkv() -> String {
    "struct ExCfg { n_tokens: u32, nh: u32, hd: u32, tpc: u32,
                row_stride: u32, attn_cols: u32, pe_stride: u32, dm: u32 };

@group(0) @binding(0) var<storage, read>       Qkv: array<u32>;
@group(0) @binding(1) var<storage, read_write> Q:   array<u32>;
@group(0) @binding(2) var<storage, read_write> K:   array<u32>;
@group(0) @binding(3) var<storage, read_write> V:   array<u32>;
@group(0) @binding(5) var<uniform>             cfg: ExCfg;

@compute @workgroup_size(256)
fn extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    // gid = (token, head, word inside the head).  The head is its own grid axis
    // rather than folded into x: `s_pad · nh` exceeds wgpu's 65535-per-dimension
    // limit at ~6 minutes of audio.
    let hd2 = cfg.hd / 2u;
    let tok = gid.x;
    if (tok >= cfg.n_tokens) { return; }
    let head = gid.y;
    let w = gid.z;
    if (w >= hd2) { return; }

    let dm2 = cfg.dm / 2u;
    let row = tok * (dm2 * 3u);
    let col = head * hd2 + w;
    let dst = tok * (cfg.attn_cols / 2u) + col;

    // No positional embedding here: it is added to the `conv_out` output by
    // `audio_add_pe`, and adding it twice would be wrong.
    Q[dst] = Qkv[row + col];
    K[dst] = Qkv[row + dm2 + col];
    V[dst] = Qkv[row + dm2 * 2u + col];
}
"
    .to_string()
}

/// Re-pack the projected Q/K/V into the per-`(head, window)` blocks the two
/// attention GEMMs read.
///
/// The projections come out token-major (`[tok][nh·hd]`), but `prefill_gemm`
/// fixes both operand row strides from `k`: the `A` row stride is `k/2` words
/// and a `transb` `B` row stride is `n/2`.  A head's slice of a token row is
/// `nh·hd` halves wide, so neither GEMM can read it in place — this kernel is
/// what makes the strides line up:
///
/// ```text
///   qp[z][row][j]   row stride hd/2 words   (scores A, m = token)
///   kt[z][j][row]   row stride wpad/2 words (scores B, transb, n = token)
///   vp[z][row][j]   row stride 64 words     (AV B, transb, n = 128 = hd_pad)
/// ```
///
/// `z = head·n_win + win` and rows are `win·wlen + row` in the global token
/// axis; rows/keys past a short last window are **zeroed**, which is what lets
/// the softmax mask them by a loop bound.  The `j` axis is padded to `hd_pad`
/// (= `GEMM_BN`) because the AV GEMM's `n` must be a whole 128-wide tile.
///
/// Bindings: 0 = Q, 1 = K, 2 = V, 3 = qp, 4 = kt, 5 = vp, 6 = `WinCfg`.
pub fn audio_win_pack() -> String {
    "struct WinCfg { wlen: u32, wpad: u32, hd: u32, n_win: u32, s: u32, acols: u32,
                pad_n: u32, _a: u32 };

@group(0) @binding(0) var<storage, read>       Q:  array<u32>;
@group(0) @binding(1) var<storage, read>       K:  array<u32>;
@group(0) @binding(2) var<storage, read>       V:  array<u32>;
@group(0) @binding(3) var<storage, read_write> Qp: array<u32>;
@group(0) @binding(4) var<storage, read_write> Kt: array<u32>;
@group(0) @binding(5) var<storage, read_write> Vp: array<u32>;
@group(0) @binding(6) var<uniform>             cfg: WinCfg;

@compute @workgroup_size(16, 16)
fn win_pack(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    let z = wid.z;
    let head = z / cfg.n_win;
    let win = z - head * cfg.n_win;
    let rp = wid.x * 16u + lid.x;          // token pair inside the window
    let jw = wid.y * 16u + lid.y;          // head-dim word (two j values)
    let hd2 = cfg.hd / 2u;
    let wpad2 = cfg.wpad / 2u;
    if (2u * rp >= cfg.wpad) { return; }
    let valid = min(cfg.wlen, cfg.s - win * cfg.wlen);
    let j0 = 2u * jw;
    let in_range = j0 < cfg.hd;            // the head-dim padding stays zero

    var q0 = vec2<f32>(0.0, 0.0);
    var k0 = q0; var v0 = q0; var q1 = q0; var k1 = q0; var v1 = q0;
    if (in_range) {
        let t0 = win * cfg.wlen + 2u * rp;
        let i0 = t0 * (cfg.acols / 2u) + head * hd2 + jw;
        if (2u * rp < valid) {
            q0 = unpack_h(Q[i0]);
            k0 = unpack_h(K[i0]);
            v0 = unpack_h(V[i0]);
        }
        if (2u * rp + 1u < valid) {
            let i1 = i0 + cfg.acols / 2u;
            q1 = unpack_h(Q[i1]);
            k1 = unpack_h(K[i1]);
            v1 = unpack_h(V[i1]);
        }
    }

    // vp: one row per token, `pad_n/2` words wide, so the head-dim padding is
    // *inside* the row and these threads write the zeros for it.
    let vp = z * (cfg.wpad * (cfg.pad_n / 2u)) + (2u * rp) * (cfg.pad_n / 2u) + jw;
    Vp[vp] = pack_h(v0);
    Vp[vp + cfg.pad_n / 2u] = pack_h(v1);

    // qp: the row is exactly `hd2` words with no padding, so a `jw` past the
    // row must not write at all — `Qp[qp]` would land in the *next* token's row
    // and this thread's zeros would race the writes of that row's own thread
    // (whichever workgroup the hardware ran last decided whether a Q row was
    // real or zero).
    if (in_range) {
        let qp = z * (cfg.wpad * hd2) + (2u * rp) * hd2 + jw;
        Qp[qp] = pack_h(q0);
        Qp[qp + hd2] = pack_h(q1);
    }

    // kt: the transpose — row `j`, and the token pair in one word.  Only the
    // real `hd` rows exist; the rest would run into the next block.
    if (in_range) {
        let kt = z * (cfg.hd * wpad2) + j0 * wpad2 + rp;
        Kt[kt] = pack_h(vec2<f32>(k0.x, k1.x));
        Kt[kt + wpad2] = pack_h(vec2<f32>(k0.y, k1.y));
    }
}
"
    .to_string()
}

/// Windowed attention softmax — one thread per `(head, window, token)` row,
/// serial over the window's valid keys.
///
/// Blocks are dense: `[z][wpad][wpad]` with `z = head·n_win + win`, so the
/// batched GEMMs' block strides are all `wpad²/2` words.
///
/// A short last window must **not** see the padded keys: each window runs over
/// `min(wlen, s - win·wlen)` tokens and everything past it is written zero.
///
/// Bindings: 0 = scores, 1 = probs, 2 = `SmCfg { s, wlen, wpad, n_win, scale }`.
pub fn audio_window_softmax() -> String {
    "struct SmCfg { s: u32, wlen: u32, wpad: u32, n_win: u32, scale: f32,
                _a: u32, _b: u32, _c: u32 };

@group(0) @binding(0) var<storage, read>       Sc:  array<u32>;
@group(0) @binding(1) var<storage, read_write> Out: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: SmCfg;

fn half_at(v: vec2<f32>, i: u32) -> f32 { return select(v.x, v.y, (i & 1u) == 1u); }

@compute @workgroup_size(128)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    // gid.x = token inside the window; gid.y = head·n_win + window
    let row = gid.x;
    if (row >= cfg.wpad) { return; }
    let z = gid.y;
    let win = z % cfg.n_win;
    let valid = min(cfg.wlen, cfg.s - win * cfg.wlen);
    let base = z * (cfg.wpad * (cfg.wpad / 2u)) + row * (cfg.wpad / 2u);
    // Keys past `valid` contribute nothing even for a live row, and a row past
    // `valid` has no consumer at all — write both as zero.
    if (row >= valid) {
        for (var w: u32 = 0u; w < cfg.wpad / 2u; w = w + 1u) { Out[base + w] = 0u; }
        return;
    }

    var mx = -3.0e38;
    for (var j: u32 = 0u; j < valid; j = j + 1u) {
        mx = max(mx, half_at(unpack_h(Sc[base + j / 2u]), j) * cfg.scale);
    }
    var sum = 0.0;
    for (var j: u32 = 0u; j < valid; j = j + 1u) {
        sum = sum + exp(half_at(unpack_h(Sc[base + j / 2u]), j) * cfg.scale - mx);
    }
    let inv = 1.0 / sum;
    for (var w: u32 = 0u; w < cfg.wpad / 2u; w = w + 1u) {
        let j0 = w * 2u;
        let j1 = j0 + 1u;
        let v = unpack_h(Sc[base + w]);
        let p0 = select(0.0, exp(v.x * cfg.scale - mx) * inv, j0 < valid);
        let p1 = select(0.0, exp(v.y * cfg.scale - mx) * inv, j1 < valid);
        Out[base + w] = pack_h(vec2<f32>(p0, p1));
    }
}
"
    .to_string()
}

/// Conv-stem epilogue: `h = conv_out + PE`.  `h` is `[tokens, d]` with token
/// stride `s_pad` (the transformer's operand layout); `dst` is the packed
/// `[tokens, d]` conv-stem output.  Both get the same value so the embedding
/// dump and the transformer consume identical bytes.
///
/// Bindings: 0 = h, 1 = PE `[tpc, d]`, 2 = dst, 3 = `PeCfg { d, tpc, s_pad }`.
pub fn audio_add_pe() -> String {
    "struct PeCfg { d: u32, tpc: u32, s_pad: u32, _a: u32 };

@group(0) @binding(0) var<storage, read>       H:   array<u32>;
@group(0) @binding(1) var<storage, read>       Pe:  array<u32>;
@group(0) @binding(2) var<storage, read_write> Dst: array<u32>;
@group(0) @binding(3) var<uniform>             cfg: PeCfg;

@compute @workgroup_size(16, 16)
fn add_pe(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    // 256 words per workgroup, so the y grid has to walk the row: `d_model/2`
    // is 448 words and the old single-workgroup row left 192 of them unwritten.
    let w = wid.y * 256u + lid.x + 16u * lid.y;
    let d2 = cfg.d / 2u;
    if (w >= d2) { return; }
    let tok = wid.x;
    let v = unpack_h(H[tok * (cfg.s_pad / 2u) + w])
          + unpack_h(Pe[(tok % cfg.tpc) * d2 + w]);
    let packed = pack_h(v);
    Dst[tok * (cfg.s_pad / 2u) + w] = packed;
}
"
    .to_string()
}

/// Gather the conv3 activation `[c][pos]` into the `conv_out` operand
/// `[tok, c·f]`.
///
/// `audio_conv`'s GEMM writes its `C` as `[m][n]` with `m` the **channel** and
/// `n` the flattened position, so a chunk's plane for channel `c` starts at
/// `c*n_all` and holds `plane_pad` positions per chunk.  Position
/// `p = f*t3 + ti` (`ti` fastest — the GEMM's tile wrote `(h_out, w_out)` in
/// row-major order), and the chunk's token `ti` is the `t3`-th part of the
/// token index.  A packed row element `j = c*f_dim + f` therefore reads
///
/// ```text
///   src = c*n_all + chunk*plane_pad + f*t3 + ti
/// ```
///
/// The positional embedding is **not** added here: it goes to the `conv_out`
/// *output* (`d_model` wide), not to this `c·f` operand — see [`audio_add_pe`].
///
/// Bindings: 0 = c3, 1 = packed, 2 = `PmCfg`.
pub fn audio_permute_pe() -> String {
    "struct PmCfg { c: u32, f: u32, t3: u32, s_pad: u32, n_all: u32, plane_pad: u32,
                tok0: u32, n_tokens: u32 };

@group(0) @binding(0) var<storage, read>       C3:  array<u32>;
@group(0) @binding(1) var<storage, read_write> Dst: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: PmCfg;

fn half_at(addr: u32) -> f32 {
    let v = unpack_h(C3[addr / 2u]);
    return select(v.x, v.y, (addr & 1u) == 1u);
}

@compute @workgroup_size(16, 16)
fn permute_pe(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let cf = cfg.c * cfg.f;
    // 256 words per workgroup, so `wid.y` walks the row: `c·f/2` is 3840 words
    // and a single block would leave all but the first 256 unwritten.
    let w = wid.y * 256u + lid.x + 16u * lid.y;
    if (w * 2u >= cf) { return; }
    // `wid.x` is the token inside this tile, spanning `plane_pad / t3` chunks;
    // the grid is padded to a GEMM `m` tile so the rows past the clip — which
    // the `conv_out` GEMM still reads — are written zero rather than left as
    // whatever the allocator handed us.
    let tok = wid.x;
    if (cfg.tok0 + tok >= cfg.n_tokens) {
        // The destination row is the *global* token: using the tile-local `tok`
        // zeroed the wrong rows whenever a round did not start at token 0 (it
        // wiped the previous round's tail and its own head).
        Dst[(cfg.tok0 + tok) * (cfg.s_pad / 2u) + w] = 0u;
        return;
    }
    let chunk = (cfg.tok0 + tok) / cfg.t3 - cfg.tok0 / cfg.t3;
    let ti = (cfg.tok0 + tok) % cfg.t3;
    var x = vec2<f32>(0.0, 0.0);
    for (var e: u32 = 0u; e < 2u; e = e + 1u) {
        let j = w * 2u + e;
        let c = j / cfg.f;
        let f = j - c * cfg.f;
        let src = c * cfg.n_all + chunk * cfg.plane_pad + f * cfg.t3 + ti;
        let v = half_at(src);
        if (e == 0u) { x.x = v; } else { x.y = v; }
    }
    Dst[(cfg.tok0 + tok) * (cfg.s_pad / 2u) + w] = pack_h(x);
}
"
    .to_string()
}

/// Flatten the per-`(head, window)` attention output into the token-major
/// `[tok, nh·hd]` operand `out_proj` wants.
///
/// The AV GEMM writes `[z][row][hd_pad]` blocks (`z = head·n_win + win`); a
/// destination word covers two adjacent `j`, which stay inside one head, so the
/// copy is word-for-word — only the two index maps (row → global token, head →
/// block) change.
///
/// Bindings: 0 = src blocks, 1 = dst, 2 = `CpCfg { cols, wlen, wpad, hd, n_win,
/// rows, ... }`.
pub fn audio_attn_flat() -> String {
    "struct CpCfg { cols: u32, wlen: u32, wpad: u32, hd: u32, n_win: u32, rows: u32,
                _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read>       Src: array<u32>;
@group(0) @binding(1) var<storage, read_write> Dst: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: CpCfg;

@compute @workgroup_size(256)
fn attn_flat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;                       // word index into the destination
    let cols2 = cfg.cols / 2u;
    if (i >= cfg.rows * cols2) { return; }
    let tok = i / cols2;
    let w = i - tok * cols2;
    let win = tok / cfg.wlen;
    let row = tok - win * cfg.wlen;
    let hd2 = cfg.hd / 2u;
    let head = w / hd2;
    let z = head * cfg.n_win + win;
    let src = z * (cfg.wpad * 64u) + row * 64u + (w - head * hd2);
    Dst[i] = Src[src];
}
"
    .to_string()
}
