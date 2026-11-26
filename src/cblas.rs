//! Apple Accelerate `cblas_sgemm` binding, used on aarch64 macOS for the
//! prefill GEMMs in the CPU decoder and audio encoder.
//!
//! Why: prefill forward is dominated by `proj` and `mlp` Linear layers — plain
//! GEMM, previously handled by the portable `gemm` crate. Single-row decode
//! (`m == 1`) is deliberately left on the hand-written `dot_transposed` /
//! `linear_gemv` path, which is memory-bandwidth bound rather than FLOP bound.
//!
//! Only the row-major, non-conjugated `f32` case used by the CPU path is
//! exposed. `b_trans` covers the `[out, in]` checkpoint layout without a
//! separate transpose pass.
//!
//! Calls are left threaded: Accelerate parallelises internally via GCD, which
//! beats nesting it inside a rayon pool. Every routed call site runs on a
//! single thread for that reason.

use std::os::raw::c_int;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;
const CBLAS_TRANS: c_int = 112;

/// `c = alpha·a·b + beta·c`, all row-major.
///
/// `a` is `(m, k)` with row stride `lda`, `c` is `(m, n)` with row stride
/// `ldc`. When `b_trans` is set, `b` holds an `(n, k)` row-major operand read
/// as its transpose (`ldb = k`); otherwise it is `(k, n)` with `ldb = n`.
///
/// # Safety
/// All pointers must address buffers large enough for the shape given by
/// `m`/`n`/`k` and the strides, with no overlap between `c` and `a`/`b`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_rm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    lda: usize,
    b_trans: bool,
    b: *const f32,
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    // Every call site here is a few thousand wide at most; the cast cannot
    // truncate. Explicit rather than `as _` so a future larger shape fails
    // loudly instead of silently wrapping.
    debug_assert!(m <= i32::MAX as usize);
    debug_assert!(n <= i32::MAX as usize);
    debug_assert!(k <= i32::MAX as usize);
    if m == 0 || n == 0 {
        return;
    }
    cblas_sgemm(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        if b_trans { CBLAS_TRANS } else { CBLAS_NO_TRANS },
        m as c_int,
        n as c_int,
        k as c_int,
        alpha,
        a,
        lda as c_int,
        b,
        ldb as c_int,
        beta,
        c,
        ldc as c_int,
    );
}
