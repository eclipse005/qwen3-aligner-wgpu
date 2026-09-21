//! Model-load attribution — `QALIGN_LOAD_TRACE=1` prints one line per load stage.
//!
//! Load sits outside the alignment clock (it happens once, before any audio is
//! timed), so it needs its own accounting: a stage that is not measured cannot
//! be argued about.  Off by default — when off, the whole cost is one relaxed
//! atomic load per stage.

use std::sync::OnceLock;
use std::time::Instant;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("QALIGN_LOAD_TRACE").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

/// Print `name` with the time since `t`, and nothing when the trace is off.
///
/// Call it as `let t = Instant::now(); …; load_trace::note("decoder: weights", t);`
/// so the stage boundary is visible in the source rather than implied by a
/// guard's scope.
pub fn note(name: &str, t: Instant) {
    note_dur(name, t.elapsed());
}

/// [`note`] for a duration accumulated across a loop.
pub fn note_dur(name: &str, d: std::time::Duration) {
    if enabled() {
        eprintln!("[load] {name:<30} {:>9.1} ms", d.as_secs_f64() * 1000.0);
    }
}

/// Bulk-transfer accounting: what [`crate::gpu::BulkUpload`] actually moved and
/// how long it spent in `queue.write_buffer` vs waiting for the queue.
///
/// Kept here rather than in `gpu.rs` so the load path can zero it, run one
/// stage, and read the result back — the split between *converting* weights and
/// *moving* them is the whole question when a load is slow.
pub mod transfer {
    use std::sync::atomic::{AtomicU64, Ordering};

    static WRITE_NS: AtomicU64 = AtomicU64::new(0);
    /// Time spent in submits that do not wait (the DMA runs on from here).
    static SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
    /// Time spent *waiting* for the queue — the transfer tail that stayed exposed.
    static WAIT_NS: AtomicU64 = AtomicU64::new(0);
    static BYTES: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn add_write(ns: u64, bytes: u64) {
        WRITE_NS.fetch_add(ns, Ordering::Relaxed);
        BYTES.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(crate) fn add_submit(ns: u64) {
        SUBMIT_NS.fetch_add(ns, Ordering::Relaxed);
    }

    pub(crate) fn add_wait(ns: u64) {
        WAIT_NS.fetch_add(ns, Ordering::Relaxed);
    }

    /// Zero the counters (call before the stage being measured).
    pub fn reset() {
        WRITE_NS.store(0, Ordering::Relaxed);
        SUBMIT_NS.store(0, Ordering::Relaxed);
        WAIT_NS.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
    }

    /// Print bytes staged, host staging time, submit time and the exposed tail.
    ///
    /// A healthy pipeline shows a small `submit` and a `wait` that is a fraction
    /// of the bytes moved; a `wait` that holds the whole transfer means the
    /// copies were never submitted early enough to overlap.
    pub fn note(name: &str) {
        if super::enabled() {
            let ns = WRITE_NS.load(Ordering::Relaxed);
            let submit = SUBMIT_NS.load(Ordering::Relaxed);
            let wait = WAIT_NS.load(Ordering::Relaxed);
            let bytes = BYTES.load(Ordering::Relaxed);
            let mib = bytes as f64 / (1024.0 * 1024.0);
            let gbs = if ns == 0 { 0.0 } else { bytes as f64 / ns as f64 };
            eprintln!(
                "[load] {name:<30} {:>9.1} ms  ({mib:.0} MiB staged, {gbs:.1} GB/s stage, \
                 {:.1} ms submit, {:.1} ms wait)",
                (ns + submit + wait) as f64 / 1e6,
                submit as f64 / 1e6,
                wait as f64 / 1e6,
            );
        }
    }
}
