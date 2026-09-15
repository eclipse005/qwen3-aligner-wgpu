# Porting guide

For anyone porting this engine to another runtime (another WGSL implementation,
a CUDA/HIP/Metal rewrite, a different model) or reusing its shape for a different
transformer.  Everything here is measured on the machine described in the README
(P104-100, Vulkan, wgpu 30) — the *numbers* are Pascal-specific, the *lessons*
are not.

The short version: **the arithmetic order is the specification, and the runtime
will fail silently if you let it.**

---

## 1. The invariants (moving any of these breaks parity)

### 1.1 Reduction trees are the reference, not an implementation detail

The text decoder must reproduce the Python reference bit-for-bit (f16 rounding
included), because that is what the frozen gold texts were generated from and
what the 12/12 gate compares.  Concretely:

* `shaders::gemv` folds 32 lane partials with `subgroupShuffleXor` (or 5 shared
  memory rounds) in the order `^16, ^8, ^4, ^2, ^1`.  That exact association is
  load-bearing; using `subgroupAdd` gives a different last ulp.
* `shaders::rms_norm` uses `block_for_reduction(hidden)` (=1024 here) virtual
  threads, and `gemv_norm` (which folds the norm into the consuming GEMV)
  reproduces the *same* tree with 256 real threads — two phantom rounds, matched
  pair by pair.
* The prefill softmax's block size is `block_for_reduction(cur)`, not a free
  choice.
* The decode attention's key stride (`T_SPLIT`) fixes each thread's key set; a
  different split changes the partial sums.

**A K-split of the decode GEMVs is therefore impossible** — `docs/decode-splitk-analysis.md`
§4 proves it for every split shape.  This is not an optimisation you are leaving
on the table; it is a wall, and the wall is documented so you don't spend a week
rediscovering it.

### 1.2 `exp` is bit-exact and integer, and it is on the critical path

`shaders::expf_bt` (three integer FMAs + `scale_pow2` + a bit-exact tail) exists
because the reference's `exp` is a specific approximation.  Replacing it with
the built-in `exp()` is a *different function*: it moved the decode by +18 %
when measured the other way round, and it changes the tokens.  Tolerances only
apply where the reference itself is not deterministic (the audio tower's
conv GEMMs, which the CUDA golden itself only matches to ~0.003).

### 1.3 Where f16 rounding happens is part of the contract

`rms_norm` writes f16, so the consuming GEMV must read the *rounded* value
(`pack2x16float(x * inv_rms * w)`), not the f32 intermediate.  The AV GEMM
accumulates in f32 and rounds **once**, at the end — which is why the slabbed
long-context attention (per-slab f16 outputs, merged in f32) is a documented,
gate-checked deviation rather than a free lunch.

### 1.4 Units and layouts are contracts, not conventions

* `bsa`/`bsb` are **words** (the shader indexes `array<u32>`), `bsc` is
  **elements** (the epilogue divides by 2).  A `bsc` in words shifted every
  batch but the first by half a block — invisible at batch 1, non-deterministic
  at batch 28 (the audio tower's attention).
* Attention slabs: `qp[z][wpad][hd]` (row stride `hd/2` words),
  `kt[z][hd][wpad]`, `vp[z][wpad][hd_pad]`, scores/probs `[z][wpad][wpad]`.
  `Layout` sections in `HANDOFF.md` and `docs/design-tiled-prefill.md` are the
  single source; `ConvLevel` (`audio_encoder_gpu.rs`) generates the conv stem's
  geometry so the kernel and the diagnostics cannot disagree.

### 1.5 Capability assumptions (survey before you design)

* **No int8 dot product in WGSL**, even where the hardware has one (Pascal has
  DP4A).  Weight quantisation is therefore not a local optimisation here.
* **Pascal's f16 arithmetic runs at 1/64 rate** — every kernel is deliberately
  f32 scalar with f16 only for storage.  Measured throughput exceeding the
  hardware f16 limit by 15× is the tell that you are already on the f32 path.
* **`enable subgroups;` is not implemented by naga**; use the `Features::SUBGROUP`
  capability bit (present on this stack) and keep a shared-memory fallback
  (`gemv` has both, and they are bit-identical).
* **`max_storage_buffer_binding_size` is 2047 MiB** and is a *driver* constant
  independent of VRAM.  This, not memory size, is what forces tiling.

---

## 2. The traps (each one cost real time here)

| # | trap | symptom | what to do |
|---|---|---|---|
| 1 | **Rust↔WGSL structs are two definitions**, and this port has `GDims` in *two* Rust files (decoder + audio tower) | adding a field to one side silently feeds the other side garbage from the uniform slot's tail: **empty transcripts, no GPU error** | one definition per uniform + a compile-time `assert!(size_of::<T>() == N)`; if a struct is shared, say so at both sites |
| 2 | **A bind group entry's offset is added to the dynamic offset** | the dispatch reads the wrong cfg slot (here: the stats kernel read the *weights* cfg) — again with no error, just wrong numbers | use *either* a static offset *or* a dynamic one, never both; assert the slot budget |
| 3 | **wgpu fails silently on validation errors**: a rejected command buffer leaves the buffers untouched | OOM → garbage tokens; an invalid bind group → an empty transcript; exit code 0 | install `device.on_uncaptured_error(...)`, set a poisoned flag, and check it before every readback/submit |
| 4 | **One buffer cannot be read-only *and* read-write in the same dispatch** | `Validation Error: … conflicting usages … is an exclusive usage` | the softmax cannot normalise in place: two buffers (`p.scores`, `p.attn`) |
| 5 | **Grid dimensions cap at 65535** | whole command buffer rejected past ~4 min of audio (softmax rows), ~6 min (audio tower `extract`) | `grid_xy()` — flatten the extra axis into `y` and pass `gx` in the uniform |
| 6 | **`0 · NaN = NaN`** | a dirty page multiplied by a softmax zero poisons a *used* row | zero the tails you never write (`k_rep`/`v_rep` past `cur`), and guard degenerate reductions (`total == 0` in the merge weights) |
| 7 | **`queue.write_buffer` lands at submit time** | every dispatch in one command buffer reads the *last* value written | per-dispatch values need distinct slots (256 B apart, dynamic offsets) — this is why there is a uniform ring |
| 8 | **TDR is wall clock, not work** | a `10m` prefill died with `map token`/`async map` after ~4 s per submit | submit + poll often enough as the per-layer time grows (here: every 2 layers above 4096 tokens) |
| 9 | **Timing without a clock reference is noise** | this card swings between 1151/1607/1911 MHz | record the SM clock with every number, run one GPU job at a time, no warm-up, and keep the phase decomposition summing to the elapsed time |
| 10 | **A capability bit is not a width.** `Features::SUBGROUP` was taken as "32 lanes" | Intel's Vulkan driver reports `subgroup 8..32`; the 32-lane xor butterfly (`shuffleXor 16/8/4/2/1`, rows mapped by `lid >> 5`) then reduced across the wrong lanes and **the whole model produced garbage with no error** — the transcript was the only symptom | gate shuffle paths on the adapter's promised width (`AdapterInfo::subgroup_min_size == subgroup_max_size == 32`), keep the shared-memory fallback (bit-identical here), and print the width in the device listing |
| 11 | **`max_storage_buffer_binding_size` varies per adapter** | Intel Vulkan: 1023 MiB; NVIDIA Vulkan: 2047; D3D12: 2047 (both) | size every allocation from the *negotiated* limit, and re-check the tiling guard per device — a clip that runs on one adapter can be refused on another |

---

## 3. Dead ends (measured — do not retry)

| attempt | measurement |
|---|---|
| `gemv` with X staged in shared memory | 255.9 vs 255.2 GB/s |
| `gemv` one warp per two rows | 200 GB/s; 0.77–0.89× on every shape |
| `gemv` split-K (2 slices + merge) | o_proj degraded to 64 vs 82 GB/s, rest 0.96–1.01× |
| prefill GEMM tile shapes (TM/TN sweep) | flat at ~2.07–2.44 TFLOP/s; the production variant is the best |
| `PREFILL_GEMM_BK = 32` | smem 35 KB ⇒ 1 workgroup/SM ⇒ prefill +27 % |
| `PREFILL_GEMM_BK = 8` | the staging loop assumes `BK ≥ 16`; produces wrong results |
| smem padding `PAD = BK+4` (20) | bank conflicts + occupancy: enc 2640 → 1010 ms, prefill 4200 → 1518 ms |
| `repeat_kv` optimisation (estimated 0.45 s) | measured 14 ms — the estimate was wrong |
| `enable subgroups;` | naga does not implement it (issue #5555); use the capability bit |
| synthetic shared-memory microbenchmarks (`smem_roof`) | strength-reduced by the compiler; the numbers are not trustworthy (marked INCONCLUSIVE) |
| flat O(s²) prefill attention past ~8 200 tokens | over the 2047 MiB binding limit; the slab path replaces it (and the *slab* path then has to avoid the [1]-style traps above) |

The pattern: **measure the op before optimising it, and write the number down
next to the idea** — this table is why nobody has to re-run those experiments.

---

## 4. What a port needs to stay honest

1. **A frozen reference output per (model, clip)** and a byte-comparison gate
   (`tools/verify_all.ps1` here).  Not a tolerance — the text, verbatim.
2. **An oracle that prints its inputs.**  The single worst bug in this port's
   history was an attention oracle that recomputed `softmax(q·k)` from the GPU's
   own (all-zero) q/k/v: zero equals zero, every row green, and the real bug
   (a uniform never written) survived three rounds of "verification".  Print
   `max|q|`, `max|k|`, `max|v|` and fail when they are zero.
3. **An ablation harness.**  `QASR_DUP=<op>` re-dispatches an idempotent kernel
   once; the time delta is that op's true cost including its launch bubble.  It
   found the slab attention's softmax eating 3.7 s of a 10.2 s prefill.
4. **Layer-by-layer golden dumps** (`cuda_ref/` + `prefill_check`) for when the
   text diverges and you need to know *where*.
5. **A clock reference and one job at a time** (§2.9), and the phase timings
   printed such that they sum to the elapsed time.

## 5. Where to look first

| question | file |
|---|---|
| how does a kernel get built and dispatched? | `src/gpu.rs`, then any `Pipes` block in `src/decoder.rs` |
| what does one decode step do, in order? | the header comment of `src/decoder.rs` |
| how is the prompt built / the output parsed? | `src/prompt.rs` (line-by-line ports of the reference) |
| how do the diagnostics work? | `transcribe --diag-enc`, `src/bin/prefill_check.rs`, `src/golden.rs` |
| why is the attention tiled? | `docs/design-tiled-prefill.md` |
| why is the decode attention shaped that way? | `docs/decode-splitk-analysis.md` |
| what has already been tried and failed? | §3 above, and `HANDOFF.md`'s dead-end lists |
