# Design: tiled causal prefill attention (kills the s² scratch)

Status: **implemented** (2026-09-15) — the scratch is `[nqh, mp, SLAB_T]` with
`SLAB_T = 1024`, gated below `s = 4096` (`QASR_SLAB=on|off` forces a path), and
`15m.wav` (12 065 tokens) now runs end to end.  *What was built differs from the
three-pass sketch below in one place — the pass structure — which the "As built"
section at the bottom records together with the measurements and the traps.*

## The problem, measured

`decoder::prefill` materializes two `[nqh, mp, np]` f16 matrices per pass:

| | 180 s clip (s = 2307) | 15 min clip (s = 12065) |
|---|---|---|
| `scores` + `attn` | 2 × 170 MB | 2 × 4.7 GB |
| cost | 821 ms of the 1772 ms prefill (46 %), all of it traffic | **refused**: each buffer is past the 2047 MiB single-binding limit |

The ablation (`QASR_DUP=p_scores|p_softmax|p_av`) splits that 821 ms into
scores 295 / causal softmax 243 / AV 283 ms.  The causal skips already landed
(see HANDOFF) cut it to ~550 ms; the remaining cost is inherent to writing and
re-reading s².

## Design

Tile the **key** dimension.  With `T` = 1024 keys per tile (padded), the largest
scratch becomes `[nqh, mp, T]` — 9.4 MB at s = 2307, and *constant* in s:

    n_tiles = ceil(cur / T)
    for t in 0..n_tiles:                       # three passes over the tiles
        # pass A: per-tile row max, combined with a plain max (exact)
        scores_tile[t] = q × K[t]ᵀ             # GEMM, k = hd, n = T
        row_max = max(row_max, tile_max)       # max is order-free, no rounding
    for t in 0..n_tiles:
        # pass B: exp against the *global* max, tile sum, keep the exp'd tile
        scores_tile[t] = q × K[t]ᵀ             # recomputed (cheap: k = hd)
        e = exp(scores_tile[t] - row_max)
        row_sum += tile_sum(e)                 # grouped by tile, in tile order
        attn_tile[t] = e
    for t in 0..n_tiles:
        # pass C: accumulate the (unnormalized) AV per tile
        attn_flat += attn_tile[t] × V[t]       # GEMM with beta = accumulate
    attn_flat *= 1 / row_sum                   # small epilogue kernel

Why this is exact-ish rather than a rewrite of the math: pass A/B use max and sum
reductions that are *reassociated* relative to the flat kernel (the softmax's
block tree is fixed by the reference, so a per-tile sum will not be bit-identical)
— the same trade the split-K decode attention already makes.  **The transcript is
the gate**, not bit-equality.

Cost: one extra scores GEMM pass (k = hd only, ~5 ms per pass at 0.6B) and one
extra small epilogue, against ~3-4x less scratch traffic.  For 15-minute audio it
is the difference between "refused" and "runs".

## Where it plugs in

* `decoder::prefill` (the layer loop's steps 5-7): the three dispatches
  (`gemm`/`softmax`/`gemm_av`) become a tile loop; the buffers are allocated
  `[nqh, mp, T]` + `[nqh, mp]` (max/sum) instead of `[nqh, mp, np]`.
* `shaders::softmax_causal`: needs a tile form (row max / exp+sum / store) that
  reads a `cur`-strided tile, or three small new entry points.
* The AV GEMM already accumulates (`gemm_acc`/beta) — pass C is that with the
  k-range restricted to a tile, so it needs a `k0` offset in `GDims` (or a
  per-tile bind group with an offset `bsa`).
* **Gate it**: only take the tiled path when `s > 4096` (≈ 5.5 min of audio).
  The six fixtures and every FLEURS clip stay on the current path, so the
  alignment gate cannot move; the long path is verified by comparing its output
  against the current path on `5m.wav` / `6m.wav` / `10m.wav` (all three run
  today) and by finally running `15m.wav`.

## Why not flash-attention (online softmax)

Rescaling the accumulators per tile multiplies the running output by a changing
factor; that is a different (and larger) arithmetic change than re-grouping the
row sum, and it does not buy anything over the three-pass form here — the scores
GEMM is only ~5 ms of a ~1.8 s prefill, so recomputing it is free.

---

# As built (2026-09-15)

## What the slab loop actually does

The three passes of the sketch all keep score data alive, so the tile loop was
built the other way round — **one slab live at a time**, each slab softmaxed
against *its own* max, with the slabs combined by the exact weights they were
divided by:

```text
for t in 0..n_slab:                        # one pass, one slab live
    t0 = t·T,  tl = min(T, cur16 − t0)
    scores[t]  = q × K[t0..t0+tl)ᵀ         # gemm_causal, row0 = t0
    (max[t], sum[t]) ← slab_stats(scores[t])   # per-slab stats
    attn[t]    = softmax_causal(scores[t])     # normalised by that slab's max
    part[t]    = attn[t] × V[t0..t0+tl)        # AV GEMM, row0 = t0, beta = 0
w[t] = exp(max[t] − M)·sum[t] / Σ_t' …     # M = max over slabs (slab_weights)
out  = Σ_t w[t]·part[t]                    # slab_merge
```

Mathematically this is the same softmax as the flat path (out = Σ e^{x−M} v /
Σ e^{x−M}); the differences are that each slab's max is not the global one and
that `part[t]` is rounded to f16 before the merge.  That is the same *class* of
trade as the split-K decode attention, and the transcript is the gate: 5-minute
`5m.wav` is byte-identical between the two paths, 6-minute `6m.wav` differs in
one contraction (`you're`/`you are` — the same word that already differs from
the Python reference), and both paths are byte-identical on all 12 fixtures
(`QASR_SLAB=on` forced).

Scratch is `[nqh, mp, T]` f16 (398 MB at 12k tokens, and *constant* in `s` up to
the next slab) plus `[n_slab, mp, nqh·hd]` for the per-slab AV outputs (597 MB
at 12k tokens) — against 2 × 4.7 GB before.  A 15-minute clip peaks at ~5.5 GiB
of the 8 GB card.

## The numbers (0.6B, one job at a time, SM clock 1873–1911 MHz)

| | flat path | slab path |
|---|---|---|
| `5m` (3 915 tok, 4 slabs) | prefill 2 848 ms | 2 857 ms, **text identical** |
| `6m` (4 695 tok, 5 slabs) | prefill 3 968 ms | 4 165 ms (+5 %) |
| `10m` (7 815 tok, 8 slabs) | **OOM** | prefill 8 115 ms |
| `15m` (12 065 tok, 12 slabs) | refused (binding limit) | prefill **16 259 ms**, decode 76 556 ms, 3 343 tokens out, 15 757 chars, RTFx **9.4** |

The flat path's 10-minute case used to run (7 203 MiB peak, with a 1 GiB KV);
it no longer does because `DECODER_MAX_SEQ` had to grow for the 15-minute
transcript (9 216 → 16 384, i.e. KV 1.05 → 1.9 GiB) and its O(s²) scratch is
3.9 GB at 7 815 tokens.  The default above 4 096 tokens is the slab path, so
this only affects `QASR_SLAB=off` A/B runs.

## Where the time went (and the one real optimisation)

The first working version was **2.6× slower than the flat path** at 6 minutes
(10 214 vs 3 968 ms).  The `QASR_DUP` ablations put 3 752 ms of that in the
softmax alone, with `slab_stats` (which has no ablation hook) hiding in the
remainder: both dispatched **one 1024-thread workgroup per score row per slab**,
and per slab most rows have no live column at all (a row is only affected by
slabs that start before it).  Two changes fixed it:

* `valid == 0` rows now take a workgroup-uniform fast path — they write the
  zeros the AV GEMM expects and skip the two barrier trees (the branch depends
  on the row alone, so skipping the barriers is legal);
* the reduction block is `SLAB_BS = 256` instead of the slab width, so a thread
  owns four columns and four times as many workgroups fit per SM.

Result: prefill 10 214 → 4 165 ms at 6 minutes, and the whole slab path is now
within 5 % of the flat one below 10 minutes while *faster* above it (its scores
and AV GEMMs beat the flat path's by ~2× thanks to the smaller working set).

## Traps worth remembering

1. **A bind group entry's offset is added to the dynamic offset.**  Writing both
   (entry `offset: slot·256`, dynamic offset `slot·256`) makes the dispatch read
   `2·slot·256` — for the stats dispatch that is the *weights* cfg, so the
   statistics came out as `valid = 1` nonsense, the merge weights were garbage
   and the transcript was empty with no error anywhere.  The slot address lives
   in the dynamic offset alone.
2. **The stats/weights row index is `head·mp + pos`, not the dispatch's flat row
   index** (the row grid spans `nqh·s`, not `nqh·mp`).
3. **The merge's two operands are laid out differently**: statistics/weights are
   head-major (`head·mp + pos`), the AV output is the GEMM's own C layout
   (`pos·nqh·hd + head·hd + …`).  One index cannot drive both.
4. **The softmax cannot normalise the slab in place**: wgpu rejects one buffer
   bound as both read-only and read-write in a single dispatch, so there are two
   slab buffers (`p.scores`, `p.attn`).
5. **The audio tower shares `shaders::prefill_gemm` and has its own `GDims`.**
   Adding `lda` to the decoder's struct without adding it to the tower's made
   the tower read garbage for the A row stride — silently, because the uniform
   slot's tail is just uninitialised bytes.  Every transcript came out empty
   *with no GPU error*, and the tell was that `--cpu-enc` worked.  `GDims` now
   has both `row0` and `lda` documented as shared.
6. **`repeat_kv` only writes rows `0..cur`,** but the AV GEMM reads V rows
   `cur..cur16` (multiplied by softmax zeros).  `0 · NaN = NaN`, so the tails of
   `k_rep`/`v_rep` are zeroed explicitly — the slab path reads a much longer tail
   than the flat one.
7. **Rows past the last position have no statistics**, so `total == 0` in
   `slab_weights`; `0 · (1/0)` would be a NaN the merge would hand to the padded
   output rows.  Guarded.
8. **TDR is wall-clock, not layers.**  Submitting every 4 layers is fine while a
   layer is ~100 ms, and loses the device once the slab path spends ~1 s per
   layer (first `10m` attempt: `map token` failure mid-prefill).  The interval is
   2 layers from 4 096 tokens up.
