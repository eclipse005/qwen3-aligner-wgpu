# Forced-aligner baseline (frozen gold)

The gate for this port is **textual identity of the aligned word sequence and the
split positions**, exactly like the ASR round.  Before any Rust is written the
official path is frozen here.

* generator: `D:\Qwen3-ASR\run_hf_aligner.py` (this repo's `tools/gold/` is its output)
* weights:   `D:\Qwen3-ASR\models\Qwen3-ForcedAligner-0.6B-tf`
  — `sha256(model.safetensors) = 00568245ceca5af1991d28562a75fe1ddc9bfeb041c27fda66947ea05c47fb86`
* env:       conda `asr` — python 3.10.0, torch 2.8.0+cu128, transformers 5.17.0,
  NVIDIA P104-100 (sm_61), driver 572.75
* gold:      `tools/gold/<clip>.tsv` — `word <TAB> start <TAB> end`, seconds, 3 decimals

## 1. How to get the weights

**The HF repo id in `HANDOFF.md` is the wrong one.**  ModelScope hosts *two*
checkpoints and they are not the same layout:

| repo id | layout | loads with |
|---|---|---|
| `Qwen/Qwen3-ForcedAligner-0.6B` | original — `thinker.*` keys, config has nested `thinker_config`, ships `chat_template.json`/`vocab.json` | the `qwen_asr` package (unimportable here) |
| `Qwen/Qwen3-ForcedAligner-0.6B-hf` | native — `model.audio_tower.*` / `model.multi_modal_projector.*` / `model.language_model.*` / `score.weight`, ships `processor_config.json`/`tokenizer.json` | `transformers` 5.17 `AutoProcessor` + `AutoModelForTokenClassification` |

The `-hf` repackaging is a pure rename: `thinker.lm_head.weight` → `score.weight`,
`thinker.model.*` → `model.language_model.*`, `thinker.audio_tower.proj{1,2}` →
`model.multi_modal_projector.linear_{1,2}`, and the tensors are **byte-identical**
(verified for `score`, `embed_tokens`, `q_proj`, `audio_tower.layers.0.fc1`,
`multi_modal_projector.linear_1`).

`modelscope` 1.38.1 on python 3.10 crashes on import-time config construction:
`modelscope-hub` 0.1.8 declares `@dataclass(slots=True)` with `init=False` fields
(`_logged_out`, `_endpoint_overridden`) and CPython < 3.11 never assigns those
defaults, so `load_token()` raises `AttributeError: 'HubConfig' object has no
attribute '_logged_out'`.  Setting `MODELSCOPE_API_TOKEN` skips `load_token()`
entirely and is enough to make it work:

```powershell
$env:MODELSCOPE_API_TOKEN = "anonymous"
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B-hf', local_dir=r'D:\Qwen3-ASR\models\Qwen3-ForcedAligner-0.6B-tf')"
```

## 2. How the gold is produced

`transformers` native path — **not** `examples/example_qwen3_forced_aligner.py`,
which imports `qwen_asr` and dies on `check_model_inputs`.  Input construction and
decoding are verbatim ports of `qwen_asr/inference/qwen3_forced_aligner.py`
(the authoritative implementation); the decoding call itself is the authoritative
API `Qwen3ASRProcessor.decode_forced_alignment`.

```
words      = processor.split_words_for_alignment(transcript, language)
input_text = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text
wav        = mono 16k float32, peak-normalised           # qwen_asr.inference.utils
inputs     = processor(text=[input_text], audio=[wav], return_tensors="pt")
logits     = model(**inputs).logits                      # (1, L, 5000)
ms         = argmax(logits, -1)[input_ids == 151705] * 80
fixed      = _fix_timestamps(ms)                         # LIS repair
word i     -> fixed[2i], fixed[2i+1]  (ms -> s, round 3)
```

Reproduce one clip / all six:

```powershell
$env:MODELSCOPE_API_TOKEN = "anonymous"
& "C:\Users\ADMIN\miniconda3\envs\asr\python.exe" D:\Qwen3-ASR\run_hf_aligner.py --wav 15s_en
& "C:\Users\ADMIN\miniconda3\envs\asr\python.exe" D:\Qwen3-ASR\run_hf_aligner.py --all --dtype float16
```

The generator refuses to freeze a gold unless it can prove the run is sane:

* the word list from `split_words_for_alignment` must equal a verbatim port of the
  upstream tokeniser — otherwise it exits without writing (`tokeniser divergence`);
* `#<timestamp>` tokens in `input_ids` must equal `2 x len(words)`, and there must
  be a non-zero number of `<|audio_pad|>` tokens (the processor expands them);
* logits must be finite and not all-zero (the `PORTING.md` §4.2 oracle guard);
* `decode_forced_alignment` must equal an independent port of upstream
  `parse_timestamp` item by item;
* no word may end before it starts, or fall outside the audio.

## 3. The six clips

Transcripts are the frozen ASR round's own outputs
(`D:\qwen3-asr-rs\docs\baseline\texts\python-hf_0.6B_<clip>.txt`, language taken
from their `detected_language` header), so the aligner is fed exactly the text the
ASR port is known to produce.

| clip | lang | words | seq_len | audio tok | ts tok | audio s | forward s (f16) |
|---|---|---|---|---|---|---|---|
| 15s_en | English | 39 | 323 | 195 | 78 | 15.000 | 0.359 |
| 30s_zh | Chinese | 150 | 844 | 392 | 300 | 30.092 | 0.586 |
| 90s_ja | Japanese | 210 | 1887 | 1161 | 420 | 89.327 | 1.235 |
| 90s_en | English | 249 | 1973 | 1170 | 498 | 90.000 | 1.308 |
| 180s_en | English | 414 | 3651 | 2292 | 828 | 176.309 | 3.137 |
| 180s_zh | Chinese | 749 | 4589 | 2340 | 1498 | 180.000 | 4.256 |

All six have `timestamp_tokens == 2 x words` and zero out-of-range spans.  The
180 s fixtures sit exactly on upstream's `MAX_FORCE_ALIGN_INPUT_SECONDS = 180`.

## 4. The baseline is float16, and that is a decision (measured)

The model ships bfloat16 and the official example loads `dtype=torch.bfloat16`,
but **Pascal has no bf16**, and bf16 is the *worst* of the three dtypes here.  All
three were run end to end over the same six clips and diffed line by line:

| clip | fp16 vs bf16 | fp16 vs fp32 | bf16 vs fp32 |
|---|---|---|---|
| 15s_en | 0 | 0 | 0 |
| 30s_zh | 2 | **0** | 2 |
| 90s_en | 2 | **0** | 2 |
| 90s_ja | 1 | 1 | 2 |
| 180s_en | 6 | 4 | 9 |
| 180s_zh | 13 | **0** | 13 |
| **total differing words** | 24 | **5** | 28 |

fp16 is the numerics the Rust port can actually reproduce on this card
(f16 storage, f32 accumulate — `PORTING.md` §1.5), it is *much* closer to the fp32
converged answer than bf16 is, and it matches the precedent the ASR round set with
`docs/baseline/python-hf-cuda-f16`.

The gold therefore lives under **one directory per dtype**, and a run is compared
against the one that matches the precision it computes in — `bf16` when the device
supports it, `f16` otherwise.  That is what the reference itself does: the caller
picks `torch_dtype`.

```
tools/gold/f16/<clip>.{tsv,json}     <- the gate on this machine (Pascal, no bf16)
tools/gold/bf16/<clip>.{tsv,json}    <- the gate on a bf16-capable device
tools/gold/fp32/<clip>.{tsv,json}    <- diagnostic reference, not a gate target
```

**The bf16 set here is emulated.**  sm_61 has no bf16 tensor core, so torch
produced it through an upconvert/round path that is not what cuBLAS does on
Ampere or later.  Re-run `run_hf_aligner.py --dtype bfloat16` on the target card
before treating that set as authoritative.

What is **dtype-independent**, and therefore the hard part of the gate: the word
sequence, the line count, the ordering, and `start <= end` for every word.  The
tokeniser fixes the words and the model emits exactly two timestamps per word, so
no numerical choice can move them.  Only the millisecond values shift, and only by
one 80 ms bucket.  Two consecutive fp16 runs are byte-identical on all six clips,
so what remains is not run-to-run noise but the set of words where the 5000-way
argmax has two buckets inside the floating-point error bar — a known, bounded,
documented tolerance.

## 5. Contracts the Rust port must reproduce

These were all load-bearing and none of them are guessable:

1. **The whole input sequence, exactly** (verified by dumping `input_ids` for `15s_en`;
   the only special ids present are these four):

   ```
   [151669]  <|audio_start|>                       x1   (never repeated)
   [151676]  <|audio_pad|>                         xN   N = num_audio_tokens
   [151670]  <|audio_end|>                         x1
   w0 [151705][151705] w1 [151705][151705] ... wk [151705][151705]
   ```

   **No BOS, no EOS, no chat template, no `<|im_start|>`** — the upstream wrapper
   builds the string by hand and hands it to the processor, which only expands
   `<|audio_pad|>`.  For `15s_en`: 323 ids = 1 + 195 + 1 + 78 timestamp + 48 word
   tokens, i.e. 39 words cost 48 BPE tokens, so **one word is not one token**.
   The timestamp count is exactly `2 x words`; the predictions gather in that
   order as `start_0, end_0, start_1, end_1, ...`.
2. **80 ms per timestamp class**, `timestamp_segment_time` from the config.
3. **`<|audio_pad|>` is expanded by the processor, not by the caller**, so `N` is
   whatever the feature extractor's output length formula gives —
   `_get_audio_token_length` above, i.e. `(mel//100)*13 + ((mel%100) post-conv)`.
   `15s` = 1500 mel frames = 15 chunks x 13 = 195.  `input_features` must be
   padded to a multiple of `n_window*2 = 100` mel frames.
4. **Tokenizer is language-dependent, and it is not the tokenizer.**
   `ja` uses nagisa, `ko` uses soynlp, everything else is CJK-per-character with
   space-separated words; punctuation is dropped and `'` is kept.  The hyphen in
   `50-minute` disappears *without* splitting the word — gold says `50minute`.
   For Japanese the gold is nagisa's morphemes (`女子`, `アナ`, `の`, `仕事`), **not**
   one CJK character per token.
   `<timestamp>` is in the vocabulary but is **not** registered as a special
   token; it still resolves atomically, which is why the BPE never merges across
   a word boundary.
5. **Two different mel counts, and they drive different things.**  The feature
   extractor right-pads the mel axis to a multiple of `n_window * 2 = 100` and
   pads with **0.0** — not the log-mel floor `(max-4)/4`, so the padded region is
   a value the encoder actually sees.  But `n_audio_tokens` is computed from the
   *unpadded* attention-mask sum, `floor(n_samples / hop)`:
   `(valid // 100) * 13 + conv3(valid % 100)`, where `conv3` applies
   `(L-1)/2+1` three times and maps 0 to 0.
   `90s_ja` is the case that shows the split: 1 429 235 samples -> 8 932 valid
   frames -> 1 161 audio tokens, in an array 9 000 wide.
6. **`_fix_timestamps` is part of the output**, not a nicety.  Its LIS repair has
   both branches exercised by the gold: short outlier blocks snap to a neighbour
   *by index distance, not value distance*, blocks longer than 2 linearly
   interpolate — `180s_zh` contains values like `127536` that are not multiples of
   80 ms.  The interpolation is `left + (right-left)/(n+1) * k` in f64 and then
   truncated with `int()`, i.e. **toward zero**, not rounded, and the pairing into
   words happens *after* the repair.  (`round(ms/1000.0, 3)` is a no-op on an
   integer millisecond count and is deliberately not reproduced.)
7. **Audio front end**: `librosa.load(..., sr=None, mono=False)` → channel-mean
   mono → `librosa.resample` to 16 k (soxr_hq, matching this repo's vendored
   resampler) → divide by peak only if peak > 1 → clip to [-1, 1].

## 6. What the model actually is

`Qwen3ASRForTokenClassification` (`modeling_layers.GenericForTokenClassification`
with `base_model_prefix = "model"`):

```
input_features (1, 128, T_mel)          T_mel a multiple of n_window*2 = 100
  audio_tower:  3x [conv2d k3 s2 p1 + gelu] -> conv_out Linear(7680 -> 1024)
                -> + sinusoid positional embedding[:time_steps]
                -> 24 encoder layers, windowed attention (n_window_infer = 800)
                -> ln_post
  multi_modal_projector: linear_1 (1024->1024) -> gelu -> linear_2 (1024->1024)
  scatter the projected frames into every input_ids == audio_token_id (151676)
  language_model: qwen3, 28 layers, hidden 1024, 16 heads / 8 kv heads,
                head_dim 128, intermediate 3072, rms_eps 1e-6, rope_theta 1e6,
                max_position_embeddings 8192
  score: Linear(1024 -> 5000), no bias        (token_classification_bias = false)
  argmax -> gather at <timestamp> -> _fix_timestamps -> pair into words
```

Differences from the ASR port that matter for the module list:

* **there is no generation loop.**  One forward over the whole sequence, no KV
  cache, no sampling, no detokenisation loop.  Everything the ASR port needed for
  incremental decode is not needed here.
* the classifier head is 5000-wide instead of a 152064-wide vocab projection, and
  `embed_tokens` is only used to embed text (audio frames are scattered in).
* the audio tower's `output_dim` is 1024 here (it equals `d_model`), so
  `proj1`/`proj2` are 1024x1024 rather than widening.
* the text tower in the *native* config carries no MRoPE section — plain
  `rope_theta = 1e6`, `rope_type = default`.  The original-layout config does list
  `mrope_section = [24, 20, 20]`, so the two configs disagree; the native one is
  what produced this gold, and `src/mrope.rs` may therefore be dead weight here.

## 7. Port status

**Complete.**  Three device paths share one gate: `cargo test` (51), the input
check (`align --check-input --all`, 6/6), and the fixtures (`align --all`).

| module | what it is |
|---|---|
| `src/words.rs` | the word splitter: nagisa for `ja`, Korean whitespace, CJK-per-character otherwise, `_clean_token`'s Unicode *general category* test (not Rust's `is_alphabetic`, which also accepts `Other_Alphabetic` marks), punctuation dropped without splitting `50-minute` |
| `src/align_input.rs` | `conv3` / `audio_token_count` / mel padding, the BPE, and the `[151669] + [151676]xN + [151670] + words` assembly |
| `src/postprocess.rs` | `_fix_timestamps` (LIS + snap-by-index / interpolate-toward-zero) and the word pairing |
| `src/align_inference.rs` | the forward: a `Backend` enum over the towers, then shared host code for the gather, the scatter, the final norm and the 5000-way head |
| `src/gold.rs` | the freeze, the dtype policy, and the margin gate |
| `src/bin/align.rs` | CLI: `--wav` / `--all` / `--eval` / `--check-input` / `--list-devices` |
| `src/shaders.rs`, `decoder.rs`, `audio_encoder{,_gpu}.rs`, `cpu_decoder.rs` | lifted from `D:\qwen3-asr-wgpu` (verified there at 12/12) |

### What the gate says

| backend | 6 fixtures | 10 languages x 20 clips | RTFx |
|---|---|---|---|
| NVIDIA Vulkan | 4/6 bit-exact, rest within a bucket | words 200/200, timestamps 195/200, **margin gate PASS** | 33.0-39.4 |
| NVIDIA D3D12 | same | — | 29.3-32.5 |
| Intel iGPU Vulkan | same (its `subgroup 8..32` gate holds) | — | 4.07-5.22 |
| CPU | **6/6** | — | 5.86-12.53 |

### The margin gate, and why the criterion is where it is

A timestamp is an argmax over 5000 buckets 80 ms apart, so a sub-ULP difference
anywhere in the towers can move one by a bucket.  Every divergence this port has
ever produced sits at a position whose top-1/top-2 logit gap is **at most
0.00781**, and at those positions the reference does not agree with *itself*:
`180s_en` index 253 is chosen differently by CUDA-fp16 than by CUDA-fp32,
CPU-fp32 and CPU-fp16, and the fp16 margin there is exactly `0.00000`.  The
next-lowest margins in that clip are 0.0098 and up.

So the gate is: **no endpoint may move where the reference's margin exceeds
`MARGIN_NOISE_FLOOR` (0.01)**; moves below it are counted and reported, never
silently accepted.

Two things about *where* it is applied, both learned by getting them wrong:

* **On the raw argmax, not the repaired value.**  `margin[i]` describes the
  reference's argmax at position `i`; after `_fix_timestamps` the value at `i`
  can come from elsewhere, so judging a repaired value against `margin[i]` is an
  index mismatch — it produced a spurious failure until it was moved.
* **fp32 is the oracle, fp16 is the upstream-default reference.**  The
  reference's CPU and CUDA runs are *identical* at the same dtype (3622
  endpoints, zero differences), so dtype — not device — is the axis that
  matters.  This port stores f16 and accumulates in f32, and therefore lands on
  the fp32 answer.  Matching fp32 costs nothing: the source weights are bf16
  (8 significand bits), which f16 holds losslessly, so storing them as f32 would
  double the weight-read bandwidth of a bandwidth-bound GEMV and add no
  information.

### Known limits

* **Cantonese is untested** — ModelScope's FLEURS mirror has no `yue_hant_hk`
  config, which `align-eval.py` reports rather than skipping silently.
* **Only Korean was run over its full split** (382 clips); the other nine
  languages were sampled at 20 each, matching the ASR round's methodology.  Full
  gold for all ten exists on disk and is regenerable.
* **Audio past ~240 s is unverified for correctness.**  It *runs* (`long_zh`
  probes at 120 s and 240 s complete), but the reference cannot produce a
  baseline that long — it OOMs at 240 s on 8 GB, and on CPU as well.  The port's
  own ceiling for Chinese is around 320 s, where `max_seq = 8192` is reached.
* **Batching is shape-compatible, not padded.**  `align_batch` loops; the
  reference runs one left-padded forward with an attention mask, which the
  lifted `prefill` does not support.
* **The two gate paths do not use the same criterion yet.**  `align --eval`
  applies the margin gate, because `align_eval.py` records `margin` in its gold.
  `align --wav/--all` still uses the plain bucket tolerance, because
  `run_hf_aligner.py` does not record it — so `180s_en` (whose two moved
  endpoints have margins of 0.00171 and 0.00213) is reported as `1 not passing`
  there and PASS in the eval path.  Adding `margin` to `run_hf_aligner.py` and
  regenerating the six-fixture gold closes this.
