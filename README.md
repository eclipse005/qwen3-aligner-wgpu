# qwen3-aligner-wgpu

Qwen3-ForcedAligner-0.6B in Rust, on **wgpu** (Vulkan / Metal / D3D12 / GL) and on
the **CPU**. Forced alignment is the inverse of transcription: audio *and* its
transcript go in, and out comes where each word sits.

The model is a single causal pass over `[audio tokens][words interleaved with
`<timestamp>` markers]`, so a clip costs one forward — no autoregressive loop.

## Running it

```
align --audio speech.wav --text transcript.txt --language English
align --audio speech.wav --text "hello world" --language English --output out.json
align --list-devices
```

Without `--output` the items print as `word<TAB>start<TAB>end`. An existing path
for `--text` is read as a file; anything else is the transcript itself.

The **checkpoint is not included.** Point `--model` (or `QALIGN_MODEL`) at the
`-hf` one:

```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B-hf',
                  local_dir=r'models/Qwen3-ForcedAligner-0.6B')
```

On ModelScope there are two repositories of that name — use the **`-hf`** one.
It stores `model.audio_tower.*` / `model.language_model.*` / `score.weight`; the
other is the original layout with `thinker.*` tensor names, which will not load
here.

## Device selection

`--device` takes a wgpu runtime, an adapter-name substring, or `cpu`:

| spec | |
|---|---|
| `auto` | the best adapter (discrete → integrated → fallback), default |
| `cpu` | the CPU backend, no GPU involved |
| `vulkan` / `dx12` / `metal` / `gl` | one runtime; `vulkan:1` picks the second adapter of it |
| `nvidia`, `intel`, … | any substring of an adapter name |

`align --list-devices` prints every adapter with the limits that decide which
kernels run — binding size, workgroup storage, and whether the subgroup width is
exactly 32 (the shuffle reduction requires that, and falls back to shared memory
when it is not).

## Notes

* Weights are stored f16 and the arithmetic is f32 throughout. Storing them as
  f32 would double the weight-read traffic of a bandwidth-bound GEMV and add no
  information: the source weights are bf16, which f16 already holds exactly.
* The text tower uses plain RoPE, not MRoPE.
* Japanese needs the `ja` feature (default on), which pulls a pure-Rust nagisa
  with its model embedded. Without it every other language still works.

## Linking as a library

```rust
use qwen3_aligner_wgpu::align_inference::Aligner;
use qwen3_aligner_wgpu::gpu::DeviceSelector;

let mut aligner = Aligner::load(DeviceSelector::parse("auto")?, std::path::Path::new("model"))?;
let items = aligner.align(std::path::Path::new("speech.wav"), "hello world", Some("English"))?;
```

`align_with_raw` and `align_samples` return the pre-repair millisecond stream
alongside the items, and take a waveform the caller already decoded.

## Licence

Apache-2.0. The vendored soxr resampler under `third_party/` is
LGPL-2.1-or-later; see that directory.
