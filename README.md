# Qwen3-Aligner wgpu

**Qwen3-ForcedAligner forced alignment in Rust with wgpu.**

**English** · [简体中文](README.zh-CN.md)

A lightweight, cross-platform Rust implementation of [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) (from the official [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) project), using [wgpu](https://github.com/gfx-rs/wgpu) for GPU acceleration.

The goal is simple: given audio **and** its transcript, produce the start and end time of every word or character — **locally and natively**, without Python or vendor-specific GPU runtimes. The entire alignment is a single forward pass.

### Features

* 🦀 Pure Rust
* 🎮 GPU acceleration with wgpu
* 🌍 Cross-platform GPU support
* 🖥️ Windows / macOS / Linux
* ⚡ CPU fallback
* 📦 Offline local inference
* ⏱️ Word-level start / end timestamps
* 🗣️ 11 languages
* 🇯🇵 Japanese word segmentation built in
* 🧩 CLI + Rust library

### Install

As a Cargo dependency:

```toml
[dependencies]
qwen3-aligner-wgpu = { git = "https://github.com/eclipse005/qwen3-aligner-wgpu.git" }
```

Or build the CLI from source:

```bash
git clone https://github.com/eclipse005/qwen3-aligner-wgpu.git
cd qwen3-aligner-wgpu
cargo build --release        # target/release/align
```

| Feature | Description |
|---------|-------------|
| `ja` (on by default) | Japanese word segmentation via a pure-Rust nagisa compiled into the binary. Disabling it keeps every other language working; only Japanese requests return an error |

### Model download

Weights are **not** included in this repository. Download the checkpoint whose name ends in **`-hf`** from ModelScope (rights remain with the original authors):

```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B-hf',
                  local_dir='models/Qwen3-ForcedAligner-0.6B')
```

ModelScope hosts two repos under the same name — use the one ending in `-hf`; the other uses the old layout and cannot be loaded.

### Quick Start

```bash
align --audio speech.wav --text "hello world" --language English
align --audio speech.wav --text transcript.txt --language English --output out.json
```

Each result item carries `text`, `start_time` and `end_time` in seconds.

| Option | Description |
|--------|-------------|
| `--audio <file>` | Audio file (WAV) |
| `--text <file/text>` | Transcript: an existing path is read as a file, otherwise the argument is used as text |
| `--language <name>` | Language such as `English`, `Chinese`; Japanese and Korean rely on it for tokenization |
| `--output <json>` | Write results as JSON; without it, results print as `word<TAB>start<TAB>end` |
| `--model <dir>` | Model directory (or the `QALIGN_MODEL` environment variable) |
| `--device <name>` | Force a device; `cpu` forces CPU. Default picks the best available |
| `--dtype <f16\|bf16>` | Storage format for 16-bit weights and activations. Defaults to `f16`, which reproduces the reference implementation most closely; `bf16` is accepted for checkpoints stored that way |
| `--list-devices` | List the devices usable on this machine |

### Library

```rust
use qwen3_aligner_wgpu::align_inference::Aligner;
use qwen3_aligner_wgpu::gpu::DeviceSelector;

let mut aligner = Aligner::load(DeviceSelector::parse("auto")?, std::path::Path::new("model"))?;
let items = aligner.align(std::path::Path::new("speech.wav"), "hello world", Some("English"))?;
for it in &items {
    println!("{:.3}s - {:.3}s  {}", it.start_time, it.end_time, it.text);
}
```

`AlignItem` carries `text`, `start_time` and `end_time` (seconds). `align` takes `&mut self` — one instance performs one alignment at a time. Also available: `align_samples` for in-memory 16 kHz mono audio, `align_with_raw` for the pre-fix raw timestamps, `load_with_dtype` to select the 16-bit storage format, and `split_words` / `decode_timestamps` / `fix_timestamps` for pipeline-level control. See `cargo doc` for the full API.

### Audio input

16 kHz mono WAV is recommended — the model's native sample rate, used without resampling. Any other sample rate is converted automatically; convert it yourself first with ffmpeg when full control matters:

```bash
ffmpeg -i input.flac -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

Only WAV is currently supported.

### Timestamp granularity

The timestamp granularity equals the tokenization granularity, which differs by language:

| Language | Tokenization |
|----------|--------------|
| Japanese | by morpheme (`女子` / `アナ` / `の` / `仕事`) |
| Korean | one token per space-separated chunk |
| Chinese | one token per character |
| Other languages / mixed | whitespace-separated; punctuation is dropped but words are not split (`50-minute` becomes `50minute`) |

### Why wgpu?

Instead of relying on CUDA, ROCm, or other vendor-specific runtimes, this project uses **wgpu** as a unified GPU abstraction.

This makes it possible to build a single Rust-based alignment runtime for different platforms and GPU vendors.

### Project Status

🚧 **Active development**

Performance and hardware compatibility are still being actively optimized and tested across different GPUs.

### Related

* [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — the official model project
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) — speech recognition (transcription)

### License

Apache-2.0. The resampler under `third_party/soxr` is LGPL-2.1-or-later; see that directory for details.

This repository is an **independent Rust inference implementation** for loading and running the officially released Qwen3-ForcedAligner weights — not an official Alibaba / Qwen release, and not affiliated with the original authors. Model weights remain under the terms of their respective owners.
