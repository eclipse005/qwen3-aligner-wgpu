# Qwen3-Aligner wgpu

**基于 wgpu 的 Qwen3-ForcedAligner 强制对齐（Rust 实现）。**

[English](README.md) · **简体中文**

[Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)（来自官方 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 项目）的轻量级跨平台 Rust 实现，使用 [wgpu](https://github.com/gfx-rs/wgpu) 进行 GPU 加速。

目标很简单：给定一段音频**和**对应的转录文本，给出每个词 / 字在音频中的起止时间——**在本地原生运行**，不依赖 Python，也不依赖特定厂商的 GPU 运行时。整个对齐是一次前向计算。

### 特性

* 🦀 纯 Rust
* 🎮 wgpu GPU 加速
* 🌍 跨平台 GPU 支持
* 🖥️ Windows / macOS / Linux
* ⚡ CPU 回退
* 📦 离线本地推理
* ⏱️ 词级起止时间戳
* 🗣️ 支持 11 种语言
* 🇯🇵 内置日语分词
* 🧩 CLI + Rust 库

### 安装

作为 Cargo 依赖：

```toml
[dependencies]
qwen3-aligner-wgpu = { git = "https://github.com/eclipse005/qwen3-aligner-wgpu.git" }
```

或从源码构建 CLI：

```bash
git clone https://github.com/eclipse005/qwen3-aligner-wgpu.git
cd qwen3-aligner-wgpu
cargo build --release        # target/release/align
```

| Feature | 说明 |
|---------|------|
| `ja`（默认开启） | 日语分词，经由编译进二进制的纯 Rust nagisa 实现。关闭后其他语言照常工作，仅日语请求会报错 |

### 模型下载

权重**不在**本仓库内。请从 ModelScope 下载名称以 **`-hf`** 结尾的检查点（版权归原作者）：

```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B-hf',
                  local_dir='models/Qwen3-ForcedAligner-0.6B')
```

ModelScope 上有两个同名仓库，请选择以 `-hf` 结尾的那个；另一个使用旧版布局，无法加载。

### 快速上手

```bash
align --audio speech.wav --text "hello world" --language English
align --audio speech.wav --text transcript.txt --language English --output out.json
```

每个结果条目包含 `text`、`start_time`、`end_time`（单位：秒）。

| 参数 | 说明 |
|------|------|
| `--audio <file>` | 音频文件（WAV） |
| `--text <file/文本>` | 转录文本：填一个存在的文件路径则读文件，否则将参数本身作为文本 |
| `--language <name>` | 语言，如 `English`、`Chinese`；日语、韩语依赖它选择分词方式 |
| `--output <json>` | 结果写成 JSON；不填则按 `词<TAB>开始<TAB>结束` 打印 |
| `--model <dir>` | 模型目录（或环境变量 `QALIGN_MODEL`） |
| `--device <name>` | 强制指定设备；`cpu` 表示强制 CPU。默认自动挑选最佳设备 |
| `--dtype <f16\|bf16>` | 16 位权重与激活的存储格式。默认 `f16`——该格式与参考实现的复现结果最接近；按该格式存储的检查点也可选 `bf16` |
| `--list-devices` | 列出本机可用设备 |

### 作为库使用

```rust
use qwen3_aligner_wgpu::align_inference::Aligner;
use qwen3_aligner_wgpu::gpu::DeviceSelector;

let mut aligner = Aligner::load(DeviceSelector::parse("auto")?, std::path::Path::new("model"))?;
let items = aligner.align(std::path::Path::new("speech.wav"), "hello world", Some("English"))?;
for it in &items {
    println!("{:.3}s - {:.3}s  {}", it.start_time, it.end_time, it.text);
}
```

`AlignItem` 包含 `text`、`start_time`、`end_time`（秒）。`align` 接收 `&mut self`——一个实例同一时刻只执行一次对齐。此外还提供：`align_samples`（内存中的 16 kHz 单声道音频）、`align_with_raw`（修复前的原始时间戳）、`load_with_dtype`（选择 16 位存储格式），以及 `split_words` / `decode_timestamps` / `fix_timestamps`（流水线级控制）。完整 API 见 `cargo doc`。

### 音频输入

推荐 **16 kHz 单声道 WAV**——这是模型的原生采样率，不做任何重采样。其他采样率会自动转换；对结果要求严格时可先用 ffmpeg 自行转换：

```bash
ffmpeg -i input.flac -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

目前仅支持 WAV。

### 时间戳粒度

时间戳的粒度等于分词的粒度，不同语言的切法不同：

| 语言 | 切分方式 |
|------|----------|
| 日语 | 按词素切（`女子` / `アナ` / `の` / `仕事`） |
| 韩语 | 按空格切，一整段算一个 |
| 中文 | 每个汉字算一个 |
| 其他语言 / 混合 | 按空格切；标点会被丢弃但不拆词（`50-minute` 切为 `50minute`） |

### 为什么选择 wgpu？

本项目不依赖 CUDA、ROCm 或其他特定厂商的运行时，而是以 **wgpu** 作为统一的 GPU 抽象层。

这使得同一套 Rust 对齐运行时可以覆盖不同平台与不同显卡厂商。

### 项目状态

🚧 **积极开发中**

性能与硬件兼容性仍在不同显卡上持续优化与测试。

### 相关项目

* [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) —— 官方模型项目
* [wgpu](https://github.com/gfx-rs/wgpu)
* [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu) —— 语音识别（转写）

### 许可证

Apache-2.0。`third_party/soxr` 下的重采样器为 LGPL-2.1-or-later，详见该目录。

本仓库是**独立的 Rust 推理实现**，用于加载并运行官方发布的 Qwen3-ForcedAligner 权重，并非 Alibaba / Qwen 官方发行版，与原作者无隶属关系。模型权重版权归原作者所有。
