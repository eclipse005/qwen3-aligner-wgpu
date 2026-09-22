# qwen3-aligner-wgpu

[Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) 的 Rust 实现，基于 wgpu（官方项目见 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)）。

强制对齐是转写的逆过程：给它一段音频**和**这段音频的转录文本，它给出每个词/字在音频里的起止时间。整个对齐是一次前向计算，没有逐字生成的循环；有 GPU 时跑在 GPU 上，没有时自动用 CPU。

## 安装

作为依赖加入 `Cargo.toml`：

```toml
[dependencies]
qwen3-aligner-wgpu = { git = "https://github.com/eclipse005/qwen-aligner-wgpu.git" }
```

构建命令行工具：

```bash
cargo build --release        # 得到 target/release/align
```

| Feature | 说明 |
|---------|------|
| `ja`（默认开启） | 日语分词。内置纯 Rust 的 nagisa，词表和模型直接编译进二进制；关掉后其它语言照常工作，只有日语会报错 |

## 模型下载

从 ModelScope 下载 **`-hf`** 那份（版权归原作者）：

```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B-hf',
                  local_dir=r'models/Qwen3-ForcedAligner-0.6B')
```

ModelScope 上有两个同名仓库，要选 `-hf` 结尾的那个；另一个是旧布局，这里加载不了。

## 使用

### 命令行

```bash
align --audio speech.wav --text transcript.txt --language English
align --audio speech.wav --text "hello world" --language English --output out.json
```

| 参数 | 说明 |
|------|------|
| `--audio <file>` | 音频文件（WAV） |
| `--text <file/文本>` | 转录文本：填一个存在的文件路径就读文件，否则把参数本身当文本 |
| `--language <name>` | 语言，如 `English`、`Chinese`；日语/韩语要靠它走对应的分词 |
| `--output <json>` | 结果写成 JSON（`text` / `start_time` / `end_time`，单位秒）；不填则按 `词<TAB>开始<TAB>结束` 打印 |
| `--model <dir>` | 模型目录（也可用环境变量 `QALIGN_MODEL`） |
| `--device <name>` | 指定设备，默认自动；`cpu` 表示强制用 CPU |
| `--raw <json>` | 另外写出**修复前**的原始毫秒流（模型自己的 argmax），用于对照参考实现 |
| `--list-devices` | 列出这台机器上可用的设备 |

### 作为库

```rust
use qwen3_aligner_wgpu::align_inference::Aligner;
use qwen3_aligner_wgpu::gpu::DeviceSelector;

let mut aligner = Aligner::load(DeviceSelector::parse("auto")?, std::path::Path::new("model"))?;
let items = aligner.align(std::path::Path::new("speech.wav"), "hello world", Some("English"))?;
for it in &items {
    println!("{:.3}s - {:.3}s  {}", it.start_time, it.end_time, it.text);
}
```

## API

### `Aligner`

| | |
|---|---|
| `Aligner::load(selector, model_dir)` | 加载模型；`selector` 用 `DeviceSelector::parse("auto")` 得到 |
| `align(audio, text, language)` | 对齐，返回 `Vec<AlignItem>` |
| `align_with_raw(audio, text, language)` | 同上，另外给出修复前的原始时间戳（毫秒） |
| `align_samples(&samples, text, language)` | 音频已解码好时用：16 kHz 单声道、取值范围 [-1, 1] |
| `supported_languages()` | 支持的语言（11 种） |

```rust
pub struct AlignItem {
    pub text: String,      // 词 / 字
    pub start_time: f64,   // 开始时间，秒
    pub end_time: f64,     // 结束时间，秒
}
```

`align` 取 `&mut self`，一个实例同时只做一次对齐。

### 其它导出

- `list_devices()`：列出可用设备
- `load_audio_wav(path, sample_rate)`：读 wav，必要时重采样
- `split_words(text, language)`：按模型的时间戳粒度分词（见下）
- `decode_timestamps(words, raw_ms)` / `fix_timestamps(ms)`：把原始时间戳整理成 `AlignItem`

## 音频输入

推荐 **16 kHz 单声道 WAV**——这是模型的原生采样率，不做任何重采样。其它采样率会自动转成 16 kHz，对结果要求严格的话可以先用 ffmpeg 转好：

```bash
ffmpeg -i input.flac -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

目前只支持 WAV。

## 分词粒度

时间戳的粒度就是分词的粒度，不同语言切法不同：

| 语言 | 怎么切 |
|------|--------|
| 日语 | 按词素切（`女子` / `アナ` / `の` / `仕事`），不是一个字一个 |
| 韩语 | 按空格切，一整段算一个 |
| 中文 / 英文 / 混合 | 每个汉字算一个，其余按空格切；标点会被丢掉但不拆词（`50-minute` 切出来是 `50minute`） |

## License

Apache-2.0。`third_party/soxr` 下的重采样器是 LGPL-2.1-or-later，协议见该目录。

## 致谢

本仓库是**独立的 Rust 推理实现**，用于加载并运行官方发布的 Qwen3-ForcedAligner 权重，**不是** Alibaba / Qwen 官方发行版，与原作者无隶属关系。使用模型权重时请遵守原作者的许可证。转写（语音识别）见同系列的 [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu)。
