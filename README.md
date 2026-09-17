# qwen3-aligner-wgpu

[Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) 的 Rust + wgpu 实现（官方项目见 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)）。强制对齐是转写的逆过程：音频**和**它的转录文本一起进，出来的是每个词/字的位置。

模型对 `[音频 token][词与 <timestamp> 交错]` 做一次因果前向，在 `<timestamp>` 位置从一个 5000 路分类头读出时间戳——一个片段一次前向，没有自回归循环。同一份 WGSL 内核由 **Vulkan / Metal / D3D12 / OpenGL(ES)** 任意一条运行时驱动，另有自带的 CPU 后端，零深度学习框架依赖。

## 多后端支持

选择轴是**运行时**，不是硬件厂商：同一张卡可以通过多条运行时走到，而那是不同的代码路径（数值也可能不同），所以厂商只出现在设备列表里。`--device` 接受下面这些写法：

| 写法 | 说明 |
|------|------|
| `auto` | 默认：独显 → 集显 → 兜底 |
| `vulkan[:N]` | Vulkan |
| `metal[:N]` | Metal（macOS / iOS） |
| `dx12[:N]` | D3D12；wgpu 在 Windows 上走 D3D12 compute，没有 DirectML（那是另一套 API） |
| `gl[:N]` | OpenGL / GLES，wgpu 的兼容运行时，能力最弱，老机器只剩它 |
| `cpu` | 自带的 CPU 后端，不创建 adapter |
| `nvidia` / `intel` … | 适配器名子串（兜底写法） |

`align --list-devices` 打印每个 adapter，以及决定 kernel 走法的 limits——binding size、workgroup storage、subgroup 宽度是否正好 32。

同一台机器（NVIDIA P104-100）上每条运行时跑同一组六条 fixture（15 s ~ 180 s）：

| 后端 | RTFx |
|------|------|
| NVIDIA Vulkan | 33.0 – 39.4 |
| D3D12 | 29.3 – 32.5 |
| Intel 集显（Vulkan） | 4.07 – 5.22 |
| CPU | 5.86 – 12.53 |

Intel 的 adapter 报的是 `subgroup 8..32`，shuffle 归约因此走宽度检查后的 shared-memory 回退——它的结果与 NVIDIA 一致，这正是那层检查存在的意义。

**一致性**：`vulkan:0` 与 `cpu` 两条路径 × 六条 fixture，输出 JSON 逐字节相同（文本与起止时间，12/12）；一条 352 s 的音频（把 KV cache 涨到 7 168 slot）同样。

## 分词器

不同语言走不同的切分路径，因为时间戳的粒度就是切分的粒度：

| 语言 | 实现 | 粒度 |
|------|------|------|
| 日语 | nagisa（纯 Rust，`ja` feature） | 形态素（`女子` / `アナ` / `の` / `仕事`），**不是**逐字 |
| 韩语 | 空格切分 | 每个空格段整块（不接词典：带词典的 `soynlp.LTokenizer` 会切出另一份词表，时间戳粒度就对不上了） |
| 中 / 英 / 混合 | 每个 CJK 字一个 token，其余按空格段 | 标点丢弃但**不切断**词：`50-minute` → `50minute` |

日语的 nagisa 权重与词表在编译期打进二进制，不需要 `<model_dir>/nagisa/` 这类目录，也不必随模型单独下载。

## 安装

```toml
[dependencies]
qwen3-aligner-wgpu = { git = "https://github.com/eclipse005/qwen-aligner-wgpu.git" }
```

| Feature | 说明 |
|---------|------|
| `ja`（默认开启） | 日语分词。关掉后其它语言照常工作，只是日语会明确报错 |

命令行工具：`cargo build --release`，产物是 `target/release/align`。

## 音频输入

**建议输入 16 kHz 单声道 WAV**：这是模型的原生采样率，不做重采样。其它采样率交给内置的 soxr（`third_party/soxr`）重采样到 16 kHz；44.1 kHz 这类非整数比会有过渡带差异，要完全一致就先转码：

```bash
ffmpeg -i input.flac -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

只读 WAV；MP3 等先用 ffmpeg 转。

## 使用

### 命令行

```bash
align --audio speech.wav --text transcript.txt --language English
align --audio speech.wav --text "hello world" --language English --output out.json
align --list-devices
```

`--text` 指向一个已存在的文件就读它，否则就把参数本身当转录文本。不给 `--output` 时按 `word<TAB>start<TAB>end` 打印；给了就写 JSON（`text` / `start_time` / `end_time`，秒）。`--model <dir>` 指定 checkpoint 目录，也可以走 `QALIGN_MODEL`。`--language` 可选，但日语/韩语要给它才会走对应的分词路径。

### 作为库

```rust
use qwen3_aligner_wgpu::align_inference::Aligner;
use qwen3_aligner_wgpu::gpu::DeviceSelector;

let mut aligner = Aligner::load(DeviceSelector::parse("auto")?, std::path::Path::new("model"))?;
let items = aligner.align(std::path::Path::new("speech.wav"), "hello world", Some("English"))?;
for it in &items {
    println!("{:.3}-{:.3} {}", it.start_time, it.end_time, it.text);
}
```

`align_with_raw` / `align_samples` 另外给出修复前的时间戳流（毫秒），后者接调用方已经解码好的波形。

语言集与参考实现一致，11 种：Chinese / Cantonese / English / French / German / Italian / Japanese / Korean / Portuguese / Russian / Spanish，`Aligner::supported_languages()` 返回它；集合之外的语言会被拒绝，而不是拿一份错的词表去对齐。

## 模型下载

权重从 ModelScope 下 **`-hf`** 那份（版权归原作者）：

```python
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B-hf',
                  local_dir=r'models/Qwen3-ForcedAligner-0.6B')
```

ModelScope 上有两个同名仓库，要的是 `-hf` 那个：它存 `model.audio_tower.*` / `model.language_model.*` / `score.weight`；另一个是原始布局（`thinker.*` 张量名），在这里加载不了。

官方项目与文档：

- 模型页：[Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- 代码与说明：[QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)

## 性能

测试环境：NVIDIA P104-100 8 GB（Vulkan），模型 Qwen3-ForcedAligner-0.6B，六条 fixture（15 s ~ 180 s）。

显存（相对显卡稳态基线的峰值增量）：六个片段 2056 – 3341 MiB，加载后常驻 2054 MiB。KV cache 按 prompt 长度分配——对齐没有生成步骤，只要 `seq` 个 slot，不预留 `seq + max_new_tokens`——容量只增不减、按 256 slot 步进，一次 resize 0.1 – 15 ms。CPU 后端 15 s 片段的峰值工作集 3795 MB。

说明：

- 权重按 f16 存、算术全程 f32。源权重是 bf16，f16 无损装得下；存成 f32 只会让带宽受限的 GEMV 多读一倍权重，而不多出任何信息。
- 文本塔用普通 RoPE，不是 MRoPE。
- `cargo test --release` 的 45 个单元测试覆盖分词边界、时间戳修复的分支、conv 长度公式、mel 滤波器组与 config 解析，不需要 checkpoint。

## 致谢 / 原版出处

本仓库是**独立的 Rust 推理实现**，用于加载并运行官方发布的 Qwen3-ForcedAligner 权重；**不是** Alibaba / Qwen 官方发行版，与原作者无隶属关系。转写（语音识别）见同系列的 [qwen3-asr-wgpu](https://github.com/eclipse005/qwen3-asr-wgpu)。

| 组件 | 原版 | 链接 | 协议（以官方页面为准） |
|------|------|------|------------------------|
| 模型权重 | Qwen3-ForcedAligner-0.6B | [Hugging Face](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | Apache-2.0 |
| 官方推理与文档 | Qwen3-ASR 仓库 | [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) | Apache-2.0 |
| 日语分词 | Python `nagisa`（本仓库通过 [nagisa-rs](https://github.com/eclipse005/nagisa-rs) 接入） | — | 见各自上游 |

使用模型权重时请遵守原作者许可证；本仓库的 Rust 推理代码以本仓库 License 为准。

## License

Apache-2.0。`third_party/soxr` 下内置的重采样器是 LGPL-2.1-or-later，协议见该目录。
