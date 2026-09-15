# HANDOFF — Qwen3-ForcedAligner → wgpu

这是**新窗口的第一份文档**：一个刚搭起来的骨架仓库，目标是把
`D:\Qwen3-ASR`（官方 Python 实现）里的**强制对齐**部分移植成 Rust + wgpu。

## 0. 现在里面有什么

从姊妹仓库 `D:\qwen3-asr-wgpu`（已完工、12/12 门禁全绿）整包搬来的**与模型无关的地基**：

| 文件 | 作用 |
|---|---|
| `src/gpu.rs` | 设备选择（**运行时**优先：`cpu`/`vulkan`/`dx12`/`metal`/`gl`，`auto` 默认）、`--list-devices`、**落盘管线缓存**（Vulkan/Metal 有效，D3D12 无此特性）、分块上传、适配器能力探测（含 subgroup 宽度门禁） |
| `src/mel.rs` | 与 torch 对齐的 log-mel 前端 + wav 读取（vendored soxr HQ 重采样，`third_party/`） |
| `src/weights.rs` | safetensors / mmap 权重访问，f16 字面布局 |
| `src/mrope.rs` | MRoPE 表（若对齐器的文本塔用同一套则直接可用） |
| `src/config.rs`、`src/cpu_tensor.rs` | 配置解析；参考路径用的小 CPU 张量 |
| `docs/PORTING.md` | **移植方法论**：11 个坑 + 死路表 + 必须保持的不变量（**先读这个**） |
| `docs/design-tiled-prefill.md` | ASR 那边长序列分块注意力的设计/实现记录（长音频对齐时会用到同样的思路） |
| `tools/verify_all.ps1` 等 | 门禁脚本、FLEURS 抓取、打分脚本（路径/模型名要按本仓库改） |

**待办第一件**：`cargo check`（本仓库还没有 `target/`，第一次会编 wgpu，几分钟）——
搬来的模块里有引用缺失的符号（例如别处的 `shaders`/`decoder` 模块），编译器的报错就是
「对齐器必须自己补哪些东西」的清单。

## 1. 移植前必须先有的东西（原版跑通 + gold）

对齐器的 `-hf` 权重**本机还没有**，HF 也不通 ⇒ 走 ModelScope：

```powershell
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B', local_dir=r'D:\Qwen3-ASR\models\Qwen3-ForcedAligner-0.6B-hf')"
```

然后**用 transformers 原生路径**（**不要**用 `examples/example_qwen3_forced_aligner.py`：
它 `from qwen_asr import ...`，而那个包在本机 env 里 import 就崩 —— vendored modeling 与
transformers 5.17 的 `check_model_inputs` 签名不匹配；ASR 那轮也是这么绕开的）：

* 权威参考实现：`D:\Qwen3-ASR\qwen_asr\inference\qwen3_forced_aligner.py`
* 权威 API/后处理：`D:\Qwen3-ASR\...(transformers 版) processing_qwen3_asr.py`
  里的 `decode_forced_alignment` 与 **`split_words_for_alignment`**
* 最短用法参考：`D:\Qwen3-ASR\examples\example_qwen3_forced_aligner.py`
* 依赖：ja 分词需要 `nagisa`、ko 需要 `soynlp`（其它 CJK 逐字，拉丁按词）——
  **分词策略直接决定时间戳粒度**，移植时必须一致

产出的 gold 建议落成 `tools/gold/<clip>.tsv`，每行 `word<TAB>start<TAB>end`。

## 2. 门禁纪律（照抄 ASR 那套，别省）

1. **先有 gold 再写代码**：官方 transformers 在 `-hf` 权重上的输出**冻结**成文本/TSV 基线。
2. **逐字对齐才算过**：移植版的时间戳文本化后与 gold **逐字符相同**（数值容差另行规定，
   但「词序列/切分位置」必须一致）。
3. **改动一律先测后改**：每次优化跑同一组用例；退步或改变输出 ⇒ **回退并记录**（`docs/PORTING.md` 的「测得但没采纳」表就是这么来的）。
4. **单作业**：跑 GPU 时不要把游戏/其它推理挂上去；记录 SM 时钟，便于跨机比较。
5. **死路也要记**：没采纳的方案写进文档，避免下一个人重踩。

## 3. 目标与顺序

* **初期（本阶段）**：就用 ASR 那 6 个 fixture 音频 + 它们的转写文本做对齐，和官方逐字比。
  （fixture 音频在 `D:\qwen3-asr-rs\tests\fixtures\`；ASR 的 gold 文本在
  `D:\qwen3-asr-rs\docs\baseline\texts\`，可直接当对齐器的输入文本。）
* **之后**：**所有支持的语言都要对齐比对**（要覆盖上游支持的语言列表，
  `get_supported_languages` 那套 API 在姊妹仓库 `src/processor.rs` 里已有形状）。
* **性能**：先正确、再分相位计时（mel / encoder / prefill / decode），最后才谈 RTFx；
  对齐器的序列长度是「音频 token + 文本 token」，通常短，长音频才需要分块注意力。

## 4. 建议的落地顺序

1. `cargo check` 修骨架 → 读 `docs/PORTING.md`。
2. 下模型 + 用 transformers 跑通 + 落 gold（**这一轮的全部产出就是 gold**）。
3. 写 `src/aligner.rs`（模型结构）与 `src/bins/align.rs`（CLI），先走 CPU（`cpu_tensor` 那套）
   或直接 GPU 单 kernel 验证，再逐步对齐数值。
4. 门禁脚本改造（`verify_all.ps1` 的模型名/基线路径）→ 6 fixture 全绿。
5. 再扩语言、再谈长音频分块、最后谈优化。
