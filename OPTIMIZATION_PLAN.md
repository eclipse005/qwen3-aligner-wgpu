# Qwen3-Aligner-WGPU Apple Silicon 优化方案

## 项目概述
这是一个基于 wgpu 的 Qwen3-ForcedAligner 实现，用于音频对齐（给定音频和文本，输出每个词/字的时间戳）。

## 当前架构分析

### 核心流程
1. **音频预处理** (mel.rs)
   - WAV 解码和重采样 (soxr)
   - Log-mel 特征提取 (rustfft)

2. **模型推理** (align_inference.rs)
   - 音频编码器 (GPU/CPU)
   - 文本解码器 (28 层 Qwen3)
   - 时间戳头部

3. **后处理**
   - 时间戳修复和对齐

## 性能瓶颈分析

### 1. GPU 路径（Metal on Apple Silicon）
- ✅ 已使用 wgpu 支持 Metal 后端
- ⚠️ 未充分利用 Apple Silicon 的统一内存架构
- ⚠️ 着色器未针对 Apple GPU 架构优化
- ⚠️ 管线编译开销大

### 2. CPU 路径
- ⚠️ 未使用 Apple Accelerate 框架的优化 BLAS
- ⚠️ soxr 重采样器性能开销大（特别是大文件）
- ⚠️ CPU 音频编码器使用通用 GEMM，未利用硬件加速

### 3. 内存管理
- ⚠️ 频繁的 CPU-GPU 数据传输
- ⚠️ 未利用统一内存优势

## 优化方案

### 阶段 1: 编译器和构建优化 (立即见效)

#### 1.1 Cargo.toml 优化
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"

# Apple Silicon 特定优化
[target.aarch64-apple-darwin]
rustflags = [
    "-C", "target-cpu=native",
    "-C", "target-feature=+neon,+fp-armv8,+crypto"
]
```

#### 1.2 添加 Apple Accelerate 框架支持
在 `Cargo.toml` 中添加：
```toml
[target.'cfg(target_os = "macos")'.dependencies]
accelerate-src = "0.3"
```

### 阶段 2: 音频处理优化

#### 2.1 优化 soxr 重采样
**问题**: 当前实现在处理大文件时性能下降明显
**解决方案**: 
- 使用分块处理（已实现 IN_BLOCK = 16KB）
- 考虑使用 Apple 的 vDSP 进行重采样

#### 2.2 优化 Mel 特征提取
**优化点**:
- 使用 Apple vDSP 加速 FFT
- 并行化 mel 滤波器组计算（已使用 rayon）
- 预分配缓冲区

#### 2.3 实现快速 PCM 解码
**已实现**: `read_pcm16_fast` 函数
**优化**: 进一步使用 SIMD 加速解码

### 阶段 3: GPU/Metal 优化

#### 3.1 Metal 着色器优化
**当前问题**: 
- 通用 WGSL 着色器未针对 Apple GPU 优化
- 未利用 Metal 特定特性

**优化方案**:
1. 使用 Metal Performance Shaders (MPS) 进行 GEMM
2. 优化工作组大小以匹配 Apple GPU 架构
3. 使用 Metal 2/3 新特性（如 simdgroup 操作）

#### 3.2 统一内存优化
**Apple Silicon 优势**: CPU 和 GPU 共享内存

**优化方案**:
- 减少不必要的内存拷贝
- 使用 wgpu 的 MAP_READ | MAP_WRITE 缓冲区
- 实现零拷贝路径（where possible）

#### 3.3 管线缓存
**已实现**: pipeline_cache 功能
**优化**: 
- 预热管线缓存
- 持久化缓存到磁盘

### 阶段 4: 计算优化

#### 4.1 使用 Accelerate 框架替换 gemm crate
```rust
// 新建 src/accelerate_gemm.rs
#[cfg(target_os = "macos")]
pub fn gemm_accelerate(
    m: usize, n: usize, k: usize,
    alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32, c: &mut [f32], ldc: usize
) {
    use accelerate_src::*;
    unsafe {
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            m as i32, n as i32, k as i32,
            alpha, a.as_ptr(), lda as i32,
            b.as_ptr(), ldb as i32,
            beta, c.as_mut_ptr(), ldc as i32
        );
    }
}
```

#### 4.2 优化音频编码器
- 使用 Accelerate vDSP 进行卷积
- NEON SIMD 优化激活函数（GELU）
- 批处理优化

#### 4.3 文本解码器优化
- 层融合以减少内存访问
- KV 缓存优化
- 使用 Metal Performance Shaders

### 阶段 5: 并行化优化

#### 5.1 音频批处理
**当前**: 单个音频序列化处理
**优化**: 支持批量音频并行处理

#### 5.2 流水线并行
- 音频加载与特征提取并行
- CPU/GPU 异步计算
- 多阶段流水线（加载 -> Mel -> 编码 -> 解码）

### 阶段 6: 内存优化

#### 6.1 对象池
- 预分配常用缓冲区
- 重用临时数组

#### 6.2 内存布局优化
- 数据对齐以优化缓存访问
- 减少内存碎片

## 实施优先级

### P0 (立即实施 - 预期提升 20-30%)
1. ✅ Cargo 编译优化
2. ✅ 添加 Accelerate 框架
3. ✅ 优化 release profile

### P1 (短期 - 预期提升 30-50%)
1. 🔄 替换 gemm crate 为 Accelerate
2. 🔄 Metal shader 优化
3. 🔄 音频重采样优化

### P2 (中期 - 预期提升 20-40%)
1. ⏳ 实现批处理
2. ⏳ 流水线并行
3. ⏳ MPS 集成

### P3 (长期 - 预期提升 10-20%)
1. ⏳ 内存池和对象重用
2. ⏳ 更激进的层融合
3. ⏳ 量化支持 (INT8)

## 预期性能提升

| 优化项 | 预期加速比 | 实施难度 |
|--------|-----------|---------|
| 编译器优化 | 1.2-1.3x | 低 |
| Accelerate GEMM | 1.5-2.0x | 中 |
| Metal 优化 | 1.3-1.5x | 高 |
| 批处理 | 2.0-3.0x | 中 |
| 流水线 | 1.2-1.4x | 高 |
| **总计** | **3-5x** | - |

## 基准测试

建议添加基准测试套件：
```bash
cargo install cargo-criterion
```

创建 `benches/align_bench.rs`

## 性能分析工具

### macOS 特定工具
```bash
# Instruments - GPU
xcrun xctrace record --template 'Metal System Trace' --launch target/release/align

# CPU Profiling
cargo instruments --release --bench align_bench -t time

# Memory
cargo instruments --release --bench align_bench -t alloc
```

## 下一步行动

1. **立即**: 实施 P0 优化
2. **本周**: 添加基准测试
3. **下周**: 开始 P1 优化实施
4. **持续**: 性能监控和调优

## 参考资料

- [Apple Accelerate Framework](https://developer.apple.com/documentation/accelerate)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [wgpu Best Practices](https://github.com/gfx-rs/wgpu/wiki/Best-Practices)
- [Rust SIMD](https://rust-lang.github.io/packed_simd/perf-guide/)
