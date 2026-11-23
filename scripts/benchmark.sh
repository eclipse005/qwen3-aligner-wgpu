#!/bin/bash
# 性能基准测试脚本

set -e

echo "================================"
echo "Qwen3-Aligner 性能基准测试"
echo "================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查是否在正确的目录
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    exit 1
fi

# 检测系统信息
echo -e "${BLUE}系统信息:${NC}"
echo "----------------------------------------"

if [ "$(uname)" == "Darwin" ]; then
    echo "操作系统: macOS"
    
    # CPU 信息
    CPU_BRAND=$(sysctl -n machdep.cpu.brand_string)
    echo "CPU: $CPU_BRAND"
    
    # CPU 核心
    CPU_CORES=$(sysctl -n hw.ncpu)
    echo "CPU 核心数: $CPU_CORES"
    
    # 内存
    MEM_SIZE=$(sysctl -n hw.memsize)
    MEM_GB=$(echo "scale=1; $MEM_SIZE / 1073741824" | bc)
    echo "内存: ${MEM_GB} GB"
    
    # 检测是否是 Apple Silicon
    ARCH=$(uname -m)
    if [ "$ARCH" == "arm64" ]; then
        echo -e "${GREEN}✓ 检测到 Apple Silicon${NC}"
        APPLE_SILICON=true
    else
        echo -e "${YELLOW}! 非 Apple Silicon (x86_64)${NC}"
        APPLE_SILICON=false
    fi
else
    echo "操作系统: $(uname)"
    echo -e "${YELLOW}! 本脚本针对 macOS Apple Silicon 优化${NC}"
    APPLE_SILICON=false
fi

echo ""

# 编译优化版本
echo -e "${BLUE}编译优化版本...${NC}"
echo "----------------------------------------"

START_TIME=$(date +%s)

if $APPLE_SILICON; then
    echo "使用 Apple Silicon 优化配置..."
    cargo build --release
else
    echo "使用通用配置..."
    cargo build --release
fi

END_TIME=$(date +%s)
BUILD_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}✓ 编译完成 (耗时: ${BUILD_TIME}s)${NC}"
echo ""

# 检查二进制文件
BINARY="./target/release/align"
if [ ! -f "$BINARY" ]; then
    echo -e "${RED}错误: 编译失败，找不到可执行文件${NC}"
    exit 1
fi

BINARY_SIZE=$(du -h "$BINARY" | cut -f1)
echo "二进制文件大小: $BINARY_SIZE"
echo ""

# 检查模型
echo -e "${BLUE}检查模型...${NC}"
echo "----------------------------------------"

MODEL_DIR="${QALIGN_MODEL:-./models/Qwen3-ForcedAligner-0.6B}"

if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${YELLOW}! 模型目录不存在: $MODEL_DIR${NC}"
    echo "请运行以下命令下载模型:"
    echo ""
    echo "pip install modelscope"
    echo "python3 << EOF"
    echo "from modelscope import snapshot_download"
    echo "snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B-hf',"
    echo "                  local_dir='models/Qwen3-ForcedAligner-0.6B')"
    echo "EOF"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ 模型已就绪: $MODEL_DIR${NC}"
echo ""

# 生成测试音频
echo -e "${BLUE}生成测试音频...${NC}"
echo "----------------------------------------"

TEST_AUDIO="./test_audio.wav"

if [ ! -f "$TEST_AUDIO" ]; then
    echo "生成 10 秒测试音频..."
    
    # 使用 ffmpeg 生成测试音频（如果可用）
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -f lavfi -i "sine=frequency=440:duration=10" \
               -ar 16000 -ac 1 -y "$TEST_AUDIO" 2>/dev/null
        echo -e "${GREEN}✓ 测试音频已生成${NC}"
    else
        echo -e "${YELLOW}! ffmpeg 未安装，跳过音频生成${NC}"
        echo "请手动提供测试音频文件: $TEST_AUDIO"
        exit 1
    fi
fi

echo ""

# 运行基准测试
echo -e "${BLUE}运行性能基准测试...${NC}"
echo "========================================"
echo ""

# 测试 1: 基本对齐
echo "测试 1: 基本对齐 (10秒音频)"
echo "----------------------------------------"

TEST_TEXT="This is a test audio file for performance benchmarking"

START=$(date +%s%3N)
$BINARY --audio "$TEST_AUDIO" --text "$TEST_TEXT" --language English > /dev/null 2>&1
END=$(date +%s%3N)
TIME1=$((END - START))

echo -e "处理时间: ${GREEN}${TIME1} ms${NC}"
REALTIME_FACTOR=$(echo "scale=1; 10000 / $TIME1" | bc)
echo -e "实时率: ${GREEN}${REALTIME_FACTOR}x${NC}"
echo ""

# 测试 2: 带性能监控
echo "测试 2: 详细性能分析"
echo "----------------------------------------"

export QWEN3_PERF_MONITOR=1
export QWEN3_ENC_PROFILE=1

START=$(date +%s%3N)
$BINARY --audio "$TEST_AUDIO" --text "$TEST_TEXT" --language English --output /tmp/result.json 2>&1 | tail -20
END=$(date +%s%3N)
TIME2=$((END - START))

echo ""
echo -e "总处理时间: ${GREEN}${TIME2} ms${NC}"
echo ""

# 测试 3: 不同设备对比
echo "测试 3: 设备性能对比"
echo "----------------------------------------"

# GPU (Metal)
echo -n "GPU (Metal): "
START=$(date +%s%3N)
$BINARY --audio "$TEST_AUDIO" --text "$TEST_TEXT" --language English --device gpu > /dev/null 2>&1 || true
END=$(date +%s%3N)
TIME_GPU=$((END - START))
echo -e "${GREEN}${TIME_GPU} ms${NC}"

# CPU (Accelerate)
echo -n "CPU (Accelerate): "
START=$(date +%s%3N)
$BINARY --audio "$TEST_AUDIO" --text "$TEST_TEXT" --language English --device cpu > /dev/null 2>&1 || true
END=$(date +%s%3N)
TIME_CPU=$((END - START))
echo -e "${GREEN}${TIME_CPU} ms${NC}"

echo ""

# 运行 Cargo 基准测试（如果有）
if [ -d "benches" ]; then
    echo "测试 4: Cargo 基准测试"
    echo "----------------------------------------"
    
    if cargo bench --bench mel_benchmark --no-run &> /dev/null; then
        echo "运行 Mel 提取基准测试..."
        cargo bench --bench mel_benchmark 2>&1 | grep -A 10 "mel_extraction"
        echo ""
    else
        echo -e "${YELLOW}跳过（需要安装 criterion）${NC}"
        echo ""
    fi
fi

# 总结
echo "========================================"
echo -e "${BLUE}基准测试总结${NC}"
echo "========================================"
echo ""
echo "编译时间: ${BUILD_TIME}s"
echo "二进制大小: $BINARY_SIZE"
echo ""
echo "性能指标 (10秒音频):"
echo "  - GPU 处理时间: ${TIME_GPU} ms"
echo "  - CPU 处理时间: ${TIME_CPU} ms"
echo "  - 实时率: ${REALTIME_FACTOR}x"
echo ""

# 计算加速比
if [ "$TIME_CPU" -gt 0 ] && [ "$TIME_GPU" -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $TIME_CPU / $TIME_GPU" | bc)
    echo -e "GPU vs CPU 加速比: ${GREEN}${SPEEDUP}x${NC}"
    echo ""
fi

# 优化建议
echo "优化建议:"
echo "----------------------------------------"

if $APPLE_SILICON; then
    echo -e "${GREEN}✓ Apple Silicon 优化已启用${NC}"
else
    echo -e "${YELLOW}! 在 Apple Silicon 上可获得更好性能${NC}"
fi

if [ "$TIME_GPU" -lt "$TIME_CPU" ]; then
    echo -e "${GREEN}✓ GPU 加速工作正常${NC}"
    echo "  建议: 使用 --device gpu 以获得最佳性能"
else
    echo -e "${YELLOW}! GPU 性能未达预期${NC}"
    echo "  建议: 检查 Metal 驱动和设备状态"
fi

echo ""
echo -e "${GREEN}测试完成！${NC}"
echo ""

# 清理
rm -f /tmp/result.json

exit 0
