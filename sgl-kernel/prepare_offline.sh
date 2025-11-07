#!/bin/bash
# SGLang Kernel 离线编译准备脚本
# 用途：在有网络的环境中准备所有离线编译所需的依赖
# 使用：./prepare_offline.sh [cuda_version]
# 示例：./prepare_offline.sh 12.8

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查网络连接
check_network() {
    echo_info "检查网络连接..."
    if ! ping -c 1 github.com &> /dev/null; then
        echo_error "无法连接到 github.com，请检查网络连接"
        exit 1
    fi
    echo_success "网络连接正常"
}

# 获取 CUDA 版本
CUDA_VERSION=${1:-12.8}
ARCH=$(uname -m)

echo_info "====================================="
echo_info "SGLang Kernel 离线编译准备工具"
echo_info "====================================="
echo_info "CUDA 版本: ${CUDA_VERSION}"
echo_info "系统架构: ${ARCH}"
echo_info ""

# 检查网络
check_network

# 创建目录
echo_info "创建目录结构..."
mkdir -p third_party
mkdir -p offline_packages

# 第三方 Git 依赖列表（从 CMakeLists.txt 提取）
declare -A GIT_DEPS=(
    ["cutlass"]="https://github.com/NVIDIA/cutlass|57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
    ["DeepGEMM"]="https://github.com/sgl-project/DeepGEMM|f4adba8a6695e635b0106ce3dae3202016ad0ee5"
    ["fmt"]="https://github.com/fmtlib/fmt|553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
    ["triton"]="https://github.com/triton-lang/triton|8f9f695ea8fde23a0c7c88e4ab256634ca27789f"
    ["flashinfer"]="https://github.com/flashinfer-ai/flashinfer|bc29697ba20b7e6bdb728ded98f04788e16ee021"
    ["sgl-attn"]="https://github.com/sgl-project/sgl-attn|f20a52329482ddca4a627b2f028f88c2959ee299"
    ["flash-attention"]="https://github.com/Dao-AILab/flash-attention|9dbed03d1a7a5862998c182c83d8265fea9dc21b"
    ["mscclpp"]="https://github.com/microsoft/mscclpp|51eca89d20f0cfb3764ccd764338d7b22cd486a6"
    ["fast-hadamard-transform"]="https://github.com/sgl-project/fast-hadamard-transform|48f3c13764dc2ec662ade842a4696a90a137f1bc"
    ["FlashMLA"]="https://github.com/sgl-project/FlashMLA|54b5f89ead3fa1ef3bb0aa79f4cf3a85f8831b13"
)

# 克隆 Git 依赖
echo_info "====================================="
echo_info "步骤 1/4: 克隆 Git 依赖"
echo_info "====================================="

cd third_party

for name in "${!GIT_DEPS[@]}"; do
    IFS='|' read -r url commit <<< "${GIT_DEPS[$name]}"

    if [ -d "$name" ]; then
        echo_warning "目录 $name 已存在，跳过克隆"
        continue
    fi

    echo_info "克隆 $name..."
    git clone --quiet "$url" "$name" || {
        echo_error "克隆 $name 失败"
        exit 1
    }

    cd "$name"
    echo_info "切换到 commit $commit..."
    git checkout --quiet "$commit" || {
        echo_error "切换 commit 失败"
        exit 1
    }
    cd ..

    echo_success "$name 完成"
done

cd ..

# 下载 Python 依赖
echo_info ""
echo_info "====================================="
echo_info "步骤 2/4: 下载 Python 依赖"
echo_info "====================================="

# 根据 CUDA 版本选择 PyTorch 索引
case "$CUDA_VERSION" in
    "13.0")
        TORCH_VERSION="2.9.0"
        TORCH_INDEX="https://download.pytorch.org/whl/cu130"
        ;;
    "12.9")
        TORCH_VERSION="2.8.0"
        TORCH_INDEX="https://download.pytorch.org/whl/cu129"
        ;;
    "12.8")
        TORCH_VERSION="2.8.0"
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        ;;
    *)
        TORCH_VERSION="2.8.0"
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        ;;
esac

echo_info "下载 PyTorch ${TORCH_VERSION} (CUDA ${CUDA_VERSION})..."

pip download \
    torch==${TORCH_VERSION} \
    scikit-build-core \
    ninja \
    setuptools==75.0.0 \
    wheel==0.41.0 \
    numpy \
    uv \
    --index-url ${TORCH_INDEX} \
    --dest offline_packages \
    --no-deps 2>&1 | grep -v "Requirement already satisfied" || true

echo_success "Python 依赖下载完成"

# 下载构建工具
echo_info ""
echo_info "====================================="
echo_info "步骤 3/4: 下载构建工具"
echo_info "====================================="

cd offline_packages

# 下载 CMake
CMAKE_VERSION="3.31.1"
CMAKE_TARBALL="cmake-${CMAKE_VERSION}-linux-${ARCH}.tar.gz"

if [ -f "$CMAKE_TARBALL" ]; then
    echo_warning "CMake 已下载，跳过"
else
    echo_info "下载 CMake ${CMAKE_VERSION}..."
    wget -q --show-progress \
        "https://cmake.org/files/v3.31/${CMAKE_TARBALL}" || {
        echo_error "下载 CMake 失败"
        exit 1
    }
    echo_success "CMake 下载完成"
fi

# 下载 ccache
CCACHE_VERSION="4.12.1"
CCACHE_TARBALL="ccache-${CCACHE_VERSION}.tar.xz"

if [ -f "$CCACHE_TARBALL" ]; then
    echo_warning "ccache 已下载，跳过"
else
    echo_info "下载 ccache ${CCACHE_VERSION}..."
    wget -q --show-progress \
        "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/${CCACHE_TARBALL}" || {
        echo_error "下载 ccache 失败"
        exit 1
    }
    echo_success "ccache 下载完成"
fi

cd ..

# 生成离线构建配置
echo_info ""
echo_info "====================================="
echo_info "步骤 4/4: 生成配置文件"
echo_info "====================================="

cat > offline_build.cmake << 'EOF'
# SGLang Kernel 离线编译 CMake 配置
# 此文件覆盖 FetchContent 使用本地第三方依赖

message(STATUS "使用离线构建模式")

# 获取当前目录
get_filename_component(SGL_KERNEL_ROOT "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

# 设置 FetchContent 源目录为本地路径
set(FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS
    "${SGL_KERNEL_ROOT}/third_party/cutlass" CACHE PATH "cutlass 源目录")
set(FETCHCONTENT_SOURCE_DIR_REPO-DEEPGEMM
    "${SGL_KERNEL_ROOT}/third_party/DeepGEMM" CACHE PATH "DeepGEMM 源目录")
set(FETCHCONTENT_SOURCE_DIR_REPO-FMT
    "${SGL_KERNEL_ROOT}/third_party/fmt" CACHE PATH "fmt 源目录")
set(FETCHCONTENT_SOURCE_DIR_REPO-TRITON
    "${SGL_KERNEL_ROOT}/third_party/triton" CACHE PATH "triton 源目录")
set(FETCHCONTENT_SOURCE_DIR_REPO-FLASHINFER
    "${SGL_KERNEL_ROOT}/third_party/flashinfer" CACHE PATH "flashinfer 源目录")
set(FETCHCONTENT_SOURCE_DIR_REPO-FLASH-ATTENTION
    "${SGL_KERNEL_ROOT}/third_party/sgl-attn" CACHE PATH "sgl-attn 源目录")
set(FETCHCONTENT_SOURCE_DIR_REPO-FLASH-ATTENTION-ORIGIN
    "${SGL_KERNEL_ROOT}/third_party/flash-attention" CACHE PATH "flash-attention 源目录")
set(FETCHCONTENT_SOURCE_DIR_REPO-MSCCLPP
    "${SGL_KERNEL_ROOT}/third_party/mscclpp" CACHE PATH "mscclpp 源目录")
set(FETCHCONTENT_SOURCE_DIR_REPO-FAST-HADAMARD-TRANSFORM
    "${SGL_KERNEL_ROOT}/third_party/fast-hadamard-transform" CACHE PATH "fast-hadamard 源目录")

# 验证所有目录存在
foreach(dep IN ITEMS
    REPO-CUTLASS REPO-DEEPGEMM REPO-FMT REPO-TRITON REPO-FLASHINFER
    REPO-FLASH-ATTENTION REPO-FLASH-ATTENTION-ORIGIN REPO-MSCCLPP
    REPO-FAST-HADAMARD-TRANSFORM)
    if(NOT EXISTS "${FETCHCONTENT_SOURCE_DIR_${dep}}")
        message(WARNING "依赖目录不存在: ${FETCHCONTENT_SOURCE_DIR_${dep}}")
    else()
        message(STATUS "找到依赖: ${dep} -> ${FETCHCONTENT_SOURCE_DIR_${dep}}")
    endif()
endforeach()

# 禁用 Git 更新检查
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "禁用 FetchContent 更新检查" FORCE)

message(STATUS "离线构建配置加载完成")
EOF

echo_success "配置文件生成完成: offline_build.cmake"

# 生成离线编译脚本
cat > build_offline.sh << 'EOF'
#!/bin/bash
# SGLang Kernel 离线编译脚本
# 在离线环境中执行此脚本进行编译

set -e

echo "====================================="
echo "SGLang Kernel 离线编译"
echo "====================================="

# 安装 CMake（如果需要）
if ! command -v cmake &> /dev/null || [ "$(cmake --version | grep -oP '\d+\.\d+' | head -1)" \< "3.31" ]; then
    echo "[INFO] 安装 CMake 3.31.1..."
    cd offline_packages
    tar -xzf cmake-3.31.1-linux-$(uname -m).tar.gz
    sudo mv cmake-3.31.1-linux-$(uname -m) /opt/cmake 2>/dev/null || \
        mv cmake-3.31.1-linux-$(uname -m) $HOME/cmake
    export PATH=/opt/cmake/bin:$HOME/cmake/bin:$PATH
    cd ..
fi

cmake --version

# 安装 ccache（如果需要）
if ! command -v ccache &> /dev/null; then
    echo "[INFO] 安装 ccache 4.12.1..."
    cd offline_packages
    tar -xf ccache-4.12.1.tar.xz
    cd ccache-4.12.1
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
    make -j$(nproc)
    make install
    export PATH=$HOME/.local/bin:$PATH
    cd ../../..
fi

ccache --version

# 安装 Python 依赖
echo "[INFO] 安装 Python 依赖..."
pip install --no-index --find-links=offline_packages \
    torch scikit-build-core ninja setuptools wheel numpy uv || \
    echo "[WARNING] 某些包可能已安装"

# 配置环境
export CCACHE_DIR=$HOME/.cache/sgl-kernel/ccache
export CCACHE_BASEDIR=$(pwd)
export CCACHE_MAXSIZE=10G
export CCACHE_COMPILERCHECK=content
export CCACHE_COMPRESS=true
export CCACHE_SLOPPINESS=file_macro,time_macros,include_file_mtime,include_file_ctime

export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# 设置 CUDA 架构（根据实际 GPU 修改）
export FLASHINFER_CUDA_ARCH_LIST='8.0 8.9 9.0'

mkdir -p $CCACHE_DIR

echo "[INFO] 开始编译..."
echo "[INFO] 并行度: $(nproc) 核心"

# 创建构建目录
mkdir -p build && cd build

# 配置
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PROJECT_INCLUDE_BEFORE=../offline_build.cmake \
    -DENABLE_CCACHE=ON \
    -DENABLE_BELOW_SM90=ON \
    -DSGL_KERNEL_ENABLE_BF16=ON \
    -DSGL_KERNEL_ENABLE_FP8=ON

# 编译
make -j$(nproc)

# 安装
make install

cd ..

echo "====================================="
echo "编译完成！"
echo "====================================="

# 验证
python -c "import sgl_kernel; print(f'sgl_kernel version: {sgl_kernel.__version__}')"

ccache -s
EOF

chmod +x build_offline.sh

echo_success "离线编译脚本生成完成: build_offline.sh"

# 统计信息
echo ""
echo_info "====================================="
echo_info "准备完成！统计信息："
echo_info "====================================="

THIRD_PARTY_SIZE=$(du -sh third_party 2>/dev/null | cut -f1)
OFFLINE_PACKAGES_SIZE=$(du -sh offline_packages 2>/dev/null | cut -f1)
THIRD_PARTY_COUNT=$(ls -1 third_party 2>/dev/null | wc -l)
OFFLINE_PACKAGES_COUNT=$(ls -1 offline_packages 2>/dev/null | wc -l)

echo_info "第三方依赖: ${THIRD_PARTY_COUNT} 个，大小: ${THIRD_PARTY_SIZE}"
echo_info "离线包: ${OFFLINE_PACKAGES_COUNT} 个，大小: ${OFFLINE_PACKAGES_SIZE}"
echo ""
echo_success "所有依赖准备完成！"
echo ""
echo_info "下一步操作："
echo_info "1. 打包整个 sgl-kernel 目录："
echo_info "   tar -czf sgl-kernel-offline.tar.gz ."
echo_info ""
echo_info "2. 传输到离线机器后，解压并执行："
echo_info "   tar -xzf sgl-kernel-offline.tar.gz"
echo_info "   cd sgl-kernel"
echo_info "   ./build_offline.sh"
echo_info ""
echo_info "详细说明请查看: OFFLINE_BUILD_GUIDE.md"
echo_info "====================================="
