# SGLang Kernel 离线编译指南

本指南详细说明如何在离线环境中编译 sgl-kernel。

## 目录
- [背景说明](#背景说明)
- [准备工作（在线环境）](#准备工作在线环境)
- [离线编译步骤](#离线编译步骤)
- [故障排除](#故障排除)

---

## 背景说明

sgl-kernel 的编译过程依赖多个外部资源：

1. **Git 依赖** - CMake FetchContent 自动下载 10+ 个 Git 仓库
2. **Python 依赖** - PyTorch、scikit-build-core、ninja 等
3. **构建工具** - CMake 3.31、ccache 4.12.1
4. **CUDA 工具链** - NVCC、CUDA 库

离线编译需要预先准备所有这些依赖。

---

## 准备工作（在线环境）

### 1. 克隆主仓库及所有依赖

在**有网络连接**的机器上执行以下操作：

#### 1.1 克隆 sgl-kernel 仓库

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang/sgl-kernel
```

#### 1.2 手动克隆所有第三方依赖

根据 `CMakeLists.txt` (line 47-125)，需要克隆以下仓库：

```bash
# 创建第三方依赖目录
mkdir -p third_party
cd third_party

# 1. cutlass
git clone https://github.com/NVIDIA/cutlass
cd cutlass
git checkout 57e3cfb47a2d9e0d46eb6335c3dc411498efa198
cd ..

# 2. DeepGEMM
git clone https://github.com/sgl-project/DeepGEMM
cd DeepGEMM
git checkout f4adba8a6695e635b0106ce3dae3202016ad0ee5
cd ..

# 3. fmt
git clone https://github.com/fmtlib/fmt
cd fmt
git checkout 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28
cd ..

# 4. triton
git clone https://github.com/triton-lang/triton
cd triton
git checkout 8f9f695ea8fde23a0c7c88e4ab256634ca27789f
cd ..

# 5. flashinfer
git clone https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer
git checkout bc29697ba20b7e6bdb728ded98f04788e16ee021
cd ..

# 6. sgl-attn
git clone https://github.com/sgl-project/sgl-attn
cd sgl-attn
git checkout f20a52329482ddca4a627b2f028f88c2959ee299
cd ..

# 7. flash-attention (origin)
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout 9dbed03d1a7a5862998c182c83d8265fea9dc21b
cd ..

# 8. mscclpp
git clone https://github.com/microsoft/mscclpp.git
cd mscclpp
git checkout 51eca89d20f0cfb3764ccd764338d7b22cd486a6
cd ..

# 9. fast-hadamard-transform
git clone https://github.com/sgl-project/fast-hadamard-transform.git
cd fast-hadamard-transform
git checkout 48f3c13764dc2ec662ade842a4696a90a137f1bc
cd ..

# 10. FlashMLA
git clone https://github.com/sgl-project/FlashMLA.git
cd FlashMLA
git checkout 54b5f89ead3fa1ef3bb0aa79f4cf3a85f8831b13
cd ..
```

#### 1.3 下载 Python 依赖包（离线 wheel）

```bash
cd /path/to/sgl-kernel

# 创建离线包目录
mkdir -p offline_packages

# 下载所有 Python 依赖（根据你的 CUDA 版本选择）
# 以 CUDA 12.8 为例：
pip download \
    torch==2.8.0 \
    scikit-build-core \
    ninja \
    setuptools==75.0.0 \
    wheel==0.41.0 \
    numpy \
    uv \
    --index-url https://download.pytorch.org/whl/cu128 \
    --dest offline_packages

# 下载 ccache 源码
cd offline_packages
wget https://github.com/ccache/ccache/releases/download/v4.12.1/ccache-4.12.1.tar.xz

# 下载 CMake
# 根据你的架构选择 (x86_64 或 aarch64)
ARCH=$(uname -m)
wget https://cmake.org/files/v3.31/cmake-3.31.1-linux-${ARCH}.tar.gz
```

#### 1.4 打包所有内容

```bash
cd /path/to/sglang

# 打包整个 sgl-kernel 目录（包含代码和依赖）
tar -czf sgl-kernel-offline.tar.gz sgl-kernel/

# 或者使用 rsync 传输到离线机器
# rsync -av sgl-kernel/ user@offline-machine:/path/to/sgl-kernel/
```

---

## 离线编译步骤

将打包的文件传输到离线机器后：

### 2. 在离线环境中编译

#### 2.1 解压并准备环境

```bash
# 解压
tar -xzf sgl-kernel-offline.tar.gz
cd sgl-kernel

# 创建缓存目录
mkdir -p ~/.cache/sgl-kernel/cmake-downloads
mkdir -p ~/.cache/sgl-kernel/ccache
```

#### 2.2 安装 CMake（离线）

```bash
cd offline_packages
ARCH=$(uname -m)
tar -xzf cmake-3.31.1-linux-${ARCH}.tar.gz
sudo mv cmake-3.31.1-linux-${ARCH} /opt/cmake

# 添加到 PATH
export PATH=/opt/cmake/bin:$PATH
cmake --version  # 验证安装
```

#### 2.3 安装 ccache（离线）

```bash
cd offline_packages
tar -xf ccache-4.12.1.tar.xz
cd ccache-4.12.1

mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install

ccache --version  # 验证安装
cd ../../..
```

#### 2.4 安装 Python 依赖（离线）

```bash
# 从离线包安装
pip install --no-index --find-links=offline_packages \
    torch \
    scikit-build-core \
    ninja \
    setuptools \
    wheel \
    numpy \
    uv
```

#### 2.5 修改 CMakeLists.txt 使用本地依赖

创建一个配置文件来覆盖 FetchContent：

```bash
cat > offline_build.cmake << 'EOF'
# 覆盖 FetchContent 使用本地目录
set(FETCHCONTENT_SOURCE_DIR_REPO-CUTLASS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")
set(FETCHCONTENT_SOURCE_DIR_REPO-DEEPGEMM "${CMAKE_CURRENT_SOURCE_DIR}/third_party/DeepGEMM")
set(FETCHCONTENT_SOURCE_DIR_REPO-FMT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmt")
set(FETCHCONTENT_SOURCE_DIR_REPO-TRITON "${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton")
set(FETCHCONTENT_SOURCE_DIR_REPO-FLASHINFER "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer")
set(FETCHCONTENT_SOURCE_DIR_REPO-FLASH-ATTENTION "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sgl-attn")
set(FETCHCONTENT_SOURCE_DIR_REPO-FLASH-ATTENTION-ORIGIN "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flash-attention")
set(FETCHCONTENT_SOURCE_DIR_REPO-MSCCLPP "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mscclpp")
set(FETCHCONTENT_SOURCE_DIR_REPO-FAST-HADAMARD-TRANSFORM "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fast-hadamard-transform")

# 禁用 ccache 的网络检查
set(ENV{CCACHE_NOHASHDIR} "1")
EOF
```

#### 2.6 配置编译环境

```bash
# 设置 ccache
export CCACHE_DIR=~/.cache/sgl-kernel/ccache
export CCACHE_BASEDIR=$(pwd)
export CCACHE_MAXSIZE=10G
export CCACHE_COMPILERCHECK=content
export CCACHE_COMPRESS=true
export CCACHE_SLOPPINESS=file_macro,time_macros,include_file_mtime,include_file_ctime

# 设置 CMake
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# 设置架构支持（根据你的 GPU）
# 例如：A100 (SM80), L40S (SM89), H100 (SM90)
export FLASHINFER_CUDA_ARCH_LIST='8.0 8.9 9.0'

# CUDA 路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 2.7 开始编译

**方法 1：使用 CMake 手动编译（推荐离线环境）**

```bash
mkdir -p build && cd build

# 配置
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PROJECT_INCLUDE_BEFORE=../offline_build.cmake \
    -DENABLE_CCACHE=ON \
    -DENABLE_BELOW_SM90=ON \
    -DSGL_KERNEL_ENABLE_BF16=ON \
    -DSGL_KERNEL_ENABLE_FP8=ON

# 编译（调整并行度避免内存不足）
make -j$(( $(nproc)/2 ))

# 安装到 Python site-packages
make install
cd ..
```

**方法 2：使用 pip（需要修改 CMakeLists.txt）**

```bash
# 确保使用 --no-build-isolation 以使用本地依赖
CMAKE_ARGS="-DCMAKE_PROJECT_INCLUDE_BEFORE=$(pwd)/offline_build.cmake" \
    pip install -e . --no-build-isolation -v
```

#### 2.8 验证安装

```bash
python -c "import sgl_kernel; print(sgl_kernel.__version__)"
python -c "from sgl_kernel import load_ops; print('Success!')"
```

---

## 故障排除

### 问题 1：CMake 仍然尝试访问网络

**解决方案**：确保 `offline_build.cmake` 中的路径正确，且所有第三方库都已克隆到 `third_party/` 目录。

验证：
```bash
ls third_party/
# 应该看到：cutlass, DeepGEMM, fmt, triton, flashinfer, 等
```

### 问题 2：内存不足（OOM）

**解决方案**：减少并行编译数

```bash
# 对于 64GB 内存的机器
export CMAKE_BUILD_PARALLEL_LEVEL=8
make -j8

# 对于 ARM64 机器（内存受限）
export CMAKE_BUILD_PARALLEL_LEVEL=2
export MAKEFLAGS='-j2'
make -j2
```

### 问题 3：找不到 CUDA

**解决方案**：检查 CUDA 安装路径

```bash
# 查找 CUDA
whereis cuda

# 设置环境变量
export CUDA_HOME=/usr/local/cuda-12.8  # 替换为实际路径
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 验证
nvcc --version
```

### 问题 4：ccache 统计显示 0 命中

这是正常的，第一次编译时 ccache 会填充缓存。后续重新编译时会看到命中率提升。

```bash
# 查看 ccache 统计
ccache -s

# 清理 ccache（如果需要）
ccache -C
```

### 问题 5：特定 GPU 架构编译失败

**解决方案**：只编译你需要的架构

```bash
# 仅编译 SM90 (H100)
export FLASHINFER_CUDA_ARCH_LIST='9.0'

# 仅编译 SM80 (A100)
export FLASHINFER_CUDA_ARCH_LIST='8.0'

# 多架构
export FLASHINFER_CUDA_ARCH_LIST='8.0 8.9 9.0'
```

---

## 高级：Docker 离线编译

如果你使用 Docker 环境，可以创建一个包含所有依赖的离线镜像：

### 1. 在线环境构建镜像

```bash
# 使用 pytorch 官方镜像
docker pull pytorch/manylinux2_28-builder:cuda12.8

# 创建 Dockerfile.offline
cat > Dockerfile.offline << 'EOF'
FROM pytorch/manylinux2_28-builder:cuda12.8

# 复制所有依赖
COPY offline_packages /offline_packages
COPY third_party /sgl-kernel/third_party
COPY . /sgl-kernel

WORKDIR /sgl-kernel

# 预安装工具
RUN cd /offline_packages && \
    tar -xzf cmake-3.31.1-linux-x86_64.tar.gz && \
    mv cmake-3.31.1-linux-x86_64 /opt/cmake && \
    tar -xf ccache-4.12.1.tar.xz && \
    cd ccache-4.12.1 && \
    mkdir build && cd build && \
    /opt/cmake/bin/cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && make install

ENV PATH=/opt/cmake/bin:$PATH

ENTRYPOINT ["/bin/bash"]
EOF

# 构建镜像
docker build -f Dockerfile.offline -t sgl-kernel-offline:latest .

# 保存镜像
docker save sgl-kernel-offline:latest | gzip > sgl-kernel-offline-image.tar.gz
```

### 2. 离线环境加载镜像

```bash
# 加载镜像
gunzip -c sgl-kernel-offline-image.tar.gz | docker load

# 运行编译
docker run --rm --gpus all \
    -v $(pwd)/dist:/sgl-kernel/dist \
    sgl-kernel-offline:latest \
    bash -c "cd /sgl-kernel && pip install /offline_packages/*.whl && make build"
```

---

## 编译选项参考

### CMake 选项

| 选项 | 默认值 | 说明 |
|------|-------|------|
| `SGL_KERNEL_ENABLE_BF16` | ON | 启用 BF16 支持 |
| `SGL_KERNEL_ENABLE_FP8` | ON | 启用 FP8 支持 |
| `SGL_KERNEL_ENABLE_FP4` | OFF | 启用 FP4 支持（需要 SM100+） |
| `SGL_KERNEL_ENABLE_FA3` | OFF | 启用 FlashAttention 3 |
| `ENABLE_BELOW_SM90` | ON | 编译 SM90 以下架构 |
| `ENABLE_CCACHE` | ON | 使用 ccache 加速编译 |

### 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `FLASHINFER_CUDA_ARCH_LIST` | 指定编译的 GPU 架构 | `'8.0 9.0'` |
| `CMAKE_BUILD_PARALLEL_LEVEL` | CMake 并行编译数 | `16` |
| `CCACHE_DIR` | ccache 缓存目录 | `~/.cache/ccache` |
| `CUDA_HOME` | CUDA 安装路径 | `/usr/local/cuda-12.8` |

---

## 性能建议

1. **使用 ccache**：第二次编译可节省 80%+ 时间
2. **减少架构数量**：只编译你的 GPU 对应的架构
3. **充足内存**：推荐 64GB+ RAM，避免 OOM
4. **并行编译**：根据内存调整 `-j` 参数
5. **使用 SSD**：避免 I/O 成为瓶颈

---

## 参考文档

- [SGLang 官方文档](https://sgl-project.github.io)
- [CMake FetchContent 文档](https://cmake.org/cmake/help/latest/module/FetchContent.html)
- [ccache 文档](https://ccache.dev/manual/latest.html)

---

## 总结

离线编译 sgl-kernel 的关键步骤：

1. ✅ 在线环境克隆所有 Git 依赖到 `third_party/`
2. ✅ 下载所有 Python wheel 包到 `offline_packages/`
3. ✅ 下载 CMake 和 ccache 安装包
4. ✅ 打包并传输到离线机器
5. ✅ 使用 `offline_build.cmake` 覆盖 FetchContent 路径
6. ✅ 配置 ccache 和编译环境
7. ✅ 执行编译

**预计编译时间**：
- 首次编译（无缓存）：1-3 小时（取决于 CPU 和 GPU 架构数量）
- 使用 ccache 后重编译：10-30 分钟

如有问题，请查阅 [故障排除](#故障排除) 部分或提交 Issue。
