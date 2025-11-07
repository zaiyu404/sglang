# 离线编译快速开始 🚀

一页纸快速参考指南，用于 sgl-kernel 离线编译。

---

## 📋 前提条件

- ✅ CUDA 工具链（nvcc, CUDA 库）
- ✅ Python ≥ 3.10
- ✅ GCC/G++ 编译器
- ✅ 64GB+ RAM（推荐）
- ✅ 50GB+ 磁盘空间

---

## 🔄 工作流程

```
[在线机器]              [离线机器]
    |                        |
 1. 准备依赖                |
    |                        |
 2. 打包传输 ──────────────> |
                             |
                          3. 解压
                             |
                          4. 编译
                             |
                          5. 安装
```

---

## 💻 在线环境：准备依赖

### 方式 1：自动化脚本（推荐）

```bash
cd sgl-kernel

# 准备所有依赖（自动）
./prepare_offline.sh 12.8

# 打包
tar -czf sgl-kernel-offline.tar.gz \
    third_party/ \
    offline_packages/ \
    offline_build.cmake \
    build_offline.sh \
    OFFLINE_BUILD_GUIDE.md \
    CMakeLists.txt \
    pyproject.toml \
    csrc/ \
    python/ \
    include/ \
    Makefile
```

### 方式 2：手动准备

```bash
# 1. 克隆 Git 依赖（10 个仓库）
mkdir -p third_party && cd third_party

git clone https://github.com/NVIDIA/cutlass && \
cd cutlass && git checkout 57e3cfb47a2d && cd ..

git clone https://github.com/sgl-project/DeepGEMM && \
cd DeepGEMM && git checkout f4adba8a6695 && cd ..

# ... 其余 8 个仓库（见 OFFLINE_BUILD_GUIDE.md）

cd ..

# 2. 下载 Python 包
mkdir -p offline_packages
pip download torch==2.8.0 scikit-build-core ninja \
    --index-url https://download.pytorch.org/whl/cu128 \
    --dest offline_packages

# 3. 下载工具
cd offline_packages
wget https://cmake.org/files/v3.31/cmake-3.31.1-linux-x86_64.tar.gz
wget https://github.com/ccache/ccache/releases/download/v4.12.1/ccache-4.12.1.tar.xz
```

---

## 📦 传输到离线机器

```bash
# 方式 1: 网络传输
rsync -av --progress sgl-kernel-offline.tar.gz user@offline-host:/path/

# 方式 2: USB/移动硬盘
cp sgl-kernel-offline.tar.gz /media/usb/

# 方式 3: SCP
scp sgl-kernel-offline.tar.gz user@offline-host:/path/
```

---

## 🔧 离线环境：编译安装

### 快速编译（使用脚本）

```bash
# 1. 解压
tar -xzf sgl-kernel-offline.tar.gz
cd sgl-kernel

# 2. 执行离线编译脚本
./build_offline.sh

# 3. 验证
python -c "import sgl_kernel; print(sgl_kernel.__version__)"
```

### 手动编译

```bash
# 1. 安装工具
cd offline_packages
tar -xzf cmake-3.31.1-linux-x86_64.tar.gz
sudo mv cmake-3.31.1-linux-x86_64 /opt/cmake

tar -xf ccache-4.12.1.tar.xz
cd ccache-4.12.1 && mkdir build && cd build
/opt/cmake/bin/cmake .. && make -j && sudo make install
cd ../../..

# 2. 安装 Python 依赖
pip install --no-index --find-links=offline_packages \
    torch scikit-build-core ninja numpy

# 3. 配置环境
export PATH=/opt/cmake/bin:$PATH
export CCACHE_DIR=$HOME/.cache/sgl-kernel/ccache
export FLASHINFER_CUDA_ARCH_LIST='8.0 8.9 9.0'  # 根据 GPU 调整

# 4. 编译
mkdir build && cd build
cmake .. -DCMAKE_PROJECT_INCLUDE_BEFORE=../offline_build.cmake
make -j$(nproc)
sudo make install
```

---

## 🎯 GPU 架构对照表

| GPU 型号 | 计算能力 | FLASHINFER_CUDA_ARCH_LIST |
|---------|---------|---------------------------|
| A100 | SM 80 | `'8.0'` |
| L40S | SM 89 | `'8.9'` |
| H100 | SM 90 | `'9.0'` |
| H100 (TMA) | SM 90a | `'9.0a'` |
| B100/B200 | SM 100a | `'10.0a'` |
| 多 GPU | - | `'8.0 8.9 9.0'` |

---

## ⚙️ 常用编译选项

```bash
# 仅编译特定架构（节省时间和空间）
export FLASHINFER_CUDA_ARCH_LIST='9.0'

# 减少并行度（节省内存）
export CMAKE_BUILD_PARALLEL_LEVEL=8
make -j8

# 启用额外特性
cmake .. \
    -DSGL_KERNEL_ENABLE_BF16=ON \
    -DSGL_KERNEL_ENABLE_FP8=ON \
    -DSGL_KERNEL_ENABLE_FP4=ON \
    -DENABLE_BELOW_SM90=OFF  # 仅编译 SM90+
```

---

## 🐛 常见问题

### 问题：内存不足（OOM）

```bash
# 减少并行数
export CMAKE_BUILD_PARALLEL_LEVEL=4
make -j4
```

### 问题：找不到 CUDA

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 问题：CMake 版本过低

```bash
# 使用离线包中的 CMake
export PATH=/opt/cmake/bin:$PATH
cmake --version  # 应显示 3.31.x
```

### 问题：第三方依赖未找到

```bash
# 验证目录结构
ls third_party/
# 应包含：cutlass, DeepGEMM, fmt, triton, flashinfer, ...

# 确保使用离线配置
cmake .. -DCMAKE_PROJECT_INCLUDE_BEFORE=$(pwd)/offline_build.cmake
```

---

## 📊 编译时间预估

| 配置 | 首次编译 | ccache 后 |
|------|---------|----------|
| 单架构（如 SM90） | 30-60 分钟 | 5-10 分钟 |
| 3 架构（8.0/8.9/9.0） | 1-2 小时 | 15-30 分钟 |
| 全架构（8.0/8.9/9.0a/10.0a/12.0a） | 2-3 小时 | 30-45 分钟 |

*基于 64 核 CPU + 128GB RAM 测试*

---

## 📝 文件清单

### 必需文件（编译）
- ✅ `third_party/` - Git 依赖（~5GB）
- ✅ `offline_packages/` - Python 包 + 工具（~3GB）
- ✅ `offline_build.cmake` - CMake 离线配置
- ✅ `CMakeLists.txt`, `pyproject.toml` - 构建配置
- ✅ `csrc/`, `python/`, `include/` - 源代码

### 可选文件（便利）
- 📄 `build_offline.sh` - 自动化编译脚本
- 📄 `OFFLINE_BUILD_GUIDE.md` - 详细指南
- 📄 `OFFLINE_QUICK_START.md` - 本文档

---

## ✅ 验证安装

```bash
# 测试导入
python3 << EOF
import sgl_kernel
print(f"版本: {sgl_kernel.__version__}")

from sgl_kernel import load_ops
print("加载成功！")
EOF

# 查看编译产物
python -c "import sgl_kernel; import os; print(os.path.dirname(sgl_kernel.__file__))"

# 检查 SO 文件
ls -lh $(python -c "import sgl_kernel; import os; print(os.path.dirname(sgl_kernel.__file__))")/*.so
```

---

## 📚 相关文档

- 📖 详细指南：[OFFLINE_BUILD_GUIDE.md](OFFLINE_BUILD_GUIDE.md)
- 🏠 项目主页：[sgl-project/sglang](https://github.com/sgl-project/sglang)
- 📦 PyPI：[sgl-kernel](https://pypi.org/project/sgl-kernel)

---

## 🆘 获取帮助

1. 查看详细文档：`OFFLINE_BUILD_GUIDE.md`
2. 提交 Issue：https://github.com/sgl-project/sglang/issues
3. 社区讨论：https://github.com/sgl-project/sglang/discussions

---

**提示**：首次编译建议先在在线环境测试流程，确保无误后再转移到离线环境。

---

_生成时间: 2025-01-07 | SGLang Kernel 版本: 0.3.x_
