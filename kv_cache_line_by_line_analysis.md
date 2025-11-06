# SGLang KV-Cache 逐行代码分析

> **深度剖析 SGLang 的 KV-Cache 实现机制**
>
> 本文档从底层到顶层，逐行解释 KV-cache 的数据结构、内存管理、分配策略和完整数据流。

---

## 目录

1. [架构总览](#一架构总览)
2. [第一层：ReqToTokenPool（请求→Token 映射）](#二第一层reqtotokenpool)
3. [第二层：KVCache 基类](#三第二层kvcache-基类)
4. [第三层：MHATokenToKVPool（物理存储）](#四第三层mhatokentokvpool)
5. [第四层：Allocator（分配器）](#五第四层allocator)
6. [第五层：RadixCache（前缀缓存）](#六第五层radixcache)
7. [完整数据流](#七完整数据流)
8. [性能优化技巧](#八性能优化技巧)

---

## 一、架构总览

### 1.1 三层架构设计

SGLang 的 KV-cache 采用**三层索引架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                         Request                              │
│                      (用户请求 ID)                            │
└────────────────────────┬────────────────────────────────────┘
                         │ 1. ReqToTokenPool
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Token Indices                             │
│           [0, 5, 12, 18, 23, ...]                            │
│           (token 在 KV-cache 中的位置索引)                    │
└────────────────────────┬────────────────────────────────────┘
                         │ 2. Allocator
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Physical KV-Cache                         │
│   K: [layer_num × [size, head_num, head_dim]]               │
│   V: [layer_num × [size, head_num, head_dim]]               │
└─────────────────────────────────────────────────────────────┘
```

**设计理由**：
- **解耦**：请求管理、索引管理、物理存储分离
- **灵活性**：不同请求可共享相同的 KV-cache（前缀共享）
- **效率**：索引操作比数据拷贝快得多

### 1.2 核心文件索引

| 层级 | 文件 | 核心类 | 作用 |
|------|------|-------|------|
| **1** | `memory_pool.py:79-127` | `ReqToTokenPool` | Request → Token 索引映射 |
| **2** | `memory_pool.py:453-555` | `KVCache` | 抽象基类，定义接口 |
| **3** | `memory_pool.py:557-843` | `MHATokenToKVPool` | 物理 KV-cache 存储 |
| **4** | `allocator.py:118-173` | `TokenToKVPoolAllocator` | Token 索引分配器 |
| **5** | `radix_cache.py` | `RadixCache` | 前缀共享（基数树） |

---

## 二、第一层：ReqToTokenPool

> **文件位置**：`python/sglang/srt/mem_cache/memory_pool.py:79-127`

### 2.1 类定义与初始化

```python
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""
```

**作用**：管理"请求 ID → Token 索引列表"的映射关系。

#### 逐行分析：`__init__`

```python
def __init__(
    self,
    size: int,                    # 最大可支持的请求数（如 512）
    max_context_len: int,         # 每个请求最大 token 数（如 4096）
    device: str,                  # 设备："cuda" 或 "cpu"
    enable_memory_saver: bool,    # 是否启用内存节省模式
):
```

**第 90-92 行**：创建内存节省适配器
```python
memory_saver_adapter = TorchMemorySaverAdapter.create(
    enable=enable_memory_saver
)
```
- 作用：在内存紧张时，延迟分配或使用更节省的方式

**第 94-100 行**：核心数据结构
```python
self.size = size                  # 保存最大请求数
self.max_context_len = max_context_len  # 保存最大上下文长度
self.device = device              # 保存设备信息

with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
    self.req_to_token = torch.zeros(
        (size, max_context_len), dtype=torch.int32, device=device
    )
```

**关键数据结构**：`self.req_to_token`
- **类型**：`torch.Tensor`，形状 `[size, max_context_len]`
- **含义**：`req_to_token[req_id, i]` 表示第 `req_id` 个请求的第 `i` 个 token 在物理 KV-cache 中的索引
- **示例**：
  ```python
  req_to_token[0] = [12, 15, 18, 0, 0, ...]  # 请求 0 的 token 索引
  req_to_token[1] = [3, 7, 11, 14, 0, ...]   # 请求 1 的 token 索引
  ```

**第 102 行**：空闲槽位管理
```python
self.free_slots = list(range(size))  # [0, 1, 2, ..., size-1]
```
- **作用**：记录哪些请求槽位是空闲的
- **初始状态**：所有槽位都可用

### 2.2 核心方法

#### 方法 1：`write` - 写入映射

```python
def write(self, indices, values):
    self.req_to_token[indices] = values
```

**逐行解释**：
- **`indices`**：请求 ID（可以是单个 int 或 tuple）
- **`values`**：token 索引列表（Tensor）
- **操作**：直接索引赋值，将 token 索引写入对应的请求槽位

**使用示例**：
```python
# 为请求 5 写入其前 3 个 token 的索引
pool.write((5, slice(0, 3)), torch.tensor([100, 101, 102]))
# 结果：req_to_token[5, 0:3] = [100, 101, 102]
```

#### 方法 2：`alloc` - 分配请求槽位

```python
def alloc(self, need_size: int) -> List[int]:
    if need_size > len(self.free_slots):  # 检查是否有足够空闲槽位
        return None

    select_index = self.free_slots[:need_size]  # 取前 need_size 个
    self.free_slots = self.free_slots[need_size:]  # 更新空闲列表

    return select_index
```

**逐行解释**：
- **第 111 行**：检查可用槽位数是否足够
- **第 114 行**：从 `free_slots` 头部取 `need_size` 个槽位
- **第 115 行**：移除已分配的槽位
- **返回值**：分配的槽位 ID 列表（如 `[0, 1, 2]`）

**示例**：
```python
# 初始：free_slots = [0, 1, 2, 3, 4, ...]
slots = pool.alloc(3)  # 分配 3 个槽位
# 结果：slots = [0, 1, 2]，free_slots = [3, 4, ...]
```

#### 方法 3：`free` - 释放请求槽位

```python
def free(self, free_index: Union[int, List[int]]):
    if isinstance(free_index, (int,)):
        self.free_slots.append(free_index)  # 单个槽位
    else:
        self.free_slots.extend(free_index)  # 多个槽位
```

**逐行解释**：
- **第 120 行**：判断是释放单个还是多个槽位
- **第 121/123 行**：将释放的槽位加回 `free_slots`

**示例**：
```python
pool.free([0, 1])  # 释放槽位 0 和 1
# 结果：free_slots = [3, 4, ..., 0, 1]（顺序可能不同）
```

### 2.3 关键设计点

**Q1：为什么不直接存储 K 和 V，而是存储索引？**
- **答**：索引只是 int32（4 字节），而 K/V 是 `[head_num × head_dim]` 的张量（数百字节）
- **好处**：
  - 索引拷贝非常快
  - 可以多个请求共享相同的 token 索引（前缀共享）
  - 索引重排不需要移动实际数据

**Q2：为什么 shape 是 `[size, max_context_len]` 而不是动态列表？**
- **答**：GPU 上固定大小的 Tensor 更高效，避免频繁的内存重分配
- **权衡**：会浪费一些内存（大部分请求用不满 max_context_len）

---

## 三、第二层：KVCache 基类

> **文件位置**：`python/sglang/srt/mem_cache/memory_pool.py:453-555`

### 3.1 抽象基类设计

```python
class KVCache(abc.ABC):
```

**作用**：定义所有 KV-cache 实现必须遵循的接口。

#### 逐行分析：`__init__`

```python
def __init__(
    self,
    size: int,                    # KV-cache 可存储的最大 token 数
    page_size: int,               # 页大小（PagedAttention 用，通常为 16）
    dtype: torch.dtype,           # 数据类型（FP16, FP8, INT8 等）
    layer_num: int,               # 模型层数
    device: str,                  # 设备
    enable_memory_saver: bool,
    start_layer: Optional[int] = None,  # 起始层（分层推理用）
    end_layer: Optional[int] = None,    # 结束层
):
```

**第 466-481 行**：基本属性初始化
```python
self.size = size                  # 最大 token 数
self.page_size = page_size        # 页大小
self.dtype = dtype                # 计算用数据类型
self.device = device              # 设备
```

**第 470-474 行**：`store_dtype` 的特殊处理
```python
if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
    # NOTE: Store as torch.uint8 because Tensor.index_put is not
    # implemented for torch.float8_e5m2
    self.store_dtype = torch.uint8
else:
    self.store_dtype = dtype
```

**关键设计**：
- **问题**：PyTorch 的 FP8 类型不支持 `index_put` 操作（索引赋值）
- **解决方案**：存储时用 `uint8`（相同内存布局），读取时 `.view(dtype)` 转回 FP8
- **示例**：
  ```python
  # 写入时
  cache_k_fp8 = cache_k.to(torch.float8_e4m3fn)
  k_buffer[loc] = cache_k_fp8.view(torch.uint8)  # 存储为 uint8

  # 读取时
  k_uint8 = k_buffer[layer_id]
  k_fp8 = k_uint8.view(torch.float8_e4m3fn)  # 转回 FP8
  ```

**第 475-477 行**：层范围管理
```python
self.layer_num = layer_num
self.start_layer = start_layer or 0
self.end_layer = end_layer or layer_num - 1
```
- **作用**：支持**分层推理**（Pipeline Parallelism）
- **示例**：32 层模型分成 2 个 GPU，每个 GPU 处理 16 层
  - GPU 0: `start_layer=0, end_layer=15`
  - GPU 1: `start_layer=16, end_layer=31`

### 3.2 抽象方法（子类必须实现）

#### 方法 1：`get_kv_buffer` - 读取 KV-cache

```python
@abc.abstractmethod
def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError()
```

**返回值**：
- `(k_buffer, v_buffer)`：该层的 K 和 V 缓存张量
- 形状：`[size, head_num, head_dim]`

#### 方法 2：`set_kv_buffer` - 写入 KV-cache

```python
@abc.abstractmethod
def set_kv_buffer(
    self,
    layer: RadixAttention,       # 当前层的 Attention 对象
    loc: torch.Tensor,            # 要写入的位置索引
    cache_k: torch.Tensor,        # 新的 K 值
    cache_v: torch.Tensor,        # 新的 V 值
) -> None:
    raise NotImplementedError()
```

**参数说明**：
- **`loc`**：形状 `[num_tokens]`，每个元素是该 token 在 KV-cache 中的索引
- **`cache_k/v`**：形状 `[num_tokens, head_num, head_dim]`

**示例**：
```python
# 写入 3 个新 token 的 KV 到位置 [100, 101, 102]
loc = torch.tensor([100, 101, 102])
cache_k = torch.randn(3, 32, 128)  # 3 tokens, 32 heads, 128 dim
kv_pool.set_kv_buffer(layer, loc, cache_k, cache_v)
```

### 3.3 辅助方法

#### `_finalize_allocation_log` - 内存统计

```python
def _finalize_allocation_log(self, num_tokens: int):
    kv_size_bytes = self.get_kv_size_bytes()  # 子类实现
    if isinstance(kv_size_bytes, tuple):
        k_size, v_size = kv_size_bytes
        k_size_GB = k_size / GB
        v_size_GB = v_size / GB
        logger.info(
            f"KV Cache is allocated. #tokens: {num_tokens}, "
            f"K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
        )
        self.mem_usage = k_size_GB + v_size_GB
```

**作用**：在初始化时打印内存占用信息。

**示例输出**：
```
KV Cache is allocated. #tokens: 131072, K size: 16.00 GB, V size: 16.00 GB
```

---

## 四、第三层：MHATokenToKVPool

> **文件位置**：`python/sglang/srt/mem_cache/memory_pool.py:557-843`

这是 **Multi-Head Attention** 的物理 KV-cache 实现。

### 4.1 初始化

```python
class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,                    # 最大 token 数（如 131072）
        page_size: int,               # 页大小（如 16）
        dtype: torch.dtype,           # 数据类型
        head_num: int,                # KV head 数量（如 32）
        head_dim: int,                # 每个 head 的维度（如 128）
        layer_num: int,               # 层数（如 32）
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,       # 是否启用替代流
        enable_kv_cache_copy: bool = False,   # 是否启用 KV 拷贝优化
    ):
```

**第 574-583 行**：调用父类初始化
```python
super().__init__(
    size,
    page_size,
    dtype,
    layer_num,
    device,
    enable_memory_saver,
    start_layer,
    end_layer,
)
self.head_num = head_num
self.head_dim = head_dim
```

**第 587 行**：创建物理缓冲区
```python
self._create_buffers()  # 详见 4.2 节
```

**第 589-592 行**：双流优化
```python
self.device_module = torch.get_device_module(self.device)
self.alt_stream = (
    self.device_module.Stream() if _is_cuda and enable_alt_stream else None
)
```

**设计思想**：
- **问题**：写入 K 和 V 是顺序操作，存在等待时间
- **解决方案**：使用两个 CUDA Stream 并行写入 K 和 V
- **效果**：在小 batch 下提升 5-10% 性能

### 4.2 核心方法：`_create_buffers`

> **这是最关键的部分！**

```python
def _create_buffers(self):
    with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.enable_custom_mem_pool
            else nullcontext()
        ):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = [
                torch.zeros(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
```

**逐行解释**：

**第 645 行**：使用内存节省适配器
```python
with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
```
- 标记这块内存用于 KV-cache，便于监控和优化

**第 646-650 行**：可选的自定义内存池
```python
with (
    torch.cuda.use_mem_pool(self.custom_mem_pool)
    if self.enable_custom_mem_pool
    else nullcontext()
):
```
- **作用**：用于分布式场景（如 Mooncake NVLink 分配器）
- **普通场景**：`nullcontext()` 不做任何事

**第 653-668 行**：分配 K 和 V 缓冲区

**关键设计点**：

**1. 为什么是 `list` 而不是一个大 Tensor？**
```python
self.k_buffer = [Tensor, Tensor, ...]  # 每层一个 Tensor
# 而不是
self.k_buffer = Tensor  # [layer_num, size, head_num, head_dim]
```

**原因**：
- **灵活性**：不同层可以有不同的 dtype（如前层 FP8，后层 FP16）
- **内存管理**：可以独立地 offload 某些层到 CPU
- **兼容性**：某些硬件（如 NPU）的内存分配器要求

**2. 为什么大小是 `size + page_size`？**
```python
(self.size + self.page_size, self.head_num, self.head_dim)
```

**原因**：
- **槽位 0** 是 padding 槽位，用于写入 padded token 的虚拟输出
- **好处**：避免分支判断（统一用索引 0 处理 padding）

**3. 形状的含义**
```python
[size + page_size, head_num, head_dim]
```

- **第 1 维**：`size + page_size` = token 索引
  - `k_buffer[i]` = 第 i 个 token 的 K
- **第 2 维**：`head_num` = KV head 数量
  - 支持 MQA/GQA（`head_num` < Q head 数量）
- **第 3 维**：`head_dim` = 每个 head 的维度

**示例**：
```python
# Llama-7B 配置
size = 131072         # 约 128K tokens
page_size = 16
head_num = 32         # 32 个 KV heads
head_dim = 128        # 每个 head 128 维
layer_num = 32        # 32 层

# 单层 K buffer 内存
single_k = (131072 + 16) * 32 * 128 * 2 bytes (FP16) = 1.05 GB
# 所有层 K + V
total_kv = 1.05 GB * 2 * 32 = 67.2 GB
```

**第 670-687 行**：存储数据指针和步长
```python
self.k_data_ptrs = torch.tensor(
    [x.data_ptr() for x in self.k_buffer],  # 每个 Tensor 的内存地址
    dtype=torch.uint64,
    device=self.device,
)
self.v_data_ptrs = torch.tensor(
    [x.data_ptr() for x in self.v_buffer],
    dtype=torch.uint64,
    device=self.device,
)
self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
self.data_strides = torch.tensor(
    [
        np.prod(x.shape[1:]) * x.dtype.itemsize  # 每个 token 的字节数
        for x in self.k_buffer + self.v_buffer
    ],
    device=self.device,
)
```

**作用**：
- **`data_ptrs`**：用于 Triton kernel 直接访问内存
- **`data_strides`**：用于计算 token 的内存偏移量

**示例**：
```python
# 访问 layer 5 的 token 100 的 K
base_ptr = k_data_ptrs[5]
stride = data_strides[5]  # = head_num * head_dim * itemsize
offset = 100 * stride
k_token_100 = base_ptr + offset
```

### 4.3 读取方法：`get_kv_buffer`

```python
def _get_key_buffer(self, layer_id: int):
    # for internal use of referencing
    if self.store_dtype != self.dtype:
        return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
    return self.k_buffer[layer_id - self.start_layer]

def get_key_buffer(self, layer_id: int):
    # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
    if self.layer_transfer_counter is not None:
        self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
    return self._get_key_buffer(layer_id)

def get_kv_buffer(self, layer_id: int):
    return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)
```

**逐行解释**：

**第 766-770 行**：`_get_key_buffer` 内部方法
```python
if self.store_dtype != self.dtype:
    return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
return self.k_buffer[layer_id - self.start_layer]
```

**关键**：
- **`layer_id - self.start_layer`**：转换全局 layer_id 到本地索引
  - 示例：GPU 1 处理 layer 16-31，`start_layer=16`
  - 访问 layer 20：`k_buffer[20 - 16]` = `k_buffer[4]`
- **`.view(self.dtype)`**：将 `uint8` 转回 FP8（如果使用 FP8）

**第 772-778 行**：`get_key_buffer` 公开方法
```python
if self.layer_transfer_counter is not None:
    self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
return self._get_key_buffer(layer_id)
```

**同步机制**：
- **问题**：分布式推理时，某层的 KV-cache 可能还在从其他节点传输
- **解决方案**：`layer_transfer_counter.wait_until()` 阻塞等待传输完成
- **场景**：PD-Disaggregation（Prefill-Decode 分离）

### 4.4 写入方法：`set_kv_buffer` ⭐⭐⭐

> **这是整个 KV-cache 系统最核心的方法！**

```python
def set_kv_buffer(
    self,
    layer: RadixAttention,          # 当前层的 Attention 对象
    loc: torch.Tensor,               # 写入位置索引 [num_tokens]
    cache_k: torch.Tensor,           # K 值 [num_tokens, head_num, head_dim]
    cache_v: torch.Tensor,           # V 值 [num_tokens, head_num, head_dim]
    k_scale: Optional[float] = None, # K 量化缩放因子
    v_scale: Optional[float] = None, # V 量化缩放因子
    layer_id_override: Optional[int] = None,  # 覆盖 layer_id（调试用）
):
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    # 第 1 步：确定 layer_id
    if layer_id_override is not None:
        layer_id = layer_id_override
    else:
        layer_id = layer.layer_id

    # 第 2 步：量化（如果需要）
    if cache_k.dtype != self.dtype:
        if k_scale is not None:
            cache_k.div_(k_scale)  # 原地除法：cache_k /= k_scale
        if v_scale is not None:
            cache_v.div_(v_scale)
        cache_k = cache_k.to(self.dtype)  # 转换数据类型
        cache_v = cache_v.to(self.dtype)

    # 第 3 步：view casting（FP8 → uint8）
    if self.store_dtype != self.dtype:
        cache_k = cache_k.view(self.store_dtype)
        cache_v = cache_v.view(self.store_dtype)

    # 第 4 步：写入缓冲区
    if get_is_capture_mode() and self.alt_stream is not None:
        # CUDA Graph 模式 + 双流优化
        current_stream = self.device_module.current_stream()
        self.alt_stream.wait_stream(current_stream)
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        with self.device_module.stream(self.alt_stream):
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v
        current_stream.wait_stream(self.alt_stream)
    else:
        # 常规模式
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
```

**逐行深度解析**：

#### 步骤 1：确定 layer_id（第 806-809 行）
```python
if layer_id_override is not None:
    layer_id = layer_id_override
else:
    layer_id = layer.layer_id
```

- **正常情况**：从 `layer.layer_id` 获取
- **特殊情况**：调试或测试时可覆盖

#### 步骤 2：量化（第 810-816 行）⭐⭐⭐

```python
if cache_k.dtype != self.dtype:
    if k_scale is not None:
        cache_k.div_(k_scale)  # 原地除法，节省内存
    if v_scale is not None:
        cache_v.div_(v_scale)
    cache_k = cache_k.to(self.dtype)
    cache_v = cache_v.to(self.dtype)
```

**量化流程**：

**假设**：`cache_k.dtype = torch.bfloat16`, `self.dtype = torch.float8_e4m3fn`

**1. 缩放**：
```python
# k_scale = 1.5（从 checkpoint 加载）
cache_k.div_(1.5)  # cache_k /= 1.5
# 效果：将数值映射到 FP8 的表示范围
```

**数学原理**：
- FP8 E4M3 范围：`[-448, 448]`
- FP16/BF16 范围：`[-65504, 65504]`
- 需要缩放因子将 FP16 的值映射到 FP8 范围
- `quantized_value = original_value / scale`

**2. 类型转换**：
```python
cache_k = cache_k.to(torch.float8_e4m3fn)
# PyTorch 自动进行最近邻舍入
```

**示例**：
```python
# 原始值 (BF16)
k_original = torch.tensor([1.5, 3.0, 450.0], dtype=torch.bfloat16)

# 量化
k_scale = 1.0
k_scaled = k_original / k_scale  # [1.5, 3.0, 450.0]
k_quantized = k_scaled.to(torch.float8_e4m3fn)
# 注意：450.0 超出 FP8 范围，会被 clamp 到 448.0

# 反量化（读取时）
k_dequantized = k_quantized.to(torch.bfloat16) * k_scale
# 结果：[1.5, 3.0, 448.0]（精度损失）
```

**为什么原地操作 `div_`？**
- **内存效率**：不创建新 Tensor
- **CUDA Graph 友好**：减少内存分配开销

#### 步骤 3：view casting（第 818-820 行）

```python
if self.store_dtype != self.dtype:
    cache_k = cache_k.view(self.store_dtype)
    cache_v = cache_v.view(self.store_dtype)
```

**作用**：将 FP8 转为 uint8 存储

**底层原理**：
```python
# FP8 和 uint8 内存布局相同（8 位）
fp8_tensor = torch.tensor([1.0], dtype=torch.float8_e4m3fn)
# 内存：0x3F (二进制：00111111)

uint8_tensor = fp8_tensor.view(torch.uint8)
# 内存：0x3F (完全相同！)

# 存储为 uint8
k_buffer[loc] = uint8_tensor  # ✅ 支持 index_put

# 读取时转回
fp8_tensor = k_buffer[loc].view(torch.float8_e4m3fn)  # ✅ 正确
```

#### 步骤 4：写入缓冲区（第 822-832 行）⭐⭐⭐

**场景 A：CUDA Graph + 双流优化**
```python
if get_is_capture_mode() and self.alt_stream is not None:
    current_stream = self.device_module.current_stream()  # 获取当前流
    self.alt_stream.wait_stream(current_stream)           # 替代流等待当前流
    self.k_buffer[layer_id - self.start_layer][loc] = cache_k  # 在当前流写 K
    with self.device_module.stream(self.alt_stream):           # 切换到替代流
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v  # 在替代流写 V
    current_stream.wait_stream(self.alt_stream)           # 当前流等待替代流完成
```

**时序图**：
```
时间 ─────────────────────────────────────────►

Stream 0 (current):
  [计算 K,V] ──► [写 K] ──────────────────────► [等待] ──► [继续]
                     │                             ▲
                     │                             │
Stream 1 (alt):      │                             │
                     └──► [等待] ──► [写 V] ────────┘

# 好处：K 和 V 的写入并行进行
```

**为什么需要同步？**
```python
self.alt_stream.wait_stream(current_stream)
# 确保 K,V 计算完成再写入

current_stream.wait_stream(self.alt_stream)
# 确保 V 写入完成再进入下一层
```

**场景 B：常规模式**
```python
else:
    self.k_buffer[layer_id - self.start_layer][loc] = cache_k
    self.v_buffer[layer_id - self.start_layer][loc] = cache_v
```

**核心操作**：高级索引赋值

**示例**：
```python
# loc = [5, 12, 18]
# cache_k.shape = [3, 32, 128]

k_buffer[layer_id][loc] = cache_k
# 等价于
k_buffer[layer_id][5] = cache_k[0]
k_buffer[layer_id][12] = cache_k[1]
k_buffer[layer_id][18] = cache_k[2]

# GPU 并行执行这些赋值操作
```

**性能优化点**：
1. **向量化**：PyTorch 自动并行处理多个索引
2. **连续内存**：如果 `loc` 是连续的，可能使用 `memcpy`
3. **异步执行**：在 GPU stream 上异步执行

---

## 五、第四层：Allocator

> **文件位置**：`python/sglang/srt/mem_cache/allocator.py:118-173`

### 5.1 Allocator 的作用

**问题**：如何管理 `[0, 1, 2, ..., size-1]` 这些 token 索引？

**设计**：用 `TokenToKVPoolAllocator` 管理空闲和已分配的索引。

### 5.2 核心数据结构

```python
class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(
        self,
        size: int,              # 总 token 数
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,       # 关联的 KV-cache
        need_sort: bool,        # 是否需要排序空闲索引
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
```

**关键数据结构**：

**1. `free_pages`**：可立即分配的空闲索引
```python
self.free_pages = [1, 2, 3, ..., size]  # 注意：从 1 开始，0 是 padding
```

**2. `release_pages`**：待释放的索引（需排序后合并）
```python
self.release_pages = []  # 初始为空
```

**为什么需要两个列表？**
- **性能优化**：`free()` 操作很频繁，直接 append 到 `release_pages`
- **延迟排序**：只在 `free_pages` 不足时才触发排序和合并
- **避免碎片**：排序后的索引更连续，提升 GPU 访问效率

### 5.3 分配：`alloc`

```python
def alloc(self, need_size: int):
    # 第 1 步：检查是否需要合并释放的索引
    if self.need_sort and need_size > len(self.free_pages):
        self.merge_and_sort_free()  # 将 release_pages 合并到 free_pages

    # 第 2 步：检查是否有足够空间
    if need_size > len(self.free_pages):
        return None

    # 第 3 步：分配
    select_index = self.free_pages[:need_size]
    self.free_pages = self.free_pages[need_size:]
    return select_index
```

**逐行解释**：

**第 146-147 行**：延迟合并
```python
if self.need_sort and need_size > len(self.free_pages):
    self.merge_and_sort_free()
```

**`merge_and_sort_free` 实现**：
```python
def merge_and_sort_free(self):
    if len(self.release_pages) > 0:
        self.free_pages = torch.cat((self.free_pages, self.release_pages))
        self.free_pages, _ = torch.sort(self.free_pages)  # 排序
        self.release_pages = torch.empty((0,), ...)       # 清空
```

**示例**：
```python
# 初始状态
free_pages = [1, 2, 3]
release_pages = [10, 5, 7]  # 乱序

# 合并后
free_pages = [1, 2, 3, 5, 7, 10]  # 排序
release_pages = []
```

**为什么需要排序？**
- **GPU 访问效率**：连续索引访问更快（cache 友好）
- **碎片整理**：避免索引碎片化

**第 152-154 行**：分配索引
```python
select_index = self.free_pages[:need_size]  # 切片，取前 need_size 个
self.free_pages = self.free_pages[need_size:]  # 更新剩余空闲索引
return select_index
```

**示例**：
```python
# free_pages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
indices = alloc(3)
# 结果：indices = [1, 2, 3]，free_pages = [4, 5, 6, 7, 8, 9, 10]
```

### 5.4 释放：`free`

```python
def free(self, free_index: torch.Tensor):
    if free_index.numel() == 0:  # 空索引，直接返回
        return

    if self.is_not_in_free_group:
        if self.need_sort:
            self.release_pages = torch.cat((self.release_pages, free_index))
        else:
            self.free_pages = torch.cat((self.free_pages, free_index))
    else:
        self.free_group.append(free_index)
```

**逐行解释**：

**第 161-164 行**：延迟合并策略
```python
if self.need_sort:
    self.release_pages = torch.cat((self.release_pages, free_index))
else:
    self.free_pages = torch.cat((self.free_pages, free_index))
```

**两种模式**：
1. **`need_sort=True`**：追加到 `release_pages`，延迟排序
2. **`need_sort=False`**：直接追加到 `free_pages`（不需要排序）

**第 165-166 行**：批量释放优化
```python
else:
    self.free_group.append(free_index)
```

**用法**：
```python
allocator.free_group_begin()  # 开始批量释放
allocator.free([1, 2, 3])
allocator.free([10, 11])
allocator.free([20])
allocator.free_group_end()  # 一次性合并所有释放的索引
# 好处：减少 torch.cat 的次数
```

---

## 六、第五层：RadixCache

> **文件位置**：`python/sglang/srt/mem_cache/radix_cache.py`

RadixCache 实现了**前缀共享**，让多个请求可以共享相同的 KV-cache。

### 6.1 核心思想

**问题**：多个请求可能有相同的前缀（如系统提示词）

**示例**：
```
Request 1: "You are a helpful assistant. What is 1+1?"
Request 2: "You are a helpful assistant. What is 2+2?"
Request 3: "You are a helpful assistant. Tell me a joke."

# 公共前缀："You are a helpful assistant."
```

**传统方法**：每个请求独立存储 KV-cache
- 浪费内存：重复存储相同的前缀
- 浪费计算：重复计算相同的前缀

**RadixCache 方法**：用基数树共享前缀
```
                "You are a helpful assistant."
                          │
           ┌──────────────┼──────────────┐
           │              │              │
     "What is 1+1?"  "What is 2+2?"  "Tell me a joke."
```

### 6.2 基数树节点结构（简化）

```python
class TreeNode:
    def __init__(self):
        self.children = {}          # 子节点：{token_id: TreeNode}
        self.parent = None          # 父节点
        self.key = []               # 当前节点的 token 序列
        self.value = None           # KV-cache 索引
        self.lock_ref = 0           # 引用计数（正在使用）
        self.last_access_time = 0   # 最后访问时间（LRU 用）
```

**关键字段**：
- **`key`**：该节点对应的 token 序列
- **`value`**：token 在 KV-cache 中的索引列表
- **`lock_ref`**：引用计数（>0 表示正在被某个请求使用）
- **`children`**：子节点字典

### 6.3 匹配流程

**场景**：新请求的 prompt 是 `[1, 2, 3, 4, 5]`

**步骤 1**：从根节点开始匹配
```python
node = root
matched_tokens = []
i = 0

while i < len(prompt):
    token = prompt[i]
    if token in node.children:
        node = node.children[token]  # 移动到子节点
        matched_tokens.extend(node.key)
        i += len(node.key)
    else:
        break  # 无法继续匹配
```

**步骤 2**：返回匹配结果
```python
# 假设匹配到了 [1, 2, 3]
cached_indices = node.value  # 如 [100, 101, 102]
new_tokens = prompt[3:]      # [4, 5]
```

**步骤 3**：分配新索引并插入树
```python
# 为新 token 分配索引
new_indices = allocator.alloc(len(new_tokens))  # [103, 104]

# 创建新节点
new_node = TreeNode()
new_node.key = new_tokens
new_node.value = new_indices
new_node.parent = node
node.children[new_tokens[0]] = new_node
```

**结果树结构**：
```
root
 └─ node([1,2,3], value=[100,101,102])
     └─ new_node([4,5], value=[103,104])
```

### 6.4 驱逐策略

当内存不足时，RadixCache 需要驱逐某些节点。

**支持的策略**：
1. **LRU**（Least Recently Used）：驱逐最久未访问的
2. **LFU**（Least Frequently Used）：驱逐访问频率最低的
3. **FIFO**：先进先出
4. **MRU**（Most Recently Used）：驱逐最近访问的（反常规）

**实现**（简化）：
```python
def evict(self, num_tokens: int):
    # 收集所有可驱逐的节点（lock_ref == 0）
    evictable_nodes = [node for node in all_nodes if node.lock_ref == 0]

    # 按策略排序
    if self.policy == "lru":
        evictable_nodes.sort(key=lambda n: n.last_access_time)
    elif self.policy == "lfu":
        evictable_nodes.sort(key=lambda n: n.access_count)

    # 驱逐直到释放足够空间
    freed = 0
    for node in evictable_nodes:
        if freed >= num_tokens:
            break
        freed += len(node.value)
        self._remove_node(node)
```

---

## 七、完整数据流

### 7.1 Prefill 阶段（首次处理 prompt）

**场景**：用户输入 prompt `"Hello world"`，假设 tokenize 后是 `[1, 2, 3]`

#### 步骤 1：请求到达 Scheduler

```python
# scheduler.py
req = Req(input_ids=[1, 2, 3])
batch = ScheduleBatch([req])
```

#### 步骤 2：分配请求槽位

```python
# ReqToTokenPool.alloc
req_pool_idx = req_to_token_pool.alloc(1)[0]  # 假设分配到槽位 5
req.req_pool_idx = req_pool_idx
```

#### 步骤 3：RadixCache 前缀匹配

```python
# radix_cache.py
matched_node, matched_len = radix_cache.match(req.input_ids)
# 假设没有匹配到任何前缀
matched_len = 0  # 需要计算所有 3 个 token
```

#### 步骤 4：分配 KV-cache 索引

```python
# allocator.py
new_indices = token_to_kv_allocator.alloc(3)
# 假设分配到：[100, 101, 102]
```

#### 步骤 5：写入 ReqToTokenPool

```python
# common.py: write_cache_indices
req_to_token_pool.write(
    (req_pool_idx, slice(0, 3)),  # 槽位 5，位置 0-3
    new_indices                     # [100, 101, 102]
)
# 结果：req_to_token[5, 0:3] = [100, 101, 102]
```

#### 步骤 6：模型前向传播

```python
# model_forward.py
for layer_id in range(num_layers):
    # Attention 计算
    q, k, v = layer.self_attn(hidden_states)  # [3, head_num, head_dim]

    # 写入 KV-cache
    layer.attn.forward(
        q, k, v,
        forward_batch,
        save_kv_cache=True
    )
```

#### 步骤 7：`set_kv_buffer` 写入物理 KV-cache

```python
# flashinfer_backend.py: forward_extend
cache_loc = forward_batch.out_cache_loc  # [100, 101, 102]

forward_batch.token_to_kv_pool.set_kv_buffer(
    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
)
```

**在 `set_kv_buffer` 内部**：
```python
# memory_pool.py: set_kv_buffer
# 量化
if k_scale is not None:
    k.div_(k_scale)
k_quantized = k.to(torch.float8_e4m3fn)

# 写入
k_buffer[layer_id][cache_loc] = k_quantized
v_buffer[layer_id][cache_loc] = v_quantized
```

**内存布局**：
```python
# Layer 0 的 K buffer
k_buffer[0][100] = k[0]  # 第 1 个 token 的 K
k_buffer[0][101] = k[1]  # 第 2 个 token 的 K
k_buffer[0][102] = k[2]  # 第 3 个 token 的 K
```

#### 步骤 8：插入 RadixCache

```python
# radix_cache.py
radix_cache.insert(
    key=req.input_ids,  # [1, 2, 3]
    value=[100, 101, 102]
)
```

**树结构**：
```
root
 └─ node(key=[1, 2, 3], value=[100, 101, 102])
```

### 7.2 Decode 阶段（逐个生成新 token）

**场景**：生成第 1 个新 token

#### 步骤 1：分配新 token 的 KV-cache 索引

```python
# allocator.py
new_idx = token_to_kv_allocator.alloc(1)[0]  # 假设分配到 103
```

#### 步骤 2：更新 ReqToTokenPool

```python
# 当前 seq_len = 3，新 token 的位置是 3
req_to_token_pool.write(
    (req_pool_idx, 3),  # 槽位 5，位置 3
    torch.tensor([new_idx])  # [103]
)
# 结果：req_to_token[5, :4] = [100, 101, 102, 103]
```

#### 步骤 3：模型前向传播

```python
# 只处理最后一个 token
hidden_states = embedding(new_token_id)  # [1, hidden_dim]

for layer_id in range(num_layers):
    # 只计算新 token 的 Q
    q = layer.q_proj(hidden_states)  # [1, head_num, head_dim]
    k = layer.k_proj(hidden_states)  # [1, head_num, head_dim]
    v = layer.v_proj(hidden_states)  # [1, head_num, head_dim]

    # Attention 计算
    output = layer.attn.forward(q, k, v, forward_batch, save_kv_cache=True)
```

#### 步骤 4：`forward_decode` 读取历史 KV-cache

```python
# flashinfer_backend.py: forward_decode
k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)
# k_buffer.shape = [size, head_num, head_dim]

# 使用 FlashInfer 的 decode kernel
output = decode_wrapper.forward(
    q,                  # [1, head_num, head_dim]（新 token）
    (k_buffer, v_buffer),  # 所有历史 K, V
    sm_scale=layer.scaling,
    k_scale=layer.k_scale_float,  # 反量化
    v_scale=layer.v_scale_float,
)
```

**FlashInfer 内部流程**（简化）：
```python
# 读取该请求的所有 token 索引
token_indices = req_to_token[req_pool_idx, :seq_len]  # [100, 101, 102, 103]

# 读取所有历史 K
K_history = k_buffer[token_indices]  # [4, head_num, head_dim]
V_history = v_buffer[token_indices]

# 反量化
K_history = K_history.to(torch.bfloat16) * k_scale
V_history = V_history.to(torch.bfloat16) * v_scale

# Attention 计算
scores = q @ K_history.transpose(-2, -1) / sqrt(head_dim)  # [1, head_num, 4]
probs = softmax(scores, dim=-1)
output = probs @ V_history  # [1, head_num, head_dim]
```

#### 步骤 5：写入新 token 的 KV

```python
# set_kv_buffer
k_buffer[layer_id][103] = k  # 新 token 的 K
v_buffer[layer_id][103] = v  # 新 token 的 V
```

#### 步骤 6：更新 RadixCache

```python
# 扩展节点
radix_cache.extend(
    node=matched_node,
    new_tokens=[new_token_id],
    new_indices=[103]
)
```

### 7.3 第二个请求的前缀共享

**场景**：新请求 `"Hello everyone"`，tokenize 后是 `[1, 4, 5]`

#### 步骤 1：RadixCache 匹配

```python
matched_node, matched_len = radix_cache.match([1, 4, 5])
# 结果：matched_node = node([1]), matched_len = 1
# 因为只有第 1 个 token 相同
```

#### 步骤 2：复用 KV-cache 索引

```python
# 复用第 1 个 token 的索引
reused_indices = matched_node.value  # [100]

# 为新 token 分配索引
new_indices = allocator.alloc(2)  # [104, 105]

# 合并
all_indices = torch.cat([reused_indices, new_indices])  # [100, 104, 105]
```

**关键**：`reused_indices[0] = 100` 指向的 KV-cache 已经被第一个请求计算好了！

#### 步骤 3：只计算新 token

```python
# Prefill 只需要计算后 2 个 token
input_ids = [4, 5]  # 跳过已缓存的 token
hidden_states = embedding(input_ids)
# ... 前向传播
```

**内存布局**：
```python
# K buffer（共享）
k_buffer[0][100] = ... # 被两个请求共享！
k_buffer[0][104] = ... # 新请求专属
k_buffer[0][105] = ... # 新请求专属
```

---

## 八、性能优化技巧

### 8.1 双流并行写入

**代码位置**：`memory_pool.py:822-829`

```python
if get_is_capture_mode() and self.alt_stream is not None:
    current_stream = self.device_module.current_stream()
    self.alt_stream.wait_stream(current_stream)
    self.k_buffer[layer_id - self.start_layer][loc] = cache_k
    with self.device_module.stream(self.alt_stream):
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
    current_stream.wait_stream(self.alt_stream)
```

**收益**：
- 小 batch（1-4）：提升 5-10%
- 大 batch（>8）：收益不明显（已被计算饱和）

### 8.2 延迟索引合并

**代码位置**：`allocator.py:146-147`

```python
if self.need_sort and need_size > len(self.free_pages):
    self.merge_and_sort_free()
```

**收益**：
- 减少 `torch.sort` 调用次数（从每次 `free` 到每次不足时）
- 批量处理更高效

### 8.3 FP8 的 uint8 存储

**代码位置**：`memory_pool.py:470-474`

```python
if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
    self.store_dtype = torch.uint8
else:
    self.store_dtype = dtype
```

**收益**：
- 绕过 PyTorch 对 FP8 的 `index_put` 限制
- 无额外开销（只是 view）

### 8.4 数据指针缓存

**代码位置**：`memory_pool.py:670-687`

```python
self.k_data_ptrs = torch.tensor(
    [x.data_ptr() for x in self.k_buffer],
    dtype=torch.uint64,
    device=self.device,
)
```

**收益**：
- Triton kernel 可以直接使用指针，避免 Python 调用开销
- 支持跨层并行访问

### 8.5 Padding 槽位 0

**代码位置**：`memory_pool.py:652`

```python
# The padded slot 0 is used for writing dummy outputs from padded tokens.
self.k_buffer = [
    torch.zeros(
        (self.size + self.page_size, ...),  # +page_size 包含槽位 0
        ...
    )
]
```

**收益**：
- 避免分支判断（统一用索引 0 处理 padding token）
- 简化 kernel 逻辑

---

## 九、总结

### 9.1 核心设计理念

1. **三层解耦**：Request → Token → Physical KV
2. **索引重定向**：用轻量级索引代替重量级数据拷贝
3. **前缀共享**：RadixCache 实现多请求 KV 复用
4. **延迟优化**：索引合并、流同步等操作按需触发
5. **硬件适配**：FP8 存储、双流并行、数据指针缓存

### 9.2 关键数据结构

| 结构 | 类型 | 形状 | 作用 |
|------|------|------|------|
| `req_to_token` | `Tensor` | `[max_req, max_len]` | Request → Token 索引 |
| `k_buffer` | `List[Tensor]` | `layer_num × [size, heads, dim]` | 物理 K 缓存 |
| `v_buffer` | `List[Tensor]` | `layer_num × [size, heads, dim]` | 物理 V 缓存 |
| `free_pages` | `Tensor` | `[N]` | 可用 token 索引 |
| `RadixCache` | `TreeNode` | N/A | 前缀共享树 |

### 9.3 关键代码路径

**Prefill**：
```
Scheduler → ReqToTokenPool.alloc → RadixCache.match
→ Allocator.alloc → Model.forward → set_kv_buffer
→ RadixCache.insert
```

**Decode**：
```
Scheduler → Allocator.alloc(1) → Model.forward
→ get_kv_buffer（读取历史）→ set_kv_buffer（写入新 token）
→ RadixCache.extend
```

### 9.4 文件索引速查

| 功能 | 文件 | 行数范围 |
|------|------|---------|
| **ReqToTokenPool** | `memory_pool.py` | 79-127 |
| **KVCache 基类** | `memory_pool.py` | 453-555 |
| **MHATokenToKVPool** | `memory_pool.py` | 557-843 |
| **set_kv_buffer** | `memory_pool.py` | 794-832 |
| **Allocator** | `allocator.py` | 118-173 |
| **RadixCache** | `radix_cache.py` | 全文 |
| **FlashInfer 集成** | `flashinfer_backend.py` | 692-830 |

---

**文档版本**：v1.0
**最后更新**：2025-01-06
**作者**：SGLang Team

**下一步阅读**：
- [KV-Cache 量化分析](kv_cache_quantization_analysis.md)
- [KV-Cache 基础概念](kv_cache_analysis.md)
