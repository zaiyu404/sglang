# KV-Cache 深度分析

## 一、KV-Cache 的由来

### 1.1 问题背景

在 Transformer 模型的**自回归生成**（Autoregressive Generation）过程中存在一个核心问题：

- **生成特点**：每次只生成一个新的 token
- **计算需求**：每生成一个新 token 时，都需要与所有历史 token 进行 attention 计算
- **计算冗余**：如果不使用缓存，每次都要重新计算所有历史 token 的 K 和 V 矩阵

**示例**：
```
步骤 1: 输入 "今天"     → 生成 "天气"
步骤 2: 输入 "今天天气"   → 生成 "很"
步骤 3: 输入 "今天天气很" → 生成 "好"
```

在步骤 3 中，"今天" 和 "天气" 的 K、V 矩阵在步骤 1 和步骤 2 中已经计算过了，但如果没有 KV-cache，就需要重复计算。

### 1.2 计算复杂度分析

**不使用 KV-cache**：
- 生成第 t 个 token 时，需要计算 t 个 token 的 K 和 V
- 总计算量：O(1 + 2 + 3 + ... + n) = O(n²)

**使用 KV-cache**：
- 生成第 t 个 token 时，只需要计算当前 token 的 K 和 V
- 总计算量：O(n)

**性能提升**：KV-cache 将生成时间复杂度从 O(n²) 降低到 O(n)，在长文本生成时效果显著。

---

## 二、KV-Cache 在 Decoder 中的作用机制

### 2.1 Attention 计算流程

Self-Attention 的计算公式：
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

**关键步骤**：
1. **线性变换**：`Q = X·W_Q, K = X·W_K, V = X·W_V`
2. **相似度计算**：`Scores = Q·K^T / √d_k`
3. **归一化**：`Weights = softmax(Scores)`
4. **加权求和**：`Output = Weights·V`

### 2.2 在 SGLang 中的具体实现

#### Prefill 阶段（初次处理输入）
```python
# python/sglang/srt/layers/attention/flashinfer_backend.py:716-719
if save_kv_cache:
    forward_batch.token_to_kv_pool.set_kv_buffer(
        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
    )
```
- 计算输入序列所有 token 的 K 和 V
- 将它们存储到 `token_to_kv_pool` 中
- 位置由 `cache_loc` 指定

#### Decode 阶段（逐个生成新 token）
```python
# python/sglang/srt/layers/attention/flashinfer_backend.py:812-828
if k is not None:
    assert v is not None
    if save_kv_cache:
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, cache_loc, k, v, layer.k_scale, layer.v_scale
        )

# 只计算当前 token 的 Q，从缓存中读取所有历史 K 和 V
o = decode_wrapper.forward(
    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),  # 读取缓存的 K、V
    sm_scale=layer.scaling,
    logits_soft_cap=layer.logit_cap,
    k_scale=layer.k_scale_float,
    v_scale=layer.v_scale_float,
)
```

**关键点**：
- 新 token 的 K、V 被追加到 KV-cache 中
- Q 每次都是即时计算，不缓存
- Attention 计算时使用缓存的所有历史 K、V

### 2.3 内存组织结构

SGLang 使用两层索引结构管理 KV-cache：

```
Request → Token Indices → KV Cache Buffer
[Req 0] → [0, 5, 12, 18] → [actual K/V tensors]
[Req 1] → [1, 6, 13, 19] → [actual K/V tensors]
```

**核心组件**：
1. **ReqToTokenPool** (`python/sglang/srt/mem_cache/memory_pool.py`): Request → Token 映射
2. **TokenToKVPool** (`python/sglang/srt/mem_cache/memory_pool.py`): Token → KV 缓存
3. **RadixCache** (`python/sglang/srt/mem_cache/radix_cache.py`): 基数树，支持前缀共享

---

## 三、对其他模型部分的影响

### 3.1 内存管理

**GPU 内存分配**：
```python
# python/sglang/srt/mem_cache/memory_pool.py
class MHATokenToKVPool:
    def __init__(self, ...):
        self.kv_data[layer_id] = torch.empty(
            (size, 2, num_kv_heads, head_dim),  # 2 表示 K 和 V
            dtype=dtype,
            device=device,
        )
```

**内存占用**：
- 每个 token 需要存储 K 和 V 两个向量
- 单个 token 内存：`2 × num_kv_heads × head_dim × sizeof(dtype)`
- 典型例子（Llama-7B，fp16）：`2 × 32 × 128 × 2 bytes = 16 KB/token`
- 生成 2048 tokens：约 32 MB（每层）

### 3.2 批处理和调度

**影响调度策略**：
```python
# python/sglang/srt/managers/scheduler.py
# KV-cache 可用性直接影响调度决策
if available_kv_cache_size < required_size:
    # 需要等待或驱逐某些请求
    evict_requests()
```

**Continuous Batching**：
- SGLang 可以同时处理多个请求，因为每个请求的 KV-cache 独立管理
- Radix Cache 允许不同请求共享相同的前缀（如系统提示词）

### 3.3 前缀缓存（Prefix Caching）

**RadixCache 实现**：
```python
# python/sglang/srt/mem_cache/radix_cache.py
class RadixCache:
    # 使用基数树（Radix Tree）共享公共前缀
    # 例如：多个请求使用相同的 system prompt
```

**优化效果**：
- 多个请求共享相同前缀的 KV-cache
- 减少重复计算和内存占用
- 特别适合批量推理场景

### 3.4 分布式推理

**PD-Disaggregation 模式**：
```python
# python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py
# Prefill 服务器生成 KV-cache，传输给 Decode 服务器
```

**影响**：
- KV-cache 需要在服务器间传输
- 传输延迟成为性能瓶颈之一
- 需要平衡计算和通信开销

### 3.5 量化支持

**KV-Cache 量化**：
```python
# python/sglang/srt/layers/quantization/kv_cache.py
class BaseKVCacheMethod:
    # 支持 FP8 等低精度格式
    # 进一步减少内存占用
```

**效果**：
- FP8 量化可减少 50% 内存占用（相比 FP16）
- 轻微的精度损失换取更大的批处理能力

---

## 四、为什么只有 KV-Cache 没有 Q-Cache

### 4.1 本质原因

#### Q（Query）的特点：
1. **每次都是新的**：Q 代表"当前 token 想要关注什么"
2. **数量少**：在 decode 阶段，每次只有 1 个 token 的 Q
3. **计算量小**：`Q = X·W_Q`，只涉及一个 token 的矩阵乘法
4. **不需要重复使用**：Q 只在当前步骤使用，下一步就不需要了

#### K、V（Key、Value）的特点：
1. **历史信息**：K 和 V 代表所有历史 token 的信息
2. **数量多**：随着生成进行，历史 token 数量线性增长
3. **重复使用**：每一步都需要与所有历史 token 的 K、V 计算 attention
4. **计算量大**：如果重复计算，总复杂度为 O(n²)

### 4.2 具体代码体现

**在 SGLang 的 forward_decode 中**：
```python
# python/sglang/srt/layers/attention/flashinfer_backend.py:820-822
o = decode_wrapper.forward(
    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),  # Q 即时计算，不缓存
    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),   # K、V 从缓存读取
    ...
)
```

**观察**：
- `q` 是临时计算的，直接传入
- `k` 和 `v` 通过 `get_kv_buffer()` 从缓存读取

### 4.3 计算量对比

假设当前已生成 n 个 token，正在生成第 n+1 个：

| 操作 | Q | K | V |
|------|---|---|---|
| **数量** | 1 个 | n 个（历史） | n 个（历史） |
| **新增计算** | 1 次矩阵乘法 | 1 次矩阵乘法 | 1 次矩阵乘法 |
| **重复使用** | 不需要 | 需要（与新 Q 计算） | 需要（加权求和） |
| **是否缓存** | ❌ 否 | ✅ 是 | ✅ 是 |

**结论**：
- 缓存 Q 没有意义，因为下一步不会用到
- 缓存 K 和 V 可以避免 O(n²) 的重复计算

### 4.4 理论类比

可以用"查询数据库"来类比：

```
Q = 查询语句（每次不同）
K = 数据库索引（固定，可复用）
V = 数据内容（固定，可复用）

Attention = 用查询语句（Q）在索引（K）中找到相关项，返回对应内容（V）
```

- 查询语句每次都不同，不需要缓存
- 数据库索引和内容是固定的，应该缓存起来

### 4.5 极端情况：如果缓存 Q 会怎样？

假设我们也缓存 Q：

```python
# 伪代码
q_cache = [q_1, q_2, ..., q_n]  # 存储所有历史 Q
```

**问题**：
1. **没有用处**：下一步生成时，这些 Q 完全不会被用到
2. **浪费内存**：存储 n 个 Q，但永远不读取
3. **增加复杂度**：需要管理额外的缓存结构

### 4.6 相关优化：MQA 和 GQA

虽然不缓存 Q，但业界有针对 KV-cache 的优化：

**Multi-Query Attention (MQA)**：
```python
# 多个 Q head，但只有 1 个 K head 和 1 个 V head
num_q_heads = 32
num_kv_heads = 1  # 大幅减少 KV-cache 大小
```

**Grouped-Query Attention (GQA)**：
```python
# 多个 Q head 共享少数几个 K/V head
num_q_heads = 32
num_kv_heads = 4  # 平衡性能和内存
```

**在 SGLang 中的支持**：
```python
# python/sglang/srt/layers/radix_attention.py:67-69
self.tp_q_head_num = num_heads
self.tp_k_head_num = num_kv_heads  # 可以 < num_heads
self.tp_v_head_num = num_kv_heads
```

---

## 五、总结

### 5.1 核心要点

1. **KV-cache 的本质**：缓存历史 token 的 Key 和 Value，避免重复计算
2. **作用位置**：在 Decoder 的 Self-Attention 层
3. **优化效果**：将生成复杂度从 O(n²) 降到 O(n)
4. **为什么不缓存 Q**：Q 每次都是新的，不会重复使用

### 5.2 在 SGLang 中的实现亮点

1. **两层索引结构**：Request → Token → KV，灵活高效
2. **Radix Cache**：前缀共享，减少重复内存占用
3. **量化支持**：FP8 等低精度格式，进一步节省内存
4. **分布式支持**：PD-Disaggregation，KV-cache 可跨服务器传输
5. **多后端支持**：FlashInfer、Triton、CUTLASS 等多种优化实现

### 5.3 关键文件索引

| 功能 | 关键文件 |
|------|---------|
| Attention 层 | `python/sglang/srt/layers/radix_attention.py` |
| KV 内存池 | `python/sglang/srt/mem_cache/memory_pool.py` |
| 前缀缓存 | `python/sglang/srt/mem_cache/radix_cache.py` |
| FlashInfer 后端 | `python/sglang/srt/layers/attention/flashinfer_backend.py` |
| 缓存分配 | `python/sglang/srt/mem_cache/allocator.py` |
| 缓存操作 | `python/sglang/srt/mem_cache/common.py` |

---

## 六、参考链接

- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [SGLang 文档：Attention Backend](docs/advanced_features/attention_backend.md)
- [SGLang 文档：HiCache 设计](docs/advanced_features/hicache_design.md)
