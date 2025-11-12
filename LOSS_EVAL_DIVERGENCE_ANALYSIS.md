# 训练Loss与评测效果背离问题分析

## 问题描述

训练1T token后，从某个点开始：
- ✅ 训练loss继续下降
- ❌ 评测效果变差

这是一个常见但严重的训练问题，需要系统性地诊断和解决。

---

## 可能原因分析

### 1. **过拟合 (Overfitting)**
最常见的原因，模型记住了训练数据但泛化能力下降。

**诊断方法：**
- 检查训练loss和验证loss的曲线
- 如果训练loss持续下降，但验证loss上升或停滞 → 过拟合

**解决方案：**
```python
# 1. 增加正则化
- 使用weight decay
- 增加dropout rate
- 使用layer normalization

# 2. 早停 (Early Stopping)
- 监控验证集指标
- 当验证集性能不再提升时停止训练
- 回退到最佳checkpoint

# 3. 减小模型容量
- 减少层数或hidden size
- 使用参数共享

# 4. 数据增强
- 增加训练数据多样性
- 使用data augmentation
```

### 2. **训练数据与评测数据分布偏移 (Distribution Shift)**
训练数据和评测数据的分布不一致。

**诊断方法：**
- 分析训练集和测试集的统计特征
- 检查数据来源、领域、时间跨度
- 计算数据分布距离 (KL divergence, JS divergence)

**解决方案：**
```python
# 1. 重新划分数据集
- 确保训练集和验证集来自相同分布
- 使用stratified sampling

# 2. 领域适应 (Domain Adaptation)
- 使用混合数据训练
- 添加目标领域数据到训练集

# 3. 数据重加权
- 对训练数据进行importance weighting
- 使训练数据分布更接近评测数据
```

### 3. **学习率过高导致的模型退化**
在训练后期，过高的学习率可能导致模型性能波动。

**诊断方法：**
- 检查学习率调度曲线
- 观察loss是否有剧烈波动
- 查看参数梯度norm的变化

**解决方案：**
```python
# 1. 调整学习率调度
- 使用cosine annealing decay
- 在发现问题的点降低学习率
- 使用warmup策略

# 2. 使用更稳定的优化器
- AdamW with gradient clipping
- 降低beta2 (如从0.999降到0.995)

# 3. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. **评测指标不合适**
Loss下降但任务性能变差，可能是优化目标与实际任务目标不一致。

**诊断方法：**
- 检查训练时使用的loss函数
- 分析评测指标的定义
- 查看具体样本的预测结果

**解决方案：**
```python
# 1. 调整训练目标
- 使用更接近下游任务的loss
- 多任务学习
- 添加auxiliary loss

# 2. 改进评测指标
- 使用更全面的评测集
- 增加多个评测维度
- 使用人工评估补充自动评测
```

### 5. **数据质量问题**
训练数据中存在噪声、标注错误或低质量样本。

**诊断方法：**
- 检查训练数据的标注一致性
- 分析loss最低的样本
- 使用数据清洗工具

**解决方案：**
```python
# 1. 数据清洗
- 过滤异常样本
- 修正标注错误
- 去重

# 2. 样本重加权
- 降低可疑样本的权重
- 使用curriculum learning

# 3. 添加数据验证
- 定期检查数据质量
- 使用数据质量评分
```

### 6. **批次大小问题**
大批次训练可能导致泛化性能下降。

**诊断方法：**
- 记录不同批次大小下的性能
- 检查批次统计信息

**解决方案：**
```python
# 1. 调整批次大小
- 尝试减小batch size
- 使用gradient accumulation保持等效batch size

# 2. 使用批次归一化技巧
- Group Normalization
- Layer Normalization
```

### 7. **训练数据重复或数据比例问题**
在1T token训练中，某些数据可能被重复使用多次。

**诊断方法：**
- 计算数据重复率
- 检查不同数据源的比例
- 分析每个epoch的数据分布

**解决方案：**
```python
# 1. 控制数据重复
- 限制高频数据的重复次数
- 使用数据采样策略

# 2. 平衡数据配比
- 调整不同来源数据的比例
- 使用动态数据混合策略
```

---

## 诊断流程

### Step 1: 数据分析
```python
# 检查训练集和验证集的统计特征
def analyze_data_distribution(train_data, eval_data):
    # 词频统计
    # 序列长度分布
    # 领域分布
    # 标注分布
    pass

# 计算分布差异
from scipy.stats import entropy

def calculate_distribution_divergence(train_dist, eval_dist):
    kl_div = entropy(train_dist, eval_dist)
    return kl_div
```

### Step 2: 训练曲线分析
```python
# 绘制完整的训练曲线
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, eval_losses, eval_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss curves
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(eval_losses, label='Eval Loss')
    axes[0].axvline(x=DIVERGENCE_POINT, color='r', linestyle='--')
    axes[0].legend()
    axes[0].set_title('Loss Curves')

    # Eval metrics
    axes[1].plot(eval_metrics)
    axes[1].axvline(x=DIVERGENCE_POINT, color='r', linestyle='--')
    axes[1].set_title('Eval Performance')

    # Loss-metric correlation
    axes[2].scatter(train_losses, eval_metrics)
    axes[2].set_title('Loss vs Performance')

    plt.tight_layout()
    plt.savefig('training_analysis.png')
```

### Step 3: 模型行为分析
```python
# 检查模型在关键点的行为
def analyze_model_behavior(model, checkpoint_paths, test_samples):
    results = []

    for ckpt_path in checkpoint_paths:
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        with torch.no_grad():
            # 计算测试样本的loss和预测
            predictions = model(test_samples)
            loss = compute_loss(predictions, labels)

            # 计算评测指标
            metrics = compute_metrics(predictions, labels)

            results.append({
                'checkpoint': ckpt_path,
                'loss': loss,
                'metrics': metrics,
                'predictions': predictions
            })

    return results
```

### Step 4: 参数变化分析
```python
# 分析参数范数和梯度变化
def analyze_parameter_changes(checkpoints):
    param_norms = []
    grad_norms = []

    for ckpt in checkpoints:
        # 参数范数
        total_norm = 0
        for p in ckpt.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        param_norms.append(total_norm)

    plt.plot(param_norms)
    plt.axvline(x=DIVERGENCE_POINT, color='r', linestyle='--')
    plt.title('Parameter Norm Changes')
    plt.savefig('param_norm_analysis.png')
```

---

## 立即可采取的措施

### 1. **紧急措施**
```bash
# 回退到分歧点之前的最佳checkpoint
# 假设最佳checkpoint在800B tokens处
BEST_CHECKPOINT="checkpoint-800B"

# 从该checkpoint继续训练，使用更保守的配置
python train.py \
    --resume_from $BEST_CHECKPOINT \
    --learning_rate 1e-5 \  # 降低学习率
    --weight_decay 0.1 \     # 增加正则化
    --gradient_clip 1.0 \    # 梯度裁剪
    --eval_steps 100 \       # 更频繁的评测
    --save_steps 100 \       # 更频繁的保存
    --early_stopping_patience 5
```

### 2. **监控增强**
```python
# 添加更详细的监控
class EnhancedTrainingMonitor:
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_metrics = []
        self.learning_rates = []
        self.grad_norms = []

    def on_step_end(self, step, loss, grad_norm, lr):
        self.train_losses.append(loss)
        self.grad_norms.append(grad_norm)
        self.learning_rates.append(lr)

    def on_eval_end(self, step, eval_loss, eval_metrics):
        self.eval_losses.append(eval_loss)
        self.eval_metrics.append(eval_metrics)

        # 检测loss-eval divergence
        if len(self.eval_metrics) > 10:
            recent_train_loss_trend = self.train_losses[-100:-1]
            recent_eval_metric_trend = self.eval_metrics[-10:-1]

            if is_decreasing(recent_train_loss_trend) and \
               is_decreasing(recent_eval_metric_trend):
                print("⚠️ WARNING: Detected loss-eval divergence!")
                self.trigger_alert()
```

### 3. **A/B测试**
```python
# 训练多个配置进行对比
configs = [
    {'lr': 5e-6, 'wd': 0.1, 'name': 'conservative'},
    {'lr': 1e-5, 'wd': 0.05, 'name': 'moderate'},
    {'lr': 2e-5, 'wd': 0.01, 'name': 'aggressive'},
]

for config in configs:
    train_with_config(
        resume_from=BEST_CHECKPOINT,
        **config
    )
```

---

## 预防措施

### 1. **建立Early Stopping机制**
```python
class EarlyStoppingWithDivergenceDetection:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_metric = -float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, eval_metric, train_loss):
        if eval_metric > self.best_metric + self.delta:
            self.best_metric = eval_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        # 检测divergence
        if train_loss < self.best_train_loss and \
           eval_metric < self.best_metric - 0.05:
            print("⚠️ Divergence detected! Consider stopping.")
```

### 2. **定期checkpoint保存**
```python
# 保存更多的中间checkpoint
# 每100B tokens保存一次，保留最近10个
SAVE_INTERVAL = "100B"
KEEP_RECENT = 10
```

### 3. **多维度评测**
```python
# 不要只看单一指标
evaluation_suite = {
    'perplexity': compute_perplexity,
    'task_accuracy': compute_task_accuracy,
    'distribution_similarity': compute_distribution_similarity,
    'sample_quality': compute_sample_quality,
    'diversity': compute_diversity,
}
```

---

## 需要收集的信息

为了更准确地诊断问题，请提供：

1. **训练配置**
   - 模型架构和大小
   - 优化器和学习率调度
   - 批次大小和gradient accumulation
   - 正则化设置 (weight decay, dropout)

2. **数据信息**
   - 训练数据规模和来源
   - 数据重复策略
   - 训练集和评测集的划分方式

3. **训练曲线**
   - 完整的训练loss曲线
   - 验证loss曲线
   - 评测指标随时间的变化
   - 学习率曲线

4. **分歧点信息**
   - 具体在哪个token/step开始出现问题
   - 该点前后的配置变化
   - 是否有数据切换或其他变动

5. **评测详情**
   - 使用的评测指标
   - 评测数据集信息
   - 具体的性能下降幅度

---

## SGLang相关实现

虽然SGLang主要是推理框架，但可以利用其评测基础设施：

### 使用SGLang进行模型评测
```python
# 在训练过程中，使用SGLang的评测工具
from sglang.test.simple_eval_common import ChatCompletionSampler, aggregate_results

# 创建sampler
sampler = ChatCompletionSampler(
    base_url="http://localhost:30000",
    model=checkpoint_path,
    temperature=0.0,
    max_tokens=2048
)

# 运行评测
eval_results = run_evaluation(sampler, eval_dataset)

# 记录结果
print(f"Score: {eval_results.score}")
print(f"Metrics: {eval_results.metrics}")
```

### 在SGLang中添加loss tracking
如果您需要在SGLang inference时追踪loss，可以修改：

1. **添加loss计算hook** (`python/sglang/srt/model_executor/model_runner.py`)
2. **记录metrics** (`python/sglang/srt/metrics/collector.py`)
3. **导出统计数据** (`python/sglang/srt/metrics/scheduler_metrics_mixin.py`)

---

## 参考资料

1. "Understanding Deep Learning Requires Rethinking Generalization" - Zhang et al.
2. "Measuring Catastrophic Forgetting in Neural Networks" - Kirkpatrick et al.
3. "On Large-Batch Training for Deep Learning" - Keskar et al.
4. "Curriculum Learning" - Bengio et al.

---

## 下一步行动

请先：
1. ✅ 确认问题出现的具体位置（token数/step数）
2. ✅ 收集该点前后的训练日志和checkpoint
3. ✅ 准备训练配置文件
4. ✅ 描述具体的评测指标变化情况

然后我可以帮您：
- 编写诊断脚本
- 分析训练曲线
- 制定具体的解决方案
- 实现监控和预警系统
