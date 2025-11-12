# Loss-Eval Divergence 诊断工具使用说明

## 快速开始

### 1. 安装依赖

```bash
pip install matplotlib numpy
```

### 2. 运行诊断

```bash
# 如果训练日志和评测日志在同一个文件中
python scripts/diagnose_loss_eval_divergence.py \
    --train_log /path/to/training.log \
    --output_dir ./analysis_output

# 如果训练日志和评测日志分开
python scripts/diagnose_loss_eval_divergence.py \
    --train_log /path/to/training.log \
    --eval_log /path/to/evaluation.log \
    --output_dir ./analysis_output \
    --window_size 10
```

### 3. 查看结果

脚本会在输出目录中生成：

- `diagnosis_report.md` - 完整的诊断报告
- `statistics.json` - 统计数据（JSON格式）
- `train_loss.png` - 训练loss曲线
- `learning_rate.png` - 学习率曲线
- `eval_metrics.png` - 评测指标曲线
- `loss_vs_eval.png` - Loss与评测的对比图
- `gradient_norm.png` - 梯度范数曲线（如果有）

## 日志格式要求

脚本会自动解析常见的训练日志格式，支持以下字段：

### 训练日志
```
step: 1000, loss: 2.345, lr: 1e-4, grad_norm: 0.123, tokens: 1000000
```

或

```
Step 1000 | Loss: 2.345 | Learning Rate: 1e-4 | Gradient Norm: 0.123
```

### 评测日志
```
step: 1000, eval_loss: 2.456, accuracy: 0.85, perplexity: 11.23
```

或

```
Evaluation at step 1000: accuracy=0.85, score=0.90
```

**注意**: 脚本使用正则表达式解析，格式比较灵活，但关键词需要匹配（不区分大小写）。

## 工作原理

### Divergence检测算法

脚本通过以下方式检测loss-eval divergence：

1. **滑动窗口**: 使用指定大小的滑动窗口分析趋势
2. **线性拟合**: 对窗口内的数据进行线性回归，计算趋势斜率
3. **背离判断**:
   - 如果训练loss趋势向下（斜率 < -0.001）
   - 同时评测指标趋势向差的方向（上升或下降，取决于指标类型）
   - 则判定为检测到divergence

### 支持的评测指标

- `eval_loss` - 越小越好
- `perplexity` - 越小越好
- `accuracy` - 越大越好
- `score` - 越大越好

可以在日志中包含任何这些指标，脚本会自动识别。

## 示例

### 示例1: 基本使用

```bash
# 假设你的训练日志格式如下：
# Step 100: loss=2.5, lr=0.0001, tokens=100000
# Step 200: loss=2.3, lr=0.00009, tokens=200000
# Eval at step 200: accuracy=0.75, eval_loss=2.4

python scripts/diagnose_loss_eval_divergence.py \
    --train_log train.log \
    --output_dir results
```

### 示例2: 自定义窗口大小

```bash
# 对于更平滑的divergence检测，可以增大窗口
python scripts/diagnose_loss_eval_divergence.py \
    --train_log train.log \
    --window_size 20 \
    --output_dir results
```

### 示例3: 分离的日志文件

```bash
# 如果训练和评测日志分开保存
python scripts/diagnose_loss_eval_divergence.py \
    --train_log logs/train_2024.log \
    --eval_log logs/eval_2024.log \
    --output_dir analysis_2024
```

## 输出解读

### 检测到Divergence

如果输出显示：
```
✅ 检测到divergence点: step 8500
   训练loss趋势: -0.003421 (下降)
   评测accuracy趋势: -0.001234 (下降)
```

这表明：
- 在step 8500处，训练loss继续下降
- 但评测准确率开始下降
- **建议**: 立即检查该checkpoint，考虑回退并调整训练策略

### 未检测到Divergence

如果输出显示：
```
ℹ️  未检测到明显的divergence
```

这表明：
- 训练过程目前正常
- Loss和评测指标保持一致的趋势
- **建议**: 继续监控，保持当前训练策略

## 常见问题

### Q1: 脚本无法解析我的日志格式

**A**: 请检查日志中是否包含以下关键词（不区分大小写）：
- `step` 或 `Step`
- `loss` 或 `Loss`
- `lr` 或 `learning_rate` 或 `Learning Rate`
- `accuracy` 或 `Accuracy`
- `eval_loss` 或 `Eval Loss`

如果格式特殊，可以修改脚本中的正则表达式模式。

### Q2: 误报Divergence

**A**: 可能原因：
1. 窗口太小，对短期波动过于敏感 → 增大 `--window_size`
2. 评测数据太少 → 增加评测频率
3. 评测指标本身有噪声 → 使用多个评测指标综合判断

### Q3: 没有生成某些图表

**A**: 检查日志中是否包含对应的数据：
- `gradient_norm.png` 需要日志中有 `grad_norm` 字段
- `learning_rate.png` 需要日志中有 `lr` 字段
- `eval_metrics.png` 需要评测日志中有评测指标

## 高级用法

### 批量分析多个实验

```bash
#!/bin/bash
# analyze_all_experiments.sh

for exp_dir in experiments/exp_*/; do
    exp_name=$(basename "$exp_dir")
    echo "Analyzing $exp_name..."

    python scripts/diagnose_loss_eval_divergence.py \
        --train_log "$exp_dir/train.log" \
        --eval_log "$exp_dir/eval.log" \
        --output_dir "analysis/$exp_name" \
        --window_size 15

    echo "Done: $exp_name"
done

echo "All experiments analyzed!"
```

### 集成到训练流程

```python
# 在训练脚本中集成实时监控
import subprocess
import time

def monitor_divergence(train_log_path, output_dir):
    """实时监控训练过程，检测divergence"""
    while training_in_progress:
        # 每1000步运行一次诊断
        time.sleep(600)  # 等待10分钟

        result = subprocess.run([
            'python', 'scripts/diagnose_loss_eval_divergence.py',
            '--train_log', train_log_path,
            '--output_dir', output_dir,
            '--window_size', '10'
        ], capture_output=True, text=True)

        # 检查是否检测到divergence
        if 'divergence点' in result.stdout:
            print("⚠️ 警告: 检测到divergence，考虑调整训练!")
            send_alert_to_team()
```

## 相关文档

- [完整分析文档](../LOSS_EVAL_DIVERGENCE_ANALYSIS.md) - 深入解析原因和解决方案
- [SGLang训练指南](https://sglang.readthedocs.io/) - SGLang官方文档

## 贡献

如果你有改进建议或发现bug，请提交issue或PR。

## 许可

MIT License
