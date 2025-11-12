#!/usr/bin/env python3
"""
诊断训练Loss与评测效果背离问题的工具脚本

使用方法:
    python diagnose_loss_eval_divergence.py \
        --train_log /path/to/train.log \
        --eval_log /path/to/eval.log \
        --output_dir ./divergence_analysis
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class TrainingLogAnalyzer:
    """分析训练日志，诊断loss-eval divergence问题"""

    def __init__(self, train_log_path: str, eval_log_path: Optional[str] = None):
        self.train_log_path = train_log_path
        self.eval_log_path = eval_log_path
        self.train_data = defaultdict(list)
        self.eval_data = defaultdict(list)
        self.divergence_point = None

    def parse_train_log(self):
        """解析训练日志"""
        print(f"📖 解析训练日志: {self.train_log_path}")

        patterns = {
            'step': r'step[:\s]+(\d+)',
            'loss': r'loss[:\s]+([\d.]+)',
            'lr': r'(?:learning_rate|lr)[:\s]+([\d.e-]+)',
            'grad_norm': r'grad_norm[:\s]+([\d.e-]+)',
            'tokens': r'tokens[:\s]+(\d+)',
        }

        with open(self.train_log_path, 'r') as f:
            for line in f:
                for key, pattern in patterns.items():
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            value = float(match.group(1))
                            self.train_data[key].append(value)
                        except ValueError:
                            continue

        print(f"  ✅ 解析完成: {len(self.train_data['step'])} 条训练记录")
        return self.train_data

    def parse_eval_log(self):
        """解析评测日志"""
        if not self.eval_log_path:
            print("⚠️  未提供评测日志路径")
            return self.eval_data

        print(f"📖 解析评测日志: {self.eval_log_path}")

        patterns = {
            'step': r'step[:\s]+(\d+)',
            'eval_loss': r'eval_loss[:\s]+([\d.]+)',
            'accuracy': r'accuracy[:\s]+([\d.]+)',
            'perplexity': r'perplexity[:\s]+([\d.]+)',
            'score': r'score[:\s]+([\d.]+)',
        }

        with open(self.eval_log_path, 'r') as f:
            for line in f:
                for key, pattern in patterns.items():
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            value = float(match.group(1))
                            self.eval_data[key].append(value)
                        except ValueError:
                            continue

        print(f"  ✅ 解析完成: {len(self.eval_data.get('step', []))} 条评测记录")
        return self.eval_data

    def detect_divergence(self, window_size: int = 10) -> Optional[int]:
        """
        检测loss-eval divergence发生的点

        逻辑: 训练loss持续下降，但评测指标开始下降
        """
        print("\n🔍 检测divergence点...")

        if not self.train_data.get('loss') or not self.eval_data:
            print("  ⚠️  数据不足，无法检测divergence")
            return None

        # 获取主要评测指标
        eval_metric_key = None
        for key in ['accuracy', 'score', 'eval_loss']:
            if key in self.eval_data and len(self.eval_data[key]) > window_size:
                eval_metric_key = key
                break

        if not eval_metric_key:
            print("  ⚠️  未找到合适的评测指标")
            return None

        train_losses = self.train_data['loss']
        eval_metrics = self.eval_data[eval_metric_key]

        # 对于eval_loss，值越小越好；对于accuracy/score，值越大越好
        should_decrease = eval_metric_key == 'eval_loss'

        # 查找divergence点
        for i in range(window_size, min(len(train_losses), len(eval_metrics))):
            # 最近window_size个训练loss的趋势
            recent_train = train_losses[i - window_size:i]
            train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]

            # 最近window_size个评测指标的趋势
            recent_eval = eval_metrics[i - window_size:i]
            eval_trend = np.polyfit(range(len(recent_eval)), recent_eval, 1)[0]

            # 检测divergence: 训练loss下降，但评测效果变差
            if train_trend < -0.001:  # 训练loss在下降
                if should_decrease:
                    # eval_loss应该下降，但在上升
                    if eval_trend > 0.001:
                        self.divergence_point = i
                        print(f"  ✅ 检测到divergence点: step {i}")
                        print(f"     训练loss趋势: {train_trend:.6f} (下降)")
                        print(f"     评测{eval_metric_key}趋势: {eval_trend:.6f} (上升)")
                        return i
                else:
                    # accuracy/score应该上升，但在下降
                    if eval_trend < -0.001:
                        self.divergence_point = i
                        print(f"  ✅ 检测到divergence点: step {i}")
                        print(f"     训练loss趋势: {train_trend:.6f} (下降)")
                        print(f"     评测{eval_metric_key}趋势: {eval_trend:.6f} (下降)")
                        return i

        print("  ℹ️  未检测到明显的divergence")
        return None

    def compute_statistics(self) -> Dict:
        """计算统计信息"""
        print("\n📊 计算统计信息...")

        stats = {
            'train': {},
            'eval': {},
            'divergence_point': self.divergence_point,
        }

        # 训练统计
        for key in ['loss', 'lr', 'grad_norm']:
            if key in self.train_data and self.train_data[key]:
                values = self.train_data[key]
                stats['train'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1],
                }

        # 评测统计
        for key in self.eval_data:
            if key != 'step' and self.eval_data[key]:
                values = self.eval_data[key]
                stats['eval'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1],
                }

                # 如果检测到divergence点，计算前后统计
                if self.divergence_point and self.divergence_point < len(values):
                    before = values[:self.divergence_point]
                    after = values[self.divergence_point:]
                    stats['eval'][key]['before_divergence'] = {
                        'mean': np.mean(before),
                        'final': before[-1] if before else None,
                    }
                    stats['eval'][key]['after_divergence'] = {
                        'mean': np.mean(after),
                        'final': after[-1] if after else None,
                    }

        return stats

    def plot_analysis(self, output_dir: str):
        """绘制分析图表"""
        print(f"\n📈 生成分析图表...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 训练loss曲线
        if 'loss' in self.train_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            steps = range(len(self.train_data['loss']))
            ax.plot(steps, self.train_data['loss'], label='Training Loss', alpha=0.7)

            if self.divergence_point:
                ax.axvline(x=self.divergence_point, color='red',
                          linestyle='--', linewidth=2, label='Divergence Point')

            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_path = output_dir / 'train_loss.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ 保存: {plot_path}")
            plt.close()

        # 2. 学习率曲线
        if 'lr' in self.train_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            steps = range(len(self.train_data['lr']))
            ax.plot(steps, self.train_data['lr'], label='Learning Rate', color='orange')

            if self.divergence_point:
                ax.axvline(x=self.divergence_point, color='red',
                          linestyle='--', linewidth=2, label='Divergence Point')

            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            plot_path = output_dir / 'learning_rate.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ 保存: {plot_path}")
            plt.close()

        # 3. 评测指标曲线
        if self.eval_data:
            eval_keys = [k for k in self.eval_data.keys() if k != 'step']

            if eval_keys:
                fig, axes = plt.subplots(len(eval_keys), 1,
                                        figsize=(12, 4 * len(eval_keys)))
                if len(eval_keys) == 1:
                    axes = [axes]

                for idx, key in enumerate(eval_keys):
                    if not self.eval_data[key]:
                        continue

                    steps = range(len(self.eval_data[key]))
                    axes[idx].plot(steps, self.eval_data[key],
                                  label=key, marker='o', markersize=4)

                    if self.divergence_point:
                        axes[idx].axvline(x=self.divergence_point, color='red',
                                         linestyle='--', linewidth=2,
                                         label='Divergence Point')

                    axes[idx].set_xlabel('Evaluation Step')
                    axes[idx].set_ylabel(key)
                    axes[idx].set_title(f'{key} Over Time')
                    axes[idx].legend()
                    axes[idx].grid(True, alpha=0.3)

                plt.tight_layout()
                plot_path = output_dir / 'eval_metrics.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"  ✅ 保存: {plot_path}")
                plt.close()

        # 4. Loss vs Eval综合视图
        if 'loss' in self.train_data and self.eval_data:
            eval_metric_key = None
            for key in ['accuracy', 'score', 'eval_loss']:
                if key in self.eval_data:
                    eval_metric_key = key
                    break

            if eval_metric_key:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                # 上图：训练loss
                train_steps = range(len(self.train_data['loss']))
                ax1.plot(train_steps, self.train_data['loss'],
                        label='Training Loss', color='blue', alpha=0.7)
                ax1.set_xlabel('Step')
                ax1.set_ylabel('Training Loss', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.grid(True, alpha=0.3)

                if self.divergence_point:
                    ax1.axvline(x=self.divergence_point, color='red',
                              linestyle='--', linewidth=2, label='Divergence Point')

                ax1.legend(loc='upper right')
                ax1.set_title('Training Loss and Evaluation Metrics')

                # 下图：评测指标
                eval_steps = range(len(self.eval_data[eval_metric_key]))
                ax2.plot(eval_steps, self.eval_data[eval_metric_key],
                        label=f'Eval {eval_metric_key}', color='green',
                        marker='o', markersize=4)
                ax2.set_xlabel('Evaluation Step')
                ax2.set_ylabel(f'Eval {eval_metric_key}', color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                ax2.grid(True, alpha=0.3)

                if self.divergence_point:
                    ax2.axvline(x=self.divergence_point, color='red',
                              linestyle='--', linewidth=2, label='Divergence Point')

                ax2.legend(loc='upper right')

                plt.tight_layout()
                plot_path = output_dir / 'loss_vs_eval.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"  ✅ 保存: {plot_path}")
                plt.close()

        # 5. 梯度范数
        if 'grad_norm' in self.train_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            steps = range(len(self.train_data['grad_norm']))
            ax.plot(steps, self.train_data['grad_norm'],
                   label='Gradient Norm', color='purple', alpha=0.7)

            if self.divergence_point:
                ax.axvline(x=self.divergence_point, color='red',
                          linestyle='--', linewidth=2, label='Divergence Point')

            ax.set_xlabel('Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_path = output_dir / 'gradient_norm.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ 保存: {plot_path}")
            plt.close()

    def generate_report(self, output_dir: str, stats: Dict):
        """生成诊断报告"""
        print(f"\n📝 生成诊断报告...")

        output_dir = Path(output_dir)
        report_path = output_dir / 'diagnosis_report.md'

        with open(report_path, 'w') as f:
            f.write("# Loss-Eval Divergence 诊断报告\n\n")
            f.write(f"**生成时间**: {np.datetime64('now')}\n\n")

            # 概览
            f.write("## 概览\n\n")
            if self.divergence_point:
                f.write(f"⚠️ **检测到divergence**: 在step {self.divergence_point}\n\n")
            else:
                f.write("✅ **未检测到明显divergence**\n\n")

            # 训练统计
            f.write("## 训练统计\n\n")
            if stats['train']:
                f.write("| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 最终值 |\n")
                f.write("|------|------|--------|--------|--------|--------|\n")
                for key, values in stats['train'].items():
                    f.write(f"| {key} | {values['mean']:.6f} | {values['std']:.6f} | "
                           f"{values['min']:.6f} | {values['max']:.6f} | {values['final']:.6f} |\n")
            f.write("\n")

            # 评测统计
            f.write("## 评测统计\n\n")
            if stats['eval']:
                f.write("| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 最终值 |\n")
                f.write("|------|------|--------|--------|--------|--------|\n")
                for key, values in stats['eval'].items():
                    f.write(f"| {key} | {values['mean']:.6f} | {values['std']:.6f} | "
                           f"{values['min']:.6f} | {values['max']:.6f} | {values['final']:.6f} |\n")
            f.write("\n")

            # Divergence分析
            if self.divergence_point:
                f.write("## Divergence点前后对比\n\n")
                for key, values in stats['eval'].items():
                    if 'before_divergence' in values and 'after_divergence' in values:
                        before = values['before_divergence']
                        after = values['after_divergence']
                        f.write(f"### {key}\n\n")
                        f.write(f"- **Divergence前**: 均值={before['mean']:.6f}, "
                               f"最终={before['final']:.6f}\n")
                        f.write(f"- **Divergence后**: 均值={after['mean']:.6f}, "
                               f"最终={after['final']:.6f}\n")
                        if before['final'] and after['final']:
                            change = (after['final'] - before['final']) / before['final'] * 100
                            f.write(f"- **变化**: {change:+.2f}%\n")
                        f.write("\n")

            # 诊断建议
            f.write("## 诊断建议\n\n")
            if self.divergence_point:
                f.write("### 🚨 发现问题\n\n")
                f.write("训练loss持续下降但评测效果变差，可能的原因包括：\n\n")
                f.write("1. **过拟合** - 模型记住了训练数据但泛化能力下降\n")
                f.write("2. **分布偏移** - 训练数据和评测数据分布不一致\n")
                f.write("3. **学习率过高** - 导致模型在后期不稳定\n")
                f.write("4. **数据质量问题** - 训练数据存在噪声或错误\n\n")

                f.write("### 💡 建议措施\n\n")
                f.write("1. **立即措施**\n")
                f.write(f"   - 回退到step {self.divergence_point}之前的checkpoint\n")
                f.write("   - 降低学习率（如减半）\n")
                f.write("   - 增加正则化（weight decay）\n\n")

                f.write("2. **调查措施**\n")
                f.write("   - 检查训练数据和评测数据的分布差异\n")
                f.write("   - 分析divergence点前后的样本预测结果\n")
                f.write("   - 查看是否有配置或数据变更\n\n")

                f.write("3. **预防措施**\n")
                f.write("   - 实施early stopping机制\n")
                f.write("   - 增加评测频率\n")
                f.write("   - 保存更多中间checkpoint\n\n")
            else:
                f.write("### ✅ 训练状态良好\n\n")
                f.write("未检测到明显的loss-eval divergence，但仍建议：\n\n")
                f.write("1. 持续监控训练和评测指标\n")
                f.write("2. 定期检查样本预测质量\n")
                f.write("3. 保持checkpoint备份\n\n")

            # 参考图表
            f.write("## 参考图表\n\n")
            f.write("- [训练Loss曲线](train_loss.png)\n")
            f.write("- [学习率曲线](learning_rate.png)\n")
            f.write("- [评测指标曲线](eval_metrics.png)\n")
            f.write("- [Loss vs Eval综合视图](loss_vs_eval.png)\n")
            if 'grad_norm' in self.train_data:
                f.write("- [梯度范数曲线](gradient_norm.png)\n")
            f.write("\n")

        print(f"  ✅ 保存报告: {report_path}")

        # 保存JSON格式的统计数据
        stats_path = output_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"  ✅ 保存统计数据: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="诊断训练Loss与评测效果背离问题"
    )
    parser.add_argument(
        '--train_log',
        type=str,
        required=True,
        help='训练日志文件路径'
    )
    parser.add_argument(
        '--eval_log',
        type=str,
        default=None,
        help='评测日志文件路径（可选，如果与训练日志分开）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./divergence_analysis',
        help='输出目录'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=10,
        help='检测divergence的窗口大小'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Loss-Eval Divergence 诊断工具")
    print("=" * 60)

    # 创建分析器
    analyzer = TrainingLogAnalyzer(
        train_log_path=args.train_log,
        eval_log_path=args.eval_log or args.train_log
    )

    # 解析日志
    analyzer.parse_train_log()
    analyzer.parse_eval_log()

    # 检测divergence
    analyzer.detect_divergence(window_size=args.window_size)

    # 计算统计信息
    stats = analyzer.compute_statistics()

    # 生成图表
    analyzer.plot_analysis(args.output_dir)

    # 生成报告
    analyzer.generate_report(args.output_dir, stats)

    print("\n" + "=" * 60)
    print(f"✅ 诊断完成！结果保存在: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
