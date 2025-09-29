#!/usr/bin/env python3
"""
对比baseline_tsv.csv和example_run.csv的模拟结果
验证LLMServingSim模拟的合理性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_compare_data():
    """加载并对比两个数据文件"""
    print("🔍 加载对比数据...")

    # 加载数据
    baseline_df = pd.read_csv('output/baseline_tsv.csv')
    example_df = pd.read_csv('output/example_run.csv')

    print(f"baseline_tsv.csv: {len(baseline_df)} 个请求")
    print(f"example_run.csv: {len(example_df)} 个请求")

    return baseline_df, example_df

def analyze_time_units(df, name):
    """分析时间单位和数据范围"""
    print(f"\n📊 {name} 数据分析:")

    # 转换为合理单位显示
    df_copy = df.copy()
    df_copy['latency_ms'] = df_copy['latency'] / 1e6
    df_copy['ttft_ms'] = df_copy['TTFT'] / 1e6
    df_copy['tpot_ms'] = df_copy['TPOT'] / 1e6
    df_copy['queue_delay_ms'] = df_copy['queuing_delay'] / 1e6

    print(f"  • 延迟范围: {df_copy['latency_ms'].min():.1f} - {df_copy['latency_ms'].max():.1f} ms")
    print(f"  • TTFT范围: {df_copy['ttft_ms'].min():.1f} - {df_copy['ttft_ms'].max():.1f} ms")
    print(f"  • TPOT范围: {df_copy['tpot_ms'].min():.1f} - {df_copy['tpot_ms'].max():.1f} ms")
    print(f"  • 队列延迟范围: {df_copy['queue_delay_ms'].min():.1f} - {df_copy['queue_delay_ms'].max():.1f} ms")

    return df_copy

def compare_performance_metrics(baseline_df, example_df):
    """对比性能指标"""
    print("\n📈 性能指标对比:")

    # 计算关键指标
    def calculate_metrics(df, name):
        total_time = (df['end_time'].max() - df['arrival'].min()) / 1e9
        total_output = df['output'].sum()
        throughput = total_output / total_time if total_time > 0 else 0

        metrics = {
            'name': name,
            'requests': len(df),
            'total_time_s': total_time,
            'avg_latency_ms': df['latency'].mean() / 1e6,
            'avg_ttft_ms': df['TTFT'].mean() / 1e6,
            'avg_tpot_ms': df['TPOT'].mean() / 1e6,
            'throughput_tps': throughput,
            'latency_std_ms': df['latency'].std() / 1e6,
            'avg_queue_delay_ms': df['queuing_delay'].mean() / 1e6
        }
        return metrics

    baseline_metrics = calculate_metrics(baseline_df, "Baseline (20请求)")
    example_metrics = calculate_metrics(example_df, "Example (12请求)")

    # 打印对比表格
    print("\n| 指标 | Baseline | Example | 差异 |")
    print("|------|----------|---------|------|")

    comparisons = [
        ('请求数', f"{baseline_metrics['requests']}", f"{example_metrics['requests']}", f"{baseline_metrics['requests'] - example_metrics['requests']}"),
        ('总时间(s)', f"{baseline_metrics['total_time_s']:.2f}", f"{example_metrics['total_time_s']:.2f}", f"{baseline_metrics['total_time_s'] - example_metrics['total_time_s']:.2f}"),
        ('平均延迟(ms)', f"{baseline_metrics['avg_latency_ms']:.1f}", f"{example_metrics['avg_latency_ms']:.1f}", f"{baseline_metrics['avg_latency_ms'] - example_metrics['avg_latency_ms']:.1f}"),
        ('平均TTFT(ms)', f"{baseline_metrics['avg_ttft_ms']:.1f}", f"{example_metrics['avg_ttft_ms']:.1f}", f"{baseline_metrics['avg_ttft_ms'] - example_metrics['avg_ttft_ms']:.1f}"),
        ('平均TPOT(ms)', f"{baseline_metrics['avg_tpot_ms']:.1f}", f"{example_metrics['avg_tpot_ms']:.1f}", f"{baseline_metrics['avg_tpot_ms'] - example_metrics['avg_tpot_ms']:.1f}"),
        ('吞吐量(tok/s)', f"{baseline_metrics['throughput_tps']:.1f}", f"{example_metrics['throughput_tps']:.1f}", f"{baseline_metrics['throughput_tps'] - example_metrics['throughput_tps']:.1f}"),
        ('延迟标准差(ms)', f"{baseline_metrics['latency_std_ms']:.1f}", f"{example_metrics['latency_std_ms']:.1f}", f"{baseline_metrics['latency_std_ms'] - example_metrics['latency_std_ms']:.1f}"),
        ('队列延迟(ms)', f"{baseline_metrics['avg_queue_delay_ms']:.1f}", f"{example_metrics['avg_queue_delay_ms']:.1f}", f"{baseline_metrics['avg_queue_delay_ms'] - example_metrics['avg_queue_delay_ms']:.1f}")
    ]

    for metric, baseline_val, example_val, diff in comparisons:
        print(f"| {metric} | {baseline_val} | {example_val} | {diff} |")

    return baseline_metrics, example_metrics

def analyze_simulation_reasonableness(baseline_metrics, example_metrics):
    """分析模拟结果的合理性"""
    print("\n🤔 模拟结果合理性分析:")

    # 检查关键指标的合理性
    checks = []

    # 1. 延迟合理性检查
    baseline_latency = baseline_metrics['avg_latency_ms']
    example_latency = example_metrics['avg_latency_ms']

    if 10 < baseline_latency < 100000:  # 10ms to 100s
        checks.append("✅ Baseline延迟在合理范围内")
    else:
        checks.append(f"⚠️ Baseline延迟异常: {baseline_latency:.1f}ms")

    if 10 < example_latency < 100000:
        checks.append("✅ Example延迟在合理范围内")
    else:
        checks.append(f"⚠️ Example延迟异常: {example_latency:.1f}ms")

    # 2. TTFT合理性检查 (通常50-500ms)
    baseline_ttft = baseline_metrics['avg_ttft_ms']
    example_ttft = example_metrics['avg_ttft_ms']

    if 50 < baseline_ttft < 2000:
        checks.append("✅ Baseline TTFT在合理范围内")
    else:
        checks.append(f"⚠️ Baseline TTFT异常: {baseline_ttft:.1f}ms")

    if 50 < example_ttft < 2000:
        checks.append("✅ Example TTFT在合理范围内")
    else:
        checks.append(f"⚠️ Example TTFT异常: {example_ttft:.1f}ms")

    # 3. TPOT合理性检查 (通常10-100ms per token)
    baseline_tpot = baseline_metrics['avg_tpot_ms']
    example_tpot = example_metrics['avg_tpot_ms']

    if 10 < baseline_tpot < 500:
        checks.append("✅ Baseline TPOT在合理范围内")
    else:
        checks.append(f"⚠️ Baseline TPOT异常: {baseline_tpot:.1f}ms")

    if 10 < example_tpot < 500:
        checks.append("✅ Example TPOT在合理范围内")
    else:
        checks.append(f"⚠️ Example TPOT异常: {example_tpot:.1f}ms")

    # 4. 吞吐量合理性检查
    baseline_throughput = baseline_metrics['throughput_tps']
    example_throughput = example_metrics['throughput_tps']

    if 10 < baseline_throughput < 1000:
        checks.append("✅ Baseline吞吐量在合理范围内")
    else:
        checks.append(f"⚠️ Baseline吞吐量异常: {baseline_throughput:.1f} tok/s")

    if 10 < example_throughput < 1000:
        checks.append("✅ Example吞吐量在合理范围内")
    else:
        checks.append(f"⚠️ Example吞吐量异常: {example_throughput:.1f} tok/s")

    for check in checks:
        print(f"  {check}")

    # 5. 对比分析
    print(f"\n📊 对比分析:")
    latency_ratio = baseline_latency / example_latency if example_latency > 0 else float('inf')
    throughput_ratio = baseline_throughput / example_throughput if example_throughput > 0 else float('inf')

    print(f"  • Baseline延迟是Example的 {latency_ratio:.1f} 倍")
    print(f"  • Baseline吞吐量是Example的 {throughput_ratio:.1f} 倍")

    if 0.5 < latency_ratio < 2.0:
        print("  ✅ 延迟表现相对一致")
    else:
        print("  ⚠️ 延迟表现差异较大")

    if 0.5 < throughput_ratio < 2.0:
        print("  ✅ 吞吐量表现相对一致")
    else:
        print("  ⚠️ 吞吐量表现差异较大")

def create_comparison_charts(baseline_df, example_df):
    """创建对比图表"""
    # 转换单位
    baseline_df['latency_ms'] = baseline_df['latency'] / 1e6
    example_df['latency_ms'] = example_df['latency'] / 1e6

    baseline_df['ttft_ms'] = baseline_df['TTFT'] / 1e6
    example_df['ttft_ms'] = example_df['TTFT'] / 1e6

    baseline_df['tpot_ms'] = baseline_df['TPOT'] / 1e6
    example_df['tpot_ms'] = example_df['TPOT'] / 1e6

    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LLMServingSim Simulation Results Comparison', fontsize=16, fontweight='bold')

    # 1. 延迟分布对比
    ax1 = axes[0, 0]
    ax1.hist(baseline_df['latency_ms'], bins=10, alpha=0.7, label='Baseline (20 req)', color='blue')
    ax1.hist(example_df['latency_ms'], bins=10, alpha=0.7, label='Example (12 req)', color='orange')
    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('Request Count')
    ax1.set_title('Latency Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. TTFT vs TPOT 散点图对比
    ax2 = axes[0, 1]
    ax2.scatter(baseline_df['ttft_ms'], baseline_df['tpot_ms'], alpha=0.7, label='Baseline', color='blue')
    ax2.scatter(example_df['ttft_ms'], example_df['tpot_ms'], alpha=0.7, label='Example', color='orange')
    ax2.set_xlabel('TTFT (ms)')
    ax2.set_ylabel('TPOT (ms)')
    ax2.set_title('TTFT vs TPOT Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 时间序列对比
    ax3 = axes[1, 0]
    baseline_time = (baseline_df['arrival'] - baseline_df['arrival'].min()) / 1e9
    example_time = (example_df['arrival'] - example_df['arrival'].min()) / 1e9

    ax3.plot(baseline_time, baseline_df['latency_ms'], 'o-', label='Baseline', alpha=0.7, color='blue')
    ax3.plot(example_time, example_df['latency_ms'], 'o-', label='Example', alpha=0.7, color='orange')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('Timeline Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 输入长度vs延迟对比
    ax4 = axes[1, 1]
    ax4.scatter(baseline_df['input'], baseline_df['latency_ms'], alpha=0.7, label='Baseline', color='blue')
    ax4.scatter(example_df['input'], example_df['latency_ms'], alpha=0.7, label='Example', color='orange')
    ax4.set_xlabel('Input Length (tokens)')
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title('Input Length vs Latency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/simulation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("📊 对比图表已保存: output/simulation_comparison.png")

def main():
    """主函数"""
    print("🔍 开始对比LLMServingSim模拟结果...")

    # 加载数据
    baseline_df, example_df = load_and_compare_data()

    # 分析数据
    baseline_converted = analyze_time_units(baseline_df, "baseline_tsv.csv")
    example_converted = analyze_time_units(example_df, "example_run.csv")

    # 对比性能指标
    baseline_metrics, example_metrics = compare_performance_metrics(baseline_df, example_df)

    # 分析合理性
    analyze_simulation_reasonableness(baseline_metrics, example_metrics)

    # 创建对比图表
    create_comparison_charts(baseline_df, example_df)

    print("\n✅ 对比分析完成！")
    print("\n📁 生成的文件:")
    print("  • output/simulation_comparison.png")

if __name__ == "__main__":
    main()