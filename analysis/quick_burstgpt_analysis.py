#!/usr/bin/env python3
"""
BurstGPT快速分析 - 生成基本统计信息和简单图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置论文级图表样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_preprocess_data(data_path):
    """加载并预处理数据"""
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    # 转换时间戳为秒（BurstGPT的时间戳已经是秒单位）
    df['timestamp_sec'] = df['Timestamp']

    # 计算时间间隔
    df = df.sort_values('timestamp_sec')
    df['time_diff'] = df['timestamp_sec'].diff().fillna(0)

    print(f"Data loading completed: {len(df)} records")
    print(f"Time range: {df['timestamp_sec'].min():.1f} - {df['timestamp_sec'].max():.1f} seconds")
    print(f"Time span: {(df['timestamp_sec'].max() - df['timestamp_sec'].min())/3600:.2f} hours")

    return df

def analyze_basic_statistics(df):
    """分析基本统计信息"""
    print("\n=== Basic Statistics ===")

    # 时间间隔统计
    time_diffs = df['time_diff'].values[1:]  # 去除第一个0
    print(f"Time interval statistics:")
    print(f"  Mean: {np.mean(time_diffs):.3f} seconds")
    print(f"  Median: {np.median(time_diffs):.3f} seconds")
    print(f"  Std deviation: {np.std(time_diffs):.3f} seconds")
    print(f"  Min: {np.min(time_diffs):.3f} seconds")
    print(f"  Max: {np.max(time_diffs):.3f} seconds")
    print(f"  Coefficient of variation (burstiness metric): {np.std(time_diffs)/np.mean(time_diffs):.3f}")

    # Token统计
    req_tokens = df['Request tokens'].values
    resp_tokens = df['Response tokens'].values

    print(f"\nRequest Token statistics:")
    print(f"  Mean: {np.mean(req_tokens):.1f}")
    print(f"  Median: {np.median(req_tokens):.1f}")
    print(f"  90th percentile: {np.percentile(req_tokens, 90):.1f}")
    print(f"  95th percentile: {np.percentile(req_tokens, 95):.1f}")
    print(f"  99th percentile: {np.percentile(req_tokens, 99):.1f}")
    print(f"  Max: {np.max(req_tokens):.1f}")

    print(f"\nResponse Token statistics:")
    print(f"  Mean: {np.mean(resp_tokens):.1f}")
    print(f"  Median: {np.median(resp_tokens):.1f}")
    print(f"  90th percentile: {np.percentile(resp_tokens, 90):.1f}")
    print(f"  95th percentile: {np.percentile(resp_tokens, 95):.1f}")
    print(f"  99th percentile: {np.percentile(resp_tokens, 99):.1f}")
    print(f"  Max: {np.max(resp_tokens):.1f}")

    return time_diffs, req_tokens, resp_tokens

def create_basic_figures(df, time_diffs, req_tokens, resp_tokens):
    """创建基本图表"""
    print("\n=== Generating Basic Figures ===")

    # 创建输出目录
    import os
    os.makedirs('analysis/burstgpt_analysis', exist_ok=True)

    # 图1: 时间间隔分布
    plt.figure(figsize=(10, 6))
    plt.hist(time_diffs[time_diffs < np.percentile(time_diffs, 99)], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Inter-arrival Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Request Inter-arrival Time Distribution')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/burstgpt_analysis/01_interarrival_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 图2: Token CDF比较
    plt.figure(figsize=(10, 6))
    sorted_req = np.sort(req_tokens)
    sorted_resp = np.sort(resp_tokens)
    cdf_req = np.arange(1, len(sorted_req) + 1) / len(sorted_req)
    cdf_resp = np.arange(1, len(sorted_resp) + 1) / len(sorted_resp)

    plt.plot(sorted_req, cdf_req, 'b-', linewidth=2, label='Request Tokens')
    plt.plot(sorted_resp, cdf_resp, 'r-', linewidth=2, label='Response Tokens')
    plt.xlabel('Token Count')
    plt.ylabel('Cumulative Distribution Function (CDF)')
    plt.title('Token Long-tail CDF Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('analysis/burstgpt_analysis/02_token_cdf.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 图3: 按小时分布
    plt.figure(figsize=(10, 6))
    hours = (df['timestamp_sec'] // 3600) % 24
    hourly_stats = df.groupby(hours).size()

    plt.bar(hourly_stats.index, hourly_stats.values, color='lightblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Hour')
    plt.ylabel('Request Count')
    plt.title('24-hour Request Distribution Pattern')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)

    # 标记峰谷
    peak_hours = hourly_stats.nlargest(3).index.tolist()
    valley_hours = hourly_stats.nsmallest(3).index.tolist()

    for hour in peak_hours:
        plt.axvline(x=hour, color='red', linestyle='--', alpha=0.7, linewidth=2)
    for hour in valley_hours:
        plt.axvline(x=hour, color='green', linestyle='--', alpha=0.7, linewidth=2)

    plt.tight_layout()
    plt.savefig('analysis/burstgpt_analysis/03_hourly_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 图4: Token相关性
    plt.figure(figsize=(10, 6))
    plt.scatter(req_tokens[::100], resp_tokens[::100], alpha=0.6, s=20)  # 采样以避免过密
    plt.xlabel('Request Token Count')
    plt.ylabel('Response Token Count')
    plt.title('Request vs Response Token Correlation')

    # 添加相关系数
    corr = np.corrcoef(req_tokens, resp_tokens)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/burstgpt_analysis/04_token_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Basic figures have been generated to analysis/burstgpt_analysis/ directory")

def main():
    """主函数"""
    # 数据路径
    data_path = '/mnt/f/LLMServingSim/dataset/BurstGPT_1.csv'

    # 加载数据
    df = load_and_preprocess_data(data_path)

    # 分析基本统计
    time_diffs, req_tokens, resp_tokens = analyze_basic_statistics(df)

    # 创建基本图表
    create_basic_figures(df, time_diffs, req_tokens, resp_tokens)

    # 打印总结
    print("\n=== Analysis Summary ===")
    print(f"1. Burstiness: CV={np.std(time_diffs)/np.mean(time_diffs):.3f}")
    print(f"2. Token long-tail: Request 99th percentile={np.percentile(req_tokens, 99):.0f}, Response 99th percentile={np.percentile(resp_tokens, 99):.0f}")
    print(f"3. Token correlation: {np.corrcoef(req_tokens, resp_tokens)[0, 1]:.3f}")

    # 计算峰谷比
    hours = (df['timestamp_sec'] // 3600) % 24
    hourly_stats = df.groupby(hours).size()
    peak_hours = hourly_stats.nlargest(3).index.tolist()
    valley_hours = hourly_stats.nsmallest(3).index.tolist()
    print(f"4. Temporal patterns: Peak hours={peak_hours}, Valley hours={valley_hours}, Peak-valley ratio={hourly_stats.max()/hourly_stats.min():.2f}")

if __name__ == "__main__":
    main()