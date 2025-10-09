#!/usr/bin/env python3
"""
BurstGPT数据分析 - 为论文Section 2提供实证分析
分析突发性、并发性、长尾性等特征，为预测和调度模块提供依据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置论文级图表样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class BurstGPTAnalyzer:
    def __init__(self, data_path):
        """初始化分析器"""
        self.data_path = data_path
        self.df = None
        self.load_data()

    def load_data(self):
        """加载并预处理数据"""
        print(f"Loading data: {self.data_path}")
        self.df = pd.read_csv(self.data_path)

        # 转换时间戳为秒（BurstGPT的时间戳已经是秒单位）
        self.df['timestamp_sec'] = self.df['Timestamp']

        # 计算时间间隔
        self.df = self.df.sort_values('timestamp_sec')
        self.df['time_diff'] = self.df['timestamp_sec'].diff().fillna(0)

        print(f"Data loading completed: {len(self.df)} records")
        print(f"Time range: {self.df['timestamp_sec'].min():.1f} - {self.df['timestamp_sec'].max():.1f} seconds")
        print(f"Time span: {(self.df['timestamp_sec'].max() - self.df['timestamp_sec'].min())/3600:.2f} hours")

    def analyze_burstiness(self):
        """分析突发性特征"""
        print("\n=== Burstiness Analysis ===")

        # 1. 请求到达时间间隔分析
        time_diffs = self.df['time_diff'].values[1:]  # 去除第一个0

        # 基本统计
        print(f"Time interval statistics:")
        print(f"  Mean: {np.mean(time_diffs):.3f} seconds")
        print(f"  Median: {np.median(time_diffs):.3f} seconds")
        print(f"  Std deviation: {np.std(time_diffs):.3f} seconds")
        print(f"  Min: {np.min(time_diffs):.3f} seconds")
        print(f"  Max: {np.max(time_diffs):.3f} seconds")

        # 突发性系数 (变异系数)
        cv = np.std(time_diffs) / np.mean(time_diffs)
        print(f"  Coefficient of variation (burstiness metric): {cv:.3f}")

        # 2. 突发检测 - 基于滑动窗口
        window_size = 60  # 60秒窗口
        self.df['time_window'] = (self.df['timestamp_sec'] // window_size) * window_size
        burst_stats = self.df.groupby('time_window').size()

        print(f"\nBurst statistics (per {window_size} seconds):")
        print(f"  Average requests: {burst_stats.mean():.2f}")
        print(f"  Max requests: {burst_stats.max():.2f}")
        print(f"  Burst intensity ratio (max/avg): {burst_stats.max()/burst_stats.mean():.2f}")

        return {
            'time_diffs': time_diffs,
            'cv': cv,
            'burst_stats': burst_stats,
            'window_size': window_size
        }

    def analyze_token_distribution(self):
        """分析Token分布的长尾性"""
        print("\n=== Token Long-tail Analysis ===")

        # Request Token分析
        req_tokens = self.df['Request tokens'].values
        resp_tokens = self.df['Response tokens'].values

        print(f"Request Token statistics:")
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

        return {
            'req_tokens': req_tokens,
            'resp_tokens': resp_tokens
        }

    def analyze_concurrency(self):
        """分析并发性模式"""
        print("\n=== Concurrency Analysis ===")

        # 假设每个请求的处理时间与response tokens成正比
        # 保守估计: 100 tokens/秒的处理速度
        processing_rate = 100  # tokens per second

        # 计算每个请求的处理时间
        processing_times = self.df['Response tokens'] / processing_rate

        # 模拟并发情况
        timeline = []
        for _, row in self.df.iterrows():
            start_time = row['timestamp_sec']
            end_time = start_time + processing_times.iloc[_]
            timeline.append((start_time, end_time))

        # 统计并发数
        time_points = np.arange(0, self.df['timestamp_sec'].max(), 1.0)
        concurrency = []

        for t in time_points:
            concurrent = sum(1 for start, end in timeline if start <= t < end)
            concurrency.append(concurrent)

        concurrency = np.array(concurrency)

        print(f"Concurrency statistics:")
        print(f"  Average concurrency: {np.mean(concurrency):.2f}")
        print(f"  Max concurrency: {np.max(concurrency):.2f}")
        print(f"  95th percentile concurrency: {np.percentile(concurrency, 95):.2f}")

        # 找到高并发时段
        high_concurrency_threshold = np.percentile(concurrency, 95)
        high_concurrency_periods = time_points[concurrency >= high_concurrency_threshold]

        return {
            'concurrency': concurrency,
            'time_points': time_points,
            'high_concurrency_threshold': high_concurrency_threshold,
            'high_concurrency_periods': high_concurrency_periods
        }

    def analyze_temporal_patterns(self):
        """分析时间周期性和峰谷性"""
        print("\n=== Temporal Pattern Analysis ===")

        # 按小时统计
        hours = (self.df['timestamp_sec'] // 3600) % 24
        hourly_stats = self.df.groupby(hours).size()

        print(f"Hourly distribution:")
        peak_hours = hourly_stats.nlargest(3).index.tolist()
        valley_hours = hourly_stats.nsmallest(3).index.tolist()
        print(f"  Peak hours: {peak_hours}")
        print(f"  Valley hours: {valley_hours}")
        print(f"  Peak-valley ratio: {hourly_stats.max()/hourly_stats.min():.2f}")

        # 检测周期性
        from scipy.fft import fft, fftfreq

        # 创建时间序列信号
        total_time = int(self.df['timestamp_sec'].max())
        time_series = np.zeros(total_time)

        for _, row in self.df.iterrows():
            time_idx = int(row['timestamp_sec'])
            if time_idx < total_time:
                time_series[time_idx] += 1

        # FFT分析
        fft_values = fft(time_series)
        frequencies = fftfreq(len(time_series), 1.0)  # 频率 (Hz)

        # 找到主要频率成分
        power_spectrum = np.abs(fft_values)**2
        positive_freq_idx = frequencies > 0

        # 找到最强的几个频率
        top_powers = np.argsort(power_spectrum[positive_freq_idx])[-5:][::-1]
        top_frequencies = frequencies[positive_freq_idx][top_powers]
        top_periods = 1.0 / top_frequencies  # 周期（秒）

        print(f"\nMain periodic components:")
        for i, (freq, period) in enumerate(zip(top_frequencies, top_periods)):
            if period < 86400:  # 只显示小于一天的周期
                print(f"  Period{i+1}: {period/3600:.2f} hours (frequency: {freq*3600:.3f} /hour)")

        return {
            'hourly_stats': hourly_stats,
            'peak_hours': peak_hours,
            'valley_hours': valley_hours,
            'time_series': time_series,
            'top_periods': top_periods[top_periods < 86400]  # 只返回小于一天的周期
        }

    def create_paper_figures(self, burst_results, token_results, concurrency_results, temporal_results):
        """创建论文级图表"""
        print("\n=== Generating Paper Figures ===")

        # 创建输出目录
        import os
        os.makedirs('analysis/burstgpt_analysis', exist_ok=True)

        # 图1: 突发性分析
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1.1 时间间隔分布
        time_diffs = burst_results['time_diffs']
        ax1.hist(time_diffs[time_diffs < np.percentile(time_diffs, 99)], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Inter-arrival Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Request Inter-arrival Time Distribution')
        ax1.set_yscale('log')

        # 1.2 突发模式时间序列
        burst_stats = burst_results['burst_stats']
        ax2.plot(burst_stats.index, burst_stats.values, 'b-', linewidth=1.5)
        ax2.axhline(y=burst_stats.mean(), color='r', linestyle='--', label=f'Mean: {burst_stats.mean():.1f}')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Requests per minute')
        ax2.set_title('Burst Pattern Time Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 1.3 时间间隔CDF
        sorted_diffs = np.sort(time_diffs)
        cdf = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
        ax3.plot(sorted_diffs, cdf, 'b-', linewidth=2)
        ax3.set_xlabel('Inter-arrival Time (seconds)')
        ax3.set_ylabel('Cumulative Distribution Function (CDF)')
        ax3.set_title('Request Inter-arrival CDF')
        ax3.grid(True, alpha=0.3)

        # 1.4 突发强度分布
        ax4.hist(burst_stats.values, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('Requests per minute')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Burst Intensity Distribution')
        ax4.axvline(x=burst_stats.mean(), color='r', linestyle='--', label=f'Mean: {burst_stats.mean():.1f}')
        ax4.legend()

        plt.tight_layout()
        plt.savefig('analysis/burstgpt_analysis/01_burstiness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 图2: Token长尾性分析
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 2.1 Request Token分布
        req_tokens = token_results['req_tokens']
        ax1.hist(req_tokens[req_tokens < np.percentile(req_tokens, 99)], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax1.set_xlabel('Request Token Count')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Request Token Distribution')
        ax1.set_yscale('log')

        # 2.2 Response Token分布
        resp_tokens = token_results['resp_tokens']
        ax2.hist(resp_tokens[resp_tokens < np.percentile(resp_tokens, 99)], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Response Token Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Response Token Distribution')
        ax2.set_yscale('log')

        # 2.3 Token CDF比较
        sorted_req = np.sort(req_tokens)
        sorted_resp = np.sort(resp_tokens)
        cdf_req = np.arange(1, len(sorted_req) + 1) / len(sorted_req)
        cdf_resp = np.arange(1, len(sorted_resp) + 1) / len(sorted_resp)

        ax3.plot(sorted_req, cdf_req, 'b-', linewidth=2, label='Request Tokens')
        ax3.plot(sorted_resp, cdf_resp, 'r-', linewidth=2, label='Response Tokens')
        ax3.set_xlabel('Token Count')
        ax3.set_ylabel('Cumulative Distribution Function (CDF)')
        ax3.set_title('Token Long-tail CDF Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 2.4 Token相关性
        ax4.scatter(req_tokens[::100], resp_tokens[::100], alpha=0.6, s=20)  # 采样以避免过密
        ax4.set_xlabel('Request Token Count')
        ax4.set_ylabel('Response Token Count')
        ax4.set_title('Request vs Response Token Correlation')

        # 添加相关系数
        corr = np.corrcoef(req_tokens, resp_tokens)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig('analysis/burstgpt_analysis/02_token_longtail_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 图3: 并发性分析
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        concurrency = concurrency_results['concurrency']
        time_points = concurrency_results['time_points']

        # 3.1 并发时间序列
        ax1.plot(time_points/3600, concurrency, 'b-', linewidth=1)
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Concurrent Requests')
        ax1.set_title('Concurrent Requests Time Series')
        ax1.grid(True, alpha=0.3)

        # 3.2 并发数分布
        ax2.hist(concurrency, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax2.set_xlabel('Concurrent Requests')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Concurrency Distribution')
        ax2.set_yscale('log')

        # 3.3 高并发时段分析
        high_concurrency_threshold = concurrency_results['high_concurrency_threshold']
        high_concurrency_periods = concurrency_results['high_concurrency_periods']

        ax3.hist(high_concurrency_periods/3600, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('High Concurrency Period Frequency')
        ax3.set_title(f'High Concurrency Periods Distribution (>={high_concurrency_threshold:.0f} concurrent)')

        # 3.4 并发统计
        concurrency_percentiles = [50, 75, 90, 95, 99]
        percentiles_values = [np.percentile(concurrency, p) for p in concurrency_percentiles]

        bars = ax4.bar(range(len(concurrency_percentiles)), percentiles_values, color='steelblue', alpha=0.7)
        ax4.set_xlabel('Percentile')
        ax4.set_ylabel('Concurrent Requests')
        ax4.set_title('Concurrency Percentile Distribution')
        ax4.set_xticks(range(len(concurrency_percentiles)))
        ax4.set_xticklabels([f'{p}%' for p in concurrency_percentiles])

        # 添加数值标签
        for bar, value in zip(bars, percentiles_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.0f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('analysis/burstgpt_analysis/03_concurrency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 图4: 时间模式分析
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        hourly_stats = temporal_results['hourly_stats']

        # 4.1 按小时分布
        ax1.bar(hourly_stats.index, hourly_stats.values, color='lightblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Request Count')
        ax1.set_title('24-hour Request Distribution Pattern')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3)

        # 标记峰谷
        peak_hours = temporal_results['peak_hours']
        valley_hours = temporal_results['valley_hours']

        for hour in peak_hours:
            ax1.axvline(x=hour, color='red', linestyle='--', alpha=0.7, linewidth=2)
        for hour in valley_hours:
            ax1.axvline(x=hour, color='green', linestyle='--', alpha=0.7, linewidth=2)

        # 4.2 时间序列热力图
        # 创建小时-分钟热力图
        heatmap_data = np.zeros((24, 60))  # 24小时 x 60分钟

        for _, row in self.df.iterrows():
            hour = int(row['timestamp_sec'] // 3600) % 24
            minute = int((row['timestamp_sec'] % 3600) // 60)
            if 0 <= hour < 24 and 0 <= minute < 60:
                heatmap_data[hour, minute] += 1

        im = ax2.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', origin='lower')
        ax2.set_xlabel('Minute')
        ax2.set_ylabel('Hour')
        ax2.set_title('Request Time Heatmap (Hour-Minute)')
        plt.colorbar(im, ax=ax2, label='Request Count')

        # 4.3 周期性分析
        time_series = temporal_results['time_series']

        # 显示部分时间序列
        sample_length = min(len(time_series), 3600*6)  # 最多显示6小时
        ax3.plot(range(sample_length)//3600, time_series[:sample_length], 'b-', linewidth=1)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Request Count')
        ax3.set_title('Request Time Series Sample')
        ax3.grid(True, alpha=0.3)

        # 4.4 峰谷统计
        peak_valley_stats = []
        hours = list(range(24))
        for hour in hours:
            count = hourly_stats.get(hour, 0)
            peak_valley_stats.append(count)

        # 按值排序
        sorted_hours = sorted(hours, key=lambda h: peak_valley_stats[h], reverse=True)
        sorted_counts = [peak_valley_stats[h] for h in sorted_hours]

        ax4.bar(range(24), sorted_counts, color='lightgreen', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Sorted Hours')
        ax4.set_ylabel('Request Count')
        ax4.set_title('Hourly Request Volume Sorting (Peak-Valley Comparison)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis/burstgpt_analysis/04_temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Paper-quality figures have been generated to analysis/burstgpt_analysis/ directory")

def main():
    """主函数"""
    # 数据路径
    data_path = '/mnt/f/LLMServingSim/dataset/BurstGPT_1.csv'

    # 创建分析器
    analyzer = BurstGPTAnalyzer(data_path)

    # 执行分析
    burst_results = analyzer.analyze_burstiness()
    token_results = analyzer.analyze_token_distribution()
    concurrency_results = analyzer.analyze_concurrency()
    temporal_results = analyzer.analyze_temporal_patterns()

    # 生成论文图表
    analyzer.create_paper_figures(burst_results, token_results, concurrency_results, temporal_results)

    # 打印总结
    print("\n=== Analysis Summary ===")
    print(f"1. Burstiness: CV={burst_results['cv']:.3f}, burst intensity ratio={burst_results['burst_stats'].max()/burst_results['burst_stats'].mean():.2f}")
    print(f"2. Token long-tail: Request 99th percentile={np.percentile(token_results['req_tokens'], 99):.0f}, Response 99th percentile={np.percentile(token_results['resp_tokens'], 99):.0f}")
    print(f"3. Concurrency: Max concurrent={np.max(concurrency_results['concurrency']):.0f}, 95th percentile={np.percentile(concurrency_results['concurrency'], 95):.0f}")
    print(f"4. Temporal patterns: Peak hours={temporal_results['peak_hours']}, peak-valley ratio={temporal_results['hourly_stats'].max()/temporal_results['hourly_stats'].min():.2f}")

if __name__ == "__main__":
    main()