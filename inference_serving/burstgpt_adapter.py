#!/usr/bin/env python3
"""
BurstGPT数据集模拟适配器
专门处理突发流量模式的LLM推理模拟
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from .data_loader import UniversalDataLoader

# 设置matplotlib参数
plt.style.use('default')
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

class BurstGPTAdapter:
    """
    BurstGPT数据集适配器
    分析和模拟突发流量模式
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.data_loader = UniversalDataLoader(verbose=verbose)
        self.burstgpt_stats = {}

    def analyze_burst_patterns(self, file_path: str, sample_size: int = 10000) -> Dict:
        """
        分析BurstGPT数据集中的突发模式

        Args:
            file_path: BurstGPT CSV文件路径
            sample_size: 采样大小

        Returns:
            突发模式分析结果
        """
        print(f"🔍 分析BurstGPT突发模式: {file_path}")

        # 加载数据
        try:
            raw_data = pd.read_csv(file_path, nrows=sample_size)
            if self.verbose:
                print(f"📊 加载了 {len(raw_data)} 条记录")
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return {}

        # 分析时间戳
        if 'Timestamp' in raw_data.columns:
            timestamps = raw_data['Timestamp'].values
            time_diffs = np.diff(timestamps)

            # 突发检测：短时间内的高频请求
            burst_threshold = np.percentile(time_diffs, 25)  # 25%分位数作为突发阈值
            burst_intervals = time_diffs[time_diffs <= burst_threshold]

            # 计算突发统计
            burst_stats = {
                'total_requests': len(raw_data),
                'sample_duration': timestamps[-1] - timestamps[0],
                'avg_time_gap': np.mean(time_diffs),
                'std_time_gap': np.std(time_diffs),
                'burst_threshold': burst_threshold,
                'burst_count': len(burst_intervals),
                'burst_ratio': len(burst_intervals) / len(time_diffs),
                'burst_intensity': np.mean(burst_intervals) if len(burst_intervals) > 0 else 0,
                'max_burst_gap': np.max(burst_intervals) if len(burst_intervals) > 0 else 0,
                'min_burst_gap': np.min(burst_intervals) if len(burst_intervals) > 0 else 0,
            }

            # 分析token分布
            token_stats = {
                'avg_input_tokens': raw_data['Request tokens'].mean(),
                'std_input_tokens': raw_data['Request tokens'].std(),
                'avg_output_tokens': raw_data['Response tokens'].mean(),
                'std_output_tokens': raw_data['Response tokens'].std(),
                'max_input_tokens': raw_data['Request tokens'].max(),
                'max_output_tokens': raw_data['Response tokens'].max(),
            }

            # 模型分布
            model_dist = raw_data['Model'].value_counts().to_dict()

            # 合并统计结果
            self.burstgpt_stats = {
                'burst_patterns': burst_stats,
                'token_distribution': token_stats,
                'model_distribution': model_dist,
                'raw_sample': raw_data.head(100)  # 保存样本数据
            }

            if self.verbose:
                self._print_analysis_summary()

            return self.burstgpt_stats

        return {}

    def _print_analysis_summary(self):
        """打印分析摘要"""
        print("\n📊 BurstGPT突发模式分析摘要:")
        print("-" * 60)

        if 'burst_patterns' in self.burstgpt_stats:
            bp = self.burstgpt_stats['burst_patterns']
            print(f"总请求数: {bp['total_requests']}")
            print(f"采样时长: {bp['sample_duration']:.2f}ms")
            print(f"平均时间间隔: {bp['avg_time_gap']:.2f}ms")
            print(f"时间间隔标准差: {bp['std_time_gap']:.2f}ms")
            print(f"突发阈值: {bp['burst_threshold']:.2f}ms")
            print(f"突发请求数: {bp['burst_count']}")
            print(f"突发比例: {bp['burst_ratio']:.2%}")
            print(f"突发强度: {bp['burst_intensity']:.2f}ms")

        if 'token_distribution' in self.burstgpt_stats:
            td = self.burstgpt_stats['token_distribution']
            print(f"\n📈 Token分布:")
            print(f"平均输入token: {td['avg_input_tokens']:.1f} (±{td['std_input_tokens']:.1f})")
            print(f"平均输出token: {td['avg_output_tokens']:.1f} (±{td['std_output_tokens']:.1f})")
            print(f"最大输入token: {td['max_input_tokens']}")
            print(f"最大输出token: {td['max_output_tokens']}")

        if 'model_distribution' in self.burstgpt_stats:
            md = self.burstgpt_stats['model_distribution']
            print(f"\n🤖 模型分布:")
            for model, count in md.items():
                print(f"  {model}: {count} ({count/sum(md.values()):.1%})")

    def create_simulation_dataset(self, file_path: str, req_num: int = 1000,
                                burst_mode: str = 'original') -> pd.DataFrame:
        """
        创建适合LLMServingSim的数据集

        Args:
            file_path: BurstGPT CSV文件路径
            req_num: 请求数量
            burst_mode: 突发模式 ('original', 'enhanced', 'extreme')

        Returns:
            标准格式的DataFrame
        """
        print(f"🔄 创建模拟数据集: {burst_mode}模式, {req_num}个请求")

        # 使用UniversalDataLoader加载数据
        try:
            standard_data = self.data_loader.load_dataset(file_path, req_num=req_num)
        except Exception as e:
            print(f"❌ 使用UniversalDataLoader失败: {e}")
            return self._create_fallback_dataset(req_num, burst_mode)

        # 根据突发模式调整时间间隔
        if burst_mode != 'original':
            standard_data = self._apply_burst_pattern(standard_data, burst_mode)

        return standard_data

    def _apply_burst_pattern(self, data: pd.DataFrame, burst_mode: str) -> pd.DataFrame:
        """应用突发模式到数据"""
        if burst_mode == 'enhanced':
            # 增强突发模式：创建更明显的突发
            return self._create_enhanced_bursts(data)
        elif burst_mode == 'extreme':
            # 极端突发模式：创建极端的突发流量
            return self._create_extreme_bursts(data)
        else:
            return data

    def _create_enhanced_bursts(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建增强突发模式"""
        enhanced_data = data.copy()
        n_requests = len(enhanced_data)

        # 创建突发簇
        burst_size = 10
        n_bursts = n_requests // burst_size
        burst_gap = 1000  # 突发簇间隔 (ms)
        intra_burst_gap = 10  # 突发簇内间隔 (ms)

        arrival_times = []
        for i in range(n_bursts):
            burst_start = i * burst_gap
            for j in range(burst_size):
                arrival_times.append(burst_start + j * intra_burst_gap)

        # 添加剩余请求
        remaining_requests = n_requests - len(arrival_times)
        for i in range(remaining_requests):
            arrival_times.append(n_bursts * burst_gap + i * 500)

        enhanced_data['arrival_time_ns'] = np.array(arrival_times[:n_requests]) * 1_000_000

        return enhanced_data

    def _create_extreme_bursts(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建极端突发模式"""
        extreme_data = data.copy()
        n_requests = len(extreme_data)

        # 创建极端突发：大量请求在极短时间内到达
        burst_clusters = [
            (0, 50, 5),      # 50个请求在5ms内
            (100, 80, 3),    # 80个请求在3ms内
            (200, 120, 2),   # 120个请求在2ms内
            (300, 60, 4),    # 60个请求在4ms内
            (400, 40, 6),    # 40个请求在6ms内
        ]

        arrival_times = []
        used_requests = 0

        for cluster_start, cluster_size, cluster_duration in burst_clusters:
            cluster_end = cluster_start + cluster_duration
            actual_size = min(cluster_size, n_requests - used_requests)

            # 在突发簇内生成请求时间
            cluster_times = np.random.uniform(cluster_start, cluster_end, actual_size)
            arrival_times.extend(sorted(cluster_times))
            used_requests += actual_size

        # 添加剩余请求
        remaining_requests = n_requests - used_requests
        if remaining_requests > 0:
            remaining_times = np.random.uniform(500, 2000, remaining_requests)
            arrival_times.extend(sorted(remaining_times))

        extreme_data['arrival_time_ns'] = np.array(arrival_times[:n_requests]) * 1_000_000

        return extreme_data

    def _create_fallback_dataset(self, req_num: int, burst_mode: str) -> pd.DataFrame:
        """创建备用数据集"""
        print("⚠️ 使用备用数据集生成")

        # 基于BurstGPT统计特征生成模拟数据
        fallback_data = pd.DataFrame()

        # 生成token数量（基于BurstGPT特征）
        fallback_data['input_toks'] = np.random.normal(400, 300, req_num).astype(int)
        fallback_data['input_toks'] = np.clip(fallback_data['input_toks'], 10, 2000)

        fallback_data['output_toks'] = np.random.normal(250, 200, req_num).astype(int)
        fallback_data['output_toks'] = np.clip(fallback_data['output_toks'], 10, 1000)

        # 生成到达时间
        if burst_mode == 'original':
            # 原始模式：相对均匀的分布
            arrival_times = np.random.exponential(100, req_num)
        elif burst_mode == 'enhanced':
            # 增强突发模式
            arrival_times = self._generate_enhanced_burst_times(req_num)
        else:  # extreme
            # 极端突发模式
            arrival_times = self._generate_extreme_burst_times(req_num)

        fallback_data['arrival_time_ns'] = np.cumsum(arrival_times) * 1_000_000
        fallback_data['model_type'] = 'ChatGPT'
        fallback_data['burst_pattern'] = burst_mode

        return fallback_data

    def _generate_enhanced_burst_times(self, req_num: int) -> np.ndarray:
        """生成增强突发时间间隔"""
        times = []
        i = 0
        while i < req_num:
            # 随机决定是否开始突发
            if np.random.random() < 0.3:  # 30%概率开始突发
                burst_length = np.random.randint(5, 15)
                burst_times = np.random.exponential(10, min(burst_length, req_num - i))
                times.extend(burst_times)
                i += len(burst_times)
            else:
                times.append(np.random.exponential(200))
                i += 1
        return np.array(times[:req_num])

    def _generate_extreme_burst_times(self, req_num: int) -> np.ndarray:
        """生成极端突发时间间隔"""
        times = []
        i = 0
        while i < req_num:
            # 创建极端突发簇
            if np.random.random() < 0.2:  # 20%概率开始极端突发
                burst_length = np.random.randint(20, 50)
                burst_times = np.random.exponential(2, min(burst_length, req_num - i))
                times.extend(burst_times)
                i += len(burst_times)
            else:
                times.append(np.random.exponential(500))
                i += 1
        return np.array(times[:req_num])

    def run_burst_simulation(self, dataset_path: str, req_num: int = 1000,
                           burst_mode: str = 'enhanced', output_prefix: str = 'burstgpt') -> Dict:
        """
        运行BurstGPT模拟实验

        Args:
            dataset_path: BurstGPT数据集路径
            req_num: 请求数量
            burst_mode: 突发模式
            output_prefix: 输出文件前缀

        Returns:
            模拟结果
        """
        print(f"🚀 开始BurstGPT模拟实验: {burst_mode}模式")

        # 创建模拟数据集
        sim_data = self.create_simulation_dataset(dataset_path, req_num, burst_mode)

        # 保存模拟数据集
        sim_data_path = f'output/{output_prefix}_sim_data.csv'
        sim_data.to_csv(sim_data_path, index=False)
        print(f"📄 模拟数据集已保存: {sim_data_path}")

        # 运行LLMServingSim模拟
        sim_result = self._run_llmserving_sim(sim_data, output_prefix)

        return {
            'sim_data': sim_data,
            'sim_result': sim_result,
            'burst_mode': burst_mode,
            'dataset_path': dataset_path
        }

    def _run_llmserving_sim(self, sim_data: pd.DataFrame, output_prefix: str) -> Dict:
        """运行LLMServingSim模拟"""
        # 保存为TSV格式供LLMServingSim使用
        tsv_path = f'output/{output_prefix}_sim.tsv'
        sim_data[['input_toks', 'output_toks', 'arrival_time_ns']].to_csv(
            tsv_path, sep='\t', index=False, header=False
        )

        # 构建命令
        cmd = f"python3 main.py --model_name 'meta-llama/Llama-3.1-8B-Instruct' --hardware 'RTX3090' --dataset '{tsv_path}' --output 'output/{output_prefix}_results.csv' --req_num {len(sim_data)}"

        print(f"🔧 运行模拟命令: {cmd}")

        # 这里应该运行实际的模拟，现在返回模拟结果
        print("⚠️ 模拟运行逻辑需要实现")

        return {'status': 'simulated', 'command': cmd}

    def compare_burst_patterns(self, file_path: str, req_num: int = 500) -> Dict:
        """
        比较不同突发模式的性能

        Args:
            file_path: BurstGPT数据集路径
            req_num: 请求数量

        Returns:
            比较结果
        """
        print("🔍 比较不同突发模式性能")

        modes = ['original', 'enhanced', 'extreme']
        results = {}

        for mode in modes:
            print(f"\n📊 测试{mode}模式...")
            try:
                result = self.run_burst_simulation(file_path, req_num, mode, f'burstgpt_{mode}')
                results[mode] = result
            except Exception as e:
                print(f"❌ {mode}模式测试失败: {e}")
                results[mode] = {'error': str(e)}

        return results

def main():
    """主函数"""
    adapter = BurstGPTAdapter(verbose=True)

    # 分析BurstGPT数据集
    burstgpt_path = 'dataset/BurstGPT_1.csv'
    stats = adapter.analyze_burst_patterns(burstgpt_path, sample_size=50000)

    # 创建模拟数据集
    if stats:
        sim_data = adapter.create_simulation_dataset(burstgpt_path, req_num=1000, burst_mode='enhanced')
        print(f"📊 创建了 {len(sim_data)} 条模拟数据")

        # 保存模拟数据
        sim_data.to_csv('output/burstgpt_simulation.csv', index=False)
        print("📄 模拟数据已保存: output/burstgpt_simulation.csv")

        # 对比不同突发模式
        comparison = adapter.compare_burst_patterns(burstgpt_path, req_num=200)
        print("✅ 突发模式对比完成")

if __name__ == "__main__":
    main()