"""
并发Batch推理时间窗口配置

基于vLLM、TGI、NVIDIA Triton等生产系统实践，实现真实的连续批处理策略。

生产系统关键配置:
- vLLM: max_num_batched_tokens=8000, max_num_seqs=256
- TGI: max-batch-total-tokens=8192, max_queue_delay_microseconds=100
- Triton: dynamic_batching with max_queue_delay_microseconds=100

核心策略: Continuous Batching (连续批处理)而非固定时间窗口
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum


class BatchStrategy(Enum):
    """批处理策略枚举"""
    REAL_TIME = "real_time"          # 实时处理，最低延迟
    STANDARD = "standard"            # 标准批处理，平衡延迟和吞吐量
    LARGE_BATCH = "large_batch"      # 大规模批处理，优化吞吐量
    PERIODIC = "periodic"            # 周期性调度，资源优化
    ADAPTIVE = "adaptive"            # 自适应，根据负载动态调整


@dataclass
class ContinuousBatchConfig:
    """连续批处理配置 - 基于生产系统实践"""
    max_num_batched_tokens: int       # 单个batch最大token数 (参考vLLM)
    max_num_seqs: int                 # 最大并发序列数 (参考vLLM)
    max_queue_delay_microseconds: int # 最大队列延迟微秒 (参考TGI)
    max_prefill_tokens: int          # 预填充阶段最大token数 (参考TGI)
    max_total_tokens: int            # 单个请求最大token数
    priority_threshold: float        # 优先级阈值
    description: str                 # 描述

    def get_optimal_batch_size(self, avg_tokens_per_request: int) -> int:
        """根据平均token数获取最优batch大小"""
        # 基于token限制计算batch大小
        token_limited_batch = self.max_num_batched_tokens // max(avg_tokens_per_request, 1)
        # 取并发限制和token限制的最小值
        return min(self.max_num_seqs, token_limited_batch)


class ContinuousBatchManager:
    """连续批处理管理器 - 基于生产系统实践"""

    def __init__(self):
        # 基于生产系统实践的配置
        self.configs: Dict[BatchStrategy, ContinuousBatchConfig] = {
            BatchStrategy.REAL_TIME: ContinuousBatchConfig(
                max_num_batched_tokens=2048,    # 2K tokens (低延迟)
                max_num_seqs=16,               # 低并发
                max_queue_delay_microseconds=50,   # 50微秒延迟
                max_prefill_tokens=512,         # 预填充限制
                max_total_tokens=4096,          # 单请求最大token数
                priority_threshold=0.8,         # 高优先级
                description="实时处理，基于vLLM低延迟配置"
            ),

            BatchStrategy.STANDARD: ContinuousBatchConfig(
                max_num_batched_tokens=8192,    # 8K tokens (标准配置)
                max_num_seqs=64,               # 中等并发
                max_queue_delay_microseconds=100,  # 100微秒延迟 (TGI标准)
                max_prefill_tokens=2048,        # 预填充限制
                max_total_tokens=8192,          # 单请求最大token数
                priority_threshold=0.6,         # 中高优先级
                description="标准批处理，基于TGI/vLLM生产配置"
            ),

            BatchStrategy.LARGE_BATCH: ContinuousBatchConfig(
                max_num_batched_tokens=16384,   # 16K tokens (大批量)
                max_num_seqs=128,              # 高并发
                max_queue_delay_microseconds=200,  # 200微秒延迟
                max_prefill_tokens=4096,        # 预填充限制
                max_total_tokens=16384,         # 单请求最大token数
                priority_threshold=0.4,         # 中等优先级
                description="大规模批处理，高吞吐量配置"
            ),

            BatchStrategy.PERIODIC: ContinuousBatchConfig(
                max_num_batched_tokens=32768,   # 32K tokens (周期性大批量)
                max_num_seqs=256,              # 最高并发
                max_queue_delay_microseconds=500,  # 500微秒延迟
                max_prefill_tokens=8192,        # 预填充限制
                max_total_tokens=32768,         # 单请求最大token数
                priority_threshold=0.2,         # 低优先级
                description="周期性调度，最大化吞吐量"
            ),

            BatchStrategy.ADAPTIVE: ContinuousBatchConfig(
                max_num_batched_tokens=8192,    # 动态调整token限制
                max_num_seqs=64,               # 动态并发限制
                max_queue_delay_microseconds=100,  # 动态延迟调整
                max_prefill_tokens=2048,        # 动态预填充
                max_total_tokens=8192,          # 动态单请求限制
                priority_threshold=0.5,         # 动态阈值
                description="自适应调整，基于负载动态优化"
            )
        }

        self.current_strategy = BatchStrategy.STANDARD
        self.adaptive_params = {
            'load_threshold_high': 0.8,        # 高负载阈值
            'load_threshold_low': 0.3,         # 低负载阈值
            'token_scaling_factor': 2.0,       # token限制缩放因子
            'max_batch_tokens': 65536,        # 最大batch tokens
            'min_batch_tokens': 1024,          # 最小batch tokens
            'scaling_step_size': 1024          # 缩放步长
        }

    def get_config(self, strategy: BatchStrategy = None) -> ContinuousBatchConfig:
        """获取指定策略的配置"""
        if strategy is None:
            strategy = self.current_strategy
        return self.configs[strategy]

    def select_strategy(self,
                       current_load: float,
                       avg_response_time: float,
                       queue_length: int) -> BatchStrategy:
        """根据当前系统状态选择最优策略"""

        if self.current_strategy == BatchStrategy.ADAPTIVE:
            return self._adaptive_strategy_selection(current_load, avg_response_time, queue_length)

        # 静态策略选择逻辑
        if current_load > 0.8 or queue_length > 100:
            return BatchStrategy.PERIODIC     # 高负载时使用周期性调度
        elif current_load > 0.6 or avg_response_time > 0.5:
            return BatchStrategy.LARGE_BATCH   # 中高负载时使用大batch
        elif current_load > 0.3 or avg_response_time > 0.1:
            return BatchStrategy.STANDARD      # 中等负载时使用标准batch
        else:
            return BatchStrategy.REAL_TIME     # 低负载时使用实时处理

    def _adaptive_strategy_selection(self,
                                   current_load: float,
                                   avg_response_time: float,
                                   queue_length: int) -> BatchStrategy:
        """自适应策略选择"""
        config = self.configs[BatchStrategy.ADAPTIVE]

        # 根据负载动态调整窗口大小
        if current_load > self.adaptive_params['load_threshold_high']:
            # 高负载，增大窗口
            new_window_size = min(
                config.window_size_seconds * self.adaptive_params['window_scaling_factor'],
                self.adaptive_params['max_window_size']
            )
            config.window_size_seconds = new_window_size
            return BatchStrategy.ADAPTIVE

        elif current_load < self.adaptive_params['load_threshold_low']:
            # 低负载，减小窗口
            new_window_size = max(
                config.window_size_seconds / self.adaptive_params['window_scaling_factor'],
                self.adaptive_params['min_window_size']
            )
            config.window_size_seconds = new_window_size
            return BatchStrategy.ADAPTIVE

        else:
            # 中等负载，选择固定策略
            return self.select_strategy(current_load, avg_response_time, queue_length)

    def simulate_continuous_batching(self,
                                    timestamps: List[float],
                                    input_tokens: List[int],
                                    output_tokens: List[int],
                                    strategy: BatchStrategy = None) -> Dict[str, any]:
        """模拟连续批处理行为"""
        config = self.get_config(strategy)

        if not timestamps or len(timestamps) != len(input_tokens):
            return {'error': 'Invalid input data'}

        # 按时间排序
        sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
        sorted_timestamps = [timestamps[i] for i in sorted_indices]
        sorted_input_tokens = [input_tokens[i] for i in sorted_indices]
        sorted_output_tokens = [output_tokens[i] for i in sorted_indices]

        batch_stats = []
        current_batch = []
        current_batch_tokens = 0
        batch_start_time = None

        for i, (ts, in_tok, out_tok) in enumerate(zip(sorted_timestamps, sorted_input_tokens, sorted_output_tokens)):
            total_tokens = in_tok + out_tok

            # 检查是否可以加入当前batch
            can_join_batch = (
                len(current_batch) < config.max_num_seqs and
                current_batch_tokens + total_tokens <= config.max_num_batched_tokens and
                total_tokens <= config.max_total_tokens
            )

            # 如果是第一个请求，或者可以加入当前batch
            if not current_batch:
                current_batch = [(ts, in_tok, out_tok)]
                current_batch_tokens = total_tokens
                batch_start_time = ts
            elif can_join_batch:
                current_batch.append((ts, in_tok, out_tok))
                current_batch_tokens += total_tokens
            else:
                # 完成当前batch，开始新batch
                if current_batch:
                    batch_end_time = current_batch[-1][0]
                    batch_delay = (batch_end_time - batch_start_time) * 1e6  # 转换为微秒

                    batch_stats.append({
                        'start_time': batch_start_time,
                        'end_time': batch_end_time,
                        'num_requests': len(current_batch),
                        'total_tokens': current_batch_tokens,
                        'avg_tokens_per_request': current_batch_tokens / len(current_batch),
                        'delay_microseconds': batch_delay,
                        'utilization': min(1.0, current_batch_tokens / config.max_num_batched_tokens)
                    })

                # 开始新batch
                current_batch = [(ts, in_tok, out_tok)]
                current_batch_tokens = total_tokens
                batch_start_time = ts

        # 处理最后一个batch
        if current_batch:
            batch_end_time = current_batch[-1][0]
            batch_delay = (batch_end_time - batch_start_time) * 1e6

            batch_stats.append({
                'start_time': batch_start_time,
                'end_time': batch_end_time,
                'num_requests': len(current_batch),
                'total_tokens': current_batch_tokens,
                'avg_tokens_per_request': current_batch_tokens / len(current_batch),
                'delay_microseconds': batch_delay,
                'utilization': min(1.0, current_batch_tokens / config.max_num_batched_tokens)
            })

        # 计算统计信息
        if batch_stats:
            total_requests = sum(stat['num_requests'] for stat in batch_stats)
            total_tokens = sum(stat['total_tokens'] for stat in batch_stats)
            avg_batch_size = total_requests / len(batch_stats)
            avg_utilization = sum(stat['utilization'] for stat in batch_stats) / len(batch_stats)
            max_delay = max(stat['delay_microseconds'] for stat in batch_stats)

            return {
                'total_batches': len(batch_stats),
                'total_requests_processed': total_requests,
                'total_tokens_processed': total_tokens,
                'avg_batch_size': avg_batch_size,
                'avg_utilization': avg_utilization,
                'max_delay_microseconds': max_delay,
                'batch_details': batch_stats,
                'config_used': config.description
            }
        else:
            return {'error': 'No batches created'}

    def estimate_concurrency(self,
                           timestamps: List[float],
                           strategy: BatchStrategy = None) -> Dict[str, float]:
        """估算并发度指标"""
        config = self.get_config(strategy)
        windows = self.calculate_batch_windows(timestamps, strategy)

        if not windows:
            return {'max_concurrency': 0, 'avg_concurrency': 0, 'total_windows': 0}

        concurrency_stats = []
        for window_start, window_end in windows:
            concurrent_count = sum(
                1 for ts in timestamps
                if window_start <= ts <= window_end
            )
            concurrency_stats.append(concurrent_count)

        return {
            'max_concurrency': max(concurrency_stats),
            'avg_concurrency': sum(concurrency_stats) / len(concurrency_stats),
            'total_windows': len(windows),
            'window_size': config.window_size_seconds
        }

    def get_optimal_batch_size(self,
                             input_tokens: List[int],
                             output_tokens: List[int],
                             strategy: BatchStrategy = None) -> int:
        """根据token数量获取最优批处理大小"""
        config = self.get_config(strategy)

        total_tokens = sum(input_tokens) + sum(output_tokens)
        if total_tokens == 0:
            return 1

        # 基于token限制计算batch大小
        token_limited_batch = config.max_total_tokens // max(
            (sum(input_tokens) + sum(output_tokens)) // max(len(input_tokens), 1),
            1
        )

        # 取并发限制和token限制的最小值
        optimal_batch = min(config.max_concurrent_requests, token_limited_batch)

        return max(1, optimal_batch)  # 至少为1

    def predict_next_window(self,
                           historical_data: List[float],
                           strategy: BatchStrategy = None) -> Dict[str, any]:
        """预测下一个时间窗口的特征"""
        config = self.get_config(strategy)

        if len(historical_data) < 2:
            return {
                'expected_requests': 1,
                'expected_tokens': 1000,
                'confidence': 0.5
            }

        # 简单的移动平均预测
        recent_intervals = historical_data[-10:]  # 最近10个间隔
        avg_interval = sum(recent_intervals) / len(recent_intervals)

        expected_requests = config.window_size_seconds / avg_interval
        expected_tokens = int(expected_requests * 1000)  # 假设平均1000 tokens/request

        confidence = min(0.9, len(recent_intervals) / 10.0)  # 基于数据量的置信度

        return {
            'expected_requests': expected_requests,
            'expected_tokens': expected_tokens,
            'confidence': confidence,
            'avg_interval': avg_interval
        }


# 全局实例
batch_window_manager = BatchWindowManager()


def get_batch_strategy_by_use_case(use_case: str) -> BatchStrategy:
    """根据使用场景获取推荐的批处理策略"""
    use_case_mapping = {
        'chat_application': BatchStrategy.REAL_TIME,
        'api_service': BatchStrategy.STANDARD,
        'batch_processing': BatchStrategy.LARGE_BATCH,
        'background_tasks': BatchStrategy.PERIODIC,
        'variable_load': BatchStrategy.ADAPTIVE,
        'real_time_inference': BatchStrategy.REAL_TIME,
        'offline_training': BatchStrategy.PERIODIC
    }

    return use_case_mapping.get(use_case, BatchStrategy.STANDARD)


def print_config_summary():
    """打印配置摘要"""
    print("=== 并发Batch推理时间窗口配置摘要 ===")

    for strategy, config in batch_window_manager.configs.items():
        print(f"\n{strategy.value.upper()} 策略:")
        print(f"  窗口大小: {config.window_size_seconds:.3f}s")
        print(f"  最大并发: {config.max_concurrent_requests}")
        print(f"  最大Token数: {config.max_total_tokens}")
        print(f"  优先级阈值: {config.priority_threshold}")
        print(f"  描述: {config.description}")


if __name__ == "__main__":
    print_config_summary()