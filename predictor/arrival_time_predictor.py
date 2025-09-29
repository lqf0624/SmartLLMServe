"""
请求到达时间预测器

基于历史请求数据预测下一个请求的到达时间，支持多种预测方法：
1. 统计方法：基于历史间隔的移动平均
2. 时间序列：使用LSTM/DLinear预测间隔
3. 泊松过程：基于到达率的随机过程
4. 自适应方法：根据数据特征自动选择最佳方法
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from .lstm_model import LSTMPredictor
from .dlinear_predictor_adapter import DLinearPredictorAdapter

logger = logging.getLogger(__name__)


class PredictionMethod(Enum):
    """预测方法枚举"""
    STATISTICAL = "statistical"          # 统计方法
    TIME_SERIES = "time_series"         # 时间序列
    POISSON_PROCESS = "poisson_process" # 泊松过程
    ADAPTIVE = "adaptive"               # 自适应方法


class ArrivalTimePredictor:
    """请求到达时间预测器"""

    def __init__(self,
                 prediction_method: PredictionMethod = PredictionMethod.ADAPTIVE,
                 window_size: int = 100,
                 confidence_level: float = 0.95):
        """
        初始化预测器

        Args:
            prediction_method: 预测方法
            window_size: 历史窗口大小
            confidence_level: 置信度
        """
        self.prediction_method = prediction_method
        self.window_size = window_size
        self.confidence_level = confidence_level

        # 历史数据
        self.arrival_intervals: List[float] = []
        self.arrival_timestamps: List[float] = []

        # 时间序列模型
        self.lstm_predictor = None
        self.dlinear_predictor = None
        self.scaler = StandardScaler()

        # 统计参数
        self.alpha = 0.1  # 指数平滑参数
        self.last_prediction = None
        self.prediction_errors = []

        logger.info(f"ArrivalTimePredictor initialized with method: {prediction_method.value}")

    def update_history(self, timestamps: List[float]) -> None:
        """
        更新历史数据

        Args:
            timestamps: 新的请求时间戳列表
        """
        if not timestamps:
            return

        # 确保时间戳是有序的
        sorted_timestamps = sorted(timestamps)

        # 如果是第一次更新，直接添加
        if not self.arrival_timestamps:
            self.arrival_timestamps.extend(sorted_timestamps)
            # 计算间隔
            for i in range(1, len(self.arrival_timestamps)):
                interval = self.arrival_timestamps[i] - self.arrival_timestamps[i-1]
                self.arrival_intervals.append(interval)
        else:
            # 找到最后一个时间戳
            last_timestamp = self.arrival_timestamps[-1]

            # 只添加比最后一个时间戳新的数据
            new_timestamps = [ts for ts in sorted_timestamps if ts > last_timestamp]

            if new_timestamps:
                # 添加从最后一个时间戳到第一个新时间戳的间隔
                first_interval = new_timestamps[0] - last_timestamp
                self.arrival_intervals.append(first_interval)

                # 添加新的时间戳和间隔
                self.arrival_timestamps.extend(new_timestamps)
                for i in range(1, len(new_timestamps)):
                    interval = new_timestamps[i] - new_timestamps[i-1]
                    self.arrival_intervals.append(interval)

        # 限制历史数据大小
        if len(self.arrival_intervals) > self.window_size * 2:
            self.arrival_intervals = self.arrival_intervals[-self.window_size:]
            self.arrival_timestamps = self.arrival_timestamps[-(self.window_size + 1):]

        logger.debug(f"Updated history: {len(self.arrival_intervals)} intervals, "
                    f"last timestamp: {self.arrival_timestamps[-1] if self.arrival_timestamps else 'None'}")

    def predict_next_arrival(self, current_time: float = None) -> Dict[str, Any]:
        """
        预测下一个请求的到达时间

        Args:
            current_time: 当前时间，如果为None则使用最后一个历史时间戳

        Returns:
            预测结果字典
        """
        if len(self.arrival_intervals) < 5:
            # 历史数据不足，使用默认预测
            return self._default_prediction()

        if current_time is None:
            current_time = self.arrival_timestamps[-1] if self.arrival_timestamps else 0

        # 根据预测方法选择预测算法
        if self.prediction_method == PredictionMethod.STATISTICAL:
            return self._statistical_prediction(current_time)
        elif self.prediction_method == PredictionMethod.TIME_SERIES:
            return self._time_series_prediction(current_time)
        elif self.prediction_method == PredictionMethod.POISSON_PROCESS:
            return self._poisson_process_prediction(current_time)
        elif self.prediction_method == PredictionMethod.ADAPTIVE:
            return self._adaptive_prediction(current_time)
        else:
            return self._default_prediction()

    def _default_prediction(self) -> Dict[str, Any]:
        """默认预测"""
        return {
            'predicted_interval': 1.0,  # 默认1秒
            'predicted_arrival_time': None,
            'confidence': 0.5,
            'method': 'default',
            'uncertainty_range': (0.1, 10.0),
            'prediction_metadata': {
                'reason': 'Insufficient historical data',
                'history_size': len(self.arrival_intervals)
            }
        }

    def _statistical_prediction(self, current_time: float) -> Dict[str, Any]:
        """统计方法预测"""
        recent_intervals = self.arrival_intervals[-min(self.window_size, len(self.arrival_intervals)):]

        # 计算统计量
        mean_interval = np.mean(recent_intervals)
        median_interval = np.median(recent_intervals)
        std_interval = np.std(recent_intervals)

        # 指数加权移动平均
        if self.last_prediction:
            ewma_interval = self.alpha * mean_interval + (1 - self.alpha) * self.last_prediction
        else:
            ewma_interval = mean_interval

        # 根据分布特征选择预测值
        if std_interval / mean_interval > 1.0:  # 高变异性
            # 使用中位数更稳健
            predicted_interval = median_interval
            method_used = 'statistical_median'
        else:
            # 使用指数加权移动平均
            predicted_interval = ewma_interval
            method_used = 'statistical_ewma'

        # 计算置信区间
        if len(recent_intervals) >= 10:
            # 使用t分布计算置信区间
            confidence_interval = stats.t.interval(
                self.confidence_level,
                len(recent_intervals) - 1,
                loc=predicted_interval,
                scale=std_interval / np.sqrt(len(recent_intervals))
            )
        else:
            # 使用正态分布近似
            confidence_interval = stats.norm.interval(
                self.confidence_level,
                loc=predicted_interval,
                scale=std_interval
            )

        predicted_arrival_time = current_time + predicted_interval

        # 更新最后一次预测
        self.last_prediction = predicted_interval

        return {
            'predicted_interval': predicted_interval,
            'predicted_arrival_time': predicted_arrival_time,
            'confidence': self.confidence_level,
            'method': method_used,
            'uncertainty_range': confidence_interval,
            'prediction_metadata': {
                'mean_interval': mean_interval,
                'median_interval': median_interval,
                'std_interval': std_interval,
                'coefficient_of_variation': std_interval / mean_interval,
                'history_size': len(recent_intervals)
            }
        }

    def _time_series_prediction(self, current_time: float) -> Dict[str, Any]:
        """时间序列预测"""
        if len(self.arrival_intervals) < 20:
            # 数据不足，回退到统计方法
            return self._statistical_prediction(current_time)

        try:
            # 准备时间序列数据
            intervals_array = np.array(self.arrival_intervals[-self.window_size:])

            # 使用DLinear预测（更轻量级）
            if self.dlinear_predictor is None:
                self.dlinear_predictor = DLinearPredictorAdapter()

            # 创建模拟的时间序列数据
            mock_data = pd.DataFrame({
                'arrival_time_ns': np.arange(len(intervals_array)) * 1e9,  # 模拟时间戳
                'input_toks': np.full(len(intervals_array), 100),         # 模拟输入tokens
                'output_toks': np.full(len(intervals_array), 300),        # 模拟输出tokens
            })

            # 训练模型
            self.dlinear_predictor.train(mock_data)

            # 预测下一个间隔
            prediction_result = self.dlinear_predictor.predict_future(mock_data, steps=1)

            if hasattr(prediction_result, 'predictions') and len(prediction_result.predictions) > 0:
                predicted_interval = abs(prediction_result.predictions[0])  # 确保为正数
                predicted_interval = max(0.001, predicted_interval)  # 最小间隔
            else:
                # 回退到统计方法
                return self._statistical_prediction(current_time)

            predicted_arrival_time = current_time + predicted_interval

            # 计算置信区间（基于历史误差）
            recent_intervals = self.arrival_intervals[-20:]
            std_interval = np.std(recent_intervals)
            confidence_interval = (
                max(0.001, predicted_interval - 1.96 * std_interval),
                predicted_interval + 1.96 * std_interval
            )

            return {
                'predicted_interval': predicted_interval,
                'predicted_arrival_time': predicted_arrival_time,
                'confidence': min(0.9, self.confidence_level * 0.9),  # 时间序列预测置信度稍低
                'method': 'time_series_dlinear',
                'uncertainty_range': confidence_interval,
                'prediction_metadata': {
                    'model_type': 'DLinear',
                    'history_size': len(recent_intervals),
                    'prediction_steps': 1
                }
            }

        except Exception as e:
            logger.warning(f"Time series prediction failed: {e}, falling back to statistical method")
            return self._statistical_prediction(current_time)

    def _poisson_process_prediction(self, current_time: float) -> Dict[str, Any]:
        """泊松过程预测"""
        recent_intervals = self.arrival_intervals[-min(self.window_size, len(self.arrival_intervals)):]

        # 计算到达率（requests per time unit）
        total_time = sum(recent_intervals)
        arrival_rate = len(recent_intervals) / total_time if total_time > 0 else 1.0

        # 泊松过程的间隔服从指数分布
        # 参数λ = arrival_rate
        predicted_interval = np.random.exponential(1.0 / arrival_rate) if arrival_rate > 0 else 1.0

        # 但我们使用期望值作为预测
        predicted_interval = 1.0 / arrival_rate if arrival_rate > 0 else 1.0

        # 计算置信区间（基于指数分布）
        alpha = 1 - self.confidence_level
        lower_bound = -np.log(1 - alpha/2) / arrival_rate if arrival_rate > 0 else 0.1
        upper_bound = -np.log(alpha/2) / arrival_rate if arrival_rate > 0 else 10.0

        predicted_arrival_time = current_time + predicted_interval

        return {
            'predicted_interval': predicted_interval,
            'predicted_arrival_time': predicted_arrival_time,
            'confidence': self.confidence_level,
            'method': 'poisson_process',
            'uncertainty_range': (lower_bound, upper_bound),
            'prediction_metadata': {
                'arrival_rate': arrival_rate,
                'process_type': 'Poisson',
                'history_size': len(recent_intervals)
            }
        }

    def _adaptive_prediction(self, current_time: float) -> Dict[str, Any]:
        """自适应预测方法"""
        recent_intervals = self.arrival_intervals[-min(50, len(self.arrival_intervals)):]

        if len(recent_intervals) < 10:
            return self._statistical_prediction(current_time)

        # 分析数据特征
        mean_interval = np.mean(recent_intervals)
        std_interval = np.std(recent_intervals)
        cv = std_interval / mean_interval if mean_interval > 0 else 0

        # 选择最佳方法
        if cv < 0.3:  # 低变异性，使用统计方法
            return self._statistical_prediction(current_time)
        elif cv < 0.8:  # 中等变异性，使用时间序列
            return self._time_series_prediction(current_time)
        else:  # 高变异性，使用泊松过程
            return self._poisson_process_prediction(current_time)

    def predict_multiple_arrivals(self, current_time: float = None, num_predictions: int = 10) -> List[Dict[str, Any]]:
        """
        预测多个连续的请求到达时间

        Args:
            current_time: 当前时间
            num_predictions: 预测数量

        Returns:
            预测结果列表
        """
        if current_time is None:
            current_time = self.arrival_timestamps[-1] if self.arrival_timestamps else 0

        predictions = []
        current_prediction_time = current_time

        for i in range(num_predictions):
            prediction = self.predict_next_arrival(current_prediction_time)
            predictions.append(prediction)
            current_prediction_time = prediction['predicted_arrival_time']

        return predictions

    def get_arrival_pattern_analysis(self) -> Dict[str, Any]:
        """
        分析到达模式

        Returns:
            模式分析结果
        """
        if len(self.arrival_intervals) < 5:
            return {'error': 'Insufficient data for pattern analysis'}

        recent_intervals = self.arrival_intervals[-min(self.window_size, len(self.arrival_intervals)):]

        # 基本统计
        mean_interval = np.mean(recent_intervals)
        median_interval = np.median(recent_intervals)
        std_interval = np.std(recent_intervals)
        min_interval = np.min(recent_intervals)
        max_interval = np.max(recent_intervals)

        # 分布特征
        cv = std_interval / mean_interval if mean_interval > 0 else 0
        skewness = stats.skew(recent_intervals)
        kurtosis = stats.kurtosis(recent_intervals)

        # 趋势分析
        if len(recent_intervals) >= 10:
            # 简单线性趋势
            x = np.arange(len(recent_intervals))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_intervals)
            trend = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
        else:
            trend = 'unknown'
            slope = 0

        # 周期性检测（简单方法）
        autocorr = np.corrcoef(recent_intervals[:-1], recent_intervals[1:])[0, 1] if len(recent_intervals) > 1 else 0

        return {
            'basic_statistics': {
                'mean_interval': mean_interval,
                'median_interval': median_interval,
                'std_interval': std_interval,
                'min_interval': min_interval,
                'max_interval': max_interval,
                'coefficient_of_variation': cv
            },
            'distribution_features': {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'normality_test_pvalue': stats.normaltest(recent_intervals).pvalue if len(recent_intervals) >= 8 else None
            },
            'trend_analysis': {
                'trend': trend,
                'slope': slope,
                'r_squared': r_value**2 if 'r_value' in locals() else 0
            },
            'autocorrelation': autocorr,
            'pattern_summary': self._summarize_pattern(cv, trend, autocorr),
            'history_size': len(recent_intervals)
        }

    def _summarize_pattern(self, cv: float, trend: str, autocorr: float) -> str:
        """总结到达模式"""
        if cv < 0.3:
            variability = "low variability (regular)"
        elif cv < 0.8:
            variability = "moderate variability"
        else:
            variability = "high variability (irregular)"

        if abs(autocorr) > 0.3:
            correlation = f"with {'positive' if autocorr > 0 else 'negative'} autocorrelation"
        else:
            correlation = "with no significant autocorrelation"

        return f"{variability}, {trend} trend, {correlation}"


# 全局实例
arrival_time_predictor = ArrivalTimePredictor()


def predict_next_batch_window(timestamps: List[float],
                            input_tokens: List[int],
                            output_tokens: List[int],
                            max_wait_time: float = 0.1,
                            max_batch_tokens: int = 8192) -> Dict[str, Any]:
    """
    预测下一个批处理窗口

    Args:
        timestamps: 历史时间戳
        input_tokens: 输入token列表
        output_tokens: 输出token列表
        max_wait_time: 最大等待时间（秒）
        max_batch_tokens: 最大batch token数

    Returns:
        批处理窗口预测
    """
    predictor = ArrivalTimePredictor()
    predictor.update_history(timestamps)

    # 预测下一个请求到达时间
    next_arrival = predictor.predict_next_arrival()

    # 模拟批处理收集
    batch_requests = []
    batch_tokens = 0
    window_start_time = timestamps[-1] if timestamps else 0
    current_time = window_start_time

    # 预测最多10个请求或达到限制
    for i in range(10):
        if current_time - window_start_time > max_wait_time:
            break

        if batch_tokens >= max_batch_tokens:
            break

        # 添加预测的请求
        avg_tokens = np.mean(input_tokens[-10:]) if input_tokens else 500
        predicted_tokens = avg_tokens * 2  # 输入+输出

        if batch_tokens + predicted_tokens <= max_batch_tokens:
            batch_requests.append({
                'arrival_time': current_time,
                'predicted_tokens': predicted_tokens
            })
            batch_tokens += predicted_tokens

            # 预测下一个请求
            next_arrival = predictor.predict_next_arrival(current_time)
            current_time = next_arrival['predicted_arrival_time']
        else:
            break

    return {
        'window_start_time': window_start_time,
        'window_end_time': current_time,
        'window_duration': current_time - window_start_time,
        'predicted_requests': len(batch_requests),
        'predicted_total_tokens': batch_tokens,
        'batch_efficiency': batch_tokens / max_batch_tokens if max_batch_tokens > 0 else 0,
        'next_arrival_prediction': next_arrival,
        'batch_details': batch_requests
    }