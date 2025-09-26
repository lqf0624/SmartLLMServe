"""
多模型集成预测模块

实现LSTM和DLinear的智能集成预测，包括：
- 动态权重优化
- 自适应模型选择
- 性能监控和权重调整
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import logging
from enum import Enum
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from datetime import datetime, timedelta

from .workload_predictor import ModelType, PredictionHorizon
from .lstm_model import LSTMPredictor
from .dlinear_predictor_adapter import create_dlinear_predictor

logger = logging.getLogger(__name__)

# 类型注解的延迟导入
if False:  # 用于类型检查，实际运行时使用延迟导入
    from .workload_predictor import PredictionResult


class WeightStrategy(Enum):
    """权重策略枚举"""
    STATIC = "static"           # 静态权重
    DYNAMIC = "dynamic"         # 动态权重
    PERFORMANCE_BASED = "performance_based"  # 基于性能的权重
    ADAPTIVE = "adaptive"       # 自适应权重


class EnsemblePredictor:
    """
    多模型集成预测器

    实现LSTM和DLinear模型的智能集成，支持多种权重策略和动态调整。
    """

    def __init__(self,
                 models: List[ModelType] = None,
                 weight_strategy: WeightStrategy = WeightStrategy.PERFORMANCE_BASED,
                 prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM,
                 sequence_length: int = 50,
                 performance_window: int = 100,
                 min_weight: float = 0.1,
                 max_weight: float = 0.9):
        """
        初始化集成预测器

        Args:
            models: 模型列表，默认为LSTM和DLinear
            weight_strategy: 权重策略
            prediction_horizon: 预测时间范围
            sequence_length: 序列长度
            performance_window: 性能评估窗口大小
            min_weight: 最小权重限制
            max_weight: 最大权重限制
        """
        self.models = models or [ModelType.LSTM, ModelType.DLINEAR]
        self.weight_strategy = weight_strategy
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length
        self.performance_window = performance_window
        self.min_weight = min_weight
        self.max_weight = max_weight

        # 初始化模型实例
        self.model_instances = {}
        self._initialize_models()

        # 权重管理
        self.weights = self._initialize_weights()
        self.performance_history = {model.value: [] for model in self.models}

        # 集成配置
        self.enable_confidence_weighting = True
        self.enable_dynamic_adjustment = True
        self.adjustment_threshold = 0.05  # 5%的性能变化触发调整

        logger.info(f"EnsemblePredictor initialized with models: {[m.value for m in self.models]}")
        logger.info(f"Weight strategy: {weight_strategy.value}")

    def _initialize_models(self):
        """初始化各个模型实例"""
        for model_type in self.models:
            try:
                if model_type == ModelType.LSTM:
                    self.model_instances[model_type] = LSTMPredictor(
                        sequence_length=self.sequence_length,
                        prediction_horizon=self._get_prediction_steps(),
                        hidden_size=64,
                        num_layers=2
                    )
                elif model_type == ModelType.DLINEAR:
                    self.model_instances[model_type] = create_dlinear_predictor(
                        prediction_horizon=self.prediction_horizon,
                        sequence_length=self.sequence_length
                    )
                else:
                    logger.warning(f"Unsupported model type: {model_type}")
                    continue

                logger.info(f"Model {model_type.value} initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize model {model_type.value}: {e}")
                continue

    def _get_prediction_steps(self) -> int:
        """获取预测步数"""
        steps_map = {
            PredictionHorizon.SHORT_TERM: 10,
            PredictionHorizon.MEDIUM_TERM: 30,
            PredictionHorizon.LONG_TERM: 60,
            PredictionHorizon.EXTENDED_TERM: 120
        }
        return steps_map.get(self.prediction_horizon, 10)

    def _initialize_weights(self) -> Dict[str, float]:
        """初始化模型权重"""
        if self.weight_strategy == WeightStrategy.STATIC:
            # 均匀权重
            weight = 1.0 / len(self.models)
            return {model.value: weight for model in self.models}
        else:
            # 性能基于权重将在训练后更新
            weight = 1.0 / len(self.models)
            return {model.value: weight for model in self.models}

    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        训练集成模型

        Args:
            data: 训练数据
            validation_split: 验证集比例

        Returns:
            训练结果字典
        """
        logger.info(f"Training ensemble models with {len(data)} samples")

        results = {}
        val_size = int(len(data) * validation_split)
        train_data = data.iloc[:-val_size] if val_size > 0 else data
        val_data = data.iloc[-val_size:] if val_size > 0 else data

        # 训练各个模型（优化版本：减少重复计算）
        for model_type, model_instance in self.model_instances.items():
            logger.info(f"Training {model_type.value} model...")

            try:
                if hasattr(model_instance, 'train'):
                    # 性能优化：减少验证数据集大小以加快训练
                    opt_validation_split = min(validation_split, 0.15)  # 最多15%用于验证

                    if model_type == ModelType.LSTM:
                        # LSTM优化：使用较少的epochs进行初始训练
                        original_epochs = getattr(model_instance, 'epochs', 100)
                        if hasattr(model_instance, 'epochs'):
                            model_instance.epochs = min(50, original_epochs)  # 限制epochs数量

                        model_result = model_instance.train(
                            train_data, validation_split=opt_validation_split
                        )

                        # 恢复原始epochs设置
                        if hasattr(model_instance, 'epochs'):
                            model_instance.epochs = original_epochs
                    else:
                        # DLinear等模型的训练（优化版本）
                        model_result = self._train_model(model_type, model_instance, train_data, opt_validation_split)
                else:
                    logger.warning(f"Model {model_type.value} has no train method")
                    continue

                results[model_type.value] = model_result
                logger.info(f"{model_type.value} training completed")

            except Exception as e:
                logger.error(f"Failed to train {model_type.value}: {e}")
                continue

        # 权重优化（限制验证集大小以提高性能）
        if len(val_data) > 0 and self.weight_strategy != WeightStrategy.STATIC:
            # 使用较小的验证集进行权重优化
            opt_val_size = min(len(val_data), 100)  # 最多100个样本用于权重优化
            opt_val_data = val_data.iloc[-opt_val_size:] if opt_val_size < len(val_data) else val_data
            self._optimize_weights(opt_val_data)

        # 集成结果
        ensemble_result = {
            'individual_results': results,
            'ensemble_weights': self.weights.copy(),
            'weight_strategy': self.weight_strategy.value,
            'training_samples': len(train_data),
            'validation_samples': len(val_data),
            'active_models': list(self.model_instances.keys())
        }

        logger.info("Ensemble training completed")
        return ensemble_result

    def _train_model(self, model_type: ModelType, model_instance, data: pd.DataFrame, validation_split: float = 0.0) -> Dict[str, Any]:
        """训练特定模型的辅助方法"""
        # 这里根据不同模型类型实现具体的训练逻辑
        if model_type == ModelType.DLINEAR:
            # DLinear模型的训练逻辑
            return self._train_dlinear_model(model_instance, data, validation_split)
        else:
            return {'status': 'trained', 'model_type': model_type.value}

    def _train_dlinear_model(self, model_instance, data: pd.DataFrame, validation_split: float = 0.0) -> Dict[str, Any]:
        """训练DLinear模型"""
        try:
            # 使用适配器的训练方法
            if hasattr(model_instance, 'train'):
                result = model_instance.train(data, validation_split=validation_split)
                return result
            else:
                return {'status': 'trained', 'model_type': 'DLinear'}
        except Exception as e:
            logger.error(f"DLinear training failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _optimize_weights(self, validation_data: pd.DataFrame):
        """基于验证集优化权重"""
        logger.info("Optimizing ensemble weights based on validation performance")

        model_performances = {}
        for model_type, model_instance in self.model_instances.items():
            try:
                # 在验证集上进行预测
                if hasattr(model_instance, 'predict') or hasattr(model_instance, 'predict_future'):
                    steps = self._get_prediction_steps()
                    historical_data = validation_data.iloc[:-steps]

                    # LSTM 使用 predict_future，其他模型使用 predict
                    if model_type == ModelType.LSTM:
                        raw_prediction = model_instance.predict_future(historical_data, steps=steps)
                        # 延迟导入以避免循环导入
                        from .workload_predictor import PredictionResult
                        # 转换为 PredictionResult 格式
                        prediction = PredictionResult(
                            predicted_requests=[],  # LSTM 不返回具体请求，只返回数值
                            confidence=raw_prediction.get('confidence', 0.85),
                            prediction_metadata=raw_prediction
                        )
                    else:
                        prediction = model_instance.predict(historical_data, steps=steps)

                    # 计算性能指标
                    performance = self._calculate_model_performance(
                        validation_data, prediction
                    )
                    model_performances[model_type.value] = performance
                else:
                    logger.warning(f"Model {model_type.value} has no predict method")
                    model_performances[model_type.value] = 0.5  # 默认性能

            except Exception as e:
                logger.error(f"Failed to evaluate {model_type.value}: {e}")
                model_performances[model_type.value] = 0.1  # 低性能惩罚

        # 根据策略调整权重
        if self.weight_strategy == WeightStrategy.PERFORMANCE_BASED:
            self._update_weights_performance_based(model_performances)
        elif self.weight_strategy == WeightStrategy.DYNAMIC:
            self._update_weights_dynamic(model_performances)
        elif self.weight_strategy == WeightStrategy.ADAPTIVE:
            self._update_weights_adaptive(model_performances)

        logger.info(f"Optimized weights: {self.weights}")

    def _calculate_model_performance(self, actual_data: pd.DataFrame, prediction: 'PredictionResult') -> float:
        """计算模型性能得分"""
        try:
            # 提取实际值和预测值
            actual_rates = self._extract_request_rates(actual_data)
            predicted_rates = self._extract_predicted_rates(prediction)

            if len(actual_rates) == 0 or len(predicted_rates) == 0:
                return 0.5  # 默认性能

            # 对齐数据
            min_len = min(len(actual_rates), len(predicted_rates))
            actual_values = actual_rates.iloc[:min_len].values
            predicted_values = predicted_rates.iloc[:min_len].values

            # 计算多个指标
            mae = mean_absolute_error(actual_values, predicted_values)
            mse = mean_squared_error(actual_values, predicted_values)

            # 综合性能得分 (0-1，越高越好)
            mae_score = 1.0 / (1.0 + mae)  # MAE转换为得分
            mse_score = 1.0 / (1.0 + mse)  # MSE转换为得分

            performance = (mae_score + mse_score) / 2.0
            return max(0.0, min(1.0, performance))

        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return 0.5

    def _extract_request_rates(self, data: pd.DataFrame) -> pd.Series:
        """从数据中提取请求速率"""
        if len(data) == 0:
            return pd.Series(dtype=float)

        # 计算每分钟的请求速率
        rates = data.resample('1min', on='arrival_time_ns').size()
        return rates

    def _extract_predicted_rates(self, prediction: 'PredictionResult') -> pd.Series:
        """从预测结果中提取预测速率"""
        if not prediction.predicted_requests:
            return pd.Series(dtype=float)

        # 创建预测时间序列
        pred_df = pd.DataFrame(prediction.predicted_requests)
        pred_df['arrival_time_ns'] = pd.to_datetime(pred_df['arrival_time_ns'])

        # 计算预测速率
        rates = pred_df.resample('1min', on='arrival_time_ns').size()
        return rates

    def _update_weights_performance_based(self, performances: Dict[str, float]):
        """基于性能更新权重"""
        total_performance = sum(performances.values())
        if total_performance == 0:
            # 如果所有模型性能都很差，使用均匀权重
            weight = 1.0 / len(performances)
            self.weights = {model: weight for model in performances.keys()}
            return

        # 基于性能分配权重
        for model in performances.keys():
            raw_weight = performances[model] / total_performance
            # 应用权重限制
            self.weights[model] = max(self.min_weight, min(self.max_weight, raw_weight))

        # 归一化权重
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {model: weight / total_weight for model, weight in self.weights.items()}

    def _update_weights_dynamic(self, performances: Dict[str, float]):
        """动态更新权重"""
        # 基于历史性能和当前性能的加权平均
        for model, current_perf in performances.items():
            if model in self.performance_history:
                # 获取历史性能
                history = self.performance_history[model][-self.performance_window:]
                avg_historical_perf = np.mean(history) if history else current_perf

                # 动态调整：当前性能和历史性能的加权平均
                alpha = 0.3  # 当前性能权重
                adjusted_perf = alpha * current_perf + (1 - alpha) * avg_historical_perf
            else:
                adjusted_perf = current_perf

            # 更新性能历史
            self.performance_history[model].append(adjusted_perf)

            # 限制历史长度
            if len(self.performance_history[model]) > self.performance_window:
                self.performance_history[model] = self.performance_history[model][-self.performance_window:]

        # 使用调整后的性能更新权重
        adjusted_performances = {
            model: self.performance_history[model][-1] if self.performance_history[model] else 0.5
            for model in performances.keys()
        }
        self._update_weights_performance_based(adjusted_performances)

    def _update_weights_adaptive(self, performances: Dict[str, float]):
        """自适应更新权重"""
        # 计算性能差异
        if len(performances) < 2:
            self._update_weights_performance_based(performances)
            return

        perf_values = list(performances.values())
        max_perf = max(perf_values)
        min_perf = min(perf_values)
        perf_range = max_perf - min_perf

        if perf_range < self.adjustment_threshold:
            # 性能差异很小，使用均匀权重
            weight = 1.0 / len(performances)
            self.weights = {model: weight for model in performances.keys()}
            return

        # 自适应权重：性能越好，权重越高
        total_adaptive_weight = 0
        adaptive_weights = {}

        for model, perf in performances.items():
            # 线性映射到权重
            if perf_range > 0:
                adaptive_weight = (perf - min_perf) / perf_range
            else:
                adaptive_weight = 0.5

            # 应用权重限制
            adaptive_weight = max(self.min_weight, min(self.max_weight, adaptive_weight))
            adaptive_weights[model] = adaptive_weight
            total_adaptive_weight += adaptive_weight

        # 归一化
        if total_adaptive_weight > 0:
            self.weights = {model: weight / total_adaptive_weight for model, weight in adaptive_weights.items()}

    def predict(self, data: pd.DataFrame, steps: int = None) -> 'PredictionResult':
        """
        集成预测

        Args:
            data: 历史数据
            steps: 预测步数

        Returns:
            集成预测结果
        """
        if steps is None:
            steps = self._get_prediction_steps()

        logger.info(f"Making ensemble prediction with {steps} steps")

        # 各个模型进行预测
        individual_predictions = {}
        individual_confidences = {}

        for model_type, model_instance in self.model_instances.items():
            try:
                if hasattr(model_instance, 'predict') or hasattr(model_instance, 'predict_future'):
                    # LSTM 使用 predict_future，其他模型使用 predict
                    if model_type == ModelType.LSTM:
                        raw_prediction = model_instance.predict_future(data, steps=steps)
                        # 延迟导入以避免循环导入
                        from .workload_predictor import PredictionResult
                        # 转换为 PredictionResult 格式
                        prediction = PredictionResult(
                            predicted_requests=[],  # LSTM 不返回具体请求，只返回数值
                            confidence=raw_prediction.get('confidence', 0.85),
                            prediction_metadata=raw_prediction
                        )
                    else:
                        prediction = model_instance.predict(data, steps=steps)
                    individual_predictions[model_type.value] = prediction
                    individual_confidences[model_type.value] = prediction.confidence

                    logger.info(f"{model_type.value} prediction completed, confidence: {prediction.confidence:.3f}")
                else:
                    logger.warning(f"Model {model_type.value} has no predict method")

            except Exception as e:
                logger.error(f"Failed to predict with {model_type.value}: {e}")
                continue

        if not individual_predictions:
            logger.error("No models available for prediction")
            # 返回空预测结果
            return PredictionResult(
                predicted_requests=[],
                confidence=0.0,
                prediction_metadata={'error': 'No models available for prediction'}
            )

        # 集成预测结果
        ensemble_result = self._combine_predictions(individual_predictions, data, steps)

        # 更新性能历史（如果需要）
        if self.enable_dynamic_adjustment and len(individual_predictions) > 1:
            self._update_prediction_performance(individual_predictions, data)

        logger.info(f"Ensemble prediction completed, confidence: {ensemble_result.confidence:.3f}")
        return ensemble_result

    def _combine_predictions(self, predictions: Dict[str, 'PredictionResult'],
                           data: pd.DataFrame, steps: int) -> 'PredictionResult':
        """组合多个模型的预测结果"""
        if not predictions:
            # 没有成功的预测，生成合成预测
            final_requests = self._generate_synthetic_predictions(data, 10, steps)
            return PredictionResult(
                predicted_requests=final_requests,
                confidence=0.5,
                prediction_metadata={
                    'ensemble_weights': self.weights.copy(),
                    'individual_predictions': {},
                    'individual_confidences': {},
                    'weight_strategy': self.weight_strategy.value,
                    'active_models': [],
                    'prediction_time': datetime.now().isoformat(),
                    'steps': steps,
                    'fallback_reason': 'No successful predictions'
                }
            )

        # 提取各个模型的预测请求数量
        predicted_counts = {}
        for model_name, pred_result in predictions.items():
            count = len(pred_result.predicted_requests)
            predicted_counts[model_name] = count

        # 基于权重计算集成预测数量
        weighted_count = 0
        total_weight = 0

        for model_name, count in predicted_counts.items():
            weight = self.weights.get(model_name, 0.0)
            weighted_count += weight * count
            total_weight += weight

        # 确保至少有一个预测
        final_count = max(1, int(round(weighted_count / max(total_weight, 1e-9))))

        # 选择权重最高的模型作为基础，然后调整数量
        # 首先找到在predictions中存在的权重最高的模型
        available_models = set(predictions.keys()) & set(self.weights.keys())
        if not available_models:
            # 如果没有共同的模型，使用第一个可用的模型
            best_model = list(predictions.keys())[0]
        else:
            best_model = max(available_models, key=lambda x: self.weights.get(x, 0.0))

        base_prediction = predictions[best_model]

        # 如果基础模型没有预测请求，尝试使用其他模型
        if not base_prediction.predicted_requests:
            for model_name, prediction in predictions.items():
                if prediction.predicted_requests:
                    base_prediction = prediction
                    break

        # 调整预测请求数量
        if not base_prediction.predicted_requests:
            # 所有模型都没有预测请求，生成合成预测
            final_requests = self._generate_synthetic_predictions(data, final_count, steps)
        elif final_count != len(base_prediction.predicted_requests):
            if final_count < len(base_prediction.predicted_requests):
                # 减少请求数量
                final_requests = base_prediction.predicted_requests[:final_count]
            else:
                # 增加请求数量（复制最后一个）
                final_requests = base_prediction.predicted_requests.copy()
                last_request = final_requests[-1].copy()

                # 为新增的请求调整时间
                last_time = pd.to_datetime(last_request['arrival_time_ns'])
                for i in range(final_count - len(base_prediction.predicted_requests)):
                    new_request = last_request.copy()
                    new_request['arrival_time_ns'] = (last_time + pd.Timedelta(seconds=i+1)).value
                    new_request['request_id'] = f"{best_model}_ensemble_{i}"
                    final_requests.append(new_request)
        else:
            final_requests = base_prediction.predicted_requests.copy()

        # 创建集成元数据
        metadata = {
            'ensemble_weights': self.weights.copy(),
            'individual_predictions': {k: len(v.predicted_requests) for k, v in predictions.items()},
            'individual_confidences': {k: v.confidence for k, v in predictions.items()},
            'weight_strategy': self.weight_strategy.value,
            'active_models': list(predictions.keys()),
            'prediction_time': datetime.now().isoformat(),
            'steps': steps
        }

        # 计算集成置信度
        individual_confidences = {k: v.confidence for k, v in predictions.items()}
        ensemble_confidence = self._calculate_ensemble_confidence(individual_confidences)

        # 延迟导入以避免循环导入
        from .workload_predictor import PredictionResult
        return PredictionResult(
            predicted_requests=final_requests,
            confidence=ensemble_confidence,
            prediction_metadata=metadata
        )

    def _calculate_ensemble_confidence(self, confidences: Dict[str, float]) -> float:
        """计算集成置信度"""
        if not confidences:
            return 0.5

        if not self.enable_confidence_weighting:
            return np.mean(list(confidences.values()))

        # 基于权重和置信度的加权平均
        weighted_confidence = 0.0
        total_weight = 0.0

        for model_name, confidence in confidences.items():
            weight = self.weights.get(model_name, 0.0)
            weighted_confidence += weight * confidence
            total_weight += weight

        # 如果总权重为0，使用简单平均
        if total_weight < 1e-9:
            return np.mean(list(confidences.values()))

        return weighted_confidence / total_weight

    def _generate_synthetic_predictions(self, data: pd.DataFrame, count: int, steps: int) -> List[Dict]:
        """生成合成预测请求"""
        synthetic_requests = []

        # 基于历史数据的统计信息生成预测
        if len(data) > 0:
            avg_input_toks = data['input_toks'].mean()
            avg_output_toks = data['output_toks'].mean()
            last_time = pd.to_datetime(data['arrival_time_ns'].iloc[-1])

            for i in range(count):
                # 生成时间间隔（基于历史数据的平均间隔）
                time_interval = pd.Timedelta(seconds=(i + 1) * steps / count)
                arrival_time = last_time + time_interval

                # 生成token数量（基于历史数据的均值和方差）
                input_toks = max(1, int(avg_input_toks + np.random.normal(0, avg_input_toks * 0.2)))
                output_toks = max(1, int(avg_output_toks + np.random.normal(0, avg_output_toks * 0.2)))

                synthetic_request = {
                    'request_id': f"ensemble_synthetic_{i}",
                    'arrival_time_ns': arrival_time.value,
                    'input_tokens': input_toks,
                    'output_tokens': output_toks,
                    'burst_pattern': 'steady',
                    'model_type': 'ChatGPT',
                    'request_type': 'medium',
                    'priority': 'medium',
                    'estimated_compute_time': 0.1,
                    'memory_requirement_mb': 100.0,
                    'predicted_accuracy': 0.85
                }
                synthetic_requests.append(synthetic_request)

        return synthetic_requests

    def _update_prediction_performance(self, predictions: Dict[str, 'PredictionResult'], data: pd.DataFrame):
        """更新预测性能历史"""
        # 这里可以在线更新性能，但需要实际值进行对比
        # 在实际应用中，可以在获得实际值后调用此方法
        pass

    def get_ensemble_info(self) -> Dict[str, Any]:
        """获取集成预测器信息"""
        return {
            'models': [model.value for model in self.models],
            'active_models': list(self.model_instances.keys()),
            'weight_strategy': self.weight_strategy.value,
            'weights': self.weights.copy(),
            'performance_history': {
                model: history[-10:] if history else []  # 最近10次性能
                for model, history in self.performance_history.items()
            },
            'configuration': {
                'prediction_horizon': self.prediction_horizon.value,
                'sequence_length': self.sequence_length,
                'performance_window': self.performance_window,
                'min_weight': self.min_weight,
                'max_weight': self.max_weight
            }
        }

    def update_weights(self, new_weights: Dict[str, float]):
        """手动更新权重"""
        # 验证权重
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:  # 允许1%的误差
            # 归一化权重
            new_weights = {k: v / total_weight for k, v in new_weights.items()}

        # 应用权重限制
        for model, weight in new_weights.items():
            new_weights[model] = max(self.min_weight, min(self.max_weight, weight))

        self.weights = new_weights
        logger.info(f"Weights updated manually: {self.weights}")

    def save(self, path: str):
        """保存集成预测器"""
        import pickle
        save_data = {
            'models': self.models,
            'weight_strategy': self.weight_strategy,
            'weights': self.weights,
            'performance_history': self.performance_history,
            'configuration': {
                'prediction_horizon': self.prediction_horizon.value,
                'sequence_length': self.sequence_length,
                'performance_window': self.performance_window,
                'min_weight': self.min_weight,
                'max_weight': self.max_weight
            }
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"EnsemblePredictor saved to {path}")

    @classmethod
    def load(cls, path: str):
        """加载集成预测器"""
        import pickle

        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        # 重建实例
        instance = cls(
            models=save_data['models'],
            weight_strategy=save_data['weight_strategy'],
            prediction_horizon=PredictionHorizon(save_data['configuration']['prediction_horizon']),
            sequence_length=save_data['configuration']['sequence_length'],
            performance_window=save_data['configuration']['performance_window'],
            min_weight=save_data['configuration']['min_weight'],
            max_weight=save_data['configuration']['max_weight']
        )

        # 恢复状态
        instance.weights = save_data['weights']
        instance.performance_history = save_data['performance_history']

        logger.info(f"EnsemblePredictor loaded from {path}")
        return instance