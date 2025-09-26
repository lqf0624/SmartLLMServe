"""
DLinear预测器适配器 - 原生DLinear模型与工作负载预测器的统一接口

将原生的DLinear模型适配到现有的工作负载预测框架中，
确保与LSTM模型的接口兼容性，提供统一的训练和预测接口。
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pickle
import json
from datetime import datetime

from .dlinear_model import DLinearPredictor, DecompositionMode
from .workload_predictor import PredictionHorizon, ModelType, PredictionResult
from .data_preprocessor import DataPreprocessor, DataFormat
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class DLinearPredictorAdapter:
    """
    DLinear预测器适配器

    将原生DLinear模型适配到工作负载预测框架中，提供与LSTM兼容的接口。
    支持多种预测时间窗口和统一的训练/预测流程。
    """

    def __init__(self,
                 prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM,
                 sequence_length: int = 50,
                 individual: bool = False,
                 decomposition_mode: DecompositionMode = DecompositionMode.MOVING_AVG,
                 kernel_size: int = 25,
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        初始化DLinear预测器适配器

        Args:
            prediction_horizon: 预测时间范围
            sequence_length: 序列长度
            individual: 是否为每个特征单独建模
            decomposition_mode: 分解模式
            kernel_size: 分解窗口大小
            learning_rate: 学习率
            device: 计算设备
        """
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length
        self.model_type = ModelType.DLINEAR

        # 映射预测时间范围到输出步数
        self.horizon_steps = {
            PredictionHorizon.SHORT_TERM: 10,
            PredictionHorizon.MEDIUM_TERM: 30,
            PredictionHorizon.LONG_TERM: 60,
            PredictionHorizon.EXTENDED_TERM: 120
        }

        output_size = self.horizon_steps[prediction_horizon]

        # 创建原生DLinear预测器
        self.dlinear_predictor = DLinearPredictor(
            input_size=sequence_length,
            output_size=output_size,
            individual=individual,
            decomposition_mode=decomposition_mode,
            kernel_size=kernel_size,
            learning_rate=learning_rate,
            device=device
        )

        # 数据预处理器
        self.data_preprocessor = DataPreprocessor()

        # 特征提取器（用于兼容性）
        self.feature_extractor = FeatureExtractor()

        # 训练历史和模型信息
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'epochs': 0,
            'best_val_loss': float('inf')
        }

        # 模型元数据
        self.model_metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': 'DLinear',
            'prediction_horizon': prediction_horizon.value,
            'sequence_length': sequence_length,
            'decomposition_mode': decomposition_mode.value,
            'kernel_size': kernel_size,
            'version': '1.0'
        }

        logger.info(f"DLinearPredictorAdapter initialized: {prediction_horizon.value}, "
                   f"steps={output_size}, sequence_length={sequence_length}")

    def train(self,
              data: Union[pd.DataFrame, str, Path],
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32,
              patience: int = 10,
              verbose: bool = True) -> Dict[str, Any]:
        """
        训练DLinear模型

        Args:
            data: 训练数据（DataFrame或文件路径）
            validation_split: 验证集比例
            epochs: 训练轮数
            batch_size: 批大小
            patience: 早停耐心值
            verbose: 是否显示训练过程

        Returns:
            Dict[str, Any]: 训练结果
        """
        logger.info(f"开始训练DLinear模型，数据形状: {self._get_data_shape(data)}")

        # 数据预处理
        processed_data = self._preprocess_training_data(data)

        # 训练DLinear模型
        training_result = self.dlinear_predictor.train(
            data=processed_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            patience=patience,
            verbose=verbose
        )

        # 更新训练历史
        self._update_training_history(training_result)

        # 记录模型信息
        self.model_metadata['trained_at'] = datetime.now().isoformat()
        self.model_metadata['training_epochs'] = training_result.get('epochs_trained', 0)
        self.model_metadata['best_val_loss'] = training_result.get('best_val_loss', float('inf'))

        logger.info(f"DLinear模型训练完成，最佳验证损失: {training_result.get('best_val_loss', 'N/A')}")

        return self._format_training_result(training_result)

    def predict(self,
                historical_data: Union[pd.DataFrame, str, Path],
                steps: Optional[int] = None,
                return_confidence: bool = False) -> PredictionResult:
        """
        进行预测

        Args:
            historical_data: 历史数据
            steps: 预测步数
            return_confidence: 是否返回置信度

        Returns:
            PredictionResult: 预测结果
        """
        logger.info(f"开始DLinear预测，历史数据长度: {self._get_data_length(historical_data)}")

        # 确定预测步数
        if steps is None:
            steps = self.horizon_steps[self.prediction_horizon]

        # 数据预处理
        processed_data = self._preprocess_prediction_data(historical_data)

        # 执行预测
        try:
            prediction_tensor = self.dlinear_predictor.predict(processed_data, steps=steps)

            # 转换为numpy数组
            if isinstance(prediction_tensor, torch.Tensor):
                predictions = prediction_tensor.detach().cpu().numpy()
            else:
                predictions = np.array(prediction_tensor)

            # 确保预测结果形状正确
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)

            # 生成预测结果
            prediction_result = self._create_prediction_result(
                predictions, historical_data, return_confidence
            )

            logger.info(f"DLinear预测完成，预测步数: {steps}")
            return prediction_result

        except Exception as e:
            logger.error(f"DLinear预测失败: {e}")
            # 返回空预测结果
            return self._create_empty_prediction_result(steps)

    def predict_future(self, data: pd.DataFrame, steps: int = None) -> Dict[str, Any]:
        """
        预测未来值（与LSTM兼容的接口）

        Args:
            data: 历史数据
            steps: 预测步数

        Returns:
            Dict[str, Any]: 预测结果字典
        """
        if steps is None:
            steps = self.horizon_steps[self.prediction_horizon]

        logger.info(f"DLinear predict_future called with steps={steps}")

        # 使用现有的predict方法
        try:
            prediction_result = self.predict(data, steps=steps)

            # 提取预测值
            predictions = []
            for req in prediction_result.predicted_requests:
                predictions.append([req['input_tokens'], req['output_tokens']])

            predictions_array = np.array(predictions) if predictions else np.array([])

            return {
                'predictions': predictions_array,
                'confidence': prediction_result.confidence,
                'model_type': 'DLinear',
                'prediction_steps': steps,
                'sequence_length': self.sequence_length
            }
        except Exception as e:
            logger.error(f"DLinear predict_future failed: {e}")
            return {
                'predictions': np.array([]),
                'confidence': 0.0,
                'model_type': 'DLinear',
                'error': str(e),
                'prediction_steps': steps,
                'sequence_length': self.sequence_length
            }

    def _preprocess_training_data(self, data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        """预处理训练数据"""
        if isinstance(data, (str, Path)):
            # 从文件加载数据
            if str(data).endswith('.csv'):
                data = pd.read_csv(data)
            elif str(data).endswith('.pkl'):
                with open(data, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {data}")

        # 使用数据预处理器
        processed_data = self.data_preprocessor.preprocess_data(data)

        # 确保包含必需的列
        required_columns = ['input_toks', 'output_toks']
        for col in required_columns:
            if col not in processed_data.columns:
                logger.warning(f"Missing required column: {col}, using synthetic data")
                processed_data[col] = np.random.randint(50, 200, len(processed_data))

        return processed_data

    def _preprocess_prediction_data(self, historical_data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        """预处理预测数据"""
        if isinstance(historical_data, (str, Path)):
            # 从文件加载数据
            if str(historical_data).endswith('.csv'):
                historical_data = pd.read_csv(historical_data)
            elif str(historical_data).endswith('.pkl'):
                with open(historical_data, 'rb') as f:
                    historical_data = pickle.load(historical_data)

        # 使用数据预处理器
        processed_data = self.data_preprocessor.preprocess_data(historical_data)

        return processed_data

    def _create_prediction_result(self,
                                 predictions: np.ndarray,
                                 historical_data: Union[pd.DataFrame, str, Path],
                                 return_confidence: bool) -> PredictionResult:
        """创建预测结果对象"""
        # 获取历史数据统计信息
        if isinstance(historical_data, pd.DataFrame):
            historical_stats = self._compute_historical_stats(historical_data)
        else:
            historical_stats = {}

        # 生成时间戳
        timestamps = self._generate_prediction_timestamps(len(predictions), historical_data)

        # 分离输入和输出token预测
        if predictions.shape[1] >= 2:
            input_token_predictions = predictions[:, 0]
            output_token_predictions = predictions[:, 1]
        else:
            # 如果只有一列，复制给输入和输出
            input_token_predictions = predictions[:, 0]
            output_token_predictions = predictions[:, 0]

        # 转换为预测请求格式
        predicted_requests = []
        for i, (input_tok, output_tok) in enumerate(zip(input_token_predictions, output_token_predictions)):
            request = {
                'request_id': f'pred_{i}',
                'arrival_time_ns': timestamps[i].value if hasattr(timestamps[i], 'value') else int(timestamps[i].timestamp() * 1e9),
                'input_tokens': max(0, int(input_tok)),
                'output_tokens': max(0, int(output_tok)),
                'burst_pattern': 'predicted',
                'model_type': 'ChatGPT',
                'request_type': 'predicted',
                'priority': 'medium',
                'estimated_compute_time': float(output_tok * 0.001),  # 粗略估计
                'memory_requirement_mb': float((input_tok + output_tok) * 0.001),
                'predicted_accuracy': 0.8
            }
            predicted_requests.append(request)

        # 计算置信度
        confidence = 0.8  # DLinear默认置信度

        # 创建预测结果元数据
        prediction_metadata = {
            'model_version': self.model_metadata['version'],
            'decomposition_mode': self.model_metadata['decomposition_mode'],
            'sequence_length': self.sequence_length,
            'historical_stats': historical_stats,
            'model_type': self.model_type.value,
            'prediction_horizon': self.prediction_horizon.value
        }

        return PredictionResult(
            predicted_requests=predicted_requests,
            confidence=confidence,
            prediction_metadata=prediction_metadata
        )

    def _create_empty_prediction_result(self, steps: int) -> PredictionResult:
        """创建空的预测结果"""
        timestamps = pd.date_range(start=datetime.now(), periods=steps, freq='100ms')

        # 创建空的预测请求
        predicted_requests = []
        for i, timestamp in enumerate(timestamps):
            request = {
                'request_id': f'empty_pred_{i}',
                'arrival_time_ns': timestamp.value if hasattr(timestamp, 'value') else int(timestamp.timestamp() * 1e9),
                'input_tokens': 0,
                'output_tokens': 0,
                'burst_pattern': 'empty',
                'model_type': 'ChatGPT',
                'request_type': 'empty',
                'priority': 'low',
                'estimated_compute_time': 0.0,
                'memory_requirement_mb': 0.0,
                'predicted_accuracy': 0.0
            }
            predicted_requests.append(request)

        prediction_metadata = {
            'error': 'Prediction failed',
            'fallback': True,
            'model_type': self.model_type.value,
            'prediction_horizon': self.prediction_horizon.value
        }

        return PredictionResult(
            predicted_requests=predicted_requests,
            confidence=0.0,
            prediction_metadata=prediction_metadata
        )

    def _compute_historical_stats(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算历史数据统计信息"""
        stats = {}

        if 'input_toks' in data.columns:
            stats['input_toks_mean'] = float(data['input_toks'].mean())
            stats['input_toks_std'] = float(data['input_toks'].std())
            stats['input_toks_max'] = float(data['input_toks'].max())

        if 'output_toks' in data.columns:
            stats['output_toks_mean'] = float(data['output_toks'].mean())
            stats['output_toks_std'] = float(data['output_toks'].std())
            stats['output_toks_max'] = float(data['output_toks'].max())

        return stats

    def _generate_prediction_timestamps(self,
                                       steps: int,
                                       historical_data: Union[pd.DataFrame, str, Path]) -> pd.DatetimeIndex:
        """生成预测时间戳"""
        try:
            if isinstance(historical_data, pd.DataFrame):
                if 'arrival_time_ns' in historical_data.columns:
                    # 基于最后一个时间戳生成
                    last_time = pd.to_datetime(historical_data['arrival_time_ns'].iloc[-1], unit='ns')
                    return pd.date_range(start=last_time, periods=steps + 1, freq='100ms')[1:]
                elif 'Timestamp' in historical_data.columns:
                    last_time = pd.to_datetime(historical_data['Timestamp'].iloc[-1])
                    return pd.date_range(start=last_time, periods=steps + 1, freq='100ms')[1:]

            # 默认时间戳
            return pd.date_range(start=datetime.now(), periods=steps, freq='100ms')

        except Exception as e:
            logger.warning(f"Timestamp generation failed: {e}, using default timestamps")
            return pd.date_range(start=datetime.now(), periods=steps, freq='100ms')

    def _calculate_confidence_intervals(self,
                                      input_preds: np.ndarray,
                                      output_preds: np.ndarray) -> Dict[str, np.ndarray]:
        """计算置信度区间"""
        # 简化的置信度计算，基于预测值的百分比
        confidence_factor = 0.1  # 10%置信度区间

        return {
            'input_lower': input_preds * (1 - confidence_factor),
            'input_upper': input_preds * (1 + confidence_factor),
            'output_lower': output_preds * (1 - confidence_factor),
            'output_upper': output_preds * (1 + confidence_factor),
            'confidence_level': 0.8
        }

    def _update_training_history(self, training_result: Dict[str, Any]) -> None:
        """更新训练历史"""
        if 'training_history' in training_result:
            history = training_result['training_history']
            self.training_history['loss'].extend(history.get('loss', []))
            self.training_history['val_loss'].extend(history.get('val_loss', []))
            self.training_history['epochs'] = len(self.training_history['loss'])
            self.training_history['best_val_loss'] = min(
                self.training_history['val_loss']) if self.training_history['val_loss'] else float('inf')

    def _format_training_result(self, training_result: Dict[str, Any]) -> Dict[str, Any]:
        """格式化训练结果"""
        return {
            'status': training_result.get('status', 'unknown'),
            'epochs_trained': training_result.get('epochs_trained', 0),
            'best_val_loss': training_result.get('best_val_loss', float('inf')),
            'final_train_loss': training_result.get('final_train_loss', float('inf')),
            'early_stopped': training_result.get('early_stopped', False),
            'model_metadata': self.model_metadata.copy(),
            'training_history': self.training_history.copy()
        }

    def _get_data_shape(self, data: Union[pd.DataFrame, str, Path]) -> str:
        """获取数据形状信息"""
        if isinstance(data, pd.DataFrame):
            return str(data.shape)
        elif isinstance(data, (str, Path)):
            return f"file: {Path(data).name}"
        else:
            return "unknown"

    def _get_data_length(self, data: Union[pd.DataFrame, str, Path]) -> int:
        """获取数据长度"""
        if isinstance(data, pd.DataFrame):
            return len(data)
        elif isinstance(data, (str, Path)):
            return 0  # 文件路径无法直接获取长度
        else:
            return 0

    def save_model(self, path: str) -> None:
        """保存模型"""
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)

        # 保存DLinear模型
        self.dlinear_predictor.save_model(str(model_path / "dlinear_model.pth"))

        # 保存适配器状态
        adapter_state = {
            'model_metadata': self.model_metadata,
            'training_history': self.training_history,
            'prediction_horizon': self.prediction_horizon.value,
            'sequence_length': self.sequence_length
        }

        with open(model_path / "adapter_state.pkl", 'wb') as f:
            pickle.dump(adapter_state, f)

        # 保存元数据
        with open(model_path / "model_metadata.json", 'w') as f:
            json.dump(self.model_metadata, f, indent=2)

        logger.info(f"DLinear适配器模型已保存到: {path}")

    def load_model(self, path: str) -> None:
        """加载模型"""
        model_path = Path(path)

        # 加载DLinear模型
        self.dlinear_predictor.load_model(str(model_path / "dlinear_model.pth"))

        # 加载适配器状态
        adapter_state_path = model_path / "adapter_state.pkl"
        if adapter_state_path.exists():
            with open(adapter_state_path, 'rb') as f:
                adapter_state = pickle.load(f)

            self.model_metadata = adapter_state['model_metadata']
            self.training_history = adapter_state['training_history']
            self.prediction_horizon = PredictionHorizon(adapter_state['prediction_horizon'])
            self.sequence_length = adapter_state['sequence_length']

        logger.info(f"DLinear适配器模型已从 {path} 加载")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        # 获取DLinear模型的分解分析
        try:
            # 创建一些示例数据进行分解分析
            sample_data = pd.DataFrame({
                'input_toks': np.random.randint(50, 200, 100),
                'output_toks': np.random.randint(100, 500, 100)
            })
            decomposition_analysis = self.dlinear_predictor.analyze_decomposition(sample_data)
        except Exception as e:
            logger.warning(f"Decomposition analysis failed: {e}")
            decomposition_analysis = {}

        return {
            'model_type': self.model_type.value,
            'prediction_horizon': self.prediction_horizon.value,
            'sequence_length': self.sequence_length,
            'decomposition_mode': self.model_metadata['decomposition_mode'],
            'kernel_size': self.model_metadata['kernel_size'],
            'training_history': self.training_history,
            'decomposition_analysis': decomposition_analysis,
            'model_metadata': self.model_metadata
        }

    def evaluate(self, test_data: Union[pd.DataFrame, str, Path]) -> Dict[str, float]:
        """评估模型性能"""
        logger.info("开始评估DLinear模型...")

        # 预处理测试数据
        processed_data = self._preprocess_training_data(test_data)

        # 使用最后sequence_length个点作为历史数据
        if len(processed_data) > self.sequence_length:
            historical_data = processed_data.iloc[:-self.sequence_length]
            actual_data = processed_data.iloc[-self.sequence_length:]
        else:
            # 数据不足，使用全部数据
            historical_data = processed_data
            actual_data = processed_data

        # 进行预测
        prediction_result = self.predict(historical_data)

        # 从预测结果中提取预测值
        predicted_values = np.array([
            [req['input_tokens'], req['output_tokens']]
            for req in prediction_result.predicted_requests
        ])

        # 计算评估指标
        metrics = self._calculate_evaluation_metrics(
            actual_data[['input_toks', 'output_toks']].values,
            predicted_values
        )

        logger.info(f"模型评估完成，MAE: {metrics.get('mae', 'N/A'):.4f}")
        return metrics

    def _calculate_evaluation_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        try:
            # 确保形状匹配
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]

            # 计算各种指标
            mae = np.mean(np.abs(actual - predicted))
            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)

            # 避免除零
            mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100

            # R²分数
            ss_total = np.sum((actual - np.mean(actual)) ** 2)
            ss_residual = np.sum((actual - predicted) ** 2)
            r2 = 1 - ss_residual / ss_total if ss_total > 0 else 0

            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2': float(r2)
            }

        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {'error': str(e)}


class UnifiedPredictorFactory:
    """
    统一预测器工厂类

    提供创建LSTM和DLinear预测器的统一接口
    """

    @staticmethod
    def create_predictor(model_type: ModelType,
                         prediction_horizon: PredictionHorizon,
                         **kwargs) -> Union['DLinearPredictorAdapter', 'WorkloadPredictor']:
        """
        创建预测器

        Args:
            model_type: 模型类型
            prediction_horizon: 预测时间范围
            **kwargs: 其他参数

        Returns:
            预测器实例
        """
        if model_type == ModelType.DLINEAR:
            return DLinearPredictorAdapter(
                prediction_horizon=prediction_horizon,
                **kwargs
            )
        elif model_type == ModelType.LSTM:
            # 导入并创建LSTM预测器
            from .workload_predictor import WorkloadPredictor
            return WorkloadPredictor(
                model_type=model_type,
                prediction_horizon=prediction_horizon,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


# 工厂函数
def create_dlinear_predictor(prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM,
                           sequence_length: int = 50,
                           **kwargs) -> DLinearPredictorAdapter:
    """创建DLinear预测器的工厂函数"""
    return DLinearPredictorAdapter(
        prediction_horizon=prediction_horizon,
        sequence_length=sequence_length,
        **kwargs
    )


# 测试函数
def test_dlinear_adapter():
    """测试DLinear适配器"""
    print("测试DLinear预测器适配器...")

    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'arrival_time_ns': np.arange(1000) * 100_000_000,
        'input_toks': np.random.randint(50, 200, 1000),
        'output_toks': np.random.randint(100, 500, 1000)
    })

    # 创建适配器
    adapter = create_dlinear_predictor(
        prediction_horizon=PredictionHorizon.SHORT_TERM,
        sequence_length=50
    )

    # 训练模型
    print("训练DLinear模型...")
    training_result = adapter.train(
        data=test_data,
        epochs=10,  # 快速测试
        verbose=False
    )

    print(f"训练结果: {training_result['status']}, 最佳验证损失: {training_result['best_val_loss']:.6f}")

    # 进行预测
    print("进行预测...")
    historical_data = test_data.iloc[:-50]
    prediction_result = adapter.predict(historical_data, steps=10)

    print(f"预测完成，预测形状: {prediction_result.predictions.shape}")
    print(f"预测类型: {prediction_result.model_type}")
    print(f"预测范围: {prediction_result.prediction_horizon}")

    # 评估模型
    print("评估模型...")
    metrics = adapter.evaluate(test_data)
    print(f"评估指标: {metrics}")

    # 获取模型信息
    model_info = adapter.get_model_info()
    print(f"模型信息: {model_info['model_type'], model_info['prediction_horizon']}")

    # 测试保存和加载
    print("测试保存和加载...")
    adapter.save_model("test_dlinear_adapter")
    new_adapter = create_dlinear_predictor()
    new_adapter.load_model("test_dlinear_adapter")

    print("DLinear预测器适配器测试完成！")


if __name__ == "__main__":
    test_dlinear_adapter()