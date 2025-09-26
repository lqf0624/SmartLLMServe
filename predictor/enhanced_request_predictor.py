"""
增强版请求预测器

集成可扩展模型接口，支持：
1. 动态模型注册和管理
2. GPU设备支持
3. 新模型的轻松集成
4. 模型性能比较
5. 统一的预测接口

这是为GPU环境设计的版本，可以很容易地添加新的预测模型。
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union
import logging
from .model_interface import (
    ModelConfig, ModelManager, ModelType, ModelDevice,
    PredictionResult, TrainingResult, model_registry
)
from .custom_models import TransformerPredictor, GRUPredictor
from .lstm_model import LSTMPredictor
from .dlinear_predictor_adapter import DLinearPredictorAdapter
from .workload_predictor import PredictionHorizon

logger = logging.getLogger(__name__)


class EnhancedRequestPredictor:
    """增强版请求预测器 - 支持可扩展模型接口"""

    def __init__(self, device: Union[str, ModelDevice] = None):
        """
        初始化增强版预测器

        Args:
            device: 计算设备，自动检测最佳设备
        """
        # 自动选择最佳设备
        if device is None:
            if torch.cuda.is_available():
                device = ModelDevice.CUDA
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = ModelDevice.MPS
            else:
                device = ModelDevice.CPU

        if isinstance(device, str):
            device = ModelDevice(device)

        self.device = device
        self.model_manager = ModelManager()
        self.supported_models = self._initialize_supported_models()

        logger.info(f"EnhancedRequestPredictor initialized on {device.value}")

    def _initialize_supported_models(self) -> Dict[str, ModelConfig]:
        """初始化支持的模型配置"""
        return {
            'lstm': ModelConfig(
                model_type=ModelType.LSTM,
                model_name='lstm_default',
                device=self.device,
                sequence_length=50,
                prediction_horizon=10,
                epochs=30,
                learning_rate=0.001,
                batch_size=32
            ),
            'dlinear': ModelConfig(
                model_type=ModelType.DLINEAR,
                model_name='dlinear_default',
                device=self.device,
                sequence_length=50,
                prediction_horizon=10,
                epochs=100,
                learning_rate=0.001,
                early_stopping=True,
                patience=10
            ),
            'transformer': ModelConfig(
                model_type=ModelType.TRANSFORMER,
                model_name='transformer_default',
                device=self.device,
                sequence_length=50,
                prediction_horizon=10,
                epochs=50,
                learning_rate=0.0005,
                batch_size=16,
                custom_params={
                    'd_model': 64,
                    'nhead': 4,
                    'num_layers': 2
                }
            ),
            'gru': ModelConfig(
                model_type=ModelType.GRU,
                model_name='gru_default',
                device=self.device,
                sequence_length=30,
                prediction_horizon=5,
                epochs=30,
                learning_rate=0.001,
                custom_params={
                    'hidden_size': 64,
                    'num_layers': 2
                }
            )
        }

    def register_existing_models(self):
        """注册现有的LSTM和DLinear模型到新接口"""
        from .lstm_model import LSTMPredictor
        from .dlinear_predictor_adapter import DLinearPredictorAdapter

        # 注册LSTM适配器
        class LSTMPredictorAdapter(LSTMPredictor):
            def __init__(self, config: ModelConfig):
                # 保持原有的初始化方式
                super().__init__(
                    sequence_length=config.sequence_length,
                    prediction_horizon=config.prediction_horizon,
                    hidden_size=config.custom_params.get('hidden_size', 64),
                    num_layers=config.custom_params.get('num_layers', 2),
                    device=config.device.value,
                    learning_rate=config.learning_rate
                )
                self.config = config

            def predict(self, data: pd.DataFrame, steps: int = None) -> PredictionResult:
                if steps is None:
                    steps = self.config.prediction_horizon

                # 使用原有的predict方法
                result = super().predict(data, steps=steps)

                # 转换为标准格式
                return PredictionResult(
                    timestamps=[result.predictions[0]] if result.predictions else [],
                    request_tokens=[result.predictions[0]] if result.predictions else [],
                    response_tokens=[result.predictions[0]] if result.predictions else [],
                    confidence=result.get('confidence', 0.8),
                    model_name=self.config.model_name,
                    prediction_time=0.01,
                    metadata={'raw_result': result.__dict__}
                )

            def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> TrainingResult:
                # 使用原有的train方法
                result = super().train(data, validation_split)

                return TrainingResult(
                    success=result.get('success', False),
                    training_time=result.get('training_time', 0),
                    final_loss=result.get('train_loss', float('inf')),
                    best_loss=result.get('val_loss', float('inf')),
                    epochs_trained=result.get('epochs_trained', 0),
                    early_stopped=False,
                    metadata=result
                )

        # 注册DLinear适配器
        class DLinearPredictorAdapterNew(DLinearPredictorAdapter):
            def __init__(self, config: ModelConfig):
                horizon_map = {
                    5: PredictionHorizon.SHORT_TERM,
                    10: PredictionHorizon.SHORT_TERM,
                    30: PredictionHorizon.MEDIUM_TERM,
                    60: PredictionHorizon.LONG_TERM
                }

                horizon = horizon_map.get(config.prediction_horizon, PredictionHorizon.SHORT_TERM)

                super().__init__(
                    prediction_horizon=horizon,
                    sequence_length=config.sequence_length,
                    device=config.device.value
                )
                self.config = config

            def predict(self, data: pd.DataFrame, steps: int = None) -> PredictionResult:
                if steps is None:
                    steps = self.config.prediction_horizon

                # 使用原有的predict_future方法
                result = super().predict_future(data, steps=steps)

                # 转换为标准格式
                if hasattr(result, 'predictions') and result.predictions:
                    return PredictionResult(
                        timestamps=[0.0],  # DLinear不直接预测时间戳
                        request_tokens=[int(result.predictions[0])],
                        response_tokens=[int(result.predictions[0])],
                        confidence=result.get('confidence', 0.7),
                        model_name=self.config.model_name,
                        prediction_time=0.005,
                        metadata={'raw_result': result.__dict__}
                    )
                else:
                    return PredictionResult(
                        timestamps=[],
                        request_tokens=[],
                        response_tokens=[],
                        confidence=0.0,
                        model_name=self.config.model_name,
                        prediction_time=0.0,
                        metadata={'error': 'No predictions available'}
                    )

            def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> TrainingResult:
                # 使用原有的train方法
                result = super().train(data, validation_split)

                return TrainingResult(
                    success=True,
                    training_time=result.get('training_time', 0),
                    final_loss=result.get('final_train_loss', float('inf')),
                    best_loss=result.get('best_val_loss', float('inf')),
                    epochs_trained=result.get('epochs_trained', 0),
                    early_stopped=result.get('early_stopped', False),
                    metadata=result.__dict__
                )

        # 注册适配器
        from .model_interface import register_model

        @register_model(ModelType.LSTM)
        class RegisteredLSTM(LSTMPredictorAdapter):
            pass

        @register_model(ModelType.DLINEAR)
        class RegisteredDLinear(DLinearPredictorAdapterNew):
            pass

        logger.info("Registered existing LSTM and DLinear models to enhanced interface")

    def add_model(self, model_name: str, model_config: ModelConfig = None) -> bool:
        """
        添加预测模型

        Args:
            model_name: 模型名称 ('lstm', 'dlinear', 'transformer', 'gru')
            model_config: 自定义模型配置，为None时使用默认配置

        Returns:
            是否成功添加
        """
        try:
            if model_config is None:
                if model_name not in self.supported_models:
                    logger.error(f"Unsupported model: {model_name}")
                    return False
                model_config = self.supported_models[model_name]

            # 确保模型类已注册
            if model_config.model_type not in model_registry.get_available_model_types():
                if model_config.model_type in [ModelType.LSTM, ModelType.DLINEAR]:
                    self.register_existing_models()
                else:
                    logger.error(f"Model type {model_config.model_type.value} not registered")
                    return False

            # 添加模型到管理器
            model = self.model_manager.add_model(model_config)
            logger.info(f"Added model: {model_name} ({model_config.model_type.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to add model {model_name}: {e}")
            return False

    def add_models(self, model_names: List[str]) -> Dict[str, bool]:
        """
        批量添加模型

        Args:
            model_names: 模型名称列表

        Returns:
            添加结果字典
        """
        results = {}
        for model_name in model_names:
            results[model_name] = self.add_model(model_name)
        return results

    def train_models(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, TrainingResult]:
        """
        训练所有已添加的模型

        Args:
            data: 训练数据
            validation_split: 验证集比例

        Returns:
            训练结果字典
        """
        if not self.model_manager.active_models:
            logger.warning("No models added. Call add_model() first.")
            return {}

        logger.info(f"Training {len(self.model_manager.active_models)} models...")
        return self.model_manager.train_all_models(data, validation_split)

    def predict_next_request(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        预测下一个请求 - 使用所有已训练的模型

        Args:
            historical_data: 历史数据

        Returns:
            预测结果
        """
        if len(historical_data) < 10:
            return {
                'error': 'Insufficient historical data for prediction',
                'min_required': 10,
                'actual': len(historical_data)
            }

        # 使用模型管理器进行预测
        predictions = self.model_manager.predict_all(historical_data, steps=1)

        # 格式化输出以保持兼容性
        formatted_predictions = {}
        for model_name, pred in predictions.items():
            if pred.metadata.get('error'):
                formatted_predictions[model_name] = {
                    'error': pred.metadata['error']
                }
            else:
                formatted_predictions[model_name] = {
                    'timestamp': pred.timestamps[0] if pred.timestamps else 0,
                    'request_tokens': pred.request_tokens[0] if pred.request_tokens else 0,
                    'response_tokens': pred.response_tokens[0] if pred.response_tokens else 0,
                    'confidence': pred.confidence,
                    'method': model_name
                }

        # 添加摘要信息
        summary = self._generate_summary(historical_data, predictions)

        return {
            'predictions': formatted_predictions,
            'summary': summary,
            'model_info': {name: model.get_model_info()
                          for name, model in self.model_manager.active_models.items()}
        }

    def predict_multiple_requests(self, historical_data: pd.DataFrame,
                                 num_requests: int = 5) -> Dict[str, Any]:
        """
        预测多个连续请求

        Args:
            historical_data: 历史数据
            num_requests: 预测请求数量

        Returns:
            多个请求的预测结果
        """
        if len(historical_data) < 10:
            return {
                'error': 'Insufficient historical data for prediction',
                'min_required': 10,
                'actual': len(historical_data)
            }

        predictions = self.model_manager.predict_all(historical_data, steps=num_requests)

        # 格式化多步预测结果
        formatted_predictions = {}
        for model_name, pred in predictions.items():
            if pred.metadata.get('error'):
                formatted_predictions[model_name] = {
                    'error': pred.metadata['error']
                }
            else:
                formatted_predictions[model_name] = [
                    {
                        'timestamp': pred.timestamps[i] if i < len(pred.timestamps) else 0,
                        'request_tokens': pred.request_tokens[i] if i < len(pred.request_tokens) else 0,
                        'response_tokens': pred.response_tokens[i] if i < len(pred.response_tokens) else 0,
                        'confidence': pred.confidence,
                        'step': i + 1
                    }
                    for i in range(num_requests)
                ]

        return {
            'predictions': formatted_predictions,
            'num_requests': num_requests,
            'historical_data_size': len(historical_data)
        }

    def compare_models(self, historical_data: pd.DataFrame, steps: int = 1) -> Dict[str, Any]:
        """
        比较所有模型的性能

        Args:
            historical_data: 历史数据
            steps: 预测步数

        Returns:
            模型比较结果
        """
        return self.model_manager.compare_models(historical_data, steps)

    def get_available_models(self) -> List[str]:
        """获取可用的模型类型"""
        return list(self.supported_models.keys())

    def get_active_models(self) -> List[str]:
        """获取已激活的模型"""
        return list(self.model_manager.active_models.keys())

    def save_model(self, model_name: str, path: str) -> bool:
        """保存指定模型"""
        model = self.model_manager.active_models.get(model_name)
        if model:
            return model.save_model(path)
        return False

    def load_model(self, model_name: str, path: str, model_config: ModelConfig) -> bool:
        """加载指定模型"""
        try:
            model = self.model_manager.add_model(model_config)
            return model.load_model(path)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def switch_device(self, device: Union[str, ModelDevice]):
        """切换所有模型的设备"""
        if isinstance(device, str):
            device = ModelDevice(device)

        for model in self.model_manager.active_models.values():
            model.to_device(device)

        self.device = device
        logger.info(f"Switched all models to {device.value}")

    def _generate_summary(self, historical_data: pd.DataFrame,
                         predictions: Dict[str, PredictionResult]) -> Dict[str, Any]:
        """生成预测摘要"""
        summary = {
            'historical_data_size': len(historical_data),
            'last_request': {
                'timestamp': historical_data['Timestamp'].iloc[-1],
                'request_tokens': int(historical_data['Request tokens'].iloc[-1]),
                'response_tokens': int(historical_data['Response tokens'].iloc[-1])
            },
            'models_used': len(predictions),
            'next_prediction_summary': {}
        }

        # 计算预测统计
        valid_predictions = [p for p in predictions.values() if not p.metadata.get('error')]
        if valid_predictions:
            confidences = [p.confidence for p in valid_predictions]
            request_tokens = [p.request_tokens[0] if p.request_tokens else 0 for p in valid_predictions]
            response_tokens = [p.response_tokens[0] if p.response_tokens else 0 for p in valid_predictions]

            summary['next_prediction_summary'] = {
                'avg_confidence': np.mean(confidences),
                'avg_request_tokens': int(np.mean(request_tokens)),
                'avg_response_tokens': int(np.mean(response_tokens)),
                'best_model': max(predictions.keys(),
                                 key=lambda k: predictions[k].confidence if not predictions[k].metadata.get('error') else 0)
            }

        return summary


# 快速启动函数
def create_enhanced_predictor(models: List[str] = None, device: str = None) -> EnhancedRequestPredictor:
    """
    快速创建增强版预测器

    Args:
        models: 要添加的模型列表，为None时添加默认模型
        device: 指定设备，为None时自动选择

    Returns:
        配置好的预测器实例
    """
    if models is None:
        models = ['lstm', 'dlinear']  # 默认模型

    predictor = EnhancedRequestPredictor(device=device)

    # 添加模型
    for model_name in models:
        predictor.add_model(model_name)

    return predictor


# GPU环境使用示例
def gpu_usage_example():
    """
    GPU环境使用示例代码

    这段代码可以保存为.py文件在GPU服务器上运行：
    """
    example_code = '''
    # gpu_predictor_example.py - GPU环境下的预测器使用示例

    import pandas as pd
    import torch
    from predictor.enhanced_request_predictor import create_enhanced_predictor
    from predictor.model_interface import ModelConfig, ModelType, ModelDevice

    def main():
        # 1. 检查GPU可用性
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")

        # 2. 创建增强版预测器（自动选择GPU）
        predictor = create_enhanced_predictor(
            models=['lstm', 'dlinear', 'transformer', 'gru'],
            device='cuda'  # 可以指定'cuda', 'cpu', 'mps'
        )

        print(f"Predictor device: {predictor.device}")
        print(f"Active models: {predictor.get_active_models()}")

        # 3. 加载数据
        data = pd.read_csv('dataset/BurstGPT_1_cleaned.csv', nrows=10000)
        print(f"Loaded data: {data.shape}")

        # 4. 训练模型
        print("Training models...")
        training_results = predictor.train_models(data, validation_split=0.2)

        print("Training Results:")
        for model_name, result in training_results.items():
            status = "Success" if result.success else "Failed"
            print(f"  {model_name}: {status} (Loss: {result.final_loss:.4f})")

        # 5. 预测下一个请求
        print("\\nPredicting next request...")
        historical_data = data.iloc[:8000]
        prediction = predictor.predict_next_request(historical_data)

        print("Next Request Predictions:")
        for model_name, pred in prediction.get('predictions', {}).items():
            if 'error' not in pred:
                print(f"  {model_name}:")
                print(f"    Timestamp: {pred['timestamp']:.3f}")
                print(f"    Request tokens: {pred['request_tokens']}")
                print(f"    Response tokens: {pred['response_tokens']}")
                print(f"    Confidence: {pred['confidence']:.3f}")
            else:
                print(f"  {model_name}: {pred['error']}")

        # 6. 比较模型性能
        print("\\nComparing models...")
        comparison = predictor.compare_models(historical_data, steps=1)

        if 'performance_summary' in comparison:
            summary = comparison['performance_summary']
            print(f"Best model: {summary.get('best_model', 'Unknown')}")
            print(f"Average confidence: {summary.get('avg_confidence', 0):.3f}")

        # 7. 保存最佳模型
        best_model_name = comparison.get('performance_summary', {}).get('best_model')
        if best_model_name:
            model_path = f"models/{best_model_name}_gpu_trained.pth"
            if predictor.save_model(best_model_name, model_path):
                print(f"\\nSaved best model to: {model_path}")

        # 8. 测试多步预测
        print("\\nTesting multi-step prediction...")
        multi_pred = predictor.predict_multiple_requests(historical_data, num_requests=3)
        print(f"Multi-step prediction for {multi_pred.get('num_requests', 0)} requests completed")

    if __name__ == "__main__":
        main()
    '''

    return example_code