"""
可扩展预测模型接口

为GPU环境下的新模型集成提供标准接口，支持：
1. 动态模型注册和加载
2. 统一的训练和预测接口
3. 模型配置管理
4. GPU设备支持
5. 模型性能评估
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
import torch
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelDevice(Enum):
    """支持的设备类型"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class ModelType(Enum):
    """模型类型枚举"""
    LSTM = "lstm"
    DLINEAR = "dlinear"
    TRANSFORMER = "transformer"
    GRU = "gru"
    CNN = "cnn"
    PROPHET = "prophet"
    CUSTOM = "custom"  # 自定义模型


@dataclass
class ModelConfig:
    """模型配置基类"""
    model_type: ModelType
    model_name: str
    device: ModelDevice = ModelDevice.CPU
    sequence_length: int = 50
    prediction_horizon: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping: bool = True
    patience: int = 10

    # 自定义参数
    custom_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class PredictionResult:
    """标准化预测结果"""
    timestamps: List[float]
    request_tokens: List[int]
    response_tokens: List[int]
    confidence: float
    model_name: str
    prediction_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TrainingResult:
    """标准化训练结果"""
    success: bool
    training_time: float
    final_loss: float
    best_loss: float
    epochs_trained: int
    early_stopped: bool
    model_path: str = None
    error_message: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BasePredictor(ABC):
    """预测模型抽象基类"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.model_name
        self.model_type = config.model_type
        self.device = torch.device(config.device.value)
        self.is_trained = False
        self.model = None
        self.training_history = {}

        logger.info(f"Initialized {self.model_name} predictor on {self.device}")

    @abstractmethod
    def prepare_data(self, data: pd.DataFrame) -> Any:
        """
        准备训练数据

        Args:
            data: 原始数据DataFrame

        Returns:
            处理后的数据格式（模型特定）
        """
        pass

    @abstractmethod
    def build_model(self) -> Any:
        """
        构建模型架构

        Returns:
            构建好的模型
        """
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> TrainingResult:
        """
        训练模型

        Args:
            data: 训练数据
            validation_split: 验证集比例

        Returns:
            训练结果
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, steps: int = None) -> PredictionResult:
        """
        预测未来时间步

        Args:
            data: 历史数据
            steps: 预测步数

        Returns:
            预测结果
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> bool:
        """
        保存模型

        Args:
            path: 保存路径

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> bool:
        """
        加载模型

        Args:
            path: 模型路径

        Returns:
            是否成功
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }

    def to_device(self, device: Union[str, ModelDevice, torch.device]):
        """
        将模型移动到指定设备

        Args:
            device: 目标设备
        """
        if isinstance(device, str):
            device = ModelDevice(device)
        if isinstance(device, ModelDevice):
            device = torch.device(device.value)

        self.device = device
        if self.model is not None:
            self.model.to(device)
        logger.info(f"Moved {self.model_name} to {device}")

    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据

        Args:
            data: 输入数据

        Returns:
            是否有效
        """
        required_columns = ['Timestamp', 'Request tokens', 'Response tokens']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        if len(data) < self.config.sequence_length:
            logger.error(f"Insufficient data: need {self.config.sequence_length}, got {len(data)}")
            return False

        return True


class ModelRegistry:
    """模型注册表"""

    def __init__(self):
        self._models: Dict[str, BasePredictor] = {}
        self._model_classes: Dict[ModelType, type] = {}
        logger.info("ModelRegistry initialized")

    def register_model_class(self, model_type: ModelType, model_class: type):
        """
        注册模型类

        Args:
            model_type: 模型类型
            model_class: 模型类（必须继承BasePredictor）
        """
        if not issubclass(model_class, BasePredictor):
            raise ValueError(f"Model class must inherit from BasePredictor")

        self._model_classes[model_type] = model_class
        logger.info(f"Registered model class: {model_type.value} -> {model_class.__name__}")

    def create_model(self, config: ModelConfig) -> BasePredictor:
        """
        创建模型实例

        Args:
            config: 模型配置

        Returns:
            模型实例
        """
        if config.model_type not in self._model_classes:
            raise ValueError(f"Model type {config.model_type.value} not registered")

        model_class = self._model_classes[config.model_type]
        model = model_class(config)

        self._models[config.model_name] = model
        logger.info(f"Created model instance: {config.model_name}")
        return model

    def get_model(self, model_name: str) -> Optional[BasePredictor]:
        """
        获取模型实例

        Args:
            model_name: 模型名称

        Returns:
            模型实例或None
        """
        return self._models.get(model_name)

    def list_models(self) -> List[str]:
        """
        列出所有已注册的模型名称

        Returns:
            模型名称列表
        """
        return list(self._models.keys())

    def remove_model(self, model_name: str) -> bool:
        """
        移除模型

        Args:
            model_name: 模型名称

        Returns:
            是否成功
        """
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Removed model: {model_name}")
            return True
        return False

    def get_available_model_types(self) -> List[ModelType]:
        """
        获取可用的模型类型

        Returns:
            模型类型列表
        """
        return list(self._model_classes.keys())


# 全局模型注册表实例
model_registry = ModelRegistry()


def register_model(model_type: ModelType):
    """
    模型注册装饰器

    Usage:
        @register_model(ModelType.CUSTOM)
        class MyCustomModel(BasePredictor):
            pass
    """
    def decorator(model_class):
        model_registry.register_model_class(model_type, model_class)
        return model_class
    return decorator


class ModelManager:
    """模型管理器 - 提供高级管理功能"""

    def __init__(self, registry: ModelRegistry = None):
        self.registry = registry or model_registry
        self.active_models: Dict[str, BasePredictor] = {}

    def add_model(self, config: ModelConfig) -> BasePredictor:
        """
        添加并初始化模型

        Args:
            config: 模型配置

        Returns:
            模型实例
        """
        model = self.registry.create_model(config)
        self.active_models[config.model_name] = model
        return model

    def train_all_models(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, TrainingResult]:
        """
        训练所有模型

        Args:
            data: 训练数据
            validation_split: 验证集比例

        Returns:
            训练结果字典
        """
        results = {}

        for model_name, model in self.active_models.items():
            try:
                logger.info(f"Training {model_name}...")
                result = model.train(data, validation_split)
                results[model_name] = result
                logger.info(f"Training {model_name} completed: {'Success' if result.success else 'Failed'}")
            except Exception as e:
                logger.error(f"Training {model_name} failed: {e}")
                results[model_name] = TrainingResult(
                    success=False,
                    training_time=0,
                    final_loss=float('inf'),
                    best_loss=float('inf'),
                    epochs_trained=0,
                    early_stopped=False,
                    error_message=str(e)
                )

        return results

    def predict_all(self, data: pd.DataFrame, steps: int = None) -> Dict[str, PredictionResult]:
        """
        使用所有模型进行预测

        Args:
            data: 历史数据
            steps: 预测步数

        Returns:
            预测结果字典
        """
        results = {}

        for model_name, model in self.active_models.items():
            if model.is_trained:
                try:
                    prediction = model.predict(data, steps)
                    results[model_name] = prediction
                except Exception as e:
                    logger.error(f"Prediction {model_name} failed: {e}")
                    results[model_name] = PredictionResult(
                        timestamps=[],
                        request_tokens=[],
                        response_tokens=[],
                        confidence=0.0,
                        model_name=model_name,
                        prediction_time=0,
                        metadata={'error': str(e)}
                    )

        return results

    def compare_models(self, data: pd.DataFrame, steps: int = None) -> Dict[str, Any]:
        """
        比较所有模型的性能

        Args:
            data: 测试数据
            steps: 预测步数

        Returns:
            比较结果
        """
        predictions = self.predict_all(data, steps)

        comparison = {
            'model_count': len(predictions),
            'predictions': predictions,
            'model_info': {name: model.get_model_info() for name, model in self.active_models.items()},
            'performance_summary': {}
        }

        # 计算基本统计
        valid_predictions = {k: v for k, v in predictions.items()
                           if not v.metadata.get('error')}

        if valid_predictions:
            confidences = [pred.confidence for pred in valid_predictions.values()]
            comparison['performance_summary'] = {
                'avg_confidence': np.mean(confidences),
                'max_confidence': np.max(confidences),
                'min_confidence': np.min(confidences),
                'best_model': max(valid_predictions.keys(),
                                 key=lambda k: valid_predictions[k].confidence)
            }

        return comparison