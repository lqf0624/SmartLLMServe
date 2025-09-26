"""
Main workload predictor interface.

This module provides a unified interface for different prediction models,
including LSTM, statistical models, and ensemble methods.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import logging
from enum import Enum
import warnings

from .lstm_model import LSTMPredictor
from .feature_extractor import FeatureExtractor, FeatureType
from .data_preprocessor import DataPreprocessor, DataFormat, create_data_preprocessor
# 延迟导入以避免循环导入

logger = logging.getLogger(__name__)


class PredictionHorizon(Enum):
    """Prediction time horizons."""
    SHORT_TERM = "short_term"    # 100ms - 1s (10 steps)
    MEDIUM_TERM = "medium_term"  # 1s - 10s (30 steps)
    LONG_TERM = "long_term"      # 10s - 60s (60 steps)
    EXTENDED_TERM = "extended_term"  # 60s - 120s (120 steps)


class ModelType(Enum):
    """Available prediction models."""
    LSTM = "lstm"
    DLINEAR = "dlinear"
    ARIMA = "arima"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


class PredictionResult:
    """
    Standardized prediction result format for scheduling decisions.

    This class provides a unified interface that can be used by:
    - LSTM models (current)
    - Large language models (future)
    - Statistical models
    """

    def __init__(self,
                 predicted_requests: List[Dict[str, Any]],
                 confidence: float,
                 prediction_metadata: Dict[str, Any]):
        """
        Initialize prediction result.

        Args:
            predicted_requests: List of predicted request attributes
            confidence: Overall confidence score (0-1)
            prediction_metadata: Additional metadata
        """
        self.predicted_requests = predicted_requests
        self.confidence = confidence
        self.prediction_metadata = prediction_metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'predicted_requests': self.predicted_requests,
            'confidence': self.confidence,
            'metadata': self.prediction_metadata
        }

    def get_resource_requirements(self) -> Dict[str, float]:
        """Extract resource requirements for scheduling."""
        if not self.predicted_requests:
            return {'memory': 0.0, 'compute': 0.0, 'tokens': 0.0}

        total_input_tokens = sum(req.get('input_tokens', 0) for req in self.predicted_requests)
        total_output_tokens = sum(req.get('output_tokens', 0) for req in self.predicted_requests)

        # Estimate resource requirements based on token counts
        memory_mb = (total_input_tokens + total_output_tokens) * 0.001  # Rough estimate
        compute_units = (total_input_tokens + total_output_tokens) * 0.0001  # Rough estimate

        return {
            'memory_mb': memory_mb,
            'compute_units': compute_units,
            'total_tokens': total_input_tokens + total_output_tokens,
            'request_count': len(self.predicted_requests)
        }

    def get_timing_info(self) -> Dict[str, Any]:
        """Extract timing information for scheduling."""
        if not self.predicted_requests:
            return {'start_time': 0, 'end_time': 0, 'duration': 0}

        timestamps = [req.get('arrival_time_ns', 0) for req in self.predicted_requests]
        return {
            'start_time_ns': min(timestamps),
            'end_time_ns': max(timestamps),
            'duration_ns': max(timestamps) - min(timestamps),
            'time_range_ms': (max(timestamps) - min(timestamps)) / 1e6
        }

    def get_burst_analysis(self) -> Dict[str, Any]:
        """Extract burst pattern analysis for scheduling."""
        if not self.predicted_requests:
            return {'burst_level': 'low', 'intensity': 0.0}

        # Analyze request density and patterns
        timestamps = [req.get('arrival_time_ns', 0) for req in self.predicted_requests]
        if len(timestamps) < 2:
            return {'burst_level': 'low', 'intensity': 0.0}

        time_diffs = np.diff(timestamps)
        avg_interval = np.mean(time_diffs)
        max_interval = np.max(time_diffs)

        # Determine burst level
        if avg_interval < 50e6:  # < 50ms average interval
            burst_level = 'high'
        elif avg_interval < 200e6:  # < 200ms average interval
            burst_level = 'medium'
        else:
            burst_level = 'low'

        intensity = 1.0 / (avg_interval / 1e9 + 1e-9)  # requests per second

        return {
            'burst_level': burst_level,
            'intensity': intensity,
            'avg_interval_ms': avg_interval / 1e6,
            'max_interval_ms': max_interval / 1e6
        }


class BasePredictor(ABC):
    """Abstract base class for all predictors."""

    @abstractmethod
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the predictor."""
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, steps: int) -> PredictionResult:
        """Make predictions."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the predictor."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the predictor."""
        pass


class WorkloadPredictor:
    """
    Main workload predictor with multiple model support.

    This class provides a unified interface for different prediction models,
    automatic model selection, and ensemble prediction capabilities.
    """

    def __init__(self,
                 model_type: ModelType = ModelType.LSTM,
                 prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM,
                 sequence_length: int = 50,
                 feature_window: int = 10,
                 enable_ensemble: bool = False,
                 ensemble_strategy: Optional[str] = "PERFORMANCE_BASED",
                 device: Optional[str] = None):
        """
        Initialize workload predictor.

        Args:
            model_type: Type of prediction model
            prediction_horizon: Prediction time horizon
            sequence_length: Length of input sequence for time series models
            feature_window: Window size for feature extraction
            enable_ensemble: Whether to use ensemble prediction
            ensemble_strategy: Strategy for ensemble weight optimization
            device: Force specific device ('cpu', 'cuda')
        """
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length
        self.feature_window = feature_window
        self.enable_ensemble = enable_ensemble
        self.ensemble_strategy = ensemble_strategy
        self.device = device

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            window_size=feature_window,
            normalize=True
        )

        # Initialize data preprocessor
        self.data_preprocessor = create_data_preprocessor()

        # Initialize prediction models
        self.models = {}
        self.active_models = []

        # Performance tracking
        self.prediction_history = []
        self.accuracy_metrics = {}

        # Model configuration based on horizon
        self.horizon_config = {
            PredictionHorizon.SHORT_TERM: {'steps': 10, 'freq': '100ms', 'description': '短期预测 (1秒)'},
            PredictionHorizon.MEDIUM_TERM: {'steps': 30, 'freq': '1s', 'description': '中期预测 (3秒)'},
            PredictionHorizon.LONG_TERM: {'steps': 60, 'freq': '10s', 'description': '长期预测 (6秒)'},
            PredictionHorizon.EXTENDED_TERM: {'steps': 120, 'freq': '10s', 'description': '扩展预测 (12秒)'}
        }

        # Initialize models
        self._initialize_models()

        logger.info(f"WorkloadPredictor initialized with {model_type.value} model")

    def _initialize_models(self):
        """Initialize prediction models."""
        # Get configuration for current horizon
        config = self.horizon_config[self.prediction_horizon]
        prediction_steps = config['steps']

        # Initialize LSTM model
        if self.model_type in [ModelType.LSTM, ModelType.ENSEMBLE]:
            self.models[ModelType.LSTM] = LSTMPredictor(
                sequence_length=self.sequence_length,
                prediction_horizon=prediction_steps,
                hidden_size=64,
                num_layers=2,
                learning_rate=0.001,
                batch_size=32,
                epochs=100,
                device=self.device
            )
            self.active_models.append(ModelType.LSTM)

        # Initialize DLinear model
        if self.model_type in [ModelType.DLINEAR, ModelType.ENSEMBLE]:
            from .dlinear_predictor_adapter import DLinearPredictorAdapter
            self.models[ModelType.DLINEAR] = DLinearPredictorAdapter(
                prediction_horizon=self.prediction_horizon,
                sequence_length=self.sequence_length,
                device=self.device or 'cpu'
            )
            self.active_models.append(ModelType.DLINEAR)

        # Initialize statistical models (placeholder for future implementation)
        if self.model_type in [ModelType.ARIMA, ModelType.ENSEMBLE]:
            # Placeholder for ARIMA implementation
            self.models[ModelType.ARIMA] = None
            if self.model_type == ModelType.ARIMA:
                self.active_models.append(ModelType.ARIMA)

        if self.model_type in [ModelType.PROPHET, ModelType.ENSEMBLE]:
            # Placeholder for Prophet implementation
            self.models[ModelType.PROPHET] = None
            if self.model_type == ModelType.PROPHET:
                self.active_models.append(ModelType.PROPHET)

        # Initialize ensemble weights if needed
        if self.enable_ensemble:
            self.ensemble_weights = self._initialize_ensemble_weights()

            # Initialize EnsemblePredictor for advanced ensemble functionality
            if len(self.active_models) >= 2:
                # 延迟导入以避免循环导入
                from .ensemble_predictor import EnsemblePredictor, WeightStrategy

                # 转换字符串为WeightStrategy枚举
                strategy_map = {
                    "STATIC": WeightStrategy.STATIC,
                    "PERFORMANCE_BASED": WeightStrategy.PERFORMANCE_BASED,
                    "DYNAMIC": WeightStrategy.DYNAMIC,
                    "ADAPTIVE": WeightStrategy.ADAPTIVE
                }
                weight_strategy = strategy_map.get(self.ensemble_strategy, WeightStrategy.PERFORMANCE_BASED)

                self.ensemble_predictor = EnsemblePredictor(
                    models=self.active_models,
                    weight_strategy=weight_strategy,
                    prediction_horizon=self.prediction_horizon,
                    sequence_length=self.sequence_length
                )
                logger.info(f"Advanced EnsemblePredictor initialized with strategy: {self.ensemble_strategy}")
            else:
                self.ensemble_predictor = None
                logger.warning("Insufficient models for advanced ensemble prediction")
        else:
            self.ensemble_weights = None
            self.ensemble_predictor = None

    def _initialize_ensemble_weights(self) -> Dict[ModelType, float]:
        """Initialize ensemble model weights."""
        weights = {}
        active_count = len(self.active_models)

        if active_count > 0:
            weight = 1.0 / active_count
            for model_type in self.active_models:
                weights[model_type] = weight

        return weights

    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the workload predictor.

        Args:
            data: Training data
            validation_split: Validation split ratio

        Returns:
            Training results
        """
        logger.info(f"Training {self.model_type.value} predictor...")

        results = {}

        # 数据预处理
        try:
            logger.info("开始训练数据预处理...")
            processed_data = self.data_preprocessor.preprocess_data(
                data,
                target_format=DataFormat.STANDARD
            )
            logger.info(f"训练数据预处理完成: {len(processed_data)} 条记录")
        except Exception as e:
            logger.error(f"训练数据预处理失败: {e}")
            # 使用原始数据继续，但记录警告
            processed_data = data
            logger.warning("使用原始数据继续训练，可能会影响准确性")

        # 数据验证
        if not self._validate_input_data(processed_data):
            raise ValueError("训练数据验证失败")

        # Extract features
        features = self.feature_extractor.extract_features(processed_data)
        logger.info(f"Extracted features shape: {features.shape}")

        # Train each active model
        for model_type in self.active_models:
            try:
                if model_type == ModelType.LSTM and self.models[model_type] is not None:
                    # Train LSTM model
                    history = self.models[model_type].train(processed_data, validation_split)
                    final_train_loss = float(history['train_loss'][-1]) if history and history.get('train_loss') else None
                    final_val_loss = float(history['val_loss'][-1]) if history and history.get('val_loss') else None

                    results[model_type.value] = {
                        'status': 'success',
                        'history': history,
                        'final_train_loss': final_train_loss,
                        'final_val_loss': final_val_loss
                    }
                    if final_train_loss is not None:
                        logger.info(f"LSTM training completed - Final train loss: {final_train_loss:.4f}")

                elif model_type == ModelType.DLINEAR and self.models[model_type] is not None:
                    # Train DLinear model
                    training_result = self.models[model_type].train(
                        processed_data,
                        validation_split=validation_split,
                        epochs=100,  # 使用默认epochs
                        verbose=False
                    )
                    results[model_type.value] = {
                        'status': training_result.get('status', 'unknown'),
                        'history': training_result.get('training_history', {}),
                        'final_train_loss': training_result.get('final_train_loss'),
                        'final_val_loss': training_result.get('best_val_loss'),
                        'epochs_trained': training_result.get('epochs_trained', 0)
                    }
                    logger.info(f"DLinear training completed - Best val loss: {results[model_type.value]['final_val_loss']:.4f}")

                elif model_type in [ModelType.ARIMA, ModelType.PROPHET]:
                    # Placeholder for statistical models
                    results[model_type.value] = {
                        'status': 'placeholder',
                        'message': f'{model_type.value} model not yet implemented'
                    }
                    logger.warning(f"{model_type.value} model not yet implemented")

            except Exception as e:
                logger.error(f"Training failed for {model_type.value}: {e}")
                results[model_type.value] = {
                    'status': 'failed',
                    'error': str(e)
                }

        # Train ensemble if enabled
        if self.enable_ensemble and len(self.active_models) > 1:
            results['ensemble'] = self._train_ensemble(data)

        return results

    def _train_ensemble(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble model using advanced EnsemblePredictor."""
        if self.ensemble_predictor is not None:
            try:
                # 使用高级集成预测器进行训练
                ensemble_result = self.ensemble_predictor.train(data, validation_split=0.2)

                # 更新权重以保持一致性
                if 'ensemble_weights' in ensemble_result:
                    # 将字符串格式的权重转换回ModelType格式
                    new_weights = {}
                    for model_str, weight in ensemble_result['ensemble_weights'].items():
                        for model_type in ModelType:
                            if model_type.value == model_str:
                                new_weights[model_type] = weight
                                break
                    self.ensemble_weights = new_weights

                logger.info("Advanced ensemble training completed")
                return ensemble_result

            except Exception as e:
                logger.error(f"Advanced ensemble training failed: {e}")
                # 降级到简单集成方法
                return self._train_simple_ensemble(data)
        else:
            # 使用简单集成方法
            return self._train_simple_ensemble(data)

    def _train_simple_ensemble(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Simple ensemble training fallback."""
        # 使用当前权重
        ensemble_result = {
            'status': 'success',
            'weights': {model_type.value: weight for model_type, weight in self.ensemble_weights.items()},
            'method': 'simple_weights',
            'active_models': [model.value for model in self.active_models]
        }
        logger.info("Simple ensemble training completed")
        return ensemble_result

    def predict(self, data: pd.DataFrame, steps: Optional[int] = None) -> PredictionResult:
        """
        Make predictions.

        Args:
            data: Historical data for prediction
            steps: Number of steps to predict (default from horizon)

        Returns:
            Prediction results
        """
        if not self.active_models:
            raise ValueError("No active models available for prediction")

        # 数据预处理
        try:
            logger.info("开始数据预处理...")
            processed_data = self.data_preprocessor.preprocess_data(
                data,
                target_format=DataFormat.STANDARD
            )
            logger.info(f"数据预处理完成: {len(processed_data)} 条记录")
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            # 使用原始数据继续，但记录警告
            processed_data = data
            logger.warning("使用原始数据继续预测，可能会影响准确性")

        # 数据验证
        if not self._validate_input_data(processed_data):
            raise ValueError("输入数据验证失败")

        # Use default steps if not specified
        if steps is None:
            steps = self.horizon_config[self.prediction_horizon]['steps']

        logger.info(f"Making {steps}-step prediction with {self.model_type.value} model...")

        predictions = {}
        confidences = {}

        # Get predictions from each model
        for model_type in self.active_models:
            try:
                if model_type == ModelType.LSTM and self.models[model_type] is not None:
                    # LSTM prediction
                    if hasattr(self.models[model_type], 'is_trained') and self.models[model_type].is_trained:
                        pred_result = self.models[model_type].predict_future(processed_data, steps)
                        predictions[model_type.value] = pred_result['predictions']
                        confidences[model_type.value] = pred_result.get('confidence', 0.85)
                    else:
                        logger.warning(f"LSTM model not trained, using synthetic predictions")
                        predictions[model_type.value] = self._generate_synthetic_rate_predictions(processed_data, steps)
                        confidences[model_type.value] = 0.3

                elif model_type == ModelType.DLINEAR and self.models[model_type] is not None:
                    # DLinear prediction - 直接使用适配器返回的PredictionResult
                    try:
                        dlinear_result = self.models[model_type].predict(processed_data, steps=steps)
                        predictions[model_type.value] = dlinear_result  # 存储完整的PredictionResult
                        confidences[model_type.value] = 0.8  # DLinear默认置信度
                        logger.info(f"DLinear prediction completed successfully")
                    except Exception as e:
                        logger.warning(f"DLinear prediction failed: {e}, using synthetic predictions")
                        predictions[model_type.value] = self._generate_synthetic_rate_predictions(processed_data, steps)
                        confidences[model_type.value] = 0.3

                elif model_type in [ModelType.ARIMA, ModelType.PROPHET]:
                    # Placeholder for statistical models
                    predictions[model_type.value] = np.ones(steps) * 5.0  # Default rate
                    confidences[model_type.value] = 0.5

            except Exception as e:
                logger.error(f"Prediction failed for {model_type.value}: {e}")
                predictions[model_type.value] = self._generate_synthetic_predictions(processed_data, steps)
                confidences[model_type.value] = 0.0

        # Generate structured predictions
        predicted_requests = self._generate_structured_predictions(
            processed_data, predictions, steps
        )

        # Combine predictions
        if self.enable_ensemble and len(predictions) > 1:
            final_confidence = self._combine_confidences(confidences)
        else:
            # Use single model prediction
            model_value = self.active_models[0].value
            final_confidence = confidences.get(model_value, 0.5)

        # Create structured result
        metadata = {
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'horizon': self.prediction_horizon.value,
            'steps': steps,
            'prediction_time': pd.Timestamp.now().isoformat(),
            'model_type': self.model_type.value
        }

        result = PredictionResult(
            predicted_requests=predicted_requests,
            confidence=final_confidence,
            prediction_metadata=metadata
        )

        # Store in history
        self.prediction_history.append(result.to_dict())

        return result

    def _generate_structured_predictions(self,
                                    data: pd.DataFrame,
                                    predictions: Dict[str, np.ndarray],
                                    steps: int) -> List[Dict[str, Any]]:
        """
        Generate structured prediction results with both numerical and categorical information.

        Args:
            data: Historical data
            predictions: Raw prediction values from models
            steps: Number of prediction steps

        Returns:
            List of predicted request dictionaries
        """
        if not data.empty:
            last_time = data['arrival_time_ns'].iloc[-1]
            last_input_tokens = data['input_toks'].iloc[-1]
            last_output_tokens = data['output_toks'].iloc[-1]

            # Get patterns from recent data
            recent_data = data.tail(min(10, len(data)))
            avg_input_tokens = recent_data['input_toks'].mean()
            avg_output_tokens = recent_data['output_toks'].mean()

            # Extract burst patterns
            burst_patterns = recent_data['burst_pattern'] if 'burst_pattern' in recent_data.columns else ['steady']
            most_common_burst = pd.Series(burst_patterns).mode().iloc[0] if len(burst_patterns) > 0 else 'steady'

            # Extract model types
            model_types = recent_data['model_type'] if 'model_type' in recent_data.columns else ['ChatGPT']
            most_common_model = pd.Series(model_types).mode().iloc[0] if len(model_types) > 0 else 'ChatGPT'
        else:
            last_time = 1_000_000_000  # Default start time
            last_input_tokens = 100
            last_output_tokens = 300
            avg_input_tokens = 100
            avg_output_tokens = 300
            most_common_burst = 'steady'
            most_common_model = 'ChatGPT'

        # Use DLinear predictions if available, otherwise LSTM predictions, otherwise generate synthetic predictions
        if ModelType.DLINEAR.value in predictions:
            dlinear_result = predictions[ModelType.DLINEAR.value]
            # DLinear已经返回了完整的PredictionResult，直接使用
            if hasattr(dlinear_result, 'predicted_requests'):
                predicted_requests = dlinear_result.predicted_requests
            else:
                # 回退到合成预测
                predicted_requests = self._generate_synthetic_predictions(
                    last_time, avg_input_tokens, avg_output_tokens,
                    most_common_burst, most_common_model, steps
                )
        elif ModelType.LSTM.value in predictions:
            lstm_pred = predictions[ModelType.LSTM.value]
            # Convert arrival rate predictions to actual request patterns
            predicted_requests = self._convert_rate_to_requests(
                lstm_pred, last_time, avg_input_tokens, avg_output_tokens,
                most_common_burst, most_common_model, steps
            )
        else:
            # Fallback: generate synthetic predictions
            predicted_requests = self._generate_synthetic_predictions(
                last_time, avg_input_tokens, avg_output_tokens,
                most_common_burst, most_common_model, steps
            )

        return predicted_requests

    def _convert_rate_to_requests(self,
                                 rate_predictions: np.ndarray,
                                 start_time: int,
                                 avg_input_tokens: float,
                                 avg_output_tokens: float,
                                 burst_pattern: str,
                                 model_type: str,
                                 steps: int) -> List[Dict[str, Any]]:
        """Convert arrival rate predictions to individual request predictions."""
        requests = []
        current_time = start_time

        # Determine time intervals based on burst pattern
        if burst_pattern == 'burst':
            base_interval = 50e6  # 50ms
            interval_variation = 0.3
        elif burst_pattern == 'sparse':
            base_interval = 500e6  # 500ms
            interval_variation = 0.5
        else:  # steady
            base_interval = 200e6  # 200ms
            interval_variation = 0.2

        for i, rate in enumerate(rate_predictions[:steps]):
            # Normalize rate to reasonable request count per time step
            expected_requests = max(1, int(rate * 0.1))  # Scale factor

            for j in range(expected_requests):
                # Add some randomness to token counts
                input_tokens = int(avg_input_tokens * np.random.uniform(0.8, 1.2))
                output_tokens = int(avg_output_tokens * np.random.uniform(0.8, 1.2))

                # Calculate arrival time with variation
                interval_variation_factor = 1.0 + np.random.uniform(-interval_variation, interval_variation)
                interval = base_interval * interval_variation_factor
                current_time += int(interval)

                # Determine request type based on token characteristics
                if input_tokens < 100 and output_tokens < 100:
                    request_type = 'short'
                elif input_tokens < 500 and output_tokens < 500:
                    request_type = 'medium'
                else:
                    request_type = 'long'

                request = {
                    'request_id': f"pred_{i}_{j}",
                    'arrival_time_ns': current_time,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'burst_pattern': burst_pattern,
                    'model_type': model_type,
                    'request_type': request_type,
                    'priority': self._calculate_priority(input_tokens, output_tokens, burst_pattern),
                    'estimated_compute_time': self._estimate_compute_time(input_tokens, output_tokens),
                    'memory_requirement_mb': (input_tokens + output_tokens) * 0.001,
                    'predicted_accuracy': max(0.7, min(0.95, 0.85 + np.random.normal(0, 0.05)))
                }

                requests.append(request)

        return requests

    def _generate_synthetic_predictions(self,
                                      start_time: int,
                                      avg_input_tokens: float,
                                      avg_output_tokens: float,
                                      burst_pattern: str,
                                      model_type: str,
                                      steps: int) -> List[Dict[str, Any]]:
        """Generate synthetic predictions when model predictions are not available."""
        requests = []
        current_time = start_time

        # Determine request count based on burst pattern
        if burst_pattern == 'burst':
            requests_per_step = 3
            interval = 50e6
        elif burst_pattern == 'sparse':
            requests_per_step = 1
            interval = 500e6
        else:  # steady
            requests_per_step = 2
            interval = 200e6

        for i in range(steps):
            for j in range(requests_per_step):
                # Add some randomness
                input_tokens = int(avg_input_tokens * np.random.uniform(0.7, 1.3))
                output_tokens = int(avg_output_tokens * np.random.uniform(0.7, 1.3))

                current_time += int(interval * np.random.uniform(0.8, 1.2))

                requests.append({
                    'arrival_time_ns': current_time,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'model_type': model_type,
                    'request_type': 'synthetic',
                    'priority': 'medium',
                    'estimated_compute_time': (input_tokens + output_tokens) * 0.0001,
                    'memory_requirement_mb': (input_tokens + output_tokens) * 0.001,
                    'predicted_accuracy': 0.7
                })

        return requests

  
    def _generate_synthetic_rate_predictions(self,
                                           data: pd.DataFrame,
                                           steps: int) -> np.ndarray:
        """
        Generate synthetic arrival rate predictions based on data patterns.

        Args:
            data: Historical data
            steps: Number of prediction steps

        Returns:
            Synthetic prediction array (arrival rates)
        """
        if not data.empty:
            # Calculate average arrival rate from historical data
            if len(data) > 1:
                time_diffs = data['arrival_time_ns'].diff().dropna()
                avg_interval = time_diffs.mean() if len(time_diffs) > 0 else 100_000_000
                avg_rate = 1e9 / avg_interval if avg_interval > 0 else 10.0
            else:
                avg_rate = 10.0

            # Add some randomness
            base_rate = avg_rate * np.random.uniform(0.8, 1.2)
        else:
            base_rate = 10.0

        # Generate synthetic arrival rate predictions
        synthetic_predictions = np.random.poisson(base_rate, steps)
        return synthetic_predictions.astype(float)

    def _extract_prediction_array(self, prediction_result) -> np.ndarray:
        """
        Extract prediction array from PredictionResult object.

        Args:
            prediction_result: PredictionResult object

        Returns:
            numpy array with arrival rate predictions
        """
        if hasattr(prediction_result, 'predicted_requests'):
            # 从predicted_requests中提取数据并转换为到达率
            requests = prediction_result.predicted_requests
            if requests:
                # 将token数量转换为到达率（假设平均间隔为200ms）
                avg_interval = 200e6  # 200ms in nanoseconds
                arrival_rates = []

                for req in requests:
                    input_tokens = req.get('input_tokens', 100)
                    # 简单转换：token数量 -> 请求大小 -> 到达率
                    # 假设每100ms一个请求，rate = 10 (requests per second)
                    rate = 10.0 * (input_tokens / 100.0)  # 标准化到合理范围
                    arrival_rates.append(max(1.0, min(50.0, rate)))  # 限制在1-50范围内

                return np.array(arrival_rates)

        # 回退到合成预测
        return np.array([10.0] * 10)  # 默认到达率为10 requests/second

    def _calculate_priority(self, input_tokens: int, output_tokens: int, burst_pattern: str) -> str:
        """Calculate request priority based on characteristics."""
        total_tokens = input_tokens + output_tokens

        if burst_pattern == 'burst':
            return 'high'
        elif total_tokens > 1000:
            return 'medium'
        else:
            return 'low'

    def _estimate_compute_time(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate compute time in milliseconds."""
        # Rough estimate: 1ms per 100 tokens
        return (input_tokens + output_tokens) * 0.01

    def _estimate_confidence(self, prediction_result: Any, model_type: ModelType) -> float:
        """Estimate prediction confidence."""
        if model_type == ModelType.LSTM:
            # For LSTM, extract confidence from result or calculate
            if isinstance(prediction_result, dict) and 'confidence' in prediction_result:
                confidence = prediction_result['confidence']
            elif 'predictions' in prediction_result:
                pred_array = prediction_result['predictions']
                if len(pred_array) > 1:
                    variance = np.var(pred_array)
                    # Higher variance = lower confidence
                    confidence = max(0.1, 1.0 - variance / (np.mean(pred_array) + 1e-9))
                else:
                    confidence = 0.8
            else:
                confidence = 0.8
        else:
            confidence = 0.5

        return min(1.0, max(0.0, confidence))

    def _combine_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions from multiple models."""
        if not self.ensemble_weights:
            # Simple average
            return np.mean(list(predictions.values()), axis=0)

        # Weighted average
        combined = np.zeros(len(list(predictions.values())[0]))
        total_weight = 0.0

        for model_type, weight in self.ensemble_weights.items():
            if model_type.value in predictions:
                combined += weight * predictions[model_type.value]
                total_weight += weight

        return combined / max(total_weight, 1e-9)

    def _combine_confidences(self, confidences: Dict[str, float]) -> float:
        """Combine confidence scores."""
        if not confidences:
            return 0.5

        if not self.ensemble_weights:
            return np.mean(list(confidences.values()))

        # Weighted average
        combined = 0.0
        total_weight = 0.0

        for model_type, weight in self.ensemble_weights.items():
            if model_type.value in confidences:
                combined += weight * confidences[model_type.value]
                total_weight += weight

        return combined / max(total_weight, 1e-9)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration."""
        return {
            'model_type': self.model_type.value,
            'prediction_horizon': self.prediction_horizon.value,
            'sequence_length': self.sequence_length,
            'feature_window': self.feature_window,
            'enable_ensemble': self.enable_ensemble,
            'active_models': [model.value for model in self.active_models],
            'ensemble_weights': {k.value: v for k, v in self.ensemble_weights.items()} if self.ensemble_weights else None,
            'feature_count': len(self.feature_extractor.get_feature_names()),
            'feature_names': self.feature_extractor.get_feature_names()
        }

    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of prediction history."""
        if not self.prediction_history:
            return {'message': 'No predictions made yet'}

        # Calculate statistics
        predictions = [pred['prediction'] for pred in self.prediction_history]
        confidences = [pred['confidence'] for pred in self.prediction_history]

        return {
            'total_predictions': len(self.prediction_history),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'min_confidence': np.min(confidences) if confidences else 0.0,
            'max_confidence': np.max(confidences) if confidences else 0.0,
            'avg_prediction_length': np.mean([len(pred) for pred in predictions]) if predictions else 0.0,
            'last_prediction_time': self.prediction_history[-1].get('timestamp', 'unknown')
        }

    def save(self, path: str) -> None:
        """Save the workload predictor."""
        import joblib

        save_data = {
            'model_type': self.model_type.value,
            'prediction_horizon': self.prediction_horizon.value,
            'sequence_length': self.sequence_length,
            'feature_window': self.feature_window,
            'enable_ensemble': self.enable_ensemble,
            'active_models': [model.value for model in self.active_models],
            'feature_extractor': self.feature_extractor,
            'ensemble_weights': {k.value: v for k, v in self.ensemble_weights.items()} if self.ensemble_weights else None,
            'prediction_history': self.prediction_history,
            'accuracy_metrics': self.accuracy_metrics
        }

        # Save individual models
        for model_type in self.active_models:
            if self.models[model_type] is not None:
                model_path = f"{path}_{model_type.value}.pt"
                self.models[model_type].save(model_path)
                save_data[f'{model_type.value}_path'] = model_path

        joblib.dump(save_data, path)
        logger.info(f"WorkloadPredictor saved to {path}")

    def load(self, path: str) -> None:
        """Load the workload predictor."""
        import joblib

        try:
            save_data = joblib.load(path)

            # Restore configuration
            self.model_type = ModelType(save_data['model_type'])
            self.prediction_horizon = PredictionHorizon(save_data['prediction_horizon'])
            self.sequence_length = save_data['sequence_length']
            self.feature_window = save_data['feature_window']
            self.enable_ensemble = save_data['enable_ensemble']
            self.active_models = [ModelType(model) for model in save_data['active_models']]
            self.feature_extractor = save_data['feature_extractor']
            ensemble_weights = save_data.get('ensemble_weights', {})
            if ensemble_weights is not None:
                self.ensemble_weights = {ModelType(k): v for k, v in ensemble_weights.items()}
            else:
                self.ensemble_weights = {}
            self.prediction_history = save_data.get('prediction_history', [])
            self.accuracy_metrics = save_data.get('accuracy_metrics', {})

            # Load individual models
            for model_type in self.active_models:
                model_path = save_data.get(f'{model_type.value}_path')
                if model_path and self.models[model_type] is not None:
                    self.models[model_type].load(model_path)

            logger.info(f"WorkloadPredictor loaded from {path}")

        except Exception as e:
            logger.error(f"Failed to load WorkloadPredictor from {path}: {e}")
            raise

    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据，增强错误处理机制

        Args:
            data: 输入数据

        Returns:
            bool: 数据是否有效
        """
        try:
            # 基本数据类型检查
            if not isinstance(data, pd.DataFrame):
                logger.error("数据必须是pandas DataFrame")
                return False

            # 空数据检查
            if data.empty:
                logger.error("数据为空")
                return False

            # 最小数据量检查
            if len(data) < 3:
                logger.error(f"数据量不足，至少需要3条记录，当前只有{len(data)}条")
                return False

            # 必需列检查（支持多种列名格式）
            required_cols_patterns = [
                ['arrival_time_ns', 'input_toks', 'output_toks'],  # 标准格式
                ['Timestamp', 'Request tokens', 'Response tokens'],  # BurstGPT格式
                ['timestamp', 'input_tokens', 'output_tokens']  # 通用格式
            ]

            found_valid_format = False
            for cols in required_cols_patterns:
                if all(col in data.columns for col in cols):
                    found_valid_format = True
                    break

            if not found_valid_format:
                logger.error(f"数据格式不支持，可用列：{list(data.columns)}")
                return False

            # 数据类型检查
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < 2:
                logger.error("数值列不足，无法进行预测")
                return False

            # 时间戳检查
            time_cols = [col for col in data.columns if 'time' in col.lower()]
            if time_cols:
                time_col = time_cols[0]
                if data[time_col].isnull().any():
                    logger.warning(f"{time_col}列包含空值，将进行填充")

            # Token数量检查
            token_cols = [col for col in data.columns if 'token' in col.lower()]
            for col in token_cols:
                if data[col].isnull().any():
                    logger.warning(f"{col}列包含空值，将进行填充")
                if (data[col] <= 0).any():
                    logger.warning(f"{col}列包含非正值，将进行修正")

            # 数据一致性检查
            if len(data) > 1:
                # 检查时间戳单调性
                if time_cols:
                    time_col = time_cols[0]
                    if not data[time_col].is_monotonic_increasing:
                        logger.warning("时间戳非单调递增，将进行排序")

                # 检查数据范围
                for col in token_cols:
                    q99 = data[col].quantile(0.99)
                    if (data[col] > q99 * 2).any():
                        logger.warning(f"{col}列可能存在异常值")

            logger.info("数据验证通过")
            return True

        except Exception as e:
            logger.error(f"数据验证过程出错: {e}")
            return False

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data.

        Args:
            data: Input DataFrame

        Returns:
            True if data is valid
        """
        required_columns = ['input_toks', 'output_toks', 'arrival_time_ns']

        if not isinstance(data, pd.DataFrame):
            logger.error("Data must be a pandas DataFrame")
            return False

        if data.empty:
            logger.error("DataFrame is empty")
            return False

        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Check for reasonable values
        if (data['input_toks'] <= 0).any():
            logger.warning("Some input_toks values are <= 0")

        if (data['output_toks'] <= 0).any():
            logger.warning("Some output_toks values are <= 0")

        if (data['arrival_time_ns'] <= 0).any():
            logger.warning("Some arrival_time_ns values are <= 0")

        return True

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models."""
        importance = {}

        # Get importance from LSTM model if available
        if ModelType.LSTM in self.models and self.models[ModelType.LSTM] is not None:
            if hasattr(self.models[ModelType.LSTM].model, 'feature_importances_'):
                lstm_importance = self.feature_extractor.get_feature_importance(
                    self.models[ModelType.LSTM].model
                )
                importance.update({f'lstm_{k}': v for k, v in lstm_importance.items()})

        return importance

    def visualize_prediction(self,
                           historical_data: Optional[pd.DataFrame] = None,
                           save_dir: str = "prediction_plots",
                           show_plots: bool = True) -> None:
        """
        Create comprehensive visualization of prediction results.

        Args:
            historical_data: Historical data for comparison
            save_dir: Directory to save plots
            show_plots: Whether to display plots
        """
        try:
            from .visualization import PredictionVisualizer

            # Create visualizer
            visualizer = PredictionVisualizer()

            # Create dashboard
            if self.prediction_history:
                # Get the latest prediction
                latest_prediction_dict = self.prediction_history[-1]
                latest_prediction = PredictionResult(
                    predicted_requests=latest_prediction_dict['predicted_requests'],
                    confidence=latest_prediction_dict['confidence'],
                    prediction_metadata=latest_prediction_dict['metadata']
                )

                # Create comprehensive comparison visualization (if no errors)
                if historical_data is not None:
                    try:
                        visualizer.plot_comprehensive_comparison(
                            historical_data=historical_data,
                            prediction_result=latest_prediction,
                            save_path=os.path.join(save_dir, "comprehensive_comparison.png"),
                            show_plot=show_plots
                        )
                    except Exception as e:
                        logger.warning(f"Comprehensive comparison failed: {e}")
                        # Fall back to individual plots

                # Create individual detailed plots
                visualizer.plot_prediction_vs_actual(
                    prediction_result=latest_prediction,
                    historical_data=historical_data,
                    save_path=os.path.join(save_dir, "prediction_overview.png"),
                    show_plot=False
                )

                # Create enhanced time series (if no errors)
                try:
                    visualizer.plot_enhanced_time_series(
                        historical_data=historical_data,
                        prediction_result=latest_prediction,
                        save_path=os.path.join(save_dir, "enhanced_time_series.png"),
                        show_plot=False
                    )
                except Exception as e:
                    logger.warning(f"Enhanced time series failed: {e}")

                visualizer.plot_resource_forecast(
                    prediction_result=latest_prediction,
                    save_path=os.path.join(save_dir, "resource_forecast.png"),
                    show_plot=False
                )

                # Create error analysis if historical data is available (if no errors)
                if historical_data is not None:
                    try:
                        visualizer.plot_error_analysis(
                            historical_data=historical_data,
                            prediction_result=latest_prediction,
                            save_path=os.path.join(save_dir, "error_analysis.png"),
                            show_plot=False
                        )
                    except Exception as e:
                        logger.warning(f"Error analysis failed: {e}")

                logger.info(f"Prediction visualization completed in {save_dir}")
            else:
                logger.warning("No prediction history available for visualization")

        except ImportError:
            logger.warning("Visualization libraries not available. Install matplotlib and seaborn.")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")

    def generate_prediction_report(self,
                                 prediction_result: PredictionResult,
                                 historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate comprehensive prediction report with metrics and insights.

        Args:
            prediction_result: Prediction result to analyze
            historical_data: Historical data for comparison

        Returns:
            Dictionary containing prediction report
        """
        report = {
            'prediction_summary': {},
            'resource_analysis': {},
            'burst_analysis': {},
            'scheduling_recommendations': [],
            'confidence_assessment': {}
        }

        # Basic prediction summary
        report['prediction_summary'] = {
            'total_predicted_requests': len(prediction_result.predicted_requests),
            'prediction_confidence': prediction_result.confidence,
            'prediction_horizon': prediction_result.prediction_metadata.get('horizon', 'unknown'),
            'model_type': prediction_result.prediction_metadata.get('model_type', 'unknown'),
            'prediction_time': prediction_result.prediction_metadata.get('prediction_time', 'unknown')
        }

        # Resource analysis
        resource_req = prediction_result.get_resource_requirements()
        timing_info = prediction_result.get_timing_info()

        report['resource_analysis'] = {
            'total_memory_required_mb': resource_req['memory_mb'],
            'total_compute_units': resource_req['compute_units'],
            'total_tokens': resource_req['total_tokens'],
            'prediction_duration_ms': timing_info['time_range_ms'],
            'peak_request_rate': len(prediction_result.predicted_requests) / max(timing_info['time_range_ms'] / 1000, 0.001)
        }

        # Burst analysis
        burst_info = prediction_result.get_burst_analysis()
        report['burst_analysis'] = {
            'burst_level': burst_info['burst_level'],
            'request_intensity': burst_info['intensity'],
            'average_interval_ms': burst_info['avg_interval_ms'],
            'burst_risk_score': self._calculate_burst_risk_score(burst_info)
        }

        # Scheduling recommendations
        report['scheduling_recommendations'] = self._generate_scheduling_recommendations(
            resource_req, burst_info, prediction_result.confidence
        )

        # Confidence assessment
        report['confidence_assessment'] = {
            'overall_confidence': prediction_result.confidence,
            'confidence_level': self._get_confidence_level(prediction_result.confidence),
            'reliability_score': self._calculate_reliability_score(prediction_result)
        }

        return report

    def _calculate_burst_risk_score(self, burst_info: Dict[str, Any]) -> float:
        """Calculate burst risk score (0-1)."""
        intensity_score = min(burst_info['intensity'] / 50.0, 1.0)  # Normalize to 0-1
        interval_score = 1.0 - min(burst_info['avg_interval_ms'] / 1000.0, 1.0)  # Shorter intervals = higher risk

        if burst_info['burst_level'] == 'high':
            level_multiplier = 1.0
        elif burst_info['burst_level'] == 'medium':
            level_multiplier = 0.7
        else:
            level_multiplier = 0.3

        return (intensity_score + interval_score) / 2.0 * level_multiplier

    def _generate_scheduling_recommendations(self,
                                           resource_req: Dict[str, float],
                                           burst_info: Dict[str, Any],
                                           confidence: float) -> List[str]:
        """Generate scheduling recommendations based on prediction."""
        recommendations = []

        # Memory-based recommendations
        if resource_req['memory_mb'] > 1000:
            recommendations.append("High memory usage predicted - consider memory optimization strategies")

        # Compute-based recommendations
        if resource_req['compute_units'] > 100:
            recommendations.append("High computational load expected - enable GPU acceleration if available")

        # Burst-based recommendations
        if burst_info['burst_level'] == 'high':
            recommendations.append("High burst pattern detected - enable aggressive batching and dynamic scaling")
        elif burst_info['burst_level'] == 'medium':
            recommendations.append("Medium burst pattern - use adaptive batching strategies")

        # Confidence-based recommendations
        if confidence < 0.7:
            recommendations.append("Low prediction confidence - enable fallback mechanisms")
        elif confidence > 0.9:
            recommendations.append("High prediction confidence - optimize for predicted workload")

        # General recommendations
        if resource_req['request_count'] > 50:
            recommendations.append("High request volume - enable load balancing")

        return recommendations

    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description."""
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        else:
            return "low"

    def _calculate_reliability_score(self, prediction_result: PredictionResult) -> float:
        """Calculate reliability score based on various factors."""
        base_score = prediction_result.confidence

        # Adjust based on request diversity
        request_types = set(req['request_type'] for req in prediction_result.predicted_requests)
        diversity_factor = min(len(request_types) / 3.0, 1.0)  # Max 3 types

        # Adjust based on prediction consistency
        if len(prediction_result.predicted_requests) > 1:
            token_variation = np.std([req['input_tokens'] + req['output_tokens']
                                   for req in prediction_result.predicted_requests])
            consistency_factor = 1.0 / (1.0 + token_variation / 1000.0)
        else:
            consistency_factor = 1.0

        return base_score * diversity_factor * consistency_factor