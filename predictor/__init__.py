"""
SmartLLMServe Predictor Module

This module contains time series prediction models for LLM workload forecasting.
主要使用DLinear模型进行轻量级的时间序列预测。
"""

from .dlinear_model import DLinearPredictor, DLinearModel
from .feature_extractor import FeatureExtractor
from .data_preprocessor import DataPreprocessor
from .model_interface import ModelConfig, ModelType
from .multi_task_loss import MultiTaskLoss, LossType

__all__ = [
    'DLinearPredictor',
    'DLinearModel',
    'FeatureExtractor',
    'DataPreprocessor',
    'ModelConfig',
    'ModelType',
    'MultiTaskLoss',
    'LossType'
]

__version__ = '0.2.0'
__author__ = 'SmartLLMServe Team'