"""
统一的多任务损失函数模块

针对LLM服务工作负载预测的特定需求，设计统一的多任务损失函数：
- 到达时间间隔预测
- 请求token数量预测
- 响应token数量预测

确保所有模型(LSTM/DLinear)优化相同的目标。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LossType(Enum):
    """损失函数类型"""
    MSE = "mse"                    # 均方误差
    MAE = "mae"                    # 平均绝对误差
    HUBER = "huber"                # Huber损失
    LOG_COSH = "log_cosh"          # LogCosh损失
    WEIGHTED_MSE = "weighted_mse"  # 加权均方误差


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数

    针对三个预测目标：
    1. time_interval: 下一个请求的到达时间间隔
    2. input_tokens: 下一个请求的输入token数量
    3. output_tokens: 下一个请求的输出token数量
    """

    def __init__(self,
                 time_weight: float = 1.0,
                 input_token_weight: float = 1.0,
                 output_token_weight: float = 1.0,
                 time_loss_type: LossType = LossType.MSE,
                 token_loss_type: LossType = LossType.MSE,
                 huber_delta: float = 1.0,
                 normalize_weights: bool = True):
        """
        初始化多任务损失函数

        Args:
            time_weight: 时间间隔预测的权重
            input_token_weight: 输入token预测的权重
            output_token_weight: 输出token预测的权重
            time_loss_type: 时间损失的函数类型
            token_loss_type: Token损失的函数类型
            huber_delta: Huber损失的delta参数
            normalize_weights: 是否标准化权重
        """
        super().__init__()

        # 设置权重
        self.time_weight = time_weight
        self.input_token_weight = input_token_weight
        self.output_token_weight = output_token_weight

        # 标准化权重
        if normalize_weights:
            total = time_weight + input_token_weight + output_token_weight
            self.time_weight /= total
            self.input_token_weight /= total
            self.output_token_weight /= total

        # 损失函数类型
        self.time_loss_type = time_loss_type
        self.token_loss_type = token_loss_type
        self.huber_delta = huber_delta

        logger.info(f"MultiTaskLoss initialized - Time: {time_weight}, "
                   f"InputToken: {input_token_weight}, OutputToken: {output_token_weight}")

    def _get_loss_function(self, loss_type: LossType) -> nn.Module:
        """获取指定类型的损失函数"""
        if loss_type == LossType.MSE:
            return nn.MSELoss(reduction='mean')
        elif loss_type == LossType.MAE:
            return nn.L1Loss(reduction='mean')
        elif loss_type == LossType.HUBER:
            return nn.HuberLoss(delta=self.huber_delta, reduction='mean')
        elif loss_type == LossType.LOG_COSH:
            return LogCoshLoss(reduction='mean')
        elif loss_type == LossType.WEIGHTED_MSE:
            return WeightedMSELoss(reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失

        Args:
            predictions: 模型预测 [batch_size, 3] 或 [batch_size, seq_len, 3]
                       格式: [time_interval, input_tokens, output_tokens]
            targets: 真实目标 [batch_size, 3] 或 [batch_size, seq_len, 3]
                    格式: [time_interval, input_tokens, output_tokens]
            mask: 可选的掩码 [batch_size] 或 [batch_size, seq_len]

        Returns:
            包含各项损失的字典
        """
        # 确保输入维度正确
        if predictions.dim() == 3:
            # 序列预测：[batch_size, seq_len, 3]
            batch_size, seq_len, _ = predictions.shape

            # 展平处理所有时间步
            predictions = predictions.reshape(-1, 3)  # [batch_size*seq_len, 3]
            targets = targets.reshape(-1, 3)  # [batch_size*seq_len, 3]
            if mask is not None:
                mask = mask.reshape(-1)  # [batch_size*seq_len]
        else:
            batch_size = predictions.shape[0]
            seq_len = 1

        # 提取各个预测目标
        pred_time = predictions[:, 0]  # [batch_size*seq_len]
        pred_input_tokens = predictions[:, 1]  # [batch_size*seq_len]
        pred_output_tokens = predictions[:, 2]  # [batch_size*seq_len]

        target_time = targets[:, 0]  # [batch_size]
        target_input_tokens = targets[:, 1]  # [batch_size]
        target_output_tokens = targets[:, 2]  # [batch_size]

        # 获取损失函数
        time_loss_fn = self._get_loss_function(self.time_loss_type)
        token_loss_fn = self._get_loss_function(self.token_loss_type)

        # 计算各项损失
        time_loss = time_loss_fn(pred_time, target_time)
        input_token_loss = token_loss_fn(pred_input_tokens, target_input_tokens)
        output_token_loss = token_loss_fn(pred_output_tokens, target_output_tokens)

        # 应用掩码（如果提供）
        if mask is not None:
            mask_float = mask.float()
            time_loss = (time_loss * mask_float).sum() / (mask_float.sum() + 1e-8)
            input_token_loss = (input_token_loss * mask_float).sum() / (mask_float.sum() + 1e-8)
            output_token_loss = (output_token_loss * mask_float).sum() / (mask_float.sum() + 1e-8)

        # 计算加权总损失
        total_loss = (self.time_weight * time_loss +
                     self.input_token_weight * input_token_loss +
                     self.output_token_weight * output_token_loss)

        # 返回损失详情
        return {
            'total_loss': total_loss,
            'time_loss': time_loss,
            'input_token_loss': input_token_loss,
            'output_token_loss': output_token_loss,
            'time_weight': torch.tensor(self.time_weight),
            'input_token_weight': torch.tensor(self.input_token_weight),
            'output_token_weight': torch.tensor(self.output_token_weight)
        }

    def get_loss_components(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        获取损失组件的详细信息（用于监控）

        Returns:
            包含各项损失指标的字典
        """
        with torch.no_grad():
            loss_dict = self.forward(predictions, targets)

            # 计算相对误差百分比
            pred_time = predictions[:, 0]
            pred_input_tokens = predictions[:, 1]
            pred_output_tokens = predictions[:, 2]

            target_time = targets[:, 0]
            target_input_tokens = targets[:, 1]
            target_output_tokens = targets[:, 2]

            # 避免除零
            time_mape = torch.mean(torch.abs((target_time - pred_time) / (target_time + 1e-8))) * 100
            input_token_mape = torch.mean(torch.abs((target_input_tokens - pred_input_tokens) / (target_input_tokens + 1e-8))) * 100
            output_token_mape = torch.mean(torch.abs((target_output_tokens - pred_output_tokens) / (target_output_tokens + 1e-8))) * 100

            return {
                'total_loss': loss_dict['total_loss'].item(),
                'time_loss': loss_dict['time_loss'].item(),
                'input_token_loss': loss_dict['input_token_loss'].item(),
                'output_token_loss': loss_dict['output_token_loss'].item(),
                'time_mape': time_mape.item(),
                'input_token_mape': input_token_mape.item(),
                'output_token_mape': output_token_mape.item()
            }


class LogCoshLoss(nn.Module):
    """LogCosh损失函数 - 对异常值比MSE更鲁棒"""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        errors = predictions - targets
        loss = torch.log(torch.cosh(errors))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedMSELoss(nn.Module):
    """加权MSE损失 - 可以给不同样本不同权重"""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 给较大误差更高权重，关注难样本
        errors = predictions - targets
        weights = 1.0 + torch.abs(errors) / (torch.abs(targets) + 1e-8)
        weighted_errors = weights * (errors ** 2)

        if self.reduction == 'mean':
            return weighted_errors.mean()
        elif self.reduction == 'sum':
            return weighted_errors.sum()
        else:
            return weighted_errors


class AdaptiveLossWeights(nn.Module):
    """自适应损失权重 - 根据训练动态调整权重"""

    def __init__(self,
                 initial_weights: Dict[str, float] = None,
                 adaptation_rate: float = 0.01,
                 min_weight: float = 0.1,
                 max_weight: float = 2.0):
        super().__init__()

        if initial_weights is None:
            initial_weights = {'time': 1.0, 'input_token': 1.0, 'output_token': 1.0}

        self.weights = nn.ParameterDict({
            'time': nn.Parameter(torch.tensor(initial_weights['time'])),
            'input_token': nn.Parameter(torch.tensor(initial_weights['input_token'])),
            'output_token': nn.Parameter(torch.tensor(initial_weights['output_token']))
        })

        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.loss_history = {'time': [], 'input_token': [], 'output_token': []}

    def update_weights(self, loss_dict: Dict[str, float]):
        """根据损失历史更新权重"""
        for key in ['time', 'input_token', 'output_token']:
            loss_key = f'{key}_loss'
            if loss_key in loss_dict:
                self.loss_history[key].append(loss_dict[loss_key])

                # 保持最近100次的损失历史
                if len(self.loss_history[key]) > 100:
                    self.loss_history[key].pop(0)

                # 如果某个任务的损失持续较高，增加其权重
                if len(self.loss_history[key]) >= 10:
                    recent_avg = np.mean(self.loss_history[key][-10:])
                    overall_avg = np.mean(self.loss_history[key])

                    if recent_avg > overall_avg * 1.1:  # 最近损失比总体高10%
                        self.weights[key].data *= (1 + self.adaptation_rate)
                    elif recent_avg < overall_avg * 0.9:  # 最近损失比总体低10%
                        self.weights[key].data *= (1 - self.adaptation_rate)

                    # 限制权重范围
                    self.weights[key].data.clamp_(self.min_weight, self.max_weight)

    def get_normalized_weights(self) -> Dict[str, float]:
        """获取标准化后的权重"""
        weights = {k: v.item() for k, v in self.weights.items()}
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


def create_loss_function(loss_config: Dict[str, Any]) -> MultiTaskLoss:
    """
    根据配置创建损失函数

    Args:
        loss_config: 损失函数配置

    Returns:
        配置好的多任务损失函数
    """
    return MultiTaskLoss(
        time_weight=loss_config.get('time_weight', 1.0),
        input_token_weight=loss_config.get('input_token_weight', 1.0),
        output_token_weight=loss_config.get('output_token_weight', 1.0),
        time_loss_type=LossType(loss_config.get('time_loss_type', 'mse')),
        token_loss_type=LossType(loss_config.get('token_loss_type', 'mse')),
        huber_delta=loss_config.get('huber_delta', 1.0),
        normalize_weights=loss_config.get('normalize_weights', True)
    )


# 预定义的损失函数配置
LOSS_CONFIGS = {
    'balanced': {
        'time_weight': 1.0,
        'input_token_weight': 1.0,
        'output_token_weight': 1.0,
        'time_loss_type': 'mse',
        'token_loss_type': 'mse',
        'description': '平衡的多任务损失'
    },
    'time_focused': {
        'time_weight': 2.0,
        'input_token_weight': 1.0,
        'output_token_weight': 1.0,
        'time_loss_type': 'mae',
        'token_loss_type': 'mse',
        'description': '重点关注时间预测'
    },
    'token_focused': {
        'time_weight': 0.5,
        'input_token_weight': 2.0,
        'output_token_weight': 2.0,
        'time_loss_type': 'mse',
        'token_loss_type': 'huber',
        'description': '重点关注token预测'
    },
    'robust': {
        'time_weight': 1.0,
        'input_token_weight': 1.0,
        'output_token_weight': 1.0,
        'time_loss_type': 'huber',
        'token_loss_type': 'huber',
        'description': '对异常值鲁棒的损失函数'
    }
}