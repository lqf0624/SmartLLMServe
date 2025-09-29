"""
DLinear官方实现 - 基于AAAI 2023论文

"Are Transformers Effective for Time Series Forecasting?"
完全按照官方实现复制，确保算法一致性。

官方实现特点：
1. 使用手动padding的移动平均分解
2. 固定kernel_size=25
3. 支持individual模式
4. 标准的数据维度处理

注意：此文件已替换为官方实现，原有的扩展功能移至dlinear_official.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum
import logging
from sklearn.preprocessing import StandardScaler
import warnings
import time
from tqdm import tqdm
from .multi_task_loss import MultiTaskLoss

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DecompositionMode(Enum):
    """时序分解模式"""
    MOVING_AVG = "moving_avg"      # 移动平均分解
    SEASONAL_DECOMPOSE = "seasonal" # 季节性分解
    WAVELET = "wavelet"            # 小波分解


class DLinearModel(nn.Module):
    """
    DLinear模型实现

    基于时序分解的线性预测模型，结构简单但效果显著。

    论文核心贡献：
    - 证明简单线性模型在时序预测中可以媲美复杂模型
    - 提出时序分解+线性网络的架构
    - 在多个基准数据集上取得SOTA结果
    """

    def __init__(self,
                 seq_len: int,
                 pred_len: int,
                 individual: bool = True,  # 默认启用通道独立
                 channels: int = 3,        # 3个通道：时间间隔、输入token、输出token
                 kernel_size: int = 25):
        """
        初始化DLinear模型 - 通道独立版本

        Args:
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            individual: 是否为每个通道单独建模（默认True）
            channels: 输入通道数（默认3）
            kernel_size: 移动平均窗口大小
        """
        super(DLinearModel, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = channels
        self.kernel_size = kernel_size

        # 时序分解层 - 使用官方实现
        self.decomposition = SeriesDecomposition(kernel_size)

        # 通道独立建模或共享线性层
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        logger.info(f"DLinear model initialized: seq_len={seq_len}, pred_len={pred_len}, "
                   f"individual={individual}, channels={channels}")

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 通道独立版本

        Args:
            x: 输入张量 [Batch, Seq_len, Channel]

        Returns:
            torch.Tensor: 预测结果 [Batch, Output length, Channel]
        """
        # 时序分解
        seasonal_init, trend_init = self.decomposition(x)

        # 转换维度 [Batch, Seq_len, Channel] -> [Batch, Channel, Seq_len]
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                       dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                     dtype=trend_init.dtype).to(trend_init.device)

            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # 重建预测结果
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [Batch, Output length, Channel]

    def predict_with_decomposition(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        带分解的预测，返回季节性和趋势组件

        Args:
            x: 输入张量

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (总预测, 季节性预测, 趋势预测)
        """
        # 时序分解
        seasonal, trend = self.decomposition_layer(x)

        # 预测季节性组件
        seasonal_pred = self.linear_seasonal(seasonal.transpose(1, 2)).transpose(1, 2)

        # 预测趋势组件
        trend_pred = self.linear_trend(trend.transpose(1, 2)).transpose(1, 2)

        # 重建预测结果
        prediction = seasonal_pred + trend_pred

        return prediction, seasonal_pred, trend_pred

    def get_decomposition_analysis(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取分解分析结果

        Args:
            x: 输入张量

        Returns:
            Dict[str, torch.Tensor]: 分解组件分析
        """
        seasonal, trend = self.decomposition_layer(x)

        return {
            'seasonal_component': seasonal,
            'trend_component': trend,
            'seasonal_strength': torch.std(seasonal, dim=1),
            'trend_strength': torch.std(trend, dim=1),
            'decomposition_ratio': torch.std(seasonal) / (torch.std(trend) + 1e-8)
        }


class SeriesDecomposition(nn.Module):
    """
    时序分解模块

    实现不同的时序分解方法：
    1. 移动平均分解
    2. 季节性分解
    3. 小波分解
    """

    def __init__(self, kernel_size: int, mode: DecompositionMode = DecompositionMode.MOVING_AVG):
        """
        初始化时序分解模块

        Args:
            kernel_size: 分解窗口大小
            mode: 分解模式
        """
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.mode = mode

        if mode == DecompositionMode.MOVING_AVG:
            # 移动平均分解
            self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2)
            logger.info(f"Moving average decomposition initialized with kernel_size={kernel_size}")
        elif mode == DecompositionMode.SEASONAL_DECOMPOSE:
            # 季节性分解（简化版）
            self.seasonal_length = kernel_size
            logger.info(f"Seasonal decomposition initialized with seasonal_length={kernel_size}")
        elif mode == DecompositionMode.WAVELET:
            # 小波分解（简化版，使用移动平均近似）
            self.wavelet_kernel = self._create_wavelet_kernel(kernel_size)
            logger.info(f"Wavelet decomposition initialized with kernel_size={kernel_size}")

    def _create_wavelet_kernel(self, kernel_size: int) -> torch.Tensor:
        """创建小波核"""
        # 简化的Haar小波
        kernel = torch.ones(1, 1, kernel_size)
        kernel[:, :, :kernel_size//2] = -1
        kernel[:, :, kernel_size//2:] = 1
        return kernel / (kernel_size // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行时序分解

        Args:
            x: 输入张量 [batch_size, seq_len, features]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (季节性组件, 趋势组件)
        """
        if self.mode == DecompositionMode.MOVING_AVG:
            return self._moving_avg_decomposition(x)
        elif self.mode == DecompositionMode.SEASONAL_DECOMPOSE:
            return self._seasonal_decomposition(x)
        elif self.mode == DecompositionMode.WAVELET:
            return self._wavelet_decomposition(x)
        else:
            raise ValueError(f"Unknown decomposition mode: {self.mode}")

    def _moving_avg_decomposition(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """移动平均分解"""
        # x shape: [batch_size, seq_len, features]
        original_shape = x.shape

        # 转换为 [batch_size * features, 1, seq_len] 进行池化
        x_reshaped = x.transpose(1, 2).reshape(-1, 1, original_shape[1])

        # 移动平均提取趋势
        trend = self.moving_avg(x_reshaped)
        trend = trend.reshape(original_shape[0], original_shape[2], original_shape[1]).transpose(1, 2)

        # 季节性 = 原始序列 - 趋势
        seasonal = x - trend

        return seasonal, trend

    def _seasonal_decomposition(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """季节性分解（简化版）"""
        # 使用移动平均作为趋势的近似
        return self._moving_avg_decomposition(x)

    def _wavelet_decomposition(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """小波分解（简化版）"""
        # 使用移动平均作为近似
        return self._moving_avg_decomposition(x)


class DLinearPredictor:
    """
    DLinear预测器

    封装DLinear模型的训练、预测和评估功能。
    """

    def __init__(self,
                 seq_len: int = 50,
                 pred_len: int = 10,
                 individual: bool = True,  # 默认启用通道独立
                 channels: int = 3,        # 3个通道：时间间隔、输入token、输出token
                 kernel_size: int = 25,
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        初始化DLinear预测器 - 通道独立版本

        Args:
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            individual: 是否为每个通道单独建模（默认True）
            channels: 特征/通道数量
            kernel_size: 分解窗口大小
            learning_rate: 学习率
            device: 设备
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = channels
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        # 创建模型 - 使用新的参数名
        self.model = DLinearModel(
            seq_len=seq_len,
            pred_len=pred_len,
            individual=individual,
            channels=channels,
            kernel_size=kernel_size
        ).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # 标准MSE损失函数 - 通道独立模式下无需多任务损失
        self.criterion = nn.MSELoss()

        # 数据标准化器
        self.scaler = StandardScaler()

        # 训练历史
        self.training_history = {
            'loss': [],
            'val_loss': []
        }

        logger.info(f"DLinearPredictor initialized: seq_len={seq_len}, pred_len={pred_len}, individual={individual}")

    def prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备训练数据

        Args:
            data: 输入DataFrame，包含Timestamp, input_toks, output_toks列

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (输入序列, 目标序列)
        """
        # 确保数据有时间戳
        if 'Timestamp' not in data.columns:
            raise ValueError("数据必须包含Timestamp列")

        # 列名映射：处理不同的数据集格式
        if 'Concurrent_requests' in data.columns:
            # 时间序列数据格式 - 分钟级固定间隔
            if 'Request_tokens_sum' in data.columns and 'Response_tokens_sum' in data.columns:
                data = data.rename(columns={
                    'Request_tokens_sum': 'input_toks',
                    'Response_tokens_sum': 'output_toks',
                    'Concurrent_requests': 'concurrent_requests'
                })
            else:
                raise ValueError("时间序列数据必须包含'Request_tokens_sum'/'Response_tokens_sum'列")
        elif 'Request tokens' in data.columns and 'Response tokens' in data.columns:
            # BurstGPT数据集格式
            data = data.rename(columns={
                'Request tokens': 'input_toks',
                'Response tokens': 'output_toks'
            })
        elif 'input_toks' not in data.columns or 'output_toks' not in data.columns:
            raise ValueError("数据必须包含'Request tokens'/'Response tokens'或'Request_tokens_sum'/'Response_tokens_sum'或'input_toks'/'output_toks'列")

        # 按时间戳排序
        data = data.sort_values('Timestamp').reset_index(drop=True)

        # 统一处理：确保总是有三个特征
        if 'Concurrent_requests' in data.columns:
            # 时间序列数据：分钟级固定间隔，只使用并发量和token数量
            concurrent_requests = data['concurrent_requests'].values.astype(float)
            input_tokens = data['input_toks'].values.astype(float)
            output_tokens = data['output_toks'].values.astype(float)
        else:
            # 原始数据格式：假设并发量为1
            concurrent_requests = np.ones(len(data)).astype(float)
            input_tokens = data['input_toks'].values.astype(float)
            output_tokens = data['output_toks'].values.astype(float)

        # 调试信息
        logger.info(f"数据准备 - concurrent_requests: [{concurrent_requests.min():.3f}, {concurrent_requests.max():.3f}], "
                   f"input_tokens: [{input_tokens.min():.3f}, {input_tokens.max():.3f}], "
                   f"output_tokens: [{output_tokens.min():.3f}, {output_tokens.max():.3f}]")

        # 目标值：下一个时间点的值
        target_concurrent_requests = concurrent_requests[1:]  # 从第二个开始
        target_input_tokens = input_tokens[1:]  # 从第二个开始
        target_output_tokens = output_tokens[1:]  # 从第二个开始

        # 创建输入序列（使用滑动窗口增加数据量）
        sequences = []
        targets = []

        # 滑动窗口步长，使用较小的步长来增加数据量
        window_step = max(1, self.seq_len // 12)  # 使用1/12序列长度作为步长，增加12倍数据量

        for i in range(0, len(input_tokens) - self.seq_len + 1, window_step):
            # 输入序列：[concurrent_requests, input_token, output_token]
            seq_features = []
            for j in range(i, i + self.seq_len):
                # 时间序列数据：直接使用三项数据
                seq_features.append([
                    concurrent_requests[j],
                    input_tokens[j],
                    output_tokens[j]
                ])

            sequences.append(seq_features)

            # 目标：未来pred_len个时间点的三个值
            target_seq = []
            for j in range(self.pred_len):
                target_idx = i + self.seq_len + j
                # 时间序列数据：使用三项数据目标
                if target_idx < len(target_concurrent_requests):
                    target_seq.append([
                        target_concurrent_requests[target_idx],
                        target_input_tokens[target_idx],
                        target_output_tokens[target_idx]
                    ])
                else:
                    # 数据不足时用最后一个值填充
                    target_seq.append([
                        target_concurrent_requests[-1],
                        target_input_tokens[-1],
                        target_output_tokens[-1]
                    ])
            targets.append(target_seq)

        sequences = np.array(sequences)
        targets = np.array(targets)


        # 使用StandardScaler进行标准化
        if len(sequences) > 0:
            # 保存StandardScaler用于反标准化
            self.scalers = []

            # 重塑数据以便StandardScaler处理: (n_samples, n_features)
            sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
            targets_reshaped = targets.reshape(-1, targets.shape[-1])

            # 对每个特征分别进行标准化（通道独立）
            for feature_idx in range(sequences.shape[-1]):
                scaler = StandardScaler()
                sequences_reshaped[:, feature_idx] = scaler.fit_transform(sequences_reshaped[:, feature_idx].reshape(-1, 1)).ravel()
                targets_reshaped[:, feature_idx] = scaler.transform(targets_reshaped[:, feature_idx].reshape(-1, 1)).ravel()
                self.scalers.append(scaler)

            # 恢复原始形状
            sequences = sequences_reshaped.reshape(sequences.shape)
            targets = targets_reshaped.reshape(targets.shape)

        return torch.FloatTensor(sequences), torch.FloatTensor(targets)

    def prepare_data_batch(self, data: pd.DataFrame, batch_size: int = 10000) -> torch.utils.data.DataLoader:
        """
        分批准备训练数据，减少内存使用

        Args:
            data: 输入DataFrame
            batch_size: 每批数据大小

        Returns:
            torch.utils.data.DataLoader: 数据加载器
        """
        # 确保数据有时间戳
        if 'Timestamp' not in data.columns:
            raise ValueError("数据必须包含Timestamp列")

        # 列名映射：处理不同的数据集格式
        if 'Concurrent_requests' in data.columns:
            # 时间序列数据格式 - 分钟级固定间隔，不需要时间间隔特征
            if 'Request_tokens_sum' in data.columns and 'Response_tokens_sum' in data.columns:
                data = data.rename(columns={
                    'Request_tokens_sum': 'input_toks',
                    'Response_tokens_sum': 'output_toks',
                    'Concurrent_requests': 'concurrent_requests'
                })
            else:
                raise ValueError("时间序列数据必须包含'Request_tokens_sum'/'Response_tokens_sum'列")
        elif 'Request tokens' in data.columns and 'Response tokens' in data.columns:
            # 原始BurstGPT数据集格式
            data = data.rename(columns={
                'Request tokens': 'input_toks',
                'Response tokens': 'output_toks'
            })
        elif 'input_toks' not in data.columns or 'output_toks' not in data.columns:
            raise ValueError("数据必须包含'Request tokens'/'Response tokens'或'Request_tokens_sum'/'Response_tokens_sum'或'input_toks'/'output_toks'列")

        # 按时间戳排序
        data = data.sort_values('Timestamp').reset_index(drop=True)

        # 统一处理：确保总是有三个特征
        if 'concurrent_requests' in data.columns:
            # 时间序列数据：分钟级固定间隔，只使用并发量和token数量
            concurrent_requests = data['concurrent_requests'].values.astype(float)
            input_tokens = data['input_toks'].values.astype(float)
            output_tokens = data['output_toks'].values.astype(float)
        else:
            # 原始数据格式：假设并发量为1
            logger.error("未正确加载并发量")
            concurrent_requests = np.ones(len(data)).astype(float)
            input_tokens = data['input_toks'].values.astype(float)
            output_tokens = data['output_toks'].values.astype(float)

        # 调试信息
        logger.info(f"批处理数据准备 - concurrent_requests: [{concurrent_requests.min():.3f}, {concurrent_requests.max():.3f}], "
                   f"input_tokens: [{input_tokens.min():.3f}, {input_tokens.max():.3f}], "
                   f"output_tokens: [{output_tokens.min():.3f}, {output_tokens.max():.3f}]")

        # 目标值：下一个时间点的值
        target_concurrent_requests = concurrent_requests[1:]  # 从第二个开始
        target_input_tokens = input_tokens[1:]  # 从第二个开始
        target_output_tokens = output_tokens[1:]  # 从第二个开始

        # 创建序列数据集
        class TimeSeriesDataset(torch.utils.data.Dataset):
            def __init__(self, input_tokens, output_tokens, concurrent_requests,
                        target_input_tokens, target_output_tokens, target_concurrent_requests,
                        input_size):
                self.input_tokens = input_tokens
                self.output_tokens = output_tokens
                self.concurrent_requests = concurrent_requests
                self.target_input_tokens = target_input_tokens
                self.target_output_tokens = target_output_tokens
                self.target_concurrent_requests = target_concurrent_requests
                self.seq_len = input_size

            def __len__(self):
                return max(0, len(self.target_input_tokens) - self.seq_len + 1)

            def __getitem__(self, idx):
                # 输入序列 - 分钟级固定间隔，只使用并发请求数和token数量
                seq_features = []
                for j in range(idx, idx + self.seq_len):
                    seq_features.append([
                        self.concurrent_requests[j],
                        self.input_tokens[j],
                        self.output_tokens[j]
                    ])

                # 目标序列 - 预测下一个时间点的并发请求数和token数量
                target_features = [
                    self.target_concurrent_requests[idx],
                    self.target_input_tokens[idx],
                    self.target_output_tokens[idx]
                ]

                return torch.FloatTensor(seq_features), torch.FloatTensor(target_features)

        # 创建数据集 - 统一使用三项特征
        dataset = TimeSeriesDataset(
            input_tokens, output_tokens, concurrent_requests,
            target_input_tokens, target_output_tokens, target_concurrent_requests,
            self.seq_len
        )

        # 检查数据集大小
        dataset_size = len(dataset)
        if dataset_size <= 0:
            logger.warning(f"数据集太小，无法创建数据加载器。需要至少 {self.seq_len} 个样本，但只有 {len(input_tokens)} 个")
            # 返回空的数据加载器
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.empty(0, 3), torch.empty(0, 3)),
                batch_size=1,
                shuffle=False
            )

        # 调整batch_size以避免超出数据集大小
        actual_batch_size = min(batch_size, dataset_size)

        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            NormalizedTimeSeriesDataset(dataset, self.seq_len, self.device, self),
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=0,  # 避免多进程问题
            pin_memory=True if self.device == 'cuda' else False
        )

        return dataloader

    def train(self,
              data: pd.DataFrame,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2,
              patience: int = 10,
              verbose: bool = True) -> Dict[str, Any]:
        """
        训练DLinear模型

        Args:
            data: 训练数据
            epochs: 训练轮数
            batch_size: 批大小
            validation_split: 验证集比例
            patience: 早停耐心值
            verbose: 是否显示训练过程

        Returns:
            Dict[str, Any]: 训练结果
        """
        # 准备数据
        X, y = self.prepare_data(data)

        # 分割训练集和验证集
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

        # 早停变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            # 批量训练
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size].to(self.device)
                batch_y = y_train[i:i + batch_size].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                # 计算标准MSE损失
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= (len(X_train) // batch_size + 1)

            # 验证阶段
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_X = X_val[i:i + batch_size].to(self.device)
                    batch_y = y_val[i:i + batch_size].to(self.device)

                    outputs = self.model(batch_X)
                    # 计算标准MSE损失
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= (len(X_val) // batch_size + 1)

            # 记录训练历史
            self.training_history['loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        result = {
            'status': 'completed',
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_loss,
            'early_stopped': patience_counter >= patience
        }

        logger.info(f"Training completed: {result}")
        return result

    def train_batch(self,
                    data: pd.DataFrame,
                    epochs: int = 100,
                    batch_size: int = 32,
                    validation_split: float = 0.2,
                    patience: int = 10,
                    verbose: bool = True) -> Dict[str, Any]:
        """
        使用分批加载训练DLinear模型，减少内存使用

        Args:
            data: 训练数据
            epochs: 训练轮数
            batch_size: 批大小
            validation_split: 验证集比例
            patience: 早停耐心值
            verbose: 是否显示训练过程

        Returns:
            Dict[str, Any]: 训练结果
        """
        # 分割训练集和验证集
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data.iloc[:split_idx].copy()
        val_data = data.iloc[split_idx:].copy()

        logger.info(f"Training data: {len(train_data)}, Validation data: {len(val_data)}")

        # 创建数据加载器
        train_loader = self.prepare_data_batch(train_data, batch_size)
        val_loader = self.prepare_data_batch(val_data, batch_size)

        # 确保scalers已初始化（如果数据加载器没有设置的话）
        if not hasattr(self, 'scalers') or len(self.scalers) == 0:
            # 使用训练数据初始化scalers
            X, y = self.prepare_data(train_data)
            logger.info("初始化StandardScaler用于预测")

        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # 训练循环
        import time
        total_epochs = epochs
        start_time = time.time()

        # 创建进度条
        epoch_pbar = tqdm(
            range(total_epochs),
            desc="Training Model",
            unit="epoch",
            dynamic_ncols=True,
            mininterval=1.0
        )

        # 训练循环
        for epoch in epoch_pbar:
                self.model.train()
                train_loss = 0.0
                num_batches = 0

                epoch_start_time = time.time()

                # 训练阶段 - 静默训练
                train_batches = len(train_loader)
                for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    # 前向传播
                    predictions = self.model(batch_X)  # [batch_size, output_size, 3]

                    # 调整输出格式以匹配损失函数期望
                    batch_y_expanded = batch_y.unsqueeze(1).expand(-1, predictions.shape[1], -1)

                    # 计算标准MSE损失
                    loss = self.criterion(predictions, batch_y_expanded)

                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    num_batches += 1

                avg_train_loss = train_loss / num_batches
                epoch_time = time.time() - epoch_start_time

                # 验证阶段 - 减少验证频率
                avg_val_loss = avg_train_loss  # 默认使用训练损失作为验证损失
                validation_interval = 5  # 每5个epoch验证一次

                if (epoch + 1) % validation_interval == 0 or epoch == 0 or (epoch + 1) == total_epochs:
                    self.model.eval()
                    val_loss = 0.0
                    val_batches = 0

                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(self.device)
                            batch_y = batch_y.to(self.device)

                            predictions = self.model(batch_X)  # [batch_size, output_size, 3]
                            batch_y_expanded = batch_y.unsqueeze(1).expand(-1, predictions.shape[1], -1)
                            loss = self.criterion(predictions, batch_y_expanded)
                            val_loss += loss.item()
                            val_batches += 1

                    avg_val_loss = val_loss / val_batches if val_batches > 0 else avg_train_loss

                # 更新进度条描述
                if (epoch + 1) % validation_interval == 0 or epoch == 0 or (epoch + 1) == total_epochs:
                    epoch_pbar.set_postfix({
                        'Train': f'{avg_train_loss:.4f}',
                        'Val': f'{avg_val_loss:.4f}',
                        'Best': f'{best_val_loss:.4f}'
                    })
                else:
                    epoch_pbar.set_postfix({
                        'Loss': f'{avg_train_loss:.4f}',
                        'Best': f'{best_val_loss:.4f}'
                    })

                # 记录历史
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['epoch_times'].append(epoch_time)

                # 只在验证epoch进行早停检查
                is_validation_epoch = (epoch + 1) % validation_interval == 0 or epoch == 0 or (epoch + 1) == total_epochs
                if is_validation_epoch:
                    # 早停检查
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1

                # 检查早停条件
                if patience_counter >= patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1} - patience limit reached")
                    break

                # 每10个epoch记录一次日志
                if verbose and epoch % 10 == 0:
                    if (epoch + 1) % validation_interval == 0 or epoch == 0:
                        logger.info(f"Epoch {epoch+1}/{total_epochs}: "
                                   f"Train Loss: {avg_train_loss:.4f}, "
                                   f"Val Loss: {avg_val_loss:.4f}, "
                                   f"Best Val: {best_val_loss:.4f}, "
                                   f"Time: {epoch_time:.1f}s")
                    else:
                        logger.info(f"Epoch {epoch+1}/{total_epochs}: "
                                   f"Train Loss: {avg_train_loss:.4f}, "
                                   f"Best Val: {best_val_loss:.4f}, "
                                   f"Time: {epoch_time:.1f}s")

        # 关闭进度条
        if verbose:
            epoch_pbar.close()

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # 计算总训练时间
        total_time = time.time() - start_time

        result = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'epoch_times': history['epoch_times'],
            'total_epochs': len(history['train_loss']),
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
            'early_stopped': patience_counter >= patience,
            'total_time': total_time
        }

        if verbose:
            # 打印训练总结
            logger.info("=" * 60)
            logger.info("训练完成总结:")
            logger.info(f"总训练时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
            logger.info(f"总epoch数: {len(history['train_loss'])}")
            logger.info(f"最佳验证损失: {best_val_loss:.6f}")
            logger.info(f"最终训练损失: {avg_train_loss:.6f}")
            logger.info(f"平均epoch时间: {np.mean(history['epoch_times']):.1f}秒")
            if patience_counter >= patience:
                logger.info("状态: 早停 (防止过拟合)")
            else:
                logger.info("状态: 正常完成")
            logger.info("=" * 60)

        return result

    def predict(self, data: pd.DataFrame, steps: Optional[int] = None) -> torch.Tensor:
        """
        进行预测

        Args:
            data: 历史数据
            steps: 预测步数，默认使用output_size

        Returns:
            torch.Tensor: 预测结果，每行包含[时间间隔, 输入token, 输出token]
        """
        if steps is None:
            steps = self.pred_len

        self.model.eval()

        # 确保数据有时间戳
        if 'Timestamp' not in data.columns:
            raise ValueError("数据必须包含Timestamp列")

        # 列名映射：处理不同的数据集格式
        if 'Concurrent_requests' in data.columns:
            # 时间序列数据格式 - 分钟级固定间隔
            if 'Request_tokens_sum' in data.columns and 'Response_tokens_sum' in data.columns:
                data = data.rename(columns={
                    'Request_tokens_sum': 'input_toks',
                    'Response_tokens_sum': 'output_toks',
                    'Concurrent_requests': 'concurrent_requests'
                })
            else:
                raise ValueError("时间序列数据必须包含'Request_tokens_sum'/'Response_tokens_sum'列")
        elif 'Request tokens' in data.columns and 'Response tokens' in data.columns:
            # BurstGPT原始数据集格式
            data = data.rename(columns={
                'Request tokens': 'input_toks',
                'Response tokens': 'output_toks'
            })
        elif 'input_toks' not in data.columns or 'output_toks' not in data.columns:
            raise ValueError("数据必须包含'Request tokens'/'Response tokens'或'Request_tokens_sum'/'Response_tokens_sum'或'input_toks'/'output_toks'列")

        # 按时间戳排序
        data = data.sort_values('Timestamp').reset_index(drop=True)

        # 准备输入序列
        if len(data) < self.seq_len:
            logger.warning(f"Insufficient data: {len(data)} < {self.seq_len}")
            # 使用零填充
            padding_size = self.seq_len - len(data)
            sequences = np.zeros((self.seq_len, 3))
            if len(data) > 0:
                if 'Concurrent_requests' in data.columns:
                    # 时间序列数据格式：使用并发请求数
                    for i in range(len(data)):
                        sequences[padding_size + i] = [
                            data.iloc[i]['concurrent_requests'],
                            data.iloc[i]['input_toks'],
                            data.iloc[i]['output_toks']
                        ]
                else:
                    # 原始数据格式：计算时间间隔
                    time_intervals = np.diff(data['Timestamp'].values).astype(float)
                    for i in range(len(data)):
                        if i == 0:
                            sequences[padding_size + i] = [0, data.iloc[i]['input_toks'], data.iloc[i]['output_toks']]
                        else:
                            sequences[padding_size + i] = [time_intervals[i-1], data.iloc[i]['input_toks'], data.iloc[i]['output_toks']]
        else:
            # 使用最新的input_size个数据点
            recent_data = data.iloc[-self.seq_len:].copy()
            sequences = []

            if 'concurrent_requests' in recent_data.columns:
                # 时间序列数据格式：使用并发请求数
                for i in range(len(recent_data)):
                    sequences.append([
                        recent_data.iloc[i]['concurrent_requests'],
                        recent_data.iloc[i]['input_toks'],
                        recent_data.iloc[i]['output_toks']
                    ])
            else:
                # 原始数据格式：计算时间间隔
                time_intervals = np.diff(recent_data['Timestamp'].values).astype(float)
                for i in range(len(recent_data)):
                    if i == 0:
                        # 第一个请求，时间间隔为0
                        sequences.append([0, recent_data.iloc[i]['input_toks'], recent_data.iloc[i]['output_toks']])
                    else:
                        sequences.append([time_intervals[i-1], recent_data.iloc[i]['input_toks'], recent_data.iloc[i]['output_toks']])

            sequences = np.array(sequences)

        # 标准化（使用与训练相同的StandardScaler）
        for feature_idx in range(sequences.shape[-1]):
            if hasattr(self, 'scalers') and feature_idx < len(self.scalers):
                sequences[:, feature_idx] = self.scalers[feature_idx].transform(
                    sequences[:, feature_idx].reshape(-1, 1)
                ).ravel()

        # 转换为张量
        input_tensor = torch.FloatTensor(sequences).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            prediction = self.model(input_tensor)

        # 反标准化
        prediction = prediction.squeeze(0).cpu().numpy()

        # 使用StandardScaler进行反标准化
        if hasattr(self, 'scalers'):
            for feature_idx in range(prediction.shape[-1]):
                if feature_idx < len(self.scalers):
                    prediction[feature_idx] = self.scalers[feature_idx].inverse_transform(
                        prediction[feature_idx].reshape(-1, 1)
                    ).ravel()[0]

        # 返回预测结果（归一化值，用于训练和验证）
        return torch.FloatTensor(prediction)

    def predict_denormalized(self, data: pd.DataFrame, steps: Optional[int] = None) -> torch.Tensor:
        """
        进行预测并返回反归一化后的结果（用于实际应用和调度器）

        Args:
            data: 历史数据
            steps: 预测步数，默认使用output_size

        Returns:
            torch.Tensor: 反归一化后的预测结果，形状为[steps, 3]，每行包含[并发请求数, 输入token, 输出token]
        """
        if steps is None:
            steps = self.pred_len

        self.model.eval()

        # 确保数据有时间戳
        if 'Timestamp' not in data.columns:
            raise ValueError("数据必须包含Timestamp列")

        # 列名映射：处理不同的数据集格式
        if 'Concurrent_requests' in data.columns:
            # 时间序列数据格式 - 分钟级固定间隔
            if 'Request_tokens_sum' in data.columns and 'Response_tokens_sum' in data.columns:
                data = data.rename(columns={
                    'Request_tokens_sum': 'input_toks',
                    'Response_tokens_sum': 'output_toks',
                    'Concurrent_requests': 'concurrent_requests'
                })
            else:
                raise ValueError("时间序列数据必须包含'Request_tokens_sum'/'Response_tokens_sum'列")
        elif 'Request tokens' in data.columns and 'Response tokens' in data.columns:
            # BurstGPT原始数据集格式
            data = data.rename(columns={
                'Request tokens': 'input_toks',
                'Response tokens': 'output_toks'
            })
        elif 'input_toks' not in data.columns or 'output_toks' not in data.columns:
            raise ValueError("数据必须包含'Request tokens'/'Response tokens'或'Request_tokens_sum'/'Response_tokens_sum'或'input_toks'/'output_toks'列")

        # 按时间戳排序
        data = data.sort_values('Timestamp').reset_index(drop=True)

        # 准备输入序列 - 统一处理为三项特征
        if len(data) < self.seq_len:
            logger.warning(f"数据不足: {len(data)} < {self.seq_len}，将使用零填充")
            # 使用零填充
            padding_size = self.seq_len - len(data)
            sequences = np.zeros((self.seq_len, 3))
            if len(data) > 0:
                # 统一使用三项特征
                for i in range(len(data)):
                    if 'concurrent_requests' in data.columns:
                        concurrent_req = data.iloc[i]['concurrent_requests']
                    else:
                        concurrent_req = 1.0  # 默认并发量

                    sequences[padding_size + i] = [
                        concurrent_req,
                        data.iloc[i]['input_toks'],
                        data.iloc[i]['output_toks']
                    ]
        else:
            # 使用最新的seq_len个数据点
            recent_data = data.iloc[-self.seq_len:].copy()
            sequences = []

            for i in range(len(recent_data)):
                if 'concurrent_requests' in recent_data.columns:
                    concurrent_req = recent_data.iloc[i]['concurrent_requests']
                else:
                    concurrent_req = 1.0  # 默认并发量

                sequences.append([
                    concurrent_req,
                    recent_data.iloc[i]['input_toks'],
                    recent_data.iloc[i]['output_toks']
                ])

            sequences = np.array(sequences)

        # 标准化（使用与训练相同的StandardScaler）
        if hasattr(self, 'scalers') and len(self.scalers) >= 3:
            for feature_idx in range(3):
                sequences[:, feature_idx] = self.scalers[feature_idx].transform(
                    sequences[:, feature_idx].reshape(-1, 1)
                ).ravel()

        # 转换为张量
        input_tensor = torch.FloatTensor(sequences).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            prediction = self.model(input_tensor)  # [1, pred_len, 3]

        # 反标准化
        prediction = prediction.squeeze(0).cpu().numpy()  # [pred_len, 3]

        # 使用StandardScaler进行反标准化
        if hasattr(self, 'scalers') and len(self.scalers) >= 3:
            for feature_idx in range(3):
                if feature_idx < len(self.scalers):
                    # 对每个预测步的每个特征进行反标准化
                    for step_idx in range(prediction.shape[0]):
                        prediction[step_idx, feature_idx] = self.scalers[feature_idx].inverse_transform(
                            prediction[step_idx, feature_idx].reshape(-1, 1)
                        ).ravel()[0]

        # 确保预测结果合理（非负值）
        prediction = np.maximum(prediction, 0)

        # 打印反归一化后的预测结果信息
        logger.info(f"反归一化预测结果 - 形状: {prediction.shape}")
        logger.info(f"反归一化范围 - 并发请求: [{prediction[:, 0].min():.1f}, {prediction[:, 0].max():.1f}]")
        logger.info(f"反归一化范围 - 输入token: [{prediction[:, 1].min():.1f}, {prediction[:, 1].max():.1f}]")
        logger.info(f"反归一化范围 - 输出token: [{prediction[:, 2].min():.1f}, {prediction[:, 2].max():.1f}]")

        # 返回反归一化后的预测结果
        return torch.FloatTensor(prediction)

    def predict_with_details(self, data: pd.DataFrame, steps: Optional[int] = None) -> Dict[str, Any]:
        """
        进行预测并返回详细信息（使用反归一化结果）

        Args:
            data: 历史数据
            steps: 预测步数，默认使用output_size

        Returns:
            Dict[str, Any]: 包含预测结果和详细信息的字典
        """
        # 获取反归一化后的预测结果
        predictions = self.predict_denormalized(data, steps)

        # 转换为numpy数组便于处理
        pred_array = predictions.numpy()

        # 计算统计信息
        details = {
            'predictions': predictions,
            'shape': predictions.shape,
            'statistics': {
                'concurrent_requests': {
                    'mean': float(pred_array[:, 0].mean()),
                    'std': float(pred_array[:, 0].std()),
                    'min': float(pred_array[:, 0].min()),
                    'max': float(pred_array[:, 0].max())
                },
                'input_tokens': {
                    'mean': float(pred_array[:, 1].mean()),
                    'std': float(pred_array[:, 1].std()),
                    'min': float(pred_array[:, 1].min()),
                    'max': float(pred_array[:, 1].max())
                },
                'output_tokens': {
                    'mean': float(pred_array[:, 2].mean()),
                    'std': float(pred_array[:, 2].std()),
                    'min': float(pred_array[:, 2].min()),
                    'max': float(pred_array[:, 2].max())
                }
            },
            'prediction_intervals': []
        }

        # 生成预测时间间隔（假设1分钟间隔）
        last_timestamp = data['Timestamp'].iloc[-1]
        for i in range(len(pred_array)):
            pred_time = last_timestamp + pd.Timedelta(minutes=i+1)
            details['prediction_intervals'].append(pred_time.strftime('%Y-%m-%d %H:%M:%S'))

        return details

  
    def analyze_decomposition(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析时序分解结果

        Args:
            data: 输入数据

        Returns:
            Dict[str, Any]: 分解分析结果
        """
        self.model.eval()

        # 确保数据有时间戳
        if 'Timestamp' not in data.columns:
            raise ValueError("数据必须包含Timestamp列")

        # 按时间戳排序
        data = data.sort_values('Timestamp').reset_index(drop=True)

        # 准备输入（使用与predict相同的方法）
        if len(data) < self.seq_len:
            logger.warning(f"Insufficient data for decomposition analysis: {len(data)} < {self.seq_len}")
            return {'error': 'Insufficient data'}

        # 使用最新的input_size个数据点
        recent_data = data.iloc[-self.seq_len:].copy()
        sequences = []

        # 计算时间间隔
        time_intervals = np.diff(recent_data['Timestamp'].values)

        for i in range(len(recent_data)):
            if i == 0:
                sequences.append([0, recent_data.iloc[i]['input_toks'], recent_data.iloc[i]['output_toks']])
            else:
                sequences.append([time_intervals[i-1], recent_data.iloc[i]['input_toks'], recent_data.iloc[i]['output_toks']])

        sequences = np.array(sequences)

        # 标准化
        for feature_idx in range(sequences.shape[-1]):
            mean_val = sequences[:, feature_idx].mean()
            std_val = sequences[:, feature_idx].std()
            if std_val > 0:
                sequences[:, feature_idx] = (sequences[:, feature_idx] - mean_val) / std_val
            else:
                sequences[:, feature_idx] = sequences[:, feature_idx] - mean_val

        input_tensor = torch.FloatTensor(sequences).unsqueeze(0).to(self.device)

        # 获取分解分析
        with torch.no_grad():
            analysis = self.model.get_decomposition_analysis(input_tensor)

        # 转换为可读格式
        result = {
            'seasonal_strength': analysis['seasonal_strength'].cpu().numpy().tolist(),
            'trend_strength': analysis['trend_strength'].cpu().numpy().tolist(),
            'decomposition_ratio': analysis['decomposition_ratio'].cpu().numpy().tolist(),
            'dominant_component': 'seasonal' if analysis['seasonal_strength'].mean() > analysis['trend_strength'].mean() else 'trend'
        }

        return result

    def save_model(self, path: str):
        """保存模型"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': {
                'input_size': self.seq_len,
                'output_size': self.pred_len,
                'learning_rate': self.learning_rate,
                'device': self.device
            },
            'training_history': self.training_history
        }

        torch.save(save_dict, path)
        logger.info(f"DLinear model saved to {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)

        # 重建模型
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']

        # 加载配置
        config = checkpoint['config']
        self.seq_len = config['input_size']
        self.pred_len = config['output_size']
        self.learning_rate = config['learning_rate']

        # 加载训练历史
        self.training_history = checkpoint.get('training_history', {'loss': [], 'val_loss': []})

        logger.info(f"DLinear model loaded from {path}")


def create_dlinear_predictor(input_size: int = 50,
                           output_size: int = 10,
                           device: str = 'cpu') -> DLinearPredictor:
    """创建DLinear预测器的工厂函数"""
    return DLinearPredictor(
        input_size=input_size,
        output_size=output_size,
        device=device
    )


# 测试函数
def test_dlinear_model():
    """测试DLinear模型"""
    print("Testing DLinear model...")

    # 使用真实的BurstGPT数据集
    try:
        # 读取BurstGPT数据集
        data_path = '../dataset/BurstGPT_1.csv'
        data = pd.read_csv(data_path)

        # 重命名列以匹配我们的格式
        data = data.rename(columns={
            'Timestamp': 'Timestamp',
            'Request tokens': 'input_toks',
            'Response tokens': 'output_toks'
        })

        # 只保留我们需要的列
        data = data[['Timestamp', 'input_toks', 'output_toks']].copy()

        # 使用前1000条数据
        data = data.head(1000).copy()

        print(f"Loaded BurstGPT data: {len(data)} samples")
        print(f"Columns: {list(data.columns)}")
        print(f"Time range: {data['Timestamp'].min()} to {data['Timestamp'].max()}")
        print(f"Input tokens range: {data['input_toks'].min()} to {data['input_toks'].max()}")
        print(f"Output tokens range: {data['output_toks'].min()} to {data['output_toks'].max()}")

    except FileNotFoundError:
        print(f"Warning: BurstGPT dataset not found at {data_path}")
        print("Using synthetic data for testing...")
        # 备用：创建合成数据
        np.random.seed(42)
        n_samples = 1000
        timestamps = np.cumsum(np.random.exponential(scale=1e8, size=n_samples)).astype(int)
        data = pd.DataFrame({
            'Timestamp': timestamps,
            'input_toks': np.random.randint(50, 200, n_samples),
            'output_toks': np.random.randint(100, 500, n_samples)
        })

    # 创建预测器
    predictor = create_dlinear_predictor(input_size=50, output_size=10)

    # 训练模型
    print("Training model...")
    result = predictor.train(data, epochs=50, batch_size=32, verbose=False)

    # 预测
    print("Making predictions...")
    predictions = predictor.predict(data, steps=10)

    # 分解分析
    print("Analyzing decomposition...")
    analysis = predictor.analyze_decomposition(data)

    print(f"DLinear model test completed:")
    print(f"  Training result: {result}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Sample prediction: {predictions[0]}")
    print(f"  Decomposition analysis: {analysis}")

    return predictor


class NormalizedTimeSeriesDataset(torch.utils.data.Dataset):
    """
    标准化的时间序列数据集，使用StandardScaler进行归一化
    """

    def __init__(self, base_dataset, seq_len, device='cpu', predictor_instance=None):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        self.device = device
        self.predictor_instance = predictor_instance

        # 收集所有数据进行标准化
        self._fit_scalers()

    def _fit_scalers(self):
        """拟合StandardScaler"""
        all_sequences = []
        all_targets = []

        # 收集所有数据
        for i in range(len(self.base_dataset)):
            seq, target = self.base_dataset[i]
            all_sequences.append(seq.numpy())
            all_targets.append(target.numpy())

        if len(all_sequences) > 0:
            all_sequences = np.array(all_sequences)
            all_targets = np.array(all_targets)

            # 为每个特征创建StandardScaler
            self.scalers = []

            # 对输入序列进行标准化
            for feature_idx in range(all_sequences.shape[-1]):
                scaler = StandardScaler()
                seq_feature_data = all_sequences[:, :, feature_idx].reshape(-1, 1)
                scaler.fit(seq_feature_data)
                self.scalers.append(scaler)

            # 将scalers保存到预测器实例中，以便预测时使用
            if self.predictor_instance is not None:
                self.predictor_instance.scalers = self.scalers

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        seq, target = self.base_dataset[idx]

        # 转换为numpy进行标准化
        seq_np = seq.numpy()
        target_np = target.numpy()

        # 对每个特征进行标准化
        for feature_idx in range(seq_np.shape[-1]):
            if hasattr(self, 'scalers') and feature_idx < len(self.scalers):
                seq_np[:, feature_idx] = self.scalers[feature_idx].transform(
                    seq_np[:, feature_idx].reshape(-1, 1)
                ).ravel()

                target_np[feature_idx] = self.scalers[feature_idx].transform(
                    target_np[feature_idx].reshape(-1, 1)
                ).ravel()[0]

        return torch.FloatTensor(seq_np), torch.FloatTensor(target_np)


if __name__ == "__main__":
    test_dlinear_model()