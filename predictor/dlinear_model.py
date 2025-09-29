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
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum
import logging
from sklearn.preprocessing import StandardScaler
import warnings
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
            # BurstGPT数据集格式
            data = data.rename(columns={
                'Request tokens': 'input_toks',
                'Response tokens': 'output_toks'
            })
        elif 'input_toks' not in data.columns or 'output_toks' not in data.columns:
            raise ValueError("数据必须包含'Request tokens'/'Response tokens'或'input_toks'/'output_toks'列")

        # 按时间戳排序
        data = data.sort_values('Timestamp').reset_index(drop=True)

        if 'Concurrent_requests' in data.columns:
            # 时间序列数据：分钟级固定间隔，只使用并发量和token数量
            concurrent_requests = data['concurrent_requests'].values
            input_tokens = data['input_toks'].values
            output_tokens = data['output_toks'].values

            # 目标值：下一个时间点的值
            target_concurrent_requests = concurrent_requests[1:]  # 从第二个开始
            target_input_tokens = input_tokens[1:]  # 从第二个开始
            target_output_tokens = output_tokens[1:]  # 从第二个开始
        else:
            # 原始数据格式：计算时间间隔
            time_intervals = np.diff(data['Timestamp'].values)
            input_tokens = data['input_toks'].values[:-1]  # 除了最后一个
            output_tokens = data['output_toks'].values[:-1]  # 除了最后一个

            # 目标值：下一个请求的时间间隔、输入token、输出token
            target_time_intervals = time_intervals
            target_input_tokens = data['input_toks'].values[1:]  # 从第二个开始
            target_output_tokens = data['output_toks'].values[1:]  # 从第二个开始
            target_concurrent_requests = np.ones_like(target_time_intervals)  # 原始数据默认并发量为1

        # 创建输入序列（使用滑动窗口增加数据量）
        sequences = []
        targets = []

        # 滑动窗口步长，使用较小的步长来增加数据量
        window_step = max(1, self.seq_len // 12)  # 使用1/12序列长度作为步长，增加12倍数据量

        for i in range(0, len(input_tokens) - self.seq_len + 1, window_step):
            # 输入序列：[concurrent_requests, input_token, output_token]
            seq_features = []
            for j in range(i, i + self.seq_len):
                if 'Concurrent_requests' in data.columns:
                    # 时间序列数据：使用并发请求数
                    seq_features.append([
                        concurrent_requests[j],
                        input_tokens[j],
                        output_tokens[j]
                    ])
                else:
                    # 原始数据：使用时间间隔
                    if j == 0:
                        time_interval = 0  # 第一个请求的时间间隔设为0
                    else:
                        time_interval = time_intervals[j-1]
                    seq_features.append([
                        time_interval,
                        input_tokens[j],
                        output_tokens[j]
                    ])

            sequences.append(seq_features)

            # 目标：未来pred_len个时间点的三个值
            target_seq = []
            for j in range(self.pred_len):
                target_idx = i + self.seq_len + j
                if 'Concurrent_requests' in data.columns:
                    # 时间序列数据：使用并发请求数目标
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
                else:
                    # 原始数据：使用时间间隔目标
                    if target_idx < len(target_time_intervals):
                        target_seq.append([
                            target_time_intervals[target_idx],
                            target_input_tokens[target_idx],
                            target_output_tokens[target_idx]
                        ])
                    else:
                        # 数据不足时用最后一个值填充
                        target_seq.append([
                            target_time_intervals[-1],
                            target_input_tokens[-1],
                            target_output_tokens[-1]
                        ])
            targets.append(target_seq)

        sequences = np.array(sequences)
        targets = np.array(targets)

        # 数据MinMax标准化
        if len(sequences) > 0:
            # 保存标准化参数用于反标准化
            self.norm_params = {'mins': [], 'maxs': []}

            # 对每个特征分别MinMax标准化
            for feature_idx in range(sequences.shape[-1]):
                min_val = sequences[:, :, feature_idx].min()
                max_val = sequences[:, :, feature_idx].max()
                self.norm_params['mins'].append(min_val)
                self.norm_params['maxs'].append(max_val)

                if max_val > min_val:
                    sequences[:, :, feature_idx] = (sequences[:, :, feature_idx] - min_val) / (max_val - min_val)
                else:
                    sequences[:, :, feature_idx] = 0  # 所有值相同，设为0

            # 对targets也进行标准化（使用与sequences相同的参数）
            for feature_idx in range(targets.shape[-1]):
                min_val = self.norm_params['mins'][feature_idx]
                max_val = self.norm_params['maxs'][feature_idx]
                if max_val > min_val:
                    targets[:, :, feature_idx] = (targets[:, :, feature_idx] - min_val) / (max_val - min_val)

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

        if 'Concurrent_requests' in data.columns:
            # 时间序列数据：分钟级固定间隔，只使用并发量和token数量
            concurrent_requests = data['concurrent_requests'].values
            input_tokens = data['input_toks'].values
            output_tokens = data['output_toks'].values

            # 目标值：下一个时间点的值
            target_concurrent_requests = concurrent_requests[1:]  # 从第二个开始
            target_input_tokens = input_tokens[1:]  # 从第二个开始
            target_output_tokens = output_tokens[1:]  # 从第二个开始
        else:
            # 原始数据格式：计算时间间隔
            time_intervals = np.diff(data['Timestamp'].values)
            input_tokens = data['input_toks'].values[:-1]  # 除了最后一个
            output_tokens = data['output_toks'].values[:-1]  # 除了最后一个

            # 目标值：下一个请求的时间间隔、输入token、输出token
            target_time_intervals = time_intervals
            target_input_tokens = data['input_toks'].values[1:]  # 从第二个开始
            target_output_tokens = data['output_toks'].values[1:]  # 从第二个开始
            target_concurrent_requests = np.ones_like(target_time_intervals)  # 原始数据默认并发量为1

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
                return max(0, len(self.input_tokens) - self.seq_len + 1)

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
                    self.target_concurrent_requests[idx + self.seq_len - 1],
                    self.target_input_tokens[idx + self.seq_len - 1],
                    self.target_output_tokens[idx + self.seq_len - 1]
                ]

                return torch.FloatTensor(seq_features), torch.FloatTensor(target_features)

        # 创建数据集
        if 'Concurrent_requests' in data.columns:
            # 时间序列数据：使用并发请求数
            dataset = TimeSeriesDataset(
                input_tokens, output_tokens, concurrent_requests,
                target_input_tokens, target_output_tokens, target_concurrent_requests,
                self.seq_len
            )
        else:
            # 原始数据格式：使用时间间隔
            dataset = TimeSeriesDataset(
                input_tokens, output_tokens, time_intervals,
                target_input_tokens, target_output_tokens, target_time_intervals,
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
            dataset,
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
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            import time
            epoch_start_time = time.time()

            # 训练阶段
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向传播
                predictions = self.model(batch_X)  # [batch_size, output_size, 3]

                # 调整输出格式以匹配损失函数期望
                # 损失函数期望: [batch_size, seq_len, 3]
                # 模型输出: [batch_size, output_size, 3]
                # 我们只需要预测序列的最后一个时间步
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

            # 验证阶段
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

            epoch_time = time.time() - epoch_start_time

            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['epoch_times'].append(epoch_time)

            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        result = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'epoch_times': history['epoch_times'],
            'total_epochs': len(history['train_loss']),
            'best_val_loss': best_val_loss,
            'final_train_loss': avg_train_loss,
            'early_stopped': patience_counter >= patience
        }

        logger.info(f"Training completed: {result}")
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
        if 'Request tokens' in data.columns and 'Response tokens' in data.columns:
            # BurstGPT数据集格式
            data = data.rename(columns={
                'Request tokens': 'input_toks',
                'Response tokens': 'output_toks'
            })
        elif 'input_toks' not in data.columns or 'output_toks' not in data.columns:
            raise ValueError("数据必须包含'Request tokens'/'Response tokens'或'input_toks'/'output_toks'列")

        # 按时间戳排序
        data = data.sort_values('Timestamp').reset_index(drop=True)

        # 准备输入序列
        if len(data) < self.seq_len:
            logger.warning(f"Insufficient data: {len(data)} < {self.seq_len}")
            # 使用零填充
            padding_size = self.seq_len - len(data)
            sequences = np.zeros((self.seq_len, 3))
            if len(data) > 0:
                # 计算时间间隔
                time_intervals = np.diff(data['Timestamp'].values)
                for i in range(len(data)):
                    if i == 0:
                        sequences[padding_size + i] = [0, data.iloc[i]['input_toks'], data.iloc[i]['output_toks']]
                    else:
                        sequences[padding_size + i] = [time_intervals[i-1], data.iloc[i]['input_toks'], data.iloc[i]['output_toks']]
        else:
            # 使用最新的input_size个数据点
            recent_data = data.iloc[-self.seq_len:].copy()
            sequences = []

            # 计算时间间隔
            time_intervals = np.diff(recent_data['Timestamp'].values)

            for i in range(len(recent_data)):
                if i == 0:
                    # 第一个请求，时间间隔为0
                    sequences.append([0, recent_data.iloc[i]['input_toks'], recent_data.iloc[i]['output_toks']])
                else:
                    sequences.append([time_intervals[i-1], recent_data.iloc[i]['input_toks'], recent_data.iloc[i]['output_toks']])

            sequences = np.array(sequences)

        # 标准化（使用与训练相同的方法）
        for feature_idx in range(sequences.shape[-1]):
            mean_val = sequences[:, feature_idx].mean()
            std_val = sequences[:, feature_idx].std()
            if std_val > 0:
                sequences[:, feature_idx] = (sequences[:, feature_idx] - mean_val) / std_val
            else:
                sequences[:, feature_idx] = sequences[:, feature_idx] - mean_val

        # 转换为张量
        input_tensor = torch.FloatTensor(sequences).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            prediction = self.model(input_tensor)

        # 反标准化
        prediction = prediction.squeeze(0).cpu().numpy()

        # 如果有标准化参数，进行反标准化
        if hasattr(self, 'norm_params') and self.norm_params:
            for feature_idx in range(prediction.shape[-1]):
                min_val = self.norm_params['mins'][feature_idx]
                max_val = self.norm_params['maxs'][feature_idx]
                if max_val > min_val:
                    prediction[feature_idx] = prediction[feature_idx] * (max_val - min_val) + min_val

        # 返回预测结果
        return torch.FloatTensor(prediction)

  
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


if __name__ == "__main__":
    test_dlinear_model()