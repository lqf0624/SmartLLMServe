"""
DLinear模型实现 - 基于AAAI 2023论文

"Are Transformers Effective for Time Series Forecasting?"
实现时序分解+线性网络的高效预测模型。

修改说明：
- 适配多任务Loss函数，直接预测三个目标值：时间间隔、输入token、输出token
- 使用真实的时间序列数据而非预处理后的特征
- 集成新的多任务损失函数

核心思想：
1. 将时间序列分解为趋势和季节性组件
2. 使用单层线性网络分别预测
3. 重建最终预测结果
4. 直接优化LLM服务的核心预测目标
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
                 input_size: int,
                 output_size: int,
                 individual: bool = False,
                 decomposition_mode: DecompositionMode = DecompositionMode.MOVING_AVG,
                 kernel_size: int = 25,
                 dropout: float = 0.1):
        """
        初始化DLinear模型

        Args:
            input_size: 输入序列长度
            output_size: 输出预测长度
            individual: 是否为每个特征单独建模
            decomposition_mode: 分解模式
            kernel_size: 移动平均窗口大小
            dropout: Dropout比率
        """
        super(DLinearModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.individual = individual
        self.decomposition_mode = decomposition_mode
        self.kernel_size = kernel_size

        # 时序分解层
        self.decomposition_layer = SeriesDecomposition(kernel_size, mode=decomposition_mode)

        # 线性预测层
        if individual:
            # 为每个特征单独建模
            self.linear_seasonal = nn.Linear(input_size, output_size)
            self.linear_trend = nn.Linear(input_size, output_size)
        else:
            # 共享线性层
            self.linear_seasonal = nn.Linear(input_size, output_size)
            self.linear_trend = nn.Linear(input_size, output_size)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

        logger.info(f"DLinear model initialized: input_size={input_size}, output_size={output_size}, "
                   f"individual={individual}, decomposition_mode={decomposition_mode.value}")

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, input_size, features]

        Returns:
            torch.Tensor: 预测结果 [batch_size, output_size, features]
        """
        # 时序分解
        seasonal, trend = self.decomposition_layer(x)

        # 预测季节性组件
        seasonal_pred = self.linear_seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        seasonal_pred = self.dropout(seasonal_pred)

        # 预测趋势组件
        trend_pred = self.linear_trend(trend.transpose(1, 2)).transpose(1, 2)
        trend_pred = self.dropout(trend_pred)

        # 重建预测结果
        prediction = seasonal_pred + trend_pred

        return prediction

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
                 input_size: int = 50,
                 output_size: int = 10,
                 individual: bool = False,
                 decomposition_mode: DecompositionMode = DecompositionMode.MOVING_AVG,
                 kernel_size: int = 25,
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        初始化DLinear预测器

        Args:
            input_size: 输入序列长度
            output_size: 输出预测长度
            individual: 是否为每个特征单独建模
            decomposition_mode: 分解模式
            kernel_size: 分解窗口大小
            learning_rate: 学习率
            device: 设备
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device

        # 创建模型
        self.model = DLinearModel(
            input_size=input_size,
            output_size=output_size,
            individual=individual,
            decomposition_mode=decomposition_mode,
            kernel_size=kernel_size
        ).to(device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # 多任务损失函数
        self.criterion = MultiTaskLoss(
            time_weight=1.0,
            input_token_weight=0.5,
            output_token_weight=0.5
        )

        # 数据标准化器
        self.scaler = StandardScaler()

        # 训练历史
        self.training_history = {
            'loss': [],
            'val_loss': []
        }

        logger.info(f"DLinearPredictor initialized: input_size={input_size}, output_size={output_size}")

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

        # 按时间戳排序
        data = data.sort_values('Timestamp').reset_index(drop=True)

        # 计算时间间隔（纳秒）
        time_intervals = np.diff(data['Timestamp'].values)

        # 提取特征和目标
        input_tokens = data['input_toks'].values[:-1]  # 除了最后一个
        output_tokens = data['output_toks'].values[:-1]  # 除了最后一个

        # 目标值：下一个请求的时间间隔、输入token、输出token
        target_time_intervals = time_intervals
        target_input_tokens = data['input_toks'].values[1:]  # 从第二个开始
        target_output_tokens = data['output_toks'].values[1:]  # 从第二个开始

        # 创建输入序列（包含时间间隔、输入token、输出token）
        sequences = []
        targets = []

        for i in range(len(input_tokens) - self.input_size + 1):
            # 输入序列：[time_interval, input_token, output_token]
            seq_features = []
            for j in range(i, i + self.input_size):
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

            # 目标：未来output_size个请求的三个值
            target_seq = []
            for j in range(self.output_size):
                target_idx = i + self.input_size + j
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

        # 标准化
        if len(sequences) > 0:
            # 对每个特征分别标准化
            for feature_idx in range(sequences.shape[-1]):
                mean_val = sequences[:, :, feature_idx].mean()
                std_val = sequences[:, :, feature_idx].std()
                if std_val > 0:
                    sequences[:, :, feature_idx] = (sequences[:, :, feature_idx] - mean_val) / std_val
                else:
                    sequences[:, :, feature_idx] = sequences[:, :, feature_idx] - mean_val

        return torch.FloatTensor(sequences), torch.FloatTensor(targets)

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

                # 计算多任务损失
                loss_dict = self.criterion(outputs, batch_y)
                loss = loss_dict['total_loss']
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
                    # 计算多任务损失
                    loss_dict = self.criterion(outputs, batch_y)
                    loss = loss_dict['total_loss']
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
            steps = self.output_size

        self.model.eval()

        # 确保数据有时间戳
        if 'Timestamp' not in data.columns:
            raise ValueError("数据必须包含Timestamp列")

        # 按时间戳排序
        data = data.sort_values('Timestamp').reset_index(drop=True)

        # 准备输入序列
        if len(data) < self.input_size:
            logger.warning(f"Insufficient data: {len(data)} < {self.input_size}")
            # 使用零填充
            padding_size = self.input_size - len(data)
            sequences = np.zeros((self.input_size, 3))
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
            recent_data = data.iloc[-self.input_size:].copy()
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
        if len(data) < self.input_size:
            logger.warning(f"Insufficient data for decomposition analysis: {len(data)} < {self.input_size}")
            return {'error': 'Insufficient data'}

        # 使用最新的input_size个数据点
        recent_data = data.iloc[-self.input_size:].copy()
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
                'input_size': self.input_size,
                'output_size': self.output_size,
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
        self.input_size = config['input_size']
        self.output_size = config['output_size']
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