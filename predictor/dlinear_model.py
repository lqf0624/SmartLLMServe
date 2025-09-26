"""
DLinear模型实现 - 基于AAAI 2023论文

"Are Transformers Effective for Time Series Forecasting?"
实现时序分解+线性网络的高效预测模型。

核心思想：
1. 将时间序列分解为趋势和季节性组件
2. 使用单层线性网络分别预测
3. 重建最终预测结果
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

        # 损失函数
        self.criterion = nn.MSELoss()

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
            data: 输入DataFrame，包含input_toks和output_toks列

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (输入序列, 目标序列)
        """
        # 提取特征
        features = data[['input_toks', 'output_toks']].values

        # 标准化
        features_scaled = self.scaler.fit_transform(features)

        # 创建序列
        sequences = []
        targets = []

        for i in range(len(features_scaled) - self.input_size - self.output_size + 1):
            seq = features_scaled[i:i + self.input_size]
            target = features_scaled[i + self.input_size:i + self.input_size + self.output_size]
            sequences.append(seq)
            targets.append(target)

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

    def predict(self, data: pd.DataFrame, steps: Optional[int] = None) -> torch.Tensor:
        """
        进行预测

        Args:
            data: 历史数据
            steps: 预测步数，默认使用output_size

        Returns:
            torch.Tensor: 预测结果
        """
        if steps is None:
            steps = self.output_size

        self.model.eval()

        # 准备输入序列
        if len(data) < self.input_size:
            # 数据不足时使用零填充
            logger.warning(f"Insufficient data: {len(data)} < {self.input_size}")
            padding_size = self.input_size - len(data)
            features = np.zeros((self.input_size, 2))
            if len(data) > 0:
                features[-len(data):] = data[['input_toks', 'output_toks']].values
        else:
            features = data[['input_toks', 'output_toks']].values[-self.input_size:]

        # 标准化
        features_scaled = self.scaler.transform(features)

        # 转换为张量
        input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            prediction = self.model(input_tensor)

        # 反标准化
        prediction = prediction.squeeze(0).cpu().numpy()
        prediction = self.scaler.inverse_transform(prediction)

        # 如果需要调整预测步数
        if steps != self.output_size:
            if steps < self.output_size:
                prediction = prediction[:steps]
            else:
                # 对于更长的预测，使用递归预测
                prediction = self._recursive_predict(data, steps)

        return torch.FloatTensor(prediction)

    def _recursive_predict(self, data: pd.DataFrame, steps: int) -> np.ndarray:
        """递归预测长序列"""
        predictions = []
        current_data = data.copy()

        for _ in range(0, steps, self.output_size):
            # 预测下一个output_size步
            pred = self.predict(current_data, min(self.output_size, steps - len(predictions)))
            predictions.append(pred.numpy())

            # 更新数据（使用预测的值）
            new_data = pd.DataFrame({
                'input_toks': pred[:, 0],
                'output_toks': pred[:, 1]
            })
            current_data = pd.concat([current_data, new_data], ignore_index=True)

        # 合并所有预测
        result = np.vstack(predictions)
        return result[:steps]  # 确保返回正确的步数

    def analyze_decomposition(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析时序分解结果

        Args:
            data: 输入数据

        Returns:
            Dict[str, Any]: 分解分析结果
        """
        self.model.eval()

        # 准备输入
        if len(data) < self.input_size:
            features = np.zeros((self.input_size, 2))
            if len(data) > 0:
                features[-len(data):] = data[['input_toks', 'output_toks']].values
        else:
            features = data[['input_toks', 'output_toks']].values[-self.input_size:]

        features_scaled = self.scaler.transform(features)
        input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)

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

    # 创建测试数据
    np.random.seed(42)
    data = pd.DataFrame({
        'input_toks': np.random.randint(50, 200, 1000),
        'output_toks': np.random.randint(100, 500, 1000)
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
    print(f"  Decomposition analysis: {analysis}")

    return predictor


if __name__ == "__main__":
    test_dlinear_model()