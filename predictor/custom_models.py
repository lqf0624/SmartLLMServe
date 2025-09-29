"""
轻量化自定义预测模型

专注于LSTM和DLinear的高效实现，删除Transformer和GRU模型以确保轻量化。
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from .model_interface import (
    BasePredictor, ModelConfig, PredictionResult, TrainingResult,
    ModelType, ModelDevice, register_model
)
from .multi_task_loss import MultiTaskLoss, LOSS_CONFIGS

logger = logging.getLogger(__name__)


@register_model(ModelType.CUSTOM)
class LightweightPredictor(BasePredictor):
    """
    轻量级预测模型示例

    展示如何使用新的多任务损失函数来实现高效的预测模型。
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.scaler = None
        self.multi_task_loss = None

        # 初始化多任务损失函数
        loss_config = LOSS_CONFIGS['balanced']  # 使用平衡的损失配置
        self.multi_task_loss = MultiTaskLoss(
            time_weight=loss_config['time_weight'],
            input_token_weight=loss_config['input_token_weight'],
            output_token_weight=loss_config['output_token_weight'],
            time_loss_type=loss_config['time_loss_type'],
            token_loss_type=loss_config['token_loss_type']
        )

    def prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备多任务预测数据

        Returns:
            features: [batch_size, seq_len, feature_dim]
            targets: [batch_size, 3] - [time_interval, input_tokens, output_tokens]
        """
        features = []
        targets = []

        # 确保数据按时序排序
        data = data.sort_values('arrival_time_ns').reset_index(drop=True)

        # 计算时间间隔
        time_intervals = data['arrival_time_ns'].diff().fillna(0).values / 1e9  # 转换为秒

        for i in range(len(data) - self.config.sequence_length):
            # 输入特征序列
            seq_data = data.iloc[i:i + self.config.sequence_length]

            # 提取特征：时间间隔、input_tokens、output_tokens
            seq_intervals = time_intervals[i:i + self.config.sequence_length]
            seq_input_tokens = seq_data['input_toks'].values
            seq_output_tokens = seq_data['output_toks'].values

            # 组合特征 [seq_len, 3]
            feature_seq = np.column_stack([
                seq_intervals,
                seq_input_tokens,
                seq_output_tokens
            ])

            # 目标：下一个请求的三个属性
            next_interval = time_intervals[i + self.config.sequence_length] if i + self.config.sequence_length < len(time_intervals) else 0
            next_input_tokens = data.iloc[i + self.config.sequence_length]['input_toks'] if i + self.config.sequence_length < len(data) else 0
            next_output_tokens = data.iloc[i + self.config.sequence_length]['output_toks'] if i + self.config.sequence_length < len(data) else 0

            target = np.array([next_interval, next_input_tokens, next_output_tokens])

            features.append(feature_seq)
            targets.append(target)

        if not features:
            # 如果没有足够的序列数据，返回空张量
            return torch.empty((0, self.config.sequence_length, 3)), torch.empty((0, 3))

        return torch.FloatTensor(np.array(features)), torch.FloatTensor(np.array(targets))

    def build_model(self):
        """构建轻量级模型"""
        class LightweightModel(nn.Module):
            def __init__(self, input_size=3, hidden_size=32, num_layers=2, output_size=3):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers

                # LSTM层
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                   batch_first=True, dropout=0.2)

                # 输出层 - 直接预测三个目标
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size // 2, output_size)
                )

            def forward(self, x):
                # LSTM处理
                lstm_out, (h_n, c_n) = self.lstm(x)

                # 取最后一个时间步的输出
                last_output = lstm_out[:, -1, :]

                # 预测三个目标
                output = self.fc(last_output)

                return output

        self.model = LightweightModel(
            input_size=3,
            hidden_size=self.config.hidden_size if hasattr(self.config, 'hidden_size') else 32,
            num_layers=self.config.num_layers if hasattr(self.config, 'num_layers') else 2,
            output_size=3
        ).to(self.config.device.value)

        logger.info(f"Lightweight model built with {sum(p.numel() for p in self.model.parameters())} parameters")

    def train(self, data: pd.DataFrame) -> TrainingResult:
        """使用多任务损失函数训练模型"""
        try:
            # 准备数据
            features, targets = self.prepare_data(data)

            if features.shape[0] == 0:
                logger.warning("Insufficient data for training")
                return TrainingResult(
                    success=False,
                    message="Insufficient data for training",
                    epochs_trained=0,
                    final_loss=float('inf')
                )

            # 构建模型
            self.build_model()

            # 优化器
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            # 训练参数
            epochs = self.config.epochs if hasattr(self.config, 'epochs') else 100
            batch_size = self.config.batch_size

            # 训练历史
            train_losses = []
            val_losses = []

            # 分割训练/验证集
            val_size = min(100, int(len(features) * 0.2))
            if len(features) > val_size:
                train_features = features[:-val_size]
                train_targets = targets[:-val_size]
                val_features = features[-val_size:]
                val_targets = targets[-val_size:]
            else:
                train_features = features
                train_targets = targets
                val_features = features
                val_targets = targets

            logger.info(f"Starting training with {len(train_features)} samples, validating on {len(val_features)} samples")

            # 训练循环
            for epoch in range(epochs):
                # 训练阶段
                self.model.train()
                epoch_loss = 0.0

                # 批量训练
                for i in range(0, len(train_features), batch_size):
                    batch_features = train_features[i:i + batch_size].to(self.config.device.value)
                    batch_targets = train_targets[i:i + batch_size].to(self.config.device.value)

                    optimizer.zero_grad()

                    # 前向传播
                    predictions = self.model(batch_features)

                    # 计算多任务损失
                    loss_dict = self.multi_task_loss(predictions, batch_targets)
                    loss = loss_dict['total_loss']

                    # 反向传播
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_train_loss = epoch_loss / (len(train_features) // batch_size + 1)
                train_losses.append(avg_train_loss)

                # 验证阶段
                if epoch % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_predictions = self.model(val_features.to(self.config.device.value))
                        val_loss_dict = self.multi_task_loss(val_predictions, val_targets.to(self.config.device.value))
                        val_loss = val_loss_dict['total_loss'].item()
                        val_losses.append(val_loss)

                        logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

                        # 详细损失信息
                        loss_details = self.multi_task_loss.get_loss_components(val_predictions, val_targets)
                        logger.info(f"  - Time Loss: {loss_details['time_loss']:.4f}, "
                                   f"Input Token Loss: {loss_details['input_token_loss']:.4f}, "
                                   f"Output Token Loss: {loss_details['output_token_loss']:.4f}")

            self.is_trained = True
            logger.info("Lightweight model training completed")

            return TrainingResult(
                success=True,
                message="Training completed successfully",
                epochs_trained=epochs,
                final_loss=train_losses[-1] if train_losses else float('inf'),
                metadata={
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'model_type': 'lightweight',
                    'parameters': sum(p.numel() for p in self.model.parameters())
                }
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                message=f"Training failed: {str(e)}",
                epochs_trained=0,
                final_loss=float('inf')
            )

    def predict_future(self, data: pd.DataFrame, steps: int = 1) -> PredictionResult:
        """预测未来值"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained yet")
            return PredictionResult(
                timestamps=[],
                request_tokens=[],
                response_tokens=[],
                confidence=0.0,
                model_name=self.config.model_name,
                prediction_time=0.0,
                metadata={'error': 'Model not trained'}
            )

        try:
            self.model.eval()

            with torch.no_grad():
                # 准备最后一个序列
                features, _ = self.prepare_data(data)

                if features.shape[0] == 0:
                    logger.warning("Insufficient data for prediction")
                    return PredictionResult(
                        timestamps=[],
                        request_tokens=[],
                        response_tokens=[],
                        confidence=0.0,
                        model_name=self.config.model_name,
                        prediction_time=0.0,
                        metadata={'error': 'Insufficient data'}
                    )

                # 使用最后一个序列进行预测
                last_seq = features[-1:].to(self.config.device.value)
                prediction = self.model(last_seq).cpu().numpy()[0]

                # 解析预测结果
                predicted_interval = max(0.001, prediction[0])  # 确保时间间隔为正
                predicted_input_tokens = max(1, int(prediction[1]))  # 确保token数量为正整数
                predicted_output_tokens = max(1, int(prediction[2]))

                # 生成预测时间戳
                last_timestamp = data['arrival_time_ns'].iloc[-1]
                predicted_timestamp = last_timestamp + int(predicted_interval * 1e9)

                logger.info(f"Prediction: Interval={predicted_interval:.3f}s, "
                           f"InputTokens={predicted_input_tokens}, OutputTokens={predicted_output_tokens}")

                return PredictionResult(
                    timestamps=[predicted_timestamp],
                    request_tokens=[predicted_input_tokens],
                    response_tokens=[predicted_output_tokens],
                    confidence=0.8,  # 轻量级模型的置信度
                    model_name=self.config.model_name,
                    prediction_time=0.001,  # 推理很快
                    metadata={
                        'predicted_interval': predicted_interval,
                        'model_type': 'lightweight',
                        'prediction_steps': steps
                    }
                )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return PredictionResult(
                timestamps=[],
                request_tokens=[],
                response_tokens=[],
                confidence=0.0,
                model_name=self.config.model_name,
                prediction_time=0.0,
                metadata={'error': str(e)}
            )

    def save_model(self, path: str) -> bool:
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'training_history': getattr(self, 'training_history', {})
            }, path)
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """加载模型"""
        try:
            checkpoint = torch.load(path, map_location=self.config.device.value)
            self.build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


# 示例用法
def create_lightweight_model(model_name: str = "lightweight_predictor") -> LightweightPredictor:
    """创建轻量级预测模型的便捷函数"""
    config = ModelConfig(
        model_type=ModelType.CUSTOM,
        model_name=model_name,
        sequence_length=50,
        prediction_horizon=1,
        batch_size=32,
        epochs=100
    )

    return LightweightPredictor(config)


if __name__ == "__main__":
    # 基础测试
    logger.info("Testing lightweight predictor...")

    # 创建示例数据
    test_data = pd.DataFrame({
        'arrival_time_ns': np.arange(100) * 1_000_000_000,  # 1秒间隔
        'input_toks': np.random.randint(100, 1000, 100),
        'output_toks': np.random.randint(50, 500, 100),
        'model_type': ['test'] * 100
    })

    # 创建和训练模型
    model = create_lightweight_model()
    result = model.train(test_data)

    if result.success:
        logger.info(f"Training successful! Final loss: {result.final_loss:.4f}")

        # 测试预测
        prediction = model.predict_future(test_data)
        logger.info(f"Prediction: {prediction.request_tokens}, {prediction.response_tokens}")
    else:
        logger.error(f"Training failed: {result.message}")