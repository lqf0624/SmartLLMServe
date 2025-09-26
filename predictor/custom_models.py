"""
自定义预测模型示例

展示如何在GPU环境中集成新的预测模型。
这些模型可以很容易地在离线GPU服务器上添加和训练。
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from .model_interface import (
    BasePredictor, ModelConfig, PredictionResult, TrainingResult,
    ModelType, ModelDevice, register_model
)

logger = logging.getLogger(__name__)


@register_model(ModelType.TRANSFORMER)
class TransformerPredictor(BasePredictor):
    """基于Transformer的预测模型"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.scaler = None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备序列数据"""
        # 提取时间序列特征
        features = []
        targets = []

        for i in range(len(data) - self.config.sequence_length - self.config.prediction_horizon + 1):
            # 输入序列
            seq = data.iloc[i:i + self.config.sequence_length]
            input_features = torch.tensor([
                seq['Timestamp'].values,
                seq['Request tokens'].values,
                seq['Response tokens'].values
            ], dtype=torch.float32).T  # [seq_len, 3]

            # 目标序列
            target_seq = data.iloc[i + self.config.sequence_length:
                              i + self.config.sequence_length + self.config.prediction_horizon]
            target = torch.tensor([
                target_seq['Request tokens'].values,
                target_seq['Response tokens'].values
            ], dtype=torch.float32).T  # [horizon, 2]

            features.append(input_features)
            targets.append(target)

        return torch.stack(features), torch.stack(targets)

    def build_model(self):
        """构建Transformer模型"""
        class TransformerModel(nn.Module):
            def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=2,
                         output_dim=2, seq_len=50, horizon=10):
                super().__init__()

                self.input_projection = nn.Linear(input_dim, d_model)
                self.pos_encoder = nn.Embedding(seq_len, d_model)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                self.output_projection = nn.Linear(d_model, output_dim * horizon)
                self.horizon = horizon

            def forward(self, x):
                # x shape: [batch_size, seq_len, input_dim]
                batch_size, seq_len, _ = x.shape

                # 输入投影
                x = self.input_projection(x)  # [batch_size, seq_len, d_model]

                # 位置编码
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
                pos_encoding = self.pos_encoder(positions)  # [1, seq_len, d_model]
                x = x + pos_encoding

                # Transformer编码
                x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
                encoded = self.transformer(x)  # [seq_len, batch_size, d_model]
                encoded = encoded.transpose(0, 1)  # [batch_size, seq_len, d_model]

                # 输出投影
                output = self.output_projection(encoded.mean(dim=1))  # [batch_size, horizon * 2]
                output = output.view(batch_size, self.horizon, 2)  # [batch_size, horizon, 2]

                return output

        self.model = TransformerModel(
            input_dim=3,
            d_model=self.config.custom_params.get('d_model', 64),
            nhead=self.config.custom_params.get('nhead', 4),
            num_layers=self.config.custom_params.get('num_layers', 2),
            output_dim=2,
            seq_len=self.config.sequence_length,
            horizon=self.config.prediction_horizon
        ).to(self.device)

        return self.model

    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> TrainingResult:
        """训练Transformer模型"""
        import time
        from sklearn.model_selection import train_test_split

        start_time = time.time()

        # 准备数据
        X, y = self.prepare_data(data)

        # 分割训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # 构建模型
        if self.model is None:
            self.build_model()

        # 训练设置
        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        # 训练循环
        for epoch in range(self.config.epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for i in range(0, len(X_train), self.config.batch_size):
                batch_X = X_train[i:i + self.config.batch_size].to(self.device)
                batch_y = y_train[i:i + self.config.batch_size].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(X_train) // self.config.batch_size
            train_losses.append(train_loss)

            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(X_val), self.config.batch_size):
                    batch_X = X_val[i:i + self.config.batch_size].to(self.device)
                    batch_y = y_val[i:i + self.config.batch_size].to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(X_val) // self.config.batch_size
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            # 早停检查
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if self.config.early_stopping and patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        training_time = time.time() - start_time
        self.is_trained = True

        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': len(train_losses) - patience_counter,
            'best_loss': best_loss
        }

        return TrainingResult(
            success=True,
            training_time=training_time,
            final_loss=train_losses[-1],
            best_loss=best_loss,
            epochs_trained=len(train_losses),
            early_stopped=patience_counter >= self.config.patience,
            metadata={'train_losses': train_losses, 'val_losses': val_losses}
        )

    def predict(self, data: pd.DataFrame, steps: int = None) -> PredictionResult:
        """预测未来时间步"""
        if steps is None:
            steps = self.config.prediction_horizon

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()
        with torch.no_grad():
            # 准备最后一个序列
            last_seq = data.iloc[-self.config.sequence_length:]
            X = torch.tensor([
                last_seq['Timestamp'].values,
                last_seq['Request tokens'].values,
                last_seq['Response tokens'].values
            ], dtype=torch.float32).T.unsqueeze(0).to(self.device)  # [1, seq_len, 3]

            # 预测
            prediction = self.model(X)  # [1, horizon, 2]
            prediction = prediction.squeeze(0).cpu().numpy()  # [horizon, 2]

            # 计算时间戳
            last_timestamp = data['Timestamp'].iloc[-1]
            avg_interval = data['Timestamp'].diff().mean()
            timestamps = [last_timestamp + (i + 1) * avg_interval for i in range(steps)]

            return PredictionResult(
                timestamps=timestamps[:steps],
                request_tokens=prediction[:steps, 0].astype(int).tolist(),
                response_tokens=prediction[:steps, 1].astype(int).tolist(),
                confidence=0.85,  # Transformer通常有较高置信度
                model_name=self.model_name,
                prediction_time=0.01,  # 推理很快
                metadata={'prediction_steps': steps}
            )

    def save_model(self, path: str) -> bool:
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'training_history': self.training_history
            }, path)
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """加载模型"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.training_history = checkpoint.get('training_history', {})
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


@register_model(ModelType.GRU)
class GRUPredictor(BasePredictor):
    """基于GRU的预测模型 - 更轻量的选择"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备序列数据"""
        sequences = []
        targets = []

        for i in range(len(data) - self.config.sequence_length - self.config.prediction_horizon + 1):
            # 输入序列
            seq = data.iloc[i:i + self.config.sequence_length][['Request tokens', 'Response tokens']].values
            sequences.append(seq)

            # 目标序列
            target = data.iloc[i + self.config.sequence_length:
                          i + self.config.sequence_length + self.config.prediction_horizon][['Request tokens', 'Response tokens']].values
            targets.append(target[0])  # 只预测下一个时间步

        return torch.FloatTensor(sequences), torch.FloatTensor(targets)

    def build_model(self):
        """构建GRU模型"""
        class GRUModel(nn.Module):
            def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.gru(x)
                out = self.fc(out[:, -1, :])  # 只用最后一个时间步
                return out

        self.model = GRUModel(
            input_size=2,
            hidden_size=self.config.custom_params.get('hidden_size', 64),
            num_layers=self.config.custom_params.get('num_layers', 2),
            output_size=2
        ).to(self.device)

        return self.model

    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> TrainingResult:
        """训练GRU模型"""
        import time
        from sklearn.model_selection import train_test_split

        start_time = time.time()

        # 准备数据
        X, y = self.prepare_data(data)

        # 分割训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # 构建模型
        if self.model is None:
            self.build_model()

        # 训练设置
        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        # 训练循环
        for epoch in range(self.config.epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for i in range(0, len(X_train), self.config.batch_size):
                batch_X = X_train[i:i + self.config.batch_size].to(self.device)
                batch_y = y_train[i:i + self.config.batch_size].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(X_train) // self.config.batch_size
            train_losses.append(train_loss)

            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(X_val), self.config.batch_size):
                    batch_X = X_val[i:i + self.config.batch_size].to(self.device)
                    batch_y = y_val[i:i + self.config.batch_size].to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(X_val) // self.config.batch_size
            val_losses.append(val_loss)

            # 早停检查
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if self.config.early_stopping and patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        training_time = time.time() - start_time
        self.is_trained = True

        return TrainingResult(
            success=True,
            training_time=training_time,
            final_loss=train_losses[-1],
            best_loss=best_loss,
            epochs_trained=len(train_losses),
            early_stopped=patience_counter >= self.config.patience,
            metadata={'train_losses': train_losses, 'val_losses': val_losses}
        )

    def predict(self, data: pd.DataFrame, steps: int = None) -> PredictionResult:
        """预测未来时间步"""
        if steps is None:
            steps = 1  # GRU默认预测1步

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()
        with torch.no_grad():
            # 准备最后一个序列
            last_seq = data.iloc[-self.config.sequence_length:][['Request tokens', 'Response tokens']].values
            X = torch.FloatTensor(last_seq).unsqueeze(0).to(self.device)  # [1, seq_len, 2]

            # 预测
            prediction = self.model(X)  # [1, 2]
            prediction = prediction.squeeze(0).cpu().numpy()  # [2]

            # 计算时间戳
            last_timestamp = data['Timestamp'].iloc[-1]
            avg_interval = data['Timestamp'].diff().mean()
            timestamps = [last_timestamp + avg_interval]

            return PredictionResult(
                timestamps=timestamps[:steps],
                request_tokens=[int(max(1, prediction[0]))],  # 确保正数
                response_tokens=[int(max(1, prediction[1]))],
                confidence=0.75,  # GRU置信度适中
                model_name=self.model_name,
                prediction_time=0.005,  # GRU推理很快
                metadata={'prediction_steps': steps}
            )

    def save_model(self, path: str) -> bool:
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__
            }, path)
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """加载模型"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


# 使用示例和说明
def create_gpu_model_examples():
    """
    创建GPU模型使用示例

    这个函数展示了如何在GPU环境中集成新模型：
    1. 创建模型配置
    2. 初始化模型管理器
    3. 训练和比较多个模型
    """

    # 示例：在GPU服务器上运行的代码
    example_code = '''
    # 1. 导入必要的模块
    from predictor.model_interface import ModelConfig, ModelDevice, ModelType, ModelManager
    from predictor.custom_models import TransformerPredictor, GRUPredictor
    import pandas as pd

    # 2. 配置GPU设备
    device = ModelDevice.CUDA if torch.cuda.is_available() else ModelDevice.CPU

    # 3. 创建模型配置
    transformer_config = ModelConfig(
        model_type=ModelType.TRANSFORMER,
        model_name="transformer_v1",
        device=device,
        sequence_length=50,
        prediction_horizon=10,
        epochs=50,
        custom_params={
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4
        }
    )

    gru_config = ModelConfig(
        model_type=ModelType.GRU,
        model_name="gru_v1",
        device=device,
        sequence_length=30,
        prediction_horizon=5,
        epochs=30,
        custom_params={
            'hidden_size': 128,
            'num_layers': 3
        }
    )

    # 4. 初始化模型管理器
    model_manager = ModelManager()

    # 5. 添加模型
    transformer_model = model_manager.add_model(transformer_config)
    gru_model = model_manager.add_model(gru_config)

    # 6. 加载数据
    data = pd.read_csv('dataset/BurstGPT_1_cleaned.csv', nrows=10000)

    # 7. 训练所有模型
    training_results = model_manager.train_all_models(data, validation_split=0.2)

    # 8. 进行预测
    historical_data = data.iloc[:8000]
    predictions = model_manager.predict_all(historical_data, steps=5)

    # 9. 比较模型性能
    comparison = model_manager.compare_models(historical_data, steps=5)

    # 10. 保存最佳模型
    best_model_name = comparison['performance_summary']['best_model']
    best_model = model_manager.active_models[best_model_name]
    best_model.save_model(f'models/{best_model_name}_best.pth')

    print(f"Best model: {best_model_name}")
    print(f"Training results: {training_results}")
    print(f"Predictions: {predictions}")
    '''

    return example_code