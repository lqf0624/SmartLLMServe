#!/usr/bin/env python3
"""
DLinear时间序列预测模型 - 简化版本
使用标准的Dataset+DataLoader实现真正的随机采样
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import logging
from typing import Dict, Any, Optional, Tuple, List
import time
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """简化的时间序列数据集"""

    def __init__(self, input_tokens, output_tokens, concurrent_requests,
                 target_input_tokens, target_output_tokens, target_concurrent_requests,
                 seq_len):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.concurrent_requests = concurrent_requests
        self.target_input_tokens = target_input_tokens
        self.target_output_tokens = target_output_tokens
        self.target_concurrent_requests = target_concurrent_requests
        self.seq_len = seq_len

        # 计算可用的起始索引范围
        self.max_start_idx = len(self.target_input_tokens) - self.seq_len
        if self.max_start_idx <= 0:
            self.valid_indices = []
        else:
            self.valid_indices = list(range(self.max_start_idx))

        logger.info(f"时间序列数据集: 可用序列数={len(self.valid_indices)}, 序列长度={self.seq_len}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if idx >= len(self.valid_indices):
            idx = idx % len(self.valid_indices)

        start_idx = self.valid_indices[idx]

        # 输入序列 - 保持时间窗口内的顺序
        seq_features = []
        for j in range(start_idx, start_idx + self.seq_len):
            if j < len(self.concurrent_requests):
                seq_features.append([
                    self.concurrent_requests[j],
                    self.input_tokens[j],
                    self.output_tokens[j]
                ])
            else:
                # 边界情况，使用最后一个有效值
                seq_features.append([
                    self.concurrent_requests[-1],
                    self.input_tokens[-1],
                    self.output_tokens[-1]
                ])

        # 目标序列 - 预测下一个时间点的值
        target_idx = start_idx + self.seq_len
        if target_idx < len(self.target_concurrent_requests):
            target_features = [
                self.target_concurrent_requests[target_idx],
                self.target_input_tokens[target_idx],
                self.target_output_tokens[target_idx]
            ]
        else:
            # 边界情况，使用最后一个目标值
            target_features = [
                self.target_concurrent_requests[-1],
                self.target_input_tokens[-1],
                self.target_output_tokens[-1]
            ]

        return torch.FloatTensor(seq_features), torch.FloatTensor(target_features)


class DLinearPredictor:
    """DLinear预测器 - 简化版本"""

    def __init__(self, seq_len=120, pred_len=3, channels=3, individual=True,
                 learning_rate=0.001, device='auto', kernel_size=25):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size

        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 创建模型
        self.model = self._create_model().to(self.device)

        # 使用AdamW优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 标准化器
        self.scalers = None

        logger.info(f"DLinear预测器初始化完成: seq_len={seq_len}, pred_len={pred_len}, "
                   f"channels={channels}, individual={individual}, device={self.device}")

    def _create_model(self):
        """创建DLinear模型"""
        class DLinear(nn.Module):
            def __init__(self, seq_len, pred_len, channels, individual):
                super().__init__()
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.channels = channels
                self.individual = individual

                if individual:
                    # 通道独立的线性层 - 预测单个时间步
                    self.Linear = nn.ModuleList()
                    for i in range(channels):
                        self.Linear.append(nn.Linear(seq_len, 1))  # 预测1个时间步
                else:
                    # 通道共享的线性层
                    self.Linear = nn.Linear(seq_len, 1)  # 预测1个时间步

            def forward(self, x):
                # x: [batch_size, seq_len, channels]
                if self.individual:
                    out = []
                    for i in range(self.channels):
                        # 每个通道独立预测下一个时间步
                        pred = self.Linear[i](x[:, :, i])  # [batch_size, 1]
                        out.append(pred)
                    out = torch.cat(out, dim=1)  # [batch_size, channels] - 下一个时间步的3个特征
                else:
                    # 共享权重，每个通道独立预测
                    out = []
                    for i in range(self.channels):
                        pred = self.Linear(x[:, :, i])  # [batch_size, 1]
                        out.append(pred)
                    out = torch.cat(out, dim=1)  # [batch_size, channels]

                return out  # [batch_size, channels]

        return DLinear(self.seq_len, self.pred_len, self.channels, self.individual)

    def prepare_data(self, data: pd.DataFrame):
        """准备数据"""
        logger.info("准备数据...")

        # 支持多种列名格式
        if 'Concurrent_requests' in data.columns:
            # 分钟级聚合数据的列名
            concurrent_requests = data['Concurrent_requests'].values.astype(float)
            input_tokens = data['Request_tokens_sum'].values.astype(float)
            output_tokens = data['Response_tokens_sum'].values.astype(float)
            logger.info(f"数据准备 - 分钟级聚合数据, concurrent_requests: [{concurrent_requests.min():.3f}, {concurrent_requests.max():.3f}], "
                       f"input_tokens: [{input_tokens.min():.3f}, {input_tokens.max():.3f}], "
                       f"output_tokens: [{output_tokens.min():.3f}, {output_tokens.max():.3f}]")
        elif 'concurrent_requests' in data.columns:
            # 原始数据的列名
            concurrent_requests = data['concurrent_requests'].values.astype(float)
            input_tokens = data['input_toks'].values.astype(float)
            output_tokens = data['output_toks'].values.astype(float)
            logger.info(f"数据准备 - 原始数据, concurrent_requests: [{concurrent_requests.min():.3f}, {concurrent_requests.max():.3f}], "
                       f"input_tokens: [{input_tokens.min():.3f}, {input_tokens.max():.3f}], "
                       f"output_tokens: [{output_tokens.min():.3f}, {output_tokens.max():.3f}]")
        else:
            # 没有并发量数据，使用默认值
            concurrent_requests = np.ones(len(data)).astype(float)
            if 'Request_tokens_sum' in data.columns:
                input_tokens = data['Request_tokens_sum'].values.astype(float)
                output_tokens = data['Response_tokens_sum'].values.astype(float)
            else:
                input_tokens = data['input_toks'].values.astype(float)
                output_tokens = data['output_toks'].values.astype(float)
            logger.info(f"数据准备 - 使用默认并发量=1, "
                       f"input_tokens: [{input_tokens.min():.3f}, {input_tokens.max():.3f}], "
                       f"output_tokens: [{output_tokens.min():.3f}, {output_tokens.max():.3f}]")

        # 目标值：下一个时间点的值
        target_concurrent_requests = concurrent_requests[1:]  # 从第二个开始
        target_input_tokens = input_tokens[1:]  # 从第二个开始
        target_output_tokens = output_tokens[1:]  # 从第二个开始

        # 标准化
        self.scalers = []
        features_to_scale = [concurrent_requests, input_tokens, output_tokens]
        scaled_features = []

        for feature in features_to_scale:
            scaler = RobustScaler()
            scaled_feature = scaler.fit_transform(feature.reshape(-1, 1)).ravel()
            scaled_features.append(scaled_feature)
            self.scalers.append(scaler)

        concurrent_requests, input_tokens, output_tokens = scaled_features

        return (input_tokens, output_tokens, concurrent_requests,
                target_input_tokens, target_output_tokens, target_concurrent_requests)

    def create_dataloader(self, data_tuple, batch_size=32, shuffle=True):
        """创建数据加载器"""
        (input_tokens, output_tokens, concurrent_requests,
         target_input_tokens, target_output_tokens, target_concurrent_requests) = data_tuple

        dataset = TimeSeriesDataset(
            input_tokens, output_tokens, concurrent_requests,
            target_input_tokens, target_output_tokens, target_concurrent_requests,
            self.seq_len
        )

        if len(dataset) == 0:
            logger.warning("数据集为空")
            return None

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,  # 使用DataLoader的内置shuffle功能
            num_workers=0,
            pin_memory=False
        )

        return dataloader

    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)  # 在归一化空间计算loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def validate_epoch(self, dataloader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_mae = 0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)  # 在归一化空间计算loss

                # 计算MSE和MAE - 也在归一化空间
                mse = nn.MSELoss()(outputs, batch_y)
                mae = nn.L1Loss()(outputs, batch_y)

                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
                num_batches += 1

        return (total_loss / num_batches if num_batches > 0 else 0,
                total_mse / num_batches if num_batches > 0 else 0,
                total_mae / num_batches if num_batches > 0 else 0)

    def train_batch(self, data: pd.DataFrame, validation_split: float = 0.2,
                   epochs: int = 100, batch_size: int = 32, patience: int = 10,
                   verbose: bool = True) -> Dict[str, Any]:
        """批量训练模型"""
        logger.info(f"开始批量训练: {epochs} epochs, batch_size={batch_size}, validation_split={validation_split}")

        # 准备数据
        data_tuple = self.prepare_data(data)

        # 分割训练和验证集
        total_samples = len(data_tuple[0]) - 1  # 减去1因为目标值从第二个开始
        val_size = int(total_samples * validation_split)
        train_size = total_samples - val_size

        # 创建训练集
        train_tuple = tuple(arr[:train_size] for arr in data_tuple)
        train_loader = self.create_dataloader(train_tuple, batch_size, shuffle=True)

        # 创建验证集
        if val_size > 0:
            val_tuple = tuple(arr[train_size-1:train_size-1+val_size] for arr in data_tuple)
            val_loader = self.create_dataloader(val_tuple, batch_size, shuffle=False)
        else:
            val_loader = None

        if train_loader is None:
            logger.error("训练数据加载失败")
            return {}

        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'learning_rates': []
        }

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练
            train_loss = self.train_epoch(train_loader)

            # 验证
            if val_loader is not None:
                val_loss, val_mse, val_mae = self.validate_epoch(val_loader)
                self.scheduler.step(val_loss)

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                history['val_loss'].append(val_loss)
                history['val_mse'].append(val_mse)
                history['val_mae'].append(val_mae)
            else:
                val_loss = train_loss
                history['val_loss'].append(train_loss)
                history['val_mse'].append(train_loss)
                history['val_mae'].append(train_loss)

            history['train_loss'].append(train_loss)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            epoch_time = time.time() - epoch_start

            if verbose and (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    logger.info(f"Epoch {epoch+1}/{epochs}: "
                               f"Train Loss: {train_loss:.4f}, "
                               f"Val Loss: {val_loss:.4f}, "
                               f"Val MSE: {val_mse:.4f}, "
                               f"Val MAE: {val_mae:.4f}, "
                               f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                               f"Time: {epoch_time:.1f}s")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}: "
                               f"Train Loss: {train_loss:.4f}, "
                               f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                               f"Time: {epoch_time:.1f}s")

            # 早停
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # 恢复最佳模型
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)

        total_time = time.time() - start_time
        logger.info(f"训练完成，总时间: {total_time:.1f}s")

        return history

    def predict(self, data: pd.DataFrame, steps: Optional[int] = None, return_details: bool = False) -> torch.Tensor:
        """进行预测"""
        if steps is None:
            steps = 1  # 预测下一个时间步

        if len(data) < self.seq_len:
            logger.warning(f"数据不足: {len(data)} < {self.seq_len}，将使用零填充")
            # 使用零填充
            padding_size = self.seq_len - len(data)
            sequences = np.zeros((self.seq_len, 3))
            if len(data) > 0:
                # 统一使用三项特征
                for i in range(len(data)):
                    if 'Concurrent_requests' in data.columns:
                        # 分钟级聚合数据
                        concurrent_req = data.iloc[i]['Concurrent_requests']
                        input_tok = data.iloc[i]['Request_tokens_sum']
                        output_tok = data.iloc[i]['Response_tokens_sum']
                    elif 'concurrent_requests' in data.columns:
                        # 原始数据
                        concurrent_req = data.iloc[i]['concurrent_requests']
                        input_tok = data.iloc[i]['input_toks']
                        output_tok = data.iloc[i]['output_toks']
                    else:
                        # 默认值
                        concurrent_req = 1.0
                        if 'Request_tokens_sum' in data.columns:
                            input_tok = data.iloc[i]['Request_tokens_sum']
                            output_tok = data.iloc[i]['Response_tokens_sum']
                        else:
                            input_tok = data.iloc[i]['input_toks']
                            output_tok = data.iloc[i]['output_toks']

                    sequences[padding_size + i] = [
                        concurrent_req,
                        input_tok,
                        output_tok
                    ]
        else:
            # 使用最新的seq_len个数据点
            recent_data = data.iloc[-self.seq_len:].copy()
            sequences = []

            for i in range(len(recent_data)):
                if 'Concurrent_requests' in recent_data.columns:
                    # 分钟级聚合数据
                    concurrent_req = recent_data.iloc[i]['Concurrent_requests']
                    input_tok = recent_data.iloc[i]['Request_tokens_sum']
                    output_tok = recent_data.iloc[i]['Response_tokens_sum']
                elif 'concurrent_requests' in recent_data.columns:
                    # 原始数据
                    concurrent_req = recent_data.iloc[i]['concurrent_requests']
                    input_tok = recent_data.iloc[i]['input_toks']
                    output_tok = recent_data.iloc[i]['output_toks']
                else:
                    # 默认值
                    concurrent_req = 1.0
                    if 'Request_tokens_sum' in recent_data.columns:
                        input_tok = recent_data.iloc[i]['Request_tokens_sum']
                        output_tok = recent_data.iloc[i]['Response_tokens_sum']
                    else:
                        input_tok = recent_data.iloc[i]['input_toks']
                        output_tok = recent_data.iloc[i]['output_toks']

                sequences.append([
                    concurrent_req,
                    input_tok,
                    output_tok
                ])

            sequences = np.array(sequences)

        # 标准化
        if hasattr(self, 'scalers') and self.scalers is not None and len(self.scalers) >= 3:
            for feature_idx in range(3):
                sequences[:, feature_idx] = self.scalers[feature_idx].transform(
                    sequences[:, feature_idx].reshape(-1, 1)
                ).ravel()

        # 转换为张量
        input_tensor = torch.FloatTensor(sequences).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            prediction = self.model(input_tensor)  # [1, 3]

        # 反标准化
        prediction = prediction.squeeze(0).cpu().numpy()  # [3]

        if hasattr(self, 'scalers') and self.scalers is not None and len(self.scalers) >= 3:
            for feature_idx in range(3):
                if feature_idx < len(self.scalers):
                    prediction[feature_idx] = self.scalers[feature_idx].inverse_transform(
                        prediction[feature_idx].reshape(-1, 1)
                    ).ravel()[0]

        # 确保预测结果合理（非负值）
        prediction = np.maximum(prediction, 0)

        logger.info(f"预测结果: 并发请求={prediction[0]:.1f}, "
                   f"输入token={prediction[1]:.1f}, 输出token={prediction[2]:.1f}")

        # 返回反归一化后的预测结果
        return torch.FloatTensor(prediction)

    def create_prediction_comparison(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                                   num_samples: int = 100) -> Dict[str, Any]:
        """创建预测对比图和统计信息"""
        logger.info(f"创建预测对比，样本数: {num_samples}")

        # 选择样本进行对比
        if len(val_data) > num_samples:
            # 随机选择样本
            sample_indices = np.random.choice(len(val_data) - self.seq_len, num_samples, replace=False)
        else:
            sample_indices = list(range(len(val_data) - self.seq_len))

        predictions = []
        actuals = []
        errors = []

        for idx in sample_indices:
            # 获取输入序列
            input_data = val_data.iloc[idx:idx + self.seq_len]

            # 预测
            pred = self.predict(input_data)
            predictions.append(pred.numpy())

            # 获取真实值
            actual_idx = idx + self.seq_len
            if actual_idx < len(val_data):
                if 'Concurrent_requests' in val_data.columns:
                    # 分钟级聚合数据
                    actual = [
                        val_data.iloc[actual_idx]['Concurrent_requests'],
                        val_data.iloc[actual_idx]['Request_tokens_sum'],
                        val_data.iloc[actual_idx]['Response_tokens_sum']
                    ]
                elif 'concurrent_requests' in val_data.columns:
                    # 原始数据
                    actual = [
                        val_data.iloc[actual_idx]['concurrent_requests'],
                        val_data.iloc[actual_idx]['input_toks'],
                        val_data.iloc[actual_idx]['output_toks']
                    ]
                else:
                    # 默认值
                    if 'Request_tokens_sum' in val_data.columns:
                        actual = [
                            1.0,
                            val_data.iloc[actual_idx]['Request_tokens_sum'],
                            val_data.iloc[actual_idx]['Response_tokens_sum']
                        ]
                    else:
                        actual = [
                            1.0,
                            val_data.iloc[actual_idx]['input_toks'],
                            val_data.iloc[actual_idx]['output_toks']
                        ]
                actuals.append(actual)

                # 计算误差
                error = [abs(pred[i] - actual[i]) for i in range(3)]
                errors.append(error)

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        errors = np.array(errors)

        # 计算统计信息
        stats = {
            'predictions': predictions,
            'actuals': actuals,
            'errors': errors,
            'mse_per_feature': [np.mean(errors[:, i]**2) for i in range(3)],
            'mae_per_feature': [np.mean(errors[:, i]) for i in range(3)],
            'feature_names': ['concurrent_requests', 'input_tokens', 'output_tokens']
        }

        logger.info(f"预测对比统计:")
        for i, name in enumerate(stats['feature_names']):
            logger.info(f"  {name}: MSE={stats['mse_per_feature'][i]:.4f}, MAE={stats['mae_per_feature'][i]:.4f}")

        return stats


def main():
    """测试函数"""
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000

    test_data = pd.DataFrame({
        'input_toks': np.random.poisson(100, n_samples),
        'output_toks': np.random.poisson(200, n_samples),
        'concurrent_requests': np.random.poisson(5, n_samples)
    })

    # 创建模型
    model = DLinearPredictor(
        seq_len=50,
        pred_len=1,
        channels=3,
        individual=True,
        learning_rate=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 训练模型
    history = model.train_batch(
        data=test_data,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        patience=10,
        verbose=True
    )

    print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"最终验证损失: {history['val_loss'][-1]:.4f}")

    # 测试预测
    test_input = test_data.iloc[:50]
    prediction = model.predict(test_input)
    print(f"测试预测: {prediction}")

    # 测试预测对比功能
    print("\n=== 测试预测对比功能 ===")
    comparison_stats = model.create_prediction_comparison(test_data, test_data.iloc[-200:], num_samples=50)
    print(f"预测对比完成，MSE: {comparison_stats['mse_per_feature']}")
    print(f"预测对比完成，MAE: {comparison_stats['mae_per_feature']}")


if __name__ == "__main__":
    main()