#!/usr/bin/env python3
"""
预测器训练脚本

使用BurstGPT数据集训练多任务预测器，支持wandb日志记录和GPU加速。
"""

import os
import sys
import argparse
import logging
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor.dlinear_model import DLinearPredictor
import wandb

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictorTrainer:
    """预测器训练器"""

    def __init__(self, config: Dict):
        """
        初始化训练器

        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.output_dir = Path(config.get('output_dir', './training_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化wandb
        self.wandb_run = None
        if config.get('use_wandb', False):
            self._init_wandb()

    def _init_wandb(self):
        """初始化wandb"""
        wandb_config = {
            'model_type': self.config.get('model_type', 'DLinear'),
            'sequence_length': self.config.get('sequence_length', 50),
            'prediction_horizon': self.config.get('prediction_horizon', 10),
            'batch_size': self.config.get('batch_size', 64),
            'learning_rate': self.config.get('learning_rate', 0.005),
            'epochs': self.config.get('epochs', 100),
            'channels': self.config.get('channels', 3),
            'individual': self.config.get('individual', True),
            'kernel_size': self.config.get('kernel_size', 25),
            'loss_function': 'MSE'
        }

        # 设置wandb目录为项目根目录
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 直接使用项目根目录作为wandb目录，避免嵌套
        wandb_dir = project_root

        self.wandb_run = wandb.init(
            project=self.config.get('wandb_project', 'smartllm-serve'),
            entity=self.config.get('wandb_entity', None),
            name=self.config.get('wandb_run_name', f'predictor_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
            config=wandb_config,
            tags=['predictor', 'burstgpt', 'channel-independent'],
            reinit=True,
            dir=wandb_dir  # 指定wandb目录为项目根目录
        )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载训练和验证数据，支持缓存"""
        logger.info("加载数据集")

        train_path = self.config['train_data_path']
        val_path = self.config['val_data_path']

        # 检查是否有缓存文件
        cache_dir = self.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        train_cache_path = cache_dir / f"train_{hash(train_path)}.pkl"
        val_cache_path = cache_dir / f"val_{hash(val_path)}.pkl"

        # 如果缓存存在且是最新的，直接加载
        if (train_cache_path.exists() and val_cache_path.exists() and
            train_cache_path.stat().st_mtime > os.path.getmtime(train_path) and
            val_cache_path.stat().st_mtime > os.path.getmtime(val_path)):
            logger.info("从缓存加载数据...")
            train_data = pd.read_pickle(train_cache_path)
            val_data = pd.read_pickle(val_cache_path)
        else:
            logger.info("加载原始数据集...")
            train_data = pd.read_csv(train_path)
            val_data = pd.read_csv(val_path)

            # 保存到缓存
            logger.info("保存数据到缓存...")
            train_data.to_pickle(train_cache_path)
            val_data.to_pickle(val_cache_path)

        logger.info(f"训练数据: {len(train_data)} 条")
        logger.info(f"验证数据: {len(val_data)} 条")

        return train_data, val_data

    def create_model(self) -> DLinearPredictor:
        """创建模型"""
        logger.info("创建DLinear预测器 - 通道独立版本")

        model_config = {
            'seq_len': self.config.get('sequence_length', 50),
            'pred_len': self.config.get('prediction_horizon', 10),
            'individual': self.config.get('individual', True),
            'channels': self.config.get('channels', 3),
            'kernel_size': self.config.get('kernel_size', 25),
            'learning_rate': self.config.get('learning_rate', 0.001),
            'device': str(self.device)
        }

        # 创建模型 - 使用改进的损失函数
        model = DLinearPredictor(**model_config, loss_weights=self.config.get('loss_weights'))

        logger.info(f"模型创建完成，参数数量: {sum(p.numel() for p in model.model.parameters()):,}")
        logger.info(f"模型配置: individual={model_config['individual']}, channels={model_config['channels']}")
        return model

    def train(self):
        """执行训练过程"""
        logger.info("开始训练")

        # 加载数据
        train_data, val_data = self.load_data()

        # 创建模型
        model = self.create_model()

        # 训练模型（使用新的简化训练方法）
        logger.info("开始训练...")

        history = model.train_batch(
            train_data,
            validation_split=0.2,  # 从训练数据中分割验证集
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 64),
            patience=self.config.get('early_stopping_patience', 10),
            verbose=True
        )

        # 记录指标到wandb
        if self.wandb_run:
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            best_val_loss = min(history['val_loss'])

            # 记录每个epoch的指标
            for epoch, (train_loss, val_loss, val_mse, val_mae, lr) in enumerate(
                zip(history['train_loss'], history['val_loss'],
                    history['val_mse'], history['val_mae'],
                    history['learning_rates'])
            ):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mse': val_mse,
                    'val_mae': val_mae,
                    'learning_rate': lr,
                })

            # 记录最终指标
            wandb.log({
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'best_val_loss': best_val_loss,
                'total_epochs': len(history['train_loss']),
                'final_val_mse': history['val_mse'][-1],
                'final_val_mae': history['val_mae'][-1],
                'final_learning_rate': history['learning_rates'][-1],
            })

            # 保存模型到wandb
            model_path = str(self.output_dir / 'best_model.pt')
            if os.path.exists(model_path):
                wandb.save(model_path)

        # 保存模型引用以便在其他方法中使用
        self.model = model

        logger.info("训练完成")
        return model, history

    def evaluate(self, model: DLinearPredictor, val_data: pd.DataFrame, history: Dict = None) -> Dict:
        """评估模型性能"""
        logger.info("开始评估")

        # 保存历史记录用于图表生成
        if history:
            self.history = history

        # 创建完整的预测对比
        prediction_comparison = self._create_prediction_comparison(model, val_data)

        # 进行单步预测
        predictions = model.predict(val_data, steps=self.config.get('prediction_horizon', 10))

        # 计算各种指标
        metrics = self._calculate_metrics(predictions, val_data)

        # 合并预测对比统计到指标中
        metrics.update(prediction_comparison)

        # 记录到wandb
        if self.wandb_run:
            wandb.log({
                'eval_metrics': metrics,
            })

        # 创建预测对比图表（独立于wandb）
        self._log_prediction_plots(predictions, val_data, prediction_comparison)

        return metrics

    def _calculate_metrics(self, predictions: torch.Tensor, val_data: pd.DataFrame) -> Dict:
        """计算评估指标"""
        # 处理预测结果
        if isinstance(predictions, torch.Tensor):
            pred_values = predictions.cpu().numpy()
        else:
            pred_values = np.array(predictions)

        # pred_values 形状应该是 [1, 3, pred_len]，我们需要计算平均预测值来与真实值比较
        if len(pred_values.shape) == 3:  # [1, 3, pred_len]
            # 计算时间维度上的平均预测值
            pred_avg = pred_values.mean(axis=2).squeeze(0)  # [3]
        else:
            # 兼容旧格式
            pred_avg = pred_values.flatten()[:3]  # [3]

        # 获取真实值（最后三个数据点作为真实值）
        if len(val_data) >= 3:
            if 'Concurrent_requests' in val_data.columns:
                # 分钟级聚合数据
                true_values = np.array([
                    val_data.iloc[-3]['Concurrent_requests'],
                    val_data.iloc[-2]['Concurrent_requests'],
                    val_data.iloc[-1]['Concurrent_requests']
                ])
            elif 'concurrent_requests' in val_data.columns:
                # 原始数据
                true_values = np.array([
                    val_data.iloc[-3]['concurrent_requests'],
                    val_data.iloc[-2]['concurrent_requests'],
                    val_data.iloc[-1]['concurrent_requests']
                ])
            else:
                true_values = np.array([1.0, 1.0, 1.0])  # 默认值

            # 计算误差指标
            mse = np.mean((pred_avg - true_values) ** 2)
            mae = np.mean(np.abs(pred_avg - true_values))
            rmse = np.sqrt(mse)

            # 计算每个特征的误差
            feature_errors = {}
            feature_names = ['Concurrent_Requests', 'Input_Tokens', 'Output_Tokens']
            for i, name in enumerate(feature_names):
                feature_mse = (pred_avg[i] - true_values[i]) ** 2
                feature_mae = np.abs(pred_avg[i] - true_values[i])
                feature_errors[name] = {'mse': feature_mse, 'mae': feature_mae}
        else:
            mse = mae = rmse = 0.0
            true_values = np.array([])
            feature_errors = {}

        metrics = {
            'prediction_shape': pred_values.shape,
            'mean_prediction': np.mean(pred_avg),
            'std_prediction': np.std(pred_avg),
            'min_prediction': np.min(pred_avg),
            'max_prediction': np.max(pred_avg),
            'validation_data_size': len(val_data),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'feature_errors': feature_errors,
            'true_values': true_values.tolist() if len(true_values) > 0 else [],
            'predicted_values': pred_avg.tolist()
        }

        return metrics

    def _create_prediction_comparison(self, model: DLinearPredictor, val_data: pd.DataFrame,
                                   num_samples: int = 100) -> Dict[str, Any]:
        """创建预测对比数据，用于生成预测值vs真实值曲线"""

        # 确保有足够的数据
        if len(val_data) <= model.seq_len:
            logger.warning(f"验证数据不足: {len(val_data)} <= {model.seq_len}")
            return {}

        # 选择样本进行对比 - 使用滑动窗口方式
        max_start_idx = len(val_data) - model.seq_len - 1  # -1确保有真实值可对比

        if max_start_idx <= 0:
            logger.warning(f"数据不足以进行预测对比: max_start_idx={max_start_idx}")
            return {}

        # 均匀选择样本点
        if max_start_idx > num_samples:
            step = max_start_idx // num_samples
            sample_indices = list(range(0, max_start_idx, step))[:num_samples]
        else:
            sample_indices = list(range(max_start_idx))

        predictions = []
        actuals = []
        errors = []

        
        for idx in sample_indices:
            # 获取输入序列
            input_data = val_data.iloc[idx:idx + model.seq_len]

            # 预测未来pred_len个时间点
            pred = model.predict(input_data, steps=model.pred_len)

            # 展平为 [3 * pred_len] 用于存储
            pred_flat = pred.squeeze(0).flatten().numpy()  # [3 * pred_len]
            predictions.append(pred_flat)

            # 获取真实值（预测目标的未来pred_len个时间点）
            actual_start_idx = idx + model.seq_len
            actual_end_idx = min(actual_start_idx + model.pred_len, len(val_data))

            if actual_start_idx < len(val_data):
                if 'Concurrent_requests' in val_data.columns:
                    # 分钟级聚合数据
                    actual_concurrent = val_data.iloc[actual_start_idx:actual_end_idx]['Concurrent_requests'].values
                    actual_input = val_data.iloc[actual_start_idx:actual_end_idx]['Request_tokens_sum'].values
                    actual_output = val_data.iloc[actual_start_idx:actual_end_idx]['Response_tokens_sum'].values
                elif 'concurrent_requests' in val_data.columns:
                    # 原始数据
                    actual_concurrent = val_data.iloc[actual_start_idx:actual_end_idx]['concurrent_requests'].values
                    actual_input = val_data.iloc[actual_start_idx:actual_end_idx]['input_toks'].values
                    actual_output = val_data.iloc[actual_start_idx:actual_end_idx]['output_toks'].values
                else:
                    # 默认值
                    if 'Request_tokens_sum' in val_data.columns:
                        actual_concurrent = np.ones(model.pred_len)
                        actual_input = val_data.iloc[actual_start_idx:actual_end_idx]['Request_tokens_sum'].values
                        actual_output = val_data.iloc[actual_start_idx:actual_end_idx]['Response_tokens_sum'].values
                    else:
                        actual_concurrent = np.ones(model.pred_len)
                        actual_input = np.zeros(model.pred_len)
                        actual_output = np.zeros(model.pred_len)

                # 确保所有数组长度一致
                min_len = min(len(actual_concurrent), len(actual_input), len(actual_output), model.pred_len)
                actual_concurrent = actual_concurrent[:min_len]
                actual_input = actual_input[:min_len]
                actual_output = actual_output[:min_len]

                # 交错排列为 [feature1_time1, feature1_time2, ..., feature2_time1, feature2_time2, ...]
                actual = []
                for t in range(min_len):
                    actual.extend([actual_concurrent[t], actual_input[t], actual_output[t]])

                actuals.append(actual)

                # 计算误差 - 修改为支持多时间步预测
                # pred_flat 是展平的 [3 * pred_len]，actual 也是展平的 [3 * pred_len]
                error = [abs(pred_flat[i] - actual[i]) for i in range(len(actual))]
                errors.append(error)

        if len(predictions) == 0:
            logger.warning("没有生成任何预测对比数据")
            return {}

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        errors = np.array(errors)

        # 计算统计信息 - 修改为支持多时间步预测
        errors = np.array(errors)
        total_features = model.pred_len * 3  # pred_len个时间步 * 3个特征

        stats = {
            'comparison_predictions': predictions.tolist(),
            'comparison_actuals': actuals.tolist(),
            'comparison_errors': errors.tolist(),
            'num_comparison_samples': len(predictions),
            'mse_per_feature': [np.mean(errors[:, i]**2) for i in range(3)],  # 按特征类型聚合
            'mae_per_feature': [np.mean(errors[:, i::3]) for i in range(3)],  # 每个特征类型在所有时间步上的平均误差
            'rmse_per_feature': [np.sqrt(np.mean(errors[:, i::3]**2)) for i in range(3)],
            'feature_names': ['concurrent_requests', 'input_tokens', 'output_tokens'],
            'sample_indices': sample_indices,
            'seq_len': model.seq_len,  # 添加实际序列长度
            'pred_len': model.pred_len  # 添加预测长度
        }

  
        return stats

    def _log_prediction_plots(self, predictions: torch.Tensor, val_data: pd.DataFrame,
                             prediction_comparison: Dict = None):
        """记录关键预测图表到wandb - 包含预测值vs真实值对比图"""
        try:
            import matplotlib.pyplot as plt

            # 处理预测值
            if isinstance(predictions, torch.Tensor):
                pred_values = predictions.cpu().numpy()
            else:
                pred_values = np.array(predictions)

            # 创建多个核心图表

            # 图1: 训练vs验证损失对比（最重要的图表）
            if hasattr(self, 'history') and 'train_loss' in self.history:
                fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))

                # 绘制训练和验证损失
                epochs = range(len(self.history['train_loss']))
                ax1.plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2, color='blue')
                ax1.plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2, color='red')

                # 标记最佳验证损失
                best_epoch = np.argmin(self.history['val_loss'])
                best_val_loss = self.history['val_loss'][best_epoch]
                ax1.scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5,
                           label=f'Best: {best_val_loss:.4f}')

                ax1.set_title('Training vs Validation Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(self.output_dir / 'loss_comparison.png', dpi=150, bbox_inches='tight')

                # 上传到wandb
                if self.wandb_run:
                    wandb.log({
                        'loss_comparison': wandb.Image(str(self.output_dir / 'loss_comparison.png'))
                    })

                plt.close()

          # 图2: 真实值vs预测值时间序列对比图（核心新增图表）
            seq_len = self.config.get('sequence_length', 120)
            self._create_time_series_prediction_plot(predictions, val_data, prediction_comparison, seq_len)

            # 图3: 预测值vs真实值散点图（新增图表）
            self._create_scatter_prediction_plot(predictions, val_data, prediction_comparison)

            # 图4: 多样化预测对比图
            if prediction_comparison:
                self._create_diverse_prediction_plots(prediction_comparison, val_data)

            # 图5: 预测质量概览（2×2布局）
            self._create_prediction_overview_plot(predictions)

        except Exception as e:
            logger.warning(f"Failed to create prediction plots: {e}")
            logger.warning(f"Error details: {str(e)}")

    def _create_time_series_prediction_plot(self, predictions: torch.Tensor, val_data: pd.DataFrame,
                                         prediction_comparison: Dict = None, seq_len: int = 120):
        """创建时间序列预测对比图 - 使用预测对比数据生成完整曲线"""
        try:
            import matplotlib.pyplot as plt

            # 如果有预测对比数据，使用它来创建完整的预测vs真实值曲线
            if prediction_comparison and 'comparison_predictions' in prediction_comparison:
                self._create_detailed_time_series_plot(prediction_comparison, val_data)
            else:
                # 回退到简化版本
                self._create_simple_time_series_plot(predictions, val_data)

        except Exception as e:
            logger.warning(f"Failed to create time series plot: {e}")

    def _create_detailed_time_series_plot(self, prediction_comparison: Dict, val_data: pd.DataFrame, seq_len: int = None):
        """创建详细的时间序列对比图，显示历史数据序列和未来预测"""
        try:
            import matplotlib.pyplot as plt

            predictions = np.array(prediction_comparison['comparison_predictions'])
            actuals = np.array(prediction_comparison['comparison_actuals'])
            sample_indices = prediction_comparison['sample_indices']
            feature_names = prediction_comparison['feature_names']
            pred_len = prediction_comparison.get('pred_len', 3)  # 获取预测长度，默认为3

            # 动态获取实际的序列长度
            actual_seq_len = prediction_comparison.get('seq_len', seq_len)
            seq_len = actual_seq_len

            # 创建多个对比图 - 选择有代表性的时间点
            num_examples = min(4, len(sample_indices))  # 最多展示4个例子
            example_indices = np.linspace(0, len(sample_indices)-1, num_examples, dtype=int)

            # 确定数据列名
            col_mapping = {
                'concurrent_requests': 'Concurrent_requests' if 'Concurrent_requests' in val_data.columns else 'concurrent_requests',
                'input_tokens': 'Request_tokens_sum' if 'Request_tokens_sum' in val_data.columns else 'input_toks',
                'output_tokens': 'Response_tokens_sum' if 'Response_tokens_sum' in val_data.columns else 'output_toks'
            }

            # 创建多子图 - 每行一个例子，每列一个特征
            fig, axes = plt.subplots(num_examples, 3, figsize=(18, 4 * num_examples))
            if num_examples == 1:
                axes = axes.reshape(1, -1)

            for example_idx, sample_idx in enumerate(example_indices):
                # sample_indices存储的是滑动窗口的起始索引
                start_idx = sample_indices[sample_idx]
                # 序列结束索引 = 起始索引 + seq_len - 1
                actual_idx = start_idx + seq_len - 1

                # 确保显示窗口：seq_len个历史点 + pred_len个预测点
                end_idx = min(len(val_data), actual_idx + pred_len + 1)  # +pred_len+1为了显示预测段

                for feature_idx, feature_name in enumerate(feature_names):
                    true_col = col_mapping[feature_name]

                    if true_col in val_data.columns:
                        # 获取历史数据段 (从start_idx到start_idx+seq_len)
                        history_data = val_data.iloc[start_idx:start_idx + seq_len][true_col].values

                        # 获取未来真实值（确保有足够的数据）
                        future_end_idx = min(actual_idx + pred_len + 1, len(val_data))
                        future_data = val_data.iloc[actual_idx+1:future_end_idx][true_col].values

                        # 时间轴 - 修复连接问题
                        history_time = list(range(len(history_data)))
                        # 预测时间应该紧接着历史数据的最后一个点
                        pred_time = list(range(len(history_data) - 1, len(history_data) - 1 + pred_len))
                        # 未来真实值时间轴也应该紧接着历史数据
                        future_time = list(range(len(history_data) - 1, len(history_data) - 1 + len(future_data)))

                        # 绘制历史真实值（蓝色）
                        axes[example_idx, feature_idx].plot(history_time, history_data, 'b-',
                                                           label='Actual', linewidth=2, alpha=0.8)

                        # 绘制未来真实值（相同蓝色连接）
                        if len(future_data) > 0:
                            axes[example_idx, feature_idx].plot(future_time, future_data, 'b-',
                                                               linewidth=2, alpha=0.8)

                        # 绘制预测值 - 修正预测值提取方式
                        # predictions的形状应该是 [num_samples, pred_len * num_features]
                        feature_pred_start = feature_idx * pred_len
                        feature_pred_end = feature_pred_start + pred_len

                        if sample_idx < predictions.shape[0] and feature_pred_end <= predictions.shape[1]:
                            pred_values = predictions[sample_idx, feature_pred_start:feature_pred_end]
                        else:
                            pred_values = np.zeros(pred_len)  # 使用默认值

                        if len(pred_values) >= pred_len:
                            axes[example_idx, feature_idx].plot(pred_time, pred_values, 'r--',
                                                               label='Prediction', linewidth=2, alpha=0.8, marker='o', markersize=4)

                        # 添加分割线 - 修正位置
                        axes[example_idx, feature_idx].axvline(x=len(history_data)-1.5, color='orange',
                                                              linestyle=':', alpha=0.7, label='Prediction Start')

                        # 计算预测误差
                        if len(future_data) > 0 and len(pred_values) >= pred_len:
                            # 使用完整的预测长度和可用的未来数据
                            actual_for_error = future_data[:min(len(future_data), pred_len)]
                            pred_for_error = pred_values[:len(actual_for_error)]

                            if len(actual_for_error) > 0:
                                mse = np.mean((actual_for_error - pred_for_error) ** 2)
                                mae = np.mean(np.abs(actual_for_error - pred_for_error))

                                # 计算数据范围用于标准化误差显示
                                all_data = np.concatenate([history_data, future_data])
                                data_range = np.max(all_data) - np.min(all_data)
                                if data_range > 0:
                                    normalized_mae = mae / data_range
                                else:
                                    normalized_mae = 0
                            else:
                                mse = mae = normalized_mae = 0
                        else:
                            mse = mae = normalized_mae = 0

                        # 设置标题和标签 - 根据特征类型选择不同的误差显示方式
                        if feature_name == 'concurrent_requests':
                            # 并发请求数显示具体误差值
                            axes[example_idx, feature_idx].set_title(
                                f'{feature_name.replace("_", " ").title()}\n'
                                f'MAE: {mae:.2f}'
                            )
                        else:
                            # Token相关特征显示标准化误差
                            axes[example_idx, feature_idx].set_title(
                                f'{feature_name.replace("_", " ").title()}'
                            )

                            # 在子图底部添加误差信息（小字体）
                            if mae > 0:
                                axes[example_idx, feature_idx].text(
                                    0.5, 0.02, f'NMAE: {normalized_mae:.4f}',
                                    transform=axes[example_idx, feature_idx].transAxes,
                                    ha='center', va='bottom', fontsize=8, alpha=0.7
                                )

                        if example_idx == num_examples - 1:
                            axes[example_idx, feature_idx].set_xlabel('Time Steps')
                        if feature_idx == 0:
                            axes[example_idx, feature_idx].set_ylabel(f'Example {example_idx + 1}\nValue')

                        axes[example_idx, feature_idx].grid(True, alpha=0.3)

                        # 只在第一个例子显示图例
                        if example_idx == 0:
                            axes[example_idx, feature_idx].legend(loc='upper right')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'detailed_time_series_prediction.png', dpi=150, bbox_inches='tight')

            if self.wandb_run:
                wandb.log({
                    'detailed_time_series_prediction': wandb.Image(str(self.output_dir / 'detailed_time_series_prediction.png'))
                })

            plt.close()
            logger.info(f"详细时间序列预测图已生成 - 展示{num_examples}个例子，窗口大小{seq_len}，预测长度{pred_len}")

        except Exception as e:
            logger.warning(f"Failed to create detailed time series plot: {e}")

    def _create_simple_time_series_plot(self, predictions: torch.Tensor, val_data: pd.DataFrame):
        """创建简化版时间序列预测对比图"""
        try:
            import matplotlib.pyplot as plt

            # 处理预测值
            if isinstance(predictions, torch.Tensor):
                pred_values = predictions.cpu().numpy()
            else:
                pred_values = np.array(predictions)

            # 从验证数据中获取真实值的时间序列
            if len(val_data) >= 50:  # 需要足够的数据点来展示时间序列
                # 获取最后50个真实值用于对比
                recent_data = val_data.tail(50)

                # 确定列名
                concurrent_col = 'Concurrent_requests' if 'Concurrent_requests' in val_data.columns else 'concurrent_requests'
                input_token_col = 'Request_tokens_sum' if 'Request_tokens_sum' in val_data.columns else 'input_toks'
                output_token_col = 'Response_tokens_sum' if 'Response_tokens_sum' in val_data.columns else 'output_toks'

                # 创建时间序列图
                fig, axes = plt.subplots(3, 1, figsize=(15, 12))

                # 转换索引为时间戳或序号
                time_points = range(len(recent_data))

                # 子图1: 并发请求数对比
                if concurrent_col in recent_data.columns:
                    true_concurrent = recent_data[concurrent_col].values
                    pred_concurrent = pred_values[0]  # 使用预测的第一个值作为未来的预测

                    axes[0].plot(time_points, true_concurrent, 'b-', label='Actual', linewidth=2, alpha=0.8)
                    # 在未来时间点显示预测值
                    pred_time = list(time_points) + [max(time_points) + 1]
                    pred_series = list(true_concurrent) + [pred_concurrent]
                    axes[0].plot(pred_time, pred_series, 'r--', label='Predicted', linewidth=2, alpha=0.8)
                    axes[0].axvline(x=max(time_points), color='gray', linestyle=':', alpha=0.7, label='Prediction Point')

                    axes[0].set_title('Concurrent Requests: Actual vs Predicted')
                    axes[0].set_ylabel('Requests')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)

                # 子图2: 输入tokens对比
                if input_token_col in recent_data.columns:
                    true_input = recent_data[input_token_col].values
                    pred_input = pred_values[1] if len(pred_values) > 1 else pred_values[0]

                    axes[1].plot(time_points, true_input, 'g-', label='Actual', linewidth=2, alpha=0.8)
                    pred_time = list(time_points) + [max(time_points) + 1]
                    pred_series = list(true_input) + [pred_input]
                    axes[1].plot(pred_time, pred_series, 'r--', label='Predicted', linewidth=2, alpha=0.8)
                    axes[1].axvline(x=max(time_points), color='gray', linestyle=':', alpha=0.7, label='Prediction Point')

                    axes[1].set_title('Input Tokens: Actual vs Predicted')
                    axes[1].set_ylabel('Tokens')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)

                # 子图3: 输出tokens对比
                if output_token_col in recent_data.columns:
                    true_output = recent_data[output_token_col].values
                    pred_output = pred_values[2] if len(pred_values) > 2 else pred_values[0]

                    axes[2].plot(time_points, true_output, 'm-', label='Actual', linewidth=2, alpha=0.8)
                    pred_time = list(time_points) + [max(time_points) + 1]
                    pred_series = list(true_output) + [pred_output]
                    axes[2].plot(pred_time, pred_series, 'r--', label='Predicted', linewidth=2, alpha=0.8)
                    axes[2].axvline(x=max(time_points), color='gray', linestyle=':', alpha=0.7, label='Prediction Point')

                    axes[2].set_title('Output Tokens: Actual vs Predicted')
                    axes[2].set_ylabel('Tokens')
                    axes[2].set_xlabel('Time')
                    axes[2].legend()
                    axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(self.output_dir / 'simple_time_series_prediction.png', dpi=150, bbox_inches='tight')

                if self.wandb_run:
                    wandb.log({
                        'simple_time_series_prediction': wandb.Image(str(self.output_dir / 'simple_time_series_prediction.png'))
                    })

                plt.close()

        except Exception as e:
            logger.warning(f"Failed to create simple time series plot: {e}")

    def _create_scatter_prediction_plot(self, predictions: torch.Tensor, val_data: pd.DataFrame,
                                     prediction_comparison: Dict = None):
        """创建预测值vs真实值散点图"""
        try:
            import matplotlib.pyplot as plt

            # 如果有预测对比数据，使用它来创建散点图
            if prediction_comparison and 'comparison_predictions' in prediction_comparison:
                self._create_detailed_scatter_plot(prediction_comparison)
            else:
                # 回退到简化版本
                self._create_simple_scatter_plot(predictions, val_data)

        except Exception as e:
            logger.warning(f"Failed to create scatter plot: {e}")

    def _create_detailed_scatter_plot(self, prediction_comparison: Dict):
        """创建详细散点图，使用预测对比数据"""
        try:
            import matplotlib.pyplot as plt

            predictions = np.array(prediction_comparison['comparison_predictions'])
            actuals = np.array(prediction_comparison['comparison_actuals'])
            feature_names = prediction_comparison['feature_names']
            mse_per_feature = prediction_comparison['mse_per_feature']
            mae_per_feature = prediction_comparison['mae_per_feature']
            rmse_per_feature = prediction_comparison['rmse_per_feature']

            # 创建2×2布局的散点图
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()

            # 为每个特征创建散点图
            # 注意：predictions和actuals的格式是 [pred_len*3]，按特征交错排列
            # 即 [concurrent_t1, concurrent_t2, ..., input_t1, input_t2, ..., output_t1, output_t2, ...]
            pred_len = len(predictions[0]) // 3  # 计算预测长度

            for i, feature_name in enumerate(feature_names):
                if i < len(axes):
                    # 计算该特征在数组中的起始和结束索引
                    start_idx = i * pred_len
                    end_idx = (i + 1) * pred_len

                    # 提取该特征所有时间步的数据
                    true_vals = actuals[:, start_idx:end_idx].flatten()
                    pred_vals = predictions[:, start_idx:end_idx].flatten()

                    # 绘制散点图
                    axes[i].scatter(true_vals, pred_vals, alpha=0.6, s=40, color='blue', label='Predictions')

                    # 绘制y=x线（完美预测线）
                    min_val = min(min(true_vals), min(pred_vals))
                    max_val = max(max(true_vals), max(pred_vals))
                    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

                    # 计算R²
                    ss_res = np.sum((true_vals - pred_vals) ** 2)
                    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                    # 设置图表属性
                    axes[i].set_xlabel('True Values')
                    axes[i].set_ylabel('Predicted Values')
                    axes[i].set_title(f'{feature_name.replace("_", " ").title()}\n'
                                    f'R² = {r2:.4f}, MSE = {mse_per_feature[i]:.4f}')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)

                    # 添加统计信息
                    stats_text = (f'Points: {len(true_vals)}\n'
                                f'Mean True: {np.mean(true_vals):.2f}\n'
                                f'Mean Pred: {np.mean(pred_vals):.2f}\n'
                                f'RMSE: {rmse_per_feature[i]:.4f}')
                    axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                               verticalalignment='top', bbox=dict(boxstyle='round',
                               facecolor='wheat', alpha=0.8), fontsize=9)

            # 隐藏最后一个子图（如果只有3个特征）
            if len(feature_names) < 4:
                axes[3].axis('off')

                # 在最后一个子图位置添加总体统计
                overall_stats_text = "Overall Prediction Performance\n" + "="*30 + "\n"
                for i, name in enumerate(feature_names):
                    overall_stats_text += f"{name.replace('_', ' ').title()}:\n"
                    overall_stats_text += f"  MSE: {mse_per_feature[i]:.4f}\n"
                    overall_stats_text += f"  MAE: {mae_per_feature[i]:.4f}\n"
                    overall_stats_text += f"  RMSE: {rmse_per_feature[i]:.4f}\n\n"

                overall_stats_text += f"Total Samples: {len(predictions)}"

                axes[3].text(0.1, 0.9, overall_stats_text, transform=axes[3].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round',
                           facecolor='lightblue', alpha=0.8), fontsize=10,
                           family='monospace')
                axes[3].set_title('Overall Statistics')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'detailed_scatter_prediction.png', dpi=150, bbox_inches='tight')

            if self.wandb_run:
                wandb.log({
                    'detailed_scatter_prediction': wandb.Image(str(self.output_dir / 'detailed_scatter_prediction.png'))
                })

            plt.close()
            logger.info("详细散点预测图已生成")

        except Exception as e:
            logger.warning(f"Failed to create detailed scatter plot: {e}")

    def _create_simple_scatter_plot(self, predictions: torch.Tensor, val_data: pd.DataFrame):
        """创建简化版散点图"""
        try:
            import matplotlib.pyplot as plt

            # 处理预测值
            if isinstance(predictions, torch.Tensor):
                pred_values = predictions.cpu().numpy()
            else:
                pred_values = np.array(predictions)

            # 获取真实值
            if len(val_data) >= 10:  # 需要足够的数据点
                recent_data = val_data.tail(10)

                # 确定列名
                concurrent_col = 'Concurrent_requests' if 'Concurrent_requests' in val_data.columns else 'concurrent_requests'
                input_token_col = 'Request_tokens_sum' if 'Request_tokens_sum' in val_data.columns else 'input_toks'
                output_token_col = 'Response_tokens_sum' if 'Response_tokens_sum' in val_data.columns else 'output_toks'

                # 收集真实值和预测值
                true_values = []
                pred_values_extended = []

                if concurrent_col in recent_data.columns:
                    true_concurrent = recent_data[concurrent_col].values
                    pred_concurrent = pred_values[0] if len(pred_values) > 0 else true_concurrent[-1]
                    true_values.extend(true_concurrent)
                    pred_values_extended.extend([pred_concurrent] * len(true_concurrent))

                if input_token_col in recent_data.columns:
                    true_input = recent_data[input_token_col].values
                    pred_input = pred_values[1] if len(pred_values) > 1 else true_input[-1]
                    true_values.extend(true_input)
                    pred_values_extended.extend([pred_input] * len(true_input))

                if output_token_col in recent_data.columns:
                    true_output = recent_data[output_token_col].values
                    pred_output = pred_values[2] if len(pred_values) > 2 else true_output[-1]
                    true_values.extend(true_output)
                    pred_values_extended.extend([pred_output] * len(true_output))

                if len(true_values) > 0 and len(pred_values_extended) > 0:
                    # 创建散点图
                    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

                    # 绘制散点图
                    ax.scatter(true_values, pred_values_extended, alpha=0.6, s=50, color='blue', label='Predictions')

                    # 绘制y=x线（完美预测线）
                    min_val = min(min(true_values), min(pred_values_extended))
                    max_val = max(max(true_values), max(pred_values_extended))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

                    # 计算R²
                    true_array = np.array(true_values)
                    pred_array = np.array(pred_values_extended)
                    ss_res = np.sum((true_array - pred_array) ** 2)
                    ss_tot = np.sum((true_array - np.mean(true_array)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                    # 计算RMSE
                    rmse = np.sqrt(np.mean((true_array - pred_array) ** 2))
                    mae = np.mean(np.abs(true_array - pred_array))

                    # 设置图表属性
                    ax.set_xlabel('True Values')
                    ax.set_ylabel('Predicted Values')
                    ax.set_title(f'True vs Predicted Values\nR² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    # 添加统计信息框
                    stats_text = f'Number of points: {len(true_values)}\nMean True: {np.mean(true_values):.2f}\nMean Pred: {np.mean(pred_array):.2f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'simple_scatter_prediction.png', dpi=150, bbox_inches='tight')

                    if self.wandb_run:
                        wandb.log({
                            'simple_scatter_prediction': wandb.Image(str(self.output_dir / 'simple_scatter_prediction.png'))
                        })

                    plt.close()

        except Exception as e:
            logger.warning(f"Failed to create simple scatter plot: {e}")

    def _create_diverse_prediction_plots(self, prediction_comparison: Dict, val_data: pd.DataFrame):
        """创建多样化的预测对比图"""
        try:
            import matplotlib.pyplot as plt

            predictions = np.array(prediction_comparison['comparison_predictions'])
            actuals = np.array(prediction_comparison['comparison_actuals'])
            sample_indices = prediction_comparison['sample_indices']
            feature_names = prediction_comparison['feature_names']

            # 图1: 不同负载水平的预测表现
            self._create_load_level_analysis(predictions, actuals, sample_indices, val_data)

            # 图2: 预测误差分布分析
            self._create_error_distribution_analysis(predictions, actuals, feature_names)

            # 图3: 时间序列连续预测展示
            self._create_continuous_prediction_series(predictions, actuals, sample_indices, val_data)

        
        except Exception as e:
            logger.warning(f"Failed to create diverse prediction plots: {e}")

    def _create_load_level_analysis(self, predictions: np.ndarray, actuals: np.ndarray,
                                  sample_indices: list, val_data: pd.DataFrame):
        """分析不同负载水平下的预测表现"""
        try:
            import matplotlib.pyplot as plt

            # 注意：predictions和actuals的格式是 [pred_len*3]，按特征交错排列
            # 即 [concurrent_t1, concurrent_t2, ..., input_t1, input_t2, ..., output_t1, output_t2, ...]
            pred_len = len(predictions[0]) // 3  # 计算预测长度

            # 根据并发请求数将样本分为不同负载级别
            # 需要从交错数据中提取concurrent_requests的平均值
            concurrent_requests = []
            for sample_idx in range(len(actuals)):
                # 提取concurrent_requests的所有时间步数据并计算平均值
                concurrent_data = actuals[sample_idx, 0:pred_len]
                concurrent_avg = np.mean(concurrent_data)
                concurrent_requests.append(concurrent_avg)
            concurrent_requests = np.array(concurrent_requests)

            # 定义负载级别
            low_load = concurrent_requests <= 10
            medium_load = (concurrent_requests > 10) & (concurrent_requests <= 50)
            high_load = concurrent_requests > 50

            load_levels = [
                ('Low Load (≤10)', low_load, 'lightblue'),
                ('Medium Load (11-50)', medium_load, 'orange'),
                ('High Load (>50)', high_load, 'red')
            ]

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            for feature_idx, feature_name in enumerate(['concurrent_requests', 'input_tokens', 'output_tokens']):
                ax = axes[feature_idx]

                # 计算该特征在数组中的起始和结束索引
                start_idx = feature_idx * pred_len
                end_idx = (feature_idx + 1) * pred_len

                for level_name, mask, color in load_levels:
                    if np.any(mask):
                        # 提取该特征所有时间步的数据并计算平均值
                        actual_vals = actuals[mask, start_idx:end_idx]
                        pred_vals = predictions[mask, start_idx:end_idx]

                        # 展平所有样本和时间步的数据
                        actual_flat = actual_vals.flatten()
                        pred_flat = pred_vals.flatten()

                        # 计算该级别的R² - 使用展平的数据
                        ss_res = np.sum((actual_flat - pred_flat) ** 2)
                        ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                        ax.scatter(actual_flat, pred_flat, alpha=0.6, s=40,
                                 color=color, label=f'{level_name} (R²={r2:.3f})')

                # 绘制完美预测线 - 使用该特征的所有数据
                feature_actuals = actuals[:, start_idx:end_idx].flatten()
                feature_predictions = predictions[:, start_idx:end_idx].flatten()
                min_val = min(np.min(feature_actuals), np.min(feature_predictions))
                max_val = max(np.max(feature_actuals), np.max(feature_predictions))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--',
                       linewidth=2, label='Perfect Prediction', alpha=0.8)

                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'{feature_name.replace("_", " ").title()} - Load Level Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'load_level_analysis.png', dpi=150, bbox_inches='tight')

            if self.wandb_run:
                wandb.log({
                    'load_level_analysis': wandb.Image(str(self.output_dir / 'load_level_analysis.png'))
                })

            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create load level analysis: {e}")

    def _create_error_distribution_analysis(self, predictions: np.ndarray, actuals: np.ndarray,
                                         feature_names: list):
        """分析预测误差分布"""
        try:
            import matplotlib.pyplot as plt

            # 计算绝对误差和相对误差
            abs_errors = np.abs(predictions - actuals)
            rel_errors = np.zeros_like(predictions)

            # 计算相对误差（避免除零）
            for i in range(actuals.shape[1]):
                mask = actuals[:, i] > 0
                rel_errors[mask, i] = abs_errors[mask, i] / actuals[mask, i] * 100

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 第一行：绝对误差分布
            for i, feature_name in enumerate(feature_names):
                axes[0, i].hist(abs_errors[:, i], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, i].set_title(f'{feature_name.replace("_", " ").title()} - Absolute Error')
                axes[0, i].set_xlabel('Absolute Error')
                axes[0, i].set_ylabel('Frequency')
                axes[0, i].grid(True, alpha=0.3)

                # 添加统计信息
                mean_error = np.mean(abs_errors[:, i])
                median_error = np.median(abs_errors[:, i])
                axes[0, i].axvline(mean_error, color='red', linestyle='--',
                                  label=f'Mean: {mean_error:.2f}')
                axes[0, i].axvline(median_error, color='orange', linestyle='--',
                                  label=f'Median: {median_error:.2f}')
                axes[0, i].legend()

            # 第二行：相对误差分布
            for i, feature_name in enumerate(feature_names):
                valid_errors = rel_errors[:, i][rel_errors[:, i] < 200]  # 过滤极端值
                axes[1, i].hist(valid_errors, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
                axes[1, i].set_title(f'{feature_name.replace("_", " ").title()} - Relative Error (%)')
                axes[1, i].set_xlabel('Relative Error (%)')
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].grid(True, alpha=0.3)

                # 添加统计信息
                mean_rel_error = np.mean(valid_errors)
                median_rel_error = np.median(valid_errors)
                axes[1, i].axvline(mean_rel_error, color='red', linestyle='--',
                                  label=f'Mean: {mean_rel_error:.1f}%')
                axes[1, i].axvline(median_rel_error, color='orange', linestyle='--',
                                  label=f'Median: {median_rel_error:.1f}%')
                axes[1, i].legend()

            plt.tight_layout()
            plt.savefig(self.output_dir / 'error_distribution_analysis.png', dpi=150, bbox_inches='tight')

            if self.wandb_run:
                wandb.log({
                    'error_distribution_analysis': wandb.Image(str(self.output_dir / 'error_distribution_analysis.png'))
                })

            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create error distribution analysis: {e}")

    def _create_continuous_prediction_series(self, predictions: np.ndarray, actuals: np.ndarray,
                                           sample_indices: list, val_data: pd.DataFrame):
        """创建连续时间序列预测展示"""
        try:
            import matplotlib.pyplot as plt

            # 选择一个连续的时间段展示
            start_sample = len(sample_indices) // 3  # 从中间开始
            num_continuous = min(20, len(sample_indices) - start_sample)  # 连续20个预测点

            if num_continuous < 5:
                return

            fig, axes = plt.subplots(3, 1, figsize=(16, 12))

            # 确定数据列名
            col_mapping = {
                'concurrent_requests': 'Concurrent_requests' if 'Concurrent_requests' in val_data.columns else 'concurrent_requests',
                'input_tokens': 'Request_tokens_sum' if 'Request_tokens_sum' in val_data.columns else 'input_toks',
                'output_tokens': 'Response_tokens_sum' if 'Response_tokens_sum' in val_data.columns else 'output_toks'
            }

            feature_names = ['concurrent_requests', 'input_tokens', 'output_tokens']

            for feature_idx, feature_name in enumerate(feature_names):
                true_col = col_mapping[feature_name]

                if true_col in val_data.columns:
                    # 获取连续时间段的数据
                    continuous_indices = sample_indices[start_sample:start_sample + num_continuous]

                    # 获取对应的真实值（需要偏移1，因为预测的是下一个时间点）
                    actual_values = []
                    for idx in continuous_indices:
                        actual_idx = idx + 1  # 预测目标的位置
                        if actual_idx < len(val_data):
                            actual_values.append(val_data.iloc[actual_idx][true_col])
                        else:
                            actual_values.append(val_data.iloc[-1][true_col])

                    # 获取预测值 - 直接使用预测对比数据中的预测值
                    # predictions是列表，每个元素是展平的 [3 * pred_len] 数组
                    # 顺序: [pred_len个concurrent_requests, pred_len个input_tokens, pred_len个output_tokens]
                    pred_len = self.model.pred_len
                    pred_values = []
                    for sample_idx in range(start_sample, start_sample + num_continuous):
                        pred_flat = predictions[sample_idx]
                        # 计算该特征在展平数组中的起始位置
                        feature_start = feature_idx * pred_len
                        # 取第一个预测时间步的值作为代表
                        pred_values.append(pred_flat[feature_start])
                    pred_values = np.array(pred_values)

                    # 创建时间轴
                    time_points = range(len(actual_values))

                    # 绘制真实值和预测值
                    axes[feature_idx].plot(time_points, actual_values, 'b-o',
                                         label='Actual', linewidth=2, markersize=4, alpha=0.8)
                    axes[feature_idx].plot(time_points, pred_values, 'r--s',
                                         label='Predicted', linewidth=2, markersize=4, alpha=0.8)

                    # 计算这个时间段的平均误差
                    mae = np.mean(np.abs(np.array(actual_values) - pred_values))
                    rmse = np.sqrt(np.mean((np.array(actual_values) - pred_values) ** 2))

                    axes[feature_idx].set_title(
                        f'{feature_name.replace("_", " ").title()} - Continuous Prediction\n'
                        f'MAE: {mae:.2f}, RMSE: {rmse:.2f}'
                    )
                    axes[feature_idx].set_xlabel('Time Steps')
                    axes[feature_idx].set_ylabel('Value')
                    axes[feature_idx].legend()
                    axes[feature_idx].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'continuous_prediction_series.png', dpi=150, bbox_inches='tight')

            if self.wandb_run:
                wandb.log({
                    'continuous_prediction_series': wandb.Image(str(self.output_dir / 'continuous_prediction_series.png'))
                })

            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create continuous prediction series: {e}")

    def _create_prediction_overview_plot(self, predictions: torch.Tensor):
        """创建预测质量概览图"""
        try:
            import matplotlib.pyplot as plt

            # 处理预测值
            if isinstance(predictions, torch.Tensor):
                pred_values = predictions.cpu().numpy()
            else:
                pred_values = np.array(predictions)

            if pred_values.size > 0:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # 左上：验证MSE和MAE趋势
                if hasattr(self, 'history') and 'val_mse' in self.history:
                    epochs = range(len(self.history['val_mse']))
                    axes[0, 0].plot(epochs, self.history['val_mse'], label='MSE', linewidth=2, color='red')
                    axes[0, 0].plot(epochs, self.history['val_mae'], label='MAE', linewidth=2, color='orange')
                    axes[0, 0].set_title('Validation Error Trends')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Error')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)

                # 右上：学习率变化
                if hasattr(self, 'history') and 'learning_rates' in self.history:
                    epochs = range(len(self.history['learning_rates']))
                    axes[0, 1].semilogy(epochs, self.history['learning_rates'], linewidth=2, color='green')
                    axes[0, 1].set_title('Learning Rate Schedule')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Learning Rate (log scale)')
                    axes[0, 1].grid(True, alpha=0.3)

                # 左下：预测值分布
                axes[1, 0].hist(pred_values.flatten(), bins=25, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Prediction Distribution')
                axes[1, 0].set_xlabel('Prediction Value')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True, alpha=0.3)

                # 右下：预测统计
                stats_values = [np.mean(pred_values), np.std(pred_values),
                               np.min(pred_values), np.max(pred_values)]
                bars = axes[1, 1].bar(['Mean', 'Std', 'Min', 'Max'], stats_values,
                                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold'],
                                      alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Prediction Statistics')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].grid(True, alpha=0.3)

                # 添加数值标签
                for bar, value in zip(bars, stats_values):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.3f}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(self.output_dir / 'prediction_overview.png', dpi=150, bbox_inches='tight')

                if self.wandb_run:
                    wandb.log({
                        'prediction_overview': wandb.Image(str(self.output_dir / 'prediction_overview.png'))
                    })

                plt.close()

        except Exception as e:
            logger.warning(f"Failed to create overview plot: {e}")

    def save_config(self):
        """保存训练配置"""
        config_path = self.output_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"配置保存到: {config_path}")

        if self.wandb_run:
            try:
                # 使用相对路径避免wandb路径问题
                import os
                original_cwd = os.getcwd()
                os.chdir(self.output_dir.parent)
                wandb.save('training_config.json')
                os.chdir(original_cwd)
            except Exception as e:
                logger.warning(f"wandb保存配置失败: {e}")

    def cleanup(self):
        """清理资源"""
        if self.wandb_run:
            self.wandb_run.finish()
            logger.info("wandb run finished")


def load_default_config() -> Dict:
    """加载默认配置"""
    return {
        'model_type': 'DLinear',
        'sequence_length': 50,
        'prediction_horizon': 10,
        'batch_size': 32,
        'learning_rate': 0.0005,
        'epochs': 100,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'early_stopping_patience': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': './training_output',
        'use_wandb': True,
        'wandb_project': 'smartllm-serve',
        'wandb_entity': None,
        'wandb_run_name': None,
        'loss_weights': {
            'time_weight': 2.0,
            'input_token_weight': 1.0,
            'output_token_weight': 1.0,
            'time_loss_type': 'huber',
            'token_loss_type': 'huber',
            'normalize_weights': True
        }
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练预测器模型')

    # 数据参数
    parser.add_argument('--train_data', type=str, required=True, help='训练数据路径')
    parser.add_argument('--val_data', type=str, required=True, help='验证数据路径')
    parser.add_argument('--output_dir', type=str, default='./training_output', help='输出目录')

    # 模型参数
    parser.add_argument('--sequence_length', type=int, default=50, help='序列长度')
    parser.add_argument('--prediction_horizon', type=int, default=10, help='预测步数')
    parser.add_argument('--hidden_size', type=int, default=64, help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--early_stopping_patience', type=int, default=50, help='早停耐心值')

    # 损失函数参数
    parser.add_argument('--time_weight', type=float, default=2.0, help='时间间隔损失权重')
    parser.add_argument('--input_token_weight', type=float, default=1.0, help='输入token损失权重')
    parser.add_argument('--output_token_weight', type=float, default=1.0, help='输出token损失权重')
    parser.add_argument('--loss_type', type=str, default='huber', choices=['mse', 'mae', 'huber'], help='损失函数类型')

    # 设备参数
    parser.add_argument('--device', type=str, default='auto', help='设备 (auto/cpu/cuda)')

    # Wandb参数
    parser.add_argument('--use_wandb', action='store_true', help='使用wandb记录')
    parser.add_argument('--wandb_project', type=str, default='smartllm-serve', help='wandb项目名')
    parser.add_argument('--wandb_entity', type=str, help='wandb实体名')
    parser.add_argument('--wandb_run_name', type=str, help='wandb运行名')

    # 配置文件
    parser.add_argument('--config', type=str, help='配置文件路径')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = load_default_config()

    # 更新配置
    config.update({
        'train_data_path': args.train_data,
        'val_data_path': args.val_data,
        'output_dir': args.output_dir,
        'sequence_length': args.sequence_length,
        'prediction_horizon': args.prediction_horizon,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'wandb_run_name': args.wandb_run_name,
        # 更新损失函数权重
        'loss_weights': {
            'time_weight': args.time_weight,
            'input_token_weight': args.input_token_weight,
            'output_token_weight': args.output_token_weight,
            'time_loss_type': args.loss_type,
            'token_loss_type': args.loss_type,
            'normalize_weights': True
        }
    })

    # 设置设备
    if args.device == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config['device'] = args.device

    logger.info(f"使用设备: {config['device']}")

    # 创建训练器
    trainer = PredictorTrainer(config)

    try:
        # 保存配置
        trainer.save_config()

        # 训练模型
        model, history = trainer.train()

        # 保存历史记录
        trainer.history = history

        # 评估模型
        val_data = pd.read_csv(args.val_data)
        metrics = trainer.evaluate(model, val_data, history)

        logger.info("训练和评估完成")
        # 只输出关键指标
        key_metrics = {k: v for k, v in metrics.items() if 'loss' in k or 'mse' in k or 'mae' in k}
        logger.info(f"关键指标: {key_metrics}")

    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

    finally:
        # 清理资源
        trainer.cleanup()


if __name__ == "__main__":
    main()