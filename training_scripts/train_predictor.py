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
from typing import Dict, Tuple
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

        self.wandb_run = wandb.init(
            project=self.config.get('wandb_project', 'smartllm-serve'),
            entity=self.config.get('wandb_entity', None),
            name=self.config.get('wandb_run_name', f'predictor_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
            config=wandb_config,
            tags=['predictor', 'burstgpt', 'channel-independent'],
            reinit=True
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

        # 创建模型 - 通道独立版本使用标准MSE损失
        model = DLinearPredictor(**model_config)

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

        logger.info("训练完成")
        return model, history

    def evaluate(self, model: DLinearPredictor, val_data: pd.DataFrame) -> Dict:
        """评估模型性能"""
        logger.info("开始评估")

        # 进行预测
        predictions = model.predict(val_data, steps=self.config.get('prediction_horizon', 10))

        # 计算各种指标
        metrics = self._calculate_metrics(predictions, val_data)

        # 记录到wandb
        if self.wandb_run:
            wandb.log({
                'eval_metrics': metrics,
            })

            # 创建预测对比图表
            self._log_prediction_plots(predictions, val_data)

        return metrics

    def _calculate_metrics(self, predictions: torch.Tensor, val_data: pd.DataFrame) -> Dict:
        """计算评估指标"""
        # 处理预测结果
        if isinstance(predictions, torch.Tensor):
            pred_values = predictions.cpu().numpy()
        else:
            pred_values = np.array(predictions)

        metrics = {
            'prediction_shape': pred_values.shape,
            'mean_prediction': np.mean(pred_values),
            'std_prediction': np.std(pred_values),
            'min_prediction': np.min(pred_values),
            'max_prediction': np.max(pred_values),
            'validation_data_size': len(val_data)
        }

        return metrics

    def _log_prediction_plots(self, predictions: torch.Tensor, val_data: pd.DataFrame):
        """记录关键预测图表到wandb - 只在评估时生成少量核心图表"""
        try:
            import matplotlib.pyplot as plt

            # 处理预测值
            if isinstance(predictions, torch.Tensor):
                pred_values = predictions.cpu().numpy()
            else:
                pred_values = np.array(predictions)

            # 只创建1-2个核心图表

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

            # 图2: 预测质量概览（2×2布局）
            if pred_values.size > 0:
                fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

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
            logger.warning(f"Failed to create prediction plots: {e}")
            logger.warning(f"Error details: {str(e)}")

    def save_config(self):
        """保存训练配置"""
        config_path = self.output_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"配置保存到: {config_path}")

        if self.wandb_run:
            wandb.save(str(config_path))

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
        'batch_size': 64,
        'learning_rate': 0.001,
        'epochs': 100,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1,
        'early_stopping_patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': './training_output',
        'use_wandb': True,
        'wandb_project': 'smartllm-serve',
        'wandb_entity': None,
        'wandb_run_name': None,
        'loss_weights': {
            'time_weight': 1.0,
            'input_token_weight': 0.5,
            'output_token_weight': 0.5,
            'time_loss_type': 'mse',
            'token_loss_type': 'mse',
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
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='早停耐心值')

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
        'wandb_run_name': args.wandb_run_name
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
        metrics = trainer.evaluate(model, val_data)

        logger.info("训练和评估完成")
        logger.info(f"最终指标: {metrics}")

    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

    finally:
        # 清理资源
        trainer.cleanup()


if __name__ == "__main__":
    main()