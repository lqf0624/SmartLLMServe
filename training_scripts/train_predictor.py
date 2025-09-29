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
from typing import Dict, List, Tuple, Optional
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
            'learning_rate': self.config.get('learning_rate', 0.001),
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

        # 训练模型（使用分批训练减少内存使用）
        logger.info("开始分批训练...")
        history = model.train_batch(
            train_data,
            validation_split=0.2,  # 从训练数据中分割验证集
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 64),
            patience=self.config.get('early_stopping_patience', 10)
        )

        # 记录最终指标到wandb
        if self.wandb_run:
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            best_val_loss = min(history['val_loss'])

            wandb.log({
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'best_val_loss': best_val_loss,
                'total_epochs': len(history['train_loss'])
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
        predictions = model.predict_future(val_data, steps=self.config.get('prediction_horizon', 10))

        # 计算各种指标
        metrics = self._calculate_metrics(predictions, val_data)

        # 记录到wandb
        if self.wandb_run:
            wandb.log({
                'eval_metrics': metrics,
                'prediction_confidence': predictions['confidence']
            })

            # 创建预测对比图表
            self._log_prediction_plots(predictions, val_data)

        return metrics

    def _calculate_metrics(self, predictions: Dict, val_data: pd.DataFrame) -> Dict:
        """计算评估指标"""
        # 这里可以添加更复杂的指标计算
        pred_values = np.array(predictions['predictions'])
        confidence = predictions['confidence']

        metrics = {
            'mean_confidence': np.mean(confidence),
            'std_confidence': np.std(confidence),
            'prediction_shape': pred_values.shape,
            'total_predicted_requests': pred_values.shape[0] * pred_values.shape[1] if len(pred_values.shape) > 1 else pred_values.shape[0]
        }

        return metrics

    def _log_prediction_plots(self, predictions: Dict, val_data: pd.DataFrame):
        """记录预测图表到wandb"""
        try:
            import matplotlib.pyplot as plt

            # 创建预测对比图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # 图1: 损失曲线
            if hasattr(self, 'history'):
                axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
                axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
                axes[0, 0].set_title('Training Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)

            # 图2: 预测置信度分布
            confidence = predictions['confidence']
            axes[0, 1].hist(confidence, bins=20, alpha=0.7)
            axes[0, 1].set_title('Prediction Confidence Distribution')
            axes[0, 1].set_xlabel('Confidence')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)

            # 保存图表
            plt.tight_layout()
            plt.savefig(self.output_dir / 'training_summary.png', dpi=150, bbox_inches='tight')

            # 上传到wandb
            if self.wandb_run:
                wandb.log({
                    'training_summary': wandb.Image(str(self.output_dir / 'training_summary.png'))
                })

            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create prediction plots: {e}")

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