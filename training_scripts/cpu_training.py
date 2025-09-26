#!/usr/bin/env python3
"""
CPU环境训练脚本

专门为CPU环境优化的训练脚本，专注于LSTM和DLinear模型的快速训练
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from typing import Dict, List, Any
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor.request_predictor import RequestPredictor
from predictor.model_interface import ModelConfig, ModelType

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CPUTrainer:
    """CPU环境训练器"""

    def __init__(self):
        self.predictor = None
        self.data = None
        self.training_results = {}
        self.saved_models = []

    def setup_cpu_configs(self, models: List[str] = None) -> Dict[str, ModelConfig]:
        """设置CPU优化的训练配置"""
        if models is None:
            models = ['lstm', 'dlinear']

        print(f"\n配置CPU训练模型: {models}")
        print("使用设备: cpu")

        model_configs = {}

        if 'lstm' in models:
            model_configs['lstm'] = ModelConfig(
                model_type=ModelType.LSTM,
                model_name='cpu_lstm',
                device='cpu',
                sequence_length=80,  # 减少序列长度以加快CPU训练
                prediction_horizon=15,
                epochs=50,  # 减少训练轮数
                batch_size=32,  # 减少批量大小
                learning_rate=0.001,
                early_stopping=True,
                patience=10
            )

        if 'dlinear' in models:
            model_configs['dlinear'] = ModelConfig(
                model_type=ModelType.DLINEAR,
                model_name='cpu_dlinear',
                device='cpu',
                sequence_length=80,
                prediction_horizon=15,
                epochs=80,
                learning_rate=0.001,
                early_stopping=True,
                patience=15
            )

        # 显示配置信息
        for name, config in model_configs.items():
            print(f"\n{name.upper()}:")
            print(f"  类型: {config.model_type.value}")
            print(f"  序列长度: {config.sequence_length}")
            print(f"  预测步数: {config.prediction_horizon}")
            print(f"  训练轮数: {config.epochs}")
            print(f"  批量大小: {config.batch_size}")
            print(f"  学习率: {config.learning_rate}")

        return model_configs

    def load_data(self, data_path: str = None, sample_size: int = 50000) -> bool:
        """加载训练数据"""
        print("\n加载数据")
        print("-" * 30)

        try:
            # 确定数据路径
            if data_path is None:
                possible_paths = [
                    'dataset/BurstGPT_1_cleaned.csv',
                    'dataset/BurstGPT_1.csv',
                    'BurstGPT_1_cleaned.csv',
                    'BurstGPT_1.csv'
                ]

                data_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        data_path = path
                        break

                if data_path is None:
                    print("错误: 未找到数据文件")
                    return False

            print(f"数据文件: {data_path}")

            # 加载数据
            data = pd.read_csv(data_path)
            print(f"原始数据: {data.shape[0]:,} 条记录")

            # CPU环境使用更小的数据集以加快训练
            if len(data) > sample_size:
                data = data.sample(n=sample_size, random_state=42)
                print(f"采样到: {data.shape[0]:,} 条记录")

            # 数据质量检查
            required_columns = ['Timestamp', 'Request tokens', 'Response tokens']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"错误: 缺少必需列: {missing_columns}")
                return False

            # 清理异常值
            data = data[data['Request tokens'] > 0]
            data = data[data['Response tokens'] > 0]
            if 'time_interval' in data.columns:
                data = data[data['time_interval'] >= 0]
            print(f"清理后数据: {data.shape[0]:,} 条记录")

            self.data = data
            return True

        except Exception as e:
            print(f"数据加载失败: {e}")
            logger.error(f"Data loading failed: {e}")
            return False

    def initialize_predictor(self) -> bool:
        """初始化预测器"""
        print("\n初始化CPU预测器")
        print("-" * 30)

        try:
            self.predictor = RequestPredictor()
            print(f"预测器初始化成功")
            return True

        except Exception as e:
            print(f"预测器初始化失败: {e}")
            logger.error(f"Predictor initialization failed: {e}")
            return False

    def train_models(self) -> bool:
        """训练所有模型"""
        print("\n模型训练 (CPU)")
        print("-" * 30)

        if self.predictor is None or self.data is None:
            print("错误: 预测器或数据未初始化")
            return False

        print(f"\n开始训练 LSTM + DLinear 模型...")
        print(f"训练数据: {self.data.shape[0]:,} 条记录")

        # 训练模型
        try:
            self.training_results = self.predictor.train_models(self.data)

            # 输出训练结果
            print(f"\n训练结果:")
            successful_models = []

            for model_name, result in self.training_results.items():
                if isinstance(result, dict):
                    # 检查字典中的状态
                    if result.get('success') or result.get('status') == 'completed':
                        print(f"[OK] {model_name}:")
                        if 'train_loss' in result:
                            print(f"    训练损失: {result['train_loss']:.4f}")
                        if 'val_loss' in result:
                            print(f"    验证损失: {result['val_loss']:.4f}")
                        if 'epochs_trained' in result:
                            print(f"    训练轮数: {result['epochs_trained']}")
                        successful_models.append(model_name)
                    else:
                        error_msg = result.get('message', '训练失败')
                        print(f"[ERROR] {model_name}: {error_msg}")
                else:
                    # 检查对象属性
                    if hasattr(result, 'success') and result.success:
                        print(f"[OK] {model_name}:")
                        if hasattr(result, 'final_loss'):
                            print(f"    最终损失: {result.final_loss:.4f}")
                        if hasattr(result, 'epochs_trained'):
                            print(f"    训练轮数: {result.epochs_trained}")
                        successful_models.append(model_name)
                    else:
                        error_msg = getattr(result, 'error_message', '未知错误') if hasattr(result, 'error_message') else '训练失败'
                        print(f"[ERROR] {model_name}: {error_msg}")

            print(f"\n成功训练: {len(successful_models)}/{len(self.training_results)} 个模型")
            return len(successful_models) > 0

        except Exception as e:
            print(f"训练过程失败: {e}")
            logger.error(f"Training failed: {e}")
            return False

    def save_models(self, model_dir: str = "models") -> bool:
        """保存训练好的模型"""
        print("\n模型保存")
        print("-" * 30)

        if not self.training_results:
            print("错误: 没有训练结果")
            return False

        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        print(f"模型目录: {model_dir}")

        # 保存成功训练的模型
        self.saved_models = []
        for model_name, result in self.training_results.items():
            if result.success:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = f"{model_dir}/{model_name}_cpu_{timestamp}.pth"

                    if self.predictor.save_model(model_name, model_path):
                        self.saved_models.append((model_name, model_path))
                        print(f"[OK] 保存模型: {model_path}")
                    else:
                        print(f"[ERROR] 保存失败: {model_name}")

                except Exception as e:
                    print(f"[ERROR] 保存模型 {model_name} 失败: {e}")

        print(f"\n成功保存: {len(self.saved_models)} 个模型")
        return len(self.saved_models) > 0

    def run_cpu_training(self, models: List[str] = None, data_path: str = None,
                        sample_size: int = 50000, model_dir: str = "models") -> bool:
        """运行CPU训练流程"""
        print("CPU环境预测模型训练")
        print("=" * 40)

        try:
            # 1. 数据加载
            if not self.load_data(data_path, sample_size):
                return False

            # 2. 预测器初始化
            if not self.initialize_predictor():
                return False

            # 3. 模型训练
            if not self.train_models():
                return False

            # 5. 模型保存
            if not self.save_models(model_dir):
                return False

            # 6. 生成摘要
            self.generate_summary()

            print("\nCPU训练完成!")
            return True

        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            logger.error(f"Training process failed: {e}")
            return False

    def generate_summary(self):
        """生成训练摘要"""
        print("\n训练摘要")
        print("-" * 30)

        print(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"设备: CPU")
        print(f"数据规模: {self.data.shape[0]:,} 条记录" if self.data is not None else "数据规模: Unknown")

        if self.training_results:
            successful_count = len([r for r in self.training_results.values() if r.success])
            total_count = len(self.training_results)

            print(f"\n训练结果:")
            print(f"成功训练: {successful_count}/{total_count} 个模型")
            print(f"成功率: {successful_count/total_count*100:.1f}%")

            # 最佳模型
            best_model = None
            best_loss = float('inf')
            for model_name, result in self.training_results.items():
                if result.success and result.final_loss < best_loss:
                    best_loss = result.final_loss
                    best_model = model_name

            if best_model:
                print(f"最佳模型: {best_model} (损失: {best_loss:.4f})")

        if self.saved_models:
            print(f"\n保存的模型:")
            for model_name, model_path in self.saved_models:
                file_size = os.path.getsize(model_path) / 1024**2  # MB
                print(f"  {model_name}: {model_path} ({file_size:.1f} MB)")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='CPU环境预测模型训练')
    parser.add_argument('--models', nargs='+',
                       choices=['lstm', 'dlinear'],
                       default=['lstm', 'dlinear'],
                       help='要训练的模型 (默认: lstm dlinear)')
    parser.add_argument('--data-path', type=str,
                       help='数据文件路径')
    parser.add_argument('--sample-size', type=int, default=50000,
                       help='数据采样大小 (默认: 50000)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='模型保存目录 (默认: models)')

    args = parser.parse_args()

    # 创建CPU训练器
    trainer = CPUTrainer()

    # 运行训练
    success = trainer.run_cpu_training(
        models=args.models,
        data_path=args.data_path,
        sample_size=args.sample_size,
        model_dir=args.model_dir
    )

    if success:
        print("\n[OK] CPU训练成功完成!")
        exit(0)
    else:
        print("\n[ERROR] CPU训练失败!")
        exit(1)


if __name__ == "__main__":
    main()