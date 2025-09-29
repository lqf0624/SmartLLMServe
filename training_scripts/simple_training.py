#!/usr/bin/env python3
"""
简化的训练脚本

专注于LSTM和DLinear模型的生产训练，移除了高级模型复杂性
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor.request_predictor import RequestPredictor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleTrainer:
    """简化的训练器"""

    def __init__(self):
        self.predictor = None
        self.data = None
        self.training_results = None

    def load_data(self, data_path: str = None, sample_size: int = 100000) -> bool:
        """加载训练数据"""
        print("加载数据")
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

            # 数据采样
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
            print(f"清理后数据: {data.shape[0]:,} 条记录")

            self.data = data
            return True

        except Exception as e:
            print(f"数据加载失败: {e}")
            logger.error(f"Data loading failed: {e}")
            return False

    def train_models(self) -> bool:
        """训练LSTM和DLinear模型"""
        print("\n模型训练")
        print("-" * 30)

        if self.data is None:
            print("错误: 数据未加载")
            return False

        print(f"开始训练 LSTM + DLinear 模型...")
        print(f"训练数据: {self.data.shape[0]:,} 条记录")

        try:
            # 创建预测器
            self.predictor = RequestPredictor()

            # 训练模型
            self.training_results = self.predictor.train_models(self.data)

            # 显示训练结果
            print("\n训练结果:")
            successful_count = 0

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
                        successful_count += 1
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
                        successful_count += 1
                    else:
                        error_msg = getattr(result, 'error_message', '未知错误') if hasattr(result, 'error_message') else '训练失败'
                        print(f"[ERROR] {model_name}: {error_msg}")

            print(f"\n成功训练: {successful_count}/{len(self.training_results)} 个模型")
            return successful_count > 0

        except Exception as e:
            print(f"训练过程失败: {e}")
            logger.error(f"Training failed: {e}")
            return False

    def test_prediction(self) -> bool:
        """测试预测功能"""
        print("\n测试预测")
        print("-" * 30)

        if self.predictor is None or self.data is None:
            print("错误: 预测器或数据未初始化")
            return False

        try:
            # 准备历史数据
            historical_data = self.data.iloc[:min(1000, len(self.data))]

            # 进行预测
            prediction = self.predictor.predict_next_request(historical_data)

            if 'predictions' in prediction:
                print("预测结果:")
                for model_name, pred in prediction['predictions'].items():
                    if 'error' not in pred:
                        print(f"  {model_name}:")
                        print(f"    时间戳: {pred.get('timestamp', 'N/A')}")
                        print(f"    请求tokens: {pred.get('request_tokens', 'N/A')}")
                        print(f"    响应tokens: {pred.get('response_tokens', 'N/A')}")
                        if 'confidence' in pred:
                            print(f"    置信度: {pred['confidence']:.3f}")
                    else:
                        print(f"  {model_name}: 预测失败")
                return True
            else:
                print("预测失败")
                return False

        except Exception as e:
            print(f"预测测试失败: {e}")
            logger.error(f"Prediction test failed: {e}")
            return False

    def run_simple_training(self, data_path: str = None, sample_size: int = 100000) -> bool:
        """运行简化训练流程"""
        print("简化预测模型训练")
        print("=" * 40)

        try:
            # 1. 数据加载
            if not self.load_data(data_path, sample_size):
                return False

            # 2. 模型训练
            if not self.train_models():
                return False

            # 3. 测试预测
            if not self.test_prediction():
                return False

            # 4. 生成摘要
            self.generate_summary()

            print("\n简化训练完成!")
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
        print(f"数据规模: {self.data.shape[0]:,} 条记录" if self.data is not None else "数据规模: Unknown")

        if self.training_results:
            successful_count = 0
            total_count = len(self.training_results)

            for result in self.training_results.values():
                if isinstance(result, dict):
                    if result.get('success') or result.get('status') == 'completed':
                        successful_count += 1
                else:
                    if hasattr(result, 'success') and result.success:
                        successful_count += 1

            print(f"成功训练: {successful_count}/{total_count} 个模型")
            print(f"成功率: {successful_count/total_count*100:.1f}%")

            # 显示模型性能
            for model_name, result in self.training_results.items():
                if isinstance(result, dict):
                    if result.get('success') or result.get('status') == 'completed':
                        if 'train_loss' in result:
                            print(f"{model_name}: 训练损失 = {result['train_loss']:.4f}")
                        if 'val_loss' in result:
                            print(f"{model_name}: 验证损失 = {result['val_loss']:.4f}")
                else:
                    if hasattr(result, 'success') and result.success:
                        if hasattr(result, 'final_loss'):
                            print(f"{model_name}: 损失 = {result.final_loss:.4f}")

        print("\n使用方法:")
        print("1. 直接使用训练好的预测器:")
        print("   predictor = RequestPredictor()")
        print("   prediction = predictor.predict_next_request(historical_data)")
        print("2. 预测结果包含LSTM和DLinear的独立预测")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='简化预测模型训练')
    parser.add_argument('--data-path', type=str,
                       help='数据文件路径')
    parser.add_argument('--sample-size', type=int, default=100000,
                       help='数据采样大小 (默认: 100000)')

    args = parser.parse_args()

    # 创建简化训练器
    trainer = SimpleTrainer()

    # 运行训练
    success = trainer.run_simple_training(
        data_path=args.data_path,
        sample_size=args.sample_size
    )

    if success:
        print("\n[OK] 简化训练成功完成!")
        exit(0)
    else:
        print("\n[ERROR] 简化训练失败!")
        exit(1)


if __name__ == "__main__":
    main()