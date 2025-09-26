#!/usr/bin/env python3
"""
简化版可视化模块测试脚本

测试简化的预测对比和基础指标功能。
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from predictor.workload_predictor import WorkloadPredictor, ModelType, PredictionHorizon, PredictionResult
from predictor.simple_visualization import SimplePredictionVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data(size: int = 500) -> pd.DataFrame:
    """创建测试数据"""
    logger.info(f"创建测试数据，大小: {size}")

    # 生成时间戳
    base_time = pd.Timestamp('2023-01-01')
    timestamps = [base_time + pd.Timedelta(milliseconds=i*100) for i in range(size)]

    # 生成有趋势和季节性的数据
    t = np.arange(size)
    trend = 30 + 0.05 * t
    seasonal = 8 * np.sin(2 * np.pi * t / 80) + 4 * np.sin(2 * np.pi * t / 25)
    noise = np.random.normal(0, 3, size)

    input_toks = np.maximum(10, trend + seasonal + noise).astype(int)
    output_toks = (input_toks * np.random.uniform(2, 4, size)).astype(int)

    return pd.DataFrame({
        'arrival_time_ns': timestamps,
        'input_toks': input_toks,
        'output_toks': output_toks,
        'model_type': ['ChatGPT'] * size
    })


def create_prediction_result(test_data: pd.DataFrame, num_predictions: int = 20) -> PredictionResult:
    """创建模拟预测结果"""
    logger.info(f"创建预测结果，预测数量: {num_predictions}")

    # 获取最后时间戳
    last_time = pd.to_datetime(test_data['arrival_time_ns'].iloc[-1])

    # 生成预测请求
    predicted_requests = []
    for i in range(num_predictions):
        pred_time = last_time + pd.Timedelta(seconds=i*5)
        predicted_requests.append({
            'request_id': f'pred_{i}',
            'arrival_time_ns': pred_time.value,
            'input_tokens': np.random.randint(40, 120),
            'output_tokens': np.random.randint(80, 250),
            'burst_pattern': 'steady',
            'model_type': 'ChatGPT',
            'request_type': 'short',
            'priority': 'medium',
            'estimated_compute_time': 0.08,
            'memory_requirement_mb': 80.0,
            'predicted_accuracy': 0.82
        })

    return PredictionResult(
        predicted_requests=predicted_requests,
        confidence=0.82,
        prediction_metadata={
            'model_type': 'Test Model',
            'prediction_horizon': 'short_term',
            'sequence_length': 50
        }
    )


def test_simple_visualization():
    """测试简化版可视化功能"""
    logger.info("=== 测试简化版可视化功能 ===")

    # 创建测试数据
    test_data = create_test_data(400)

    # 创建可视化器
    visualizer = SimplePredictionVisualizer()

    # 创建预测结果
    prediction_result = create_prediction_result(test_data, 15)

    try:
        # 测试基础预测对比
        logger.info("测试基础预测对比...")
        metrics = visualizer.plot_prediction_comparison(
            test_data, prediction_result, "Test Model",
            save_path="test_simple_prediction.png",
            show_plot=False
        )

        logger.info(f"基础指标: {metrics}")

        # 测试多模型对比
        logger.info("测试多模型对比...")
        prediction_results = {
            'model1': create_prediction_result(test_data, 15),
            'model2': create_prediction_result(test_data, 15)
        }

        model_names = {'model1': 'LSTM', 'model2': 'DLinear'}
        multi_metrics = visualizer.plot_multi_model_comparison(
            prediction_results, test_data, model_names,
            save_path="test_multi_model_comparison.png",
            show_plot=False
        )

        logger.info(f"多模型指标: {multi_metrics}")

        # 测试报告生成
        logger.info("测试报告生成...")
        report = visualizer.generate_simple_report(
            prediction_results, test_data, model_names,
            save_path="test_prediction_report.txt"
        )

        logger.info(f"报告生成完成，长度: {len(report)} 字符")

        logger.info("✅ 简化版可视化功能测试通过")

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_lstm_prediction():
    """测试LSTM预测器的简化可视化"""
    logger.info("=== 测试LSTM预测器简化可视化 ===")

    try:
        # 创建测试数据
        test_data = create_test_data(300)

        # 创建LSTM预测器
        predictor = WorkloadPredictor(
            model_type=ModelType.LSTM,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=50
        )

        # 快速训练
        logger.info("训练LSTM模型...")
        training_results = predictor.train(test_data, validation_split=0.2)
        logger.info(f"训练完成，最终损失: {training_results.get('final_val_loss', 'N/A')}")

        # 进行预测
        historical_data = test_data.iloc[:-20]
        prediction_result = predictor.predict(historical_data, steps=10)

        # 使用简化可视化
        visualizer = SimplePredictionVisualizer()
        metrics = visualizer.plot_prediction_comparison(
            test_data, prediction_result, "LSTM Model",
            save_path="test_lstm_prediction.png",
            show_plot=False
        )

        logger.info(f"LSTM预测指标: {metrics}")

        logger.info("✅ LSTM预测器简化可视化测试通过")

    except Exception as e:
        logger.error(f"❌ LSTM预测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def cleanup_test_files():
    """清理测试文件"""
    import os
    test_files = [
        "test_simple_prediction.png",
        "test_multi_model_comparison.png",
        "test_lstm_prediction.png",
        "test_prediction_report.txt"
    ]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"清理测试文件: {file}")


def main():
    """主测试函数"""
    logger.info("开始简化版可视化测试...")

    try:
        # 运行测试
        success1 = test_simple_visualization()
        success2 = test_lstm_prediction()

        if success1 and success2:
            logger.info("🎉 所有简化版可视化测试通过！")
            return True
        else:
            logger.error("❌ 部分测试失败")
            return False

    except Exception as e:
        logger.error(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理测试文件
        cleanup_test_files()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)