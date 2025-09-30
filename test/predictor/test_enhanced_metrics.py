#!/usr/bin/env python3
"""
增强评估指标测试脚本

测试新添加的全面评估指标功能，包括高级指标、统计检验、业务指标和鲁棒性分析。
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
from predictor.visualization import PredictionVisualizer
from predictor.dlinear_predictor_adapter import create_dlinear_predictor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data(size: int = 1000) -> pd.DataFrame:
    """创建测试数据"""
    logger.info(f"创建测试数据，大小: {size}")

    # 生成时间戳
    base_time = pd.Timestamp('2023-01-01')
    timestamps = [base_time + pd.Timedelta(milliseconds=i*100) for i in range(size)]

    # 生成有趋势和季节性的数据
    t = np.arange(size)
    trend = 50 + 0.1 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 100) + 5 * np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, 5, size)

    input_toks = np.maximum(10, trend + seasonal + noise).astype(int)
    output_toks = (input_toks * np.random.uniform(2, 4, size)).astype(int)

    # 添加一些突发
    burst_positions = np.random.choice(size, size//50, replace=False)
    input_toks[burst_positions] = np.random.randint(200, 500, len(burst_positions))
    output_toks[burst_positions] = input_toks[burst_positions] * np.random.randint(2, 5, len(burst_positions))

    return pd.DataFrame({
        'arrival_time_ns': timestamps,
        'input_toks': input_toks,
        'output_toks': output_toks,
        'model_type': ['ChatGPT'] * size
    })


def test_comprehensive_metrics():
    """测试全面评估指标功能"""
    logger.info("=== 测试全面评估指标功能 ===")

    # 创建测试数据
    test_data = create_test_data(800)

    # 创建可视化器
    visualizer = PredictionVisualizer()

    # 创建一个简单的预测结果（模拟）

    # 模拟预测请求
    predicted_requests = []
    base_time = pd.Timestamp('2023-01-01 00:01:00')
    for i in range(20):
        req_time = base_time + pd.Timedelta(seconds=i*5)
        predicted_requests.append({
            'request_id': f'pred_{i}',
            'arrival_time_ns': req_time.value,
            'input_tokens': np.random.randint(50, 150),
            'output_tokens': np.random.randint(100, 300),
            'burst_pattern': 'steady',
            'model_type': 'ChatGPT',
            'request_type': 'short',
            'priority': 'medium',
            'estimated_compute_time': 0.1,
            'memory_requirement_mb': 100.0,
            'predicted_accuracy': 0.85
        })

    # 创建预测结果
    prediction_result = PredictionResult(
        predicted_requests=predicted_requests,
        confidence=0.85,
        prediction_metadata={
            'model_type': 'LSTM',
            'prediction_horizon': 'short_term',
            'sequence_length': 50
        }
    )

    # 测试全面指标计算
    logger.info("测试全面指标计算...")
    comprehensive_metrics = visualizer.calculate_comprehensive_metrics(
        test_data, prediction_result
    )

    logger.info(f"基础指标: {comprehensive_metrics['basic_metrics']}")
    logger.info(f"高级指标: {comprehensive_metrics['advanced_metrics']}")
    logger.info(f"业务指标: {comprehensive_metrics['business_metrics']}")
    logger.info(f"鲁棒性指标: {comprehensive_metrics['robustness_metrics']}")
    logger.info(f"总体评分: {comprehensive_metrics['overall_score']:.3f}")

    # 测试指标分析图
    logger.info("测试指标分析图...")
    visualizer.plot_comprehensive_metrics_analysis(
        test_data, prediction_result, "Test Model",
        save_path="test_comprehensive_metrics.png",
        show_plot=False
    )

    # 测试报告生成
    logger.info("测试报告生成...")
    report = visualizer.generate_metrics_report(test_data, prediction_result, "Test Model")

    logger.info(f"性能等级: {report['summary']['performance_grade']}")
    logger.info(f"置信水平: {report['summary']['confidence_level']:.2f}")
    logger.info("改进建议:")
    for rec in report['recommendations']:
        logger.info(f"  - {rec}")

    logger.info("✅ 全面评估指标功能测试通过")


def test_lstm_dlinear_comparison():
    """测试LSTM vs DLinear的全面对比"""
    logger.info("=== 测试LSTM vs DLinear全面对比 ===")

    # 创建测试数据
    test_data = create_test_data(600)

    models = {
        'LSTM': ModelType.LSTM,
        'DLinear': ModelType.DLINEAR
    }

    results = {}

    for model_name, model_type in models.items():
        logger.info(f"测试 {model_name} 模型...")

        try:
            # 创建预测器
            predictor = WorkloadPredictor(
                model_type=model_type,
                prediction_horizon=PredictionHorizon.SHORT_TERM,
                sequence_length=50
            )

            # 快速训练
            training_results = predictor.train(test_data, validation_split=0.2)
            logger.info(f"{model_name} 训练完成")

            # 预测
            historical_data = test_data.iloc[:-30]
            prediction_result = predictor.predict(historical_data, steps=10)

            # 计算全面指标
            visualizer = PredictionVisualizer()
            comprehensive_metrics = visualizer.calculate_comprehensive_metrics(
                test_data, prediction_result
            )

            results[model_name] = {
                'metrics': comprehensive_metrics,
                'prediction_result': prediction_result,
                'training_results': training_results
            }

            logger.info(f"{model_name} 总体评分: {comprehensive_metrics['overall_score']:.3f}")

        except Exception as e:
            logger.error(f"{model_name} 测试失败: {e}")
            continue

    # 对比分析
    if len(results) >= 2:
        logger.info("\n模型对比分析:")
        for model_name, result in results.items():
            metrics = result['metrics']
            logger.info(f"\n{model_name}:")
            logger.info(f"  总体评分: {metrics['overall_score']:.3f}")
            logger.info(f"  基础指标 MAE: {metrics['basic_metrics'].get('MAE', 'N/A'):.2f}")
            logger.info(f"  高级指标 SMAPE: {metrics['advanced_metrics'].get('SMAPE', 'N/A'):.1f}%")
            logger.info(f"  业务指标 资源效率: {metrics['business_metrics'].get('Resource_Efficiency', 'N/A'):.2f}")
            logger.info(f"  鲁棒性评分: {metrics['robustness_metrics'].get('Overall_Robustness', 'N/A'):.2f}")

        # 找出最佳模型
        best_model = max(results.keys(), key=lambda x: results[x]['metrics']['overall_score'])
        logger.info(f"\n🏆 最佳模型: {best_model} (评分: {results[best_model]['metrics']['overall_score']:.3f})")

    logger.info("✅ LSTM vs DLinear 对比测试完成")


def test_metrics_visualization():
    """测试指标可视化功能"""
    logger.info("=== 测试指标可视化功能 ===")

    # 创建测试数据
    test_data = create_test_data(500)

    # 创建可视化器
    visualizer = PredictionVisualizer()

    # 创建预测结果
    predicted_requests = []
    base_time = pd.Timestamp('2023-01-01 00:01:00')
    for i in range(15):
        req_time = base_time + pd.Timedelta(seconds=i*3)
        predicted_requests.append({
            'request_id': f'pred_{i}',
            'arrival_time_ns': req_time.value,
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

    prediction_result = PredictionResult(
        predicted_requests=predicted_requests,
        confidence=0.82,
        prediction_metadata={
            'model_type': 'Test Model',
            'prediction_horizon': 'short_term',
            'sequence_length': 50
        }
    )

    # 测试各种可视化
    logger.info("测试全面指标分析图...")
    visualizer.plot_comprehensive_metrics_analysis(
        test_data, prediction_result, "Visualization Test",
        save_path="test_metrics_visualization.png",
        show_plot=False
    )

    # 测试增强的真实vs预测对比
    logger.info("测试增强的真实vs预测对比...")
    visualizer.plot_enhanced_actual_vs_predicted(
        test_data, prediction_result, "Enhanced Comparison",
        save_path="test_enhanced_comparison.png",
        show_plot=False
    )

    logger.info("✅ 指标可视化功能测试通过")


def cleanup_test_files():
    """清理测试文件"""
    import os
    test_files = [
        "test_comprehensive_metrics.png",
        "test_metrics_visualization.png",
        "test_enhanced_comparison.png"
    ]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"清理测试文件: {file}")


def main():
    """主测试函数"""
    logger.info("开始增强评估指标测试...")

    try:
        # 运行所有测试
        test_comprehensive_metrics()
        test_lstm_dlinear_comparison()
        test_metrics_visualization()

        logger.info("🎉 所有增强评估指标测试通过！")

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理测试文件
        cleanup_test_files()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)