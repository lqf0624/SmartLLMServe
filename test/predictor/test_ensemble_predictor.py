#!/usr/bin/env python3
"""
多模型集成预测器测试脚本

测试新的EnsemblePredictor类和WorkloadPredictor的集成功能。
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
from predictor.ensemble_predictor import EnsemblePredictor, WeightStrategy
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
    trend = 30 + 0.1 * t
    seasonal = 8 * np.sin(2 * np.pi * t / 80) + 4 * np.sin(2 * np.pi * t / 25)
    noise = np.random.normal(0, 3, size)

    input_toks = np.maximum(15, trend + seasonal + noise).astype(int)
    output_toks = (input_toks * np.random.uniform(2, 4, size)).astype(int)

    return pd.DataFrame({
        'arrival_time_ns': timestamps,
        'input_toks': input_toks,
        'output_toks': output_toks,
        'model_type': ['ChatGPT'] * size
    })


def test_ensemble_predictor_standalone():
    """测试独立的EnsemblePredictor"""
    logger.info("=== 测试独立EnsemblePredictor ===")

    try:
        # 创建测试数据
        test_data = create_test_data(400)

        # 测试不同的权重策略
        strategies = [
            WeightStrategy.STATIC,
            WeightStrategy.PERFORMANCE_BASED,
            WeightStrategy.DYNAMIC,
            WeightStrategy.ADAPTIVE
        ]

        results = {}

        for strategy in strategies:
            logger.info(f"测试权重策略: {strategy.value}")

            # 创建集成预测器
            ensemble_predictor = EnsemblePredictor(
                models=[ModelType.LSTM, ModelType.DLINEAR],
                weight_strategy=strategy,
                prediction_horizon=PredictionHorizon.SHORT_TERM,
                sequence_length=40,
                performance_window=50
            )

            # 训练模型
            training_results = ensemble_predictor.train(test_data, validation_split=0.2)
            logger.info(f"训练完成，权重: {training_results.get('ensemble_weights', {})}")

            # 进行预测
            historical_data = test_data.iloc[:-30]
            prediction_result = ensemble_predictor.predict(historical_data, steps=10)

            logger.info(f"预测完成，请求数量: {len(prediction_result.predicted_requests)}")
            logger.info(f"预测置信度: {prediction_result.confidence:.3f}")

            results[strategy.value] = {
                'training_results': training_results,
                'prediction_result': prediction_result,
                'ensemble_info': ensemble_predictor.get_ensemble_info()
            }

        # 对比不同策略的结果
        logger.info("\n策略对比分析:")
        for strategy_name, result in results.items():
            weights = result['ensemble_info']['weights']
            confidence = result['prediction_result'].confidence
            logger.info(f"  {strategy_name}: 权重={weights}, 置信度={confidence:.3f}")

        logger.info("✅ 独立EnsemblePredictor测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 独立EnsemblePredictor测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workload_predictor_ensemble():
    """测试WorkloadPredictor的集成功能"""
    logger.info("=== 测试WorkloadPredictor集成功能 ===")

    try:
        # 创建测试数据
        test_data = create_test_data(300)

        # 创建启用集成的WorkloadPredictor
        predictor = WorkloadPredictor(
            model_type=ModelType.ENSEMBLE,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=40,
            enable_ensemble=True,
            ensemble_strategy=WeightStrategy.PERFORMANCE_BASED
        )

        logger.info(f"WorkloadPredictor初始化完成，激活模型: {[m.value for m in predictor.active_models]}")

        # 训练模型
        logger.info("开始训练...")
        training_results = predictor.train(test_data, validation_split=0.2)

        logger.info("训练结果:")
        for model_name, result in training_results.items():
            if isinstance(result, dict):
                status = result.get('status', 'unknown')
                logger.info(f"  {model_name}: {status}")
                if 'ensemble_weights' in result:
                    logger.info(f"    权重: {result['ensemble_weights']}")

        # 进行预测
        logger.info("\n开始预测...")
        historical_data = test_data.iloc[:-20]
        prediction_result = predictor.predict(historical_data, steps=8)

        logger.info(f"预测完成:")
        logger.info(f"  请求数量: {len(prediction_result.predicted_requests)}")
        logger.info(f"  置信度: {prediction_result.confidence:.3f}")
        logger.info(f"  预测元数据: {list(prediction_result.prediction_metadata.keys())}")

        # 测试可视化
        logger.info("\n生成可视化...")
        visualizer = SimplePredictionVisualizer()
        metrics = visualizer.plot_prediction_comparison(
            test_data, prediction_result, "Ensemble Model",
            save_path="test_ensemble_prediction.png",
            show_plot=False
        )

        logger.info(f"基础指标: {metrics}")

        logger.info("✅ WorkloadPredictor集成功能测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ WorkloadPredictor集成功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_strategy_comparison():
    """测试多策略对比"""
    logger.info("=== 测试多策略对比 ===")

    try:
        # 创建测试数据
        test_data = create_test_data(350)

        strategies = [
            WeightStrategy.STATIC,
            WeightStrategy.PERFORMANCE_BASED,
            WeightStrategy.ADAPTIVE
        ]

        strategy_results = {}

        for strategy in strategies:
            logger.info(f"\n测试策略: {strategy.value}")

            # 创建WorkloadPredictor
            predictor = WorkloadPredictor(
                model_type=ModelType.ENSEMBLE,
                prediction_horizon=PredictionHorizon.SHORT_TERM,
                sequence_length=35,
                enable_ensemble=True,
                ensemble_strategy=strategy
            )

            # 训练
            training_results = predictor.train(test_data, validation_split=0.25)

            # 预测
            historical_data = test_data.iloc[:-25]
            prediction_result = predictor.predict(historical_data, steps=12)

            # 收集结果
            strategy_results[strategy.value] = {
                'training_results': training_results,
                'prediction_result': prediction_result,
                'active_models': predictor.active_models,
                'weights': predictor.ensemble_weights
            }

        # 生成对比报告
        logger.info("\n=== 多策略对比报告 ===")
        for strategy_name, result in strategy_results.items():
            weights = result['weights']
            confidence = result['prediction_result'].confidence
            request_count = len(result['prediction_result'].predicted_requests)

            logger.info(f"\n{strategy_name}:")
            logger.info(f"  活跃模型: {[m.value for m in result['active_models']]}")
            logger.info(f"  权重分配: {weights}")
            logger.info(f"  预测置信度: {confidence:.3f}")
            logger.info(f"  预测请求数: {request_count}")

        # 找出最佳策略
        best_strategy = max(strategy_results.keys(),
                          key=lambda x: strategy_results[x]['prediction_result'].confidence)
        logger.info(f"\n🏆 最佳策略: {best_strategy} (置信度: {strategy_results[best_strategy]['prediction_result'].confidence:.3f})")

        logger.info("✅ 多策略对比测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 多策略对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_files():
    """清理测试文件"""
    import os
    test_files = ["test_ensemble_prediction.png"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"清理测试文件: {file}")


def main():
    """主测试函数"""
    logger.info("开始多模型集成预测器测试...")

    try:
        # 运行所有测试
        test1 = test_ensemble_predictor_standalone()
        test2 = test_workload_predictor_ensemble()
        test3 = test_multi_strategy_comparison()

        if test1 and test2 and test3:
            logger.info("🎉 所有多模型集成预测器测试通过！")
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