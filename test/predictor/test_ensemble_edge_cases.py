#!/usr/bin/env python3
"""
集成预测器边缘情况和错误处理测试

测试EnsemblePredictor在各种异常情况下的稳定性和错误处理能力。
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import traceback
from pathlib import Path
from typing import List, Dict, Any

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


def create_basic_test_data(size: int = 100) -> pd.DataFrame:
    """创建基础测试数据"""
    base_time = pd.Timestamp('2023-01-01')
    timestamps = [base_time + pd.Timedelta(milliseconds=i*100) for i in range(size)]

    # 生成简单的测试数据
    input_toks = np.random.poisson(100, size).astype(int)
    output_toks = (input_toks * np.random.uniform(2, 4, size)).astype(int)

    return pd.DataFrame({
        'arrival_time_ns': timestamps,
        'input_toks': input_toks,
        'output_toks': output_toks,
        'model_type': ['ChatGPT'] * size
    })


def test_ensemble_empty_models():
    """测试空模型列表的处理"""
    logger.info("=== 测试空模型列表处理 ===")

    try:
        # 测试空模型列表
        ensemble_predictor = EnsemblePredictor(
            models=[],
            weight_strategy=WeightStrategy.STATIC,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=30
        )

        # 应该能够处理空模型情况
        test_data = create_basic_test_data(50)

        # 训练应该失败但不会崩溃
        try:
            training_results = ensemble_predictor.train(test_data)
            logger.info("空模型训练结果: 有合理的错误处理")
        except Exception as e:
            logger.info(f"空模型训练失败（预期）: {e}")

        # 预测应该返回合成预测
        historical_data = test_data.iloc[:-10]
        prediction_result = ensemble_predictor.predict(historical_data, steps=5)

        logger.info(f"空模型预测结果:")
        logger.info(f"  请求数量: {len(prediction_result.predicted_requests)}")
        logger.info(f"  置信度: {prediction_result.confidence:.3f}")
        logger.info(f"  模型类型: {prediction_result.prediction_metadata.get('model_type', 'N/A')}")

        # 验证合成预测的有效性
        assert len(prediction_result.predicted_requests) > 0, "应该生成合成预测"
        assert prediction_result.confidence > 0, "应该有合理的置信度"

        logger.info("✅ 空模型列表处理测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 空模型列表处理测试失败: {e}")
        traceback.print_exc()
        return False


def test_ensemble_insufficient_data():
    """测试数据不足情况"""
    logger.info("=== 测试数据不足情况 ===")

    try:
        # 创建数据不足的情况
        test_data = create_basic_test_data(20)  # 非常少的数据

        ensemble_predictor = EnsemblePredictor(
            models=[ModelType.LSTM, ModelType.DLINEAR],
            weight_strategy=WeightStrategy.STATIC,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=30  # 序列长度大于数据量
        )

        # 训练应该处理数据不足
        try:
            training_results = ensemble_predictor.train(test_data, validation_split=0.2)
            logger.info("数据不足训练处理成功")
        except Exception as e:
            logger.info(f"数据不足训练失败（预期）: {e}")

        # 预测应该能够处理
        historical_data = test_data.iloc[:-5]
        prediction_result = ensemble_predictor.predict(historical_data, steps=10)

        logger.info(f"数据不足预测结果:")
        logger.info(f"  请求数量: {len(prediction_result.predicted_requests)}")
        logger.info(f"  置信度: {prediction_result.confidence:.3f}")

        # 验证结果的有效性
        assert prediction_result is not None, "应该返回预测结果"
        assert prediction_result.predicted_requests is not None, "应该有预测请求数组"

        logger.info("✅ 数据不足情况测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 数据不足情况测试失败: {e}")
        traceback.print_exc()
        return False


def test_ensemble_malformed_data():
    """测试异常数据处理"""
    logger.info("=== 测试异常数据处理 ===")

    try:
        # 创建异常数据
        test_data = create_basic_test_data(100)

        # 添加异常值
        test_data.loc[10, 'input_toks'] = -100  # 负值
        test_data.loc[20, 'output_toks'] = 0   # 零值
        test_data.loc[30, 'input_toks'] = 100000  # 极大值
        test_data.loc[40, 'arrival_time_ns'] = None  # 缺失时间戳

        ensemble_predictor = EnsemblePredictor(
            models=[ModelType.LSTM],
            weight_strategy=WeightStrategy.STATIC,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=30
        )

        # 训练应该能够处理异常数据
        training_results = ensemble_predictor.train(test_data, validation_split=0.2)
        logger.info("异常数据训练处理成功")

        # 预测应该能够处理
        historical_data = test_data.iloc[:-10]
        prediction_result = ensemble_predictor.predict(historical_data, steps=5)

        logger.info(f"异常数据处理结果:")
        logger.info(f"  请求数量: {len(prediction_result.predicted_requests)}")
        logger.info(f"  置信度: {prediction_result.confidence:.3f}")

        # 验证结果的有效性
        assert prediction_result is not None, "应该返回预测结果"
        assert len(prediction_result.predicted_requests) > 0, "应该有预测请求"

        logger.info("✅ 异常数据处理测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 异常数据处理测试失败: {e}")
        traceback.print_exc()
        return False


def test_ensemble_weight_extremes():
    """测试权重极端情况"""
    logger.info("=== 测试权重极端情况 ===")

    try:
        test_data = create_basic_test_data(150)

        # 测试极端权重分配
        extreme_weights = {
            ModelType.LSTM: 0.0,    # LSTM权重为0
            ModelType.DLINEAR: 1.0  # DLinear权重为1
        }

        ensemble_predictor = EnsemblePredictor(
            models=[ModelType.LSTM, ModelType.DLINEAR],
            weight_strategy=WeightStrategy.STATIC,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=30
        )

        # 设置极端权重
        ensemble_predictor.weights = extreme_weights

        # 训练
        training_results = ensemble_predictor.train(test_data, validation_split=0.2)
        logger.info("极端权重训练成功")

        # 预测
        historical_data = test_data.iloc[:-10]
        prediction_result = ensemble_predictor.predict(historical_data, strpseps=5)

        logger.info(f"极端权重预测结果:")
        logger.info(f"  请求数量: {len(prediction_result.predicted_requests)}")
        logger.info(f"  置信度: {prediction_result.confidence:.3f}")
        logger.info(f"  使用的权重: {ensemble_predictor.weights}")

        # 验证权重分配
        assert ModelType.DLINEAR in ensemble_predictor.weights, "应该包含DLinear权重"
        assert ensemble_predictor.weights[ModelType.DLINEAR] > 0, "DLinear权重应该大于0"

        logger.info("✅ 权重极端情况测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 权重极端情况测试失败: {e}")
        traceback.print_exc()
        return False


def test_ensemble_concurrent_predictions():
    """测试并发预测"""
    logger.info("=== 测试并发预测 ===")

    try:
        import threading
        import time

        test_data = create_basic_test_data(200)

        ensemble_predictor = EnsemblePredictor(
            models=[ModelType.LSTM, ModelType.DLINEAR],
            weight_strategy=WeightStrategy.STATIC,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=30
        )

        # 训练
        training_results = ensemble_predictor.train(test_data, validation_split=0.2)
        logger.info("并发预测训练成功")

        # 并发预测测试
        results = []
        errors = []

        def predict_worker(worker_id):
            try:
                historical_data = test_data.iloc[:-15]
                prediction_result = ensemble_predictor.predict(historical_data, steps=5)
                results.append({
                    'worker_id': worker_id,
                    'prediction_count': len(prediction_result.predicted_requests),
                    'confidence': prediction_result.confidence
                })
                logger.info(f"Worker {worker_id} 预测完成")
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # 创建多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=predict_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        logger.info(f"并发预测结果:")
        for result in results:
            logger.info(f"  Worker {result['worker_id']}: {result['prediction_count']} requests, confidence={result['confidence']:.3f}")

        if errors:
            logger.warning(f"并发预测错误: {errors}")

        # 验证并发结果
        assert len(results) == 3, "应该有3个预测结果"
        assert len(errors) == 0, "不应该有错误"

        logger.info("✅ 并发预测测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 并发预测测试失败: {e}")
        traceback.print_exc()
        return False


def test_ensemble_model_failure():
    """测试模型失败情况"""
    logger.info("=== 测试模型失败情况 ===")

    try:
        test_data = create_basic_test_data(150)

        ensemble_predictor = EnsemblePredictor(
            models=[ModelType.LSTM, ModelType.DLINEAR],
            weight_strategy=WeightStrategy.STATIC,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=30
        )

        # 训练
        training_results = ensemble_predictor.train(test_data, validation_split=0.2)
        logger.info("模型失败测试训练成功")

        # 模拟模型失败 - 破坏一个模型的权重
        original_weights = ensemble_predictor.weights.copy()
        ensemble_predictor.weights[ModelType.LSTM] = 0.0  # 让LSTM失效

        # 预测应该仍然能够工作
        historical_data = test_data.iloc[:-10]
        prediction_result = ensemble_predictor.predict(historical_data, steps=5)

        logger.info(f"模型失败预测结果:")
        logger.info(f"  请求数量: {len(prediction_result.predicted_requests)}")
        logger.info(f"  置信度: {prediction_result.confidence:.3f}")
        logger.info(f"  原始权重: {original_weights}")
        logger.info(f"  当前权重: {ensemble_predictor.weights}")

        # 验证预测仍然有效
        assert prediction_result is not None, "应该返回预测结果"
        assert len(prediction_result.predicted_requests) > 0, "应该有预测请求"

        logger.info("✅ 模型失败情况测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 模型失败情况测试失败: {e}")
        traceback.print_exc()
        return False


def test_ensemble_memory_usage():
    """测试内存使用情况"""
    logger.info("=== 测试内存使用情况 ===")

    try:
        import psutil
        import os

        # 获取初始内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建较大的测试数据
        test_data = create_basic_test_data(1000)

        ensemble_predictor = EnsemblePredictor(
            models=[ModelType.LSTM, ModelType.DLINEAR],
            weight_strategy=WeightStrategy.STATIC,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=50
        )

        # 训练
        training_results = ensemble_predictor.train(test_data, validation_split=0.2)

        # 获取训练后内存使用
        training_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 预测
        historical_data = test_data.iloc[:-50]
        prediction_result = ensemble_predictor.predict(historical_data, steps=20)

        # 获取预测后内存使用
        prediction_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_increase = prediction_memory - initial_memory

        logger.info(f"内存使用情况:")
        logger.info(f"  初始内存: {initial_memory:.2f} MB")
        logger.info(f"  训练后内存: {training_memory:.2f} MB")
        logger.info(f"  预测后内存: {prediction_memory:.2f} MB")
        logger.info(f"  内存增长: {memory_increase:.2f} MB")

        # 验证内存使用是否合理
        assert memory_increase < 500, f"内存增长过多: {memory_increase} MB"

        logger.info("✅ 内存使用情况测试通过")
        return True

    except Exception as e:
        logger.error(f"❌ 内存使用情况测试失败: {e}")
        traceback.print_exc()
        return False


def cleanup_test_files():
    """清理测试文件"""
    test_files = ["test_ensemble_edge_*.png"]
    for pattern in test_files:
        import glob
        for file in glob.glob(pattern):
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"清理测试文件: {file}")


def main():
    """主测试函数"""
    logger.info("开始集成预测器边缘情况和错误处理测试...")

    test_functions = [
        ("空模型列表处理", test_ensemble_empty_models),
        ("数据不足情况", test_ensemble_insufficient_data),
        ("异常数据处理", test_ensemble_malformed_data),
        ("权重极端情况", test_ensemble_weight_extremes),
        ("并发预测", test_ensemble_concurrent_predictions),
        ("模型失败情况", test_ensemble_model_failure),
        ("内存使用情况", test_ensemble_memory_usage)
    ]

    passed_tests = 0
    total_tests = len(test_functions)

    try:
        for test_name, test_func in test_functions:
            logger.info(f"\n{'='*60}")
            logger.info(f"开始测试: {test_name}")
            logger.info(f"{'='*60}")

            try:
                if test_func():
                    logger.info(f"✅ {test_name} 测试通过")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} 测试失败")
            except Exception as e:
                logger.error(f"❌ {test_name} 测试崩溃: {e}")
                traceback.print_exc()

        logger.info(f"\n{'='*60}")
        logger.info("测试总结")
        logger.info(f"{'='*60}")
        logger.info(f"通过测试: {passed_tests}/{total_tests}")
        logger.info(f"通过率: {passed_tests/total_tests*100:.1f}%")

        if passed_tests == total_tests:
            logger.info("🎉 所有边缘情况测试通过！")
            return True
        else:
            logger.error("❌ 部分测试失败")
            return False

    except Exception as e:
        logger.error(f"❌ 测试过程出错: {e}")
        traceback.print_exc()
        return False

    finally:
        cleanup_test_files()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)