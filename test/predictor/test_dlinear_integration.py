#!/usr/bin/env python3
"""
DLinear集成测试脚本

测试DLinear模型与工作负载预测器的完整集成，包括训练、预测和评估功能。
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from predictor.workload_predictor import WorkloadPredictor, ModelType, PredictionHorizon
from predictor.dlinear_predictor_adapter import create_dlinear_predictor
from predictor.data_preprocessor import DataPreprocessor

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


def test_dlinear_adapter():
    """测试DLinear适配器"""
    logger.info("=== 测试DLinear适配器 ===")

    # 创建测试数据
    test_data = create_test_data(500)

    # 创建DLinear适配器
    adapter = create_dlinear_predictor(
        prediction_horizon=PredictionHorizon.SHORT_TERM,
        sequence_length=50
    )

    # 测试训练
    logger.info("测试DLinear训练...")
    training_result = adapter.train(
        data=test_data,
        epochs=20,  # 快速测试
        verbose=False
    )

    assert training_result['status'] == 'completed', "DLinear训练失败"
    logger.info(f"DLinear训练成功，最佳验证损失: {training_result['best_val_loss']:.6f}")

    # 测试预测
    logger.info("测试DLinear预测...")
    historical_data = test_data.iloc[:-30]
    prediction_result = adapter.predict(historical_data, steps=10)

    assert len(prediction_result.predicted_requests) == 10, f"预测请求数量错误: {len(prediction_result.predicted_requests)}"
    logger.info(f"DLinear预测成功，预测了 {len(prediction_result.predicted_requests)} 个请求")

    # 测试评估
    logger.info("测试DLinear评估...")
    metrics = adapter.evaluate(test_data)
    logger.info(f"DLinear评估指标: {metrics}")

    # 测试模型信息
    model_info = adapter.get_model_info()
    logger.info(f"DLinear模型信息: {model_info['model_type']}, {model_info['prediction_horizon']}")

    # 测试保存和加载
    logger.info("测试保存和加载...")
    adapter.save_model("test_dlinear_adapter")
    loaded_adapter = create_dlinear_predictor()
    loaded_adapter.load_model("test_dlinear_adapter")

    logger.info("✅ DLinear适配器测试通过")


def test_workload_predictor_dlinear():
    """测试工作负载预测器中的DLinear集成"""
    logger.info("=== 测试工作负载预测器DLinear集成 ===")

    # 创建测试数据
    test_data = create_test_data(800)

    # 创建使用DLinear的工作负载预测器
    predictor = WorkloadPredictor(
        model_type=ModelType.DLINEAR,
        prediction_horizon=PredictionHorizon.SHORT_TERM,
        sequence_length=50
    )

    # 测试训练
    logger.info("测试工作负载预测器DLinear训练...")
    training_results = predictor.train(test_data, validation_split=0.2)

    assert 'dlinear' in training_results, "DLinear训练结果缺失"
    assert training_results['dlinear']['status'] == 'completed', "DLinear训练失败"
    logger.info(f"工作负载预测器DLinear训练成功: {training_results['dlinear']['status']}")

    # 测试预测
    logger.info("测试工作负载预测器DLinear预测...")
    historical_data = test_data.iloc[:-50]
    prediction_result = predictor.predict(historical_data, steps=10)

    assert len(prediction_result.predicted_requests) == 10, f"预测步数错误: {len(prediction_result.predicted_requests)}"
    logger.info(f"工作负载预测器DLinear预测成功，预测了 {len(prediction_result.predicted_requests)} 个请求")

    # 测试模型信息
    model_info = predictor.get_model_info()
    logger.info(f"工作负载预测器模型信息: {model_info['model_type']}")

    logger.info("✅ 工作负载预测器DLinear集成测试通过")


def test_ensemble_dlinear():
    """测试集成模式下的DLinear"""
    logger.info("=== 测试集成模式DLinear ===")

    # 创建测试数据
    test_data = create_test_data(600)

    # 创建集成预测器
    predictor = WorkloadPredictor(
        model_type=ModelType.ENSEMBLE,
        prediction_horizon=PredictionHorizon.MEDIUM_TERM,
        sequence_length=50
    )

    # 检查是否包含DLinear模型
    assert ModelType.DLINEAR in predictor.models, "集成模式未包含DLinear模型"
    assert ModelType.DLINEAR in predictor.active_models, "DLinear模型未激活"
    logger.info("集成模式成功包含DLinear模型")

    # 测试训练
    logger.info("测试集成模式训练...")
    training_results = predictor.train(test_data, validation_split=0.2)

    assert 'dlinear' in training_results, "集成训练结果中缺少DLinear"
    assert 'lstm' in training_results, "集成训练结果中缺少LSTM"
    logger.info("集成模式训练成功")

    # 测试预测
    logger.info("测试集成模式预测...")
    historical_data = test_data.iloc[:-50]
    prediction_result = predictor.predict(historical_data, steps=15)

    assert len(prediction_result.predicted_requests) > 0, "集成预测未产生结果"
    logger.info(f"集成模式预测成功，预测了 {len(prediction_result.predicted_requests)} 个请求")

    logger.info("✅ 集成模式DLinear测试通过")


def test_multi_horizon_dlinear():
    """测试多时间范围DLinear预测"""
    logger.info("=== 测试多时间范围DLinear预测 ===")

    # 创建测试数据
    test_data = create_test_data(1000)

    horizons = [
        PredictionHorizon.SHORT_TERM,    # 10步
        PredictionHorizon.MEDIUM_TERM,  # 30步
        PredictionHorizon.LONG_TERM,     # 60步
    ]

    for horizon in horizons:
        logger.info(f"测试 {horizon.value} DLinear预测...")

        # 创建DLinear预测器
        predictor = WorkloadPredictor(
            model_type=ModelType.DLINEAR,
            prediction_horizon=horizon,
            sequence_length=50
        )

        # 快速训练
        training_results = predictor.train(test_data, validation_split=0.2)
        assert training_results['dlinear']['status'] == 'completed', f"{horizon.value} 训练失败"

        # 预测
        historical_data = test_data.iloc[:-50]
        prediction_result = predictor.predict(historical_data)

        expected_steps = predictor.horizon_config[horizon]['steps']
        assert len(prediction_result.predicted_requests) == expected_steps, f"{horizon.value} 预测步数错误"

        logger.info(f"✅ {horizon.value} DLinear预测测试通过")

    logger.info("✅ 多时间范围DLinear预测测试通过")


def test_dlinear_vs_lstm():
    """对比DLinear和LSTM的性能"""
    logger.info("=== 对比DLinear和LSTM性能 ===")

    # 创建测试数据
    test_data = create_test_data(800)

    models = [ModelType.DLINEAR, ModelType.LSTM]
    results = {}

    for model_type in models:
        logger.info(f"测试 {model_type.value} 模型...")

        # 创建预测器
        predictor = WorkloadPredictor(
            model_type=model_type,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=50
        )

        # 训练
        training_results = predictor.train(test_data, validation_split=0.2)
        train_loss = training_results[model_type.value].get('final_train_loss', float('inf'))
        val_loss = training_results[model_type.value].get('final_val_loss', float('inf'))

        # 预测
        historical_data = test_data.iloc[:-50]
        prediction_result = predictor.predict(historical_data)

        results[model_type.value] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'prediction_steps': len(prediction_result.predicted_requests),
            'training_time': training_results[model_type.value].get('epochs_trained', 0)
        }

        logger.info(f"{model_type.value}: 训练损失={train_loss:.6f}, 验证损失={val_loss:.6f}")

    # 打印对比结果
    logger.info("\n模型对比结果:")
    for model_name, result in results.items():
        logger.info(f"  {model_name}:")
        logger.info(f"    训练损失: {result['train_loss']:.6f}")
        logger.info(f"    验证损失: {result['val_loss']:.6f}")
        logger.info(f"    训练轮数: {result['training_time']}")
        logger.info(f"    预测步数: {result['prediction_steps']}")

    logger.info("✅ DLinear vs LSTM 对比测试完成")


def cleanup_test_files():
    """清理测试文件"""
    import shutil
    test_dirs = ["test_dlinear_adapter"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            logger.info(f"清理测试目录: {test_dir}")


def main():
    """主测试函数"""
    logger.info("开始DLinear集成测试...")

    try:
        # 运行所有测试
        test_dlinear_adapter()
        test_workload_predictor_dlinear()
        test_ensemble_dlinear()
        test_multi_horizon_dlinear()
        test_dlinear_vs_lstm()

        logger.info("🎉 所有DLinear集成测试通过！")

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