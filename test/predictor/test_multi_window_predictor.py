#!/usr/bin/env python3
"""
多窗口预测器测试脚本

测试MultiWindowPredictor的功能，验证扩展的预测窗口是否正常工作。
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from predictor.multi_window_predictor import MultiWindowPredictor, create_multi_window_predictor
from predictor.workload_predictor import PredictionHorizon, ModelType

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data():
    """加载测试数据"""
    try:
        # 尝试加载清洗后的数据
        data_path = "dataset/BurstGPT_1_cleaned.csv"
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, nrows=500)
            logger.info(f"加载清洗后的数据: {data.shape}")
        else:
            # 回退到原始数据
            data_path = "dataset/BurstGPT_1.csv"
            data = pd.read_csv(data_path, nrows=500)
            logger.info(f"加载原始数据: {data.shape}")

        # 确保必要的列存在
        required_columns = ['Timestamp', 'Request tokens', 'Response tokens']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            logger.error(f"缺少必要的列: {missing_columns}")
            return None

        return data

    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return None

def test_prediction_horizons():
    """测试预测窗口扩展"""
    logger.info("=== 测试预测窗口扩展 ===")

    # 检查PredictionHorizon枚举
    horizons = list(PredictionHorizon)
    logger.info(f"支持的预测窗口: {[h.value for h in horizons]}")

    # 验证窗口配置
    expected_horizons = ['short_term', 'medium_term', 'long_term', 'extended_term']
    actual_horizons = [h.value for h in horizons]

    for expected in expected_horizons:
        if expected in actual_horizons:
            logger.info(f"✓ 预测窗口 {expected} 已实现")
        else:
            logger.error(f"✗ 预测窗口 {expected} 缺失")

    return len(actual_horizons) == 4

def test_multi_window_predictor_creation():
    """测试多窗口预测器创建"""
    logger.info("=== 测试多窗口预测器创建 ===")

    try:
        # 使用工厂函数创建预测器
        predictor = create_multi_window_predictor()

        # 验证预测器属性
        assert hasattr(predictor, 'supported_horizons'), "缺少supported_horizons属性"
        assert hasattr(predictor, 'predictors'), "缺少predictors属性"
        assert hasattr(predictor, 'integration_weights'), "缺少integration_weights属性"

        logger.info(f"✓ 支持的预测窗口数量: {len(predictor.supported_horizons)}")
        logger.info(f"✓ 预测器数量: {len(predictor.predictors)}")
        logger.info(f"✓ 集成权重: {predictor.integration_weights}")

        return True

    except Exception as e:
        logger.error(f"✗ 创建多窗口预测器失败: {e}")
        return False

def test_multi_window_prediction():
    """测试多窗口预测功能"""
    logger.info("=== 测试多窗口预测功能 ===")

    # 加载测试数据
    data = load_test_data()
    if data is None:
        logger.error("✗ 无法加载测试数据")
        return False

    try:
        # 创建预测器
        predictor = create_multi_window_predictor()

        # 测试多窗口预测
        logger.info("开始多窗口预测...")
        results = predictor.predict_multi_window(
            historical_data=data,
            horizons=None,  # 使用所有支持的窗口
            integration_enabled=True
        )

        # 验证结果
        logger.info(f"✓ 预测结果数量: {len(results)}")

        for horizon, result in results.items():
            logger.info(f"  - {horizon.value}: {len(result.predicted_requests)} 个预测请求, 置信度: {result.confidence:.3f}")

        # 验证结果格式
        for horizon, result in results.items():
            assert hasattr(result, 'predicted_requests'), f"缺少predicted_requests属性: {horizon}"
            assert hasattr(result, 'confidence'), f"缺少confidence属性: {horizon}"
            assert hasattr(result, 'metadata'), f"缺少metadata属性: {horizon}"

        logger.info("✓ 多窗口预测测试通过")
        return True

    except Exception as e:
        logger.error(f"✗ 多窗口预测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_methods():
    """测试集成方法"""
    logger.info("=== 测试集成方法 ===")

    # 加载测试数据
    data = load_test_data()
    if data is None:
        logger.error("✗ 无法加载测试数据")
        return False

    integration_methods = ['weighted_average', 'voting', 'stacking']

    for method in integration_methods:
        try:
            # 创建使用特定集成方法的预测器
            predictor = MultiWindowPredictor(
                base_predictor_config={
                    'model_type': ModelType.LSTM,
                    'prediction_horizon': PredictionHorizon.MEDIUM_TERM,
                    'sequence_length': 30,
                    'device': 'cpu'
                },
                enable_multi_window=True,
                integration_method=method
            )

            # 进行预测
            results = predictor.predict_multi_window(
                historical_data=data,
                integration_enabled=True
            )

            logger.info(f"✓ 集成方法 {method} 测试通过")

        except Exception as e:
            logger.error(f"✗ 集成方法 {method} 测试失败: {e}")
            return False

    logger.info("✓ 所有集成方法测试通过")
    return True

def test_adaptive_window_selection():
    """测试自适应窗口选择"""
    logger.info("=== 测试自适应窗口选择 ===")

    # 加载测试数据
    data = load_test_data()
    if data is None:
        logger.error("✗ 无法加载测试数据")
        return False

    try:
        # 创建预测器
        predictor = create_multi_window_predictor()

        # 测试自适应窗口选择
        optimal_horizon = predictor.select_optimal_horizon(data)
        logger.info(f"✓ 选择的最优预测窗口: {optimal_horizon.value}")

        # 验证返回值类型
        assert isinstance(optimal_horizon, PredictionHorizon), "返回值类型错误"

        return True

    except Exception as e:
        logger.error(f"✗ 自适应窗口选择测试失败: {e}")
        return False

def test_performance_reporting():
    """测试性能报告"""
    logger.info("=== 测试性能报告 ===")

    # 加载测试数据
    data = load_test_data()
    if data is None:
        logger.error("✗ 无法加载测试数据")
        return False

    try:
        # 创建预测器
        predictor = create_multi_window_predictor()

        # 进行一些预测
        results = predictor.predict_multi_window(data)

        # 生成性能报告
        report = predictor.get_performance_report()

        logger.info("✓ 性能报告生成成功")
        logger.info(f"  - 总预测次数: {report['overall_stats']['total_predictions']}")
        logger.info(f"  - 平均延迟: {report['overall_stats']['average_latency']:.3f}s")

        # 验证报告结构
        assert 'overall_stats' in report, "缺少overall_stats"
        assert 'predictor_stats' in report, "缺少predictor_stats"
        assert 'horizon_performance' in report, "缺少horizon_performance"

        return True

    except Exception as e:
        logger.error(f"✗ 性能报告测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始多窗口预测器功能测试...")

    tests = [
        ("预测窗口扩展", test_prediction_horizons),
        ("多窗口预测器创建", test_multi_window_predictor_creation),
        ("多窗口预测功能", test_multi_window_prediction),
        ("集成方法", test_integration_methods),
        ("自适应窗口选择", test_adaptive_window_selection),
        ("性能报告", test_performance_reporting)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试: {test_name}")
        logger.info(f"{'='*50}")

        try:
            if test_func():
                logger.info(f"✓ {test_name} 测试通过")
                passed += 1
            else:
                logger.error(f"✗ {test_name} 测试失败")
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} 测试异常: {e}")
            failed += 1

    logger.info(f"\n{'='*50}")
    logger.info("测试总结")
    logger.info(f"{'='*50}")
    logger.info(f"通过: {passed}")
    logger.info(f"失败: {failed}")
    logger.info(f"总计: {passed + failed}")

    if failed == 0:
        logger.info("🎉 所有测试通过！多窗口预测器功能正常。")
        return True
    else:
        logger.error(f"❌ {failed} 个测试失败，需要修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)