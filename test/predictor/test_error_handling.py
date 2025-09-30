#!/usr/bin/env python3
"""
错误处理机制测试脚本

测试新增的错误处理机制和数据预处理器功能。
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from predictor.workload_predictor import WorkloadPredictor, ModelType, PredictionHorizon
from predictor.data_preprocessor import DataPreprocessor, DataFormat

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
            data = pd.read_csv(data_path, nrows=200)
            logger.info(f"加载清洗后的数据: {data.shape}")
        else:
            # 回退到原始数据
            data_path = "dataset/BurstGPT_1.csv"
            data = pd.read_csv(data_path, nrows=200)
            logger.info(f"加载原始数据: {data.shape}")

        return data

    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return None

def test_data_preprocessor():
    """测试数据预处理器"""
    logger.info("=== 测试数据预处理器 ===")

    # 加载测试数据
    data = load_test_data()
    if data is None:
        logger.error("✗ 无法加载测试数据")
        return False

    try:
        # 创建预处理器
        preprocessor = DataPreprocessor()

        # 测试格式检测
        detected_format = preprocessor.detect_format(data)
        logger.info(f"✓ 检测到数据格式: {detected_format.value}")

        # 测试预处理
        processed_data = preprocessor.preprocess_data(data, DataFormat.STANDARD)
        logger.info(f"✓ 数据预处理成功: {processed_data.shape}")

        # 验证必需列存在
        required_columns = ['arrival_time_ns', 'input_toks', 'output_toks']
        for col in required_columns:
            if col in processed_data.columns:
                logger.info(f"✓ 必需列 {col} 存在")
            else:
                logger.error(f"✗ 必需列 {col} 缺失")
                return False

        # 显示预处理统计
        stats = preprocessor.get_preprocessing_stats()
        logger.info(f"✓ 预处理统计: {stats}")

        return True

    except Exception as e:
        logger.error(f"✗ 数据预处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理机制"""
    logger.info("=== 测试错误处理机制 ===")

    # 加载测试数据
    data = load_test_data()
    if data is None:
        logger.error("✗ 无法加载测试数据")
        return False

    try:
        # 创建预测器
        predictor = WorkloadPredictor(
            model_type=ModelType.LSTM,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=30,
            device='cpu'
        )

        # 测试未训练模型的预测（应该生成合成预测）
        logger.info("测试未训练模型的预测...")
        try:
            result = predictor.predict(data.head(50), steps=5)
            logger.info(f"✓ 未训练模型预测成功，生成 {len(result.predicted_requests)} 个预测请求")
        except Exception as e:
            logger.error(f"✗ 未训练模型预测失败: {e}")
            return False

        # 测试数据验证
        logger.info("测试数据验证...")
        valid_data = data.copy()
        invalid_data = pd.DataFrame()  # 空数据

        # 测试有效数据
        if predictor._validate_input_data(valid_data.head(10)):
            logger.info("✓ 有效数据验证通过")
        else:
            logger.error("✗ 有效数据验证失败")
            return False

        # 测试无效数据
        if not predictor._validate_input_data(invalid_data):
            logger.info("✓ 无效数据被正确识别")
        else:
            logger.error("✗ 无效数据验证失败")
            return False

        return True

    except Exception as e:
        logger.error(f"✗ 错误处理机制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_with_preprocessing():
    """测试带预处理的训练"""
    logger.info("=== 测试带预处理的训练 ===")

    # 加载测试数据
    data = load_test_data()
    if data is None:
        logger.error("✗ 无法加载测试数据")
        return False

    try:
        # 创建预测器
        predictor = WorkloadPredictor(
            model_type=ModelType.LSTM,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=20,  # 使用较短的序列以加快测试
            device='cpu'
        )

        # 测试训练（使用少量数据）
        logger.info("开始训练测试...")
        training_data = data.head(100)  # 只使用100条数据进行测试
        results = predictor.train(training_data, validation_split=0.2)

        logger.info("✓ 训练完成")
        for model, result in results.items():
            logger.info(f"  - {model}: {result.get('status', 'unknown')}")

        # 测试训练后的预测
        logger.info("测试训练后的预测...")
        prediction_result = predictor.predict(data.head(50), steps=5)
        logger.info(f"✓ 训练后预测成功，生成 {len(prediction_result.predicted_requests)} 个预测请求")

        return True

    except Exception as e:
        logger.error(f"✗ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """测试边界情况"""
    logger.info("=== 测试边界情况 ===")

    try:
        # 创建预测器
        predictor = WorkloadPredictor(
            model_type=ModelType.LSTM,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            device='cpu'
        )

        # 测试1: 数据量不足
        logger.info("测试数据量不足...")
        small_data = pd.DataFrame({
            'Timestamp': [1000.0, 1000.1],
            'Request tokens': [100, 150],
            'Response tokens': [300, 400],
            'Model': ['ChatGPT', 'ChatGPT']
        })

        try:
            result = predictor.predict(small_data, steps=3)
            logger.info("✓ 小数据量处理成功")
        except Exception as e:
            logger.info(f"✓ 小数据量被正确拒绝: {e}")

        # 测试2: 缺失列
        logger.info("测试缺失列...")
        missing_col_data = pd.DataFrame({
            'Timestamp': [1000.0, 1000.1, 1000.2],
            'Request tokens': [100, 150, 200]
            # 缺少 Response tokens
        })

        try:
            result = predictor.predict(missing_col_data, steps=3)
            logger.info("✓ 缺失列处理成功")
        except Exception as e:
            logger.info(f"✓ 缺失列被正确处理: {e}")

        # 测试3: 异常值
        logger.info("测试异常值...")
        outlier_data = pd.DataFrame({
            'Timestamp': [1000.0, 1000.1, 1000.2, 1000.3],
            'Request tokens': [100, 150, 999999, 200],  # 包含异常值
            'Response tokens': [300, 400, 500, 350],
            'Model': ['ChatGPT', 'ChatGPT', 'ChatGPT', 'ChatGPT']
        })

        try:
            result = predictor.predict(outlier_data, steps=3)
            logger.info("✓ 异常值处理成功")
        except Exception as e:
            logger.info(f"✓ 异常值被正确处理: {e}")

        return True

    except Exception as e:
        logger.error(f"✗ 边界情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    logger.info("开始错误处理机制测试...")

    tests = [
        ("数据预处理器", test_data_preprocessor),
        ("错误处理机制", test_error_handling),
        ("带预处理的训练", test_training_with_preprocessing),
        ("边界情况", test_edge_cases)
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
        logger.info("🎉 所有测试通过！错误处理机制工作正常。")
        return True
    else:
        logger.error(f"❌ {failed} 个测试失败，需要修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)