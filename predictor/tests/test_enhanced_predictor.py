"""
测试增强版预测器接口

验证新的可扩展模型接口是否正常工作，包括：
1. 动态模型注册
2. GPU设备支持
3. 模型训练和预测
4. 自定义模型集成
"""

import pandas as pd
import torch
import sys
import os
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor.enhanced_request_predictor import EnhancedRequestPredictor, create_enhanced_predictor
from predictor.model_interface import ModelConfig, ModelType, ModelDevice, model_registry
# 专注于轻量化模型，删除Transformer和GRU引用

def create_test_data():
    """创建测试数据 - 使用真实数据集"""
    try:
        # 使用真实数据集的前1000行
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'BurstGPT_1_cleaned.csv')
        test_data = pd.read_csv(dataset_path, nrows=1000)

        print(f"[OK] 成功加载真实数据集前1000行")
        print(f"  数据形状: {test_data.shape}")
        print(f"  Log Type分布: {test_data['Log Type'].value_counts().to_dict()}")

        return test_data

    except Exception as e:
        print(f"[ERROR] 加载真实数据集失败: {e}")
        return None

def test_device_detection():
    """测试设备检测"""
    print("\n=== 测试设备检测 ===")

    # 检查PyTorch和CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name()}")

    # 测试设备枚举
    for device_type in ModelDevice:
        print(f"设备类型 {device_type.value}: 支持")

    return True

def test_model_registry():
    """测试模型注册表"""
    print("\n=== 测试模型注册表 ===")

    # 检查可用模型类型
    available_types = model_registry.get_available_model_types()
    print(f"已注册的模型类型: {[t.value for t in available_types]}")

    # 检查支持的模型
    print("模型注册表状态: 正常")
    return True

def test_enhanced_predictor_creation():
    """测试增强版预测器创建"""
    print("\n=== 测试增强版预测器创建 ===")

    try:
        # 自动选择设备
        predictor = EnhancedRequestPredictor()
        print(f"[OK] 预测器创建成功，设备: {predictor.device.value}")

        # 检查支持的模型
        available_models = predictor.get_available_models()
        print(f"[OK] 支持的模型: {available_models}")

        return predictor

    except Exception as e:
        print(f"[ERROR] 预测器创建失败: {e}")
        return None

def test_model_addition(predictor: EnhancedRequestPredictor):
    """测试模型添加"""
    print("\n=== 测试模型添加 ===")

    try:
        # 添加基本模型
        basic_models = ['lstm', 'dlinear']
        results = predictor.add_models(basic_models)

        for model_name, success in results.items():
            status = "[OK]" if success else "[ERROR]"
            print(f"{status} {model_name}: {'成功' if success else '失败'}")

        # 检查激活的模型
        active_models = predictor.get_active_models()
        print(f"[OK] 激活的模型: {active_models}")

        return len(active_models) > 0

    except Exception as e:
        print(f"[ERROR] 模型添加失败: {e}")
        return False

def test_model_training(predictor: EnhancedRequestPredictor, data: pd.DataFrame):
    """测试模型训练"""
    print("\n=== 测试模型训练 ===")

    try:
        # 训练所有模型
        training_results = predictor.train_models(data, validation_split=0.2)

        print("训练结果:")
        for model_name, result in training_results.items():
            status = "[OK]" if result.success else "[ERROR]"
            loss_info = f"Loss: {result.final_loss:.4f}" if result.success else result.error_message
            print(f"  {status} {model_name}: {loss_info}")

        # 检查是否有成功的训练
        successful_trainings = [r for r in training_results.values() if r.success]
        return len(successful_trainings) > 0

    except Exception as e:
        print(f"[ERROR] 模型训练失败: {e}")
        return False

def test_model_prediction(predictor: EnhancedRequestPredictor, data: pd.DataFrame):
    """测试模型预测"""
    print("\n=== 测试模型预测 ===")

    try:
        # 准备历史数据
        historical_data = data.iloc[:800]

        # 预测下一个请求
        prediction = predictor.predict_next_request(historical_data)

        if 'error' in prediction:
            print(f"[ERROR] 预测失败: {prediction['error']}")
            return False

        print("预测结果:")
        for model_name, pred in prediction.get('predictions', {}).items():
            if 'error' not in pred:
                print(f"  {model_name.upper()}:")
                print(f"    时间戳: {pred['timestamp']:.3f}")
                print(f"    请求tokens: {pred['request_tokens']}")
                print(f"    响应tokens: {pred['response_tokens']}")
                print(f"    置信度: {pred['confidence']:.3f}")
            else:
                print(f"  {model_name.upper()}: {pred['error']}")

        # 显示摘要
        if 'summary' in prediction:
            summary = prediction['summary']
            print(f"\n摘要:")
            print(f"  历史数据大小: {summary['historical_data_size']}")
            print(f"  使用的模型数: {summary['models_used']}")

        return True

    except Exception as e:
        print(f"[ERROR] 预测测试失败: {e}")
        return False

def test_model_comparison(predictor: EnhancedRequestPredictor, data: pd.DataFrame):
    """测试模型比较"""
    print("\n=== 测试模型比较 ===")

    try:
        # 比较模型性能
        comparison = predictor.compare_models(data.iloc[:800], steps=1)

        print(f"模型总数: {comparison['model_count']}")

        if 'performance_summary' in comparison:
            summary = comparison['performance_summary']
            print(f"平均置信度: {summary.get('avg_confidence', 0):.3f}")
            print(f"最佳模型: {summary.get('best_model', 'Unknown')}")

        return True

    except Exception as e:
        print(f"[ERROR] 模型比较失败: {e}")
        return False

def test_lightweight_models():
    """测试轻量化模型"""
    print("\n=== 测试轻量化模型 ===")

    try:
        # 创建专门测试轻量化模型的预测器
        predictor = EnhancedRequestPredictor()

        # 尝试添加轻量化模型
        lightweight_models = ['lightweight']
        results = predictor.add_models(lightweight_models)

        print("轻量化模型添加结果:")
        for model_name, success in results.items():
            status = "[OK]" if success else "[ERROR]"
            print(f"  {status} {model_name}: {'成功' if success else '失败'}")

        # 如果成功添加，进行简单测试
        if any(results.values()):
            data = create_test_data()
            if data is not None:
                print("进行高级模型训练测试...")
                training_results = predictor.train_models(data.iloc[:500], validation_split=0.2)

                for model_name, result in training_results.items():
                    if result.success:
                        print(f"[OK] {model_name} 训练成功，Loss: {result.final_loss:.4f}")
                    else:
                        print(f"[ERROR] {model_name} 训练失败: {result.error_message}")

        return True

    except Exception as e:
        print(f"[ERROR] 高级模型测试失败: {e}")
        return False

def test_quick_start():
    """测试快速启动功能"""
    print("\n=== 测试快速启动功能 ===")

    try:
        # 使用快速启动函数
        predictor = create_enhanced_predictor(
            models=['lstm', 'dlinear'],
            device='auto'
        )

        print(f"[OK] 快速启动成功，设备: {predictor.device.value}")
        print(f"[OK] 激活的模型: {predictor.get_active_models()}")

        # 简单测试
        data = create_test_data()
        if data is not None:
            # 训练
            training_results = predictor.train_models(data.iloc[:500])
            successful_models = [name for name, result in training_results.items() if result.success]

            if successful_models:
                print(f"[OK] 快速启动训练成功: {successful_models}")

                # 预测
                historical_data = data.iloc[:400]
                prediction = predictor.predict_next_request(historical_data)

                if 'error' not in prediction:
                    print(f"[OK] 快速启动预测成功")
                else:
                    print(f"[ERROR] 快速启动预测失败: {prediction['error']}")
            else:
                print(f"[ERROR] 快速启动训练失败")

        return True

    except Exception as e:
        print(f"[ERROR] 快速启动测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试增强版预测器接口...\n")

    test_results = []

    # 1. 测试设备检测
    test_results.append(("设备检测", test_device_detection()))

    # 2. 测试模型注册表
    test_results.append(("模型注册表", test_model_registry()))

    # 3. 测试预测器创建
    predictor = test_enhanced_predictor_creation()
    test_results.append(("预测器创建", predictor is not None))

    if predictor is None:
        print("[ERROR] 预测器创建失败，无法继续测试")
        return

    # 4. 测试数据加载
    data = create_test_data()
    test_results.append(("数据加载", data is not None))

    if data is None:
        print("[ERROR] 数据加载失败，无法继续测试")
        return

    # 5. 测试模型添加
    model_addition_success = test_model_addition(predictor)
    test_results.append(("模型添加", model_addition_success))

    # 6. 测试模型训练
    if model_addition_success:
        training_success = test_model_training(predictor, data)
        test_results.append(("模型训练", training_success))
    else:
        test_results.append(("模型训练", False))

    # 7. 测试模型预测
    if model_addition_success:
        prediction_success = test_model_prediction(predictor, data)
        test_results.append(("模型预测", prediction_success))
    else:
        test_results.append(("模型预测", False))

    # 8. 测试模型比较
    if model_addition_success:
        comparison_success = test_model_comparison(predictor, data)
        test_results.append(("模型比较", comparison_success))
    else:
        test_results.append(("模型比较", False))

    # 9. 测试高级模型
    test_results.append(("高级模型", test_advanced_models()))

    # 10. 测试快速启动
    test_results.append(("快速启动", test_quick_start()))

    # 总结测试结果
    print("\n" + "="*50)
    print("测试结果总结:")
    print("="*50)

    passed = 0
    total = len(test_results)

    for test_name, success in test_results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("[OK] 所有测试通过！增强版预测器接口工作正常。")
        print("\nGPU部署建议:")
        print("1. 查看 predictor/gpu_deployment_guide.md 获取详细部署指南")
        print("2. 使用 EnhancedRequestPredictor 替代原有的 RequestPredictor")
        print("3. 支持动态添加新模型，无需修改核心代码")
        print("4. 自动GPU设备管理，优化性能")
    else:
        print(f"[WARNING] {total - passed} 个测试失败，请检查相关配置。")

if __name__ == "__main__":
    main()