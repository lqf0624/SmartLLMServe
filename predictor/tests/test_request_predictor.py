"""
简化的请求预测器测试

测试RequestPredictor的功能，包括：
1. 数据预处理
2. 模型训练
3. 单步预测
4. 多步预测
5. 模型对比
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor.request_predictor import RequestPredictor

def create_test_data():
    """创建测试数据 - 使用BurstGPT_1_cleaned.csv的前1000行"""
    try:
        # 读取真实数据集的前1000行
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'BurstGPT_1_cleaned.csv')
        test_data = pd.read_csv(dataset_path, nrows=1000)

        # 确保必需的列存在
        required_columns = ['Timestamp', 'Request tokens', 'Response tokens']
        missing_columns = [col for col in required_columns if col not in test_data.columns]

        if missing_columns:
            raise ValueError(f"数据集缺少必需的列: {missing_columns}")

        print(f"[OK] 成功加载真实数据集前1000行")
        print(f"  数据形状: {test_data.shape}")
        print(f"  包含列: {list(test_data.columns)}")
        print(f"  Log Type分布: {test_data['Log Type'].value_counts().to_dict()}")

        return test_data

    except Exception as e:
        print(f"[ERROR] 加载真实数据集失败: {e}")
        print(f"使用模拟数据作为备用...")

        # 备用模拟数据
        np.random.seed(42)
        timestamps = list(range(0, 1000, 5))
        request_tokens = np.random.randint(100, 500, len(timestamps))
        response_tokens = np.random.randint(200, 800, len(timestamps))

        test_data = pd.DataFrame({
            'Timestamp': timestamps,
            'Request tokens': request_tokens,
            'Response tokens': response_tokens
        })

        return test_data

def test_data_preprocessing():
    """测试数据预处理"""
    print("=== 测试数据预处理 ===")

    predictor = RequestPredictor()
    test_data = create_test_data()

    try:
        processed_data = predictor.prepare_input_data(test_data)
        print(f"[OK] 数据预处理成功")
        print(f"  原始数据大小: {len(test_data)}")
        print(f"  处理后数据大小: {len(processed_data)}")
        print(f"  包含列: {list(processed_data.columns)}")
        print(f"  时间范围: {processed_data['timestamp'].min():.2f} - {processed_data['timestamp'].max():.2f}")
        return processed_data
    except Exception as e:
        print(f"[ERROR] 数据预处理失败: {e}")
        return None

def test_model_training():
    """测试模型训练"""
    print("\n=== 测试模型训练 ===")

    predictor = RequestPredictor()
    test_data = create_test_data()

    try:
        training_results = predictor.train_models(test_data)

        print("训练结果:")
        for model_name, result in training_results.items():
            if result['success']:
                print(f"[OK] {model_name.upper()} 训练成功")
                if 'train_loss' in result:
                    print(f"  训练损失: {result['train_loss']:.4f}")
                if 'val_loss' in result:
                    print(f"  验证损失: {result['val_loss']:.4f}")
                if 'training_time' in result:
                    print(f"  训练时间: {result['training_time']:.2f}s")
            else:
                print(f"[ERROR] {model_name.upper()} 训练失败: {result['error']}")

        return predictor
    except Exception as e:
        print(f"[ERROR] 模型训练失败: {e}")
        return None

def test_single_prediction():
    """测试单步预测"""
    print("\n=== 测试单步预测 ===")

    predictor = RequestPredictor()
    test_data = create_test_data()

    # 训练模型
    training_results = predictor.train_models(test_data)

    if not (predictor.lstm_trained or predictor.dlinear_trained):
        print("[ERROR] 模型训练失败，无法进行预测")
        return

    # 使用前80%的数据预测下一个请求
    split_point = int(len(test_data) * 0.8)
    historical_data = test_data.iloc[:split_point]
    actual_next = test_data.iloc[split_point]

    try:
        prediction = predictor.predict_next_request(historical_data)

        print(f"实际下一个请求:")
        print(f"  Timestamp: {actual_next['Timestamp']:.3f}")
        print(f"  Request tokens: {actual_next['Request tokens']}")
        print(f"  Response tokens: {actual_next['Response tokens']}")

        print(f"\n预测结果:")

        for model_name, pred in prediction.get('predictions', {}).items():
            if 'error' not in pred:
                print(f"\n{model_name.upper()} 预测:")
                print(f"  Timestamp: {pred['timestamp']:.3f}")
                print(f"  Request tokens: {pred['request_tokens']}")
                print(f"  Response tokens: {pred['response_tokens']}")
                print(f"  Confidence: {pred['confidence']:.3f}")

                # 计算预测误差
                timestamp_error = abs(pred['timestamp'] - actual_next['Timestamp'])
                request_tokens_error = abs(pred['request_tokens'] - actual_next['Request tokens'])
                response_tokens_error = abs(pred['response_tokens'] - actual_next['Response tokens'])

                print(f"  Timestamp误差: {timestamp_error:.3f}")
                print(f"  Request tokens误差: {request_tokens_error}")
                print(f"  Response tokens误差: {response_tokens_error}")
            else:
                print(f"\n{model_name.upper()} 预测失败: {pred['error']}")

    except Exception as e:
        print(f"[ERROR] 单步预测失败: {e}")

def test_multiple_predictions():
    """测试多步预测"""
    print("\n=== 测试多步预测 ===")

    predictor = RequestPredictor()
    test_data = create_test_data()

    # 训练模型
    training_results = predictor.train_models(test_data)

    if not (predictor.lstm_trained or predictor.dlinear_trained):
        print("[ERROR] 模型训练失败，无法进行预测")
        return

    # 使用前70%的数据预测接下来5个请求
    split_point = int(len(test_data) * 0.7)
    historical_data = test_data.iloc[:split_point]
    actual_next_5 = test_data.iloc[split_point:split_point+5]

    try:
        predictions = predictor.predict_multiple_requests(historical_data, num_requests=5)

        print(f"实际接下来5个请求:")
        for i, (_, actual) in enumerate(actual_next_5.iterrows()):
            print(f"  {i+1}. Timestamp: {actual['Timestamp']:.3f}, "
                  f"Request: {actual['Request tokens']}, Response: {actual['Response tokens']}")

        print(f"\n预测结果:")

        for model_name, preds in predictions.get('predictions', {}).items():
            print(f"\n{model_name.upper()} 预测 (共{len(preds)}个):")
            for i, pred in enumerate(preds):
                print(f"  {i+1}. Timestamp: {pred['timestamp']:.3f}, "
                      f"Request: {pred['request_tokens']}, Response: {pred['response_tokens']}")

                if i < len(actual_next_5):
                    actual = actual_next_5.iloc[i]
                    timestamp_error = abs(pred['timestamp'] - actual['Timestamp'])
                    request_tokens_error = abs(pred['request_tokens'] - actual['Request tokens'])
                    response_tokens_error = abs(pred['response_tokens'] - actual['Response tokens'])
                    print(f"     误差: T={timestamp_error:.3f}, R={request_tokens_error}, Re={response_tokens_error}")

    except Exception as e:
        print(f"[ERROR] 多步预测失败: {e}")

def test_model_comparison():
    """测试模型对比"""
    print("\n=== 测试模型对比 ===")

    predictor = RequestPredictor()
    test_data = create_test_data()

    try:
        comparison = predictor.compare_models(test_data)

        if 'error' in comparison:
            print(f"[ERROR] 模型对比失败: {comparison['error']}")
            return

        print("训练结果:")
        for model_name, result in comparison['training_results'].items():
            status = "[OK] 成功" if result['success'] else "[ERROR] 失败"
            print(f"  {model_name.upper()}: {status}")
            if not result['success']:
                print(f"    原因: {result.get('error', 'Unknown error')}")

        if 'model_performance' in comparison:
            print("\n模型性能对比:")

            for model_name, perf in comparison['model_performance'].items():
                print(f"\n{model_name.upper()}:")
                print(f"  平均时间戳误差: {perf['avg_timestamp_error']:.3f}")
                print(f"  平均请求token误差: {perf['avg_request_tokens_error']:.1f}")
                print(f"  平均响应token误差: {perf['avg_response_tokens_error']:.1f}")
                print(f"  平均置信度: {perf['avg_confidence']:.3f}")
                print(f"  评估预测数: {perf['predictions_evaluated']}")

            # 找出更好的模型
            if len(comparison['model_performance']) >= 2:
                models = list(comparison['model_performance'].keys())
                lstm_perf = comparison['model_performance'].get('lstm')
                dlinear_perf = comparison['model_performance'].get('dlinear')

                if lstm_perf and dlinear_perf:
                    lstm_score = (lstm_perf['avg_confidence'] -
                                lstm_perf['avg_timestamp_error'] * 0.1 -
                                lstm_perf['avg_request_tokens_error'] * 0.001 -
                                lstm_perf['avg_response_tokens_error'] * 0.001)

                    dlinear_score = (dlinear_perf['avg_confidence'] -
                                   dlinear_perf['avg_timestamp_error'] * 0.1 -
                                   dlinear_perf['avg_request_tokens_error'] * 0.001 -
                                   dlinear_perf['avg_response_tokens_error'] * 0.001)

                    print(f"\n综合评分:")
                    print(f"  LSTM: {lstm_score:.3f}")
                    print(f"  DLinear: {dlinear_score:.3f}")

                    if lstm_score > dlinear_score:
                        print(f"  [OK] LSTM 在这个数据集上表现更好")
                    else:
                        print(f"  [OK] DLinear 在这个数据集上表现更好")

    except Exception as e:
        print(f"[ERROR] 模型对比失败: {e}")

def main():
    """主测试函数"""
    print("开始测试简化的请求预测器...\n")

    # 1. 测试数据预处理
    processed_data = test_data_preprocessing()
    if processed_data is None:
        return

    # 2. 测试模型训练
    predictor = test_model_training()
    if predictor is None:
        return

    # 3. 测试单步预测
    test_single_prediction()

    # 4. 测试多步预测
    test_multiple_predictions()

    # 5. 测试模型对比
    test_model_comparison()

    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()