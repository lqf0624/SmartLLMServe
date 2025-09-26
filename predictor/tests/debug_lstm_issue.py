"""
调试LSTM问题的脚本
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor.request_predictor import RequestPredictor

def debug_lstm_training():
    """调试LSTM训练问题"""
    print("=== 调试LSTM训练问题 ===")

    try:
        # 使用真实数据集的前1000行
        dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'BurstGPT_1_cleaned.csv')
        test_data = pd.read_csv(dataset_path, nrows=1000)

        print(f"[OK] 成功加载真实数据集前1000行")
        print(f"  数据形状: {test_data.shape}")
        print(f"  Log Type分布: {test_data['Log Type'].value_counts().to_dict()}")

    except Exception as e:
        print(f"[ERROR] 加载真实数据集失败: {e}")
        print("使用模拟数据作为备用...")

        # 备用模拟数据
        np.random.seed(42)
        timestamps = list(range(0, 1000, 5))
        test_data = pd.DataFrame({
            'Timestamp': timestamps,
            'Request tokens': np.random.randint(100, 500, len(timestamps)),
            'Response tokens': np.random.randint(200, 800, len(timestamps))
        })

    print(f"测试数据:")
    print(f"  列: {list(test_data.columns)}")
    print(f"  形状: {test_data.shape}")
    print(f"  前3行:")
    print(test_data.head(3))

    predictor = RequestPredictor()

    try:
        # 步骤1：测试数据预处理
        print("\n步骤1：数据预处理")
        prepared_data = predictor.prepare_input_data(test_data)
        print(f"  预处理后列: {list(prepared_data.columns)}")
        print(f"  预处理后形状: {prepared_data.shape}")

        # 步骤2：测试LSTM数据准备
        print("\n步骤2：LSTM数据准备")
        lstm_data = predictor._prepare_lstm_data(prepared_data)
        print(f"  LSTM数据列: {list(lstm_data.columns)}")
        print(f"  LSTM数据形状: {lstm_data.shape}")

        # 检查LSTM需要的列
        required_cols = ['arrival_time_ns', 'input_toks', 'output_toks', 'burst_pattern']
        print(f"  检查必需列:")
        for col in required_cols:
            if col in lstm_data.columns:
                print(f"    [OK] {col}")
            else:
                print(f"    [ERROR] {col}")

        # 步骤3：直接测试LSTM训练（跳过train_models）
        print("\n步骤3：直接测试LSTM训练")
        try:
            lstm_history = predictor.lstm_predictor.train(lstm_data, validation_split=0.2)
            print(f"  [OK] LSTM训练成功")
            print(f"  训练轮数: {len(lstm_history.get('train_loss', []))}")
        except Exception as e:
            print(f"  [ERROR] LSTM训练失败: {e}")
            import traceback
            traceback.print_exc()

        # 步骤4：测试完整的train_models方法
        print("\n步骤4：测试完整的train_models方法")
        try:
            training_results = predictor.train_models(test_data)
            print(f"  [OK] train_models成功")
            print(f"  结果: {training_results}")
        except Exception as e:
            print(f"  [ERROR] train_models失败: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"[ERROR] 调试过程失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lstm_training()