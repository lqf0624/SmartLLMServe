"""
简单的LSTM测试，用于诊断问题
"""

import pandas as pd
import numpy as np
from predictor.request_predictor import RequestPredictor

def create_simple_test_data():
    """创建简单的测试数据 - 使用BurstGPT_1_cleaned.csv的前1000行"""
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
        print("使用模拟数据作为备用...")

        # 备用模拟数据
        np.random.seed(42)
        timestamps = list(range(0, 1000, 10))
        request_tokens = np.random.randint(100, 500, len(timestamps))
        response_tokens = np.random.randint(200, 800, len(timestamps))

        test_data = pd.DataFrame({
            'Timestamp': timestamps,
            'Request tokens': request_tokens,
            'Response tokens': response_tokens
        })

        return test_data

def test_lstm_preparation():
    """测试LSTM数据准备"""
    print("=== 测试LSTM数据准备 ===")

    predictor = RequestPredictor()
    test_data = create_simple_test_data()

    print(f"原始数据列: {list(test_data.columns)}")
    print(f"原始数据形状: {test_data.shape}")

    try:
        # 测试数据预处理
        prepared_data = predictor.prepare_input_data(test_data)
        print(f"[OK] 数据预处理成功")
        print(f"预处理后列: {list(prepared_data.columns)}")
        print(f"预处理后形状: {prepared_data.shape}")

        # 测试LSTM数据准备
        lstm_data = predictor._prepare_lstm_data(prepared_data)
        print(f"[OK] LSTM数据准备成功")
        print(f"LSTM数据列: {list(lstm_data.columns)}")
        print(f"LSTM数据形状: {lstm_data.shape}")

        # 检查LSTM需要的列
        required_lstm_columns = ['arrival_time_ns', 'input_toks', 'output_toks', 'burst_pattern']
        missing_lstm_cols = [col for col in required_lstm_columns if col not in lstm_data.columns]

        if missing_lstm_cols:
            print(f"[ERROR] LSTM缺少列: {missing_lstm_cols}")
        else:
            print(f"[OK] LSTM包含所有必需列")

        return prepared_data, lstm_data

    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_lstm_training():
    """测试LSTM训练"""
    print("\n=== 测试LSTM训练 ===")

    prepared_data, lstm_data = test_lstm_preparation()

    if prepared_data is None:
        return

    try:
        # 直接测试LSTM训练
        predictor = RequestPredictor()

        print("开始LSTM训练...")
        lstm_history = predictor.lstm_predictor.train(lstm_data, validation_split=0.2)
        print(f"[OK] LSTM训练成功")
        print(f"训练轮数: {len(lstm_history.get('train_loss', []))}")

        return True

    except Exception as e:
        print(f"[ERROR] LSTM训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_lstm_training()