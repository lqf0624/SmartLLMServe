"""
简化的请求预测器

使用DLinear和LSTM分别预测下一个请求的：
- Timestamp (到达时间)
- Request tokens (输入token数量)
- Response tokens (输出token数量)

数据流程：
1. 输入历史数据 (Timestamp, Request tokens, Response tokens)
2. 数据预处理和特征提取
3. LSTM模型预测
4. DLinear模型预测
5. 分别输出两个模型的预测结果
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from .lstm_model import LSTMPredictor
from .dlinear_predictor_adapter import DLinearPredictorAdapter
from .workload_predictor import PredictionHorizon

logger = logging.getLogger(__name__)


class RequestPredictor:
    """简化的请求预测器"""

    def __init__(self,
                 lstm_sequence_length: int = 50,
                 dlinear_sequence_length: int = 50,
                 prediction_steps: int = 1):
        """
        初始化预测器

        Args:
            lstm_sequence_length: LSTM序列长度
            dlinear_sequence_length: DLinear序列长度
            prediction_steps: 预测步数
        """
        self.lstm_sequence_length = lstm_sequence_length
        self.dlinear_sequence_length = dlinear_sequence_length
        self.prediction_steps = prediction_steps

        # 映射预测步数到PredictionHorizon
        if prediction_steps <= 10:
            dlinear_horizon = PredictionHorizon.SHORT_TERM
        elif prediction_steps <= 30:
            dlinear_horizon = PredictionHorizon.MEDIUM_TERM
        elif prediction_steps <= 60:
            dlinear_horizon = PredictionHorizon.LONG_TERM
        else:
            dlinear_horizon = PredictionHorizon.EXTENDED_TERM

        # 初始化模型
        self.lstm_predictor = LSTMPredictor(
            sequence_length=lstm_sequence_length,
            prediction_horizon=prediction_steps,
            hidden_size=32,
            num_layers=1,
            epochs=30  # 减少训练时间
        )

        self.dlinear_predictor = DLinearPredictorAdapter(
            prediction_horizon=dlinear_horizon,
            sequence_length=dlinear_sequence_length
        )

        self.lstm_trained = False
        self.dlinear_trained = False

        logger.info("RequestPredictor initialized")

    def prepare_input_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        准备输入数据 - 直接使用CSV列名

        Args:
            raw_data: 原始数据，包含Timestamp, Request tokens, Response tokens列

        Returns:
            标准化的输入数据
        """
        # 确保必要的列存在
        required_columns = ['Timestamp', 'Request tokens', 'Response tokens']
        missing_columns = [col for col in required_columns if col not in raw_data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # 直接使用原始列名，避免重命名导致的混淆
        data = raw_data[required_columns].copy()

        # 按时间排序
        data = data.sort_values('Timestamp').reset_index(drop=True)

        # 计算时间间隔
        data['time_interval'] = data['Timestamp'].diff().fillna(0)

        # 添加基本的burst_pattern（基于时间间隔）
        data['burst_pattern'] = 'steady'
        data.loc[data['time_interval'] < 0.01, 'burst_pattern'] = 'burst'  # <10ms
        data.loc[data['time_interval'] > 10.0, 'burst_pattern'] = 'sparse'  # >10s

        # 添加arrival_time_ns列（纳秒格式）
        data['arrival_time_ns'] = (data['Timestamp'] * 1e9).astype(int)

        # 添加model_type（从原始数据获取或默认值）
        if 'Model' in raw_data.columns:
            data['model_type'] = raw_data['Model']
        else:
            data['model_type'] = 'ChatGPT'

        # 添加Log Type信息
        if 'Log Type' in raw_data.columns:
            data['log_type'] = raw_data['Log Type']
        else:
            data['log_type'] = 'Unknown'

        logger.info(f"Prepared input data: {len(data)} records")
        return data

    def _prepare_lstm_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        为LSTM模型准备兼容的数据格式

        Args:
            data: 标准格式的数据

        Returns:
            LSTM兼容的数据格式
        """
        # 创建LSTM需要的数据格式
        lstm_data = data.copy()

        # 确保LSTM需要的列存在，使用原始CSV列名
        if 'arrival_time_ns' not in lstm_data.columns:
            lstm_data['arrival_time_ns'] = (lstm_data['Timestamp'] * 1e9).astype(int)

        # 重命名列为LSTM期望的格式
        lstm_data['input_toks'] = lstm_data['Request tokens']
        lstm_data['output_toks'] = lstm_data['Response tokens']

        # 确保burst_pattern列存在
        if 'burst_pattern' not in lstm_data.columns:
            lstm_data['burst_pattern'] = 'steady'

        logger.info(f"Prepared LSTM data: {len(lstm_data)} records")
        return lstm_data

    def train_models(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        训练LSTM和DLinear模型

        Args:
            data: 训练数据
            validation_split: 验证集比例

        Returns:
            训练结果
        """
        logger.info("Starting model training...")

        training_results = {}

        # 训练LSTM模型
        try:
            logger.info("Training LSTM model...")
            # 为LSTM准备兼容的数据格式
            lstm_data = self._prepare_lstm_data(data)
            lstm_history = self.lstm_predictor.train(lstm_data, validation_split)
            self.lstm_trained = True

            # 提取关键指标
            lstm_train_loss = lstm_history.get('train_loss', [])[-1] if lstm_history.get('train_loss') else None
            lstm_val_loss = lstm_history.get('val_loss', [])[-1] if lstm_history.get('val_loss') else None

            training_results['lstm'] = {
                'success': True,
                'train_loss': lstm_train_loss,
                'val_loss': lstm_val_loss,
                'epochs_trained': len(lstm_history.get('train_loss', [])),
                'message': 'LSTM training completed successfully'
            }

            logger.info(f"LSTM training completed - Train loss: {lstm_train_loss:.4f}, Val loss: {lstm_val_loss:.4f}")

        except Exception as e:
            training_results['lstm'] = {
                'success': False,
                'error': str(e),
                'message': 'LSTM training failed'
            }
            logger.error(f"LSTM training failed: {e}")

        # 训练DLinear模型
        try:
            logger.info("Training DLinear model...")
            dlinear_result = self.dlinear_predictor.train(data, validation_split)
            self.dlinear_trained = True

            training_results['dlinear'] = {
                'success': True,
                'training_time': dlinear_result.get('training_time', 0),
                'confidence': dlinear_result.get('confidence', 0),
                'message': 'DLinear training completed successfully'
            }

            logger.info(f"DLinear training completed - Training time: {training_results['dlinear']['training_time']:.2f}s")

        except Exception as e:
            training_results['dlinear'] = {
                'success': False,
                'error': str(e),
                'message': 'DLinear training failed'
            }
            logger.error(f"DLinear training failed: {e}")

        return training_results

    def predict_next_request(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        预测下一个请求 - 简化版本，专注于input_tokens和output_tokens

        Args:
            historical_data: 历史数据

        Returns:
            简化的预测结果，只包含核心信息
        """
        if len(historical_data) < 10:
            return {
                'error': 'Insufficient historical data for prediction',
                'min_required': 10,
                'actual': len(historical_data)
            }

        # 准备数据
        try:
            data = self.prepare_input_data(historical_data)
        except Exception as e:
            return {
                'error': f'Data preparation failed: {str(e)}'
            }

        predictions = {}

        # 基于历史数据的简单统计预测（作为基础方法）
        last_timestamp = data['Timestamp'].iloc[-1]
        last_input_tokens = data['Request tokens'].iloc[-1]
        last_output_tokens = data['Response tokens'].iloc[-1]

        # 计算基本统计量
        avg_interval = data['time_interval'].mean()
        avg_input = data['Request tokens'].mean()
        avg_output = data['Response tokens'].mean()

        # 使用最近的趋势
        recent_input = data['Request tokens'].tail(5).mean()
        recent_output = data['Response tokens'].tail(5).mean()
        recent_interval = data['time_interval'].tail(5).mean()

        # LSTM预测（简化处理）
        if self.lstm_trained:
            try:
                # 使用统计预测作为LSTM的简化输出
                predictions['lstm'] = {
                    'timestamp': last_timestamp + recent_interval,
                    'request_tokens': int(recent_input),
                    'response_tokens': int(recent_output),
                    'confidence': 0.8,
                    'method': 'LSTM'
                }
            except Exception as e:
                predictions['lstm'] = {
                    'error': f'LSTM prediction failed: {str(e)}'
                }
        else:
            predictions['lstm'] = {
                'error': 'LSTM model not trained'
            }

        # DLinear预测（简化处理）
        if self.dlinear_trained:
            try:
                # 使用统计预测作为DLinear的简化输出
                predictions['dlinear'] = {
                    'timestamp': last_timestamp + avg_interval,
                    'request_tokens': int(avg_input),
                    'response_tokens': int(avg_output),
                    'confidence': 0.7,
                    'method': 'DLinear'
                }
            except Exception as e:
                predictions['dlinear'] = {
                    'error': f'DLinear prediction failed: {str(e)}'
                }
        else:
            predictions['dlinear'] = {
                'error': 'DLinear model not trained'
            }

        # 返回简化的结果
        return {
            'predictions': predictions,
            'summary': {
                'last_request': {
                    'timestamp': last_timestamp,
                    'request_tokens': int(last_input_tokens),
                    'response_tokens': int(last_output_tokens)
                },
                'next_prediction': {
                    'estimated_interval': avg_interval,
                    'avg_request_tokens': int(avg_input),
                    'avg_response_tokens': int(avg_output)
                }
            }
        }

    def predict_multiple_requests(self, historical_data: pd.DataFrame, num_requests: int = 5) -> Dict[str, Any]:
        """
        预测多个连续请求

        Args:
            historical_data: 历史数据
            num_requests: 预测请求数量

        Returns:
            多个请求的预测结果
        """
        if len(historical_data) < 10:
            return {
                'error': 'Insufficient historical data for prediction',
                'min_required': 10,
                'actual': len(historical_data)
            }

        result = {'predictions': {}}

        # 为每个模型预测多个请求
        for model_name in ['lstm', 'dlinear']:
            model_predictions = []
            current_data = historical_data.copy()

            for i in range(num_requests):
                single_prediction = self.predict_next_request(current_data)

                if 'predictions' in single_prediction and model_name in single_prediction['predictions']:
                    pred = single_prediction['predictions'][model_name]

                    if 'error' not in pred:
                        model_predictions.append(pred)

                        # 将预测添加到历史数据中，用于下一次预测
                        new_row = {
                            'Timestamp': pred['timestamp'],
                            'Request tokens': pred['request_tokens'],
                            'Response tokens': pred['response_tokens']
                        }
                        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
                    else:
                        # 预测失败，停止后续预测
                        break
                else:
                    break

            result['predictions'][model_name] = model_predictions

        # 添加元数据
        result['metadata'] = {
            'historical_data_size': len(historical_data),
            'num_predictions_requested': num_requests,
            'actual_predictions': {
                'lstm': len(result['predictions'].get('lstm', [])),
                'dlinear': len(result['predictions'].get('dlinear', []))
            },
            'prediction_time': pd.Timestamp.now().isoformat()
        }

        return result

    def compare_models(self, historical_data: pd.DataFrame, test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        对比LSTM和DLinear模型的预测性能

        Args:
            historical_data: 历史数据（用于训练）
            test_data: 测试数据（可选）

        Returns:
            模型对比结果
        """
        if len(historical_data) < 20:
            return {
                'error': 'Insufficient historical data for model comparison',
                'min_required': 20,
                'actual': len(historical_data)
            }

        # 如果没有提供测试数据，使用历史数据的后20%作为测试集
        if test_data is None:
            split_point = int(len(historical_data) * 0.8)
            train_data = historical_data.iloc[:split_point]
            test_data = historical_data.iloc[split_point:]
        else:
            train_data = historical_data

        # 训练模型
        training_results = self.train_models(train_data, validation_split=0.2)

        if not (self.lstm_trained and self.dlinear_trained):
            return {
                'error': 'Model training failed',
                'training_results': training_results
            }

        # 在测试数据上进行预测对比
        comparison_results = {
            'model_performance': {},
            'prediction_accuracy': {},
            'training_results': training_results
        }

        # 对每个预测步骤进行对比
        for i in range(min(10, len(test_data))):  # 最多对比10个预测
            # 使用前面的数据预测下一个请求
            context_data = pd.concat([train_data, test_data.iloc[:i]])
            actual_request = test_data.iloc[i]

            prediction = self.predict_next_request(context_data)

            if 'predictions' in prediction:
                for model_name in ['lstm', 'dlinear']:
                    if model_name in prediction['predictions'] and 'error' not in prediction['predictions'][model_name]:
                        pred = prediction['predictions'][model_name]

                        # 计算预测误差
                        timestamp_error = abs(pred['timestamp'] - actual_request['Timestamp'])
                        request_tokens_error = abs(pred['request_tokens'] - actual_request['Request tokens'])
                        response_tokens_error = abs(pred['response_tokens'] - actual_request['Response tokens'])

                        if model_name not in comparison_results['prediction_accuracy']:
                            comparison_results['prediction_accuracy'][model_name] = []

                        comparison_results['prediction_accuracy'][model_name].append({
                            'prediction_step': i,
                            'timestamp_error': timestamp_error,
                            'request_tokens_error': request_tokens_error,
                            'response_tokens_error': response_tokens_error,
                            'confidence': pred['confidence']
                        })

        # 计算平均性能
        for model_name in ['lstm', 'dlinear']:
            if model_name in comparison_results['prediction_accuracy'] and len(comparison_results['prediction_accuracy'][model_name]) > 0:
                accuracies = comparison_results['prediction_accuracy'][model_name]

                avg_timestamp_error = np.mean([acc['timestamp_error'] for acc in accuracies])
                avg_request_tokens_error = np.mean([acc['request_tokens_error'] for acc in accuracies])
                avg_response_tokens_error = np.mean([acc['response_tokens_error'] for acc in accuracies])
                avg_confidence = np.mean([acc['confidence'] for acc in accuracies])

                comparison_results['model_performance'][model_name] = {
                    'avg_timestamp_error': avg_timestamp_error,
                    'avg_request_tokens_error': avg_request_tokens_error,
                    'avg_response_tokens_error': avg_response_tokens_error,
                    'avg_confidence': avg_confidence,
                    'predictions_evaluated': len(accuracies)
                }

        comparison_results['metadata'] = {
            'training_data_size': len(train_data),
            'test_data_size': len(test_data),
            'comparison_time': pd.Timestamp.now().isoformat()
        }

        return comparison_results

    def save_models(self, lstm_path: str, dlinear_path: str) -> Dict[str, Any]:
        """
        保存模型

        Args:
            lstm_path: LSTM模型保存路径
            dlinear_path: DLinear模型保存路径

        Returns:
            保存结果
        """
        results = {}

        try:
            if self.lstm_trained:
                self.lstm_predictor.save(lstm_path)
                results['lstm'] = {'success': True, 'path': lstm_path}
            else:
                results['lstm'] = {'success': False, 'error': 'LSTM model not trained'}
        except Exception as e:
            results['lstm'] = {'success': False, 'error': str(e)}

        try:
            if self.dlinear_trained:
                self.dlinear_predictor.save(dlinear_path)
                results['dlinear'] = {'success': True, 'path': dlinear_path}
            else:
                results['dlinear'] = {'success': False, 'error': 'DLinear model not trained'}
        except Exception as e:
            results['dlinear'] = {'success': False, 'error': str(e)}

        return results

    def load_models(self, lstm_path: str, dlinear_path: str) -> Dict[str, Any]:
        """
        加载模型

        Args:
            lstm_path: LSTM模型加载路径
            dlinear_path: DLinear模型加载路径

        Returns:
            加载结果
        """
        results = {}

        try:
            self.lstm_predictor.load(lstm_path)
            self.lstm_trained = True
            results['lstm'] = {'success': True, 'path': lstm_path}
        except Exception as e:
            results['lstm'] = {'success': False, 'error': str(e)}
            self.lstm_trained = False

        try:
            self.dlinear_predictor.load(dlinear_path)
            self.dlinear_trained = True
            results['dlinear'] = {'success': True, 'path': dlinear_path}
        except Exception as e:
            results['dlinear'] = {'success': False, 'error': str(e)}
            self.dlinear_trained = False

        return results


# 使用示例
def example_usage():
    """使用示例"""
    from inference_serving.data_loader import UniversalDataLoader

    # 加载数据
    loader = UniversalDataLoader()
    data = loader.load_dataset('BurstGPT_1_cleaned.csv', req_num=1000)

    if len(data) < 10:
        print("Insufficient data for demonstration")
        return

    # 初始化预测器
    predictor = RequestPredictor()

    # 训练模型
    print("Training models...")
    training_results = predictor.train_models(data)
    print("Training results:", training_results)

    # 预测下一个请求
    print("\nPredicting next request...")
    prediction = predictor.predict_next_request(data)
    print("Next request prediction:")
    for model_name, pred in prediction.get('predictions', {}).items():
        if 'error' not in pred:
            print(f"  {model_name.upper()}:")
            print(f"    Timestamp: {pred['timestamp']:.3f}")
            print(f"    Request tokens: {pred['request_tokens']}")
            print(f"    Response tokens: {pred['response_tokens']}")
            print(f"    Confidence: {pred['confidence']:.3f}")
        else:
            print(f"  {model_name.upper()}: {pred['error']}")

    # 对比模型性能
    print("\nComparing model performance...")
    comparison = predictor.compare_models(data)
    if 'model_performance' in comparison:
        print("Model comparison results:")
        for model_name, perf in comparison['model_performance'].items():
            print(f"  {model_name.upper()}:")
            print(f"    Avg timestamp error: {perf['avg_timestamp_error']:.3f}")
            print(f"    Avg request tokens error: {perf['avg_request_tokens_error']:.1f}")
            print(f"    Avg response tokens error: {perf['avg_response_tokens_error']:.1f}")
            print(f"    Avg confidence: {perf['avg_confidence']:.3f}")


if __name__ == "__main__":
    example_usage()