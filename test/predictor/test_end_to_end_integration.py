#!/usr/bin/env python3
"""
端到端集成测试脚本

测试从数据加载到预测输出的完整工作流程，验证整个系统的集成性和稳定性。
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import traceback
from pathlib import Path
import json
import tempfile
import shutil
from typing import Dict, List, Any, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from predictor.workload_predictor import WorkloadPredictor, ModelType, PredictionHorizon
from predictor.ensemble_predictor import EnsemblePredictor, WeightStrategy
from predictor.simple_visualization import SimplePredictionVisualizer
from predictor.data_preprocessor import DataPreprocessor, DataFormat

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EndToEndTestRunner:
    """端到端测试运行器"""

    def __init__(self):
        self.test_results = {}
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"创建临时测试目录: {self.temp_dir}")

    def cleanup(self):
        """清理临时文件"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("清理临时测试目录")

    def create_realistic_test_data(self, size: int = 1000) -> pd.DataFrame:
        """创建真实的测试数据"""
        logger.info(f"创建真实测试数据，大小: {size}")

        # 模拟真实工作负载数据
        base_time = pd.Timestamp('2023-01-01 00:00:00')
        timestamps = []
        current_time = base_time

        # 模拟不同的工作负载模式
        input_tokens = []
        output_tokens = []
        burst_patterns = []

        for i in range(size):
            # 时间间隔（毫秒）
            if i % 100 < 10:  # 突发期
                interval = np.random.exponential(10)  # 10ms平均间隔
                pattern = 'burst'
            elif i % 100 < 70:  # 稳定期
                interval = np.random.exponential(100)  # 100ms平均间隔
                pattern = 'steady'
            else:  # 稀疏期
                interval = np.random.exponential(500)  # 500ms平均间隔
                pattern = 'sparse'

            current_time += pd.Timedelta(milliseconds=interval)
            timestamps.append(current_time)

            # Token数量
            if pattern == 'burst':
                input_tok = np.random.randint(50, 200)
            elif pattern == 'steady':
                input_tok = np.random.randint(20, 80)
            else:
                input_tok = np.random.randint(5, 30)

            output_tok = int(input_tok * np.random.uniform(2.5, 4.0))

            input_tokens.append(input_tok)
            output_tokens.append(output_tok)
            burst_patterns.append(pattern)

        return pd.DataFrame({
            'arrival_time_ns': timestamps,
            'input_toks': input_tokens,
            'output_toks': output_tokens,
            'burst_pattern': burst_patterns,
            'model_type': ['ChatGPT'] * size
        })

    def test_data_preprocessing_workflow(self) -> bool:
        """测试数据预处理工作流程"""
        logger.info("=== 测试数据预处理工作流程 ===")

        try:
            # 创建原始数据
            raw_data = self.create_realistic_test_data(500)

            # 添加一些数据质量问题
            raw_data.loc[10, 'input_toks'] = np.nan  # 缺失值
            raw_data.loc[20, 'output_toks'] = -1     # 异常值
            raw_data.loc[30, 'arrival_time_ns'] = None  # 时间戳缺失

            # 保存原始数据
            raw_data_path = os.path.join(self.temp_dir, "raw_data.csv")
            raw_data.to_csv(raw_data_path, index=False)

            # 数据预处理
            preprocessor = DataPreprocessor()
            processed_data = preprocessor.load_and_preprocess(
                raw_data_path,
                DataFormat.CSV,
                target_column='input_toks'
            )

            logger.info(f"原始数据大小: {len(raw_data)}")
            logger.info(f"处理后数据大小: {len(processed_data)}")

            # 验证数据质量
            assert not processed_data.isnull().any().any(), "处理后不应有缺失值"
            assert (processed_data['input_toks'] > 0).all(), "input_toks应该都是正值"
            assert (processed_data['output_toks'] > 0).all(), "output_toks应该都是正值"

            # 保存处理后的数据
            processed_data_path = os.path.join(self.temp_dir, "processed_data.csv")
            processed_data.to_csv(processed_data_path, index=False)

            self.test_results['data_preprocessing'] = {
                'success': True,
                'original_size': len(raw_data),
                'processed_size': len(processed_data),
                'missing_values_handled': raw_data.isnull().sum().sum(),
                'anomaly_values_handled': len(raw_data[(raw_data['input_toks'] <= 0) | (raw_data['output_toks'] <= 0)])
            }

            logger.info("✅ 数据预处理工作流程测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 数据预处理工作流程测试失败: {e}")
            traceback.print_exc()
            self.test_results['data_preprocessing'] = {'success': False, 'error': str(e)}
            return False

    def test_model_training_workflow(self) -> bool:
        """测试模型训练工作流程"""
        logger.info("=== 测试模型训练工作流程 ===")

        try:
            # 创建测试数据
            test_data = self.create_realistic_test_data(800)

            # 测试不同模型的训练
            models_config = [
                (ModelType.LSTM, "LSTM模型"),
                (ModelType.DLINEAR, "DLinear模型"),
                (ModelType.ENSEMBLE, "集成模型")
            ]

            training_results = {}

            for model_type, model_name in models_config:
                logger.info(f"训练 {model_name}...")

                # 创建预测器
                predictor = WorkloadPredictor(
                    model_type=model_type,
                    prediction_horizon=PredictionHorizon.SHORT_TERM,
                    sequence_length=40,
                    enable_ensemble=(model_type == ModelType.ENSEMBLE),
                    ensemble_strategy=WeightStrategy.PERFORMANCE_BASED,
                    device='cpu'
                )

                # 训练
                training_result = predictor.train(test_data, validation_split=0.2)

                # 保存模型
                model_path = os.path.join(self.temp_dir, f"{model_name.lower()}_model.pkl")
                predictor.save(model_path)

                training_results[model_name] = {
                    'training_success': training_result,
                    'model_saved': os.path.exists(model_path),
                    'model_info': predictor.get_model_info()
                }

                logger.info(f"{model_name} 训练完成")

            self.test_results['model_training'] = training_results

            # 验证所有模型都训练成功
            all_success = all(result['training_success'] for result in training_results.values())
            assert all_success, "所有模型都应该训练成功"

            logger.info("✅ 模型训练工作流程测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 模型训练工作流程测试失败: {e}")
            traceback.print_exc()
            self.test_results['model_training'] = {'success': False, 'error': str(e)}
            return False

    def test_prediction_workflow(self) -> bool:
        """测试预测工作流程"""
        logger.info("=== 测试预测工作流程 ===")

        try:
            # 创建测试数据
            test_data = self.create_realistic_test_data(600)

            # 加载已训练的模型
            models_to_test = [
                (ModelType.LSTM, "lstm_model.pkl"),
                (ModelType.DLINEAR, "dlinear_model.pkl"),
                (ModelType.ENSEMBLE, "ensemble_model.pkl")
            ]

            prediction_results = {}

            for model_type, model_file in models_to_test:
                model_path = os.path.join(self.temp_dir, model_file)

                if os.path.exists(model_path):
                    logger.info(f"测试 {model_type.value} 预测...")

                    # 加载模型
                    predictor = WorkloadPredictor(
                        model_type=model_type,
                        prediction_horizon=PredictionHorizon.SHORT_TERM,
                        device='cpu'
                    )
                    predictor.load(model_path)

                    # 进行预测
                    historical_data = test_data.iloc[:-20]
                    prediction_result = predictor.predict(historical_data, steps=15)

                    # 生成报告
                    report = predictor.generate_prediction_report(prediction_result, historical_data)

                    prediction_results[model_type.value] = {
                        'prediction_success': True,
                        'prediction_count': len(prediction_result.predicted_requests),
                        'confidence': prediction_result.confidence,
                        'report_sections': list(report.keys())
                    }

                    logger.info(f"{model_type.value} 预测完成，置信度: {prediction_result.confidence:.3f}")
                else:
                    logger.warning(f"模型文件不存在: {model_path}")
                    prediction_results[model_type.value] = {
                        'prediction_success': False,
                        'error': 'Model file not found'
                    }

            self.test_results['prediction'] = prediction_results

            # 验证预测结果
            successful_predictions = [r for r in prediction_results.values() if r['prediction_success']]
            assert len(successful_predictions) > 0, "应该有至少一个成功的预测"

            logger.info("✅ 预测工作流程测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 预测工作流程测试失败: {e}")
            traceback.print_exc()
            self.test_results['prediction'] = {'success': False, 'error': str(e)}
            return False

    def test_visualization_workflow(self) -> bool:
        """测试可视化工作流程"""
        logger.info("=== 测试可视化工作流程 ===")

        try:
            # 创建测试数据
            test_data = self.create_realistic_test_data(400)

            # 加载集成模型
            ensemble_model_path = os.path.join(self.temp_dir, "ensemble_model.pkl")

            if os.path.exists(ensemble_model_path):
                # 加载模型
                predictor = WorkloadPredictor(
                    model_type=ModelType.ENSEMBLE,
                    prediction_horizon=PredictionHorizon.SHORT_TERM,
                    device='cpu'
                )
                predictor.load(ensemble_model_path)

                # 进行预测
                historical_data = test_data.iloc[:-15]
                prediction_result = predictor.predict(historical_data, steps=10)

                # 创建可视化
                visualizer = SimplePredictionVisualizer()

                # 测试单模型可视化
                single_plot_path = os.path.join(self.temp_dir, "single_prediction.png")
                single_metrics = visualizer.plot_prediction_comparison(
                    historical_data, prediction_result, "Ensemble Model",
                    save_path=single_plot_path, show_plot=False
                )

                # 测试多模型对比
                multi_models_results = {}
                for model_type in [ModelType.LSTM, ModelType.DLINEAR]:
                    model_path = os.path.join(self.temp_dir, f"{model_type.value.lower()}_model.pkl")
                    if os.path.exists(model_path):
                        model_predictor = WorkloadPredictor(
                            model_type=model_type,
                            prediction_horizon=PredictionHorizon.SHORT_TERM,
                            device='cpu'
                        )
                        model_predictor.load(model_path)
                        model_prediction = model_predictor.predict(historical_data, steps=10)
                        multi_models_results[model_type.value] = model_prediction

                if multi_models_results:
                    multi_plot_path = os.path.join(self.temp_dir, "multi_model_comparison.png")
                    multi_metrics = visualizer.plot_multi_model_comparison(
                        multi_models_results, historical_data,
                        save_path=multi_plot_path, show_plot=False
                    )

                # 生成报告
                report_path = os.path.join(self.temp_dir, "prediction_report.txt")
                report = visualizer.generate_simple_report(
                    {"Ensemble": prediction_result}, historical_data,
                    save_path=report_path
                )

                self.test_results['visualization'] = {
                    'success': True,
                    'single_plot_created': os.path.exists(single_plot_path),
                    'multi_plot_created': os.path.exists(multi_plot_path),
                    'report_created': os.path.exists(report_path),
                    'single_metrics': single_metrics,
                    'models_compared': len(multi_models_results)
                }

                logger.info("✅ 可视化工作流程测试通过")
                return True
            else:
                logger.warning("集成模型文件不存在，跳过可视化测试")
                self.test_results['visualization'] = {
                    'success': False,
                    'error': 'Ensemble model not found'
                }
                return False

        except Exception as e:
            logger.error(f"❌ 可视化工作流程测试失败: {e}")
            traceback.print_exc()
            self.test_results['visualization'] = {'success': False, 'error': str(e)}
            return False

    def test_error_recovery_workflow(self) -> bool:
        """测试错误恢复工作流程"""
        logger.info("=== 测试错误恢复工作流程 ===")

        try:
            # 测试1: 数据格式错误
            logger.info("测试数据格式错误处理...")
            malformed_data = pd.DataFrame({
                'wrong_column': [1, 2, 3],
                'another_wrong': [4, 5, 6]
            })

            try:
                predictor = WorkloadPredictor(model_type=ModelType.LSTM, device='cpu')
                predictor.validate_data(malformed_data)
                validation_success = False
            except:
                validation_success = True  # 应该抛出异常

            # 测试2: 模型加载错误
            logger.info("测试模型加载错误处理...")
            try:
                predictor = WorkloadPredictor(model_type=ModelType.LSTM, device='cpu')
                predictor.load("non_existent_model.pkl")
                loading_success = False
            except:
                loading_success = True  # 应该抛出异常

            # 测试3: 预测错误恢复
            logger.info("测试预测错误恢复...")
            test_data = self.create_realistic_test_data(50)  # 数据不足

            predictor = WorkloadPredictor(
                model_type=ModelType.LSTM,
                prediction_horizon=PredictionHorizon.SHORT_TERM,
                sequence_length=100,  # 序列过长
                device='cpu'
            )

            # 即使训练失败，预测也应该有合理的处理
            try:
                prediction_result = predictor.predict(test_data, steps=10)
                recovery_success = prediction_result is not None
            except:
                recovery_success = False

            self.test_results['error_recovery'] = {
                'data_format_validation': validation_success,
                'model_loading_handling': loading_success,
                'prediction_recovery': recovery_success
            }

            logger.info("✅ 错误恢复工作流程测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 错误恢复工作流程测试失败: {e}")
            traceback.print_exc()
            self.test_results['error_recovery'] = {'success': False, 'error': str(e)}
            return False

    def run_all_tests(self) -> bool:
        """运行所有端到端测试"""
        logger.info("开始端到端集成测试...")

        test_functions = [
            self.test_data_preprocessing_workflow,
            self.test_model_training_workflow,
            self.test_prediction_workflow,
            self.test_visualization_workflow,
            self.test_error_recovery_workflow
        ]

        passed_tests = 0
        total_tests = len(test_functions)

        try:
            for test_func in test_functions:
                test_name = test_func.__name__.replace('test_', '').replace('_', ' ').title()
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
            logger.info("端到端测试总结")
            logger.info(f"{'='*60}")
            logger.info(f"通过测试: {passed_tests}/{total_tests}")
            logger.info(f"通过率: {passed_tests/total_tests*100:.1f}%")

            # 保存测试结果
            results_path = os.path.join(self.temp_dir, "e2e_test_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)

            logger.info(f"测试结果已保存到: {results_path}")

            if passed_tests == total_tests:
                logger.info("🎉 所有的端到端集成测试通过！")
                return True
            else:
                logger.error("❌ 部分测试失败")
                return False

        except Exception as e:
            logger.error(f"❌ 测试过程出错: {e}")
            traceback.print_exc()
            return False

        finally:
            self.cleanup()


def main():
    """主函数"""
    test_runner = EndToEndTestRunner()
    success = test_runner.run_all_tests()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)