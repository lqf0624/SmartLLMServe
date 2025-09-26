#!/usr/bin/env python3
"""
数据质量场景测试脚本

测试预测器在各种数据质量场景下的表现，包括缺失值、异常值、数据分布等。
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from predictor.workload_predictor import WorkloadPredictor, ModelType, PredictionHorizon
from predictor.ensemble_predictor import EnsemblePredictor, WeightStrategy
from predictor.data_preprocessor import DataPreprocessor, DataFormat

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityTestRunner:
    """数据质量测试运行器"""

    def __init__(self):
        self.test_results = {}
        self.scenarios = self._create_test_scenarios()

    def _create_base_data(self, size: int = 500) -> pd.DataFrame:
        """创建基础测试数据"""
        base_time = pd.Timestamp('2023-01-01')
        timestamps = [base_time + pd.Timedelta(milliseconds=i*100) for i in range(size)]

        # 生成稳定的基准数据
        t = np.arange(size)
        base_input = 50 + 10 * np.sin(2 * np.pi * t / 100) + np.random.normal(0, 3, size)
        base_output = base_input * np.random.uniform(2.5, 3.5, size)

        return pd.DataFrame({
            'arrival_time_ns': timestamps,
            'input_toks': np.maximum(5, base_input).astype(int),
            'output_toks': np.maximum(10, base_output).astype(int),
            'model_type': ['ChatGPT'] * size
        })

    def _create_test_scenarios(self) -> Dict[str, pd.DataFrame]:
        """创建各种测试场景"""
        scenarios = {}
        base_size = 500

        # 场景1: 完美数据（基准）
        scenarios['perfect_data'] = self._create_base_data(base_size)

        # 场景2: 随机缺失值
        data_missing = self._create_base_data(base_size)
        missing_indices = np.random.choice(base_size, base_size//10, replace=False)
        data_missing.loc[missing_indices, 'input_toks'] = np.nan
        scenarios['random_missing'] = data_missing

        # 场景3: 连续缺失值
        data_consecutive_missing = self._create_base_data(base_size)
        data_consecutive_missing.loc[100:120, 'input_toks'] = np.nan
        data_consecutive_missing.loc[100:120, 'output_toks'] = np.nan
        scenarios['consecutive_missing'] = data_consecutive_missing

        # 场景4: 异常值 - 极大值
        data_outliers_high = self._create_base_data(base_size)
        outlier_indices = np.random.choice(base_size, base_size//20, replace=False)
        data_outliers_high.loc[outlier_indices, 'input_toks'] = np.random.randint(500, 1000, len(outlier_indices))
        scenarios['high_outliers'] = data_outliers_high

        # 场景5: 异常值 - 极小值
        data_outliers_low = self._create_base_data(base_size)
        outlier_indices = np.random.choice(base_size, base_size//20, replace=False)
        data_outliers_low.loc[outlier_indices, 'input_toks'] = np.random.randint(0, 5, len(outlier_indices))
        scenarios['low_outliers'] = data_outliers_low

        # 场景6: 时间戳异常
        data_time_anomalies = self._create_base_data(base_size)
        data_time_anomalies.loc[200:210, 'arrival_time_ns'] = None
        data_time_anomalies.loc[300, 'arrival_time_ns'] = pd.Timestamp('2022-01-01')  # 过去的时间
        scenarios['time_anomalies'] = data_time_anomalies

        # 场景7: 数据量很少
        scenarios['small_dataset'] = self._create_base_data(50)

        # 场景8: 数据分布偏斜
        data_skewed = self._create_base_data(base_size)
        # 创建长尾分布
        skewed_values = np.random.exponential(20, base_size)
        data_skewed['input_toks'] = np.maximum(5, skewed_values).astype(int)
        data_skewed['output_toks'] = (data_skewed['input_toks'] * np.random.uniform(2.5, 3.5, base_size)).astype(int)
        scenarios['skewed_distribution'] = data_skewed

        # 场景9: 高波动性
        data_volatile = self._create_base_data(base_size)
        volatility = np.random.normal(0, 30, base_size)
        data_volatile['input_toks'] = np.maximum(5, data_volatile['input_toks'] + volatility).astype(int)
        data_volatile['output_toks'] = (data_volatile['input_toks'] * np.random.uniform(2.5, 3.5, base_size)).astype(int)
        scenarios['high_volatility'] = data_volatile

        # 场景10: 多种问题混合
        data_mixed = self._create_base_data(base_size)
        # 添加缺失值
        mixed_missing = np.random.choice(base_size, base_size//15, replace=False)
        data_mixed.loc[mixed_missing, 'input_toks'] = np.nan
        # 添加异常值
        mixed_outliers = np.random.choice(base_size, base_size//25, replace=False)
        data_mixed.loc[mixed_outliers, 'input_toks'] = np.random.randint(400, 800, len(mixed_outliers))
        # 添加时间异常
        data_mixed.loc[150:155, 'arrival_time_ns'] = None
        scenarios['mixed_issues'] = data_mixed

        return scenarios

    def test_scenario_compatibility(self, scenario_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """测试单个场景的兼容性"""
        logger.info(f"测试场景: {scenario_name}")

        results = {
            'scenario_name': scenario_name,
            'data_size': len(data),
            'missing_values': data.isnull().sum().sum(),
            'zero_values': len(data[(data['input_toks'] <= 0) | (data['output_toks'] <= 0)]),
            'negative_values': len(data[(data['input_toks'] < 0) | (data['output_toks'] < 0)]),
            'outliers': self._detect_outliers(data),
            'time_issues': self._detect_time_issues(data)
        }

        return results

    def _detect_outliers(self, data: pd.DataFrame) -> int:
        """检测异常值"""
        Q1 = data['input_toks'].quantile(0.25)
        Q3 = data['input_toks'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return len(data[(data['input_toks'] < lower_bound) | (data['input_toks'] > upper_bound)])

    def _detect_time_issues(self, data: pd.DataFrame) -> int:
        """检测时间问题"""
        issues = 0
        if data['arrival_time_ns'].isnull().any():
            issues += data['arrival_time_ns'].isnull().sum()
        # 检查时间戳是否单调递增
        try:
            time_diffs = data['arrival_time_ns'].diff().dropna()
            if (time_diffs < pd.Timedelta(0)).any():
                issues += (time_diffs < pd.Timedelta(0)).sum()
        except:
            issues += len(data)
        return issues

    def test_model_performance_on_scenario(self, scenario_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """测试模型在特定场景下的表现"""
        logger.info(f"测试模型在 {scenario_name} 场景下的表现")

        results = {}

        # 测试不同模型
        models_to_test = [
            (ModelType.LSTM, "LSTM"),
            (ModelType.DLINEAR, "DLinear"),
            (ModelType.ENSEMBLE, "Ensemble")
        ]

        for model_type, model_name in models_to_test:
            try:
                # 创建预测器
                predictor = WorkloadPredictor(
                    model_type=model_type,
                    prediction_horizon=PredictionHorizon.SHORT_TERM,
                    sequence_length=min(30, len(data)//2),
                    enable_ensemble=(model_type == ModelType.ENSEMBLE),
                    ensemble_strategy=WeightStrategy.STATIC,
                    device='cpu'
                )

                # 训练
                training_result = predictor.train(data, validation_split=0.2)

                # 预测
                if len(data) > 20:
                    historical_data = data.iloc[:-10]
                    prediction_result = predictor.predict(historical_data, steps=5)
                else:
                    prediction_result = predictor.predict(data, steps=3)

                results[model_name] = {
                    'training_success': training_result,
                    'prediction_success': True,
                    'prediction_count': len(prediction_result.predicted_requests),
                    'confidence': prediction_result.confidence,
                    'error': None
                }

                logger.info(f"  {model_name}: 置信度 {prediction_result.confidence:.3f}")

            except Exception as e:
                results[model_name] = {
                    'training_success': False,
                    'prediction_success': False,
                    'prediction_count': 0,
                    'confidence': 0.0,
                    'error': str(e)
                }
                logger.warning(f"  {model_name}: 失败 - {e}")

        return results

    def test_data_preprocessing_effectiveness(self, scenario_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """测试数据预处理的有效性"""
        logger.info(f"测试 {scenario_name} 的数据预处理效果")

        try:
            # 保存原始数据
            original_stats = {
                'size': len(data),
                'missing': data.isnull().sum().sum(),
                'min_input': data['input_toks'].min(),
                'max_input': data['input_toks'].max(),
                'mean_input': data['input_toks'].mean(),
                'std_input': data['input_toks'].std()
            }

            # 数据预处理
            preprocessor = DataPreprocessor()

            # 保存临时文件
            temp_file = f"temp_{scenario_name}.csv"
            data.to_csv(temp_file, index=False)

            try:
                processed_data = preprocessor.load_and_preprocess(
                    temp_file,
                    DataFormat.CSV,
                    target_column='input_toks'
                )

                processed_stats = {
                    'size': len(processed_data),
                    'missing': processed_data.isnull().sum().sum(),
                    'min_input': processed_data['input_toks'].min(),
                    'max_input': processed_data['input_toks'].max(),
                    'mean_input': processed_data['input_toks'].mean(),
                    'std_input': processed_data['input_toks'].std()
                }

                # 清理临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)

                return {
                    'preprocessing_success': True,
                    'original_stats': original_stats,
                    'processed_stats': processed_stats,
                    'improvements': {
                        'missing_reduction': original_stats['missing'] - processed_stats['missing'],
                        'size_change': processed_stats['size'] - original_stats['size']
                    }
                }

            except Exception as e:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                return {
                    'preprocessing_success': False,
                    'error': str(e)
                }

        except Exception as e:
            return {
                'preprocessing_success': False,
                'error': str(e)
            }

    def generate_scenario_analysis_report(self) -> str:
        """生成场景分析报告"""
        report = []
        report.append("# 数据质量场景测试报告")
        report.append(f"测试时间: {pd.Timestamp.now()}")
        report.append("")
        report.append("## 测试场景概览")
        report.append("")

        for scenario_name, scenario_results in self.test_results.items():
            report.append(f"### {scenario_name}")
            report.append("")

            # 场景基本信息
            if 'compatibility' in scenario_results:
                comp = scenario_results['compatibility']
                report.append("**数据特征:**")
                report.append(f"- 数据量: {comp['data_size']}")
                report.append(f"- 缺失值: {comp['missing_values']}")
                report.append(f"- 零/负值: {comp['zero_values']}")
                report.append(f"- 异常值: {comp['outliers']}")
                report.append(f"- 时间问题: {comp['time_issues']}")
                report.append("")

            # 预处理效果
            if 'preprocessing' in scenario_results:
                prep = scenario_results['preprocessing']
                if prep['preprocessing_success']:
                    orig = prep['original_stats']
                    proc = prep['processed_stats']
                    imp = prep['improvements']
                    report.append("**预处理效果:**")
                    report.append(f"- 缺失值减少: {imp['missing_reduction']}")
                    report.append(f"- 数据量变化: {imp['size_change']}")
                    report.append(f"- 处理后数据范围: [{proc['min_input']}, {proc['max_input']}]")
                    report.append("")
                else:
                    report.append("**预处理效果:** 失败")
                    report.append(f"- 错误: {prep['error']}")
                    report.append("")

            # 模型表现
            if 'model_performance' in scenario_results:
                report.append("**模型表现:**")
                for model_name, perf in scenario_results['model_performance'].items():
                    if perf['prediction_success']:
                        report.append(f"- {model_name}: 置信度 {perf['confidence']:.3f}, {perf['prediction_count']} 个预测")
                    else:
                        report.append(f"- {model_name}: 失败 ({perf['error']})")
                report.append("")

            report.append("---")
            report.append("")

        return "\n".join(report)

    def run_all_scenario_tests(self) -> bool:
        """运行所有场景测试"""
        logger.info("开始数据质量场景测试...")

        passed_scenarios = 0
        total_scenarios = len(self.scenarios)

        try:
            for scenario_name, scenario_data in self.scenarios.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"测试场景: {scenario_name}")
                logger.info(f"{'='*60}")

                scenario_results = {}

                # 1. 场景兼容性测试
                compatibility_results = self.test_scenario_compatibility(scenario_name, scenario_data)
                scenario_results['compatibility'] = compatibility_results

                # 2. 数据预处理测试
                preprocessing_results = self.test_data_preprocessing_effectiveness(scenario_name, scenario_data)
                scenario_results['preprocessing'] = preprocessing_results

                # 3. 模型表现测试
                model_results = self.test_model_performance_on_scenario(scenario_name, scenario_data)
                scenario_results['model_performance'] = model_results

                self.test_results[scenario_name] = scenario_results

                # 评估场景是否通过
                successful_models = [m for m in model_results.values() if m['prediction_success']]
                if len(successful_models) > 0:
                    logger.info(f"✅ {scenario_name} 场景测试通过")
                    passed_scenarios += 1
                else:
                    logger.warning(f"❌ {scenario_name} 场景测试失败")

            logger.info(f"\n{'='*60}")
            logger.info("数据质量场景测试总结")
            logger.info(f"{'='*60}")
            logger.info(f"通过场景: {passed_scenarios}/{total_scenarios}")
            logger.info(f"通过率: {passed_scenarios/total_scenarios*100:.1f}%")

            # 生成报告
            report = self.generate_scenario_analysis_report()
            report_path = "data_quality_scenario_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"场景分析报告已保存到: {report_path}")

            if passed_scenarios >= total_scenarios * 0.8:  # 80%通过率
                logger.info("🎉 数据质量场景测试基本通过！")
                return True
            else:
                logger.error("❌ 数据质量场景测试通过率过低")
                return False

        except Exception as e:
            logger.error(f"❌ 场景测试过程出错: {e}")
            traceback.print_exc()
            return False

    def identify_data_quality_patterns(self) -> Dict[str, Any]:
        """识别数据质量模式"""
        logger.info("分析数据质量模式...")

        patterns = {
            'failure_patterns': [],
            'success_patterns': [],
            'critical_issues': [],
            'recommendations': []
        }

        for scenario_name, results in self.test_results.items():
            # 分析失败模式
            if 'model_performance' in results:
                failed_models = [model for model, perf in results['model_performance'].items() if not perf['prediction_success']]
                if failed_models:
                    patterns['failure_patterns'].append({
                        'scenario': scenario_name,
                        'failed_models': failed_models,
                        'reason': 'Model training/prediction failed'
                    })

                # 分析成功模式
                successful_models = [model for model, perf in results['model_performance'].items() if perf['prediction_success']]
                if successful_models:
                    patterns['success_patterns'].append({
                        'scenario': scenario_name,
                        'successful_models': successful_models,
                        'avg_confidence': np.mean([results['model_performance'][model]['confidence'] for model in successful_models])
                    })

            # 分析关键问题
            if 'compatibility' in results:
                comp = results['compatibility']
                if comp['missing_values'] > len(results['compatibility']) * 0.3:  # 30%缺失值
                    patterns['critical_issues'].append({
                        'scenario': scenario_name,
                        'issue': 'High missing values',
                        'severity': 'High'
                    })
                if comp['outliers'] > len(results['compatibility']) * 0.1:  # 10%异常值
                    patterns['critical_issues'].append({
                        'scenario': scenario_name,
                        'issue': 'High outlier count',
                        'severity': 'Medium'
                    })

        # 生成建议
        if patterns['critical_issues']:
            patterns['recommendations'].append("建议实现更强的异常值检测和处理机制")
        if any('missing' in issue['issue'] for issue in patterns['critical_issues']):
            patterns['recommendations'].append("建议改进缺失值填充策略")
        patterns['recommendations'].append("建议在数据预处理阶段增加数据质量评估")

        return patterns


def main():
    """主函数"""
    test_runner = DataQualityTestRunner()
    success = test_runner.run_all_scenario_tests()

    if success:
        # 分析数据质量模式
        patterns = test_runner.identify_data_quality_patterns()
        logger.info("数据质量模式分析完成")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)