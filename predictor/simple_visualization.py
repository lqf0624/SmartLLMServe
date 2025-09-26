"""
简化版可视化模块，专注于预测器核心功能

只提供基础的预测对比和指标计算功能：
- 历史窗口、预测窗口、实际值的对比
- 基础评估指标（MSE、MAE、RMSE、MAPE）
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from .workload_predictor import PredictionResult

logger = logging.getLogger(__name__)

# 设置样式
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SimplePredictionVisualizer:
    """
    简化版预测结果可视化工具

    专注于核心预测功能展示：
    - 历史数据窗口
    - 预测数据窗口
    - 实际值对比（如果有）
    - 基础评估指标
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """初始化可视化器"""
        self.figsize = figsize
        self.dpi = dpi
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def plot_prediction_comparison(self,
                                 historical_data: pd.DataFrame,
                                 prediction_result: PredictionResult,
                                 model_name: str = "Prediction Model",
                                 save_path: Optional[str] = None,
                                 show_plot: bool = True) -> Dict[str, float]:
        """
        绘制历史窗口、预测窗口和实际值的对比图

        Args:
            historical_data: 历史数据
            prediction_result: 预测结果
            model_name: 模型名称
            save_path: 保存路径
            show_plot: 是否显示图表

        Returns:
            基础评估指标字典
        """
        logger.info(f"创建预测对比图 - 模型: {model_name}")

        # 准备数据
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)

        # 上图：历史窗口对比
        if len(hist_rates) > 0:
            ax1.plot(hist_rates.index, hist_rates.values,
                    'o-', label='Historical Data', color=self.colors[0], linewidth=2, markersize=4)

            # 在历史数据末尾添加预测数据
            if len(pred_rates) > 0:
                pred_start_idx = len(hist_rates) - len(pred_rates)
                if pred_start_idx >= 0:
                    hist_for_comparison = hist_rates.iloc[pred_start_idx:]
                    pred_aligned = pred_rates.iloc[:len(hist_for_comparison)]

                    if len(pred_aligned) > 0:
                        ax1.plot(hist_for_comparison.index, pred_aligned.values,
                                's--', label='Predicted Values', color=self.colors[1], linewidth=2, markersize=4)

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Request Rate (req/min)')
        ax1.set_title(f'{model_name} - Historical Window Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 下图：预测窗口详情
        if len(pred_rates) > 0:
            ax2.plot(pred_rates.index, pred_rates.values,
                    's-', label='Prediction Window', color=self.colors[1], linewidth=2, markersize=6)

            # 如果有实际值用于对比
            # 这里预留未来实际值对比的接口

        ax2.set_xlabel('Time')
        ax2.set_ylabel('Request Rate (req/min)')
        ax2.set_title(f'{model_name} - Prediction Window Detail')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # 计算并显示基础指标
        metrics = self._calculate_basic_metrics(hist_rates, pred_rates)

        # 在图表上添加指标信息
        metrics_text = f"""Basic Metrics:
MAE: {metrics['MAE']:.4f}
MSE: {metrics['MSE']:.4f}
RMSE: {metrics['RMSE']:.4f}
MAPE: {metrics['MAPE']:.2f}%
R²: {metrics['R2']:.4f}
Data Points: {metrics['data_points']}"""

        fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"预测对比图已保存到: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        logger.info(f"基础指标计算完成: {metrics}")
        return metrics

    def plot_multi_model_comparison(self,
                                  prediction_results: Dict[str, PredictionResult],
                                  historical_data: pd.DataFrame,
                                  model_names: Optional[Dict[str, str]] = None,
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True) -> Dict[str, Dict[str, float]]:
        """
        多模型预测对比

        Args:
            prediction_results: 多个预测结果
            historical_data: 历史数据
            model_names: 模型名称映射
            save_path: 保存路径
            show_plot: 是否显示图表

        Returns:
            各模型的指标字典
        """
        logger.info(f"创建多模型对比图 - 模型数量: {len(prediction_results)}")

        if model_names is None:
            model_names = {key: f"Model_{key}" for key in prediction_results.keys()}

        # 准备历史数据
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)

        # 上图：历史数据
        if len(hist_rates) > 0:
            ax1.plot(hist_rates.index, hist_rates.values,
                    'o-', label='Historical Data', color='black', linewidth=2, markersize=4)

        # 下图：各模型预测对比
        all_metrics = {}
        for i, (model_key, pred_result) in enumerate(prediction_results.items()):
            pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                         for req in pred_result.predicted_requests]
            pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

            if len(pred_rates) > 0:
                color = self.colors[i % len(self.colors)]
                model_name = model_names.get(model_key, model_key)

                ax2.plot(pred_rates.index, pred_rates.values,
                        's-', label=model_name, color=color, linewidth=2, markersize=4)

                # 计算指标
                metrics = self._calculate_basic_metrics(hist_rates, pred_rates)
                all_metrics[model_key] = metrics

        # 设置图表
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Request Rate (req/min)')
        ax1.set_title('Historical Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        ax2.set_xlabel('Time')
        ax2.set_ylabel('Request Rate (req/min)')
        ax2.set_title('Multi-Model Prediction Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"多模型对比图已保存到: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        logger.info(f"多模型对比完成，指标: {all_metrics}")
        return all_metrics

    def generate_simple_report(self,
                             prediction_results: Dict[str, PredictionResult],
                             historical_data: pd.DataFrame,
                             model_names: Optional[Dict[str, str]] = None,
                             save_path: Optional[str] = None) -> str:
        """
        生成简单的文本报告

        Args:
            prediction_results: 预测结果
            historical_data: 历史数据
            model_names: 模型名称
            save_path: 保存路径

        Returns:
            报告文本
        """
        logger.info("生成简单预测报告")

        if model_names is None:
            model_names = {key: f"Model_{key}" for key in prediction_results.keys()}

        # 准备历史数据
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PREDICTION PERFORMANCE REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Data Points: {len(historical_data)}")
        report_lines.append(f"Models Tested: {len(prediction_results)}")
        report_lines.append("")

        # 为每个模型生成报告
        for model_key, pred_result in prediction_results.items():
            model_name = model_names.get(model_key, model_key)
            report_lines.append(f"Model: {model_name}")
            report_lines.append("-" * 40)

            # 计算预测数据
            pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                         for req in pred_result.predicted_requests]
            pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

            # 计算指标
            metrics = self._calculate_basic_metrics(hist_rates, pred_rates)

            report_lines.append(f"  Prediction Confidence: {pred_result.confidence:.3f}")
            report_lines.append(f"  Predicted Requests: {len(pred_result.predicted_requests)}")
            report_lines.append(f"  Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
            report_lines.append(f"  Mean Squared Error (MSE): {metrics['MSE']:.4f}")
            report_lines.append(f"  Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
            report_lines.append(f"  Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
            report_lines.append(f"  R-squared (R²): {metrics['R2']:.4f}")
            report_lines.append(f"  Data Points Compared: {metrics['data_points']}")
            report_lines.append("")

        # 模型对比总结
        if len(prediction_results) > 1:
            report_lines.append("Model Comparison Summary")
            report_lines.append("-" * 40)

            best_mae = min(metrics['MAE'] for metrics in [self._calculate_basic_metrics(
                hist_rates, self._calculate_predicted_rates(
                    [pd.to_datetime(req['arrival_time_ns'], unit='ns') for req in result.predicted_requests]
                )) for result in prediction_results.values()
            ])

            best_models = [model_names[key] for key, result in prediction_results.items()
                          if self._calculate_basic_metrics(hist_rates,
                                self._calculate_predicted_rates(
                                    [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                                     for req in result.predicted_requests]
                                ))['MAE'] == best_mae]

            report_lines.append(f"Best Model by MAE: {', '.join(best_models)}")
            report_lines.append(f"Best MAE: {best_mae:.4f}")

        report_lines.append("=" * 60)

        report_text = "\n".join(report_lines)

        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"预测报告已保存到: {save_path}")

        return report_text

    def _calculate_request_rates(self, data: pd.DataFrame, window_seconds: int = 60) -> pd.Series:
        """计算请求速率"""
        if len(data) == 0:
            return pd.Series(dtype=float)

        # 计算时间窗口内的请求数量
        rates = data.resample(f'{window_seconds}s', on='arrival_time_ns').size()
        return rates

    def _calculate_predicted_rates(self, pred_times: List[datetime], window_seconds: int = 60) -> pd.Series:
        """计算预测请求速率"""
        if len(pred_times) == 0:
            return pd.Series(dtype=float)

        # 创建预测时间序列
        pred_df = pd.DataFrame({'arrival_time_ns': pred_times})
        pred_df['arrival_time_ns'] = pd.to_datetime(pred_df['arrival_time_ns'])

        # 计算时间窗口内的请求数量
        rates = pred_df.resample(f'{window_seconds}s', on='arrival_time_ns').size()
        return rates

    def _calculate_basic_metrics(self, hist_rates: pd.Series, pred_rates: pd.Series) -> Dict[str, float]:
        """计算基础评估指标"""
        if len(hist_rates) == 0 or len(pred_rates) == 0:
            return {
                'MAE': float('nan'),
                'MSE': float('nan'),
                'RMSE': float('nan'),
                'MAPE': float('nan'),
                'R2': float('nan'),
                'data_points': 0
            }

        # 对齐数据
        if len(pred_rates) <= len(hist_rates):
            # 使用历史数据的最后部分进行对比
            hist_aligned = hist_rates.iloc[-len(pred_rates):]
            pred_aligned = pred_rates
        else:
            # 预测数据比历史数据长，只比较重叠部分
            overlap_len = len(hist_rates)
            hist_aligned = hist_rates
            pred_aligned = pred_rates.iloc[:overlap_len]

        # 确保数据长度一致
        min_len = min(len(hist_aligned), len(pred_aligned))
        if min_len == 0:
            return {
                'MAE': float('nan'),
                'MSE': float('nan'),
                'RMSE': float('nan'),
                'MAPE': float('nan'),
                'R2': float('nan'),
                'data_points': 0
            }

        hist_values = hist_aligned.iloc[:min_len].values
        pred_values = pred_aligned.iloc[:min_len].values

        # 计算各种指标
        errors = hist_values - pred_values
        abs_errors = np.abs(errors)

        mae = np.mean(abs_errors)
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)

        # MAPE - 避免除零
        mape = np.mean(abs_errors / np.maximum(np.abs(hist_values), 1e-8)) * 100

        # R-squared
        if np.var(hist_values) > 0:
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((hist_values - np.mean(hist_values)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = float('nan')

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'data_points': min_len
        }