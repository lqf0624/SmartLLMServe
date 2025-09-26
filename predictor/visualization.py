"""
Visualization module for workload prediction results.

This module provides comprehensive visualization capabilities for:
- Historical vs predicted workload comparison
- Resource requirement analysis
- Burst pattern visualization
- Confidence intervals and uncertainty
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from .workload_predictor import PredictionResult

logger = logging.getLogger(__name__)

# 添加统计相关导入
try:
    import scipy.stats as stats
    import statsmodels.api as sm
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    logger.warning("scipy/statsmodels not available, some statistical metrics will be disabled")

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PredictionVisualizer:
    """
    Comprehensive visualization toolkit for workload prediction results.

    Provides multiple types of plots for different analysis needs:
    - Time series comparison
    - Resource requirement forecasting
    - Burst pattern analysis
    - Model performance visualization
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        logger.info("PredictionVisualizer initialized")

    def plot_prediction_vs_actual(self,
                               prediction_result: PredictionResult,
                               historical_data: pd.DataFrame,
                               save_path: Optional[str] = None,
                               show_plot: bool = True) -> None:
        """
        Create detailed prediction vs actual comparison plot.

        Args:
            prediction_result: Prediction result to visualize
            historical_data: Historical data for comparison
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        fig.suptitle('Prediction vs Actual Analysis', fontsize=16, fontweight='bold')

        # 1. Request Rate: Actual vs Predicted
        self._plot_request_rate_comparison(axes[0, 0], historical_data, prediction_result)

        # 2. Token Count: Actual vs Predicted
        self._plot_token_comparison(axes[0, 1], historical_data, prediction_result)

        # 3. Cumulative Workload
        self._plot_cumulative_workload(axes[0, 2], historical_data, prediction_result)

        # 4. Prediction Error Analysis
        self._plot_prediction_errors(axes[1, 0], historical_data, prediction_result)

        # 5. Time Series with Confidence Intervals
        self._plot_time_series_with_confidence(axes[1, 1], historical_data, prediction_result)

        # 6. Resource Utilization Forecast
        self._plot_resource_forecast_comparison(axes[1, 2], historical_data, prediction_result)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Prediction overview saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_time_series_comparison(self,
                                  historical_data: pd.DataFrame,
                                  prediction_result: PredictionResult,
                                  window_minutes: int = 30,
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True) -> None:
        """
        Plot historical vs predicted time series.

        Args:
            historical_data: Historical request data
            prediction_result: Prediction result
            window_minutes: Time window to show in minutes
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Convert historical data to time series
        if historical_data is not None and not historical_data.empty:
            hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
            hist_counts = self._create_time_series(hist_times, window_minutes)

            # Plot historical data
            ax.plot(hist_counts.index, hist_counts.values,
                   label='Historical', color=self.colors[0], linewidth=2, alpha=0.8)

        # Convert predicted data to time series
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_counts = self._create_time_series(pd.DatetimeIndex(pred_times), window_minutes)

        # Plot predicted data
        ax.plot(pred_counts.index, pred_counts.values,
               label='Predicted', color=self.colors[1], linewidth=2, linestyle='--')

        # Add confidence interval
        if len(pred_counts) > 0:
            confidence_lower = pred_counts.values * (1 - prediction_result.confidence * 0.3)
            confidence_upper = pred_counts.values * (1 + prediction_result.confidence * 0.3)
            ax.fill_between(pred_counts.index, confidence_lower, confidence_upper,
                          alpha=0.2, color=self.colors[1], label=f'Confidence ({prediction_result.confidence:.2f})')

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Request Count', fontsize=12)
        ax.set_title('Historical vs Predicted Workload', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Time series comparison saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_resource_forecast(self,
                             prediction_result: PredictionResult,
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
        """
        Plot resource requirement forecast.

        Args:
            prediction_result: Prediction result
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        fig.suptitle('Resource Requirement Forecast', fontsize=16, fontweight='bold')

        # Extract resource information
        resource_req = prediction_result.get_resource_requirements()
        timing_info = prediction_result.get_timing_info()
        burst_analysis = prediction_result.get_burst_analysis()

        # 1. Memory usage over time
        self._plot_memory_usage(axes[0, 0], prediction_result)

        # 2. Compute units over time
        self._plot_compute_usage(axes[0, 1], prediction_result)

        # 3. Token distribution
        self._plot_token_forecast(axes[1, 0], prediction_result)

        # 4. Resource summary
        self._plot_resource_summary(axes[1, 1], resource_req, burst_analysis)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Resource forecast saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_burst_patterns(self,
                           historical_data: Optional[pd.DataFrame] = None,
                           prediction_result: Optional[PredictionResult] = None,
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
        """
        Visualize burst patterns in historical and predicted data.

        Args:
            historical_data: Historical data
            prediction_result: Prediction result
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        fig.suptitle('Burst Pattern Analysis', fontsize=16, fontweight='bold')

        # Historical burst patterns
        if historical_data is not None and not historical_data.empty:
            self._plot_historical_bursts(axes[0, 0], historical_data)
            self._plot_inter_arrival_times(axes[0, 1], historical_data)

        # Predicted burst patterns
        if prediction_result is not None:
            self._plot_predicted_bursts(axes[1, 0], prediction_result)
            self._plot_burst_intensity(axes[1, 1], prediction_result)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Burst patterns saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_model_performance(self,
                             training_history: Dict[str, List[float]],
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
        """
        Plot model training performance.

        Args:
            training_history: Training history from model
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        fig.suptitle('Model Training Performance', fontsize=16, fontweight='bold')

        # Loss curves
        if 'train_loss' in training_history and 'val_loss' in training_history:
            epochs = range(1, len(training_history['train_loss']) + 1)

            axes[0].plot(epochs, training_history['train_loss'],
                        label='Training Loss', color=self.colors[0], linewidth=2)
            axes[0].plot(epochs, training_history['val_loss'],
                        label='Validation Loss', color=self.colors[1], linewidth=2)

            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training vs Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Final metrics
        if training_history:
            metrics = ['Train Loss', 'Val Loss']
            values = [
                training_history['train_loss'][-1] if training_history['train_loss'] else 0,
                training_history['val_loss'][-1] if training_history['val_loss'] else 0
            ]

            bars = axes[1].bar(metrics, values, color=[self.colors[0], self.colors[1]])
            axes[1].set_ylabel('Loss Value')
            axes[1].set_title('Final Model Performance')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.4f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Model performance saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    # Helper methods for specific plots
    def _plot_request_rate_comparison(self, ax, historical_data: pd.DataFrame,
                                   prediction_result: PredictionResult):
        """Plot request rate comparison between actual and predicted."""
        # Calculate historical request rates
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data)

        # Calculate predicted request rates
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times)

        # Plot historical rates
        ax.plot(hist_rates.index, hist_rates.values, 'o-',
               label='Actual Rate', color=self.colors[0], linewidth=2, markersize=4)

        # Plot predicted rates
        if len(pred_rates) > 0:
            ax.plot(pred_rates.index, pred_rates.values, 's--',
                   label='Predicted Rate', color=self.colors[1], linewidth=2, markersize=4)

            # Add prediction confidence band
            confidence_band = pred_rates.values * prediction_result.confidence
            ax.fill_between(pred_rates.index,
                           pred_rates.values - confidence_band,
                           pred_rates.values + confidence_band,
                           alpha=0.2, color=self.colors[1],
                           label=f'Confidence ({prediction_result.confidence:.2f})')

        ax.set_xlabel('Time')
        ax.set_ylabel('Requests per Second')
        ax.set_title('Request Rate: Actual vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _calculate_request_rates(self, data: pd.DataFrame, window_seconds: int = 60) -> pd.Series:
        """Calculate request rates from data."""
        if data.empty:
            return pd.Series()

        times = pd.to_datetime(data['arrival_time_ns'], unit='ns')
        # Group by time windows and count requests
        return times.dt.floor(f'{window_seconds}s').value_counts().sort_index()

    def _calculate_predicted_rates(self, pred_times: List[pd.Timestamp],
                                 window_seconds: int = 60) -> pd.Series:
        """Calculate predicted request rates."""
        if not pred_times:
            return pd.Series()

        pred_series = pd.Series(pred_times)
        return pred_series.dt.floor(f'{window_seconds}s').value_counts().sort_index()

    def _plot_token_comparison(self, ax, historical_data: pd.DataFrame,
                           prediction_result: PredictionResult):
        """Plot token count comparison between actual and predicted."""
        # Calculate historical token statistics
        hist_tokens = historical_data['input_toks'] + historical_data['output_toks']

        # Calculate predicted token statistics
        pred_tokens = [req['input_tokens'] + req['output_tokens']
                      for req in prediction_result.predicted_requests]

        # Create statistical comparison
        stats_df = pd.DataFrame({
            'Actual': hist_tokens,
            'Predicted': pred_tokens
        })

        # Plot box plots for comparison
        positions = [1, 2]
        box_data = [stats_df['Actual'].dropna(), stats_df['Predicted'].dropna()]

        box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True,
                            labels=['Actual', 'Predicted'])

        # Color the boxes
        colors = [self.colors[0], self.colors[1]]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add statistical annotations
        actual_mean = stats_df['Actual'].mean()
        pred_mean = stats_df['Predicted'].mean()
        actual_std = stats_df['Actual'].std()
        pred_std = stats_df['Predicted'].std()

        ax.text(positions[0], max(stats_df['Actual'].max(), stats_df['Predicted'].max()) * 0.95,
               f'Mean: {actual_mean:.0f}\\nStd: {actual_std:.0f}',
               ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.text(positions[1], max(stats_df['Actual'].max(), stats_df['Predicted'].max()) * 0.95,
               f'Mean: {pred_mean:.0f}\\nStd: {pred_std:.0f}',
               ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_ylabel('Total Tokens per Request')
        ax.set_title('Token Count: Actual vs Predicted')
        ax.grid(True, alpha=0.3)

    def _plot_resource_requirements(self, ax, prediction_result: PredictionResult):
        """Plot resource requirements over time."""
        times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                for req in prediction_result.predicted_requests]
        memory = [req['memory_requirement_mb'] for req in prediction_result.predicted_requests]
        compute = [req['estimated_compute_time'] for req in prediction_result.predicted_requests]

        ax2 = ax.twinx()

        line1 = ax.plot(times, memory, 'o-', label='Memory (MB)', color=self.colors[0], markersize=4)
        line2 = ax2.plot(times, compute, 's-', label='Compute Time (ms)', color=self.colors[1], markersize=4)

        ax.set_xlabel('Time')
        ax.set_ylabel('Memory (MB)', color=self.colors[0])
        ax2.set_ylabel('Compute Time (ms)', color=self.colors[1])
        ax.set_title('Resource Requirements Over Time')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_burst_analysis(self, ax, prediction_result: PredictionResult):
        """Plot burst analysis."""
        burst_info = prediction_result.get_burst_analysis()

        metrics = ['Intensity', 'Avg Interval', 'Max Interval']
        values = [
            burst_info['intensity'],
            burst_info['avg_interval_ms'],
            burst_info['max_interval_ms']
        ]

        bars = ax.bar(metrics, values, color=[self.colors[0], self.colors[1], self.colors[2]])
        ax.set_ylabel('Value')
        ax.set_title(f'Burst Analysis (Level: {burst_info["burst_level"]})')

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}', ha='center', va='bottom')

        ax.grid(True, alpha=0.3)

    def _create_time_series(self, times: pd.DatetimeIndex, window_minutes: int) -> pd.Series:
        """Create time series from timestamps."""
        if len(times) == 0:
            return pd.Series()

        # Create time bins
        start_time = times.min()
        end_time = times.max()

        if window_minutes > 0:
            bins = pd.date_range(start=start_time, end=end_time,
                               freq=f'{window_minutes}min')
            return pd.cut(times, bins=bins).value_counts().sort_index()
        else:
            return pd.Series([len(times)], index=[start_time])

    def _plot_memory_usage(self, ax, prediction_result: PredictionResult):
        """Plot memory usage over time."""
        times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                for req in prediction_result.predicted_requests]
        memory = [req['memory_requirement_mb'] for req in prediction_result.predicted_requests]

        ax.plot(times, memory, 'o-', color=self.colors[0], markersize=4)
        ax.fill_between(times, memory, alpha=0.3, color=self.colors[0])

        ax.set_xlabel('Time')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage Forecast')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_compute_usage(self, ax, prediction_result: PredictionResult):
        """Plot compute usage over time."""
        times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                for req in prediction_result.predicted_requests]
        compute = [req['estimated_compute_time'] for req in prediction_result.predicted_requests]

        ax.plot(times, compute, 's-', color=self.colors[1], markersize=4)
        ax.fill_between(times, compute, alpha=0.3, color=self.colors[1])

        ax.set_xlabel('Time')
        ax.set_ylabel('Compute Time (ms)')
        ax.set_title('Compute Usage Forecast')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_token_forecast(self, ax, prediction_result: PredictionResult):
        """Plot token forecast."""
        times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                for req in prediction_result.predicted_requests]
        input_tokens = [req['input_tokens'] for req in prediction_result.predicted_requests]
        output_tokens = [req['output_tokens'] for req in prediction_result.predicted_requests]

        ax.plot(times, input_tokens, 'o-', label='Input Tokens', color=self.colors[0], markersize=4)
        ax.plot(times, output_tokens, 's-', label='Output Tokens', color=self.colors[1], markersize=4)

        ax.set_xlabel('Time')
        ax.set_ylabel('Token Count')
        ax.set_title('Token Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_resource_summary(self, ax, resource_req: Dict[str, float], burst_analysis: Dict[str, Any]):
        """Plot resource summary."""
        metrics = ['Memory (MB)', 'Compute Units', 'Total Tokens', 'Request Count']
        values = [
            resource_req['memory_mb'],
            resource_req['compute_units'],
            resource_req['total_tokens'] / 1000,  # Scale down
            resource_req['request_count']
        ]

        bars = ax.bar(metrics, values, color=self.colors[:4])
        ax.set_ylabel('Value (tokens in thousands)')
        ax.set_title('Resource Summary')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}', ha='center', va='bottom')

        ax.grid(True, alpha=0.3)

    def _plot_historical_bursts(self, ax, historical_data: pd.DataFrame):
        """Plot historical burst patterns."""
        if 'burst_pattern' not in historical_data.columns:
            ax.text(0.5, 0.5, 'No burst pattern data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Historical Burst Patterns')
            return

        pattern_counts = historical_data['burst_pattern'].value_counts()

        bars = ax.bar(pattern_counts.index, pattern_counts.values,
                     color=[self.colors[0], self.colors[1], self.colors[2]])
        ax.set_xlabel('Burst Pattern')
        ax.set_ylabel('Count')
        ax.set_title('Historical Burst Patterns')

        for bar, count in zip(bars, pattern_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{count}', ha='center', va='bottom')

        ax.grid(True, alpha=0.3)

    def _plot_inter_arrival_times(self, ax, historical_data: pd.DataFrame):
        """Plot inter-arrival time distribution."""
        times = historical_data['arrival_time_ns'].sort_values()
        inter_arrival = np.diff(times) / 1e9  # Convert to seconds

        ax.hist(inter_arrival, bins=30, alpha=0.7, color=self.colors[0], edgecolor='black')
        ax.set_xlabel('Inter-arrival Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Inter-arrival Time Distribution')
        ax.grid(True, alpha=0.3)

    def _plot_predicted_bursts(self, ax, prediction_result: PredictionResult):
        """Plot predicted burst patterns."""
        patterns = [req['burst_pattern'] for req in prediction_result.predicted_requests]
        pattern_counts = pd.Series(patterns).value_counts()

        bars = ax.bar(pattern_counts.index, pattern_counts.values,
                     color=[self.colors[1], self.colors[2], self.colors[3]])
        ax.set_xlabel('Burst Pattern')
        ax.set_ylabel('Count')
        ax.set_title('Predicted Burst Patterns')

        for bar, count in zip(bars, pattern_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{count}', ha='center', va='bottom')

        ax.grid(True, alpha=0.3)

    def _plot_burst_intensity(self, ax, prediction_result: PredictionResult):
        """Plot burst intensity over time."""
        burst_info = prediction_result.get_burst_analysis()

        # Create intensity timeline
        times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                for req in prediction_result.predicted_requests]

        # Calculate rolling intensity
        window_size = max(1, len(times) // 10)
        intensity_values = []
        time_points = []

        for i in range(window_size, len(times)):
            window_times = times[i-window_size:i]
            window_duration = (window_times[-1] - window_times[0]).total_seconds()
            if window_duration > 0:
                intensity = window_size / window_duration  # requests per second
                intensity_values.append(intensity)
                time_points.append(times[i])

        if intensity_values:
            ax.plot(time_points, intensity_values, 'o-', color=self.colors[2], markersize=4)
            ax.axhline(y=burst_info['intensity'], color='red', linestyle='--',
                       label=f'Avg Intensity: {burst_info["intensity"]:.1f}')

        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity (requests/sec)')
        ax.set_title('Burst Intensity Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def plot_comprehensive_comparison(self,
                                     historical_data: pd.DataFrame,
                                     prediction_result: PredictionResult,
                                     save_path: Optional[str] = None,
                                     show_plot: bool = True) -> None:
        """
        Create comprehensive actual vs predicted comparison plots.

        Args:
            historical_data: Historical actual data
            prediction_result: Prediction result
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        fig.suptitle('Comprehensive Actual vs Predicted Comparison', fontsize=16, fontweight='bold')

        # 1. Request rate comparison with confidence intervals
        self._plot_enhanced_request_rate_comparison(axes[0, 0], historical_data, prediction_result)

        # 2. Token distribution comparison
        self._plot_enhanced_token_comparison(axes[0, 1], historical_data, prediction_result)

        # 3. Cumulative workload comparison
        self._plot_cumulative_workload_comparison(axes[0, 2], historical_data, prediction_result)

        # 4. Time series with prediction intervals
        self._plot_time_series_with_intervals(axes[1, 0], historical_data, prediction_result)

        # 5. Error analysis
        self._plot_prediction_error_analysis(axes[1, 1], historical_data, prediction_result)

        # 6. Resource forecast comparison
        self._plot_resource_forecast_comparison(axes[1, 2], historical_data, prediction_result)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Comprehensive comparison saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _plot_enhanced_request_rate_comparison(self, ax, historical_data: pd.DataFrame,
                                            prediction_result: PredictionResult):
        """Enhanced request rate comparison with statistical analysis."""
        # Calculate historical rates
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=30)

        # Calculate predicted rates
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=30)

        # Validate data lengths
        if len(hist_rates) == 0:
            ax.text(0.5, 0.5, 'No historical data available',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Request Rate Comparison (No Data)')
            return

        # Plot actual rates
        ax.plot(hist_rates.index, hist_rates.values, 'o-',
               label='Actual Rate', color=self.colors[0], linewidth=2, markersize=4)

        # Plot predicted rates with confidence band
        if len(pred_rates) > 0:
            ax.plot(pred_rates.index, pred_rates.values, 's--',
                   label='Predicted Rate', color=self.colors[1], linewidth=2, markersize=4)

            # Add confidence band
            confidence_upper = pred_rates.values * (1 + (1 - prediction_result.confidence))
            confidence_lower = pred_rates.values * max(0, (1 - (1 - prediction_result.confidence)))

            ax.fill_between(pred_rates.index, confidence_lower, confidence_upper,
                           alpha=0.2, color=self.colors[1],
                           label=f'{int(prediction_result.confidence*100)}% Confidence')

        # Add statistical summary
        if len(hist_rates) > 0 and len(pred_rates) > 0:
            actual_mean = hist_rates.mean()
            pred_mean = pred_rates.mean()
            mape = abs(actual_mean - pred_mean) / max(actual_mean, 1e-6) * 100

            ax.text(0.02, 0.98, f'MAE: {abs(actual_mean - pred_mean):.2f}\nMAPE: {mape:.1f}%',
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Time')
        ax.set_ylabel('Requests per Second')
        ax.set_title('Request Rate: Actual vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_enhanced_token_comparison(self, ax, historical_data: pd.DataFrame,
                                      prediction_result: PredictionResult):
        """Enhanced token comparison with distribution analysis."""
        # Calculate token statistics
        hist_input_tokens = historical_data['input_toks']
        hist_output_tokens = historical_data['output_toks']
        hist_total_tokens = hist_input_tokens + hist_output_tokens

        pred_input_tokens = [req['input_tokens'] for req in prediction_result.predicted_requests]
        pred_output_tokens = [req['output_tokens'] for req in prediction_result.predicted_requests]
        pred_total_tokens = [inp + out for inp, out in zip(pred_input_tokens, pred_output_tokens)]

        # Create comparison DataFrame
        comparison_data = pd.DataFrame({
            'Actual_Total': hist_total_tokens,
            'Predicted_Total': pred_total_tokens,
            'Actual_Input': hist_input_tokens,
            'Predicted_Input': pred_input_tokens,
            'Actual_Output': hist_output_tokens,
            'Predicted_Output': pred_output_tokens
        })

        # Plot violin plots for distribution comparison
        positions = [1, 2, 3, 4]
        plot_data = [
            comparison_data['Actual_Total'].dropna(),
            comparison_data['Predicted_Total'].dropna(),
            comparison_data['Actual_Input'].dropna(),
            comparison_data['Predicted_Input'].dropna()
        ]

        parts = ax.violinplot(plot_data, positions=positions, showmeans=True, showmedians=True)

        # Color the violins
        colors = [self.colors[0], self.colors[1], self.colors[2], self.colors[3]]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)

        # Add statistical annotations
        actual_mean = comparison_data['Actual_Total'].mean()
        pred_mean = comparison_data['Predicted_Total'].mean()
        actual_std = comparison_data['Actual_Total'].std()
        pred_std = comparison_data['Predicted_Total'].std()

        ax.set_xticks(positions)
        ax.set_xticklabels(['Actual Total', 'Predicted Total', 'Actual Input', 'Predicted Input'])
        ax.set_ylabel('Token Count')
        ax.set_title('Token Distribution Comparison')
        ax.grid(True, alpha=0.3)

        # Add summary statistics
        stats_text = f'Actual: μ={actual_mean:.0f}, σ={actual_std:.0f}\nPredicted: μ={pred_mean:.0f}, σ={pred_std:.0f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _plot_cumulative_workload_comparison(self, ax, historical_data: pd.DataFrame,
                                           prediction_result: PredictionResult):
        """Plot cumulative workload comparison."""
        # Historical cumulative workload
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_sorted = historical_data.sort_values('arrival_time_ns')
        hist_cumulative = np.arange(1, len(hist_sorted) + 1)

        # Predicted cumulative workload
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_sorted = sorted(prediction_result.predicted_requests, key=lambda x: x['arrival_time_ns'])
        pred_cumulative = np.arange(1, len(pred_sorted) + 1)

        # Plot cumulative curves
        ax.plot(hist_times, hist_cumulative, 'o-', label='Actual Cumulative',
               color=self.colors[0], linewidth=2, markersize=3)
        ax.plot(pred_times, pred_cumulative, 's--', label='Predicted Cumulative',
               color=self.colors[1], linewidth=2, markersize=3)

        # Calculate and display divergence
        if len(hist_cumulative) > 0 and len(pred_cumulative) > 0:
            final_actual = hist_cumulative[-1]
            final_predicted = pred_cumulative[-1]
            divergence = abs(final_actual - final_predicted) / max(final_actual, 1) * 100

            ax.text(0.02, 0.98, f'Final Divergence: {divergence:.1f}%',
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Request Count')
        ax.set_title('Cumulative Workload Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_time_series_with_intervals(self, ax, historical_data: pd.DataFrame,
                                       prediction_result: PredictionResult):
        """Plot time series with prediction intervals."""
        # Create time bins for comparison
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        time_bins = pd.date_range(start=hist_times.min(), end=hist_times.max(), freq='1min')

        # Calculate actual request counts per bin
        hist_counts = pd.cut(hist_times, bins=time_bins).value_counts().sort_index()

        # Generate predicted timeline
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]

        # Create prediction intervals
        if len(pred_times) > 0:
            pred_bins = pd.date_range(start=min(pred_times), end=max(pred_times), freq='1min')
            pred_counts = pd.cut(pred_times, bins=pred_bins).value_counts().sort_index()

            # Plot actual counts
            ax.bar(hist_counts.index, hist_counts.values, width=0.8/len(time_bins),
                   alpha=0.7, label='Actual Requests', color=self.colors[0])

            # Plot predicted counts with confidence intervals
            pred_mean = pred_counts.mean()
            pred_std = pred_counts.std()

            ax.plot(pred_counts.index, pred_counts.values, 's--',
                   label='Predicted Requests', color=self.colors[1], markersize=4)

            # Add confidence bands
            upper_bound = pred_counts + pred_std
            lower_bound = pred_counts - pred_std
            ax.fill_between(pred_counts.index, lower_bound, upper_bound,
                           alpha=0.2, color=self.colors[1], label='±1 Std Dev')

        ax.set_xlabel('Time')
        ax.set_ylabel('Request Count')
        ax.set_title('Time Series with Prediction Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_prediction_error_analysis(self, ax, historical_data: pd.DataFrame,
                                      prediction_result: PredictionResult):
        """Plot prediction error analysis."""
        # Calculate errors for different metrics
        errors = []

        # Request rate error
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=30)
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=30)

        if len(hist_rates) > 0 and len(pred_rates) > 0:
            rate_mae = abs(hist_rates.mean() - pred_rates.mean())
            rate_mape = abs(hist_rates.mean() - pred_rates.mean()) / max(hist_rates.mean(), 1e-6) * 100
            errors.append(('Request Rate', rate_mae, rate_mape))

        # Token count error
        hist_tokens = historical_data['input_toks'] + historical_data['output_toks']
        pred_tokens = [req['input_tokens'] + req['output_tokens']
                      for req in prediction_result.predicted_requests]

        if len(hist_tokens) > 0 and len(pred_tokens) > 0:
            token_mae = abs(hist_tokens.mean() - np.mean(pred_tokens))
            token_mape = abs(hist_tokens.mean() - np.mean(pred_tokens)) / max(hist_tokens.mean(), 1e-6) * 100
            errors.append(('Token Count', token_mae, token_mape))

        # Create error visualization
        if errors:
            metrics = [e[0] for e in errors]
            mae_values = [e[1] for e in errors]
            mape_values = [e[2] for e in errors]

            x = np.arange(len(metrics))
            width = 0.35

            bars1 = ax.bar(x - width/2, mae_values, width, label='MAE', color=self.colors[0], alpha=0.8)
            bars2 = ax.bar(x + width/2, mape_values, width, label='MAPE (%)', color=self.colors[1], alpha=0.8)

            ax.set_xlabel('Metrics')
            ax.set_ylabel('Error Value')
            ax.set_title('Prediction Error Analysis')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bars, values in [(bars1, mae_values), (bars2, mape_values)]:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=8)

    def _plot_resource_forecast_comparison(self, ax, historical_data: pd.DataFrame,
                                         prediction_result: PredictionResult):
        """Plot resource forecast comparison."""
        # Calculate historical resource usage
        hist_memory = (historical_data['input_toks'] + historical_data['output_toks']) * 0.001  # Simplified estimate
        hist_compute = historical_data['input_toks'] * 0.0001  # Simplified estimate

        # Predicted resource usage
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_memory = [req['memory_requirement_mb'] for req in prediction_result.predicted_requests]
        pred_compute = [req['estimated_compute_time'] for req in prediction_result.predicted_requests]

        # Create time series comparison
        ax2 = ax.twinx()

        # Memory comparison
        ax.hist(hist_memory, bins=20, alpha=0.5, label='Historical Memory',
               color=self.colors[0], density=True)
        ax.hist(pred_memory, bins=20, alpha=0.5, label='Predicted Memory',
               color=self.colors[1], density=True)

        # Compute comparison
        ax2.hist(hist_compute, bins=20, alpha=0.5, label='Historical Compute',
                color=self.colors[2], density=True)
        ax2.hist(pred_compute, bins=20, alpha=0.5, label='Predicted Compute',
                color=self.colors[3], density=True)

        ax.set_xlabel('Resource Value')
        ax.set_ylabel('Memory (MB) - Density')
        ax2.set_ylabel('Compute Time (ms) - Density')
        ax.set_title('Resource Usage Distribution Comparison')

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax.grid(True, alpha=0.3)

    def plot_enhanced_time_series(self,
                                  historical_data: pd.DataFrame,
                                  prediction_result: PredictionResult,
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True) -> None:
        """
        Create enhanced time series visualization with multiple views.

        Args:
            historical_data: Historical actual data
            prediction_result: Prediction result
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        fig.suptitle('Enhanced Time Series Analysis', fontsize=16, fontweight='bold')

        # 1. Multi-scale request rate analysis
        self._plot_multi_scale_analysis(axes[0, 0], historical_data, prediction_result)

        # 2. Trend and seasonality decomposition
        self._plot_trend_decomposition(axes[0, 1], historical_data, prediction_result)

        # 3. Rolling statistics comparison
        self._plot_rolling_statistics(axes[1, 0], historical_data, prediction_result)

        # 4. Prediction horizon analysis
        self._plot_prediction_horizon_analysis(axes[1, 1], historical_data, prediction_result)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Enhanced time series saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _plot_multi_scale_analysis(self, ax, historical_data: pd.DataFrame,
                                 prediction_result: PredictionResult):
        """Plot multi-scale time series analysis."""
        # Historical data
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')

        # Multiple time windows
        windows = [30, 60, 300]  # 30s, 1min, 5min
        colors = [self.colors[0], self.colors[1], self.colors[2]]

        for i, window in enumerate(windows):
            hist_rates = self._calculate_request_rates(historical_data, window_seconds=window)
            ax.plot(hist_rates.index, hist_rates.values, 'o-',
                   label=f'{window}s window', color=colors[i], alpha=0.7, linewidth=1.5)

        # Predicted data with overlay
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(pred_rates) > 0:
            ax.plot(pred_rates.index, pred_rates.values, 's--',
                   label='Predicted (60s)', color=self.colors[3], linewidth=2, markersize=4)

            # Add prediction confidence band
            confidence_band = pred_rates.values * (1 - prediction_result.confidence)
            ax.fill_between(pred_rates.index, pred_rates.values - confidence_band,
                           pred_rates.values + confidence_band,
                           alpha=0.2, color=self.colors[3])

        ax.set_xlabel('Time')
        ax.set_ylabel('Requests per Second')
        ax.set_title('Multi-Scale Request Rate Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_trend_decomposition(self, ax, historical_data: pd.DataFrame,
                                prediction_result: PredictionResult):
        """Plot trend and seasonality decomposition."""
        # Calculate request rates
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        if len(hist_rates) < 10:
            ax.text(0.5, 0.5, 'Insufficient data for trend analysis',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Trend Decomposition (Insufficient Data)')
            return

        # Simple trend calculation using rolling mean
        trend_window = max(5, len(hist_rates) // 10)
        trend = hist_rates.rolling(window=trend_window, center=True).mean()

        # Plot components with proper alignment
        valid_trend = trend.dropna()
        if len(valid_trend) > 0:
            # Get corresponding times for valid trend points
            valid_indices = trend.dropna().index
            valid_times = hist_rates.loc[valid_indices].index

            ax.plot(valid_times, valid_trend.values,
                   label='Trend', color=self.colors[0], linewidth=2)

        # Add predicted trend
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(pred_rates) > 1:
            pred_trend = pred_rates.rolling(window=min(3, len(pred_rates)//2), center=True).mean()
            ax.plot(pred_trend.index, pred_trend.dropna(),
                   label='Predicted Trend', color=self.colors[1], linewidth=2, linestyle='--')

        ax.set_xlabel('Time')
        ax.set_ylabel('Requests per Second')
        ax.set_title('Trend Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_rolling_statistics(self, ax, historical_data: pd.DataFrame,
                               prediction_result: PredictionResult):
        """Plot rolling statistics comparison."""
        # Historical rolling statistics
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=30)

        if len(hist_rates) < 10:
            ax.text(0.5, 0.5, 'Insufficient data for rolling statistics',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Rolling Statistics (Insufficient Data)')
            return

        # Calculate rolling statistics
        window = max(5, len(hist_rates) // 8)
        rolling_mean = hist_rates.rolling(window=window).mean()
        rolling_std = hist_rates.rolling(window=window).std()

        # Plot rolling statistics with proper alignment
        valid_mean = rolling_mean.dropna()
        if len(valid_mean) > 0:
            valid_mean_indices = valid_mean.index
            valid_mean_times = hist_rates.loc[valid_mean_indices].index

            ax.plot(valid_mean_times, valid_mean.values,
                   label='Rolling Mean', color=self.colors[0], linewidth=2)

            # Plot confidence interval
            valid_std = rolling_std.loc[valid_mean_indices]
            ax.fill_between(valid_mean_times,
                           valid_mean.values - valid_std.values,
                           valid_mean.values + valid_std.values,
                           alpha=0.2, color=self.colors[0], label='±1 Std Dev')

        # Predicted rolling statistics
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=30)

        if len(pred_rates) > 3:
            pred_window = max(3, len(pred_rates) // 4)
            pred_rolling_mean = pred_rates.rolling(window=pred_window).mean()
            ax.plot(pred_rolling_mean.index, pred_rolling_mean.dropna(),
                   label='Predicted Rolling Mean', color=self.colors[1], linewidth=2, linestyle='--')

        ax.set_xlabel('Time')
        ax.set_ylabel('Requests per Second')
        ax.set_title('Rolling Statistics Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_prediction_horizon_analysis(self, ax, historical_data: pd.DataFrame,
                                        prediction_result: PredictionResult):
        """Plot prediction horizon accuracy analysis."""
        # Calculate historical rates at different horizons
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        if len(hist_rates) < 5:
            ax.text(0.5, 0.5, 'Insufficient data for horizon analysis',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Prediction Horizon Analysis (Insufficient Data)')
            return

        # Simulate prediction at different horizons
        horizons = [1, 3, 5, 10]  # Different prediction steps
        accuracies = []

        for horizon in horizons:
            if len(hist_rates) > horizon:
                # Simple simulation: use past values to predict future
                predicted_values = hist_rates.iloc[:-horizon].values
                actual_values = hist_rates.iloc[horizon:].values

                if len(predicted_values) > 0 and len(actual_values) > 0:
                    # Calculate accuracy (1 - normalized error)
                    mae = np.mean(np.abs(predicted_values - actual_values[:len(predicted_values)]))
                    accuracy = 1 / (1 + mae / max(hist_rates.mean(), 1e-6))
                    accuracies.append(accuracy)
                else:
                    accuracies.append(0)
            else:
                accuracies.append(0)

        # Plot horizon vs accuracy
        ax.plot(horizons[:len(accuracies)], accuracies, 'o-', linewidth=2, markersize=6)
        ax.fill_between(horizons[:len(accuracies)], accuracies, alpha=0.3)

        # Mark the current model's confidence
        ax.axhline(y=prediction_result.confidence, color='red', linestyle='--',
                   label=f'Model Confidence: {prediction_result.confidence:.2f}')

        ax.set_xlabel('Prediction Horizon (steps)')
        ax.set_ylabel('Prediction Accuracy')
        ax.set_title('Prediction Horizon vs Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add horizon annotations
        for i, (horizon, accuracy) in enumerate(zip(horizons[:len(accuracies)], accuracies)):
            ax.text(horizon, accuracy + 0.01, f'{accuracy:.2f}',
                   ha='center', va='bottom', fontsize=8)

    def plot_error_analysis(self,
                          historical_data: pd.DataFrame,
                          prediction_result: PredictionResult,
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> None:
        """
        Create comprehensive error analysis visualization.

        Args:
            historical_data: Historical actual data
            prediction_result: Prediction result
            save_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        fig.suptitle('Comprehensive Error Analysis', fontsize=16, fontweight='bold')

        # 1. Residual analysis
        self._plot_residual_analysis(axes[0, 0], historical_data, prediction_result)

        # 2. Error distribution
        self._plot_error_distribution(axes[0, 1], historical_data, prediction_result)

        # 3. Quantile-Quantile plot
        self._plot_qq_analysis(axes[0, 2], historical_data, prediction_result)

        # 4. Error over time
        self._plot_error_over_time(axes[1, 0], historical_data, prediction_result)

        # 5. Error metrics summary
        self._plot_error_metrics_summary(axes[1, 1], historical_data, prediction_result)

        # 6. Confidence calibration
        self._plot_confidence_calibration(axes[1, 2], historical_data, prediction_result)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"Error analysis saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _plot_residual_analysis(self, ax, historical_data: pd.DataFrame,
                               prediction_result: PredictionResult):
        """Plot residual analysis."""
        # Calculate aligned actual vs predicted values
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            ax.text(0.5, 0.5, 'Insufficient data for residual analysis',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Residual Analysis (Insufficient Data)')
            return

        # Align time series by resampling to common timestamps
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        # Calculate residuals
        residuals = hist_aligned - pred_aligned

        # Plot residuals vs predicted values
        ax.scatter(pred_aligned, residuals, alpha=0.6, color=self.colors[0])

        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)

        # Add trend line
        if len(residuals) > 1:
            z = np.polyfit(pred_aligned, residuals, 1)
            p = np.poly1d(z)
            ax.plot(pred_aligned, p(pred_aligned), "r--", alpha=0.8, linewidth=2,
                   label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')

        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals (Actual - Predicted)')
        ax.set_title('Residual Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        residual_std = residuals.std()
        ax.text(0.02, 0.98, f'Residual Std: {residual_std:.3f}',
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _plot_error_distribution(self, ax, historical_data: pd.DataFrame,
                              prediction_result: PredictionResult):
        """Plot error distribution analysis."""
        # Calculate errors for different metrics
        errors = self._calculate_prediction_errors(historical_data, prediction_result)

        if not errors:
            ax.text(0.5, 0.5, 'No errors to analyze',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Error Distribution (No Data)')
            return

        # Create error distributions
        metric_names = list(errors.keys())
        error_values = list(errors.values())

        # Create histogram for each error type
        for i, (metric, error_vals) in enumerate(zip(metric_names, error_values)):
            if error_vals:
                ax.hist(error_vals, bins=20, alpha=0.7, label=metric,
                       color=self.colors[i % len(self.colors)], density=True)

                # Add normal distribution overlay
                if len(error_vals) > 5:
                    mu, sigma = np.mean(error_vals), np.std(error_vals)
                    x = np.linspace(min(error_vals), max(error_vals), 100)
                    y = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*((x-mu)/sigma)**2)
                    ax.plot(x, y, '--', alpha=0.8, color=self.colors[i % len(self.colors)])

        ax.set_xlabel('Error Value')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_qq_analysis(self, ax, historical_data: pd.DataFrame,
                        prediction_result: PredictionResult):
        """Plot Quantile-Quantile analysis."""
        # Calculate errors
        errors = self._calculate_prediction_errors(historical_data, prediction_result)

        if not errors or not any(errors.values()):
            ax.text(0.5, 0.5, 'No data for Q-Q analysis',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Q-Q Plot (No Data)')
            return

        # Use the first available error metric
        first_metric = next(iter(errors.values()))
        if not first_metric:
            ax.text(0.5, 0.5, 'No valid errors for Q-Q analysis',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Q-Q Plot (No Valid Data)')
            return

        error_values = np.array(first_metric)
        error_values = error_values[~np.isnan(error_values)]  # Remove NaN values

        if len(error_values) < 5:
            ax.text(0.5, 0.5, 'Insufficient data for Q-Q analysis',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Q-Q Plot (Insufficient Data)')
            return

        # Calculate theoretical quantiles
        theoretical_quantiles = np.percentile(np.random.normal(0, 1, 10000),
                                            np.linspace(0, 100, len(error_values)))
        sample_quantiles = np.percentile(error_values, np.linspace(0, 100, len(error_values)))

        # Plot Q-Q plot
        ax.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, color=self.colors[0])

        # Add reference line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Fit')

        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.set_title('Q-Q Plot (Normality Check)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_error_over_time(self, ax, historical_data: pd.DataFrame,
                            prediction_result: PredictionResult):
        """Plot error evolution over time."""
        # Calculate time-aligned errors
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            ax.text(0.5, 0.5, 'Insufficient data for error timeline',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Error Over Time (Insufficient Data)')
            return

        # Align and calculate errors
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        errors = hist_aligned - pred_aligned
        absolute_errors = np.abs(errors)

        # Plot errors over time
        ax.plot(common_times, errors, 'o-', label='Errors', color=self.colors[0], alpha=0.7)
        ax.plot(common_times, absolute_errors, 's--', label='Absolute Errors', color=self.colors[1], alpha=0.7)

        # Add zero line
        ax.axhline(y=0, color='red', linestyle=':', alpha=0.5)

        # Add rolling mean of errors
        if len(errors) > 5:
            rolling_mean = errors.rolling(window=min(5, len(errors)//2)).mean()
            ax.plot(common_times, rolling_mean, '-', label='Rolling Mean', color=self.colors[2], linewidth=2)

        ax.set_xlabel('Time')
        ax.set_ylabel('Error Value')
        ax.set_title('Error Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_error_metrics_summary(self, ax, historical_data: pd.DataFrame,
                                 prediction_result: PredictionResult):
        """Plot comprehensive error metrics summary."""
        # Calculate various error metrics
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            ax.text(0.5, 0.5, 'Insufficient data for metrics',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Error Metrics Summary (Insufficient Data)')
            return

        # Align data
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        errors = hist_aligned - pred_aligned
        absolute_errors = np.abs(errors)

        # Calculate metrics
        mae = np.mean(absolute_errors)
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        mape = np.mean(absolute_errors / np.maximum(hist_aligned, 1e-6)) * 100

        # Create metrics visualization
        metrics = ['MAE', 'RMSE', 'MAPE (%)']
        values = [mae, rmse, mape]

        bars = ax.bar(metrics, values, color=self.colors[:3], alpha=0.8)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Error Value')
        ax.set_title('Error Metrics Summary')
        ax.grid(True, alpha=0.3)

        # Add additional statistics as text
        stats_text = f'Sample Count: {len(errors)}\nError Std: {errors.std():.3f}\nBias: {errors.mean():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _plot_confidence_calibration(self, ax, historical_data: pd.DataFrame,
                                   prediction_result: PredictionResult):
        """Plot confidence calibration analysis."""
        # Calculate prediction intervals and actual coverage
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            ax.text(0.5, 0.5, 'Insufficient data for calibration',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Confidence Calibration (Insufficient Data)')
            return

        # Align data
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        errors = np.abs(hist_aligned - pred_aligned)

        # Test different confidence levels
        confidence_levels = np.linspace(0.5, 0.99, 10)
        actual_coverages = []

        for conf_level in confidence_levels:
            # Calculate prediction interval based on confidence
            prediction_interval = pred_aligned * (1 - conf_level)
            within_interval = errors <= prediction_interval
            actual_coverage = np.mean(within_interval) if len(within_interval) > 0 else 0
            actual_coverages.append(actual_coverage)

        # Plot calibration curve
        ax.plot(confidence_levels, actual_coverages, 'o-', linewidth=2, markersize=6,
               color=self.colors[0], label='Actual Coverage')

        # Plot perfect calibration line
        ax.plot([0.5, 1.0], [0.5, 1.0], '--', color='red', alpha=0.8,
               label='Perfect Calibration')

        # Mark the model's stated confidence
        model_confidence = prediction_result.confidence
        if model_confidence >= 0.5:
            idx = np.argmin(np.abs(confidence_levels - model_confidence))
            actual_at_model_conf = actual_coverages[idx]
            ax.plot(model_confidence, actual_at_model_conf, 'ro', markersize=10,
                   label=f'Model: {model_confidence:.2f} → {actual_at_model_conf:.2f}')

        ax.set_xlabel('Stated Confidence Level')
        ax.set_ylabel('Actual Coverage')
        ax.set_title('Confidence Calibration Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add calibration score
        calibration_error = np.mean(np.array(confidence_levels) - np.array(actual_coverages))
        ax.text(0.02, 0.98, f'Calibration Error: {calibration_error:.3f}',
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _calculate_prediction_errors(self, historical_data: pd.DataFrame,
                                   prediction_result: PredictionResult) -> Dict[str, List[float]]:
        """Calculate prediction errors for different metrics."""
        errors = {}

        # Request rate errors
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) > 0 and len(pred_rates) > 0:
            # Align data
            common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
            hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
            pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

            rate_errors = (hist_aligned - pred_aligned).tolist()
            errors['Request Rate'] = rate_errors

        # Token count errors
        hist_tokens = historical_data['input_toks'] + historical_data['output_toks']
        pred_tokens = [req['input_tokens'] + req['output_tokens']
                      for req in prediction_result.predicted_requests]

        if len(hist_tokens) > 0 and len(pred_tokens) > 0:
            # Resample to match lengths
            min_len = min(len(hist_tokens), len(pred_tokens))
            hist_sampled = hist_tokens.iloc[:min_len] if hasattr(hist_tokens, 'iloc') else hist_tokens[:min_len]
            pred_sampled = pred_tokens[:min_len]

            token_errors = (hist_sampled - pred_sampled).tolist()
            errors['Token Count'] = token_errors

        return errors

    def create_dashboard(self,
                        prediction_result: PredictionResult,
                        historical_data: Optional[pd.DataFrame] = None,
                        training_history: Optional[Dict[str, List[float]]] = None,
                        save_dir: str = "prediction_plots") -> None:
        """
        Create a complete dashboard with all visualization plots.

        Args:
            prediction_result: Prediction result
            historical_data: Historical data for comparison
            training_history: Model training history
            save_dir: Directory to save plots
        """
        import os

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Generate all plots
        plots = [
            (self.plot_prediction_overview,
             f"{save_dir}/01_prediction_overview.png"),
            (self.plot_time_series_comparison,
             f"{save_dir}/02_time_series_comparison.png"),
            (self.plot_resource_forecast,
             f"{save_dir}/03_resource_forecast.png"),
            (self.plot_burst_patterns,
             f"{save_dir}/04_burst_patterns.png"),
        ]

        for plot_func, save_path in plots:
            try:
                if plot_func == self.plot_prediction_overview:
                    plot_func(prediction_result, historical_data, save_path, show_plot=False)
                elif plot_func == self.plot_time_series_comparison:
                    plot_func(historical_data, prediction_result, save_path=save_path, show_plot=False)
                elif plot_func == self.plot_resource_forecast:
                    plot_func(prediction_result, save_path=save_path, show_plot=False)
                elif plot_func == self.plot_burst_patterns:
                    plot_func(historical_data, prediction_result, save_path=save_path, show_plot=False)
            except Exception as e:
                logger.error(f"Failed to create {save_path}: {e}")

        # Add model performance plot if training history is available
        if training_history:
            try:
                self.plot_model_performance(training_history,
                                          f"{save_dir}/05_model_performance.png",
                                          show_plot=False)
            except Exception as e:
                logger.error(f"Failed to create model performance plot: {e}")

        logger.info(f"Dashboard created in {save_dir}/")
        print(f"Dashboard created with {len(plots)} plots in {save_dir}/")

    # ====================== 新增多维度对比分析功能 ======================

    def plot_multi_model_comparison(self,
                                   model_predictions: Dict[str, PredictionResult],
                                   historical_data: pd.DataFrame,
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True) -> None:
        """
        创建多模型对比分析图，对比不同模型的预测性能。

        Args:
            model_predictions: 字典格式的模型预测结果 {model_name: PredictionResult}
            historical_data: 历史真实数据
            save_path: 保存路径
            show_plot: 是否显示图表
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 14), dpi=self.dpi)
        fig.suptitle('多模型预测性能对比分析', fontsize=18, fontweight='bold')

        # 1. 预测准确度对比
        self._plot_model_accuracy_comparison(axes[0, 0], model_predictions, historical_data)

        # 2. 时间序列预测对比
        self._plot_multi_model_time_series(axes[0, 1], model_predictions, historical_data)

        # 3. 误差分布对比
        self._plot_multi_model_error_distribution(axes[0, 2], model_predictions, historical_data)

        # 4. 预测置信度对比
        self._plot_model_confidence_comparison(axes[1, 0], model_predictions)

        # 5. 资源预测对比
        self._plot_model_resource_comparison(axes[1, 1], model_predictions, historical_data)

        # 6. 综合性能指标对比
        self._plot_model_performance_metrics(axes[1, 2], model_predictions, historical_data)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"多模型对比图保存到 {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_multi_horizon_analysis(self,
                                   prediction_results: Dict[str, PredictionResult],
                                   historical_data: pd.DataFrame,
                                   save_path: Optional[str] = None,
                                   show_plot: bool = True) -> None:
        """
        创建多时间范围预测分析图。

        Args:
            prediction_results: 不同时间范围的预测结果 {horizon_name: PredictionResult}
            historical_data: 历史数据
            save_path: 保存路径
            show_plot: 是否显示图表
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        fig.suptitle('多时间范围预测分析', fontsize=16, fontweight='bold')

        # 1. 不同时间范围的预测精度对比
        self._plot_horizon_accuracy_comparison(axes[0, 0], prediction_results, historical_data)

        # 2. 预测稳定性分析
        self._plot_horizon_stability_analysis(axes[0, 1], prediction_results)

        # 3. 累积预测对比
        self._plot_horizon_cumulative_comparison(axes[1, 0], prediction_results, historical_data)

        # 4. 时间范围vs准确度趋势
        self._plot_horizon_accuracy_trend(axes[1, 1], prediction_results, historical_data)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"多时间范围分析图保存到 {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_enhanced_actual_vs_predicted(self,
                                        historical_data: pd.DataFrame,
                                        prediction_result: PredictionResult,
                                        model_name: str = "Model",
                                        save_path: Optional[str] = None,
                                        show_plot: bool = True) -> None:
        """
        增强的真实值vs预测值对比分析。

        Args:
            historical_data: 历史真实数据
            prediction_result: 预测结果
            model_name: 模型名称
            save_path: 保存路径
            show_plot: 是否显示图表
        """
        fig, axes = plt.subplots(3, 3, figsize=(21, 15), dpi=self.dpi)
        fig.suptitle(f'{model_name} - 真实值 vs 预测值多维度对比分析', fontsize=16, fontweight='bold')

        # 1. 请求率对比（增强版）
        self._plot_enhanced_request_rate_comparison_v2(axes[0, 0], historical_data, prediction_result)

        # 2. Token分布对比
        self._plot_enhanced_token_distribution_comparison(axes[0, 1], historical_data, prediction_result)

        # 3. 时间序列对比（带置信区间）
        self._plot_enhanced_time_series_comparison(axes[0, 2], historical_data, prediction_result)

        # 4. 误差热力图
        self._plot_prediction_error_heatmap(axes[1, 0], historical_data, prediction_result)

        # 5. 滚动统计对比
        self._plot_enhanced_rolling_comparison(axes[1, 1], historical_data, prediction_result)

        # 6. 预测vs真实散点图
        self._plot_prediction_vs_actual_scatter(axes[1, 2], historical_data, prediction_result)

        # 7. 分位数对比
        self._plot_quantile_comparison(axes[2, 0], historical_data, prediction_result)

        # 8. 预测区间覆盖率
        self._plot_prediction_interval_coverage(axes[2, 1], historical_data, prediction_result)

        # 9. 综合评估指标
        self._plot_comprehensive_evaluation_metrics(axes[2, 2], historical_data, prediction_result)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            logger.info(f"增强对比分析图保存到 {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _plot_model_accuracy_comparison(self, ax, model_predictions: Dict[str, PredictionResult],
                                     historical_data: pd.DataFrame):
        """绘制模型准确度对比。"""
        model_names = list(model_predictions.keys())
        metrics = ['MAE', 'RMSE', 'MAPE', 'R²']

        # 计算每个模型的指标
        model_metrics = {}
        for model_name, pred_result in model_predictions.items():
            metrics_dict = self._calculate_model_metrics(historical_data, pred_result)
            model_metrics[model_name] = metrics_dict

        # 创建对比图
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)

        for i, model_name in enumerate(model_names):
            values = [model_metrics[model_name].get(metric, 0) for metric in metrics]
            ax.bar(x + i * width, values, width, label=model_name, alpha=0.8)

        ax.set_xlabel('评估指标')
        ax.set_ylabel('指标值')
        ax.set_title('模型预测准确度对比')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_multi_model_time_series(self, ax, model_predictions: Dict[str, PredictionResult],
                                   historical_data: pd.DataFrame):
        """绘制多模型时间序列对比。"""
        # 历史数据
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        ax.plot(hist_rates.index, hist_rates.values, 'o-', linewidth=3,
               label='真实值', color='black', markersize=4)

        # 各模型预测
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_predictions)))
        for i, (model_name, pred_result) in enumerate(model_predictions.items()):
            pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                         for req in pred_result.predicted_requests]
            pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

            if len(pred_rates) > 0:
                ax.plot(pred_rates.index, pred_rates.values, '--', linewidth=2,
                       label=f'{model_name}预测', color=colors[i], alpha=0.8)

        ax.set_xlabel('时间')
        ax.set_ylabel('请求率 (requests/second)')
        ax.set_title('多模型时间序列预测对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_multi_model_error_distribution(self, ax, model_predictions: Dict[str, PredictionResult],
                                          historical_data: pd.DataFrame):
        """绘制多模型误差分布对比。"""
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_predictions)))

        for i, (model_name, pred_result) in enumerate(model_predictions.items()):
            errors = self._calculate_aligned_errors(historical_data, pred_result)
            if errors:
                ax.hist(errors, bins=20, alpha=0.6, label=f'{model_name}',
                       color=colors[i], density=True)

        ax.set_xlabel('误差值')
        ax.set_ylabel('密度')
        ax.set_title('多模型误差分布对比')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_model_confidence_comparison(self, ax, model_predictions: Dict[str, PredictionResult]):
        """绘制模型置信度对比。"""
        model_names = list(model_predictions.keys())
        confidences = [pred_result.confidence for pred_result in model_predictions.values()]

        bars = ax.bar(model_names, confidences, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))

        ax.set_ylabel('置信度')
        ax.set_title('模型预测置信度对比')
        ax.set_ylim(0, 1)

        # 添加数值标签
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{conf:.3f}', ha='center', va='bottom')

        ax.grid(True, alpha=0.3)

    def _plot_model_resource_comparison(self, ax, model_predictions: Dict[str, PredictionResult],
                                     historical_data: pd.DataFrame):
        """绘制模型资源预测对比。"""
        model_names = list(model_predictions.keys())
        resource_metrics = ['内存预测(MB)', '计算时间预测(ms)', 'Token预测']

        for i, model_name in enumerate(model_predictions):
            pred_result = model_predictions[model_name]
            resource_req = pred_result.get_resource_requirements()

            values = [
                resource_req['memory_mb'],
                resource_req['compute_units'] * 10,  # 放大显示
                resource_req['total_tokens'] / 1000  # 缩小显示
            ]

            x = np.arange(len(resource_metrics))
            width = 0.8 / len(model_names)
            ax.bar(x + i * width, values, width, label=model_name, alpha=0.7)

        ax.set_xlabel('资源类型')
        ax.set_ylabel('预测值')
        ax.set_title('模型资源预测对比')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(resource_metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_model_performance_metrics(self, ax, model_predictions: Dict[str, PredictionResult],
                                      historical_data: pd.DataFrame):
        """绘制模型综合性能指标雷达图。"""
        model_names = list(model_predictions.keys())
        metrics = ['准确度', '稳定性', '置信度', '效率', '覆盖度']

        # 计算每个模型的综合评分
        model_scores = {}
        for model_name in model_names:
            scores = self._calculate_comprehensive_scores(historical_data, model_predictions[model_name])
            model_scores[model_name] = scores

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        ax = plt.subplot(111, projection='polar')

        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        for i, model_name in enumerate(model_names):
            scores = list(model_scores[model_name].values())
            scores += scores[:1]  # 闭合图形

            ax.plot(angles, scores, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, scores, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('模型综合性能雷达图', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        ax.grid(True)

    def _plot_horizon_accuracy_comparison(self, ax, prediction_results: Dict[str, PredictionResult],
                                        historical_data: pd.DataFrame):
        """绘制不同时间范围的准确度对比。"""
        horizon_names = list(prediction_results.keys())
        metrics = ['MAE', 'RMSE', 'MAPE']

        horizon_metrics = {}
        for horizon_name, pred_result in prediction_results.items():
            metrics_dict = self._calculate_model_metrics(historical_data, pred_result)
            horizon_metrics[horizon_name] = metrics_dict

        x = np.arange(len(metrics))
        width = 0.8 / len(horizon_names)

        for i, horizon_name in enumerate(horizon_names):
            values = [horizon_metrics[horizon_name].get(metric, 0) for metric in metrics]
            ax.bar(x + i * width, values, width, label=horizon_name, alpha=0.8)

        ax.set_xlabel('评估指标')
        ax.set_ylabel('指标值')
        ax.set_title('不同时间范围预测准确度对比')
        ax.set_xticks(x + width * (len(horizon_names) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_horizon_stability_analysis(self, ax, prediction_results: Dict[str, PredictionResult]):
        """绘制预测稳定性分析。"""
        horizon_names = list(prediction_results.keys())
        stability_metrics = ['预测方差', '置信度一致性', '变化幅度']

        stability_values = {}
        for horizon_name, pred_result in prediction_results.items():
            # 计算稳定性指标
            token_values = [req['input_tokens'] + req['output_tokens']
                           for req in pred_result.predicted_requests]
            variance = np.var(token_values) if token_values else 0
            confidence = pred_result.confidence
            change_amplitude = np.std(token_values) if token_values else 0

            stability_values[horizon_name] = [variance/1000, confidence, change_amplitude/100]

        x = np.arange(len(stability_metrics))
        width = 0.8 / len(horizon_names)

        for i, horizon_name in enumerate(horizon_names):
            values = stability_values[horizon_name]
            ax.bar(x + i * width, values, width, label=horizon_name, alpha=0.8)

        ax.set_xlabel('稳定性指标')
        ax.set_ylabel('指标值')
        ax.set_title('预测稳定性分析')
        ax.set_xticks(x + width * (len(horizon_names) - 1) / 2)
        ax.set_xticklabels(stability_metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_horizon_cumulative_comparison(self, ax, prediction_results: Dict[str, PredictionResult],
                                          historical_data: pd.DataFrame):
        """绘制累积预测对比。"""
        # 历史累积数据
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_sorted = historical_data.sort_values('arrival_time_ns')
        hist_cumulative = np.arange(1, len(hist_sorted) + 1)

        ax.plot(hist_times, hist_cumulative, 'o-', linewidth=3,
               label='真实累积', color='black', markersize=3)

        # 各时间范围的预测
        colors = plt.cm.autumn(np.linspace(0, 1, len(prediction_results)))
        for i, (horizon_name, pred_result) in enumerate(prediction_results.items()):
            pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                         for req in pred_result.predicted_requests]
            pred_sorted = sorted(pred_result.predicted_requests, key=lambda x: x['arrival_time_ns'])
            pred_cumulative = np.arange(1, len(pred_sorted) + 1)

            ax.plot(pred_times, pred_cumulative, '--', linewidth=2,
                   label=f'{horizon_name}预测', color=colors[i], alpha=0.8)

        ax.set_xlabel('时间')
        ax.set_ylabel('累积请求数')
        ax.set_title('累积预测对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_horizon_accuracy_trend(self, ax, prediction_results: Dict[str, PredictionResult],
                                   historical_data: pd.DataFrame):
        """绘制时间范围vs准确度趋势。"""
        horizon_names = list(prediction_results.keys())
        accuracy_trends = []

        for horizon_name in prediction_results.keys():
            metrics = self._calculate_model_metrics(historical_data, prediction_results[horizon_name])
            # 综合准确度评分
            accuracy_score = (1 / (1 + metrics.get('MAE', 1))) * \
                           (1 / (1 + metrics.get('RMSE', 1))) * \
                           (1 / (1 + metrics.get('MAPE', 100) / 100))
            accuracy_trends.append(accuracy_score)

        # 提取预测步数信息
        prediction_steps = []
        for horizon_name in horizon_names:
            if 'short' in horizon_name.lower():
                prediction_steps.append(10)
            elif 'medium' in horizon_name.lower():
                prediction_steps.append(30)
            elif 'long' in horizon_name.lower():
                prediction_steps.append(60)
            else:
                prediction_steps.append(len(prediction_results[horizon_name].predicted_requests))

        ax.plot(prediction_steps, accuracy_trends, 'o-', linewidth=2, markersize=8)
        ax.fill_between(prediction_steps, accuracy_trends, alpha=0.3)

        # 添加趋势线
        if len(prediction_steps) > 2:
            z = np.polyfit(prediction_steps, accuracy_trends, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(min(prediction_steps), max(prediction_steps), 100)
            ax.plot(x_trend, p(x_trend), '--', color='red', alpha=0.8, label='趋势线')

        ax.set_xlabel('预测步数')
        ax.set_ylabel('准确度评分')
        ax.set_title('预测时间范围vs准确度趋势')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加数据点标注
        for i, (step, accuracy) in enumerate(zip(prediction_steps, accuracy_trends)):
            ax.annotate(f'{accuracy:.3f}', (step, accuracy), textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=8)

    def _plot_enhanced_request_rate_comparison_v2(self, ax, historical_data: pd.DataFrame,
                                                 prediction_result: PredictionResult):
        """增强的请求率对比（第二版）。"""
        # 计算历史和预测的请求率
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=30)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=30)

        if len(hist_rates) == 0:
            ax.text(0.5, 0.5, '无历史数据', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('请求率对比（无数据）')
            return

        # 绘制真实值
        ax.plot(hist_rates.index, hist_rates.values, 'o-', linewidth=3,
               label='真实值', color=self.colors[0], markersize=4, alpha=0.9)

        # 绘制预测值和置信区间
        if len(pred_rates) > 0:
            ax.plot(pred_rates.index, pred_rates.values, 's--', linewidth=2,
                   label='预测值', color=self.colors[1], markersize=4)

            # 95% 置信区间
            confidence_upper = pred_rates.values * (1 + (1 - prediction_result.confidence))
            confidence_lower = pred_rates.values * max(0, (1 - (1 - prediction_result.confidence)))
            ax.fill_between(pred_rates.index, confidence_lower, confidence_upper,
                           alpha=0.2, color=self.colors[1], label='95% 置信区间')

        # 添加统计信息
        if len(hist_rates) > 0 and len(pred_rates) > 0:
            actual_mean = hist_rates.mean()
            pred_mean = pred_rates.mean() if len(pred_rates) > 0 else 0
            mape = abs(actual_mean - pred_mean) / max(actual_mean, 1e-6) * 100

            stats_text = f'真实均值: {actual_mean:.2f}\n预测均值: {pred_mean:.2f}\nMAPE: {mape:.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('时间')
        ax.set_ylabel('请求率 (requests/second)')
        ax.set_title('请求率：真实值 vs 预测值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_enhanced_token_distribution_comparison(self, ax, historical_data: pd.DataFrame,
                                                   prediction_result: PredictionResult):
        """增强的Token分布对比。"""
        # 计算Token统计
        hist_input = historical_data['input_toks']
        hist_output = historical_data['output_toks']
        hist_total = hist_input + hist_output

        pred_input = [req['input_tokens'] for req in prediction_result.predicted_requests]
        pred_output = [req['output_tokens'] for req in prediction_result.predicted_requests]
        pred_total = [inp + out for inp, out in zip(pred_input, pred_output)]

        # 创建对比DataFrame
        comparison_df = pd.DataFrame({
            '真实输入': hist_input,
            '真实输出': hist_output,
            '真实总计': hist_total,
            '预测输入': pred_input,
            '预测输出': pred_output,
            '预测总计': pred_total
        })

        # 绘制箱形图对比
        positions = np.arange(6)
        box_data = [comparison_df[col].dropna() for col in comparison_df.columns]
        box_plot = ax.boxplot(box_data, positions=positions, patch_artist=True, showmeans=True)

        # 设置颜色
        colors = [self.colors[0], self.colors[0], self.colors[0],
                 self.colors[1], self.colors[1], self.colors[1]]
        alphas = [0.7, 0.7, 0.9, 0.7, 0.7, 0.9]

        for patch, color, alpha in zip(box_plot['boxes'], colors, alphas):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)

        ax.set_xticks(positions)
        ax.set_xticklabels(comparison_df.columns, rotation=45)
        ax.set_ylabel('Token数量')
        ax.set_title('Token分布对比分析')

        # 添加统计摘要
        actual_mean = comparison_df['真实总计'].mean()
        pred_mean = comparison_df['预测总计'].mean()
        stats_text = f'真实均值: {actual_mean:.0f}\n预测均值: {pred_mean:.0f}\n差异: {abs(actual_mean-pred_mean):.0f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.grid(True, alpha=0.3)

    def _plot_enhanced_time_series_comparison(self, ax, historical_data: pd.DataFrame,
                                             prediction_result: PredictionResult):
        """增强的时间序列对比。"""
        # 创建时间分箱
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        time_bins = pd.date_range(start=hist_times.min(), end=hist_times.max(), freq='30s')

        # 计算真实请求计数
        hist_counts = pd.cut(hist_times, bins=time_bins).value_counts().sort_index()

        # 预测数据
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]

        # 创建预测分箱
        if pred_times:
            pred_bins = pd.date_range(start=min(pred_times), end=max(pred_times), freq='30s')
            pred_counts = pd.cut(pred_times, bins=pred_bins).value_counts().sort_index()

            # 计算预测统计量
            pred_mean = pred_counts.mean()
            pred_std = pred_counts.std()

            # 绘制真实值
            ax.bar(hist_counts.index, hist_counts.values, width=25/len(time_bins),
                   alpha=0.7, label='真实请求', color=self.colors[0])

            # 绘制预测值和区间
            ax.plot(pred_counts.index, pred_counts.values, 's--',
                   label='预测请求', color=self.colors[1], markersize=4)

            # 置信区间
            upper_bound = np.maximum(0, pred_counts + pred_std)
            lower_bound = np.maximum(0, pred_counts - pred_std)
            ax.fill_between(pred_counts.index, lower_bound, upper_bound,
                           alpha=0.2, color=self.colors[1], label='±1 标准差')

        ax.set_xlabel('时间')
        ax.set_ylabel('请求数量')
        ax.set_title('时间序列对比（30秒窗口）')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_prediction_error_heatmap(self, ax, historical_data: pd.DataFrame,
                                      prediction_result: PredictionResult):
        """绘制预测误差热力图。"""
        # 计算对齐的误差
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            ax.text(0.5, 0.5, '数据不足', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('预测误差热力图（数据不足）')
            return

        # 对齐数据
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        # 计算误差矩阵
        errors = hist_aligned - pred_aligned
        absolute_errors = np.abs(errors)

        # 创建热力图数据
        time_indices = np.arange(len(common_times))
        error_matrix = absolute_errors.values.reshape(-1, 1)

        # 绘制热力图
        im = ax.imshow(error_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('绝对误差')

        # 设置时间标签
        time_step = max(1, len(common_times) // 10)
        ax.set_xticks(time_indices[::time_step])
        ax.set_xticklabels([str(t.time())[:8] for t in common_times[::time_step]], rotation=45)

        ax.set_xlabel('时间')
        ax.set_ylabel('误差强度')
        ax.set_title('预测误差热力图')

    def _plot_enhanced_rolling_comparison(self, ax, historical_data: pd.DataFrame,
                                        prediction_result: PredictionResult):
        """增强的滚动统计对比。"""
        # 计算滚动统计
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=30)

        if len(hist_rates) < 10:
            ax.text(0.5, 0.5, '数据不足', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('滚动统计对比（数据不足）')
            return

        # 真实数据滚动统计
        window = max(5, len(hist_rates) // 8)
        hist_rolling_mean = hist_rates.rolling(window=window).mean()
        hist_rolling_std = hist_rates.rolling(window=window).std()

        # 预测数据
        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=30)

        if len(pred_rates) > 3:
            pred_window = max(3, len(pred_rates) // 6)
            pred_rolling_mean = pred_rates.rolling(window=pred_window).mean()

            # 绘制对比
            valid_hist_mean = hist_rolling_mean.dropna()
            valid_hist_times = hist_rates.loc[valid_hist_mean.index].index

            ax.plot(valid_hist_times, valid_hist_mean.values, '-', linewidth=3,
                   label='真实滚动均值', color=self.colors[0])

            ax.plot(pred_rolling_mean.index, pred_rolling_mean.dropna(), '--', linewidth=2,
                   label='预测滚动均值', color=self.colors[1])

            # 添加置信带
            pred_rolling_std = pred_rates.rolling(window=pred_window).std()
            upper_bound = pred_rolling_mean + pred_rolling_std
            lower_bound = pred_rolling_mean - pred_rolling_std

            ax.fill_between(pred_rolling_mean.index, lower_bound.dropna(), upper_bound.dropna(),
                           alpha=0.2, color=self.colors[1], label='±1 标准差')

        ax.set_xlabel('时间')
        ax.set_ylabel('请求率 (requests/second)')
        ax.set_title(f'滚动统计对比（窗口={window//2:.0f}分钟）')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_prediction_vs_actual_scatter(self, ax, historical_data: pd.DataFrame,
                                         prediction_result: PredictionResult):
        """绘制预测vs真实值散点图。"""
        # 计算对齐的数据
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            ax.text(0.5, 0.5, '数据不足', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('预测vs真实值散点图（数据不足）')
            return

        # 对齐数据
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        # 绘制散点图
        ax.scatter(hist_aligned, pred_aligned, alpha=0.6, s=50, color=self.colors[0])

        # 添加完美预测线
        min_val = min(hist_aligned.min(), pred_aligned.min())
        max_val = max(hist_aligned.max(), pred_aligned.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='完美预测')

        # 添加趋势线
        if len(hist_aligned) > 2:
            z = np.polyfit(hist_aligned, pred_aligned, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min_val, max_val, 100)
            ax.plot(x_trend, p(x_trend), 'g--', alpha=0.8, linewidth=2, label=f'趋势线 (y={z[0]:.2f}x+{z[1]:.2f})')

        # 计算 R²
        if len(hist_aligned) > 2:
            residuals = hist_aligned - pred_aligned
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((hist_aligned - np.mean(hist_aligned))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            ax.text(0.02, 0.98, f'R² = {r_squared:.3f}', transform=ax.transAxes,
                   va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title('预测值 vs 真实值散点图')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_quantile_comparison(self, ax, historical_data: pd.DataFrame,
                                 prediction_result: PredictionResult):
        """绘制分位数对比。"""
        # 计算对齐的数据
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            ax.text(0.5, 0.5, '数据不足', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('分位数对比（数据不足）')
            return

        # 对齐数据
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        # 计算分位数
        quantiles = np.arange(0.1, 1.0, 0.1)
        hist_quantiles = np.quantile(hist_aligned, quantiles)
        pred_quantiles = np.quantile(pred_aligned, quantiles)

        # 绘制分位数对比
        ax.plot(quantiles, hist_quantiles, 'o-', linewidth=2, markersize=6,
               label='真实值分位数', color=self.colors[0])
        ax.plot(quantiles, pred_quantiles, 's--', linewidth=2, markersize=6,
               label='预测值分位数', color=self.colors[1])

        # 添加完美预测线
        ax.plot([0.1, 1.0], [hist_quantiles[0], hist_quantiles[-1]], 'r:', alpha=0.8, label='完美预测')

        ax.set_xlabel('分位数')
        ax.set_ylabel('请求率')
        ax.set_title('分位数对比分析')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加分位数标签
        for i, (q, h, p) in enumerate(zip(quantiles[::2], hist_quantiles[::2], pred_quantiles[::2])):
            ax.annotate(f'{q:.1f}', (q, h), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    def _plot_prediction_interval_coverage(self, ax, historical_data: pd.DataFrame,
                                          prediction_result: PredictionResult):
        """绘制预测区间覆盖率。"""
        # 计算对齐的数据
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            ax.text(0.5, 0.5, '数据不足', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('预测区间覆盖率（数据不足）')
            return

        # 对齐数据
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        # 计算误差
        errors = np.abs(hist_aligned - pred_aligned)

        # 测试不同置信水平的覆盖率
        confidence_levels = np.arange(0.5, 0.96, 0.05)
        actual_coverages = []

        for conf_level in confidence_levels:
            # 计算预测区间
            interval_width = pred_aligned * (1 - conf_level)
            within_interval = errors <= interval_width
            coverage = np.mean(within_interval) if len(within_interval) > 0 else 0
            actual_coverages.append(coverage)

        # 绘制覆盖率曲线
        ax.plot(confidence_levels, actual_coverages, 'o-', linewidth=2, markersize=6,
               color=self.colors[0], label='实际覆盖率')

        # 绘制完美校准线
        ax.plot([0.5, 0.95], [0.5, 0.95], 'r--', alpha=0.8, label='完美校准')

        # 标记模型置信度
        model_conf = prediction_result.confidence
        if 0.5 <= model_conf <= 0.95:
            idx = np.argmin(np.abs(confidence_levels - model_conf))
            actual_cov = actual_coverages[idx]
            ax.plot(model_conf, actual_cov, 'ro', markersize=10,
                   label=f'模型: {model_conf:.2f}→{actual_cov:.2f}')

        ax.set_xlabel('置信水平')
        ax.set_ylabel('实际覆盖率')
        ax.set_title('预测区间覆盖率分析')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加校准误差
        calibration_error = np.mean(np.array(confidence_levels) - np.array(actual_coverages))
        ax.text(0.02, 0.98, f'校准误差: {calibration_error:.3f}', transform=ax.transAxes,
               va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    def _plot_comprehensive_evaluation_metrics(self, ax, historical_data: pd.DataFrame,
                                            prediction_result: PredictionResult):
        """绘制综合评估指标。"""
        # 计算各种指标
        metrics = self._calculate_model_metrics(historical_data, prediction_result)

        # 选择关键指标进行可视化
        key_metrics = ['MAE', 'RMSE', 'MAPE']
        if 'R²' in metrics:
            key_metrics.append('R²')

        values = [metrics.get(metric, 0) for metric in key_metrics]

        # 创建指标对比图
        bars = ax.bar(key_metrics, values, color=self.colors[:len(key_metrics)], alpha=0.8)

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('指标值')
        ax.set_title('综合评估指标')
        ax.grid(True, alpha=0.3)

        # 添加评估等级
        mae_score = max(0, 1 - metrics.get('MAE', 1) / 10)  # 假设MAE<10为优秀
        mape_score = max(0, 1 - metrics.get('MAPE', 100) / 100)  # MAPE<100%为合格

        overall_score = (mae_score + mape_score) / 2
        grade = "A" if overall_score > 0.8 else "B" if overall_score > 0.6 else "C"

        ax.text(0.02, 0.98, f'综合评分: {overall_score:.2f}\n等级: {grade}',
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # ====================== 辅助方法 ======================

    def _calculate_model_metrics(self, historical_data: pd.DataFrame,
                               prediction_result: PredictionResult) -> Dict[str, float]:
        """计算模型评估指标。"""
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            return {'MAE': 0, 'RMSE': 0, 'MAPE': 0, 'R²': 0}

        # 对齐数据
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        # 计算指标
        errors = hist_aligned - pred_aligned
        absolute_errors = np.abs(errors)

        mae = np.mean(absolute_errors)
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)

        # MAPE - 避免除零
        mape = np.mean(absolute_errors / np.maximum(hist_aligned, 1e-6)) * 100

        # R²
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((hist_aligned - np.mean(hist_aligned))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r_squared}

    def _calculate_aligned_errors(self, historical_data: pd.DataFrame,
                                prediction_result: PredictionResult) -> List[float]:
        """计算对齐的误差。"""
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            return []

        # 对齐数据
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        return (hist_aligned - pred_aligned).tolist()

    def _calculate_comprehensive_scores(self, historical_data: pd.DataFrame,
                                       prediction_result: PredictionResult) -> Dict[str, float]:
        """计算综合评分。"""
        metrics = self._calculate_model_metrics(historical_data, prediction_result)

        # 准确度评分 (0-1)
        accuracy_score = 1 / (1 + metrics.get('MAE', 1))

        # 稳定性评分 - 基于预测值的一致性
        pred_values = [req['input_tokens'] + req['output_tokens']
                      for req in prediction_result.predicted_requests]
        stability_score = 1 / (1 + np.std(pred_values) / 100) if pred_values else 0.5

        # 置信度评分
        confidence_score = prediction_result.confidence

        # 效率评分 - 基于计算复杂度
        efficiency_score = min(1.0, 1000 / len(prediction_result.predicted_requests)) if prediction_result.predicted_requests else 0.5

        # 覆盖度评分 - 基于预测范围
        coverage_score = min(1.0, len(prediction_result.predicted_requests) / 100) if prediction_result.predicted_requests else 0.5

        return {
            '准确度': accuracy_score,
            '稳定性': stability_score,
            '置信度': confidence_score,
            '效率': efficiency_score,
            '覆盖度': coverage_score
        }

    def calculate_comprehensive_metrics(self,
                                      historical_data: pd.DataFrame,
                                      prediction_result: PredictionResult) -> Dict[str, Any]:
        """
        计算全面的高级评估指标

        Args:
            historical_data: 历史数据
            prediction_result: 预测结果

        Returns:
            包含所有评估指标的字典
        """
        # 基础指标
        basic_metrics = self._calculate_model_metrics(historical_data, prediction_result)

        # 高级指标
        advanced_metrics = self._calculate_advanced_metrics(historical_data, prediction_result)

        # 统计检验指标
        statistical_metrics = self._calculate_statistical_metrics(historical_data, prediction_result)

        # 业务指标
        business_metrics = self._calculate_business_metrics(historical_data, prediction_result)

        # 鲁棒性指标
        robustness_metrics = self._calculate_robustness_metrics(historical_data, prediction_result)

        return {
            'basic_metrics': basic_metrics,
            'advanced_metrics': advanced_metrics,
            'statistical_metrics': statistical_metrics,
            'business_metrics': business_metrics,
            'robustness_metrics': robustness_metrics,
            'overall_score': self._calculate_overall_score(basic_metrics, advanced_metrics)
        }

    def _calculate_advanced_metrics(self,
                                   historical_data: pd.DataFrame,
                                   prediction_result: PredictionResult) -> Dict[str, float]:
        """计算高级评估指标"""
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            return self._get_default_advanced_metrics()

        # 对齐数据
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        errors = hist_aligned - pred_aligned
        absolute_errors = np.abs(errors)

        # 1. SMAPE (Symmetric Mean Absolute Percentage Error)
        numerator = np.abs(hist_aligned - pred_aligned)
        denominator = (np.abs(hist_aligned) + np.abs(pred_aligned)) / 2
        smape = np.mean(numerator / np.maximum(denominator, 1e-6)) * 100

        # 2. MASE (Mean Absolute Scaled Error)
        if len(hist_aligned) > 1:
            # 计算季节性误差（使用简单的滞后1预测）
            seasonal_errors = np.abs(hist_aligned.iloc[1:].values - hist_aligned.iloc[:-1].values)
            mase = np.mean(absolute_errors) / np.mean(seasonal_errors) if np.mean(seasonal_errors) > 0 else float('inf')
        else:
            mase = float('inf')

        # 3. Theil's U Statistic
        if len(hist_aligned) > 1:
            actual_changes = np.diff(hist_aligned)
            predicted_changes = np.diff(pred_aligned)
            if np.sum(actual_changes**2) > 0:
                theil_u = np.sqrt(np.sum((actual_changes - predicted_changes)**2) / np.sum(actual_changes**2))
            else:
                theil_u = float('inf')
        else:
            theil_u = float('inf')

        # 4. Directional Accuracy
        if len(hist_aligned) > 1:
            actual_direction = np.sign(np.diff(hist_aligned))
            predicted_direction = np.sign(np.diff(pred_aligned))
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        else:
            directional_accuracy = 0

        # 5. Coverage Probability (基于置信区间)
        if hasattr(prediction_result, 'confidence_intervals'):
            # 假设有置信区间信息
            lower_bound = pred_aligned - 1.96 * np.std(errors)
            upper_bound = pred_aligned + 1.96 * np.std(errors)
            coverage = np.mean((hist_aligned >= lower_bound) & (hist_aligned <= upper_bound)) * 100
        else:
            # 使用简单的标准差作为置信区间
            std_error = np.std(errors)
            lower_bound = pred_aligned - 1.96 * std_error
            upper_bound = pred_aligned + 1.96 * std_error
            coverage = np.mean((hist_aligned >= lower_bound) & (hist_aligned <= upper_bound)) * 100

        # 6. Mean Scaled Error
        if np.mean(hist_aligned) > 0:
            scaled_error = np.mean(errors) / np.mean(hist_aligned)
        else:
            scaled_error = 0

        # 7. Peak Accuracy
        if len(hist_aligned) > 0:
            peak_threshold = np.percentile(hist_aligned, 90)  # 前10%的峰值
            peak_mask = hist_aligned >= peak_threshold
            if np.sum(peak_mask) > 0:
                peak_mae = np.mean(absolute_errors[peak_mask])
                peak_accuracy = 1 / (1 + peak_mae / max(np.mean(hist_aligned), 1e-6))
            else:
                peak_accuracy = 0
        else:
            peak_accuracy = 0

        return {
            'SMAPE': smape,
            'MASE': mase,
            'Theil_U': theil_u,
            'Directional_Accuracy': directional_accuracy,
            'Coverage_Probability': coverage,
            'Scaled_Error': scaled_error,
            'Peak_Accuracy': peak_accuracy
        }

    def _calculate_statistical_metrics(self,
                                     historical_data: pd.DataFrame,
                                     prediction_result: PredictionResult) -> Dict[str, Any]:
        """计算统计检验指标"""
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            return self._get_default_statistical_metrics()

        # 对齐数据
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        errors = hist_aligned - pred_aligned

        # 1. Jarque-Bera正态性检验
        if STATS_AVAILABLE and len(errors) > 10:
            jb_stat, jb_pvalue = stats.jarque_bera(errors)
            is_normal = jb_pvalue > 0.05
        else:
            jb_stat, jb_pvalue = float('nan'), float('nan')
            is_normal = False

        # 2. Ljung-Box自相关检验
        if STATS_AVAILABLE and len(errors) > 10:
            try:
                lb_stat, lb_pvalue = stats.acorr_ljungbox(errors, lags=min(10, len(errors)//4))
                # 使用最后一个lag的p值
                lb_pvalue = lb_pvalue.iloc[-1] if hasattr(lb_pvalue, 'iloc') else lb_pvalue
                no_autocorrelation = lb_pvalue > 0.05
            except:
                lb_stat, lb_pvalue = float('nan'), float('nan')
                no_autocorrelation = False
        else:
            lb_stat, lb_pvalue = float('nan'), float('nan')
            no_autocorrelation = False

        # 3. Breusch-Pagan异方差性检验
        if STATS_AVAILABLE and len(hist_aligned) > 10:
            try:
                # 使用pred_aligned作为解释变量
                X = sm.add_constant(pred_aligned)
                model = sm.OLS(errors**2, X).fit()
                bp_stat = model.nobs * model.rsquared
                bp_pvalue = stats.chi2.sf(bp_stat, 1)
                is_homoscedastic = bp_pvalue > 0.05
            except:
                bp_stat, bp_pvalue = float('nan'), float('nan')
                is_homoscedastic = False
        else:
            bp_stat, bp_pvalue = float('nan'), float('nan')
            is_homoscedastic = False

        # 4. Diebold-Mariano预测准确性检验
        if STATS_AVAILABLE and len(hist_aligned) > 10:
            # 与朴素预测对比（使用最后一个观测值）
            naive_forecast = np.full_like(pred_aligned, hist_aligned.iloc[-1])
            dm_loss_diff = (hist_aligned - pred_aligned)**2 - (hist_aligned - naive_forecast)**2
            dm_mean = np.mean(dm_loss_diff)
            dm_var = np.var(dm_loss_diff)
            if dm_var > 0:
                dm_stat = dm_mean / np.sqrt(dm_var / len(dm_loss_diff))
                dm_pvalue = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
                better_than_naive = dm_pvalue < 0.05 and dm_mean < 0
            else:
                dm_stat, dm_pvalue = float('nan'), float('nan')
                better_than_naive = False
        else:
            dm_stat, dm_pvalue = float('nan'), float('nan')
            better_than_naive = False

        return {
            'Jarque_Bera_Stat': jb_stat,
            'Jarque_Bera_PValue': jb_pvalue,
            'Errors_Are_Normal': is_normal,
            'Ljung_Box_Stat': lb_stat,
            'Ljung_Box_PValue': lb_pvalue,
            'No_Autocorrelation': no_autocorrelation,
            'Breusch_Pagan_Stat': bp_stat,
            'Breusch_Pagan_PValue': bp_pvalue,
            'Is_Homoscedastic': is_homoscedastic,
            'Diebold_Mariano_Stat': dm_stat,
            'Diebold_Mariano_PValue': dm_pvalue,
            'Better_Than_Naive': better_than_naive
        }

    def _calculate_business_metrics(self,
                                  historical_data: pd.DataFrame,
                                  prediction_result: PredictionResult) -> Dict[str, float]:
        """计算业务相关指标"""
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            return self._get_default_business_metrics()

        # 对齐数据
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        errors = hist_aligned - pred_aligned
        absolute_errors = np.abs(errors)

        # 1. 资源利用率优化
        # 假设每个请求需要固定资源，预测准确性影响资源分配效率
        optimal_resource = hist_aligned
        predicted_resource = pred_aligned
        resource_efficiency = 1 - np.mean(absolute_errors) / np.mean(optimal_resource) if np.mean(optimal_resource) > 0 else 0

        # 2. 服务水平协议(SLA)达成率
        # 假设SLA要求预测误差在20%以内
        sla_threshold = 0.2
        relative_errors = absolute_errors / np.maximum(hist_aligned, 1e-6)
        sla_achievement = np.mean(relative_errors <= sla_threshold) * 100

        # 3. 成本节约
        # 假设过度分配成本是不足分配成本的2倍
        over_allocation_cost = np.maximum(pred_aligned - hist_aligned, 0) * 2
        under_allocation_cost = np.maximum(hist_aligned - pred_aligned, 0) * 1
        total_cost = np.sum(over_allocation_cost + under_allocation_cost)
        optimal_cost = np.sum(hist_aligned) * 1.5  # 理想成本
        cost_saving = max(0, 1 - total_cost / optimal_cost) if optimal_cost > 0 else 0

        # 4. 峰值处理能力
        peak_threshold = np.percentile(hist_aligned, 95)
        peak_periods = hist_aligned >= peak_threshold
        if np.sum(peak_periods) > 0:
            peak_accuracy = 1 - np.mean(absolute_errors[peak_periods]) / np.mean(hist_aligned[peak_periods]) if np.mean(hist_aligned[peak_periods]) > 0 else 0
        else:
            peak_accuracy = 0

        # 5. 稳定性指标
        # 预测的稳定性（变化程度）
        pred_changes = np.abs(np.diff(pred_aligned))
        hist_changes = np.abs(np.diff(hist_aligned))
        if len(hist_changes) > 0 and np.mean(hist_changes) > 0:
            stability_ratio = np.mean(pred_changes) / np.mean(hist_changes)
            stability_score = 1 / (1 + abs(stability_ratio - 1))  # 越接近1越好
        else:
            stability_score = 0

        return {
            'Resource_Efficiency': resource_efficiency,
            'SLA_Achievement': sla_achievement,
            'Cost_Saving': cost_saving,
            'Peak_Handling': peak_accuracy,
            'Stability_Score': stability_score
        }

    def _calculate_robustness_metrics(self,
                                    historical_data: pd.DataFrame,
                                    prediction_result: PredictionResult) -> Dict[str, float]:
        """计算鲁棒性指标"""
        hist_times = pd.to_datetime(historical_data['arrival_time_ns'], unit='ns')
        hist_rates = self._calculate_request_rates(historical_data, window_seconds=60)

        pred_times = [pd.to_datetime(req['arrival_time_ns'], unit='ns')
                     for req in prediction_result.predicted_requests]
        pred_rates = self._calculate_predicted_rates(pred_times, window_seconds=60)

        if len(hist_rates) == 0 or len(pred_rates) == 0:
            return self._get_default_robustness_metrics()

        # 对齐数据
        common_times = sorted(set(hist_rates.index) | set(pred_rates.index))
        hist_aligned = hist_rates.reindex(common_times, method='nearest').fillna(0)
        pred_aligned = pred_rates.reindex(common_times, method='nearest').fillna(0)

        errors = hist_aligned - pred_aligned
        absolute_errors = np.abs(errors)

        # 1. 抗异常值能力
        # 在数据中加入噪声测试鲁棒性
        noise_levels = [0.05, 0.1, 0.2]
        robustness_scores = []

        for noise_level in noise_levels:
            # 添加高斯噪声
            noisy_hist = hist_aligned + np.random.normal(0, noise_level * np.std(hist_aligned), len(hist_aligned))
            noisy_errors = np.abs(noisy_hist - pred_aligned)
            robustness = 1 - np.mean(noisy_errors) / np.mean(absolute_errors) if np.mean(absolute_errors) > 0 else 0
            robustness_scores.append(max(0, robustness))

        overall_robustness = np.mean(robustness_scores)

        # 2. 预测一致性
        # 检查预测在不同时间窗口下的一致性
        window_sizes = [30, 60, 120]  # 秒
        window_metrics = []

        for window_size in window_sizes:
            hist_window = self._calculate_request_rates(historical_data, window_seconds=window_size)
            pred_window = self._calculate_predicted_rates(pred_times, window_seconds=window_size)

            if len(hist_window) > 0 and len(pred_window) > 0:
                window_errors = np.abs(hist_window - pred_window)
                window_mae = np.mean(window_errors)
                window_metrics.append(window_mae)

        consistency_score = 1 / (1 + np.std(window_metrics)) if window_metrics and np.std(window_metrics) > 0 else 1

        # 3. 极端情况处理
        # 测试在极端值下的表现
        extreme_threshold = np.percentile(hist_aligned, 99)
        extreme_mask = hist_aligned >= extreme_threshold

        if np.sum(extreme_mask) > 0:
            extreme_errors = absolute_errors[extreme_mask]
            normal_errors = absolute_errors[~extreme_mask]

            if len(normal_errors) > 0:
                extreme_ratio = np.mean(extreme_errors) / np.mean(normal_errors)
                extreme_handling = 1 / (1 + extreme_ratio)  # 越接近1越好
            else:
                extreme_handling = 0
        else:
            extreme_handling = 1  # 没有极端值，默认满分

        # 4. 模型退化检测
        # 检查预测误差是否有增长趋势
        if len(errors) > 10:
            time_index = np.arange(len(errors))
            if STATS_AVAILABLE:
                slope, _, _, _, _ = stats.linregress(time_index, absolute_errors)
            else:
                # 简单线性回归
                slope = np.cov(time_index, absolute_errors)[0, 1] / np.var(time_index)
            degradation_rate = abs(slope) / np.mean(absolute_errors) if np.mean(absolute_errors) > 0 else 0
            degradation_score = 1 / (1 + degradation_rate)
        else:
            degradation_score = 1

        return {
            'Overall_Robustness': overall_robustness,
            'Consistency_Score': consistency_score,
            'Extreme_Handling': extreme_handling,
            'Degradation_Score': degradation_score
        }

    def _calculate_overall_score(self, basic_metrics: Dict[str, float],
                              advanced_metrics: Dict[str, float]) -> float:
        """计算总体评分"""
        # 基础指标权重 (40%)
        basic_weights = {
            'MAE': 0.3,
            'RMSE': 0.3,
            'MAPE': 0.2,
            'R²': 0.2
        }

        basic_score = 0
        for metric, weight in basic_weights.items():
            if metric in basic_metrics:
                if metric == 'R²':
                    # R²越高越好
                    basic_score += weight * min(1.0, basic_metrics[metric])
                else:
                    # 其他指标越低越好
                    basic_score += weight * (1 / (1 + basic_metrics[metric]))

        # 高级指标权重 (60%)
        advanced_weights = {
            'SMAPE': 0.15,
            'Directional_Accuracy': 0.15,
            'Coverage_Probability': 0.1,
            'Peak_Accuracy': 0.1,
            'MASE': 0.1
        }

        advanced_score = 0
        for metric, weight in advanced_weights.items():
            if metric in advanced_metrics:
                if metric in ['Directional_Accuracy', 'Coverage_Probability', 'Peak_Accuracy']:
                    # 百分比指标，越高越好
                    advanced_score += weight * (advanced_metrics[metric] / 100)
                else:
                    # 其他指标越低越好
                    advanced_score += weight * (1 / (1 + advanced_metrics[metric]))

        # 综合评分
        overall_score = 0.4 * basic_score + 0.6 * advanced_score

        return min(1.0, max(0.0, overall_score))

    def _get_default_advanced_metrics(self) -> Dict[str, float]:
        """获取默认高级指标"""
        return {
            'SMAPE': 100.0,
            'MASE': float('inf'),
            'Theil_U': float('inf'),
            'Directional_Accuracy': 0.0,
            'Coverage_Probability': 0.0,
            'Scaled_Error': 1.0,
            'Peak_Accuracy': 0.0
        }

    def _get_default_statistical_metrics(self) -> Dict[str, Any]:
        """获取默认统计指标"""
        return {
            'Jarque_Bera_Stat': float('nan'),
            'Jarque_Bera_PValue': float('nan'),
            'Errors_Are_Normal': False,
            'Ljung_Box_Stat': float('nan'),
            'Ljung_Box_PValue': float('nan'),
            'No_Autocorrelation': False,
            'Breusch_Pagan_Stat': float('nan'),
            'Breusch_Pagan_PValue': float('nan'),
            'Is_Homoscedastic': False,
            'Diebold_Mariano_Stat': float('nan'),
            'Diebold_Mariano_PValue': float('nan'),
            'Better_Than_Naive': False
        }

    def _get_default_business_metrics(self) -> Dict[str, float]:
        """获取默认业务指标"""
        return {
            'Resource_Efficiency': 0.0,
            'SLA_Achievement': 0.0,
            'Cost_Saving': 0.0,
            'Peak_Handling': 0.0,
            'Stability_Score': 0.0
        }

    def _get_default_robustness_metrics(self) -> Dict[str, float]:
        """获取默认鲁棒性指标"""
        return {
            'Overall_Robustness': 0.0,
            'Consistency_Score': 0.0,
            'Extreme_Handling': 0.0,
            'Degradation_Score': 0.0
        }

    def plot_comprehensive_metrics_analysis(self,
                                           historical_data: pd.DataFrame,
                                           prediction_result: PredictionResult,
                                           model_name: str = "Model",
                                           save_path: Optional[str] = None,
                                           show_plot: bool = True) -> None:
        """
        创建全面的评估指标分析图

        Args:
            historical_data: 历史数据
            prediction_result: 预测结果
            model_name: 模型名称
            save_path: 保存路径
            show_plot: 是否显示图表
        """
        # 计算全面指标
        comprehensive_metrics = self.calculate_comprehensive_metrics(historical_data, prediction_result)

        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comprehensive Metrics Analysis - {model_name}', fontsize=16, fontweight='bold')

        # 1. 基础指标雷达图
        self._plot_basic_metrics_radar(axes[0, 0], comprehensive_metrics['basic_metrics'], model_name)

        # 2. 高级指标条形图
        self._plot_advanced_metrics_bar(axes[0, 1], comprehensive_metrics['advanced_metrics'])

        # 3. 业务指标仪表盘
        self._plot_business_metrics_gauge(axes[0, 2], comprehensive_metrics['business_metrics'])

        # 4. 鲁棒性指标热力图
        self._plot_robustness_heatmap(axes[1, 0], comprehensive_metrics['robustness_metrics'])

        # 5. 统计检验结果
        self._plot_statistical_tests(axes[1, 1], comprehensive_metrics['statistical_metrics'])

        # 6. 总体评分
        self._plot_overall_score(axes[1, 2], comprehensive_metrics['overall_score'], model_name)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive metrics analysis saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _plot_basic_metrics_radar(self, ax, basic_metrics: Dict[str, float], model_name: str):
        """绘制基础指标雷达图"""
        metrics = ['MAE', 'RMSE', 'MAPE', 'R²']
        values = []

        for metric in metrics:
            if metric in basic_metrics:
                if metric == 'R²':
                    # R²越高越好，直接使用
                    values.append(min(1.0, basic_metrics[metric]))
                else:
                    # 其他指标越低越好，转换为0-1分数
                    values.append(1 / (1 + basic_metrics[metric]))
            else:
                values.append(0)

        # 雷达图设置
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]

        ax.clear()
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Basic Metrics (Normalized)')
        ax.grid(True)

        # 添加数值标签
        for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            ax.text(angle, value + 0.05, f'{value:.2f}', ha='center', va='bottom', fontsize=8)

    def _plot_advanced_metrics_bar(self, ax, advanced_metrics: Dict[str, float]):
        """绘制高级指标条形图"""
        metrics = ['SMAPE', 'Directional_Accuracy', 'Coverage_Probability', 'Peak_Accuracy']
        values = []
        colors = []

        for metric in metrics:
            if metric in advanced_metrics:
                values.append(advanced_metrics[metric])
                # Directional_Accuracy和Coverage_Probability是百分比，越高越好
                if metric in ['Directional_Accuracy', 'Coverage_Probability', 'Peak_Accuracy']:
                    colors.append('green')
                else:
                    # SMAPE越低越好
                    colors.append('orange' if advanced_metrics[metric] < 20 else 'red')
            else:
                values.append(0)
                colors.append('gray')

        ax.clear()
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_title('Advanced Metrics')
        ax.set_ylabel('Value')

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}{"%" if value > 1 else ""}',
                   ha='center', va='bottom', fontsize=8)

        ax.set_ylim(0, max(max(values) * 1.2, 1))

    def _plot_business_metrics_gauge(self, ax, business_metrics: Dict[str, float]):
        """绘制业务指标仪表盘"""
        metrics = list(business_metrics.keys())
        values = list(business_metrics.values())

        ax.clear()

        # 创建半圆形仪表盘
        theta = np.linspace(0, np.pi, 100)

        # 绘制每个指标的仪表
        for i, (metric, value) in enumerate(zip(metrics, values)):
            radius = 0.8 - i * 0.15
            # 背景弧
            ax.plot(theta, np.full_like(theta, radius), 'lightgray', linewidth=8)
            # 值弧
            value_angle = value * np.pi
            ax.plot(theta[:int(value_angle*100/np.pi)], np.full(int(value_angle*100/np.pi), radius),
                   'green' if value > 0.7 else 'orange' if value > 0.4 else 'red', linewidth=8)

            # 添加标签
            ax.text(-0.1, radius, metric, ha='right', va='center', fontsize=8)
            ax.text(0.1, radius, f'{value:.2f}', ha='left', va='center', fontsize=8, fontweight='bold')

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1)
        ax.set_title('Business Metrics')
        ax.set_aspect('equal')
        ax.axis('off')

    def _plot_robustness_heatmap(self, ax, robustness_metrics: Dict[str, float]):
        """绘制鲁棒性指标热力图"""
        metrics = list(robustness_metrics.keys())
        values = list(robustness_metrics.values())

        # 创建2D数组用于热力图
        data = np.array(values).reshape(-1, 1)

        ax.clear()
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # 设置刻度
        ax.set_xticks([0])
        ax.set_xticklabels(['Score'])
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics)

        # 添加数值标签
        for i, value in enumerate(values):
            ax.text(0, i, f'{value:.2f}', ha='center', va='center',
                   color='white' if value < 0.5 else 'black', fontweight='bold')

        ax.set_title('Robustness Metrics')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Score', rotation=270, labelpad=15)

    def _plot_statistical_tests(self, ax, statistical_metrics: Dict[str, Any]):
        """绘制统计检验结果"""
        test_results = []
        test_names = []
        colors = []

        # 提取关键检验结果
        tests = [
            ('Normality', statistical_metrics.get('Errors_Are_Normal', False)),
            ('No Autocorrelation', statistical_metrics.get('No_Autocorrelation', False)),
            ('Homoscedasticity', statistical_metrics.get('Is_Homoscedastic', False)),
            ('Better Than Naive', statistical_metrics.get('Better_Than_Naive', False))
        ]

        for name, result in tests:
            test_names.append(name)
            test_results.append(1 if result else 0)
            colors.append('green' if result else 'red')

        ax.clear()
        bars = ax.bar(test_names, test_results, color=colors, alpha=0.7)
        ax.set_title('Statistical Tests')
        ax.set_ylabel('Pass/Fail')
        ax.set_ylim(0, 1.2)

        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # 添加通过/失败标签
        for bar, result in zip(bars, test_results):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   'Pass' if result else 'Fail',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    def _plot_overall_score(self, ax, overall_score: float, model_name: str):
        """绘制总体评分"""
        ax.clear()

        # 创建圆形评分显示
        circle = plt.Circle((0.5, 0.5), 0.4, fill=False, linewidth=3, color='lightgray')
        ax.add_patch(circle)

        # 绘制评分弧
        score_angle = overall_score * 2 * np.pi
        theta = np.linspace(0, score_angle, 100)
        x_arc = 0.5 + 0.4 * np.cos(theta - np.pi/2)
        y_arc = 0.5 + 0.4 * np.sin(theta - np.pi/2)
        ax.plot(x_arc, y_arc, linewidth=8,
                color='green' if overall_score > 0.8 else 'orange' if overall_score > 0.6 else 'red')

        # 添加评分文本
        ax.text(0.5, 0.5, f'{overall_score:.3f}', ha='center', va='center',
                fontsize=24, fontweight='bold')
        ax.text(0.5, 0.3, model_name, ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.2, 'Overall Score', ha='center', va='center', fontsize=10, alpha=0.7)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

    def generate_metrics_report(self,
                               historical_data: pd.DataFrame,
                               prediction_result: PredictionResult,
                               model_name: str = "Model") -> Dict[str, Any]:
        """
        生成详细的评估指标报告

        Args:
            historical_data: 历史数据
            prediction_result: 预测结果
            model_name: 模型名称

        Returns:
            包含详细评估报告的字典
        """
        comprehensive_metrics = self.calculate_comprehensive_metrics(historical_data, prediction_result)

        report = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'overall_score': comprehensive_metrics['overall_score'],
                'performance_grade': self._get_performance_grade(comprehensive_metrics['overall_score']),
                'prediction_horizon': len(prediction_result.predicted_requests),
                'confidence_level': prediction_result.confidence
            },
            'basic_metrics': comprehensive_metrics['basic_metrics'],
            'advanced_metrics': comprehensive_metrics['advanced_metrics'],
            'business_metrics': comprehensive_metrics['business_metrics'],
            'robustness_metrics': comprehensive_metrics['robustness_metrics'],
            'statistical_metrics': comprehensive_metrics['statistical_metrics'],
            'recommendations': self._generate_recommendations(comprehensive_metrics)
        }

        return report

    def _get_performance_grade(self, score: float) -> str:
        """根据评分获取性能等级"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        elif score >= 0.6:
            return "Poor"
        else:
            return "Very Poor"

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """基于评估指标生成改进建议"""
        recommendations = []

        # 基于基础指标的建议
        basic = metrics['basic_metrics']
        if basic.get('MAE', float('inf')) > 10:
            recommendations.append("考虑增加训练数据量或调整模型参数以降低MAE")

        if basic.get('RMSE', float('inf')) > 15:
            recommendations.append("RMSE较高，建议检查数据质量或使用集成方法")

        if basic.get('MAPE', 100) > 20:
            recommendations.append("预测误差较大，建议重新评估特征工程或尝试不同模型")

        # 基于高级指标的建议
        advanced = metrics['advanced_metrics']
        if advanced.get('Directional_Accuracy', 0) < 60:
            recommendations.append("方向准确性较低，建议增加趋势分析特征")

        if advanced.get('Coverage_Probability', 0) < 80:
            recommendations.append("置信区间覆盖不足，建议调整置信区间计算方法")

        # 基于业务指标的建议
        business = metrics['business_metrics']
        if business.get('SLA_Achievement', 0) < 90:
            recommendations.append("SLA达成率低，建议优化预测模型或增加安全边际")

        if business.get('Resource_Efficiency', 0) < 0.7:
            recommendations.append("资源利用效率低，建议改进资源分配策略")

        # 基于鲁棒性指标的建议
        robustness = metrics['robustness_metrics']
        if robustness.get('Overall_Robustness', 0) < 0.7:
            recommendations.append("模型鲁棒性不足，建议增加数据预处理或使用集成方法")

        if robustness.get('Consistency_Score', 0) < 0.8:
            recommendations.append("预测一致性较差，建议检查数据质量或调整预测窗口")

        if not recommendations:
            recommendations.append("模型性能良好，建议定期监控和重训练")

        return recommendations

    def create_enhanced_dashboard(self,
                                prediction_results: Dict[str, PredictionResult],
                                historical_data: pd.DataFrame,
                                model_names: Optional[Dict[str, str]] = None,
                                save_dir: str = "enhanced_prediction_plots") -> None:
        """
        创建增强的完整可视化面板。

        Args:
            prediction_results: 预测结果字典 {key: PredictionResult}
            historical_data: 历史数据
            model_names: 模型名称映射
            save_dir: 保存目录
        """
        import os

        # 创建目录
        os.makedirs(save_dir, exist_ok=True)

        # 默认模型名称
        if model_names is None:
            model_names = {key: f"Model_{key}" for key in prediction_results.keys()}

        logger.info(f"创建增强可视化面板，保存到 {save_dir}")

        # 1. 创建多模型对比图
        if len(prediction_results) > 1:
            try:
                self.plot_multi_model_comparison(
                    prediction_results, historical_data,
                    f"{save_dir}/01_multi_model_comparison.png",
                    show_plot=False
                )
            except Exception as e:
                logger.error(f"创建多模型对比图失败: {e}")

        # 2. 为每个模型创建详细对比图
        for model_key, pred_result in prediction_results.items():
            model_name = model_names.get(model_key, model_key)
            try:
                self.plot_enhanced_actual_vs_predicted(
                    historical_data, pred_result, model_name,
                    f"{save_dir}/02_{model_key}_enhanced_comparison.png",
                    show_plot=False
                )
            except Exception as e:
                logger.error(f"创建{model_name}增强对比图失败: {e}")

        # 3. 如果有多个时间范围，创建时间范围分析
        if any('short' in key.lower() or 'medium' in key.lower() or 'long' in key.lower()
               for key in prediction_results.keys()):
            try:
                self.plot_multi_horizon_analysis(
                    prediction_results, historical_data,
                    f"{save_dir}/03_multi_horizon_analysis.png",
                    show_plot=False
                )
            except Exception as e:
                logger.error(f"创建多时间范围分析图失败: {e}")

        # 4. 创建综合误差分析
        for model_key, pred_result in prediction_results.items():
            try:
                self.plot_error_analysis(
                    historical_data, pred_result,
                    f"{save_dir}/04_{model_key}_error_analysis.png",
                    show_plot=False
                )
            except Exception as e:
                logger.error(f"创建{model_key}误差分析图失败: {e}")

        logger.info(f"增强可视化面板创建完成！")
        print(f"✅ 增强可视化面板已创建，包含 {len(prediction_results)} 个模型的对比分析")
        print(f"📁 结果保存在: {save_dir}/")