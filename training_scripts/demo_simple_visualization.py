#!/usr/bin/env python3
"""
简化版可视化模块演示

展示核心预测器功能的可视化效果：
- 历史窗口、预测窗口对比
- 基础评估指标（MSE、MAE、RMSE、MAPE、R²）
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from predictor.workload_predictor import WorkloadPredictor, ModelType, PredictionHorizon
from predictor.simple_visualization import SimplePredictionVisualizer


def create_demo_data():
    """创建演示数据"""
    print("创建演示数据...")

    # 生成时间戳
    base_time = pd.Timestamp('2023-01-01 10:00:00')
    timestamps = [base_time + pd.Timedelta(milliseconds=i*100) for i in range(200)]

    # 生成有趋势和季节性的数据
    t = np.arange(200)
    trend = 25 + 0.1 * t
    seasonal = 8 * np.sin(2 * np.pi * t / 50) + 4 * np.sin(2 * np.pi * t / 20)
    noise = np.random.normal(0, 3, 200)

    input_toks = np.maximum(15, trend + seasonal + noise).astype(int)
    output_toks = (input_toks * np.random.uniform(2, 4, 200)).astype(int)

    return pd.DataFrame({
        'arrival_time_ns': timestamps,
        'input_toks': input_toks,
        'output_toks': output_toks,
        'model_type': ['ChatGPT'] * 200
    })


def main():
    """主演示函数"""
    print("=== LLM预测器 - 简化版可视化演示 ===\n")

    # 创建数据
    demo_data = create_demo_data()
    print(f"[OK] 演示数据创建完成: {len(demo_data)} 条记录")

    # 创建LSTM预测器
    print("\n[STEP] 初始化LSTM预测器...")
    predictor = WorkloadPredictor(
        model_type=ModelType.LSTM,
        prediction_horizon=PredictionHorizon.SHORT_TERM,
        sequence_length=30
    )

    # 训练模型
    print("[STEP] 训练预测模型...")
    training_results = predictor.train(demo_data, validation_split=0.2)
    print(f"[OK] 模型训练完成")

    # 进行预测
    print("\n[STEP] 进行预测...")
    historical_data = demo_data.iloc[:-30]
    prediction_result = predictor.predict(historical_data, steps=15)
    print(f"[OK] 预测完成: {len(prediction_result.predicted_requests)} 个预测请求")

    # 创建简化可视化器
    print("\n[STEP] 生成可视化结果...")
    visualizer = SimplePredictionVisualizer()

    # 生成基础对比图
    print("1. 基础预测对比图...")
    metrics = visualizer.plot_prediction_comparison(
        demo_data, prediction_result, "LSTM Model",
        save_path="demo_prediction_comparison.png",
        show_plot=False
    )
    print(f"   [OK] 对比图已保存")
    print(f"   [METRICS] MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.1f}%")

    # 生成报告
    print("\n2. 生成预测报告...")
    prediction_results = {'lstm': prediction_result}
    report = visualizer.generate_simple_report(
        prediction_results, demo_data, {'lstm': 'LSTM Model'},
        save_path="demo_prediction_report.txt"
    )
    print(f"   [OK] 报告已保存")

    print("\n=== 演示完成 ===")
    print("[FILES] 生成的文件:")
    print("   - demo_prediction_comparison.png: 预测对比图")
    print("   - demo_prediction_report.txt: 详细报告")

    print("\n[FEATURES] 核心功能:")
    print("   [OK] 历史数据窗口展示")
    print("   [OK] 预测数据窗口展示")
    print("   [OK] 基础评估指标计算 (MAE, MSE, RMSE, MAPE, R²)")
    print("   [OK] 多模型对比支持")
    print("   [OK] 简洁的文本报告生成")


if __name__ == "__main__":
    main()