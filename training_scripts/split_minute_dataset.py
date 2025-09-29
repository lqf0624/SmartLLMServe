#!/usr/bin/env python3
"""
基于分钟级时间序列的BurstGPT数据集划分脚本

将1分钟窗口的时间序列数据按时间顺序划分为训练、验证、RL训练和测试集。
保留所有零值以学习完整的业务模式。
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_minute_dataset(input_path: str, output_dir: str) -> None:
    """
    划分分钟级时间序列数据集

    Args:
        input_path: 输入分钟级时间序列CSV文件路径
        output_dir: 输出目录
    """
    logger.info(f"开始划分分钟级时间序列数据集: {input_path}")

    # 加载分钟级时间序列数据
    data = pd.read_csv(input_path)
    logger.info(f"总数据量: {len(data)} 条记录")

    # 确保数据按时间排序
    data = data.sort_values('Timestamp').reset_index(drop=True)

    # 计算划分点
    total_samples = len(data)
    train_end = int(total_samples * 0.6)  # 60% 训练
    val_end = int(total_samples * 0.7)     # 10% 验证 (70% - 60%)
    rl_end = int(total_samples * 0.9)      # 20% RL训练 (90% - 70%)
    # 剩余10% 为测试

    # 划分数据集
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    rl_data = data.iloc[val_end:rl_end]
    test_data = data.iloc[rl_end:]

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存数据集
    datasets = [
        (train_data, 'burstgpt_predictor_train_minute.csv'),
        (val_data, 'burstgpt_predictor_val_minute.csv'),
        (rl_data, 'burstgpt_rl_train_minute.csv'),
        (test_data, 'burstgpt_final_test_minute.csv')
    ]

    for dataset, filename in datasets:
        file_path = output_path / filename
        dataset.to_csv(file_path, index=False)
        logger.info(f"保存 {filename}: {len(dataset)} 条记录")

    # 统计信息
    logger.info("=== 分钟级数据集划分统计 ===")
    logger.info(f"训练集: {len(train_data)} 条 ({len(train_data)/total_samples*100:.1f}%)")
    logger.info(f"验证集: {len(val_data)} 条 ({len(val_data)/total_samples*100:.1f}%)")
    logger.info(f"RL训练集: {len(rl_data)} 条 ({len(rl_data)/total_samples*100:.1f}%)")
    logger.info(f"测试集: {len(test_data)} 条 ({len(test_data)/total_samples*100:.1f}%)")

    # 时间范围统计
    logger.info("=== 时间范围统计 ===")
    logger.info(f"训练集: {train_data['Timestamp'].min()} - {train_data['Timestamp'].max()} 秒")
    logger.info(f"验证集: {val_data['Timestamp'].min()} - {val_data['Timestamp'].max()} 秒")
    logger.info(f"RL训练集: {rl_data['Timestamp'].min()} - {rl_data['Timestamp'].max()} 秒")
    logger.info(f"测试集: {test_data['Timestamp'].min()} - {test_data['Timestamp'].max()} 秒")

    # 零值和并发量统计
    logger.info("=== 零值和并发量统计 (1分钟窗口) ===")
    for name, dataset in [("训练集", train_data), ("验证集", val_data), ("RL训练集", rl_data), ("测试集", test_data)]:
        zero_ratio = (dataset['Concurrent_requests'] == 0).sum() / len(dataset) * 100
        nonzero_ratio = (dataset['Concurrent_requests'] > 0).sum() / len(dataset) * 100
        avg_concurrent = dataset['Concurrent_requests'].mean()
        max_concurrent = dataset['Concurrent_requests'].max()

        nonzero_data = dataset[dataset['Concurrent_requests'] > 0]
        if len(nonzero_data) > 0:
            avg_nonzero = nonzero_data['Concurrent_requests'].mean()
            median_nonzero = nonzero_data['Concurrent_requests'].median()
        else:
            avg_nonzero = 0
            median_nonzero = 0

        logger.info(f"{name} - 零值比例: {zero_ratio:.1f}%, 非零比例: {nonzero_ratio:.1f}%, "
                   f"平均并发量: {avg_concurrent:.2f}, 平均非零并发量: {avg_nonzero:.2f}, "
                   f"中位数非零并发量: {median_nonzero:.1f}, 最大并发量: {max_concurrent}")

def main():
    """主函数"""
    input_path = 'dataset/BurstGPT_1_compressed_60s.csv'
    output_dir = 'dataset/split'

    if not os.path.exists(input_path):
        logger.error(f"输入文件不存在: {input_path}")
        logger.info("请先运行: python3 training_scripts/compress_timeseries.py")
        return

    split_minute_dataset(input_path, output_dir)
    logger.info("分钟级时间序列数据集划分完成！")
    logger.info("保留所有零值以学习完整的业务模式，包括夜间低谷等周期性特征。")

if __name__ == "__main__":
    main()