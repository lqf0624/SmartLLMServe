#!/usr/bin/env python3
"""
压缩稠密时间序列数据

通过10秒时间窗口聚合减少数据量，提高可视化和训练效率。
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compress_timeseries(input_path: str, output_path: str, window_size: int = 10) -> None:
    """
    压缩时间序列数据

    Args:
        input_path: 输入稠密时间序列CSV文件路径
        output_path: 输出压缩后的CSV文件路径
        window_size: 时间窗口大小（秒）
    """
    logger.info(f"开始压缩时间序列数据: {input_path}")
    logger.info(f"时间窗口大小: {window_size} 秒")

    # 加载稠密时间序列数据
    data = pd.read_csv(input_path)
    logger.info(f"原始数据: {len(data)} 条记录")

    # 创建时间窗口
    data['time_window'] = (data['Timestamp'] // window_size) * window_size

    # 按时间窗口聚合
    compressed = data.groupby('time_window').agg({
        'Concurrent_requests': 'sum',           # 窗口内总并发量
        'Request_tokens_sum': 'sum',             # 窗口内总输入token
        'Response_tokens_sum': 'sum',            # 窗口内总输出token
        'Timestamp': 'count'                     # 窗口内秒数（用于计算间隔）
    }).reset_index()

    # 重命名列
    compressed.columns = ['Timestamp', 'Concurrent_requests', 'Request_tokens_sum', 'Response_tokens_sum', 'seconds_in_window']

    # 计算实际时间间隔
    compressed['Time_interval'] = compressed['Timestamp'].diff().fillna(window_size)

    # 计算平均值（可选）
    compressed['Avg_concurrent_requests'] = compressed['Concurrent_requests'] / compressed['seconds_in_window']
    compressed['Avg_request_tokens_per_sec'] = compressed['Request_tokens_sum'] / compressed['seconds_in_window']
    compressed['Avg_response_tokens_per_sec'] = compressed['Response_tokens_sum'] / compressed['seconds_in_window']

    # 重新排序列
    compressed = compressed[[
        'Timestamp', 'Time_interval', 'Concurrent_requests', 'Request_tokens_sum', 'Response_tokens_sum',
        'Avg_concurrent_requests', 'Avg_request_tokens_per_sec', 'Avg_response_tokens_per_sec'
    ]]

    logger.info(f"压缩后数据: {len(compressed)} 条记录")
    logger.info(f"压缩比: {len(data)/len(compressed):.1f}:1")
    logger.info(f"时间范围: {compressed['Timestamp'].min()} - {compressed['Timestamp'].max()} 秒")

    # 统计信息
    total_windows = len(compressed)
    windows_with_requests = (compressed['Concurrent_requests'] > 0).sum()
    windows_without_requests = (compressed['Concurrent_requests'] == 0).sum()

    logger.info("压缩数据统计:")
    logger.info(f"  总窗口数: {total_windows}")
    logger.info(f"  有请求的窗口数: {windows_with_requests} ({windows_with_requests/total_windows*100:.1f}%)")
    logger.info(f"  无请求的窗口数: {windows_without_requests} ({windows_without_requests/total_windows*100:.1f}%)")
    logger.info(f"  平均并发量: {compressed['Concurrent_requests'].mean():.2f}")
    logger.info(f"  最大并发量: {compressed['Concurrent_requests'].max()}")

    # 保存数据
    compressed.to_csv(output_path, index=False)
    logger.info(f"压缩完成，保存到: {output_path}")

def main():
    """主函数"""
    input_path = 'dataset/BurstGPT_1_dense_timeseries.csv'
    output_path = 'dataset/BurstGPT_1_compressed_60s.csv'
    window_size = 60  # 1分钟

    if not os.path.exists(input_path):
        logger.error(f"输入文件不存在: {input_path}")
        logger.info("请先运行: python3 training_scripts/convert_to_dense_timeseries.py --convert-original")
        return

    compress_timeseries(input_path, output_path, window_size)
    logger.info("1分钟窗口压缩完成！")

if __name__ == "__main__":
    main()