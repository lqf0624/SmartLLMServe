#!/usr/bin/env python3
"""
将BurstGPT数据转换为稠密时间序列格式

以秒为单位创建稠密时间序列，每秒都有数据点，
没有请求的秒数值为0，适合DLinear预测。
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_dense_timeseries(input_path: str, output_path: str) -> None:
    """
    将BurstGPT数据转换为稠密时间序列格式

    Args:
        input_path: 输入CSV文件路径
        output_path: 输出CSV文件路径
    """
    logger.info(f"开始转换数据: {input_path}")

    # 加载原始数据
    data = pd.read_csv(input_path)
    logger.info(f"原始数据: {len(data)} 条记录")

    # 获取时间范围
    min_time = int(data['Timestamp'].min())
    max_time = int(data['Timestamp'].max())
    logger.info(f"时间范围: {min_time} - {max_time} 秒 (共 {max_time - min_time + 1} 秒)")

    # 创建稠密时间轴
    all_seconds = np.arange(min_time, max_time + 1)
    dense_data = pd.DataFrame({'Timestamp': all_seconds})

    # 按时间戳聚合原始数据
    aggregated = data.groupby('Timestamp').agg({
        'Request tokens': ['sum', 'count'],
        'Response tokens': 'sum'
    }).reset_index()

    # 重命名列
    aggregated.columns = ['Timestamp', 'Request_tokens_sum', 'Concurrent_requests', 'Response_tokens_sum']

    # 将聚合数据合并到稠密时间轴
    dense_data = dense_data.merge(aggregated, on='Timestamp', how='left')

    # 填充缺失值（没有请求的秒数）
    dense_data['Concurrent_requests'] = dense_data['Concurrent_requests'].fillna(0).astype(int)
    dense_data['Request_tokens_sum'] = dense_data['Request_tokens_sum'].fillna(0).astype(int)
    dense_data['Response_tokens_sum'] = dense_data['Response_tokens_sum'].fillna(0).astype(int)

    # 计算时间间隔（固定为1秒）
    dense_data['Time_interval'] = 1.0

    # 重新排序列
    dense_data = dense_data[['Timestamp', 'Time_interval', 'Concurrent_requests', 'Request_tokens_sum', 'Response_tokens_sum']]

    logger.info(f"稠密时间序列数据: {len(dense_data)} 条记录")
    logger.info(f"时间范围: {dense_data['Timestamp'].min()} - {dense_data['Timestamp'].max()} 秒")

    # 统计信息
    total_seconds = len(dense_data)
    seconds_with_requests = (dense_data['Concurrent_requests'] > 0).sum()
    seconds_without_requests = (dense_data['Concurrent_requests'] == 0).sum()

    logger.info("稠密数据统计:")
    logger.info(f"  总秒数: {total_seconds}")
    logger.info(f"  有请求的秒数: {seconds_with_requests} ({seconds_with_requests/total_seconds*100:.1f}%)")
    logger.info(f"  无请求的秒数: {seconds_without_requests} ({seconds_without_requests/total_seconds*100:.1f}%)")

    logger.info("并发量统计:")
    logger.info(f"  均值: {dense_data['Concurrent_requests'].mean():.2f}")
    logger.info(f"  最大值: {dense_data['Concurrent_requests'].max()}")
    logger.info(f"  非零值占比: {(dense_data['Concurrent_requests'] > 0).sum()/len(dense_data)*100:.1f}%")

    logger.info("请求Token统计:")
    logger.info(f"  每秒总和均值: {dense_data['Request_tokens_sum'].mean():.1f}")
    logger.info(f"  每秒总和最大值: {dense_data['Request_tokens_sum'].max()}")

    logger.info("响应Token统计:")
    logger.info(f"  每秒总和均值: {dense_data['Response_tokens_sum'].mean():.1f}")
    logger.info(f"  每秒总和最大值: {dense_data['Response_tokens_sum'].max()}")

    # 保存数据
    dense_data.to_csv(output_path, index=False)
    logger.info(f"转换完成，保存到: {output_path}")

def convert_original_dataset():
    """转换原始BurstGPT数据集"""
    input_path = Path('dataset/BurstGPT_1.csv')
    output_path = Path('dataset/BurstGPT_1_dense_timeseries.csv')

    if input_path.exists():
        convert_to_dense_timeseries(str(input_path), str(output_path))
        return True
    else:
        logger.error(f"原始数据集不存在: {input_path}")
        return False

def analyze_dense_patterns():
    """分析稠密时间序列模式"""
    logger.info("=== 分析稠密时间序列模式 ===")

    # 加载转换后的数据
    train_data = pd.read_csv('dataset/BurstGPT_1_dense_timeseries.csv')

    # 分析并发量分布
    concurrent = train_data['Concurrent_requests']

    print(f"稠密时间序列统计:")
    print(f"  总秒数: {len(concurrent)}")
    print(f"  有请求的秒数: {(concurrent > 0).sum()} ({(concurrent > 0).sum()/len(concurrent)*100:.1f}%)")
    print(f"  无请求的秒数: {(concurrent == 0).sum()} ({(concurrent == 0).sum()/len(concurrent)*100:.1f}%)")

    print(f"\n并发量统计 (仅包含有请求的秒数):")
    nonzero_concurrent = concurrent[concurrent > 0]
    print(f"  均值: {nonzero_concurrent.mean():.2f}")
    print(f"  中位数: {nonzero_concurrent.median():.2f}")
    print(f"  标准差: {nonzero_concurrent.std():.2f}")
    print(f"  最大值: {nonzero_concurrent.max()}")

    # 分析分布
    print(f"\n并发量分布:")
    print(f"  0个请求: {(concurrent == 0).sum()} ({(concurrent == 0).sum()/len(concurrent)*100:.1f}%)")
    print(f"  1个请求: {(concurrent == 1).sum()} ({(concurrent == 1).sum()/len(concurrent)*100:.1f}%)")
    print(f"  2-5个请求: {((concurrent >= 2) & (concurrent <= 5)).sum()} ({((concurrent >= 2) & (concurrent <= 5)).sum()/len(concurrent)*100:.1f}%)")
    print(f"  6-10个请求: {((concurrent >= 6) & (concurrent <= 10)).sum()} ({((concurrent >= 6) & (concurrent <= 10)).sum()/len(concurrent)*100:.1f}%)")
    print(f"  >10个请求: {(concurrent > 10).sum()} ({(concurrent > 10).sum()/len(concurrent)*100:.1f}%)")

    # 分析自相关性（时序特性）
    print(f"\n时序特性:")
    print(f"  自相关性(滞后1): {concurrent.autocorr(lag=1):.3f}")
    print(f"  自相关性(滞后5): {concurrent.autocorr(lag=5):.3f}")
    print(f"  自相关性(滞后10): {concurrent.autocorr(lag=10):.3f}")

    # 分析零值连续性
    zero_streaks = []
    current_streak = 0
    for val in concurrent:
        if val == 0:
            current_streak += 1
        else:
            if current_streak > 0:
                zero_streaks.append(current_streak)
            current_streak = 0

    if zero_streaks:
        print(f"\n零值连续性分析:")
        print(f"  平均零值连续长度: {np.mean(zero_streaks):.1f} 秒")
        print(f"  最长零值连续长度: {max(zero_streaks)} 秒")
        print(f"  零值连续段数量: {len(zero_streaks)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='转换BurstGPT数据为稠密时间序列格式')
    parser.add_argument('--convert-original', action='store_true', help='转换原始BurstGPT数据集')
    parser.add_argument('--analyze', action='store_true', help='分析稠密时间序列模式')

    args = parser.parse_args()

    if args.convert_original:
        if convert_original_dataset():
            logger.info("原始数据集稠密转换完成")
        else:
            logger.error("原始数据集稠密转换失败")

    if args.analyze:
        analyze_dense_patterns()

    if not args.convert_original and not args.analyze:
        print("请指定 --convert-original 或 --analyze 参数")
        print("示例:")
        print("  python3 convert_to_dense_timeseries.py --convert-original")
        print("  python3 convert_to_dense_timeseries.py --analyze")