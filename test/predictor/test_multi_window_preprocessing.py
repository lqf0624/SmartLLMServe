#!/usr/bin/env python3
"""
多窗口预处理性能测试脚本

测试新创建的时序窗口管理器和预处理配置管理器的性能和功能。
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from predictor.time_series_window_manager import TimeSeriesWindowManager, create_window_manager
from predictor.preprocessing_config_manager import PreprocessingConfigManager, create_config_manager
from predictor.workload_predictor import PredictionHorizon, ModelType
from predictor.data_preprocessor import DataPreprocessor, DataFormat

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(size: int = 2000, burst_patterns: bool = True) -> pd.DataFrame:
    """创建测试数据"""
    logger.info(f"创建测试数据，大小: {size}")

    # 生成基础时间序列
    base_times = np.arange(size) * 100_000_000  # 100ms间隔

    # 生成请求token数量（带突发模式）
    if burst_patterns:
        # 创建突发模式
        base_tokens = np.random.randint(50, 150, size)

        # 添加突发
        burst_positions = np.random.choice(size, size//10, replace=False)
        burst_tokens = np.random.randint(200, 500, len(burst_positions))
        base_tokens[burst_positions] = burst_tokens

        # 添加稀疏期
        sparse_positions = np.random.choice(size, size//5, replace=False)
        sparse_tokens = np.random.randint(10, 50, len(sparse_positions))
        base_tokens[sparse_positions] = sparse_tokens
    else:
        base_tokens = np.random.randint(80, 120, size)

    # 生成响应token数量（与输入相关）
    output_tokens = base_tokens * np.random.uniform(2, 4, size)
    output_tokens = output_tokens.astype(int)

    # 创建数据框
    data = pd.DataFrame({
        'arrival_time_ns': base_times,
        'input_toks': base_tokens,
        'output_toks': output_tokens,
        'model_type': np.random.choice(['ChatGPT', 'GPT-4'], size)
    })

    # 添加一些噪声和异常值
    noise_indices = np.random.choice(size, size//20, replace=False)
    data.loc[noise_indices, 'input_toks'] = np.random.randint(500, 1000, len(noise_indices))

    logger.info(f"测试数据创建完成，形状: {data.shape}")
    return data

def test_window_manager_performance():
    """测试窗口管理器性能"""
    logger.info("=== 测试窗口管理器性能 ===")

    # 创建测试数据
    test_data = create_test_data(size=5000)

    # 创建窗口管理器
    manager = create_window_manager()

    # 测试不同时间范围
    horizons = [PredictionHorizon.SHORT_TERM, PredictionHorizon.MEDIUM_TERM,
                PredictionHorizon.LONG_TERM, PredictionHorizon.EXTENDED_TERM]

    performance_results = {}

    for horizon in horizons:
        logger.info(f"测试 {horizon.value} 窗口性能...")

        start_time = time.time()
        windows = manager.create_multi_horizon_windows(test_data, [horizon])
        creation_time = time.time() - start_time

        # 测试特征提取
        start_time = time.time()
        features = manager.extract_window_features(windows)
        feature_time = time.time() - start_time

        # 获取统计信息
        stats = manager.get_window_statistics(windows)

        performance_results[horizon.value] = {
            'creation_time': creation_time,
            'feature_time': feature_time,
            'total_time': creation_time + feature_time,
            'window_count': stats[horizon.value]['window_count'],
            'avg_window_size': stats[horizon.value]['avg_window_size'],
            'memory_usage_mb': stats[horizon.value]['memory_usage_mb']
        }

        logger.info(f"  创建时间: {creation_time:.3f}s")
        logger.info(f"  特征时间: {feature_time:.3f}s")
        logger.info(f"  窗口数量: {stats[horizon.value]['window_count']}")
        logger.info(f"  平均窗口大小: {stats[horizon.value]['avg_window_size']:.1f}")

    # 测试多时间范围并行处理
    logger.info("\n测试多时间范围并行处理...")
    start_time = time.time()
    all_windows = manager.create_multi_horizon_windows(test_data, horizons)
    parallel_time = time.time() - start_time

    logger.info(f"多时间范围并行处理时间: {parallel_time:.3f}s")

    # 获取性能报告
    performance_report = manager.get_performance_report()
    logger.info(f"缓存命中率: {performance_report['cache_statistics']['cache_hit_rate']:.3f}")

    return performance_results, performance_report

def test_config_manager_functionality():
    """测试配置管理器功能"""
    logger.info("=== 测试配置管理器功能 ===")

    # 创建配置管理器
    manager = create_config_manager()

    # 测试模板应用
    templates = ['fast', 'balanced', 'high_quality']
    template_results = {}

    for template in templates:
        logger.info(f"测试模板: {template}")

        start_time = time.time()
        manager.apply_template(template)
        apply_time = time.time() - start_time

        # 验证配置
        is_valid, errors = manager.validate_config()

        # 获取配置摘要
        summary = manager.get_config_summary()

        template_results[template] = {
            'apply_time': apply_time,
            'is_valid': is_valid,
            'errors': errors,
            'strategy': summary['strategy'],
            'feature_count': summary['feature_count']
        }

        logger.info(f"  应用时间: {apply_time:.3f}s")
        logger.info(f"  配置有效: {is_valid}")
        if errors:
            logger.warning(f"  错误: {errors}")

    # 测试数据优化
    logger.info("\n测试数据优化...")
    test_data_sizes = [100, 1000, 5000, 10000]
    optimization_results = []

    for size in test_data_sizes:
        start_time = time.time()
        manager.optimize_for_data(data_size=size, data_quality_score=0.7, computational_budget="medium")
        opt_time = time.time() - start_time

        summary = manager.get_config_summary()
        optimization_results.append({
            'data_size': size,
            'optimization_time': opt_time,
            'strategy': summary['strategy']
        })

        logger.info(f"  数据大小: {size}, 优化时间: {opt_time:.3f}s")

    # 测试配置保存和加载
    logger.info("\n测试配置保存和加载...")
    config_path = "test_config_performance.json"

    start_time = time.time()
    manager.save_config(config_path)
    save_time = time.time() - start_time

    start_time = time.time()
    new_manager = create_config_manager()
    new_manager.load_config(config_path)
    load_time = time.time() - start_time

    logger.info(f"保存时间: {save_time:.3f}s")
    logger.info(f"加载时间: {load_time:.3f}s")

    # 清理测试文件
    if os.path.exists(config_path):
        os.remove(config_path)

    return template_results, optimization_results, {'save_time': save_time, 'load_time': load_time}

def test_end_to_end_preprocessing():
    """测试端到端预处理性能"""
    logger.info("=== 测试端到端预处理性能 ===")

    # 创建测试数据
    test_data = create_test_data(size=3000)

    # 创建管理器
    window_manager = create_window_manager()
    config_manager = create_config_manager()

    # 测试不同配置的端到端性能
    configs = ['fast', 'balanced', 'high_quality']
    e2e_results = {}

    for config_name in configs:
        logger.info(f"测试配置: {config_name}")

        # 应用配置
        config_manager.apply_template(config_name)

        start_time = time.time()

        # 1. 数据预处理
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_data(test_data)

        # 2. 窗口创建
        horizons = [PredictionHorizon.SHORT_TERM, PredictionHorizon.MEDIUM_TERM]
        windows = window_manager.create_multi_horizon_windows(processed_data, horizons)

        # 3. 特征提取
        features = window_manager.extract_window_features(windows)

        total_time = time.time() - start_time

        # 收集统计信息
        stats = window_manager.get_window_statistics(windows)
        config_summary = config_manager.get_config_summary()

        e2e_results[config_name] = {
            'total_time': total_time,
            'processing_time_ratio': {
                'data_preprocessing': 0.2,  # 估算
                'window_creation': 0.5,     # 估算
                'feature_extraction': 0.3   # 估算
            },
            'total_windows': sum(stats[h.value]['window_count'] for h in horizons),
            'total_features': sum(len(f) for f in features.values()),
            'memory_usage_mb': sum(stats[h.value]['memory_usage_mb'] for h in horizons),
            'config_strategy': config_summary['strategy']
        }

        logger.info(f"  总时间: {total_time:.3f}s")
        logger.info(f"  总窗口数: {e2e_results[config_name]['total_windows']}")
        logger.info(f"  总特征数: {e2e_results[config_name]['total_features']}")

    return e2e_results

def test_scalability():
    """测试可扩展性"""
    logger.info("=== 测试可扩展性 ===")

    data_sizes = [500, 1000, 2000, 5000]
    scalability_results = []

    for size in data_sizes:
        logger.info(f"测试数据大小: {size}")

        # 创建测试数据
        test_data = create_test_data(size=size)

        # 创建管理器
        window_manager = create_window_manager()
        config_manager = create_config_manager()

        config_manager.apply_template('balanced')

        start_time = time.time()

        # 端到端处理
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_data(test_data)

        horizons = [PredictionHorizon.SHORT_TERM, PredictionHorizon.MEDIUM_TERM]
        windows = window_manager.create_multi_horizon_windows(processed_data, horizons)

        total_time = time.time() - start_time

        stats = window_manager.get_window_statistics(windows)

        scalability_results.append({
            'data_size': size,
            'processing_time': total_time,
            'total_windows': sum(stats[h.value]['window_count'] for h in horizons),
            'throughput_windows_per_second': sum(stats[h.value]['window_count'] for h in horizons) / total_time if total_time > 0 else 0,
            'memory_usage_mb': sum(stats[h.value]['memory_usage_mb'] for h in horizons)
        })

        logger.info(f"  处理时间: {total_time:.3f}s")
        logger.info(f"  窗口吞吐量: {scalability_results[-1]['throughput_windows_per_second']:.1f} 窗口/秒")

    return scalability_results

def generate_performance_report(window_results, config_results, e2e_results, scalability_results):
    """生成性能报告"""
    logger.info("=== 生成性能报告 ===")

    report = {
        'test_timestamp': str(pd.Timestamp.now()),
        'window_manager_performance': window_results,
        'config_manager_performance': config_results,
        'end_to_end_performance': e2e_results,
        'scalability_analysis': scalability_results,
        'summary': {}
    }

    # 计算汇总统计
    report['summary']['fastest_config'] = min(e2e_results.items(), key=lambda x: x[1]['total_time'])[0]
    report['summary']['most_windows_config'] = max(e2e_results.items(), key=lambda x: x[1]['total_windows'])[0]
    report['summary']['avg_throughput'] = np.mean([r['throughput_windows_per_second'] for r in scalability_results])

    # 保存报告
    report_path = "multi_window_preprocessing_performance_report.json"
    import json
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"性能报告已保存到: {report_path}")

    # 打印关键结果
    print("\n" + "="*60)
    print("性能测试结果汇总")
    print("="*60)

    print(f"\n窗口管理器性能:")
    for horizon, results in window_results.items():
        print(f"  {horizon}: {results['total_time']:.3f}s ({results['window_count']} 窗口)")

    print(f"\n配置管理器性能:")
    for template, results in config_results[0].items():
        print(f"  {template}: {results['apply_time']:.3f}s")

    print(f"\n端到端预处理性能:")
    for config, results in e2e_results.items():
        print(f"  {config}: {results['total_time']:.3f}s ({results['total_windows']} 窗口)")

    print(f"\n可扩展性分析:")
    for result in scalability_results:
        print(f"  数据大小 {result['data_size']}: {result['throughput_windows_per_second']:.1f} 窗口/秒")

    print(f"\n汇总:")
    print(f"  最快配置: {report['summary']['fastest_config']}")
    print(f"  最多功能配置: {report['summary']['most_windows_config']}")
    print(f"  平均吞吐量: {report['summary']['avg_throughput']:.1f} 窗口/秒")

    return report

def main():
    """主测试函数"""
    logger.info("开始多窗口预处理性能测试...")

    try:
        # 测试窗口管理器性能
        window_results, window_report = test_window_manager_performance()

        # 测试配置管理器功能
        config_results = test_config_manager_functionality()

        # 测试端到端预处理性能
        e2e_results = test_end_to_end_preprocessing()

        # 测试可扩展性
        scalability_results = test_scalability()

        # 生成性能报告
        report = generate_performance_report(
            window_results, config_results, e2e_results, scalability_results
        )

        logger.info("🎉 多窗口预处理性能测试完成！")
        return True

    except Exception as e:
        logger.error(f"❌ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)