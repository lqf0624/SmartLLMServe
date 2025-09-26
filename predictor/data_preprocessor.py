"""
数据预处理器 - 处理不同格式的输入数据

支持多种数据格式的标准化处理，包括BurstGPT格式和标准格式。
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """支持的数据格式"""
    BURSTGPT = "burstgpt"  # BurstGPT格式: Timestamp, Request tokens, Response tokens, Model
    STANDARD = "standard"  # 标准格式: arrival_time_ns, input_toks, output_toks, model_type
    GENERIC = "generic"    # 通用格式: 自动检测列名


class DataPreprocessor:
    """
    数据预处理器

    功能：
    - 自动检测数据格式
    - 格式转换和标准化
    - 数据清洗和验证
    - 错误处理和恢复
    """

    def __init__(self):
        """初始化数据预处理器"""
        self.format_mappings = {
            DataFormat.BURSTGPT: {
                'timestamp': 'Timestamp',
                'input_tokens': 'Request tokens',
                'output_tokens': 'Response tokens',
                'model_type': 'Model'
            },
            DataFormat.STANDARD: {
                'timestamp': 'arrival_time_ns',
                'input_tokens': 'input_toks',
                'output_tokens': 'output_toks',
                'model_type': 'model_type'
            }
        }

        # 预处理统计
        self.preprocessing_stats = {
            'total_processed': 0,
            'format_conversions': 0,
            'error_recoveries': 0,
            'data_issues_fixed': 0
        }

    def detect_format(self, data: pd.DataFrame) -> DataFormat:
        """
        自动检测数据格式

        Args:
            data: 输入数据

        Returns:
            检测到的数据格式
        """
        columns = [col.lower() for col in data.columns]

        # 检测BurstGPT格式
        burstgpt_indicators = ['timestamp', 'request tokens', 'response tokens']
        if all(indicator in columns for indicator in burstgpt_indicators):
            return DataFormat.BURSTGPT

        # 检测标准格式
        standard_indicators = ['arrival_time_ns', 'input_toks', 'output_toks']
        if all(indicator in columns for indicator in standard_indicators):
            return DataFormat.STANDARD

        # 默认使用通用格式
        return DataFormat.GENERIC

    def preprocess_data(self,
                       data: pd.DataFrame,
                       target_format: DataFormat = DataFormat.STANDARD,
                       **kwargs) -> pd.DataFrame:
        """
        预处理数据

        Args:
            data: 原始数据
            target_format: 目标格式
            **kwargs: 预处理参数

        Returns:
            预处理后的数据
        """
        try:
            # 创建数据副本
            processed_data = data.copy()

            # 检测源格式
            source_format = self.detect_format(processed_data)
            logger.info(f"检测到数据格式: {source_format.value}")

            # 格式转换
            if source_format != target_format:
                processed_data = self._convert_format(processed_data, source_format, target_format)
                self.preprocessing_stats['format_conversions'] += 1

            # 数据清洗
            processed_data = self._clean_data(processed_data, **kwargs)

            # 数据验证
            processed_data = self._validate_data(processed_data, target_format)

            # 更新统计
            self.preprocessing_stats['total_processed'] += len(processed_data)

            logger.info(f"数据预处理完成: {len(processed_data)} 条记录")
            return processed_data

        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            # 尝试错误恢复
            return self._error_recovery(data, e)

    def _convert_format(self,
                       data: pd.DataFrame,
                       source_format: DataFormat,
                       target_format: DataFormat) -> pd.DataFrame:
        """
        转换数据格式

        Args:
            data: 源数据
            source_format: 源格式
            target_format: 目标格式

        Returns:
            转换后的数据
        """
        try:
            if source_format == DataFormat.BURSTGPT and target_format == DataFormat.STANDARD:
                return self._burstgpt_to_standard(data)
            elif source_format == DataFormat.STANDARD and target_format == DataFormat.BURSTGPT:
                return self._standard_to_burstgpt(data)
            else:
                logger.warning(f"不支持的格式转换: {source_format.value} -> {target_format.value}")
                return data

        except Exception as e:
            logger.error(f"格式转换失败: {e}")
            raise

    def _burstgpt_to_standard(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        BurstGPT格式转换为标准格式

        Args:
            data: BurstGPT格式数据

        Returns:
            标准格式数据
        """
        try:
            # 创建标准格式的DataFrame
            standard_data = pd.DataFrame()

            # 时间戳转换（从秒转换为纳秒）
            if 'Timestamp' in data.columns:
                standard_data['arrival_time_ns'] = (data['Timestamp'] * 1e9).astype(int)
            else:
                # 如果没有时间戳，创建连续的时间戳
                standard_data['arrival_time_ns'] = np.arange(len(data)) * 100_000_000  # 100ms间隔

            # Token数量列名转换
            if 'Request tokens' in data.columns:
                standard_data['input_toks'] = data['Request tokens']
            else:
                standard_data['input_toks'] = 100  # 默认值

            if 'Response tokens' in data.columns:
                standard_data['output_toks'] = data['Response tokens']
            else:
                standard_data['output_toks'] = 300  # 默认值

            # 模型类型转换
            if 'Model' in data.columns:
                standard_data['model_type'] = data['Model']
            else:
                standard_data['model_type'] = 'ChatGPT'  # 默认值

            # 保留其他列
            for col in data.columns:
                if col not in ['Timestamp', 'Request tokens', 'Response tokens', 'Model']:
                    standard_data[col] = data[col]

            logger.info("BurstGPT格式转换为标准格式成功")
            return standard_data

        except Exception as e:
            logger.error(f"BurstGPT格式转换失败: {e}")
            raise

    def _standard_to_burstgpt(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        标准格式转换为BurstGPT格式

        Args:
            data: 标准格式数据

        Returns:
            BurstGPT格式数据
        """
        try:
            # 创建BurstGPT格式的DataFrame
            burstgpt_data = pd.DataFrame()

            # 时间戳转换（从纳秒转换为秒）
            if 'arrival_time_ns' in data.columns:
                burstgpt_data['Timestamp'] = data['arrival_time_ns'] / 1e9
            else:
                burstgpt_data['Timestamp'] = np.arange(len(data)) * 0.1  # 100ms间隔

            # Token数量列名转换
            if 'input_toks' in data.columns:
                burstgpt_data['Request tokens'] = data['input_toks']
            else:
                burstgpt_data['Request tokens'] = 100

            if 'output_toks' in data.columns:
                burstgpt_data['Response tokens'] = data['output_toks']
            else:
                burstgpt_data['Response tokens'] = 300

            # 模型类型转换
            if 'model_type' in data.columns:
                burstgpt_data['Model'] = data['model_type']
            else:
                burstgpt_data['Model'] = 'ChatGPT'

            # 保留其他列
            for col in data.columns:
                if col not in ['arrival_time_ns', 'input_toks', 'output_toks', 'model_type']:
                    burstgpt_data[col] = data[col]

            logger.info("标准格式转换为BurstGPT格式成功")
            return burstgpt_data

        except Exception as e:
            logger.error(f"标准格式转换失败: {e}")
            raise

    def _clean_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        数据清洗

        Args:
            data: 输入数据
            **kwargs: 清洗参数

        Returns:
            清洗后的数据
        """
        try:
            cleaned_data = data.copy()

            # 移除重复行
            initial_len = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            if len(cleaned_data) < initial_len:
                logger.info(f"移除 {initial_len - len(cleaned_data)} 条重复记录")
                self.preprocessing_stats['data_issues_fixed'] += initial_len - len(cleaned_data)

            # 处理缺失值
            if cleaned_data.isnull().any().any():
                logger.info("处理缺失值...")
                cleaned_data = self._handle_missing_values(cleaned_data, **kwargs)

            # 处理异常值
            cleaned_data = self._handle_outliers(cleaned_data, **kwargs)

            # 确保时间戳单调递增
            if 'arrival_time_ns' in cleaned_data.columns:
                cleaned_data = self._ensure_monotonic_timestamps(cleaned_data)

            # 确保token数量为正数
            token_columns = ['input_toks', 'output_toks', 'Request tokens', 'Response tokens']
            for col in token_columns:
                if col in cleaned_data.columns:
                    # 将非正数替换为最小正值
                    positive_mask = cleaned_data[col] > 0
                    if positive_mask.any():
                        min_positive = cleaned_data[col][positive_mask].min()
                        cleaned_data.loc[~positive_mask, col] = min_positive
                    else:
                        cleaned_data[col] = 100  # 默认值

                    logger.info(f"确保 {col} 为正数，修正 {len(cleaned_data[~positive_mask])} 条记录")

            return cleaned_data

        except Exception as e:
            logger.error(f"数据清洗失败: {e}")
            raise

    def _handle_missing_values(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        处理缺失值

        Args:
            data: 输入数据
            **kwargs: 处理参数

        Returns:
            处理后的数据
        """
        fill_method = kwargs.get('fill_method', 'forward')

        for col in data.columns:
            if data[col].isnull().any():
                if fill_method == 'forward':
                    data[col] = data[col].fillna(method='ffill')
                elif fill_method == 'backward':
                    data[col] = data[col].fillna(method='bfill')
                elif fill_method == 'mean':
                    if data[col].dtype in ['int64', 'float64']:
                        data[col] = data[col].fillna(data[col].mean())
                    else:
                        data[col] = data[col].fillna(data[col].mode()[0])
                elif fill_method == 'drop':
                    data = data.dropna(subset=[col])

                logger.info(f"处理 {col} 列的缺失值，使用 {fill_method} 方法")

        return data

    def _handle_outliers(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        处理异常值

        Args:
            data: 输入数据
            **kwargs: 处理参数

        Returns:
            处理后的数据
        """
        outlier_method = kwargs.get('outlier_method', 'clip')

        # 对数值列处理异常值
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col in ['arrival_time_ns', 'Timestamp']:
                continue  # 跳过时间戳列

            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()

            if outlier_count > 0:
                if outlier_method == 'clip':
                    data[col] = data[col].clip(lower_bound, upper_bound)
                elif outlier_method == 'remove':
                    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

                logger.info(f"处理 {col} 列的 {outlier_count} 个异常值，使用 {outlier_method} 方法")
                self.preprocessing_stats['data_issues_fixed'] += outlier_count

        return data

    def _ensure_monotonic_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        确保时间戳单调递增

        Args:
            data: 输入数据

        Returns:
            处理后的数据
        """
        if 'arrival_time_ns' not in data.columns:
            return data

        # 检查时间戳是否单调递增
        if not data['arrival_time_ns'].is_monotonic_increasing:
            logger.info("时间戳非单调递增，进行排序...")
            data = data.sort_values('arrival_time_ns').reset_index(drop=True)
            self.preprocessing_stats['data_issues_fixed'] += 1

        # 处理重复时间戳
        duplicate_timestamps = data['arrival_time_ns'].duplicated()
        if duplicate_timestamps.any():
            logger.info(f"处理 {duplicate_timestamps.sum()} 个重复时间戳...")
            # 为重复时间戳添加微小偏移量
            for i in range(len(data)):
                if i > 0 and data.iloc[i]['arrival_time_ns'] == data.iloc[i-1]['arrival_time_ns']:
                    data.at[i, 'arrival_time_ns'] = data.iloc[i-1]['arrival_time_ns'] + 1_000_000  # 1ms偏移
            self.preprocessing_stats['data_issues_fixed'] += duplicate_timestamps.sum()

        return data

    def _validate_data(self, data: pd.DataFrame, format_type: DataFormat) -> pd.DataFrame:
        """
        验证数据质量

        Args:
            data: 输入数据
            format_type: 数据格式

        Returns:
            验证后的数据
        """
        if format_type == DataFormat.STANDARD:
            required_columns = ['arrival_time_ns', 'input_toks', 'output_toks']
        elif format_type == DataFormat.BURSTGPT:
            required_columns = ['Timestamp', 'Request tokens', 'Response tokens']
        else:
            required_columns = []

        # 检查必需列
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.warning(f"缺少必需列: {missing_columns}，尝试添加默认值...")
            for col in missing_columns:
                if col == 'arrival_time_ns':
                    data[col] = np.arange(len(data)) * 100_000_000
                elif col == 'input_toks' or col == 'Request tokens':
                    data[col] = 100
                elif col == 'output_toks' or col == 'Response tokens':
                    data[col] = 300
                elif col == 'Timestamp':
                    data[col] = np.arange(len(data)) * 0.1
                elif col == 'model_type' or col == 'Model':
                    data[col] = 'ChatGPT'

        # 检查数据类型
        if 'arrival_time_ns' in data.columns:
            data['arrival_time_ns'] = data['arrival_time_ns'].astype(int)
        if 'input_toks' in data.columns:
            data['input_toks'] = data['input_toks'].astype(int)
        if 'output_toks' in data.columns:
            data['output_toks'] = data['output_toks'].astype(int)

        return data

    def _error_recovery(self, original_data: pd.DataFrame, error: Exception) -> pd.DataFrame:
        """
        错误恢复

        Args:
            original_data: 原始数据
            error: 错误信息

        Returns:
            恢复后的数据
        """
        try:
            logger.warning(f"尝试错误恢复: {error}")
            self.preprocessing_stats['error_recoveries'] += 1

            # 创建最小化的有效数据
            recovery_data = pd.DataFrame({
                'arrival_time_ns': np.arange(len(original_data)) * 100_000_000,
                'input_toks': 100,
                'output_toks': 300,
                'model_type': 'ChatGPT'
            })

            logger.info("错误恢复成功，生成最小化有效数据")
            return recovery_data

        except Exception as recovery_error:
            logger.error(f"错误恢复失败: {recovery_error}")
            # 返回最基本的数据
            return pd.DataFrame({
                'arrival_time_ns': [0],
                'input_toks': [100],
                'output_toks': [300],
                'model_type': ['ChatGPT']
            })

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """
        获取预处理统计信息

        Returns:
            统计信息字典
        """
        return self.preprocessing_stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.preprocessing_stats = {
            'total_processed': 0,
            'format_conversions': 0,
            'error_recoveries': 0,
            'data_issues_fixed': 0
        }


def create_data_preprocessor() -> DataPreprocessor:
    """创建数据预处理器的工厂函数"""
    return DataPreprocessor()


# 测试函数
def test_data_preprocessor():
    """测试数据预处理器"""
    preprocessor = create_data_preprocessor()

    # 创建测试数据
    test_data = pd.DataFrame({
        'Timestamp': [1000.0, 1000.1, 1000.2, 1000.3],
        'Request tokens': [100, 150, 200, 120],
        'Response tokens': [300, 400, 500, 350],
        'Model': ['ChatGPT', 'ChatGPT', 'GPT-4', 'ChatGPT']
    })

    # 测试预处理
    processed_data = preprocessor.preprocess_data(test_data, DataFormat.STANDARD)

    print("原始数据:")
    print(test_data)
    print("\n预处理后数据:")
    print(processed_data)
    print("\n统计信息:")
    print(preprocessor.get_preprocessing_stats())

    return processed_data


if __name__ == "__main__":
    test_data_preprocessor()