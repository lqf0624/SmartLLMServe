"""
BurstGPT数据清洗和预处理模块

处理数据质量问题，为模型训练准备高质量数据集。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BurstGPTDataCleaner:
    """
    BurstGPT数据清洗器

    功能：
    - 处理重复时间戳
    - 处理零值和异常值
    - 数据标准化
    - 时间序列对齐
    """

    def __init__(self,
                 remove_zero_tokens: bool = True,
                 handle_duplicates: str = 'adjust',
                 handle_outliers: str = 'clip',
                 scale_features: bool = True):
        """
        初始化数据清洗器

        Args:
            remove_zero_tokens: 是否移除零token记录
            handle_duplicates: 重复时间戳处理方法 ('adjust', 'remove', 'merge')
            handle_outliers: 异常值处理方法 ('clip', 'remove', 'transform')
            scale_features: 是否标准化特征
        """
        self.remove_zero_tokens = remove_zero_tokens
        self.handle_duplicates = handle_duplicates
        self.handle_outliers = handle_outliers
        self.scale_features = scale_features

        # 统计信息
        self.original_stats = {}
        self.cleaning_stats = {}
        self.scaler = RobustScaler() if scale_features else None

        logger.info("BurstGPT数据清洗器初始化完成")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的数据清洗流程

        Args:
            df: 原始数据DataFrame

        Returns:
            pd.DataFrame: 清洗后的数据
        """
        logger.info("开始数据清洗流程...")

        # 记录原始统计信息
        self.original_stats = self._get_data_stats(df)
        logger.info(f"原始数据: {self.original_stats['total_records']} 条记录")

        cleaned_df = df.copy()

        # 1. 处理重复时间戳
        cleaned_df = self._handle_duplicate_timestamps(cleaned_df)

        # 2. 处理零值记录
        cleaned_df = self._handle_zero_tokens(cleaned_df)

        # 3. 处理异常值
        cleaned_df = self._handle_outliers(cleaned_df)

        # 4. 时间序列对齐和特征工程
        cleaned_df = self._align_time_series(cleaned_df)

        # 5. 数据标准化
        cleaned_df = self._scale_features(cleaned_df)

        # 记录清洗后统计信息
        self.cleaning_stats = self._get_data_stats(cleaned_df)

        # 生成清洗报告
        self._generate_cleaning_report()

        logger.info(f"清洗完成: {self.cleaning_stats['total_records']} 条记录")
        logger.info(f"数据保留率: {self.cleaning_stats['total_records']/self.original_stats['total_records']*100:.2f}%")

        return cleaned_df

    def _handle_duplicate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理重复时间戳"""
        logger.info("处理重复时间戳...")

        duplicate_count = df['Timestamp'].duplicated().sum()
        logger.info(f"发现 {duplicate_count} 个重复时间戳")

        if duplicate_count == 0:
            return df

        if self.handle_duplicates == 'adjust':
            # 为重复时间戳添加微小偏移量
            df = self._adjust_duplicate_timestamps(df)
        elif self.handle_duplicates == 'remove':
            # 保留每个时间戳的第一个记录
            df = df.drop_duplicates(subset=['Timestamp'], keep='first')
        elif self.handle_duplicates == 'merge':
            # 合并相同时间戳的记录
            df = self._merge_duplicate_timestamps(df)

        return df

    def _adjust_duplicate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """调整重复时间戳"""
        # 按时间戳分组
        grouped = df.groupby('Timestamp')

        adjusted_dfs = []
        for timestamp, group in grouped:
            if len(group) == 1:
                adjusted_dfs.append(group)
            else:
                # 为重复记录添加微小时间偏移
                for i, (_, row) in enumerate(group.iterrows()):
                    new_row = row.copy()
                    new_row['Timestamp'] += i * 0.001  # 添加1ms的偏移
                    adjusted_dfs.append(new_row.to_frame().T)

        result = pd.concat(adjusted_dfs, ignore_index=True)
        result = result.sort_values('Timestamp').reset_index(drop=True)
        return result

    def _merge_duplicate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """合并重复时间戳的记录"""
        # 按时间戳分组并聚合
        aggregated = df.groupby('Timestamp').agg({
            'Model': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'Request tokens': 'sum',
            'Response tokens': 'sum',
            'Total tokens': 'sum',
            'Log Type': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        }).reset_index()

        return aggregated

    def _handle_zero_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理零值记录"""
        logger.info("处理零值记录...")

        zero_request = (df['Request tokens'] == 0).sum()
        zero_response = (df['Response tokens'] == 0).sum()

        logger.info(f"零请求token: {zero_request} 个")
        logger.info(f"零响应token: {zero_response} 个")

        if not self.remove_zero_tokens:
            # 使用插值填充零值
            df = self._impute_zero_tokens(df)
        else:
            # 移除零值记录
            zero_mask = (df['Request tokens'] == 0) | (df['Response tokens'] == 0)
            df = df[~zero_mask]

        return df

    def _impute_zero_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """插值填充零值"""
        # 为零值使用中位数填充
        median_request = df[df['Request tokens'] > 0]['Request tokens'].median()
        median_response = df[df['Response tokens'] > 0]['Response tokens'].median()

        df.loc[df['Request tokens'] == 0, 'Request tokens'] = median_request
        df.loc[df['Response tokens'] == 0, 'Response tokens'] = median_response

        # 重新计算Total tokens
        df['Total tokens'] = df['Request tokens'] + df['Response tokens']

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        logger.info("处理异常值...")

        # 使用IQR方法检测异常值
        q1_request = df['Request tokens'].quantile(0.25)
        q3_request = df['Request tokens'].quantile(0.75)
        iqr_request = q3_request - q1_request

        q1_response = df['Response tokens'].quantile(0.25)
        q3_response = df['Response tokens'].quantile(0.75)
        iqr_response = q3_response - q1_response

        # 计算异常值边界
        lower_bound_request = max(0, q1_request - 1.5 * iqr_request)
        upper_bound_request = q3_request + 1.5 * iqr_request

        lower_bound_response = max(0, q1_response - 1.5 * iqr_response)
        upper_bound_response = q3_response + 1.5 * iqr_response

        outliers_before = ((df['Request tokens'] < lower_bound_request) |
                          (df['Request tokens'] > upper_bound_request) |
                          (df['Response tokens'] < lower_bound_response) |
                          (df['Response tokens'] > upper_bound_response)).sum()

        logger.info(f"发现 {outliers_before} 个异常值")

        if self.handle_outliers == 'clip':
            # 截断异常值
            df['Request tokens'] = df['Request tokens'].clip(lower_bound_request, upper_bound_request)
            df['Response tokens'] = df['Response tokens'].clip(lower_bound_response, upper_bound_response)
        elif self.handle_outliers == 'remove':
            # 移除异常值
            outlier_mask = ((df['Request tokens'] < lower_bound_request) |
                           (df['Request tokens'] > upper_bound_request) |
                           (df['Response tokens'] < lower_bound_response) |
                           (df['Response tokens'] > upper_bound_response))
            df = df[~outlier_mask]
        elif self.handle_outliers == 'transform':
            # 对数变换处理异常值
            df['Request tokens'] = np.log1p(df['Request tokens'])
            df['Response tokens'] = np.log1p(df['Response tokens'])

        # 重新计算Total tokens
        df['Total tokens'] = df['Request tokens'] + df['Response tokens']

        return df

    def _align_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """时间序列对齐"""
        logger.info("时间序列对齐...")

        # 确保时间戳单调递增
        df = df.sort_values('Timestamp').reset_index(drop=True)

        # 计算时间间隔
        time_diffs = np.diff(df['Timestamp'].values)

        # 处理过大的时间间隔
        median_interval = np.median(time_diffs)
        large_gap_threshold = median_interval * 10  # 10倍中位数为大间隔

        # 这里可以添加插值逻辑来填补大间隔
        # 目前暂时保留原样

        # 添加时间特征
        df = self._add_temporal_features(df)

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        # 计算时间间隔
        df['time_interval'] = df['Timestamp'].diff().fillna(0)

        # 添加相对时间特征
        df['relative_time'] = (df['Timestamp'] - df['Timestamp'].min()) / (df['Timestamp'].max() - df['Timestamp'].min())

        # 添加时间窗口特征
        window_size = 100
        df['arrival_rate_window'] = df.rolling(window=window_size, on='Timestamp')['Timestamp'].count()

        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据标准化"""
        if not self.scale_features:
            return df

        logger.info("数据标准化...")

        # 选择需要标准化的数值特征
        numeric_features = ['Request tokens', 'Response tokens', 'Total tokens', 'time_interval']
        available_features = [col for col in numeric_features if col in df.columns]

        if available_features:
            # 使用RobustScaler处理异常值
            scaled_data = self.scaler.fit_transform(df[available_features])

            for i, col in enumerate(available_features):
                df[f'{col}_scaled'] = scaled_data[:, i]

        return df

    def _get_data_stats(self, df: pd.DataFrame) -> Dict:
        """获取数据统计信息"""
        stats = {
            'total_records': len(df),
            'time_span': df['Timestamp'].max() - df['Timestamp'].min(),
            'duplicate_timestamps': df['Timestamp'].duplicated().sum(),
            'zero_request_tokens': (df['Request tokens'] == 0).sum(),
            'zero_response_tokens': (df['Response tokens'] == 0).sum(),
            'avg_request_tokens': df['Request tokens'].mean(),
            'avg_response_tokens': df['Response tokens'].mean(),
            'model_distribution': df['Model'].value_counts().to_dict()
        }
        return stats

    def _generate_cleaning_report(self):
        """生成清洗报告"""
        logger.info("=== 数据清洗报告 ===")
        logger.info(f"原始记录数: {self.original_stats['total_records']:,}")
        logger.info(f"清洗后记录数: {self.cleaning_stats['total_records']:,}")
        logger.info(f"数据保留率: {self.cleaning_stats['total_records']/self.original_stats['total_records']*100:.2f}%")
        logger.info(f"重复时间戳处理: {self.original_stats['duplicate_timestamps']} -> {self.cleaning_stats['duplicate_timestamps']}")
        logger.info(f"零请求token: {self.original_stats['zero_request_tokens']} -> {self.cleaning_stats['zero_request_tokens']}")
        logger.info(f"零响应token: {self.original_stats['zero_response_tokens']} -> {self.cleaning_stats['zero_response_tokens']}")

    def save_cleaned_data(self, df: pd.DataFrame, output_path: str):
        """保存清洗后的数据"""
        df.to_csv(output_path, index=False)
        logger.info(f"清洗后的数据已保存到: {output_path}")

    def load_and_clean(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载并清洗数据的便捷方法

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径（可选）

        Returns:
            pd.DataFrame: 清洗后的数据
        """
        logger.info(f"加载数据: {input_path}")

        # 加载数据
        df = pd.read_csv(input_path)

        # 清洗数据
        cleaned_df = self.clean_data(df)

        # 保存清洗后的数据
        if output_path:
            self.save_cleaned_data(cleaned_df, output_path)

        return cleaned_df


def create_preprocessing_pipeline() -> BurstGPTDataCleaner:
    """
    创建标准的数据预处理管道

    Returns:
        BurstGPTDataCleaner: 配置好的数据清洗器
    """
    return BurstGPTDataCleaner(
        remove_zero_tokens=True,
        handle_duplicates='adjust',
        handle_outliers='clip',
        scale_features=True
    )


if __name__ == "__main__":
    # 示例用法
    cleaner = create_preprocessing_pipeline()

    # 加载并清洗数据
    cleaned_data = cleaner.load_and_clean(
        input_path="../dataset/BurstGPT_1.csv",
        output_path="../dataset/BurstGPT_1_cleaned.csv"
    )

    print("数据清洗完成！")