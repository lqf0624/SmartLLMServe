import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import os


class UniversalDataLoader:
    """
    Universal data loader for multiple LLM serving dataset formats.

    Supports:
    - Original TSV format (ShareGPT)
    - BurstGPT CSV format
    - Custom formats with configurable mappings
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.supported_formats = {
            '.tsv': self._load_tsv_format,
            '.csv': self._load_csv_format
        }

    def load_dataset(self, file_path: str, req_num: Optional[int] = None) -> pd.DataFrame:
        """
        Load and normalize dataset from file path.

        Args:
            file_path: Path to dataset file
            req_num: Maximum number of requests to load

        Returns:
            Normalized DataFrame with standard columns
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Get file extension
        _, ext = os.path.splitext(file_path.lower())

        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(self.supported_formats.keys())}")

        # Load data using format-specific loader
        loader_func = self.supported_formats[ext]
        raw_data = loader_func(file_path)

        # Normalize to standard format
        normalized_data = self._normalize_to_standard_format(raw_data, ext)

        # Limit number of requests if specified
        if req_num is not None and len(normalized_data) > req_num:
            normalized_data = normalized_data.head(req_num)

        if self.verbose:
            print(f"DataLoader: Loaded {len(normalized_data)} requests from {file_path}")
            print(f"DataLoader: Format detected: {ext}")
            print(f"DataLoader: Columns: {list(normalized_data.columns)}")

        return normalized_data

    def _load_tsv_format(self, file_path: str) -> pd.DataFrame:
        """Load original TSV format (ShareGPT style)."""
        return pd.read_csv(file_path, sep='\t')

    def _load_csv_format(self, file_path: str) -> pd.DataFrame:
        """Load CSV format (BurstGPT style)."""
        return pd.read_csv(file_path)

    def _normalize_to_standard_format(self, data: pd.DataFrame, source_format: str) -> pd.DataFrame:
        """
        Normalize different data formats to standard format.

        Standard format columns:
        - input_toks: Number of input tokens
        - output_toks: Number of output tokens
        - arrival_time_ns: Arrival time in nanoseconds
        - model_type: Model type (optional)
        - burst_pattern: Burst pattern identifier (optional)
        """

        if source_format == '.tsv':
            return self._normalize_tsv_data(data)
        elif source_format == '.csv':
            return self._normalize_csv_data(data)
        else:
            raise ValueError(f"Normalization not implemented for format: {source_format}")

    def _normalize_tsv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize TSV data to standard format."""
        normalized = pd.DataFrame()

        # Map columns directly
        normalized['input_toks'] = data['input_toks'].astype(int)
        normalized['output_toks'] = data['output_toks'].astype(int)
        normalized['arrival_time_ns'] = data['arrival_time_ns'].astype(int)

        # Add optional columns with default values
        normalized['model_type'] = 'unknown'
        normalized['burst_pattern'] = self._detect_burst_pattern(normalized['arrival_time_ns'])

        return normalized

    def _normalize_csv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize BurstGPT CSV data to standard format."""
        normalized = pd.DataFrame()

        # Map BurstGPT columns to standard format
        normalized['input_toks'] = data['Request tokens'].astype(int)
        normalized['output_toks'] = data['Response tokens'].astype(int)

        # Convert timestamp to nanoseconds (assuming BurstGPT uses milliseconds)
        if 'Timestamp' in data.columns:
            normalized['arrival_time_ns'] = (data['Timestamp'] * 1_000_000).astype(int)
        else:
            # If no timestamp, generate sequential arrival times
            normalized['arrival_time_ns'] = self._generate_sequential_times(len(data))

        # Extract model type
        normalized['model_type'] = data.get('Model', 'unknown')

        # Detect burst patterns
        normalized['burst_pattern'] = self._detect_burst_pattern(normalized['arrival_time_ns'])

        return normalized

    def _detect_burst_pattern(self, arrival_times: pd.Series) -> pd.Series:
        """
        Detect burst patterns in arrival times.

        Returns:
            Series with burst pattern identifiers
        """
        if len(arrival_times) == 0:
            return pd.Series(dtype=str)

        # Calculate inter-arrival times
        sorted_times = arrival_times.sort_values()
        inter_arrival = sorted_times.diff().dropna()

        if len(inter_arrival) == 0:
            return pd.Series(['steady'] * len(arrival_times), index=arrival_times.index)

        # Calculate statistics
        mean_interval = inter_arrival.mean()
        std_interval = inter_arrival.std()

        # Define burst threshold (e.g., intervals < 0.5 * mean)
        burst_threshold = 0.5 * mean_interval if mean_interval > 0 else 0

        # Classify patterns
        burst_patterns = []
        for i, time in enumerate(arrival_times):
            if i == 0:
                burst_patterns.append('start')
            else:
                prev_time = arrival_times.iloc[i-1]
                interval = time - prev_time

                if interval <= burst_threshold:
                    burst_patterns.append('burst')
                elif interval > 2 * mean_interval:
                    burst_patterns.append('sparse')
                else:
                    burst_patterns.append('steady')

        return pd.Series(burst_patterns, index=arrival_times.index)

    def _generate_sequential_times(self, num_requests: int, base_interval: int = 100_000_000) -> pd.Series:
        """
        Generate sequential arrival times when timestamps are not available.

        Args:
            num_requests: Number of requests
            base_interval: Base interval in nanoseconds (default: 100ms)

        Returns:
            Series with sequential arrival times
        """
        times = []
        current_time = 0

        for i in range(num_requests):
            # Add some randomness to make it more realistic
            interval = base_interval + np.random.normal(0, base_interval * 0.1)
            current_time += max(interval, 1_000_000)  # Minimum 1ms interval
            times.append(int(current_time))

        return pd.Series(times)

    def get_dataset_info(self, file_path: str) -> Dict:
        """
        Get basic information about a dataset without loading it fully.

        Args:
            file_path: Path to dataset file

        Returns:
            Dictionary with dataset information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        _, ext = os.path.splitext(file_path.lower())

        # Load just the first few rows to get info
        if ext == '.tsv':
            sample_data = pd.read_csv(file_path, sep='\t', nrows=10)
        elif ext == '.csv':
            sample_data = pd.read_csv(file_path, nrows=10)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        info = {
            'file_path': file_path,
            'format': ext,
            'columns': list(sample_data.columns),
            'sample_size': len(sample_data),
            'estimated_total_rows': self._estimate_total_rows(file_path)
        }

        return info

    def _estimate_total_rows(self, file_path: str) -> int:
        """Estimate total number of rows in file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            # Count lines minus header
            return sum(1 for _ in f) - 1


def create_data_loader(verbose: bool = False) -> UniversalDataLoader:
    """Factory function to create data loader instance."""
    return UniversalDataLoader(verbose=verbose)