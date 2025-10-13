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

        # Check if this is minute-level aggregated data
        if 'Request_tokens_sum' in data.columns and 'Response_tokens_sum' in data.columns:
            return self._normalize_minute_level_data(data)

        # Check for original BurstGPT format
        if 'Request tokens' in data.columns and 'Response tokens' in data.columns:
            return self._normalize_original_burstgpt_data(data)

        raise ValueError("Unsupported CSV format. Expected columns for BurstGPT format.")

    def _normalize_minute_level_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize minute-level aggregated BurstGPT data using realistic token distribution."""
        normalized = pd.DataFrame()

        # Performance model limits
        MAX_INPUT_TOKENS = 2048
        MAX_OUTPUT_TOKENS = 2048

        # Generate requests from minute-level data
        requests = []
        for idx, row in data.iterrows():
            # Skip rows with no activity
            if row['Concurrent_requests'] == 0 or row['Request_tokens_sum'] == 0:
                continue

            # Extract minute-level statistics
            minute_concurrent = int(row['Concurrent_requests'])  # Total number of requests in this minute
            total_input_tokens = int(row['Request_tokens_sum'])  # SUM of input tokens for all requests
            total_output_tokens = int(row['Response_tokens_sum'])  # SUM of output tokens for all requests

            # Calculate base average tokens per request
            base_avg_input = total_input_tokens / minute_concurrent
            base_avg_output = total_output_tokens / minute_concurrent

            print(f"   处理分钟级数据: {minute_concurrent} 个请求")
            print(f"   基础平均值: input={base_avg_input:.1f}, output={base_avg_output:.1f}")

            # Generate realistic token distribution based on BurstGPT analysis
            input_tokens_list = self._generate_realistic_token_distribution(
                minute_concurrent, total_input_tokens, base_avg_input)
            output_tokens_list = self._generate_realistic_token_distribution(
                minute_concurrent, total_output_tokens, base_avg_output)

            # Generate arrival times within the minute
            # Convert minute timestamp to nanoseconds
            base_timestamp_ns = int(row['Timestamp'] * 60 * 1e9)  # minutes to nanoseconds
            minute_duration_ns = 60 * 1e9  # 60 seconds in nanoseconds

            # Use uniform distribution for arrival times across the minute
            arrival_positions = np.random.uniform(0, 1, minute_concurrent)
            sorted_indices = np.argsort(arrival_positions)

            # Generate requests
            minute_requests = 0
            for i in range(minute_concurrent):
                request_idx = sorted_indices[i]
                arrival_time_ns = base_timestamp_ns + int(arrival_positions[request_idx] * minute_duration_ns)

                # Apply performance model limits
                input_tokens = min(input_tokens_list[i], MAX_INPUT_TOKENS)
                output_tokens = min(output_tokens_list[i], MAX_OUTPUT_TOKENS)

                # Allow 0 tokens (as seen in real BurstGPT data)
                input_tokens = max(0, int(input_tokens))
                output_tokens = max(0, int(output_tokens))

                # Only add request if it has at least some tokens
                if input_tokens > 0 or output_tokens > 0:
                    requests.append({
                        'input_toks': input_tokens,
                        'output_toks': output_tokens,
                        'arrival_time_ns': arrival_time_ns,
                        'model_type': 'meta-llama/Llama-3.1-8B-Instruct'
                    })
                    minute_requests += 1

            # Verify token totals (approximately, allowing for performance model limits)
            if minute_requests > 0:
                actual_input_total = sum(req['input_toks'] for req in requests[-minute_requests:])
                actual_output_total = sum(req['output_toks'] for req in requests[-minute_requests:])

                print(f"   生成请求: {minute_requests} 个 (过滤了0-token请求)")
                print(f"   Token验证: 原始input={total_input_tokens}, 实际input={actual_input_total}")
                print(f"   Token验证: 原始output={total_output_tokens}, 实际output={actual_output_total}")

        if not requests:
            # If no requests found, generate minimal requests
            return self._generate_minimal_requests(len(data))

        normalized = pd.DataFrame(requests)

        # Sort by arrival time
        normalized = normalized.sort_values('arrival_time_ns').reset_index(drop=True)

        # Detect burst patterns
        normalized['burst_pattern'] = self._detect_burst_pattern(normalized['arrival_time_ns'])

        return normalized

    def _generate_realistic_token_distribution(self, num_requests: int, total_tokens: int, base_avg: float) -> List[float]:
        """
        Generate realistic token distribution based on BurstGPT analysis.

        Key insights from BurstGPT data:
        - High concurrency → lower average tokens per request
        - Many 0-token requests in high-concurrency minutes
        - Token distribution follows log-normal pattern for non-zero requests
        """
        if num_requests == 0:
            return []

        tokens = []
        remaining_tokens = total_tokens

        # Determine strategy based on concurrency level and average tokens
        if base_avg < 50:
            # Very high concurrency scenario (like 1250 requests/min with avg=62 tokens)
            # Many 0-token requests, few with normal tokens
            zero_token_ratio = 0.4  # 40% zero-token requests

            for i in range(num_requests):
                if i == num_requests - 1:
                    # Last request gets remaining tokens
                    tokens.append(max(0, remaining_tokens))
                elif np.random.random() < zero_token_ratio and remaining_tokens > 0:
                    # Zero-token request
                    tokens.append(0)
                else:
                    # Small token request (log-normal distribution)
                    generated = np.random.lognormal(3.5, 1.2)  # exp(3.5) ≈ 33 tokens avg
                    generated = min(generated, 500)  # Cap at 500 for high concurrency
                    allocated = min(generated, remaining_tokens // max(1, num_requests - i - 1))
                    tokens.append(max(0, allocated))
                    remaining_tokens -= allocated

        elif base_avg < 200:
            # Medium-high concurrency (like 111 requests/min with avg=629 tokens)
            # Fewer zero-token requests, moderate token sizes
            zero_token_ratio = 0.1  # 10% zero-token requests

            for i in range(num_requests):
                if i == num_requests - 1:
                    tokens.append(max(0, remaining_tokens))
                elif np.random.random() < zero_token_ratio and remaining_tokens > 10:
                    tokens.append(0)
                else:
                    # Medium token request
                    generated = np.random.lognormal(5.0, 1.4)  # exp(5.0) ≈ 148 tokens avg
                    generated = min(generated, 1500)
                    allocated = min(generated, remaining_tokens // max(1, num_requests - i - 1))
                    tokens.append(max(0, allocated))
                    remaining_tokens -= allocated

        elif base_avg < 600:
            # Medium concurrency (like 4 requests/min with avg=935 tokens)
            # Almost no zero-token requests, larger token sizes
            zero_token_ratio = 0.02  # 2% zero-token requests

            for i in range(num_requests):
                if i == num_requests - 1:
                    tokens.append(max(0, remaining_tokens))
                elif np.random.random() < zero_token_ratio and remaining_tokens > 50:
                    tokens.append(0)
                else:
                    # Larger token request
                    generated = np.random.lognormal(6.0, 1.3)  # exp(6.0) ≈ 403 tokens avg
                    generated = min(generated, 2500)
                    allocated = min(generated, remaining_tokens // max(1, num_requests - i - 1))
                    tokens.append(max(0, allocated))
                    remaining_tokens -= allocated

        else:
            # Low concurrency (like 2-4 requests/min with avg=450-935 tokens)
            # Very few zero-token requests, large token sizes
            zero_token_ratio = 0.01  # 1% zero-token requests

            for i in range(num_requests):
                if i == num_requests - 1:
                    tokens.append(max(0, remaining_tokens))
                elif np.random.random() < zero_token_ratio and remaining_tokens > 100:
                    tokens.append(0)
                else:
                    # Large token request
                    generated = np.random.lognormal(6.5, 1.2)  # exp(6.5) ≈ 665 tokens avg
                    generated = min(generated, 3500)
                    allocated = min(generated, remaining_tokens // max(1, num_requests - i - 1))
                    tokens.append(max(0, allocated))
                    remaining_tokens -= allocated

        # Ensure we exactly match the total by adjusting the last few requests
        token_sum = sum(tokens)
        if token_sum != total_tokens and len(tokens) > 1:
            diff = total_tokens - token_sum
            i = len(tokens) - 1
            while diff != 0 and i >= 0:
                adjustment = 1 if diff > 0 else -1
                new_val = max(0, tokens[i] + adjustment)
                actual_adjustment = new_val - tokens[i]
                tokens[i] = new_val
                diff -= actual_adjustment
                i -= 1

        return tokens

    def _normalize_original_burstgpt_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize original BurstGPT data to standard format."""
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

    def _generate_minimal_requests(self, num_requests: int) -> pd.DataFrame:
        """Generate minimal requests when no activity data is available."""
        requests = []
        base_interval_ns = 60_000_000_000  # 1 minute in nanoseconds

        for i in range(min(num_requests, 10)):  # Generate at most 10 requests
            requests.append({
                'input_toks': 128,  # Default input length
                'output_toks': 256,  # Default output length
                'arrival_time_ns': i * base_interval_ns,
                'model_type': 'meta-llama/Llama-3.1-8B-Instruct'
            })

        normalized = pd.DataFrame(requests)
        normalized['burst_pattern'] = 'steady'

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