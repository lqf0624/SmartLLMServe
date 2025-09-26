"""
Feature extraction module for LLM workload prediction.

This module provides comprehensive feature extraction from raw request data
to support time series prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features that can be extracted."""
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"
    WORKLOAD = "workload"
    CONTEXTUAL = "contextual"


class FeatureExtractor:
    """
    Comprehensive feature extractor for LLM workload prediction.

    Extracts multiple types of features from raw request data:
    - Temporal features: time-based patterns
    - Statistical features: request characteristics
    - Workload features: system-level metrics
    - Contextual features: burst patterns and model types
    """

    def __init__(self,
                 window_size: int = 10,
                 feature_types: Optional[List[FeatureType]] = None,
                 normalize: bool = True):
        """
        Initialize feature extractor.

        Args:
            window_size: Window size for rolling statistics
            feature_types: Types of features to extract
            normalize: Whether to normalize features
        """
        self.window_size = window_size
        self.feature_types = feature_types or list(FeatureType)
        self.normalize = normalize

        # Scalers for different feature groups
        self.scalers = {
            'temporal': StandardScaler(),
            'statistical': StandardScaler(),
            'workload': StandardScaler(),
            'contextual': MinMaxScaler()
        }

        # Feature names for different groups
        self.feature_names = {
            FeatureType.TEMPORAL: ['arrival_rate', 'time_of_day', 'day_of_week', 'is_weekend'],
            FeatureType.STATISTICAL: ['avg_input_tokens', 'avg_output_tokens', 'token_ratio', 'complexity_score'],
            FeatureType.WORKLOAD: ['request_density', 'burst_intensity', 'queue_pressure', 'resource_demand'],
            FeatureType.CONTEXTUAL: ['burst_pattern_numeric', 'model_type_numeric', 'request_type_numeric']
        }

        # Burst pattern mapping
        self.burst_mapping = {'burst': 1.0, 'steady': 0.5, 'sparse': 0.0, 'start': 0.5}

        # Model type mapping (can be extended)
        self.model_mapping = {
            'ChatGPT': 0.0,
            'GPT-4': 0.2,
            'LLaMA': 0.4,
            'GPT-3.5': 0.6,
            'Claude': 0.8,
            'Unknown': 1.0
        }

        logger.info(f"FeatureExtractor initialized with window_size={window_size}")

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract all features from raw data.

        Args:
            data: Input DataFrame with request data

        Returns:
            np.ndarray: Extracted features
        """
        if data.empty:
            logger.warning("Empty DataFrame provided")
            return np.array([])

        # Sort by arrival time
        data_sorted = data.sort_values('arrival_time_ns').copy()

        all_features = []

        # Extract features by type
        if FeatureType.TEMPORAL in self.feature_types:
            temporal_features = self._extract_temporal_features(data_sorted)
            all_features.append(temporal_features)

        if FeatureType.STATISTICAL in self.feature_types:
            statistical_features = self._extract_statistical_features(data_sorted)
            all_features.append(statistical_features)

        if FeatureType.WORKLOAD in self.feature_types:
            workload_features = self._extract_workload_features(data_sorted)
            all_features.append(workload_features)

        if FeatureType.CONTEXTUAL in self.feature_types:
            contextual_features = self._extract_contextual_features(data_sorted)
            all_features.append(contextual_features)

        # Combine all features
        if all_features:
            combined_features = np.concatenate(all_features, axis=1)

            # Normalize if requested
            if self.normalize:
                combined_features = self._normalize_features(combined_features)

            return combined_features
        else:
            return np.array([])

    def _extract_temporal_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract temporal features."""
        features = []

        # Calculate time differences
        time_diffs = data['arrival_time_ns'].diff().fillna(0)

        for i in range(len(data)):
            row_features = []

            # Arrival rate (requests per second)
            window_start = max(0, i - self.window_size + 1)
            recent_times = data.iloc[window_start:i + 1]['arrival_time_ns']

            if len(recent_times) > 1:
                time_span = (recent_times.iloc[-1] - recent_times.iloc[0]) / 1e9
                arrival_rate = len(recent_times) / max(time_span, 1e-9)
            else:
                arrival_rate = 1.0

            row_features.append(arrival_rate)

            # Time of day (normalized 0-1)
            timestamp = data.iloc[i]['arrival_time_ns']
            time_of_day = (timestamp % (24 * 3600 * 1e9)) / (24 * 3600 * 1e9)
            row_features.append(time_of_day)

            # Day of week (normalized 0-1)
            day_of_week = ((timestamp // (24 * 3600 * 1e9)) % 7) / 7
            row_features.append(day_of_week)

            # Weekend indicator
            is_weekend = 1.0 if day_of_week >= 5/7 else 0.0
            row_features.append(is_weekend)

            features.append(row_features)

        return np.array(features)

    def _extract_statistical_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract statistical features."""
        features = []

        for i in range(len(data)):
            row_features = []

            # Current request tokens
            input_tokens = data.iloc[i]['input_toks']
            output_tokens = data.iloc[i]['output_toks']

            # Token statistics
            avg_tokens = (input_tokens + output_tokens) / 2
            token_ratio = output_tokens / max(input_tokens, 1)

            row_features.extend([avg_tokens, token_ratio])

            # Rolling statistics
            window_start = max(0, i - self.window_size + 1)
            window_data = data.iloc[window_start:i + 1]

            # Average input tokens in window
            avg_input_tokens = window_data['input_toks'].mean()
            row_features.append(avg_input_tokens)

            # Average output tokens in window
            avg_output_tokens = window_data['output_toks'].mean()
            row_features.append(avg_output_tokens)

            # Complexity score (based on token variance)
            if len(window_data) > 1:
                complexity_score = window_data['input_toks'].var() + window_data['output_toks'].var()
            else:
                complexity_score = 0.0

            row_features.append(complexity_score)

            features.append(row_features)

        return np.array(features)

    def _extract_workload_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract workload-related features."""
        features = []

        for i in range(len(data)):
            row_features = []

            # Request density in recent window
            window_start = max(0, i - self.window_size + 1)
            window_data = data.iloc[window_start:i + 1]

            # Calculate time span for window
            if len(window_data) > 1:
                time_span = (window_data['arrival_time_ns'].iloc[-1] -
                           window_data['arrival_time_ns'].iloc[0]) / 1e9
                request_density = len(window_data) / max(time_span, 1e-9)
            else:
                request_density = 1.0

            row_features.append(request_density)

            # Burst intensity (inter-arrival time variance)
            if len(window_data) > 1:
                time_diffs = window_data['arrival_time_ns'].diff().dropna()
                burst_intensity = time_diffs.var() / 1e18  # Normalize by 1e18
            else:
                burst_intensity = 0.0

            row_features.append(burst_intensity)

            # Queue pressure (estimated based on recent request rate)
            current_time = data.iloc[i]['arrival_time_ns']
            future_requests = data[data['arrival_time_ns'] > current_time].head(5)
            queue_pressure = len(future_requests) / max(len(future_requests), 1)
            row_features.append(queue_pressure)

            # Resource demand (estimated based on token requirements)
            total_tokens = data.iloc[i]['input_toks'] + data.iloc[i]['output_toks']
            resource_demand = total_tokens / 1000.0  # Normalize by 1000 tokens
            row_features.append(resource_demand)

            features.append(row_features)

        return np.array(features)

    def _extract_contextual_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract contextual features."""
        features = []

        for i in range(len(data)):
            row_features = []

            # Burst pattern
            burst_pattern = data.iloc[i].get('burst_pattern', 'steady')
            burst_numeric = self.burst_mapping.get(burst_pattern, 0.5)
            row_features.append(burst_numeric)

            # Model type
            model_type = data.iloc[i].get('model_type', 'Unknown')
            model_numeric = self.model_mapping.get(model_type, 1.0)
            row_features.append(model_numeric)

            # Request type (based on token characteristics)
            input_tokens = data.iloc[i]['input_toks']
            output_tokens = data.iloc[i]['output_toks']

            if input_tokens < 100 and output_tokens < 100:
                request_type = 0.0  # Short
            elif input_tokens < 500 and output_tokens < 500:
                request_type = 0.5  # Medium
            else:
                request_type = 1.0  # Long

            row_features.append(request_type)

            features.append(row_features)

        return np.array(features)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features by type."""
        if features.size == 0:
            return features

        normalized_features = []
        start_idx = 0

        for feature_type in self.feature_types:
            feature_names = self.feature_names[feature_type]
            num_features = len(feature_names)

            if start_idx + num_features <= features.shape[1]:
                feature_group = features[:, start_idx:start_idx + num_features]

                # Fit and transform with appropriate scaler
                scaler_key = feature_type.value
                if scaler_key in self.scalers:
                    if hasattr(self.scalers[scaler_key], 'mean_'):
                        # Scaler is already fitted
                        normalized_group = self.scalers[scaler_key].transform(feature_group)
                    else:
                        # Fit scaler
                        normalized_group = self.scalers[scaler_key].fit_transform(feature_group)
                else:
                    normalized_group = feature_group

                normalized_features.append(normalized_group)
                start_idx += num_features

        if normalized_features:
            return np.concatenate(normalized_features, axis=1)
        else:
            return features

    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        all_names = []
        for feature_type in self.feature_types:
            all_names.extend(self.feature_names[feature_type])
        return all_names

    def get_feature_importance(self, model: Any) -> Dict[str, float]:
        """
        Get feature importance from a trained model.

        Args:
            model: Trained model with feature_importances_ attribute

        Returns:
            Dict mapping feature names to importance scores
        """
        if hasattr(model, 'feature_importances_'):
            feature_names = self.get_feature_names()
            importances = model.feature_importances_

            return dict(zip(feature_names, importances))
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return {}

    def create_sequences(self, features: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.

        Args:
            features: Extracted features
            sequence_length: Length of each sequence

        Returns:
            Tuple of (sequences, targets)
        """
        if len(features) < sequence_length + 1:
            logger.warning(f"Not enough data for sequences. Need {sequence_length + 1}, got {len(features)}")
            return np.array([]), np.array([])

        sequences = []
        targets = []

        for i in range(len(features) - sequence_length):
            seq = features[i:i + sequence_length]
            # Target is the next time step's arrival rate (first feature)
            target = features[i + sequence_length, 0]

            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def save_scalers(self, path: str) -> None:
        """Save fitted scalers."""
        import joblib
        scalers_data = {
            'scalers': self.scalers,
            'feature_types': [ft.value for ft in self.feature_types],
            'window_size': self.window_size
        }
        joblib.dump(scalers_data, path)
        logger.info(f"Scalers saved to {path}")

    def load_scalers(self, path: str) -> None:
        """Load fitted scalers."""
        import joblib
        try:
            scalers_data = joblib.load(path)
            self.scalers = scalers_data['scalers']
            self.feature_types = [FeatureType(ft) for ft in scalers_data['feature_types']]
            self.window_size = scalers_data['window_size']
            logger.info(f"Scalers loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load scalers from {path}: {e}")
            raise