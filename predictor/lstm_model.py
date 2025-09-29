"""
LSTM-based time series prediction model for LLM workload forecasting.

This module implements a lightweight LSTM model that can run on both CPU and GPU,
with automatic device detection and fallback mechanisms.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import logging
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction of LLM workloads.

    Features:
    - Multi-horizon forecasting
    - Automatic CPU/GPU detection
    - Configurable architecture
    - Robust error handling
    """

    def __init__(self,
                 input_size: int = 5,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 3,
                 dropout: float = 0.2,
                 device: Optional[str] = None):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            output_size: Number of output predictions
            dropout: Dropout rate
            device: Force specific device ('cpu', 'cuda')
        """
        super(LSTMModel, self).__init__()

        # Device detection and validation
        self.device = self._setup_device(device)
        logger.info(f"Using device: {self.device}")

        # Model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        # Move model to device
        self.to(self.device)

    def _setup_device(self, device: Optional[str]) -> str:
        """
        Setup and validate device.

        Args:
            device: Requested device

        Returns:
            str: Actual device to use
        """
        if device is not None:
            if device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return 'cpu'
            return device

        # Auto-detect
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU")
            return 'cuda'
        else:
            logger.info("CUDA not available, using CPU")
            return 'cpu'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_size)

        Returns:
            torch.Tensor: Output predictions
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Take the last time step output
        out = out[:, -1, :]

        # Fully connected layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions in evaluation mode.

        Args:
            x: Input tensor

        Returns:
            np.ndarray: Predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(self.device)

            # Add batch dimension if needed
            if x.dim() == 2:
                x = x.unsqueeze(0)

            predictions = self.forward(x)
            return predictions.cpu().numpy()

    def save_model(self, path: str) -> None:
        """
        Save model state.

        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.lstm.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.fc2.out_features
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model state.

        Args:
            path: Path to load model from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise


class LSTMPredictor:
    """
    Complete LSTM prediction pipeline with training and inference.
    """

    def __init__(self,
                 sequence_length: int = 50,
                 prediction_horizon: int = 10,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 device: Optional[str] = None):
        """
        Initialize LSTM predictor.

        Args:
            sequence_length: Length of input sequence
            prediction_horizon: Number of steps to predict
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Training epochs
            device: Force specific device
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.epochs = epochs

        # Initialize model
        self.model = LSTMModel(
            input_size=5,  # arrival_rate, avg_tokens, burst_indicator, time_of_day, day_of_week
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=prediction_horizon,
            device=device
        )

        # Training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Data preprocessing
        self.scaler = StandardScaler()
        self.is_trained = False

        logger.info("LSTM Predictor initialized")

    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for training.

        Args:
            data: Input DataFrame with time series data

        Returns:
            Tuple of (sequences, targets)
        """
        # Extract features
        features = self._extract_features(data)

        # Normalize features
        features_normalized = self.scaler.fit_transform(features)

        # Create sequences
        sequences = []
        targets = []

        for i in range(len(features_normalized) - self.sequence_length - self.prediction_horizon + 1):
            seq = features_normalized[i:i + self.sequence_length]
            target = features_normalized[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, 0]  # Predict arrival rate
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features from raw data.

        Args:
            data: Input DataFrame

        Returns:
            np.ndarray: Feature matrix
        """
        features = []

        # Sort by arrival time
        data_sorted = data.sort_values('arrival_time_ns')

        # Calculate time differences
        time_diffs = data_sorted['arrival_time_ns'].diff().fillna(0)

        for i in range(len(data_sorted)):
            # Arrival rate (requests per second)
            window_size = min(10, i + 1)
            recent_times = data_sorted.iloc[max(0, i - window_size + 1):i + 1]['arrival_time_ns']
            if len(recent_times) > 1:
                time_diff_ns = recent_times.iloc[-1] - recent_times.iloc[0]
                if hasattr(time_diff_ns, 'value'):
                    time_diff_ns = time_diff_ns.value
                arrival_rate = len(recent_times) / (time_diff_ns / 1e9 + 1e-9)
            else:
                arrival_rate = 1.0

            # Average tokens
            avg_tokens = (data_sorted.iloc[i]['input_toks'] + data_sorted.iloc[i]['output_toks']) / 2

            # Burst indicator (from existing pattern detection)
            burst_map = {'burst': 1.0, 'steady': 0.5, 'sparse': 0.0, 'start': 0.5}
            burst_indicator = burst_map.get(data_sorted.iloc[i].get('burst_pattern', 'steady'), 0.5)

            # Time of day (normalized)
            timestamp = pd.to_datetime(data_sorted.iloc[i]['arrival_time_ns'])
            timestamp_ns = timestamp.value if hasattr(timestamp, 'value') else timestamp.timestamp() * 1e9
            time_of_day = (timestamp_ns % (24 * 3600 * 1e9)) / (24 * 3600 * 1e9)

            # Day of week (normalized)
            day_of_week = ((timestamp_ns // (24 * 3600 * 1e9)) % 7) / 7

            features.append([arrival_rate, avg_tokens, burst_indicator, time_of_day, day_of_week])

        return np.array(features)

    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the LSTM model.

        Args:
            data: Training data
            validation_split: Validation split ratio

        Returns:
            Dict with training history
        """
        logger.info("Starting LSTM training...")

        # Prepare sequences
        sequences, targets = self.prepare_sequences(data)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, targets, test_size=validation_split, random_state=42
        )

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.model.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.model.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.model.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.model.device)

        # Training history
        history = {'train_loss': [], 'val_loss': []}

        # Training loop
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for i in range(0, len(X_train_tensor), self.batch_size):
                batch_X = X_train_tensor[i:i + self.batch_size]
                batch_y = y_train_tensor[i:i + self.batch_size]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()

            # Record history
            avg_train_loss = train_loss / (len(X_train_tensor) // self.batch_size + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)

            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        self.is_trained = True
        logger.info("LSTM training completed")

        return history

    def predict_future(self, data: pd.DataFrame, steps: int = None) -> Dict[str, Any]:
        """
        Predict future values.

        Args:
            data: Historical data
            steps: Number of steps to predict

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if steps is None:
            steps = self.prediction_horizon

        # Prepare last sequence
        features = self._extract_features(data)
        features_normalized = self.scaler.transform(features)

        # Take last sequence
        if len(features_normalized) >= self.sequence_length:
            last_sequence = features_normalized[-self.sequence_length:]
        else:
            # Pad if not enough data
            pad_length = self.sequence_length - len(features_normalized)
            last_sequence = np.pad(features_normalized, ((pad_length, 0), (0, 0)), mode='constant')

        # Make prediction
        sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.model.device)
        predictions = self.model.predict(sequence_tensor)

        # Inverse transform
        predictions_flat = predictions.flatten()
        if len(predictions_flat) > 0:
            # Create a full feature array for inverse transformation
            predictions_full = np.zeros((len(predictions_flat), 5))
            predictions_full[:, 0] = predictions_flat
            predictions_denormalized = self.scaler.inverse_transform(predictions_full)[:, 0]
        else:
            predictions_denormalized = np.array([])

        return {
            'predictions': predictions_denormalized[:steps],
            'confidence': 0.85,  # Default confidence for LSTM
            'model_type': 'LSTM',
            'prediction_steps': steps,
            'sequence_length': self.sequence_length
        }

    def save(self, path: str) -> None:
        """
        Save the complete predictor.

        Args:
            path: Path to save predictor
        """
        save_dict = {
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'is_trained': self.is_trained
        }
        torch.save(save_dict, path)
        logger.info(f"Predictor saved to {path}")

    def load(self, path: str) -> None:
        """
        Load the complete predictor with backward compatibility.

        Args:
            path: Path to load predictor from
        """
        try:
            save_dict = torch.load(path, map_location=self.model.device, weights_only=False)

            # Check if we need to handle model architecture compatibility
            try:
                self.model.load_state_dict(save_dict['model_state'])
            except RuntimeError as e:
                if "Missing key(s) in state_dict" in str(e) or "size mismatch" in str(e):
                    logger.warning(f"Model architecture mismatch detected. Creating compatible model...")
                    # Try to create a compatible model based on saved state
                    self._create_compatible_model(save_dict['model_state'])
                    self.model.load_state_dict(save_dict['model_state'])
                else:
                    raise

            self.scaler = save_dict['scaler']
            self.sequence_length = save_dict['sequence_length']
            self.prediction_horizon = save_dict['prediction_horizon']
            self.is_trained = save_dict['is_trained']
            logger.info(f"Predictor loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load predictor from {path}: {e}")
            raise

    def _create_compatible_model(self, state_dict: dict) -> None:
        """
        Create a model compatible with the saved state dict.

        Args:
            state_dict: Model state dict to inspect
        """
        # Inspect the state dict to determine the original architecture
        lstm_keys = [k for k in state_dict.keys() if k.startswith('lstm.')]

        # Extract architecture info from state dict
        if 'lstm.weight_ih_l0' in state_dict:
            input_size = state_dict['lstm.weight_ih_l0'].shape[1]
            hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4  # LSTM has 4 matrices per layer

            # Check number of layers
            layer_keys = [k for k in lstm_keys if 'weight_ih_l' in k]
            num_layers = len(layer_keys)

            # Determine output size from fc2 layer
            if 'fc2.weight' in state_dict:
                output_size = state_dict['fc2.weight'].shape[0]
            else:
                output_size = 3  # Default

            logger.info(f"Detected original architecture: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, output_size={output_size}")

            # Recreate model with compatible architecture
            compatible_model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                device=self.model.device
            )

            # Replace current model
            self.model = compatible_model