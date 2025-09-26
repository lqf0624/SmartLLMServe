#!/usr/bin/env python3
"""
Simple test script to generate visualization plots.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from predictor.workload_predictor import WorkloadPredictor, ModelType, PredictionHorizon

def create_test_data():
    """Create synthetic test data."""
    np.random.seed(42)

    # Generate timestamps
    base_time = 1_000_000_000
    timestamps = []
    current_time = base_time

    for i in range(100):
        if i % 20 < 5:  # Burst pattern
            interval = np.random.exponential(0.01)
        else:
            interval = np.random.exponential(0.1)

        current_time += int(interval * 1e9)
        timestamps.append(current_time)

    # Generate tokens
    input_tokens = np.random.poisson(100, 100)
    output_tokens = np.random.poisson(300, 100)

    # Create DataFrame
    data = pd.DataFrame({
        'input_toks': input_tokens,
        'output_toks': output_tokens,
        'arrival_time_ns': timestamps,
        'burst_pattern': ['burst' if i % 20 < 5 else 'steady' for i in range(100)],
        'model_type': ['ChatGPT'] * 100
    })

    return data

def main():
    """Generate visualizations."""
    print("Creating test data...")
    test_data = create_test_data()

    print("Initializing workload predictor...")
    predictor = WorkloadPredictor(
        model_type=ModelType.LSTM,
        prediction_horizon=PredictionHorizon.SHORT_TERM,
        sequence_length=20,
        feature_window=10,
        enable_ensemble=False,
        device='cpu'
    )

    print("Training predictor...")
    predictor.train(test_data, validation_split=0.2)

    print("Making predictions...")
    prediction_result = predictor.predict(test_data, steps=10)

    print("Generating comprehensive comparison visualization...")
    predictor.visualize_prediction(
        historical_data=test_data,
        save_dir="predictor/demo_visualization",
        show_plots=False
    )

    print("Visualization complete! Check predictor/demo_visualization/ directory")

    # List generated files
    if os.path.exists("predictor/demo_visualization"):
        files = os.listdir("predictor/demo_visualization")
        print(f"Generated files: {files}")

if __name__ == "__main__":
    main()