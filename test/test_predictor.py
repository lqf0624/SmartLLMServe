#!/usr/bin/env python3
"""
Test script for workload predictor functionality.

Tests the complete prediction pipeline including:
- Feature extraction
- LSTM model training and prediction
- Workload predictor interface
- Model saving and loading
"""

import os
import sys
import numpy as np
import pandas as pd
import logging

# Add parent directories to path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from predictor.workload_predictor import WorkloadPredictor, ModelType, PredictionHorizon, PredictionResult
from predictor.feature_extractor import FeatureExtractor, FeatureType
from predictor.lstm_model import LSTMPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_feature_extractor():
    """Test feature extraction functionality."""
    print("=== Testing Feature Extractor ===")

    try:
        # Create synthetic test data
        test_data = create_synthetic_data(50)

        # Test basic feature extraction
        extractor = FeatureExtractor(window_size=5, normalize=True)
        features = extractor.extract_features(test_data)

        print(f"[SUCCESS] Feature extraction completed")
        print(f"   Input data shape: {test_data.shape}")
        print(f"   Extracted features shape: {features.shape}")
        print(f"   Feature names: {extractor.get_feature_names()}")

        # Test sequence creation
        sequences, targets = extractor.create_sequences(features, sequence_length=10)
        print(f"   Sequences shape: {sequences.shape}")
        print(f"   Targets shape: {targets.shape}")

        # Test different feature types
        temporal_extractor = FeatureExtractor(
            window_size=5,
            feature_types=[FeatureType.TEMPORAL],
            normalize=True
        )
        temporal_features = temporal_extractor.extract_features(test_data)
        print(f"   Temporal features only: {temporal_features.shape}")

        return True

    except Exception as e:
        print(f"[FAILED] Feature extraction test failed: {e}")
        return False


def test_lstm_predictor():
    """Test LSTM predictor functionality."""
    print("\n=== Testing LSTM Predictor ===")

    try:
        # Create test data
        test_data = create_synthetic_data(100)

        # Initialize LSTM predictor
        predictor = LSTMPredictor(
            sequence_length=20,
            prediction_horizon=5,
            hidden_size=32,
            num_layers=1,
            learning_rate=0.01,
            batch_size=16,
            epochs=2,  # Reduced for testing
            device='cpu'
        )

        print(f"[SUCCESS] LSTM predictor initialized")

        # Test training
        print("[INFO] Starting training (reduced epochs for testing)...")
        history = predictor.train(test_data, validation_split=0.2)

        print(f"[SUCCESS] Training completed")
        print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"   Final val loss: {history['val_loss'][-1]:.4f}")

        # Test prediction
        predictions = predictor.predict_future(test_data, steps=5)
        print(f"[SUCCESS] Prediction completed")
        print(f"   Predictions: {predictions['predictions']}")
        print(f"   Confidence: {predictions['confidence']}")
        print(f"   Model type: {predictions['model_type']}")

        # Test model saving/loading
        test_model_path = "../predictor/test_lstm_model.pt"
        predictor.save(test_model_path)
        print(f"[SUCCESS] Model saved to {test_model_path}")

        # Create new predictor and load
        new_predictor = LSTMPredictor(
            sequence_length=20,
            prediction_horizon=5,
            device='cpu'
        )
        new_predictor.load(test_model_path)
        print(f"[SUCCESS] Model loaded successfully")

        # Test prediction with loaded model
        loaded_predictions = new_predictor.predict_future(test_data, steps=5)
        print(f"[SUCCESS] Prediction with loaded model: {loaded_predictions['predictions']}")

        # Clean up
        if os.path.exists(test_model_path):
            os.remove(test_model_path)

        return True

    except Exception as e:
        print(f"[FAILED] LSTM predictor test failed: {e}")
        return False


def test_workload_predictor():
    """Test workload predictor interface."""
    print("\n=== Testing Workload Predictor Interface ===")

    try:
        # Create test data
        test_data = create_synthetic_data(150)

        # Test LSTM workload predictor
        workload_pred = WorkloadPredictor(
            model_type=ModelType.LSTM,
            prediction_horizon=PredictionHorizon.SHORT_TERM,
            sequence_length=30,
            feature_window=10,
            enable_ensemble=False,
            device='cpu'
        )

        print(f"[SUCCESS] Workload predictor initialized")
        print(f"   Model info: {workload_pred.get_model_info()}")

        # Test data validation
        is_valid = workload_pred.validate_data(test_data)
        print(f"[SUCCESS] Data validation: {is_valid}")

        # Test training
        print("[INFO] Starting workload predictor training...")
        results = workload_pred.train(test_data, validation_split=0.2)

        print(f"[SUCCESS] Training completed")
        for model_name, result in results.items():
            print(f"   {model_name}: {result['status']}")
            if result['status'] == 'success':
                print(f"     Final train loss: {result.get('final_train_loss', 'N/A')}")
                print(f"     Final val loss: {result.get('final_val_loss', 'N/A')}")

        # Test prediction
        prediction_result = workload_pred.predict(test_data, steps=10)
        print(f"[SUCCESS] Prediction completed")
        print(f"   Prediction type: {type(prediction_result)}")
        print(f"   Confidence: {prediction_result.confidence:.4f}")
        print(f"   Predicted requests: {len(prediction_result.predicted_requests)}")

        # Test prediction result methods
        resource_req = prediction_result.get_resource_requirements()
        timing_info = prediction_result.get_timing_info()
        burst_analysis = prediction_result.get_burst_analysis()

        print(f"   Resource requirements: {resource_req}")
        print(f"   Timing info: {timing_info}")
        print(f"   Burst analysis: {burst_analysis}")

        # Show sample predicted request
        if prediction_result.predicted_requests:
            sample_req = prediction_result.predicted_requests[0]
            print(f"   Sample request: {sample_req}")

        # Test visualization (without displaying plots to avoid blocking)
        print("[INFO] Testing prediction visualization...")
        try:
            workload_pred.visualize_prediction(
                historical_data=test_data,
                save_dir="../predictor/test_visualization",
                show_plots=False  # Don't show plots during testing
            )
            print("[SUCCESS] Visualization completed")
        except Exception as e:
            print(f"[INFO] Visualization test failed (expected if no display): {e}")

        # Test prediction report generation
        print("[INFO] Testing prediction report generation...")
        try:
            report = workload_pred.generate_prediction_report(prediction_result, test_data)
            print("[SUCCESS] Prediction report generated")
            print(f"   Report sections: {list(report.keys())}")
            print(f"   Scheduling recommendations: {len(report['scheduling_recommendations'])}")
            print(f"   Confidence level: {report['confidence_assessment']['confidence_level']}")
            print(f"   Burst risk score: {report['burst_analysis']['burst_risk_score']:.2f}")
        except Exception as e:
            print(f"[FAILED] Report generation failed: {e}")

        # Test model saving/loading
        test_save_path = "../predictor/test_workload_predictor.pkl"
        workload_pred.save(test_save_path)
        print(f"[SUCCESS] Workload predictor saved to {test_save_path}")

        # Create new predictor and load
        new_workload_pred = WorkloadPredictor(
            model_type=ModelType.LSTM,
            device='cpu'
        )
        new_workload_pred.load(test_save_path)
        print(f"[SUCCESS] Workload predictor loaded successfully")

        # Test prediction with loaded model
        loaded_prediction = new_workload_pred.predict(test_data, steps=10)
        print(f"[SUCCESS] Prediction with loaded model: {len(loaded_prediction.predicted_requests)} requests")

        # Test prediction summary
        summary = new_workload_pred.get_prediction_summary()
        print(f"[SUCCESS] Prediction summary: {summary}")

        # Clean up
        if os.path.exists(test_save_path):
            os.remove(test_save_path)

        # Remove any saved model files
        for model_file in ["test_workload_predictor_lstm.pt"]:
            model_path = os.path.join(original_dir, "predictor", model_file)
            if os.path.exists(model_path):
                os.remove(model_path)

        # Clean up visualization directory
        import shutil
        viz_dir = os.path.join(original_dir, "predictor", "test_visualization")
        if os.path.exists(viz_dir):
            shutil.rmtree(viz_dir)

        return True

    except Exception as e:
        print(f"[FAILED] Workload predictor test failed: {e}")
        return False


def test_ensemble_predictor():
    """Test ensemble prediction functionality."""
    print("\n=== Testing Ensemble Predictor ===")

    try:
        # Create test data
        test_data = create_synthetic_data(100)

        # Test ensemble predictor (with only LSTM for now)
        ensemble_pred = WorkloadPredictor(
            model_type=ModelType.ENSEMBLE,
            prediction_horizon=PredictionHorizon.MEDIUM_TERM,
            sequence_length=25,
            feature_window=8,
            enable_ensemble=True,
            device='cpu'
        )

        print(f"[SUCCESS] Ensemble predictor initialized")

        # Test training
        print("[INFO] Starting ensemble training...")
        results = ensemble_pred.train(test_data, validation_split=0.2)

        print(f"[SUCCESS] Ensemble training completed")
        for model_name, result in results.items():
            print(f"   {model_name}: {result['status']}")

        # Test prediction
        prediction_result = ensemble_pred.predict(test_data, steps=15)
        print(f"[SUCCESS] Ensemble prediction completed")
        print(f"   Predicted requests: {len(prediction_result.predicted_requests)}")
        print(f"   Confidence: {prediction_result.confidence:.4f}")

        return True

    except Exception as e:
        print(f"[FAILED] Ensemble predictor test failed: {e}")
        return False


def create_synthetic_data(num_requests: int) -> pd.DataFrame:
    """Create synthetic test data."""
    np.random.seed(42)  # For reproducible results

    # Generate timestamps with some patterns
    base_time = 1_000_000_000  # Start at 1 second
    timestamps = []
    current_time = base_time

    for i in range(num_requests):
        # Create some burst patterns
        if i % 20 < 5:  # Burst every 20 requests
            interval = np.random.exponential(0.01)  # 10ms average
        else:
            interval = np.random.exponential(0.1)  # 100ms average

        current_time += int(interval * 1e9)  # Convert to nanoseconds
        timestamps.append(current_time)

    # Generate token counts
    input_tokens = np.random.poisson(100, num_requests)
    output_tokens = np.random.poisson(300, num_requests)

    # Create burst patterns
    burst_patterns = []
    for i in range(num_requests):
        if i % 20 < 5:
            burst_patterns.append('burst')
        elif i % 20 < 15:
            burst_patterns.append('steady')
        else:
            burst_patterns.append('sparse')

    # Create DataFrame
    data = pd.DataFrame({
        'input_toks': input_tokens,
        'output_toks': output_tokens,
        'arrival_time_ns': timestamps,
        'burst_pattern': burst_patterns,
        'model_type': ['ChatGPT'] * num_requests
    })

    return data


def main():
    """Run all predictor tests."""
    print("Testing Workload Predictor")
    print("=" * 50)

    # Change to astra-sim directory for relative path handling
    original_dir = os.getcwd()
    astra_dir = os.path.join(original_dir, "astra-sim")

    if os.path.exists(astra_dir):
        os.chdir(astra_dir)
        print(f"Changed to directory: {os.getcwd()}")
    else:
        print(f"Warning: astra-sim directory not found, using current directory")

    tests = [
        ("Feature Extractor", test_feature_extractor),
        ("LSTM Predictor", test_lstm_predictor),
        ("Workload Predictor", test_workload_predictor),
        ("Ensemble Predictor", test_ensemble_predictor)
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"[CRASHED] Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")

    # Return to original directory
    os.chdir(original_dir)

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)