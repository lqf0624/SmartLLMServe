#!/usr/bin/env python3
"""
Test script for universal data loader functionality.
Tests both original TSV format and BurstGPT CSV format.
"""

import os
import sys
sys.path.append('../inference_serving')

from data_loader import create_data_loader


def test_tsv_format():
    """Test loading original TSV format - DEPRECATED: Use BurstGPT instead."""
    print("=== Testing TSV Format (DEPRECATED) ===")
    print("[INFO] TSV format testing removed - using BurstGPT dataset for all tests")
    return True


def test_burstgpt_format():
    """Test loading BurstGPT CSV format."""
    print("\n=== Testing BurstGPT CSV Format ===")

    loader = create_data_loader(verbose=True)

    # Test with BurstGPT file
    csv_file = "../dataset/BurstGPT_1.csv"

    if os.path.exists(csv_file):
        try:
            data = loader.load_dataset(csv_file, req_num=5)
            print(f"[SUCCESS] BurstGPT CSV loading successful")
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Sample data:\n{data.head()}")

            print("\n[INFO] Loaded dataset entries (BurstGPT format):")
            print("=" * 60)
            for idx, row in data.iterrows():
                print(f"Entry {idx + 1}:")
                print(f"  Input tokens: {row['input_toks']}")
                print(f"  Output tokens: {row['output_toks']}")
                print(f"  Arrival time (ns): {row['arrival_time_ns']}")
                print(f"  Model type: {row.get('model_type', 'N/A')}")
                print(f"  Burst pattern: {row.get('burst_pattern', 'N/A')}")
                print("-" * 40)

            return True
        except Exception as e:
            print(f"[FAILED] BurstGPT CSV loading failed: {e}")
            return False
    else:
        print(f"[ERROR] BurstGPT CSV file not found: {csv_file}")
        return False


def test_dataset_info():
    """Test dataset information extraction."""
    print("\n=== Testing Dataset Info Extraction ===")

    loader = create_data_loader()

    # Only test with BurstGPT dataset
    files_to_test = [
        "../dataset/BurstGPT_1.csv"
    ]

    success_count = 0
    for file_path in files_to_test:
        if os.path.exists(file_path):
            try:
                info = loader.get_dataset_info(file_path)
                print(f"[SUCCESS] Info for {file_path}:")
                for key, value in info.items():
                    print(f"   {key}: {value}")
                success_count += 1
            except Exception as e:
                print(f"[FAILED] Info extraction failed for {file_path}: {e}")
        else:
            print(f"[ERROR] File not found: {file_path}")

    return success_count == len(files_to_test)


def test_burst_detection():
    """Test burst pattern detection."""
    print("\n=== Testing Burst Pattern Detection ===")

    # Create synthetic test data
    import pandas as pd
    import numpy as np

    # Steady pattern
    steady_times = [i * 100_000_000 for i in range(10)]  # 100ms intervals

    # Burst pattern
    burst_times = []
    base = 1_000_000_000  # Start at 1 second
    for i in range(5):
        burst_start = base + i * 500_000_000  # Every 500ms
        burst_times.extend([burst_start + j * 10_000_000 for j in range(3)])  # 3 requests 10ms apart

    all_times = steady_times + burst_times
    test_series = pd.Series(all_times)

    loader = create_data_loader(verbose=True)
    patterns = loader._detect_burst_pattern(test_series)

    print(f"[SUCCESS] Burst detection test")
    print(f"   Time points: {all_times}")
    print(f"   Detected patterns: {list(patterns)}")

    # Check if burst patterns are detected
    burst_count = sum(1 for p in patterns if p == 'burst')
    print(f"   Burst patterns detected: {burst_count}")

    # Return True if we detected some burst patterns
    return burst_count > 0


def main():
    """Run all tests."""
    print("Testing Universal Data Loader")
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
        ("BurstGPT Format", test_burstgpt_format),
        ("Dataset Info", test_dataset_info),
        ("Burst Detection", test_burst_detection)
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