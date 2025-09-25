# Test Directory

This directory contains test scripts and utilities for the SmartLLMServe project.

## Directory Structure

```
test/
├── README.md                    # This file
├── test_data_loader.py          # Data loading functionality tests
└── [future test files...]        # Additional test modules
```

## Running Tests

### Data Loader Tests
Tests the universal data loader functionality with multiple dataset formats.

```bash
# Run from project root directory
cd test
python test_data_loader.py

# Or run from test directory
python test_data_loader.py
```

### Expected Output
All tests should pass:
```
Testing Universal Data Loader
==================================================
[PASS] TSV Format
[PASS] BurstGPT Format
[PASS] Dataset Info
[PASS] Burst Detection

Overall: 4/4 tests passed
```

## Test Coverage

### Current Tests
- **Data Loading**: TSV and CSV format compatibility
- **Dataset Info**: Metadata extraction functionality
- **Burst Detection**: Pattern detection algorithms

### Future Tests
- Time series prediction models
- RL scheduling algorithms
- Integration tests
- Performance benchmarks

## Test Data

Tests use the following datasets from `../dataset/`:
- `share-gpt-req100-rate10.tsv` - Original TSV format
- `BurstGPT_1.csv` - Burst workload CSV format

## Adding New Tests

1. Create test file with prefix `test_`
2. Follow existing naming conventions
3. Include comprehensive test cases
4. Add documentation for new functionality
5. Update this README when adding new test modules

## Test Conventions

- Use `[SUCCESS]`, `[FAILED]`, `[ERROR]`, `[INFO]` prefixes for output
- Return boolean values for pass/fail status
- Include detailed error messages
- Test both success and failure scenarios
- Use relative paths from test directory

## Continuous Integration

These tests are designed to be run as part of CI/CD pipeline to ensure:
- Backward compatibility with original LLMServingSim
- New features work as expected
- Performance regressions are caught early