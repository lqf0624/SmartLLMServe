# SmartLLMServe 🚀

**AI-Powered LLM Inference Serving with Predictive Scheduling**

> An intelligent enhancement to LLMServingSim featuring time series prediction and reinforcement learning for optimized LLM serving performance.

## 🎯 Project Overview

SmartLLMServe builds upon the excellent [LLMServingSim](https://github.com/casys-kaist/LLMServingSim) foundation to create a next-generation LLM serving optimization framework. By integrating advanced time series prediction and reinforcement learning algorithms, we enable proactive resource management and intelligent scheduling decisions.

### Key Innovation

- **🧠 Predictive Workload Analysis**: Forecast request patterns and resource demands
- **⚡ RL-Optimized Scheduling**: Make intelligent batching and resource allocation decisions
- **📊 Multi-Format Data Support**: Handle diverse workload patterns including burst scenarios
- **🎪 Real-time Adaptation**: Dynamically adjust to changing workload conditions

## 📋 Current Status

**Based on**: LLMServingSim v0.2.1 (MIT License)
**Enhancement Target**: v0.3.1 - Enhanced Prediction Framework
**Development Phase**: Phase 2 Complete ✅ - Ready for Phase 3 (RL Integration)

### ✅ **Completed Features (v0.3.1)**
- **Multi-Task Loss Function**: Optimized for arrival time, input tokens, and output tokens prediction
- **Lightweight Architecture**: DLinear-based models for efficient time series forecasting
- **Real Data Focus**: Exclusive use of BurstGPT dataset for authentic workload patterns
- **Enhanced Feature Engineering**: Comprehensive temporal and statistical feature extraction
- **Advanced Visualization System**: Professional prediction analysis with confidence intervals
- **Comprehensive Testing Suite**: Real-data validation across all components
- **Multi-GPU Support**: Scalable simulation infrastructure
- **Performance Analysis Tools**: System limitation identification and optimization insights

### 🚧 **Next Phase**: RL Scheduling Framework (v0.3.2)

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Workload Predictor│───▶│   RL Scheduler  │
│                 │    │                  │    │                 │
│ • BurstGPT      │    │ • Time Series     │    │ • State Space   │    │   Forecasting     │    │   Design        │
│ • Custom Datasets│    │ • Resource Demand │    │ • Action Space  │
│                 │    │   Prediction      │    │   Definition     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLMServingSim Core                           │
│                                                                 │
│  • Memory Management  • Request Scheduling  • Simulation Engine │
│  • KV Cache Optimization  • Batch Processing   • Performance     │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Development Roadmap

### Phase 1: Enhanced Data Infrastructure (v0.3.0) ✅ **COMPLETED**
- [x] **Universal Data Loader**
  - [x] Support for multiple dataset formats (TSV, CSV)
  - [x] BurstGPT dataset integration
  - [x] Automatic format detection and normalization
  - [x] Burst pattern detection algorithms
  - [x] Comprehensive test coverage

- [x] **Feature Engineering Pipeline**
  - [x] Time series feature extraction
  - [x] Request pattern analysis
  - [x] Resource demand characterization
  - [x] Real-time feature computation

### Phase 2: Workload Prediction Module (v0.3.1) ✅ **COMPLETED**
- [x] **Advanced Multi-Task Loss Function**
  - [x] Optimized for arrival time, input tokens, output tokens
  - [x] Configurable weights for different prediction targets
  - [x] Robust to outliers and workload variations
  - [x] Support for both sequence and point predictions

- [x] **Lightweight Model Architecture**
  - [x] DLinear-based time series decomposition
  - [x] Removed heavy Transformer/GRU components
  - [x] Efficient CPU/GPU deployment
  - [x] Maintains high prediction accuracy

- [x] **Real Data Integration**
  - [x] Exclusive BurstGPT dataset usage
  - [x] Removed synthetic ShareGPT dependencies
  - [x] Authentic workload pattern analysis
  - [x] Production-relevant validation

- [x] **Enhanced Visualization System**
  - [x] Professional prediction accuracy analysis
  - [x] Multi-scale time series decomposition
  - [x] Confidence interval estimation
  - [x] Resource requirement forecasting

### Phase 3: RL Scheduling Framework (v0.3.2)
- [ ] **Reinforcement Learning Environment**
  - [ ] State space definition (system status + predictions)
  - [ ] Action space design (batching, memory management)
  - [ ] Reward function formulation (multi-objective optimization)
  - [ ] Environment simulation and validation

- [ ] **RL Algorithm Implementation**
  - [ ] PPO/A3C for discrete action spaces
  - [ ] SAC/DDPG for continuous action spaces
  - [ ] Multi-objective RL formulations
  - [ ] Exploration strategies

### Phase 4: Integration and Optimization (v0.3.3)
- [ ] **Prediction-Orchestrated Scheduling**
  - [ ] Integration of predictor and RL scheduler
  - [ ] Real-time decision making pipeline
  - [ ] Fallback mechanisms for prediction failures
  - [ ] Performance monitoring and adaptation

- [ ] **Comprehensive Evaluation**
  - [ ] A/B testing against baseline scheduler
  - [ ] Performance benchmarking suite
  - [ ] Resource efficiency analysis
  - [ ] Real-world scenario validation

## 🎯 Technical Objectives

### Performance Goals
- **🎯 20-40% reduction** in average TTFT (Time to First Token)
- **⚡ 15-30% improvement** in overall throughput
- **💾 25-50% reduction** in memory fragmentation
- **📊 Real-time adaptation** to workload changes

### Research Contributions
- **Novel prediction-scheduling integration** framework
- **Multi-objective RL formulation** for LLM serving
- **Burst workload characterization** and handling
- **Online learning capabilities** for dynamic environments

## 📊 Dataset Support

### Currently Supported
- **BurstGPT**: Real-world production workload with authentic arrival patterns and multiple model types
- **Custom Datasets**: Flexible format support for research scenarios (TSV/CSV)

### Data Philosophy
- **Real Data Only**: Exclusively use production-relevant datasets (removed synthetic ShareGPT data)
- **Authentic Patterns**: Focus on real-world workload characteristics and burst scenarios
- **Research Value**: Prioritize datasets that provide genuine insights for LLM serving optimization

### Data Format Compatibility
```python
# Original TSV Format
input_toks    output_toks    arrival_time_ns
10            70             46926808

# BurstGPT CSV Format
Timestamp    Model    Request tokens    Response tokens    Total tokens    Log Type
5            ChatGPT  472               230                702              Conversation log
```

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.9+
- GCC/G++ 7.5.0
- Conda (recommended)

### Quick Start
```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/your-username/SmartLLMServe.git
cd SmartLLMServe

# Setup environment
conda env create -p ./env -f ./environment.yml
conda activate ./env

# Build dependencies
cd astra-sim
./build/astra_analytical/build.sh
cd extern/graph_frontend/chakra
pip install .
cd ../../..
```

### Running Experiments
```bash
# Baseline scheduling with BurstGPT dataset
python main.py --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
               --hardware 'RTX3090' --npu_num 1 --npu_group 1 \
               --dataset 'dataset/BurstGPT_1.csv' \
               --output 'output/baseline_burstgpt.csv' \
               --req_num 1000

# Multi-GPU simulation with enhanced prediction
python multi_gpu_simulation.py --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
                              --hardware 'RTX3090' --npu_num 2 --npu_group 1 \
                              --dataset 'dataset/BurstGPT_1.csv' \
                              --output 'output/multi_gpu_results.csv'

# Performance analysis and system limitation identification
python analysis/scripts/demo_limitations.py

# Quick test of multi-GPU setup
python run_multi_gpu_quick_test.py
```

### Testing Prediction Module
```bash
# Test comprehensive predictor functionality with real BurstGPT data
cd test
python test_predictor.py

# Expected: All core prediction tests should pass
# - Feature Extractor: Real feature extraction from BurstGPT
# - LSTM Predictor: Multi-task loss optimization
# - Workload Predictor: End-to-end prediction pipeline
# - Ensemble Predictor: Multi-model integration

# Generates professional visualization plots in predictor/test_visualization/
```

### Performance Analysis
```bash
# Run comprehensive system analysis
python analysis/scripts/demo_limitations.py

# Output: System limitation analysis and improvement insights
# - Performance bottleneck identification
# - Resource utilization analysis
# - Scheduling efficiency metrics
# - Optimization recommendations
```

### Multi-GPU Testing
```bash
# Quick multi-GPU simulation test
python run_multi_gpu_quick_test.py

# Full multi-GPU simulation
python multi_gpu_simulation.py --config config/multi_gpu_config.json
```

## 🤝 Contributing

We welcome contributions in:
- **Time Series Prediction**: Novel forecasting algorithms
- **Reinforcement Learning**: Advanced RL formulations for scheduling
- **Performance Modeling**: Resource utilization prediction
- **Workload Characterization**: Real-world trace analysis
- **System Integration**: Production-ready deployment strategies

### Development Guidelines
1. Fork the repository and create a feature branch
2. Follow the existing code style and architecture patterns
3. Add comprehensive tests for new features
4. Update documentation and examples
5. Submit pull requests with clear descriptions

## 📊 Visualization Capabilities

### Enhanced Prediction Analysis
Our visualization system provides professional-grade charts for workload prediction analysis:

#### 🎯 **Actual vs Predicted Comparison**
- **Request Rate Analysis**: Side-by-side comparison with confidence intervals
- **Statistical Annotations**: MAE, MAPE, and other accuracy metrics
- **Multi-Scale Visualization**: 30s, 1min, 5min time windows for pattern detection

#### 📈 **Time Series Analysis**
- **Trend Decomposition**: Long-term pattern identification
- **Rolling Statistics**: Moving averages and standard deviations
- **Seasonality Detection**: Periodic pattern analysis
- **Prediction Horizon Analysis**: Accuracy vs forecast length

#### 📉 **Error Analysis**
- **Residual Analysis**: Prediction error patterns
- **Distribution Analysis**: Error histograms with normal overlays
- **Q-Q Plots**: Normality testing for prediction errors
- **Confidence Calibration**: Stated vs actual confidence comparison

#### 💾 **Resource Forecasting**
- **Memory Requirements**: Predicted memory usage over time
- **Compute Demand**: Estimated processing time forecasts
- **Burst Analysis**: Workload intensity and interval predictions
- **Resource Allocation**: Optimized resource suggestions

### Generated Output Files
```
predictor/test_visualization/
├── 01_prediction_overview.png      # Main prediction vs actual comparison
├── 02_time_series_comparison.png   # Multi-scale time series analysis
├── 03_resource_forecast.png        # Memory and compute requirements
└── 04_burst_patterns.png           # Burst pattern analysis
```

## 📈 Performance Monitoring

### Key Metrics
- **TTFT (Time to First Token)**: Latency for initial response
- **TPOT (Time Per Output Token)**: Token generation throughput
- **Queue Delay**: Waiting time before processing
- **Memory Utilization**: KV cache and memory efficiency
- **GPU Utilization**: Compute resource usage

### Real-time Analytics
- Live performance dashboards
- Adaptive parameter tuning
- Scheduling decision visualization
- Resource allocation tracking

## 📄 License

This project is based on [LLMServingSim](https://github.com/casys-kaist/LLMServingSim) and licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original LLMServingSim**: CASYS Lab at KAIST for the excellent simulation framework
- **Research Community**: Contributors to LLM serving optimization research
- **Open Source Contributors**: Developers of the underlying tools and libraries

## 📞 Contact

- **Project Maintainers**: [Your Name/Team]
- **Issues & Feature Requests**: [GitHub Issues](https://github.com/your-username/SmartLLMServe/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/SmartLLMServe/discussions)

---

**Last Updated**: September 29, 2025
**Current Version**: v0.3.1 - Enhanced Prediction Framework with Multi-Task Loss
**Next Milestone**: v0.3.2 - RL Scheduling Integration