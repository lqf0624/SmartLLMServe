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
**Enhancement Target**: v0.3.0 - Predictive Scheduling Framework
**Development Phase**: Phase 2/4 - Workload Prediction Module

### ✅ **Completed Features (v0.3.0a)**
- **Universal Data Loader**: Multi-format dataset support (TSV/CSV)
- **BurstGPT Integration**: Real-world workload patterns
- **Pattern Detection**: Automatic burst/steady/sparse classification
- **Comprehensive Testing**: Full test coverage in `test/` directory

### 🚧 **Next Phase**: Time Series Prediction Module (v0.3.1)

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Workload Predictor│───▶│   RL Scheduler  │
│                 │    │                  │    │                 │
│ • ShareGPT      │    │ • Time Series     │    │ • State Space   │
│ • BurstGPT      │    │   Forecasting     │    │   Design        │
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

- [ ] **Feature Engineering Pipeline**
  - [ ] Time series feature extraction
  - [ ] Request pattern analysis
  - [ ] Resource demand characterization
  - [ ] Real-time feature computation

### Phase 2: Workload Prediction Module (v0.3.1)
- [ ] **Time Series Prediction Models**
  - [ ] LSTM/Transformer-based forecasting
  - [ ] Statistical baseline models (ARIMA, Prophet)
  - [ ] Ensemble prediction methods
  - [ ] Online learning capability

- [ ] **Multi-Horizon Forecasting**
  - [ ] Short-term prediction (next 100ms)
  - [ ] Medium-term prediction (next 1s)
  - [ ] Long-term trend analysis
  - [ ] Confidence interval estimation

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
- **ShareGPT**: Standard conversation traces with Poisson arrival patterns
- **BurstGPT**: Real-world bursty workload with multiple model types
- **Custom Datasets**: Flexible format support for research scenarios

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
# Original scheduling with TSV dataset (baseline)
python main.py --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
               --hardware 'RTX3090' --npu_num 1 --npu_group 1 \
               --dataset 'dataset/share-gpt-req100-rate10.tsv' \
               --output 'output/baseline_tsv.csv'

# Original scheduling with BurstGPT dataset (new capability)
python main.py --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
               --hardware 'RTX3090' --npu_num 1 --npu_group 1 \
               --dataset 'dataset/BurstGPT_1.csv' \
               --output 'output/baseline_burstgpt.csv' \
               --req_num 1000

# With predictive scheduling (coming in v0.3.1)
python main.py --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
               --hardware 'RTX3090' --npu_num 1 --npu_group 1 \
               --dataset 'dataset/BurstGPT_1.csv' \
               --scheduler 'predictive_rl' \
               --output 'output/predictive_results.csv'
```

### Testing Data Loading
```bash
# Test the universal data loader functionality
cd test
python test_data_loader.py

# Expected: All 4 tests should pass
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

**Last Updated**: September 25, 2025
**Current Version**: v0.3.0a - Data Infrastructure Complete
**Next Milestone**: v0.3.1 - Time Series Prediction Module