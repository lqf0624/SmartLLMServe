# 项目结构说明

## 📁 目录结构

```
LLMServingSim/
├── 📁 predictor/                    # 预测模型核心代码
│   ├── __init__.py
│   ├── request_predictor.py        # 主要预测器类
│   ├── lstm_model.py               # LSTM模型实现
│   ├── dlinear_model.py            # DLinear模型实现
│   ├── dlinear_predictor_adapter.py # DLinear适配器
│   ├── model_interface.py          # 模型接口定义
│   ├── data_preprocessor.py        # 数据预处理
│   ├── feature_extractor.py        # 特征提取
│   ├── simple_visualization.py     # 简化可视化
│   ├── visualization.py            # 完整可视化
│   └── 📁 tests/                   # 预测器测试文件
│       ├── test_request_predictor.py
│       ├── debug_lstm_issue.py
│       ├── test_lstm_simple.py
│       └── test_enhanced_predictor.py
├── 📁 training_scripts/            # 训练脚本
│   ├── simple_training.py         # 简化训练脚本（推荐）
│   ├── production_training.py     # 生产环境训练脚本
│   ├── cpu_training.py            # CPU优化训练脚本
│   ├── demo_simple_visualization.py # 可视化演示
│   └── training_scripts_guide.md  # 训练脚本使用指南
├── 📁 test/                        # 测试目录
│   ├── test_predictor.py
│   ├── test_data_loader.py
│   ├── test_visualization_demo.py
│   └── 📁 predictor/              # 预测器专项测试
├── 📁 docs/                        # 文档目录
│   ├── main.py                    # 原主文件（保留参考）
│   ├── performance_optimization_report.md
│   ├── demo_prediction_comparison.png
│   └── demo_prediction_report.txt
├── 📁 dataset/                     # 数据集目录
│   ├── BurstGPT_1_cleaned.csv      # 清理后的数据集
│   └── BurstGPT_1.csv             # 原始数据集
├── 📁 models/                      # 训练好的模型保存目录
├── 📁 output/                      # 输出结果目录
├── CLAUDE.md                      # 项目主要文档
├── README.md                      # 项目说明
└── environment.yml                # 环境配置
```

## 🚀 快速开始

### 1. 训练模型（推荐使用简化脚本）
```bash
# 进入项目根目录
cd LLMServingSim

# 使用简化训练脚本（推荐）
python training_scripts/simple_training.py --sample-size 100000

# 或使用生产环境训练脚本
python training_scripts/production_training.py --sample-size 100000

# CPU环境使用优化脚本
python training_scripts/cpu_training.py --sample-size 50000
```

### 2. 使用训练好的预测器
```python
from predictor.request_predictor import RequestPredictor
import pandas as pd

# 创建预测器
predictor = RequestPredictor()

# 加载历史数据
historical_data = pd.read_csv('dataset/BurstGPT_1_cleaned.csv', nrows=1000)

# 进行预测
prediction = predictor.predict_next_request(historical_data)

# 查看结果
print("预测结果:")
for model_name, pred in prediction['predictions'].items():
    if 'error' not in pred:
        print(f"{model_name}: 时间戳={pred['timestamp']}, "
              f"请求tokens={pred['request_tokens']}, "
              f"响应tokens={pred['response_tokens']}")
```

## 📋 训练脚本说明

| 脚本 | 用途 | 特点 |
|------|------|------|
| `simple_training.py` | **简化训练** | 最简单的方式，推荐使用 |
| `production_training.py` | **生产训练** | 支持CPU/GPU自动检测 |
| `cpu_training.py` | **CPU训练** | CPU环境参数优化 |

## 🔧 核心功能

### 预测模型
- **LSTM**: 长短期记忆网络，适合复杂时间序列模式
- **DLinear**: 基于移动平均分解的线性模型，适合趋势预测

### 预测内容
- 下一个请求的到达时间（Timestamp）
- 请求的token数量（Request tokens）
- 响应的token数量（Response tokens）

### 技术特点
- 支持CPU和GPU自动检测
- 自动数据预处理和特征提取
- 提供训练损失和验证指标
- 包含预测测试功能

## 📊 使用建议

1. **首次使用**: 推荐使用 `simple_training.py`
2. **生产环境**: 使用 `production_training.py`
3. **CPU环境**: 使用 `cpu_training.py` 获得更好性能
4. **模型选择**: LSTM适合复杂模式，DLinear适合线性趋势

## 📝 注意事项

- 确保数据文件 `dataset/BurstGPT_1_cleaned.csv` 存在
- 训练数据至少需要60条记录（sequence_length + prediction_horizon）
- GPU环境会自动检测，无需额外配置
- 训练好的模型保存在 `models/` 目录