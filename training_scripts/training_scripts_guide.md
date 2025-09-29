# 训练脚本使用指南

## 📁 训练脚本位置

### 生产训练脚本（简化版）

| 脚本文件 | 用途 | 特点 |
|---------|------|------|
| `simple_training.py` | **简化训练脚本** | 仅LSTM和DLinear，推荐使用 |
| `production_training.py` | **生产环境训练脚本** | 支持CPU和GPU自动检测 |
| `cpu_training.py` | **CPU专用训练脚本** | CPU环境优化 |

### 测试和验证脚本

| 脚本文件 | 用途 |
|---------|------|
| `test_request_predictor.py` | 基础训练测试 |
| `show_existing_plots.py` | 结果可视化 |

## 🚀 快速开始

### 方案1：简化训练（推荐）
```bash
# 最简单的训练方式
python simple_training.py

# 指定数据路径和采样大小
python simple_training.py --data-path dataset/BurstGPT_1_cleaned.csv --sample-size 100000
```

### 方案2：生产环境训练
```bash
# 自动检测CPU/GPU
python production_training.py

# 指定数据路径和采样大小
python production_training.py --data-path dataset/BurstGPT_1_cleaned.csv --sample-size 100000
```

### 方案3：CPU专用训练
```bash
# CPU环境快速训练
python cpu_training.py

# CPU环境小数据集训练
python cpu_training.py --sample-size 50000
```

## 📋 详细使用方法

### 1. 简化训练脚本详解（推荐）

**simple_training.py** - 主要特点：
- 仅支持LSTM和DLinear模型
- 自动设备检测和优化
- 最简化的API和配置
- 包含预测测试功能

```bash
# 基本用法
python simple_training.py

# 指定数据路径和采样大小
python simple_training.py \
  --data-path dataset/BurstGPT_1_cleaned.csv \
  --sample-size 100000
```

**参数说明：**
- `--data-path`: 数据文件路径
- `--sample-size`: 数据采样大小

### 2. 生产环境训练脚本详解

**production_training.py** - 主要特点：
- 自动检测CPU/GPU设备
- 完整的生产级配置
- 支持模型保存功能

```bash
# 基本用法
python production_training.py

# 高级用法
python production_training.py \
  --data-path dataset/BurstGPT_1_cleaned.csv \
  --sample-size 100000 \
  --model-dir models
```

**参数说明：**
- `--data-path`: 数据文件路径
- `--sample-size`: 数据采样大小
- `--model-dir`: 模型保存目录

### 3. CPU专用训练脚本详解

**cpu_training.py** - CPU优化特点：
- 减少序列长度 (80 vs 100)
- 减少训练轮数 (50 vs 100)
- 减少批量大小 (32 vs 64)
- 更小的数据集默认值

```bash
# CPU快速训练
python cpu_training.py

# CPU小数据集训练
python cpu_training.py --sample-size 30000
```

### 4. 自定义训练脚本

创建你的训练脚本 `custom_training.py`：

```python
#!/usr/bin/env python3
"""
自定义训练脚本示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_training import SimpleTrainer

def main():
    # 创建简化训练器
    trainer = SimpleTrainer()

    # 运行自定义训练
    success = trainer.run_simple_training(
        data_path='dataset/BurstGPT_1_cleaned.csv',
        sample_size=50000  # 使用5万条数据
    )

    if success:
        print("自定义训练完成!")
    else:
        print("训练失败!")

if __name__ == "__main__":
    main()
```

## 🔧 生产环境部署

### 1. 训练配置建议

**GPU环境配置：**
```bash
# 大规模GPU训练
python production_training.py \
  --sample-size 200000 \
  --models lstm dlinear
```

**CPU环境配置：**
```bash
# 快速CPU训练
python cpu_training.py \
  --sample-size 50000 \
  --models lstm
```

### 2. 模型管理

训练完成后，模型将保存在 `models/` 目录：

```
models/
├── production_lstm_20250926_143000.pth
├── production_dlinear_20250926_143000.pth
└── cpu_lstm_20250926_143000.pth
```

### 3. 生产环境使用

训练好的预测器可以直接用于预测：

```python
from predictor.request_predictor import RequestPredictor
import pandas as pd

# 创建预测器（自动使用训练好的模型）
predictor = RequestPredictor()

# 进行预测
historical_data = pd.read_csv('dataset/BurstGPT_1_cleaned.csv', nrows=1000)
prediction = predictor.predict_next_request(historical_data)

# 查看预测结果
print("预测结果:")
for model_name, pred in prediction['predictions'].items():
    if 'error' not in pred:
        print(f"{model_name}:")
        print(f"  时间戳: {pred['timestamp']}")
        print(f"  请求tokens: {pred['request_tokens']}")
        print(f"  响应tokens: {pred['response_tokens']}")
        if 'confidence' in pred:
            print(f"  置信度: {pred['confidence']}")
```

## 📊 训练结果

### 训练输出示例
```
🚀 简化预测模型训练
========================================

加载数据
------------------------------
数据文件: dataset/BurstGPT_1_cleaned.csv
原始数据: 1,404,294 条记录
采样到: 100,000 条记录
清理后数据: 99,987 条记录

模型训练
------------------------------
开始训练 LSTM + DLinear 模型...
训练数据: 99,987 条记录

训练结果:
[OK] lstm:
    最终损失: 0.0234
    训练轮数: 78
[OK] dlinear:
    最终损失: 0.0189
    训练轮数: 95

成功训练: 2/2 个模型

测试预测
------------------------------
预测结果:
  lstm:
    时间戳: 1234567.89
    请求tokens: 150
    响应tokens: 300
    置信度: 0.856
  dlinear:
    时间戳: 1234567.92
    请求tokens: 145
    响应tokens: 295
    置信度: 0.912

训练摘要
------------------------------
训练时间: 2025-09-26 14:30:00
数据规模: 99,987 条记录
成功训练: 2/2 个模型
成功率: 100.0%
lstm: 损失 = 0.0234
dlinear: 损失 = 0.0189

使用方法:
1. 直接使用训练好的预测器:
   predictor = RequestPredictor()
   prediction = predictor.predict_next_request(historical_data)
2. 预测结果包含LSTM和DLinear的独立预测
```

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**
```bash
# 减少数据采样大小
python simple_training.py --sample-size 50000

# 或使用CPU训练
python cpu_training.py --sample-size 30000
```

2. **数据文件未找到**
```bash
# 确保数据文件存在
ls dataset/BurstGPT_1_cleaned.csv

# 或指定正确路径
python simple_training.py --data-path /path/to/your/data.csv
```

3. **模型训练失败**
```bash
# 使用CPU训练进行调试
python cpu_training.py --sample-size 10000
```

### 性能优化建议

1. **GPU环境：**
   - 使用更大的数据集 (100000+)
   - 自动优化批量大小和训练轮数

2. **CPU环境：**
   - 减少数据采样大小 (30000-50000)
   - 使用cpu_training.py优化参数

## 📝 最佳实践

1. **推荐使用简化脚本：**
   - `python simple_training.py` - 最简单的方式
   - 自动处理所有配置和优化

2. **数据准备：**
   - 确保数据文件存在且格式正确
   - 检查必需列：Timestamp, Request tokens, Response tokens
   - 清理异常值和缺失数据

3. **模型对比：**
   - LSTM：适合复杂的时间序列模式
   - DLinear：适合线性趋势和季节性模式
   - 两个模型独立预测，可以选择更准确的

4. **训练监控：**
   - 观察训练损失值
   - 监控训练成功率
   - 检查预测测试结果

5. **生产部署：**
   - 使用RequestPredictor直接进行预测
   - 模型自动包含在预测器中
   - 无需手动加载模型文件

---

**总结：**
- **推荐使用** `simple_training.py` - 最简化的训练方式
- **生产环境** 使用 `production_training.py` - 支持模型保存
- **CPU环境** 使用 `cpu_training.py` - CPU优化参数
- **所有脚本** 专注于LSTM和DLinear模型，移除了高级模型复杂性