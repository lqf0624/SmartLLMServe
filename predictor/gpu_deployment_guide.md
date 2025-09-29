# GPU环境预测模型部署指南

## 概述

本指南提供了在无法联网的GPU服务器上部署和使用预测模型的完整方案。新的接口设计支持动态模型注册、GPU设备管理和离线训练。

## 核心特性

### 1. 可扩展模型接口
- **抽象基类**: `BasePredictor` 为所有模型提供统一接口
- **模型注册表**: `ModelRegistry` 支持动态模型注册
- **配置管理**: `ModelConfig` 统一模型配置
- **设备管理**: 自动GPU/CPU设备切换

### 2. 预置模型类型
- **LSTM**: 长短期记忆网络（已实现）
- **DLinear**: 线性分解模型（已实现）
- **Transformer**: 注意力机制模型（新实现）
- **GRU**: 门控循环单元（新实现）
- **自定义模型**: 支持轻松集成新模型

### 3. GPU优化
- **自动设备检测**: 自动选择最佳计算设备
- **内存管理**: 智能GPU内存分配
- **批量训练**: 支持大批量GPU训练
- **模型并行**: 支持多GPU训练（扩展）

## 快速开始

### 1. 基础使用

```python
# 1. 导入模块
from predictor.enhanced_request_predictor import create_enhanced_predictor
import pandas as pd

# 2. 创建预测器（自动选择GPU）
predictor = create_enhanced_predictor(
    models=['lstm', 'dlinear', 'transformer', 'gru'],
    device='cuda'  # 或 'auto' 自动选择
)

# 3. 加载数据
data = pd.read_csv('dataset/BurstGPT_1_cleaned.csv', nrows=10000)

# 4. 训练模型
training_results = predictor.train_models(data, validation_split=0.2)

# 5. 预测
historical_data = data.iloc[:8000]
prediction = predictor.predict_next_request(historical_data)
```

### 2. 高级配置

```python
from predictor.enhanced_request_predictor import EnhancedRequestPredictor
from predictor.model_interface import ModelConfig, ModelType, ModelDevice

# 创建预测器
predictor = EnhancedRequestPredictor(device='cuda')

# 自定义模型配置
transformer_config = ModelConfig(
    model_type=ModelType.TRANSFORMER,
    model_name='large_transformer',
    device=ModelDevice.CUDA,
    sequence_length=100,
    prediction_horizon=20,
    epochs=100,
    batch_size=64,
    learning_rate=0.0001,
    custom_params={
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'dropout': 0.1
    }
)

# 添加模型
predictor.add_model('transformer', transformer_config)

# 训练和预测
training_results = predictor.train_models(data)
predictions = predictor.predict_next_request(historical_data)
```

## 自定义模型集成

### 1. 创建自定义模型

```python
from predictor.model_interface import BasePredictor, ModelConfig, register_model
import torch
import torch.nn as nn

@register_model(ModelType.CUSTOM)  # 注册模型类型
class MyCustomModel(BasePredictor):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None

    def prepare_data(self, data: pd.DataFrame):
        # 实现数据预处理
        pass

    def build_model(self):
        # 构建模型架构
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(3, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                )

            def forward(self, x):
                return self.layers(x)

        self.model = MyModel().to(self.device)
        return self.model

    def train(self, data: pd.DataFrame, validation_split: float = 0.2):
        # 实现训练逻辑
        # 必须返回 TrainingResult
        pass

    def predict(self, data: pd.DataFrame, steps: int = None):
        # 实现预测逻辑
        # 必须返回 PredictionResult
        pass

    def save_model(self, path: str) -> bool:
        torch.save(self.model.state_dict(), path)
        return True

    def load_model(self, path: str) -> bool:
        self.build_model()
        self.model.load_state_dict(torch.load(path))
        return True
```

### 2. 使用自定义模型

```python
# 配置自定义模型
custom_config = ModelConfig(
    model_type=ModelType.CUSTOM,
    model_name='my_custom_model',
    device='cuda',
    sequence_length=50,
    prediction_horizon=10,
    custom_params={
        'my_param1': 'value1',
        'my_param2': 123
    }
)

# 添加到预测器
predictor = EnhancedRequestPredictor(device='cuda')
predictor.add_model('custom', custom_config)

# 正常使用
training_results = predictor.train_models(data)
predictions = predictor.predict_next_request(historical_data)
```

## GPU部署配置

### 1. 环境要求

```bash
# 系统要求
- Ubuntu 18.04+ 或 CentOS 7+
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA 11.0+
- cuDNN 8.0+

# Python环境
python >= 3.8
torch >= 1.12.0
pandas >= 1.3.0
numpy >= 1.19.0
scikit-learn >= 1.0.0
```

### 2. 安装脚本

```bash
#!/bin/bash
# install_gpu_env.sh - GPU环境安装脚本

# 安装PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install pandas numpy scikit-learn matplotlib seaborn

# 验证安装
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
```

### 3. 性能优化配置

```python
# GPU优化配置
gpu_optimized_config = ModelConfig(
    model_type=ModelType.TRANSFORMER,
    model_name='gpu_optimized_transformer',
    device='cuda',
    sequence_length=200,  # 更长序列
    prediction_horizon=50,  # 更长预测
    batch_size=256,  # 更大批量
    epochs=200,
    learning_rate=0.0001,
    custom_params={
        'd_model': 512,
        'nhead': 16,
        'num_layers': 12,
        'dropout': 0.1,
        # GPU特定优化
        'gradient_checkpointing': True,
        'mixed_precision': True,
        'amp': True  # 自动混合精度
    }
)
```

## 模型管理

### 1. 模型保存和加载

```python
# 保存训练好的模型
model_path = 'models/best_transformer_model.pth'
predictor.save_model('transformer', model_path)

# 加载已保存的模型
config = ModelConfig(
    model_type=ModelType.TRANSFORMER,
    model_name='loaded_transformer',
    device='cuda'
)
predictor.load_model('transformer', model_path, config)
```

### 2. 批量模型管理

```python
# 添加多个模型
models_to_add = ['lstm', 'dlinear', 'transformer', 'gru']
results = predictor.add_models(models_to_add)

# 训练所有模型
training_results = predictor.train_models(data)

# 比较模型性能
comparison = predictor.compare_models(historical_data)

# 获取最佳模型
best_model = comparison['performance_summary']['best_model']
print(f"Best performing model: {best_model}")
```

## 离线部署方案

### 1. 完整离线脚本

```python
# offline_gpu_training.py
import torch
import pandas as pd
from predictor.enhanced_request_predictor import EnhancedRequestPredictor
from predictor.model_interface import ModelConfig, ModelType

def offline_training():
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建预测器
    predictor = EnhancedRequestPredictor(device=device)

    # 配置模型（针对离线环境优化）
    models_config = {
        'lstm': ModelConfig(
            model_type=ModelType.LSTM,
            model_name='offline_lstm',
            device=device,
            sequence_length=100,
            prediction_horizon=20,
            epochs=50,
            batch_size=128
        ),
        'transformer': ModelConfig(
            model_type=ModelType.TRANSFORMER,
            model_name='offline_transformer',
            device=device,
            sequence_length=100,
            prediction_horizon=20,
            epochs=30,
            batch_size=64,
            custom_params={'d_model': 128, 'nhead': 4}
        )
    }

    # 添加模型
    for name, config in models_config.items():
        predictor.add_model(name, config)

    # 加载数据
    data = pd.read_csv('dataset/BurstGPT_1_cleaned.csv')
    print(f"Loaded {len(data)} records")

    # 训练模型
    training_results = predictor.train_models(data, validation_split=0.2)

    # 保存所有模型
    for model_name in predictor.get_active_models():
        model_path = f'models/{model_name}_offline_final.pth'
        if predictor.save_model(model_name, model_path):
            print(f"Saved {model_name} to {model_path}")

    print("Offline training completed!")

if __name__ == "__main__":
    offline_training()
```

### 2. 推理服务

```python
# inference_service.py
import torch
import pandas as pd
from predictor.enhanced_request_predictor import EnhancedRequestPredictor
from predictor.model_interface import ModelConfig, ModelType

class InferenceService:
    def __init__(self):
        self.predictor = EnhancedRequestPredictor(device='cuda')
        self.load_models()

    def load_models(self):
        """加载预训练模型"""
        models_to_load = [
            ('lstm', 'models/offline_lstm_final.pth'),
            ('transformer', 'models/offline_transformer_final.pth')
        ]

        for model_name, model_path in models_to_load:
            config = ModelConfig(
                model_type=ModelType.LSTM if model_name == 'lstm' else ModelType.TRANSFORMER,
                model_name=f'inference_{model_name}',
                device='cuda'
            )
            self.predictor.load_model(model_name, model_path, config)
            print(f"Loaded {model_name} from {model_path}")

    def predict_next_request(self, historical_data):
        """预测下一个请求"""
        return self.predictor.predict_next_request(historical_data)

    def get_model_info(self):
        """获取模型信息"""
        return {
            name: model.get_model_info()
            for name, model in self.predictor.model_manager.active_models.items()
        }

# 使用示例
if __name__ == "__main__":
    service = InferenceService()

    # 模拟实时数据
    historical_data = pd.read_csv('dataset/BurstGPT_1_cleaned.csv', nrows=1000)

    # 预测
    prediction = service.predict_next_request(historical_data)
    print("Prediction result:", prediction)
```

## 性能监控

### 1. GPU监控

```python
import torch
import psutil
import time

def monitor_gpu_performance():
    """GPU性能监控"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("GPU Performance Monitor")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
    print(f"Utilization: {torch.cuda.utilization()}%")
```

### 2. 训练监控

```python
def monitor_training_progress(predictor):
    """监控训练进度"""
    for model_name, model in predictor.model_manager.active_models.items():
        if hasattr(model, 'training_history'):
            history = model.training_history
            print(f"{model_name}:")
            print(f"  Final Loss: {history.get('final_loss', 'N/A')}")
            print(f"  Best Loss: {history.get('best_loss', 'N/A')}")
            print(f"  Epochs: {history.get('epochs_trained', 'N/A')}")
```

## 故障排除

### 1. 常见问题

**CUDA内存不足**
```python
# 减少批量大小
config = ModelConfig(
    batch_size=32,  # 从64减少到32
    sequence_length=50,  # 从100减少到50
)
```

**模型不收敛**
```python
# 调整学习率和训练参数
config = ModelConfig(
    learning_rate=0.0001,  # 降低学习率
    epochs=200,  # 增加训练轮数
    early_stopping=True,
    patience=20
)
```

### 2. 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.INFO)

# 检查设备兼容性
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## 总结

新的预测模型接口提供了：

1. **可扩展性**: 轻松集成新模型
2. **GPU优化**: 自动设备管理和内存优化
3. **离线支持**: 完整的离线训练和推理方案
4. **统一接口**: 所有模型使用相同的API
5. **性能监控**: 内置性能监控和调试工具

这个设计确保了在无法联网的GPU服务器上可以顺利地进行模型训练、评估和部署。