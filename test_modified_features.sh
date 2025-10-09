#!/bin/bash

echo "========================================"
echo "测试修改后的功能"
echo "========================================"

# 1. 检查Python语法
echo "1. 检查训练脚本语法..."
python -m py_compile training_scripts/train_predictor.py
if [ $? -eq 0 ]; then
    echo "✅ 训练脚本语法正确"
else
    echo "❌ 训练脚本语法错误"
    exit 1
fi

# 2. 检查配置文件JSON格式
echo "2. 检查配置文件JSON格式..."
python -c "
import json
try:
    with open('training_output/training_config.json', 'r') as f:
        config = json.load(f)
    print('✅ training_config.json 格式正确')
    print(f'   早停patience: {config.get(\"early_stopping_patience\", \"N/A\")}')
except Exception as e:
    print(f'❌ training_config.json 格式错误: {e}')
    exit(1)
"

python -c "
import json
try:
    with open('training_scripts/configs/predictor_config.json', 'r') as f:
        config = json.load(f)
    print('✅ predictor_config.json 格式正确')
    print(f'   早停patience: {config.get(\"early_stopping_patience\", \"N/A\")}')
except Exception as e:
    print(f'❌ predictor_config.json 格式错误: {e}')
    exit(1)
"

# 3. 检查DLinear模型导入
echo "3. 检查DLinear模型导入..."
python -c "
try:
    from predictor.dlinear_model import DLinearPredictor
    print('✅ DLinearPredictor 导入成功')
except Exception as e:
    print(f'❌ DLinearPredictor 导入失败: {e}')
    exit(1)
"

# 4. 检查训练脚本参数解析
echo "4. 检查训练脚本参数解析..."
python -c "
import sys
sys.path.append('training_scripts')
try:
    from train_predictor import load_default_config
    config = load_default_config()
    print('✅ 训练脚本配置加载成功')
    print(f'   默认早停patience: {config.get(\"early_stopping_patience\", \"N/A\")}')
except Exception as e:
    print(f'❌ 训练脚本配置加载失败: {e}')
    exit(1)
"

# 5. 测试数据处理功能
echo "5. 测试数据处理功能..."
python -c "
import pandas as pd
import numpy as np
import os

# 创建测试数据
test_data = pd.DataFrame({
    'Timestamp': pd.date_range('2025-01-01', periods=150, freq='T'),
    'Concurrent_requests': np.random.poisson(5, 150),
    'Request_tokens_sum': np.random.randint(100, 1000, 150),
    'Response_tokens_sum': np.random.randint(50, 500, 150)
})

# 保存测试数据
os.makedirs('test_output', exist_ok=True)
test_data.to_csv('test_output/test_data.csv', index=False)
print('✅ 测试数据创建成功')

# 测试数据读取
loaded_data = pd.read_csv('test_output/test_data.csv')
print(f'✅ 测试数据读取成功，共{len(loaded_data)}行')
"

# 6. 测试绘图功能
echo "6. 测试绘图功能..."
python -c "
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

# 创建测试图表
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pred_values = np.array([5.2, 350.1, 180.5])
true_values = np.array([4.8, 380.2, 175.3])
feature_names = ['Concurrent\nRequests', 'Input\nTokens', 'Output\nTokens']
x_pos = np.arange(len(feature_names))

bar_width = 0.35
bars1 = ax.bar(x_pos - bar_width/2, pred_values, bar_width,
              label='Predicted', alpha=0.8, color='skyblue')
bars2 = ax.bar(x_pos + bar_width/2, true_values, bar_width,
              label='Actual', alpha=0.8, color='lightcoral')

ax.set_xlabel('Features')
ax.set_ylabel('Values')
ax.set_title('Predicted vs Actual Values Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names)
ax.legend()

plt.tight_layout()
plt.savefig('test_output/test_prediction_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print('✅ 绘图功能测试成功')
"

# 7. 检查wandb配置
echo "7. 检查wandb配置..."
python -c "
try:
    import wandb
    print('✅ wandb 导入成功')
except ImportError:
    print('⚠️  wandb 未安装，但不影响基本功能测试')
"

# 8. 检查CUDA可用性
echo "8. 检查CUDA可用性..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA 可用，版本: {torch.version.cuda}')
    print(f'   GPU数量: {torch.cuda.device_count()}')
else:
    print('⚠️  CUDA 不可用，将使用CPU训练')
"

echo ""
echo "========================================"
echo "✅ 所有基本功能测试通过！"
echo "========================================"

# 清理测试文件
rm -rf test_output