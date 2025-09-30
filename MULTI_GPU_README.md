# 多卡环境模拟实验指南

本目录包含多卡环境LLM推理模拟的完整工具集。

## 文件结构

```
LLMServingSim/
├── multi_gpu_simulation.py          # 主要的多卡模拟脚本
├── run_multi_gpu_quick_test.py      # 快速测试脚本
├── main.py                          # 主模拟程序
├── analysis/                        # 分析脚本目录
│   ├── scripts/                     # 整理后的分析脚本
│   └── scripts_info.py             # 脚本信息说明
└── output/multi_gpu_results/        # 实验结果输出目录
```

## 快速开始

### 1. 环境检查
确保以下依赖已安装：
- Python 3.8+
- pandas, numpy, matplotlib
- LLMServingSim 依赖项

### 2. 快速测试
运行快速测试验证环境：

```bash
python run_multi_gpu_quick_test.py
```

这将测试1-4卡配置，使用小规模数据集进行验证。

### 3. 完整实验
运行完整的多卡模拟实验：

```bash
python multi_gpu_simulation.py
```

## 配置说明

### GPU配置选项
- `single_gpu`: 1卡基准测试
- `dual_gpu`: 2卡并行
- `quad_gpu`: 4卡并行
- `octa_gpu`: 8卡并行
- `hexadeca_gpu`: 16卡并行

### 数据集选项
- `burstgpt_full`: 完整BurstGPT数据集（1.4M+请求）
- `burstgpt_sample`: BurstGPT采样数据集（10K请求）
- `sharegpt`: ShareGPT基准数据集（100请求）

### 实验参数
- `--hardware`: 硬件类型（默认: RTX3090）
- `--npu_num`: GPU数量
- `--npu_group`: GPU组配置
- `--dataset`: 数据集路径
- `--output`: 输出文件路径
- `--scheduler`: 调度器类型（可选）

## 输出结果

实验完成后，在 `output/multi_gpu_results/` 目录下会生成：

1. **CSV文件**: 每个实验配置的详细结果
2. **实验报告**: Markdown格式的实验总结
3. **性能图表**: PNG格式的性能对比图
4. **JSON数据**: 完整的实验数据

### 性能指标
- 平均延迟 (ms)
- 吞吐量 (tokens/s)
- 执行时间 (秒)
- 请求处理数量
- GPU利用效率

## 使用示例

### 自定义实验配置
```python
# 在 multi_gpu_simulation.py 中修改
datasets = ["burstgpt_sample"]  # 指定数据集
gpus = ["single_gpu", "dual_gpu"]  # 指定GPU配置
```

### 单独运行特定配置
```bash
# 运行双GPU BurstGPT实验
python main.py \
    --hardware RTX3090 \
    --npu_num 2 \
    --npu_group 1 \
    --dataset dataset/BurstGPT_1_cleaned.csv \
    --output output/custom_experiment.csv
```

## 性能分析

### 自动生成的分析内容
1. **延迟分析**: 随GPU数量变化的延迟趋势
2. **吞吐量分析**: 并行扩展效率
3. **执行时间分析**: 不同配置的时间开销
4. **效率分析**: GPU利用效率评估

### 手动分析脚本
使用 `analysis/scripts/` 目录下的脚本进行深度分析：

```bash
# 调度器局限性分析
python analysis/scripts/analyze_scheduler_limitations.py

# 模拟结果对比
python analysis/scripts/compare_simulation_results.py
```

## 故障排除

### 常见问题
1. **超时错误**: 增加 `timeout` 参数或使用更小的数据集
2. **内存不足**: 减少数据集大小或GPU数量
3. **配置错误**: 检查 `--npu_num` 和 `--npu_group` 参数

### 调试技巧
1. 先运行快速测试验证环境
2. 使用小规模数据集进行测试
3. 检查输出目录权限
4. 查看详细错误日志

## 预期结果

### 理想性能特征
1. **线性扩展**: 吞吐量应随GPU数量线性增长
2. **延迟优化**: 增加GPU数量应降低平均延迟
3. **高效率**: GPU利用效率应保持在较高水平
4. **稳定性**: 不同配置下性能应相对稳定

### SmartLLMServe 验证点
1. **突发处理**: 验证系统对突发流量的处理能力
2. **预测调度**: 评估预测调度的性能提升
3. **内存管理**: 测试KV缓存管理效果
4. **负载均衡**: 验证多GPU间的负载分配

## 扩展实验

### 添加新配置
在 `MultiGPUSimulator` 类中修改 `get_gpu_configs()` 方法。

### 自定义数据集
在 `get_dataset_configs()` 方法中添加新的数据集配置。

### 性能优化
根据实验结果调整调度器参数和内存管理策略。

## 参考资料

- [LLMServingSim 文档](./docs/)
- [SmartLLMServe 项目说明](./README.md)
- [分析脚本说明](./analysis/scripts_info.py)

---

**注意**: 大规模实验可能需要较长时间运行，建议在空闲时段执行。