#!/usr/bin/env python3
"""
多卡环境快速测试脚本

用于快速验证多卡模拟功能，使用小规模数据集进行测试
"""

import subprocess
import os
import time
from pathlib import Path

def run_quick_test():
    """运行快速测试"""
    print("🚀 多卡环境快速测试")
    print("=" * 50)

    # 测试配置
    test_configs = [
        {
            "name": "single_gpu_test",
            "npu_num": 1,
            "npu_group": 1,
            "description": "单GPU测试"
        },
        {
            "name": "dual_gpu_test",
            "npu_num": 2,
            "npu_group": 1,
            "description": "双GPU测试"
        },
        {
            "name": "quad_gpu_test",
            "npu_num": 4,
            "npu_group": 1,
            "description": "四GPU测试"
        }
    ]

    # 使用小规模数据集进行测试
    dataset_path = "dataset/BurstGPT_1.csv"  # 使用真实的BurstGPT数据集

    results = []

    for config in test_configs:
        print(f"\n🔍 测试配置: {config['description']}")

        output_file = f"output/quick_test_{config['name']}.csv"

        # 构建命令
        cmd = [
            "python", "main.py",
            "--hardware", "RTX3090",
            "--npu_num", str(config["npu_num"]),
            "--npu_group", str(config["npu_group"]),
            "--dataset", dataset_path,
            "--output", output_file
        ]

        print(f"   命令: {' '.join(cmd)}")

        start_time = time.time()

        try:
            # 运行测试
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2分钟超时
                cwd="/mnt/f/LLMServingSim"
            )

            end_time = time.time()
            execution_time = end_time - start_time

            if result.returncode == 0:
                print(f"✅ 测试成功: {execution_time:.2f}秒")

                # 检查输出文件
                if os.path.exists(output_file):
                    # 简单分析结果
                    with open(output_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # 有数据行
                            print(f"   生成了 {len(lines)-1} 个请求的结果")
                        else:
                            print(f"   ⚠️ 输出文件为空")

                results.append({
                    "config": config,
                    "status": "success",
                    "execution_time": execution_time,
                    "output_file": output_file
                })

            else:
                print(f"❌ 测试失败: {result.stderr}")
                results.append({
                    "config": config,
                    "status": "failed",
                    "execution_time": execution_time,
                    "error": result.stderr
                })

        except subprocess.TimeoutExpired:
            print(f"⏰ 测试超时: 120秒")
            results.append({
                "config": config,
                "status": "timeout",
                "execution_time": 120,
                "error": "测试超时"
            })

        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append({
                "config": config,
                "status": "error",
                "execution_time": time.time() - start_time,
                "error": str(e)
            })

    # 汇总结果
    print(f"\n📊 快速测试结果汇总")
    print("=" * 50)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    print(f"✅ 成功: {len(successful)}/{len(results)}")
    print(f"❌ 失败: {len(failed)}/{len(results)}")

    for result in results:
        status_symbol = "✅" if result["status"] == "success" else "❌"
        print(f"{status_symbol} {result['config']['description']}: {result['execution_time']:.2f}秒")

    if successful:
        print(f"\n🎉 多卡环境测试通过！可以进行大规模实验")
        print("💡 使用完整脚本进行详细实验:")
        print("   python multi_gpu_simulation.py")
    else:
        print(f"\n⚠️ 多卡环境测试失败，需要检查环境配置")

if __name__ == "__main__":
    run_quick_test()