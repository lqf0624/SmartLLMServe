#!/usr/bin/env python3
"""
多卡环境LLM推理模拟脚本

支持多种GPU配置和数据集的并行模拟实验
适用于大规模推理环境性能测试和优化分析

作者: SmartLLMServe Team
日期: 2025-09-28
"""

import subprocess
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class MultiGPUSimulator:
    """多卡模拟器类"""

    def __init__(self, base_config: Dict = None):
        self.base_config = base_config or {
            "base_hardware": "RTX3090",
            "base_dataset": "dataset/BurstGPT_1.csv",
            "output_dir": "output/multi_gpu_results",
            "timeout": 600,  # 10分钟超时
            "verbose": True
        }
        self.results = {}
        self.ensure_output_dir()

    def ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.base_config["output_dir"], exist_ok=True)

    def get_gpu_configs(self) -> List[Dict]:
        """获取GPU配置列表"""
        return [
            {
                "npu_num": 1,
                "npu_group": 1,
                "name": "single_gpu",
                "description": "单GPU基准测试"
            },
            {
                "npu_num": 2,
                "npu_group": 1,
                "name": "dual_gpu",
                "description": "双GPU并行"
            },
            {
                "npu_num": 4,
                "npu_group": 1,
                "name": "quad_gpu",
                "description": "四GPU并行"
            },
            {
                "npu_num": 8,
                "npu_group": 1,
                "name": "octa_gpu",
                "description": "八GPU并行"
            },
            {
                "npu_num": 16,
                "npu_group": 1,
                "name": "hexadeca_gpu",
                "description": "十六GPU并行"
            }
        ]

    def get_dataset_configs(self) -> List[Dict]:
        """获取数据集配置列表"""
        return [
            {
                "path": "dataset/BurstGPT_1_cleaned.csv",
                "name": "burstgpt_full",
                "description": "完整BurstGPT数据集",
                "sample_size": None
            },
            {
                "path": "dataset/BurstGPT_1_cleaned.csv",
                "name": "burstgpt_sample",
                "description": "BurstGPT采样数据集",
                "sample_size": 10000
            },
            {
                "path": "dataset/BurstGPT_1.csv",
                "name": "burstgpt_full",
                "description": "BurstGPT完整数据集（真实生产环境）",
                "sample_size": None
            }
        ]

    def create_sample_dataset(self, original_path: str, sample_size: int, output_path: str) -> bool:
        """创建采样数据集"""
        try:
            df = pd.read_csv(original_path)
            if len(df) > sample_size:
                sampled_df = df.sample(n=sample_size, random_state=42)
                sampled_df.to_csv(output_path, index=False)
                print(f"✅ 创建采样数据集: {output_path} ({sample_size} 个请求)")
                return True
            else:
                print(f"ℹ️ 数据集已小于采样大小，直接使用原文件")
                return False
        except Exception as e:
            print(f"❌ 创建采样数据集失败: {e}")
            return False

    def run_single_simulation(self, config: Dict) -> Dict:
        """运行单个模拟实验"""
        experiment_name = f"{config['dataset_name']}_{config['gpu_name']}"
        output_file = f"{self.base_config['output_dir']}/{experiment_name}.csv"

        print(f"\n🚀 开始实验: {experiment_name}")
        print(f"   GPU配置: {config['gpu_num']}卡 ({config['gpu_description']})")
        print(f"   数据集: {config['dataset_description']}")

        # 准备数据集路径
        dataset_path = config["dataset_path"]
        if config["sample_size"] and config["sample_size"] > 0:
            sample_path = f"temp_sample_{config['dataset_name']}.csv"
            if self.create_sample_dataset(dataset_path, config["sample_size"], sample_path):
                dataset_path = sample_path

        # 构建命令
        cmd = [
            "python", "main.py",
            "--hardware", self.base_config["base_hardware"],
            "--npu_num", str(config["gpu_num"]),
            "--npu_group", str(config["gpu_group"]),
            "--dataset", dataset_path,
            "--output", output_file
        ]

        # 添加调度器参数
        if config.get("scheduler"):
            cmd.extend(["--scheduler", config["scheduler"]])

        print(f"   命令: {' '.join(cmd)}")

        # 记录开始时间
        start_time = time.time()

        try:
            # 运行模拟
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.base_config["timeout"],
                cwd="/mnt/f/LLMServingSim"
            )

            # 记录结束时间
            end_time = time.time()
            execution_time = end_time - start_time

            # 清理临时文件
            if config["sample_size"] and config["sample_size"] > 0 and os.path.exists(sample_path):
                os.remove(sample_path)

            # 处理结果
            if result.returncode == 0:
                print(f"✅ 实验完成: {execution_time:.2f}秒")

                # 解析输出文件
                if os.path.exists(output_file):
                    stats = self.analyze_results(output_file)
                    return {
                        "status": "success",
                        "execution_time": execution_time,
                        "stats": stats,
                        "config": config,
                        "output_file": output_file
                    }
                else:
                    print(f"⚠️ 输出文件不存在: {output_file}")
                    return {
                        "status": "warning",
                        "execution_time": execution_time,
                        "error": "输出文件不存在",
                        "config": config
                    }
            else:
                print(f"❌ 实验失败: {result.stderr}")
                return {
                    "status": "failed",
                    "execution_time": execution_time,
                    "error": result.stderr,
                    "config": config
                }

        except subprocess.TimeoutExpired:
            print(f"⏰ 实验超时: {self.base_config['timeout']}秒")
            return {
                "status": "timeout",
                "execution_time": self.base_config["timeout"],
                "error": "实验超时",
                "config": config
            }
        except Exception as e:
            print(f"❌ 实验异常: {e}")
            return {
                "status": "error",
                "execution_time": time.time() - start_time,
                "error": str(e),
                "config": config
            }

    def analyze_results(self, result_file: str) -> Dict:
        """分析模拟结果"""
        try:
            df = pd.read_csv(result_file)

            # 基本统计信息
            stats = {
                "total_requests": len(df),
                "avg_latency_ms": df["latency"].mean() / 1e6,
                "avg_ttft_ms": df["TTFT"].mean() / 1e6,
                "avg_tpot_ms": df["TPOT"].mean() / 1e6,
                "avg_queue_delay_ms": df["queuing_delay"].mean() / 1e6,
                "throughput_tokens_per_sec": df["output"].sum() / (df["end_time"].max() - df["arrival"].min()) * 1e9,
                "total_execution_time_sec": (df["end_time"].max() - df["arrival"].min()) / 1e9,
                "latency_std_ms": df["latency"].std() / 1e6,
                "total_output_tokens": df["output"].sum()
            }

            return stats

        except Exception as e:
            print(f"❌ 分析结果失败: {e}")
            return {"error": str(e)}

    def run_comprehensive_experiment(self, datasets: List[str] = None, gpus: List[str] = None) -> Dict:
        """运行综合实验"""
        print("🎯 开始多卡环境综合模拟实验")
        print("=" * 60)

        # 获取配置
        gpu_configs = self.get_gpu_configs()
        dataset_configs = self.get_dataset_configs()

        # 过滤配置
        if datasets:
            dataset_configs = [d for d in dataset_configs if d["name"] in datasets]
        if gpus:
            gpu_configs = [g for g in gpu_configs if g["name"] in gpus]

        all_results = {}

        for dataset_config in dataset_configs:
            dataset_results = {}
            print(f"\n📊 数据集: {dataset_config['description']}")

            for gpu_config in gpu_configs:
                # 构建实验配置
                experiment_config = {
                    "dataset_name": dataset_config["name"],
                    "dataset_description": dataset_config["description"],
                    "dataset_path": dataset_config["path"],
                    "sample_size": dataset_config["sample_size"],
                    "gpu_name": gpu_config["name"],
                    "gpu_description": gpu_config["description"],
                    "gpu_num": gpu_config["npu_num"],
                    "gpu_group": gpu_config["npu_group"]
                }

                # 运行实验
                result = self.run_single_simulation(experiment_config)
                dataset_results[gpu_config["name"]] = result

            all_results[dataset_config["name"]] = dataset_results

        self.results = all_results
        return all_results

    def generate_summary_report(self) -> str:
        """生成实验总结报告"""
        if not self.results:
            return "没有可用的实验结果"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.base_config['output_dir']}/experiment_report_{timestamp}.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 多卡环境模拟实验报告\n\n")
            f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for dataset_name, dataset_results in self.results.items():
                f.write(f"## {dataset_name} 数据集\n\n")

                # 创建性能对比表格
                f.write("| GPU配置 | 状态 | 执行时间(秒) | 平均延迟(ms) | 吞吐量(tokens/s) | 请求数 |\n")
                f.write("|---------|------|-------------|-------------|------------------|--------|\n")

                for gpu_name, result in dataset_results.items():
                    status = "✅ 成功" if result["status"] == "success" else f"❌ {result['status']}"
                    exec_time = f"{result['execution_time']:.2f}"

                    if result["status"] == "success" and "stats" in result:
                        avg_latency = f"{result['stats']['avg_latency_ms']:.2f}"
                        throughput = f"{result['stats']['throughput_tokens_per_sec']:.2f}"
                        requests = str(result['stats']['total_requests'])
                    else:
                        avg_latency = "N/A"
                        throughput = "N/A"
                        requests = "N/A"

                    f.write(f"| {gpu_name} | {status} | {exec_time} | {avg_latency} | {throughput} | {requests} |\n")

                f.write("\n")

            # 总结和建议
            f.write("## 实验总结\n\n")
            successful_experiments = sum(1 for d in self.results.values() for r in d.values() if r["status"] == "success")
            total_experiments = sum(len(d) for d in self.results.values())
            f.write(f"- **实验完成率**: {successful_experiments}/{total_experiments} ({successful_experiments/total_experiments*100:.1f}%)\n")

            if successful_experiments > 0:
                f.write("- **成功的实验配置**: \n")
                for dataset_name, dataset_results in self.results.items():
                    for gpu_name, result in dataset_results.items():
                        if result["status"] == "success":
                            f.write(f"  - {dataset_name} + {gpu_name}\n")

            f.write("\n## 建议的后续实验\n")
            f.write("1. **扩展GPU配置**: 测试更多的GPU并行模式\n")
            f.write("2. **调度器优化**: 对比不同调度策略的性能\n")
            f.write("3. **内存管理**: 分析不同GPU配置下的内存使用情况\n")
            f.write("4. **真实数据**: 使用更真实的大规模数据集进行验证\n")

        print(f"📄 实验报告已保存: {report_path}")
        return report_path

    def create_performance_charts(self):
        """创建性能对比图表"""
        if not self.results:
            print("❌ 没有可用的实验结果")
            return

        # 准备数据
        data = []
        for dataset_name, dataset_results in self.results.items():
            for gpu_name, result in dataset_results.items():
                if result["status"] == "success" and "stats" in result:
                    data.append({
                        "dataset": dataset_name,
                        "gpu_config": gpu_name,
                        "gpu_num": int(gpu_name.split('_')[0]) if '_' in gpu_name else 1,
                        "avg_latency_ms": result["stats"]["avg_latency_ms"],
                        "throughput": result["stats"]["throughput_tokens_per_sec"],
                        "execution_time": result["execution_time"]
                    })

        if not data:
            print("❌ 没有成功的实验数据")
            return

        df = pd.DataFrame(data)

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('多卡环境性能对比分析', fontsize=16, fontweight='bold')

        # 1. 延迟 vs GPU数量
        ax1 = axes[0, 0]
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            ax1.plot(dataset_df['gpu_num'], dataset_df['avg_latency_ms'],
                    marker='o', label=dataset, linewidth=2)
        ax1.set_xlabel('GPU数量')
        ax1.set_ylabel('平均延迟 (ms)')
        ax1.set_title('延迟随GPU数量变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 吞吐量 vs GPU数量
        ax2 = axes[0, 1]
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            ax2.plot(dataset_df['gpu_num'], dataset_df['throughput'],
                    marker='s', label=dataset, linewidth=2)
        ax2.set_xlabel('GPU数量')
        ax2.set_ylabel('吞吐量 (tokens/s)')
        ax2.set_title('吞吐量随GPU数量变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 执行时间 vs GPU数量
        ax3 = axes[1, 0]
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            ax3.plot(dataset_df['gpu_num'], dataset_df['execution_time'],
                    marker='^', label=dataset, linewidth=2)
        ax3.set_xlabel('GPU数量')
        ax3.set_ylabel('执行时间 (秒)')
        ax3.set_title('执行时间随GPU数量变化')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 性能效率分析
        ax4 = axes[1, 1]
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            # 计算效率指标 (吞吐量/GPU数量)
            efficiency = dataset_df['throughput'] / dataset_df['gpu_num']
            ax4.plot(dataset_df['gpu_num'], efficiency,
                    marker='d', label=dataset, linewidth=2)
        ax4.set_xlabel('GPU数量')
        ax4.set_ylabel('每GPU吞吐量 (tokens/s)')
        ax4.set_title('GPU利用效率')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        chart_path = f"{self.base_config['output_dir']}/performance_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"📊 性能对比图表已保存: {chart_path}")
        plt.close()

        return chart_path

    def save_results_json(self):
        """保存结果为JSON格式"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"{self.base_config['output_dir']}/experiment_results_{timestamp}.json"

        # 准备JSON数据
        json_data = {
            "experiment_time": datetime.now().isoformat(),
            "base_config": self.base_config,
            "results": self.results
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"💾 实验结果已保存: {json_path}")
        return json_path

def main():
    """主函数"""
    print("🎯 多卡环境LLM推理模拟器")
    print("=" * 60)

    # 创建模拟器实例
    simulator = MultiGPUSimulator()

    # 运行实验（可以选择特定的数据集和GPU配置）
    datasets = ["burstgpt_sample"]  # 使用采样数据集快速测试
    gpus = ["single_gpu", "dual_gpu", "quad_gpu"]  # 测试1-4卡配置

    # 运行综合实验
    results = simulator.run_comprehensive_experiment(datasets=datasets, gpus=gpus)

    # 生成报告和图表
    if results:
        report_path = simulator.generate_summary_report()
        chart_path = simulator.create_performance_charts()
        json_path = simulator.save_results_json()

        print(f"\n🎉 实验完成！")
        print(f"📄 报告: {report_path}")
        print(f"📊 图表: {chart_path}")
        print(f"💾 数据: {json_path}")
    else:
        print("❌ 实验失败，没有生成结果")

if __name__ == "__main__":
    main()