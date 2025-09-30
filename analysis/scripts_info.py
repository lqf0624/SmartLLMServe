#!/usr/bin/env python3
"""
Analysis Scripts README
整理后的分析脚本目录

脚本功能说明:
1. analyze_scheduler_limitations.py - LLM调度器局限性分析
2. compare_simulation_results.py - 模拟结果对比分析
3. corrected_analysis.py - 修正后的分析脚本（正确时间单位）
4. create_english_charts.py - 英文版本图表生成
5. demo_limitations.py - 调度器局限性演示
6. final_summary.py - 最终分析总结
7. multi_gpu_experiment.py - 多GPU实验脚本

使用方法:
python scripts/脚本名称.py
"""

import os
from pathlib import Path

def list_analysis_scripts():
    """列出所有分析脚本"""
    scripts_dir = Path(__file__).parent / "scripts"

    print("📊 Available Analysis Scripts:")
    print("=" * 50)

    for script in scripts_dir.glob("*.py"):
        print(f"• {script.name}")

    print("\n📋 Analysis Directory Contents:")
    print("=" * 50)
    for item in Path(__file__).parent.iterdir():
        if item.is_file():
            print(f"• {item.name}")
        elif item.is_dir():
            print(f"📁 {item.name}/")

if __name__ == "__main__":
    list_analysis_scripts()