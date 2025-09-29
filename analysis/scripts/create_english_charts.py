#!/usr/bin/env python3
"""
Generate English version charts to avoid font issues
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_english_charts():
    """Create English version charts with CORRECTED units"""
    # Load data
    full_df = pd.read_csv('output/baseline_tsv_full.csv')

    # Convert nanoseconds to milliseconds for display
    full_df['latency_ms'] = full_df['latency'] / 1e6
    full_df['ttft_ms'] = full_df['TTFT'] / 1e6
    full_df['tpot_ms'] = full_df['TPOT'] / 1e6
    full_df['queue_delay_ms'] = full_df['queuing_delay'] / 1e6

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LLM Scheduler Performance Analysis (Corrected)', fontsize=16, fontweight='bold')

    # 1. Latency Distribution
    ax1 = axes[0, 0]
    ax1.hist(full_df['latency_ms'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(full_df['latency_ms'].mean(), color='red', linestyle='--', label=f'Avg: {full_df["latency_ms"].mean():.1f}ms')
    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('Request Count')
    ax1.set_title('Request Latency Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. TTFT vs TPOT
    ax2 = axes[0, 1]
    scatter = ax2.scatter(full_df['ttft_ms'], full_df['tpot_ms'], c=full_df['input'], cmap='viridis', alpha=0.7)
    ax2.set_xlabel('TTFT (ms)')
    ax2.set_ylabel('TPOT (ms)')
    ax2.set_title('TTFT vs TPOT (colored by input length)')
    plt.colorbar(scatter, ax=ax2, label='Input Length')
    ax2.grid(True, alpha=0.3)

    # 3. Queue Delay Over Time
    ax3 = axes[1, 0]
    ax3.plot(range(len(full_df)), full_df['queue_delay_ms'], marker='o', markersize=3, alpha=0.7)
    ax3.set_xlabel('Request ID')
    ax3.set_ylabel('Queue Delay (ms)')
    ax3.set_title('Queue Delay Over Requests')
    ax3.grid(True, alpha=0.3)

    # 4. Processing Efficiency
    ax4 = axes[1, 1]
    theoretical_time = full_df['output'] * 0.1  # Assume 0.1ms per token
    actual_time = full_df['latency_ms']
    efficiency = (theoretical_time / actual_time) * 100

    ax4.plot(range(len(full_df)), efficiency, marker='o', markersize=3, alpha=0.7, color='green')
    ax4.set_xlabel('Request ID')
    ax4.set_ylabel('Processing Efficiency (%)')
    ax4.set_title('Processing Efficiency Analysis')
    ax4.axhline(efficiency.mean(), color='red', linestyle='--', label=f'Avg: {efficiency.mean():.1f}%')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/current_system_limitations_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create CORRECTED summary chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Calculate CORRECTED key metrics
    total_time_sec = (full_df['end_time'].max() - full_df['arrival'].min()) / 1e9
    metrics = {
        'Avg Latency (ms)': full_df['latency_ms'].mean(),
        'Avg TTFT (ms)': full_df['ttft_ms'].mean(),
        'Avg TPOT (ms)': full_df['tpot_ms'].mean(),
        'Throughput (tok/s)': full_df['output'].sum() / total_time_sec,
        'Latency Std (ms)': full_df['latency_ms'].std(),
        'Queue Delay (ms)': full_df['queue_delay_ms'].mean()
    }

    categories = list(metrics.keys())
    values = list(metrics.values())
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('Corrected System Performance Metrics')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('output/corrected_metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ CORRECTED English charts created:")
    print("  - output/current_system_limitations_corrected.png")
    print("  - output/corrected_metrics_summary.png")

if __name__ == "__main__":
    create_english_charts()