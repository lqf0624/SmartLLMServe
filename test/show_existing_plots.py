#!/usr/bin/env python3
"""
Script to display the existing visualization plots.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_plots():
    """Display the existing visualization plots."""
    viz_dir = "predictor/test_visualization"

    if not os.path.exists(viz_dir):
        print("Visualization directory not found!")
        return

    # List of plot files
    plot_files = [
        "01_prediction_overview.png",
        "02_time_series_comparison.png",
        "03_resource_forecast.png",
        "04_burst_patterns.png"
    ]

    # Create a figure to display the plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, plot_file in enumerate(plot_files):
        plot_path = os.path.join(viz_dir, plot_file)
        if os.path.exists(plot_path):
            try:
                # Load and display the image
                img = mpimg.imread(plot_path)
                axes[i].imshow(img)
                axes[i].set_title(plot_file.replace('.png', '').replace('_', ' ').title())
                axes[i].axis('off')
                print(f"Loaded: {plot_file}")
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading {plot_file}\n{str(e)}",
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(plot_file.replace('.png', '').replace('_', ' ').title() + " (Error)")
        else:
            axes[i].text(0.5, 0.5, "File not found",
                       transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(plot_file.replace('.png', '').replace('_', ' ').title() + " (Missing)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Showing existing visualization plots...")
    print("These plots demonstrate the enhanced visualization system:")
    print("1. Prediction Overview - Shows predicted vs actual values")
    print("2. Time Series Comparison - Temporal pattern analysis")
    print("3. Resource Forecast - Memory and compute requirements")
    print("4. Burst Patterns - Workload burst analysis")
    print()
    show_plots()