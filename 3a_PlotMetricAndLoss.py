"""
Training History Visualization for Custom Metric Models

This script creates publication-quality plots showing the training progression
of neural network models for the Lorenz system. It displays both the Mean
Squared Error (MSE) loss and the prediction horizon metric across training epochs.

The visualization uses a dual-panel layout with:
- Top panel: Log-scale MSE loss over epochs
- Bottom panel: Linear-scale prediction horizon over epochs

A vertical line indicates the epoch where maximum prediction horizon was achieved,
with the corresponding MSE value displayed.

Example arguments:
-m A --width 10 --height 12 --dpi 600

Author: Anthony S. Miller, University of Essex
Date: 17/10/2025
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import glob


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        dict: Dictionary containing parsed arguments
    """
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Visualize training history for custom metric models"
    )
    parser.add_argument(
        "-m", "--model_label",
        default="A",
        type=str,
        help="Model label (e.g., 'A', 'B', 'C')"
    )
    parser.add_argument(
        "-d", "--data_directory",
        default="./_CustomMetricFiles/",
        type=str,
        help="Directory containing training history files"
    )
    parser.add_argument(
        "-e", "--epochs",
        default=None,
        type=int,
        help="Number of epochs (if None, will auto-detect from files)"
    )
    parser.add_argument(
        "-b", "--batch_size",
        default=None,
        type=int,
        help="Batch size used (if None, will auto-detect from files)"
    )
    parser.add_argument(
        "--width",
        default=8,
        type=float,
        help="Figure width in cm (default: 8)"
    )
    parser.add_argument(
        "--height",
        default=8,
        type=float,
        help="Figure height in cm (default: 8)"
    )
    parser.add_argument(
        "--dpi",
        default=300,
        type=int,
        help="DPI for saved figure (default: 300)"
    )
    return vars(parser.parse_args())


def cm_to_inch(cm):
    """
    Convert centimeters to inches for matplotlib figure sizing.

    Parameters:
        cm (float): Size in centimeters

    Returns:
        float: Size in inches
    """
    return cm * 0.3937


def find_training_files(data_directory, model_label):
    """
    Find training history files for a specific model.

    Parameters:
        data_directory (str): Directory containing training files
        model_label (str): Model label (e.g., 'A', 'B')

    Returns:
        tuple: (loss_file, metric_file) paths, or (None, None) if not found
    """
    model_dir = os.path.join(data_directory, f'Model{model_label}')

    if not os.path.exists(model_dir):
        print(f'Error: Model directory not found: {model_dir}')
        return None, None

    # Search for loss and metric files
    loss_files = glob.glob(os.path.join(model_dir, '*loss*.csv'))
    metric_files = glob.glob(os.path.join(model_dir, '*prediction_horizon*.csv'))

    if not loss_files:
        print(f'Error: No loss file found in {model_dir}')
        return None, None

    if not metric_files:
        print(f'Error: No prediction horizon file found in {model_dir}')
        return None, None

    # Use the first file found (should only be one)
    loss_file = loss_files[0]
    metric_file = metric_files[0]

    print(f'Found loss file: {os.path.basename(loss_file)}')
    print(f'Found metric file: {os.path.basename(metric_file)}')

    return loss_file, metric_file


def load_training_data(loss_file, metric_file):
    """
    Load training history data from CSV files.

    Parameters:
        loss_file (str): Path to loss history file
        metric_file (str): Path to metric history file

    Returns:
        tuple: (loss_df, metric_df) DataFrames with epoch and value columns
    """
    # Load data (files have headers: epoch,metric_name)
    loss_df = pd.read_csv(loss_file)
    metric_df = pd.read_csv(metric_file)

    # Rename columns for clarity
    loss_df.columns = ['epoch', 'loss']
    metric_df.columns = ['epoch', 'prediction_horizon']

    print(f'\nLoaded {len(loss_df)} epochs of training data')
    print(f'Loss range: {loss_df["loss"].min():.6e} - {loss_df["loss"].max():.6e}')
    print(
        f'Prediction horizon range: {metric_df["prediction_horizon"].min():.4f} - {metric_df["prediction_horizon"].max():.4f}')

    return loss_df, metric_df


def find_optimal_epoch(loss_df, metric_df):
    """
    Find the epoch where maximum prediction horizon was achieved.

    Parameters:
        loss_df (pd.DataFrame): Loss history dataframe
        metric_df (pd.DataFrame): Metric history dataframe

    Returns:
        tuple: (max_ph, optimal_epoch, mse_at_optimal) where:
            - max_ph: Maximum prediction horizon achieved
            - optimal_epoch: Epoch where maximum was achieved
            - mse_at_optimal: MSE loss at that epoch
    """
    # Find maximum prediction horizon
    max_ph = metric_df['prediction_horizon'].max()
    optimal_epoch = metric_df['prediction_horizon'].idxmax()

    # Get corresponding MSE at that epoch
    mse_at_optimal = loss_df.loc[optimal_epoch, 'loss']

    print(f'\nOptimal training point:')
    print(f'  Epoch: {optimal_epoch}')
    print(f'  Max prediction horizon: {max_ph:.4f} Lyapunov times')
    print(f'  MSE at optimal epoch: {mse_at_optimal:.6e}')

    return max_ph, optimal_epoch, mse_at_optimal


def create_dual_panel_plot(loss_df, metric_df, max_ph, optimal_epoch,
                           mse_at_optimal, model_label, fig_width_cm=8,
                           fig_height_cm=8):
    """
    Create a dual-panel plot showing loss and prediction horizon over epochs.

    Parameters:
        loss_df (pd.DataFrame): Loss history dataframe
        metric_df (pd.DataFrame): Metric history dataframe
        max_ph (float): Maximum prediction horizon
        optimal_epoch (int): Epoch where maximum was achieved
        mse_at_optimal (float): MSE at optimal epoch
        model_label (str): Model identifier
        fig_width_cm (float): Figure width in cm
        fig_height_cm (float): Figure height in cm

    Returns:
        tuple: (fig, (ax1, ax2)) - Figure and axes objects
    """
    # Create figure with shared x-axis
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(cm_to_inch(fig_width_cm), cm_to_inch(fig_height_cm))
    )

    # Get standard tick font size for custom labels
    tick_size = ax2.xaxis.get_ticklabels()[0].get_fontsize()

    # Top panel: Loss (log scale)
    ax1.set_yscale('log')
    loss_label = f'MSE={mse_at_optimal:.2e}'
    ax1.plot(loss_df['epoch'], loss_df['loss'], color='orange',
             linewidth=1.5, label=loss_label)
    ax1.axvline(x=optimal_epoch, color='green', linestyle=':',
                linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Log(MSE)', fontsize=10)
    ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Bottom panel: Prediction Horizon (linear scale)
    metric_label = f'PH={max_ph:.2f}'
    ax2.plot(metric_df['epoch'], metric_df['prediction_horizon'],
             color='blue', linewidth=1.5, label=metric_label)
    ax2.axvline(x=optimal_epoch, color='green', linestyle=':',
                linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Prediction Horizon\n(Lyapunov times)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add custom tick mark and label for optimal epoch
    if optimal_epoch > 0:
        label_offset = 0.30  # Position below axis
        extra_line = 0.10  # Extra line length below label

        # Draw extended tick line
        ax2.plot(
            [optimal_epoch, optimal_epoch],
            [0, -label_offset + extra_line],
            transform=ax2.get_xaxis_transform(),
            color="black",
            linewidth=1.0,
            clip_on=False,
            zorder=5
        )

        # Add label
        ax2.text(
            optimal_epoch, -label_offset, str(optimal_epoch),
            ha="center", va="center",
            fontsize=tick_size,
            transform=ax2.get_xaxis_transform(),
            zorder=6
        )

    # Add overall title
    fig.suptitle(f'Model {model_label} Training History',
                 fontsize=11, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    return fig, (ax1, ax2)


def save_figure(fig, output_path, dpi=300):
    """
    Save figure to file.

    Parameters:
        fig: Matplotlib figure object
        output_path (str): Path where figure should be saved
        dpi (int): Resolution in dots per inch
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f'\nFigure saved to: {output_path}')


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    model_label = args["model_label"]
    data_directory = args["data_directory"]
    fig_width_cm = args["width"]
    fig_height_cm = args["height"]
    dpi = args["dpi"]

    print(f'\n{"=" * 80}')
    print(f'Training History Visualization - Model {model_label}')
    print(f'{"=" * 80}\n')

    # Find training files
    loss_file, metric_file = find_training_files(data_directory, model_label)

    if loss_file is None or metric_file is None:
        print('\nError: Could not find required training files')
        return

    # Load training data
    loss_df, metric_df = load_training_data(loss_file, metric_file)

    # Find optimal epoch
    max_ph, optimal_epoch, mse_at_optimal = find_optimal_epoch(loss_df, metric_df)

    # Create plot
    print(f'\nCreating visualization...')
    fig, axes = create_dual_panel_plot(
        loss_df, metric_df, max_ph, optimal_epoch, mse_at_optimal,
        model_label, fig_width_cm, fig_height_cm
    )

    # Save figure
    model_dir = os.path.join(data_directory, f'Model{model_label}')
    plotting_dir = os.path.join(model_dir, 'Plotting')

    # Extract epochs and batch size from filename for output name
    loss_filename = os.path.basename(loss_file)
    # Expected format: Model{X}_loss_epochs{Y}_batch{Z}.csv
    parts = loss_filename.replace('.csv', '').split('_')
    epochs_str = next((p for p in parts if p.startswith('epochs')), 'epochs500')
    batch_str = next((p for p in parts if p.startswith('batch')), 'batch2000')

    output_filename = f'TrainingHistory_Model{model_label}_{epochs_str}_{batch_str}.png'
    output_path = os.path.join(plotting_dir, output_filename)

    save_figure(fig, output_path, dpi)

    # Show plot
    plt.show()

    print(f'\n{"=" * 80}')
    print('Visualization Complete')
    print(f'{"=" * 80}')


if __name__ == "__main__":
    main()