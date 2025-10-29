"""
Prediction Horizon Statistical Analysis and Visualization

This script performs comprehensive statistical analysis on prediction horizons
for neural network models of the Lorenz system. It computes key statistics,
identifies top-performing initial conditions, and generates publication-quality
visualizations including probability density functions with multiple distributional fits.

The prediction horizon is defined as the time until the relative error between
predicted and ground truth trajectories exceeds 0.4, measured in Lyapunov time units.

Key features:
- Automated file matching for prediction horizon data
- Statistical analysis (mean, variance, standard deviation)
- Top-performing initial condition identification
- Mean and standard deviation visualization
- Probability density function estimation with multiple distributional fits
- Optimal binning using Freedman-Diaconis, Scott, or Sturges rules

Author: Anthony S. Miller, University of Essex
Date: 17/10/2025
"""

import json
import numpy as np
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import datetime
import seaborn as sns
from scipy import stats
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings

warnings.simplefilter("ignore")


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        dict: Dictionary containing parsed arguments
    """
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Analyze prediction horizons and generate statistical visualizations"
    )
    parser.add_argument(
        "-m", "--model_name",
        default="ModelA",
        type=str,
        help="Model identifier"
    )
    parser.add_argument(
        "-d", "--prediction_directory",
        default="./_PredictionOutput/",
        type=str,
        help="Base directory containing prediction results"
    )
    parser.add_argument(
        "-ic", "--initial_conditions",
        default="IC_x0_1000_LP_P.json",
        type=str,
        help="JSON file containing initial conditions"
    )
    parser.add_argument(
        "--perturbation",
        default=1e-5,
        type=float,
        help="Perturbation magnitude applied to initial conditions (default: 1e-5)"
    )
    parser.add_argument(
        "--error_threshold",
        default=0.4,
        type=float,
        help="Relative error threshold for prediction horizon (default: 0.4)"
    )
    parser.add_argument(
        "--bin_method",
        default="fd",
        choices=["fd", "scott", "sturges"],
        help="Binning method for histograms (default: fd - Freedman-Diaconis)"
    )
    parser.add_argument(
        "--top_n",
        default=10,
        type=int,
        help="Number of top prediction horizons to save (default: 10)"
    )
    return vars(parser.parse_args())


# Configure plot aesthetics
rcParams.update({'font.size': 14})
plt.rcParams['figure.figsize'] = [12, 12]


# ===========================
# File Matching Utilities
# ===========================

def extract_triplet_from_filename(filename):
    """
    Extract initial condition triplet from filename.

    Expected format: RelativeError_IC####_ModelX_YYYYMMDD.csv
    or similar with IC values embedded.

    Parameters:
        filename (str): Filename to parse

    Returns:
        list or None: [x, y, z] values if found, None otherwise
    """
    # Match pattern like [x_y_z]
    match = re.search(r"\[([0-9eE\._\-]+)\]", filename)
    if not match:
        return None

    parts = match.group(1).split('_')
    if len(parts) != 3:
        return None

    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def triplet_match(triplet1, triplet2, tolerance=1e-7):
    """
    Check if two triplets match within tolerance.

    Parameters:
        triplet1 (list): First triplet [x, y, z]
        triplet2 (list): Second triplet [x, y, z]
        tolerance (float): Maximum allowed difference per component

    Returns:
        bool: True if all components match within tolerance
    """
    if len(triplet1) != 3 or len(triplet2) != 3:
        return False

    return all(abs(a - b) < tolerance for a, b in zip(triplet1, triplet2))


def find_matching_file(filenames, base_directory, target_triplet, tolerance=1e-7):
    """
    Find file matching a specific initial condition triplet.

    Parameters:
        filenames (list): List of filenames to search
        base_directory (str): Base directory for constructing full paths
        target_triplet (list): Target initial condition [x, y, z]
        tolerance (float): Matching tolerance

    Returns:
        str or None: Full path to matching file, or None if not found
    """
    for filename in filenames:
        triplet_in_file = extract_triplet_from_filename(filename)

        if triplet_in_file and triplet_match(target_triplet, triplet_in_file, tolerance):
            return os.path.join(base_directory, filename)

    return None


# ===========================
# Data Loading and Processing
# ===========================

def load_initial_conditions(filename):
    """
    Load initial conditions from JSON file.

    Parameters:
        filename (str): Path to JSON file

    Returns:
        list: List of initial condition triplets
    """
    print(f'Loading initial conditions from: {filename}')

    with open(filename, 'r') as f:
        data = json.load(f)

    # Handle different possible key names
    if 'perturbedICs' in data:
        initial_conditions = data['perturbedICs']
    elif 'ICs' in data:
        initial_conditions = data['ICs']
    else:
        raise KeyError(f"Could not find 'perturbedICs' or 'ICs' key in {filename}")

    print(f'Loaded {len(initial_conditions)} initial conditions')

    return initial_conditions


def apply_perturbation(initial_condition, perturbation):
    """
    Apply uniform perturbation to all components of initial condition.

    Parameters:
        initial_condition (list): Original IC [x, y, z]
        perturbation (float): Perturbation magnitude

    Returns:
        list: Perturbed IC [x+ε, y+ε, z+ε]
    """
    return [component + perturbation for component in initial_condition]


def find_prediction_horizon(dataframe, error_threshold=0.4):
    """
    Find the prediction horizon where error first exceeds threshold.

    Parameters:
        dataframe (pd.DataFrame): DataFrame with columns [Time, PredictionHorizon, Error, IC, Model]
        error_threshold (float): Error threshold (default: 0.4)

    Returns:
        tuple: (index, row) where threshold is exceeded, or (None, None) if never exceeded
    """
    for idx, row in dataframe.iterrows():
        if row.Error >= error_threshold - 1e-10:  # Small tolerance for floating point comparison
            return idx, row

    return None, None


def extract_prediction_horizons(initial_conditions, horizon_dir, perturbation,
                                error_threshold=0.4):
    """
    Extract prediction horizons for all initial conditions.

    Parameters:
        initial_conditions (list): List of initial condition triplets
        horizon_dir (str): Directory containing prediction horizon CSV files
        perturbation (float): Perturbation magnitude
        error_threshold (float): Error threshold for prediction horizon

    Returns:
        list: List of (prediction_horizon, perturbed_ic) tuples
    """
    print(f'\n{"=" * 80}')
    print('Extracting Prediction Horizons')
    print(f'{"=" * 80}\n')

    csv_files = [f for f in os.listdir(horizon_dir) if f.endswith('.csv')]
    print(f'Found {len(csv_files)} CSV files in horizon directory')

    horizon_data = []

    for ic_idx, original_ic in enumerate(initial_conditions):
        # Apply perturbation
        perturbed_ic = apply_perturbation(original_ic, perturbation)

        # Find matching file
        match_path = find_matching_file(csv_files, horizon_dir, perturbed_ic)

        if not match_path:
            print(f'ALERT: IC {ic_idx:04d}: No matching file found')
            continue

        # Load CSV
        try:
            df = pd.read_csv(match_path)

            # Ensure required columns exist
            required_cols = ['Time', 'PredictionHorizon', 'Error', 'IC', 'Model']
            if not all(col in df.columns for col in required_cols):
                print(f'ALERT: IC {ic_idx:04d}: Missing required columns in {os.path.basename(match_path)}')
                continue

        except Exception as e:
            print(f'ALERT: IC {ic_idx:04d}: Could not read file: {e}')
            continue

        # Find prediction horizon
        index, row = find_prediction_horizon(df, error_threshold)

        if row is None:
            print(f'INFO: IC {ic_idx:04d}: Error never exceeded {error_threshold}')
            continue

        print(f'    IC {ic_idx:04d}: PH = {row.PredictionHorizon:.4f}, Error = {row.Error:.4f}')
        horizon_data.append((row.PredictionHorizon, perturbed_ic))

    print(f'\n{"=" * 80}')
    print(f'Successfully extracted {len(horizon_data)} prediction horizons')
    print(f'{"=" * 80}\n')

    return horizon_data


# ===========================
# Statistical Analysis
# ===========================

def compute_statistics(horizon_data):
    """
    Compute statistical measures for prediction horizons.

    Parameters:
        horizon_data (list): List of (prediction_horizon, ic) tuples

    Returns:
        dict: Dictionary containing statistical measures
    """
    if not horizon_data:
        return None

    horizons = [ph for ph, _ in horizon_data]

    stats_dict = {
        'count': len(horizons),
        'mean': np.mean(horizons),
        'variance': np.var(horizons, ddof=0),
        'std_dev': np.std(horizons, ddof=0),
        'min': np.min(horizons),
        'max': np.max(horizons),
        'median': np.median(horizons),
        'q25': np.percentile(horizons, 25),
        'q75': np.percentile(horizons, 75)
    }

    print(f'{"=" * 80}')
    print('Statistical Summary')
    print(f'{"=" * 80}')
    print(f'Sample size: {stats_dict["count"]}')
    print(f'Mean: {stats_dict["mean"]:.4f}')
    print(f'Standard deviation: {stats_dict["std_dev"]:.4f}')
    print(f'Variance: {stats_dict["variance"]:.4f}')
    print(f'Median: {stats_dict["median"]:.4f}')
    print(f'Range: [{stats_dict["min"]:.4f}, {stats_dict["max"]:.4f}]')
    print(f'25th percentile: {stats_dict["q25"]:.4f}')
    print(f'75th percentile: {stats_dict["q75"]:.4f}')
    print(f'{"=" * 80}\n')

    return stats_dict


def save_top_results(horizon_data, stats_dict, output_path, model_name,
                     perturbation, top_n=10):
    """
    Save top prediction horizon results to CSV.

    Parameters:
        horizon_data (list): List of (prediction_horizon, ic) tuples
        stats_dict (dict): Statistical summary dictionary
        output_path (str): Output file path
        model_name (str): Model identifier
        perturbation (float): Perturbation magnitude
        top_n (int): Number of top results to save
    """
    # Sort by prediction horizon (descending)
    sorted_data = sorted(horizon_data, key=lambda x: x[0], reverse=True)[:top_n]

    # Create DataFrame
    df = pd.DataFrame(sorted_data, columns=['PredictionHorizon', 'InitialCondition'])

    # Format IC for readability
    df['InitialCondition'] = df['InitialCondition'].apply(
        lambda ic: ', '.join(f'{v:.6f}' for v in ic)
    )

    # Add statistics and metadata
    df['Mean'] = stats_dict['mean']
    df['StdDev'] = stats_dict['std_dev']
    df['Model'] = model_name
    df['Perturbation'] = f'{perturbation:.2e}'

    # Reorder columns
    df = df[['PredictionHorizon', 'InitialCondition', 'Mean', 'StdDev', 'Model', 'Perturbation']]

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f'    Saved top {top_n} prediction horizons to: {output_path}')


def save_all_results(horizon_data, stats_dict, output_path, model_name, perturbation):
    """
    Save all prediction horizon results to CSV.

    Parameters:
        horizon_data (list): List of (prediction_horizon, ic) tuples
        stats_dict (dict): Statistical summary dictionary
        output_path (str): Output file path
        model_name (str): Model identifier
        perturbation (float): Perturbation magnitude
    """
    # Sort by prediction horizon (descending)
    sorted_data = sorted(horizon_data, key=lambda x: x[0], reverse=True)

    # Create DataFrame
    df = pd.DataFrame(sorted_data, columns=['PredictionHorizon', 'InitialCondition'])

    # Format IC for readability
    df['InitialCondition'] = df['InitialCondition'].apply(
        lambda ic: ', '.join(f'{v:.6f}' for v in ic)
    )

    # Add statistics and metadata
    df['Mean'] = stats_dict['mean']
    df['StdDev'] = stats_dict['std_dev']
    df['Model'] = model_name
    df['Perturbation'] = f'{perturbation:.2e}'

    # Reorder columns
    df = df[['PredictionHorizon', 'InitialCondition', 'Mean', 'StdDev', 'Model', 'Perturbation']]

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f'    Saved all {len(horizon_data)} prediction horizons to: {output_path}')


# ===========================
# Visualization Functions
# ===========================

def calculate_optimal_bins(data, method='fd'):
    """
    Calculate optimal number of bins using standard methods.

    Parameters:
        data (np.ndarray): Data array
        method (str): Binning method - 'fd' (Freedman-Diaconis),
                     'scott', or 'sturges'

    Returns:
        int: Optimal number of bins
    """
    data = np.asarray(data)
    n = len(data)

    if method == 'fd':
        # Freedman-Diaconis rule
        iqr = stats.iqr(data)
        bin_width = 2 * iqr * n ** (-1 / 3)
    elif method == 'scott':
        # Scott's rule
        bin_width = 3.5 * np.std(data, ddof=1) * n ** (-1 / 3)
    elif method == 'sturges':
        # Sturges' rule
        return int(np.ceil(np.log2(n) + 1))
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'fd', 'scott', 'sturges'.")

    if bin_width <= 0:
        return 10  # Fallback

    data_range = data.max() - data.min()
    return max(1, int(np.ceil(data_range / bin_width)))


def plot_mean_std_dev(horizon_data, stats_dict, output_path, model_name):
    """
    Create plot showing mean and standard deviation of prediction horizons.

    Parameters:
        horizon_data (list): List of (prediction_horizon, ic) tuples
        stats_dict (dict): Statistical summary dictionary
        output_path (str): Output file path
        model_name (str): Model identifier
    """
    horizons = [ph for ph, _ in horizon_data]
    mean = stats_dict['mean']
    std_dev = stats_dict['std_dev']

    # Create mean and std dev lines
    x_range = range(len(horizons))
    mean_line = [mean] * len(horizons)
    std_dev_upper = [mean + std_dev] * len(horizons)
    std_dev_lower = [mean - std_dev] * len(horizons)

    # Create plot
    plt.figure(figsize=(16, 10))
    plt.plot(x_range, horizons, 'o', markersize=4, label='Data Points', alpha=0.6)
    plt.plot(x_range, mean_line, 'r-', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.plot(x_range, std_dev_upper, 'g--', linewidth=2, label=f'+1 Std Dev: {mean + std_dev:.2f}')
    plt.plot(x_range, std_dev_lower, 'g--', linewidth=2, label=f'-1 Std Dev: {mean - std_dev:.2f}')

    # Fill between std dev lines
    plt.fill_between(x_range, std_dev_lower, std_dev_upper,
                     color='green', alpha=0.1, label='±1 Std Dev Region')

    plt.legend(loc='lower right', fontsize=12)
    plt.title(f'Prediction Horizon per Initial Condition - {model_name}',
              fontsize=16, fontweight='bold')
    plt.xlabel('Initial Condition Index', fontsize=14)
    plt.ylabel('Prediction Horizon (Lyapunov times)', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'    Saved mean/std dev plot to: {output_path}')


def plot_probability_density(horizon_data, stats_dict, output_path, bin_method='fd'):
    """
    Create probability density function plot with fitted normal distribution.

    Parameters:
        horizon_data (list): List of (prediction_horizon, ic) tuples
        stats_dict (dict): Statistical summary dictionary
        output_path (str): Output file path
        bin_method (str): Binning method for histogram
    """
    horizons = np.array([ph for ph, _ in horizon_data])
    mean = stats_dict['mean']
    std_dev = stats_dict['std_dev']

    # Calculate optimal bins
    nbins = calculate_optimal_bins(horizons, method=bin_method)
    print(f'Using {nbins} bins based on {bin_method} rule')

    # Create figure
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # Plot histogram
    sns.histplot(horizons, stat='density', bins=nbins, ax=ax,
                 color='lightblue', edgecolor='black', alpha=0.7)

    # Fit normal distribution
    bandwidth = len(horizons) ** (-1 / 5) * std_dev
    x = np.linspace(0, horizons.max() + bandwidth * 3, nbins)

    params = stats.norm.fit(horizons)
    y = stats.norm.pdf(x, *params)

    ax.plot(x, y, color='#282828', linewidth=2, label='Normal Distribution Fit')

    # Plot mean line
    ax.axvline(x=mean, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean:.3f}')

    # Shade ±1 std dev region
    std_lower = mean - std_dev
    std_upper = mean + std_dev

    # Find indices for shading
    mask = (x >= std_lower) & (x <= std_upper)
    x_shade = x[mask]
    y_shade = stats.norm.pdf(x_shade, *params)

    ax.fill_between(x_shade, y_shade, color='red', alpha=0.3,
                    label=f'±1 Std Dev Region')

    # Calculate probability within ±1 std dev
    prob_within_std = np.trapz(y_shade, x_shade)
    print(f'Probability within ±1 std dev: {prob_within_std:.4f}')

    ax.set_xlabel('Prediction Horizon (Lyapunov times)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Probability Density of Prediction Horizons',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'    Saved probability density plot to: {output_path}')


def plot_multiple_distribution_fits(horizon_data, output_path, bin_method='fd'):
    """
    Create PDF plot with multiple distributional fits for comparison.

    Parameters:
        horizon_data (list): List of (prediction_horizon, ic) tuples
        output_path (str): Output file path
        bin_method (str): Binning method for histogram
    """
    horizons = np.array([ph for ph, _ in horizon_data])

    # Calculate optimal bins
    nbins = calculate_optimal_bins(horizons, method=bin_method)

    # Distribution families to fit
    distributions = [
        (stats.norm, 'Normal', 'blue'),
        (stats.lognorm, 'Log-Normal', 'green'),
        (stats.gamma, 'Gamma', 'orange'),
        (stats.beta, 'Beta', 'purple')
    ]

    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Plot histogram
    sns.histplot(horizons, stat='density', bins=nbins, ax=ax,
                 color='lightgray', edgecolor='black', alpha=0.5, label='Histogram')

    # Fit and plot each distribution
    bandwidth = len(horizons) ** (-1 / 5) * np.std(horizons, ddof=1)
    x = np.linspace(0, horizons.max() + bandwidth * 3, 200)

    for dist_func, dist_name, color in distributions:
        try:
            params = dist_func.fit(horizons)
            y = dist_func.pdf(x, *params)
            ax.plot(x, y, label=dist_name, color=color, linewidth=2)
        except Exception as e:
            print(f'    Could not fit {dist_name} distribution: {e}')

    # Plot mean
    mean = np.mean(horizons)
    ax.axvline(x=mean, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean:.3f}')

    ax.set_xlabel('Prediction Horizon (Lyapunov times)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Probability Density with Multiple Distribution Fits',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'    Saved multiple distribution fits plot to: {output_path}')


# ===========================
# Main Execution
# ===========================

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    model_name = args["model_name"]
    prediction_dir = args["prediction_directory"]
    ic_file = args["initial_conditions"]
    perturbation = args["perturbation"]
    error_threshold = args["error_threshold"]
    bin_method = args["bin_method"]
    top_n = args["top_n"]

    # Setup paths
    data_dir = './_DataFiles/'
    ic_path = os.path.join(data_dir, ic_file)

    # Construct paths for model-specific directories
    perturbation_str = f'{perturbation:.0e}'.replace('-', 'minus').replace('+', 'plus')
    model_base_dir = os.path.join(prediction_dir, model_name, 'RelativeErrors')
    plot_dir = os.path.join(prediction_dir, model_name, 'Plotting')

    # Ensure output directory exists
    os.makedirs(plot_dir, exist_ok=True)

    print(f'\n{"=" * 80}')
    print('Prediction Horizon Statistical Analysis')
    print(f'{"=" * 80}')
    print(f'Model: {model_name}')
    print(f'Perturbation: {perturbation:.2e}')
    print(f'Error threshold: {error_threshold}')
    print(f'Binning method: {bin_method}')
    print(f'{"=" * 80}\n')

    # Load initial conditions
    initial_conditions = load_initial_conditions(ic_path)

    # Extract prediction horizons
    horizon_data = extract_prediction_horizons(
        initial_conditions, model_base_dir, perturbation, error_threshold
    )

    if not horizon_data:
        print('     No valid prediction horizons found. Exiting.')
        return

    # Compute statistics
    stats_dict = compute_statistics(horizon_data)

    # Display top results
    sorted_data = sorted(horizon_data, key=lambda x: x[0], reverse=True)
    print(f'{"=" * 80}')
    print(f'Top {top_n} Longest Prediction Horizons')
    print(f'{"=" * 80}')
    for i, (ph, ic) in enumerate(sorted_data[:top_n], start=1):
        ic_str = ', '.join(f'{val:.6f}' for val in ic)
        print(f'{i:2d}. Horizon: {ph:.4f} | IC: [{ic_str}]')
    print(f'{"=" * 80}\n')

    # Generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d")

    # Save results to CSV
    top_results_path = os.path.join(
        plot_dir, f'top{top_n}_prediction_horizons_{model_name}_{timestamp}.csv'
    )
    save_top_results(horizon_data, stats_dict, top_results_path,
                     model_name, perturbation, top_n)

    all_results_path = os.path.join(
        plot_dir, f'all_prediction_horizons_{model_name}_{timestamp}.csv'
    )
    save_all_results(horizon_data, stats_dict, all_results_path,
                     model_name, perturbation)

    # Generate visualizations
    print(f'\n{"=" * 80}')
    print('Generating Visualizations')
    print(f'{"=" * 80}\n')

    mean_std_plot_path = os.path.join(
        plot_dir, f'prediction_horizons_mean_stddev_{model_name}_{timestamp}.png'
    )
    plot_mean_std_dev(horizon_data, stats_dict, mean_std_plot_path, model_name)

    pdf_plot_path = os.path.join(
        plot_dir, f'prediction_horizons_pdf_{model_name}_{timestamp}.png'
    )
    plot_probability_density(horizon_data, stats_dict, pdf_plot_path, bin_method)

    multi_fit_plot_path = os.path.join(
        plot_dir, f'prediction_horizons_multi_fit_{model_name}_{timestamp}.png'
    )
    plot_multiple_distribution_fits(horizon_data, multi_fit_plot_path, bin_method)

    print(f'\n{"=" * 80}')
    print('Analysis Complete')
    print(f'{"=" * 80}')
    print(f'Results directory: {plot_dir}')
    print(f'{"=" * 80}')


if __name__ == "__main__":
    main()