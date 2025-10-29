"""
Relative Error Computation for Lorenz System Predictions

This script computes the relative error between ground truth trajectories
(computed using Runge-Kutta integration) and predicted trajectories (from
trained neural network models) for the Lorenz system.

The relative error is calculated at each time step as:
    error(t) = ||y_true(t) - y_pred(t)|| / ||y_true(t)||

Results are saved as CSV files containing time, Lyapunov time, relative error,
and identifying information for each trajectory.

Key features:
- Robust file matching using triplet tolerance comparison
- Automatic perturbation application to match prediction inputs
- Lyapunov time conversion for physically meaningful time scales
- CSV output for further analysis and visualization

Example arguments:
-m ModelA -d ./_PredictionOutput/ModelA/ -ic IC_x0_1000_TTs.json --perturbation 1e-5 -t 20 --tolerance 1e-8

Author: Anthony S. Miller, University of Essex
Date: 17/10/2025
"""

import json
import numpy as np
import os
import re
import pandas as pd
import datetime
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
        description="Compute relative errors between ground truth and predicted trajectories"
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
        help="Directory containing prediction output files"
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
        "-t", "--integration_time",
        default=20,
        type=int,
        help="Integration time T (default: 20)"
    )
    parser.add_argument(
        "--tolerance",
        default=1e-7,
        type=float,
        help="Tolerance for IC matching (default: 1e-7)"
    )
    return vars(parser.parse_args())


def ensure_directory_exists(directory):
    """
    Create directory if it doesn't exist.

    Parameters:
        directory (str): Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def calculate_lyapunov_time_array(t, lle=0.9036):
    """
    Convert time array to Lyapunov time units.

    Parameters:
        t (np.ndarray): Time array
        lle (float): Largest Lyapunov exponent (default: 0.9036)

    Returns:
        list: Time array in Lyapunov units
    """
    return [time_point * lle for time_point in t]


def extract_triplet_from_filename(filename):
    """
    Extract initial condition triplet from filename.

    Filenames are expected to contain initial conditions in the format:
    'lorenz63_IC####_...' where the IC values are embedded in the name.

    Parameters:
        filename (str): Filename to parse

    Returns:
        list or None: [x, y, z] values if found, None otherwise
    """
    # Try to match pattern like [[x y z]]
    match = re.search(r"\[\[([^\]]+)\]\]", filename)
    if not match:
        return None

    parts = match.group(1).strip().split()
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


def load_trajectory_from_csv(filepath):
    """
    Load trajectory data from CSV file.

    Parameters:
        filepath (str): Path to CSV file

    Returns:
        np.ndarray: Trajectory data with shape (n_points, 3)
    """
    df = pd.read_csv(filepath, header=None)
    return df.to_numpy()


def compute_relative_error(ground_truth, predicted):
    """
    Compute relative error between ground truth and predicted trajectories.

    The relative error at each time step is defined as:
        error(t) = ||y_true(t) - y_pred(t)|| / ||y_true(t)||

    Parameters:
        ground_truth (np.ndarray): Ground truth trajectory (n_points, 3)
        predicted (np.ndarray): Predicted trajectory (n_points, 3)

    Returns:
        list: Relative errors at each time step
    """
    if len(ground_truth) != len(predicted):
        raise ValueError(f"Trajectory lengths don't match: "
                         f"ground_truth={len(ground_truth)}, predicted={len(predicted)}")

    relative_errors = [
        np.linalg.norm(true_point - pred_point) / np.linalg.norm(true_point)
        for true_point, pred_point in zip(ground_truth, predicted)
    ]

    return relative_errors


def create_error_dataframe(time_array, lyapunov_time_array, relative_errors,
                           initial_condition, model_name):
    """
    Create DataFrame containing error analysis results.

    Parameters:
        time_array (np.ndarray): Time points
        lyapunov_time_array (list): Lyapunov time points
        relative_errors (list): Relative errors at each time point
        initial_condition (list): Initial condition [x, y, z]
        model_name (str): Model identifier

    Returns:
        pd.DataFrame: DataFrame with columns [Time, PredictionHorizon, Error, IC, Model]
    """
    # Format IC string for identification
    ic_str = "_".join(f"{val:.15g}" for val in initial_condition)

    # Create DataFrame
    df = pd.DataFrame({
        'Time': time_array,
        'PredictionHorizon': lyapunov_time_array,
        'Error': relative_errors,
        'IC': ic_str,
        'Model': model_name
    })

    return df


def process_single_initial_condition(ic_data):
    """
    Process a single initial condition: compute and save relative error.

    Parameters:
        ic_data (tuple): Tuple containing all necessary data

    Returns:
        tuple: (success, ic_index, message)
    """
    (ic_idx, original_ic, perturbed_ic, ground_truth_files, predicted_files,
     base_dir, time_array, lyapunov_time_array, model_name, output_dir,
     tolerance) = ic_data

    try:
        # Find matching files for this IC
        ground_truth_path = find_matching_file(
            ground_truth_files, base_dir, perturbed_ic, tolerance
        )
        predicted_path = find_matching_file(
            predicted_files, base_dir, perturbed_ic, tolerance
        )

        # Check if both files were found
        if not ground_truth_path or not predicted_path:
            status_gt = 'Found' if ground_truth_path else 'Not Found'
            status_pred = 'Found' if predicted_path else 'Not Found'
            message = (f"Missing files for IC {ic_idx}: "
                       f"Ground truth {status_gt} | Predicted {status_pred}")
            return False, ic_idx, message

        # Load trajectories
        ground_truth = load_trajectory_from_csv(ground_truth_path)
        predicted = load_trajectory_from_csv(predicted_path)

        # Compute relative error
        relative_errors = compute_relative_error(ground_truth, predicted)

        # Create error DataFrame
        error_df = create_error_dataframe(
            time_array, lyapunov_time_array, relative_errors,
            perturbed_ic, model_name
        )

        # Generate output filename
        ic_str = "_".join(f"{val:.15g}" for val in perturbed_ic)
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        output_filename = f"RelativeError_IC{ic_idx:04d}_{model_name}_{timestamp}.csv"
        output_path = os.path.join(output_dir, output_filename)

        # Save to CSV
        error_df.to_csv(output_path, index=False)

        message = f"Saved: {output_filename}"
        return True, ic_idx, message

    except Exception as e:
        message = f"Error processing IC {ic_idx}: {str(e)}"
        return False, ic_idx, message


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    model_name = args["model_name"]
    prediction_dir = args["prediction_directory"]
    ic_file = args["initial_conditions"]
    perturbation = args["perturbation"]
    integration_time = args["integration_time"]
    tolerance = args["tolerance"]

    # Configuration
    data_dir = './_DataFiles/'
    ic_path = os.path.join(data_dir, ic_file)

    # Lorenz system parameters
    LARGEST_LYAPUNOV_EXPONENT = 0.9036
    dt = 0.01
    T = integration_time

    # Setup output directory
    perturbation_str = f'{perturbation:.0e}'.replace('-', 'minus').replace('+', 'plus')
    output_dir = os.path.join(prediction_dir, model_name, 'RelativeErrors')
    ensure_directory_exists(output_dir)

    print(f'\n{"=" * 80}')
    print('Relative Error Computation for Lorenz System Predictions')
    print(f'{"=" * 80}')
    print(f'Model: {model_name}')
    print(f'Integration time: T = {T}')
    print(f'Perturbation: {perturbation:.2e}')
    print(f'Matching tolerance: {tolerance:.2e}')
    print(f'Output directory: {output_dir}')
    print(f'{"=" * 80}\n')

    # Setup time arrays
    t = np.arange(0, T + dt, dt)
    lyapunov_time = calculate_lyapunov_time_array(t, LARGEST_LYAPUNOV_EXPONENT)

    print(f'Time steps: {len(t)}')
    print(f'Lyapunov time range: 0 - {lyapunov_time[-1]:.4f}')

    # Load initial conditions
    initial_conditions = load_initial_conditions(ic_path)

    # Get list of prediction files
    prediction_base_dir = os.path.join(prediction_dir, model_name)

    if not os.path.exists(prediction_base_dir):
        raise FileNotFoundError(f"Prediction directory not found: {prediction_base_dir}")

    all_files = [f for f in os.listdir(prediction_base_dir) if f.endswith('.csv')]
    ground_truth_files = [f for f in all_files if 'groundtruth' in f.lower()]
    predicted_files = [f for f in all_files if 'predicted' in f.lower()]

    print(f'\nFound {len(ground_truth_files)} ground truth files')
    print(f'Found {len(predicted_files)} predicted files')

    if not ground_truth_files or not predicted_files:
        raise FileNotFoundError("No prediction files found. Check directory structure.")

    # Process each initial condition
    print(f'\n{"=" * 80}')
    print(f'Processing {len(initial_conditions)} Initial Conditions')
    print(f'{"=" * 80}\n')

    successful = 0
    failed = 0

    for ic_idx, original_ic in enumerate(initial_conditions):
        # Apply perturbation (to match what was used in prediction)
        perturbed_ic = apply_perturbation(original_ic, perturbation)

        # Prepare data for processing
        ic_data = (
            ic_idx, original_ic, perturbed_ic,
            ground_truth_files, predicted_files,
            prediction_base_dir, t, lyapunov_time,
            model_name, output_dir, tolerance
        )

        # Process this IC
        success, idx, message = process_single_initial_condition(ic_data)

        if success:
            print(f'Success: IC {idx:04d}: {message}')
            successful += 1
        else:
            print(f'Failed: IC {idx:04d}: {message}')
            failed += 1

    # Summary
    print(f'\n{"=" * 80}')
    print('Processing Complete')
    print(f'{"=" * 80}')
    print(f'Successfully processed: {successful}/{len(initial_conditions)}')
    if failed > 0:
        print(f'Failed: {failed}/{len(initial_conditions)}')
    print(f'Results saved to: {output_dir}')
    print(f'{"=" * 80}')


if __name__ == "__main__":
    main()