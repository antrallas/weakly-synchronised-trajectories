"""
Kuramoto Order Parameter (KOP) Analysis for Lorenz System Trajectories

This script performs Kuramoto Order Parameter (KOP) analysis on ensembles of
Lorenz system trajectories to assess phase synchronization across the ensemble.
KOP quantifies the degree of phase coherence in the oscillator ensemble at each
time point.

The analysis uses a memory-efficient streaming approach with ijson to handle
large trajectory datasets, and implements parallel processing with early stopping
when KOP values fall below a specified threshold. Low KOP values indicate
desirable trajectory diversity for training robust prediction models.

When a suitable subset is found (KOP below threshold), the script automatically
saves both the trajectory keys and the full trajectory data for immediate use
in training.

Key features:
- Memory-efficient streaming JSON parsing for large datasets
- Parallel processing of multiple random trajectory samples
- Phase angle extraction using Hilbert transform
- KOP computation across ensemble at each time step
- Early stopping when low synchronization threshold is met
- Automatic saving of selected trajectory subset
- Automatic visualization and CSV logging

Kuramoto Order Parameter is defined as:
    KOP(t) = |⟨e^(iθ_j(t))⟩_j| = |1/N ∑_j e^(iθ_j(t))|
where θ_j(t) is the instantaneous phase of trajectory j at time t.
KOP ranges from 0 (no synchronization) to 1 (perfect synchronization).

Example  arguments:
-f IC_x0_10000_TTs.json -s 1000 -t 0.053 -n 100 -p 8
This set of arguments would find 1000 low-KOP trajectories from a set of 10000 trajectories.

Author: Anthony S. Miller, University of Essex
Date: 17/10/2025
"""

import os
import json
import random
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import ijson
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, freeze_support
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
        description="Perform KOP analysis on Lorenz system trajectory ensembles"
    )
    parser.add_argument(
        "-f", "--trajectory_file",
        default="IC_x0_10000_TTs.json",
        type=str,
        help="JSON file containing trajectory data"
    )
    parser.add_argument(
        "-o", "--output_directory",
        default="./KOP_Analysis/",
        type=str,
        help="Directory for output files"
    )
    parser.add_argument(
        "-n", "--num_iterations",
        default=100,
        type=int,
        help="Maximum number of iterations (default: 100)"
    )
    parser.add_argument(
        "-s", "--sample_size",
        default=1000,
        type=int,
        help="Number of trajectories to sample per iteration (default: 1000)"
    )
    parser.add_argument(
        "-t", "--threshold",
        default=0.053,
        type=float,
        help="KOP threshold for early stopping (default: 0.053)"
    )
    parser.add_argument(
        "-p", "--num_workers",
        default=4,
        type=int,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--integration_time",
        default=20,
        type=int,
        help="Integration time T (default: 20)"
    )
    parser.add_argument(
        "--dt",
        default=0.01,
        type=float,
        help="Time step (default: 0.01)"
    )
    parser.add_argument(
        "--save_subset",
        action="store_true",
        default=True,
        help="Save selected trajectory subset when threshold is met (default: True)"
    )
    return vars(parser.parse_args())


# ===========================
# Memory-Efficient JSON Parsing
# ===========================

def get_all_trajectory_keys(file_path):
    """
    Extract all trajectory keys from large JSON file using streaming parser.

    This function uses ijson to stream through the JSON file without loading
    the entire file into memory, making it suitable for very large datasets.

    Parameters:
        file_path (str): Path to JSON file containing trajectories

    Returns:
        list: List of trajectory keys (typically integer strings)
    """
    print(f'Extracting trajectory keys from: {file_path}')
    keys = []

    with open(file_path, 'r') as f:
        parser = ijson.parse(f)
        for prefix, event, value in parser:
            if prefix == '' and event == 'map_key':
                keys.append(value)

    print(f'Found {len(keys)} trajectories in file')
    return keys


def load_selected_trajectories(file_path, selected_keys):
    """
    Load only selected trajectories from JSON file using streaming parser.

    This memory-efficient approach only loads the requested trajectories
    rather than the entire dataset as trajectory sets can be multi-gigabit files.

    Parameters:
        file_path (str): Path to JSON file
        selected_keys (list): List of trajectory keys to load

    Returns:
        dict: Dictionary mapping keys to trajectory arrays
    """
    selected_data = {}
    selected_keys_set = set(selected_keys)

    with open(file_path, 'r') as f:
        objects = ijson.kvitems(f, '')
        for key, value in objects:
            if key in selected_keys_set:
                selected_data[key] = value
                # Early exit once all selected trajectories are loaded
                if len(selected_data) == len(selected_keys):
                    break

    return selected_data


# ===========================
# Kuramoto Order Parameter Computation
# ===========================

def extract_x_component(trajectory_data):
    """
    Extract x-component time series from all trajectories.

    The x-component is used for phase analysis as it represents one of the
    three state variables of the Lorenz system and exhibits oscillatory behavior
    suitable for phase extraction.

    Parameters:
        trajectory_data (dict): Dictionary of trajectories, where each trajectory
                               is a list of [x, y, z] points

    Returns:
        np.ndarray: Array of shape (n_trajectories, n_time_steps) containing
                   x-component values
    """
    num_trajectories = len(trajectory_data)
    num_time_steps = len(next(iter(trajectory_data.values())))

    X_values = np.zeros((num_trajectories, num_time_steps))

    for i, trajectory in enumerate(trajectory_data.values()):
        X_values[i, :] = [point[0] for point in trajectory]

    return X_values


def compute_kuramoto_order_parameter(X_values):
    """
    Compute Kuramoto Order Parameter (KOP) across trajectory ensemble.

    KOP measures the phase synchronization between trajectories at each time
    point, treating each trajectory as an oscillator in the Kuramoto model.
    The computation involves:

    1. Hilbert transform to obtain analytic signal for each trajectory
    2. Phase angle extraction from complex analytic signal: θ_j(t) = arg(H[x_j(t)])
    3. Complex mean of phase vectors: ⟨e^(iθ)⟩ = 1/N ∑_j e^(iθ_j)
    4. KOP = magnitude of complex mean = |⟨e^(iθ)⟩|

    KOP interpretation:
    - KOP ~ 0: No phase synchronization (desirable for diverse training set)
    - KOP ~ 1: Perfect phase synchronization (undesirable, indicates redundancy trajectories)

    Low KOP values indicate good trajectory diversity, which is important for
    training robust prediction models for chaotic systems.

    Parameters:
        X_values (np.ndarray): Array of shape (n_trajectories, n_time_steps)

    Returns:
        np.ndarray: KOP values at each time step, shape (n_time_steps,)
    """
    num_trajectories, num_time_steps = X_values.shape
    KOP_values = np.zeros(num_time_steps)

    for t in range(num_time_steps):
        # Extract x-values at time t across all trajectories
        x_t = X_values[:, t]

        # Compute analytic signal using Hilbert transform
        analytic_signal = scipy.signal.hilbert(x_t)

        # Extract instantaneous phase angles
        phase_angles = np.angle(analytic_signal)

        # Compute complex order parameter (mean of unit phase vectors)
        # This is the core of the Kuramoto order parameter calculation
        order_parameter = np.mean(np.exp(1j * phase_angles))

        # KOP is the magnitude of the complex order parameter
        KOP_values[t] = np.abs(order_parameter)

    return KOP_values


# ===========================
# Output and Visualization
# ===========================

def write_summary_to_csv(csv_path, lock, iteration, mean_kop, max_kop,
                         min_kop, below_threshold):
    """
    Write iteration summary to CSV log file (thread-safe).

    Parameters:
        csv_path (str): Path to CSV log file
        lock: Multiprocessing lock for thread-safe writing
        iteration (int): Iteration number
        mean_kop (float): Mean KOP across time
        max_kop (float): Maximum KOP value
        min_kop (float): Minimum KOP value
        below_threshold (bool): Whether all KOP values are below threshold
    """
    with lock:
        write_header = not os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    'Iteration', 'Mean_KOP', 'Max_KOP', 'Min_KOP', 'Below_Threshold'
                ])

            writer.writerow([
                iteration,
                f"{mean_kop:.5f}",
                f"{max_kop:.5f}",
                f"{min_kop:.5f}",
                below_threshold
            ])


def save_kop_results(output_dir, iteration, selected_keys, KOP_values):
    """
    Save KOP results to JSON file.

    Parameters:
        output_dir (str): Output directory
        iteration (int): Iteration number
        selected_keys (list): Keys of trajectories used in this iteration
        KOP_values (np.ndarray): KOP values over time
    """
    result = {
        'iteration': iteration,
        'selected_keys': selected_keys,
        'kop_values': KOP_values.tolist(),
        'statistics': {
            'mean': float(np.mean(KOP_values)),
            'std': float(np.std(KOP_values)),
            'max': float(np.max(KOP_values)),
            'min': float(np.min(KOP_values))
        }
    }

    json_filename = os.path.join(output_dir, f"kop_result_iter{iteration}.json")

    with open(json_filename, 'w') as f:
        json.dump(result, f, indent=2)


def create_kop_plot(output_dir, iteration, time_array, KOP_values, threshold):
    """
    Create and save KOP visualization plot.

    Parameters:
        output_dir (str): Output directory
        iteration (int): Iteration number
        time_array (np.ndarray): Time points
        KOP_values (np.ndarray): KOP values over time
        threshold (float): Threshold value to visualize
    """
    plt.figure(figsize=(10, 6))

    # Plot KOP over time
    plt.plot(time_array, KOP_values, label=f"KOP (Iteration {iteration})",
             color='blue', linewidth=1.5)

    # Add threshold line
    plt.axhline(y=threshold, color='red', linestyle='--',
                linewidth=2, label=f"Threshold = {threshold}")

    # Labels and formatting
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Kuramoto Order Parameter (KOP)", fontsize=12)
    plt.title(f"KOP Over Time - Iteration {iteration}", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([0, 1.05])  # KOP ranges from 0 to 1
    plt.tight_layout()

    # Save plot
    plot_filename = os.path.join(output_dir, f"kop_plot_iter{iteration}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_selected_trajectory_subset(output_dir, iteration, selected_keys,
                                    original_json_path, output_filename=None):
    """
    Save the selected trajectory subset that meets the KOP threshold.

    This function saves both:
    1. The list of selected trajectory keys (indices in original file)
    2. The actual trajectory data for those keys (ready for training)

    Parameters:
        output_dir (str): Output directory
        iteration (int): Iteration number where threshold was met
        selected_keys (list): Keys of the selected trajectories
        original_json_path (str): Path to original trajectory file
        output_filename (str): Optional custom filename for the subset
    """
    print(f'\n{"=" * 80}')
    print('Saving Selected Trajectory Subset')
    print(f'{"=" * 80}')

    # Save the keys
    keys_output = {
        'iteration': iteration,
        'num_trajectories': len(selected_keys),
        'selected_keys': selected_keys,
        'source_file': os.path.basename(original_json_path),
        'description': 'Trajectory subset with KOP below threshold - suitable for training'
    }

    if output_filename is None:
        keys_filename = f'selected_trajectories_iter{iteration}_keys.json'
    else:
        keys_filename = f'{output_filename}_keys.json'

    keys_path = os.path.join(output_dir, keys_filename)

    with open(keys_path, 'w') as f:
        json.dump(keys_output, f, indent=2)

    print(f'   Saved selected trajectory keys to: {keys_filename}')
    print(f'   Keys represent indices in the original trajectory file')

    # Also save the actual trajectory data
    print(f'\nLoading and saving trajectory data for selected subset...')
    trajectory_data = load_selected_trajectories(original_json_path, selected_keys)

    if output_filename is None:
        data_filename = f'selected_trajectories_iter{iteration}_data.json'
    else:
        data_filename = f'{output_filename}_data.json'

    data_path = os.path.join(output_dir, data_filename)

    # Save with 'trajectories' key for compatibility with training scripts
    output_data = {'trajectories': trajectory_data}

    with open(data_path, 'w') as f:
        json.dump(trajectory_data, f, indent=2)

    print(f'    Saved trajectory data to: {data_filename}')
    print(f'   {len(selected_keys)} complete trajectories ready for training')
    print(f'{"=" * 80}')


# ===========================
# Iteration Processing
# ===========================

def process_single_iteration(iteration_data):
    """
    Process a single KOP analysis iteration.

    This function performs one complete KOP analysis:
    1. Randomly samples trajectories from the dataset
    2. Extracts x-component time series
    3. Computes KOP across the ensemble (Kuramoto order parameter)
    4. Saves results (JSON, plot, CSV log)
    5. Checks if KOP is below threshold (indicating low synchronization)

    Low KOP values across all time points indicate that the sampled trajectories
    exhibit diverse phase dynamics, which is desirable for training robust models
    for chaotic systems.

    Parameters:
        iteration_data (tuple): Tuple containing:
            - iteration (int): Iteration number
            - json_path (str): Path to trajectory JSON file
            - output_dir (str): Output directory
            - all_keys (list): All available trajectory keys
            - sample_size (int): Number of trajectories to sample
            - threshold (float): KOP threshold for early stopping
            - lock: Multiprocessing lock
            - csv_path (str): Path to summary CSV file
            - time_array (np.ndarray): Time points for plotting

    Returns:
        tuple: (iteration, below_threshold, mean_kop, selected_keys)
    """
    (iteration, json_path, output_dir, all_keys, sample_size,
     threshold, lock, csv_path, time_array) = iteration_data

    try:
        # Randomly sample trajectories
        selected_keys = random.sample(all_keys, sample_size)

        # Load selected trajectories
        trajectory_data = load_selected_trajectories(json_path, selected_keys)

        # Extract x-component time series
        X_values = extract_x_component(trajectory_data)

        # Compute Kuramoto Order Parameter
        KOP_values = compute_kuramoto_order_parameter(X_values)

        # Calculate statistics
        mean_kop = np.mean(KOP_values)
        max_kop = np.max(KOP_values)
        min_kop = np.min(KOP_values)
        below_threshold = np.all(KOP_values < threshold)

        # Save results
        save_kop_results(output_dir, iteration, selected_keys, KOP_values)
        create_kop_plot(output_dir, iteration, time_array, KOP_values, threshold)
        write_summary_to_csv(csv_path, lock, iteration, mean_kop,
                             max_kop, min_kop, below_threshold)

        return (iteration, below_threshold, mean_kop, selected_keys)

    except Exception as e:
        print(f'\nError in iteration {iteration}: {e}')
        return (iteration, False, None, None)


# ===========================
# Parallel Execution Driver
# ===========================

def run_parallel_kop_analysis(json_path, output_dir, num_iterations=100,
                              sample_size=1000, threshold=0.053,
                              num_workers=4, integration_time=20, dt=0.01,
                              save_subset=True):
    """
    Run parallel KOP analysis with early stopping capability.

    This function orchestrates the parallel execution of multiple KOP analysis
    iterations. It implements early stopping when a sample achieves KOP values
    below the threshold for all time points, indicating low phase synchronization
    and good trajectory diversity.

    When a suitable subset is found, both the trajectory keys and full data are
    saved automatically for use in training.

    Parameters:
        json_path (str): Path to trajectory JSON file
        output_dir (str): Output directory for results
        num_iterations (int): Maximum number of iterations
        sample_size (int): Number of trajectories per sample (e.g., 1000)
        threshold (float): KOP threshold for early stopping (lower is better)
        num_workers (int): Number of parallel processes
        integration_time (float): Total integration time
        dt (float): Time step
        save_subset (bool): Whether to save the selected trajectory subset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'kop_summary_log.csv')

    print(f'\n{"=" * 80}')
    print('Kuramoto Order Parameter (KOP) Analysis')
    print(f'{"=" * 80}')
    print(f'Trajectory file: {json_path}')
    print(f'Output directory: {output_dir}')
    print(f'Maximum iterations: {num_iterations}')
    print(f'Sample size per iteration: {sample_size}')
    print(f'KOP threshold: {threshold} (lower values indicate better diversity)')
    print(f'Parallel workers: {num_workers}')
    print(f'Save subset when found: {save_subset}')
    print(f'{"=" * 80}\n')

    # Load all trajectory keys
    all_keys = get_all_trajectory_keys(json_path)

    if len(all_keys) < sample_size:
        raise ValueError(f"Not enough trajectories: found {len(all_keys)}, "
                         f"need {sample_size}")

    print(f'Total trajectories available: {len(all_keys)}')
    print(f'Will sample {sample_size} trajectories per iteration\n')

    # Create time array
    time_array = np.arange(0, integration_time + dt, dt)

    # Setup parallel processing with manager for shared state
    with Manager() as manager:
        lock = manager.Lock()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all iterations
            futures = {}
            for i in range(1, num_iterations + 1):
                iteration_data = (
                    i, json_path, output_dir, all_keys, sample_size,
                    threshold, lock, csv_path, time_array
                )
                future = executor.submit(process_single_iteration, iteration_data)
                futures[future] = i

            # Process results as they complete
            early_stop_triggered = False
            winning_iteration = None
            winning_keys = None

            for future in tqdm(as_completed(futures), total=num_iterations,
                               desc="Processing iterations"):
                iteration, below_threshold, mean_kop, selected_keys_result = future.result()

                if mean_kop is not None:
                    status = "Finished" if below_threshold else "Working"
                    print(f'{status} Iteration {iteration}: Mean KOP = {mean_kop:.5f}')

                # Check for early stopping condition
                if below_threshold and not early_stop_triggered:
                    print(f'\n{"=" * 80}')
                    print(f'EARLY STOPPING TRIGGERED')
                    print(f'KOP below threshold ({threshold}) for all time points '
                          f'in iteration {iteration}')
                    print(f'Found {sample_size} trajectories with low synchronization!')
                    print(f'{"=" * 80}\n')

                    early_stop_triggered = True
                    winning_iteration = iteration
                    winning_keys = selected_keys_result

                    # Cancel remaining futures
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break

        # Save the selected subset if found
        if early_stop_triggered and save_subset and winning_keys:
            save_selected_trajectory_subset(
                output_dir, winning_iteration, winning_keys, json_path
            )

        # Final summary
        completed_count = sum(1 for f in futures if f.done() and not f.cancelled())

        print(f'\n{"=" * 80}')
        print('Analysis Complete')
        print(f'{"=" * 80}')
        print(f'Completed iterations: {completed_count}/{num_iterations}')

        if early_stop_triggered:
            print(f'Status: Early stopping triggered - low synchronization found')
            print(f'Selected subset: {sample_size} trajectories from iteration {winning_iteration}')
            if save_subset:
                print(f'Subset files saved and ready for training')
        else:
            print(f'Status: All iterations completed (threshold not met)')
            print(f'Consider: Lowering threshold or increasing iterations')

        print(f'Results directory: {output_dir}')
        print(f'Summary CSV: {csv_path}')
        print(f'{"=" * 80}')


# ===========================
# Main Entry Point
# ===========================

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    # Setup paths
    data_dir = './_DataFiles/'
    trajectory_file = os.path.join(data_dir, args["trajectory_file"])
    output_dir = args["output_directory"]

    # Validate input file exists
    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

    # Run parallel KOP analysis
    run_parallel_kop_analysis(
        json_path=trajectory_file,
        output_dir=output_dir,
        num_iterations=args["num_iterations"],
        sample_size=args["sample_size"],
        threshold=args["threshold"],
        num_workers=args["num_workers"],
        integration_time=args["integration_time"],
        dt=args["dt"],
        save_subset=args["save_subset"]
    )


if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing compatibility
    main()
