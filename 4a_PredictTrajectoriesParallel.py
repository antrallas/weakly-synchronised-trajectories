"""
Parallel Neural Network Prediction for Lorenz System Trajectories

This script generates predictions for the Lorenz system using trained neural network
models. It processes multiple initial conditions in parallel, comparing model predictions
against ground truth trajectories computed using the Runge-Kutta method.

Key features:
- Parallel processing of multiple initial conditions
- Recursive model prediction for long-term forecasting
- Ground truth comparison using RK45 integration
- Automatic perturbation of initial conditions
- CSV output for both predicted and actual trajectories

The script applies a small perturbation to initial conditions before prediction to
ensure they haven't been seen during training, maintaining valid test conditions.

Example arguments:
-f ann_x0_1000_MSE_ModelA.keras -m ModelA -t 20 -n 1000 --perturbation 1e-15 -p 8

Author: Anthony S. Miller, University of Essex
Date: 17/10/2025
"""

import numpy as np
import tensorflow as tf
import keras
from scipy.integrate import solve_ivp
import datetime
from tqdm import tqdm
import time
import json
from json import JSONEncoder
import multiprocessing as mp
from multiprocessing import Process
import sys
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings

warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class NumpyArrayEncoder(JSONEncoder):
    """
    Custom JSON encoder for NumPy arrays.

    Converts NumPy arrays to Python lists for JSON serialization.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        dict: Dictionary containing parsed arguments
    """
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Generate predictions using trained neural network models"
    )
    parser.add_argument(
        "-f", "--model_file",
        default="ann_x0_1000_MSE_Model_A.keras",
        type=str,
        help="Trained neural network model file (.keras)"
    )
    parser.add_argument(
        "-m", "--model_name",
        default="ModelA",
        type=str,
        help="Model identifier for output files"
    )
    parser.add_argument(
        "-ic", "--initial_conditions",
        default="IC_x0_1000_LP_P.json",
        type=str,
        help="JSON file containing perturbed initial conditions"
    )
    parser.add_argument(
        "-n", "--num_trajectories",
        default=1000,
        type=int,
        help="Number of trajectories to predict"
    )
    parser.add_argument(
        "-p", "--num_processes",
        default=4,
        type=int,
        help="Number of parallel processes (default: 4)"
    )
    parser.add_argument(
        "-t", "--integration_time",
        default=20,
        type=int,
        help="Integration time T (default: 20)"
    )
    parser.add_argument(
        "--perturbation",
        default=1e-15,
        type=float,
        help="Perturbation magnitude for initial conditions (default: 1e-15)"
    )
    parser.add_argument(
        "-o", "--output_directory",
        default="./_PredictionOutput/",
        type=str,
        help="Base directory for prediction output"
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
        print(f'Created directory: {directory}')


def load_initial_conditions(filename, perturbation=1e-15):
    """
    Load initial conditions from JSON file and apply perturbation.

    Parameters:
        filename (str): Path to initial conditions JSON file
        perturbation (float): Magnitude of perturbation to apply

    Returns:
        np.ndarray: Array of perturbed initial conditions with shape (n, 1, 3)
    """
    print(f'Loading initial conditions from: {filename}')

    with open(filename, 'r') as f:
        data = json.load(f)
        initial_conditions = np.asarray(data["perturbedICs"])

    print(f'Loaded {len(initial_conditions)} initial conditions')
    print(f'Applying perturbation of magnitude: {perturbation:.2e}')

    # Apply perturbation to all three components
    perturbed_ics = []
    for ic in initial_conditions:
        perturbed_ic = [
            ic[0] + perturbation,
            ic[1] + perturbation,
            ic[2] + perturbation
        ]
        perturbed_ics.append(perturbed_ic)

    # Expand dimensions for Keras predict function (requires shape: n, 1, 3)
    perturbed_array = np.expand_dims(perturbed_ics, axis=1)

    print(f'Prepared {len(perturbed_array)} perturbed initial conditions')

    return perturbed_array


def load_trained_model(model_path):
    """
    Load a trained Keras model from file.

    Parameters:
        model_path (str): Path to trained model file

    Returns:
        keras.Model: Loaded model
    """
    print(f'Loading trained model from: {model_path}')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = keras.models.load_model(model_path)
    print(f'Model loaded successfully')
    model.summary()

    return model


def lorenz(t, state, sigma, beta, rho):
    """
    Parameters:
        t (float): Time (not used, required by solve_ivp interface)
        state (array-like): Current state [x, y, z]
        sigma (float): Prandtl number
        beta (float): Geometrical factor
        rho (float): Rayleigh number

    Returns:
        list: Time derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return [dx, dy, dz]


def compute_ground_truth_trajectory(initial_condition, t, t_span, lorenz_params):
    """
    Compute ground truth trajectory using Runge-Kutta integration.

    Parameters:
        initial_condition (array-like): Initial state [x, y, z]
        t (np.ndarray): Time points for evaluation
        t_span (tuple): Time span (start, end)
        lorenz_params (tuple): Lorenz system parameters (sigma, beta, rho)

    Returns:
        np.ndarray: Ground truth trajectory with shape (n_points, 3)
    """
    # Solve ODE using RK45 method
    solution = solve_ivp(
        lorenz, t_span, initial_condition,
        args=lorenz_params,
        method='RK45',
        t_eval=t
    )

    # Transpose to shape (n_points, 3)
    trajectory = solution.y[:, :].T

    return trajectory


def predict_trajectory_recursively(model, initial_condition, num_steps):
    """
    Generate predicted trajectory using recursive model predictions.

    Parameters:
        model (keras.Model): Trained prediction model
        initial_condition (np.ndarray): Initial state with shape (1, 3)
        num_steps (int): Number of time steps to predict

    Returns:
        np.ndarray: Predicted trajectory with shape (num_steps + 1, 3)
    """
    # Initialize trajectory array
    trajectory = np.zeros((num_steps + 1, 3))
    trajectory[0] = initial_condition[0]

    # Current state for recursive prediction
    current_state = initial_condition.copy()

    # Predict recursively
    for step in range(num_steps):
        # Predict next state
        next_state = model.predict(current_state, verbose=0)

        # Store prediction
        trajectory[step + 1] = next_state[0]

        # Update current state for next iteration
        current_state = next_state

    return trajectory


def save_trajectory_to_csv(trajectory, filepath, trajectory_type='predicted'):
    """
    Save trajectory data to CSV file.

    Parameters:
        trajectory (np.ndarray): Trajectory data with shape (n_points, 3)
        filepath (str): Output file path
        trajectory_type (str): Type descriptor for output
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(filepath)
    if output_dir:
        ensure_directory_exists(output_dir)

    # Save to CSV
    with open(filepath, 'w') as f:
        # Write header
        f.write('x,y,z\n')

        # Write data
        for point in trajectory:
            f.write(f'{point[0]},{point[1]},{point[2]}\n')


def process_single_initial_condition(ic_data):
    """
    Process a single initial condition: compute ground truth and prediction.

    Computes both the ground truth trajectory (using RK45) and the model
    prediction (recursive), then saves both to CSV files.

    Parameters:
        ic_data (tuple): Tuple containing:
            - ic_idx (int): Initial condition index
            - initial_condition (np.ndarray): Initial state
            - model (keras.Model): Trained model
            - t (np.ndarray): Time points
            - t_span (tuple): Time span
            - lorenz_params (tuple): Lorenz parameters
            - output_dir (str): Output directory
            - model_name (str): Model identifier

    Returns:
        tuple: (ic_idx, success) where success is True if processing completed
    """
    (ic_idx, initial_condition, model, t, t_span, lorenz_params,
     output_dir, model_name) = ic_data

    try:
        print(f'\nProcessing initial condition {ic_idx}')
        print(f'IC: {initial_condition[0]}')

        # Get timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d")

        # Prepare IC string for filename (clean format)
        ic_str = f"{initial_condition[0][0]:.6f}_{initial_condition[0][1]:.6f}_{initial_condition[0][2]:.6f}"

        # Compute ground truth trajectory
        print(f'Computing ground truth trajectory (RK45)...')
        ground_truth = compute_ground_truth_trajectory(
            initial_condition[0], t, t_span, lorenz_params
        )

        # Save ground truth
        gt_filename = f'lorenz63_IC{ic_idx:04d}_groundtruth_{timestamp}_{model_name}.csv'
        gt_filepath = os.path.join(output_dir, gt_filename)
        save_trajectory_to_csv(ground_truth, gt_filepath, 'ground_truth')
        print(f'Saved ground truth: {gt_filename}')

        # Small delay to avoid potential resource conflicts
        time.sleep(1)

        # Predict trajectory using model
        print(f'Computing predicted trajectory (recursive NN)...')
        predicted = predict_trajectory_recursively(
            model, initial_condition, len(t) - 1
        )

        # Save prediction
        pred_filename = f'lorenz63_IC{ic_idx:04d}_predicted_{timestamp}_{model_name}.csv'
        pred_filepath = os.path.join(output_dir, pred_filename)
        save_trajectory_to_csv(predicted, pred_filepath, 'predicted')
        print(f'Saved prediction: {pred_filename}')

        print(f'Completed processing IC {ic_idx}')

        return ic_idx, True

    except Exception as e:
        print(f'Error processing IC {ic_idx}: {e}')
        return ic_idx, False


def parallel_prediction_worker(worker_data):
    """
    Worker function for parallel processing of initial conditions.

    Parameters:
        worker_data (tuple): Tuple containing:
            - worker_id (int): Worker identifier
            - ic_chunk (list): Chunk of initial conditions to process
            - (various other parameters passed through)

    Returns:
        dict: Results dictionary with worker_id and list of (ic_idx, success) tuples
    """
    (worker_id, ic_chunk, model, t, t_span, lorenz_params,
     output_dir, model_name) = worker_data

    print(f'\nWorker {worker_id} started with {len(ic_chunk)} initial conditions')

    results = []

    for ic_data in tqdm(ic_chunk, desc=f"Worker {worker_id}", position=worker_id):
        ic_idx, initial_condition = ic_data

        # Prepare data for processing
        process_data = (ic_idx, initial_condition, model, t, t_span,
                        lorenz_params, output_dir, model_name)

        # Process this initial condition
        result = process_single_initial_condition(process_data)
        results.append(result)

    print(f'\nWorker {worker_id} completed')

    return {'worker_id': worker_id, 'results': results}


def run_parallel_predictions(initial_conditions, model, t, t_span, lorenz_params,
                             output_dir, model_name, num_processes):
    """
    Run predictions for multiple initial conditions in parallel.

    Parameters:
        initial_conditions (np.ndarray): Array of initial conditions
        model (keras.Model): Trained model
        t (np.ndarray): Time points
        t_span (tuple): Time span
        lorenz_params (tuple): Lorenz parameters
        output_dir (str): Output directory
        model_name (str): Model identifier
        num_processes (int): Number of parallel processes

    Returns:
        list: List of (ic_idx, success) results
    """
    print(f'\n{"=" * 80}')
    print(f'Running Parallel Predictions')
    print(f'Total initial conditions: {len(initial_conditions)}')
    print(f'Parallel processes: {num_processes}')
    print(f'{"=" * 80}\n')

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Create enumerated list of initial conditions
    ic_list = [(i, ic) for i, ic in enumerate(initial_conditions)]

    # Split into chunks for parallel processing
    chunks = np.array_split(ic_list, num_processes)

    # Prepare worker data
    worker_data_list = [
        (i, chunks[i], model, t, t_span, lorenz_params, output_dir, model_name)
        for i in range(len(chunks))
    ]

    # Create process pool
    manager = mp.Manager()
    worker_pool = []

    for worker_data in worker_data_list:
        p = Process(target=parallel_prediction_worker, args=(worker_data,))
        worker_pool.append(p)
        p.start()

    # Wait for all workers to complete
    for p in worker_pool:
        p.join()

    print(f'\n{"=" * 80}')
    print('All Predictions Complete')
    print(f'{"=" * 80}')


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Configuration
    model_file = args["model_file"]
    model_name = args["model_name"]
    ic_file = args["initial_conditions"]
    num_trajectories = args["num_trajectories"]
    num_processes = args["num_processes"]
    integration_time = args["integration_time"]
    perturbation = args["perturbation"]
    base_output_dir = args["output_directory"]

    # Lorenz system parameters
    dt = 0.01
    T = integration_time
    t = np.arange(0, T + dt, dt)
    t_span = (0.0, T + dt)

    sigma = 10.0  # Prandtl number
    beta = 8.0 / 3.0  # Geometrical factor
    rho = 28.0  # Rayleigh number
    lorenz_params = (sigma, beta, rho)

    # Setup paths
    model_dir = './_Models/'
    data_dir = './_DataFiles/'
    model_path = os.path.join(model_dir, model_file)
    ic_path = os.path.join(data_dir, ic_file)

    # Create output directory
    perturbation_str = f'{perturbation:.0e}'.replace('-', 'minus').replace('+', 'plus')
    output_dir = os.path.join(base_output_dir, f'{model_name}_{perturbation_str}')
    ensure_directory_exists(output_dir)

    print(f'\n{"=" * 80}')
    print('Neural Network Prediction for Lorenz System')
    print(f'{"=" * 80}')
    print(f'Model: {model_name}')
    print(f'Integration time: T = {T}')
    print(f'Time step: dt = {dt}')
    print(f'Number of trajectories: {num_trajectories}')
    print(f'Perturbation: {perturbation:.2e}')
    print(f'Parallel processes: {num_processes}')
    print(f'Output directory: {output_dir}')
    print(f'{"=" * 80}\n')

    # Load model
    model = load_trained_model(model_path)

    # Load and perturb initial conditions
    initial_conditions = load_initial_conditions(ic_path, perturbation)

    # Limit to requested number of trajectories
    if len(initial_conditions) > num_trajectories:
        initial_conditions = initial_conditions[:num_trajectories]
        print(f'Limited to first {num_trajectories} initial conditions')

    # Run parallel predictions
    run_parallel_predictions(
        initial_conditions, model, t, t_span, lorenz_params,
        output_dir, model_name, num_processes
    )

    print(f'\n{"=" * 80}')
    print('Processing Complete')
    print(f'Results saved to: {output_dir}')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()