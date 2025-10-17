"""
Parallel Lorenz System Trajectory Generator

This script generates training trajectories for the Lorenz system using parallel
processing. The process consists of two phases:

Phase 1: Integrate initial conditions to the attractor and apply perturbations
Phase 2: Generate training trajectories from the perturbed states

Key features:
- Multiprocessing support for parallel trajectory computation
- Batch processing for memory management
- Checkpoint/resume capability for long-running jobs
- Flexible phase execution (run phases independently or together)

Example arguments:
-i IC_x0_10_baseline.json -p IC_x0_10_baseline_LP_P.json -o IC_x0_10_baseline_TTs.json -n 2 -b 10 --phase both --checkpoint baseline_checkpoint

Author: Anthony S. Miller, University of Essex
Date: 17/10/2025
"""

import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
import time
import json
from json import JSONEncoder
from multiprocessing import Pool, cpu_count
import os
import sys

warnings.simplefilter("ignore")


class NumpyArrayEncoder(JSONEncoder):
    """
    Custom JSON encoder for NumPy arrays.

    Converts NumPy arrays to Python lists for JSON serialization.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        dict: Dictionary containing parsed arguments
    """
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Generate Lorenz system trajectories using parallel processing"
    )
    parser.add_argument(
        "-i", "--inputfile",
        default="IC_x0_100_baseline.json",
        type=str,
        help="Input file containing initial conditions"
    )
    parser.add_argument(
        "-o", "--outputfilename",
        default="IC_x0_100_TTs.json",
        type=str,
        help="Output file for training trajectories"
    )
    parser.add_argument(
        "-p", "--perturbedfilename",
        default="IC_x0_100_LP_P.json",
        type=str,
        help="Output file for perturbed initial conditions on attractor"
    )
    parser.add_argument(
        "-n", "--num_processes",
        default=None,
        type=int,
        help="Number of parallel processes (default: number of CPU cores)"
    )
    parser.add_argument(
        "-b", "--batch_size",
        default=100,
        type=int,
        help="Batch size for processing trajectories (for memory management)"
    )
    parser.add_argument(
        "--phase",
        default="both",
        choices=["1", "2", "both"],
        help="Which phase to run: 1 (attractor only), 2 (training only), or both"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        nargs='?',
        const='default',
        help="Enable checkpointing with optional base filename (e.g., --checkpoint or --checkpoint job123)"
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


def lorenz(t, state, sigma, beta, rho):
    """
    Lorenz system differential equations.

    Parameters:
        t (float): Time (not used, required by solve_ivp interface)
        state (array-like): Current state [x, y, z]
        sigma (float): Prandtl number (ratio of momentum to thermal diffusivity)
        beta (float): Geometrical factor related to physical dimensions
        rho (float): Rayleigh number (related to temperature difference)

    Returns:
        list: Time derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return [dx, dy, dz]


def compute_attractor_trajectory(args):
    """
    Compute a single trajectory to the attractor and return the perturbed final state.

    This function is designed to be called in parallel for each initial condition.
    It integrates the Lorenz system for a long time to ensure convergence to the
    attractor, then applies a small perturbation to the x-component.

    Parameters:
        args (tuple): Tuple of (x0, t1_span, t1, lorenz_params, perturb_magnitude)

    Returns:
        list or None: Perturbed final state [x + δ, y, z], or None if computation fails
    """
    x0, t1_span, t1, lorenz_params, perturb_magnitude = args

    try:
        # Integrate to attractor
        solution = solve_ivp(lorenz, t1_span, x0, args=lorenz_params, method='RK45', t_eval=t1)

        # Extract final state
        final_state = solution.y[:, -1]

        # Perturb x-component only
        perturbed_state = [
            final_state[0] + perturb_magnitude,
            final_state[1],
            final_state[2]
        ]

        return perturbed_state
    except Exception as e:
        print(f"Error in compute_attractor_trajectory: {e}")
        return None


def compute_training_trajectory(args):
    """
    Compute a single training trajectory from a perturbed initial condition.

    This function is designed to be called in parallel. It takes an indexed
    initial condition and returns the trajectory with its index for proper ordering.

    Parameters:
        args (tuple): Tuple of (trajectory_index, x0, t2_span, t2, lorenz_params)

    Returns:
        tuple: (trajectory_index, trajectory_data) or (trajectory_index, None) if failed
    """
    traj_idx, x0, t2_span, t2, lorenz_params = args

    try:
        # Integrate for training period
        solution = solve_ivp(lorenz, t2_span, x0, args=lorenz_params, method='RK45', t_eval=t2)

        # Return index and transposed trajectory (n_points × 3)
        return (traj_idx, solution.y[:, :].T)
    except Exception as e:
        print(f"Error in compute_training_trajectory for index {traj_idx}: {e}")
        return (traj_idx, None)


def save_checkpoint(data, filename):
    """
    Save checkpoint data to file for resume capability.

    Parameters:
        data: Data to checkpoint (typically dict or list)
        filename (str): Checkpoint file path
    """
    with open(filename, 'w') as f:
        json.dump(data, f, cls=NumpyArrayEncoder)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename):
    """
    Load checkpoint data from file if it exists.

    Parameters:
        filename (str): Checkpoint file path

    Returns:
        dict or None: Checkpoint data if file exists, None otherwise
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def load_initial_conditions(filename):
    """
    Load initial conditions from JSON file.

    Parameters:
        filename (str): Path to JSON file containing initial conditions

    Returns:
        np.ndarray: Array of initial conditions with shape (n_conditions, 3)
    """
    print(f'Reading initial conditions from: {filename}')
    with open(filename, 'r') as f:
        data = json.load(f)
        return np.asarray(data["ICs"])


def save_json(data, filename, data_key):
    """
    Save data to JSON file with custom numpy encoder.

    Parameters:
        data: Data to save (will be wrapped in dictionary)
        filename (str): Output filename
        data_key (str): Key name for data in JSON
    """
    print(f'Saving to: {filename}')
    encoded_data = {data_key: data}
    with open(filename, 'w') as f:
        json.dump(encoded_data, f, cls=NumpyArrayEncoder)
    print(f"Successfully saved to: {filename}")


def verify_file(filename, expected_count, data_key):
    """
    Verify that a file was written correctly.

    Parameters:
        filename (str): Path to file to verify
        expected_count (int): Expected number of items
        data_key (str): Key to check in JSON file

    Returns:
        bool: True if verification successful, False otherwise
    """
    print("Verifying saved file...")
    with open(filename, 'r') as f:
        verify_data = json.load(f)
        verify_count = len(verify_data[data_key])

    if verify_count == expected_count:
        print(f"✓ File verification successful: {verify_count} items confirmed")
        return True
    else:
        print(f"✗ WARNING: File verification failed! Expected {expected_count}, found {verify_count}")
        return False


def phase_1_attractor_integration(initial_conditions, args, num_processes,
                                  t1_span, t1, lorenz_params, perturb_magnitude,
                                  checkpoint_file):
    """
    Phase 1: Integrate initial conditions to attractor and apply perturbations.

    This phase processes all initial conditions in parallel, integrating each for
    a long time to ensure convergence to the Lorenz attractor. The final states
    are perturbed and saved for use in Phase 2.

    Parameters:
        initial_conditions (np.ndarray): Array of initial conditions
        args (dict): Command line arguments
        num_processes (int): Number of parallel processes to use
        t1_span (tuple): Time span for integration
        t1 (np.ndarray): Time evaluation points
        lorenz_params (tuple): Lorenz system parameters (sigma, beta, rho)
        perturb_magnitude (float): Perturbation magnitude
        checkpoint_file (str): Path to checkpoint file

    Returns:
        tuple: (perturbed_states, computation_time)
    """
    print("\n" + "=" * 50)
    print("PHASE 1: Computing Attractor Trajectories")
    print("=" * 50)

    print(f'Total ICs to process: {len(initial_conditions)}')
    print(f'Generating trajectories of {len(t1)} time points using {num_processes} processes...')

    tic = time.perf_counter()

    # Initialize checkpoint handling
    perturbed_states = []
    start_idx = 0

    # Resume from checkpoint if available
    if args["checkpoint"]:
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            perturbed_states = checkpoint_data.get("perturbed_x0s", [])
            start_idx = len(perturbed_states)
            print(f"Resuming from checkpoint: {start_idx} trajectories already completed")

    # Process trajectories in batches for memory management
    batch_size = args["batch_size"]
    total_batches = (len(initial_conditions) + batch_size - 1) // batch_size

    for i in range(start_idx, len(initial_conditions), batch_size):
        batch_ics = initial_conditions[i:i + batch_size]
        batch_num = i // batch_size + 1
        end_idx = min(i + batch_size - 1, len(initial_conditions) - 1)

        print(f"\nProcessing batch {batch_num}/{total_batches} (indices {i}-{end_idx})")

        # Prepare arguments for parallel processing
        batch_args = [(x0, t1_span, t1, lorenz_params, perturb_magnitude) for x0 in batch_ics]

        # Parallel computation of attractor trajectories
        with Pool(processes=num_processes) as pool:
            batch_results = list(tqdm(
                pool.imap(compute_attractor_trajectory, batch_args),
                total=len(batch_args),
                desc="Computing attractor trajectories"
            ))

        # Filter out failed computations
        batch_results = [r for r in batch_results if r is not None]
        perturbed_states.extend(batch_results)

        # Save checkpoint periodically
        if args["checkpoint"] and (batch_num % 5 == 0 or i + batch_size >= len(initial_conditions)):
            save_checkpoint({"perturbed_x0s": perturbed_states}, checkpoint_file)

    toc = time.perf_counter()
    computation_time = toc - tic

    print(f"\nPhase 1 Complete!")
    print(f"Computed {len(perturbed_states)} attractor trajectories in {computation_time:.4f} seconds")
    print(f"Speed improvement: ~{num_processes}x faster than serial processing")

    return perturbed_states, computation_time


def phase_2_training_trajectories(perturbed_states, args, num_processes,
                                  t2_span, t2, lorenz_params, checkpoint_file):
    """
    Phase 2: Generate training trajectories from perturbed initial conditions.

    This phase takes the perturbed states from Phase 1 (which are already on the
    attractor) and integrates them for a shorter time period to generate training
    data. Processing is done in parallel with checkpoint support.

    Parameters:
        perturbed_states (np.ndarray): Array of perturbed initial conditions
        args (dict): Command line arguments
        num_processes (int): Number of parallel processes to use
        t2_span (tuple): Time span for integration
        t2 (np.ndarray): Time evaluation points
        lorenz_params (tuple): Lorenz system parameters (sigma, beta, rho)
        checkpoint_file (str): Path to checkpoint file

    Returns:
        tuple: (trajectories, computation_time)
    """
    print("\n" + "=" * 50)
    print("PHASE 2: Computing Training Trajectories")
    print("=" * 50)

    print(f'Processing {len(perturbed_states)} perturbed ICs')
    print(f'Generating training trajectories using {num_processes} processes...')

    tic = time.perf_counter()

    # Initialize checkpoint handling
    trajectories = {}
    start_idx = 0

    # Resume from checkpoint if available
    if args["checkpoint"]:
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            trajectories = checkpoint_data
            start_idx = len(trajectories)
            print(f"Resuming from checkpoint: {start_idx} trajectories already completed")

    # Prepare indexed data for parallel processing
    indexed_data = list(enumerate(perturbed_states))[start_idx:]

    # Process trajectories in batches
    batch_size = args["batch_size"]
    total_batches = (len(indexed_data) + batch_size - 1) // batch_size

    for i in range(0, len(indexed_data), batch_size):
        batch = indexed_data[i:i + batch_size]
        batch_num = i // batch_size + 1
        current_indices = [idx for idx, _ in batch]

        print(f"\nProcessing batch {batch_num}/{total_batches} "
              f"(trajectory indices {min(current_indices)}-{max(current_indices)})")

        # Prepare arguments for parallel processing
        batch_args = [(idx, x0, t2_span, t2, lorenz_params) for idx, x0 in batch]

        # Parallel computation of training trajectories
        with Pool(processes=num_processes) as pool:
            batch_results = list(tqdm(
                pool.imap(compute_training_trajectory, batch_args),
                total=len(batch_args),
                desc="Computing training trajectories"
            ))

        # Store results, filtering out failed computations
        for traj_idx, traj_data in batch_results:
            if traj_data is not None:
                trajectories[traj_idx] = traj_data

        # Save checkpoint periodically
        if args["checkpoint"] and (batch_num % 5 == 0 or i + batch_size >= len(indexed_data)):
            save_checkpoint(trajectories, checkpoint_file)

    toc = time.perf_counter()
    computation_time = toc - tic

    print(f"\nPhase 2 Complete!")
    print(f"Computed {len(trajectories)} training trajectories in {computation_time:.4f} seconds")

    return trajectories, computation_time


def cleanup_checkpoints(checkpoint_files):
    """
    Remove checkpoint files after successful completion.

    Parameters:
        checkpoint_files (list): List of checkpoint file paths to remove
    """
    print("\nCleaning up checkpoint files...")
    for checkpoint_file in checkpoint_files:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"Removed: {checkpoint_file}")
    print("Checkpoints cleaned up successfully")


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    # Configuration: file paths
    data_dir = './_DataFiles/'
    ensure_directory_exists(data_dir)

    input_file = data_dir + args["inputfile"]
    output_trajectories_file = data_dir + args["outputfilename"]
    output_perturbed_file = data_dir + args["perturbedfilename"]

    checkpoint_dir = data_dir + 'checkpoints/'
    phase1_checkpoint_file = None
    phase2_checkpoint_file = None

    # Setup checkpoint directory and filenames if needed
    if args["checkpoint"]:
        ensure_directory_exists(checkpoint_dir)

        checkpoint_base = 'checkpoint' if args["checkpoint"] == 'default' else args["checkpoint"]
        phase1_checkpoint_file = checkpoint_dir + f'{checkpoint_base}_phase1.json'
        phase2_checkpoint_file = checkpoint_dir + f'{checkpoint_base}_phase2.json'

        print(f"Checkpoint files: {phase1_checkpoint_file} and {phase2_checkpoint_file}")

    # Configuration: Lorenz system parameters
    sigma = 10.0  # Prandtl number
    beta = 8.0 / 3.0  # Geometrical factor
    rho = 28.0  # Rayleigh number
    lorenz_params = (sigma, beta, rho)

    # Configuration: integration parameters
    dt = 0.01  # Integration time step (0.01 time units)
    T1 = 5000  # Attractor convergence time (5000 time units = 500,000 points)
    T2 = 20  # Training trajectory length (20 time units = 2,000 points)

    t1 = np.arange(0, T1 + dt, dt)  # Time points for attractor integration
    t2 = np.arange(0, T2 + dt, dt)  # Time points for training trajectories

    t1_span = (0.0, T1 + dt)
    t2_span = (0.0, T2 + dt)

    # Configuration: perturbation magnitude
    PERTURB = 10e-2  # 0.1 units perturbation in x-direction

    # Determine number of parallel processes
    if args["num_processes"] is None:
        num_processes = cpu_count()
    else:
        num_processes = min(args["num_processes"], cpu_count())

    print(f"Using {num_processes} parallel processes")
    print(f"Running phase(s): {args['phase']}")

    # Track timing for summary
    phase1_time = 0
    phase2_time = 0

    # ========== PHASE 1: Attractor Integration ==========
    if args["phase"] in ["1", "both"]:
        # Load initial conditions
        initial_conditions = load_initial_conditions(input_file)

        # Compute attractor trajectories and perturbations
        perturbed_states, phase1_time = phase_1_attractor_integration(
            initial_conditions, args, num_processes,
            t1_span, t1, lorenz_params, PERTURB,
            phase1_checkpoint_file
        )

        # Save perturbed states
        print('\nSaving perturbed initial conditions...')
        save_json(perturbed_states, output_perturbed_file, "perturbedICs")

        # Verify the save was successful
        if not verify_file(output_perturbed_file, len(perturbed_states), "perturbedICs"):
            print("ERROR: File verification failed!")
            sys.exit(1)

        print("\n" + "=" * 50)
        print("PHASE 1 COMPLETED - All perturbed ICs saved to disk")
        print("=" * 50)

    # ========== PHASE 2: Training Trajectory Generation ==========
    if args["phase"] in ["2", "both"]:
        # Load perturbed initial conditions
        print(f'\nReading perturbed initial conditions from: {output_perturbed_file}')

        if not os.path.exists(output_perturbed_file):
            print(f"ERROR: Perturbed ICs file not found: {output_perturbed_file}")
            print("Please run Phase 1 first to generate the perturbed initial conditions.")
            sys.exit(1)

        with open(output_perturbed_file, 'r') as f:
            perturbed_data = json.load(f)
            perturbed_states = np.asarray(perturbed_data["perturbedICs"])

        print(f'Successfully loaded {len(perturbed_states)} perturbed ICs')

        # Generate training trajectories
        training_trajectories, phase2_time = phase_2_training_trajectories(
            perturbed_states, args, num_processes,
            t2_span, t2, lorenz_params,
            phase2_checkpoint_file
        )

        # Save training trajectories
        print('\nSaving training trajectories...')
        save_json(training_trajectories, output_trajectories_file, "trajectories")

        # Print summary statistics
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Total trajectories processed: {len(training_trajectories)}")
        print(f"Phase 2 time: {phase2_time:.2f} seconds")

        if args["phase"] == "both":
            print(f"Total time for both phases: {phase1_time + phase2_time:.2f} seconds")

    # Clean up checkpoint files if both phases completed successfully
    if args["checkpoint"] and args["phase"] == "both":
        cleanup_checkpoints([phase1_checkpoint_file, phase2_checkpoint_file])


if __name__ == "__main__":
    main()
