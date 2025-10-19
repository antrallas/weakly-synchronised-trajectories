"""
Neural Network Hyperparameter Tuning for Lorenz System

This script performs hyperparameter optimization for an Artificial Neural Network (ANN)
designed to learn the dynamics of the Lorenz system. The optimization uses Keras Tuner
with the Hyperband algorithm to search for optimal network architectures and training
parameters.

Key features:
- Hyperband optimization for efficient hyperparameter search
- Multiple independent tuning iterations for robustness
- Parallel execution of tuning iterations
- Early stopping to prevent overfitting
- Comprehensive hyperparameter space including:
  * Network architecture (layer sizes, activations)
  * Optimization algorithms and learning rates

Example arguments:
-i IC_x0_1000_TTs.json -d GridSearchTuningOutput_1000 -p ann_tuning_1000-Run -r _results-ann_tuning-1000-Run- -n 8 -m 10   

Author: Anthony S. Miller, University of Essex
Date: 17/10/2025
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras_tuner import HyperModel
import keras_tuner as kt
from keras.losses import MeanSquaredError, MeanSquaredLogarithmicError, MeanAbsoluteError
from keras.callbacks import EarlyStopping
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
import os
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.simplefilter("ignore")
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class NumpyArrayEncoder(json.JSONEncoder):
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
        description="Hyperparameter tuning for Lorenz system ANN"
    )
    parser.add_argument(
        "-i", "--TrajectoryDataFile",
        default="IC_x0_10_TTs.json",
        type=str,
        help="Input file containing training trajectories"
    )
    parser.add_argument(
        "-d", "--GridTuningOutput",
        default="GridSearchTuningOutput_10",
        type=str,
        help="Directory for tuning output"
    )
    parser.add_argument(
        "-p", "--ProjectName",
        default="ann_tuning",
        type=str,
        help="Base project name for tuning runs"
    )
    parser.add_argument(
        "-r", "--ResultsFileName",
        default="results_ann_tuning",
        type=str,
        help="Base results filename for tuning runs"
    )
    parser.add_argument(
        "-n", "--num_processes",
        default=None,
        type=int,
        help="Number of parallel processes for tuning iterations (default: number of CPU cores)"
    )
    parser.add_argument(
        "-m", "--max_iterations",
        default=10,
        type=int,
        help="Number of independent tuning iterations to run"
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


class ANNHyperModel(HyperModel):
    """
    Hyperparameter model for the Lorenz system ANN.
    
    This class defines the search space for neural network architecture
    and training parameters. The network has two hidden layers with
    tunable sizes and activation functions, followed by a linear output
    layer for regression.
    """
    
    def build(self, hp):
        """
        Build a Keras model with hyperparameters to tune.
        
        Tunable hyperparameters:
        - layer_1_units: Number of units in first hidden layer (5-30, step 5)
        - layer_2_units: Number of units in second hidden layer (5-30, step 5)
        - layer_1_activation: Activation function for first layer
        - layer_2_activation: Activation function for second layer
        - learning_rate: Learning rate for optimizer (1e-2, 1e-3, 1e-4)
        - optimizer: Optimization algorithm (SGD, RMSprop, Adam, etc.)
        
        Parameters:
            hp: HyperParameters object from Keras Tuner
            
        Returns:
            tf.keras.Model: Compiled Keras model
        """
        model = tf.keras.Sequential()
        
        # Tune the number of units in each hidden layer
        hp_units_1 = hp.Int('layer_1_units', min_value=5, max_value=30, step=5)
        hp_units_2 = hp.Int('layer_2_units', min_value=5, max_value=30, step=5)
        
        # Tune the activation functions
        activation_1 = hp.Choice("layer_1_activation", ["tanh", "relu", "softmax", "sigmoid"])
        activation_2 = hp.Choice("layer_2_activation", ["tanh", "relu", "softmax", "sigmoid"])
        
        # First hidden layer
        model.add(tf.keras.layers.Dense(
            units=hp_units_1,
            input_dim=3,  # Lorenz system has 3 state variables (x, y, z)
            activation=activation_1,
        ))
        
        # Second hidden layer
        model.add(tf.keras.layers.Dense(
            units=hp_units_2,
            activation=activation_2,
        ))
        
        # Output layer (must be linear for regression)
        model.add(tf.keras.layers.Dense(units=3, activation="linear"))
        
        # Tune the learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        # Tune the optimizer
        optimizer_name = hp.Choice('optimizer', 
                                   values=['sgd', 'rmsprop', 'adam', 'adamw', 
                                          'nadam', 'adamax', 'adadelta'])
        
        # Create optimizer instance with tuned learning rate
        optimizer_map = {
            'sgd': tf.keras.optimizers.SGD,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'adam': tf.keras.optimizers.Adam,
            'adamw': tf.keras.optimizers.AdamW,
            'nadam': tf.keras.optimizers.Nadam,
            'adamax': tf.keras.optimizers.Adamax,
            'adadelta': tf.keras.optimizers.Adadelta
        }
        
        optimizer = optimizer_map[optimizer_name](learning_rate=hp_learning_rate)
        
        # Compile model with Mean Squared Error loss
        model.compile(
            optimizer=optimizer,
            loss=MeanSquaredError(),
            metrics=[MeanSquaredError()]
        )
        
        return model


def load_trajectory_data(filename):
    """
    Load training trajectories from JSON file.
    
    Parameters:
        filename (str): Path to trajectory data file
        
    Returns:
        dict: Dictionary mapping trajectory indices to trajectory arrays
    """
    print(f'Reading training trajectories from: {filename}')
    with open(filename, 'r') as f:
        trajectories = json.load(f)
    print(f'Loaded {len(trajectories)} trajectories')
    return trajectories


def prepare_training_data(trajectories, dt=0.01, T=20):
    """
    Prepare training data from trajectories for the neural network.
    
    The neural network learns a one-step prediction: given state x(t),
    predict state x(t + dt). This function transforms trajectory data
    into input-output pairs suitable for supervised learning.
    
    Parameters:
        trajectories (dict): Dictionary of trajectory arrays
        dt (float): Integration time step
        T (float): Total integration time
        
    Returns:
        tuple: (nn_input, nn_output) arrays with shape (n_samples, 3)
    """
    print('Preparing training data...')
    
    t = np.arange(0, T + dt, dt)
    num_trajectories = len(trajectories)
    time_steps_per_traj = len(t) - 1
    
    # Preallocate arrays
    nn_input = np.zeros((num_trajectories * time_steps_per_traj, 3))
    nn_output = np.zeros_like(nn_input)
    
    # Convert trajectories to input-output pairs
    for idx in range(num_trajectories):
        trajectory = np.asarray(trajectories[str(idx)])
        
        start_idx = idx * time_steps_per_traj
        end_idx = (idx + 1) * time_steps_per_traj
        
        # Input: x(t), Output: x(t + dt)
        nn_input[start_idx:end_idx, :] = trajectory[:-1, :]
        nn_output[start_idx:end_idx, :] = trajectory[1:, :]
    
    print(f'Prepared {len(nn_input)} training samples')
    return nn_input, nn_output


def train_val_split(input_data, output_data, train_fraction=0.8):
    """
    Split data into training and validation sets.
    
    Parameters:
        input_data (np.ndarray): Input features
        output_data (np.ndarray): Output targets
        train_fraction (float): Fraction of data to use for training (default: 0.8)
        
    Returns:
        tuple: (input_train, output_train, input_val, output_val)
    """
    split_idx = int(len(input_data) * train_fraction)
    
    input_train = input_data[:split_idx]
    output_train = output_data[:split_idx]
    input_val = input_data[split_idx:]
    output_val = output_data[split_idx:]
    
    print(f'Training set: {len(input_train)} samples')
    print(f'Validation set: {len(input_val)} samples')
    
    return input_train, output_train, input_val, output_val


def run_single_tuning_iteration(iteration_params):
    """
    Run a single hyperparameter tuning iteration.
    
    This function is designed to be called in parallel. Each iteration
    performs an independent Hyperband search to find optimal hyperparameters.
    
    Parameters:
        iteration_params (tuple): Tuple containing:
            - iteration (int): Iteration number
            - nn_input_train (np.ndarray): Training input data
            - nn_output_train (np.ndarray): Training output data
            - nn_input_val (np.ndarray): Validation input data
            - nn_output_val (np.ndarray): Validation output data
            - tuning_output_dir (str): Directory for tuning output
            - project_name (str): Base project name
            - results_filename (str): Base results filename
            - max_epochs (int): Maximum epochs for tuning
            - hyperband_iterations (int): Number of Hyperband iterations
            - batch_size (int): Training batch size
            - early_stopping_patience (int): Patience for early stopping
            
    Returns:
        tuple: (iteration, results_df, best_hyperparameters)
    """
    (iteration, nn_input_train, nn_output_train, nn_input_val, nn_output_val,
     tuning_output_dir, project_name, results_filename, max_epochs,
     hyperband_iterations, batch_size, early_stopping_patience) = iteration_params
    
    print(f'\n{"=" * 50}')
    print(f'Starting Tuning Iteration {iteration}')
    print(f'{"=" * 50}')
    
    # Create project name for this iteration
    iteration_project_name = f"{project_name}_iter{iteration}"
    
    # Initialize hypermodel
    hypermodel = ANNHyperModel()
    
    # Initialize Hyperband tuner
    tuner = kt.Hyperband(
        hypermodel,
        objective='val_mean_squared_error',
        max_epochs=max_epochs,
        factor=3,
        directory=tuning_output_dir + 'hp_tuning',
        project_name=iteration_project_name,
        hyperband_iterations=hyperband_iterations,
        overwrite=False  # Don't overwrite previous runs
    )
    
    # Setup early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
        start_from_epoch=10
    )
    
    # Run hyperparameter search
    print(f'Running Hyperband search for iteration {iteration}...')
    tuner.search(
        nn_input_train, nn_output_train,
        epochs=365,
        validation_data=(nn_input_val, nn_output_val),
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Get best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    
    print(f'\nBest hyperparameters for iteration {iteration}:')
    for param_name in (['layer_1_units', 'layer_2_units'] +
                       ['layer_1_activation', 'layer_2_activation'] +
                       ['learning_rate', 'optimizer']):
        print(f'  {param_name}: {best_hyperparameters.get(param_name)}')
    
    # Get best trials and save results
    trials = tuner.oracle.get_best_trials(num_trials=30)
    
    results = []
    for trial in trials:
        trial_results = trial.hyperparameters.get_config()["values"].copy()
        trial_results["Score"] = trial.score
        results.append(trial_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_filepath = f"{tuning_output_dir}hp_tuning/{results_filename}_iter{iteration}.csv"
    results_df.to_csv(results_filepath, index=False, na_rep='NaN')
    print(f'Saved results to: {results_filepath}')
    
    print(f'Completed iteration {iteration}')
    
    return iteration, results_df, best_hyperparameters


def run_parallel_tuning(nn_input_train, nn_output_train, nn_input_val, nn_output_val,
                       tuning_output_dir, project_name, results_filename,
                       max_iterations, num_processes):
    """
    Run multiple hyperparameter tuning iterations in parallel.
    
    Each iteration performs an independent search, allowing for exploration
    of the hyperparameter space and providing multiple candidate models.
    
    Parameters:
        nn_input_train (np.ndarray): Training input data
        nn_output_train (np.ndarray): Training output data
        nn_input_val (np.ndarray): Validation input data
        nn_output_val (np.ndarray): Validation output data
        tuning_output_dir (str): Directory for tuning output
        project_name (str): Base project name
        results_filename (str): Base results filename
        max_iterations (int): Number of tuning iterations
        num_processes (int): Number of parallel processes
        
    Returns:
        list: List of (iteration, results_df, best_hyperparameters) tuples
    """
    print(f'\n{"=" * 50}')
    print(f'Running {max_iterations} Tuning Iterations in Parallel')
    print(f'Using {num_processes} parallel processes')
    print(f'{"=" * 50}\n')
    
    # Tuning configuration
    MAX_EPOCHS = 50
    HYPERBAND_ITERATIONS = 3
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 5
    
    # Prepare parameters for each iteration
    iteration_params_list = [
        (i, nn_input_train, nn_output_train, nn_input_val, nn_output_val,
         tuning_output_dir, project_name, results_filename, MAX_EPOCHS,
         HYPERBAND_ITERATIONS, BATCH_SIZE, EARLY_STOPPING_PATIENCE)
        for i in range(max_iterations)
    ]
    
    # Run iterations in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_single_tuning_iteration, iteration_params_list)
    
    print(f'\n{"=" * 50}')
    print('All Tuning Iterations Complete')
    print(f'{"=" * 50}')
    
    return results


def summarize_tuning_results(results, tuning_output_dir):
    """
    Summarize results across all tuning iterations.
    
    Parameters:
        results (list): List of (iteration, results_df, best_hyperparameters) tuples
        tuning_output_dir (str): Directory for tuning output
    """
    print('\n' + '=' * 50)
    print('Summary of Best Hyperparameters Across Iterations')
    print('=' * 50)
    
    # Collect all best hyperparameters
    all_best_params = []
    for iteration, results_df, best_hp in results:
        params = {
            'iteration': iteration,
            'layer_1_units': best_hp.get('layer_1_units'),
            'layer_2_units': best_hp.get('layer_2_units'),
            'layer_1_activation': best_hp.get('layer_1_activation'),
            'layer_2_activation': best_hp.get('layer_2_activation'),
            'learning_rate': best_hp.get('learning_rate'),
            'optimizer': best_hp.get('optimizer'),
            'best_score': results_df['Score'].min()
        }
        all_best_params.append(params)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_best_params)
    
    # Save summary
    summary_filepath = f"{tuning_output_dir}hp_tuning/tuning_summary.csv"
    summary_df.to_csv(summary_filepath, index=False)
    print(f'Saved summary to: {summary_filepath}')
    
    # Print summary statistics
    print('\nBest overall score:', summary_df['best_score'].min())
    print('Mean score across iterations:', summary_df['best_score'].mean())
    print('Std dev of scores:', summary_df['best_score'].std())
    
    print('\nMost common hyperparameters:')
    for col in ['layer_1_units', 'layer_2_units', 'layer_1_activation', 
                'layer_2_activation', 'optimizer']:
        if col in summary_df.columns:
            mode_value = summary_df[col].mode()[0]
            print(f'  {col}: {mode_value}')


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configuration: file paths
    data_dir = './_DataFiles/'
    data_file = data_dir + args["TrajectoryDataFile"]
    tuning_output_dir = './_GridSearching/' + args["GridTuningOutput"] + '/'
    
    ensure_directory_exists(tuning_output_dir)
    ensure_directory_exists(tuning_output_dir + 'hp_tuning')
    
    project_name = args["ProjectName"]
    results_filename = args["ResultsFileName"]
    max_iterations = args["max_iterations"]
    
    # Determine number of parallel processes
    if args["num_processes"] is None:
        num_processes = min(cpu_count(), max_iterations)  # Don't use more processes than iterations
    else:
        num_processes = min(args["num_processes"], cpu_count(), max_iterations)
    
    # Load and prepare data
    trajectories = load_trajectory_data(data_file)
    nn_input, nn_output = prepare_training_data(trajectories)
    
    # Split into training and validation sets
    nn_input_train, nn_output_train, nn_input_val, nn_output_val = train_val_split(
        nn_input, nn_output
    )
    
    # Run parallel tuning
    results = run_parallel_tuning(
        nn_input_train, nn_output_train, nn_input_val, nn_output_val,
        tuning_output_dir, project_name, results_filename,
        max_iterations, num_processes
    )
    
    # Summarize results
    summarize_tuning_results(results, tuning_output_dir)
    
    print('\n' + '=' * 50)
    print('Hyperparameter Tuning Complete')
    print('=' * 50)


if __name__ == "__main__":
    main()