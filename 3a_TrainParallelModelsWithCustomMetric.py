"""
Parallel Neural Network Training with Custom Lyapunov-Based Metric

This script trains multiple neural network models in parallel for the Lorenz system
using a custom metric based on the Lyapunov time horizon. The training process monitors
both Mean Squared Error (MSE) and the prediction horizon, where the prediction horizon
is defined as the time until the relative error exceeds a threshold of 0.4.

Key features:
- Parallel training of multiple model configurations
- Custom metric based on Lyapunov time and prediction horizon
- Configurable model architectures from tuning results
- Batch processing aligned with trajectory length for accurate metrics
- Training history export for analysis

The Lyapunov time is calculated using the largest Lyapunov exponent (LLE = 0.9036)
for the Lorenz system, providing a physically meaningful measure of predictability.

Example arguments:
-f IC_x0_10000_TTs.json -m top10_best_models_from_tuning.json -s A -e 500 -p 6

Author: Anthony S. Miller, University of Essex
Date: 17/10/2025
"""

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
import os
import json
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, Nadam, Adamax, SGD, RMSprop, Adadelta, AdamW
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        dict: Dictionary containing parsed arguments
    """
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Train neural networks in parallel with custom Lyapunov-based metric"
    )
    parser.add_argument(
        "-f", "--trajectoryfile",
        default="IC_x0_1000_TTs.json",
        type=str,
        help="Input file containing training trajectories"
    )
    parser.add_argument(
        "-m", "--modelsfile",
        default="top10MSE.json",
        type=str,
        help="Input file containing model configurations from tuning"
    )
    parser.add_argument(
        "-e", "--epochs",
        default=500,
        type=int,
        help="Maximum number of training epochs (default: 500)"
    )
    parser.add_argument(
        "-t", "--integration_time",
        default=20,
        type=int,
        choices=[20, 60],
        help="Integration time T (20 or 60, determines trajectory length)"
    )
    parser.add_argument(
        "-s", "--start_letter",
        default="A",
        type=str,
        help="Starting letter for model labels (default: 'A')"
    )
    parser.add_argument(
        "-p", "--num_processes",
        default=1,
        type=int,
        help="Number of parallel processes for training models (default: 1 for sequential)"
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


def validate_start_letter(start_letter):
    """
    Validate and convert starting letter to uppercase.

    Parameters:
        start_letter (str): Starting letter for model labeling

    Returns:
        str: Uppercase version of the letter
    """
    if len(start_letter) != 1 or not start_letter.isalpha():
        raise ValueError(f"Start letter must be a single letter A-Z, got: '{start_letter}'")
    return start_letter.upper()


def get_model_letter(model_idx, start_letter='A'):
    """
    Generate model letter label based on index and starting letter.

    Parameters:
        model_idx (int): Model index (0-based)
        start_letter (str): Starting letter (default: 'A')

    Returns:
        str: Model letter label
    """
    start_ord = ord(start_letter.upper())
    target_ord = start_ord + model_idx

    if target_ord <= ord('Z'):
        return chr(target_ord)
    else:
        excess = target_ord - ord('Z') - 1
        first_letter = chr(ord('A') + (excess // 26))
        second_letter = chr(ord('A') + (excess % 26))
        return first_letter + second_letter


def calculate_lyapunov_time_array(t, lle=0.9036):
    """
    Convert time array to Lyapunov time units.

    Lyapunov time provides a natural time scale for chaotic systems,
    representing the time it takes for nearby trajectories to diverge
    by a factor of e.

    Parameters:
        t (np.ndarray): Time array
        lle (float): Largest Lyapunov exponent (default: 0.9036 for Lorenz system)

    Returns:
        np.ndarray: Time array in Lyapunov units
    """
    return t * lle


def get_max_prediction_horizon(integration_time, lle=0.9036):
    """
    Calculate maximum prediction horizon based on integration time.

    The maximum prediction horizon is reached when the entire trajectory
    is predicted without exceeding the error threshold.

    Parameters:
        integration_time (float): Total integration time T
        lle (float): Largest Lyapunov exponent

    Returns:
        float: Maximum prediction horizon in Lyapunov time units
    """
    return integration_time * lle


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


def load_model_configurations(filename):
    """
    Load model configurations from tuning results.

    Parameters:
        filename (str): Path to model configurations JSON file

    Returns:
        list: List of model configurations
    """
    print(f'Reading model configurations from: {filename}')
    with open(filename, 'r') as f:
        data = json.load(f)
        configurations = data["models"]
    print(f'Loaded {len(configurations)} model configurations')
    return configurations


def prepare_training_data(trajectories, dt=0.01, T=20):
    """
    Prepare training data from trajectories.

    Parameters:
        trajectories (dict): Dictionary of trajectory arrays
        dt (float): Integration time step
        T (float): Total integration time

    Returns:
        tuple: (nn_input, nn_output, num_trajectories, time_steps_per_trajectory)
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

        nn_input[start_idx:end_idx, :] = trajectory[:-1, :]
        nn_output[start_idx:end_idx, :] = trajectory[1:, :]

    print(f'Prepared {len(nn_input)} training samples')
    print(f'Batch size (trajectory length): {time_steps_per_traj}')

    return nn_input, nn_output, num_trajectories, time_steps_per_traj


def get_optimizer(optimizer_name, learning_rate):
    """
    Create an optimizer instance with the specified learning rate.

    Parameters:
        optimizer_name (str): Name of the optimizer
        learning_rate (float): Learning rate value

    Returns:
        keras.optimizers.Optimizer: Configured optimizer instance
    """
    optimizer_map = {
        'adam': Adam,
        'nadam': Nadam,
        'adamax': Adamax,
        'sgd': SGD,
        'adadelta': Adadelta,
        'adamw': AdamW,
        'rmsprop': RMSprop
    }

    if optimizer_name.lower() not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer_map[optimizer_name.lower()](learning_rate=learning_rate)


class PredictionHorizonMetric(tf.keras.metrics.Metric):
    """
    Custom metric for calculating prediction horizon based on relative error.

    This metric computes the time (in Lyapunov units) until the relative error
    between predicted and true trajectories exceeds a threshold of 0.4. This
    provides a physically meaningful measure of model performance for chaotic
    systems.

    The prediction horizon is defined as the first time where:
        ||y_true - y_pred|| / ||y_true|| >= 0.4

    Attributes:
        pred_horizon (tf.Variable): Stores the prediction horizon value
        time_array (np.ndarray): Time points for the trajectory
        lyapunov_time_array (np.ndarray): Time points in Lyapunov units
        max_horizon (float): Maximum prediction horizon (when threshold never exceeded)
        error_threshold (float): Relative error threshold (default: 0.4)
    """

    def __init__(self, time_array, lyapunov_time_array, max_horizon,
                 error_threshold=0.4, name='prediction_horizon', **kwargs):
        """
        Initialize the prediction horizon metric.

        Parameters:
            time_array (np.ndarray): Time points for trajectory
            lyapunov_time_array (np.ndarray): Time points in Lyapunov units
            max_horizon (float): Maximum prediction horizon
            error_threshold (float): Relative error threshold (default: 0.4)
            name (str): Metric name
        """
        super().__init__(name=name, **kwargs)
        self.pred_horizon = self.add_variable(
            shape=(),
            initializer='zeros',
            name='pred_horizon'
        )
        self.time_array = time_array
        self.lyapunov_time_array = lyapunov_time_array
        self.max_horizon = max_horizon
        self.error_threshold = error_threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update metric state by computing prediction horizon for current batch.

        Parameters:
            y_true (tf.Tensor): True values
            y_pred (tf.Tensor): Predicted values
            sample_weight: Optional sample weights (unused)
        """
        # Convert tensors to numpy arrays
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()

        # Calculate relative errors for each time step
        relative_errors = [
            np.linalg.norm(true - pred) / np.linalg.norm(true)
            for true, pred in zip(y_true_np, y_pred_np)
        ]

        # Create dataframe with time, Lyapunov time, and errors
        df_error = pd.DataFrame({
            'Time': self.time_array,
            'LyapunovTime': self.lyapunov_time_array,
            'RelativeError': relative_errors
        })

        # Find first point where error exceeds threshold
        exceeds_threshold = df_error[df_error['RelativeError'] >= self.error_threshold]

        if len(exceeds_threshold) > 0:
            # Get Lyapunov time of first exceedance
            horizon = exceeds_threshold.iloc[0]['LyapunovTime']
        else:
            # No exceedance - use maximum horizon
            horizon = self.max_horizon

        # Update metric value
        self.pred_horizon.assign(horizon)

    def result(self):
        """
        Return the current metric value.

        Returns:
            tf.Variable: Prediction horizon value
        """
        return self.pred_horizon

    def reset_states(self):
        """Reset metric state at the start of each epoch."""
        self.pred_horizon.assign(0.0)


def save_training_history(history, model_label, output_dir, epochs, batch_size):
    """
    Save training history to CSV files.

    Parameters:
        history: Keras training history object
        model_label (str): Model identifier
        output_dir (str): Output directory
        epochs (int): Number of epochs trained
        batch_size (int): Batch size used
    """
    ensure_directory_exists(output_dir)

    for metric_name in history.history.keys():
        filename = f'{output_dir}Model{model_label}_{metric_name}_epochs{epochs}_batch{batch_size}.csv'

        with open(filename, 'w') as f:
            # Write header
            f.write(f'epoch,{metric_name}\n')

            # Write data
            for epoch, value in enumerate(history.history[metric_name]):
                f.write(f'{epoch},{value}\n')


def build_and_train_model(model_params):
    """
    Build and train a single model with custom prediction horizon metric.

    This function is designed to be called in parallel. It constructs a model,
    trains it with the custom Lyapunov-based metric, and saves the training history.

    Parameters:
        model_params (tuple): Tuple containing:
            - model_idx (int): Model index
            - config (list): Model configuration
            - nn_input (np.ndarray): Training input data
            - nn_output (np.ndarray): Training output data
            - time_array (np.ndarray): Time points
            - lyapunov_time_array (np.ndarray): Lyapunov time points
            - max_horizon (float): Maximum prediction horizon
            - epochs (int): Maximum training epochs
            - batch_size (int): Batch size
            - output_dir (str): Directory for saving results
            - start_letter (str): Starting letter for model labels

    Returns:
        tuple: (model_idx, model_label, final_loss, final_prediction_horizon)
    """
    (model_idx, config, nn_input, nn_output, time_array, lyapunov_time_array,
     max_horizon, epochs, batch_size, output_dir, start_letter) = model_params

    # Generate model label
    model_label = get_model_letter(model_idx, start_letter)

    print(f'\n{"=" * 80}')
    print(f'Training Model {model_label}')
    print(f'{"=" * 80}')

    # Extract configuration
    layer1_units = int(config[0])
    layer2_units = int(config[1])
    activation_1 = config[2]
    activation_2 = config[3]
    learning_rate = float(config[4])
    optimizer_name = config[5]
    tuning_score = config[6] if len(config) > 6 else None

    print(f'Architecture: [{layer1_units}, {layer2_units}]')
    print(f'Activations: [{activation_1}, {activation_2}]')
    print(f'Optimizer: {optimizer_name} (lr={learning_rate})')
    if tuning_score:
        print(f'Tuning score: {tuning_score:.6e}')

    try:
        # Build model
        model = Sequential([
            Dense(layer1_units, input_dim=3, activation=activation_1),
            Dense(layer2_units, activation=activation_2),
            Dense(3, activation='linear')
        ])

        # Get optimizer
        optimizer = get_optimizer(optimizer_name, learning_rate)

        # Create custom metric
        custom_metric = PredictionHorizonMetric(
            time_array=time_array,
            lyapunov_time_array=lyapunov_time_array,
            max_horizon=max_horizon,
            error_threshold=0.4
        )

        # Compile model
        model.compile(
            loss=MeanSquaredError(),
            optimizer=optimizer,
            metrics=[custom_metric],
            run_eagerly=True  # Required for custom metric
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='loss',
                mode='min',
                patience=5,
                restore_best_weights=True,
                start_from_epoch=10,
                verbose=1
            )
        ]

        # Train model
        print(f'\nTraining for up to {epochs} epochs...')
        history = model.fit(
            nn_input, nn_output,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=False,  # Critical: maintain trajectory order
            verbose=1
        )

        # Display summary
        model.summary()

        # Get final metrics
        final_loss = history.history['loss'][-1]
        final_prediction_horizon = history.history['prediction_horizon'][-1]

        # Save training history
        model_output_dir = f'{output_dir}Model{model_label}/'
        save_training_history(history, model_label, model_output_dir, epochs, batch_size)

        print(f'\nModel {model_label} complete!')
        print(f'Final loss: {final_loss:.6e}')
        print(f'Final prediction horizon: {final_prediction_horizon:.4f} Lyapunov times')

        return model_idx, model_label, final_loss, final_prediction_horizon

    except Exception as e:
        print(f'\nError training Model {model_label}: {e}')
        return model_idx, model_label, None, None


def train_models_sequential(model_configs, nn_input, nn_output, time_array,
                            lyapunov_time_array, max_horizon, epochs, batch_size,
                            output_dir, start_letter):
    """
    Train multiple models sequentially.

    Parameters:
        model_configs (list): List of model configurations
        nn_input (np.ndarray): Training input data
        nn_output (np.ndarray): Training output data
        time_array (np.ndarray): Time points
        lyapunov_time_array (np.ndarray): Lyapunov time points
        max_horizon (float): Maximum prediction horizon
        epochs (int): Maximum training epochs
        batch_size (int): Batch size
        output_dir (str): Output directory
        start_letter (str): Starting letter for model labels

    Returns:
        list: List of (model_idx, model_label, final_loss, final_horizon) tuples
    """
    print(f'\n{"=" * 80}')
    print(f'Training {len(model_configs)} Models Sequentially')
    print(f'Model labels will start from: {start_letter}')
    print(f'{"=" * 80}\n')

    results = []

    for model_idx, config in enumerate(tqdm(model_configs, desc="Training models")):
        params = (model_idx, config, nn_input, nn_output, time_array,
                 lyapunov_time_array, max_horizon, epochs, batch_size,
                 output_dir, start_letter)

        result = build_and_train_model(params)
        results.append(result)

    return results


def train_models_parallel(model_configs, nn_input, nn_output, time_array,
                          lyapunov_time_array, max_horizon, epochs, batch_size,
                          output_dir, start_letter, num_processes):
    """
    Train multiple models in parallel.

    Parameters:
        model_configs (list): List of model configurations
        nn_input (np.ndarray): Training input data
        nn_output (np.ndarray): Training output data
        time_array (np.ndarray): Time points
        lyapunov_time_array (np.ndarray): Lyapunov time points
        max_horizon (float): Maximum prediction horizon
        epochs (int): Maximum training epochs
        batch_size (int): Batch size
        output_dir (str): Output directory
        start_letter (str): Starting letter for model labels
        num_processes (int): Number of parallel processes

    Returns:
        list: List of (model_idx, model_label, final_loss, final_horizon) tuples
    """
    print(f'\n{"=" * 80}')
    print(f'Training {len(model_configs)} Models in Parallel')
    print(f'Using {num_processes} parallel processes')
    print(f'Model labels will start from: {start_letter}')
    print(f'{"=" * 80}\n')

    # Prepare parameters for each model
    model_params_list = []
    for model_idx, config in enumerate(model_configs):
        params = (model_idx, config, nn_input, nn_output, time_array,
                 lyapunov_time_array, max_horizon, epochs, batch_size,
                 output_dir, start_letter)
        model_params_list.append(params)

    # Train models in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(build_and_train_model, model_params_list)

    return results


def save_training_summary(results, output_dir):
    """
    Save summary of training results to CSV.

    Parameters:
        results (list): List of training results
        output_dir (str): Output directory
    """
    summary_data = []
    for model_idx, model_label, final_loss, final_horizon in results:
        summary_data.append({
            'model_index': model_idx,
            'model_label': model_label,
            'final_loss': final_loss,
            'final_prediction_horizon': final_horizon
        })

    summary_df = pd.DataFrame(summary_data)
    summary_filepath = f'{output_dir}training_summary.csv'
    summary_df.to_csv(summary_filepath, index=False)
    print(f'\nSaved training summary to: {summary_filepath}')

    # Print best model by prediction horizon
    valid_results = summary_df[summary_df['final_prediction_horizon'].notna()]
    if len(valid_results) > 0:
        best_idx = valid_results['final_prediction_horizon'].idxmax()
        best_model = valid_results.iloc[best_idx]
        print(f'\nBest model by prediction horizon: {best_model["model_label"]} '
              f'(horizon: {best_model["final_prediction_horizon"]:.4f} Lyapunov times)')


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Configuration: Lorenz system parameters
    LARGEST_LYAPUNOV_EXPONENT = 0.9036
    dt = 0.01
    T = args["integration_time"]

    # Setup directories
    data_dir = './_DataFiles/'
    models_dir = './_TrainModelData/_BestModelFile/'
    output_dir = './_CustomMetricFiles/'

    ensure_directory_exists(output_dir)

    # File paths
    trajectory_file = data_dir + args["trajectoryfile"]
    models_file = models_dir + args["modelsfile"]

    epochs = args["epochs"]
    start_letter = validate_start_letter(args["start_letter"])
    num_processes = args["num_processes"]

    print(f'\n{"=" * 80}')
    print('Neural Network Training with Custom Lyapunov Metric')
    print(f'{"=" * 80}')
    print(f'Integration time: T = {T}')
    print(f'Largest Lyapunov exponent: {LARGEST_LYAPUNOV_EXPONENT}')
    print(f'Maximum epochs: {epochs}')
    print(f'{"=" * 80}\n')

    # Calculate time arrays
    t = np.arange(0, T + dt, dt)
    lyapunov_time = calculate_lyapunov_time_array(t, LARGEST_LYAPUNOV_EXPONENT)
    max_prediction_horizon = get_max_prediction_horizon(T, LARGEST_LYAPUNOV_EXPONENT)

    print(f'Time steps: {len(t)}')
    print(f'Maximum prediction horizon: {max_prediction_horizon:.4f} Lyapunov times')

    # Load data
    trajectories = load_trajectory_data(trajectory_file)
    model_configs = load_model_configurations(models_file)

    # Prepare training data
    nn_input, nn_output, num_trajectories, batch_size = prepare_training_data(
        trajectories, dt, T
    )

    # Time arrays for metric (excluding last point)
    time_array_for_metric = t[:-1]
    lyapunov_time_for_metric = lyapunov_time[:-1]

    # Train models (parallel or sequential)
    if num_processes > 1:
        # Limit processes to number of models or CPU cores
        num_processes = min(num_processes, len(model_configs), cpu_count())
        results = train_models_parallel(
            model_configs, nn_input, nn_output,
            time_array_for_metric, lyapunov_time_for_metric,
            max_prediction_horizon, epochs, batch_size,
            output_dir, start_letter, num_processes
        )
    else:
        results = train_models_sequential(
            model_configs, nn_input, nn_output,
            time_array_for_metric, lyapunov_time_for_metric,
            max_prediction_horizon, epochs, batch_size,
            output_dir, start_letter
        )

    # Save training summary
    save_training_summary(results, output_dir)

    print(f'\n{"=" * 80}')
    print('Training Complete')
    print(f'Models labeled: {get_model_letter(0, start_letter)} - '
          f'{get_model_letter(len(model_configs)-1, start_letter)}')
    print(f'{"=" * 80}')


if __name__ == "__main__":
    main()