"""
Neural Network Model Training for Lorenz System

This script trains multiple neural network models based on hyperparameter configurations
identified during the tuning phase. The models learn to predict one-step-ahead dynamics
of the Lorenz system.

Key features:
- Parallel training of multiple model configurations
- Early stopping to prevent overfitting
- Automatic model saving with descriptive filenames
- Flexible model labeling with customizable starting letters
- Support for various optimizers and architectures

Example arguments:
-f selected_trajectories_iter10_1000_TTs.json -o top10_1000TT_MSE.json -n 1000 -m "Model_1000" -p 10 -e 365 -s A 

Note: -p sets the numbers of processes based on the number of models we have in the options file. If you
      set -p to 10, then all is well and 1 process is kicked off for each model, however if you set this to be >10 
      or more accurately, more than the number of models you have in the options file, then you will have some 
      processes that are consuming memory whilst sat idle. 

Author: Anthony S. Miller, University of Essex
Date: 17/10/2025
"""

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, Nadam, Adamax, SGD, RMSprop, Adadelta, AdamW
import json
from keras.callbacks import EarlyStopping
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count

warnings.simplefilter("ignore")
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        dict: Dictionary containing parsed arguments
    """
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Train neural network models for Lorenz system prediction"
    )
    parser.add_argument(
        "-f", "--trajectoryfile",
        default="IC_x0_1000_TTs.json",
        type=str,
        help="Input file containing training trajectories"
    )
    parser.add_argument(
        "-o", "--optionsfile",
        default="top10MSE.json",
        type=str,
        help="Input file containing model configurations from tuning"
    )
    parser.add_argument(
        "-n", "--num_trajectories",
        default=1000,
        type=int,
        help="Number of trajectories in the training data"
    )
    parser.add_argument(
        "-m", "--model_label",
        default="Epochs365_model",
        type=str,
        help="Label for model filename (e.g., 'Epochs365_model-09052025-LowPLV')"
    )
    parser.add_argument(
        "-p", "--num_processes",
        default=1,
        type=int,
        help="Number of parallel processes for training models (default: 1 for sequential)"
    )
    parser.add_argument(
        "-e", "--epochs",
        default=365,
        type=int,
        help="Number of training epochs (default: 365)"
    )
    parser.add_argument(
        "-s", "--start_letter",
        default="A",
        type=str,
        help="Starting letter for model labels (default: 'A'). Use 'K' to start from K, etc."
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


def validate_start_letter(start_letter):
    """
    Validate and convert starting letter to uppercase.
    
    Parameters:
        start_letter (str): Starting letter for model labeling
        
    Returns:
        str: Uppercase version of the letter
        
    Raises:
        ValueError: If input is not a single letter A-Z
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
        str: Model letter label (e.g., 'A', 'B', ..., 'Z', 'AA', 'AB', ...)
        
    Examples:
        >>> get_model_letter(0, 'A')
        'A'
        >>> get_model_letter(0, 'K')
        'K'
        >>> get_model_letter(5, 'K')
        'P'
    """
    start_ord = ord(start_letter.upper())
    target_ord = start_ord + model_idx
    
    # Handle wrap-around if going past 'Z'
    if target_ord <= ord('Z'):
        return chr(target_ord)
    else:
        # For labels beyond Z, use AA, AB, AC, etc.
        excess = target_ord - ord('Z') - 1
        first_letter = chr(ord('A') + (excess // 26))
        second_letter = chr(ord('A') + (excess % 26))
        return first_letter + second_letter


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
    
    The configurations include hyperparameters such as layer sizes,
    activation functions, learning rates, and optimizers.
    
    Parameters:
        filename (str): Path to model configurations JSON file
        
    Returns:
        np.ndarray: Array of model configurations
    """
    print(f'Reading model configurations from: {filename}')
    with open(filename, 'r') as f:
        data = json.load(f)
        configurations = np.asarray(data["models"])
    print(f'Loaded {len(configurations)} model configurations')
    return configurations


def prepare_training_data(trajectories, num_trajectories, dt=0.01, T=20):
    """
    Prepare training data from trajectories for the neural network.
    
    The neural network learns a one-step prediction: given state x(t),
    predict state x(t + dt). This function transforms trajectory data
    into input-output pairs suitable for supervised learning.
    
    Parameters:
        trajectories (dict): Dictionary of trajectory arrays
        num_trajectories (int): Number of trajectories to process
        dt (float): Integration time step
        T (float): Total integration time
        
    Returns:
        tuple: (nn_input, nn_output) arrays with shape (n_samples, 3)
    """
    print('Preparing training data...')
    
    t = np.arange(0, T + dt, dt)
    time_steps_per_traj = len(t) - 1
    
    # Preallocate arrays
    nn_input = np.zeros((num_trajectories * time_steps_per_traj, 3))
    nn_output = np.zeros_like(nn_input)
    
    # Sort keys to ensure consistent processing order
    sorted_keys = sorted(trajectories.keys(), key=lambda x: int(x))
    
    # Convert trajectories to input-output pairs
    for idx, key in enumerate(sorted_keys[:num_trajectories]):
        trajectory = np.asarray(trajectories[key])
        
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
        raise ValueError(f"Unknown optimizer: {optimizer_name}. "
                        f"Available: {list(optimizer_map.keys())}")
    
    return optimizer_map[optimizer_name.lower()](learning_rate=learning_rate)


def build_and_train_model(model_params):
    """
    Build and train a single neural network model.
    
    This function is designed to be called in parallel. It constructs a model
    based on the provided hyperparameters, trains it, and saves the result.
    
    Parameters:
        model_params (tuple): Tuple containing:
            - model_idx (int): Model index for identification
            - config (array): Model configuration [layer1_units, layer2_units,
                             activation1, activation2, learning_rate, optimizer, score]
            - nn_input_train (np.ndarray): Training input data
            - nn_output_train (np.ndarray): Training output data
            - nn_input_val (np.ndarray): Validation input data
            - nn_output_val (np.ndarray): Validation output data
            - model_filepath (str): Path where model should be saved
            - epochs (int): Number of training epochs
            - batch_size (int): Training batch size
            - early_stopping_patience (int): Patience for early stopping
            - start_letter (str): Starting letter for model labels
            
    Returns:
        tuple: (model_idx, model_label, final_loss, final_val_loss)
    """
    (model_idx, config, nn_input_train, nn_output_train, nn_input_val, nn_output_val,
     model_filepath, epochs, batch_size, early_stopping_patience, start_letter) = model_params
    
    # Extract hyperparameters from configuration
    layer1_units = int(config[0])
    layer2_units = int(config[1])
    activation_1 = config[2]
    activation_2 = config[3]
    learning_rate = float(config[4])
    optimizer_name = config[5]
    # config[6] is the score from tuning (optional)
    
    # Generate model label using starting letter
    model_label = get_model_letter(model_idx, start_letter)
    
    print(f'\n{"=" * 50}')
    print(f'Training Model {model_label}')
    print(f'{"=" * 50}')
    print(f'Architecture: [{layer1_units}, {layer2_units}]')
    print(f'Activations: [{activation_1}, {activation_2}]')
    print(f'Optimizer: {optimizer_name} (lr={learning_rate})')
    print(f'Epochs: {epochs}')
    
    # Build model
    model = Sequential([
        Dense(layer1_units, input_dim=3, activation=activation_1),
        Dense(layer2_units, activation=activation_2),
        Dense(3, activation='linear')
    ])
    
    # Get optimizer
    optimizer = get_optimizer(optimizer_name, learning_rate)
    
    # Compile model
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            verbose=1,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        nn_input_train, nn_output_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(nn_input_val, nn_output_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Get final losses
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    # Save model
    model.save(model_filepath)
    print(f'Saved Model {model_label} to: {model_filepath}')
    print(f'Final training loss: {final_loss:.6e}')
    print(f'Final validation loss: {final_val_loss:.6e}')
    
    # Print model summary
    model.summary()
    
    return model_idx, model_label, final_loss, final_val_loss


def train_models_sequential(model_configs, nn_input_train, nn_output_train,
                           nn_input_val, nn_output_val, model_dir, num_trajectories,
                           model_label, epochs, batch_size, early_stopping_patience,
                           start_letter='A'):
    """
    Train multiple models sequentially.
    
    Parameters:
        model_configs (np.ndarray): Array of model configurations
        nn_input_train (np.ndarray): Training input data
        nn_output_train (np.ndarray): Training output data
        nn_input_val (np.ndarray): Validation input data
        nn_output_val (np.ndarray): Validation output data
        model_dir (str): Directory to save models
        num_trajectories (int): Number of trajectories used
        model_label (str): Label for model filenames
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        early_stopping_patience (int): Patience for early stopping
        start_letter (str): Starting letter for model labels (default: 'A')
        
    Returns:
        list: List of (model_idx, model_label, final_loss, final_val_loss) tuples
    """
    print(f'\n{"=" * 50}')
    print(f'Training {len(model_configs)} Models Sequentially')
    print(f'Model labels will start from: {start_letter}')
    print(f'{"=" * 50}\n')
    
    results = []
    
    for model_idx, config in enumerate(tqdm(model_configs, desc="Training models")):
        # Generate model letter
        model_letter = get_model_letter(model_idx, start_letter)
        model_filepath = f'{model_dir}ann_x0_{num_trajectories}_MSE_{model_label}_Model{model_letter}.keras'
        
        # Prepare parameters
        params = (model_idx, config, nn_input_train, nn_output_train,
                 nn_input_val, nn_output_val, model_filepath, epochs,
                 batch_size, early_stopping_patience, start_letter)
        
        # Train model
        result = build_and_train_model(params)
        results.append(result)
    
    return results


def train_models_parallel(model_configs, nn_input_train, nn_output_train,
                         nn_input_val, nn_output_val, model_dir, num_trajectories,
                         model_label, epochs, batch_size, early_stopping_patience,
                         num_processes, start_letter='A'):
    """
    Train multiple models in parallel.
    
    Parameters:
        model_configs (np.ndarray): Array of model configurations
        nn_input_train (np.ndarray): Training input data
        nn_output_train (np.ndarray): Training output data
        nn_input_val (np.ndarray): Validation input data
        nn_output_val (np.ndarray): Validation output data
        model_dir (str): Directory to save models
        num_trajectories (int): Number of trajectories used
        model_label (str): Label for model filenames
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        early_stopping_patience (int): Patience for early stopping
        num_processes (int): Number of parallel processes
        start_letter (str): Starting letter for model labels (default: 'A')
        
    Returns:
        list: List of (model_idx, model_label, final_loss, final_val_loss) tuples
    """
    print(f'\n{"=" * 50}')
    print(f'Training {len(model_configs)} Models in Parallel')
    print(f'Using {num_processes} parallel processes')
    print(f'Model labels will start from: {start_letter}')
    print(f'{"=" * 50}\n')
    
    # Prepare parameters for each model
    model_params_list = []
    for model_idx, config in enumerate(model_configs):
        model_letter = get_model_letter(model_idx, start_letter)
        model_filepath = f'{model_dir}ann_x0_{num_trajectories}_MSE_{model_label}_Model{model_letter}.keras'
        
        params = (model_idx, config, nn_input_train, nn_output_train,
                 nn_input_val, nn_output_val, model_filepath, epochs,
                 batch_size, early_stopping_patience, start_letter)
        model_params_list.append(params)
    
    # Train models in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(build_and_train_model, model_params_list)
    
    return results


def save_training_summary(results, output_dir):
    """
    Save summary of training results to CSV.
    
    Parameters:
        results (list): List of (model_idx, model_label, final_loss, final_val_loss) tuples
        output_dir (str): Directory to save summary file
    """
    import pandas as pd
    
    summary_data = []
    for model_idx, model_label, final_loss, final_val_loss in results:
        summary_data.append({
            'model_index': model_idx,
            'model_label': model_label,
            'final_training_loss': final_loss,
            'final_validation_loss': final_val_loss
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filepath = f'{output_dir}training_summary.csv'
    summary_df.to_csv(summary_filepath, index=False)
    print(f'\nSaved training summary to: {summary_filepath}')
    
    # Print best model
    best_idx = summary_df['final_validation_loss'].idxmin()
    best_model = summary_df.iloc[best_idx]
    print(f'\nBest model: {best_model["model_label"]} '
          f'(validation loss: {best_model["final_validation_loss"]:.6e})')


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configuration: file paths
    base_dir = './_TrainModelData'
    ic_dir = base_dir + '/_ICFile/'
    best_model_dir = base_dir + '/_BestModelFile/'
    model_dir = './_Models/'
    
    ensure_directory_exists(model_dir)
    
    trajectory_file = ic_dir + args["trajectoryfile"]
    options_file = best_model_dir + args["optionsfile"]
    
    num_trajectories = args["num_trajectories"]
    model_label = args["model_label"]
    num_processes = args["num_processes"]
    epochs = args["epochs"]
    start_letter = validate_start_letter(args["start_letter"])
    
    # Training configuration
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 5
    
    # Load data
    trajectories = load_trajectory_data(trajectory_file)
    model_configs = load_model_configurations(options_file)
    
    # Prepare training data
    nn_input, nn_output = prepare_training_data(trajectories, num_trajectories)
    nn_input_train, nn_output_train, nn_input_val, nn_output_val = train_val_split(
        nn_input, nn_output
    )
    
    # Train models (parallel or sequential)
    if num_processes > 1:
        # Limit processes to number of models or CPU cores
        num_processes = min(num_processes, len(model_configs), cpu_count())
        results = train_models_parallel(
            model_configs, nn_input_train, nn_output_train,
            nn_input_val, nn_output_val, model_dir, num_trajectories,
            model_label, epochs, BATCH_SIZE, EARLY_STOPPING_PATIENCE,
            num_processes, start_letter
        )
    else:
        results = train_models_sequential(
            model_configs, nn_input_train, nn_output_train,
            nn_input_val, nn_output_val, model_dir, num_trajectories,
            model_label, epochs, BATCH_SIZE, EARLY_STOPPING_PATIENCE,
            start_letter
        )
    
    # Save training summary
    save_training_summary(results, model_dir)
    
    print('\n' + '=' * 50)
    print('Model Training Complete')
    print(f'Models labeled: {get_model_letter(0, start_letter)} - {get_model_letter(len(model_configs)-1, start_letter)}')
    print('=' * 50)


if __name__ == "__main__":
    main()