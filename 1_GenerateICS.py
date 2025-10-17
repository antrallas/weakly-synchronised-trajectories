"""
Initial Condition Generator for Chaotic System Analysis

This script generates random initial conditions uniformly distributed within
a cubic domain for numerical experiments with dynamical systems.

Author: Anthony S. Miller, University of Essex
Date: 17/10/2025
"""

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
from json import JSONEncoder
import warnings
import os

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
        description="Generate initial conditions for dynamical system experiments"
    )
    parser.add_argument(
        "-n", "--number",
        default=100,
        type=int,
        help="Number of initial conditions to generate"
    )
    parser.add_argument(
        "-s", "--seed",
        default=42,
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-d", "--datasetName",
        default="baseline",
        type=str,
        help="Name identifier for the dataset (used in output filename)"
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


def generate_initial_conditions(number, seed):
    """
    Generate random initial conditions uniformly distributed in a cubic domain.

    Parameters:
        number (int): Number of initial conditions to generate
        seed (int): Random seed for reproducibility

    Returns:
        np.ndarray: Array of shape (number, 3) containing initial conditions
    """
    np.random.seed(seed)
    # Generate uniform random samples in [-15, 15] for each dimension
    x0s = 30 * (np.random.random((number, 3)) - 0.5)
    return x0s


def save_initial_conditions(x0s, filename):
    """
    Save initial conditions to JSON file.

    Parameters:
        x0s (np.ndarray): Array of initial conditions
        filename (str): Output filename
    """
    encoded_data = {"ICs": x0s}
    print(f'Saving {len(x0s)} initial conditions to {filename}...')
    with open(filename, 'w') as f:
        json.dump(encoded_data, f, cls=NumpyArrayEncoder, indent=2)
    print('Save complete.')


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    number = args["number"]
    seed = args["seed"]
    dataset_name = args["datasetName"]

    # Configuration
    data_dir = './_DataFiles/'

    # Ensure output directory exists
    ensure_directory_exists(data_dir)
    
    # Seed reference guide (for reproducibility of published results):
    # Training Set (Original): seed = 42, datasetName = "baseline"
    # Prediction Set 1: seed = 65, datasetName = "prediction_set1"
    # Prediction Set 2: seed = 85, datasetName = "prediction_set2"
    # Prediction Set 3: seed = 33, datasetName = "prediction_set3"
    # Prediction Set 4: seed = 17, datasetName = "prediction_set4"
    # Prediction Set 5: seed = 93, datasetName = "prediction_set5"
    
    # Construct filename based on parameters
    filename = f'{data_dir}IC_x0_{number}_{dataset_name}.json'
    
    # Generate initial conditions
    x0s = generate_initial_conditions(number, seed)
    
    # Save to file
    save_initial_conditions(x0s, filename)


if __name__ == "__main__":
    main()