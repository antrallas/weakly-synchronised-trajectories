"""

This script aggregates results from multiple hyperparameter tuning iterations,
identifies the best performing model configurations, and exports them for
model training.

The script:
1. Loads all tuning result CSV files from a directory
2. Combines them into a single dataset
3. Sorts by validation score to identify top performers
4. Exports the top N configurations to JSON format

Example arguments:
-d ../_TuningData-1000TT/results -o top10_MSE_test.json -n 10

Author: Anthony S. Miller, University of Essex
Date: 17/10/2025
"""

import pandas as pd
import glob
import json
from json import JSONEncoder
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os


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
        description="Aggregate tuning results and identify best model configurations"
    )
    parser.add_argument(
        "-d", "--tuning_directory",
        default="./_TuningData/",
        type=str,
        help="Directory containing tuning result CSV files"
    )
    parser.add_argument(
        "-o", "--output_file",
        default="top10_MSE.json",
        type=str,
        help="Output JSON file for best model configurations"
    )
    parser.add_argument(
        "-n", "--top_n",
        default=10,
        type=int,
        help="Number of top models to select (default: 10)"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="*.csv",
        type=str,
        help="File pattern to match CSV files (default: '*.csv')"
    )
    parser.add_argument(
        "--save_combined",
        action="store_true",
        help="Save the combined sorted results to CSV"
    )
    return vars(parser.parse_args())


def load_csv_files(directory, pattern="*.csv"):
    """
    Load and combine all CSV files matching the pattern from a directory.
    
    Parameters:
        directory (str): Path to directory containing CSV files
        pattern (str): Glob pattern for matching files (default: '*.csv')
        
    Returns:
        pd.DataFrame: Combined dataframe from all CSV files
    """
    search_pattern = os.path.join(directory, pattern)
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern: {search_pattern}")
    
    print(f'Found {len(csv_files)} CSV files in: {directory}')
    print(f'Loading and combining files...')
    
    dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
            print(f'  Loaded: {os.path.basename(csv_file)} ({len(df)} rows)')
        except Exception as e:
            print(f'  Warning: Could not load {os.path.basename(csv_file)}: {e}')
    
    if not dataframes:
        raise ValueError("No valid CSV files could be loaded")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f'\nCombined dataset: {len(combined_df)} total rows')
    
    return combined_df


def sort_and_select_top_models(df, top_n=10, score_column='Score', ascending=True):
    """
    Sort models by score and select the top N performers.
    
    Parameters:
        df (pd.DataFrame): Dataframe containing model results
        top_n (int): Number of top models to select
        score_column (str): Column name to sort by (default: 'Score')
        ascending (bool): Sort order (True for lower is better, False for higher is better)
        
    Returns:
        tuple: (sorted_df, top_n_df) - Full sorted dataframe and top N models
    """
    if score_column not in df.columns:
        raise ValueError(f"Score column '{score_column}' not found in dataframe. "
                        f"Available columns: {list(df.columns)}")
    
    print(f'\nSorting by {score_column} ({"ascending" if ascending else "descending"})...')
    sorted_df = df.sort_values(by=[score_column], ascending=ascending)
    
    # Select top N models
    top_n_df = sorted_df.head(top_n)
    
    print(f'Selected top {len(top_n_df)} models')
    
    return sorted_df, top_n_df


def display_dataframe(df, title="DataFrame"):
    """
    Display dataframe with full formatting for easy viewing.
    
    Parameters:
        df (pd.DataFrame): Dataframe to display
        title (str): Title to print above the dataframe
    """
    print(f'\n{"=" * 80}')
    print(f'{title}')
    print(f'{"=" * 80}')
    
    with pd.option_context('display.max_rows', None, 
                          'display.max_columns', None, 
                          'display.width', 1000,
                          'display.precision', 6):
        print(df)
    
    print(f'{"=" * 80}\n')


def export_top_models_to_json(df, output_file, columns_to_export):
    """
    Export top model configurations to JSON format.
    
    Parameters:
        df (pd.DataFrame): Dataframe containing top models
        output_file (str): Path to output JSON file
        columns_to_export (list): List of column names to include in export
    """
    # Verify all columns exist
    missing_columns = [col for col in columns_to_export if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in dataframe: {missing_columns}")
    
    print(f'Exporting model configurations...')
    print(f'Columns: {columns_to_export}')
    
    # Convert to numpy array
    json_data = df[columns_to_export].to_numpy()
    
    # Create JSON structure
    encoded_data = {"models": json_data}
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(encoded_data, f, cls=NumpyArrayEncoder, indent=2)
    
    print(f'Saved {len(json_data)} model configurations to: {output_file}')


def display_summary_statistics(df, score_column='Score'):
    """
    Display summary statistics for the model scores.
    
    Parameters:
        df (pd.DataFrame): Dataframe containing model results
        score_column (str): Column name containing scores
    """
    print(f'\n{"=" * 80}')
    print('Summary Statistics')
    print(f'{"=" * 80}')
    print(f'Total models evaluated: {len(df)}')
    print(f'Best score: {df[score_column].min():.6e}')
    print(f'Worst score: {df[score_column].max():.6e}')
    print(f'Mean score: {df[score_column].mean():.6e}')
    print(f'Median score: {df[score_column].median():.6e}')
    print(f'Std deviation: {df[score_column].std():.6e}')
    print(f'{"=" * 80}\n')


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    tuning_directory = args["tuning_directory"]
    output_file = os.path.join(tuning_directory, args["output_file"])
    top_n = args["top_n"]
    pattern = args["pattern"]
    save_combined = args["save_combined"]
    
    # Ensure output directory exists
    os.makedirs(tuning_directory, exist_ok=True)
    
    print(f'\n{"=" * 80}')
    print('Hyperparameter Tuning Results Aggregator')
    print(f'{"=" * 80}\n')
    
    # Load all CSV files
    try:
        combined_df = load_csv_files(tuning_directory, pattern)
    except Exception as e:
        print(f'Error loading CSV files: {e}')
        return
    
    # Sort and select top models
    sorted_df, top_n_df = sort_and_select_top_models(combined_df, top_n)
    
    # Display results
    display_summary_statistics(sorted_df)
    display_dataframe(sorted_df.head(20), f"Top 20 Models (sorted by Score)")
    display_dataframe(top_n_df, f"Top {top_n} Models Selected for Training")
    
    # Define columns to export
    columns_to_export = [
        "layer_1_units", 
        "layer_2_units", 
        "layer_1_activation", 
        "layer_2_activation", 
        "learning_rate", 
        "optimizer", 
        "Score"
    ]
    
    # Export top models to JSON
    try:
        export_top_models_to_json(top_n_df, output_file, columns_to_export)
    except Exception as e:
        print(f'Error exporting to JSON: {e}')
        return
    
    # Optionally save combined sorted results
    if save_combined:
        combined_output = os.path.join(tuning_directory, "combined_sorted_results.csv")
        sorted_df.to_csv(combined_output, index=False)
        print(f'Saved combined sorted results to: {combined_output}')
    
    print('\n' + '=' * 80)
    print('Processing Complete')
    print('=' * 80)


if __name__ == "__main__":
    main()